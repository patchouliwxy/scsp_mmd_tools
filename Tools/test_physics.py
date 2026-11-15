"""Independent unit and Blender Bullet simulation tests for Tools/pe_physics.py."""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import bpy
from mathutils import Euler, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pe_physics
import pmx_io as pmx


def _import_mmd_tools():
    import importlib, os
    ext = Path(os.environ.get("APPDATA", "")) / "Blender Foundation/Blender/5.3/extensions/user_default"
    if (ext / "mmd_tools" / "__init__.py").is_file() and str(ext) not in sys.path:
        sys.path.insert(0, str(ext))
    mod = importlib.import_module("mmd_tools")
    if not hasattr(bpy.types.Object, "mmd_type"):
        mod.register()
    from mmd_tools.core.pmx.importer import PMXImporter
    return PMXImporter


STUDIO_DIR = Path(__file__).resolve().parents[1] / "Outputs" / "studio"


def _find_studio_pmx(prefix: str) -> Path | None:
    if not STUDIO_DIR.is_dir():
        return None
    for p in sorted(STUDIO_DIR.glob(f"{prefix}*/*.pmx")):
        if "_work" not in p.parts:
            return p
    return None


class TestPhysicsProtocol(unittest.TestCase):
    def test_non_collision_mask_bits(self) -> None:
        mask = pe_physics.non_collision_mask({0, 2, 15})
        self.assertEqual(mask & (1 << 0), 0)
        self.assertNotEqual(mask & (1 << 1), 0)
        self.assertEqual(mask & (1 << 2), 0)
        self.assertEqual(mask & (1 << 15), 0)
        self.assertEqual(pe_physics.non_collision_mask(set()), 0xFFFF)

    def test_capsule_rotation_roundtrip(self) -> None:
        for vec in (
            Vector((0.0, -1.0, 0.0)),
            Vector((1.0, -2.0, 0.5)),
            Vector((-0.7, -0.3, -1.2)),
            Vector((0.2, 1.0, -0.4)),
        ):
            d_pmx = vec.normalized()
            rot_pmx = pe_physics._pmx_rot_from_dir(d_pmx)
            e_bl = Euler(Vector(rot_pmx).xzy * -1.0, "YXZ")
            z_bl = e_bl.to_matrix() @ Vector((0.0, 0.0, 1.0))
            d_bl = Vector(d_pmx).xzy.normalized()
            self.assertGreater(d_bl.dot(z_bl), 0.9999)


class TestPhysicsSimulation(unittest.TestCase):
    def _run_sim_check(self, model_prefix: str, pose_test: bool = False) -> None:
        src_pmx = _find_studio_pmx(model_prefix)
        if src_pmx is None:
            self.skipTest(f"Studio model not found: {model_prefix}")

        with tempfile.TemporaryDirectory() as tmp:
            out_pmx = Path(tmp) / f"{model_prefix}_physics.pmx"
            rep = Path(tmp) / "report.json"
            payload = pe_physics.process_pmx_file(src_pmx, out_pmx, rep)
            self.assertTrue(payload["roundtrip_audit"]["passed"])
            self.assertGreater(payload["summary"]["dynamic_rigid_count"], 0)

            bpy.ops.wm.read_factory_settings(use_empty=True)
            try:
                PMXImporter = _import_mmd_tools()
            except Exception as exc:
                self.skipTest(f"mmd_tools unavailable for Blender Bullet sim test: {exc}")
            importer = PMXImporter()
            importer.execute(filepath=str(out_pmx), scale=0.08, types={"MESH", "ARMATURE", "PHYSICS"})

            scene = bpy.context.scene
            self.assertIsNotNone(scene.rigidbody_world)
            scene.rigidbody_world.enabled = True
            scene.rigidbody_world.substeps_per_frame = 6
            scene.rigidbody_world.solver_iterations = 20
            for obj in bpy.data.objects:
                if obj.rigid_body_constraint is not None:
                    obj.rigid_body_constraint.disable_collisions = True

            rigid_objs = [o for o in bpy.data.objects if o.rigid_body is not None]
            self.assertGreater(len(rigid_objs), 0)

            scene.frame_set(1)
            init_pos = {o.name: o.matrix_world.translation.copy() for o in rigid_objs}

            # Static stability check over 60 frames
            for f in range(2, 62):
                scene.frame_set(f)

            max_drift_mmd = 0.0
            max_bust_drift = 0.0
            for o in rigid_objs:
                pos = o.matrix_world.translation
                self.assertTrue(all(math.isfinite(c) for c in pos), f"Non-finite position for {o.name}")
                if not o.rigid_body.kinematic:
                    drift_mmd = (pos - init_pos[o.name]).length / 0.08
                    max_drift_mmd = max(max_drift_mmd, drift_mmd)
                    if "OPAI_" in o.name:
                        max_bust_drift = max(max_bust_drift, drift_mmd)

            self.assertLess(max_bust_drift, 0.5, f"{model_prefix} bust drift too large: {max_bust_drift:.3f} MMD units")
            self.assertLess(max_drift_mmd, 1.5, f"{model_prefix} static drift too large: {max_drift_mmd:.3f} MMD units")

            if pose_test:
                bpy.ops.mmd_tools.build_rig()
                arm = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
                self.assertIsNotNone(arm)
                bpy.context.view_layer.objects.active = arm
                bpy.ops.object.mode_set(mode="POSE")
                if "頭" in arm.pose.bones:
                    pb = arm.pose.bones["頭"]
                    pb.rotation_mode = "XYZ"
                    pb.rotation_euler = (0.0, 0.0, math.radians(55.0))
                    pb.keyframe_insert(data_path="rotation_euler", frame=70)
                if "左足" in arm.pose.bones:
                    pb = arm.pose.bones["左足"]
                    pb.rotation_mode = "XYZ"
                    pb.rotation_euler = (math.radians(-75.0), 0.0, 0.0)
                    pb.keyframe_insert(data_path="rotation_euler", frame=80)
                bpy.ops.object.mode_set(mode="OBJECT")

                for f in range(62, 100):
                    scene.frame_set(f)

                for o in rigid_objs:
                    pos = o.matrix_world.translation
                    self.assertTrue(all(math.isfinite(c) for c in pos), f"Exploded during pose for {o.name}")
                    self.assertLess((pos - init_pos[o.name]).length / 0.08, 25.0, f"Flew away during pose: {o.name}")

    def test_chiyoko_static_and_pose(self) -> None:
        self._run_sim_check("Sonoda Chiyoko", pose_test=True)

    def test_kogane_static(self) -> None:
        self._run_sim_check("Tsukioka Kogane", pose_test=False)

    def test_mamimi_static(self) -> None:
        self._run_sim_check("Tanaka Mamimi", pose_test=False)

    def test_mano_static(self) -> None:
        self._run_sim_check("Sakuragi Mano", pose_test=False)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    unittest.main(argv=[sys.argv[0]] + argv)
