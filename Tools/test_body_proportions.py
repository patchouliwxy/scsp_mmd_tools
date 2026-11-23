"""Small Blender synthetic test for the assembled-rig body_proportions API."""
import json
import math
import sys
import tempfile
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from body_proportions import BindingError, apply, bake_bust, initialize, restore


def close(actual, expected, tolerance=1e-6):
    assert abs(actual - expected) <= tolerance, (actual, expected)


def make_rig():
    data = bpy.data.armatures.new("AssembledData")
    obj = bpy.data.objects.new("Assembled", data)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    specs = [
        ("Head", (0, 0, 1), (0, 0, 1.5), None),
        ("LeftArm", (.2, 0, .7), (.5, 0, .7), None),
        ("RightArm", (-.2, 0, .7), (-.5, 0, .7), None),
        ("LeftHand", (.5, 0, .7), (.8, 0, .7), "LeftArm"),
        ("RightHand", (-.5, 0, .7), (-.8, 0, .7), "RightArm"),
        ("p01__Head", (0, 0, 1.7), (0, 0, 2), "Head"),
        ("p01__Top", (0, 0, 2), (0, 0, 2.2), "p01__Head"),
    ]
    bones = {}
    for name, head, tail, _ in specs:
        bones[name] = obj.data.edit_bones.new(name)
        bones[name].head, bones[name].tail = head, tail
    for name, _, _, parent in specs:
        if parent:
            bones[name].parent = bones[parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.select_set(False)
    return obj


def make_mesh(name, rig, skinned, baseline):
    data = bpy.data.meshes.new(name + "Data")
    data.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    if skinned:
        group = obj.vertex_groups.new(name="Head")
        group.add([0, 1, 2], 1, "REPLACE")
        obj.modifiers.new("Armature", "ARMATURE").object = rig
    obj.shape_key_add(name="Basis", from_mix=False)
    intermediate = obj.shape_key_add(name="SafeChannel@50pct", from_mix=False)
    curve = intermediate.driver_add("value")
    curve.driver.type = "SCRIPTED"
    curve.driver.expression = "0.0"
    key = obj.shape_key_add(name="FaceModules.bust_up", from_mix=False)
    key.data[0].co.z = .25
    key.value = baseline
    return obj


def snapshot(root, rig, meshes):
    return {
        "root": [float(root.scale[i]) for i in range(3)],
        "bones": {name: [float(rig.pose.bones[name].scale[i]) for i in range(3)]
                  for name in ("Head", "LeftArm", "RightArm", "LeftHand", "RightHand")},
        "bust": {mesh.name: float(mesh.data.shape_keys.key_blocks["FaceModules.bust_up"].value)
                 for mesh in meshes},
        "vertices": {mesh.name: [tuple(v.co) for v in mesh.data.vertices] for mesh in meshes},
    }


def same(left, right):
    assert left["root"] == right["root"]
    assert left["bones"] == right["bones"]
    assert left["bust"] == right["bust"]
    assert left["vertices"] == right["vertices"]


def run():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    root = bpy.data.objects.new("SCSP_Character", None)
    bpy.context.scene.collection.objects.link(root)
    rig = make_rig()
    skinned = make_mesh("Skinned", rig, True, .1)
    static = make_mesh("Static", rig, False, .2)
    meshes = [skinned, static]
    params = {"height": 1.01, "bust": .522, "head": .93, "arm": .95, "hand": .9}

    report = initialize(root, rig, meshes, params, "synthetic explicit factors")
    json.dumps(report)
    assert len([obj for obj in bpy.data.objects if obj.type == "ARMATURE"]) == 1
    assert {obj.name for obj in meshes} == {"Skinned", "Static"}
    assert rig.data.bones["p01__Head"].parent.name == "Head"
    assert not [m for m in static.modifiers if m.type == "ARMATURE"]
    first = snapshot(root, rig, meshes)
    close(first["root"][0], 1.01)
    close(first["bones"]["Head"][0], .93)
    close(first["bones"]["LeftArm"][0], .95)
    close(first["bones"]["RightHand"][0], .9)
    close(rig.pose.bones["p01__Head"].matrix.to_scale()[0], .93, 1e-5)
    assert all(close(value, .522) is None for value in first["bust"].values())

    apply(root, rig, meshes, params)
    same(first, snapshot(root, rig, meshes))
    path = Path(tempfile.gettempdir()) / "scsp_body_proportions_short.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(path), check_existing=False)
    bpy.ops.wm.open_mainfile(filepath=str(path), load_ui=False, use_scripts=False)
    root = bpy.data.objects["SCSP_Character"]
    rig = bpy.data.objects["Assembled"]
    meshes = [bpy.data.objects["Skinned"], bpy.data.objects["Static"]]
    apply(root, rig, meshes, params)
    same(first, snapshot(root, rig, meshes))
    restore(root, rig, meshes)
    restored = snapshot(root, rig, meshes)
    close(restored["root"][0], 1)
    close(restored["bones"]["Head"][0], 1)
    close(restored["bust"]["Skinned"], .1)
    close(restored["bust"]["Static"], .2)
    meshes[0].data.shape_keys.key_blocks["FaceModules.bust_up"].driver_add("value")
    try: apply(root, rig, meshes, params)
    except BindingError: bust_driver_rejected = True
    else: raise AssertionError("bust_up driver was accepted")
    path.unlink(missing_ok=True)
    bust_bake_result = test_bake_bust()
    preserve_bust_result = test_bake_bust_preserve_morph()
    print("SCSP_BODY_PROPORTIONS_TEST " + json.dumps({
        "status": "PASS", "blender": bpy.app.version_string, "armatures": 1,
        "meshes": 2, "static_mesh_without_modifier": True,
        "safe_shape_drivers_allowed": True, "reapply_idempotent": True,
        "reload_baseline": True, "bust_driver_rejected": bust_driver_rejected,
        "bust_bake": bust_bake_result,
        "preserve_bust_morph": preserve_bust_result,
        "restored": restored,
    }, sort_keys=True))


def test_bake_bust():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    root = bpy.data.objects.new("SCSP_Character", None)
    bpy.context.scene.collection.objects.link(root)
    rig = make_rig()

    explicit = make_mesh("Skinned", rig, True, 0.0)
    explicit.data.normals_split_custom_set([(0.0, 0.0, 1.0)] * len(explicit.data.loops))
    safe = explicit.data.shape_keys.key_blocks["SafeChannel@50pct"]
    for idx in range(len(explicit.data.vertices)):
        safe.data[idx].co.y += 0.3
    explicit.data.shape_keys.animation_data.drivers[0].driver.expression = "0.5"

    carrier_data = bpy.data.meshes.new("ClothBsBData")
    carrier_data.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    carrier_data.update()
    carrier_data.normals_split_custom_set([(0.0, 0.0, 1.0)] * len(carrier_data.loops))
    carrier = bpy.data.objects.new("01_dress__cloth_BS_b", carrier_data)
    bpy.context.scene.collection.objects.link(carrier)
    group = carrier.vertex_groups.new(name="Head")
    group.add([0, 1, 2], 1.0, "REPLACE")
    carrier.modifiers.new("Armature", "ARMATURE").object = rig
    carrier.shape_key_add(name="Basis", from_mix=False)
    bs_key = carrier.shape_key_add(name="cloth_BS_b1", from_mix=False)
    bs_key.data[0].co.z = 0.25
    carrier.hide_viewport = True

    meshes = [explicit, carrier]
    params = {"height": 1.01, "bust": .522, "head": .93, "arm": .95, "hand": .9}
    initialize(root, rig, meshes, params, "synthetic bust bake")

    # Capture expected normals in local space at bust=0.522 before baking.
    expected = {}
    for obj in meshes:
        old_hide = obj.hide_viewport
        obj.hide_viewport = False
        for mod in obj.modifiers:
            mod.show_viewport = False
        if "SafeChannel@50pct" in obj.data.shape_keys.key_blocks:
            obj.data.shape_keys.key_blocks["SafeChannel@50pct"].mute = True
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(dg).to_mesh()
        expected[obj.name] = (
            [tuple(v.co) for v in ev.vertices],
            [tuple(c.vector) for c in ev.corner_normals],
        )
        obj.evaluated_get(dg).to_mesh_clear()
        if "SafeChannel@50pct" in obj.data.shape_keys.key_blocks:
            obj.data.shape_keys.key_blocks["SafeChannel@50pct"].mute = False
        for mod in obj.modifiers:
            mod.show_viewport = True
        obj.hide_viewport = old_hide

    report = bake_bust(root, rig, meshes)
    assert report["status"] == "baked"
    assert report["baked_count"] == 2
    assert carrier.hide_viewport is True

    for obj in meshes:
        exp_verts, exp_normals = expected[obj.name]
        for actual, target in zip(obj.data.vertices, exp_verts):
            for idx in range(3):
                close(actual.co[idx], target[idx], 1e-6)
        assert obj.data.has_custom_normals
        for corner, target in zip(obj.data.corner_normals, exp_normals):
            for idx in range(3):
                close(corner.vector[idx], target[idx], 1e-4)
        assert abs(obj.data.corner_normals[0].vector[0]) > 0.05

    assert [k.name for k in explicit.data.shape_keys.key_blocks] == ["Basis", "SafeChannel@50pct"]
    basis = explicit.data.shape_keys.key_blocks["Basis"]
    safe = explicit.data.shape_keys.key_blocks["SafeChannel@50pct"]
    close(safe.value, 0.5, 1e-5)
    for idx in range(len(explicit.data.vertices)):
        close(safe.data[idx].co.y - basis.data[idx].co.y, 0.3, 1e-6)
    assert carrier.data.shape_keys is None

    for fn in (
        lambda: apply(root, rig, meshes, params),
        lambda: restore(root, rig, meshes),
        lambda: bake_bust(root, rig, meshes),
    ):
        try:
            fn()
        except BindingError:
            pass
        else:
            raise AssertionError("Operation succeeded after bust bake")

    return {"baked_count": report["baked_count"], "carrier_keys_removed": True}


def test_bake_bust_preserve_morph():
    from mesh_processing import merge

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    root = bpy.data.objects.new("SCSP_Character", None)
    bpy.context.scene.collection.objects.link(root)
    rig = make_rig()
    rig.name = "SCSP_Skeleton"

    explicit = make_mesh("01_dress__Top_BS_b", rig, True, 0.0)
    explicit["scsp_source_fbx"] = "M_Dress_test.fbx"
    explicit["scsp_source_object"] = "Top_BS_b"
    explicit.data.normals_split_custom_set([(0.0, 0.0, 1.0)] * len(explicit.data.loops))
    safe = explicit.data.shape_keys.key_blocks["SafeChannel@50pct"]
    for idx in range(len(explicit.data.vertices)):
        safe.data[idx].co.y += 0.3
    explicit.data.shape_keys.animation_data.drivers[0].driver.expression = "0.5"
    # Give explicit & carrier different interpolations to verify bake_bust normalizes settings before merge.
    explicit.data.shape_keys.key_blocks["FaceModules.bust_up"].interpolation = "KEY_BSPLINE"

    carrier_data = bpy.data.meshes.new("ClothBsBData")
    carrier_data.from_pydata([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [], [(0, 1, 2)])
    carrier_data.update()
    carrier_data.normals_split_custom_set([(0.0, 0.0, 1.0)] * len(carrier_data.loops))
    carrier = bpy.data.objects.new("01_dress__cloth_BS_b", carrier_data)
    bpy.context.scene.collection.objects.link(carrier)
    carrier["scsp_source_fbx"] = "M_Dress_test.fbx"
    carrier["scsp_source_object"] = "cloth_BS_b"
    group = carrier.vertex_groups.new(name="Head")
    group.add([0, 1, 2], 1.0, "REPLACE")
    carrier.modifiers.new("Armature", "ARMATURE").object = rig
    carrier.shape_key_add(name="Basis", from_mix=False)
    bs_key = carrier.shape_key_add(name="cloth_BS_b1", from_mix=False)
    bs_key.data[0].co.z = 0.25

    meshes = [explicit, carrier]
    params = {"height": 1.01, "bust": 0.522, "head": 0.93, "arm": 0.95, "hand": 0.9}
    initialize(root, rig, meshes, params, "synthetic bust preserve")

    report = bake_bust(root, rig, meshes, preserve_bust_morph=True)
    assert report["status"] == "baked"
    assert report["preserve_bust_morph"] is True
    assert report["baked_count"] == 2

    for obj in meshes:
        assert "bust_up" in obj.data.shape_keys.key_blocks
        k = obj.data.shape_keys.key_blocks["bust_up"]
        close(k.value, 0.0)
        close(k.slider_min, 0.0)
        close(k.slider_max, 1.0)
        assert k.mute is False
        assert k.interpolation == "KEY_LINEAR"
        # At 0.0 (Basis), rest vertex 0 is already baked at 0.522 * 0.25 = 0.1305
        close(obj.data.vertices[0].co.z, 0.522 * 0.25, 1e-6)
        # At 1.0 (bust_up target), vertex 0 reaches original max 0.25 (remaining headroom = 0.478 * 0.25)
        close(k.data[0].co.z, 0.25, 1e-6)

    # Clear the test driver and reset root.scale (normally done by finalize_skeleton) before mesh_processing.merge
    explicit.data.shape_keys.driver_remove('key_blocks["SafeChannel@50pct"].value')
    root.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()
    merged, _ = merge(meshes, root, rig)
    assert len(merged) == 1
    dress = merged[0]
    assert "bust_up" in dress.data.shape_keys.key_blocks
    dk = dress.data.shape_keys.key_blocks["bust_up"]
    close(dk.value, 0.0)
    # Vertex 0 of first piece and vertex 3 of second piece both move from 0.1305 to 0.25 when bust_up = 1.0
    close(dress.data.vertices[0].co.z, 0.522 * 0.25, 1e-5)
    close(dress.data.vertices[3].co.z, 0.522 * 0.25, 1e-5)
    close(dk.data[0].co.z, 0.25, 1e-5)
    close(dk.data[3].co.z, 0.25, 1e-5)
    return {"baked_count": report["baked_count"], "merged_keys": list(dress.data.shape_keys.key_blocks.keys())}


if __name__ == "__main__":
    run()
