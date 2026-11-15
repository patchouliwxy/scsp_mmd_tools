"""Experimental FBX preprocessing. Run in a fresh Blender background process.

No PMX/MMR, auto weights, rest-pose baking, or source-file modifications.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mathutils import Matrix, Vector


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mesh_fingerprint(obj):
    h = hashlib.sha256()
    for v in obj.data.vertices:
        h.update(repr((v.index, tuple(v.co), [(g.group, g.weight) for g in v.groups])).encode())
    h.update(repr([g.name for g in obj.vertex_groups]).encode())
    if obj.data.shape_keys:
        for key in obj.data.shape_keys.key_blocks:
            h.update(key.name.encode())
            for point in key.data:
                h.update(repr(tuple(point.co)).encode())
    return h.hexdigest()


def evaluated_vertices(meshes):
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    result = {}
    for obj in meshes:
        ev = obj.evaluated_get(dg)
        mesh = ev.to_mesh()
        try:
            result[obj.name] = [ev.matrix_world @ v.co for v in mesh.vertices]
        finally:
            ev.to_mesh_clear()
    return result


def distance(before, after):
    errors = []
    for name, vertices in before.items():
        if len(vertices) != len(after[name]):
            raise ValueError("Topology changed: " + name)
        errors.extend((a - b).length for a, b in zip(vertices, after[name]))
    return {"max_m": max(errors, default=0), "mean_m": sum(errors) / max(1, len(errors)),
            "vertices": len(errors)}


def select_skin_armature(armatures, meshes):
    """Select one real rig; empty unbound FBX containers remain references."""
    bound = {m.object for mesh in meshes for m in mesh.modifiers
             if m.type == 'ARMATURE' and m.object is not None}
    effective = [arm for arm in armatures if len(arm.data.bones) or arm in bound]
    ignored = [arm for arm in armatures if arm not in effective]
    for arm in ignored:
        if arm.animation_data or arm.constraints:
            raise ValueError('Empty armature has runtime dependencies: ' + arm.name)
    if len(effective) > 1:
        raise ValueError('Multiple non-empty or skin-bound armatures require explicit handling: '
                         + ', '.join(arm.name for arm in effective))
    return (effective[0] if effective else None), [
        {'object': arm.name, 'bones': 0, 'skin_bound': False,
         'policy': 'Retained as source reference; not selected as deformation rig'}
        for arm in ignored]


def _normalize_inverse_geogp_offsets(objects):
    """Fix head sub-meshes exported with an unapplied inverse face_geoGp offset (e.g. M_Head_10)."""
    by_source = {obj.get("scsp_source_object"): obj for obj in objects}
    face = by_source.get("face")
    face_geogp = by_source.get("face_geoGp")
    if face is None or face_geogp is None:
        return []
    gp_loc = Vector(face_geogp.location)
    if gp_loc.length <= 1e-4 or Vector(face.location).length >= 1e-4:
        return []
    repaired = []
    for obj in objects:
        if obj.type != 'MESH':
            continue
        loc = Vector(obj.location)
        if (loc + gp_loc).length < 1e-4:
            obj.location = (0.0, 0.0, 0.0)
            obj["scsp_repaired_inverse_geogp_offset"] = list(loc)
            repaired.append({"object": obj.name, "source_object": obj.get("scsp_source_object"),
                             "cleared_location": list(loc)})
    if repaired:
        bpy.context.view_layer.update()
    return repaired


def import_parts(config, report):
    parts = []
    for i, entry in enumerate(config["parts"]):
        path = Path(entry["path"]).resolve(strict=True)
        existing = set(bpy.data.objects)
        importer = config.get("importer", "ufbx")
        if importer == "ufbx":
            from ufbx_importer import import_fbx_file
            import_fbx_file(path)
        elif importer == "better_fbx":
            options=dict(use_auto_bone_orientation=False, use_only_deform_bones=False,
                use_fix_bone_poses=False, use_animation=False, use_vertex_animation=False,
                use_detect_deform_bone=True, use_optimize_for_blender=False,
                use_fix_attributes=False, use_fix_mesh_scaling=False,
                use_import_materials=True, use_triangulate=False, my_scale=1.0,
                my_fbx_unit="m", my_import_normal="Import")
            if config.get("better_fbx_headless", False):
                from better_fbx_headless import import_file
                detail=import_file(path,Path(config["better_fbx_addons_dir"])/"better_fbx",**options)
                report.setdefault('headless_imports',[]).append(detail)
            else:
                result=bpy.ops.better_import.fbx(filepath=str(path),files=[{"name":path.name}],**options)
                if result != {'FINISHED'}:
                    raise RuntimeError("Better FBX import did not finish: " + str(result))
        else:
            bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False, use_image_search=True,
                                     automatic_bone_orientation=False, ignore_leaf_bones=False,
                                     use_prepost_rot=True, bake_space_transform=False,
                                     global_scale=1.0)
        objects = [o for o in bpy.data.objects if o not in existing]
        label = f"{i:02d}_{entry['kind']}"
        collection = bpy.data.collections.new(label)
        bpy.context.scene.collection.children.link(collection)
        # FBX skin names are case-sensitive; never rename the bone or vertex groups.
        for obj in objects:
            original = obj.name
            for old in list(obj.users_collection):
                old.objects.unlink(obj)
            collection.objects.link(obj)
            obj.name = label + "__" + original
            obj["scsp_source_object"] = original
            obj["scsp_source_fbx"] = str(path)
        repaired_offsets = _normalize_inverse_geogp_offsets(objects) if entry['kind'] == 'head' else []
        armatures = [o for o in objects if o.type == 'ARMATURE']
        meshes = [o for o in objects if o.type == 'MESH']
        arm, ignored_armatures = select_skin_armature(armatures, meshes)
        if arm and (arm.animation_data or any(pb.constraints for pb in arm.pose.bones)):
            raise ValueError("Animated/constrained input needs explicit handling: " + arm.name)
        info = {"kind": entry["kind"], "path": str(path), "sha256": sha(path),
                "armature": arm.name if arm else None, "bones": len(arm.data.bones) if arm else 0, "meshes": [],
                "ignored_empty_armatures": ignored_armatures,
                "repaired_inverse_offsets": repaired_offsets,
                "bone_rest": {b.name: {"parent": b.parent.name if b.parent else None,
                    "matrix_world": [list(r) for r in (arm.matrix_world @ b.matrix_local)]}
                    for b in arm.data.bones} if arm else {}}
        for mesh in meshes:
            mods = [m for m in mesh.modifiers if m.type == 'ARMATURE']
            if any(m.object != arm or m.use_deform_preserve_volume for m in mods):
                raise ValueError("Unexpected skin target or preserve-volume input: " + mesh.name)
            info["meshes"].append({"name": mesh.name, "vertices": len(mesh.data.vertices),
                "skin_modifiers": len(mods), "raw_fingerprint": mesh_fingerprint(mesh),
                "shape_keys": [k.name for k in mesh.data.shape_keys.key_blocks] if mesh.data.shape_keys else []})
        report["sources"].append(info)
        parts.append({"kind": entry["kind"], "objects": objects, "arm": arm, "meshes": meshes})
    return parts


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--import-only', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if bpy.data.filepath:
        raise ValueError("Use a fresh --factory-startup process")
    config_path = Path(args.config).resolve(strict=True)
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / n).exists() for n in ('01-import.blend', '02-controls.blend', '03-proportions.blend')):
        raise FileExistsError("Use a new output folder; existing stage files are never overwritten")
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    report = {"schema": 1, "experimental": True, "config_sha256": sha(config_path),
              "blender": bpy.app.version_string, "sources": [], "stages": {},
              "scope": "FBX import, independent control assembly, explicit body proportions; no PMX/MMR"}
    importer_name = config.get("importer", "ufbx")
    if importer_name == "ufbx":
        dll_path = (Path(__file__).resolve().parent / "ufbx_bridge.dll").resolve(strict=True)
        report['importer'] = {'name': 'ufbx', 'license': 'MIT', 'dll': str(dll_path)}
    elif importer_name == "better_fbx":
        import addon_utils
        addon_dir = Path(config["better_fbx_addons_dir"]).resolve(strict=True)
        sys.path.insert(0, str(addon_dir))
        if not config.get('better_fbx_headless',False):
            addon_utils.enable('better_fbx', default_set=True, persistent=False)
        import better_fbx
        report['importer'] = {'name': 'better_fbx', 'version': list(better_fbx.bl_info['version']),
                              'path': str(addon_dir / 'better_fbx')}
    parts = import_parts(config, report)
    meshes = [m for p in parts for m in p['meshes']]
    raw = {m.name: mesh_fingerprint(m) for m in meshes}
    initial = evaluated_vertices(meshes)
    report['stages']['import'] = {"objects": len(bpy.data.objects), "meshes": len(meshes),
                                  "vertices": sum(len(v) for v in initial.values())}
    from import_processing import prepare_materials
    report['parts_materials']=prepare_materials(parts)
    report['materials_prepared']=True
    bpy.ops.wm.save_as_mainfile(filepath=str(output / '01-import.blend'))
    if not args.import_only:
        raise NotImplementedError("Control assembly implementation pending source audit")
    report['source_hashes_unchanged'] = all(sha(s['path']) == s['sha256'] for s in report['sources'])
    report['raw_meshes_unchanged'] = all(mesh_fingerprint(m) == raw[m.name] for m in meshes)
    if not report['source_hashes_unchanged'] or not report['raw_meshes_unchanged']:
        raise AssertionError("Source preservation check failed")
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print("SCSP_PREPROCESS_RESULT " + json.dumps({"output": str(output), "stages": report['stages']}))


if __name__ == '__main__':
    run()
