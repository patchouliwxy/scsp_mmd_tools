"""Bake a world-Y A-pose into rest space while preserving absolute shape keys.

Public API: bake(rig, meshes, degrees=45) -> JSON-safe report.
The caller supplies the already consolidated SCSP armature and every target
mesh. The function does not create, merge, rename, reparent, or delete bones.
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

SCHEMA = 1
IDENTITY = Matrix.Identity(4)
EPS = 1e-8
WORLD_TOLERANCE = 1e-5
BASELINE_KEY = "scsp_body_proportions_baseline"
BASELINE_SCHEMA_KEY = "scsp_body_proportions_schema"
ARCHIVE_KEY = "scsp_body_proportions_historical"
BAKED_FLAG = "scsp_body_proportions_baked"
ARCHIVE_SCHEMA_KEY = "scsp_body_proportions_schema_apose_archive"
ARCHIVE_REASON_KEY = "scsp_body_proportions_apose_archive_reason"
BAKE_KEY = "scsp_apose_bake"


class AposeBakeError(ValueError):
    """The supplied rig or mesh set cannot satisfy the A-pose bake contract."""


def _matrix_rows(matrix):
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def _vector_values(vector):
    return [float(vector[index]) for index in range(3)]


def _matrix_error(left, right):
    return max(abs(left[row][column] - right[row][column])
               for row in range(4) for column in range(4))


def _meshes(value):
    if isinstance(value, bpy.types.Object):
        value = [value]
    result, seen = [], set()
    for obj in value:
        if not isinstance(obj, bpy.types.Object) or obj.type != "MESH":
            raise AposeBakeError("meshes must contain only MESH objects")
        if obj.name not in seen:
            seen.add(obj.name)
            result.append(obj)
    if not result:
        raise AposeBakeError("meshes must not be empty")
    return result


def _armature_modifier(obj, rig):
    modifiers = [modifier for modifier in obj.modifiers if modifier.type == "ARMATURE"]
    if any(modifier.object != rig for modifier in modifiers):
        raise AposeBakeError("Armature modifier points outside supplied rig: " + obj.name)
    if len(modifiers) > 1:
        raise AposeBakeError("Ambiguous Armature modifiers: " + obj.name)
    if modifiers:
        modifier = modifiers[0]
        if not modifier.use_vertex_groups or modifier.use_bone_envelopes:
            raise AposeBakeError("A-pose bake requires vertex-group LBS only: " + obj.name)
        if modifier.use_deform_preserve_volume:
            raise AposeBakeError("Preserve Volume is incompatible with per-vertex LBS: " + obj.name)
    return modifiers[0] if modifiers else None


def _assert_ready(rig, meshes):
    if not isinstance(rig, bpy.types.Object) or rig.type != "ARMATURE":
        raise AposeBakeError("rig must be one ARMATURE object")
    if rig.name != "SCSP_Skeleton":
        raise AposeBakeError("Expected consolidated SCSP_Skeleton armature")
    if rig.mode != "OBJECT":
        raise AposeBakeError("bake() must be called in Object Mode")
    if rig.parent is None or rig.parent.name != "SCSP_Character":
        raise AposeBakeError("SCSP_Skeleton must be parented to SCSP_Character")
    if rig.animation_data and (rig.animation_data.action or any(
            track.strips for track in rig.animation_data.nla_tracks)):
        raise AposeBakeError("An armature action or NLA track is active")
    if rig.data.animation_data and (rig.data.animation_data.action or any(
            track.strips for track in rig.data.animation_data.nla_tracks)):
        raise AposeBakeError("Armature data has an active action or NLA track")
    connected = [bone.name for bone in rig.data.bones if bone.use_connect]
    if connected:
        raise AposeBakeError("Connected bones are unsupported: " + ", ".join(connected))
    constrained = [bone.name for bone in rig.pose.bones if bone.constraints]
    if constrained:
        raise AposeBakeError("Pose constraints are unsupported: " + ", ".join(constrained))
    required = [side + suffix for side in ("Left", "Right")
                for suffix in ("Arm", "Elbow", "Hand")]
    missing = [name for name in required if name not in rig.data.bones]
    if missing:
        raise AposeBakeError("Missing Arm/Elbow/Hand bones: " + ", ".join(missing))
    for obj in meshes:
        modifier = _armature_modifier(obj, rig)
        if modifier and not obj.data.vertices:
            raise AposeBakeError("Skinned mesh has no vertices: " + obj.name)
    return {obj.name: _armature_modifier(obj, rig) for obj in meshes}


def _mesh_state(obj):
    shape_keys = obj.data.shape_keys
    keys = [] if shape_keys is None else [{
        "name": key.name,
        "coords": tuple(tuple(float(value) for value in item.co) for item in key.data),
        "value": float(key.value),
        "relative_key": key.relative_key.name if key.relative_key else None,
    } for key in shape_keys.key_blocks]
    drivers = [] if shape_keys is None or shape_keys.animation_data is None else [
        driver.data_path for driver in shape_keys.animation_data.drivers]
    uvs = []
    for layer in obj.data.uv_layers:
        uvs.append((layer.name, tuple(tuple(float(value) for value in item.uv)
                                      for item in layer.data)))
    weights = tuple(tuple(sorted((entry.group, float(entry.weight))
                                 for entry in vertex.groups))
                    for vertex in obj.data.vertices)
    return {
        "groups": tuple(group.name for group in obj.vertex_groups),
        "weights": weights,
        "uvs": tuple(uvs),
        "keys": tuple(keys),
        "drivers": tuple(drivers),
        "relative": shape_keys.use_relative if shape_keys else None,
        "loops": int(len(obj.data.loops)),
        "verts": int(len(obj.data.vertices)),
    }


def _pose_snapshot(rig):
    return ({bone.name: bone.matrix.copy() for bone in rig.pose.bones},
            {bone.name: bone.matrix_basis.copy() for bone in rig.pose.bones})


def _bone_flags(rig):
    result = {}
    for bone in rig.data.bones:
        result[bone.name] = {
            "parent": bone.parent.name if bone.parent else None,
            "use_connect": bool(bone.use_connect),
            "use_deform": bool(bone.use_deform),
            "inherit_scale": bone.inherit_scale,
            "use_inherit_rotation": bool(bone.use_inherit_rotation),
            "use_local_location": bool(bone.use_local_location),
            "hide": bool(bone.hide),
        }
    return result


def _archive_body_proportions(rig):
    archived = []
    reason = ("The current body proportions and arm pose were baked into rest "
              "coordinates and every shape key. Reapplying the old baseline would "
              "apply the same proportions twice.")
    for owner in bpy.data.objects:
        raw = owner.get(BASELINE_KEY)
        if not isinstance(raw, str):
            continue
        try:
            baseline = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AposeBakeError("Invalid body-proportion baseline on " + owner.name) from exc
        if baseline.get("armature") != rig.name:
            continue
        owner[ARCHIVE_KEY] = raw
        owner[ARCHIVE_SCHEMA_KEY] = owner.get(BASELINE_SCHEMA_KEY, 1)
        owner[ARCHIVE_REASON_KEY] = reason
        del owner[BASELINE_KEY]
        if BASELINE_SCHEMA_KEY in owner:
            del owner[BASELINE_SCHEMA_KEY]
        archived.append(owner.name)
    root = rig.parent
    rig[ARCHIVE_REASON_KEY] = reason
    rig[BAKED_FLAG] = True
    root[ARCHIVE_REASON_KEY] = reason
    root[BAKED_FLAG] = True
    return archived, reason


def _rotate_arm(rig, name, degrees):
    pose_bone = rig.pose.bones[name]
    head = pose_bone.matrix.translation.copy()
    armature_world = rig.matrix_world.to_quaternion()
    world_rotation = Quaternion((0.0, 1.0, 0.0), math.radians(degrees))
    local_rotation = armature_world.conjugated() @ world_rotation @ armature_world
    to_head = Matrix.Translation(head)
    from_head = Matrix.Translation(-head)
    pose_bone.matrix = (to_head @ local_rotation.to_matrix().to_4x4()
                        @ from_head @ pose_bone.matrix)
    bpy.context.view_layer.update()


def _weights_and_transforms(obj, rig, pose_matrices, old_rest):
    group_names = {group.index: group.name for group in obj.vertex_groups}
    bone_names = {bone.name for bone in rig.data.bones if bone.use_deform}
    outer = obj.matrix_world.inverted() @ rig.matrix_world
    inner = rig.matrix_world.inverted() @ obj.matrix_world
    transforms, unweighted = [], 0
    for vertex in obj.data.vertices:
        influences = []
        for entry in vertex.groups:
            name = group_names.get(entry.group)
            weight = float(entry.weight)
            if name in bone_names and weight > 0.0:
                influences.append((name, weight))
        total = sum(weight for _, weight in influences)
        if total <= EPS:
            transforms.append(IDENTITY.copy())
            unweighted += 1
            continue
        accumulated = Matrix((
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        ))
        for name, weight in influences:
            matrix = outer @ pose_matrices[name] @ old_rest[name].inverted() @ inner
            accumulated += matrix * (weight / total)
        transforms.append(accumulated)
    return transforms, unweighted


def _transform_geometry(obj, transforms, geometry_before):
    shape_keys = obj.data.shape_keys
    changed_keys = 0
    if shape_keys is None:
        for vertex, matrix in zip(obj.data.vertices, transforms):
            vertex.co = matrix @ vertex.co
    else:
        for key, old_key in zip(shape_keys.key_blocks, geometry_before["keys"]):
            changed = False
            for point, matrix, old_co in zip(key.data, transforms, old_key["coords"]):
                point.co = matrix @ point.co
                changed = changed or (point.co - Vector(old_co)).length > EPS
            changed_keys += int(changed)
    obj.data.update()
    return changed_keys


def _sync_custom_normals(obj, transforms, old_normals):
    if len(old_normals) != len(obj.data.loops):
        raise AposeBakeError("Source corner-normal count mismatch: " + obj.name)
    fallback_normals = None
    result = []
    for index, old_normal in enumerate(old_normals):
        if old_normal.length <= EPS:
            if fallback_normals is None:
                fallback_normals = {}
                for poly in obj.data.polygons:
                    pn = poly.normal.copy()
                    if pn.length <= EPS:
                        pn = Vector((0.0, 0.0, 1.0))
                    else:
                        pn.normalize()
                    for li in poly.loop_indices:
                        fallback_normals[li] = pn
            result.append(fallback_normals.get(index, Vector((0.0, 0.0, 1.0))))
            continue
        vertex_index = obj.data.loops[index].vertex_index
        normal = transforms[vertex_index].to_3x3().inverted_safe().transposed() @ old_normal
        if normal.length <= EPS:
            raise AposeBakeError("Degenerate transformed corner normal: " + obj.name)
        normal.normalize()
        result.append(normal)
    obj.data.normals_split_custom_set(result)
    obj.data.update()
    return len(result)


def _evaluated_world_positions(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        matrix = evaluated.matrix_world.copy()
        return tuple(matrix @ vertex.co for vertex in mesh.vertices)
    finally:
        evaluated.to_mesh_clear()


def _evaluation_error(before, after):
    if len(before) != len(after):
        return float("inf")
    return max((left - right).length for left, right in zip(before, after)) if before else 0.0


def _uv_weight_shape_checks(obj, before, geometry_after):
    if tuple(group.name for group in obj.vertex_groups) != before["groups"]:
        raise AposeBakeError("Vertex-group names changed: " + obj.name)
    weights = tuple(tuple(sorted((entry.group, float(entry.weight))
                                 for entry in vertex.groups))
                    for vertex in obj.data.vertices)
    if weights != before["weights"]:
        raise AposeBakeError("Vertex weights changed: " + obj.name)
    uvs = tuple((layer.name, tuple(tuple(float(value) for value in item.uv)
                                   for item in layer.data)) for layer in obj.data.uv_layers)
    if uvs != before["uvs"]:
        raise AposeBakeError("UV data changed: " + obj.name)
    if len(geometry_after["keys"]) != len(before["keys"]):
        raise AposeBakeError("Shape-key count changed: " + obj.name)
    for old, current in zip(before["keys"], geometry_after["keys"]):
        if old["name"] != current["name"] or old["relative_key"] != current["relative_key"]:
            raise AposeBakeError("Shape-key relationships changed: " + obj.name)
        if abs(old["value"] - current["value"]) > 1e-7:
            raise AposeBakeError("Shape-key value changed: " + obj.name)
    if obj.data.shape_keys and tuple(geometry_after["drivers"]) != before["drivers"]:
        raise AposeBakeError("Shape-key drivers changed: " + obj.name)


def bake(rig, meshes, degrees=45):
    """Bake both arms down by `degrees` and return a JSON-safe audit report."""
    if isinstance(degrees, bool):
        raise AposeBakeError("degrees must be a finite number")
    degrees = float(degrees)
    if not math.isfinite(degrees) or not 0.0 < degrees < 90.0:
        raise AposeBakeError("degrees must be finite and between 0 and 90")
    mesh_objects = _meshes(meshes)
    modifiers = _assert_ready(rig, mesh_objects)
    root_scale_before = list(rig.parent.scale) if rig.parent else None
    flags_before = _bone_flags(rig)
    old_rest = {bone.name: bone.matrix_local.copy() for bone in rig.data.bones}
    old_length = {bone.name: float(bone.length) for bone in rig.data.bones}
    geometry_before = {obj.name: _mesh_state(obj) for obj in mesh_objects}
    normal_before = {obj.name: tuple(corner.vector.copy() for corner in obj.data.corner_normals)
                     for obj in mesh_objects if modifiers[obj.name] is not None}
    static = [obj for obj in mesh_objects if modifiers[obj.name] is None]
    static_before = {obj.name: obj.matrix_world.copy() for obj in static}
    archived, archive_reason = _archive_body_proportions(rig)

    _rotate_arm(rig, "LeftArm", degrees)
    _rotate_arm(rig, "RightArm", -degrees)
    bpy.context.view_layer.update()
    posed, _ = _pose_snapshot(rig)
    static_posed = {obj.name: obj.matrix_world.copy() for obj in static}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    world_before = {obj.name: _evaluated_world_positions(obj, depsgraph)
                    for obj in mesh_objects}

    changed_meshes, changed_keys, normal_counts, unweighted = 0, 0, {}, 0
    for obj in mesh_objects:
        if modifiers[obj.name] is None:
            continue
        transforms, count = _weights_and_transforms(
            obj, rig, posed, old_rest)
        unweighted += count
        keys_changed = _transform_geometry(obj, transforms, geometry_before[obj.name])
        changed_keys += keys_changed
        changed_meshes += int(keys_changed > 0 or obj.data.shape_keys is None)
        normal_counts[obj.name] = _sync_custom_normals(obj, transforms, normal_before[obj.name])

    new_rest = {}
    new_tails = {}
    for bone in rig.data.bones:
        pose_matrix = posed[bone.name]
        location = pose_matrix.translation.copy()
        rotation = pose_matrix.to_quaternion()
        frame = Matrix.Translation(location) @ rotation.to_matrix().to_4x4()
        new_rest[bone.name] = frame
        new_tails[bone.name] = pose_matrix @ Vector((0.0, old_length[bone.name], 0.0))

    active = bpy.context.view_layer.objects.active
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    try:
        for name, frame in new_rest.items():
            edit_bone = rig.data.edit_bones[name]
            edit_bone.matrix = frame
            edit_bone.head = frame.translation
            edit_bone.tail = new_tails[name]
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
        if active and active.name in bpy.data.objects:
            bpy.context.view_layer.objects.active = active
    for pose_bone in rig.pose.bones:
        pose_bone.matrix_basis = IDENTITY.copy()
    bpy.context.view_layer.update()
    for obj in static:
        obj.matrix_world = static_posed[obj.name]
    bpy.context.view_layer.update()

    actual_new_rest = {bone.name: bone.matrix_local.copy() for bone in rig.data.bones}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    world_after = {obj.name: _evaluated_world_positions(obj, depsgraph)
                   for obj in mesh_objects}
    world_errors = {name: _evaluation_error(world_before[name], world_after[name])
                    for name in world_before}
    if max(world_errors.values()) > WORLD_TOLERANCE:
        raise AposeBakeError("World-space mesh changed beyond 1e-5 m")
    basis_errors = {bone.name: _matrix_error(bone.matrix_basis, IDENTITY)
                    for bone in rig.pose.bones}
    if max(basis_errors.values()) > 1e-6:
        raise AposeBakeError("Pose basis is not identity after bake")
    flags_after = _bone_flags(rig)
    if flags_before != flags_after:
        raise AposeBakeError("Bone parent or flags changed")
    if root_scale_before is not None and max(
            abs(left - right) for left, right in zip(root_scale_before, rig.parent.scale)) > EPS:
        raise AposeBakeError("Root scale changed")

    geometry_after = {obj.name: _mesh_state(obj) for obj in mesh_objects}
    for obj in mesh_objects:
        _uv_weight_shape_checks(obj, geometry_before[obj.name], geometry_after[obj.name])
    arm_records = {}
    for side, signed_degrees in (("Left", degrees), ("Right", -degrees)):
        name = side + "Arm"
        old_direction = old_rest[name].to_3x3() @ Vector((0.0, 1.0, 0.0))
        new_direction = actual_new_rest[name].to_3x3() @ Vector((0.0, 1.0, 0.0))
        old_world = rig.matrix_world.to_quaternion() @ old_direction
        new_world = rig.matrix_world.to_quaternion() @ new_direction
        arm_records[side] = {
            "bone": name,
            "world_axis": [0.0, 1.0, 0.0],
            "requested_degrees": signed_degrees,
            "old_world_direction": _vector_values(old_world.normalized()),
            "new_world_direction": _vector_values(new_world.normalized()),
            "old_to_new_degrees": math.degrees(old_world.angle(new_world)),
            "elbow_pose_basis_error": basis_errors[side + "Elbow"],
            "old_length": old_length[name],
            "new_length": float(rig.data.bones[name].length),
            "new_tail": _vector_values(rig.data.bones[name].tail_local),
            "old_rest": _matrix_rows(old_rest[name]),
            "new_rest": _matrix_rows(actual_new_rest[name]),
        }

    report = {
        "schema": SCHEMA,
        "status": "baked",
        "armature": rig.name,
        "bone_count": len(rig.data.bones),
        "mesh_count": len(mesh_objects),
        "changed_meshes": changed_meshes,
        "changed_shape_keys": changed_keys,
        "shape_key_count": sum(len(geometry_after[obj.name]["keys"]) for obj in mesh_objects),
        "shape_driver_count": sum(len(geometry_after[obj.name]["drivers"]) for obj in mesh_objects),
        "normal_meshes_processed": len(normal_counts),
        "normal_loops_processed": sum(normal_counts.values()),
        "normal_loop_counts": normal_counts,
        "unweighted_vertices_unchanged": unweighted,
        "static_meshes": [obj.name for obj in static],
        "static_world_before": {name: _matrix_rows(matrix) for name, matrix in static_before.items()},
        "static_world_posed": {name: _matrix_rows(matrix) for name, matrix in static_posed.items()},
        "max_world_mesh_error_m": max(world_errors.values()),
        "world_mesh_errors_m": world_errors,
        "max_pose_basis_error": max(basis_errors.values()),
        "root_scale_before": root_scale_before,
        "root_scale_after": list(rig.parent.scale) if rig.parent else None,
        "arms": arm_records,
        "old_rest": {name: _matrix_rows(matrix) for name, matrix in old_rest.items()},
        "new_rest": {name: _matrix_rows(matrix) for name, matrix in actual_new_rest.items()},
        "body_proportions_archive": {
            "owners": archived,
            "reason": archive_reason,
            "active_baseline_markers_removed": True,
            "baked_flag": True,
            "baked_flag_owners": ["SCSP_Character", rig.name],
            "historical_json_preserved": True,
        },
        "checks": {
            "shape_keys": True,
            "shape_driver_paths": True,
            "uv_layers": True,
            "vertex_weights": True,
            "bone_parent_and_flags": True,
            "pose_basis_identity": True,
            "world_mesh_tolerance_m": WORLD_TOLERANCE,
            "custom_normals_transformed": True,
        },
    }
    compact = {
        "schema": SCHEMA,
        "degrees": degrees,
        "maximum_world_error_m": report["max_world_mesh_error_m"],
        "normal_meshes_processed": report["normal_meshes_processed"],
        "normal_loops_processed": report["normal_loops_processed"],
        "changed_meshes": changed_meshes,
        "changed_shape_keys": changed_keys,
    }
    rig[BAKE_KEY] = json.dumps(compact, ensure_ascii=False, sort_keys=True)
    bpy.context.view_layer.update()
    return report


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--degrees", type=float, default=45.0)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    source = os.path.abspath(bpy.data.filepath)
    output = os.path.abspath(args.save)
    if source == output:
        raise AposeBakeError("Refusing to save over the source blend")
    rig = bpy.data.objects.get("SCSP_Skeleton")
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    report = bake(rig, meshes, args.degrees)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=output, check_existing=False)
    report["source_blend"] = source
    report["output_blend"] = output
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    print("APOSE_REPORT_BEGIN")
    print(json.dumps({key: report[key] for key in (
        "status", "bone_count", "mesh_count", "changed_meshes", "changed_shape_keys",
        "shape_key_count", "shape_driver_count", "normal_meshes_processed",
        "normal_loops_processed", "max_world_mesh_error_m", "source_blend",
        "output_blend")}, ensure_ascii=False, sort_keys=True))
    print("APOSE_REPORT_END")


if __name__ == "__main__":
    _main()