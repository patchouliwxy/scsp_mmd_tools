"""Orient one neutral, assembled SCSP armature along its joint topology.

The source rest frames use the imported vertical Y convention.  This module
changes only rest-frame orientation and length; bone heads, parent links,
names, skin bindings, and static attachment bones remain intact.  It does not
apply body proportions or create MMD axes.
"""
import json
import math
from collections import defaultdict

import bpy
from mathutils import Matrix, Vector

SCHEMA = 1
REST_MAP_KEY = "scsp_bone_orientation_rest_map"
STATIC_MARKER = "__static_"
IDENTITY = Matrix.Identity(4)
EPS = 1e-8
# Attached FBX part roots can differ from their parent by sub-micrometre noise.
# Such offsets are not joint directions and must not shorten a display bone.
COINCIDENT_JOINT_TOLERANCE = 1e-5


class OrientationError(ValueError):
    """The supplied rig cannot be oriented without breaking the contract."""


def _animated(owner):
    data = getattr(owner, "animation_data", None)
    return bool(data and (data.action or any(track.strips for track in data.nla_tracks)
                          or data.drivers))


def _assert_neutral(rig):
    if not isinstance(rig, bpy.types.Object) or rig.type != "ARMATURE":
        raise OrientationError("rig must be one ARMATURE object")
    if rig.mode != "OBJECT":
        raise OrientationError("orient() must be called in Object Mode")
    if REST_MAP_KEY in rig:
        raise OrientationError("Armature has already been oriented: " + rig.name)
    if "scsp_body_proportions_baseline" in rig:
        raise OrientationError("Bone orientation must run before body proportions")
    if _animated(rig) or _animated(rig.data):
        raise OrientationError("Animated armature/data cannot be reoriented without action remapping")
    for pose_bone in rig.pose.bones:
        if pose_bone.constraints:
            raise OrientationError("Constrained pose bone: " + pose_bone.name)
        error = max(abs(pose_bone.matrix_basis[r][c] - IDENTITY[r][c])
                    for r in range(4) for c in range(4))
        if error > 1e-5:
            raise OrientationError("Non-neutral pose bone: " + pose_bone.name)
    connected = [bone.name for bone in rig.data.bones if bone.use_connect]
    if connected:
        raise OrientationError("Connected bones would force heads: " + ", ".join(connected))


def _as_list(matrix):
    return [[float(matrix[r][c]) for c in range(4)] for r in range(4)]


def _basis_list(matrix):
    rotation = matrix.to_3x3()
    return [[float(rotation[r][c]) for c in range(3)] for r in range(3)]


def _old_to_new(old_matrix, new_matrix):
    return old_matrix.to_3x3().inverted() @ new_matrix.to_3x3()


def _rest_rotation(bone):
    rotation = bone.matrix_local.to_3x3()
    loc, quat, scale = bone.matrix_local.decompose()
    rebuilt = Matrix.LocRotScale(loc, quat, scale).to_3x3()
    error = max(abs(rotation[r][c] - rebuilt[r][c]) for r in range(3) for c in range(3))
    if min(scale) <= 0 or max(abs(value - 1.0) for value in scale) > 5e-4 or error > 5e-4:
        raise OrientationError("Non-rigid source rest frame: " + bone.name)
    return quat.to_matrix()


def _child_direction(heads, bone_name, child_name):
    vector = heads[child_name] - heads[bone_name]
    length = vector.length
    if length <= EPS:
        raise OrientationError(f"Zero-length intended direction: {bone_name}->{child_name}")
    return vector / length, length


def _frame(head, direction, up_hint, forward_hint):
    y_axis = Vector(direction).normalized()
    z_axis = up_hint - y_axis * y_axis.dot(up_hint)
    if z_axis.length <= 1e-6:
        z_axis = forward_hint - y_axis * y_axis.dot(forward_hint)
    if z_axis.length <= 1e-6:
        alternate = Vector((1.0, 0.0, 0.0))
        if abs(y_axis.dot(alternate)) > 0.999:
            alternate = Vector((0.0, 1.0, 0.0))
        z_axis = alternate - y_axis * y_axis.dot(alternate)
    z_axis.normalize()
    x_axis = y_axis.cross(z_axis).normalized()
    z_axis = x_axis.cross(y_axis).normalized()
    frame = Matrix.Identity(4)
    for row in range(3):
        frame[row][0] = x_axis[row]
        frame[row][1] = y_axis[row]
        frame[row][2] = z_axis[row]
        frame[row][3] = head[row]
    return frame


def _topological(names, parents, children):
    order, remaining = [], set(names)
    while remaining:
        ready = sorted(name for name in remaining
                       if name not in parents or parents[name] is None or parents[name] not in remaining)
        if not ready:
            raise OrientationError("Bone hierarchy contains a cycle")
        order.extend(ready)
        remaining.difference_update(ready)
    return order


def _weighted_vertex_centroids(rig, heads):
    """Compute weighted vertex-offset centroid (in armature space) and max span per bone."""
    weighted_sums = {name: Vector((0.0, 0.0, 0.0)) for name in heads}
    weight_totals = {name: 0.0 for name in heads}
    max_spans = {name: 0.0 for name in heads}
    inv_rig = rig.matrix_world.inverted()
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not any(
            m.type == "ARMATURE" and m.object == rig for m in obj.modifiers
        ):
            continue
        to_armature = inv_rig @ obj.matrix_world
        group_names = {
            group.index: group.name
            for group in obj.vertex_groups
            if group.name in heads
        }
        if not group_names:
            continue
        for vertex in obj.data.vertices:
            local_co = None
            for entry in vertex.groups:
                name = group_names.get(entry.group)
                if name is None or entry.weight <= 0.05:
                    continue
                if local_co is None:
                    local_co = to_armature @ vertex.co
                offset = local_co - heads[name]
                weighted_sums[name] += offset * entry.weight
                weight_totals[name] += entry.weight
                dist = offset.length
                if dist > max_spans[name]:
                    max_spans[name] = dist
    centroids = {}
    for name, total in weight_totals.items():
        if total > 1e-6:
            centroids[name] = (weighted_sums[name] / total, max_spans[name])
    return centroids


def _skin_snapshot(rig):
    result = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not any(m.type == "ARMATURE" and m.object == rig
                                         for m in obj.modifiers):
            continue
        groups = tuple(group.name for group in obj.vertex_groups)
        weights = tuple(tuple((entry.group, float(entry.weight)) for entry in vertex.groups)
                        for vertex in obj.data.vertices)
        result.append((obj, groups, weights))
    return result


def _unique_eye_bone(rig, source_bone):
    matches = [bone for bone in rig.data.bones
               if bone.get("scsp_source_bone") == source_bone]
    if len(matches) > 1:
        raise OrientationError("Ambiguous eye-bone metadata for " + source_bone + ": "
                               + ", ".join(bone.name for bone in matches))
    if len(matches) == 1:
        return matches[0]
    matches = [bone for bone in rig.data.bones if bone.name == source_bone]
    if len(matches) != 1:
        raise OrientationError("Missing or ambiguous exact fallback eye bone: " + source_bone)
    return matches[0]


def _unique_eye_sheet(rig, source_object):
    matches = [obj for obj in bpy.data.objects
               if obj.type == "MESH" and obj.get("scsp_source_object") == source_object
               and any(mod.type == "ARMATURE" and mod.object == rig
                       for mod in obj.modifiers)]
    if len(matches) != 1:
        raise OrientationError("Missing or ambiguous eye sheet for " + source_object + " on "
                               + rig.name + ": " + ", ".join(obj.name for obj in matches))
    return matches[0]


def _chains(bones=None):
    chains = {
        "spine": ["root", "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head"],
    }
    for side in ("Left", "Right"):
        chains[side + "Arm"] = [side + "Shoulder", side + "Arm", side + "ForeArm",
                                side + "Hand", side + "HandMiddle1"]
        leg = [side + "UpLeg", side + "Leg", side + "Foot", side + "Toe"]
        if bones is None or (side + "Toe_null") in bones:
            leg.append(side + "Toe_null")
        chains[side + "Leg"] = leg
        for finger in ("Index", "Thumb", "Middle", "Ring", "Pinky"):
            seq = [f"{side}Hand{finger}{index}" for index in range(1, 4)]
            tip = f"{side}Hand{finger}4"
            if bones is None or tip in bones:
                seq.append(tip)
            chains[f"{side}{finger}"] = seq
        chains[side + "RingPalm"] = [side + "Ring", side + "HandRing1"]
    return chains


def _assists():
    assists = {}
    for side in ("Left", "Right"):
        shoulder = (side + "Shoulder", side + "Arm")
        arm = (side + "Arm", side + "ForeArm")
        forearm = (side + "ForeArm", side + "Hand")
        assists[side + "Shoulder_rot"] = shoulder
        for index in (1, 2, 3):
            assists[f"{side}Arm{index}_rot"] = arm
        assists[side + "Elbow"] = forearm
        assists[side + "ForeArm_rot"] = forearm
    return assists


def _angle_degrees(a, b):
    return math.degrees(a.normalized().angle(b.normalized()))


def orient(rig):
    """Orient the neutral assembled rig and return a JSON-safe audit report."""
    _assert_neutral(rig)
    bones = {bone.name: bone for bone in rig.data.bones}
    names = tuple(bones)
    heads = {name: bones[name].head_local.copy() for name in names}
    parents = {name: bones[name].parent.name if bones[name].parent else None for name in names}
    old_matrices = {name: bones[name].matrix_local.copy() for name in names}
    old_lengths = {name: float(bones[name].length) for name in names}
    old_y = {name: (_rest_rotation(bones[name]) @ Vector((0.0, 1.0, 0.0))).normalized()
             for name in names}
    children = defaultdict(list)
    for name in names:
        if parents[name]:
            children[parents[name]].append(name)
    for values in children.values():
        values.sort()

    world_to_armature = rig.matrix_world.to_quaternion().inverted()
    world_front = Vector((0.0, -1.0, 0.0))
    directions, lengths, targets, origins = {}, {}, {}, {}
    main_records = []
    chain_map = _chains(bones)
    for chain_name, chain in chain_map.items():
        missing = [name for name in chain if name not in bones]
        if missing:
            raise OrientationError(chain_name + " is missing: " + ", ".join(missing))
        for parent_name, child_name in zip(chain, chain[1:]):
            directions[parent_name], lengths[parent_name] = _child_direction(
                heads, parent_name, child_name)
            targets[parent_name], origins[parent_name] = child_name, "main"
            main_records.append((chain_name, parent_name, child_name))

    for side in ("Left", "Right"):
        toe_name = side + "Toe"
        if toe_name not in directions and (side + "Toe_null") not in bones:
            has_child = any(
                STATIC_MARKER not in child_name
                and (heads[child_name] - heads[toe_name]).length > COINCIDENT_JOINT_TOLERANCE
                for child_name in children[toe_name]
            )
            if not has_child:
                directions[toe_name] = (world_to_armature @ world_front).normalized()
                lengths[toe_name] = old_lengths[toe_name]
                targets[toe_name], origins[toe_name] = None, "toe_front"

    eye_records = []
    for side in ("L", "R"):
        source_bone = f"EYE_{side}"
        source_object = f"eyesclera_{side}"
        eye_bone = _unique_eye_bone(rig, source_bone)
        bone_name = eye_bone.name
        sheet = _unique_eye_sheet(rig, source_object)
        sheet_center_world = sum((sheet.matrix_world @ Vector(corner)
                                  for corner in sheet.bound_box), Vector()) / 8.0
        sheet_center = rig.matrix_world.inverted() @ sheet_center_world
        direction = (world_to_armature @ world_front).normalized()
        directions[bone_name] = direction
        lengths[bone_name] = old_lengths[bone_name]
        targets[bone_name], origins[bone_name] = None, "eye_front"
        eye_records.append((bone_name, source_bone, sheet.name,
                            sheet_center.copy(), direction.copy()))

    assist_records = []
    for bone_name, (parent_name, child_name) in _assists().items():
        if bone_name not in bones:
            raise OrientationError("Expected assistance bone is missing: " + bone_name)
        directions[bone_name], _ = _child_direction(heads, parent_name, child_name)
        lengths[bone_name] = old_lengths[bone_name]
        targets[bone_name], origins[bone_name] = child_name, "assist"
        assist_records.append((bone_name, f"{parent_name}->{child_name}"))

    centroids = _weighted_vertex_centroids(rig, heads)
    order = _topological(names, parents, children)
    coincident_children = []
    hub_names = set()
    for name in order:
        if name in directions:
            continue
        if STATIC_MARKER in name:
            origins[name] = "static"
            continue
        candidates = []
        for child_name in children[name]:
            if STATIC_MARKER in child_name:
                continue
            vector = heads[child_name] - heads[name]
            if vector.length > COINCIDENT_JOINT_TOLERANCE:
                candidates.append((float(vector.length), child_name, vector / vector.length))
            else:
                coincident_children.append({'bone': name, 'child': child_name,
                    'distance_m': float(vector.length)})
        if len(candidates) == 1:
            distance, child_name, direction = candidates[0]
            directions[name], lengths[name] = direction, distance
            targets[name], origins[name] = child_name, "child"
        elif len(candidates) >= 2:
            # Multi-child hub bones (e.g. p02__HAIR, p03__acce) must not tilt toward one arbitrary child.
            hub_names.add(name)
            mean_vec = sum((item[2] for item in candidates), Vector((0.0, 0.0, 0.0)))
            if parents[name] and parents[name] in directions:
                directions[name] = directions[parents[name]].copy()
            elif mean_vec.length > EPS:
                directions[name] = mean_vec.normalized()
            else:
                raise OrientationError("Cannot derive hub direction: " + name)
            lengths[name] = max(min(old_lengths[name], 0.04), 1e-4)
            targets[name], origins[name] = None, "hub"
        elif parents[name] and parents[name] in directions:
            parent_name = parents[name]
            centroid_info = centroids.get(name)
            from_hub_or_hair = (parent_name in hub_names) or ("hair" in name.lower())
            if from_hub_or_hair and centroid_info is not None and centroid_info[0].length > 3e-3:
                offset_vec, max_span = centroid_info
                directions[name] = offset_vec.normalized()
                # Scale display length to fit the actual weighted hair/part span cleanly.
                lengths[name] = max(min(old_lengths[name], offset_vec.length * 1.25, max_span), 0.012)
                targets[name], origins[name] = None, "mesh"
            else:
                directions[name] = directions[parent_name].copy()
                if parent_name in hub_names:
                    cap = centroid_info[1] if centroid_info is not None else 0.03
                    lengths[name] = max(min(old_lengths[name], max(cap, 0.015)), 0.012)
                else:
                    lengths[name] = old_lengths[name]
                targets[name], origins[name] = None, "parent"
        else:
            raise OrientationError("Cannot derive a nonzero branch direction: " + name)

    up_hint = (world_to_armature @ Vector((0.0, 0.0, 1.0))).normalized()
    forward_hint = (world_to_armature @ Vector((0.0, -1.0, 0.0))).normalized()
    new_matrices = {
        name: _frame(heads[name], directions[name], up_hint, forward_hint)
        for name in directions
    }

    mapping_bones = {}
    for name in names:
        if name not in new_matrices:
            continue
        old_matrix = old_matrices[name]
        new_matrix = new_matrices[name]
        mapping_bones[name] = {
            "old_rest_matrix": _as_list(old_matrix),
            "new_rest_matrix": _as_list(new_matrix),
            "old_to_new_basis": _basis_list(_old_to_new(old_matrix, new_matrix)),
            "old_length": old_lengths[name],
            "new_length": float(lengths[name]),
            "direction_origin": origins[name],
            "intended_child": targets[name],
        }
    mapping = {
        "schema": SCHEMA,
        "armature": rig.name,
        "frame": "armature-space rest matrices; rig.matrix_world is unchanged",
        "pose_equivalence": "P_new = P_old * R_old^-1 * R_new",
        "bones": mapping_bones,
    }
    rig[REST_MAP_KEY] = json.dumps(mapping, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":"))

    skins_before = _skin_snapshot(rig)
    selected_before = tuple(bpy.context.selected_objects)
    active_before = bpy.context.view_layer.objects.active
    try:
        bpy.ops.object.select_all(action="DESELECT")
        rig.select_set(True)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode="EDIT")
        for name in order:
            if name not in new_matrices:
                continue
            edit_bone = rig.data.edit_bones[name]
            edit_bone.use_connect = False
            edit_bone.length = max(float(lengths[name]), 1e-5)
            edit_bone.matrix = new_matrices[name]
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        if rig.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        if REST_MAP_KEY in rig:
            del rig[REST_MAP_KEY]
        raise
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_before:
            if obj.name in bpy.data.objects:
                obj.select_set(True)
        if active_before and active_before.name in bpy.data.objects:
            bpy.context.view_layer.objects.active = active_before
        bpy.context.view_layer.update()

    updated = {bone.name: bone for bone in rig.data.bones}
    head_error = max((updated[name].head_local - heads[name]).length for name in names)
    parent_errors = [name for name in names
                     if (updated[name].parent.name if updated[name].parent else None) != parents[name]]
    matrix_error = max(max(abs(updated[name].matrix_local[r][c] - new_matrices[name][r][c])
                           for r in range(4) for c in range(4))
                       for name in new_matrices)
    static_error = max((max(abs(updated[name].matrix_local[r][c] - old_matrices[name][r][c])
                            for r in range(4) for c in range(4))
                        for name in names if STATIC_MARKER in name), default=0.0)
    skins_after = _skin_snapshot(rig)
    if head_error > 5e-5 or parent_errors or matrix_error > 5e-5 or static_error > 5e-5:
        raise AssertionError(f"Orientation invariant failed: head={head_error} parents={parent_errors} matrix={matrix_error} static={static_error}")
    if [(obj.name, groups, weights) for obj, groups, weights in skins_before] != [
            (obj.name, groups, weights) for obj, groups, weights in skins_after]:
        raise AssertionError("Skin vertex groups or weights changed")

    main_report = []
    for chain_name, parent_name, child_name in main_records:
        main_report.append({
            "chain": chain_name,
            "bone": parent_name,
            "intended_child": child_name,
            "before_angle_deg": _angle_degrees(old_y[parent_name], directions[parent_name]),
            "after_angle_deg": _angle_degrees(
                new_matrices[parent_name].to_3x3() @ Vector((0.0, 1.0, 0.0)),
                directions[parent_name]),
        })
    eye_report = []
    for bone_name, source_bone, sheet_name, sheet_center, direction in eye_records:
        eye_report.append({
            "bone": bone_name,
            "source_bone": source_bone,
            "source_sheet": sheet_name,
            "bone_lookup": "unique scsp_source_bone metadata; exact EYE name fallback",
            "sheet_lookup": "unique scsp_source_object metadata with armature modifier targeting rig",
            "sheet_center_armature": [float(value) for value in sheet_center],
            "head_offset_from_sheet_center_m": [
                float(value) for value in heads[bone_name] - sheet_center],
            "direction_armature": [float(value) for value in direction],
            "direction_world": [float(value) for value in
                                (rig.matrix_world.to_quaternion() @ direction).normalized()],
            "policy": "explicit face-front -Y; original head and length retained",
            "before_angle_deg": _angle_degrees(old_y[bone_name], direction),
            "after_angle_deg": _angle_degrees(
                new_matrices[bone_name].to_3x3() @ Vector((0.0, 1.0, 0.0)), direction),
        })

    assist_report = []
    for bone_name, segment in assist_records:
        assist_report.append({
            "bone": bone_name,
            "segment": segment,
            "before_angle_deg": _angle_degrees(old_y[bone_name], directions[bone_name]),
            "after_angle_deg": _angle_degrees(
                new_matrices[bone_name].to_3x3() @ Vector((0.0, 1.0, 0.0)),
                directions[bone_name]),
        })

    origin_counts = {label: sum(value == label for value in origins.values())
                     for label in ("main", "assist", "eye_front", "toe_front", "child", "hub", "mesh", "parent", "source_y", "static")}
    prefix_counts = {prefix: sum(name.startswith(prefix) and name not in new_matrices
                                 for name in names)
                     for prefix in ("p00__", "p02__", "p03__")}
    skinned_meshes = len(skins_after)
    vertex_groups = sum(len(groups) for _, groups, _ in skins_after)
    weighted_records = sum(len(weights) for _, _, weights in skins_after)
    return {
        "schema": SCHEMA,
        "status": "oriented",
        "armature": rig.name,
        "policy": ("joint-to-child rest direction; deterministic world-up roll; "
                   "source identities, heads, hierarchy, and skin bindings retained"),
        "counts": {
            "bones": len(names),
            "oriented": len(new_matrices),
            "main_edges": len(main_records),
            "assist_bones": len(assist_records),
            "eye_bones": len(eye_records),
            "static_unchanged": origin_counts["static"],
            "source_prefix_static_unchanged": prefix_counts,
            **origin_counts,
        },
        "main_angles": main_report,
        "eye_bones": eye_report,
        "assist_angles": assist_report,
        "roll": {
            "reference": "physical world +Z projected normal to bone Y; world -Y fallback",
            "up_hint_armature": [float(value) for value in up_hint],
            "forward_hint_armature": [float(value) for value in forward_hint],
            "uses_recalculate_roll": False,
        },
        "coincident_child_links_ignored": coincident_children,
        "coincident_joint_tolerance_m": COINCIDENT_JOINT_TOLERANCE,
        "rest_remap": {
            "rig_property": REST_MAP_KEY,
            "bones": len(mapping_bones),
            "pose_equivalence": mapping["pose_equivalence"],
            "stored": True,
        },
        "invariants": {
            "bone_names_unchanged": set(updated) == set(names),
            "max_head_error_m": float(head_error),
            "parent_map_unchanged": not parent_errors,
            "max_new_rest_matrix_error": float(matrix_error),
            "static_rest_matrix_max_error": float(static_error),
            "skinned_meshes": skinned_meshes,
            "vertex_groups": vertex_groups,
            "weighted_vertex_records": weighted_records,
            "vertex_groups_and_weights_exact": True,
            "object_transforms_changed": False,
        },
    }
