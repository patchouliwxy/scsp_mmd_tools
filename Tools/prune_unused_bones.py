"""Prune runtime-unused bones from the SCSP working rig.

``prune(rig, meshes)`` mutates only ``SCSP_Skeleton`` and returns a JSON-safe
report. Source/reference armatures are never edited.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

import bpy


SCHEMA = 1
TARGET_OBJECT = "SCSP_Skeleton"
SOURCE_IDENTITY_PROPERTIES = (
    "scsp_source_bone",
    "scsp_mmd_source_identity",
    "scsp_mmd_source_name",
)


def _identity_value(owner: Any) -> str | None:
    for property_name in SOURCE_IDENTITY_PROPERTIES:
        try:
            value = owner.get(property_name)
        except (AttributeError, ReferenceError, TypeError):
            continue
        if value:
            return str(value)
    return None


def _source_identity(rig: Any, bone: Any) -> str:
    identity = _identity_value(bone)
    if identity:
        return identity
    pose_bone = rig.pose.bones.get(bone.name)
    if pose_bone is not None:
        identity = _identity_value(pose_bone)
        if identity:
            return identity
    return bone.name


def _standard_mmd_names() -> set[str]:
    names = {
        "全ての親",
        "センター",
        "グルーブ",
        "両目",
        "下半身",
        "上半身",
        "上半身2",
        "上半身3",
        "首",
        "頭",
        "左目",
        "右目",
        "左肩",
        "右肩",
        "左腕",
        "右腕",
        "左ひじ",
        "右ひじ",
        "左手首",
        "右手首",
        "左足",
        "右足",
        "左ひざ",
        "右ひざ",
        "左足首",
        "右足首",
        "左つま先",
        "右つま先",
        "左足ＩＫ",
        "右足ＩＫ",
        "左つま先ＩＫ",
        "右つま先ＩＫ",
        "左足D",
        "右足D",
        "左ひざD",
        "右ひざD",
        "左足首D",
        "右足首D",
        "左足先EX",
        "右足先EX",
        "左腕捩",
        "右腕捩",
        "左手捩",
        "右手捩",
        "左足捩",
        "右足捩",
        "上半身捩",
        "下半身捩",
        "root",
        "Hips",
        "Spine",
        "Spine1",
        "Spine2",
        "Neck",
        "Head",
        "p00__EYE_L",
        "p00__EYE_R",
        "LeftEye",
        "RightEye",
        "LeftShoulder",
        "RightShoulder",
        "LeftArm",
        "RightArm",
        "LeftElbow",
        "RightElbow",
        "LeftHand",
        "RightHand",
        "LeftUpLeg",
        "RightUpLeg",
        "LeftLeg",
        "RightLeg",
        "LeftFoot",
        "RightFoot",
        "LeftToe",
        "LeftToeBase",
        "RightToe",
        "RightToeBase",
    }
    sides = (("Left", "左"), ("Right", "右"))
    fingers = (
        ("Index", "人指"),
        ("Middle", "中指"),
        ("Ring", "薬指"),
        ("Pinky", "小指"),
    )
    for english_side, japanese_side in sides:
        names.add(f"左親指０" if english_side == "Left" else "右親指０")
        for number in ("1", "2", "3"):
            names.add(f"{english_side}HandThumb{number}")
            names.add(f"{japanese_side}親指{('０','１','２')[int(number) - 1]}")
        for english_finger, japanese_finger in fingers:
            for number, japanese_number in zip(("1", "2", "3"), ("１", "２", "３")):
                names.add(f"{english_side}Hand{english_finger}{number}")
                names.add(f"{japanese_side}{japanese_finger}{japanese_number}")
    return names


STANDARD_MMD_NAMES = frozenset(_standard_mmd_names())


def _mesh_list(meshes: Iterable[Any]) -> list[Any]:
    result = []
    seen = set()
    for mesh in meshes:
        if not isinstance(mesh, bpy.types.Object) or mesh.type != "MESH":
            raise TypeError("meshes must contain Blender mesh objects")
        if mesh.name not in seen:
            result.append(mesh)
            seen.add(mesh.name)
    return sorted(result, key=lambda item: item.name)


def _add_reference(
    references: dict[str, list[dict[str, Any]]],
    name: str,
    kind: str,
    **details: Any,
) -> None:
    record = {"kind": kind, **details}
    if name and record not in references[name]:
        references[name].append(record)


def _positive_weight_references(
    meshes: list[Any],
    bone_names: set[str],
) -> dict[str, list[dict[str, Any]]]:
    result = defaultdict(list)
    for mesh in meshes:
        names_by_index = {group.index: group.name for group in mesh.vertex_groups}
        stats: dict[str, dict[str, float | int]] = {}
        for vertex in mesh.data.vertices:
            for membership in vertex.groups:
                if membership.weight <= 0.0:
                    continue
                group_name = names_by_index.get(membership.group)
                if group_name not in bone_names:
                    continue
                item = stats.setdefault(group_name, {"vertices": 0, "max_weight": 0.0})
                item["vertices"] += 1
                item["max_weight"] = max(float(item["max_weight"]), float(membership.weight))
        for bone_name, item in sorted(stats.items()):
            _add_reference(
                result,
                bone_name,
                "positive_weight",
                mesh=mesh.name,
                vertices=int(item["vertices"]),
                max_weight=float(item["max_weight"]),
            )
    return result


def _constraint_bone_targets(constraint: Any, rig: Any, bone_names: set[str]):
    for target_attribute, subtarget_attribute in (
        ("target", "subtarget"),
        ("pole_target", "pole_subtarget"),
    ):
        try:
            target = getattr(constraint, target_attribute, None)
            subtarget = getattr(constraint, subtarget_attribute, "")
        except ReferenceError:
            continue
        if target == rig and subtarget in bone_names:
            yield subtarget, target_attribute, subtarget_attribute
    try:
        target = getattr(constraint, "target", None)
        target_bone = getattr(constraint, "target_bone", "")
    except ReferenceError:
        return
    if target == rig and target_bone in bone_names:
        yield target_bone, "target", "target_bone"


def _constraint_references(
    rig: Any,
    bone_names: set[str],
) -> dict[str, list[dict[str, Any]]]:
    result = defaultdict(list)
    for obj in bpy.data.objects:
        for constraint in obj.constraints:
            for bone_name, target_attribute, subtarget_attribute in (
                _constraint_bone_targets(constraint, rig, bone_names)
            ):
                _add_reference(
                    result,
                    bone_name,
                    "constraint_target",
                    owner_id=obj.name,
                    owner_bone=None,
                    constraint=constraint.name,
                    target_attribute=target_attribute,
                    subtarget_attribute=subtarget_attribute,
                )
        if obj.type != "ARMATURE":
            continue
        for pose_bone in obj.pose.bones:
            for constraint in pose_bone.constraints:
                if obj == rig:
                    _add_reference(
                        result,
                        pose_bone.name,
                        "constrained_owner",
                        constraint=constraint.name,
                    )
                for bone_name, target_attribute, subtarget_attribute in (
                    _constraint_bone_targets(constraint, rig, bone_names)
                ):
                    _add_reference(
                        result,
                        bone_name,
                        "constraint_target",
                        owner_id=obj.name,
                        owner_bone=pose_bone.name,
                        constraint=constraint.name,
                        target_attribute=target_attribute,
                        subtarget_attribute=subtarget_attribute,
                    )
    return result


_BONE_PATH = re.compile(
    r"""bones\[(?:"((?:\\.|[^"])*)"|'((?:\\.|[^'])*)')\]"""
)


def _bone_paths(data_path: str) -> set[str]:
    names = set()
    for match in _BONE_PATH.finditer(data_path or ""):
        value = match.group(1) if match.group(1) is not None else match.group(2)
        names.add(value.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\"))
    return names


def _iter_animation_ids():
    for property_definition in bpy.data.bl_rna.properties:
        if property_definition.type != "COLLECTION":
            continue
        try:
            collection = getattr(bpy.data, property_definition.identifier)
        except (AttributeError, ReferenceError):
            continue
        if not hasattr(collection, "__iter__"):
            continue
        for item in collection:
            if hasattr(item, "animation_data"):
                yield item


def _driver_references(
    rig: Any,
    bone_names: set[str],
) -> dict[str, list[dict[str, Any]]]:
    result = defaultdict(list)
    for owner in _iter_animation_ids():
        animation_data = getattr(owner, "animation_data", None)
        if animation_data is None:
            continue
        for driver in animation_data.drivers:
            data_path = str(getattr(driver, "data_path", ""))
            if owner in (rig,rig.data):
                for bone_name in _bone_paths(data_path):
                    if bone_name in bone_names:
                        _add_reference(
                            result,
                            bone_name,
                            "driver_owner_path",
                            owner_id=rig.name,
                            owner_type="ARMATURE",
                            data_path=data_path,
                        )
            for variable in driver.driver.variables:
                for target in variable.targets:
                    try:
                        target_id = target.id
                        target_path = str(getattr(target, "data_path", ""))
                        bone_target = str(getattr(target, "bone_target", ""))
                    except ReferenceError:
                        continue
                    if target_id in (rig, rig.data):
                        for bone_name in _bone_paths(target_path):
                            if bone_name in bone_names:
                                _add_reference(
                                    result,
                                    bone_name,
                                    "driver_target",
                                    variable=variable.name,
                                    target_id=target_id.name,
                                    data_path=target_path,
                                )
                    if target_id == rig and bone_target in bone_names:
                        _add_reference(
                            result,
                            bone_name,
                            "driver_transform_bone_target",
                            variable=variable.name,
                            bone_target=bone_target,
                        )
    return result


def _action_curve_paths(action: Any):
    seen = set()
    try:
        curves = action.fcurves
    except (AttributeError, RuntimeError):
        curves = ()
    for curve in curves:
        path = str(curve.data_path)
        if path not in seen:
            seen.add(path)
            yield path
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for channelbag in getattr(strip, "channelbags", ()):
                for curve in getattr(channelbag, "fcurves", ()):
                    path = str(curve.data_path)
                    if path not in seen:
                        seen.add(path)
                        yield path


def _action_references(
    rig: Any,
    bone_names: set[str],
) -> dict[str, list[dict[str, Any]]]:
    result = defaultdict(list)
    for action in bpy.data.actions:
        for data_path in _action_curve_paths(action):
            for bone_name in _bone_paths(data_path):
                if bone_name in bone_names:
                    _add_reference(
                        result,
                        bone_name,
                        "action_pose_path",
                        action=action.name,
                        data_path=data_path,
                    )
    for owner in bpy.data.objects:
        animation_data = getattr(owner, "animation_data", None)
        if animation_data is None:
            continue
        for track in animation_data.nla_tracks:
            for strip in track.strips:
                action = getattr(strip, "action", None)
                if action is None:
                    continue
                for data_path in _action_curve_paths(action):
                    for bone_name in _bone_paths(data_path):
                        if bone_name in bone_names:
                            _add_reference(
                                result,
                                bone_name,
                                "nla_pose_path",
                                owner_id=owner.name,
                                track=track.name,
                                strip=strip.name,
                                action=action.name,
                                data_path=data_path,
                            )
    return result


def _object_parent_references(
    rig: Any,
    bone_names: set[str],
) -> dict[str, list[dict[str, Any]]]:
    result = defaultdict(list)
    for obj in bpy.data.objects:
        if obj.parent != rig or obj.parent_type != "BONE":
            continue
        if obj.parent_bone in bone_names:
            _add_reference(
                result,
                obj.parent_bone,
                "object_bone_parent",
                object=obj.name,
            )
    return result


def _merge_references(
    rig: Any,
    bone_names: set[str],
    meshes: list[Any],
) -> dict[str, list[dict[str, Any]]]:
    references: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sources = (
        _positive_weight_references(meshes, bone_names),
        _object_parent_references(rig, bone_names),
        _constraint_references(rig, bone_names),
        _driver_references(rig, bone_names),
        _action_references(rig, bone_names),
    )
    for source in sources:
        for bone_name, reasons in source.items():
            for reason in reasons:
                _add_reference(references, bone_name, **reason)
    for bone_name in bone_names:
        if bone_name in STANDARD_MMD_NAMES:
            _add_reference(
                references,
                bone_name,
                "standard_mmd_role",
                role_name=bone_name,
            )
    return references


def _close_parents(
    bones: dict[str, Any],
    references: dict[str, list[dict[str, Any]]],
) -> set[str]:
    retained = set(references)
    pending = list(sorted(retained))
    while pending:
        bone = bones.get(pending.pop())
        if bone is None or bone.parent is None:
            continue
        parent_name = bone.parent.name
        _add_reference(
            references,
            parent_name,
            "ancestor_of",
            child=bone.name,
        )
        if parent_name not in retained:
            retained.add(parent_name)
            pending.append(parent_name)
    return retained


def _matrix_values(matrix: Any) -> tuple[float, ...]:
    return tuple(float(matrix[row][column]) for row in range(4) for column in range(4))


def _mesh_geometry(mesh: Any) -> tuple[Any, ...]:
    vertices = tuple(tuple(float(value) for value in vertex.co) for vertex in mesh.data.vertices)
    shapes = {}
    if mesh.data.shape_keys is not None:
        for key in mesh.data.shape_keys.key_blocks:
            shapes[key.name] = (
                float(key.value),
                tuple(tuple(float(value) for value in point.co) for point in key.data),
            )
    return vertices, shapes


def _weight_values(meshes: list[Any]) -> dict[str, tuple[Any, ...]]:
    result = {}
    for mesh in meshes:
        names_by_index = {group.index: group.name for group in mesh.vertex_groups}
        values = []
        for vertex in mesh.data.vertices:
            for membership in vertex.groups:
                if membership.weight > 0.0:
                    values.append(
                        (vertex.index, names_by_index[membership.group], float(membership.weight))
                    )
        result[mesh.name] = tuple(sorted(values))
    return result


def _bone_state(bone: Any) -> tuple[Any, ...]:
    return (
        _matrix_values(bone.matrix_local),
        tuple(float(value) for value in bone.head_local),
        tuple(float(value) for value in bone.tail_local),
        bone.parent.name if bone.parent else None,
        bool(bone.use_deform),
        bool(bone.use_connect),
    )


def _delete_bones(rig: Any, names: list[str]) -> None:
    view_layer = bpy.context.view_layer
    selected_before = tuple(view_layer.objects.selected)
    active_before = view_layer.objects.active
    try:
        view_layer.objects.active = rig
        rig.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            for name in names:
                edit_bone = rig.data.edit_bones.get(name)
                if edit_bone is None:
                    raise RuntimeError(f"Bone disappeared before deletion: {name}")
                rig.data.edit_bones.remove(edit_bone)
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")
    finally:
        for obj in view_layer.objects:
            obj.select_set(False)
        for obj in selected_before:
            obj.select_set(True)
        view_layer.objects.active = active_before


def _remove_empty_groups(
    meshes: list[Any],
    identities: dict[str, str],
) -> list[dict[str, str]]:
    removed = []
    for mesh in meshes:
        positive_groups = set()
        names_by_index = {group.index: group.name for group in mesh.vertex_groups}
        for vertex in mesh.data.vertices:
            for membership in vertex.groups:
                if membership.weight > 0.0:
                    positive_groups.add(membership.group)
        for group in sorted(mesh.vertex_groups, key=lambda item: item.index, reverse=True):
            if group.index in positive_groups or group.name not in identities:
                continue
            removed.append(
                {
                    "mesh": mesh.name,
                    "group": group.name,
                    "source_identity": identities.get(group.name, group.name),
                }
            )
            mesh.vertex_groups.remove(group)
    return sorted(removed, key=lambda item: (item["mesh"], item["group"]))


def _validate(
    rig: Any,
    meshes: list[Any],
    retained_before: dict[str, tuple[Any, ...]],
    pose_before: dict[str, tuple[Any, ...]],
    geometry_before: dict[str, tuple[Any, ...]],
    weights_before: dict[str, tuple[Any, ...]],
    matrices_before: dict[str, tuple[float, ...]],
    deleted_names: set[str],
) -> dict[str, Any]:
    bpy.context.view_layer.update()
    bone_errors = [
        name for name, expected in retained_before.items()
        if rig.data.bones.get(name) is None or _bone_state(rig.data.bones[name]) != expected
    ]
    pose_errors = [
        name for name, expected in pose_before.items()
        if rig.pose.bones.get(name) is None
        or _matrix_values(rig.pose.bones[name].matrix_basis) != expected
    ]
    geometry_errors = [
        mesh.name for mesh in meshes
        if _mesh_geometry(mesh) != geometry_before[mesh.name]
    ]
    weight_errors = [
        mesh.name for mesh in meshes
        if _weight_values([mesh])[mesh.name] != weights_before[mesh.name]
    ]
    matrix_errors = [
        name for name, expected in matrices_before.items()
        if name not in bpy.data.objects
        or _matrix_values(bpy.data.objects[name].matrix_world) != expected
    ]
    remaining_empty_groups = []
    for mesh in meshes:
        positive_groups = set()
        for vertex in mesh.data.vertices:
            for membership in vertex.groups:
                if membership.weight > 0.0:
                    positive_groups.add(membership.group)
        remaining_empty_groups.extend(
            f"{mesh.name}:{group.name}"
            for group in mesh.vertex_groups
            if group.name in deleted_names
        )
    errors = {
        "retained_bones_changed": bone_errors,
        "pose_transforms_changed": pose_errors,
        "mesh_geometry_or_shape_keys_changed": geometry_errors,
        "positive_weights_changed": weight_errors,
        "object_world_matrices_changed": matrix_errors,
        "empty_vertex_groups_remaining": remaining_empty_groups,
    }
    failed = {key: value for key, value in errors.items() if value}
    if failed:
        raise AssertionError(f"Bone prune validation failed: {failed}")
    return {
        "retained_bones_unchanged": True,
        "pose_transforms_unchanged": True,
        "mesh_geometry_unchanged": True,
        "shape_keys_unchanged": True,
        "positive_weights_unchanged": True,
        "object_world_matrices_unchanged": True,
        "empty_vertex_groups_remaining": 0,
    }


def prune(rig: Any, meshes: Iterable[Any]) -> dict[str, Any]:
    """Delete unused bones, remove zero-weight groups and return a JSON report."""
    if not isinstance(rig, bpy.types.Object) or rig.type != "ARMATURE":
        raise TypeError("rig must be one Blender ARMATURE object")
    if rig.name != TARGET_OBJECT:
        raise ValueError(f"Only {TARGET_OBJECT} may be pruned; got {rig.name!r}")
    if rig.mode != "OBJECT":
        raise ValueError("prune() requires Object mode")
    meshes = _mesh_list(meshes)
    bones = {bone.name: bone for bone in rig.data.bones}
    identities = {
        name: _source_identity(rig, bone)
        for name, bone in bones.items()
    }

    references = _merge_references(rig, set(bones), meshes)
    retained = _close_parents(bones, references)
    candidates = set(bones) - retained

    for name in candidates:
        retained_children = [child.name for child in bones[name].children if child.name in retained]
        if retained_children:
            raise AssertionError(f"Candidate would orphan retained children: {name} {retained_children}")

    depth = {}
    for name in bones:
        current = bones[name]
        depth[name] = 0
        while current.parent is not None:
            depth[name] += 1
            current = current.parent
    deletion_order = sorted(candidates, key=lambda name: (-depth[name], name))

    retained_before = {name: _bone_state(bones[name]) for name in retained}
    pose_before = {
        name: _matrix_values(rig.pose.bones[name].matrix_basis)
        for name in retained
    }
    geometry_before = {mesh.name: _mesh_geometry(mesh) for mesh in meshes}
    weights_before = _weight_values(meshes)
    matrices_before = {
        obj.name: _matrix_values(obj.matrix_world)
        for obj in bpy.data.objects
    }
    groups_before = sum(len(mesh.vertex_groups) for mesh in meshes)

    retained_records = []
    for name in sorted(retained):
        bone = bones[name]
        retained_records.append(
            {
                "name": name,
                "source_identity": identities[name],
                "parent_before": bone.parent.name if bone.parent else None,
                "reasons": references[name],
            }
        )
    candidate_records = [
        {
            "name": name,
            "source_identity": identities[name],
            "parent_before": bones[name].parent.name if bones[name].parent else None,
            "children_before": sorted(child.name for child in bones[name].children),
        }
        for name in sorted(candidates)
    ]
    deleted_records = [
        {
            "name": name,
            "source_identity": identities[name],
            "parent": bones[name].parent.name if bones[name].parent else None,
            "children": sorted(child.name for child in bones[name].children),
        }
        for name in deletion_order
    ]

    if deletion_order:
        _delete_bones(rig, deletion_order)
    removed_groups = _remove_empty_groups(meshes, {name: identities[name] for name in candidates})
    validation = _validate(
        rig,
        meshes,
        retained_before,
        pose_before,
        geometry_before,
        weights_before,
        matrices_before,
        candidates,
    )

    return {
        "schema": SCHEMA,
        "operation": "prune_unused_bones",
        "status": "deleted" if deletion_order or removed_groups else "unchanged",
        "armature": rig.name,
        "scope": "SCSP_Skeleton only; source/reference armatures untouched",
        "weight_policy": "positive means weight > 0.0",
        "before": {
            "bones": len(bones),
            "vertex_groups": groups_before,
            "candidate_bones": len(candidates),
        },
        "after": {
            "bones": len(rig.data.bones),
            "vertex_groups": sum(len(mesh.vertex_groups) for mesh in meshes),
        },
        "candidate_bones": candidate_records,
        "retained_bones": retained_records,
        "deleted_bones": deleted_records,
        "vertex_groups_removed": removed_groups,
        "validation": validation,
    }
