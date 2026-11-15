"""Rename the isolated SCSP working rig to MMD-standard bone names.

The operation changes names and metadata only. It does not create MMD control
bones, IK chains, PMX data, geometry, rest frames, hierarchy, or weights.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import bpy


SCHEMA = 1
RESOURCE = Path(__file__).resolve().parents[1] / "Resources" / "mmd_bone_names.json"
SOURCE_PROPERTY = "scsp_source_bone"
ROLE_PROPERTY = "scsp_working_role"
SOURCE_NAME_PROPERTY = "scsp_mmd_source_name"
SOURCE_IDENTITY_PROPERTY = "scsp_mmd_source_identity"
WORKING_ROLE_PROPERTY = "scsp_mmd_working_role"
NAME_J_PROPERTY = "scsp_mmd_name_j"
NAME_E_PROPERTY = "scsp_mmd_name_e"
REPORT_PROPERTY = "scsp_mmd_bone_names_report"


class MmdBoneNameError(ValueError):
    """The rig cannot be renamed without ambiguity or a contract violation."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_resources() -> dict[str, Any]:
    data = json.loads(RESOURCE.read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise MmdBoneNameError(f"Unsupported bone-name mapping schema in {RESOURCE}")
    mappings = data.get("mappings")
    if not isinstance(mappings, list) or not mappings:
        raise MmdBoneNameError(f"Bone-name resource has no mappings: {RESOURCE}")
    for index, rule in enumerate(mappings):
        if not isinstance(rule, dict):
            raise MmdBoneNameError(f"Mapping {index} is not an object")
        for key in ("role", "source_names", "japanese", "english"):
            if key not in rule:
                raise MmdBoneNameError(f"Mapping {index} is missing {key}")
        if not isinstance(rule["source_names"], list) or not rule["source_names"]:
            raise MmdBoneNameError(f"Mapping {index} source_names must be a non-empty list")
    return data


def _as_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def _as_matrix_rows(matrix: Any) -> list[list[float]]:
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def _max_matrix_error(left: Any, right: Any) -> float:
    return max(
        abs(float(left[row][column]) - float(right[row][column]))
        for row in range(4)
        for column in range(4)
    )


def _bone_snapshot(bone: Any) -> dict[str, Any]:
    return {
        "matrix_local": bone.matrix_local.copy(),
        "head_local": bone.head_local.copy(),
        "tail_local": bone.tail_local.copy(),
        "parent": bone.parent.name if bone.parent else None,
        "use_deform": bool(bone.use_deform),
        "use_connect": bool(bone.use_connect),
    }


def _pose_snapshot(rig: Any) -> dict[str, Any]:
    return {pose_bone.name: pose_bone.matrix_basis.copy() for pose_bone in rig.pose.bones}


def _group_snapshot(mesh: Any) -> tuple[tuple[str, ...], tuple[tuple[tuple[int, float], ...], ...]]:
    names = tuple(group.name for group in mesh.vertex_groups)
    weights = tuple(
        tuple((entry.group, float(entry.weight)) for entry in vertex.groups)
        for vertex in mesh.data.vertices
    )
    return names, weights


def _mesh_list(meshes: Iterable[Any]) -> list[Any]:
    result = []
    seen: set[str] = set()
    for mesh in meshes:
        if not isinstance(mesh, bpy.types.Object) or mesh.type != "MESH":
            raise TypeError("rename(rig, meshes) accepts Blender mesh objects only")
        if mesh.name not in seen:
            result.append(mesh)
            seen.add(mesh.name)
    return sorted(result, key=lambda item: item.name)


def _string_property(owner: Any, name: str) -> str | None:
    value = owner.get(name)
    return str(value) if value is not None else None


def _rule_candidates(rule: dict[str, Any], bones: list[Any]) -> list[Any]:
    source_names = set(rule["source_names"])
    source_metadata = set(rule.get("source_metadata", []))
    required_role = rule.get("working_role")
    matches = []
    for bone in bones:
        working_role = _string_property(bone, ROLE_PROPERTY)
        if required_role and working_role != required_role:
            continue
        source_identity = _string_property(bone, SOURCE_PROPERTY)
        if bone.name in source_names:
            matches.append(bone)
        elif source_metadata and source_identity in source_metadata:
            matches.append(bone)
    return matches


def _safe_mmd_name(name: str, resources: dict[str, Any]) -> bool:
    if not name or "\0" in name:
        return False
    try:
        encoded = name.encode(resources["name_encoding"])
    except UnicodeEncodeError:
        return False
    return len(encoded) <= int(resources["name_max_bytes"])


def _unique_temporary(prefix: str, occupied: set[str], offset: int) -> str:
    for index in range(offset, offset + 10000):
        candidate = f"{prefix}{index:04d}"
        if candidate not in occupied:
            occupied.add(candidate)
            return candidate
    raise MmdBoneNameError(f"Cannot allocate temporary name with prefix {prefix}")


def _mmd_field_owners(bone: Any, pose_bone: Any) -> list[tuple[str, Any]]:
    owners = []
    for label, owner in (("bone", bone), ("pose_bone", pose_bone)):
        try:
            mmd_bone = owner.mmd_bone
        except (AttributeError, ReferenceError):
            continue
        if mmd_bone is not None:
            owners.append((label, mmd_bone))
    return owners


def _write_metadata(
    bone: Any,
    pose_bone: Any,
    record: dict[str, Any],
) -> dict[str, Any]:
    for owner in (bone, pose_bone):
        owner[SOURCE_NAME_PROPERTY] = record["source_name"]
        owner[SOURCE_IDENTITY_PROPERTY] = record["source_identity"]
        if record["working_role"] is not None:
            owner[WORKING_ROLE_PROPERTY] = record["working_role"]
        owner[NAME_J_PROPERTY] = record["name_j"]
        owner[NAME_E_PROPERTY] = record["name_e"]

    plugin_fields = []
    plugin_errors = []
    for owner_name, mmd_bone in _mmd_field_owners(bone, pose_bone):
        for field_name, value in (("name_j", record["name_j"]), ("name_e", record["name_e"])):
            try:
                setattr(mmd_bone, field_name, value)
                plugin_fields.append(f"{owner_name}.{field_name}")
            except (AttributeError, ReferenceError, RuntimeError, TypeError) as exc:
                plugin_errors.append(f"{owner_name}.{field_name}: {exc}")
    return {
        "mmd_plugin_registered": bool(plugin_fields),
        "mmd_plugin_fields": plugin_fields,
        "mmd_plugin_errors": plugin_errors,
    }


def _rename_groups(
    meshes: list[Any],
    bone_map: dict[str, str],
) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    updates = []
    future_names = {}
    for mesh in meshes:
        current = [group.name for group in mesh.vertex_groups]
        projected = [bone_map.get(name, name) for name in current]
        if len(projected) != len(set(projected)):
            duplicates = sorted(name for name, count in
                                ((name, projected.count(name)) for name in set(projected))
                                if count > 1)
            raise MmdBoneNameError(
                f"{mesh.name}: vertex-group rename collision on {duplicates}"
            )
        future_names[mesh.name] = projected

        occupied = set(current) | set(projected)
        temporary = {}
        for offset, old in enumerate(bone_map):
            group = mesh.vertex_groups.get(old)
            if group is None:
                continue
            temp = _unique_temporary(f"__SCSP_MMD_VG_TMP_{offset:02d}_", occupied, 0)
            group.name = temp
            temporary[old] = temp
            updates.append({"mesh": mesh.name, "source_name": old, "new_name": bone_map[old]})
        for old, temp in temporary.items():
            group = mesh.vertex_groups.get(temp)
            if group is None:
                raise MmdBoneNameError(f"{mesh.name}: temporary vertex group disappeared: {temp}")
            group.name = bone_map[old]
    return updates, future_names


def _rename_bones(rig: Any, bone_map: dict[str, str]) -> None:
    selected_before = tuple(bpy.context.selected_objects)
    active_before = bpy.context.view_layer.objects.active
    occupied = {bone.name for bone in rig.data.bones} | set(bone_map.values())
    temporary = {
        old: _unique_temporary("__SCSP_MMD_BONE_TMP_", occupied, offset)
        for offset, old in enumerate(bone_map)
    }
    try:
        bpy.ops.object.select_all(action="DESELECT")
        rig.select_set(True)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.object.mode_set(mode="EDIT")
        try:
            for old, temp in temporary.items():
                edit_bone = rig.data.edit_bones.get(old)
                if edit_bone is None:
                    raise MmdBoneNameError(f"Bone disappeared before rename: {old}")
                edit_bone.name = temp
            for old, temp in temporary.items():
                edit_bone = rig.data.edit_bones.get(temp)
                if edit_bone is None:
                    raise MmdBoneNameError(f"Temporary bone disappeared: {temp}")
                edit_bone.name = bone_map[old]
        finally:
            bpy.ops.object.mode_set(mode="OBJECT")
    finally:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in selected_before:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = active_before


def rename(rig: Any, meshes: Iterable[Any]) -> dict[str, Any]:
    """Apply MMD names to an existing working rig and return a JSON-safe report."""
    if not isinstance(rig, bpy.types.Object) or rig.type != "ARMATURE":
        raise TypeError("rig must be one Blender ARMATURE object")
    if rig.mode != "OBJECT":
        raise MmdBoneNameError("rename() requires Object mode")
    resources = _load_resources()
    meshes = _mesh_list(meshes)
    bones = list(rig.data.bones)
    source_pairs = []
    for rule in resources["mappings"]:
        candidates = _rule_candidates(rule, bones)
        if len(candidates) != 1:
            names = [bone.name for bone in candidates]
            raise MmdBoneNameError(
                f"Role {rule['role']!r} requires exactly one source bone; found {names}"
            )
        bone = candidates[0]
        if rule['role']=='all_parent':
            if bone.parent is not None or bone.head_local.length>1e-5 or not any(c.name=='Hips' for c in bone.children):
                raise MmdBoneNameError('Root does not satisfy existing all-parent role')
        source_pairs.append((rule, bone))

    if len({bone.name for _, bone in source_pairs}) != len(source_pairs):
        grouped = defaultdict(list)
        for rule, bone in source_pairs:
            grouped[bone.name].append(rule["role"])
        raise MmdBoneNameError(f"One source bone resolves to multiple roles: {dict(grouped)}")

    for rule, _ in source_pairs:
        if not _safe_mmd_name(rule["japanese"], resources):
            raise MmdBoneNameError(f"MMD name exceeds resource limits: {rule['japanese']}")

    target_owners = defaultdict(list)
    for rule, bone in source_pairs:
        target_owners[rule["japanese"]].append(bone.name)
    collisions = {
        name: owners for name, owners in target_owners.items() if len(owners) != 1
    }
    if collisions:
        raise MmdBoneNameError(f"MMD target-name collision: {collisions}")

    bone_map = {
        bone.name: rule["japanese"]
        for rule, bone in source_pairs
    }
    mapped_old_names = set(bone_map)
    for old, new in bone_map.items():
        for bone in bones:
            if bone.name == new and new not in mapped_old_names:
                raise MmdBoneNameError(
                    f"MMD target {new!r} for {old!r} already belongs to unmapped bone {bone.name!r}"
                )

    bone_before = {bone.name: _bone_snapshot(bone) for bone in bones}
    pose_before = _pose_snapshot(rig)
    group_before = {mesh.name: _group_snapshot(mesh) for mesh in meshes}
    object_parent_refs = [
        (obj, obj.parent_bone)
        for obj in bpy.data.objects
        if obj.parent == rig
        and obj.parent_type == "BONE"
        and obj.parent_bone in bone_map
    ]

    records = []
    for rule, bone in source_pairs:
        pose_bone = rig.pose.bones.get(bone.name)
        if pose_bone is None:
            raise MmdBoneNameError(f"Pose bone missing for source bone: {bone.name}")
        records.append({
            "role": rule["role"],
            "source_name": bone.name,
            "source_identity": _string_property(bone, SOURCE_PROPERTY) or bone.name,
            "working_role": _string_property(bone, ROLE_PROPERTY),
            "rule_working_role": rule.get("working_role"),
            "name_j": rule["japanese"],
            "name_e": rule["english"],
            "parent_before": bone.parent.name if bone.parent else None,
            "source_lookup": (
                "unique source metadata" if rule.get("source_metadata")
                else "required working role + exact source name" if rule.get("working_role")
                else "exact working name"
            ),
        })

    group_updates, expected_groups = _rename_groups(meshes, bone_map)
    _rename_bones(rig, bone_map)

    parent_updates = []
    for obj, old_parent_bone in object_parent_refs:
        new_parent_bone = bone_map[old_parent_bone]
        obj.parent_bone = new_parent_bone
        parent_updates.append({
            "object": obj.name,
            "source_bone": old_parent_bone,
            "new_bone": new_parent_bone,
        })

    for record in records:
        new_name = record["name_j"]
        bone = rig.data.bones.get(new_name)
        pose_bone = rig.pose.bones.get(new_name)
        if bone is None or pose_bone is None:
            raise AssertionError(f"Renamed bone is missing: {record['source_name']} -> {new_name}")
        record.update(_write_metadata(bone, pose_bone, record))

    after_names = {bone.name for bone in rig.data.bones}
    if len(after_names) != len(bones):
        raise AssertionError("Bone count changed during rename")

    matrix_error = 0.0
    head_error = 0.0
    tail_error = 0.0
    parent_errors = []
    for old_name, before in bone_before.items():
        new_name = bone_map.get(old_name, old_name)
        bone = rig.data.bones.get(new_name)
        if bone is None:
            raise AssertionError(f"Bone lost under rename: {old_name} -> {new_name}")
        matrix_error = max(
            matrix_error, _max_matrix_error(before["matrix_local"], bone.matrix_local)
        )
        head_error = max(
            head_error, max(abs(a - b) for a, b in zip(before["head_local"], bone.head_local))
        )
        tail_error = max(
            tail_error, max(abs(a - b) for a, b in zip(before["tail_local"], bone.tail_local))
        )
        expected_parent = (
            bone_map.get(before["parent"], before["parent"]) if before["parent"] else None
        )
        actual_parent = bone.parent.name if bone.parent else None
        if actual_parent != expected_parent:
            parent_errors.append(
                {"bone": new_name, "expected_parent": expected_parent, "parent": actual_parent}
            )
        if bool(bone.use_deform) != before["use_deform"] or bool(bone.use_connect) != before["use_connect"]:
            raise AssertionError(f"Bone structural flags changed during rename: {new_name}")
    if matrix_error > 1e-7 or head_error > 1e-7 or tail_error > 1e-7 or parent_errors:
        raise AssertionError(
            f"Rename changed rest/hierarchy: matrix={matrix_error} head={head_error} "
            f"tail={tail_error} parents={parent_errors}"
        )

    pose_error = 0.0
    for old_name, before in pose_before.items():
        new_name = bone_map.get(old_name, old_name)
        pose_error = max(
            pose_error, _max_matrix_error(before, rig.pose.bones[new_name].matrix_basis)
        )
    if pose_error > 1e-7:
        raise AssertionError(f"Rename changed pose transforms: {pose_error}")

    weights_errors = []
    for mesh in meshes:
        names_before, weights_before = group_before[mesh.name]
        names_after, weights_after = _group_snapshot(mesh)
        expected_names = tuple(bone_map.get(name, name) for name in names_before)
        if names_after != expected_names:
            weights_errors.append({
                "mesh": mesh.name,
                "expected": list(expected_names),
                "actual": list(names_after),
            })
        if weights_after != weights_before:
            weights_errors.append({"mesh": mesh.name, "weights": "changed"})
        if tuple(expected_groups[mesh.name]) != names_after:
            weights_errors.append({
                "mesh": mesh.name,
                "projected_groups": expected_groups[mesh.name],
                "actual": list(names_after),
            })
    if weights_errors:
        raise AssertionError(f"Vertex-group rename validation failed: {weights_errors}")

    object_parent_errors = [
        obj.name for obj, _ in object_parent_refs
        if obj.parent != rig
        or obj.parent_type != "BONE"
        or obj.parent_bone != bone_map[next(old for item, old in object_parent_refs if item == obj)]
    ]
    if object_parent_errors:
        raise AssertionError(f"Bone-parent object references failed: {object_parent_errors}")

    mapped_names = set(bone_map)
    preserved = [
        {
            "name": bone.name,
            "source_identity": _string_property(bone, SOURCE_PROPERTY) or bone.name,
            "working_role": _string_property(bone, ROLE_PROPERTY),
        }
        for bone in rig.data.bones
        if bone.name not in set(bone_map.values())
    ]
    report = {
        "schema": SCHEMA,
        "operation": "rename_mmd_bones",
        "status": "renamed",
        "saved_blend": False,
        "armature": rig.name,
        "resources": {
            "path": str(RESOURCE),
            "sha256": _sha256(RESOURCE),
            "schema": SCHEMA,
        },
        "summary": {
            "bones": len(bones),
            "mapped": len(records),
            "preserved": len(preserved),
            "vertex_groups_renamed": len(group_updates),
            "bone_parent_objects_updated": len(parent_updates),
            "mmd_plugin_bones": sum(record["mmd_plugin_registered"] for record in records),
            "mmd_plugin_errors": sum(len(record["mmd_plugin_errors"]) for record in records),
        },
        "mapping": dict(sorted(bone_map.items())),
        "records": records,
        "preserved": sorted(preserved, key=lambda item: item["name"]),
        "vertex_group_updates": group_updates,
        "bone_parent_object_updates": parent_updates,
        "not_created_controls": resources["not_created_controls"],
        "metadata": {
            "source_identity_property": SOURCE_PROPERTY,
            "existing_source_identity_preserved": True,
            "source_name_property_on_bone_and_pose_bone": SOURCE_NAME_PROPERTY,
            "source_identity_property_on_bone_and_pose_bone": SOURCE_IDENTITY_PROPERTY,
            "name_j_property": NAME_J_PROPERTY,
            "name_e_property": NAME_E_PROPERTY,
            "official_plugin_binding": "set only when bone or pose_bone exposes mmd_bone",
        },
        "invariants": {
            "bone_count_unchanged": len(after_names) == len(bones),
            "rest_matrix_max_abs_error": matrix_error,
            "head_max_abs_error": head_error,
            "tail_max_abs_error": tail_error,
            "parent_errors": parent_errors,
            "pose_basis_max_abs_error": pose_error,
            "vertex_groups_and_weights_exact": True,
            "bone_parent_object_references_updated": True,
            "geometry_topology_unchanged": True,
            "rest_pose_unchanged": True,
            "hierarchy_unchanged": True,
        },
        "policy": (
            "Rename only the existing working bones and corresponding bindings. "
            "Hips maps to 下半身, never センター. Missing center/IK/control bones are "
            "reported and not created. Auxiliary and source-copy bones keep their names."
        ),
        "limitations": [
            "No PMX or MMD runtime artifact is created by this module.",
            "MMD Tools name_j/name_e fields are written only if the add-on has registered them.",
            "Custom scsp_mmd_name_j/name_e properties preserve the intended names for a later exporter.",
        ],
    }
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    rig[REPORT_PROPERTY] = json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return report


def _cli() -> None:
    rigs = [obj for obj in bpy.data.objects if obj.type == "ARMATURE" and obj.name == "SCSP_Skeleton"]
    if len(rigs) != 1:
        raise MmdBoneNameError("Expected exactly one SCSP_Skeleton armature")
    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    print(
        "SCSP_MMD_BONE_NAMES="
        + json.dumps(rename(rigs[0], meshes), ensure_ascii=False, separators=(",", ":"))
    )


if __name__ == "__main__":
    _cli()
