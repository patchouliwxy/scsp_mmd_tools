"""Explicitly merge eight arm-twist helper groups into their arm main groups."""
import math
from collections import defaultdict
from collections.abc import Iterable
import bpy
SCHEMA = 1
TOLERANCE = 1e-5
MAPPING = (
    ("LeftArm1_rot", "LeftArm"),
    ("LeftArm2_rot", "LeftArm"),
    ("LeftArm3_rot", "LeftArm"),
    ("LeftForeArm_rot", "LeftForeArm"),
    ("RightArm1_rot", "RightArm"),
    ("RightArm2_rot", "RightArm"),
    ("RightArm3_rot", "RightArm"),
    ("RightForeArm_rot", "RightForeArm"),
)
ELBOW_MAPPING = (("LeftElbow", "LeftForeArm"), ("RightElbow", "RightForeArm"))
class TwistTransferError(ValueError):
    """The rig, mapping, or mesh binding cannot be transferred safely."""
def _rig(value):
    if not isinstance(value, bpy.types.Object) or value.type != "ARMATURE":
        raise TwistTransferError("rig must be one ARMATURE object")
    return value
def _meshes(value):
    if isinstance(value, bpy.types.Object):
        values = [value]
    elif isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TwistTransferError("meshes must contain one or more MESH objects")
    else:
        values = list(value)
    result = list(dict.fromkeys(values))
    if not result or any(
        not isinstance(obj, bpy.types.Object) or obj.type != "MESH" for obj in result
    ):
        raise TwistTransferError("meshes must contain one or more MESH objects")
    return result
def _bone(rig, canonical_name):
    matches = [bone for bone in rig.data.bones if bone.name == canonical_name
               or str(bone.get("scsp_source_bone", "")) == canonical_name]
    if len(matches) != 1:
        raise TwistTransferError(
            f"Expected one bone for {canonical_name!r}; found "
            f"{[bone.name for bone in matches]}"
        )
    return matches[0]
def _mapping(rig, mapping_spec):
    result = []
    for source_name, target_name in mapping_spec:
        source, target = _bone(rig, source_name), _bone(rig, target_name)
        result.append((source, target, source_name, target_name))
    sources = {entry[0].name for entry in result}
    targets = {entry[1].name for entry in result}
    expected_targets = {target_name for _, target_name in mapping_spec}
    if (len(sources) != len(mapping_spec) or len(targets) != len(expected_targets)
            or sources & targets):
        raise TwistTransferError("The explicit mapping does not resolve uniquely")
    return result
def _skinned(mesh, rig):
    modifiers = [modifier for modifier in mesh.modifiers if modifier.type == "ARMATURE"]
    if not modifiers:
        return False
    if len(modifiers) != 1 or modifiers[0].object != rig:
        raise TwistTransferError(f"Armature modifier on {mesh.name} is ambiguous or "
                                 f"does not point to {rig.name}")
    return True
def _weights(mesh):
    names = {group.index: group.name for group in mesh.vertex_groups}
    return [
        {names[group.group]: float(group.weight) for group in vertex.groups}
        for vertex in mesh.data.vertices
    ]
def _totals(snapshot, names):
    return {name: sum(values.get(name, 0.0) for values in snapshot) for name in names}
def _counts(snapshot, names):
    return {
        name: sum(values.get(name, 0.0) > 0.0 for values in snapshot)
        for name in names
    }
def _apply(mesh, before, mapping):
    groups = {group.name: group for group in mesh.vertex_groups}
    target_names = sorted({entry[1].name for entry in mapping})
    for vertex_index, values in enumerate(before):
        targets = {name: values.get(name, 0.0) for name in target_names}
        for source, target, _, _ in mapping:
            amount = values.get(source.name, 0.0)
            if amount > 0.0:
                groups[source.name].add([vertex_index], 0.0, "REPLACE")
                targets[target.name] += amount
        for name, weight in targets.items():
            if weight != values.get(name, 0.0):
                group = groups.get(name)
                if group is None:
                    group = mesh.vertex_groups.new(name=name)
                    groups[name] = group
                group.add([vertex_index], weight, "REPLACE")
def _mesh_report(mesh, rig, mapping):
    if not _skinned(mesh, rig):
        return {"mesh": mesh.name, "status": "skipped_static",
                "vertex_count": len(mesh.data.vertices), "armature_modifier": None}
    source_names = [entry[0].name for entry in mapping]
    target_names = sorted({entry[1].name for entry in mapping})
    names = source_names + target_names
    before = _weights(mesh)
    for index, values in enumerate(before):
        if any(not math.isfinite(values.get(name, 0.0))
               or values.get(name, 0.0) < 0.0 for name in source_names):
            raise TwistTransferError(f"Invalid source weight on {mesh.name}[{index}]")
    group_names = {group.name for group in mesh.vertex_groups}
    before_totals, before_counts = _totals(before, names), _counts(before, names)
    _apply(mesh, before, mapping)
    after = _weights(mesh)
    after_totals = _totals(after, names)
    after_counts = _counts(after, names)
    sources = [{"source": name, "vertex_count_before": before_counts[name],
                "vertex_count_after": after_counts[name],
                "total_weight_before": before_totals[name],
                "total_weight_after": after_totals[name]} for name in source_names]
    target_sources = defaultdict(list)
    for source, target, _, _ in mapping:
        target_sources[target.name].append(source.name)
    targets, target_errors = [], []
    for name in target_names:
        source_group = target_sources[name]
        errors = [abs(after_values.get(name, 0.0) - before_values.get(name, 0.0)
                      - sum(before_values.get(source, 0.0) for source in source_group))
                  for before_values, after_values in zip(before, after)]
        error = max(errors, default=0.0)
        target_errors.append(error)
        targets.append({
            "target": name, "sources": source_group,
            "source_vertex_count": len({
                index for index, values in enumerate(before)
                if any(values.get(source, 0.0) > 0.0 for source in source_group)
            }),
            "target_vertex_count_before": before_counts[name],
            "target_vertex_count_after": after_counts[name],
            "source_total_weight": sum(before_totals[source] for source in source_group),
            "total_weight_before": before_totals[name],
            "total_weight_after": after_totals[name],
            "weight_sum_max_error": error})
    ignored = set(names)
    unrelated_error = max((
        abs(after_values.get(name, 0.0) - weight)
        for before_values, after_values in zip(before, after)
        for name, weight in before_values.items() if name not in ignored
    ), default=0.0)
    source_after_max = max(
        (abs(values.get(name, 0.0)) for values in after for name in source_names),
        default=0.0,
    )
    weight_error = max(target_errors, default=0.0)
    groups_preserved = group_names <= {group.name for group in mesh.vertex_groups}
    passed = (weight_error <= TOLERANCE and unrelated_error <= TOLERANCE
              and source_after_max <= TOLERANCE and groups_preserved)
    return {
        "mesh": mesh.name,
        "status": "applied" if passed else "failed_validation",
        "armature_modifier": rig.name,
        "vertex_count": len(mesh.data.vertices),
        "sources": sources,
        "targets": targets,
        "source_total_weight_before": sum(before_totals[name] for name in source_names),
        "source_total_weight_after": sum(after_totals[name] for name in source_names),
        "target_total_weight_before": sum(before_totals[name] for name in target_names),
        "target_total_weight_after": sum(after_totals[name] for name in target_names),
        "weight_sum_max_error": weight_error,
        "unrelated_group_max_error": unrelated_error,
        "unrelated_groups_consistent": unrelated_error <= TOLERANCE,
        "source_weights_cleared": source_after_max <= TOLERANCE,
        "vertex_groups_preserved": groups_preserved,
        "normalized": False}
def _transfer(rig, meshes, mapping_spec, mode, scope):
    rig, meshes = _rig(rig), _meshes(meshes)
    mapping = _mapping(rig, mapping_spec)
    bones_before = {bone.name for bone in rig.data.bones}
    enabled = []
    for target in {entry[1] for entry in mapping}:
        if not target.use_deform:
            target.use_deform = True
            enabled.append(target.name)
    bpy.context.view_layer.update()
    mesh_reports = [_mesh_report(mesh, rig, mapping) for mesh in meshes]
    bpy.context.view_layer.update()
    failed = [r for r in mesh_reports if r["status"] == "failed_validation"]
    return {"schema": SCHEMA, "status": "failed_validation" if failed else "applied",
        "mode": mode, "scope": scope,
        "rig": rig.name,
        "mapping": [{"source_name": source_name, "source_bone": source.name,
                     "target_name": target_name, "target_bone": target.name}
                    for source, target, source_name, target_name in mapping],
        "target_deform_enabled": enabled,
        "bones_preserved": bones_before == {bone.name for bone in rig.data.bones},
        "bone_count": len(rig.data.bones),
        "skinned_meshes": sum(r["status"] == "applied" for r in mesh_reports),
        "static_meshes_skipped": sum(r["status"] == "skipped_static"
                                     for r in mesh_reports),
        "meshes": mesh_reports, "normalized": False,
        "weight_sum_tolerance": TOLERANCE}
def transfer(rig, meshes):
    """Merge the authorized eight twist groups; exact names, no normalization."""
    return _transfer(
        rig, meshes, MAPPING, "twist",
        "Explicit eight-bone arm twist-helper merge; no automatic weights, "
        "normalization, bone/group deletion, or game-formula recovery.")
def transfer_elbows(rig, meshes):
    """Merge exactly LeftElbow/RightElbow groups without parent-based inference."""
    return _transfer(
        rig, meshes, ELBOW_MAPPING, "elbows",
        "Explicit LeftElbow->LeftForeArm and RightElbow->RightForeArm merge; "
        "exact names only, no parent inference, automatic weights, normalization, "
        "bone/group deletion, or game-formula recovery.")


def transfer_elbows_to_arm(rig, meshes):
    """Explicit comparison: preserve the current parent-arm influence."""
    return _transfer(rig, meshes, (("LeftElbow", "LeftArm"), ("RightElbow", "RightArm")),
        "elbows_to_arm", "Elbow support weights to corresponding upper Arm; comparison only.")


def transfer_forearm_helpers_to_elbow(rig, meshes):
    """Merge six upper-arm helpers by MAPPING and both ForeArm groups to Elbow."""
    mapping = (
        tuple(entry for entry in MAPPING if entry[1] in {"LeftArm", "RightArm"})
        + (
            ("LeftForeArm_rot", "LeftElbow"),
            ("LeftForeArm", "LeftElbow"),
            ("RightForeArm_rot", "RightElbow"),
            ("RightForeArm", "RightElbow"),
        )
    )
    return _transfer(
        rig, meshes, mapping, "forearm_helpers_to_elbow",
        "Explicit six upper-arm twist helpers to Arm plus both ForeArm_rot and "
        "ForeArm groups to corresponding Elbow; retain existing Elbow weights, "
        "no Elbow source transfer, inference, normalization, or deletion.")
