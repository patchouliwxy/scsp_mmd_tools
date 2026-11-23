"""Explicit body proportions for one already assembled Blender armature.
The caller owns assembly. This module never creates, merges, renames, reparents,
or deletes objects; it only saves baselines and applies/restores proportions.
No PMX/MMR or recovered native-game formula is implemented.
"""
import json
import math
from collections.abc import Iterable, Mapping

import bpy
from mathutils import Matrix

SCHEMA = 1
BASELINE_KEY = "scsp_body_proportions_baseline"
SCHEMA_KEY = "scsp_body_proportions_schema"
BUST_BAKED_FLAG = "scsp_bust_baked"
PARAMETER_KEYS = ("height", "bust", "head", "arm", "hand")
CANONICAL_BONES = ("Head", "LeftArm", "RightArm", "LeftHand", "RightHand")
IDENTITY = Matrix.Identity(4)

class BindingError(ValueError):
    """The assembled rig or one of its proportion bindings is unusable."""

def _number(value, label):
    if isinstance(value, bool):
        raise BindingError(label + " must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise BindingError(label + " must be a finite number") from exc
    if not math.isfinite(value):
        raise BindingError(label + " must be a finite number")
    return value

def _parameters(values):
    if not isinstance(values, Mapping): raise BindingError("parameters must be a mapping")
    missing = [key for key in PARAMETER_KEYS if key not in values]
    if missing: raise BindingError("Missing body parameters: " + ", ".join(missing))
    result = {key: _number(values[key], "parameters." + key) for key in PARAMETER_KEYS}
    for key in ("height", "head", "arm", "hand"):
        if not 0.25 <= result[key] <= 4: raise BindingError(f"parameters.{key} outside [0.25,4]")
    if not 0 <= result["bust"] <= 1: raise BindingError("parameters.bust outside [0,1]")
    return result

def _objects(value, label):
    if isinstance(value, bpy.types.Object): return [value]
    if isinstance(value, Mapping): value = value.values()
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise BindingError(label + " must be an object or iterable")
    result, seen = [], set()
    for item in value:
        if not isinstance(item, bpy.types.Object): raise BindingError(label + " contains non-object")
        if item not in seen: seen.add(item); result.append(item)
    if not result: raise BindingError(label + " is empty")
    return result

def _uniform(scale, label):
    values = [_number(scale[i], f"{label}[{i}]") for i in range(3)]
    if min(values) <= 0 or max(values) / min(values) > 1.001:
        raise BindingError(label + " must be finite positive uniform")
    return values

def _animation(owner, include_drivers=True):
    data = getattr(owner, "animation_data", None)
    return bool(data and (data.action or any(t.strips for t in data.nla_tracks)
                          or (include_drivers and data.drivers)))

def _one_armature(value):
    if not isinstance(value, bpy.types.Object) or value.type != "ARMATURE":
        raise BindingError("armature must be one assembled ARMATURE object")
    return value

def _bust_name(name):
    leaf = str(name).rsplit(".", 1)[-1].casefold().replace("_", "")
    return leaf == "bustup"


def _bust_keys(mesh):
    keys = mesh.data.shape_keys
    if not keys:
        return [], "missing"
    explicit = [key for key in keys.key_blocks[1:] if _bust_name(key.name)]
    if explicit:
        return explicit, "explicit_bust_up_name"
    # Better FBX may retain the single shape target name on a *_BS_b carrier.
    # Limit this fallback to that carrier's own <source>1 target, never arbitrary keys.
    source = str(mesh.get("scsp_source_object", mesh.name)).rsplit("__", 1)[-1]
    endpoints = list(keys.key_blocks[1:])
    if source.casefold().endswith("_bs_b") and len(endpoints) == 1:
        key = endpoints[0]
        if key.name.casefold() == (source + "1").casefold():
            return [key], "single_bs_b_carrier_target_name"
    return [], "missing"


def _assert_neutral(root, rig, meshes):
    if root.type != "EMPTY": raise BindingError("root must be an EMPTY object")
    _uniform(root.scale, "root.scale")
    if _animation(root) or _animation(rig) or _animation(rig.data):
        raise BindingError("Root or assembled rig is animated/driven")
    for pose_bone in rig.pose.bones:
        if pose_bone.constraints: raise BindingError("Constrained pose bone: " + pose_bone.name)
        error = max(abs(pose_bone.matrix_basis[r][c] - IDENTITY[r][c])
                    for r in range(4) for c in range(4))
        if error > 1e-5: raise BindingError("Non-neutral pose bone: " + pose_bone.name)
    for mesh in meshes:
        if mesh.type != "MESH": raise BindingError("Expected MESH object: " + mesh.name)

def _audit_meshes(meshes, rig):
    bindings, bust = [], []
    for mesh in meshes:
        if _animation(mesh) or _animation(mesh.data):
            raise BindingError("Animated/driven object or mesh data: " + mesh.name)
        keys = mesh.data.shape_keys
        bust_keys, recognition_rule = _bust_keys(mesh)
        if keys and _animation(keys, include_drivers=False):
            raise BindingError("Active shape-key action/NLA: " + mesh.name)
        if keys and keys.animation_data:
            drivers = {driver.data_path for driver in keys.animation_data.drivers}
            bust_paths = {key.path_from_id("value") for key in bust_keys}
            if drivers & bust_paths: raise BindingError("bust_up is driven: " + mesh.name)
        modifiers = [m for m in mesh.modifiers if m.type == "ARMATURE"]
        if len(modifiers) > 1: raise BindingError("Ambiguous Armature modifiers: " + mesh.name)
        if modifiers and modifiers[0].object != rig:
            raise BindingError("Armature modifier points outside supplied rig: " + mesh.name)
        bindings.append({"mesh": mesh.name, "binding": "skinned" if modifiers else "static"})
        found = []
        if mesh.data.shape_keys:
            for index, key in enumerate(mesh.data.shape_keys.key_blocks):
                if key in bust_keys:
                    if found: raise BindingError("Ambiguous bust_up keys on " + mesh.name)
                    if len(key.data) != len(mesh.data.vertices):
                        raise BindingError("Shape-key vertex mismatch: " + mesh.name)
                    found.append({"mesh": mesh.name, "key": key.name, "index": index,
                                  "baseline": _number(key.value, mesh.name + "." + key.name),
                                  "recognition_rule": recognition_rule})
        bust.extend(found)
    return bindings, bust

def _canonical_scales(rig):
    scales = {}
    for name in CANONICAL_BONES:
        if len([bone for bone in rig.data.bones if bone.name == name]) != 1 or name not in rig.pose.bones:
            raise BindingError("Missing or ambiguous canonical bone: " + name)
        scales[name] = _uniform(rig.pose.bones[name].scale, name + ".scale")
    return scales

def _store(root, rig, baseline):
    encoded = json.dumps(baseline, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    for owner in (root, rig):
        owner[SCHEMA_KEY], owner[BASELINE_KEY] = SCHEMA, encoded

def _load(root, armature, meshes):
    if root.type != "EMPTY": raise BindingError("root must be an EMPTY object")
    rig = _one_armature(armature)
    encoded = root.get(BASELINE_KEY)
    if not isinstance(encoded, str): raise BindingError("Missing saved baseline")
    try: baseline = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise BindingError("Body-proportion baseline is not valid JSON") from exc
    if baseline.get("schema") != SCHEMA: raise BindingError("Unsupported baseline schema")
    if root.name != baseline["root"] or rig.name != baseline["armature"]:
        raise BindingError("Root or armature binding changed")
    if int(root.get(SCHEMA_KEY, -1)) != SCHEMA or int(rig.get(SCHEMA_KEY, -1)) != SCHEMA:
        raise BindingError("Baseline marker changed")
    if rig.get(BASELINE_KEY) != encoded: raise BindingError("Armature baseline copy changed")
    mesh_objects = _objects(meshes, "meshes")
    if {mesh.name for mesh in mesh_objects} != {entry["mesh"] for entry in baseline["meshes"]}:
        raise BindingError("Supplied meshes do not match saved bindings")
    current_meshes, current_bust = _audit_meshes(mesh_objects, rig)
    if current_meshes != baseline["meshes"]: raise BindingError("Mesh binding changed")
    if [(b["mesh"], b["key"], b["index"]) for b in current_bust] != [
            (b["mesh"], b["key"], b["index"]) for b in baseline["bust"]]:
        raise BindingError("bust_up binding changed")
    if any(name not in rig.pose.bones for name in CANONICAL_BONES):
        raise BindingError("Canonical bone changed")
    return baseline, rig

def _state(root, rig, baseline):
    return {
        "root_scale": [float(root.scale[i]) for i in range(3)],
        "bone_scale": {name: [float(rig.pose.bones[name].scale[i]) for i in range(3)]
                       for name in CANONICAL_BONES},
        "bust": [{"mesh": b["mesh"], "key": b["key"], "value": float(
            bpy.data.objects[b["mesh"]].data.shape_keys.key_blocks[b["index"]].value)}
            for b in baseline["bust"]],
        "skinned_meshes": sum(e["binding"] == "skinned" for e in baseline["meshes"]),
        "static_meshes": sum(e["binding"] == "static" for e in baseline["meshes"]),
    }

def _bust_report(baseline, requested):
    binding_count = len(baseline["bust"])
    report = {
        "source_channel": "bust_up",
        "requested_blender_value": requested,
        "requested_unity_percent": requested * 100,
        "applied_blender_value": requested if binding_count else None,
        "applied_unity_percent": requested * 100 if binding_count else None,
        "binding_count": binding_count,
        "bindings": [{"mesh": b["mesh"], "key": b["key"],
                      "recognition_rule": b.get("recognition_rule", "explicit_bust_up_name")}
                     for b in baseline["bust"]],
        "automatic_character_adaptation": True,
    }
    if binding_count:
        return {**report, "status": "applied"}, []
    warning = ("Source meshes contain no bust_up shape key; bust was skipped "
               "and no shape key was created")
    return {
        **report,
        "status": "skipped_missing_source_channel",
        "reason": warning,
    }, [warning]

def _assert_unbaked(root):
    """Reject proportion edits or rebaking once rest geometry has been baked."""
    if root.get("scsp_body_proportions_baked") or root.get(BUST_BAKED_FLAG):
        raise BindingError(
            "Body proportions were baked into this rest mesh; rebuild from source to change them"
        )


def _evaluate_bust_geometry(obj, bust_keys):
    """Evaluate local rest coordinates and custom corner normals with only bust keys active."""
    keys = obj.data.shape_keys
    basis = keys.key_blocks[0]
    if not keys.use_relative or keys.reference_key != basis:
        raise BindingError("Relative Basis shape key required to bake bust: " + obj.name)

    bust_set = set(bust_keys)
    for key in keys.key_blocks[1:]:
        if key not in bust_set and key.relative_key in bust_set:
            raise BindingError(f"Shape key {key.name} is relative to bust key on {obj.name}")

    old_show_only = bool(obj.show_only_shape_key)
    old_hide_viewport = bool(obj.hide_viewport)
    old_hide_get = bool(obj.hide_get())
    old_modifiers = [(modifier, bool(modifier.show_viewport)) for modifier in obj.modifiers]
    old_non_bust = [
        (key, bool(key.mute)) for key in keys.key_blocks[1:] if key not in bust_set
    ]
    has_custom_normals = bool(obj.data.has_custom_normals)

    try:
        obj.show_only_shape_key = False
        obj.hide_viewport = False
        obj.hide_set(False)
        for modifier, _ in old_modifiers:
            modifier.show_viewport = False
        for key, _ in old_non_bust:
            key.mute = True
        bpy.context.view_layer.update()

        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(depsgraph)
        eval_mesh = eval_obj.to_mesh()
        try:
            if len(eval_mesh.vertices) != len(obj.data.vertices):
                raise BindingError("Evaluated vertex count mismatch while baking bust: " + obj.name)
            if len(eval_mesh.loops) != len(obj.data.loops):
                raise BindingError("Evaluated loop count mismatch while baking bust: " + obj.name)
            baked_co = [0.0] * (len(eval_mesh.vertices) * 3)
            eval_mesh.vertices.foreach_get("co", baked_co)
            baked_normals = (
                [tuple(float(v) for v in corner.vector) for corner in eval_mesh.corner_normals]
                if has_custom_normals
                else None
            )
        finally:
            eval_obj.to_mesh_clear()
    finally:
        for key, was_muted in old_non_bust:
            key.mute = was_muted
        for modifier, was_shown in old_modifiers:
            modifier.show_viewport = was_shown
        obj.show_only_shape_key = old_show_only
        obj.hide_viewport = old_hide_viewport
        obj.hide_set(old_hide_get)

    vertex_floats = len(obj.data.vertices) * 3
    old_basis_co = [0.0] * vertex_floats
    basis.data.foreach_get("co", old_basis_co)

    if not any(key.vertex_group for key in bust_keys) and all(
        key.relative_key == basis for key in bust_keys
    ):
        expected = list(old_basis_co)
        key_buf = [0.0] * vertex_floats
        for key in bust_keys:
            weight = float(key.value)
            if abs(weight) <= 1e-12 or bool(getattr(key, "mute", False)):
                continue
            key.data.foreach_get("co", key_buf)
            for idx in range(vertex_floats):
                expected[idx] += weight * (key_buf[idx] - old_basis_co[idx])
        eval_err = max((abs(a - b) for a, b in zip(baked_co, expected)), default=0.0)
        if eval_err > 1e-4:
            raise BindingError(
                f"Bust shape-key evaluation mismatch on {obj.name}: {eval_err:.6g}"
            )

    delta_co = [new - old for new, old in zip(baked_co, old_basis_co)]
    max_disp = 0.0
    for idx in range(0, vertex_floats, 3):
        dx, dy, dz = delta_co[idx], delta_co[idx + 1], delta_co[idx + 2]
        disp = math.sqrt(dx * dx + dy * dy + dz * dz)
        if disp > max_disp:
            max_disp = disp

    return baked_co, delta_co, baked_normals, has_custom_normals, max_disp


def initialize(root, armature, meshes, parameters, provenance):
    """Save bindings for the already assembled rig, then apply parameters."""
    if not isinstance(provenance, str) or not provenance.strip():
        raise BindingError("Explicit parameter provenance required")
    rig = _one_armature(armature)
    _assert_unbaked(root)
    if BASELINE_KEY in root: raise BindingError("Already initialized; use apply or restore")
    config = _parameters(parameters)
    mesh_objects = _objects(meshes, "meshes")
    _assert_neutral(root, rig, mesh_objects)
    mesh_bindings, bust = _audit_meshes(mesh_objects, rig)
    baseline = {
        "schema": SCHEMA, "scope": "explicit Blender body proportions; no PMX/MMR",
        "provenance": provenance.strip(), "root": root.name, "armature": rig.name,
        "root_scale": [float(root.scale[i]) for i in range(3)],
        "bone_scale": _canonical_scales(rig), "bust": bust, "meshes": mesh_bindings,
        "initial_parameters": config,
    }
    _store(root, rig, baseline)
    report = apply(root, rig, mesh_objects, config)
    report["status"] = "initialized"
    return report

def apply(root, armature, meshes, parameters):
    """Apply parameters from saved baselines rather than current values."""
    _assert_unbaked(root)
    config = _parameters(parameters)
    mesh_objects = _objects(meshes, "meshes")
    baseline, rig = _load(root, armature, mesh_objects)
    factors = {"Head": config["head"], "LeftArm": config["arm"], "RightArm": config["arm"],
               "LeftHand": config["hand"], "RightHand": config["hand"]}
    root.scale = [value * config["height"] for value in baseline["root_scale"]]
    for name, factor in factors.items():
        rig.pose.bones[name].scale = [value * factor for value in baseline["bone_scale"][name]]
    for binding in baseline["bust"]:
        key = bpy.data.objects[binding["mesh"]].data.shape_keys.key_blocks[binding["index"]]
        key.value = config["bust"]
    bpy.context.view_layer.update()
    bust_report, warnings = _bust_report(baseline, config["bust"])
    return {"status": "applied", "schema": SCHEMA, "provenance": baseline["provenance"],
            "parameters": config, "bust": bust_report,
            "bust_blender_value": bust_report["applied_blender_value"],
            "bust_unity_percent": bust_report["applied_unity_percent"],
            "head_factor_applied_once": True, "warnings": warnings,
            "state": _state(root, rig, baseline)}

def restore(root, armature, meshes):
    """Restore root, canonical pose scales, and bust_up values to baseline."""
    _assert_unbaked(root)
    mesh_objects = _objects(meshes, "meshes")
    baseline, rig = _load(root, armature, mesh_objects)
    root.scale = baseline["root_scale"]
    for name in CANONICAL_BONES:
        rig.pose.bones[name].scale = baseline["bone_scale"][name]
    for binding in baseline["bust"]:
        key = bpy.data.objects[binding["mesh"]].data.shape_keys.key_blocks[binding["index"]]
        key.value = binding["baseline"]
    bpy.context.view_layer.update()
    return {"status": "restored", "schema": SCHEMA, "provenance": baseline["provenance"],
            "state": _state(root, rig, baseline)}


def bake_bust(root, armature, meshes, preserve_bust_morph=False):
    """Bake active bust shape keys into rest coordinates and custom corner normals."""
    _assert_unbaked(root)
    preserve_bust_morph = bool(preserve_bust_morph)
    mesh_objects = _objects(meshes, "meshes")
    baseline, rig = _load(root, armature, mesh_objects)

    bindings_by_mesh = {}
    for binding in baseline["bust"]:
        bindings_by_mesh.setdefault(binding["mesh"], []).append(binding)

    baked_records = []
    for obj in mesh_objects:
        mesh_bindings = bindings_by_mesh.get(obj.name)
        if not mesh_bindings:
            continue
        keys = obj.data.shape_keys
        bust_keys = [keys.key_blocks[b["index"]] for b in mesh_bindings]
        applied_values = [float(key.value) for key in bust_keys]

        vertex_floats = len(obj.data.vertices) * 3
        orig_targets = []
        if preserve_bust_morph:
            if len(bust_keys) != 1:
                raise BindingError("Preserving bust_up requires single bust key on " + obj.name)
            buf = [0.0] * vertex_floats
            bust_keys[0].data.foreach_get("co", buf)
            orig_targets.append(buf)

        baked_co, delta_co, baked_normals, has_custom_normals, max_disp = (
            _evaluate_bust_geometry(obj, bust_keys)
        )

        basis = keys.key_blocks[0]
        basis.data.foreach_set("co", baked_co)
        obj.data.vertices.foreach_set("co", baked_co)

        bust_set = set(bust_keys)
        other_buf = [0.0] * vertex_floats
        for key in keys.key_blocks[1:]:
            if key in bust_set:
                continue
            key.data.foreach_get("co", other_buf)
            key.data.foreach_set("co", [c + d for c, d in zip(other_buf, delta_co)])

        if preserve_bust_morph:
            bust_key = bust_keys[0]
            bust_key.name = "bust_up"
            bust_key.data.foreach_set("co", orig_targets[0])
            bust_key.value = 0.0
            bust_key.slider_min = 0.0
            bust_key.slider_max = 1.0
            bust_key.mute = False
            bust_key.interpolation = "KEY_LINEAR"
            removed_empty_basis = False
        else:
            for key in bust_keys:
                obj.shape_key_remove(key)

            removed_empty_basis = False
            if obj.data.shape_keys and len(obj.data.shape_keys.key_blocks) == 1:
                obj.shape_key_remove(obj.data.shape_keys.key_blocks[0])
                removed_empty_basis = True

        obj.data.update()
        if has_custom_normals and baked_normals is not None:
            obj.data.normals_split_custom_set(baked_normals)
            obj.data.update()

        remaining_keys = (
            [key.name for key in obj.data.shape_keys.key_blocks[1:]]
            if obj.data.shape_keys
            else []
        )
        for binding, applied_value in zip(mesh_bindings, applied_values):
            baked_records.append({
                "mesh": obj.name,
                "key": binding["key"],
                "recognition_rule": binding.get("recognition_rule", "explicit_bust_up_name"),
                "baked_value": applied_value,
                "max_vertex_displacement_m": max_disp,
                "custom_normals_updated": has_custom_normals,
                "removed_empty_basis": removed_empty_basis,
                "preserved_morph": preserve_bust_morph,
                "preserved_morph_name": "bust_up" if preserve_bust_morph else None,
                "remaining_shape_keys": remaining_keys,
            })

    root[BUST_BAKED_FLAG] = True
    rig[BUST_BAKED_FLAG] = True
    bpy.context.view_layer.update()
    return {
        "status": "baked" if baked_records else "skipped_no_bust_keys",
        "schema": SCHEMA,
        "provenance": baseline["provenance"],
        "preserve_bust_morph": preserve_bust_morph,
        "baked_count": len(baked_records),
        "baked_bindings": baked_records,
    }
