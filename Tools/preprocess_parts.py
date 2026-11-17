"""Bounded, non-destructive preprocessing for imported SCSP parts.

Run this module inside Blender after the Better FBX import stage.  The public
entry point is ``prepare_parts(parts)``; it returns a JSON-serializable report.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import bpy


REPORT_SCHEMA = 1
WEIGHT_EPSILON = 1e-8
ALPHA_SAMPLE_LIMIT = 4096
ALPHA_TRANSPARENT_LEVEL = 0.01
ALPHA_OPAQUE_LEVEL = 0.99
ALPHA_MIXED_FRACTION = 0.01

EYE_BINDINGS = {
    "eyesclera_l": "EYE_L",
    "eyesclera_r": "EYE_R",
}

_NON_COLOR_NAME_PARTS = {
    "ao",
    "mask",
    "metal",
    "metallic",
    "normal",
    "nrm",
    "occlusion",
    "plx",
    "rim",
    "rmask",
    "rough",
    "roughness",
    "sd",
    "spec",
    "specular",
    "spmap",
}

_GENERIC_MATERIAL_PARTS = {
    "m",
    "mat",
    "material",
    "00",
}


def _object_pointer(obj):
    return int(obj.as_pointer())


def _object_label(obj):
    return str(getattr(obj, "name", obj))


def _source_name(obj):
    return str(obj.get("scsp_source_object", "") or _object_label(obj))


def _validate_and_plan(parts):
    if not isinstance(parts, (list, tuple)):
        raise TypeError("parts must be a list or tuple of mappings")

    plan = []
    seen_meshes = set()
    for index, part in enumerate(parts):
        if not isinstance(part, dict):
            raise TypeError(f"parts[{index}] must be a mapping")

        kind = str(part.get("kind", "")).strip()
        if not kind:
            raise ValueError(f"parts[{index}].kind must be a non-empty string")

        arm = part.get("arm")
        if arm is not None and not isinstance(arm, bpy.types.Object):
            raise TypeError(f"parts[{index}].arm must be a Blender object or None")
        if arm is not None and arm.type != "ARMATURE":
            raise ValueError(f"parts[{index}].arm must target an armature")

        objects = list(part.get("objects") or [])
        meshes = list(part.get("meshes") or [])
        for obj in objects + meshes:
            if not isinstance(obj, bpy.types.Object):
                raise TypeError(f"parts[{index}] contains a non-Blender object")

        object_ids = {_object_pointer(obj) for obj in objects}
        for mesh in meshes:
            if mesh.type != "MESH":
                raise ValueError(f"parts[{index}] mesh is not a mesh: {mesh.name}")
            pointer = _object_pointer(mesh)
            if pointer in seen_meshes:
                raise ValueError(f"Mesh appears in more than one part: {mesh.name}")
            seen_meshes.add(pointer)
            if pointer not in object_ids:
                raise ValueError(
                    f"Mesh is missing from its part objects list: {mesh.name}"
                )
            source = str(mesh.get("scsp_source_object", "")).strip()
            source_fbx = str(mesh.get("scsp_source_fbx", "")).strip()
            if not source or not source_fbx:
                raise ValueError(
                    "Each mesh needs scsp_source_object and scsp_source_fbx "
                    f"custom properties: {mesh.name}"
                )

        source_dirs = set()
        for mesh in meshes:
            path = Path(str(mesh["scsp_source_fbx"]))
            if path.parent.is_dir():
                source_dirs.add(path.parent.resolve())

        plan.append(
            {
                "kind": kind,
                "arm": arm,
                "objects": objects,
                "meshes": meshes,
                "source_dirs": sorted(source_dirs, key=lambda item: str(item).lower()),
            }
        )
    return plan


def _topology_snapshot(meshes):
    snapshot = {}
    for mesh in meshes:
        shape_names = []
        if mesh.data.shape_keys:
            shape_names = [key.name for key in mesh.data.shape_keys.key_blocks]
        snapshot[_object_pointer(mesh)] = {
            "name": mesh.name,
            "vertices": len(mesh.data.vertices),
            "edges": len(mesh.data.edges),
            "polygons": len(mesh.data.polygons),
            "loops": len(mesh.data.loops),
            "shape_keys": shape_names,
        }
    return snapshot


_EYE_PARTS_CARD_NAMES = {"m_eyelash", "m_eyebrow", "m_faceline", "m_mouthline"}
_EYE_PARTS_MAT_PREFIXES = ("m_eyelash_", "m_eyebrow_", "m_faceline_", "m_mouthline_")
_EYE_PARTS_TARGET_PREFIXES = ("m_eyelash", "m_eyebrow", "m_faceline", "m_mouthline")


def _accessory_card_kind(obj):
    """Identify accessory material-override carrier cards ('face' or 'eyeparts')."""
    if obj.type != "MESH":
        return None
    names = {_source_name(obj).casefold()}
    raw = obj.get("scsp_merge_sources")
    if raw:
        try:
            sources = json.loads(raw)
            if len(sources) == 1:
                names.add(str(sources[0].get("source_object", "")).casefold())
        except (TypeError, ValueError):
            pass
    path = str(obj.get("scsp_source_fbx", "")).replace("\\", "/").lower()
    accessory = (
        obj.get("scsp_merged_part") == "acce"
        or "_acce__" in obj.name.lower()
        or "/m_acce_" in path
    )
    if not accessory:
        return None

    wants_face = "m_face" in names
    wants_eyeparts = bool(names & _EYE_PARTS_CARD_NAMES)
    is_geom_dup = bool(obj.get("scsp_duplicate_head_target"))
    if not wants_face and not wants_eyeparts and not is_geom_dup:
        return None

    for material in obj.data.materials:
        if material is None:
            continue
        mat_low = material.name.lower()
        if (wants_face or is_geom_dup) and mat_low.startswith("m_face"):
            return "face"
        if (wants_eyeparts or is_geom_dup) and (
            mat_low.startswith(_EYE_PARTS_MAT_PREFIXES)
            or (is_geom_dup and mat_low.startswith(_EYE_PARTS_TARGET_PREFIXES))
        ):
            return "eyeparts"
        if material.node_tree:
            for node in material.node_tree.nodes:
                if node.type != "TEX_IMAGE" or not node.image:
                    continue
                img_low = node.image.name.lower()
                if (wants_face or is_geom_dup) and "_face_" in img_low:
                    return "face"
                if (wants_eyeparts or is_geom_dup) and "_eyeparts_" in img_low:
                    return "eyeparts"
    return None


def _is_face_texture_card(obj):
    """Recognize an accessory face-texture carrier, including merged provenance."""
    return _accessory_card_kind(obj) == "face"


def _helper_reason(obj):
    card_kind = _accessory_card_kind(obj)
    if card_kind == "face":
        return "accessory_face_texture_card"
    if card_kind == "eyeparts":
        return "accessory_eyeparts_texture_card"
    if obj.get("scsp_duplicate_head_target"):
        return "accessory_head_duplicate_overlay"
    source = _source_name(obj).casefold()
    name = obj.name.casefold()
    candidates = {source, name}

    if source == "cheek":
        return "shader_only_cheek_overlay"
    for candidate in candidates:
        if (
            candidate == "roomshoes"
            or candidate.startswith("roomshoes_")
            or candidate.endswith("__roomshoes")
            or "__roomshoes_" in candidate
        ):
            return "room_shoes_helper"
        if candidate.endswith("_skin_00"):
            return "skin_placeholder"
        if candidate.endswith("_shadow"):
            return "shadow_proxy"
        if "materialholder" in candidate:
            return "material_holder"
        if "collider" in candidate:
            return "collider"
        if any(token in candidate for token in ("face_close", "closeface", "close_face")):
            return "face_close"
        if "mouth_hole" in candidate:
            return "mouth_hole"
        if "_helper" in candidate or candidate.startswith("helper_"):
            return "helper"
    return None


def _hide_objects(plan, report):
    seen = set()
    for part in plan:
        for obj in part["objects"]:
            pointer = _object_pointer(obj)
            if pointer in seen:
                continue
            seen.add(pointer)
            reason = _helper_reason(obj)
            if not reason:
                continue

            render_was_hidden = bool(obj.hide_render)
            try:
                viewport_was_hidden = bool(obj.hide_get())
            except RuntimeError:
                viewport_was_hidden = None

            obj.hide_render = True
            viewport_hidden = False
            try:
                obj.hide_set(True)
                viewport_hidden = bool(obj.hide_get())
            except RuntimeError:
                pass

            report["hidden_objects"].append(
                {
                    "name": obj.name,
                    "source_object": _source_name(obj),
                    "part_kind": part["kind"],
                    "reason": reason,
                    "render_was_hidden": render_was_hidden,
                    "viewport_was_hidden": viewport_was_hidden,
                    "hide_render": bool(obj.hide_render),
                    "hide_set": viewport_hidden,
                }
            )


def _material_slots_for_targets(plan):
    targets_by_material = {}
    for part in plan:
        for mesh in part["meshes"]:
            for slot in mesh.material_slots:
                material = slot.material
                if material is None:
                    continue
                entry = targets_by_material.setdefault(
                    _object_pointer(material),
                    {
                        "material": material,
                        "meshes": [],
                        "part_kinds": set(),
                        "source_dirs": set(),
                        "eye_meshes": [],
                    },
                )
                entry["meshes"].append(mesh)
                entry["part_kinds"].add(part["kind"])
                entry["source_dirs"].update(part["source_dirs"])
                source = _source_name(mesh).lower()
                if source in EYE_BINDINGS:
                    entry["eye_meshes"].append(mesh)
    return targets_by_material


def _material_outside_targets(material, target_meshes):
    target_ids = {_object_pointer(obj) for obj in target_meshes}
    outside = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or _object_pointer(obj) in target_ids:
            continue
        if any(slot.material == material for slot in obj.material_slots):
            outside.append(obj.name)
    return outside


def _isolate_shared_material(entry):
    material = entry["material"]
    outside = _material_outside_targets(material, entry["meshes"])
    if not outside:
        return material, False, []

    replacement = material.copy()
    replacement.name = material.name + "_scsp_part"
    for mesh in entry["meshes"]:
        for index, slot in enumerate(mesh.material_slots):
            if slot.material == material:
                mesh.material_slots[index].material = replacement
    entry["material"] = replacement
    return replacement, True, outside


def _existing_scene_links(material):
    count = 0
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if any(slot.material == material for slot in obj.material_slots):
            count += 1
    return count


def _ensure_shader(material):
    if not material.use_nodes or material.node_tree is None:
        material.use_nodes = True
    tree = material.node_tree
    bsdf = next((node for node in tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
    output = next(
        (node for node in tree.nodes if node.type == "OUTPUT_MATERIAL"), None
    )
    if output is None:
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        output.location = (340.0, 80.0)
    if bsdf is None:
        bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (40.0, 120.0)
        base = bsdf.inputs.get("Base Color")
        if base is not None and hasattr(material, "diffuse_color"):
            base.default_value = tuple(material.diffuse_color)
    if not output.inputs["Surface"].is_linked:
        tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return tree, bsdf


def _find_local_file(filename, source_dirs):
    if not filename:
        return None
    candidates = []
    wanted = filename.lower()
    for source_dir in source_dirs:
        for path in source_dir.iterdir():
            if path.is_file() and path.name.lower() == wanted:
                candidates.append(path.resolve())
    unique = sorted(set(candidates), key=lambda item: str(item).lower())
    return unique[0] if len(unique) == 1 else None


def _image_basename(image):
    if image is None:
        return ""
    if image.filepath:
        return Path(image.filepath.replace("\\", "/")).name
    if image.name:
        return image.name if Path(image.name).suffix else image.name + ".png"
    return ""


def _load_local_image(path):
    absolute = str(path.resolve())
    for image in bpy.data.images:
        if image.filepath and Path(bpy.path.abspath(image.filepath)).resolve() == Path(
            absolute
        ):
            return image
    image = bpy.data.images.load(absolute, check_existing=True)
    image.filepath = absolute
    return image


def _resolve_image_nodes(tree, source_dirs):
    resolved = {}
    for node in tree.nodes:
        if node.type != "TEX_IMAGE":
            continue
        basename = _image_basename(node.image)
        path = _find_local_file(basename, source_dirs)
        if path is None:
            continue
        node.image = _load_local_image(path)
        resolved[node.name] = path.name
    return resolved


def _normalized_tokens(value):
    tokens = re.findall(r"[a-z0-9]+", value.lower())
    while tokens and tokens[0] in {"m", "mat", "material"}:
        tokens.pop(0)
    return [token for token in tokens if token not in _GENERIC_MATERIAL_PARTS]


def _normalized_key(value):
    return "_".join(_normalized_tokens(value))


def _is_color_candidate(path):
    stem = path.stem.lower()
    if "eye_col" in stem or "eye_high" in stem or "eye_sclera" in stem:
        return False
    if stem in {"black", "white"} or stem.startswith("fo_rim"):
        return False
    parts = set(token for token in re.split(r"[^a-z0-9]+", stem) if token)
    return not bool(parts & _NON_COLOR_NAME_PARTS)


def _file_core_tokens(path):
    tokens = re.findall(r"[a-z0-9]+", path.stem.lower())
    while "col" in tokens:
        index = tokens.index("col")
        tokens = tokens[:index] + tokens[index + 1 :]
    return [token for token in tokens if token not in {"00"}]


def _match_color_file(source_material_name, source_dirs):
    files = []
    for source_dir in source_dirs:
        files.extend(path.resolve() for path in source_dir.glob("*.png"))
    files = list({str(path): path for path in files}.values())
    files = [path for path in files if _is_color_candidate(path)]
    if not files:
        return None, [], "no local color PNG candidates"

    material_tokens = _normalized_tokens(source_material_name)
    material_key = "_".join(material_tokens)
    matches = []
    for path in files:
        file_tokens = _file_core_tokens(path)
        file_key = "_".join(file_tokens)
        exact = material_key and (
            file_key == material_key or file_key.endswith("_" + material_key)
        )
        token_match = (
            len(material_tokens) >= 2
            and all(token in file_tokens for token in material_tokens)
        )
        if exact or token_match:
            matches.append(path)

    unique = sorted({str(path): path for path in matches}.values(), key=str)
    if len(unique) == 1:
        return unique[0], [path.name for path in files], None
    if len(unique) > 1:
        return None, [path.name for path in unique], "ambiguous local color PNG match"
    return None, [path.name for path in files], "no unambiguous material name match"


def _find_unique_eye_texture(keyword, source_dirs):
    matches = []
    wanted = re.compile(rf"(^|_){re.escape(keyword)}($|_)", re.IGNORECASE)
    for source_dir in source_dirs:
        for path in source_dir.glob("*.png"):
            if wanted.search(path.stem):
                matches.append(path.resolve())
    unique = sorted({str(path): path for path in matches}.values(), key=str)
    if len(unique) == 1:
        return unique[0], None
    if len(unique) > 1:
        return None, "multiple " + keyword + " PNG files"
    return None, "no " + keyword + " PNG file"


def _alpha_stats(image):
    try:
        import numpy as np

        pixel_count = image.size[0] * image.size[1]
        if pixel_count <= 0:
            return {
                "available": False,
                "has_transparency": False,
                "reason": "image has no pixels",
            }
        values = np.empty(pixel_count * 4, dtype=np.float32)
        image.pixels.foreach_get(values)
        alpha = values[3::4]
        if len(alpha) > ALPHA_SAMPLE_LIMIT:
            indices = np.linspace(
                0, len(alpha) - 1, ALPHA_SAMPLE_LIMIT, dtype=np.int64
            )
            alpha = alpha[indices]
        transparent = float(np.mean(alpha <= ALPHA_TRANSPARENT_LEVEL))
        opaque = float(np.mean(alpha >= ALPHA_OPAQUE_LEVEL))
        return {
            "available": True,
            "samples": int(len(alpha)),
            "transparent_fraction": transparent,
            "opaque_fraction": opaque,
            "has_transparency": (
                transparent >= ALPHA_MIXED_FRACTION
                and opaque >= ALPHA_MIXED_FRACTION
            ),
        }
    except Exception as exc:
        return {
            "available": False,
            "has_transparency": False,
            "reason": str(exc),
        }


def _ensure_alpha(material, tree, bsdf, image_node):
    alpha_input = bsdf.inputs.get("Alpha")
    if alpha_input is None:
        return {
            "available": False,
            "has_transparency": False,
            "connected": False,
            "reason": "Principled BSDF has no Alpha input",
        }

    stats = _alpha_stats(image_node.image)
    for link in list(alpha_input.links):
        tree.links.remove(link)
    if stats["has_transparency"]:
        tree.links.new(image_node.outputs["Alpha"], alpha_input)
        stats["connected"] = True
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "DITHERED"
        elif hasattr(material, "blend_method"):
            material.blend_method = "HASHED"
    else:
        alpha_input.default_value = 1.0
        stats["connected"] = False
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "DITHERED"
        elif hasattr(material, "blend_method"):
            material.blend_method = "OPAQUE"
    return stats


def _set_base_color(tree, bsdf, image_node):
    base_input = bsdf.inputs.get("Base Color")
    if base_input is None:
        return False
    for link in list(base_input.links):
        tree.links.remove(link)
    tree.links.new(image_node.outputs["Color"], base_input)
    return True


def _set_emission(tree, bsdf, image_node):
    emission = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
    if emission is None:
        return False
    for link in list(emission.links):
        tree.links.remove(link)
    tree.links.new(image_node.outputs["Color"], emission)
    return True


def _image_node_for_path(tree, path):
    for node in tree.nodes:
        if node.type != "TEX_IMAGE" or node.image is None:
            continue
        try:
            current = Path(bpy.path.abspath(node.image.filepath)).resolve()
        except (OSError, RuntimeError, ValueError):
            continue
        if current == path.resolve():
            return node
    return None


def _ensure_image_node(tree, path, name):
    node = _image_node_for_path(tree, path)
    if node is not None:
        return node
    node = tree.nodes.new("ShaderNodeTexImage")
    node.name = name
    node.label = path.name
    node.image = _load_local_image(path)
    return node


def _linked_image_node(bsdf, input_name):
    socket = bsdf.inputs.get(input_name)
    if socket is None or not socket.is_linked:
        return None
    for link in socket.links:
        if link.from_node.type == "TEX_IMAGE" and link.from_node.image is not None:
            return link.from_node
    return None


def _unresolved_entry(entry, material, part_kinds, expected, reason, candidates=None):
    return {
        "material": material.name,
        "source_material": str(
            material.get("scsp_source_material", material.name)
        ),
        "part_kinds": sorted(part_kinds),
        "meshes": sorted({mesh.name for mesh in entry["meshes"]}),
        "expected": expected,
        "reason": reason,
        "candidates": list(candidates or []),
        "source_dirs": [str(path) for path in sorted(entry["source_dirs"])],
    }


def _process_eye_material(entry, report):
    material, isolated, outside_users = _isolate_shared_material(entry)
    tree, bsdf = _ensure_shader(material)
    source_dirs = sorted(entry["source_dirs"])
    resolved_nodes = _resolve_image_nodes(tree, source_dirs)

    eye_col, eye_col_error = _find_unique_eye_texture("eye_col", source_dirs)
    eye_high, eye_high_error = _find_unique_eye_texture("eye_high", source_dirs)
    if eye_col is None:
        report["unresolved_textures"].append(
            _unresolved_entry(
                entry,
                material,
                entry["part_kinds"],
                "unique eye_col PNG",
                eye_col_error,
            )
        )
        return

    color_node = _ensure_image_node(tree, eye_col, "eye_col")
    base_connected = _set_base_color(tree, bsdf, color_node)
    alpha = _ensure_alpha(material, tree, bsdf, color_node)

    emission_node = None
    emission_connected = False
    if eye_high is not None:
        emission_node = _ensure_image_node(tree, eye_high, "eye_high")
        emission_connected = _set_emission(tree, bsdf, emission_node)
    else:
        report["unresolved_textures"].append(
            _unresolved_entry(
                entry,
                material,
                entry["part_kinds"],
                "unique eye_high PNG",
                eye_high_error,
            )
        )

    status = "connected" if base_connected and emission_connected else "partial"
    report["material_results"].append(
        {
            "material": material.name,
            "source_material": str(
                material.get("scsp_source_material", material.name)
            ),
            "part_kinds": sorted(entry["part_kinds"]),
            "meshes": sorted({mesh.name for mesh in entry["meshes"]}),
            "role": "eye",
            "status": status,
            "isolated_from_outside_parts": isolated,
            "outside_users": outside_users,
            "resolved_existing_images": resolved_nodes,
            "base_color_texture": eye_col.name,
            "emission_texture": eye_high.name if eye_high else None,
            "base_color_connected": base_connected,
            "emission_connected": emission_connected,
            "alpha": alpha,
            "scene_material_users_after": _existing_scene_links(material),
        }
    )


def _process_regular_material(entry, report):
    material, isolated, outside_users = _isolate_shared_material(entry)
    tree, bsdf = _ensure_shader(material)
    source_dirs = sorted(entry["source_dirs"])
    resolved_nodes = _resolve_image_nodes(tree, source_dirs)

    if not source_dirs:
        report["unresolved_textures"].append(
            _unresolved_entry(
                entry,
                material,
                entry["part_kinds"],
                "PNG in the source FBX folder",
                "source FBX folder was not found",
            )
        )
        return

    image_node = _linked_image_node(bsdf, "Base Color")
    fallback_name = None
    match_error = None
    all_candidates = []
    source_name = str(material.get("scsp_source_material", material.name))

    if image_node is not None:
        basename = _image_basename(image_node.image)
        path = _find_local_file(basename, source_dirs)
        if path is not None:
            image_node.image = _load_local_image(path)
            fallback_name = path.name
        else:
            image_node = None

    if image_node is None:
        path, all_candidates, match_error = _match_color_file(
            source_name, source_dirs
        )
        if path is not None:
            image_node = _ensure_image_node(tree, path, "base_color")
            fallback_name = path.name

    if image_node is None:
        report["unresolved_textures"].append(
            _unresolved_entry(
                entry,
                material,
                entry["part_kinds"],
                "unambiguous local color PNG",
                match_error or "no valid local image connection",
                all_candidates,
            )
        )
        return

    base_connected = _set_base_color(tree, bsdf, image_node)
    alpha = _ensure_alpha(material, tree, bsdf, image_node)
    report["material_results"].append(
        {
            "material": material.name,
            "source_material": source_name,
            "part_kinds": sorted(entry["part_kinds"]),
            "meshes": sorted({mesh.name for mesh in entry["meshes"]}),
            "role": "regular",
            "status": "connected" if base_connected else "partial",
            "isolated_from_outside_parts": isolated,
            "outside_users": outside_users,
            "resolved_existing_images": resolved_nodes,
            "base_color_texture": fallback_name or _image_basename(image_node.image),
            "base_color_connected": base_connected,
            "alpha": alpha,
            "scene_material_users_after": _existing_scene_links(material),
        }
    )


def _process_materials(plan, report):
    targets = _material_slots_for_targets(plan)
    ordered = sorted(
        targets.values(),
        key=lambda entry: (
            min(mesh.name for mesh in entry["meshes"]),
            entry["material"].name,
        ),
    )
    for entry in ordered:
        if entry["eye_meshes"]:
            _process_eye_material(entry, report)
        else:
            _process_regular_material(entry, report)


def _material_base_color_image(material):
    if material is None:
        return None
    if material.use_nodes and material.node_tree is not None:
        bsdf = next(
            (node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"),
            None,
        )
        if bsdf is not None:
            node = _linked_image_node(bsdf, "Base Color")
            if node is not None and node.image is not None:
                return node.image
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or node.image is None:
                continue
            stem = Path(node.image.name).stem.lower()
            if ("_face_" in stem or "_eyeparts_" in stem) and not stem.endswith(("_sd", "_mask", "_lmask", "_rmask")):
                return node.image
    return None


def _transfer_accessory_texture(
    plan,
    report,
    carriers,
    target_meshes,
    slot_predicate,
    *,
    kind_label="face",
):
    if not carriers or not target_meshes:
        return

    makeup_source = None
    makeup_image = None
    for carrier in carriers:
        for slot in carrier.material_slots:
            image = _material_base_color_image(slot.material)
            if image is not None:
                makeup_source = carrier
                makeup_image = image
                break
        if makeup_image is not None:
            break

    if makeup_image is None:
        return

    makeup_path = None
    if makeup_image.filepath:
        try:
            candidate = Path(bpy.path.abspath(makeup_image.filepath)).resolve()
            if candidate.is_file():
                makeup_path = candidate
        except (OSError, RuntimeError, ValueError):
            makeup_path = None

    target_set = set(target_meshes)
    isolation_meshes = [
        mesh
        for part in plan
        for mesh in part["meshes"]
        if mesh in target_set or _helper_reason(mesh) is not None
    ]

    processed_mats = set()
    for target_mesh in target_meshes:
        target_slots = [
            slot
            for slot in target_mesh.material_slots
            if slot.material is not None and slot_predicate(slot, target_mesh)
        ]
        for slot in target_slots:
            mat_ptr = _object_pointer(slot.material)
            if mat_ptr in processed_mats:
                continue
            material, isolated, _ = _isolate_shared_material(
                {"material": slot.material, "meshes": isolation_meshes}
            )
            processed_mats.add(_object_pointer(material))
            tree, bsdf = _ensure_shader(material)
            if makeup_path is not None:
                color_node = _ensure_image_node(tree, makeup_path, "base_color")
            else:
                color_node = _linked_image_node(bsdf, "Base Color")
                if color_node is None:
                    color_node = tree.nodes.new("ShaderNodeTexImage")
                    color_node.name = "base_color"
                color_node.image = makeup_image
                color_node.label = makeup_image.name
            base_connected = _set_base_color(tree, bsdf, color_node)
            alpha = _ensure_alpha(material, tree, bsdf, color_node)
            report["makeup_face_transfers"].append(
                {
                    "kind": kind_label,
                    "from_mesh": makeup_source.name,
                    "to_mesh": target_mesh.name,
                    "material": material.name,
                    "isolated": isolated,
                    "texture": makeup_path.name if makeup_path else makeup_image.name,
                    "base_color_connected": base_connected,
                    "alpha": alpha,
                }
            )


def _sorted_world_coords(obj):
    mw = obj.matrix_world
    return sorted(
        (round(co.x, 4), round(co.y, 4), round(co.z, 4))
        for co in (mw @ v.co for v in obj.data.vertices)
    )


def _mark_accessory_head_duplicates(plan):
    """Auto-detect accessory meshes that geometrically duplicate a main head mesh."""
    head_by_sig = {}
    head_coords_cache = {}
    for part in plan:
        if str(part.get("kind", "")).casefold() != "head":
            continue
        for mesh in part["meshes"]:
            if _helper_reason(mesh) is not None:
                continue
            v_count = len(mesh.data.vertices)
            p_count = len(mesh.data.polygons)
            if v_count < 16 or p_count < 8:
                continue
            head_by_sig.setdefault((v_count, p_count), []).append(mesh)

    if not head_by_sig:
        return

    for part in plan:
        if str(part.get("kind", "")).casefold() != "acce":
            continue
        for mesh in part["meshes"]:
            if mesh.data.shape_keys is not None:
                continue
            sig = (len(mesh.data.vertices), len(mesh.data.polygons))
            candidates = head_by_sig.get(sig)
            if not candidates:
                continue
            acce_coords = _sorted_world_coords(mesh)
            for head_mesh in candidates:
                h_ptr = _object_pointer(head_mesh)
                if h_ptr not in head_coords_cache:
                    head_coords_cache[h_ptr] = _sorted_world_coords(head_mesh)
                head_coords = head_coords_cache[h_ptr]
                if all(
                    abs(a[0] - b[0]) < 1e-3
                    and abs(a[1] - b[1]) < 1e-3
                    and abs(a[2] - b[2]) < 1e-3
                    for a, b in zip(acce_coords, head_coords)
                ):
                    mesh["scsp_duplicate_head_target"] = head_mesh.name
                    break


def _apply_accessory_face_makeup(plan, report):
    """Transfer accessory face/eyeparts makeup textures onto main head materials."""
    face_carriers = []
    eyeparts_carriers = []
    seen_carriers = set()
    head_face_meshes = []
    head_non_helper_meshes = []
    seen_targets = set()

    for part in plan:
        kind_cf = str(part.get("kind", "")).casefold()
        for mesh in part["meshes"]:
            ptr = _object_pointer(mesh)
            card_kind = _accessory_card_kind(mesh)
            if card_kind == "face":
                if ptr not in seen_carriers:
                    seen_carriers.add(ptr)
                    face_carriers.append(mesh)
            elif card_kind == "eyeparts":
                if ptr not in seen_carriers:
                    seen_carriers.add(ptr)
                    eyeparts_carriers.append(mesh)
            elif kind_cf == "head" and _helper_reason(mesh) is None:
                if ptr not in seen_targets:
                    seen_targets.add(ptr)
                    head_non_helper_meshes.append(mesh)
                    if _source_name(mesh).casefold() == "face":
                        head_face_meshes.append(mesh)

    _transfer_accessory_texture(
        plan,
        report,
        face_carriers,
        head_face_meshes,
        lambda slot, mesh: slot.material.name.lower().startswith("m_face") or len(mesh.material_slots) == 1,
        kind_label="face",
    )

    def _is_eyeparts_slot(slot, _mesh):
        if slot.material.name.lower().startswith(_EYE_PARTS_TARGET_PREFIXES):
            return True
        img = _material_base_color_image(slot.material)
        return img is not None and "_eyeparts_" in img.name.lower()

    _transfer_accessory_texture(
        plan,
        report,
        eyeparts_carriers,
        head_non_helper_meshes,
        _is_eyeparts_slot,
        kind_label="eyeparts",
    )


def _positive_weight_counts(mesh, arm):
    deform_bones = {
        bone.name for bone in arm.data.bones if getattr(bone, "use_deform", True)
    }
    positive_vertices = set()
    positive_deform_vertices = set()
    positive_non_deform_groups = set()
    for vertex in mesh.data.vertices:
        for membership in vertex.groups:
            if membership.weight <= WEIGHT_EPSILON:
                continue
            group_name = mesh.vertex_groups[membership.group].name
            positive_vertices.add(vertex.index)
            if group_name in deform_bones:
                positive_deform_vertices.add(vertex.index)
            else:
                positive_non_deform_groups.add(group_name)
    return {
        "vertices": len(mesh.data.vertices),
        "positive_weight_vertices": len(positive_vertices),
        "positive_deform_weight_vertices": len(positive_deform_vertices),
        "positive_non_deform_groups": sorted(positive_non_deform_groups),
    }


def _process_eye_bindings(plan, report):
    for part in plan:
        arm = part["arm"]
        for mesh in part["meshes"]:
            source = _source_name(mesh).lower()
            bone_name = EYE_BINDINGS.get(source)
            if bone_name is None:
                continue

            entry = {
                "part_kind": part["kind"],
                "mesh": mesh.name,
                "source_object": _source_name(mesh),
                "armature": arm.name if arm else None,
                "bone": bone_name,
            }
            if arm is None:
                report["eye_binding_skips"].append(
                    dict(entry, reason="part has no owning armature")
                )
                continue
            if arm.data.bones.get(bone_name) is None:
                report["eye_binding_skips"].append(
                    dict(entry, reason="owning armature has no source eye bone")
                )
                continue
            if len(mesh.data.vertices) == 0:
                report["eye_binding_skips"].append(
                    dict(entry, reason="mesh has no vertices")
                )
                continue

            counts = _positive_weight_counts(mesh, arm)
            if (
                counts["positive_weight_vertices"]
                or counts["positive_deform_weight_vertices"]
            ):
                report["eye_binding_skips"].append(
                    dict(
                        entry,
                        reason="mesh already has positive source weights",
                        existing_weights=counts,
                    )
                )
                continue

            armature_modifiers = [
                modifier
                for modifier in mesh.modifiers
                if modifier.type == "ARMATURE"
            ]
            if len(armature_modifiers) > 1:
                report["eye_binding_skips"].append(
                    dict(
                        entry,
                        reason="mesh already has multiple armature modifiers",
                        existing_weights=counts,
                    )
                )
                continue
            if (
                armature_modifiers
                and armature_modifiers[0].object is not arm
            ):
                report["eye_binding_skips"].append(
                    dict(
                        entry,
                        reason="existing armature modifier targets another armature",
                        existing_weights=counts,
                    )
                )
                continue

            group = mesh.vertex_groups.get(bone_name)
            group_created = group is None
            if group is None:
                group = mesh.vertex_groups.new(name=bone_name)
            vertex_indices = [vertex.index for vertex in mesh.data.vertices]
            group.add(vertex_indices, 1.0, "REPLACE")

            modifier_created = False
            if not armature_modifiers:
                modifier = mesh.modifiers.new(
                    name="Eye Armature", type="ARMATURE"
                )
                modifier.object = arm
                modifier_created = True
            else:
                modifier = armature_modifiers[0]

            target_arm_modifiers = [
                item
                for item in mesh.modifiers
                if item.type == "ARMATURE" and item.object is arm
            ]
            if len(target_arm_modifiers) != 1:
                raise AssertionError(
                    "Eye binding did not leave exactly one owning-armature modifier: "
                    + mesh.name
                )

            report["eye_binding_changes"].append(
                dict(
                    entry,
                    reason="no positive source weights were present",
                    vertices_assigned=len(vertex_indices),
                    weight=1.0,
                    vertex_group_created=group_created,
                    modifier=modifier.name,
                    modifier_created=modifier_created,
                    armature_modifier_count=len(target_arm_modifiers),
                    existing_weights_before=counts,
                )
            )


def _validate_topology(snapshots, meshes):
    failures = []
    after = _topology_snapshot(meshes)
    for pointer, before in snapshots.items():
        current = after[pointer]
        for key in ("vertices", "edges", "polygons", "loops", "shape_keys"):
            if before[key] != current[key]:
                failures.append(
                    {
                        "mesh": before["name"],
                        "field": key,
                        "before": before[key],
                        "after": current[key],
                    }
                )
    if failures:
        raise AssertionError("Unexpected topology or shape-key change: " + repr(failures))
    return True


def prepare_parts(parts, *, eye_bindings=True):
    """Prepare imported parts and return a JSON-serializable report.

    The function changes only objects and materials reachable from ``parts``.
    It never deletes objects, edits mesh topology, applies modifiers, merges
    vertices, changes eye pivots, or replaces shape-key data.
    """

    plan = _validate_and_plan(parts)
    meshes = [mesh for part in plan for mesh in part["meshes"]]
    snapshots = _topology_snapshot(meshes)
    empty_names_before = sorted(
        {
            obj.name
            for part in plan
            for obj in part["objects"]
            if obj.type == "EMPTY"
        }
    )

    report = {
        "schema": REPORT_SCHEMA,
        "scope": "imported parts only; non-destructive",
        "blender": bpy.app.version_string,
        "parts": [
            {
                "kind": part["kind"],
                "armature": part["arm"].name if part["arm"] else None,
                "objects": len(part["objects"]),
                "meshes": len(part["meshes"]),
                "source_dirs": [str(path) for path in part["source_dirs"]],
            }
            for part in plan
        ],
        "hidden_objects": [],
        "material_results": [],
        "makeup_face_transfers": [],
        "unresolved_textures": [],
        "eye_binding_changes": [],
        "eye_binding_skips": [],
    }

    _process_materials(plan, report)
    _mark_accessory_head_duplicates(plan)
    _apply_accessory_face_makeup(plan, report)
    if eye_bindings:
        _process_eye_bindings(plan, report)
    _hide_objects(plan, report)

    topology_unchanged = _validate_topology(snapshots, meshes)
    empty_names_after = {
        obj.name
        for part in plan
        for obj in part["objects"]
        if obj.type == "EMPTY"
    }
    deleted_empties = sorted(set(empty_names_before) - empty_names_after)
    if deleted_empties:
        raise AssertionError("Imported empties were unexpectedly deleted: " + repr(deleted_empties))

    report["validation"] = {
        "target_meshes": len(meshes),
        "topology_unchanged": topology_unchanged,
        "shape_keys_unchanged": topology_unchanged,
        "empties_before": len(empty_names_before),
        "empties_after": len(empty_names_after),
        "empties_deleted": 0,
        "outside_scene_materials_modified": 0,
        "source_weight_vertices_preserved": all(
            change["existing_weights_before"]["positive_weight_vertices"] == 0
            for change in report["eye_binding_changes"]
        ),
    }

    json.dumps(report, ensure_ascii=False, allow_nan=False)
    return report


def prepare_eye_bindings(parts):
    """Initialize missing iris bindings; owned by the eye-processing module."""
    report={"eye_binding_changes":[],"eye_binding_skips":[]}
    _process_eye_bindings(_validate_and_plan(parts),report)
    return report


__all__ = ["prepare_parts", "prepare_eye_bindings"]
