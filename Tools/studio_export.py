"""Export an SCSP preprocessed blend to a base PMX in Blender 5.3 background mode (pure Python, no mmd_tools)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pmx_io as pmx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PMX_SCALE = 12.5
MORPH_OFFSET_THRESHOLD = 1e-3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def absolute_path(raw: str, label: str, must_exist: bool = False) -> Path:
    cleaned = raw.strip().strip('"')
    if not cleaned:
        raise ValueError(f"{label} is required")
    expanded = os.path.expandvars(os.path.expanduser(cleaned))
    path = Path(expanded)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path: {cleaned}")
    resolved = path.resolve(strict=False)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def bone_name_maps() -> tuple[dict[str, str], dict[str, str]]:
    resource = PROJECT_ROOT / "Resources" / "mmd_bone_names.json"
    payload = json.loads(resource.read_text(encoding="utf-8"))
    japanese_to_english: dict[str, str] = {}
    role_to_names: dict[str, str] = {}
    for record in payload["mappings"]:
        japanese = str(record["japanese"])
        english = str(record["english"])
        japanese_to_english[japanese] = english
        role_to_names[str(record["role"])] = japanese
    return japanese_to_english, role_to_names


def parent_preserving_world(
    obj: bpy.types.Object,
    parent: bpy.types.Object,
) -> float:
    before = obj.matrix_world.copy()
    obj.parent = parent
    obj.parent_type = "OBJECT"
    obj.parent_bone = ""
    obj.matrix_world = before
    bpy.context.view_layer.update()
    maximum = max(
        abs(before[row][column] - obj.matrix_world[row][column])
        for row in range(4)
        for column in range(4)
    )
    return float(maximum)


def configure_bone_names(
    rig: bpy.types.Object,
    japanese_to_english: dict[str, str],
) -> tuple[dict[str, Any], list[tuple[str, str, str]]]:
    source_names: list[str] = []
    exported_names: list[str] = []
    bone_entries: list[tuple[str, str, str]] = []
    missing_custom_names = 0
    for pose_bone in rig.pose.bones:
        custom_japanese = str(pose_bone.get("scsp_mmd_name_j", "") or "").strip()
        custom_english = str(pose_bone.get("scsp_mmd_name_e", "") or "").strip()
        if not custom_japanese:
            missing_custom_names += 1
        name_j = custom_japanese or pose_bone.name
        name_e = (
            custom_english
            or japanese_to_english.get(name_j)
            or pose_bone.name
        )
        source_names.append(pose_bone.name)
        exported_names.append(name_j)
        bone_entries.append((pose_bone.name, name_j, name_e))
    if not rig.pose.bones:
        raise ValueError("SCSP_Skeleton has no pose bones")
    if len(set(exported_names)) != len(exported_names):
        raise ValueError("MMD bone names are not unique")
    report = {
        "count": len(exported_names),
        "bone_ids": list(range(len(exported_names))),
        "custom_name_count": len(exported_names) - missing_custom_names,
        "fallback_name_count": missing_custom_names,
        "first_names": list(zip(source_names, exported_names))[:12],
        "last_names": list(zip(source_names, exported_names))[-12:],
    }
    return report, bone_entries


def _image_extension(image: bpy.types.Image, fallback_name: str) -> str:
    suffix = Path(fallback_name).suffix.lower()
    if suffix:
        return suffix
    formats = {
        "PNG": ".png",
        "JPEG": ".jpg",
        "TARGA": ".tga",
        "BMP": ".bmp",
        "TIFF": ".tif",
        "WEBP": ".webp",
    }
    return formats.get(str(image.file_format).upper(), ".png")


def _safe_texture_name(image: bpy.types.Image) -> str:
    raw_filepath = str(image.filepath or "")
    basename = Path(raw_filepath.replace("\\", "/")).name or str(image.name)
    suffix = _image_extension(image, basename)
    stem = Path(basename).stem or str(image.name)
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .")
    return (stem or "texture") + suffix


def _stage_image(
    image: bpy.types.Image,
    material_name: str,
    texture_dir: Path,
    image_cache: dict[int, Path],
    source_cache: dict[tuple[str, str], Path],
    used_names: set[str],
    records: list[dict[str, Any]],
) -> Path:
    """Copy or unpack a Blender image once into `texture_dir` and return destination."""
    image_key = int(image.as_pointer())
    cached = image_cache.get(image_key)
    if cached is not None:
        return cached

    packed = getattr(image, "packed_file", None)
    if packed is not None:
        data = bytes(packed.data)
        source_sha = hashlib.sha256(data).hexdigest()
        source_kind = "packed"
        source_path = ""
        cache_key: tuple[str, str] | None = ("packed", source_sha)
    else:
        raw_path = bpy.path.abspath(image.filepath)
        source = Path(raw_path).resolve()
        source_path = str(source)
        if not source.is_file():
            raise FileNotFoundError(
                f"Texture for {material_name} has no packed data "
                f"and the source file is missing: {source}"
            )
        source_kind = "external"
        source_sha = sha256(source)
        cache_key = ("external", os.path.normcase(source_path))
        data = b""

    destination = source_cache.get(cache_key) if cache_key is not None else None
    if destination is None:
        name = _safe_texture_name(image)
        if name in used_names:
            stem, suffix = Path(name).stem, Path(name).suffix
            counter = 2
            while f"{stem}_{counter}{suffix}" in used_names:
                counter += 1
            name = f"{stem}_{counter}{suffix}"
        used_names.add(name)
        destination = texture_dir / name
        if packed is not None:
            destination.write_bytes(data)
        else:
            shutil.copy2(source, destination)
        destination_sha = sha256(destination)
        if source_sha != destination_sha:
            raise AssertionError(
                f"Texture bytes changed while staging: {image.name}"
            )
        if cache_key is not None:
            source_cache[cache_key] = destination
        records.append(
            {
                "image": image.name,
                "source_kind": source_kind,
                "source": source_path,
                "destination": str(destination),
                "size": destination.stat().st_size,
                "sha256": destination_sha,
            }
        )

    image_cache[image_key] = destination
    return destination


def _stage_external_file(
    source: Path,
    texture_dir: Path,
    source_cache: dict[tuple[str, str], Path],
    used_names: set[str],
    records: list[dict[str, Any]],
) -> Path:
    resolved = source.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"External texture file missing: {resolved}")
    cache_key = ("external", os.path.normcase(str(resolved)))
    cached = source_cache.get(cache_key)
    if cached is not None:
        return cached

    name = resolved.name
    if name in used_names:
        stem, suffix = Path(name).stem, Path(name).suffix
        counter = 2
        while f"{stem}_{counter}{suffix}" in used_names:
            counter += 1
        name = f"{stem}_{counter}{suffix}"
    used_names.add(name)
    destination = texture_dir / name
    shutil.copy2(resolved, destination)
    source_sha = sha256(resolved)
    dest_sha = sha256(destination)
    if source_sha != dest_sha:
        raise AssertionError(f"Texture bytes changed while staging: {resolved}")
    source_cache[cache_key] = destination
    records.append(
        {
            "image": resolved.name,
            "source_kind": "external",
            "source": str(resolved),
            "destination": str(destination),
            "size": destination.stat().st_size,
            "sha256": dest_sha,
        }
    )
    return destination


def _extract_material_base_image(material: bpy.types.Material) -> tuple[bpy.types.Image | None, float]:
    alpha = float(material.diffuse_color[3]) if len(material.diffuse_color) >= 4 else 1.0
    node_tree = getattr(material, "node_tree", None)
    if node_tree is None:
        return None, alpha

    bsdf = next((n for n in node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        tex = next((n for n in node_tree.nodes if n.type == "TEX_IMAGE" and n.image is not None), None)
        return (tex.image if tex is not None else None), alpha

    alpha_input = bsdf.inputs.get("Alpha")
    if alpha_input is not None and not alpha_input.is_linked:
        alpha = float(alpha_input.default_value)

    base_input = bsdf.inputs.get("Base Color")
    if base_input is None:
        return None, alpha

    visited: set[int] = set()
    stack = [link.from_node for link in base_input.links]
    while stack:
        node = stack.pop(0)
        ptr = int(node.as_pointer())
        if ptr in visited:
            continue
        visited.add(ptr)
        if node.type == "TEX_IMAGE" and getattr(node, "image", None) is not None:
            return node.image, alpha
        for inp in node.inputs:
            if inp.is_linked:
                for link in inp.links:
                    stack.append(link.from_node)
    return None, alpha


def source_shape_key_report(meshes: list[bpy.types.Object]) -> dict[str, Any]:
    mesh_records: list[dict[str, Any]] = []
    names: list[str] = []
    for mesh in meshes:
        block = mesh.data.shape_keys
        if block is None:
            mesh_records.append(
                {"mesh": mesh.name, "reference_key": None, "keys": []}
            )
            continue
        reference = block.reference_key.name if block.reference_key else None
        if reference != "Basis":
            raise ValueError(
                f"{mesh.name} shape key reference must be Basis, got {reference}"
            )
        keys = [key.name for key in block.key_blocks if key.name != "Basis"]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate shape key names in {mesh.name}")
        names.extend(keys)
        mesh_records.append(
            {
                "mesh": mesh.name,
                "reference_key": reference,
                "keys": keys,
            }
        )
    return {"mesh_keys": mesh_records, "keys": list(dict.fromkeys(names))}


def collect_export_objects() -> tuple[
    bpy.types.Object,
    bpy.types.Object,
    list[bpy.types.Object],
]:
    root = bpy.data.objects.get("SCSP_Character")
    rig = bpy.data.objects.get("SCSP_Skeleton")
    if root is None or root.type != "EMPTY":
        raise ValueError("Expected EMPTY object SCSP_Character")
    if rig is None or rig.type != "ARMATURE":
        raise ValueError("Expected ARMATURE object SCSP_Skeleton")
    main_meshes = [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH"
        and obj.get("scsp_merged_part")
        and not obj.get("scsp_reference_only")
    ]
    all_meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    unexpected = sorted(
        obj.name for obj in all_meshes if obj not in set(main_meshes)
    )
    if not main_meshes:
        raise ValueError("No scsp_merged_part main mesh found")
    if unexpected:
        raise ValueError(
            "Reference or unclassified meshes are present: " + ", ".join(unexpected)
        )
    for mesh in main_meshes:
        modifiers = [
            modifier
            for modifier in mesh.modifiers
            if modifier.type == "ARMATURE"
        ]
        if len(modifiers) != 1 or modifiers[0].object != rig:
            raise ValueError(
                f"{mesh.name} must have exactly one ARMATURE modifier targeting "
                "SCSP_Skeleton"
            )
    return root, rig, sorted(main_meshes, key=lambda item: item.name)


def _encode_pmx_bone_weight(
    mesh_vert: bpy.types.MeshVertex,
    group_index_to_bone: dict[int, int],
) -> pmx.BoneWeight:
    merged: dict[int, float] = {}
    for g in mesh_vert.groups:
        bone_idx = group_index_to_bone.get(g.group)
        if bone_idx is None:
            continue
        w = float(g.weight)
        if w > 1e-6:
            merged[bone_idx] = merged.get(bone_idx, 0.0) + w

    bw = pmx.BoneWeight()
    if not merged:
        bw.type = pmx.BoneWeight.BDEF1
        bw.bones = [0]
        bw.weights = []
        return bw

    items = sorted(merged.items(), key=lambda item: item[1], reverse=True)[:4]
    total = sum(w for _, w in items)
    if total <= 1e-8:
        bw.type = pmx.BoneWeight.BDEF1
        bw.bones = [0]
        bw.weights = []
        return bw
    items = [(b, w / total) for b, w in items]

    if len(items) == 1:
        bw.type = pmx.BoneWeight.BDEF1
        bw.bones = [items[0][0]]
        bw.weights = []
    elif len(items) == 2:
        bw.type = pmx.BoneWeight.BDEF2
        bw.bones = [items[0][0], items[1][0]]
        bw.weights = [items[0][1]]
    else:
        while len(items) < 4:
            items.append((0, 0.0))
        bw.type = pmx.BoneWeight.BDEF4
        bw.bones = [b for b, _ in items]
        bw.weights = [w for _, w in items]
    return bw


def export_scene_to_pmx(
    output: Path,
    texture_dir: Path,
    model_name: str,
    comment: str,
    rig: bpy.types.Object,
    meshes: list[bpy.types.Object],
    bone_entries: list[tuple[str, str, str]],
    scale: float = PMX_SCALE,
) -> tuple[list[dict[str, Any]], dict[str, float], list[dict[str, Any]]]:
    """Build and write a PMX 2.0 binary file directly from Blender objects."""
    texture_dir.mkdir(parents=True, exist_ok=False)

    model = pmx.Model()
    model.name = model_name
    model.name_e = model_name
    model.comment = comment or "exported by scsp_mmd_tools"
    model.comment_e = model.comment

    # 1. Bones
    bone_name_to_idx: dict[str, int] = {}
    rig_world = rig.matrix_world
    for idx, (orig_name, name_j, name_e) in enumerate(bone_entries):
        bone_name_to_idx[orig_name] = idx
        bone_name_to_idx[name_j] = idx

    for idx, (orig_name, name_j, name_e) in enumerate(bone_entries):
        pose_bone = rig.pose.bones[orig_name]
        b_data = pose_bone.bone
        head_w = rig_world @ b_data.head_local
        tail_w = rig_world @ b_data.tail_local
        head_pmx = (head_w.x * scale, head_w.z * scale, head_w.y * scale)
        tail_pmx = (tail_w.x * scale, tail_w.z * scale, tail_w.y * scale)
        bone = pmx.Bone()
        bone.name = name_j
        bone.name_e = name_e
        bone.location = head_pmx
        bone.parent = bone_name_to_idx[b_data.parent.name] if b_data.parent else -1
        bone.transform_order = 0
        bone.displayConnection = (
            tail_pmx[0] - head_pmx[0],
            tail_pmx[1] - head_pmx[1],
            tail_pmx[2] - head_pmx[2],
        )
        bone.isRotatable = True
        bone.isMovable = True
        bone.visible = True
        bone.isControllable = True
        bone.loopCount = 8
        bone.rotationConstraint = 0.03
        model.bones.append(bone)

    # 2. Textures, Materials, Vertices & Faces
    add_uv_count = max(0, max((len(m.data.uv_layers) for m in meshes), default=0) - 1)
    model.header.additional_uvs = add_uv_count

    image_cache: dict[int, Path] = {}
    source_cache: dict[tuple[str, str], Path] = {}
    used_names: set[str] = set()
    texture_records: list[dict[str, Any]] = []
    pmx_tex_index: dict[str, int] = {}

    def register_texture_path(dest: Path) -> int:
        key = os.path.normcase(str(dest))
        existing_idx = pmx_tex_index.get(key)
        if existing_idx is not None:
            return existing_idx
        idx = len(model.textures)
        model.textures.append(pmx.Texture(path=str(dest)))
        pmx_tex_index[key] = idx
        return idx

    material_records: list[dict[str, Any]] = []
    alpha_by_name: dict[str, float] = {}
    bl_vert_to_pmx_verts: dict[tuple[int, int], list[int]] = {}

    for mesh_idx, mesh_obj in enumerate(meshes):
        mesh_data = mesh_obj.data
        mesh_data.calc_loop_triangles()

        mw = mesh_obj.matrix_world
        normal_mat = mw.to_3x3().inverted().transposed()
        group_index_to_bone: dict[int, int] = {}
        for vg in mesh_obj.vertex_groups:
            bi = bone_name_to_idx.get(vg.name)
            if bi is not None:
                group_index_to_bone[vg.index] = bi

        # Group loop triangles by material slot index while preserving slot order.
        tris_by_slot: dict[int, list[Any]] = {}
        for tri in mesh_data.loop_triangles:
            tris_by_slot.setdefault(tri.material_index, []).append(tri)

        uv_layers = list(mesh_data.uv_layers)
        primary_uv = uv_layers[0].data if len(uv_layers) >= 1 else None
        extra_uvs = [layer.data for layer in uv_layers[1:]]

        # Precompute PMX base vertex locations & bone weights per Blender vertex index
        bl_coords: list[tuple[float, float, float]] = []
        bl_weights: list[pmx.BoneWeight] = []
        for mv in mesh_data.vertices:
            wco = mw @ mv.co
            bl_coords.append((wco.x * scale, wco.z * scale, wco.y * scale))
            bl_weights.append(_encode_pmx_bone_weight(mv, group_index_to_bone))

        mesh_vert_cache: dict[int, list[int]] = {}
        slot_indices = sorted(tris_by_slot.keys())
        for slot_idx in slot_indices:
            slot_tris = tris_by_slot[slot_idx]
            if not slot_tris:
                continue
            mat_obj = mesh_data.materials[slot_idx] if slot_idx < len(mesh_data.materials) else None
            mat_name = mat_obj.name if mat_obj is not None else f"{mesh_obj.name}_mat_{slot_idx}"

            tex_idx = -1
            toon_idx = -1
            alpha = 1.0
            is_double_sided = True
            disable_shadows = False
            toon_path_str = ""

            if mat_obj is not None:
                base_img, alpha = _extract_material_base_image(mat_obj)
                if base_img is not None:
                    dest = _stage_image(
                        base_img,
                        mat_name,
                        texture_dir,
                        image_cache,
                        source_cache,
                        used_names,
                        texture_records,
                    )
                    tex_idx = register_texture_path(dest)
                toon_raw = str(mat_obj.get("scsp_skin_toon_path", "") or "").strip()
                if toon_raw:
                    toon_dest = _stage_external_file(
                        Path(toon_raw).expanduser(),
                        texture_dir,
                        source_cache,
                        used_names,
                        texture_records,
                    )
                    toon_idx = register_texture_path(toon_dest)
                    toon_path_str = str(toon_dest)
                is_double_sided = not bool(mat_obj.use_backface_culling)
                disable_shadows = bool(mat_obj.get("scsp_disable_shadows")) or mat_name == "SCSP_EyeEffects"

            alpha_by_name[mat_name] = alpha
            material_records.append(
                {
                    "source": mat_name,
                    "export_copy": mat_name,
                    "diffuse": [1.0, 1.0, 1.0],
                    "specular": [0.0, 0.0, 0.0],
                    "ambient": [0.25, 0.25, 0.25],
                    "shininess": 1.0,
                    "sphere_mode": "OFF",
                    "alpha": alpha,
                    "toon_texture": toon_path_str or None,
                    "is_shared_toon_texture": False,
                }
            )

            pm = pmx.Material()
            pm.name = mat_name
            pm.name_e = mat_name
            pm.diffuse = (1.0, 1.0, 1.0, alpha)
            pm.specular = (0.0, 0.0, 0.0)
            pm.shininess = 1.0
            pm.ambient = (0.25, 0.25, 0.25)
            pm.is_double_sided = is_double_sided
            pm.enabled_drop_shadow = not disable_shadows
            pm.enabled_self_shadow_map = not disable_shadows
            pm.enabled_self_shadow = not disable_shadows
            pm.enabled_toon_edge = False
            pm.edge_color = (0.0, 0.0, 0.0, 1.0)
            pm.edge_size = 1.0
            pm.texture = tex_idx
            pm.sphere_texture = -1
            pm.sphere_texture_mode = 0
            pm.is_shared_toon_texture = False
            pm.toon_texture = toon_idx
            pm.comment = ""
            pm.vertex_count = len(slot_tris) * 3
            model.materials.append(pm)

            for tri in slot_tris:
                face_verts: list[int] = []
                for loop_idx, bl_vi in zip(tri.loops, tri.vertices):
                    loop = mesh_data.loops[loop_idx]
                    n_world = (normal_mat @ loop.normal).normalized()
                    pmx_n = (float(n_world.x), float(n_world.z), float(n_world.y))
                    if primary_uv is not None:
                        u, v = primary_uv[loop_idx].uv
                        pmx_uv = (float(u), float(1.0 - v))
                    else:
                        pmx_uv = (0.0, 0.0)
                    pmx_add: list[tuple[float, float, float, float]] = []
                    for k in range(add_uv_count):
                        if k < len(extra_uvs):
                            eu, ev = extra_uvs[k][loop_idx].uv
                            pmx_add.append((float(eu), float(1.0 - ev), 0.0, 0.0))
                        else:
                            pmx_add.append((0.0, 0.0, 0.0, 0.0))

                    candidates = mesh_vert_cache.get(bl_vi)
                    pmx_vi: int | None = None
                    if candidates is not None:
                        for cand_idx in candidates:
                            cv = model.vertices[cand_idx]
                            if (
                                abs(cv.uv[0] - pmx_uv[0]) > 1e-4
                                or abs(cv.uv[1] - pmx_uv[1]) > 1e-4
                            ):
                                continue
                            dn = (
                                (cv.normal[0] - pmx_n[0]) ** 2
                                + (cv.normal[1] - pmx_n[1]) ** 2
                                + (cv.normal[2] - pmx_n[2]) ** 2
                            )
                            if dn > 1e-4:  # |dn| > 0.01
                                continue
                            add_ok = True
                            for ca, na in zip(cv.additional_uvs, pmx_add):
                                if abs(ca[0] - na[0]) > 1e-4 or abs(ca[1] - na[1]) > 1e-4:
                                    add_ok = False
                                    break
                            if add_ok:
                                pmx_vi = cand_idx
                                break

                    if pmx_vi is None:
                        pmx_vi = len(model.vertices)
                        pv = pmx.Vertex()
                        pv.co = bl_coords[bl_vi]
                        pv.normal = pmx_n
                        pv.uv = pmx_uv
                        pv.additional_uvs = pmx_add
                        pv.weight = bl_weights[bl_vi]
                        pv.edge_scale = 1.0
                        model.vertices.append(pv)
                        mesh_vert_cache.setdefault(bl_vi, []).append(pmx_vi)
                        bl_vert_to_pmx_verts.setdefault((mesh_idx, bl_vi), []).append(pmx_vi)
                    face_verts.append(pmx_vi)
                # Flip winding when converting RH Z-up Blender -> LH Y-up PMX.
                model.faces.append((face_verts[2], face_verts[1], face_verts[0]))

    # 3. Vertex Morphs (Shape Keys)
    morph_by_name: dict[str, pmx.VertexMorph] = {}
    for mesh_idx, mesh_obj in enumerate(meshes):
        block = mesh_obj.data.shape_keys
        if block is None or block.reference_key is None:
            continue
        basis = block.reference_key.data
        mw3 = mesh_obj.matrix_world.to_3x3()
        v_count = len(mesh_obj.data.vertices)
        for kb in block.key_blocks:
            if kb.name == "Basis":
                continue
            morph = morph_by_name.get(kb.name)
            if morph is None:
                morph = pmx.VertexMorph(name=kb.name, name_e="", category=4)
                morph_by_name[kb.name] = morph
                model.morphs.append(morph)
            kb_data = kb.data
            for bl_vi in range(v_count):
                pmx_indices = bl_vert_to_pmx_verts.get((mesh_idx, bl_vi))
                if not pmx_indices:
                    continue
                delta = (mw3 @ (kb_data[bl_vi].co - basis[bl_vi].co)) * scale
                if delta.length <= MORPH_OFFSET_THRESHOLD:
                    continue
                pmx_off = (float(delta.x), float(delta.z), float(delta.y))
                for pvi in pmx_indices:
                    morph.offsets.append(pmx.VertexMorphOffset(index=pvi, offset=pmx_off))

    pmx.save(str(output), model, add_uv_count=add_uv_count)
    return material_records, alpha_by_name, texture_records


def validate_pmx(
    output: Path,
    expected_shape_names: list[str],
    expected_bones: list[str],
    expected_alpha: dict[str, float],
) -> dict[str, Any]:
    model = pmx.load(str(output))
    texture_records: list[dict[str, Any]] = []
    for texture in model.textures:
        path = Path(texture.path)
        resolved = path if path.is_absolute() else output.parent / path
        exists = resolved.is_file()
        texture_records.append(
            {"pmx_path": texture.path, "resolved": str(resolved), "exists": exists}
        )
        if not exists:
            raise FileNotFoundError(f"PMX texture path does not exist: {texture.path}")
    if not texture_records:
        raise ValueError("PMX contains no textures")
    if not model.vertices:
        raise ValueError("PMX contains no vertices")
    if not model.bones:
        raise ValueError("PMX contains no bones")
    if model.rigids or model.joints:
        raise ValueError("PMX unexpectedly contains physics")

    output_bone_names = [bone.name for bone in model.bones]
    if output_bone_names != expected_bones:
        raise ValueError("PMX bone names or order do not match source bone names")
    for vertex in model.vertices:
        if any(index < 0 or index >= len(model.bones) for index in vertex.weight.bones):
            raise ValueError("PMX vertex has an invalid bone index")

    morph_names = [morph.name for morph in model.morphs]
    missing_morphs = sorted(set(expected_shape_names) - set(morph_names))
    if missing_morphs:
        raise ValueError(
            "PMX is missing source shape keys: " + ", ".join(missing_morphs)
        )
    if len(morph_names) != len(set(morph_names)):
        raise ValueError("PMX morph names are not unique")

    material_records: list[dict[str, Any]] = []
    for material in model.materials:
        if material.sphere_texture_mode != 0:
            raise ValueError(f"{material.name} sphere texture mode is not OFF")
        if max(abs(value - 1.0) for value in material.diffuse[:3]) > 1e-6:
            raise ValueError(f"{material.name} diffuse is not 1,1,1")
        if max(abs(value) for value in material.specular) > 1e-6:
            raise ValueError(f"{material.name} specular is not 0,0,0")
        if max(abs(value - 0.25) for value in material.ambient) > 1e-6:
            raise ValueError(f"{material.name} ambient is not 0.25")
        if abs(material.shininess - 1.0) > 1e-6:
            raise ValueError(f"{material.name} shininess is not 1")
        expected = expected_alpha.get(material.name)
        if expected is not None and abs(material.diffuse[3] - expected) > 1e-6:
            raise ValueError(f"{material.name} alpha changed")
        material_records.append(
            {
                "name": material.name,
                "diffuse": list(material.diffuse),
                "ambient": list(material.ambient),
                "specular": list(material.specular),
                "shininess": material.shininess,
                "sphere_texture_mode": material.sphere_texture_mode,
                "texture": material.texture,
                "is_shared_toon_texture": bool(material.is_shared_toon_texture),
                "toon_texture": material.toon_texture,
            }
        )
    return {
        "model_name": model.name,
        "model_name_e": model.name_e,
        "vertices": len(model.vertices),
        "faces": len(model.faces),
        "bones": len(model.bones),
        "bone_names": output_bone_names,
        "morphs": len(model.morphs),
        "morph_names": morph_names,
        "materials": material_records,
        "textures": texture_records,
        "rigids": len(model.rigids),
        "joints": len(model.joints),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blender 5.3 SCSP preprocessed.blend to base PMX"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--comment", default="")
    parser.add_argument("--report", required=True)
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def run(args: argparse.Namespace, report: dict[str, Any]) -> dict[str, Any]:
    source = absolute_path(args.input, "input", must_exist=True)
    output = absolute_path(args.output, "output")
    report_path = absolute_path(args.report, "report")
    if source.suffix.lower() != ".blend":
        raise ValueError("--input must be a .blend file")
    if output.suffix.lower() != ".pmx":
        raise ValueError("--output must be a .pmx file")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    texture_dir = output.with_name(output.stem + "_textures")
    if texture_dir.exists():
        raise FileExistsError(f"Texture directory already exists: {texture_dir}")
    name = args.name.strip()
    if not name:
        raise ValueError("--name cannot be empty")

    source_hash = sha256(source)
    report.update(
        {
            "status": "running",
            "input": str(source),
            "input_sha256_before": source_hash,
            "output": str(output),
            "texture_dir": str(texture_dir),
            "report": str(report_path),
            "name": name,
            "comment": args.comment,
            "exporter": "scsp_pmx_io",
        }
    )

    bpy.ops.wm.open_mainfile(filepath=str(source))
    root, rig, meshes = collect_export_objects()
    report["source_scene"] = {
        "root": root.name,
        "rig": rig.name,
        "bones": len(rig.pose.bones),
        "meshes": [
            {
                "name": mesh.name,
                "part": str(mesh.get("scsp_merged_part")),
                "vertices": len(mesh.data.vertices),
                "materials": len(mesh.data.materials),
                "parent": mesh.parent.name if mesh.parent else None,
            }
            for mesh in meshes
        ],
        "pose_position": rig.data.pose_position,
    }
    source_shapes = source_shape_key_report(meshes)
    report["source_shape_keys"] = source_shapes
    if not source_shapes["keys"]:
        raise ValueError("Source meshes contain no non-Basis shape keys")

    if rig.parent != root:
        rig_world = rig.matrix_world.copy()
        rig.parent = root
        rig.matrix_world = rig_world
    bpy.context.view_layer.update()
    if rig.parent != root:
        raise ValueError("SCSP_Skeleton is not parented to SCSP_Character")

    japanese_to_english, _roles = bone_name_maps()
    bone_report, bone_entries = configure_bone_names(rig, japanese_to_english)
    report["bone_configuration"] = bone_report

    parent_errors = {
        mesh.name: parent_preserving_world(mesh, rig) for mesh in meshes
    }
    report["mesh_parent_world_error"] = parent_errors
    if max(parent_errors.values()) > 1e-5:
        raise ValueError("Mesh world transform changed while parenting to rig")

    original_pose = rig.data.pose_position
    rig.data.pose_position = "REST"
    bpy.context.view_layer.update()
    try:
        material_records, alpha_by_name, texture_records = export_scene_to_pmx(
            output=output,
            texture_dir=texture_dir,
            model_name=name,
            comment=args.comment,
            rig=rig,
            meshes=meshes,
            bone_entries=bone_entries,
            scale=PMX_SCALE,
        )
    finally:
        rig.data.pose_position = original_pose
        bpy.context.view_layer.update()

    report["materials"] = material_records
    report["source_textures"] = texture_records

    expected_bones = [name_j for _orig, name_j, _name_e in bone_entries]
    report["pmx_validation"] = validate_pmx(
        output,
        source_shapes["keys"],
        expected_bones,
        alpha_by_name,
    )
    source_hash_after = sha256(source)
    report["input_sha256_after"] = source_hash_after
    report["source_unchanged"] = source_hash_after == source_hash
    if not report["source_unchanged"]:
        raise AssertionError("Input production blend changed")
    report["output_size"] = output.stat().st_size
    report["output_sha256"] = sha256(output)
    report["status"] = "ok"
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return report


def main() -> None:
    args = parse_args()
    report_path = absolute_path(args.report, "report")
    report: dict[str, Any] = {
        "tool": str(Path(__file__).resolve()),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        result = run(args, report)
        write_json(report_path, result)
        print(
            "STUDIO_EXPORT_OK "
            + json.dumps(
                {
                    "pmx": result["output"],
                    "vertices": result["pmx_validation"]["vertices"],
                    "bones": result["pmx_validation"]["bones"],
                    "morphs": result["pmx_validation"]["morphs"],
                    "textures": len(result["pmx_validation"]["textures"]),
                },
                ensure_ascii=False,
            )
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    main()
