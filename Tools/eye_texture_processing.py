"""Integrate baked eye-colour textures into existing head eye materials."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import bpy

from eye_texture_bake import bake_eye_texture


__all__ = ["prepare"]

_COL_SUFFIX = "_eye_col.png"
_HIGH_SUFFIX = "_eye_high.png"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_root(sourcepath: Path) -> Path:
    sourcepath = sourcepath.expanduser().resolve(strict=False)
    if sourcepath.is_file():
        return sourcepath.parent
    if sourcepath.is_dir():
        return sourcepath
    raise FileNotFoundError(f"kind=head sourcepath does not exist: {sourcepath}")


def _matching_files(root: Path, suffix: str) -> list[Path]:
    return sorted(
        (
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.name.casefold().endswith(suffix)
        ),
        key=lambda path: str(path).casefold(),
    )


def _resolve_pair(root: Path, sourcepath: Path) -> tuple[Path, Path]:
    base_files = _matching_files(root, _COL_SUFFIX)
    high_files = _matching_files(root, _HIGH_SUFFIX)
    if len(base_files) != 1:
        raise ValueError(
            f"Eye source {sourcepath} must contain exactly one *{_COL_SUFFIX}; "
            f"found {len(base_files)}: {[str(path) for path in base_files]}"
        )
    if len(high_files) != 1:
        raise ValueError(
            f"Eye source {sourcepath} must contain exactly one *{_HIGH_SUFFIX}; "
            f"found {len(high_files)}: {[str(path) for path in high_files]}"
        )

    base_path = base_files[0]
    highlight_path = high_files[0]
    base_prefix = base_path.stem[: -len("_eye_col")]
    high_prefix = highlight_path.stem[: -len("_eye_high")]
    if base_prefix.casefold() != high_prefix.casefold():
        raise ValueError(
            "Eye files do not form a matching pair: "
            f"{base_path} and {highlight_path}"
        )
    return base_path, highlight_path


def _head_materials(meshes: list[Any]) -> tuple[list[Any], dict[int, list[Any]]]:
    materials: list[Any] = []
    users: dict[int, list[Any]] = {}
    seen_materials: set[int] = set()
    for mesh in meshes:
        if not isinstance(mesh, bpy.types.Object) or mesh.type != "MESH":
            raise TypeError(f"Head part contains a non-mesh Blender object: {mesh!r}")
        for slot in mesh.material_slots:
            material = slot.material
            if material is None:
                continue
            pointer = int(material.as_pointer())
            users.setdefault(pointer, []).append(mesh)
            if pointer not in seen_materials:
                seen_materials.add(pointer)
                materials.append(material)
    return materials, users


def _resolved_image_path(image: Any) -> Path | None:
    if image is None:
        return None
    try:
        return Path(bpy.path.abspath(image.filepath)).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def _same_path(left: Path | None, right: Path) -> bool:
    if left is None:
        return False
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


def _image_nodes(materials: list[Any]) -> list[tuple[Any, Any]]:
    result: list[tuple[Any, Any]] = []
    seen_nodes: set[int] = set()
    for material in materials:
        if not material.use_nodes or material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or node.image is None:
                continue
            pointer = int(node.as_pointer())
            if pointer in seen_nodes:
                continue
            seen_nodes.add(pointer)
            result.append((material, node))
    return result


def _nodes_for_path(
    materials: list[Any],
    path: Path,
) -> list[tuple[Any, Any]]:
    return [
        (material, node)
        for material, node in _image_nodes(materials)
        if _same_path(_resolved_image_path(node.image), path)
    ]


def _node_snapshot(nodes: list[tuple[Any, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "material": material.name,
            "material_pointer": int(material.as_pointer()),
            "node": node.name,
            "node_pointer": int(node.as_pointer()),
            "image": node.image.name if node.image else None,
            "image_pointer": int(node.image.as_pointer()) if node.image else None,
            "filepath": node.image.filepath if node.image else None,
        }
        for material, node in nodes
    ]


def _source_object(mesh: Any) -> str:
    return str(mesh.get("scsp_source_object", "") or mesh.name)


def _eye_sides(meshes: list[Any], materials: set[int]) -> list[str]:
    result: set[str] = set()
    for mesh in meshes:
        if not any(
            slot.material is not None
            and int(slot.material.as_pointer()) in materials
            for slot in mesh.material_slots
        ):
            continue
        source = _source_object(mesh).casefold()
        if "eyesclera_l" in source:
            result.add("left")
        if "eyesclera_r" in source:
            result.add("right")
    return sorted(result)


def _configure_baked_image(path: Path) -> Any:
    image = bpy.data.images.load(str(path), check_existing=False)
    image.colorspace_settings.name = "sRGB"
    image.alpha_mode = "STRAIGHT"
    if hasattr(image, "use_alpha"):
        image.use_alpha = True
    return image


def prepare(parts: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Bake configured eye pairs and replace their existing base image nodes."""

    configured_dir = config.get("eye_texture_output_dir")
    if not configured_dir:
        raise ValueError(
            "config.eye_texture_output_dir must be present when eye texture processing is requested"
        )
    output_dir = Path(configured_dir).expanduser().resolve(strict=False)

    heads = [
        (index, part)
        for index, part in enumerate(parts)
        if isinstance(part, dict)
        and str(part.get("kind", "")).strip().casefold() == "head"
    ]
    if not heads:
        raise ValueError(
            "config.eye_texture_output_dir is configured but parts contain no kind=head entry"
        )

    report: dict[str, Any] = {
        "schema": "scsp-eye-texture-processing/1",
        "status": "ok",
        "output_dir": str(output_dir),
        "heads": [],
        "warnings": [],
        "static_uv_limitation": (
            "Only static, normalized-UV-aligned *_eye_col/*_eye_high pairs are "
            "supported. Material _EyeOffset.zw, _HighlightShift, and runtime "
            "jitter are not recoverable from the two PNG files."
        ),
    }
    used_outputs: set[str] = set()

    for part_index, part in heads:
        raw_sourcepath = str(part.get("sourcepath", "")).strip()
        if not raw_sourcepath:
            raise ValueError(f"parts[{part_index}].sourcepath is required for kind=head")
        sourcepath = Path(raw_sourcepath).expanduser().resolve(strict=False)
        root = _source_root(sourcepath)
        base_path, highlight_path = _resolve_pair(root, sourcepath)
        output_path = (output_dir / f"{base_path.stem}_baked.png").resolve()
        output_key = os.path.normcase(str(output_path))
        if output_key in used_outputs:
            raise ValueError(f"Multiple head parts map to the same baked output: {output_path}")
        used_outputs.add(output_key)

        meshes = list(part.get("meshes") or [])
        materials, material_users = _head_materials(meshes)
        base_nodes = _nodes_for_path(materials, base_path)
        high_nodes = _nodes_for_path(materials, highlight_path)
        if not base_nodes:
            raise RuntimeError(
                "No existing head material image node references the eye base texture: "
                f"{base_path}"
            )

        base_sha_before = _sha256(base_path)
        high_sha_before = _sha256(highlight_path)
        high_snapshot_before = _node_snapshot(high_nodes)
        texture_nodes_before = len(_image_nodes(materials))
        base_materials = {
            int(material.as_pointer())
            for material, _node in base_nodes
        }
        sides_before = _eye_sides(meshes, base_materials)

        bake_report = bake_eye_texture(base_path, highlight_path, output_path)
        head_report: dict[str, Any] = {
            "part_index": part_index,
            "sourcepath": str(sourcepath),
            "source_root": str(root),
            "pair": {
                "base": str(base_path),
                "highlight": str(highlight_path),
                "base_sha256_before": base_sha_before,
                "highlight_sha256_before": high_sha_before,
            },
            "bake": bake_report,
            "material_integration": {},
        }

        if bake_report.get("status") != "ok":
            reason = str(bake_report.get("reason") or "eye texture bake did not support this pair")
            report["warnings"].append(f"{base_path}: {reason}; original base texture retained")
            base_nodes_after_source = _nodes_for_path(materials, base_path)
            high_nodes_after = _nodes_for_path(materials, highlight_path)
            head_report["material_integration"] = {
                "base_reference_count_before": len(base_nodes),
                "base_nodes_replaced": 0,
                "base_reference_count_after_source": len(base_nodes_after_source),
                "eye_sides_before": sides_before,
                "eye_sides_after": _eye_sides(meshes, base_materials),
                "high_node_count_before": len(high_nodes),
                "high_node_count_after": len(high_nodes_after),
                "high_nodes_unchanged": (
                    _node_snapshot(high_nodes_after) == high_snapshot_before
                ),
                "texture_node_count_before": texture_nodes_before,
                "texture_node_count_after": len(_image_nodes(materials)),
                "original_base_retained": (
                    len(base_nodes_after_source) == len(base_nodes)
                ),
            }
        else:
            baked_image = _configure_baked_image(output_path)
            for _material, node in base_nodes:
                node.image = baked_image

            base_nodes_after_source = _nodes_for_path(materials, base_path)
            base_nodes_after_baked = _nodes_for_path(materials, output_path)
            high_nodes_after = _nodes_for_path(materials, highlight_path)
            high_snapshot_after = _node_snapshot(high_nodes_after)
            texture_nodes_after = len(_image_nodes(materials))
            sides_after = _eye_sides(meshes, base_materials)
            all_replaced = (
                not base_nodes_after_source
                and len(base_nodes_after_baked) == len(base_nodes)
                and all(
                    node.image is baked_image
                    for _material, node in _nodes_for_path(materials, output_path)
                )
            )
            if not all_replaced:
                raise RuntimeError(
                    f"Not all existing base image references were replaced for {base_path}"
                )
            if sides_before != ["left", "right"] or sides_after != sides_before:
                raise RuntimeError(
                    "Expected existing left and right eye material references for "
                    f"{base_path}; found before={sides_before}, after={sides_after}"
                )
            if high_snapshot_after != high_snapshot_before:
                raise RuntimeError(f"Existing highlight nodes changed unexpectedly for {base_path}")
            if texture_nodes_after != texture_nodes_before:
                raise RuntimeError(f"Texture node count changed unexpectedly for {base_path}")

            head_report["material_integration"] = {
                "base_reference_count_before": len(base_nodes),
                "base_reference_count_after_baked": len(base_nodes_after_baked),
                "base_reference_count_after_source": len(base_nodes_after_source),
                "base_nodes_replaced": len(base_nodes),
                "eye_sides_before": sides_before,
                "eye_sides_after": sides_after,
                "left_and_right_references_checked": (
                    sides_before == ["left", "right"]
                    and sides_after == sides_before
                ),
                "high_node_count_before": len(high_nodes),
                "high_node_count_after": len(high_nodes_after),
                "high_nodes_unchanged": high_snapshot_after == high_snapshot_before,
                "texture_node_count_before": texture_nodes_before,
                "texture_node_count_after": texture_nodes_after,
                "second_highlight_created": False,
                "baked_image": {
                    "name": baked_image.name,
                    "filepath": baked_image.filepath,
                    "colorspace": baked_image.colorspace_settings.name,
                    "alpha_mode": baked_image.alpha_mode,
                },
                "material_users": {
                    material.name: sorted(
                        {
                            mesh.name
                            for mesh in material_users.get(int(material.as_pointer()), [])
                        }
                    )
                    for material in {material for material, _node in base_nodes}
                },
            }

        base_sha_after = _sha256(base_path)
        high_sha_after = _sha256(highlight_path)
        head_report["pair"]["base_sha256_after"] = base_sha_after
        head_report["pair"]["highlight_sha256_after"] = high_sha_after
        head_report["pair"]["source_hashes_unchanged"] = (
            base_sha_before == base_sha_after
            and high_sha_before == high_sha_after
        )
        if not head_report["pair"]["source_hashes_unchanged"]:
            raise RuntimeError(f"Eye source texture changed during processing: {sourcepath}")
        report["heads"].append(head_report)

    if report["warnings"]:
        report["status"] = "partial"
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    return report
