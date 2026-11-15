"""Per-character SCSP skin color and shadow toon adaptation for Blender -> PMX."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

try:
    import bpy
except ImportError:
    bpy = None

SCHEMA = "scsp-skin-processing/1"
TOON_WIDTH = 16
TOON_HEIGHT = 32

_HEX_RGB_RE = re.compile(r"^#?([0-9a-fA-F]{6})$")
_COPY_SUFFIX_RE = re.compile(r"\.\d{3}$")
_SKIN_NAME_RE = re.compile(r"(?:^|_)skin(?:_|\d|$)")
_FACE_COL_RE = re.compile(r"(?:^|_)face(?:_\d+)?_col$")
_EXCLUDED_MAT_TOKENS = (
    "mouth",
    "teeth",
    "tongue",
    "faceline",
    "cheek",
    "eye",
    "brow",
    "lash",
    "hair",
    "sclera",
)
_NONCOLOR_TEX_SUFFIXES = ("_sd", "_sp", "_nml", "_mask", "_rmask", "_lmask")

SKIN_RECORD_FIELDS = {
    "body": ("bodySkinBrightCode", "bodySkinDarkCode"),
    "face": ("faceSkinBrightCode", "faceSkinDarkCode"),
}


def parse_hex_rgb(code: str) -> tuple[float, float, float]:
    """Parse a `#RRGGBB` or `RRGGBB` string into linear-byte RGB floats in [0, 1]."""
    text = str(code or "").strip()
    match = _HEX_RGB_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"Invalid skin hex color code: {code!r}")
    digits = match.group(1)
    return (
        int(digits[0:2], 16) / 255.0,
        int(digits[2:4], 16) / 255.0,
        int(digits[4:6], 16) / 255.0,
    )


def face_sd_toon(shade_r: Any) -> Any:
    """Evaluate SCSP FaceSkin toon ramp (`UseWrap=0, center=0.45, width=0.05`)."""
    if np is None:
        raise RuntimeError("numpy is required for face_sd_toon")
    return np.clip((shade_r * 0.5 - 0.425) / 0.05, 0.0, 1.0).astype(np.float32)


def skin_tint(
    bright_code: str,
    dark_code: str | None = None,
    toon: Any = 1.0,
) -> Any:
    """Compute SCSP `FaceSkin.shader` tint multiplier `(lerp(dark, bright, toon) + 0.5)`."""
    bright_rgb = parse_hex_rgb(bright_code)
    if np is None:
        if dark_code is None:
            return tuple(c + 0.5 for c in bright_rgb)
        dark_rgb = parse_hex_rgb(dark_code)
        t = float(toon)
        return tuple(d * (1.0 - t) + b * t + 0.5 for b, d in zip(bright_rgb, dark_rgb))
    bright_arr = np.asarray(bright_rgb, dtype=np.float32)
    if dark_code is None:
        return bright_arr + 0.5
    dark_arr = np.asarray(parse_hex_rgb(dark_code), dtype=np.float32)
    return dark_arr * (1.0 - toon) + bright_arr * toon + 0.5


def bake_skin_pixels(
    base_rgba: Any,
    bright_code: str,
    dark_code: str | None = None,
    shade_rgba: Any = None,
) -> Any:
    """Apply SCSP character skin palette tint to Non-Color STRAIGHT RGBA pixels."""
    if np is None:
        raise RuntimeError("numpy is required for bake_skin_pixels")
    if shade_rgba is not None:
        if dark_code is None:
            raise ValueError("dark_code is required when baking with a face_sd map")
        toon = face_sd_toon(shade_rgba[..., 0:1])
        tint = skin_tint(bright_code, dark_code, toon)
    else:
        tint = skin_tint(bright_code)
    result = np.array(base_rgba, dtype=np.float32, copy=True)
    result[..., :3] = np.clip(result[..., :3] * tint, 0.0, 1.0)
    result[..., 3] = np.clip(base_rgba[..., 3], 0.0, 1.0)
    return result


def make_toon_ramp_pixels(
    bright_code: str,
    dark_code: str,
    width: int = TOON_WIDTH,
    height: int = TOON_HEIGHT,
) -> Any:
    """Generate an MMD skin toon PNG array (bottom=shadow tint ratio, top=white)."""
    if np is None:
        raise RuntimeError("numpy is required for make_toon_ramp_pixels")
    if width <= 0 or height <= 1:
        raise ValueError(f"Invalid toon ramp size: {width}x{height}")
    lit_tint = np.maximum(np.asarray(parse_hex_rgb(bright_code), dtype=np.float32) + 0.5, 1e-4)
    shadow_tint = np.maximum(np.asarray(parse_hex_rgb(dark_code), dtype=np.float32) + 0.5, 0.0)
    raw_ratio = np.clip(shadow_tint / lit_tint, 0.0, 1.0).astype(np.float32)
    # Attenuate toon depth (~50%, closer to MMD toon02) so Diffuse BSDF + Toon don't double-darken skin.
    shadow_ratio = (1.0 - (1.0 - raw_ratio) * 0.50).astype(np.float32)

    # Keep front and normal +/-40deg pitch views in pure white lit zone; shade only steep undersides/back angles.
    t = np.linspace(0.0, 1.0, height, dtype=np.float32)
    u = np.clip((t - 0.06) / 0.14, 0.0, 1.0)
    lit_factor = (u * u * (3.0 - 2.0 * u))[:, None]
    row_rgb = shadow_ratio[None, :] * (1.0 - lit_factor) + lit_factor
    ramp = np.ones((height, width, 4), dtype=np.float32)
    ramp[..., :3] = np.clip(row_rgb[:, None, :], 0.0, 1.0)
    return ramp


def classify_skin_material(
    material_name: str,
    texture_path: Path | str,
    part_kind: str = "",
) -> str | None:
    """Classify a `(material, texture)` pair as `'face'`, `'body'`, or `None`."""
    clean_mat = material_name.strip()
    if clean_mat.lower().startswith("scsp_export__"):
        clean_mat = clean_mat[len("scsp_export__") :]
    clean_mat = _COPY_SUFFIX_RE.sub("", clean_mat).casefold()
    tex_stem = Path(texture_path).stem.casefold()
    kind_cf = str(part_kind or "").strip().casefold()

    if any(token in clean_mat for token in _EXCLUDED_MAT_TOKENS):
        return None
    if tex_stem.endswith(_NONCOLOR_TEX_SUFFIXES):
        return None

    if (
        clean_mat == "m_face"
        or clean_mat.startswith("m_face_")
        or (kind_cf == "head" and _FACE_COL_RE.search(tex_stem) is not None and "face" in clean_mat)
    ):
        return "face"

    if (
        "bodyskin" in tex_stem
        or "body_skin" in tex_stem
        or "bodyskin" in clean_mat
        or "body_skin" in clean_mat
        or _SKIN_NAME_RE.search(clean_mat) is not None
    ):
        return "body"

    return None


def resolve_face_shade_path(
    col_path: Path,
    part_source_root: Path | None = None,
) -> tuple[Path | None, str | None]:
    """Locate the matching `*_face_sd.png` texture next to a face color map."""
    col_prefix = col_path.name.split("_", 1)[0].casefold()

    def _pick(candidates: list[Path], scope_label: str) -> tuple[Path | None, str | None]:
        if len(candidates) == 1:
            return candidates[0], None
        if len(candidates) > 1:
            prefixed = [
                p for p in candidates if p.name.split("_", 1)[0].casefold() == col_prefix
            ]
            if len(prefixed) == 1:
                return prefixed[0], None
            return None, (
                f"Ambiguous *_face_sd.png candidates in {scope_label} for {col_path.name}: "
                f"{[p.name for p in candidates]}"
            )
        return None, None

    if col_path.parent.is_dir():
        direct = sorted(
            p for p in col_path.parent.glob("*_face_sd.png") if p.is_file()
        )
        match, warning = _pick(direct, str(col_path.parent))
        if match is not None or warning is not None:
            return match, warning

    if part_source_root is not None and part_source_root.is_dir():
        nested = sorted(
            p for p in part_source_root.rglob("*_face_sd.png") if p.is_file()
        )
        match, warning = _pick(nested, str(part_source_root))
        if match is not None or warning is not None:
            return match, warning

    return None, (
        f"No *_face_sd.png found for {col_path.name}; baked face without face_sd shadow modulation"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_root(sourcepath: Path) -> Path | None:
    if sourcepath.is_dir():
        return sourcepath
    if sourcepath.is_file():
        return sourcepath.parent
    return None


def _resolved_image_path(image: Any) -> Path | None:
    if image is None:
        return None
    try:
        raw = str(getattr(image, "filepath", "") or "").strip()
        if not raw:
            return None
        resolved = Path(bpy.path.abspath(raw)).expanduser().resolve(strict=False)
        return resolved if resolved.is_file() else None
    except (OSError, RuntimeError, ValueError):
        return None


def _primary_color_image_nodes(material: Any) -> list[Any]:
    if not material.use_nodes or material.node_tree is None:
        return []
    tree = material.node_tree
    linked_nodes: list[Any] = []
    seen_pointers: set[int] = set()

    def _collect_upstream(socket: Any, visited_sockets: set[int]) -> None:
        if socket is None or not socket.is_linked:
            return
        sock_ptr = int(socket.as_pointer())
        if sock_ptr in visited_sockets:
            return
        visited_sockets.add(sock_ptr)
        for link in socket.links:
            src = link.from_node
            if src is None:
                continue
            if src.type == "TEX_IMAGE" and src.image is not None:
                ptr = int(src.as_pointer())
                if ptr not in seen_pointers:
                    seen_pointers.add(ptr)
                    linked_nodes.append(src)
            else:
                for inp in getattr(src, "inputs", ()):
                    _collect_upstream(inp, visited_sockets)

    visited: set[int] = set()
    for node in tree.nodes:
        if node.type == "BSDF_PRINCIPLED":
            _collect_upstream(node.inputs.get("Base Color"), visited)
    if linked_nodes:
        return linked_nodes

    fallback: list[Any] = []
    for node in tree.nodes:
        if node.type != "TEX_IMAGE" or node.image is None:
            continue
        path = _resolved_image_path(node.image)
        if path is not None and path.stem.casefold().endswith(_NONCOLOR_TEX_SUFFIXES):
            continue
        fallback.append(node)
    return fallback


def _configure_srgb_image(path: Path) -> Any:
    image = bpy.data.images.load(str(path), check_existing=False)
    image.colorspace_settings.name = "sRGB"
    image.alpha_mode = "STRAIGHT"
    if hasattr(image, "use_alpha"):
        image.use_alpha = True
    return image


def _unique_output_path(output_dir: Path, stem: str, used: set[str]) -> Path:
    candidate_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "skin"
    index = 0
    while True:
        suffix = "" if index == 0 else f"_{index}"
        candidate = (output_dir / f"{candidate_stem}{suffix}.png").resolve()
        key = os.path.normcase(str(candidate))
        if key not in used:
            used.add(key)
            return candidate
        index += 1


def _extract_skin_palette(record: dict[str, Any] | None) -> tuple[dict[str, tuple[str, str]], list[str]]:
    if not isinstance(record, dict):
        return {}, ["config.record missing; skipped character skin adaptation"]
    palette: dict[str, tuple[str, str]] = {}
    warnings: list[str] = []
    for kind, (bright_key, dark_key) in SKIN_RECORD_FIELDS.items():
        bright = str(record.get(bright_key, "") or "").strip()
        dark = str(record.get(dark_key, "") or "").strip()
        if not bright or not dark:
            warnings.append(
                f"Character record missing {bright_key}/{dark_key}; skipped {kind} skin adaptation"
            )
            continue
        try:
            parse_hex_rgb(bright)
            parse_hex_rgb(dark)
        except ValueError as exc:
            warnings.append(f"Invalid skin color for {kind} ({exc}); skipped {kind} adaptation")
            continue
        palette[kind] = (bright, dark)
    return palette, warnings


def prepare(parts: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Adapt SCSP face/body skin diffuse PNGs and generate character toon ramps."""
    if bpy is None:
        raise RuntimeError("skin_processing.prepare requires Blender (bpy)")

    from eye_texture_bake import (
        load_noncolor_pixels,
        resample_bilinear,
        write_noncolor_rgba_png,
    )

    configured_dir = config.get("skin_texture_output_dir")
    if not configured_dir:
        raise ValueError(
            "config.skin_texture_output_dir must be present when skin processing is requested"
        )
    output_dir = Path(configured_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    palette, warnings = _extract_skin_palette(config.get("record"))
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "ok" if palette else "skipped",
        "output_dir": str(output_dir),
        "palette": {
            kind: {"bright": pair[0], "dark": pair[1]}
            for kind, pair in palette.items()
        },
        "toon_ramps": {},
        "adapted_materials": [],
        "skipped_materials": [],
        "warnings": warnings,
    }
    if not palette:
        return report

    used_outputs: set[str] = set()
    owned_images: list[Any] = []
    toon_paths: dict[str, Path] = {}
    baked_cache: dict[tuple[str, str, str], tuple[Path, Any, str | None]] = {}
    seen_mat_ptrs: set[int] = set()

    try:
        for kind, (bright, dark) in palette.items():
            ramp_path = _unique_output_path(output_dir, f"skin_{kind}_toon", used_outputs)
            ramp_pixels = make_toon_ramp_pixels(bright, dark)
            write_noncolor_rgba_png(ramp_path, ramp_pixels, owned_images)
            toon_paths[kind] = ramp_path
            report["toon_ramps"][kind] = {
                "path": str(ramp_path),
                "sha256": _sha256(ramp_path),
                "width": TOON_WIDTH,
                "height": TOON_HEIGHT,
            }

        for part_index, part in enumerate(parts):
            if not isinstance(part, dict):
                continue
            part_kind = str(part.get("kind", "") or "").strip().casefold()
            raw_source = str(part.get("sourcepath", "") or "").strip()
            part_root = _source_root(Path(raw_source).expanduser().resolve(strict=False)) if raw_source else None
            for mesh in list(part.get("meshes") or []):
                if not isinstance(mesh, bpy.types.Object) or mesh.type != "MESH":
                    continue
                for slot in mesh.material_slots:
                    material = slot.material
                    if material is None:
                        continue
                    mat_ptr = int(material.as_pointer())
                    if mat_ptr in seen_mat_ptrs:
                        continue
                    seen_mat_ptrs.add(mat_ptr)

                    nodes = _primary_color_image_nodes(material)
                    if not nodes:
                        continue
                    first_path = _resolved_image_path(nodes[0].image)
                    if first_path is None:
                        warnings.append(
                            f"Material {material.name} image has no file on disk; skipped"
                        )
                        continue

                    skin_kind = classify_skin_material(material.name, first_path, part_kind)
                    if skin_kind is None:
                        report["skipped_materials"].append(
                            {
                                "material": material.name,
                                "part_index": part_index,
                                "part_kind": part_kind,
                                "source_texture": str(first_path),
                                "reason": "non-skin",
                            }
                        )
                        continue
                    if skin_kind not in palette:
                        report["skipped_materials"].append(
                            {
                                "material": material.name,
                                "part_index": part_index,
                                "part_kind": part_kind,
                                "source_texture": str(first_path),
                                "reason": f"missing-{skin_kind}-palette",
                            }
                        )
                        continue

                    bright, dark = palette[skin_kind]
                    node_updates: list[dict[str, Any]] = []
                    for node in nodes:
                        src_path = _resolved_image_path(node.image)
                        if src_path is None:
                            continue
                        shade_path: Path | None = None
                        shade_warning: str | None = None
                        if skin_kind == "face":
                            shade_path, shade_warning = resolve_face_shade_path(
                                src_path, part_root
                            )
                            if shade_warning:
                                warnings.append(shade_warning)

                        cache_key = (
                            skin_kind,
                            os.path.normcase(str(src_path)),
                            os.path.normcase(str(shade_path)) if shade_path else "",
                        )
                        if cache_key not in baked_cache:
                            _, base_pixels = load_noncolor_pixels(src_path, owned_images)
                            shade_pixels: np.ndarray | None = None
                            if shade_path is not None:
                                _, raw_shade = load_noncolor_pixels(
                                    shade_path, owned_images
                                )
                                shade_pixels = resample_bilinear(
                                    raw_shade,
                                    base_pixels.shape[0],
                                    base_pixels.shape[1],
                                )
                            baked_pixels = bake_skin_pixels(
                                base_pixels,
                                bright,
                                dark if shade_pixels is not None else None,
                                shade_pixels,
                            )
                            out_path = _unique_output_path(
                                output_dir,
                                f"{skin_kind}_{src_path.stem}_skin",
                                used_outputs,
                            )
                            write_noncolor_rgba_png(
                                out_path,
                                baked_pixels,
                                owned_images,
                                verify_alpha_from=base_pixels[..., 3],
                            )
                            baked_image = _configure_srgb_image(out_path)
                            baked_cache[cache_key] = (
                                out_path,
                                baked_image,
                                str(shade_path) if shade_path else None,
                            )

                        out_path, baked_image, used_shade = baked_cache[cache_key]
                        previous_image_name = node.image.name if node.image else None
                        # Assigning a dedicated datablock preserves any shared original (e.g. mouth/teeth).
                        node.image = baked_image
                        node_updates.append(
                            {
                                "node": node.name,
                                "previous_image": previous_image_name,
                                "source_texture": str(src_path),
                                "face_sd_texture": used_shade,
                                "baked_texture": str(out_path),
                                "baked_sha256": _sha256(out_path),
                            }
                        )

                    if node_updates:
                        toon_path = toon_paths[skin_kind]
                        material["scsp_skin_kind"] = skin_kind
                        material["scsp_skin_toon_path"] = str(toon_path)
                        report["adapted_materials"].append(
                            {
                                "material": material.name,
                                "part_index": part_index,
                                "part_kind": part_kind,
                                "skin_kind": skin_kind,
                                "toon_texture": str(toon_path),
                                "nodes": node_updates,
                            }
                        )
    finally:
        for image in reversed(owned_images):
            try:
                bpy.data.images.remove(image)
            except (ReferenceError, RuntimeError):
                pass

    return report
