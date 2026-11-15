"""Bake an SCSP eye highlight layer into one RGBA eye texture.

Run this module in Blender 5.3. It uses Blender's image pixel access and the
NumPy package bundled with Blender; Pillow is not required.
"""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile
from pathlib import Path
from typing import Any

import bpy
import numpy as np


__all__ = ["bake_eye_texture"]

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_METHOD = {
    "name": "straight-alpha source-over RGB with preserved base alpha",
    "rgb_equation": "out.rgb = base.rgb * (1 - high.a) + high.rgb * high.a",
    "alpha_equation": "out.a = base.a",
    "highlight_alpha_policy": "Use the source PNG alpha as the blend weight; do not synthesize a mask and do not brighten the whole image.",
    "color_handling": "Load inputs as Blender Non-Color and composite in PNG sample units without implicit sRGB conversion, tint, desaturation, or white fill.",
    "uv_mapping": "Treat both images as the same normalized UV layout; bilinearly resample highlight channels to the base pixel grid when dimensions differ.",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_info(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        if handle.read(8) != _PNG_SIGNATURE:
            raise ValueError(f"Not a PNG file: {path}")

        width = height = bit_depth = color_type = None
        has_trns = False
        while True:
            length_bytes = handle.read(4)
            chunk_type_bytes = handle.read(4)
            if len(length_bytes) != 4 or len(chunk_type_bytes) != 4:
                raise ValueError(f"Truncated PNG chunk stream: {path}")
            length = struct.unpack(">I", length_bytes)[0]
            chunk_type = chunk_type_bytes.decode("ascii", errors="replace")
            data = handle.read(length)
            if len(data) != length:
                raise ValueError(f"Truncated PNG chunk {chunk_type}: {path}")
            handle.read(4)

            if chunk_type == "IHDR":
                if length != 13:
                    raise ValueError(f"Invalid PNG IHDR: {path}")
                (
                    width,
                    height,
                    bit_depth,
                    color_type,
                    compression,
                    filtering,
                    interlace,
                ) = struct.unpack(">IIBBBBB", data)
                if compression != 0 or filtering != 0 or interlace not in (0, 1):
                    raise ValueError(f"Unsupported PNG coding fields: {path}")
            elif chunk_type == "tRNS":
                has_trns = True
            elif chunk_type == "IDAT":
                break
            elif chunk_type == "IEND":
                break

    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError(f"PNG has no IHDR: {path}")
    if width <= 0 or height <= 0:
        raise ValueError(f"PNG has invalid dimensions: {path}")

    if color_type in (4, 6):
        alpha_source = "channel"
    elif has_trns:
        alpha_source = "tRNS"
    else:
        alpha_source = None

    return {
        "width": int(width),
        "height": int(height),
        "bit_depth": int(bit_depth),
        "color_type": int(color_type),
        "alpha_source": alpha_source,
    }


def _alpha_stats(pixels: np.ndarray) -> dict[str, Any]:
    alpha = np.clip(pixels[..., 3], 0.0, 1.0)
    alpha_8bit = np.rint(alpha * 255.0).astype(np.uint8)
    return {
        "min": float(alpha.min()),
        "max": float(alpha.max()),
        "mean": float(alpha.mean()),
        "unique_8bit": int(np.unique(alpha_8bit).size),
        "fully_transparent_pixels": int(np.count_nonzero(alpha_8bit == 0)),
        "fully_opaque_pixels": int(np.count_nonzero(alpha_8bit == 255)),
        "pixel_count": int(alpha_8bit.size),
    }


def load_noncolor_pixels(path: Path, owned_images: list[Any]) -> tuple[Any, np.ndarray]:
    """Decode an image in Blender as Non-Color STRAIGHT float32 pixels."""
    image = bpy.data.images.load(str(path), check_existing=False)
    owned_images.append(image)
    image.alpha_mode = "STRAIGHT"
    image.colorspace_settings.name = "Non-Color"
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Blender could not decode image dimensions: {path}")
    values = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(values)
    return image, values.reshape((height, width, 4)).copy()


_load_image = load_noncolor_pixels


def resample_bilinear(
    source: np.ndarray,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Bilinearly resample an (H, W, C) float32 image onto target dimensions."""
    source_height, source_width = source.shape[:2]
    if (source_height, source_width) == (target_height, target_width):
        return source.copy()

    source_y = (
        (np.arange(target_height, dtype=np.float32) + 0.5)
        * source_height
        / target_height
        - 0.5
    )
    source_x = (
        (np.arange(target_width, dtype=np.float32) + 0.5)
        * source_width
        / target_width
        - 0.5
    )

    x_floor = np.floor(source_x)
    y_floor = np.floor(source_y)
    x0 = np.clip(x_floor.astype(np.int32), 0, source_width - 1)
    y0 = np.clip(y_floor.astype(np.int32), 0, source_height - 1)
    x1 = np.clip(x0 + 1, 0, source_width - 1)
    y1 = np.clip(y0 + 1, 0, source_height - 1)
    x_weight = np.clip(source_x - x_floor, 0.0, 1.0)[None, :, None]
    y_weight = np.clip(source_y - y_floor, 0.0, 1.0)[:, None, None]

    row0 = source[y0]
    row1 = source[y1]
    top = (
        row0[:, x0, :] * (1.0 - x_weight)
        + row0[:, x1, :] * x_weight
    )
    bottom = (
        row1[:, x0, :] * (1.0 - x_weight)
        + row1[:, x1, :] * x_weight
    )
    return top * (1.0 - y_weight) + bottom * y_weight


_resample_bilinear = resample_bilinear


def write_noncolor_rgba_png(
    output_path: Path,
    pixels: np.ndarray,
    owned_images: list[Any],
    *,
    verify_alpha_from: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Atomically write (H, W, 4) float32 RGBA data as an 8-bit Non-Color PNG."""
    clipped = np.clip(pixels, 0.0, 1.0).astype(np.float32)
    height, width = clipped.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(
        prefix=f".{output_path.stem}.",
        suffix=".png",
        dir=output_path.parent,
        delete=False,
    )
    temp_path = Path(temporary.name)
    temporary.close()
    try:
        result_image = bpy.data.images.new(
            f"{output_path.stem}_bake",
            width=width,
            height=height,
            alpha=True,
            float_buffer=False,
        )
        owned_images.append(result_image)
        result_image.alpha_mode = "STRAIGHT"
        result_image.colorspace_settings.name = "Non-Color"
        result_image.pixels.foreach_set(clipped.ravel())
        result_image.update()
        result_image.filepath_raw = str(temp_path)
        result_image.file_format = "PNG"
        result_image.save()

        output_png = _png_info(temp_path)
        if (
            output_png["width"] != width
            or output_png["height"] != height
            or output_png["bit_depth"] != 8
            or output_png["color_type"] != 6
            or output_png["alpha_source"] != "channel"
        ):
            raise RuntimeError(
                f"Blender did not write the required 8-bit RGBA PNG: {output_png}"
            )

        _, roundtrip_pixels = load_noncolor_pixels(temp_path, owned_images)
        if verify_alpha_from is not None:
            expected_alpha = np.rint(
                np.clip(verify_alpha_from, 0.0, 1.0) * 255.0
            ).astype(np.uint8)
            output_alpha_8bit = np.rint(roundtrip_pixels[..., 3] * 255.0).astype(
                np.uint8
            )
            if not np.array_equal(expected_alpha, output_alpha_8bit):
                maximum_error = int(
                    np.max(
                        np.abs(expected_alpha.astype(np.int16) - output_alpha_8bit)
                    )
                )
                raise RuntimeError(
                    f"output alpha changed during PNG save; maximum 8-bit error={maximum_error}"
                )

        os.replace(temp_path, output_path)
        return _png_info(output_path), roundtrip_pixels
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _same_path(left: Path, right: Path) -> bool:
    if os.path.normcase(str(left)) == os.path.normcase(str(right)):
        return True
    if left.exists() and right.exists():
        try:
            return os.path.samefile(left, right)
        except OSError:
            return False
    return False


def _normalize_path(value: str | os.PathLike[str], label: str) -> Path:
    path = Path(value).expanduser().resolve(strict=False)
    if not str(path):
        raise ValueError(f"{label} path is empty")
    return path


def _base_report(
    base_path: Path,
    highlight_path: Path,
    output_path: Path,
    base_png: dict[str, Any],
    highlight_png: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "scsp-eye-texture-bake/1",
        "status": "ok",
        "blender_version": bpy.app.version_string,
        "inputs": {
            "base": {
                "path": str(base_path),
                "sha256": _sha256(base_path),
                "png": base_png,
            },
            "highlight": {
                "path": str(highlight_path),
                "sha256": _sha256(highlight_path),
                "png": highlight_png,
            },
        },
        "method": dict(_METHOD),
        "output": {
            "path": str(output_path),
            "sha256": None,
            "png": None,
        },
        "validation": {},
        "warnings": [
            "The public signature has no material _EyeOffset.zw, highlight shift, or jitter arguments. Callers must use images with aligned normalized UV layouts; animated or material-shifted highlights are outside this bake.",
        ],
    }


def _unsupported(
    report: dict[str, Any],
    reason: str,
    base_alpha: dict[str, Any] | None = None,
    highlight_alpha: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report["status"] = "unsupported"
    report["output"] = None
    report["reason"] = reason
    report["validation"] = {
        "base_alpha": base_alpha,
        "highlight_alpha": highlight_alpha,
        "output_written": False,
    }
    return report


def bake_eye_texture(
    base_path: str | os.PathLike[str],
    highlight_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Merge an eye highlight into an eye-colour PNG and return a JSON report.

    The output keeps the base image's alpha byte-for-byte. A highlight without
    an explicit, varying source alpha is reported as unsupported and is not
    silently converted into a luminance mask.
    """

    base_path = _normalize_path(base_path, "base")
    highlight_path = _normalize_path(highlight_path, "highlight")
    output_path = _normalize_path(output_path, "output")

    for path, label in ((base_path, "base"), (highlight_path, "highlight")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} PNG does not exist: {path}")
        if path.suffix.lower() != ".png":
            raise ValueError(f"{label} input must be a PNG: {path}")
    if output_path.suffix.lower() != ".png":
        raise ValueError(f"output path must end in .png: {output_path}")
    if _same_path(output_path, base_path) or _same_path(output_path, highlight_path):
        raise ValueError("output path must not overwrite either input PNG")
    if _same_path(base_path, highlight_path):
        raise ValueError("base and highlight paths must refer to different files")

    base_png = _png_info(base_path)
    highlight_png = _png_info(highlight_path)
    report = _base_report(
        base_path,
        highlight_path,
        output_path,
        base_png,
        highlight_png,
    )

    owned_images: list[Any] = []
    temp_path: Path | None = None
    try:
        _, base_pixels = _load_image(base_path, owned_images)
        _, highlight_pixels = _load_image(highlight_path, owned_images)
        base_alpha = _alpha_stats(base_pixels)
        highlight_alpha = _alpha_stats(highlight_pixels)
        report["inputs"]["base"]["alpha"] = base_alpha
        report["inputs"]["highlight"]["alpha"] = highlight_alpha

        if base_png["alpha_source"] is None:
            return _unsupported(
                report,
                "base texture has no explicit alpha source",
                base_alpha,
                highlight_alpha,
            )
        if highlight_png["alpha_source"] is None:
            return _unsupported(
                report,
                "highlight texture has no explicit alpha channel or tRNS transparency",
                base_alpha,
                highlight_alpha,
            )
        if highlight_alpha["max"] <= 0.0:
            return _unsupported(
                report,
                "highlight alpha is fully transparent",
                base_alpha,
                highlight_alpha,
            )
        if highlight_alpha["unique_8bit"] <= 1:
            return _unsupported(
                report,
                "highlight alpha has no usable spatial variation; no mask rule was supplied",
                base_alpha,
                highlight_alpha,
            )

        height, width = base_pixels.shape[:2]
        resized_highlight = _resample_bilinear(highlight_pixels, height, width)
        highlight_weight = np.clip(resized_highlight[..., 3:4], 0.0, 1.0)
        output_pixels = base_pixels.copy()
        output_pixels[..., :3] = (
            base_pixels[..., :3] * (1.0 - highlight_weight)
            + resized_highlight[..., :3] * highlight_weight
        )
        output_pixels[..., 3] = base_pixels[..., 3]
        output_pixels = np.clip(output_pixels, 0.0, 1.0).astype(np.float32)

        output_info, roundtrip_pixels = write_noncolor_rgba_png(
            output_path,
            output_pixels,
            owned_images,
            verify_alpha_from=base_pixels[..., 3],
        )
        base_alpha_8bit = np.rint(base_pixels[..., 3] * 255.0).astype(np.uint8)
        output_alpha_8bit = np.rint(roundtrip_pixels[..., 3] * 255.0).astype(np.uint8)
        output_sha256 = _sha256(output_path)
        output_alpha = _alpha_stats(roundtrip_pixels)

        for source_path, original_hash in (
            (base_path, report["inputs"]["base"]["sha256"]),
            (highlight_path, report["inputs"]["highlight"]["sha256"]),
        ):
            if _sha256(source_path) != original_hash:
                raise RuntimeError(f"input changed during bake: {source_path}")

        report["output"] = {
            "path": str(output_path),
            "sha256": output_sha256,
            "png": output_info,
            "alpha": output_alpha,
        }
        report["validation"] = {
            "output_written": True,
            "dimensions": {
                "base": [int(width), int(height)],
                "highlight": [
                    int(highlight_png["width"]),
                    int(highlight_png["height"]),
                ],
                "output": [
                    int(output_info["width"]),
                    int(output_info["height"]),
                ],
            },
            "highlight_resampled_to_base_grid": (
                highlight_png["width"] != width
                or highlight_png["height"] != height
            ),
            "base_alpha_preserved_8bit": True,
            "base_visible_pixels": int(
                np.count_nonzero(base_alpha_8bit > 0)
            ),
            "output_visible_pixels": int(
                np.count_nonzero(output_alpha_8bit > 0)
            ),
            "contour_expanded": bool(
                np.any((output_alpha_8bit > 0) & (base_alpha_8bit == 0))
            ),
            "inputs_unchanged": True,
            "output_is_rgba_png": True,
        }
        return report
    finally:
        for image in reversed(owned_images):
            try:
                bpy.data.images.remove(image)
            except (ReferenceError, RuntimeError):
                pass
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
