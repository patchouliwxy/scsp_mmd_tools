"""Procedural RGBA atlas generator for SCSP MMD stencil eye effects."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TOOLS_DIR = str(Path(__file__).resolve().parent)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from eye_texture_bake import write_noncolor_rgba_png

ATLAS_SIZE = 1024
DEFAULT_ATLAS_PATH = Path(__file__).resolve().parents[1] / "Resources" / "scsp_eye_effects_atlas.png"

# UV rectangles (u_min, v_min, u_max, v_max) with generous transparent padding around each 512x512 tile.
ATLAS_UV_RECTS: dict[str, tuple[float, float, float, float]] = {
    "星目": (0.01, 0.51, 0.49, 0.99),
    "はぁと": (0.51, 0.51, 0.99, 0.99),
    "はちゅ目": (0.01, 0.01, 0.49, 0.49),
    "はぅ": (0.51, 0.01, 0.99, 0.49),
}


def _smoothstep(edge0: float | np.ndarray, edge1: float | np.ndarray, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / np.maximum(edge1 - edge0, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _segment_distance(
    px: np.ndarray,
    py: np.ndarray,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> np.ndarray:
    abx = bx - ax
    aby = by - ay
    denom = max(abx * abx + aby * aby, 1e-8)
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / denom, 0.0, 1.0)
    dx = px - (ax + abx * t)
    dy = py - (ay + aby * t)
    return np.sqrt(dx * dx + dy * dy).astype(np.float32)


def _quadrant_grid(tile_size: int) -> tuple[np.ndarray, np.ndarray]:
    # qx in [-1, 1] left->right, qy in [-1, 1] bottom->top matching Blender image/UV rows.
    coords = np.linspace(-1.0, 1.0, tile_size, dtype=np.float32)
    qx, qy = np.meshgrid(coords, coords)
    return qx, qy


def _quad_curve_distance(
    px: np.ndarray,
    py: np.ndarray,
    x0: float,
    y0: float,
    xc: float,
    yc: float,
    x1: float,
    y1: float,
    samples: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(min_distance, curve_t)` to a sampled quadratic Bezier curve."""
    ts = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    bx = (1.0 - ts) ** 2 * x0 + 2.0 * (1.0 - ts) * ts * xc + ts ** 2 * x1
    by = (1.0 - ts) ** 2 * y0 + 2.0 * (1.0 - ts) * ts * yc + ts ** 2 * y1
    min_d2 = np.full(px.shape, 1e9, dtype=np.float32)
    best_t = np.zeros(px.shape, dtype=np.float32)
    for i in range(samples - 1):
        ax, ay = float(bx[i]), float(by[i])
        cx, cy = float(bx[i + 1]), float(by[i + 1])
        abx, aby = cx - ax, cy - ay
        denom = max(abx * abx + aby * aby, 1e-8)
        seg_t = np.clip(((px - ax) * abx + (py - ay) * aby) / denom, 0.0, 1.0)
        dx = px - (ax + abx * seg_t)
        dy = py - (ay + aby * seg_t)
        d2 = dx * dx + dy * dy
        hit = d2 < min_d2
        min_d2 = np.where(hit, d2, min_d2)
        best_t = np.where(hit, float(ts[i]) + (float(ts[i + 1]) - float(ts[i])) * seg_t, best_t)
    return np.sqrt(min_d2).astype(np.float32), best_t


def _render_star_tile(tile_size: int) -> np.ndarray:
    qx, qy = _quadrant_grid(tile_size)
    # Four-point concave sparkle star (clean single star, no extra corner dots or noisy fringe).
    star_metric = (np.abs(qx) / 0.72) ** 0.52 + (np.abs(qy) / 0.82) ** 0.52
    alpha = 1.0 - _smoothstep(0.95, 1.01, star_metric)
    core_mix = 1.0 - _smoothstep(0.50, 0.84, star_metric)

    rgb = np.empty((tile_size, tile_size, 3), dtype=np.float32)
    gold = np.array([1.0, 0.89, 0.34], dtype=np.float32)
    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    rgb[:] = gold[None, None, :] * (1.0 - core_mix[..., None]) + white[None, None, :] * core_mix[..., None]
    return np.clip(np.dstack([rgb, alpha]), 0.0, 1.0).astype(np.float32)


def _render_heart_tile(tile_size: int) -> np.ndarray:
    qx, qy = _quadrant_grid(tile_size)
    nx = qx / 0.70
    ny = (qy + 0.12) / 0.70
    heart_metric = nx * nx + (1.15 * ny - 0.68 * (np.abs(nx) ** 0.70)) ** 2
    alpha = 1.0 - _smoothstep(0.58, 0.65, heart_metric)
    rim_mix = _smoothstep(0.28, 0.60, heart_metric)

    hl_dist = np.sqrt(((qx + 0.25) / 0.16) ** 2 + ((qy - 0.22) / 0.12) ** 2)
    hl_mix = (1.0 - _smoothstep(0.35, 1.0, hl_dist)) * 0.65

    inner_pink = np.array([1.0, 0.50, 0.68], dtype=np.float32)
    rim_pink = np.array([0.96, 0.24, 0.48], dtype=np.float32)
    shine = np.array([1.0, 0.90, 0.94], dtype=np.float32)

    rgb = inner_pink[None, None, :] * (1.0 - rim_mix[..., None]) + rim_pink[None, None, :] * rim_mix[..., None]
    rgb = rgb * (1.0 - hl_mix[..., None]) + shine[None, None, :] * hl_mix[..., None]
    return np.clip(np.dstack([rgb, alpha]), 0.0, 1.0).astype(np.float32)


def _render_hachume_tile(tile_size: int) -> np.ndarray:
    qx, qy = _quadrant_grid(tile_size)
    r = np.sqrt(qx * qx + qy * qy)
    theta = np.arctan2(qy, qx)
    wobble = 0.008 * np.sin(5.0 * theta + 0.4) + 0.005 * np.cos(9.0 * theta - 0.7)
    rw = r + wobble

    outer_alpha = 1.0 - _smoothstep(0.78, 0.82, rw)
    white_disc = 1.0 - _smoothstep(0.65, 0.69, rw)

    dark = np.array([0.13, 0.10, 0.13], dtype=np.float32)
    white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    rgb = dark[None, None, :] * (1.0 - white_disc[..., None]) + white[None, None, :] * white_disc[..., None]
    return np.clip(np.dstack([rgb, outer_alpha]), 0.0, 1.0).astype(np.float32)


def _render_hau_tile(tile_size: int) -> np.ndarray:
    qx, qy = _quadrant_grid(tile_size)
    # Bold comic `>` brush strokes: gently bowed horizontal arms meeting cleanly at inner apex.
    d_top, t_top = _quad_curve_distance(qx, qy, -0.82, 0.48, -0.04, 0.25, 0.74, 0.0)
    d_bot, t_bot = _quad_curve_distance(qx, qy, -0.82, -0.48, -0.04, -0.25, 0.74, 0.0)

    r_top = 0.064 + 0.038 * np.sin(np.pi * np.clip(t_top, 0.0, 0.95))
    r_bot = 0.064 + 0.038 * np.sin(np.pi * np.clip(t_bot, 0.0, 0.95))
    a_top = 1.0 - _smoothstep(r_top - 0.014, r_top + 0.014, d_top)
    a_bot = 1.0 - _smoothstep(r_bot - 0.014, r_bot + 0.014, d_bot)
    alpha = np.clip(np.maximum(a_top, a_bot), 0.0, 1.0)

    stroke_col = np.array([0.16, 0.10, 0.13], dtype=np.float32)
    rgb = np.broadcast_to(stroke_col[None, None, :], (tile_size, tile_size, 3)).copy()
    return np.clip(np.dstack([rgb, alpha]), 0.0, 1.0).astype(np.float32)


def build_eye_effects_atlas(size: int = ATLAS_SIZE) -> np.ndarray:
    """Build the 2x2 RGBA stencil atlas image array (`size` x `size` x 4 float32 in bottom-up order)."""
    half = size // 2
    canvas = np.zeros((size, size, 4), dtype=np.float32)
    # Bottom half in Blender array (rows 0..half) = UV v in [0.0, 0.5].
    canvas[0:half, 0:half] = _render_hachume_tile(half)
    canvas[0:half, half:size] = _render_hau_tile(half)
    # Top half in Blender array (rows half..size) = UV v in [0.5, 1.0].
    canvas[half:size, 0:half] = _render_star_tile(half)
    canvas[half:size, half:size] = _render_heart_tile(half)
    return canvas


def ensure_atlas_file(path: Path | None = None) -> Path:
    """Write the preset atlas PNG and return its resolved path."""
    import bpy

    target = Path(path or DEFAULT_ATLAS_PATH).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    atlas = build_eye_effects_atlas(ATLAS_SIZE)
    owned: list[bpy.types.Image] = []
    try:
        write_noncolor_rgba_png(target, atlas, owned)
    finally:
        for img in reversed(owned):
            bpy.data.images.remove(img)
    return target


if __name__ == "__main__":
    written = ensure_atlas_file()
    print("WROTE_ATLAS:", written)
