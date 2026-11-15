"""Pure-geometry brow and mouth MMD morph generation for SCSP merged head meshes."""
from __future__ import annotations

import json
from typing import Any

import numpy as np

from eye_morphs import _add_or_replace_key, _key_coords, _material_vertex_sets, _smoothstep
from rename_mmd_shapes import KEY_SOURCE_PROPERTY

SCHEMA = "scsp-facial-morphs/1"


def _source_or_named_key(
    keys: Any,
    resources: dict[str, Any],
    source_id: str,
    default_name: str,
) -> Any | None:
    names = resources.get("source_names", {})
    mapped = names.get(source_id, default_name)
    return keys.key_blocks.get(mapped) or keys.key_blocks.get(source_id)


def _eyebrow_deltas(
    basis: np.ndarray,
    eb_indices: np.ndarray,
    up_delta: np.ndarray | None,
    serious_delta: np.ndarray | None,
) -> list[tuple[str, np.ndarray]]:
    """Generate `にこり` (smily) and `怒り` (angry) on `M_eyebrow` per side."""
    nikori = np.zeros_like(basis, dtype=np.float32)
    ikari = np.zeros_like(basis, dtype=np.float32)

    eb_x = basis[eb_indices, 0]
    side_masks = [
        (eb_indices[eb_x > 1e-4], 1.0),
        (eb_indices[eb_x < -1e-4], -1.0),
    ]
    for side_idx, sign in side_masks:
        if len(side_idx) == 0:
            continue
        ax = np.abs(basis[side_idx, 0])
        x_in = float(ax.min())
        x_out = float(ax.max())
        span = max(x_out - x_in, 1e-4)
        t = np.clip((ax - x_in) / span, 0.0, 1.0).astype(np.float32)
        u_len = span / 0.037  # Normalize to typical ~37mm brow span.

        # `にこり`: warm gentle arch (higher in middle, slightly lifted ends, subtle outward relax).
        arch = np.sin(np.pi * t).astype(np.float32)
        up_lift = (0.0022 + 0.0046 * arch) * min(max(u_len, 0.75), 1.35)
        if up_delta is not None:
            dz_ref = float(np.max(np.abs(up_delta[side_idx, 2])))
            if dz_ref > 1e-4:
                up_dir = up_delta[side_idx] / dz_ref
                step = up_dir * up_lift[:, None]
            else:
                step = np.column_stack([np.zeros_like(t), np.zeros_like(t), up_lift])
        else:
            step = np.column_stack([np.zeros_like(t), np.zeros_like(t), up_lift])
        # Subtle outward relaxation at inner end and tiny forward offset to clear sloped foreheads.
        step[:, 0] += sign * (0.0010 * (1.0 - t))
        step[:, 1] -= 0.0005 * arch
        nikori[side_idx] = step.astype(np.float32)

        # `怒り`: strong inner dip + inward pinch toward nose, raised/sharp outer tail.
        if serious_delta is not None and float(np.max(np.abs(serious_delta[side_idx]))) > 1e-4:
            sd = serious_delta[side_idx].copy()
            # Boost inner drop & pinch while keeping outer tail sharp/raised (less downward dip at tip).
            inner_w = (1.0 - _smoothstep(0.10, 0.78, t))
            outer_w = _smoothstep(0.55, 1.0, t)
            ang_step = sd.copy()
            ang_step[:, 2] *= (1.0 + 0.25 * inner_w)
            ang_step[:, 2] -= 0.0026 * inner_w
            ang_step[:, 2] += 0.0024 * outer_w
            ang_step[:, 0] -= sign * (0.0016 * inner_w)
            # Keep clearance in front of upper eyelid skin.
            ang_step[:, 1] -= 0.0004 * inner_w
            ikari[side_idx] = ang_step.astype(np.float32)
        else:
            # Pure fallback parametric slant.
            z_off = -0.0115 * (1.0 - t) + 0.0065 * t - 0.0018 * np.sin(np.pi * t)
            x_off = -sign * 0.0045 * (1.0 - 0.4 * t)
            ikari[side_idx, 0] = x_off.astype(np.float32)
            ikari[side_idx, 1] = (-0.0006 * (1.0 - t)).astype(np.float32)
            ikari[side_idx, 2] = z_off.astype(np.float32)

    return [("にこり", nikori), ("怒り", ikari)]


def _mouth_region_mask(
    basis: np.ndarray,
    mat_verts: dict[str, np.ndarray],
    mouth_keys: list[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return `(ml_idx, env_indices, skin_weights, tt_idx)` covering the whole coherent mouth pocket."""
    ml_idx = mat_verts.get("M_mouthline", np.empty(0, dtype=np.int32))
    mi_idx = mat_verts.get("M_mouth_inner", np.empty(0, dtype=np.int32))
    fc_idx = mat_verts.get("M_face", np.empty(0, dtype=np.int32))
    tt_idx = mat_verts.get("M_teeth_tongue", np.empty(0, dtype=np.int32))
    if len(ml_idx) == 0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, np.empty(0, dtype=np.float32), tt_idx

    ml_pts = basis[ml_idx]
    x_c = float(0.5 * (ml_pts[:, 0].min() + ml_pts[:, 0].max()))
    w = max(float(0.5 * (ml_pts[:, 0].max() - ml_pts[:, 0].min())), 0.010)
    z_c = float(np.median(ml_pts[:, 2]))
    y_front = float(ml_pts[:, 1].min())

    candidates = np.unique(np.concatenate([ml_idx, mi_idx, fc_idx]))
    pts = basis[candidates]
    dx = np.abs(pts[:, 0] - x_c)
    dz = pts[:, 2] - z_c
    dy = pts[:, 1] - y_front

    # Loose mouth + lower-cheek + chin pocket bounds (never reaches eyes/ears).
    box_mask = (dx <= w + 0.032) & (dz >= -0.032) & (dz <= 0.028) & (dy >= -0.010) & (dy <= 0.040)
    # Also retain any vertex near the mouthline even if slightly outside nominal offsets,
    # while requiring active mouth key support for far cheek vertices when keys exist.
    near_box = (dx <= w + 0.022) & (dz >= -0.024) & (dz <= 0.022)
    if mouth_keys:
        moved = np.zeros(len(basis), dtype=bool)
        for blk in mouth_keys:
            kco = _key_coords(blk)
            moved |= np.linalg.norm(kco - basis, axis=1) > 1e-5
        box_mask &= near_box | moved[candidates]

    env_idx = candidates[box_mask]
    ep = basis[env_idx]

    # Normalized elliptical radial falloff: 1.0 on M_mouthline, smooth decay to 0.0 at cheek/chin/philtrum edges.
    rx = np.maximum(np.abs(ep[:, 0] - x_c) - (w + 0.002), 0.0) / 0.022
    rz_pos = np.maximum(ep[:, 2] - (z_c + 0.004), 0.0) / 0.018
    rz_neg = np.maximum((z_c - 0.004) - ep[:, 2], 0.0) / 0.022
    rz = np.maximum(rz_pos, rz_neg)
    r_horiz = np.sqrt(rx * rx + rz * rz).astype(np.float32)
    env_xz = 1.0 - _smoothstep(0.0, 1.0, r_horiz)
    # Keep deep oral cavity vertices anchored while mouth rim follows lips identically.
    env_y = 1.0 - _smoothstep(0.014, 0.036, (ep[:, 1] - y_front).astype(np.float32))
    env_w = (env_xz * env_y).astype(np.float32)
    # Force M_mouthline to exact 1.0 weight inside lip strip.
    is_ml = np.isin(env_idx, ml_idx)
    env_w[is_ml] = 1.0
    return ml_idx, env_idx, env_w, tt_idx


def _estimate_face_curvature_ky(ml_pts: np.ndarray, x_c: float) -> float:
    """Estimate Y = y0 + ky * (x - x_c)^2 face curve along mouthline."""
    dx2 = (ml_pts[:, 0] - x_c) ** 2
    hi = dx2 > np.quantile(dx2, 0.65)
    lo = dx2 < np.quantile(dx2, 0.35)
    if np.any(hi) and np.any(lo):
        dy = float(np.median(ml_pts[hi, 1]) - np.median(ml_pts[lo, 1]))
        ddx2 = max(float(np.median(dx2[hi]) - np.median(dx2[lo])), 1e-6)
        return float(np.clip(dy / ddx2, 0.0, 60.0))
    return 25.0


def _spatial_mouth_deform(
    ref: np.ndarray,
    basis: np.ndarray,
    env_idx: np.ndarray,
    env_w: np.ndarray,
    ml_idx: np.ndarray,
    disp_fn: Any,
) -> np.ndarray:
    """Apply continuous spatial displacement field on `ref[env_idx]`, returning delta relative to `basis`."""
    p = ref[env_idx]
    disp = disp_fn(p)
    out = ref.copy()
    out[env_idx] = p + disp * env_w[:, None]
    # Keep M_teeth_tongue matching ref exactly on non-env vertices.
    return (out - basis).astype(np.float32)


def _open_lip_midline(ref: np.ndarray, ml_idx: np.ndarray, x_c: float, half_w: float) -> tuple[float, float]:
    """Return `(z_mid, half_h)` of `M_mouthline` opening in an open-mouth key."""
    ml = ref[ml_idx]
    mid_band = ml[np.abs(ml[:, 0] - x_c) <= 0.35 * half_w]
    if len(mid_band) == 0:
        mid_band = ml
    z_top = float(np.quantile(mid_band[:, 2], 0.92))
    z_bot = float(np.quantile(mid_band[:, 2], 0.08))
    return 0.5 * (z_top + z_bot), max(0.5 * (z_top - z_bot), 0.002)


def _mouth_deltas(
    basis: np.ndarray,
    keys: Any,
    resources: dict[str, Any],
    mat_verts: dict[str, np.ndarray],
) -> list[tuple[str, np.ndarray]]:
    """Build standard MMD panel-3 mouth morphs anchored coherently across SCSP lip/inner/teeth meshes."""
    ref_blocks = {
        k: _source_or_named_key(keys, resources, src, jp)
        for k, src, jp in (
            ("A", "A", "あ"),
            ("O", "O", "お"),
            ("serious", "mouth_serious", "真面目口"),
            ("shy", "mouth_shy", "照れ口"),
            ("surp", "mouth_surp", "驚き口"),
        )
    }
    available_blocks = [blk for blk in ref_blocks.values() if blk is not None]
    ml_idx, env_idx, env_w, tt_idx = _mouth_region_mask(basis, mat_verts, available_blocks)
    if len(ml_idx) == 0 or len(env_idx) == 0:
        return []

    coords = {k: _key_coords(blk) for k, blk in ref_blocks.items() if blk is not None}
    ref_closed = coords.get("serious", basis)
    ml_c = ref_closed[ml_idx]
    x_c = float(0.5 * (ml_c[:, 0].min() + ml_c[:, 0].max()))
    w_closed = max(float(0.5 * (ml_c[:, 0].max() - ml_c[:, 0].min())), 0.010)
    z_c = float(np.median(ml_c[:, 2]))
    ky_closed = _estimate_face_curvature_ky(ml_c, x_c)

    # Basis metrics for corner utilities (`口角上げ`, `口角下げ`, `口横広げ`).
    ml_b = basis[ml_idx]
    w_basis = max(float(0.5 * (ml_b[:, 0].max() - ml_b[:, 0].min())), 0.010)
    ky_basis = _estimate_face_curvature_ky(ml_b, x_c)

    results: list[tuple[str, np.ndarray]] = []

    # 1. `ワ`: wide smiling open D-mouth derived cleanly from `あ` (coherent inner + teeth).
    if "A" in coords:
        a_ref = coords["A"]
        ml_a = a_ref[ml_idx]
        w_a = max(float(0.5 * (ml_a[:, 0].max() - ml_a[:, 0].min())), 0.010)
        z_mid_a, half_h_a = _open_lip_midline(a_ref, ml_idx, x_c, w_a)
        ky_a = _estimate_face_curvature_ky(ml_a, x_c)

        def disp_wa(p: np.ndarray) -> np.ndarray:
            dx = p[:, 0] - x_c
            u = np.clip(np.abs(dx) / w_a, 0.0, 1.35)
            up_w = _smoothstep(-0.003, 0.003, p[:, 2] - z_mid_a)
            # Noticeably widen & lift outer corners into cheerful wide D-smile (`ワ` vs `あ`).
            stretch = 0.14 * _smoothstep(0.12, 1.0, u)
            new_dx = dx * (1.0 + stretch)
            y_wrap = ky_a * (new_dx * new_dx - dx * dx) * 0.70
            dz = np.zeros_like(u)
            dz += 0.0028 * _smoothstep(0.35, 1.0, u) * up_w
            dz += 0.0012 * (1.0 - up_w) * (1.0 - 0.30 * _smoothstep(0.0, 1.0, u))
            return np.column_stack([new_dx - dx, y_wrap, dz]).astype(np.float32)

        results.append(("ワ", _spatial_mouth_deform(a_ref, basis, env_idx, env_w, ml_idx, disp_wa)))

    # 2. `にっこり`: warm symmetric happy closed U-smile from `Basis`.
    def disp_nikkori(p: np.ndarray) -> np.ndarray:
        dx = p[:, 0] - x_c
        u = np.clip(np.abs(dx) / w_basis, 0.0, 1.30)
        new_dx = dx * (1.0 + 0.08 * _smoothstep(0.08, 1.0, u))
        y_wrap = ky_basis * (new_dx * new_dx - dx * dx) * 0.85
        curve = -0.0012 * (1.0 - _smoothstep(0.15, 0.70, u)) + 0.0052 * _smoothstep(0.22, 1.05, u) ** 1.25
        return np.column_stack([new_dx - dx, y_wrap, curve]).astype(np.float32)

    results.append(("にっこり", _spatial_mouth_deform(basis, basis, env_idx, env_w, ml_idx, disp_nikkori)))

    # 3. `にやり`: wide sleek symmetric gentle closed smile matching standard MMD `にやり`.
    def disp_niyari(p: np.ndarray) -> np.ndarray:
        dx = p[:, 0] - x_c
        u = np.clip(np.abs(dx) / w_basis, 0.0, 1.30)
        new_dx = dx * (1.0 + 0.14 * _smoothstep(0.05, 1.0, u))
        y_wrap = ky_basis * (new_dx * new_dx - dx * dx) * 0.85
        dz = 0.0032 * _smoothstep(0.28, 1.02, u) ** 1.35 - 0.0003 * (1.0 - _smoothstep(0.0, 0.50, u))
        return np.column_stack([new_dx - dx, y_wrap, dz]).astype(np.float32)

    results.append(("にやり", _spatial_mouth_deform(basis, basis, env_idx, env_w, ml_idx, disp_niyari)))

    # 4. `ω`: closed cat mouth (`:3`) from `真面目口`.
    def disp_omega(p: np.ndarray) -> np.ndarray:
        dx = p[:, 0] - x_c
        u = np.clip(np.abs(dx) / w_closed, 0.0, 1.20)
        new_dx = dx * 0.85
        y_wrap = ky_closed * (new_dx * new_dx - dx * dx) * 0.5
        # Smooth rounded lobes (`bell` sigma 0.28) with center cusp & curled outer tips.
        bell = np.exp(-((u - 0.46) / 0.28) ** 2).astype(np.float32)
        center_peak = np.exp(-(u / 0.24) ** 2).astype(np.float32)
        tip_up = _smoothstep(0.64, 1.02, u)
        dz = 0.0014 * center_peak - 0.0022 * bell + 0.0028 * tip_up
        return np.column_stack([new_dx - dx, y_wrap, dz]).astype(np.float32)

    results.append(("ω", _spatial_mouth_deform(ref_closed, basis, env_idx, env_w, ml_idx, disp_omega)))

    # 5. `ω□`: open cat smile (clean `あ`/`お` base avoids zigzag artifacts from `照れ口`).
    if "A" in coords:
        base_om_open = coords["A"] * 0.85 + coords["O"] * 0.15 if "O" in coords else coords["A"] * 0.85 + basis * 0.15
        ml_oo = base_om_open[ml_idx]
        w_oo = max(float(0.5 * (ml_oo[:, 0].max() - ml_oo[:, 0].min())), 0.010)
        z_mid_oo, half_h_oo = _open_lip_midline(base_om_open, ml_idx, x_c, w_oo)
        ky_oo = _estimate_face_curvature_ky(ml_oo, x_c)

        def disp_omega_open(p: np.ndarray) -> np.ndarray:
            dx = p[:, 0] - x_c
            u = np.clip(np.abs(dx) / w_oo, 0.0, 1.25)
            up_w = _smoothstep(-0.0025, 0.0028, p[:, 2] - z_mid_oo)
            new_dx = dx * 0.86
            y_wrap = ky_oo * (new_dx * new_dx - dx * dx) * 0.5
            # Smooth rounded `ω` upper contour without jagged spikes.
            bell = np.exp(-((u - 0.45) / 0.28) ** 2).astype(np.float32)
            center_peak = np.exp(-(u / 0.26) ** 2).astype(np.float32)
            tip_up = _smoothstep(0.65, 1.02, u)
            cat_wave = (0.0016 * center_peak - 0.0022 * bell + 0.0024 * tip_up) * (0.25 + 0.75 * up_w)
            bot_round = 0.0024 * (1.0 - up_w) * (1.0 - 0.40 * _smoothstep(0.25, 1.0, u))
            return np.column_stack([new_dx - dx, y_wrap, cat_wave + bot_round]).astype(np.float32)

        delta_oo = _spatial_mouth_deform(base_om_open, basis, env_idx, env_w, ml_idx, disp_omega_open)
        if len(tt_idx):
            tt_z_mid = z_mid_oo + 0.001
            hi_tt = tt_idx[base_om_open[tt_idx, 2] > tt_z_mid]
            if len(hi_tt):
                delta_oo[hi_tt, 1] += 0.0018
                delta_oo[hi_tt, 2] -= 0.0010
        results.append(("ω□", delta_oo))

    # 6. `▲`: open triangle mouth (peaked narrow top, flatter wider bottom).
    if "A" in coords and "surp" in coords:
        base_tri = 0.55 * coords["A"] + 0.45 * coords["surp"]
        ml_tr = base_tri[ml_idx]
        w_tr = max(float(0.5 * (ml_tr[:, 0].max() - ml_tr[:, 0].min())), 0.008)
        z_mid_tr, _ = _open_lip_midline(base_tri, ml_idx, x_c, w_tr)

        def disp_tri(p: np.ndarray) -> np.ndarray:
            dx = p[:, 0] - x_c
            u = np.clip(np.abs(dx) / w_tr, 0.0, 1.30)
            up_w = _smoothstep(-0.0030, 0.0030, p[:, 2] - z_mid_tr)
            # Narrow top half, gently flare bottom half outward into triangle base.
            x_scale = 0.80 * up_w + 1.10 * (1.0 - up_w)
            new_dx = dx * x_scale
            # Lift center apex on upper lip, pull outer shoulders gently down.
            dz_up = (0.0026 * np.exp(-(u / 0.45) ** 2) - 0.0014 * _smoothstep(0.45, 1.0, u)) * up_w
            # Flatten bottom base horizontally.
            dz_bot = 0.0020 * (1.0 - up_w) * (1.0 - _smoothstep(0.35, 1.0, u))
            return np.column_stack([new_dx - dx, np.zeros_like(dx), dz_up + dz_bot]).astype(np.float32)

        results.append(("▲", _spatial_mouth_deform(base_tri, basis, env_idx, env_w, ml_idx, disp_tri)))

    # 7. `∧`: closed inverted-V worried mouth from `真面目口`.
    def disp_mountain(p: np.ndarray) -> np.ndarray:
        dx = p[:, 0] - x_c
        u = np.clip(np.abs(dx) / w_closed, 0.0, 1.25)
        new_dx = dx * 0.82
        y_wrap = ky_closed * (new_dx * new_dx - dx * dx) * 0.5
        dz = 0.0032 * np.exp(-(u / 0.48) ** 2) - 0.0044 * _smoothstep(0.28, 1.0, u)
        return np.column_stack([new_dx - dx, y_wrap, dz]).astype(np.float32)

    results.append(("∧", _spatial_mouth_deform(ref_closed, basis, env_idx, env_w, ml_idx, disp_mountain)))

    # 8. `□`: rounded rectangular open mouth from `お` + `あ`.
    if "O" in coords:
        base_sq = 0.65 * coords["O"] + 0.35 * coords["A"] if "A" in coords else coords["O"]
        ml_sq = base_sq[ml_idx]
        w_sq = max(float(0.5 * (ml_sq[:, 0].max() - ml_sq[:, 0].min())), 0.008)
        z_mid_sq, half_h_sq = _open_lip_midline(base_sq, ml_idx, x_c, w_sq)

        def disp_square(p: np.ndarray) -> np.ndarray:
            dx = p[:, 0] - x_c
            u = np.clip(np.abs(dx) / w_sq, 0.0, 1.35)
            v_norm = np.clip((p[:, 2] - z_mid_sq) / max(half_h_sq, 0.003), -1.3, 1.3)
            # Push top/bottom outer shoulders outward in X to square off oval `お` -> `□`.
            shoulder_x = _smoothstep(0.12, 0.82, u) * (1.0 - _smoothstep(0.95, 1.35, u))
            vert_box = _smoothstep(0.20, 0.88, np.abs(v_norm))
            new_dx = dx * (1.16 + 0.28 * shoulder_x * vert_box)
            # Gently flatten arched top/bottom centers slightly toward square top/bottom edges.
            center_flat = (1.0 - _smoothstep(0.25, 0.85, u)) * _smoothstep(0.60, 1.05, np.abs(v_norm))
            dz = -0.0022 * np.sign(v_norm) * center_flat
            return np.column_stack([new_dx - dx, np.zeros_like(dx), dz]).astype(np.float32)

        results.append(("□", _spatial_mouth_deform(base_sq, basis, env_idx, env_w, ml_idx, disp_square)))

    # 9. `ん`: closed pursed/flat N mouth from `真面目口`.
    def disp_n(p: np.ndarray) -> np.ndarray:
        dx = p[:, 0] - x_c
        u = np.clip(np.abs(dx) / w_closed, 0.0, 1.25)
        new_dx = dx * 0.86
        y_wrap = ky_closed * (new_dx * new_dx - dx * dx) * 0.5 + 0.0008 * (1.0 - _smoothstep(0.5, 1.1, u))
        dz = -0.0006 * _smoothstep(0.45, 1.0, u)
        return np.column_stack([new_dx - dx, y_wrap, dz]).astype(np.float32)

    results.append(("ん", _spatial_mouth_deform(ref_closed, basis, env_idx, env_w, ml_idx, disp_n)))

    # 10. `口角上げ` & 11. `口角下げ` & 12. `口横広げ`: additive utility sliders from `Basis`.
    def disp_corner_up(p: np.ndarray) -> np.ndarray:
        dx = p[:, 0] - x_c
        u = np.clip(np.abs(dx) / w_basis, 0.0, 1.30)
        new_dx = dx * (1.0 + 0.04 * _smoothstep(0.35, 1.0, u))
        y_wrap = ky_basis * (new_dx * new_dx - dx * dx)
        dz = 0.0028 * _smoothstep(0.32, 1.0, u) ** 1.15
        return np.column_stack([new_dx - dx, y_wrap, dz]).astype(np.float32)

    def disp_corner_down(p: np.ndarray) -> np.ndarray:
        dx = p[:, 0] - x_c
        u = np.clip(np.abs(dx) / w_basis, 0.0, 1.30)
        dz = -0.0072 * _smoothstep(0.40, 1.05, u) ** 1.20
        return np.column_stack([np.zeros_like(dx), np.zeros_like(dx), dz]).astype(np.float32)

    def disp_widen(p: np.ndarray) -> np.ndarray:
        dx = p[:, 0] - x_c
        new_dx = dx * 1.18
        y_wrap = ky_basis * (new_dx * new_dx - dx * dx) * 0.85
        return np.column_stack([new_dx - dx, y_wrap, np.zeros_like(dx)]).astype(np.float32)

    results.append(("口角上げ", _spatial_mouth_deform(basis, basis, env_idx, env_w, ml_idx, disp_corner_up)))
    results.append(("口角下げ", _spatial_mouth_deform(basis, basis, env_idx, env_w, ml_idx, disp_corner_down)))
    results.append(("口横広げ", _spatial_mouth_deform(basis, basis, env_idx, env_w, ml_idx, disp_widen)))

    return results


def generate(meshes: list[Any], resources: dict[str, Any]) -> dict[str, Any]:
    """Generate brow (`panel=1`) and mouth (`panel=3`) MMD morphs on merged `head` meshes."""
    report: dict[str, Any] = {"schema": SCHEMA, "generated": []}
    for mesh in meshes:
        keys = getattr(mesh.data, "shape_keys", None)
        if keys is None or keys.key_blocks.get("Basis") is None:
            continue
        mat_verts = _material_vertex_sets(mesh)
        has_brow = "M_eyebrow" in mat_verts
        has_mouth = "M_mouthline" in mat_verts
        if not has_brow and not has_mouth:
            continue

        basis_block = keys.key_blocks["Basis"]
        basis = _key_coords(basis_block)
        raw_marker = keys.get(KEY_SOURCE_PROPERTY)
        marker = json.loads(raw_marker) if raw_marker else {}

        mesh_generated: list[dict[str, Any]] = []
        if has_brow:
            up_blk = keys.key_blocks.get("上")
            ser_blk = keys.key_blocks.get("真面目")
            up_d = (_key_coords(up_blk) - basis) if up_blk is not None else None
            ser_d = (_key_coords(ser_blk) - basis) if ser_blk is not None else None
            for morph_name, delta in _eyebrow_deltas(basis, mat_verts["M_eyebrow"], up_d, ser_d):
                mesh_generated.append(_add_or_replace_key(mesh, basis_block, basis, morph_name, delta))
                marker[morph_name] = f"@generated:{morph_name}"

        if has_mouth:
            for morph_name, delta in _mouth_deltas(basis, keys, resources, mat_verts):
                mesh_generated.append(_add_or_replace_key(mesh, basis_block, basis, morph_name, delta))
                marker[morph_name] = f"@generated:{morph_name}"

        keys[KEY_SOURCE_PROPERTY] = json.dumps(marker, ensure_ascii=False, separators=(",", ":"))
        report["generated"].append({"mesh": mesh.name, "morphs": mesh_generated})
    return report
