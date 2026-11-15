"""Pure-geometry eye morph generation and cleanup for SCSP merged head meshes."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import bpy
import numpy as np

from eye_effect_cards import attach_hidden_cards, generate_effect_morphs
from rename_mmd_shapes import KEY_SOURCE_PROPERTY

SCHEMA = "scsp-eye-morphs/1"

_PUPIL_SPECS: tuple[tuple[str, float, float], ...] = (
    ("瞳小", 0.44, 0.44),
    ("瞳縦潰れ", 0.88, 0.68),
)


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize zero-length vector")
    return vec / norm


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _key_coords(block: Any) -> np.ndarray:
    count = len(block.data)
    values = np.empty(count * 3, dtype=np.float32)
    block.data.foreach_get("co", values)
    return values.reshape((count, 3))


def _material_vertex_sets(mesh: Any) -> dict[str, np.ndarray]:
    materials = [slot.material.name if slot.material else "" for slot in mesh.material_slots]
    by_name: dict[str, set[int]] = {name: set() for name in materials if name}
    for poly in mesh.data.polygons:
        if poly.material_index < len(materials):
            name = materials[poly.material_index]
            if name in by_name:
                by_name[name].update(poly.vertices)
    return {
        name: np.array(sorted(indices), dtype=np.int32)
        for name, indices in by_name.items()
        if indices
    }


def _eyeball_pivots(basis: np.ndarray, mat_verts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    pivots: dict[str, np.ndarray] = {}
    armature = bpy.data.objects.get("SCSP_Skeleton")
    bone_map = {"M_eye_L": "左目", "M_eye_R": "右目"}
    for mat_name, bone_name in bone_map.items():
        if armature is not None and armature.type == "ARMATURE":
            bone = armature.data.bones.get(bone_name)
            if bone is not None:
                pivots[mat_name] = np.array(bone.head_local, dtype=np.float32)
                continue
        if mat_name in mat_verts:
            center = basis[mat_verts[mat_name]].mean(axis=0)
            pivots[mat_name] = center + np.array([0.0, 0.038, 0.0], dtype=np.float32)
    return pivots


def pupil_scale_delta(
    basis: np.ndarray,
    indices: np.ndarray,
    pivot: np.ndarray,
    sx: float,
    sy: float,
) -> np.ndarray:
    """Compute sphere-preserving tangent scaling delta around eyeball `pivot`."""
    delta = np.zeros_like(basis, dtype=np.float32)
    if len(indices) == 0:
        return delta

    pts = basis[indices]
    vecs = pts - pivot[None, :]
    radii = np.linalg.norm(vecs, axis=1, keepdims=True)
    safe_r = np.maximum(radii, 1e-6)
    dirs = vecs / safe_r

    n_axis = _normalize(dirs.mean(axis=0))
    up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(n_axis, up_ref))) > 0.98:
        up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u_axis = _normalize(np.cross(up_ref, n_axis))
    v_axis = _normalize(np.cross(n_axis, u_axis))

    u = dirs @ u_axis
    v = dirs @ v_axis
    u_new = u * float(sx)
    v_new = v * float(sy)
    r2 = u_new * u_new + v_new * v_new
    over = r2 > 0.96
    if np.any(over):
        clamp_scale = np.sqrt(0.96 / np.maximum(r2[over], 1e-6))
        u_new[over] *= clamp_scale
        v_new[over] *= clamp_scale
        r2 = u_new * u_new + v_new * v_new
    w_new = np.sqrt(np.maximum(1.0 - r2, 0.0))

    new_dirs = (
        u_new[:, None] * u_axis[None, :]
        + v_new[:, None] * v_axis[None, :]
        + w_new[:, None] * n_axis[None, :]
    )
    new_pts = pivot[None, :] + new_dirs * radii
    delta[indices] = new_pts - pts
    return delta


def _side_close_stats(basis: np.ndarray, close_delta: np.ndarray) -> dict[str, Any]:
    mags = np.linalg.norm(close_delta, axis=1)
    max_mag = float(mags.max()) if len(mags) else 0.0
    if max_mag <= 1e-6:
        return {"empty": True}
    rel = mags / max_mag
    active = np.where(rel > 0.04)[0]
    upper = np.where((close_delta[:, 2] < -1e-4) & (rel > 0.04))[0]
    lower = np.where((close_delta[:, 2] > 1e-4) & (rel > 0.04))[0]
    slit_ref = np.where(rel > 0.45)[0]
    if len(slit_ref) == 0:
        slit_ref = active
    slit_z = float((basis[slit_ref, 2] + close_delta[slit_ref, 2]).mean())
    return {
        "empty": False,
        "mags": mags,
        "max_mag": max_mag,
        "rel": rel,
        "active": active,
        "upper": upper,
        "lower": lower,
        "slit_z": slit_z,
    }


def _find_pair_blocks(
    keys: Any,
    resources: dict[str, Any],
    left_source: str,
    left_default: str,
    right_source: str,
    right_default: str,
) -> tuple[Any | None, Any | None]:
    names = resources.get("source_names", {})
    left_name = names.get(left_source, left_default)
    right_name = names.get(right_source, right_default)
    left = keys.key_blocks.get(left_name) or keys.key_blocks.get(left_source)
    right = keys.key_blocks.get(right_name) or keys.key_blocks.get(right_source)
    return left, right


def _jito_delta(basis: np.ndarray, sides: list[tuple[np.ndarray, dict[str, Any]]]) -> np.ndarray:
    """Half-shut deadpan eyes (`じと目`) with smooth corner blending and flat upper arch."""
    delta = np.zeros_like(basis, dtype=np.float32)
    for close_d, st in sides:
        if st["empty"] or len(st["active"]) == 0:
            continue
        active = st["active"]
        cd = close_d[active]
        # Smooth continuous weight across upper lid, corners (~0.35), and subtle lower lift (~0.12)
        # so connected outer/inner eyelash wing triangles never tear at a hard Z threshold.
        up_blend = _smoothstep(-0.0035, 0.0025, -cd[:, 2])
        w = (0.12 + 0.36 * up_blend).astype(np.float32)
        step = cd * w[:, None]

        upper_mask = cd[:, 2] < -1e-4
        if np.any(upper_mask):
            rel_u = st["rel"][active[upper_mask]]
            z_half = basis[active[upper_mask], 2] + step[upper_mask, 2]
            hi = rel_u > 0.45
            top_ref = float(np.quantile(z_half[hi], 0.35)) if np.any(hi) else float(z_half.mean())
            arch_drop = np.maximum(z_half - top_ref, 0.0)
            flatten_w = _smoothstep(0.30, 0.75, rel_u) * 0.34
            step[upper_mask, 2] -= arch_drop * flatten_w
        delta[active] += step
    return delta


def _side_horiz_t(basis: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Normalized inner-nose (0.0) -> outer-temple (1.0) coordinate for one eye side."""
    ax = np.abs(basis[active, 0])
    x_lo = float(ax.min())
    x_hi = float(ax.max())
    return np.clip((ax - x_lo) / max(x_hi - x_lo, 1e-4), 0.0, 1.0).astype(np.float32)


def _joy_delta(
    basis: np.ndarray,
    sides: list[tuple[np.ndarray, dict[str, Any]]],
    sclera_indices: np.ndarray,
    smile_deltas: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """High-arched happy closed crescent eyes (`喜び`), distinct from standard `笑い`."""
    if smile_deltas is not None:
        delta = (smile_deltas[0] + smile_deltas[1]).astype(np.float32).copy()
    else:
        delta = np.zeros_like(basis, dtype=np.float32)
        for close_d, st in sides:
            if not st["empty"]:
                delta[st["active"]] += close_d[st["active"]]
    for close_d, st in sides:
        if st["empty"] or len(st["active"]) == 0:
            continue
        active = st["active"]
        t_out = _side_horiz_t(basis, active)
        bell = _smoothstep(0.05, 0.45, t_out) * _smoothstep(0.05, 0.45, 1.0 - t_out)
        # Keep upper and lower rims moving together so the crescent stays tightly sealed.
        rim_lock = _smoothstep(0.04, 0.24, st["rel"][active])
        lo_extra = 0.0008 * _smoothstep(-0.001, 0.003, close_d[active, 2])
        lift = (0.0028 + lo_extra) * bell * rim_lock
        delta[active, 2] += lift
    _tuck_sclera(delta, sclera_indices, sides)
    return delta


def _surprise_delta(
    basis: np.ndarray,
    sides: list[tuple[np.ndarray, dict[str, Any]]],
    sclera_indices: np.ndarray,
) -> np.ndarray:
    """Slightly open both eyelids while leaving socket sclera undisturbed behind the iris."""
    delta = np.zeros_like(basis, dtype=np.float32)
    sclera_mask = np.zeros(len(basis), dtype=bool)
    if len(sclera_indices):
        sclera_mask[sclera_indices] = True
    for close_d, st in sides:
        if st["empty"]:
            continue
        up = st["upper"][~sclera_mask[st["upper"]]]
        lo = st["lower"][~sclera_mask[st["lower"]]]
        if len(up):
            delta[up] += close_d[up] * (-0.08)
        if len(lo):
            delta[lo] += close_d[lo] * (-0.03)
    return delta


def _tuck_sclera(
    delta: np.ndarray,
    sclera_indices: np.ndarray,
    sides: list[tuple[np.ndarray, dict[str, Any]]],
) -> None:
    """Push moved sclera pocket vertices slightly back (+Y) so closed lids never leak white."""
    if len(sclera_indices) == 0:
        return
    sclera_mask = np.zeros(len(delta), dtype=bool)
    sclera_mask[sclera_indices] = True
    for _close_d, st in sides:
        if st["empty"]:
            continue
        active = st["active"]
        hit = active[sclera_mask[active]]
        if len(hit):
            delta[hit, 1] += 0.0025


def _calm_close_delta(
    basis: np.ndarray,
    sides: list[tuple[np.ndarray, dict[str, Any]]],
    eyelash_indices: np.ndarray,
    sclera_indices: np.ndarray,
    smile_deltas: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Create straight relaxed closed eyes (`なごみ`) by averaging downward close and upward smile."""
    delta = np.zeros_like(basis, dtype=np.float32)
    if smile_deltas is not None:
        close_sum = sum((close_d for close_d, _st in sides), np.zeros_like(basis, dtype=np.float32))
        smile_sum = smile_deltas[0] + smile_deltas[1]
        delta = (close_sum * 0.50 + smile_sum * 0.50).astype(np.float32)
        _tuck_sclera(delta, sclera_indices, sides)
        return delta

    lash_mask = np.zeros(len(basis), dtype=bool)
    lash_mask[eyelash_indices] = True
    for close_d, st in sides:
        if st["empty"]:
            continue
        active = st["active"]
        delta[active] += close_d[active]
        hit_lash = active[lash_mask[active]]
        if len(hit_lash):
            closed_z = basis[hit_lash, 2] + close_d[hit_lash, 2]
            delta[hit_lash, 2] += (st["slit_z"] - closed_z) * 0.55
    _tuck_sclera(delta, sclera_indices, sides)
    return delta


def _add_or_replace_key(
    mesh: Any,
    basis_block: Any,
    basis_coords: np.ndarray,
    name: str,
    delta: np.ndarray,
) -> dict[str, Any]:
    keys = mesh.data.shape_keys
    existing = keys.key_blocks.get(name)
    if existing is None:
        block = mesh.shape_key_add(name=name, from_mix=False)
        status = "added"
    else:
        block = existing
        status = "replaced"
    block.value = 0.0
    block.relative_key = basis_block
    if hasattr(block, "mute"):
        block.mute = False
    target = (basis_coords + delta).astype(np.float32)
    block.data.foreach_set("co", target.ravel())
    changed = int(np.count_nonzero(np.linalg.norm(delta, axis=1) > 1e-6))
    return {"name": name, "status": status, "changed_vertices": changed}


def remove_unwanted_keys(meshes: list[Any], resources: dict[str, Any]) -> list[dict[str, str]]:
    """Delete configured noisy/broken morphs (e.g. `口開け`, `追加睫毛`) cleanly."""
    removed_sources = set(resources.get("removed_sources", []))
    source_names = resources.get("source_names", {})
    removed_names = set(removed_sources) | {
        source_names[src] for src in removed_sources if src in source_names
    }
    if not removed_names:
        return []

    records: list[dict[str, str]] = []
    for mesh in meshes:
        keys = getattr(mesh.data, "shape_keys", None)
        if keys is None:
            continue
        raw_marker = keys.get(KEY_SOURCE_PROPERTY)
        marker = json.loads(raw_marker) if raw_marker else {}
        to_remove: list[tuple[Any, str | None]] = []
        for block in list(keys.key_blocks):
            if block.name == "Basis":
                continue
            source = marker.get(block.name)
            if block.name in removed_names or source in removed_sources:
                to_remove.append((block, source))
        for block, source in to_remove:
            key_name = block.name
            if keys.animation_data is not None:
                for drv in list(keys.animation_data.drivers):
                    if f'key_blocks["{key_name}"]' in drv.data_path:
                        keys.animation_data.drivers.remove(drv)
            mesh.shape_key_remove(block)
            marker.pop(key_name, None)
            records.append({"mesh": mesh.name, "key": key_name, "source": source or key_name})
        if to_remove:
            keys[KEY_SOURCE_PROPERTY] = json.dumps(marker, ensure_ascii=False, separators=(",", ":"))
    return records


def _eyeball_pupil_delta(
    basis: np.ndarray,
    mat_verts: dict[str, np.ndarray],
    pivots: dict[str, np.ndarray],
    sx: float,
    sy: float,
) -> np.ndarray:
    delta = np.zeros_like(basis, dtype=np.float32)
    for mat_name in ("M_eye_L", "M_eye_R"):
        if mat_name in mat_verts and mat_name in pivots:
            delta += pupil_scale_delta(basis, mat_verts[mat_name], pivots[mat_name], sx, sy)
    return delta


def generate(meshes: list[Any], resources: dict[str, Any]) -> dict[str, Any]:
    """Generate pure-geometry eye morphs on merged `head` meshes."""
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "generated": [],
        "skipped": [],
    }
    for mesh in meshes:
        keys = getattr(mesh.data, "shape_keys", None)
        if keys is None or keys.key_blocks.get("Basis") is None:
            continue
        mat_verts = _material_vertex_sets(mesh)
        has_eyes = "M_eye_L" in mat_verts or "M_eye_R" in mat_verts
        left_close, right_close = _find_pair_blocks(
            keys, resources, "eye_L_close", "ウィンク２", "eye_R_close", "ｳｨﾝｸ２右"
        )
        left_smile, right_smile = _find_pair_blocks(
            keys, resources, "eye_L_smile", "ウィンク", "eye_R_smile", "ウィンク右"
        )
        if not has_eyes and left_close is None and right_close is None:
            continue

        basis_block = keys.key_blocks["Basis"]
        basis = _key_coords(basis_block)
        card_indices: dict[str, dict[str, np.ndarray]] = {}
        cards_attached = False
        if "M_eye_L" in mat_verts and "M_eye_R" in mat_verts:
            init_pivots = _eyeball_pivots(basis, mat_verts)
            card_indices, cards_attached = attach_hidden_cards(mesh, basis, mat_verts, init_pivots)
            if cards_attached:
                keys = mesh.data.shape_keys
                basis_block = keys.key_blocks["Basis"]
                basis = _key_coords(basis_block)
                mat_verts = _material_vertex_sets(mesh)
                left_close, right_close = _find_pair_blocks(
                    keys, resources, "eye_L_close", "ウィンク２", "eye_R_close", "ｳｨﾝｸ２右"
                )
                left_smile, right_smile = _find_pair_blocks(
                    keys, resources, "eye_L_smile", "ウィンク", "eye_R_smile", "ウィンク右"
                )

        raw_marker = keys.get(KEY_SOURCE_PROPERTY)
        marker = json.loads(raw_marker) if raw_marker else {}

        mesh_generated: list[dict[str, Any]] = []
        wao_pupil_delta = np.zeros_like(basis, dtype=np.float32)
        pivots: dict[str, np.ndarray] = {}

        if has_eyes:
            pivots = _eyeball_pivots(basis, mat_verts)
            for morph_name, sx, sy in _PUPIL_SPECS:
                d = _eyeball_pupil_delta(basis, mat_verts, pivots, sx, sy)
                mesh_generated.append(_add_or_replace_key(mesh, basis_block, basis, morph_name, d))
                marker[morph_name] = f"@generated:{morph_name}"
            wao_pupil_delta = _eyeball_pupil_delta(basis, mat_verts, pivots, 0.58, 0.58)

        sides = None
        calm_delta = None
        if left_close is not None and right_close is not None:
            close_L = _key_coords(left_close) - basis
            close_R = _key_coords(right_close) - basis
            sides = [
                (close_L, _side_close_stats(basis, close_L)),
                (close_R, _side_close_stats(basis, close_R)),
            ]
            smile_deltas = None
            if left_smile is not None and right_smile is not None:
                smile_deltas = (_key_coords(left_smile) - basis, _key_coords(right_smile) - basis)
            eyelash_verts = mat_verts.get("M_eyelash", np.empty(0, dtype=np.int32))
            sclera_verts = mat_verts.get("SCSP_EyeSocketSclera", np.empty(0, dtype=np.int32))
            wide_d = _surprise_delta(basis, sides, sclera_verts)
            calm_delta = _calm_close_delta(basis, sides, eyelash_verts, sclera_verts, smile_deltas)
            eyelid_deltas = [
                ("じと目", _jito_delta(basis, sides)),
                ("喜び", _joy_delta(basis, sides, sclera_verts, smile_deltas)),
                ("びっくり", wide_d),
                ("わぉ?!", wide_d * 0.90 + wao_pupil_delta),
                ("なごみ", calm_delta),
            ]
            for morph_name, d in eyelid_deltas:
                mesh_generated.append(_add_or_replace_key(mesh, basis_block, basis, morph_name, d))
                marker[morph_name] = f"@generated:{morph_name}"
        else:
            report["skipped"].append({"mesh": mesh.name, "reason": "missing close eye keys"})

        if pivots:
            eff_generated, eff_markers = generate_effect_morphs(
                mesh,
                basis_block,
                basis,
                mat_verts,
                pivots,
                card_indices,
                sides,
                calm_delta,
                _add_or_replace_key,
            )
            mesh_generated.extend(eff_generated)
            marker.update(eff_markers)

        keys[KEY_SOURCE_PROPERTY] = json.dumps(marker, ensure_ascii=False, separators=(",", ":"))
        report["generated"].append({"mesh": mesh.name, "cards_attached": cards_attached, "morphs": mesh_generated})
    return report
