"""Hidden-in-head MMD stencil card geometry and effect morph builders."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import bmesh
import bpy
import numpy as np

from eye_effect_atlas import ATLAS_UV_RECTS, DEFAULT_ATLAS_PATH, _smoothstep, ensure_atlas_file

EFFECT_MATERIAL_NAME = "SCSP_EyeEffects"
_GRID_N = 4  # 4x4 vertices = 9 quads per curved card.


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize zero-length vector")
    return (vec / norm).astype(np.float32)


def _eye_axes(pts: np.ndarray, pivot: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Compute `(u_axis, v_axis, n_axis, mean_r, half_u, half_v)` for an eyeball mesh."""
    vecs = pts - pivot[None, :]
    radii = np.linalg.norm(vecs, axis=1)
    mean_r = float(radii.mean())
    dirs = vecs / np.maximum(radii[:, None], 1e-6)
    n_axis = _normalize(dirs.mean(axis=0))
    up_ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(n_axis, up_ref))) > 0.98:
        up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u_axis = _normalize(np.cross(up_ref, n_axis))
    v_axis = _normalize(np.cross(n_axis, u_axis))
    u_vals = (pts - pivot[None, :]) @ u_axis
    v_vals = (pts - pivot[None, :]) @ v_axis
    half_u = float((u_vals.max() - u_vals.min()) * 0.5)
    half_v = float((v_vals.max() - v_vals.min()) * 0.5)
    return u_axis, v_axis, n_axis, mean_r, half_u, half_v


def _curved_patch_coords(
    center: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    n_axis: np.ndarray,
    half_w: float,
    half_h: float,
    curve_radius: float,
    grid_n: int = _GRID_N,
) -> np.ndarray:
    """Build `(grid_n*grid_n, 3)` coordinates on a gentle sphere/cap facing `n_axis`."""
    lin = np.linspace(-1.0, 1.0, grid_n, dtype=np.float32)
    coords = np.empty((grid_n * grid_n, 3), dtype=np.float32)
    r = max(float(curve_radius), max(half_w, half_h) * 1.4, 1e-4)
    idx = 0
    for sy in lin:  # bottom (-1) -> top (+1)
        for sx in lin:  # left (-1) -> right (+1)
            du = float(sx) * half_w
            dv = float(sy) * half_h
            # Sagitta pullback along -n_axis keeps edges hugging curved eyes/face.
            sag = (du * du + dv * dv) / (2.0 * r)
            coords[idx] = center + du * u_axis + dv * v_axis - sag * n_axis
            idx += 1
    return coords


def _sphere_cap_patch_coords(
    pivot: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    n_axis: np.ndarray,
    radius: float,
    half_w: float,
    half_h: float,
    grid_n: int = _GRID_N,
) -> np.ndarray:
    """Project a `(grid_n x grid_n)` grid radially onto the eyeball sphere of `radius`."""
    lin = np.linspace(-1.0, 1.0, grid_n, dtype=np.float32)
    coords = np.empty((grid_n * grid_n, 3), dtype=np.float32)
    # Center direction is slightly weighted around optical n_axis; scale tangents on sphere.
    idx = 0
    for sy in lin:
        for sx in lin:
            du = float(sx) * half_w
            dv = float(sy) * half_h
            vec = n_axis * radius + u_axis * du + v_axis * dv
            coords[idx] = pivot + _normalize(vec) * radius
            idx += 1
    return coords


def _conform_front_to_face(coords: np.ndarray, face_pts: np.ndarray, clearance: float = 0.0020) -> np.ndarray:
    """Place every card vertex cleanly `clearance` meters in front (`-Y`) of nearby `M_face` skin."""
    if not len(face_pts):
        return coords
    out = coords.copy()
    fxz = face_pts[:, [0, 2]]
    for i, pt in enumerate(out):
        dxz = fxz - pt[[0, 2]][None, :]
        dist2 = np.sum(dxz * dxz, axis=1)
        near = np.where(dist2 <= 0.0085 * 0.0085)[0]
        if not len(near):
            near = np.argpartition(dist2, min(4, len(dist2) - 1))[:4]
        out[i, 1] = float(face_pts[near, 1].min()) - clearance
    return out
    return coords


def ensure_eye_effects_material(mesh: Any) -> int:
    """Create or reuse `SCSP_EyeEffects` with the packed preset atlas and return slot index."""
    for slot_idx, mat in enumerate(mesh.data.materials):
        if mat is not None and mat.name == EFFECT_MATERIAL_NAME:
            return slot_idx

    atlas_path = DEFAULT_ATLAS_PATH if DEFAULT_ATLAS_PATH.is_file() else ensure_atlas_file()
    image = None
    norm_target = str(atlas_path.resolve()).casefold()
    for existing_img in bpy.data.images:
        raw_fp = str(bpy.path.abspath(existing_img.filepath or "")).strip()
        if raw_fp and str(bpy.path.abspath(raw_fp)).casefold() == norm_target:
            image = existing_img
            break
    if image is None:
        image = bpy.data.images.load(str(atlas_path), check_existing=True)
    image.colorspace_settings.name = "sRGB"
    image.alpha_mode = "STRAIGHT"
    if hasattr(image, "use_alpha"):
        image.use_alpha = True
    if not image.packed_file:
        image.pack()

    mat = bpy.data.materials.get(EFFECT_MATERIAL_NAME)
    if mat is None:
        mat = bpy.data.materials.new(name=EFFECT_MATERIAL_NAME)
        mat.use_nodes = True
        mat.roughness = 1.0
        if hasattr(mat, "use_backface_culling"):
            mat.use_backface_culling = False
        if hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "BLENDED"
        if hasattr(mat, "blend_method"):
            mat.blend_method = "BLEND"
        if hasattr(mat, "use_transparent_shadow"):
            mat.use_transparent_shadow = True
        tree = mat.node_tree
        tree.nodes.clear()
        out_node = tree.nodes.new("ShaderNodeOutputMaterial")
        mix_shadow = tree.nodes.new("ShaderNodeMixShader")
        lp_node = tree.nodes.new("ShaderNodeLightPath")
        trans_bsdf = tree.nodes.new("ShaderNodeBsdfTransparent")
        trans_bsdf.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
        tex = tree.nodes.new("ShaderNodeTexImage")
        tex.image = image
        tex.extension = "CLIP"
        bsdf.inputs["Roughness"].default_value = 1.0
        if "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = 0.0
        tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        tree.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
        tree.links.new(lp_node.outputs["Is Shadow Ray"], mix_shadow.inputs["Fac"])
        tree.links.new(bsdf.outputs["BSDF"], mix_shadow.inputs[1])
        tree.links.new(trans_bsdf.outputs["BSDF"], mix_shadow.inputs[2])
        tree.links.new(mix_shadow.outputs["Shader"], out_node.inputs["Surface"])
        if hasattr(mat, "mmd_material"):
            mat.mmd_material.name_j = EFFECT_MATERIAL_NAME
            mat.mmd_material.name_e = "eye_effects"
            mat.mmd_material.enabled_drop_shadow = False
            mat.mmd_material.enabled_self_shadow_map = False
            mat.mmd_material.enabled_self_shadow = False
            mat.mmd_material.enabled_toon_edge = False
        mat["scsp_disable_shadows"] = True

    mesh.data.materials.append(mat)
    return len(mesh.data.materials) - 1


def _resolve_bone_groups(mesh: Any, mat_verts: dict[str, np.ndarray]) -> dict[str, int]:
    """Resolve vertex group indices on `mesh` for `head`, `left_eye`, `right_eye`."""
    vg_index = {g.name: g.index for g in mesh.vertex_groups}
    resolved: dict[str, int] = {}

    rig = bpy.data.objects.get("SCSP_Skeleton")
    if rig is not None and rig.type == "ARMATURE":
        mmd_to_vg: dict[str, str] = {}
        for pb in rig.pose.bones:
            jname = getattr(getattr(pb, "mmd_bone", None), "name_j", "")
            if jname in ("頭", "左目", "右目") and pb.name in vg_index:
                mmd_to_vg[jname] = pb.name
            elif pb.name in ("頭", "左目", "右目") and pb.name in vg_index:
                mmd_to_vg.setdefault(pb.name, pb.name)
        if "頭" in mmd_to_vg:
            resolved["head"] = vg_index[mmd_to_vg["頭"]]
        if "左目" in mmd_to_vg:
            resolved["left_eye"] = vg_index[mmd_to_vg["左目"]]
        if "右目" in mmd_to_vg:
            resolved["right_eye"] = vg_index[mmd_to_vg["右目"]]

    for key, fallbacks in (
        ("head", ("頭", "p00__Head", "Head")),
        ("left_eye", ("左目",)),
        ("right_eye", ("右目",)),
    ):
        if key not in resolved:
            for cand in fallbacks:
                if cand in vg_index:
                    resolved[key] = vg_index[cand]
                    break

    if "head" not in resolved:
        face_verts = mat_verts.get("M_face", np.empty(0, dtype=np.int32))
        counts: dict[int, int] = {}
        for vi in face_verts[:200]:
            for g in mesh.data.vertices[int(vi)].groups:
                counts[g.group] = counts.get(g.group, 0) + 1
        if counts:
            resolved["head"] = max(counts, key=lambda gid: counts[gid])
        elif vg_index:
            resolved["head"] = next(iter(vg_index.values()))
        else:
            vg = mesh.vertex_groups.new(name="頭")
            resolved["head"] = vg.index

    resolved.setdefault("left_eye", resolved["head"])
    resolved.setdefault("right_eye", resolved["head"])
    return resolved


def _skull_center(basis: np.ndarray, mat_verts: dict[str, np.ndarray], pivots: dict[str, np.ndarray]) -> np.ndarray:
    """Compute a safe interior skull center strictly inside the head cavity."""
    face_pts = basis[mat_verts["M_face"]] if "M_face" in mat_verts else basis
    min_y, max_y = float(face_pts[:, 1].min()), float(face_pts[:, 1].max())
    mid_pivot = (pivots["M_eye_L"] + pivots["M_eye_R"]) * 0.5
    center = mid_pivot.copy()
    center[0] = 0.0
    # Shift backward behind the eye pivots, strictly clamped inside front/back skull bounds.
    safe_lo = min_y + 0.045
    safe_hi = max_y - 0.035
    if safe_lo <= safe_hi:
        center[1] = float(np.clip(mid_pivot[1] + 0.025, safe_lo, safe_hi))
    else:
        center[1] = float((min_y + max_y) * 0.5)
    return center.astype(np.float32)


def attach_hidden_cards(
    mesh: Any,
    basis: np.ndarray,
    mat_verts: dict[str, np.ndarray],
    pivots: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, np.ndarray]], bool]:
    """Attach collapsed 4x4 stencil cards inside skull at Basis while preserving custom normals."""
    if any(m is not None and m.name == EFFECT_MATERIAL_NAME for m in mesh.data.materials):
        # Cards already attached on this mesh (idempotent skip of geometry duplication).
        return {}, False

    mat_idx = ensure_eye_effects_material(mesh)
    vg_map = _resolve_bone_groups(mesh, mat_verts)
    skull = _skull_center(basis, mat_verts, pivots)

    me = mesh.data
    had_custom_normals = bool(me.has_custom_normals)
    old_loop_count = len(me.loops)
    old_loop_normals = np.empty(old_loop_count * 3, dtype=np.float32)
    me.corner_normals.foreach_get("vector", old_loop_normals)
    old_loop_normals = old_loop_normals.reshape((old_loop_count, 3))

    keys = me.shape_keys
    raw_marker = keys.get("scsp_mmd_shape_source_by_current") if keys is not None else None
    key_meta = [
        (kb.name, kb.relative_key.name if kb.relative_key else "Basis", float(kb.value), bool(kb.mute))
        for kb in (keys.key_blocks if keys is not None else ())
    ]

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    deform = bm.verts.layers.deform.verify()
    uv_layer = bm.loops.layers.uv.get("UV0") or bm.loops.layers.uv.active or bm.loops.layers.uv.verify()
    shape_layers = list(bm.verts.layers.shape)

    u_world = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    n_front = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    specs: list[tuple[str, str, np.ndarray, int, bool]] = [
        ("星目", "L", pivots["M_eye_L"], vg_map["left_eye"], False),
        ("星目", "R", pivots["M_eye_R"], vg_map["right_eye"], False),
        ("はぁと", "L", pivots["M_eye_L"], vg_map["left_eye"], False),
        ("はぁと", "R", pivots["M_eye_R"], vg_map["right_eye"], False),
        ("はちゅ目", "L", skull, vg_map["head"], False),
        ("はちゅ目", "R", skull, vg_map["head"], False),
        ("はぅ", "L", skull, vg_map["head"], True),
        ("はぅ", "R", skull, vg_map["head"], False),
    ]

    card_indices: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in ("星目", "はぁと", "はちゅ目", "はぅ")
    }
    lin_u = np.linspace(0.0, 1.0, _GRID_N, dtype=np.float32)
    lin_v = np.linspace(0.0, 1.0, _GRID_N, dtype=np.float32)

    added_loops = 0
    for morph_name, side, rest_pos, vg_idx, flip_u in specs:
        u0, v0, u1, v1 = ATLAS_UV_RECTS[morph_name]
        # Collapse to 0.4 mm dot strictly inside skull cavity / eyeball pivot center.
        rest_coords = _curved_patch_coords(rest_pos, u_world, v_world, n_front, 0.0002, 0.0002, 0.01)
        new_bm_verts = []
        created_idxs: list[int] = []
        for k in range(_GRID_N * _GRID_N):
            v = bm.verts.new(rest_coords[k])
            v[deform][vg_idx] = 1.0
            for slayer in shape_layers:
                v[slayer] = rest_coords[k]
            new_bm_verts.append(v)

        bm.verts.index_update()
        created_idxs = [v.index for v in new_bm_verts]
        side_key = "left" if side == "L" else "right"
        card_indices[morph_name][side_key] = np.array(created_idxs, dtype=np.int32)

        for r in range(_GRID_N - 1):
            for c in range(_GRID_N - 1):
                i00 = r * _GRID_N + c
                i10 = i00 + 1
                i11 = i00 + _GRID_N + 1
                i01 = i00 + _GRID_N
                face_verts = (new_bm_verts[i00], new_bm_verts[i10], new_bm_verts[i11], new_bm_verts[i01])
                f = bm.faces.new(face_verts)
                f.material_index = mat_idx
                f.smooth = True
                uvs = []
                for rr, cc in ((r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c)):
                    fu = float(lin_u[cc])
                    fv = float(lin_v[rr])
                    uu = (u1 - fu * (u1 - u0)) if flip_u else (u0 + fu * (u1 - u0))
                    vv = v0 + fv * (v1 - v0)
                    uvs.append((uu, vv))
                for loop, uv_val in zip(f.loops, uvs):
                    loop[uv_layer].uv = uv_val
                added_loops += len(f.loops)

    bm.to_mesh(me)
    bm.free()
    me.update()

    if had_custom_normals and len(me.loops) == old_loop_count + added_loops:
        new_loop_normals = np.tile(n_front[None, :], (added_loops, 1))
        combined = np.vstack([old_loop_normals, new_loop_normals])
        me.normals_split_custom_set(combined)

    if me.shape_keys is not None:
        basis_kb = me.shape_keys.key_blocks.get("Basis")
        for kname, rel_name, val, is_muted in key_meta:
            kb = me.shape_keys.key_blocks.get(kname)
            if kb is not None:
                if basis_kb is not None and kb.name != "Basis":
                    kb.relative_key = me.shape_keys.key_blocks.get(rel_name, basis_kb)
                kb.value = val
                if hasattr(kb, "mute"):
                    kb.mute = is_muted
        if raw_marker is not None:
            me.shape_keys["scsp_mmd_shape_source_by_current"] = raw_marker

    return card_indices, True


def _seal_flat_closed_eyelids(
    mesh: Any,
    basis: np.ndarray,
    face_idxs: np.ndarray,
    sides: list[tuple[np.ndarray, dict[str, Any]]],
    base_delta: np.ndarray,
) -> np.ndarray:
    """Seal closed `M_face` eyelids flush with the cheek via bridged harmonic Y interpolation.

    Prevents open rim slits, dark interior/seam lines, and under-eye crease ridges behind
    face-card expressions (`はちゅ目`, `はぅ`).
    """
    sealed = base_delta.copy()
    me = mesh.data
    face_mids = [i for i, m in enumerate(me.materials) if m and m.name == "M_face"]
    if not face_mids or not len(face_idxs) or not sides:
        return sealed

    face_mid = face_mids[0]
    edge_counts: Counter[tuple[int, int]] = Counter()
    adj: dict[int, set[int]] = defaultdict(set)
    for poly in me.polygons:
        if poly.material_index == face_mid:
            vs = list(poly.vertices)
            for a, b in zip(vs, vs[1:] + vs[:1]):
                edge_counts[(min(a, b), max(a, b))] += 1
                adj[a].add(b)
                adj[b].add(a)
    rim_set = {v for (a, b), c in edge_counts.items() if c == 1 for v in (a, b)}

    face_mask = np.zeros(len(basis), dtype=bool)
    face_mask[face_idxs] = True

    for close_d, st in sides:
        if st["empty"]:
            continue
        hit = st["active"][face_mask[st["active"]]]
        if not len(hit):
            continue

        # Expand solve domain by 2 rings so Basis eye-socket curvature flattens cleanly into the cheek.
        ext = {int(v) for v in hit}
        for _ in range(2):
            ext |= {nb for v in list(ext) for nb in adj[v] if face_mask[nb]}
        solve_verts = np.array(sorted(ext), dtype=np.int32)
        pos = basis + close_d
        new_pos = pos[solve_verts].copy()

        dz = close_d[solve_verts, 2]
        is_rim = np.array([int(v) in rim_set for v in solve_verts], dtype=bool)
        up_idxs = np.where(is_rim & (dz < -0.0001))[0]
        lo_idxs = np.where(is_rim & (dz >= -0.0001))[0]

        bridge_pairs: list[tuple[int, int]] = []
        u_fade = np.zeros(len(up_idxs), dtype=np.float32)
        if len(up_idxs) and len(lo_idxs):
            ux, uz = new_pos[up_idxs, 0], new_pos[up_idxs, 2]
            lx, lz = new_pos[lo_idxs, 0], new_pos[lo_idxs, 2]
            uo, lo_o = np.argsort(ux), np.argsort(lx)
            xmin = float(min(ux.min(), lx.min()))
            xmax = float(max(ux.max(), lx.max()))
            span_x = max(xmax - xmin, 1e-4)

            def _seam_target(x_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                zu = np.interp(x_vals, ux[uo], uz[uo])
                zl = np.interp(x_vals, lx[lo_o], lz[lo_o])
                edge_t = np.clip((x_vals - xmin) / span_x, 0.0, 1.0)
                mid_w = _smoothstep(0.0, 0.008, edge_t) * _smoothstep(0.0, 0.008, 1.0 - edge_t)
                return 0.5 * (zu + zl), mid_w

            zu_mid, u_fade = _seam_target(ux)
            zl_mid, l_fade = _seam_target(lx)
            new_pos[lo_idxs, 2] = lz + (zl_mid + 0.00065 - lz) * l_fade
            new_pos[up_idxs, 2] = uz + (zu_mid - 0.00065 - uz) * u_fade

            for ui in up_idxs:
                li = int(lo_idxs[np.argmin(np.abs(new_pos[lo_idxs, 0] - new_pos[ui, 0]))])
                bridge_pairs.append((int(ui), li))
            for li in lo_idxs:
                ui = int(up_idxs[np.argmin(np.abs(new_pos[up_idxs, 0] - new_pos[li, 0]))])
                bridge_pairs.append((ui, int(li)))

        solve_pos = pos.copy()
        solve_pos[solve_verts] = new_pos
        nbr_lists = [list(adj[int(v)]) for v in solve_verts]
        for ui, li in bridge_pairs:
            u_v, l_v = int(solve_verts[ui]), int(solve_verts[li])
            if l_v not in nbr_lists[ui]:
                nbr_lists[ui].append(l_v)
            if u_v not in nbr_lists[li]:
                nbr_lists[li].append(u_v)

        # Light XZ relaxation on interior non-rim rings keeps compressed eyelid loops smooth.
        xz_sol = solve_pos[:, [0, 2]].copy()
        for _ in range(8):
            for i, v in enumerate(solve_verts):
                if is_rim[i]:
                    continue
                xz_sol[int(v)] = 0.75 * xz_sol[int(v)] + 0.25 * xz_sol[nbr_lists[i]].mean(axis=0)
        solve_pos[solve_verts, 0] = xz_sol[solve_verts, 0]
        solve_pos[solve_verts, 2] = xz_sol[solve_verts, 1]
        new_pos = solve_pos[solve_verts].copy()

        # Inverse-distance weighted harmonic Y solve clamped to the untouched cheek boundary.
        y_sol = solve_pos[:, 1].copy()
        wts: list[tuple[np.ndarray, np.ndarray]] = []
        for i, v in enumerate(solve_verts):
            nbs = nbr_lists[i]
            d = np.hypot(solve_pos[nbs, 0] - solve_pos[int(v), 0], solve_pos[nbs, 2] - solve_pos[int(v), 2])
            w = 1.0 / np.maximum(d, 0.0015)
            w /= w.sum()
            wts.append((np.asarray(nbs, dtype=np.int32), w.astype(np.float32)))

        for _ in range(250):
            for i, v in enumerate(solve_verts):
                nbs, w = wts[i]
                y_sol[int(v)] = float(np.dot(w, y_sol[nbs]))

        new_pos[:, 1] = y_sol[solve_verts]
        if len(up_idxs):
            new_pos[up_idxs, 1] += 0.00025 * u_fade

        sealed[solve_verts] = new_pos - basis[solve_verts]

    return sealed


def generate_effect_morphs(
    mesh: Any,
    basis_block: Any,
    basis: np.ndarray,
    mat_verts: dict[str, np.ndarray],
    pivots: dict[str, np.ndarray],
    card_indices: dict[str, dict[str, np.ndarray]],
    sides: list[tuple[np.ndarray, dict[str, Any]]] | None,
    calm_delta: np.ndarray | None,
    add_or_replace_fn: Any,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Build target vertex deltas for `恐ろしい子！`, `星目`, `はぁと`, `はちゅ目`, and `はぅ`."""
    results: list[dict[str, Any]] = []
    markers: dict[str, str] = {}

    eye_mats = [("left", "M_eye_L"), ("right", "M_eye_R")]
    available_eyes = [(side, m) for side, m in eye_mats if m in mat_verts and m in pivots]

    # 1. `恐ろしい子！`: pure-geometry iris retraction cleanly behind sclera inside head.
    funky_delta = np.zeros_like(basis, dtype=np.float32)
    for _side, mat_name in available_eyes:
        idxs = mat_verts[mat_name]
        piv = pivots[mat_name]
        funky_delta[idxs] = (piv[None, :] + (basis[idxs] - piv[None, :]) * 0.14) - basis[idxs]
    results.append(add_or_replace_fn(mesh, basis_block, basis, "恐ろしい子！", funky_delta))
    markers["恐ろしい子！"] = "@generated:恐ろしい子！"

    if not card_indices:
        return results, markers

    # 2. `星目` & `はぁと`: sphere-hugging iris cards weighted to `左目`/`右目`.
    for morph_name, sx_scale, sy_scale, r_off in (
        ("星目", 0.78, 0.80, 0.00035),
        ("はぁと", 0.62, 0.64, 0.00035),
    ):
        delta = np.zeros_like(basis, dtype=np.float32)
        side_cards = card_indices.get(morph_name, {})
        for side, mat_name in available_eyes:
            if side not in side_cards:
                continue
            pts = basis[mat_verts[mat_name]]
            piv = pivots[mat_name]
            u_ax, v_ax, n_ax, mean_r, half_u, half_v = _eye_axes(pts, piv)
            target = _sphere_cap_patch_coords(
                piv, u_ax, v_ax, n_ax, mean_r + r_off, half_u * sx_scale, half_v * sy_scale
            )
            c_idx = side_cards[side]
            delta[c_idx] = target - basis[c_idx]
        results.append(add_or_replace_fn(mesh, basis_block, basis, morph_name, delta))
        markers[morph_name] = f"@generated:{morph_name}"

    # 3. `はちゅ目` (`O O`) & `はぅ` (`> <`): face-aligned cards weighted to head bone.
    sclera_idxs = mat_verts.get("SCSP_EyeSocketSclera", np.empty(0, dtype=np.int32))
    sclera_pts = basis[sclera_idxs] if len(sclera_idxs) else None
    eyelash_idxs = mat_verts.get("M_eyelash", np.empty(0, dtype=np.int32))
    face_idxs = mat_verts.get("M_face", np.empty(0, dtype=np.int32))
    face_pts = basis[face_idxs] if len(face_idxs) else basis
    face_half_w = float((face_pts[:, 0].max() - face_pts[:, 0].min()) * 0.5) if len(face_pts) else 0.09
    v_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Use 100% `まばたき` (`close_L + close_R`) so eyelids seal completely shut without open slits,
    # while retracting eyeballs, sclera pocket, and eyelashes cleanly inside skull.
    full_close_delta = np.zeros_like(basis, dtype=np.float32)
    if sides:
        for close_d, _st in sides:
            full_close_delta += close_d
    elif calm_delta is not None:
        full_close_delta = calm_delta.copy()

    # Retract irises, sclera, and eyelashes deeply inside skull cavity (3.5 cm back in +Y)
    # so zero lash tips, wing corners, or inner sclera ever clip through closed lids.
    hide_parts_delta = funky_delta.copy()
    if len(eyelash_idxs):
        hide_parts_delta[eyelash_idxs] = full_close_delta[eyelash_idxs]
        hide_parts_delta[eyelash_idxs, 1] += 0.035
    if len(sclera_idxs):
        hide_parts_delta[sclera_idxs] = full_close_delta[sclera_idxs]
        hide_parts_delta[sclera_idxs, 1] += 0.035

    # Seal `M_face` eyelids completely flat onto the surrounding cheek surface via harmonic Y.
    sealed_face_delta = _seal_flat_closed_eyelids(mesh, basis, face_idxs, sides or [], full_close_delta)

    def _eye_visible_box(side: str, mat_name: str) -> tuple[np.ndarray, float, float]:
        """Measure visible eye center & extent from iris (`M_eye_*`) and eyelash main span, ignoring deep sclera bowl."""
        pts = basis[mat_verts[mat_name]]
        c = pts.mean(axis=0).astype(np.float32)
        iris_w = float(pts[:, 0].max() - pts[:, 0].min())
        iris_h = float(pts[:, 2].max() - pts[:, 2].min())
        # Closed-eye slit Z centers the card accurately over sealed eyelids.
        side_idx = 0 if side == "left" else 1
        if sides and side_idx < len(sides) and not sides[side_idx][1]["empty"]:
            c[2] = float(sides[side_idx][1]["slit_z"])
        # Clamp horizontal aperture using core eyelash span if available (excluding deep hidden sclera cup).
        eye_w = iris_w
        if len(eyelash_idxs):
            el_pts = basis[eyelash_idxs]
            el_side = el_pts[el_pts[:, 0] > 0.0] if side == "left" else el_pts[el_pts[:, 0] < 0.0]
            if len(el_side):
                lash_w = float(el_side[:, 0].max() - el_side[:, 0].min())
                eye_w = max(iris_w, min(iris_w * 1.45, lash_w * 0.72))
        return c, eye_w, iris_h

    # `はちゅ目`: sealed skin behind comic `O O` sized proportionally to visible iris height.
    hachu_delta = sealed_face_delta.copy()
    if len(eyelash_idxs):
        hachu_delta[eyelash_idxs] = hide_parts_delta[eyelash_idxs]
    if len(sclera_idxs):
        hachu_delta[sclera_idxs] = hide_parts_delta[sclera_idxs]
    for _s, mat_name in available_eyes:
        idxs = mat_verts[mat_name]
        hachu_delta[idxs] = funky_delta[idxs]

    hachu_face_pts = (basis[face_idxs] + hachu_delta[face_idxs]) if len(face_idxs) else face_pts
    hachu_cards = card_indices.get("はちゅ目", {})
    for side, mat_name in available_eyes:
        if side not in hachu_cards:
            continue
        center, eye_w, eye_h = _eye_visible_box(side, mat_name)
        # Keep circular comic `O O`: outer circle spans ~80% of atlas tile -> radius 0.54 * eye_h gives ~25mm circle.
        hachu_r = max(eye_h * 0.54, eye_w * 0.52)
        out_sign = 1.0 if side == "left" else -1.0
        n_card = _normalize(np.array([0.16 * out_sign, -0.985, 0.0], dtype=np.float32))
        u_card = _normalize(np.cross(v_world, n_card))
        target = _curved_patch_coords(
            center, u_card, v_world, n_card, hachu_r, hachu_r, face_half_w * 0.75
        )
        target = _conform_front_to_face(target, hachu_face_pts, clearance=0.00075)
        c_idx = hachu_cards[side]
        hachu_delta[c_idx] = target - basis[c_idx]
    results.append(add_or_replace_fn(mesh, basis_block, basis, "はちゅ目", hachu_delta))
    markers["はちゅ目"] = "@generated:はちゅ目"

    # `はぅ` (`> <`): 100% sealed skin (`まばたき`), eyes/lashes retracted behind skin, crisp `> <` fitted to eye width.
    hau_delta = sealed_face_delta.copy()
    if len(eyelash_idxs):
        hau_delta[eyelash_idxs] = hide_parts_delta[eyelash_idxs]
    if len(sclera_idxs):
        hau_delta[sclera_idxs] = hide_parts_delta[sclera_idxs]
    for _s, mat_name in available_eyes:
        idxs = mat_verts[mat_name]
        hau_delta[idxs] = funky_delta[idxs]

    hau_face_pts = (basis[face_idxs] + hau_delta[face_idxs]) if len(face_idxs) else face_pts
    hau_cards = card_indices.get("はぅ", {})
    for side, mat_name in available_eyes:
        if side not in hau_cards:
            continue
        center, eye_w, eye_h = _eye_visible_box(side, mat_name)
        center[2] += eye_h * 0.03
        out_sign = 1.0 if side == "left" else -1.0
        n_card = _normalize(np.array([0.16 * out_sign, -0.985, 0.0], dtype=np.float32))
        u_card = _normalize(np.cross(v_world, n_card))
        target = _curved_patch_coords(
            center, u_card, v_world, n_card, eye_w * 0.62, eye_h * 0.56, face_half_w * 0.75
        )
        target = _conform_front_to_face(target, hau_face_pts, clearance=0.00070)
        c_idx = hau_cards[side]
        hau_delta[c_idx] = target - basis[c_idx]
    results.append(add_or_replace_fn(mesh, basis_block, basis, "はぅ", hau_delta))
    markers["はぅ"] = "@generated:はぅ"

    return results, markers
