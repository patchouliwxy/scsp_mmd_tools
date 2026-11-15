"""Rebuild SCSP iris meshes as alpha-trimmed concentric polar concave bowls."""

from __future__ import annotations

from collections import deque
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import bpy
from mathutils import Vector
import numpy as np


PLANARITY_TOLERANCE_M = 1.0e-5
_ALPHA_THRESHOLD = 0.08
_RADIAL_SEGMENTS = 32
_RING_RHOS = (0.22, 0.48, 0.75, 1.00)
_DEFAULT_FRONT_CLEARANCE_M = 0.0012
_PIVOT_DEPTH_RANGE_M = (0.026, 0.044)
_BOWL_DEPTH_MAX_M = 0.0030
_BOWL_DEPTH_MIN_M = 0.0018


class IrisPlaneError(RuntimeError):
    """Raised when iris geometry inputs fail validation."""


def _basis_points(obj: Any) -> list[Vector]:
    keys = obj.data.shape_keys
    if keys and keys.key_blocks:
        coords = keys.key_blocks[0].data
        return [obj.matrix_world @ coords[i].co for i in range(len(obj.data.vertices))]
    return [obj.matrix_world @ v.co for v in obj.data.vertices]


def _fit_eye_plane(points: Sequence[Vector], reference_forward: Vector) -> tuple[Vector, Vector]:
    arr = np.array([[float(p.x), float(p.y), float(p.z)] for p in points], dtype=np.float64)
    center_np = arr.mean(axis=0)
    _, _, vt = np.linalg.svd(arr - center_np, full_matrices=False)
    normal = Vector((float(vt[-1, 0]), float(vt[-1, 1]), float(vt[-1, 2]))).normalized()
    if normal.dot(reference_forward) < 0.0:
        normal = -normal
    return Vector((float(center_np[0]), float(center_np[1]), float(center_np[2]))), normal


def _find_eye_texture_image(iris_obj: Any) -> Any:
    for slot in iris_obj.material_slots:
        mat = slot.material
        if not mat or not mat.use_nodes or not mat.node_tree:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None and node.image.size[0] > 0:
                return node.image
    source_fbx = iris_obj.get("scsp_source_fbx")
    if source_fbx:
        candidates = sorted(Path(str(source_fbx)).parent.glob("*_eye_col.png"))
        if len(candidates) == 1:
            return bpy.data.images.load(str(candidates[0]), check_existing=True)
    raise IrisPlaneError(f"{iris_obj.name}: unable to locate eye texture image for alpha trimming")


def _largest_opaque_component(image: Any, threshold: float = _ALPHA_THRESHOLD) -> tuple[np.ndarray, np.ndarray]:
    width, height = int(image.size[0]), int(image.size[1])
    channels = int(image.channels)
    if width <= 0 or height <= 0 or channels < 4:
        raise IrisPlaneError(f"Eye texture must be RGBA with positive size, got {(width, height, channels)}")
    pixels = np.array(image.pixels[:], dtype=np.float32).reshape((height, width, channels))
    alpha = pixels[:, :, 3]
    mask = alpha > threshold
    visited = np.zeros_like(mask, dtype=bool)
    best_pts: list[tuple[int, int]] = []
    ys, xs = np.where(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if visited[y0, x0]:
            continue
        queue = deque([(y0, x0)])
        visited[y0, x0] = True
        pts: list[tuple[int, int]] = []
        while queue:
            cy, cx = queue.popleft()
            pts.append((cx, cy))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        if len(pts) > len(best_pts):
            best_pts = pts
    if len(best_pts) < 64:
        raise IrisPlaneError("Eye texture opaque alpha region is too small or missing")
    comp = np.zeros_like(mask, dtype=bool)
    comp_xs = [pt[0] for pt in best_pts]
    comp_ys = [pt[1] for pt in best_pts]
    comp[comp_ys, comp_xs] = True
    return comp, alpha


def _extract_uv_triangles(iris_obj: Any) -> list[tuple[tuple[Vector, Vector, Vector], tuple[Vector, Vector, Vector]]]:
    mesh = iris_obj.data
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        raise IrisPlaneError(f"{iris_obj.name}: missing active UV layer")
    world_pts = _basis_points(iris_obj)
    uv_data = uv_layer.data
    tris = []
    for poly in mesh.polygons:
        loop_indices = list(poly.loop_indices)
        verts = [mesh.loops[li].vertex_index for li in loop_indices]
        uvs = [Vector((float(uv_data[li].uv.x), float(uv_data[li].uv.y))) for li in loop_indices]
        for idx in range(1, len(verts) - 1):
            tris.append((
                (uvs[0], uvs[idx], uvs[idx + 1]),
                (world_pts[verts[0]], world_pts[verts[idx]], world_pts[verts[idx + 1]]),
            ))
    if not tris:
        raise IrisPlaneError(f"{iris_obj.name}: mesh has no triangles")
    return tris


def _barycentric_2d(p: Vector, a: Vector, b: Vector, c: Vector) -> tuple[float, float, float] | None:
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) <= 1.0e-14:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return (u, v, w)


def _in_uv_mesh(u: float, v: float, tris: Sequence[Any]) -> bool:
    p = Vector((u, v))
    for (ua, ub, uc), _ in tris:
        bc = _barycentric_2d(p, ua, ub, uc)
        if bc is not None and bc[0] >= -1.0e-4 and bc[1] >= -1.0e-4 and bc[2] >= -1.0e-4:
            return True
    return False


def _sample_uv_to_world(u: float, v: float, tris: Sequence[Any]) -> Vector:
    p = Vector((u, v))
    best_dist = float("inf")
    best_point: Vector | None = None
    for (ua, ub, uc), (pa, pb, pc) in tris:
        bc = _barycentric_2d(p, ua, ub, uc)
        if bc is None:
            continue
        b0, b1, b2 = bc
        if b0 >= -1.0e-4 and b1 >= -1.0e-4 and b2 >= -1.0e-4:
            return pa * b0 + pb * b1 + pc * b2
        c0 = max(0.0, b0)
        c1 = max(0.0, b1)
        c2 = max(0.0, b2)
        total = c0 + c1 + c2
        if total > 0.0:
            c0, c1, c2 = c0 / total, c1 / total, c2 / total
            uv_clamped = ua * c0 + ub * c1 + uc * c2
            dist = (uv_clamped - p).length
            if dist < best_dist:
                best_dist = dist
                best_point = pa * c0 + pb * c1 + pc * c2
    if best_point is None:
        raise IrisPlaneError("Failed to map iris UV coordinate to 3D surface")
    return best_point


def _trace_opaque_boundary_uvs(
    comp: np.ndarray,
    alpha: np.ndarray,
    tris: Sequence[Any],
    segments: int = _RADIAL_SEGMENTS,
) -> tuple[tuple[float, float], list[tuple[float, float]]]:
    height, width = comp.shape
    ys, xs = np.where(comp)
    sum_w = 0.0
    sum_u = 0.0
    sum_v = 0.0
    valid_us: list[float] = []
    valid_vs: list[float] = []
    for cx, cy in zip(xs.tolist(), ys.tolist()):
        u = (cx + 0.5) / width
        v = (cy + 0.5) / height
        if _in_uv_mesh(u, v, tris):
            w = float(alpha[cy, cx])
            sum_w += w
            sum_u += w * u
            sum_v += w * v
            valid_us.append(u)
            valid_vs.append(v)
    if sum_w <= 0.0 or not valid_us:
        raise IrisPlaneError("Opaque iris texture pixels do not overlap iris UV triangles")
    uc = sum_u / sum_w
    vc = sum_v / sum_w
    max_r = max(math.hypot(max(valid_us) - min(valid_us), max(valid_vs) - min(valid_vs)) * 0.75, 0.05)

    def _is_inside(u: float, v: float) -> bool:
        px = int(u * width)
        py = int(v * height)
        if not (0 <= px < width and 0 <= py < height):
            return False
        return bool(comp[py, px]) and _in_uv_mesh(u, v, tris)

    boundary: list[tuple[float, float]] = []
    steps = 256
    for seg in range(segments):
        angle = 2.0 * math.pi * seg / segments
        du, dv = math.cos(angle), math.sin(angle)
        lo = 0.0
        hi = 0.0
        for step in range(1, steps + 1):
            r = max_r * step / steps
            if _is_inside(uc + r * du, vc + r * dv):
                lo = r
            else:
                hi = r
                break
        if hi == 0.0:
            hi = max_r
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            if _is_inside(uc + mid * du, vc + mid * dv):
                lo = mid
            else:
                hi = mid
        boundary.append((uc + lo * du, vc + lo * dv))
    return (uc, vc), boundary


def fit_eye_kinematics(
    side: str,
    hole_obj: Any,
    face_obj: Any,
    eyelash_obj: Any | None,
    iris_obj: Any,
    reference_forward: Vector,
    *,
    front_clearance: float = _DEFAULT_FRONT_CLEARANCE_M,
) -> dict[str, Any]:
    """Fit an eye rotation sphere center P behind the eye opening so +-20 deg sweep clears lids."""
    eh_pts = _basis_points(hole_obj)
    eh_center, eye_forward = _fit_eye_plane(eh_pts, reference_forward)
    eh_arr = np.array([[p.x, p.y, p.z] for p in eh_pts], dtype=np.float64)
    center_np = np.array([eh_center.x, eh_center.y, eh_center.z], dtype=np.float64)
    fwd_np = np.array([eye_forward.x, eye_forward.y, eye_forward.z], dtype=np.float64)

    d_min, d_max = _PIVOT_DEPTH_RANGE_M
    best_depth = d_min
    best_std = float("inf")
    for depth in np.linspace(d_min, d_max, 73):
        cand = center_np - float(depth) * fwd_np
        radii = np.linalg.norm(eh_arr - cand, axis=1)
        std = float(radii.std())
        if std < best_std:
            best_std = std
            best_depth = float(depth)
    pivot_np = center_np - best_depth * fwd_np
    pivot = Vector((float(pivot_np[0]), float(pivot_np[1]), float(pivot_np[2])))

    # Collect front eyelid / eyelash vertices inside the eye's angular cone to clamp rim_radius.
    lid_arrays = [np.array([[p.x, p.y, p.z] for p in _basis_points(face_obj)], dtype=np.float64)]
    if eyelash_obj is not None:
        lid_arrays.append(np.array([[p.x, p.y, p.z] for p in _basis_points(eyelash_obj)], dtype=np.float64))
    lid_pts = np.vstack(lid_arrays)
    eh_radii = np.linalg.norm(eh_arr - pivot_np, axis=1)
    eh_cos = ((eh_arr - pivot_np) @ fwd_np) / np.maximum(eh_radii, 1.0e-8)
    eh_half_ang = float(np.arccos(np.clip(eh_cos, -1.0, 1.0)).max())
    cone_ang = min(math.radians(62.0), eh_half_ang + math.radians(20.0))

    vecs = lid_pts - pivot_np
    dists = np.linalg.norm(vecs, axis=1)
    cos_ang = (vecs @ fwd_np) / np.maximum(dists, 1.0e-8)
    cone_mask = (cos_ang >= math.cos(cone_ang)) & (dists > 0.012) & (dists < 0.090)
    min_lid_radius = float(dists[cone_mask].min()) if np.any(cone_mask) else float(eh_radii.min())
    rim_radius = min(float(eh_radii.min()), min_lid_radius) - float(front_clearance)
    if rim_radius <= 0.010:
        raise IrisPlaneError(f"{side}: fitted iris sphere radius too small ({rim_radius:.5f} m)")

    return {
        "side": side,
        "pivot_world": pivot,
        "eye_forward_world": eye_forward,
        "eye_hole_center_world": eh_center,
        "pivot_depth_m": best_depth,
        "eye_hole_min_radius_m": float(eh_radii.min()),
        "lid_cone_min_radius_m": min_lid_radius,
        "rim_radius_m": rim_radius,
        "iris_name": iris_obj.name,
    }


def _assign_single_bone_weight(obj: Any, bone_name: str) -> None:
    obj.vertex_groups.clear()
    group = obj.vertex_groups.new(name=bone_name)
    group.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")


def _ensure_forward_winding(
    face_indices: tuple[int, ...],
    world_verts: Sequence[Vector],
    forward: Vector,
) -> tuple[int, ...]:
    pts = [world_verts[idx] for idx in face_indices]
    normal = (pts[1] - pts[0]).cross(pts[2] - pts[0])
    if normal.length_squared <= 1.0e-20 and len(pts) == 4:
        normal = (pts[2] - pts[0]).cross(pts[3] - pts[0])
    if normal.dot(forward) < 0.0:
        return tuple(reversed(face_indices))
    return face_indices


def rebuild_concave_iris(
    side: str,
    iris_obj: Any,
    kinematics: Mapping[str, Any],
    *,
    segments: int = _RADIAL_SEGMENTS,
    ring_rhos: Sequence[float] = _RING_RHOS,
) -> dict[str, Any]:
    """Trim transparent margin via eye alpha and rebuild as a clean polar concave bowl."""
    if iris_obj.mode != "OBJECT":
        raise IrisPlaneError(f"{iris_obj.name}: expected Object mode")
    if len(ring_rhos) < 2 or abs(float(ring_rhos[-1]) - 1.0) > 1.0e-6:
        raise IrisPlaneError("ring_rhos must end at 1.0")

    image = _find_eye_texture_image(iris_obj)
    comp, alpha = _largest_opaque_component(image)
    tris = _extract_uv_triangles(iris_obj)
    (uc, vc), boundary_uvs = _trace_opaque_boundary_uvs(comp, alpha, tris, segments=segments)

    pivot = kinematics["pivot_world"]
    eye_forward = kinematics["eye_forward_world"]
    rim_radius = float(kinematics["rim_radius_m"])

    # Measure original opaque 3D rim radius perpendicular to forward to scale bowl depth naturally.
    center_orig = _sample_uv_to_world(uc, vc, tris)
    rim_orig = [_sample_uv_to_world(u, v, tris) for u, v in boundary_uvs]
    tangent_radii = [((p - center_orig) - eye_forward * (p - center_orig).dot(eye_forward)).length for p in rim_orig]
    tangent_radius = float(sum(tangent_radii) / len(tangent_radii))
    bowl_depth = max(_BOWL_DEPTH_MIN_M, min(_BOWL_DEPTH_MAX_M, 0.22 * tangent_radius))
    pupil_radius = rim_radius - bowl_depth
    if pupil_radius <= 0.008:
        raise IrisPlaneError(f"{iris_obj.name}: bowl_depth exceeds safe pupil radius")

    world_verts: list[Vector] = []
    vertex_uvs: list[tuple[float, float]] = []

    # Vertex 0: central pupil pole.
    center_ray = (center_orig - pivot).normalized()
    world_verts.append(pivot + pupil_radius * center_ray)
    vertex_uvs.append((uc, vc))

    # Rings 0..len(ring_rhos)-1 (each has `segments` vertices).
    for rho in ring_rhos:
        r_bowl = rim_radius - bowl_depth * (1.0 - float(rho) * float(rho))
        for bu, bv in boundary_uvs:
            u = uc + float(rho) * (bu - uc)
            v = vc + float(rho) * (bv - vc)
            p_orig = _sample_uv_to_world(u, v, tris)
            ray = (p_orig - pivot).normalized()
            world_verts.append(pivot + r_bowl * ray)
            vertex_uvs.append((u, v))

    faces: list[tuple[int, ...]] = []
    # Inner triangle fan connecting pole (0) to ring 0 (1 .. segments).
    for seg in range(segments):
        nxt = (seg + 1) % segments
        faces.append(_ensure_forward_winding((0, 1 + seg, 1 + nxt), world_verts, eye_forward))
    # Concentric quad rings.
    for ring_idx in range(len(ring_rhos) - 1):
        inner_start = 1 + ring_idx * segments
        outer_start = 1 + (ring_idx + 1) * segments
        for seg in range(segments):
            nxt = (seg + 1) % segments
            quad = (
                inner_start + seg,
                outer_start + seg,
                outer_start + nxt,
                inner_start + nxt,
            )
            faces.append(_ensure_forward_winding(quad, world_verts, eye_forward))

    inverse = iris_obj.matrix_world.inverted()
    local_verts = [inverse @ pt for pt in world_verts]

    old_mesh = iris_obj.data
    materials = list(old_mesh.materials)
    new_mesh = bpy.data.meshes.new(old_mesh.name + "_ConcaveIris")
    new_mesh.from_pydata(local_verts, [], faces)
    new_mesh.update(calc_edges=True)
    if new_mesh.validate(verbose=False, clean_customdata=False):
        raise IrisPlaneError(f"{iris_obj.name}: rebuilt polar iris mesh is invalid")
    for mat in materials:
        new_mesh.materials.append(mat)
    for poly in new_mesh.polygons:
        poly.use_smooth = True

    uv_layer = new_mesh.uv_layers.new(name="UV0")
    for loop in new_mesh.loops:
        uv_layer.data[loop.index].uv = vertex_uvs[loop.vertex_index]
    new_mesh.uv_layers.active = uv_layer
    uv_layer.active_render = True

    iris_obj.data = new_mesh
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)

    _assign_single_bone_weight(iris_obj, f"EYE_{side}")
    iris_obj["scsp_iris_side"] = side
    iris_obj["scsp_eye_pivot_world"] = [float(c) for c in pivot]
    iris_obj["scsp_eye_forward_world"] = [float(c) for c in eye_forward]
    iris_obj["scsp_iris_rim_radius"] = rim_radius
    iris_obj["scsp_iris_pupil_radius"] = pupil_radius
    iris_obj["scsp_iris_bowl_depth"] = bowl_depth
    iris_obj["scsp_iris_topology"] = json.dumps(
        {
            "type": "polar_concave_bowl",
            "segments": segments,
            "rings": len(ring_rhos),
            "ring_rhos": list(ring_rhos),
            "uv_center": [uc, vc],
        },
        ensure_ascii=False,
    )

    return {
        "side": side,
        "object": iris_obj.name,
        "topology": "polar_concave_bowl",
        "vertices": len(local_verts),
        "faces": len(faces),
        "segments": segments,
        "rings": len(ring_rhos),
        "pivot_world": [float(c) for c in pivot],
        "eye_forward_world": [float(c) for c in eye_forward],
        "rim_radius_m": rim_radius,
        "pupil_radius_m": pupil_radius,
        "bowl_depth_m": bowl_depth,
        "tangent_radius_m": tangent_radius,
        "uv_center": [uc, vc],
    }


def flatten(eyes: Mapping[str, Any], forward: Vector | Sequence[float]) -> dict[str, Any]:
    """Legacy entrypoint kept only for explicit error clarity if called directly."""
    raise IrisPlaneError("flatten() is replaced by rebuild_concave_iris(); run build_eye_sockets()")


__all__ = [
    "IrisPlaneError",
    "PLANARITY_TOLERANCE_M",
    "fit_eye_kinematics",
    "rebuild_concave_iris",
    "flatten",
]
