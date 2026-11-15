"""Structured eye-socket cup geometry for Blender 5.3.

The front ring is the supplied topology loop.  The rear ring is an index-wise
translation by ``forward * depth``.  A regular quad grid closes the rear, with
fixed harmonic boundary weights so every shape key reuses the same topology.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from mathutils import Vector


SCHEMA = 1


def _vector(value: Any, label: str) -> Vector:
    try:
        result = Vector(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a 3D vector") from exc
    if len(result) != 3 or not all(math.isfinite(float(item)) for item in result):
        raise ValueError(f"{label} must contain three finite values")
    return result


def _ring(rim_points: Sequence[Vector]) -> list[Vector]:
    points = [_vector(point, f"rim_points[{index}]") for index, point in enumerate(rim_points)]
    if len(points) < 8:
        raise ValueError("rim_points must contain at least 8 vertices")
    if len(points) % 2:
        raise ValueError("rim_points must contain an even vertex count for a quad grid")
    return points


def _axes(forward: Vector) -> tuple[Vector, Vector]:
    fallback = Vector((0.0, 0.0, 1.0))
    if abs(forward.dot(fallback)) > 0.95:
        fallback = Vector((1.0, 0.0, 0.0))
    u_axis = forward.cross(fallback).normalized()
    return u_axis, u_axis.cross(forward).normalized()


def _project(points: Sequence[Vector], u_axis: Vector, v_axis: Vector) -> list[Vector]:
    return [Vector((point.dot(u_axis), point.dot(v_axis))) for point in points]


def _signed_area(points: Sequence[Vector]) -> float:
    return 0.5 * sum(
        points[index].x * points[(index + 1) % len(points)].y
        - points[(index + 1) % len(points)].x * points[index].y
        for index in range(len(points))
    )


def _orientation(a: Vector, b: Vector, c: Vector) -> float:
    return (b - a).cross(c - a)


def _crosses(a: Vector, b: Vector, c: Vector, d: Vector) -> bool:
    ab_c = _orientation(a, b, c)
    ab_d = _orientation(a, b, d)
    cd_a = _orientation(c, d, a)
    cd_b = _orientation(c, d, b)
    return ab_c * ab_d < -1.0e-14 and cd_a * cd_b < -1.0e-14


def _validate_simple(points: Sequence[Vector]) -> None:
    if abs(_signed_area(points)) <= 1.0e-12:
        raise ValueError("rim_points projected polygon has zero area")
    count = len(points)
    for first in range(count):
        second = (first + 1) % count
        for third in range(first + 1, count):
            fourth = (third + 1) % count
            if len({first, second, third, fourth}) < 4:
                continue
            if _crosses(points[first], points[second], points[third], points[fourth]):
                raise ValueError("rim_points projected polygon self-intersects")


def _perimeter(m: int, n: int) -> list[tuple[int, int]]:
    result = [(column, 0) for column in range(m)]
    result += [(m - 1, row) for row in range(1, n)]
    result += [(column, n - 1) for column in range(m - 2, -1, -1)]
    result += [(0, row) for row in range(n - 2, 0, -1)]
    return result


def _interior(m: int, n: int) -> list[tuple[int, int]]:
    return [(column, row) for row in range(1, n - 1) for column in range(1, m - 1)]


def _start_index(projected: Sequence[Vector], m: int, n: int) -> int:
    count = len(projected)
    best_start = 0
    best_area = -1.0
    for start in range(count):
        indices = (
            start,
            (start + m - 1) % count,
            (start + m + n - 2) % count,
            (start + 2 * m + n - 3) % count,
        )
        corners = [projected[index] for index in indices]
        area = abs(_signed_area(corners))
        if area > best_area + 1.0e-14:
            best_start = start
            best_area = area
    return best_start


def _harmonic_weights(m: int, n: int, boundary_order: Sequence[int]) -> list[list[float]]:
    count = len(boundary_order)
    perimeter = _perimeter(m, n)
    interior = _interior(m, n)
    hidden = {cell: index for index, cell in enumerate(interior)}
    matrix = np.eye(len(interior), dtype=float) * 4.0
    boundary = np.zeros((len(interior), count), dtype=float)
    for cell, row in hidden.items():
        column, vertical = cell
        for neighbor in (
            (column - 1, vertical),
            (column + 1, vertical),
            (column, vertical - 1),
            (column, vertical + 1),
        ):
            other = hidden.get(neighbor)
            if other is None:
                boundary[row, boundary_order[perimeter.index(neighbor)]] += 1.0
            else:
                matrix[row, other] -= 1.0
    weights = np.linalg.solve(matrix, boundary)
    weights[np.abs(weights) < 1.0e-15] = 0.0
    if float(weights.min(initial=0.0)) < -1.0e-10:
        raise ValueError("harmonic cap weights are not non-negative")
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum(axis=1, keepdims=True)
    return [[float(value) for value in row] for row in weights]


def _newell_normal(points: Sequence[Vector]) -> Vector:
    normal = Vector()
    for index, point in enumerate(points):
        following = points[(index + 1) % len(points)]
        normal.x += (point.y - following.y) * (point.z + following.z)
        normal.y += (point.z - following.z) * (point.x + following.x)
        normal.z += (point.x - following.x) * (point.y + following.y)
    return normal


def _validate_cap_quads(positions: Mapping[int, Vector], faces: Sequence[Sequence[int]], forward: Vector) -> None:
    u_axis, v_axis = _axes(forward)
    for index, face in enumerate(faces):
        points = [positions[vertex] for vertex in face]
        if _newell_normal(points).dot(forward) <= 1.0e-16:
            raise ValueError(f"cap face {index} is folded or has zero forward area")
        projected = _project(points, u_axis, v_axis)
        if _crosses(projected[0], projected[1], projected[2], projected[3]) or _crosses(
            projected[1], projected[2], projected[3], projected[0]
        ):
            raise ValueError(f"cap face {index} has crossing non-adjacent edges")


def _build_layout(rim_points: Sequence[Vector], forward: Vector) -> dict[str, Any]:
    count = len(rim_points)
    extent = count // 2 + 2
    columns, rows = extent // 2, extent - extent // 2
    u_axis, v_axis = _axes(forward)
    projected = _project(rim_points, u_axis, v_axis)
    _validate_simple(projected)
    start = _start_index(projected, columns, rows)
    boundary_order = list(range(start, count)) + list(range(start))
    perimeter = _perimeter(columns, rows)
    interior = _interior(columns, rows)
    hidden_weights = _harmonic_weights(columns, rows, boundary_order)

    grid_id: dict[tuple[int, int], int] = {}
    for position, cell in enumerate(perimeter):
        grid_id[cell] = count + boundary_order[position]
    for offset, cell in enumerate(interior):
        grid_id[cell] = 2 * count + offset

    rear = [point - forward * 0.0 for point in rim_points]
    positions = {}
    for cell, index in grid_id.items():
        if index < 2 * count:
            positions[index] = rear[index - count]
        else:
            positions[index] = sum(
                (rear[ring_index] * weight for ring_index, weight in enumerate(hidden_weights[index - 2 * count])),
                Vector(),
            )
    cap_faces = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            cap_faces.append([
                grid_id[(column, row)],
                grid_id[(column + 1, row)],
                grid_id[(column + 1, row + 1)],
                grid_id[(column, row + 1)],
            ])
    cap_normal = sum((_newell_normal([positions[index] for index in face]) for face in cap_faces), Vector())
    if cap_normal.dot(forward) < 0.0:
        cap_faces = [list(reversed(face)) for face in cap_faces]
    _validate_cap_quads(positions, cap_faces, forward)

    cap_directions = {}
    for face in cap_faces:
        for index, first in enumerate(face):
            second = face[(index + 1) % 4]
            cap_directions[tuple(sorted((first, second)))] = (first, second)
    walls = []
    for index in range(count):
        following = (index + 1) % count
        rear_first, rear_following = count + index, count + following
        wall = [index, following, rear_following, rear_first]
        if cap_directions[tuple(sorted((rear_first, rear_following)))] == (rear_following, rear_first):
            wall.reverse()
        walls.append(wall)

    faces = cap_faces + walls
    return {
        "schema": SCHEMA,
        "ring_count": count,
        "grid": [columns, rows],
        "boundary_order": boundary_order,
        "interior_weights": hidden_weights,
        "faces": faces,
        "front_open": True,
    }


def _validate_layout(layout: Mapping[str, Any], count: int) -> None:
    if int(layout.get("schema", -1)) != SCHEMA:
        raise ValueError("unsupported eye-socket layout schema")
    if int(layout.get("ring_count", -1)) != count:
        raise ValueError("layout ring_count differs from rim_points")
    columns, rows = map(int, layout["grid"])
    if 2 * (columns + rows) - 4 != count:
        raise ValueError("layout grid does not match the rim vertex count")
    if len(layout["interior_weights"]) != (columns - 2) * (rows - 2):
        raise ValueError("layout interior weight count differs from its grid")
    if len(layout["faces"]) != count + (columns - 1) * (rows - 1):
        raise ValueError("layout face count differs from its grid")
    for weights in layout["interior_weights"]:
        if len(weights) != count or abs(sum(float(value) for value in weights) - 1.0) > 1.0e-8:
            raise ValueError("layout interior weights must have one row per ring vertex and sum to one")


def make_cup(
    rim_points: Sequence[Vector],
    forward: Vector,
    depth: float = 0.012,
    layout: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one open-front, quad-closed cup for a fixed morph topology."""

    points = _ring(rim_points)
    direction = _vector(forward, "forward")
    if direction.length <= 1.0e-12:
        raise ValueError("forward must be non-zero")
    direction.normalize()
    depth = float(depth)
    if not math.isfinite(depth) or depth <= 0.0:
        raise ValueError("depth must be finite and positive")
    if layout is None:
        layout = _build_layout(points, direction)
    else:
        _validate_layout(layout, len(points))

    rear = [point - direction * depth for point in points]
    vertices = [point.copy() for point in points] + rear
    for weights in layout["interior_weights"]:
        vertices.append(sum((rear[index] * float(weight) for index, weight in enumerate(weights)), Vector()))
    faces = [list(face) for face in layout["faces"]]
    if not all(all(math.isfinite(float(item)) for item in vertex) for vertex in vertices):
        raise ValueError("generated eye-socket cup contains non-finite coordinates")
    return {
        "vertices": vertices,
        "faces": faces,
        "layout": layout,
        "validation": {
            "ring_count": len(points),
            "grid": list(layout["grid"]),
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "quad_count": sum(len(face) == 4 for face in faces),
            "front_ring_exact": all((vertices[index] - point).length <= 1.0e-12 for index, point in enumerate(points)),
            "rear_depth_m": depth,
            "weight_row_sum_max_error": max(
                abs(sum(float(value) for value in row) - 1.0)
                for row in layout["interior_weights"]
            ),
        },
    }


def _ring_any(rim_points: Sequence[Vector]) -> list[Vector]:
    points = [_vector(point, f"rim_points[{index}]") for index, point in enumerate(rim_points)]
    if len(points) < 8:
        raise ValueError("rim_points must contain at least 8 vertices")
    return points


def _nlerp_direction(first: Vector, second: Vector, weight: float) -> Vector:
    blended = first * (1.0 - weight) + second * weight
    if blended.length <= 1.0e-12:
        return first.normalized()
    return blended.normalized()


_FLARE_SUBDIV = 3


def _orbital_faces(
    vertices: Sequence[Vector],
    rim_count: int,
    dense_count: int,
    num_interior_rings: int,
    forward: Vector,
) -> list[list[int]]:
    pole_index = rim_count + num_interior_rings * dense_count
    last_ring_start = rim_count + (num_interior_rings - 1) * dense_count
    pole_point = vertices[pole_index]
    fan_normal = Vector()
    for seg in range(dense_count):
        nxt = (seg + 1) % dense_count
        a = vertices[last_ring_start + seg] - pole_point
        b = vertices[last_ring_start + nxt] - pole_point
        fan_normal += a.cross(b)
    if fan_normal.length_squared <= 1.0e-20:
        raise ValueError("orbital eye-socket pole fan has degenerate area")
    flip = fan_normal.dot(forward) < 0.0

    faces: list[list[int]] = []
    for seg in range(dense_count):
        nxt = (seg + 1) % dense_count
        if flip:
            faces.append([pole_index, last_ring_start + nxt, last_ring_start + seg])
        else:
            faces.append([pole_index, last_ring_start + seg, last_ring_start + nxt])
    for ring_idx in range(num_interior_rings - 1):
        outer_start = rim_count + ring_idx * dense_count
        inner_start = rim_count + (ring_idx + 1) * dense_count
        for seg in range(dense_count):
            nxt = (seg + 1) % dense_count
            if flip:
                faces.append([outer_start + nxt, outer_start + seg, inner_start + seg, inner_start + nxt])
            else:
                faces.append([outer_start + seg, outer_start + nxt, inner_start + nxt, inner_start + seg])
    # Bridge Ring 0 (rim_count) cleanly to dense Ring 1a (3 * rim_count) without T-junctions.
    for seg in range(rim_count):
        nxt = (seg + 1) % rim_count
        b = rim_count + _FLARE_SUBDIV * seg
        bn = rim_count + _FLARE_SUBDIV * nxt
        tris = [
            [seg, b + 1, b],
            [seg, nxt, b + 1],
            [nxt, b + 2, b + 1],
            [nxt, bn, b + 2],
        ]
        if flip:
            faces.extend([list(reversed(tri)) for tri in tris])
        else:
            faces.extend(tris)
    return faces


def _flare_rings(
    rim_points: Sequence[Vector],
    pivot: Vector,
    u2_dense: Sequence[Vector],
    rim_radius: float,
    basis_flare: Sequence[Vector] | None = None,
) -> list[Vector]:
    r1a_min = rim_radius + 0.00066
    r1a_max = rim_radius + 0.00078
    r1b_val = rim_radius + 0.00064
    r1c_val = rim_radius + 0.00058
    ring1a: list[Vector] = []
    ring1b: list[Vector] = []
    ring1c: list[Vector] = []
    count = len(rim_points)
    dense_count = count * _FLARE_SUBDIV

    def _compensated_radius(
        ring_offset: int,
        dense_idx: int,
        u_morph: Vector,
        r_base: float,
        r_ceiling: float,
    ) -> float:
        if basis_flare is None:
            return r_base
        rel_basis = basis_flare[ring_offset * dense_count + dense_idx] - pivot
        r_basis = rel_basis.length
        if r_basis <= 1.0e-8:
            return r_base
        u_basis = rel_basis / r_basis
        dot_prod = max(-0.99, min(1.0, u_basis.dot(u_morph)))
        if dot_prod >= 0.9998:
            return r_base
        cos_half = math.sqrt(0.5 * (1.0 + dot_prod))
        r_target = r_basis + 0.00018 * (1.0 - dot_prod)
        r_needed = (2.0 * r_target / cos_half) - r_basis
        return max(r_base, min(r_ceiling, r_needed))

    for idx in range(count):
        nxt = (idx + 1) % count
        rel_a = rim_points[idx] - pivot
        rel_b = rim_points[nxt] - pivot
        r0_a = rel_a.length
        r0_b = rel_b.length
        u0_a = rel_a / max(r0_a, 1.0e-8)
        u0_b = rel_b / max(r0_b, 1.0e-8)
        for sub in range(_FLARE_SUBDIV):
            dense_idx = _FLARE_SUBDIV * idx + sub
            weight = sub / float(_FLARE_SUBDIV)
            u0 = _nlerp_direction(u0_a, u0_b, weight)
            r0 = r0_a * (1.0 - weight) + r0_b * weight
            u2 = u2_dense[dense_idx]
            u1a = _nlerp_direction(u0, u2, 0.36)
            u1b = _nlerp_direction(u0, u2, 0.58)
            u1c = _nlerp_direction(u0, u2, 0.80)
            r1a_base = max(r1a_min, min(r1a_max, r0 * 0.25 + r1b_val * 0.75))
            r1a_ceil = max(r1a_base, min(r0 - 0.00035, rim_radius + 0.00125))
            r_ceil = max(r1a_base, r0 - 0.00030)
            r1a = _compensated_radius(0, dense_idx, u1a, r1a_base, r1a_ceil)
            r1b = _compensated_radius(1, dense_idx, u1b, r1b_val, r_ceil)
            r1c = _compensated_radius(2, dense_idx, u1c, r1c_val, r_ceil)
            ring1a.append(pivot + r1a * u1a)
            ring1b.append(pivot + r1b * u1b)
            ring1c.append(pivot + r1c * u1c)
    return ring1a + ring1b + ring1c


def _smooth_tangents(
    tangents: Sequence[Vector],
    forward: Vector,
    passes: int,
) -> list[Vector]:
    """Smooth unit radial tangents along a cyclic ring to balance inner spiderweb spacing."""
    current = [t.normalized() for t in tangents]
    count = len(current)
    fallback_u, _ = _axes(forward)
    for _ in range(passes):
        updated: list[Vector] = []
        for idx in range(count):
            blended = 0.25 * current[(idx - 1) % count] + 0.50 * current[idx] + 0.25 * current[(idx + 1) % count]
            blended = blended - blended.dot(forward) * forward
            if blended.length <= 1.0e-8:
                blended = current[idx]
            updated.append(blended.normalized() if blended.length > 1.0e-8 else fallback_u)
        current = updated
    return current


def _egg_faces(
    vertices: Sequence[Vector],
    ring_count: int,
    num_rings: int,
    forward: Vector,
) -> list[list[int]]:
    """Build pure-quad 1x egg-shell rings plus one central back pole triangle fan."""
    pole_index = num_rings * ring_count
    last_ring_start = (num_rings - 1) * ring_count
    pole_point = vertices[pole_index]
    fan_normal = Vector()
    for seg in range(ring_count):
        nxt = (seg + 1) % ring_count
        a = vertices[last_ring_start + seg] - pole_point
        b = vertices[last_ring_start + nxt] - pole_point
        fan_normal += a.cross(b)
    if fan_normal.length_squared <= 1.0e-20:
        raise ValueError("egg eye-socket pole fan has degenerate area")
    flip = fan_normal.dot(forward) < 0.0

    faces: list[list[int]] = []
    for ring_idx in range(num_rings - 1):
        outer_start = ring_idx * ring_count
        inner_start = (ring_idx + 1) * ring_count
        for seg in range(ring_count):
            nxt = (seg + 1) % ring_count
            if flip:
                faces.append([outer_start + nxt, outer_start + seg, inner_start + seg, inner_start + nxt])
            else:
                faces.append([outer_start + seg, outer_start + nxt, inner_start + nxt, inner_start + seg])
    for seg in range(ring_count):
        nxt = (seg + 1) % ring_count
        if flip:
            faces.append([pole_index, last_ring_start + nxt, last_ring_start + seg])
        else:
            faces.append([pole_index, last_ring_start + seg, last_ring_start + nxt])
    return faces


def _egg_cuff_rings(
    rim_points: Sequence[Vector],
    pivot: Vector,
    u2_eq: Sequence[Vector],
    rim_radius: float,
    basis_cuff: Sequence[Vector] | None = None,
) -> list[Vector]:
    """Build Ring 1a..1b midway between moving eyelid rim and egg equator with blink sag compensation."""
    count = len(rim_points)
    ring1a: list[Vector] = []
    ring1b: list[Vector] = []

    def _compensated_radius(
        ring_offset: int,
        idx: int,
        u_curr: Vector,
        r_base: float,
        r_ceil: float,
    ) -> float:
        if basis_cuff is None:
            return r_base
        rel_basis = basis_cuff[ring_offset * count + idx] - pivot
        r_basis = rel_basis.length
        if r_basis <= 1.0e-8:
            return r_base
        u_basis = rel_basis / r_basis
        dot_prod = max(-0.99, min(1.0, u_basis.dot(u_curr)))
        if dot_prod >= 0.9998:
            return r_base
        cos_half = math.sqrt(0.5 * (1.0 + dot_prod))
        r_target = r_basis + 0.00022 * (1.0 - dot_prod)
        r_needed = (2.0 * r_target / cos_half) - r_basis
        return max(r_base, min(r_ceil, r_needed))

    for idx, point in enumerate(rim_points):
        rel = point - pivot
        r0 = rel.length
        if r0 <= 1.0e-8:
            raise ValueError("rim_points cannot coincide with eye pivot")
        u0 = rel / r0
        u1a = _nlerp_direction(u0, u2_eq[idx], 0.35)
        u1b = _nlerp_direction(u0, u2_eq[idx], 0.70)
        r1a_base = min(r0 - 0.00020, rim_radius + 0.00105)
        r1b_base = min(r0 - 0.00025, rim_radius + 0.00095)
        r_ceil = max(r1a_base, r0 - 0.00020)
        r1a = _compensated_radius(0, idx, u1a, r1a_base, r_ceil)
        r1b = _compensated_radius(1, idx, u1b, r1b_base, r_ceil)
        ring1a.append(pivot + r1a * u1a)
        ring1b.append(pivot + r1b * u1b)
    return ring1a + ring1b


def make_concave_socket(
    rim_points: Sequence[Vector],
    pivot: Vector,
    forward: Vector,
    *,
    pupil_radius: float,
    rim_radius: float | None = None,
    socket_clearance: float = 0.0032,
    dome_angle_deg: float = 40.0,
    outer_flare_clearance: float = 0.00042,
    pole_extra_depth: float = 0.0020,
    layout: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a compact 1x egg-shaped concave eye-socket cup behind an iris sphere.

    Ring 0 matches the eyelid contour 1:1 with zero boundary subdivision.
    Rings 1a..1b form a smooth front egg curve between the moving eyelid and the
    equator (Ring 2 at ``dome_angle_deg``), keeping mid-blink chords clear of
    ``rim_radius`` while avoiding wide nose-bridge flare.  Ring 3..Pole close
    the rear into a rounded egg cap strictly behind ``pupil_radius``.
    """
    points = _ring_any(rim_points)
    pivot_vec = _vector(pivot, "pivot")
    direction = _vector(forward, "forward")
    if direction.length <= 1.0e-12:
        raise ValueError("forward must be non-zero")
    direction.normalize()

    count = len(points)
    if layout is not None:
        if int(layout.get("ring_count", -1)) != count or layout.get("type") != "concave_polar":
            raise ValueError("layout does not match concave eye-socket ring")
        if layout.get("variant") == "egg_cup" and "u2_directions" in layout:
            resolved_rim_r = float(layout["rim_radius_m"])
            u2_dirs = [Vector(item) for item in layout["u2_directions"]]
            basis_cuff = (
                [Vector(item) for item in layout["basis_cuff_vertices"]]
                if "basis_cuff_vertices" in layout
                else None
            )
            cuff = _egg_cuff_rings(points, pivot_vec, u2_dirs, resolved_rim_r, basis_cuff=basis_cuff)
            static_tail = [Vector(item) for item in layout["static_tail_vertices"]]
            vertices = [point.copy() for point in points] + cuff + static_tail
        elif "u2_directions" in layout and "static_tail_vertices" in layout:
            resolved_rim_r = float(layout["rim_radius_m"])
            u2_dirs = [Vector(item) for item in layout["u2_directions"]]
            basis_flare = (
                [Vector(item) for item in layout["basis_flare_vertices"]]
                if "basis_flare_vertices" in layout
                else None
            )
            flare = _flare_rings(points, pivot_vec, u2_dirs, resolved_rim_r, basis_flare=basis_flare)
            static_tail = [Vector(item) for item in layout["static_tail_vertices"]]
            vertices = [point.copy() for point in points] + flare + static_tail
        else:
            interior = [Vector(v) for v in layout["interior_vertices"]]
            vertices = [point.copy() for point in points] + interior
        faces = [list(face) for face in layout["faces"]]
        return {
            "vertices": vertices,
            "faces": faces,
            "layout": layout,
            "validation": {
                "ring_count": count,
                "vertex_count": len(vertices),
                "face_count": len(faces),
            },
        }

    base_inner_r = float(pupil_radius) - float(socket_clearance)
    if base_inner_r <= 0.005:
        raise ValueError(f"Invalid concave socket radius: {base_inner_r}")

    if rim_radius is None:
        min_rim_dist = min((point - pivot_vec).length for point in points)
        resolved_rim_r = max(float(pupil_radius) + 0.0020, min(min_rim_dist - 0.0012, float(pupil_radius) + 0.0032))
    else:
        resolved_rim_r = float(rim_radius)
    if resolved_rim_r <= float(pupil_radius):
        raise ValueError(f"rim_radius ({resolved_rim_r}) must exceed pupil_radius ({pupil_radius})")

    dome_rad = math.radians(float(dome_angle_deg))
    facet_margin = resolved_rim_r * (1.0 / math.cos(min(0.45, math.pi * 0.7 / count)) - 1.0)
    r_equator = resolved_rim_r + max(float(outer_flare_clearance), 0.00062 + facet_margin)
    r_pole = max(0.005, base_inner_r - float(pole_extra_depth))

    fallback_u, _ = _axes(direction)
    t_axes_0: list[Vector] = []
    th_egg_vals: list[float] = []
    for point in points:
        rel = point - pivot_vec
        if rel.length <= 1.0e-12:
            raise ValueError("rim_points cannot coincide with eye pivot")
        u0 = rel.normalized()
        z0 = max(-1.0, min(1.0, u0.dot(direction)))
        th0 = math.acos(z0)
        radial = u0 - z0 * direction
        t_axis = fallback_u if radial.length <= 1.0e-8 else radial.normalized()
        t_axes_0.append(t_axis)
        th_egg_vals.append(max(dome_rad, th0 + math.radians(3.0)))

    # Progressively smooth radial directions on deeper static rings so corners don't pinch the pole.
    t_eq = _smooth_tangents(t_axes_0, direction, passes=1)
    t_outer_back = _smooth_tangents(t_axes_0, direction, passes=2)
    t_inner_back = _smooth_tangents(t_axes_0, direction, passes=3)

    u2_dirs = [
        (math.cos(th) * direction + math.sin(th) * t_ax).normalized()
        for th, t_ax in zip(th_egg_vals, t_eq)
    ]
    cuff = _egg_cuff_rings(points, pivot_vec, u2_dirs, resolved_rim_r)

    ring_equator = [pivot_vec + r_equator * u2 for u2 in u2_dirs]
    ring_bowl_outer = [
        pivot_vec + base_inner_r * (
            math.cos(0.95 * th) * direction + math.sin(0.95 * th) * t_ax
        )
        for th, t_ax in zip(th_egg_vals, t_outer_back)
    ]
    ring_bowl_inner = [
        pivot_vec + (base_inner_r - 0.5 * float(pole_extra_depth)) * (
            math.cos(0.48 * th) * direction + math.sin(0.48 * th) * t_ax
        )
        for th, t_ax in zip(th_egg_vals, t_inner_back)
    ]
    pole = [pivot_vec + r_pole * direction]

    static_tail = ring_equator + ring_bowl_outer + ring_bowl_inner + pole
    vertices = [point.copy() for point in points] + cuff + static_tail
    num_rings = 6  # Ring 0 (rim) + Rings 1a..1b (cuff) + Ring 2 (equator) + Ring 3 (outer back) + Ring 4 (inner back)
    faces = _egg_faces(vertices, count, num_rings, direction)
    if not all(all(math.isfinite(float(item)) for item in vertex) for vertex in vertices):
        raise ValueError("generated egg eye-socket contains non-finite coordinates")

    built_layout = {
        "schema": SCHEMA,
        "type": "concave_polar",
        "variant": "egg_cup",
        "ring_count": count,
        "rings": num_rings,
        "faces": faces,
        "rim_radius_m": resolved_rim_r,
        "dome_angle_deg": float(dome_angle_deg),
        "outer_flare_clearance_m": float(outer_flare_clearance),
        "u2_directions": [[float(v.x), float(v.y), float(v.z)] for v in u2_dirs],
        "basis_cuff_vertices": [[float(v.x), float(v.y), float(v.z)] for v in cuff],
        "static_tail_vertices": [[float(v.x), float(v.y), float(v.z)] for v in static_tail],
        "inner_radius_m": base_inner_r,
        "pole_radius_m": r_pole,
    }
    return {
        "vertices": vertices,
        "faces": faces,
        "layout": built_layout,
        "validation": {
            "ring_count": count,
            "dense_count": count,
            "rings": num_rings,
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "quad_count": sum(len(face) == 4 for face in faces),
            "tri_count": sum(len(face) == 3 for face in faces),
            "front_ring_exact": True,
            "inner_radius_m": base_inner_r,
            "outer_equator_radius_m": r_equator,
            "dome_angle_deg": float(dome_angle_deg),
            "pole_radius_m": r_pole,
        },
    }


__all__ = ["make_cup", "make_concave_socket"]
