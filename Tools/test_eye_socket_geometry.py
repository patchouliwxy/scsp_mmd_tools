"""Synthetic Blender tests for enlarged orbital eye-socket geometry and morph stability."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eye_socket_geometry import make_concave_socket


def _elliptical_eye_rim(
    pivot: Vector,
    forward: Vector,
    count: int,
    half_width_m: float,
    half_height_m: float,
    sphere_r_m: float,
    *,
    close_fraction: float = 0.0,
) -> list[Vector]:
    right = Vector((1.0, 0.0, 0.0))
    up = Vector((0.0, 0.0, 1.0))
    points: list[Vector] = []
    for idx in range(count):
        phi = 2.0 * math.pi * idx / count
        x = half_width_m * math.cos(phi)
        y_open = half_height_m * math.sin(phi)
        y = y_open * (1.0 - close_fraction)
        z = math.sqrt(max(sphere_r_m * sphere_r_m - x * x - y * y, 1.0e-6))
        points.append(pivot + x * right + y * up + z * forward)
    return points


def _check_oriented_manifold(faces: list[list[int]]) -> None:
    directed_edges: Counter[tuple[int, int]] = Counter()
    undirected_edges: Counter[tuple[int, int]] = Counter()
    for face in faces:
        n = len(face)
        for i in range(n):
            a, b = face[i], face[(i + 1) % n]
            directed_edges[(a, b)] += 1
            undirected_edges[(min(a, b), max(a, b))] += 1
    assert max(undirected_edges.values()) == 2, "Non-manifold edge with >2 faces"
    for (a, b), count in directed_edges.items():
        assert count == 1, f"Duplicate directed edge {(a, b)} (flipped face winding)"
    boundary = [edge for edge, count in undirected_edges.items() if count == 1]
    assert boundary, "Expected open front boundary loop"
    for a, b in undirected_edges:
        if undirected_edges[(a, b)] == 2:
            assert directed_edges[(a, b)] == 1 and directed_edges[(b, a)] == 1, (
                f"Inconsistent winding across shared edge {(a, b)}"
            )


def _check_blender_mesh(name: str, vertices: list[Vector], faces: list[list[int]]) -> None:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in vertices], [], faces)
    mesh.update(calc_edges=True)
    invalid = mesh.validate(verbose=False, clean_customdata=False)
    bpy.data.meshes.remove(mesh)
    assert not invalid, f"{name}: Blender mesh.validate() reported errors"


def test_orbital_socket_for_count(count: int) -> None:
    pivot = Vector((0.0, 0.045, 1.40))
    forward = Vector((0.0, -1.0, 0.0))
    rim_radius = 0.0445
    pupil_radius = 0.0415
    socket_clearance = 0.0032
    base_inner_r = pupil_radius - socket_clearance

    rim_open = _elliptical_eye_rim(pivot, forward, count, 0.021, 0.011, 0.0462)
    basis = make_concave_socket(
        rim_open,
        pivot,
        forward,
        pupil_radius=pupil_radius,
        rim_radius=rim_radius,
        socket_clearance=socket_clearance,
        dome_angle_deg=40.0,
    )
    json.dumps(basis["layout"], allow_nan=False)
    verts = basis["vertices"]
    faces = basis["faces"]
    assert basis["validation"]["dense_count"] == count
    assert len(verts) == 6 * count + 1
    assert len(faces) == 6 * count
    _check_oriented_manifold(faces)
    _check_blender_mesh(f"TestEggCup_{count}", verts, faces)

    # Ring 3 (egg equator: Rim 0 + Cuff 1a, 1b + Equator) reaches >= 40 deg without 64 deg nose-bridge bloat.
    equator_start = 3 * count
    ring_equator = verts[equator_start : equator_start + count]
    ring_equator_angles = [
        math.degrees(math.acos(max(-1.0, min(1.0, (p - pivot).normalized().dot(forward)))))
        for p in ring_equator
    ]
    assert min(ring_equator_angles) >= 39.9, f"Expected >=40 deg egg equator, got {min(ring_equator_angles):.2f}"

    # Outer front cuff + equator (Ring 1a..Equator) stay outside rim_radius; inner cap stays <= base_inner_r.
    for ring_idx in (1, 2, 3):
        start = ring_idx * count
        for p in verts[start : start + count]:
            assert (p - pivot).length > rim_radius + 0.00035, f"Egg cuff/equator ring {ring_idx} too close to iris"
    inner_bowl_start = 4 * count
    for p in verts[inner_bowl_start:]:
        assert (p - pivot).length <= base_inner_r + 1.0e-6, "Inner egg cap intruded past base_inner_r"

    # Morph compatibility (closed eye + 0.5 linear blend check).
    rim_closed = _elliptical_eye_rim(pivot, forward, count, 0.021, 0.011, 0.0462, close_fraction=0.98)
    closed = make_concave_socket(
        rim_closed,
        pivot,
        forward,
        pupil_radius=pupil_radius,
        rim_radius=rim_radius,
        socket_clearance=socket_clearance,
        layout=basis["layout"],
    )
    assert closed["faces"] == faces
    assert len(closed["vertices"]) == len(verts)
    # Static egg shell (Equator..Pole) stays stationary during eyelid blink.
    static_start = 3 * count
    for idx in range(static_start, len(verts)):
        assert (closed["vertices"][idx] - verts[idx]).length <= 1.0e-7

    # Half-blink (0.5) linear interpolation on front rings must stay strictly outside rim_radius.
    half_verts = [a * 0.5 + b * 0.5 for a, b in zip(verts, closed["vertices"])]
    outer_limit = 4 * count
    for idx in range(count, outer_limit):
        assert (half_verts[idx] - pivot).length > rim_radius + 0.00015, (
            f"Half-blink vertex {idx} dipped inside iris rim_radius"
        )
    for face in faces:
        if all(count <= vi < outer_limit for vi in face):
            centroid = sum((half_verts[vi] for vi in face), Vector()) / len(face)
            assert (centroid - pivot).length > rim_radius + 0.00015, (
                "Half-blink front egg face centroid sagged inside iris rim_radius"
            )


def test_backwards_compat_and_validation() -> None:
    pivot = Vector((0.0, 0.045, 1.40))
    forward = Vector((0.0, -1.0, 0.0))
    rim_open = _elliptical_eye_rim(pivot, forward, 16, 0.020, 0.011, 0.0462)
    fallback = make_concave_socket(rim_open, pivot, forward, pupil_radius=0.0415)
    assert fallback["layout"]["variant"] == "egg_cup"
    try:
        make_concave_socket(rim_open, pivot, forward, pupil_radius=0.045, rim_radius=0.044)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError when rim_radius <= pupil_radius")


def run() -> None:
    test_orbital_socket_for_count(19)
    test_orbital_socket_for_count(24)
    test_orbital_socket_for_count(26)
    test_backwards_compat_and_validation()
    print("SCSP_EYE_SOCKET_GEOMETRY_TEST", json.dumps({"status": "PASS", "tested_counts": [19, 24, 26]}))


if __name__ == "__main__":
    run()
