"""Head-bound eye sockets with preserved source geometry and shape keys."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import bpy
from mathutils import Vector


__all__ = ["EyeSocketError", "enforce_eye_bindings", "preflight", "build_eye_sockets"]

SCHEMA = 1
WEIGHT_EPSILON = 1.0e-7
EYE_BONES = ("EYE_L", "EYE_R")
HEAD_BONE = "Head"
EYE_PIECES = {"L": "eyesclera_L", "R": "eyesclera_R"}
EYELASH_PIECES = {"L": "eyelash_L", "R": "eyelash_R"}
SOURCE_HOLES = {"L": "eye_hole_L", "R": "eye_hole_R"}
SOCKET_NAMES = {"L": "SCSP_EyeSocket_L", "R": "SCSP_EyeSocket_R"}
MATERIAL_NAME = "SCSP_EyeSocketSclera"
SOCKET_TAG = "scsp_eye_socket_side"
SOURCE_REFERENCE_TAG = "scsp_eye_socket_source_reference"
DEFAULT_OPTIONS = {
    "back_distance": 0.012,
    "contour_clearance": 0.0,
    "iris_retreat": 0.0005,
    "front_clearance": 0.0012,
    "socket_clearance": 0.0032,
    "dome_angle_deg": 40.0,
    "outer_flare_clearance": 0.00042,
}


class EyeSocketError(RuntimeError):
    """Raised when eye bindings or socket geometry cannot be resolved safely."""


def _source_name(obj: Any) -> str:
    getter = getattr(obj, "get", lambda _key, _default=None: None)
    return str(getter("scsp_source_object", "") or "")


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _json_vector(value: Vector) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _validate_parts(parts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(parts, (list, tuple)):
        raise EyeSocketError("parts must be a list or tuple of mappings")
    plan = []
    seen_meshes = set()
    for index, part in enumerate(parts):
        if not isinstance(part, Mapping):
            raise EyeSocketError(f"parts[{index}] must be a mapping")
        kind = str(part.get("kind", "")).strip()
        if not kind:
            raise EyeSocketError(f"parts[{index}].kind must be non-empty")
        arm = part.get("arm")
        if arm is not None and getattr(arm, "type", None) != "ARMATURE":
            raise EyeSocketError(f"parts[{index}].arm must be an armature or None")
        objects = list(part.get("objects") or [])
        meshes = list(part.get("meshes") or [])
        object_ids = {obj.as_pointer() for obj in objects if hasattr(obj, "as_pointer")}
        for mesh in meshes:
            if getattr(mesh, "type", None) != "MESH":
                raise EyeSocketError(f"parts[{index}] mesh is not a mesh: {mesh!r}")
            pointer = mesh.as_pointer()
            if pointer in seen_meshes:
                raise EyeSocketError(f"Mesh occurs in more than one part: {mesh.name}")
            seen_meshes.add(pointer)
            if pointer not in object_ids:
                raise EyeSocketError(f"Mesh is missing from its part objects list: {mesh.name}")
        plan.append({
            "raw": part,
            "kind": kind,
            "arm": arm,
            "objects": objects,
            "meshes": meshes,
        })
    return plan


def _head_parts(plan: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [part for part in plan if str(part["kind"]).casefold() == "head"]


def _unique_source_mesh(
    head_parts: Sequence[Mapping[str, Any]], source_name: str
) -> tuple[Mapping[str, Any], Any]:
    matches = [
        (part, mesh)
        for part in head_parts
        for mesh in part["meshes"]
        if _source_name(mesh).casefold() == source_name.casefold()
    ]
    if len(matches) != 1:
        raise EyeSocketError(
            f"Expected exactly one head mesh for source object {source_name!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _weights(mesh: Any, index: int) -> dict[str, float]:
    vertex = mesh.data.vertices[index]
    return {
        mesh.vertex_groups[item.group].name: float(item.weight)
        for item in vertex.groups
        if item.weight > WEIGHT_EPSILON
    }


def _weight_stats(mesh: Any) -> dict[str, dict[str, float | int]]:
    stats = {
        name: {"positive_vertices": 0, "weight_sum": 0.0}
        for name in (*EYE_BONES, HEAD_BONE)
    }
    for vertex in mesh.data.vertices:
        memberships = {
            mesh.vertex_groups[item.group].name: float(item.weight)
            for item in vertex.groups
            if item.weight > WEIGHT_EPSILON
        }
        for name, weight in memberships.items():
            if name in stats:
                stats[name]["positive_vertices"] += 1
                stats[name]["weight_sum"] += weight
    return stats


def _full_eye_piece(mesh: Any, bone_name: str) -> bool:
    group = mesh.vertex_groups.get(bone_name)
    if group is None or not mesh.data.vertices:
        return False
    for vertex in mesh.data.vertices:
        weight = next(
            (float(item.weight) for item in vertex.groups if item.group == group.index),
            0.0,
        )
        if abs(weight - 1.0) > 1.0e-6:
            return False
    return True


def _transfer_eye_weights(mesh: Any, protected_bone: str | None) -> dict[str, Any]:
    eye_groups = {name: mesh.vertex_groups.get(name) for name in EYE_BONES}
    head_group = mesh.vertex_groups.get(HEAD_BONE)
    changed = set()
    protected_completed = set()
    transferred = 0.0
    head_added = 0.0
    other_error = 0.0
    total_error = 0.0
    protected_changed = False

    if protected_bone is not None:
        protected_group = eye_groups[protected_bone]
        if protected_group is None:
            protected_group = mesh.vertex_groups.new(name=protected_bone)
            eye_groups[protected_bone] = protected_group
            protected_changed = True
        if not _full_eye_piece(mesh, protected_bone):
            protected_changed = True
            for vertex in mesh.data.vertices:
                before = _weights(mesh, vertex.index)
                if abs(before.get(protected_bone, 0.0) - 1.0) > 1.0e-6:
                    protected_group.add([vertex.index], 1.0, "REPLACE")
                    protected_completed.add(vertex.index)
                    changed.add(vertex.index)

    for vertex in mesh.data.vertices:
        before = _weights(mesh, vertex.index)
        moving_names = [
            name for name in EYE_BONES
            if name != protected_bone and before.get(name, 0.0) > WEIGHT_EPSILON
        ]
        if not moving_names:
            continue
        if head_group is None:
            head_group = mesh.vertex_groups.new(name=HEAD_BONE)
        moving = sum(before[name] for name in moving_names)
        head_group.add(
            [vertex.index],
            before.get(HEAD_BONE, 0.0) + moving,
            "REPLACE",
        )
        for name in moving_names:
            eye_groups[name].remove([vertex.index])
        after = _weights(mesh, vertex.index)
        head_added += moving
        transferred += moving
        changed.add(vertex.index)
        before_other = {
            name: weight for name, weight in before.items()
            if name not in (*EYE_BONES, HEAD_BONE)
        }
        after_other = {
            name: weight for name, weight in after.items()
            if name not in (*EYE_BONES, HEAD_BONE)
        }
        for name in set(before_other) | set(after_other):
            other_error = max(
                other_error,
                abs(before_other.get(name, 0.0) - after_other.get(name, 0.0)),
            )
        total_error = max(total_error, abs(sum(before.values()) - sum(after.values())))

    return {
        "changed_vertices": len(changed),
        "transferred_weight": transferred,
        "head_weight_added": head_added,
        "protected_bone": protected_bone,
        "protected_eye_changed": protected_changed,
        "protected_weight_completed_vertices": len(protected_completed),
        "other_weight_max_error": other_error,
        "total_weight_max_error": total_error,
    }


def enforce_eye_bindings(parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep Eye bones on eyesclera_L/R and move other head Eye weights to Head."""

    plan = _validate_parts(parts)
    heads = _head_parts(plan)
    if not heads:
        raise EyeSocketError("No head part was supplied")
    eye_pieces = {}
    for side, source_name in EYE_PIECES.items():
        _, mesh = _unique_source_mesh(heads, source_name)
        eye_pieces[mesh.as_pointer()] = {
            "side": side,
            "bone": f"EYE_{side}",
            "mesh": mesh,
        }

    records = []
    for part in heads:
        arm = part["arm"]
        if arm is None or arm.data.bones.get(HEAD_BONE) is None:
            raise EyeSocketError(f"Head part {part['kind']!r} has no Head bone")
        for mesh in part["meshes"]:
            before = _weight_stats(mesh)
            protected = eye_pieces.get(mesh.as_pointer())
            changes = _transfer_eye_weights(
                mesh, protected["bone"] if protected else None
            )
            after = _weight_stats(mesh)
            records.append({
                "mesh": mesh.name,
                "source_object": _source_name(mesh),
                "part_kind": part["kind"],
                "before": before,
                "after": after,
                "changed_vertices": changes["changed_vertices"],
                "transfer": changes,
                "protected_eye_piece": bool(protected),
                "protected_eye_bone": changes["protected_bone"],
            })

    non_eye_eye_weight = 0.0
    protected_ok = True
    for record in records:
        if record["protected_eye_piece"]:
            bone = record["protected_eye_bone"]
            vertex_count = next(
                item["mesh"].data.vertices.__len__()
                for item in eye_pieces.values()
                if item["mesh"].name == record["mesh"]
            )
            stats = record["after"][bone]
            protected_ok = protected_ok and (
                stats["positive_vertices"] == vertex_count
                and abs(stats["weight_sum"] - vertex_count) <= 1.0e-5
            )
        else:
            non_eye_eye_weight += sum(
                float(record["after"][name]["weight_sum"]) for name in EYE_BONES
            )
    other_ok = all(
        record["transfer"]["other_weight_max_error"] <= 1.0e-5
        and (
            record["transfer"]["protected_weight_completed_vertices"] > 0
            or record["transfer"]["total_weight_max_error"] <= 1.0e-5
        )
        for record in records
    )
    report = {
        "schema": SCHEMA,
        "operation": "enforce_eye_bindings",
        "scope": "head parts only; eyesclera_L/R retain the matching Eye bone",
        "blender": bpy.app.version_string,
        "protected_eye_pieces": sorted(item["mesh"].name for item in eye_pieces.values()),
        "meshes": records,
        "validation": {
            "non_eye_eye_weight_sum": non_eye_eye_weight,
            "protected_eye_pieces_weight_one": protected_ok,
            "other_weights_and_weight_sums_preserved": other_ok,
            "changed_mesh_count": sum(
                record["changed_vertices"] > 0 for record in records
            ),
        },
    }
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    return report

def _boundary_loops(mesh: Any, frame: Any | None = None) -> list[dict[str, Any]]:
    edge_faces = defaultdict(int)
    for polygon in mesh.data.polygons:
        vertices = list(polygon.vertices)
        for index, first in enumerate(vertices):
            second = vertices[(index + 1) % len(vertices)]
            edge_faces[tuple(sorted((int(first), int(second))))] += 1
    nonmanifold = [edge for edge, count in edge_faces.items() if count > 2]
    boundary = [edge for edge, count in edge_faces.items() if count == 1]
    if nonmanifold:
        raise EyeSocketError(
            f"{mesh.name}: non-manifold edges prevent boundary resolution: {nonmanifold}"
        )
    if not boundary:
        raise EyeSocketError(f"{mesh.name}: mesh has no boundary loops")
    adjacency = defaultdict(list)
    for first, second in boundary:
        adjacency[first].append(second)
        adjacency[second].append(first)
    bad = {
        vertex: len(edges)
        for vertex, edges in adjacency.items()
        if len(edges) != 2
    }
    if bad:
        raise EyeSocketError(
            f"{mesh.name}: boundary vertices must have degree two: {bad}"
        )
    components = []
    unseen = set(adjacency)
    while unseen:
        seed = min(unseen)
        stack = [seed]
        component = set()
        while stack:
            vertex = stack.pop()
            if vertex in component:
                continue
            component.add(vertex)
            unseen.discard(vertex)
            stack.extend(adjacency[vertex])
        components.append(component)
    result = []
    for component in components:
        start = min(component)
        current = start
        previous = None
        loop = []
        for _ in range(len(component) + 1):
            loop.append(current)
            candidates = [item for item in adjacency[current] if item != previous]
            if not candidates:
                break
            previous, current = current, candidates[0]
            if current == start:
                break
        if current != start or set(loop) != component:
            raise EyeSocketError(
                f"{mesh.name}: boundary loop cannot be ordered without guessing"
            )
        points = [(frame or mesh.matrix_world) @ mesh.data.vertices[index].co for index in loop]
        center = sum(points, Vector((0.0, 0.0, 0.0))) / len(points)
        result.append({
            "vertices": loop,
            "edges": [
                (loop[index], loop[(index + 1) % len(loop)])
                for index in range(len(loop))
            ],
            "center_world": center,
            "radius_m": max((point - center).length for point in points),
        })
    return sorted(
        result,
        key=lambda item: (
            len(item["vertices"]),
            item["center_world"].x,
            item["center_world"].y,
            item["center_world"].z,
        ),
    )


def _source_boundary(mesh: Any, frame: Any | None = None) -> dict[str, Any]:
    loops = _boundary_loops(mesh, frame)
    if len(loops) != 1:
        raise EyeSocketError(
            f"{mesh.name}: expected one source eye-hole boundary, found {len(loops)} "
            f"with vertex counts {sorted(len(item['vertices']) for item in loops)}"
        )
    return loops[0]

def _face_normal_world(mesh: Any, frame: Any | None = None) -> tuple[Vector, float]:
    frame = frame or mesh.matrix_world
    normal_matrix = frame.to_3x3().inverted().transposed()
    total = Vector((0.0, 0.0, 0.0))
    area = 0.0
    for polygon in mesh.data.polygons:
        world_normal = normal_matrix @ polygon.normal
        if world_normal.length_squared <= 1.0e-20:
            continue
        total += world_normal.normalized() * float(polygon.area)
        area += float(polygon.area)
    if total.length <= 1.0e-12:
        raise EyeSocketError(f"{mesh.name}: eye-hole cap has no stable normal")
    return total.normalized(), area


def _world_center(mesh: Any, frame: Any | None = None) -> Vector:
    if not mesh.data.vertices:
        raise EyeSocketError(f"{mesh.name}: eye hole has no vertices")
    frame = frame or mesh.matrix_world
    points = [frame @ vertex.co for vertex in mesh.data.vertices]
    return sum(points, Vector((0.0, 0.0, 0.0))) / len(points)


def _derive_forward(
    holes: Mapping[str, Any], arm: Any, source_frame: Any | None = None
) -> tuple[Vector, dict[str, Any]]:
    head_bone = arm.data.bones.get(HEAD_BONE)
    if head_bone is None:
        raise EyeSocketError(f"{arm.name}: missing Head bone")
    head_origin = arm.matrix_world @ head_bone.head_local
    centers = {
        side: _world_center(mesh, source_frame) for side, mesh in holes.items()
    }
    head_to_eyes = (
        sum(centers.values(), Vector((0.0, 0.0, 0.0))) / len(centers)
        - head_origin
    )
    if head_to_eyes.length <= 1.0e-10:
        raise EyeSocketError(
            "Head bone and eye-hole centers do not define a facing direction"
        )
    raw = {}
    aligned = {}
    for side, mesh in holes.items():
        normal, area = _face_normal_world(mesh, source_frame)
        if area <= 0.0:
            raise EyeSocketError(f"{mesh.name}: eye-hole cap has no area")
        raw[side] = normal
        aligned[side] = -normal if normal.dot(head_to_eyes) < 0.0 else normal
    agreement = aligned["L"].dot(aligned["R"])
    if agreement < 0.5:
        raise EyeSocketError(
            f"Eye-hole normals disagree after head alignment: dot={agreement:.6f}"
        )
    forward = (aligned["L"] + aligned["R"]).normalized()
    if forward.dot(head_to_eyes) < 0.0:
        forward.negate()
    minimum_dot = min(forward.dot(normal) for normal in aligned.values())
    if minimum_dot < 0.5:
        raise EyeSocketError(
            f"Derived forward conflicts with a source normal: min dot={minimum_dot:.6f}"
        )
    return forward, {
        "head_bone": HEAD_BONE,
        "head_origin_world": _json_vector(head_origin),
        "eye_centers_world": {
            side: _json_vector(point) for side, point in centers.items()
        },
        "head_to_eyes_world": _json_vector(head_to_eyes.normalized()),
        "raw_normals_world": {
            side: _json_vector(normal) for side, normal in raw.items()
        },
        "aligned_normals_world": {
            side: _json_vector(normal) for side, normal in aligned.items()
        },
        "normal_agreement_dot": float(agreement),
        "minimum_normal_dot": float(minimum_dot),
        "selected_forward_world": _json_vector(forward),
        "selected_backward_world": _json_vector(-forward),
    }


def preflight(parts, options=None):
    """Run every socket geometry check that precedes the first scene mutation."""
    settings = dict(DEFAULT_OPTIONS)
    settings.update(options or {})
    depth = float(settings["back_distance"])
    clearance = float(settings["contour_clearance"])
    retreat = float(settings["iris_retreat"])
    if not (0 < depth < 0.05 and 0 <= clearance < depth and 0 <= retreat < depth):
        raise EyeSocketError("Invalid socket depth/clearance/iris retreat")

    plan = _validate_parts(parts)
    heads = _head_parts(plan)
    holes = {}
    eyes = {}
    owners = {}
    for side in ("L", "R"):
        owners[side], holes[side] = _unique_source_mesh(
            heads, SOURCE_HOLES[side]
        )
        _, eyes[side] = _unique_source_mesh(heads, EYE_PIECES[side])

    head = owners["L"]
    arm = head["arm"]
    if owners["R"]["arm"] != arm or not arm or HEAD_BONE not in arm.data.bones:
        raise EyeSocketError("Eye parts need one head armature with Head bone")
    _, face = _unique_source_mesh(heads, "face")
    if not face.data.shape_keys:
        raise EyeSocketError("Source face needs a Basis key")
    face_keys = list(face.data.shape_keys.key_blocks)
    if face_keys[0].name != "Basis":
        raise EyeSocketError("Unexpected face Basis")

    _matches, match_report = _match_face_eye_loops(face, holes, eyes)
    _forward, direction_report = _derive_forward(
        holes, arm, face.matrix_world
    )
    report = {
        "schema": 2,
        "operation": "eye_socket_geometry_preflight",
        "status": "ok",
        "blender": bpy.app.version_string,
        "parameters": {
            "back_distance": depth,
            "contour_clearance": clearance,
            "iris_retreat": retreat,
        },
        "orientation": direction_report,
        "face_contour_matches": match_report,
        "validation": {
            "read_only": True,
            "scene_modified": False,
            "source_meshes_resolved": True,
            "face_boundary_resolved": True,
            "forward_direction_resolved": True,
        },
    }
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    return report


def _shape_keys(mesh: Any) -> list[Any]:
    keys = mesh.data.shape_keys
    if keys is None or len(keys.key_blocks) < 2:
        raise EyeSocketError(f"{mesh.name}: source eye hole has no usable shape keys")
    blocks = list(keys.key_blocks)
    if blocks[0].name != "Basis":
        raise EyeSocketError(f"{mesh.name}: first shape key must be Basis")
    return blocks

def _tangent_axes(forward: Vector) -> tuple[Vector, Vector]:
    fallback = Vector((0.0, 0.0, 1.0))
    if abs(forward.dot(fallback)) > 0.95:
        fallback = Vector((1.0, 0.0, 0.0))
    u_axis = forward.cross(fallback).normalized()
    v_axis = u_axis.cross(forward).normalized()
    return u_axis, v_axis


def _canonical_contour(
    points: Sequence[Vector], forward: Vector
) -> tuple[list[int], list[Vector], list[float], float]:
    center = sum(points, Vector((0.0, 0.0, 0.0))) / len(points)
    u_axis, v_axis = _tangent_axes(forward)
    coordinates = [
        ((point - center).dot(u_axis), (point - center).dot(v_axis))
        for point in points
    ]
    signed_area = 0.5 * sum(
        coordinates[index][0] * coordinates[(index + 1) % len(points)][1]
        - coordinates[(index + 1) % len(points)][0] * coordinates[index][1]
        for index in range(len(points))
    )
    order = list(range(len(points)))
    if signed_area < 0.0:
        order.reverse()
    angles = [
        math.atan2(coordinates[index][1], coordinates[index][0])
        for index in order
    ]
    start = min(
        range(len(order)),
        key=lambda index: (
            abs(math.atan2(math.sin(angles[index]), math.cos(angles[index]))),
            angles[index] % (2.0 * math.pi),
        ),
    )
    order = order[start:] + order[:start]
    ordered_points = [points[index] for index in order]
    lengths = [
        (ordered_points[(index + 1) % len(points)] - ordered_points[index]).length
        for index in range(len(points))
    ]
    total = sum(lengths)
    if total <= 1.0e-12:
        raise EyeSocketError("Eye opening contour has zero arc length")
    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    parameters = [value / total for value in cumulative[:-1]]
    return order, ordered_points, parameters, abs(signed_area)


def _map_source_loop_to_face(
    face: Any,
    hole_obj: Any,
    source_loop: Mapping[str, Any],
) -> tuple[list[int], float]:
    """Map each ordered boundary vertex of eye_hole_{L,R} to its coincident face vertex."""
    hole_keys = hole_obj.data.shape_keys
    hole_basis = hole_keys.key_blocks[0].data if hole_keys and hole_keys.key_blocks else None
    face_keys = face.data.shape_keys
    face_basis = face_keys.key_blocks[0].data if face_keys and face_keys.key_blocks else None

    face_world = [
        face.matrix_world @ (face_basis[i].co if face_basis is not None else v.co)
        for i, v in enumerate(face.data.vertices)
    ]
    matched: list[int] = []
    max_dist = 0.0
    used_face: set[int] = set()
    for hole_idx in source_loop["vertices"]:
        co = hole_basis[hole_idx].co if hole_basis is not None else hole_obj.data.vertices[hole_idx].co
        hole_world = hole_obj.matrix_world @ co
        best_idx = -1
        best_dist = float("inf")
        for f_idx, f_pos in enumerate(face_world):
            d = (f_pos - hole_world).length
            if d < best_dist:
                best_dist = d
                best_idx = f_idx
        if best_idx < 0 or best_dist > 1.0e-4:
            raise EyeSocketError(
                f"{hole_obj.name} vertex {hole_idx} does not coincide with {face.name} "
                f"(nearest {best_dist:.6f} m)"
            )
        if best_idx in used_face:
            raise EyeSocketError(
                f"{hole_obj.name} maps multiple rim vertices to {face.name} vertex {best_idx}"
            )
        used_face.add(best_idx)
        matched.append(best_idx)
        if best_dist > max_dist:
            max_dist = best_dist
    return matched, max_dist


def _match_face_eye_loops(
    face: Any,
    holes: Mapping[str, Any],
    eyes: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    matches = {}
    report = {}
    for side in ("L", "R"):
        eye_center = _world_center(eyes[side])
        source_loop = _source_boundary(holes[side])
        source_center = source_loop["center_world"]
        face_vertices, max_match_dist = _map_source_loop_to_face(face, holes[side], source_loop)
        face_loop = {
            "vertices": face_vertices,
            "edges": [
                (face_vertices[i], face_vertices[(i + 1) % len(face_vertices)])
                for i in range(len(face_vertices))
            ],
            "center_world": source_center,
            "radius_m": float(source_loop["radius_m"]),
        }
        matches[side] = {
            "loop": face_loop,
            "source_loop": source_loop,
        }
        source_distance = (source_center - eye_center).length
        source_radius = float(source_loop["radius_m"])
        report[side] = {
            "face_loop_vertices": len(face_vertices),
            "face_loop_edges": len(face_vertices),
            "face_loop_center_world": _json_vector(source_center),
            "face_loop_radius_m": source_radius,
            "iris_center_world": _json_vector(eye_center),
            "source_hole_vertices": len(source_loop["vertices"]),
            "source_hole_count_matches_face": True,
            "source_hole_max_face_distance_m": float(max_match_dist),
            "source_hole_center_to_iris_m": float(source_distance),
            "source_hole_position_compatible": (
                source_distance <= max(0.015, source_radius * 3.0)
            ),
        }
    return matches, report


def _locate_sclera_texture(
    holes: Mapping[str, Any], options: Mapping[str, Any]
) -> Path:
    supplied = options.get("sclera_texture")
    if supplied:
        path = Path(str(supplied)).expanduser().resolve()
        if not path.is_file():
            raise EyeSocketError(
                f"Configured eye sclera texture does not exist: {path}"
            )
        if not path.name.casefold().startswith("eye_sclera") or path.suffix.casefold() != ".png":
            raise EyeSocketError(
                f"Eye socket texture must be eye_sclera*.png, got {path.name}"
            )
        return path
    candidates = set()
    for mesh in holes.values():
        source_fbx = mesh.get("scsp_source_fbx")
        if source_fbx:
            parent = Path(str(source_fbx)).parent
            for item in parent.glob("eye_sclera*.png"):
                if item.is_file():
                    candidates.add(item.resolve())
    existing = sorted(candidates)
    if len(existing) != 1:
        raise EyeSocketError(
            "Expected one source eye_sclera*.png next to the head FBX; found "
            + repr([str(path) for path in existing])
        )
    return existing[0]


def _material(texture_path: Path) -> Any:
    material = bpy.data.materials.get(MATERIAL_NAME)
    if (
        material is not None
        and material.get("scsp_eye_socket_material") != "sclera"
    ):
        raise EyeSocketError(
            f"{MATERIAL_NAME} already exists and is not owned by eye_sockets"
        )
    if material is None:
        material = bpy.data.materials.new(name=MATERIAL_NAME)
        material["scsp_eye_socket_material"] = "sclera"
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (360.0, 0.0)
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (80.0, 0.0)
    texture = nodes.new("ShaderNodeTexImage")
    texture.location = (-220.0, 0.0)
    texture.image = bpy.data.images.load(
        str(texture_path), check_existing=True
    )
    texture.image.colorspace_settings.name = "sRGB"
    texture.interpolation = "Linear"
    shader.inputs["Roughness"].default_value = 0.5
    links.new(texture.outputs["Color"], shader.inputs["Base Color"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    material["scsp_eye_socket_texture"] = str(texture_path)
    return material


def _uv_layer(
    mesh: Any,
    basis_geometry: Mapping[str, Any],
    forward: Vector,
) -> None:
    fallback = Vector((0.0, 0.0, 1.0))
    if abs(forward.dot(fallback)) > 0.95:
        fallback = Vector((1.0, 0.0, 0.0))
    u_axis = forward.cross(fallback).normalized()
    v_axis = u_axis.cross(forward).normalized()
    center = basis_geometry["front_center_world"]
    projected = []
    for point in basis_geometry["world"]:
        delta = point - center
        projected.append((delta.dot(u_axis), delta.dot(v_axis)))
    radius = max(math.hypot(u, v) for u, v in projected)
    if radius <= 1.0e-12:
        raise EyeSocketError("Eye socket UV projection is degenerate")
    uv = mesh.uv_layers.new(name="EyeSocketUV")
    vertex_uvs = [
        (0.5 + 0.18 * u / radius, 0.5 + 0.18 * v / radius)
        for u, v in projected
    ]
    for loop in mesh.loops:
        uv.data[loop.index].uv = vertex_uvs[loop.vertex_index]

def _retreat_iris(eyes, backward, amount):
    """Rigid object translation only; no edit of the iris mesh or its shape data."""
    from blender_preprocess import mesh_fingerprint
    records=[]
    for side,eye in eyes.items():
        if eye.get("scsp_iris_retreat_applied"):
            raise EyeSocketError("Iris retreat already applied: "+eye.name)
        fingerprint=mesh_fingerprint(eye)
        old=eye.matrix_world.copy()
        updated=old.copy();updated.translation=old.translation+backward*amount
        eye.matrix_world=updated
        bpy.context.view_layer.update()
        if mesh_fingerprint(eye)!=fingerprint:
            raise EyeSocketError("Rigid iris retreat changed mesh data")
        displacement=eye.matrix_world.translation-old.translation
        if (displacement-backward*amount).length>1e-7:
            raise EyeSocketError("Iris retreat is not a rigid back translation")
        eye["scsp_iris_retreat_applied"]=float(amount)
        records.append({"side":side,"mesh":eye.name,"distance_m":amount,
            "translation_world":list(displacement),"mesh_weights_shapes_identical":True,
            "curvature_operation":False})
    return records


def _optional_source_mesh(heads: Sequence[Mapping[str, Any]], source_name: str) -> Any | None:
    target = source_name.casefold()
    for head in heads:
        for obj in head["meshes"]:
            if _source_name(obj).casefold() == target:
                return obj
    return None


def build_eye_sockets(parts, options=None):
    """Trim & rebuild concave polar irises, fit spherical pivots, and build concave sclera sockets."""
    from eye_socket_geometry import make_concave_socket
    from iris_planes import fit_eye_kinematics, rebuild_concave_iris

    settings = dict(DEFAULT_OPTIONS)
    settings.update(options or {})
    clearance = float(settings["contour_clearance"])
    front_clearance = float(settings["front_clearance"])
    socket_clearance = float(settings["socket_clearance"])
    dome_angle_deg = float(settings["dome_angle_deg"])
    outer_flare_clearance = float(settings["outer_flare_clearance"])
    if not (
        0.0 <= clearance < 0.01
        and 0.0 < front_clearance < 0.01
        and 0.0 < socket_clearance < 0.02
        and 30.0 <= dome_angle_deg <= 85.0
        and 0.0 <= outer_flare_clearance < 0.005
    ):
        raise EyeSocketError("Invalid socket/iris clearance parameters")

    plan = _validate_parts(parts)
    heads = _head_parts(plan)
    holes = {}
    eyes = {}
    owners = {}
    eyelashes = {}
    for side in ("L", "R"):
        owners[side], holes[side] = _unique_source_mesh(heads, SOURCE_HOLES[side])
        _, eyes[side] = _unique_source_mesh(heads, EYE_PIECES[side])
        eyelashes[side] = _optional_source_mesh(heads, EYELASH_PIECES[side])

    head = owners["L"]
    arm = head["arm"]
    if owners["R"]["arm"] != arm or not arm or HEAD_BONE not in arm.data.bones:
        raise EyeSocketError("Eye parts need one head armature with Head bone")
    _, face = _unique_source_mesh(heads, "face")
    if not face.data.shape_keys:
        raise EyeSocketError("Source face needs a Basis key")
    face_keys = list(face.data.shape_keys.key_blocks)
    if face_keys[0].name != "Basis":
        raise EyeSocketError("Unexpected face Basis")

    matches, match_report = _match_face_eye_loops(face, holes, eyes)
    forward, direction_report = _derive_forward(holes, arm, face.matrix_world)

    kinematics = {}
    iris_reports = {}
    for side in ("L", "R"):
        kin = fit_eye_kinematics(
            side,
            holes[side],
            face,
            eyelashes[side],
            eyes[side],
            forward,
            front_clearance=front_clearance,
        )
        kinematics[side] = {
            "pivot_world": [float(c) for c in kin["pivot_world"]],
            "eye_forward_world": [float(c) for c in kin["eye_forward_world"]],
            "eye_hole_center_world": [float(c) for c in kin["eye_hole_center_world"]],
            "pivot_depth_m": kin["pivot_depth_m"],
            "rim_radius_m": kin["rim_radius_m"],
        }
        iris_reports[side] = rebuild_concave_iris(side, eyes[side], kin)

    material = _material(_locate_sclera_texture(holes, settings))
    records = []
    for side in ("L", "R"):
        name = SOCKET_NAMES[side]
        if bpy.data.objects.get(name):
            raise EyeSocketError("Socket already exists: " + name)
        iris_rec = iris_reports[side]
        pivot = Vector(iris_rec["pivot_world"])
        eye_forward = Vector(iris_rec["eye_forward_world"]).normalized()
        eye_backward = -eye_forward

        indices = list(matches[side]["loop"]["vertices"])
        basis_world = [face.matrix_world @ face_keys[0].data[i].co + eye_backward * clearance for i in indices]
        basis = make_concave_socket(
            basis_world,
            pivot,
            eye_forward,
            pupil_radius=iris_rec["pupil_radius_m"],
            rim_radius=iris_rec["rim_radius_m"],
            socket_clearance=socket_clearance,
            dome_angle_deg=dome_angle_deg,
            outer_flare_clearance=outer_flare_clearance,
        )
        frame = holes[side].matrix_world.copy()
        inverse = frame.inverted()
        vertices = [inverse @ Vector(v) for v in basis["vertices"]]
        mesh = bpy.data.meshes.new(name + "_Mesh")
        mesh.from_pydata(vertices, [], basis["faces"])
        mesh.update(calc_edges=True)
        if mesh.validate(verbose=False, clean_customdata=False):
            raise EyeSocketError("Concave eye-socket topology invalid")
        for poly in mesh.polygons:
            poly.use_smooth = True
        mesh.materials.append(material)
        _uv_layer(
            mesh,
            {
                "world": [Vector(v) for v in basis["vertices"]],
                "front_center_world": sum(basis_world, Vector()) / len(basis_world),
            },
            eye_forward,
        )
        obj = bpy.data.objects.new(name, mesh)
        for collection in holes[side].users_collection:
            collection.objects.link(obj)
        obj.matrix_world = frame
        obj.shape_key_add(name="Basis", from_mix=False)
        for key in face_keys[1:]:
            rim = [face.matrix_world @ key.data[i].co + eye_backward * clearance for i in indices]
            variant = make_concave_socket(
                rim,
                pivot,
                eye_forward,
                pupil_radius=iris_rec["pupil_radius_m"],
                rim_radius=iris_rec["rim_radius_m"],
                socket_clearance=socket_clearance,
                dome_angle_deg=dome_angle_deg,
                outer_flare_clearance=outer_flare_clearance,
                layout=basis["layout"],
            )
            if variant["faces"] != basis["faces"]:
                raise EyeSocketError("Morph changed concave socket topology")
            block = obj.shape_key_add(name=key.name, from_mix=False)
            block.data.foreach_set("co", [c for v in variant["vertices"] for c in (inverse @ Vector(v))])
            block.slider_min = key.slider_min
            block.slider_max = key.slider_max
            block.value = key.value
            if not all(math.isfinite(c) for point in block.data for c in point.co):
                raise EyeSocketError("Non-finite socket morph")

        group = obj.vertex_groups.new(name=HEAD_BONE)
        group.add(list(range(len(vertices))), 1.0, "REPLACE")
        modifier = obj.modifiers.new(name="Head Armature", type="ARMATURE")
        modifier.object = arm
        obj[SOCKET_TAG] = side
        obj["scsp_eye_socket_source"] = holes[side].name
        obj["scsp_eye_socket_source_fbx"] = str(holes[side].get("scsp_source_fbx", ""))
        obj["scsp_eye_socket_layout"] = json.dumps(basis["layout"], ensure_ascii=False)
        obj["scsp_eye_socket_face_vertices"] = json.dumps(indices)
        obj["scsp_eye_socket_topology"] = "concave egg cup (1x quad rings + central pole fan)"
        head["raw"]["meshes"].append(obj)
        head["raw"]["objects"].append(obj)

        source = holes[side]
        source[SOURCE_REFERENCE_TAG] = True
        source.hide_render = True
        source.hide_set(True)
        source.hide_viewport = True

        records.append({
            "side": side,
            "object": obj.name,
            "vertices": len(vertices),
            "polygons": len(mesh.polygons),
            "quad_faces": sum(len(p.vertices) == 4 for p in mesh.polygons),
            "tri_faces": sum(len(p.vertices) == 3 for p in mesh.polygons),
            "face_contour_vertices": len(indices),
            "face_vertex_indices": indices,
            "shape_keys": [k.name for k in obj.data.shape_keys.key_blocks],
            "topology": basis.get("validation", {}),
        })

    bpy.context.view_layer.update()
    return {
        "schema": 3,
        "operation": "alpha-trimmed concave polar iris and concave polar eye socket",
        "parameters": {
            "contour_clearance": clearance,
            "front_clearance": front_clearance,
            "socket_clearance": socket_clearance,
            "dome_angle_deg": dome_angle_deg,
            "outer_flare_clearance": outer_flare_clearance,
        },
        "orientation": direction_report,
        "kinematics": kinematics,
        "iris_concave_bowls": iris_reports,
        "face_contour_matches": match_report,
        "sockets": records,
        "validation": {
            "source_topology_unchanged": True,
            "socket_count": 2,
            "all_socket_vertices_head_weight_one": True,
            "socket_eye_weight_entries": 0,
            "all_shape_keys_finite": True,
            "same_rim_indices_all_keys": True,
            "iris_polar_rebuilt": True,
            "front_cap_removed": True,
        },
    }
