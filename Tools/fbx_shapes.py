"""Restore FBX blend-shape in-between knots after a Better FBX import.

Run this module inside Blender 5.3, after the Better FBX import stage.  The
official FBX parser is used only to read the source object graph.  Shape keys
are created through Blender's normal shape-key API.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import bpy
import numpy as np
from io_scene_fbx import parse_fbx


SCHEMA = 1
MARKER_PROPERTY = "scsp_fbx_shape_restore"
MAPPING_TOLERANCE = 1.0e-5
ENDPOINT_TOLERANCE = 2.0e-5


class FbxShapeRestoreError(RuntimeError):
    """Raised when source geometry cannot be restored without guessing."""


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _clean_name(value: Any) -> str:
    return _decode(value).split("\x00", 1)[0].strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_path(value: Any) -> Path:
    return Path(str(value)).expanduser().resolve(strict=True)


def _path_key(path: Path) -> str:
    return str(path).replace("/", "\\").casefold()


def _element_array(element: Any, name: bytes) -> list[Any]:
    child = next((item for item in element.elems if item.id == name), None)
    if child is None:
        return []
    if len(child.props) != 1:
        return list(child.props)
    value = child.props[0]
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _float_array(element: Any, name: bytes) -> np.ndarray:
    values = _element_array(element, name)
    try:
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FbxShapeRestoreError(
            f"FBX array {name.decode()} is not numeric on "
            f"{_clean_name(element.props[1]) if len(element.props) > 1 else 'unknown'}"
        ) from exc


def _config_data(config: Any) -> tuple[dict[str, Any], Path | None, str | None]:
    if isinstance(config, Mapping):
        return dict(config), None, None
    path = _canonical_path(config)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FbxShapeRestoreError(f"Cannot read config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise FbxShapeRestoreError("Config root must be a JSON object")
    return data, path, _sha256(path)


def _object_index(root: Any) -> tuple[dict[str, Any], dict[str, list[str]], dict[str, list[str]]]:
    objects = next((item for item in root.elems if item.id == b"Objects"), None)
    connections = next((item for item in root.elems if item.id == b"Connections"), None)
    if objects is None or connections is None:
        raise FbxShapeRestoreError("FBX is missing Objects or Connections")

    by_id: dict[str, Any] = {}
    for element in objects.elems:
        if not element.props:
            continue
        object_id = _decode(element.props[0])
        if object_id in by_id:
            raise FbxShapeRestoreError(f"Duplicate FBX object id: {object_id}")
        by_id[object_id] = element

    parents: dict[str, list[str]] = {}
    children: dict[str, list[str]] = {}
    for connection in connections.elems:
        if len(connection.props) < 3:
            continue
        if _decode(connection.props[0]) not in {"OO", "OP"}:
            continue
        child = _decode(connection.props[1])
        parent = _decode(connection.props[2])
        parents.setdefault(child, []).append(parent)
        children.setdefault(parent, []).append(child)
    return by_id, parents, children


def _kind(element: Any) -> str:
    return _decode(element.id)


def _subkind(element: Any) -> str:
    return _clean_name(element.props[2]) if len(element.props) > 2 else ""


def _typed_ids(
    ids: Sequence[str], by_id: Mapping[str, Any], *, kind: str, subkind: str | None = None
) -> list[str]:
    result = []
    for object_id in dict.fromkeys(ids):
        element = by_id.get(object_id)
        if element is None or _kind(element) != kind:
            continue
        if subkind is not None and _subkind(element) != subkind:
            continue
        result.append(object_id)
    return result


def _parse_source(path: Path) -> dict[str, Any]:
    source_hash = _sha256(path)
    try:
        root, version = parse_fbx.parse(str(path))
    except Exception as exc:
        raise FbxShapeRestoreError(f"Cannot parse FBX {path}: {exc}") from exc

    by_id, parents, children = _object_index(root)
    shape_ids = [
        object_id
        for object_id, element in by_id.items()
        if _kind(element) == "Geometry" and _subkind(element) == "Shape"
    ]
    shape_id_set = set(shape_ids)
    channels = [
        (object_id, element)
        for object_id, element in by_id.items()
        if _kind(element) == "Deformer"
        and _subkind(element) == "BlendShapeChannel"
    ]
    geometry_ids = [
        object_id
        for object_id, element in by_id.items()
        if _kind(element) == "Geometry" and _subkind(element) == "Mesh"
    ]
    model_ids = [
        object_id
        for object_id, element in by_id.items()
        if _kind(element) == "Model" and _subkind(element) == "Mesh"
    ]

    multi_channels: list[dict[str, Any]] = []
    single_count = 0
    for channel_id, channel in channels:
        channel_name = _clean_name(channel.props[1])
        linked_shapes = [
            object_id
            for object_id in children.get(channel_id, [])
            if object_id in shape_id_set
        ]
        if len(linked_shapes) < 2:
            single_count += 1
            continue

        weights = _float_array(channel, b"FullWeights")
        if len(weights) != len(linked_shapes):
            raise FbxShapeRestoreError(
                f"{path.name}: {channel_name} has {len(linked_shapes)} shapes "
                f"but {len(weights)} FullWeights"
            )
        if not np.all(np.isfinite(weights)):
            raise FbxShapeRestoreError(f"{path.name}: {channel_name} has non-finite weights")
        if np.any(np.diff(weights) <= 0):
            raise FbxShapeRestoreError(
                f"{path.name}: {channel_name} weights are not strictly increasing: "
                f"{weights.tolist()}"
            )
        if weights[0] < 0 or abs(float(weights[-1]) - 100.0) > 1.0e-5:
            raise FbxShapeRestoreError(
                f"{path.name}: {channel_name} must span 0..100 and end at 100; "
                f"got {weights.tolist()}"
            )
        if weights[0] <= 0:
            raise FbxShapeRestoreError(
                f"{path.name}: {channel_name} first knot must be above 0; "
                f"got {weights.tolist()}"
            )

        blend_parents = _typed_ids(
            parents.get(channel_id, []), by_id, kind="Deformer", subkind="BlendShape"
        )
        if len(blend_parents) != 1:
            raise FbxShapeRestoreError(
                f"{path.name}: {channel_name} has {len(blend_parents)} BlendShape parents"
            )
        base_geometries = _typed_ids(
            parents.get(blend_parents[0], []), by_id, kind="Geometry", subkind="Mesh"
        )
        if len(base_geometries) != 1:
            raise FbxShapeRestoreError(
                f"{path.name}: {channel_name} has {len(base_geometries)} base mesh geometries"
            )
        base_models = _typed_ids(
            parents.get(base_geometries[0], []), by_id, kind="Model"
        )
        if len(base_models) != 1:
            raise FbxShapeRestoreError(
                f"{path.name}: {channel_name} has {len(base_models)} mesh models"
            )

        shape_records = []
        for weight, shape_id in zip(weights.tolist(), linked_shapes):
            shape = by_id[shape_id]
            indexes = _float_array(shape, b"Indexes")
            vertices = _float_array(shape, b"Vertices")
            if indexes.size == 0 or vertices.size == 0:
                raise FbxShapeRestoreError(
                    f"{path.name}: {channel_name} shape {_clean_name(shape.props[1])} "
                    "has no Indexes or Vertices"
                )
            rounded = np.rint(indexes)
            if np.max(np.abs(indexes - rounded)) > 1.0e-6:
                raise FbxShapeRestoreError(
                    f"{path.name}: {channel_name} has non-integer shape indexes"
                )
            if vertices.size != indexes.size * 3:
                raise FbxShapeRestoreError(
                    f"{path.name}: {channel_name} shape {_clean_name(shape.props[1])} "
                    f"has {indexes.size} indexes and {vertices.size} vertex values"
                )
            shape_records.append(
                {
                    "name": _clean_name(shape.props[1]),
                    "geometry_id": shape_id,
                    "weight_percent": float(weight),
                    "indexes": rounded.astype(np.int64),
                    "vertices": vertices.reshape((-1, 3)),
                }
            )

        multi_channels.append(
            {
                "name": channel_name,
                "channel_id": channel_id,
                "weights_percent": weights.astype(np.float64),
                "shape_records": shape_records,
                "base_geometry_id": base_geometries[0],
                "base_model_id": base_models[0],
                "source_model": _clean_name(by_id[base_models[0]].props[1]),
            }
        )

    if _sha256(path) != source_hash:
        raise FbxShapeRestoreError(f"Source FBX changed while being read: {path}")

    return {
        "path": path,
        "sha256": source_hash,
        "version": int(version),
        "shape_count": len(shape_ids),
        "channel_count": len(channels),
        "single_shape_channel_count": single_count,
        "mesh_geometry_count": len(geometry_ids),
        "mesh_model_count": len(model_ids),
        "multi_channels": multi_channels,
        "by_id": by_id,
    }


def _mesh_records(parts: Sequence[Mapping[str, Any]]) -> tuple[dict[tuple[str, str], list[Any]], dict[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[Any]] = {}
    path_counts: dict[str, int] = {}
    report_parts: list[dict[str, Any]] = []
    seen_meshes: set[Any] = set()

    for part in parts:
        if not isinstance(part, Mapping):
            raise FbxShapeRestoreError("Each parts entry must be a dictionary")
        kind = str(part.get("kind", ""))
        arm = part.get("arm")
        meshes = list(part.get("meshes") or [])
        objects = list(part.get("objects") or [])
        report_part = {
            "kind": kind,
            "armature": arm.name if arm is not None else None,
            "objects": len(objects),
            "meshes": len(meshes),
        }
        report_parts.append(report_part)

        for mesh in meshes:
            if getattr(mesh, "type", None) != "MESH":
                raise FbxShapeRestoreError(f"parts mesh is not a Blender mesh: {mesh!r}")
            if mesh in seen_meshes:
                raise FbxShapeRestoreError(f"Mesh occurs in more than one part: {mesh.name}")
            seen_meshes.add(mesh)
            source_fbx = mesh.get("scsp_source_fbx")
            source_object = mesh.get("scsp_source_object")
            if not source_fbx or not source_object:
                report_part.setdefault("ignored_meshes", []).append(mesh.name)
                continue
            path = _canonical_path(source_fbx)
            path_key = _path_key(path)
            identity = _clean_name(source_object)
            grouped.setdefault((path_key, identity), []).append(mesh)
            path_counts[path_key] = path_counts.get(path_key, 0) + 1

    return grouped, path_counts, report_parts


def _marker(mesh: Any) -> dict[str, Any]:
    raw = mesh.get(MARKER_PROPERTY)
    if not raw:
        return {"schema": SCHEMA, "channels": {}}
    if not isinstance(raw, str):
        raise FbxShapeRestoreError(
            f"{mesh.name}: {MARKER_PROPERTY} must be a JSON string"
        )
    try:
        marker = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FbxShapeRestoreError(
            f"{mesh.name}: invalid {MARKER_PROPERTY} JSON"
        ) from exc
    if marker.get("schema") != SCHEMA or not isinstance(marker.get("channels"), dict):
        raise FbxShapeRestoreError(
            f"{mesh.name}: unsupported {MARKER_PROPERTY} schema"
        )
    return marker


def _endpoint_key(mesh: Any, channel_name: str, final_shape_name: str) -> Any:
    keys = mesh.data.shape_keys
    if keys is None or len(keys.key_blocks) < 2:
        raise FbxShapeRestoreError(
            f"{mesh.name}: source {channel_name} targets a mesh without imported shape keys"
        )
    blocks = list(keys.key_blocks[1:])
    exact = [block for block in blocks if _clean_name(block.name) == channel_name]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise FbxShapeRestoreError(
            f"{mesh.name}: multiple imported keys match {channel_name}"
        )
    by_source_endpoint = [
        block for block in blocks if _clean_name(block.name) == final_shape_name
    ]
    if len(by_source_endpoint) == 1:
        return by_source_endpoint[0]
    raise FbxShapeRestoreError(
        f"{mesh.name}: cannot identify imported endpoint key for {channel_name}"
    )


def _intermediate_key_name(channel_name: str, weight_percent: float) -> str:
    weight = f"{weight_percent:.9g}"
    return f"{channel_name}@{weight}pct"


def _control_property_name(channel_name: str) -> str:
    safe = "".join(char if char.isalnum() or char == "_" else "_" for char in channel_name)
    return f"scsp_fbx_shape_{safe}_percent"


def _driver_path(key_block: Any) -> str:
    return key_block.path_from_id("value")


def _drivers_by_path(key_blocks: Any) -> dict[str, Any]:
    animation_data = key_blocks.animation_data
    if animation_data is None:
        return {}
    return {
        driver.data_path: driver
        for driver in animation_data.drivers
    }


def _fit_affine(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, int, float]:
    if source.ndim != 2 or target.ndim != 2 or source.shape != target.shape:
        raise FbxShapeRestoreError("Source and imported basis vertex arrays differ in shape")
    if source.shape[1] != 3 or source.shape[0] < 4:
        raise FbxShapeRestoreError("Too few vertices for an affine coordinate fit")
    design = np.column_stack((source, np.ones(source.shape[0], dtype=np.float64)))
    rank = int(np.linalg.matrix_rank(design, tol=1.0e-10))
    if rank != 4:
        raise FbxShapeRestoreError(
            f"Ambiguous basis coordinate fit: design matrix rank is {rank}, expected 4"
        )
    matrix, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    errors = np.linalg.norm(design @ matrix - target, axis=1)
    maximum = float(np.max(errors, initial=0.0))
    if maximum > MAPPING_TOLERANCE:
        raise FbxShapeRestoreError(
            f"Basis coordinate fit is ambiguous or inaccurate: max error {maximum:.9g} m"
        )
    return matrix, rank, maximum


def _endpoint_errors(
    final_shape: Mapping[str, Any],
    basis: np.ndarray,
    endpoint: np.ndarray,
    matrix: np.ndarray,
) -> dict[str, Any]:
    indexes = final_shape["indexes"]
    vertices = final_shape["vertices"]
    if np.any(indexes < 0) or np.any(indexes >= basis.shape[0]):
        raise FbxShapeRestoreError("Source endpoint shape indexes exceed imported vertex count")
    if np.unique(indexes).size != indexes.size:
        raise FbxShapeRestoreError("Source endpoint shape contains duplicate vertex indexes")

    transformed_delta = basis[indexes] + vertices @ matrix[:3, :]
    transformed_absolute = np.column_stack(
        (vertices, np.ones(indexes.size, dtype=np.float64))
    ) @ matrix
    delta_errors = np.linalg.norm(transformed_delta - endpoint[indexes], axis=1)
    absolute_errors = np.linalg.norm(transformed_absolute - endpoint[indexes], axis=1)
    delta_max = float(np.max(delta_errors, initial=0.0))
    absolute_max = float(np.max(absolute_errors, initial=0.0))

    untouched = np.ones(basis.shape[0], dtype=bool)
    untouched[indexes] = False
    if np.any(untouched):
        untouched_max = float(np.max(np.linalg.norm(endpoint[untouched] - basis[untouched], axis=1)))
    else:
        untouched_max = 0.0
    if untouched_max > ENDPOINT_TOLERANCE:
        raise FbxShapeRestoreError(
            "Imported endpoint changes source-untouched vertices; mesh identity mapping is ambiguous: "
            f"max error {untouched_max:.9g} m"
        )

    if delta_max <= ENDPOINT_TOLERANCE and absolute_max <= ENDPOINT_TOLERANCE:
        if abs(delta_max - absolute_max) <= 1.0e-7:
            mode = "equivalent"
            endpoint_max = min(delta_max, absolute_max)
        else:
            mode = "delta"
            endpoint_max = delta_max
    elif delta_max <= ENDPOINT_TOLERANCE:
        mode = "delta"
        endpoint_max = delta_max
    elif absolute_max <= ENDPOINT_TOLERANCE:
        mode = "absolute"
        endpoint_max = absolute_max
    else:
        raise FbxShapeRestoreError(
            "Imported 100% endpoint does not agree with either FBX shape coordinate convention: "
            f"delta max {delta_max:.9g} m, absolute max {absolute_max:.9g} m"
        )

    return {
        "mode": mode,
        "delta_max_error_m": delta_max,
        "absolute_max_error_m": absolute_max,
        "endpoint_max_error_m": endpoint_max,
        "untouched_max_error_m": untouched_max,
    }


def _hat_expression(index: int, weights: Sequence[float]) -> str:
    current = float(weights[index]) / 100.0
    if index == 0:
        up = f"min(max(p / {current:.12g}, 0.0), 1.0)"
    else:
        previous = float(weights[index - 1]) / 100.0
        up = (
            f"min(max((p - {previous:.12g}) / "
            f"{current - previous:.12g}, 0.0), 1.0)"
        )
    if index == len(weights) - 1:
        return up
    following = float(weights[index + 1]) / 100.0
    down = (
        f"min(max(({following:.12g} - p) / "
        f"{following - current:.12g}, 0.0), 1.0)"
    )
    return f"min({up}, {down})"


def _plan_channel(
    *,
    kind: str,
    mesh: Any,
    channel: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    name = channel["name"]
    shape_records = channel["shape_records"]
    final_shape = shape_records[-1]
    keys = mesh.data.shape_keys
    if keys is None or not keys.key_blocks:
        raise FbxShapeRestoreError(f"{mesh.name}: no imported shape-key basis")

    basis_key = keys.key_blocks[0]
    mesh_vertices = np.asarray([tuple(vertex.co) for vertex in mesh.data.vertices], dtype=np.float64)
    basis = np.asarray([tuple(point.co) for point in basis_key.data], dtype=np.float64)
    if mesh_vertices.shape != basis.shape:
        raise FbxShapeRestoreError(f"{mesh.name}: basis key and mesh vertex counts differ")
    basis_difference = float(
        np.max(np.linalg.norm(mesh_vertices - basis, axis=1), initial=0.0)
    )
    if basis_difference > MAPPING_TOLERANCE:
        raise FbxShapeRestoreError(
            f"{mesh.name}: shape-key basis differs from mesh vertices by {basis_difference:.9g} m"
        )

    base_geometry = source["by_id"][channel["base_geometry_id"]]
    raw_base_flat = _float_array(base_geometry, b"Vertices")
    if raw_base_flat.size % 3:
        raise FbxShapeRestoreError(f"{source['path'].name}: malformed base mesh vertices")
    raw_base = raw_base_flat.reshape((-1, 3))
    if raw_base.shape != basis.shape:
        raise FbxShapeRestoreError(
            f"{mesh.name}: imported vertex count {basis.shape[0]} does not match "
            f"source {raw_base.shape[0]}"
        )

    matrix, rank, basis_max = _fit_affine(raw_base, basis)
    endpoint_key = _endpoint_key(mesh, name, final_shape["name"])
    endpoint = np.asarray([tuple(point.co) for point in endpoint_key.data], dtype=np.float64)
    if endpoint.shape != basis.shape:
        raise FbxShapeRestoreError(f"{mesh.name}: endpoint key has the wrong vertex count")
    endpoint_checks = _endpoint_errors(final_shape, basis, endpoint, matrix)
    mode = endpoint_checks["mode"]

    marker = _marker(mesh)
    existing_channel = marker["channels"].get(name)
    control_name = _control_property_name(name)
    if control_name in mesh and existing_channel is None:
        raise FbxShapeRestoreError(
            f"{mesh.name}: control property {control_name} already exists outside a prior restore"
        )

    driver_map = _drivers_by_path(keys)
    endpoint_driver_path = _driver_path(endpoint_key)
    known_driver_keys = set()
    if existing_channel:
        known_driver_keys.update(existing_channel.get("driver_keys", []))
    if endpoint_driver_path in driver_map and endpoint_key.name not in known_driver_keys:
        raise FbxShapeRestoreError(
            f"{mesh.name}: endpoint {endpoint_key.name} already has a foreign driver"
        )

    intermediates = []
    for shape in shape_records[:-1]:
        key_name = _intermediate_key_name(name, shape["weight_percent"])
        existing_names = {_clean_name(block.name) for block in keys.key_blocks}
        if key_name in existing_names:
            if (
                existing_channel is None
                or key_name not in existing_channel.get("intermediate_keys", {})
            ):
                raise FbxShapeRestoreError(
                    f"{mesh.name}: intermediate key collision on {key_name}"
                )
        indexes = shape["indexes"]
        vertices = shape["vertices"]
        if np.any(indexes < 0) or np.any(indexes >= basis.shape[0]):
            raise FbxShapeRestoreError(
                f"{mesh.name}: {shape['name']} indexes exceed imported vertex count"
            )
        if np.unique(indexes).size != indexes.size:
            raise FbxShapeRestoreError(
                f"{mesh.name}: {shape['name']} contains duplicate vertex indexes"
            )
        if mode == "absolute":
            affected = np.column_stack(
                (vertices, np.ones(indexes.size, dtype=np.float64))
            ) @ matrix
        else:
            affected = basis[indexes] + vertices @ matrix[:3, :]
        intermediates.append(
            {
                "key_name": key_name,
                "weight_percent": shape["weight_percent"],
                "source_shape": shape["name"],
                "indexes": indexes,
                "coordinates": affected,
            }
        )

    weights = channel["weights_percent"].astype(np.float64)
    driver_plan = [
        {
            "key_name": (
                intermediates[index]["key_name"]
                if index < len(intermediates)
                else endpoint_key.name
            ),
            "expression": _hat_expression(index, weights),
        }
        for index in range(len(weights))
    ]
    for entry in driver_plan:
        block = next(
            (candidate for candidate in keys.key_blocks if candidate.name == entry["key_name"]),
            None,
        )
        if block is not None:
            path = _driver_path(block)
            if path in driver_map and entry["key_name"] not in known_driver_keys:
                raise FbxShapeRestoreError(
                    f"{mesh.name}: key {entry['key_name']} already has a foreign driver"
                )

    if existing_channel:
        initial_control = float(mesh.get(control_name, 0.0))
    else:
        last_previous = float(weights[-2] / 100.0)
        endpoint_value = min(max(float(endpoint_key.value), 0.0), 1.0)
        initial_control = (
            0.0
            if endpoint_value <= 1.0e-12
            else last_previous + endpoint_value * (1.0 - last_previous)
        )

    return {
        "kind": kind,
        "mesh": mesh,
        "source_path": source["path"],
        "source_sha256": source["sha256"],
        "source_model": channel["source_model"],
        "channel": name,
        "weights": [float(value) for value in weights],
        "shape_names": [shape["name"] for shape in shape_records],
        "index_counts": [int(shape["indexes"].size) for shape in shape_records],
        "endpoint_key": endpoint_key,
        "endpoint_before": endpoint.copy(),
        "basis_before": basis.copy(),
        "basis_key": basis_key,
        "intermediates": intermediates,
        "driver_plan": driver_plan,
        "control_name": control_name,
        "initial_control": initial_control,
        "matrix": matrix,
        "rank": rank,
        "basis_max_error_m": basis_max,
        "basis_difference_m": basis_difference,
        "endpoint_checks": endpoint_checks,
        "existing": bool(existing_channel),
        "marker": marker,
    }


def _set_driver(key_block: Any, mesh: Any, control_name: str, expression: str) -> None:
    animation_data = key_block.id_data.animation_data
    path = _driver_path(key_block)
    curve = next(
        (candidate for candidate in animation_data.drivers if candidate.data_path == path),
        None,
    ) if animation_data is not None else None
    if curve is None:
        curve = key_block.driver_add("value")
    driver = curve.driver
    driver.type = "SCRIPTED"
    while driver.variables:
        driver.variables.remove(driver.variables[0])
    variable = driver.variables.new()
    variable.name = "p"
    variable.type = "SINGLE_PROP"
    target = variable.targets[0]
    target.id_type = "OBJECT"
    target.id = mesh
    target.data_path = f'["{control_name}"]'
    driver.use_self = False
    driver.expression = expression


def _set_control(mesh: Any, name: str, value: float, channel: str) -> None:
    mesh[name] = float(value)
    try:
        ui = mesh.id_properties_ui(name)
        ui.update(
            min=0.0,
            max=1.0,
            soft_min=0.0,
            soft_max=1.0,
            subtype="PERCENTAGE",
            description=(
                f"SCSP FBX shape control for {channel}; stored as 0..1, "
                "shown as a percentage"
            ),
        )
    except (AttributeError, TypeError, ValueError):
        pass


def restore_fbx_shapes(config: Any, parts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Restore intermediate FBX blend-shape knots and return a JSON-safe report.

    ``config`` may be the parsed config mapping or its JSON path.  ``parts`` is
    the list produced by ``blender_preprocess.import_parts``.  Source ambiguity,
    missing identity metadata, and endpoint disagreement are fatal.
    """

    if bpy.app.version[:2] != (5, 3):
        raise FbxShapeRestoreError(
            f"restore_fbx_shapes requires Blender 5.3; running {bpy.app.version_string}"
        )
    config_data, config_path, config_hash = _config_data(config)
    grouped_meshes, path_counts, report_parts = _mesh_records(parts)
    if not path_counts:
        raise FbxShapeRestoreError(
            "No meshes with scsp_source_fbx and scsp_source_object were supplied"
        )

    sources: dict[str, dict[str, Any]] = {}
    for raw_path in config_data.get("parts", []):
        try:
            path = _canonical_path(raw_path["path"])
        except (KeyError, OSError) as exc:
            raise FbxShapeRestoreError(f"Invalid config source: {raw_path!r}") from exc
        path_key = _path_key(path)
        if path_key not in path_counts or path_key in sources:
            continue
        sources[path_key] = _parse_source(path)

    missing_paths = sorted(set(path_counts) - set(sources))
    if missing_paths:
        raise FbxShapeRestoreError(
            f"Supplied mesh sources are absent from config: {', '.join(missing_paths)}"
        )

    plans: list[dict[str, Any]] = []
    source_reports: list[dict[str, Any]] = []
    for path_key, source in sources.items():
        source_reports.append(
            {
                "path": str(source["path"]),
                "sha256": source["sha256"],
                "fbx_version": source["version"],
                "shape_geometry_count": source["shape_count"],
                "blend_shape_channel_count": source["channel_count"],
                "single_shape_channel_count": source["single_shape_channel_count"],
                "multi_shape_channel_count": len(source["multi_channels"]),
            }
        )
        exact_meshes = {
            identity: meshes
            for (group_path, identity), meshes in grouped_meshes.items()
            if group_path == path_key
        }
        folded_meshes: dict[str, list[Any]] = {}
        for identity, meshes in exact_meshes.items():
            folded_meshes.setdefault(identity.casefold(), []).extend(meshes)

        for channel in source["multi_channels"]:
            source_model = channel["source_model"]
            matches = exact_meshes.get(source_model, [])
            if not matches:
                matches = folded_meshes.get(source_model.casefold(), [])
            if len(matches) != 1:
                raise FbxShapeRestoreError(
                    f"{source['path'].name}: channel {channel['name']} maps to "
                    f"{len(matches)} imported meshes for source object {source_model}"
                )
            mesh = matches[0]
            kinds = {
                part.get("kind", "")
                for part in parts
                if mesh in (part.get("meshes") or [])
            }
            kind = sorted(kinds)[0] if len(kinds) == 1 else "|".join(sorted(kinds))
            plans.append(
                _plan_channel(
                    kind=kind,
                    mesh=mesh,
                    channel=channel,
                    source=source,
                )
            )

    if not plans:
        report = {
            "schema": SCHEMA,
            "operation": "restore_fbx_shapes",
            "blender": bpy.app.version_string,
            "config": {
                "path": str(config_path) if config_path else None,
                "sha256": config_hash,
            },
            "summary": {
                "parts": len(parts),
                "source_fbx_files": len(sources),
                "shape_geometry_count": sum(
                    source["shape_count"] for source in sources.values()
                ),
                "blend_shape_channel_count": sum(
                    source["channel_count"] for source in sources.values()
                ),
                "multi_shape_channels": 0,
                "restored_channels": 0,
                "added_intermediate_keys": 0,
                "updated_existing_channels": 0,
                "source_hashes_unchanged": True,
                "endpoint_geometry_preserved": True,
            },
            "parts": report_parts,
            "sources": source_reports,
            "channels": [],
            "warnings": ["No multi-shape FBX blend-shape channels were found; no changes applied."],
        }
        try:
            json.dumps(report, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise FbxShapeRestoreError(
                f"No-op restore report is not JSON serializable: {exc}"
            ) from exc
        return report

    added_keys: list[tuple[Any, Any]] = []
    try:
        for plan in plans:
            mesh = plan["mesh"]
            keys = mesh.data.shape_keys
            existing_names = {block.name for block in keys.key_blocks}
            for intermediate in plan["intermediates"]:
                key_name = intermediate["key_name"]
                if key_name in existing_names:
                    key_block = keys.key_blocks[key_name]
                else:
                    key_block = mesh.shape_key_add(name=key_name, from_mix=False)
                    added_keys.append((mesh, key_block))
                    existing_names.add(key_block.name)
                for vertex_index, coordinate in zip(
                    intermediate["indexes"], intermediate["coordinates"]
                ):
                    key_block.data[int(vertex_index)].co = coordinate
                key_block.relative_key = plan["basis_key"]
                untouched = np.ones(len(key_block.data), dtype=bool)
                untouched[intermediate["indexes"]] = False
                untouched_indexes = np.flatnonzero(untouched)
                if untouched_indexes.size:
                    unindexed_error = float(
                        np.max(
                            np.linalg.norm(
                                np.asarray(
                                    [
                                        tuple(key_block.data[int(index)].co)
                                        for index in untouched_indexes
                                    ],
                                    dtype=np.float64,
                                )
                                - np.asarray(
                                    [
                                        tuple(plan["basis_key"].data[int(index)].co)
                                        for index in untouched_indexes
                                    ],
                                    dtype=np.float64,
                                ),
                                axis=1,
                            ),
                            initial=0.0,
                        )
                    )
                else:
                    unindexed_error = 0.0
                if unindexed_error > MAPPING_TOLERANCE:
                    raise FbxShapeRestoreError(
                        f"{mesh.name}: {key_name} copied a non-basis current mix into "
                        f"unindexed vertices: max error {unindexed_error:.9g} m"
                    )
                intermediate["unindexed_basis_max_error_m"] = unindexed_error
                key_block.slider_min = 0.0
                key_block.slider_max = 1.0

            _set_control(
                mesh,
                plan["control_name"],
                plan["initial_control"],
                plan["channel"],
            )
            for driver in plan["driver_plan"]:
                key_block = (
                    keys.key_blocks[driver["key_name"]]
                    if driver["key_name"] in {block.name for block in keys.key_blocks}
                    else plan["endpoint_key"]
                )
                _set_driver(
                    key_block,
                    mesh,
                    plan["control_name"],
                    driver["expression"],
                )
                key_block.slider_min = 0.0
                key_block.slider_max = 1.0

        bpy.context.view_layer.update()
    except Exception:
        for mesh, key_block in reversed(added_keys):
            try:
                mesh.shape_key_remove(key_block)
            except (ReferenceError, RuntimeError):
                pass
        raise

    endpoint_results = []
    marker_updates: dict[Any, dict[str, Any]] = {}
    for plan in plans:
        mesh = plan["mesh"]
        endpoint_after = np.asarray(
            [tuple(point.co) for point in plan["endpoint_key"].data],
            dtype=np.float64,
        )
        basis_after = np.asarray(
            [tuple(point.co) for point in plan["basis_key"].data],
            dtype=np.float64,
        )
        endpoint_difference = float(
            np.max(
                np.linalg.norm(endpoint_after - plan["endpoint_before"], axis=1),
                initial=0.0,
            )
        )
        basis_change = float(
            np.max(
                np.linalg.norm(basis_after - plan["basis_before"], axis=1),
                initial=0.0,
            )
        )
        if endpoint_difference > 0.0 or basis_change > 0.0:
            raise FbxShapeRestoreError(
                f"{mesh.name}: endpoint or basis geometry changed during restore"
            )
        endpoint_results.append(
            {
                "kind": plan["kind"],
                "source_fbx": str(plan["source_path"]),
                "source_sha256": plan["source_sha256"],
                "source_object": plan["source_model"],
                "mesh": mesh.name,
                "channel": plan["channel"],
                "weights_percent": plan["weights"],
                "source_shapes": plan["shape_names"],
                "changed_index_counts": plan["index_counts"],
                "endpoint_key": plan["endpoint_key"].name,
                "intermediate_keys": [
                    {
                        "name": intermediate["key_name"],
                        "weight_percent": intermediate["weight_percent"],
                        "source_shape": intermediate["source_shape"],
                        "from_mix": False,
                        "relative_key": plan["basis_key"].name,
                        "unindexed_basis_max_error_m": intermediate[
                            "unindexed_basis_max_error_m"
                        ],
                    }
                    for intermediate in plan["intermediates"]
                ],
                "control": {
                    "property": plan["control_name"],
                    "range": [0.0, 1.0],
                    "display": "percentage",
                    "location": "Mesh Object Properties > Custom Properties",
                },
                "drivers": [
                    {
                        "key": driver["key_name"],
                        "expression": driver["expression"],
                        "reads_object_property": plan["control_name"],
                        "uses_self": False,
                    }
                    for driver in plan["driver_plan"]
                ],
                "mapping": {
                    "method": "affine least squares, source local to imported key local",
                    "interpretation": plan["endpoint_checks"]["mode"],
                    "design_rank": plan["rank"],
                    "basis_max_error_m": plan["basis_max_error_m"],
                    "basis_key_vs_mesh_max_error_m": plan["basis_difference_m"],
                    "delta_endpoint_max_error_m": plan["endpoint_checks"][
                        "delta_max_error_m"
                    ],
                    "absolute_endpoint_max_error_m": plan["endpoint_checks"][
                        "absolute_max_error_m"
                    ],
                    "endpoint_max_error_m": plan["endpoint_checks"][
                        "endpoint_max_error_m"
                    ],
                    "untouched_endpoint_max_error_m": plan["endpoint_checks"][
                        "untouched_max_error_m"
                    ],
                    "affine_row_major": plan["matrix"].tolist(),
                },
                "endpoint_geometry_change_m": endpoint_difference,
                "basis_geometry_change_m": basis_change,
            }
        )

        marker = plan["marker"]
        marker["channels"][plan["channel"]] = {
            "control_property": plan["control_name"],
            "weights_percent": plan["weights"],
            "intermediate_keys": {
                intermediate["key_name"]: intermediate["weight_percent"]
                for intermediate in plan["intermediates"]
            },
            "endpoint_key": plan["endpoint_key"].name,
            "driver_keys": [plan["endpoint_key"].name]
            + [intermediate["key_name"] for intermediate in plan["intermediates"]],
            "source_fbx": str(plan["source_path"]),
            "source_object": plan["source_model"],
        }
        marker_updates[mesh] = marker

    for mesh, marker in marker_updates.items():
        mesh[MARKER_PROPERTY] = json.dumps(marker, ensure_ascii=False, separators=(",", ":"))

    for source in sources.values():
        if _sha256(source["path"]) != source["sha256"]:
            raise FbxShapeRestoreError(
                f"Source FBX changed during restore: {source['path']}"
            )

    report = {
        "schema": SCHEMA,
        "operation": "restore_fbx_shapes",
        "blender": bpy.app.version_string,
        "config": {
            "path": str(config_path) if config_path else None,
            "sha256": config_hash,
        },
        "summary": {
            "parts": len(parts),
            "source_fbx_files": len(sources),
            "shape_geometry_count": sum(
                source["shape_count"] for source in sources.values()
            ),
            "blend_shape_channel_count": sum(
                source["channel_count"] for source in sources.values()
            ),
            "multi_shape_channels": sum(
                len(source["multi_channels"]) for source in sources.values()
            ),
            "restored_channels": len(plans),
            "added_intermediate_keys": len(added_keys),
            "updated_existing_channels": sum(plan["existing"] for plan in plans),
            "source_hashes_unchanged": True,
            "endpoint_geometry_preserved": all(
                result["endpoint_geometry_change_m"] == 0.0
                for result in endpoint_results
            ),
        },
        "parts": report_parts,
        "sources": source_reports,
        "channels": endpoint_results,
        "warnings": [
            "Artist controls are mesh object custom properties; their 0..1 values "
            "display as percentages.",
            "Intermediate and endpoint shape-key sliders are driver outputs. "
            "Use the documented object property as the artist control.",
        ],
    }
    try:
        json.dumps(report, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FbxShapeRestoreError(f"Restore report is not JSON serializable: {exc}") from exc
    return report
