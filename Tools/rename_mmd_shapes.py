"""Rename SCSP Blender shape keys to the PMX names produced by Unity.

The API mutates shape keys, combined keys, drivers, and F-curve paths in the
current Blender session. It intentionally never saves the .blend file.
"""

from __future__ import annotations

import hashlib
import json
import re
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import bpy


SCHEMA = 1
# Direction is current shape-key name -> original source identity. Values with
# "@combined:" are reserved combination identities. Never reverse this marker.
KEY_SOURCE_PROPERTY = "scsp_mmd_shape_source_by_current"
RESTORE_PROPERTY = "scsp_fbx_shape_restore"
RESOURCE = Path(__file__).resolve().parents[1] / "Resources" / "unity_morph_names.json"

_KEY_PATH = re.compile(r'^key_blocks\[("(?:[^"\\]|\\.)*")\]\.value$')
_resources: dict[str, Any] | None = None


def _load_resources() -> dict[str, Any]:
    global _resources
    if _resources is None:
        data = json.loads(RESOURCE.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise ValueError(f"Unsupported mapping schema in {RESOURCE}")
        _resources = data
    return _resources


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(name: str, resources: dict[str, Any]) -> bool:
    if not name or "\0" in name:
        return False
    try:
        return len(name.encode(resources["name_encoding"])) <= resources["name_max_bytes"]
    except UnicodeEncodeError:
        return False


def _marker(keys: Any) -> dict[str, str]:
    raw = keys.get(KEY_SOURCE_PROPERTY)
    if not raw:
        return {}
    value = json.loads(raw)
    return value if isinstance(value, dict) else {}


def _restore_endpoints(mesh: Any) -> dict[str, str]:
    raw = mesh.get(RESTORE_PROPERTY)
    if not raw:
        return {}
    try:
        channels = json.loads(raw).get("channels", {})
    except (TypeError, json.JSONDecodeError):
        return {}
    return {
        channel.get("endpoint_key"): source
        for source, channel in channels.items()
        if isinstance(channel, dict) and channel.get("endpoint_key")
    }


def _source_for(mesh: Any, block: Any, resources: dict[str, Any]) -> tuple[str | None, str]:
    marked = _marker(mesh.data.shape_keys).get(block.name)
    if marked and marked.startswith(("@combined:", "@generated:")):
        return None, "existing derived morph"
    if marked:
        return marked, "key marker"
    restored = _restore_endpoints(mesh).get(block.name)
    if restored:
        return restored, "restore source identifier"
    names = resources["source_names"]
    if block.name in names:
        return block.name, "exact source identifier"
    tail = block.name.rsplit(".", 1)[-1]
    if tail in names:
        return tail, "last dot component"
    return block.name, "unknown preserved"


def _panel(source: str, resources: dict[str, Any]) -> int:
    config = resources["panels"]
    if source.startswith("eyebrow_"):
        return config["eyebrow_prefix"]
    if source.startswith("eye_") or source.startswith("eyelash"):
        return config["eye_and_eyelash_prefixes"]
    if source in config["mouth_sources"] or source.startswith(config["mouth_prefix"]) or source in config["mouth_extra_sources"]:
        return config["mouth_sources_panel"]
    return config["default"]


def _path_name(data_path: str) -> str | None:
    match = _KEY_PATH.match(data_path)
    return json.loads(match.group(1)) if match else None


def _key_path(name: str) -> str:
    return f"key_blocks[{json.dumps(name, ensure_ascii=False)}].value"


def _basis_block(keys: Any) -> Any:
    basis = keys.key_blocks.get("Basis")
    if basis is None:
        raise ValueError("Shape keys must contain a key named 'Basis'")
    return basis


def _key_pointer(block: Any) -> int | None:
    return int(block.as_pointer()) if block is not None else None


def _changed_vertex_count(result: array, basis: array) -> int:
    return sum(
        any(result[index + axis] != basis[index + axis] for axis in range(3))
        for index in range(0, len(result), 3)
    )


def _metadata_source_paths(metadata: Any) -> list[str]:
    if isinstance(metadata, str):
        text = metadata.strip()
        if text.startswith(("[", "{")):
            try:
                metadata = json.loads(text)
            except json.JSONDecodeError:
                return []
    result: list[str] = []

    def visit(value: Any, key_hint: str | None = None) -> None:
        if isinstance(value, str):
            if value.strip() and key_hint in {"record", "source_fbx", "path", "fbx_path"}:
                result.append(value.strip())
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item, key_hint)
            return
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            if key in {"sources", "source_files"}:
                visit(item, "record")
            elif key in {"source_fbx", "path", "fbx_path"}:
                visit(item, key)

    visit(metadata, "record")
    return list(dict.fromkeys(result))


def _mesh_source_paths(mesh: Any) -> list[str]:
    merged = _metadata_source_paths(mesh.get("merge_metadata"))
    if merged:
        return merged
    result: list[str] = []
    for property_name in ("scsp_source_fbx", "scsp_eye_socket_source_fbx"):
        raw = mesh.get(property_name)
        if raw:
            result.extend(_metadata_source_paths(raw))
    return list(dict.fromkeys(result))


def _coordinates(block: Any) -> array:
    values = array("f", [0.0]) * (len(block.data) * 3)
    block.data.foreach_get("co", values)
    return values


def _mesh_sort(meshes: Iterable[Any]) -> list[Any]:
    result = list(dict.fromkeys(meshes))
    if any(getattr(mesh, "type", None) != "MESH" for mesh in result):
        raise TypeError("rename(meshes) accepts Blender mesh objects only")
    return sorted(result, key=lambda mesh: mesh.name)


def rename(meshes: Iterable[Any]) -> dict[str, Any]:
    """Rename shape keys across meshes and return a JSON-serializable report."""
    resources = _load_resources()
    meshes = _mesh_sort(meshes)
    unique_prefix, digest_length = resources["unique_prefix"], resources["unique_digest_length"]
    source_meshes: dict[str, list[Any]] = defaultdict(list)
    source_blocks: dict[tuple[str, str], Any] = {}
    original_key_names: dict[tuple[str, str], str] = {}
    source_states: dict[tuple[str, str], dict[str, Any]] = {}
    source_origins: dict[str, str] = {}
    unknowns: list[dict[str, str]] = []
    mesh_source_paths: dict[str, list[str]] = {}

    for mesh in meshes:
        mesh_source_paths[mesh.name] = _mesh_source_paths(mesh)
        keys = mesh.data.shape_keys
        if keys is None:
            continue
        per_source: dict[str, str] = {}
        for block in keys.key_blocks:
            if block == keys.key_blocks[0] and block.name == "Basis":
                continue
            source, origin = _source_for(mesh, block, resources)
            if source is None:
                continue
            if source in per_source:
                raise ValueError(
                    f"{mesh.name}: source identity {source!r} occurs as both "
                    f"{per_source[source]!r} and {block.name!r}"
                )
            per_source[source] = block.name
            source_meshes[source].append(mesh)
            source_blocks[(mesh.name, source)] = block
            original_key_names[(mesh.name, source)] = block.name
            relative = block.relative_key
            source_states[(mesh.name, source)] = {
                "value_before": float(block.value),
                "relative_key_before": relative.name if relative else None,
                "relative_pointer_before": _key_pointer(relative),
            }
            source_origins.setdefault(source, origin)
            if origin == "unknown preserved":
                unknowns.append({"mesh": mesh.name, "source": source})

    used = {"Basis"}
    conflicts: list[dict[str, str]] = []
    source_names: dict[str, str] = {}

    def unique(desired: str, identity: str, preserve: bool = False) -> str:
        if (preserve or _safe(desired, resources)) and desired not in used:
            used.add(desired)
            return desired
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:digest_length]
        for index in range(1000):
            candidate = f"{unique_prefix}{digest}{'' if index == 0 else index}"
            if _safe(candidate, resources) and candidate not in used:
                used.add(candidate)
                conflicts.append({"identity": identity, "desired": desired, "assigned": candidate})
                return candidate
        raise ValueError(f"Cannot allocate PMX-safe name for {identity!r}")

    for source in sorted(source_meshes):
        source_names[source] = unique(
            resources["source_names"].get(source, source),
            source,
            preserve=source_origins[source] == "unknown preserved",
        )

    combined_names: list[str] = []
    combination_plans: list[dict[str, Any]] = []
    for combination in resources["combinations"]:
        if not all(source in source_meshes for source in combination["sources"]):
            combination_plans.append({**combination, "global_missing": True})
            continue
        name = unique(combination["name"], f"combined:{combination['name']}")
        combined_names.append(name)
        combination_plans.append({**combination, "assigned_name": name, "global_missing": False})

    before_drivers: dict[str, list[dict[str, Any]]] = {}
    path_renames: dict[str, str] = {}
    for mesh in meshes:
        keys = mesh.data.shape_keys
        if keys is None:
            continue
        animation_data = keys.animation_data
        records = []
        if animation_data:
            for driver in animation_data.drivers:
                records.append({
                    "data_path": driver.data_path,
                    "target_paths": [target.data_path for variable in driver.driver.variables for target in variable.targets],
                    "valid_before": driver.is_valid,
                })
        before_drivers[mesh.name] = records
        mapping = {
            block.name: source_names[source]
            for (mesh_name, source), block in source_blocks.items()
            if mesh_name == mesh.name and block.name != source_names[source]
        }
        path_renames.update(mapping)
        for old, new in mapping.items():
            for key_name in keys.key_blocks.keys():
                if key_name == old:
                    block = keys.key_blocks[key_name]
                    block.name = new
                    break

    curve_updates: list[dict[str, str]] = []
    for action in bpy.data.actions:
        try:
            curves = action.fcurves
        except (AttributeError, RuntimeError):
            continue
        for curve in curves:
            old_name = _path_name(curve.data_path)
            new_name = path_renames.get(old_name)
            if new_name:
                curve_updates.append({"action": action.name, "old": curve.data_path})
                curve.data_path = _key_path(new_name)
                curve_updates[-1]["new"] = curve.data_path

    driver_updates: dict[str, list[dict[str, Any]]] = {}
    for mesh in meshes:
        keys = mesh.data.shape_keys
        if keys is None or keys.animation_data is None:
            continue
        after = []
        for driver in keys.animation_data.drivers:
            old_path = driver.data_path
            old_name = _path_name(old_path)
            if old_name in path_renames:
                driver.data_path = _key_path(path_renames[old_name])
            for variable in driver.driver.variables:
                for target in variable.targets:
                    old_name = _path_name(target.data_path)
                    if old_name in path_renames:
                        target.data_path = _key_path(path_renames[old_name])
            after.append({
                "old_path": old_path,
                "new_path": driver.data_path,
                "target_paths": [target.data_path for variable in driver.driver.variables for target in variable.targets],
                "target_paths_before": before_drivers[mesh.name][len(after)]["target_paths"],
                "target_paths_unchanged": (
                    before_drivers[mesh.name][len(after)]["target_paths"]
                    == [target.data_path for variable in driver.driver.variables for target in variable.targets]
                ),
                "valid_after": driver.is_valid,
            })
        driver_updates[mesh.name] = after

    combinations: list[dict[str, Any]] = []
    for plan in combination_plans:
        if plan["global_missing"]:
            combinations.append({k: v for k, v in plan.items() if k != "global_missing"} | {"status": "global sources missing"})
            continue
        assigned_name = plan["assigned_name"]
        for mesh in meshes:
            keys = mesh.data.shape_keys
            if keys is None:
                continue
            present = [source for source in plan["sources"] if (mesh.name, source) in source_blocks]
            if not present:
                continue
            existing = next((block for block in keys.key_blocks if block.name == assigned_name), None)
            added = existing is None
            value_before = None if added else float(existing.value)
            mute_before = None if added else bool(getattr(existing, "mute", False))
            relative_before = None if added else existing.relative_key.name
            relative_pointer_before = None if added else _key_pointer(existing.relative_key)
            value_at_creation = None
            initial_value = None
            if added:
                existing = mesh.shape_key_add(name=assigned_name, from_mix=False)
                value_at_creation = float(existing.value)
                existing.value = 0.0
                existing.relative_key = _basis_block(keys)
                if hasattr(existing, "mute"):
                    existing.mute = False
                initial_value = float(existing.value)
                if initial_value != 0.0:
                    raise AssertionError(
                        f"{mesh.name}/{assigned_name}: new combination is active at {initial_value}"
                    )
                if _key_pointer(existing.relative_key) != _key_pointer(_basis_block(keys)):
                    raise AssertionError(
                        f"{mesh.name}/{assigned_name}: new combination is not relative to Basis"
                    )
                if hasattr(existing, "mute") and bool(existing.mute):
                    raise AssertionError(
                        f"{mesh.name}/{assigned_name}: new combination is muted"
                    )
            elif (
                float(existing.value) != value_before
                or _key_pointer(existing.relative_key) != relative_pointer_before
            ):
                raise AssertionError(
                    f"{mesh.name}/{assigned_name}: existing combination changed before use"
                )
            changed = 0
            if added:
                basis = _coordinates(_basis_block(keys))
                result = array("f", basis)
                for source in present:
                    values = _coordinates(source_blocks[(mesh.name, source)])
                    for index, value in enumerate(values):
                        result[index] += value - basis[index]
                changed = _changed_vertex_count(result, basis)
                existing.data.foreach_set("co", result)
                if float(existing.value) != initial_value:
                    raise AssertionError(
                        f"{mesh.name}/{assigned_name}: combination activated while updating geometry"
                    )
            combinations.append({
                "name": assigned_name, "sources": plan["sources"], "english": plan["english"],
                "panel": plan["panel"], "mesh": mesh.name, "present_sources": present,
                "status": "added" if added else "existing", "changed_vertices": changed,
                "value_before": value_before,
                "value_at_creation": value_at_creation,
                "initial_value": initial_value,
                "value_after": float(existing.value),
                "mute_before": mute_before,
                "mute_after": bool(getattr(existing, "mute", False)),
                "relative_key_before": relative_before,
                "relative_key_after": existing.relative_key.name,
                "relative_relation_unchanged": (
                    added or _key_pointer(existing.relative_key) == relative_pointer_before
                ),
            })

    changed_source_values: list[dict[str, Any]] = []
    changed_source_relations: list[dict[str, Any]] = []
    for (mesh_name, source), state in source_states.items():
        block = source_blocks[(mesh_name, source)]
        relative = block.relative_key
        if float(block.value) != state["value_before"]:
            changed_source_values.append({
                "mesh": mesh_name,
                "source": source,
                "before": state["value_before"],
                "after": float(block.value),
            })
        if _key_pointer(relative) != state["relative_pointer_before"]:
            changed_source_relations.append({
                "mesh": mesh_name,
                "source": source,
                "relative_key_before": state["relative_key_before"],
                "relative_key_after": relative.name if relative else None,
            })
    if changed_source_values or changed_source_relations:
        raise AssertionError(
            f"Source key values changed: {changed_source_values}; "
            f"source key relations changed: {changed_source_relations}"
        )

    source_paths: set[Path] = set()
    mesh_records = []
    for mesh in meshes:
        keys = mesh.data.shape_keys
        if keys is None:
            continue
        mesh_source_path_list = mesh_source_paths[mesh.name]
        for source_path in mesh_source_path_list:
            if Path(source_path).is_file():
                source_paths.add(Path(source_path))
        marker = {
            source_names[source]: source
            for (mesh_name, source) in source_blocks
            if mesh_name == mesh.name
        }
        marker.update({name: f"@combined:{name}" for name in combined_names if any(b.name == name for b in keys.key_blocks)})
        keys[KEY_SOURCE_PROPERTY] = json.dumps(marker, ensure_ascii=False, separators=(",", ":"))
        records = []
        for block in keys.key_blocks:
            if block.name == "Basis":
                continue
            matched = next((source for (mesh_name, source), old in source_blocks.items() if mesh_name == mesh.name and old == block), None)
            stored = marker.get(block.name)
            source = matched or stored or block.name
            if source.startswith("@combined:"):
                source = source[1:]
            record = {
                "source": source,
                "old_name": original_key_names.get((mesh.name, matched)) if matched else None,
                "new_name": block.name,
                "panel": _panel(matched or block.name, resources),
            }
            if matched:
                state = source_states[(mesh.name, matched)]
                relative = block.relative_key
                record.update({
                    "value_before": state["value_before"],
                    "value_after": float(block.value),
                    "value_unchanged": float(block.value) == state["value_before"],
                    "relative_key_before": state["relative_key_before"],
                    "relative_key_after": relative.name if relative else None,
                    "relative_relation_unchanged": (
                        _key_pointer(relative) == state["relative_pointer_before"]
                    ),
                })
            records.append(record)
        mesh_records.append({
            "mesh": mesh.name,
            "source_fbx": (
                mesh_source_path_list[0]
                if len(mesh_source_path_list) == 1
                else mesh_source_path_list
            ),
            "source_fbx_count": len(mesh_source_path_list),
            "keys": records,
        })

    invalid_drivers = [
        {"mesh": mesh, "path": driver["new_path"]}
        for mesh, drivers in driver_updates.items()
        for driver in drivers if not driver["valid_after"]
    ]
    duplicate_keys = {
        mesh.name: len(mesh.data.shape_keys.key_blocks) - len({block.name for block in mesh.data.shape_keys.key_blocks})
        for mesh in meshes if mesh.data.shape_keys is not None
    }
    changed_targets = [
        {"mesh": mesh, "driver": driver}
        for mesh, drivers in driver_updates.items()
        for driver in drivers if not driver["target_paths_unchanged"]
    ]
    if invalid_drivers or changed_targets or any(duplicate_keys.values()):
        raise RuntimeError(
            f"Post-rename validation failed: drivers={invalid_drivers}, "
            f"target_paths={changed_targets}, duplicate_keys={duplicate_keys}"
        )

    report = {
        "schema": SCHEMA,
        "operation": "rename_mmd_shapes",
        "saved_blend": False,
        "resources": {"path": str(RESOURCE), "sha256": _sha256(RESOURCE)},
        "unity_mapping_source": resources["unity_exporter"],
        "key_source_marker": {
            "property": KEY_SOURCE_PROPERTY,
            "direction": "current key name -> original source identity",
            "combined_values": "@combined:<generated name>",
            "reverse_lookup_supported": False,
        },
        "summary": {
            "meshes": len(meshes), "source_identities": len(source_meshes),
            "renamed_keys": len(path_renames), "combined_keys_added": sum(x["status"] == "added" for x in combinations),
            "drivers": sum(len(x) for x in before_drivers.values()),
            "invalid_drivers_after": len(invalid_drivers),
            "driver_target_paths_changed": len(changed_targets),
            "duplicate_key_names": sum(duplicate_keys.values()),
        },
        "sources": [
            {
                "source": source, "new_name": source_names[source],
                "desired_name": resources["source_names"].get(source, source),
                "resolution": source_origins[source], "meshes": [mesh.name for mesh in source_meshes[source]],
                "panel": _panel(source, resources), "english": resources["english_names"].get(source),
            }
            for source in sorted(source_meshes)
        ],
        "source_files": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in sorted(source_paths, key=lambda item: str(item).casefold())
        ],
        "unknowns": unknowns, "unique_conflicts": conflicts,
        "combinations": combinations, "driver_updates": driver_updates,
        "fcurve_updates": curve_updates, "meshes": mesh_records,
    }
    json.dumps(report, ensure_ascii=False, allow_nan=False)
    return report


def _cli() -> None:
    meshes = sorted((obj for obj in bpy.data.objects if obj.type == "MESH" and obj.data.shape_keys), key=lambda obj: obj.name)
    print("SCSP_MMD_SHAPES_REPORT=" + json.dumps(rename(meshes), ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    _cli()
