"""Expression stage, called once after part merging."""
from __future__ import annotations

from typing import Any

from eye_morphs import generate as generate_eye_morphs, remove_unwanted_keys
from facial_morphs import generate as generate_facial_morphs
from rename_mmd_shapes import _load_resources, rename


def process(meshes: list[Any]) -> dict[str, Any]:
    """Clean unwanted morphs, apply canonical MMD names, and generate eye/brow/mouth morphs."""
    resources = _load_resources()
    removed = remove_unwanted_keys(meshes, resources)
    report = rename(meshes)
    report["removed_morphs"] = removed
    report["eye_morphs"] = generate_eye_morphs(meshes, resources)
    report["facial_morphs"] = generate_facial_morphs(meshes, resources)
    return report
