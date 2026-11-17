"""Repository-relative data paths, optionally overridden by PEAS_ROOT."""
import os
import sys
from pathlib import Path

SCENES = ("canonical", "concentric", "eccentric", "pipe")


def repository_root():
    return Path(os.environ.get("PEAS_ROOT", Path(__file__).resolve().parents[3])).resolve()


def data_dir(scene):
    if scene not in SCENES:
        raise ValueError(f"Unknown scenario: {scene}")
    return repository_root() / "outputs" / "data" / scene


def analysis_dir(scene):
    return repository_root() / "outputs" / "analysis" / scene


def config_file(scene):
    if scene not in SCENES:
        raise ValueError(f"Unknown scenario: {scene}")
    candidates = (
        repository_root() / "configs" / "peas" / f"{scene}.json",
        Path(sys.prefix) / "share" / "peas" / "configs" / "peas" / f"{scene}.json",
        Path(__file__).resolve().parents[2] / "share" / "peas" / "configs" / "peas" / f"{scene}.json",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])
