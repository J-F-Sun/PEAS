from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_toml

SCENARIOS = ("canonical", "concentric", "eccentric", "pipe")
SPLITS = ("train", "validation", "test")


def resolve_path(value: str, config: dict[str, Any]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(config["_config_path"]).parent / path).resolve()


def validate_geometry(scenario: str, geometry: dict[str, Any]) -> None:
    center = np.asarray(geometry["rx_center"], dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(float(geometry["rx_radius"])) or float(geometry["rx_radius"]) <= 0:
        raise ValueError("Invalid receiver geometry.")
    if scenario in ("concentric", "eccentric"):
        outer = np.asarray(geometry["outer_center"], dtype=float)
        radius, kappa = float(geometry["outer_radius"]), float(geometry["kappa"])
        if outer.shape != (3,) or not np.isfinite(outer).all() or not np.isfinite([radius, kappa]).all():
            raise ValueError("Invalid outer-sphere geometry.")
        if radius <= np.linalg.norm(center - outer) + geometry["rx_radius"] or kappa < 0:
            raise ValueError("Receiver must lie wholly inside the outer sphere; kappa >= 0.")
        distance = np.linalg.norm(center - outer)
        if (scenario == "concentric" and distance > 1e-10) or (scenario == "eccentric" and distance <= 1e-10):
            raise ValueError("Scene name does not match concentric/eccentric geometry.")
        if scenario == "eccentric" and abs(center[1]-outer[1]) + abs(center[2]-outer[2]) > 1e-10:
            raise ValueError("The first-version eccentric configuration uses the x axis.")
    if scenario == "pipe":
        r, lo, hi, k = (float(geometry[n]) for n in ("pipe_radius", "z_min", "z_max", "wall_kappa"))
        if not np.isfinite([r, lo, hi, k]).all() or r <= np.hypot(*center[:2]) + geometry["rx_radius"] or k < 0:
            raise ValueError("Invalid pipe wall or receiver placement.")
        if not lo + geometry["rx_radius"] < center[2] < hi - geometry["rx_radius"]:
            raise ValueError("Receiver must fit between pipe end caps.")
        if geometry["end_boundary"] not in ("absorbing", "reflecting"):
            raise ValueError("Unsupported pipe end boundary.")
        if abs(center[1]) > 1e-10 or abs(center[2] - (lo+hi)/2) > 1e-10:
            raise ValueError("First-version pipe split symmetry assumes RX on x and at axial midpoint.")


def load_config(path: str | Path, profile: str = "formal") -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    cfg = read_toml(path)
    cfg["_config_path"] = str(path)
    cfg["profile"] = profile
    if cfg.get("schema_version") != 1 or cfg["scenario"] not in SCENARIOS:
        raise ValueError("Expected schema_version=1 and one of the four supported scenes.")
    validate_geometry(cfg["scenario"], cfg["geometry"])
    if profile == "smoke":
        for section in ("model", "training", "time", "runtime"):
            cfg[section].update(copy.deepcopy(cfg.get("smoke", {}).get(section, {})))
    elif profile not in ("formal", "experiment"):
        raise ValueError("profile must be formal, smoke, or experiment.")
    m, t, training = cfg["model"], cfg["time"], cfg["training"]
    minimum = 2 if cfg["scenario"] == "canonical" else 5 if cfg["scenario"] == "pipe" else 3
    if m["max_entities"] < minimum or m["entity_dim"] != 13 or m["numerical_dim"] != 7:
        raise ValueError("Expected 13-field entities, 7 numerical features, and sufficient entities.")
    for field in ("num_transformer_layers", "num_heads", "key_dim", "ff_dim", "shape_hidden", "postprocess_window"):
        if int(m[field]) < 1:
            raise ValueError(f"model.{field} must be positive.")
    for field in ("shared_widths", "max_widths"):
        if not m[field] or any(int(v) < 1 for v in m[field]):
            raise ValueError(f"model.{field} must contain positive widths.")
    if not 0 <= m["dropout"] < 1 or t["n_samples"] < 2 or t["t_max"] <= 0 or t["zero_padding"] < 0:
        raise ValueError("Invalid model dropout/time settings.")
    if training["curve_loss_mode"] not in ("raw", "postprocessed") or training["epochs"] < 1 or training["batch_size"] < 1:
        raise ValueError("Invalid training budget or curve loss mode.")
    if training["monotonic_weight"] < 0 or training["gradient_clip"] < 0 or training["log_interval"] < 1:
        raise ValueError("Invalid loss/gradient/logging settings.")
    if profile != "experiment" and cfg["data"]["counts"] != {"train": 160, "validation": 20, "test": 20}:
        raise ValueError("Formal split contract is 160/20/20 positions.")
    if profile == "experiment" and (set(cfg["data"]["counts"]) != set(SPLITS) or
            any(int(v) < 1 for v in cfg["data"]["counts"].values())):
        raise ValueError("Experiments require explicit positive split counts.")
    lr = training["learning_rate_schedule"]
    if lr["warmup_epochs"] < 0 or lr["decay_epochs"] < 1 or not 0 < lr["min_lr"] <= lr["base_lr"] or not 0 < lr["initial_lr"] <= lr["base_lr"]:
        raise ValueError("Invalid learning-rate schedule.")
    schedule = training["loss_weight_schedule"]
    if schedule["warmup_epochs"] < 0 or schedule["decay_epochs"] < 1 or cfg["runtime"]["threads"] < 1:
        raise ValueError("Invalid loss schedule/runtime settings.")
    return cfg
