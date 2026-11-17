"""Deterministic scene encoders retaining the existing MCvD 13/7-field layouts."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def angles(v: np.ndarray) -> tuple[float, float, float]:
    radius = float(np.linalg.norm(v))
    if radius == 0:
        return 0.0, 0.0, 0.0
    return radius, math.atan2(float(v[1]), float(v[0])), math.asin(float(v[2]) / radius)


def validate_samples(samples: np.ndarray, scenario: str, g: dict[str, Any]) -> None:
    if samples.ndim != 2 or samples.shape[1] != 3 or not np.isfinite(samples).all() or len(samples) == 0:
        raise ValueError("samples must be a nonempty finite (N,3) array.")
    if np.any(np.linalg.norm(samples - g["rx_center"], axis=1) <= g["rx_radius"]):
        raise ValueError("Transmitter must lie outside the absorbing receiver.")
    if scenario in ("concentric", "eccentric"):
        if np.any(np.linalg.norm(samples - g["outer_center"], axis=1) >= g["outer_radius"]):
            raise ValueError("Transmitter must lie inside the outer sphere.")
    if scenario == "pipe":
        if np.any(np.linalg.norm(samples[:, :2], axis=1) >= g["pipe_radius"]) or np.any(samples[:, 2] <= g["z_min"]) or np.any(samples[:, 2] >= g["z_max"]):
            raise ValueError("Transmitter must lie inside the pipe.")


def orbit_key(x: np.ndarray, scenario: str, g: dict[str, Any]) -> tuple[float, ...]:
    if scenario in ("canonical", "concentric"):
        values = (np.linalg.norm(x - g["rx_center"]),)
    elif scenario == "eccentric":
        v = x - g["outer_center"]
        values = (v[0], np.hypot(v[1], v[2]))
    else:
        values = (x[0], abs(x[1]), abs(x[2] - (g["z_min"]+g["z_max"])/2))
    return tuple(round(float(v), 6) for v in values)


def encode_topology(x0: np.ndarray, geometry: dict[str, Any], scenario: str, delta_t: float, max_entities: int) -> tuple[np.ndarray, np.ndarray]:
    g = geometry
    transmitter = np.asarray(x0) - g["rx_center"]
    distance, theta, phi = angles(transmitter)
    tokens = np.zeros((max_entities, 13), dtype=np.float32)
    tokens[:, 0] = -1.0
    tokens[1] = [1, *transmitter, distance, theta, phi, 0, 0, 0, 0, 0, 0]
    if scenario == "canonical":
        tokens[0] = [0, 0, 0, 0, 0, 0, 0, g["rx_radius"], 0, 0, 0, 0, 0]
        numerical = [1, delta_t, 0, 0, 0, 0, 0]
    elif scenario in ("concentric", "eccentric"):
        outer = np.asarray(g["outer_center"]) - g["rx_center"]
        od, ot, op = angles(outer)
        tokens[0] = [0, 0, 0, 0, 0, 0, 0, g["rx_radius"], g["kappa"], 0, 0, 0, 0]
        tokens[2] = [2, *outer, od, ot, op, g["outer_radius"], g["kappa"], 1, 0, 0, 0]
        numerical = [1, delta_t, g["rx_radius"], g["outer_radius"], g["kappa"], 1, od]
    elif scenario == "pipe":
        zmid = (g["z_min"]+g["z_max"])/2 - g["rx_center"][2]
        tokens[0] = [0, 0, 0, 0, 0, 0, 0, g["rx_radius"], 1, 0, 0, 0, 0]
        tokens[2] = [2, 0, 0, zmid, abs(zmid), 0, 0, g["pipe_radius"], g["wall_kappa"], g["z_max"]-g["z_min"], 0, 0, 0]
        for i, z, normal in ((3, g["z_min"], -1), (4, g["z_max"], 1)):
            rz = z - g["rx_center"][2]
            tokens[i] = [3, 0, 0, rz, abs(rz), 0, np.sign(rz), g["pipe_radius"], float(g["end_boundary"] == "absorbing"), 0, normal, 0, 0]
        numerical = [1, delta_t, g["rx_radius"], g["pipe_radius"], g["z_min"]-g["rx_center"][2], g["z_max"]-g["rx_center"][2], g["wall_kappa"]]
    else:
        raise ValueError(f"Unknown scene {scenario}")
    return tokens, np.asarray(numerical, dtype=np.float32)


def encode_batch(samples: np.ndarray, times: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    validate_samples(samples, config["scenario"], config["geometry"])
    encoded = [encode_topology(x, config["geometry"], config["scenario"], float(times[1]-times[0]), config["model"]["max_entities"]) for x in samples]
    return np.stack([e[0] for e in encoded]), np.stack([e[1] for e in encoded])
