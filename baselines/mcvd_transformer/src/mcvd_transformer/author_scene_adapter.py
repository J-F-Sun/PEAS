"""Physical scene adapter for the audited author MCvD input/architecture.

Native class IDs 0--4 retain their original meanings. Classes 5--8 extend
the entity description to enclosing Robin surfaces and finite pipe caps.
They are input adaptations, not objects present in the published dataset.
All points, cylinder axes and cap normals rotate by one rigid matrix.
"""
from __future__ import annotations

import copy
import numpy as np
from .encoders import validate_samples

VERSION = "author_scene_adapter_v1"
COMMIT = "c580cca0ef956866e49fbcc84d38e331bcf24de0"
ROBIN_SHELL, ROBIN_CYLINDER, ABSORBING_CAP, REFLECTING_CAP = 5, 6, 7, 8


def author_config(base, seed, epochs=400):
    cfg = copy.deepcopy(base)
    cfg["seed"] = int(seed)
    cfg["model"] = {"max_entities": 15, "entity_dim": 13, "numerical_dim": 7,
                    "num_transformer_layers": 2, "num_heads": 16, "key_dim": 13,
                    "ff_dim": 32, "dropout": .1, "shared_widths": [400, 800, 1600],
                    "shape_hidden": 2 * (cfg["time"]["n_samples"] + 10),
                    "max_widths": [1600, 800, 400], "postprocess_window": 3}
    cfg["time"]["zero_padding"] = 10
    cfg["training"] = {"epochs": int(epochs), "batch_size": 64,
                       "curve_loss_mode": "raw", "monotonic_weight": .0001,
                       "gradient_clip": 0., "log_interval": 50,
                       "learning_rate_schedule": {"warmup_epochs": 25, "decay_epochs": 175,
                                                  "initial_lr": 1e-5, "base_lr": .001, "min_lr": 5e-5},
                       "loss_weight_schedule": {"warmup_epochs": 30, "decay_epochs": 200}}
    cfg["runtime"]["threads"] = 1
    cfg["author_source"] = {"commit": COMMIT, "version": VERSION,
                            "amplitude_epsilon": 1., "rotation": "Rz(theta) @ Ry(phi), rigid full geometry",
                            "boundary_adapter": "class5_shell_class6_cylinder_class7_8_cap"}
    return cfg


def rotation_matrix(theta_degrees, phi_degrees):
    t, p = np.deg2rad([theta_degrees, phi_degrees])
    rz = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    return rz @ ry


def point_fields(vector):
    r = float(np.linalg.norm(vector))
    longitude = float(np.mod(np.arctan2(vector[1], vector[0]), 2 * np.pi)) if r else 0.
    latitude = float(np.arcsin(np.clip(vector[2] / r, -1, 1))) if r else 0.
    return [*vector, r, longitude, latitude]


def encode_scene(samples, cfg, rng=None, rotation=None):
    """Tx is the origin; target Rx first; no Tx entity; 15 x 13 + 7.

    The numerical vector is [D, actual dt, five unobstructed LOS entries].
    Every direct Tx-to-facing-Rx segment lies inside the convex container,
    so no other object intersects those segments in these three scenes.
    """
    samples = np.asarray(samples, dtype=float)
    scene, g = cfg["scenario"], cfg["geometry"]
    validate_samples(samples, scene, g)
    if cfg["author_source"]["version"] != VERSION:
        raise ValueError("Wrong scene-adapter version")
    if cfg["model"]["max_entities"] != 15:
        raise ValueError("Author adapter requires 15 entity slots")
    if rng is not None and rotation is not None:
        raise ValueError("Choose random augmentation or an explicit rigid rotation")
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=float)
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        if not np.isclose(np.linalg.det(rotation), 1):
            raise ValueError("Expected a proper rigid rotation")
    topology = np.zeros((len(samples), 15, 13), dtype=np.float32)
    topology[:, :, 0] = -1
    for i, tx in enumerate(samples):
        q = rotation if rotation is not None else (
            rotation_matrix(rng.randint(0, 360), rng.randint(-90, 90)) if rng is not None else np.eye(3))
        rx = q @ (np.asarray(g["rx_center"]) - tx)
        topology[i, 0] = [0, *point_fields(rx), g["rx_radius"], 0, 0, 0, 0, 0]
        if scene in ("concentric", "eccentric"):
            center = q @ (np.asarray(g["outer_center"]) - tx)
            # Interior shell; class differs from an exterior solid reflector.
            topology[i, 1] = [ROBIN_SHELL, *point_fields(center), g["outer_radius"], g["kappa"], 1, 0, 0, 0]
        elif scene == "pipe":
            middle = (g["z_min"] + g["z_max"]) / 2
            center = q @ (np.array([0., 0., middle]) - tx)
            axis = q @ np.array([0., 0., 1.])
            topology[i, 1] = [ROBIN_CYLINDER, *point_fields(center), g["pipe_radius"], g["wall_kappa"],
                              g["z_max"] - g["z_min"], *axis]
            cap_type = ABSORBING_CAP if g["end_boundary"] == "absorbing" else REFLECTING_CAP
            for j, (z, sign) in enumerate(((g["z_min"], -1), (g["z_max"], 1)), 2):
                center = q @ (np.array([0., 0., z]) - tx)
                topology[i, j] = [cap_type, *point_fields(center), g["pipe_radius"], *(sign * axis), 0, 0]
        else:
            raise ValueError("This adapter supports concentric/eccentric/pipe")
        if rng is not None:
            rng.shuffle(topology[i, 1:])
    numerical = np.zeros((len(samples), 7), dtype=np.float32)
    numerical[:, 0] = 1.
    numerical[:, 1] = cfg["time"]["t_max"] / (cfg["time"]["n_samples"] - 1)
    return topology, numerical
