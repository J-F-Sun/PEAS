from __future__ import annotations
import math
import torch

def canonical_multiscale_values(t, y, tau, features):
    """E04's five inputs, shared by inference and the retained training graph."""
    ts = float(features["time_scale"])
    tau_scale = float(features["tau_scale"])
    return torch.cat([t / ts, y, tau, torch.log1p(t) / math.log(1.0 + ts),
        torch.log1p(tau / tau_scale) / math.log(1.0 + ts / tau_scale)], dim=1)


def canonical_features(t, y, config):
    if config["features"].get("preset") == "canonical5_multiscale":
        return canonical_multiscale_values(t, y, t * y**2, config["features"])
    return torch.cat([t, y, t * y**2], dim=1)


