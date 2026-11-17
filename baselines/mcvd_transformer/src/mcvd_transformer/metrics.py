from __future__ import annotations

from typing import Any

import numpy as np


def curve_metrics(prediction: np.ndarray, reference: np.ndarray, ids: np.ndarray) -> dict[str, Any]:
    p, q = np.asarray(prediction, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if p.shape != q.shape or p.ndim != 2 or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Metric inputs must be matching finite (N,T) arrays.")
    error = p-q
    relative = np.linalg.norm(error, axis=1) / np.maximum(np.linalg.norm(q, axis=1), 1e-12)
    rows = [{"id": str(pid), "mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e**2))), "relative_l2": float(r)}
            for pid, e, r in zip(ids, error, relative)]
    return {"positions": len(p), "time_points": p.shape[1], "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))), "mean_relative_l2": float(relative.mean()),
            "median_relative_l2": float(np.median(relative)), "max_relative_l2": float(relative.max()),
            "pooled_relative_l2": float(np.linalg.norm(error) / max(np.linalg.norm(q), 1e-12)), "per_position": rows}
