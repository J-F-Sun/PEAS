from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoints import load_checkpoint
from .encoders import encode_batch
from .io import save_npz
from .model import MCvDTransformer
from .postprocess import postprocess_mcvd_output
from .runtime import choose_device


@torch.no_grad()
def predict_arrays(model: MCvDTransformer, samples: np.ndarray, times: np.ndarray, config: dict[str, Any], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    topology, numerical = encode_batch(samples, times, config)
    model.eval()
    raw, processed = [], []
    batch_size = config["training"]["batch_size"]
    for start in range(0, len(samples), batch_size):
        output = model(torch.as_tensor(topology[start:start+batch_size], device=device), torch.as_tensor(numerical[start:start+batch_size], device=device))
        prediction = postprocess_mcvd_output(output["shape"], output["maximum"],
            zero_padding=config["time"]["zero_padding"], moving_average_window=config["model"]["postprocess_window"])
        processed.append(prediction.cpu().numpy().astype(np.float64))
        raw.append(output["curve"][:, config["time"]["zero_padding"]:].cpu().numpy().astype(np.float64))
    return np.concatenate(processed), np.concatenate(raw)


def load_predictor(path: Path, device_name: str) -> tuple[MCvDTransformer, dict[str, Any], torch.device]:
    checkpoint = load_checkpoint(path)
    device = choose_device(device_name)
    torch.set_num_threads(checkpoint["config"]["runtime"]["threads"])
    model = MCvDTransformer(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint, device


def read_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"x", "y", "z"}.issubset(rows[0]):
        raise ValueError("Positions CSV needs x,y,z columns and optional id.")
    samples = np.asarray([[float(row[k]) for k in ("x", "y", "z")] for row in rows], dtype=np.float64)
    ids = np.asarray([row.get("id") or f"prediction_{i+1:03d}" for i, row in enumerate(rows)])
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Prediction IDs must be unique.")
    return samples, ids


def warn_outside_training(samples: np.ndarray, checkpoint: dict[str, Any]) -> None:
    support = checkpoint["data_info"]["training_support"]
    distance = np.linalg.norm(samples - checkpoint["config"]["geometry"]["rx_center"], axis=1)
    outside = np.any((distance < support["distance_min"]) | (distance > support["distance_max"]))
    box_min, box_max = np.asarray(support["coordinate_min"]), np.asarray(support["coordinate_max"])
    outside = outside or np.any(samples < box_min) or np.any(samples > box_max)
    if outside:
        warnings.warn("Requested positions extend outside the training position range; prediction accuracy there has not been evaluated.", stacklevel=2)


def save_predictions(path: Path, *, samples: np.ndarray, ids: np.ndarray, times: np.ndarray,
                     prediction: np.ndarray, raw: np.ndarray, checkpoint: dict[str, Any], split: str = "prediction", **extra: Any) -> None:
    save_npz(path, samples=samples, ids=ids, times=times, prediction=prediction, raw_prediction=raw,
        scenario=checkpoint["config"]["scenario"], profile=checkpoint["config"]["profile"], split=split,
        data_role="mcvd_prediction", geometry_json=json.dumps(checkpoint["config"]["geometry"], sort_keys=True),
        model_epoch=checkpoint["epoch"], ams_run_id=checkpoint["data_info"]["ams_run_id"],
        dataset_fingerprint=checkpoint["data_fingerprint"], **extra)


def predict(checkpoint_path: Path, positions: Path, output: Path, device_name: str = "auto") -> dict[str, Any]:
    model, checkpoint, device = load_predictor(checkpoint_path, device_name)
    samples, ids = read_positions(positions)
    warn_outside_training(samples, checkpoint)
    times = checkpoint["times"].numpy()
    prediction, raw = predict_arrays(model, samples, times, checkpoint["config"], device)
    if not np.isfinite(prediction).all() or not np.isfinite(raw).all():
        raise ValueError("Non-finite model output; inspect the checkpoint/training history.")
    save_predictions(output, samples=samples, ids=ids, times=times, prediction=prediction, raw=raw, checkpoint=checkpoint)
    return {"output": str(output), "shape": list(prediction.shape), "profile": checkpoint["config"]["profile"]}
