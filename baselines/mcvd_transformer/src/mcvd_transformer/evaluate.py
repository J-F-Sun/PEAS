from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .data import CurveBundle, validate_bundle
from .io import save_json
from .metrics import curve_metrics
from .predict import load_predictor, predict_arrays, save_predictions


def canonical_analytic(samples: np.ndarray, times: np.ndarray, geometry: dict[str, Any]) -> np.ndarray:
    import math
    radius = float(geometry["rx_radius"])
    distance = np.linalg.norm(samples-geometry["rx_center"], axis=1)
    values = np.zeros((len(samples), len(times)), dtype=np.float64)
    for i, d in enumerate(distance):
        values[i, times > 0] = [radius/d * math.erfc((d-radius)/(2*math.sqrt(float(t)))) for t in times[times > 0]]
    return values


def aligned_reference(path: Path, samples: np.ndarray, times: np.ndarray, key: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as v:
        if not np.array_equal(v["samples"], samples) or not np.array_equal(v["times"], times):
            raise ValueError("Reference samples/time grid/order must match exactly; resample explicitly before evaluation.")
        values = v[key].copy()
    if values.shape != (len(samples), len(times)):
        raise ValueError("External reference must have shape (N,T).")
    return values


def evaluate(checkpoint_path: Path, bundle: CurveBundle, split: str, output: Path, *, device_name: str = "auto",
             reference_path: Path | None = None, reference_key: str = "reference", reference_label: str = "external_reference",
             analytic: bool = False, manifest: Path | None = None, resume_output: bool = False) -> dict[str, Any]:
    model, checkpoint, device = load_predictor(checkpoint_path, device_name)
    cfg = checkpoint["config"]
    if manifest:
        cfg["data"]["manifest"] = str(manifest.expanduser().resolve())
    validate_bundle(bundle, cfg)
    if checkpoint["data_fingerprint"] != bundle.fingerprint:
        raise ValueError("Evaluation dataset differs from the checkpoint's fixed AMS dataset.")
    if analytic and cfg["scenario"] != "canonical":
        raise ValueError("Canonical analytic reference is available only in the canonical geometry.")
    if analytic and reference_path:
        raise ValueError("Choose either canonical analytic or an external reference.")
    idx = bundle.indices(split)
    if not len(idx):
        raise ValueError(f"No completed positions in split {split}.")
    samples, ids, times = bundle.samples[idx], bundle.ids[idx], bundle.times
    prediction, raw = predict_arrays(model, samples, times, cfg, device)
    summary = {"scenario": cfg["scenario"], "profile": cfg["profile"], "split": split, "model_epoch": checkpoint["epoch"],
               "ams_run_id": bundle.metadata["ams_run_id"], "ams_data_role": bundle.metadata["data_role"],
               "against_ams": curve_metrics(prediction, bundle.values[idx], ids),
               "raw_against_ams": curve_metrics(raw, bundle.values[idx], ids)}
    extra = {"reference_ams": bundle.values[idx], **{key: value[idx] for key, value in bundle.uncertainty.items()}}
    if analytic or reference_path:
        label = "canonical_analytic" if analytic else reference_label
        reference = canonical_analytic(samples, times, cfg["geometry"]) if analytic else aligned_reference(reference_path, samples, times, reference_key)
        summary["physical_reference"] = {"label": label, "model": curve_metrics(prediction, reference, ids),
                                          "ams": curve_metrics(bundle.values[idx], reference, ids)}
        extra.update(reference=reference, reference_label=label)
    output = output.expanduser().resolve()
    status = {"complete": False, "model_epoch": checkpoint["epoch"], "dataset_fingerprint": bundle.fingerprint,
              "split": split, "profile": cfg["profile"], "reference_label": "canonical_analytic" if analytic else reference_label if reference_path else "ams_only"}
    if output.exists():
        import json
        marker = output/"status.json"
        if not resume_output or not marker.is_file() or json.loads(marker.read_text()) != status:
            raise FileExistsError(f"Evaluation directory already exists or does not match an interrupted evaluation: {output}")
    else:
        output.mkdir(parents=True)
    save_json(output/"status.json", status)
    save_predictions(output/"predictions.npz", samples=samples, ids=ids, times=times, prediction=prediction, raw=raw,
                     checkpoint=checkpoint, split=split, **extra)
    save_json(output/"metrics.json", summary)
    rows = summary["against_ams"]["per_position"]
    with (output/"position_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(output/"status.json", status | {"complete": True})
    return {"output": str(output), "profile": cfg["profile"], "against_ams": {k: v for k, v in summary["against_ams"].items() if k != "per_position"}}
