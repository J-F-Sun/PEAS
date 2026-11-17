"""AMS-MOD manifest/JSON adapter and portable, non-pickle dataset archives."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import SPLITS, resolve_path
from .encoders import encode_batch, orbit_key, validate_samples
from .io import read_toml, safe_id, save_npz


@dataclass
class CurveBundle:
    times: np.ndarray
    samples: np.ndarray
    values: np.ndarray
    ids: np.ndarray
    splits: np.ndarray
    geometry: dict[str, Any]
    simulation: dict[str, Any]
    metadata: dict[str, Any]
    uncertainty: dict[str, np.ndarray] = field(default_factory=dict)

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.splits == split)

    @property
    def counts(self) -> dict[str, int]:
        return {name: int(len(self.indices(name))) for name in SPLITS}

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for value in (self.times, self.samples, self.values):
            digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
        identity = {k: v for k, v in self.metadata.items() if not k.endswith("_path")}
        digest.update(json.dumps([self.ids.tolist(), self.splits.tolist(), self.geometry, self.simulation, identity], sort_keys=True).encode())
        return digest.hexdigest()


def manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not {"id", "split", "x", "y", "z"}.issubset(row) for row in rows):
        raise ValueError(f"Invalid positions manifest: {path}")
    for row in rows:
        safe_id(row["id"])
        if row["split"] not in SPLITS:
            raise ValueError(f"Unknown split in {path}: {row['split']}")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate position IDs in manifest.")
    return rows


def validate_bundle(bundle: CurveBundle, cfg: dict[str, Any], *, require_trainable: bool = False) -> None:
    n, nt = len(bundle.ids), len(bundle.times)
    if bundle.metadata["scenario"] != cfg["scenario"] or bundle.geometry != cfg["geometry"]:
        raise ValueError("Dataset scenario/geometry differs from configuration.")
    if bundle.values.shape != (n, nt) or bundle.samples.shape != (n, 3) or bundle.splits.shape != (n,):
        raise ValueError("Expected values=(N,T), samples=(N,3), ids/splits=(N,).")
    if nt < 2 or not np.isfinite(bundle.times).all() or not np.all(np.diff(bundle.times) > 0):
        raise ValueError("Invalid time grid.")
    if bundle.times[0] != 0 or not np.allclose(np.diff(bundle.times), bundle.times[1], rtol=1e-8, atol=1e-12):
        raise ValueError("MCvD expects an AMS uniform time grid including t=0.")
    if not np.isfinite(bundle.values).all() or np.any(bundle.values < 0) or np.any(bundle.values > 1 + 1e-8):
        raise ValueError("AMS probability labels must be finite and lie in [0,1].")
    if np.any(np.abs(bundle.values[:, 0]) > 1e-10) or np.any(np.diff(bundle.values, axis=1) < -1e-10):
        raise ValueError("AMS labels must start at zero and be cumulative probabilities.")
    if len(set(bundle.ids.tolist())) != n or not set(bundle.splits.tolist()).issubset(SPLITS):
        raise ValueError("Duplicate IDs or invalid split labels.")
    validate_samples(bundle.samples, cfg["scenario"], bundle.geometry)
    keys = [orbit_key(x, cfg["scenario"], bundle.geometry) for x in bundle.samples]
    if len(set(keys)) != n:
        raise ValueError("Duplicate physical symmetry classes within/across splits.")
    expected_manifest = resolve_path(cfg["data"]["manifest"], cfg)
    expected = {r["id"]: r for r in manifest_rows(expected_manifest)}
    for pid, split, x in zip(bundle.ids, bundle.splits, bundle.samples):
        row = expected.get(str(pid))
        if row is None or row["split"] != split or not np.array_equal(x, [float(row[k]) for k in ("x", "y", "z")]):
            raise ValueError(f"Position {pid} differs from the fixed AMS-MOD manifest.")
    for name, values in bundle.uncertainty.items():
        if values.shape != (n, nt) or not np.isfinite(values).all():
            raise ValueError(f"Invalid AMS uncertainty array {name}.")
        if name in ("q_std", "q_se") and np.any(values < 0):
            raise ValueError(f"Negative AMS uncertainty {name}.")
    if cfg["profile"] == "formal":
        if bundle.metadata["profile"] != "formal" or bundle.metadata["data_role"] != "ams_mod_estimate":
            raise ValueError("Formal training requires formal AMS-MOD estimates.")
        if bundle.counts != cfg["data"]["counts"] or not bundle.metadata.get("complete", False) or set(bundle.ids) != set(expected):
            raise ValueError(f"Formal data must be complete 160/20/20; found {bundle.counts}.")
        if nt != cfg["time"]["n_samples"] or not np.isclose(bundle.times[-1], cfg["time"]["t_max"], rtol=0, atol=1e-12):
            raise ValueError("Formal time grid differs from the configured 801-point grid.")
    elif cfg["profile"] == "experiment":
        if bundle.metadata["profile"] != "experiment" or bundle.metadata["data_role"] not in ("analytic_reference", "ams_mod_estimate"):
            raise ValueError("Experimental data need an explicit experiment profile and truthful label source.")
        if bundle.metadata["data_role"] == "analytic_reference" and (cfg["scenario"] != "canonical" or
                bundle.metadata.get("diffusion_coefficient") != 1.0 or bundle.uncertainty):
            raise ValueError("Analytic experiments currently support only D=1 canonical labels without AMS uncertainty.")
        if bundle.counts != cfg["data"]["counts"] or not bundle.metadata.get("complete", False) or set(bundle.ids) != set(expected):
            raise ValueError("Experimental data must match their complete, explicit positions manifest and split counts.")
        if nt != cfg["time"]["n_samples"] or not np.isclose(bundle.times[-1], cfg["time"]["t_max"], rtol=0, atol=1e-12):
            raise ValueError("Experimental time grid differs from configuration.")
    elif bundle.metadata["profile"] not in ("smoke", "synthetic_smoke"):
        raise ValueError("Smoke commands accept only explicitly marked smoke data.")
    if require_trainable:
        if not bundle.counts["train"] or not bundle.counts["validation"]:
            raise ValueError("Training requires nonempty train and validation splits.")
        normalized = np.isin(bundle.splits, ("train", "validation"))
        bad = bundle.ids[normalized & (bundle.values[:, -1] <= 0)]
        if len(bad):
            raise ValueError(f"q(T)=0 prevents shape normalization; inspect AMS repeats for {bad.tolist()}. No positions were dropped.")


def load_ams_run(run: Path, cfg: dict[str, Any]) -> CurveBundle:
    run = run.expanduser().resolve()
    meta = read_toml(run / "run.toml")
    if meta.get("schema_version") != 1 or meta["scenario"] != cfg["scenario"] or meta["geometry"] != cfg["geometry"]:
        raise ValueError("AMS run schema/scenario/geometry mismatch.")
    if meta["profile"] != cfg["profile"]:
        raise ValueError("AMS run profile differs from requested formal/smoke profile.")
    rows = manifest_rows(run / "positions.csv")
    records = []
    missing = []
    for row in rows:
        path = run / row["split"] / row["id"] / "curve.json"
        if not path.is_file():
            missing.append(row["id"])
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 1 or value["id"] != row["id"] or value["split"] != row["split"]:
            raise ValueError(f"AMS curve identity/schema mismatch: {path}")
        if value["x0"] != [float(row[k]) for k in ("x", "y", "z")]:
            raise ValueError(f"AMS coordinates differ from manifest: {path}")
        for key in ("scenario", "profile", "geometry", "simulation"):
            if value[key] != meta[key]:
                raise ValueError(f"AMS curve {key} differs from run metadata: {path}")
        if value["data_role"] != "ams_mod_estimate" or value["completed_repeats"] != meta["simulation"]["repeats"]:
            raise ValueError(f"AMS repeats are incomplete or data_role is wrong: {path}")
        records.append(value)
    if cfg["profile"] == "formal" and missing:
        raise ValueError(f"AMS formal run incomplete: missing {len(missing)} positions, first {missing[:5]}.")
    if not records:
        raise ValueError("No completed AMS curves found.")
    times = records[0]["times"]
    if any(v["times"] != times for v in records):
        raise ValueError("AMS curves do not share the same time grid.")
    sim = meta["simulation"]
    if len(times) != sim["output_points"] or times[-1] != sim["t_max"]:
        raise ValueError("AMS time grid differs from simulation metadata.")
    bundle = CurveBundle(
        times=np.asarray(times, dtype=np.float64),
        samples=np.asarray([v["x0"] for v in records], dtype=np.float64),
        values=np.asarray([v["q"] for v in records], dtype=np.float64),
        ids=np.asarray([v["id"] for v in records]), splits=np.asarray([v["split"] for v in records]),
        geometry=meta["geometry"], simulation=sim,
        metadata={"scenario": meta["scenario"], "profile": meta["profile"], "data_role": "ams_mod_estimate",
                  "ams_run_id": run.name, "ams_run_path": str(run), "ams_signature": meta["signature"],
                  "complete": not missing, "completed_repeats": [v["completed_repeats"] for v in records]},
        uncertainty={key: np.asarray([v[key] for v in records], dtype=np.float64) for key in ("q_std", "q_se", "ci95_low", "ci95_high")},
    )
    validate_bundle(bundle, cfg)
    return bundle


def save_bundle(path: Path, bundle: CurveBundle) -> None:
    save_npz(path, times=bundle.times, samples=bundle.samples, values=bundle.values, ids=bundle.ids,
             splits=bundle.splits, geometry_json=json.dumps(bundle.geometry, sort_keys=True),
             simulation_json=json.dumps(bundle.simulation, sort_keys=True), metadata_json=json.dumps(bundle.metadata, sort_keys=True),
             **bundle.uncertainty)


def load_bundle(path: Path, cfg: dict[str, Any]) -> CurveBundle:
    with np.load(path, allow_pickle=False) as v:
        bundle = CurveBundle(
            **{k: v[k].copy() for k in ("times", "samples", "values", "ids", "splits")},
            geometry=json.loads(str(v["geometry_json"])), simulation=json.loads(str(v["simulation_json"])),
            metadata=json.loads(str(v["metadata_json"])),
            uncertainty={k: v[k].copy() for k in ("q_std", "q_se", "ci95_low", "ci95_high") if k in v},
        )
    validate_bundle(bundle, cfg)
    return bundle


class CurveDataset(Dataset):
    def __init__(self, bundle: CurveBundle, config: dict[str, Any], split: str):
        idx = bundle.indices(split)
        topology, numerical = encode_batch(bundle.samples[idx], bundle.times, config)
        curves = np.pad(bundle.values[idx], ((0, 0), (config["time"]["zero_padding"], 0)))
        maximum = curves[:, -1:]
        if np.any(maximum <= 0):
            raise ValueError(f"Nonpositive q(T) in {split}.")
        self.items = {name: torch.tensor(value, dtype=torch.float32) for name, value in
                      {"topology": topology, "numerical": numerical, "curve": curves,
                       "maximum": maximum, "shape": curves / maximum}.items()}

    def __len__(self) -> int:
        return len(self.items["curve"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.items.items()}
