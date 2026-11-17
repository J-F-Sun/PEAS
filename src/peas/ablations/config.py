"""Build independent run contracts from the retained formal PEAS settings."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from peas.ablations import SCENES, VARIANTS
from peas.common.config import load_config
from peas.common.paths import config_file, data_dir, repository_root


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes():
    root = Path(__file__).resolve().parents[1]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob("*.py"))}


def formal_grid(scene, *, with_reference=False):
    with np.load(repository_root()/"data/reference"/f"{scene}_training_grid.npz", allow_pickle=False) as d:
        return {k:d[k].copy() for k in (("samples","times","ids","reference") if with_reference else ("samples","times","ids"))}


def formal_seeds(scene):
    with np.load(repository_root()/f"data/predictions/ablations/{scene}/no_features.npz") as d:
        return d["seeds"].tolist()


def full_prediction_path(scene, **kwargs):
    return repository_root()/f"data/predictions/comparison/{scene}/peas.npz"


def ablation_root(**kwargs):
    return repository_root()/"outputs/ablations"


def make_config(scene, variant, seed, *, source_id=None, device="auto", without_late_loss=False, split="test"):
    if scene not in ("canonical", "eccentric", "pipe") or variant not in VARIANTS:
        raise ValueError("Unsupported paper scene/variant")
    path=repository_root()/f"configs/ablations/{scene}_{variant}.json"
    c=load_config(path); c["seed"]=int(seed); c["runtime"]["device"]=device
    c["ablation"]["baseline_config_sha256"]=sha256(path)
    grid_name=f"{scene}.npz" if scene=="canonical" else f"{scene}_training_grid.npz"
    c["ablation"]["grid_sha256"]=sha256(repository_root()/"data/reference"/grid_name)
    if variant=="no_reformulation":
        if source_id is None: raise ValueError("Forward training requires --source-id")
        grid=formal_grid(scene)
        if scene=="canonical":
            with np.load(repository_root()/"data"/("validation" if split=="validation" else "reference")/"canonical.npz") as z:grid={k:z[k].copy() for k in ("samples","times","ids")}
        index=np.flatnonzero(grid["ids"]==source_id)
        if len(index)!=1: raise ValueError("Unknown source ID")
        source=grid["samples"][index[0]]; g=c["problem"]["domain"]
        if scene=="canonical":
            r=np.linalg.norm(source); clearance=min(r-1,c["steady_state"]["parameters"]["outer_radius"]-r)
        elif scene=="eccentric":
            clearance=min(np.linalg.norm(source-g["rx_center"])-g["rx_radius"],g["outer_radius"]-np.linalg.norm(source-g["outer_center"]))
        else:
            clearance=min(np.linalg.norm(source-g["rx_center"])-g["rx_radius"],g["pipe_radius"]-np.linalg.norm(source[:2]),source[2]-g["z_min"],g["z_max"]-source[2])
        if clearance<=0: raise ValueError("Source must be in fluid interior")
        c["ablation"].update(source_id=str(source_id),source=source.tolist())
        f=c["ablation"]["forward"]; f["sigma"]=min(f["sigma_max"],clearance/f["clearance_divisor"])
    elif source_id is not None: raise ValueError("Backward models use all source positions")
    return c


def run_directory(c, output_root=None):
    a = c["ablation"]
    p = Path(output_root or c["paths"]["output_root"]) / a["variant"] / c["scenario"] / f"seed_{c['seed']}"
    return p / a["source_id"] if a["source_id"] else p


def smoke_config(c):
    """Two tiny optimizer updates; outputs are explicitly excluded from collection."""
    c = copy.deepcopy(c)
    c["ablation"]["smoke"] = True
    c["training"]["epochs"] = 2
    c["training"]["lr_schedule"] = [{**c["training"]["lr_schedule"][0], "epochs": 2}]
    c["training"]["batch_sizes"] = {k: 8 for k in c["training"]["batch_sizes"]}
    c["training"]["checkpoint_every"] = 1
    c["training"]["lbfgs"] = {"enabled": False}
    c["training"]["validation"] = {"enabled": False}
    c["training"]["adaptive_sampling"] = {"enabled": False}
    return c


def contract(c):
    c = copy.deepcopy(c)
    c.pop("paths", None)
    c["runtime"].pop("device", None)
    # Relocation is permitted; content hashes, rather than host paths, identify assets.
    c["steady_state"]["checkpoint"] = None
    c["evaluation"]["reference"] = None
    return c

