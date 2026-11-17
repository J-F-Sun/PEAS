"""Portable, aligned numerical archives used by all three paper figures."""
from pathlib import Path
import json
import numpy as np
from peas.common.paths import repository_root
SCENES=("canonical","eccentric","pipe")
METHODS={"PEAS":"peas","AMS (mean of 128)":"monte_carlo","MCvD-Transformer":"mcvd_transformer","Frozen-PINN":"frozen_pinn"}
VARIANTS=("full","no_reformulation","no_steady","no_features")

def read(scene, method, group="comparison", *, version="current"):
    if scene not in SCENES: raise ValueError(scene)
    if version not in ("current", "previous"): raise ValueError(version)
    if group=="ablations" and method=="no_reformulation" and version=="previous": group="ablations_previous"
    name=METHODS[method] if group=="comparison" else method
    root=repository_root()
    with np.load(root/f"data/predictions/{group}/{scene}/{name}.npz",allow_pickle=False) as z:
        data={k:z[k].copy() for k in z.files}
    with np.load(root/f"data/reference/{scene}.npz",allow_pickle=False) as z:
        for key in ("ids","samples","times","reference"):np.testing.assert_array_equal(data[key],z[key])
    if not np.isfinite(data["prediction"]).all(): raise ValueError("Nonfinite prediction")
    return data

def errors(prediction,reference):
    p=np.asarray(prediction,dtype=np.float64);r=np.asarray(reference,dtype=np.float64)
    if p.ndim==2:p=p[None]
    if p.ndim!=3 or p.shape[1:]!=r.shape:raise ValueError("Expected run × position × time and matching reference")
    den=np.linalg.norm(r,axis=1)
    if np.any(den<=0):raise ValueError("Zero reference norm")
    return 100*np.linalg.norm(p-r,axis=2)/den

def positions(scene):
    root=repository_root()
    config=json.loads((root/f"configs/scenarios/{scene}.json").read_text())
    with np.load(root/f"data/reference/{scene}.npz") as z:
        ids=z["ids"];dist=np.linalg.norm(z["samples"]-config["domain"].get("rx_center",[0,0,0]),axis=1)
    order=np.lexsort((ids,dist))
    return ids[order],dist[order],order
