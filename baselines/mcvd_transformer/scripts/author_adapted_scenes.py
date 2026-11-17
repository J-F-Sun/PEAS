#!/usr/bin/env python3
"""Audited bounded-scene training routines; use scripts/train_mcvd_transformer.py at the project root."""
from __future__ import annotations
import copy, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mcvd_transformer.author_scene_adapter import author_config, encode_scene, rotation_matrix, VERSION, COMMIT
from mcvd_transformer.model import MCvDTransformer
from mcvd_transformer.metrics import curve_metrics
from mcvd_transformer.io import save_json, save_npz
from mcvd_transformer.runtime import initialize, choose_device, environment
from author_verified_canonical import keras_initialize, lr_for_epoch, weights_for_epoch, loss_values, source_postprocess, snapshot, atomic_torch_save
OUT = ROOT / "outputs/training/bounded"
ROTATIONS = ((0,0),(45,30),(90,-45),(135,60),(180,0),(225,-30),(270,45),(315,-60))

def physical(pred):
    return {"q0_max_abs": float(np.abs(pred[:, 0]).max()),
            "negative_value_fraction": float(np.mean(pred < -1e-9)),
            "above_one_fraction": float(np.mean(pred > 1 + 1e-9)),
            "nonmonotone_curves": int(np.any(np.diff(pred, axis=1) < -1e-9, axis=1).sum())}

def metric(pred, ref, ids):
    m = curve_metrics(pred, ref, ids)
    m["mean_max_normalized_absolute_error"] = float(np.mean(np.max(np.abs(pred-ref) / ref[:, -1:], axis=1)))
    m["endpoint_relative_mae"] = float(np.mean(np.abs(pred[:, -1]-ref[:, -1])/ref[:, -1]))
    m["physical"] = physical(pred)
    return m

def paper_text_postprocess(shape, maximum, recursive=False):
    """Frozen-weight diagnostic, not a claim of unique paper interpretation.

    Crop padding, set initial estimated shape to 0, implement printed adjacent
    max or intended recursive max, then causal 3-point average and zero/end
    anchoring. Paper does not specify averaging alignment or padding order.
    """
    s = shape[:, 10:].copy()
    s[:, 0] = 0
    if recursive:
        sp = np.maximum.accumulate(s, axis=1)
    else:
        sp = np.column_stack((np.zeros(len(s)), np.maximum(s[:, 1:], s[:, :-1])))
    padded = np.pad(sp, ((0, 0), (2, 0)))
    smooth = sum(padded[:, i:i+s.shape[1]] for i in range(3))/3
    denominator = smooth[:, -1:] - smooth[:, :1]
    valid = np.abs(denominator[:, 0]) > 1e-12
    pred = np.full_like(smooth, np.nan)
    pred[valid] = (smooth[valid]-smooth[valid, :1])/denominator[valid] * maximum[valid]
    return pred, valid

@torch.no_grad()
def infer(model, samples, cfg, rotation=None):
    model.eval()
    top, num = encode_scene(samples, cfg, rotation=rotation)
    data = {k: [] for k in ("raw", "processed", "shape", "maximum")}
    device = next(model.parameters()).device
    for i in range(0, len(top), 64):
        out = model(torch.as_tensor(top[i:i+64], device=device), torch.as_tensor(num[i:i+64], device=device))
        data["raw"].append(out["curve"][:, 10:].cpu().numpy().astype(np.float64))
        data["processed"].append(source_postprocess(out["shape"], out["maximum"]).cpu().numpy().astype(np.float64))
        data["shape"].append(out["shape"].cpu().numpy().astype(np.float64))
        data["maximum"].append(out["maximum"].cpu().numpy().astype(np.float64))
    data = {k: np.concatenate(v) for k, v in data.items()}
    if not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError("Nonfinite raw or released postprocessing inference")
    return data

def evaluate(model, cfg, bundle, folder, checkpoint):
    data = infer(model, bundle.samples, cfg)
    result = {s: {k: metric(data[k][idx], bundle.values[idx], bundle.ids[idx]) for k in ("raw", "processed")}
              for s, idx in ((s, bundle.indices(s)) for s in ("train", "validation", "test"))}
    supplementary = {}
    for key, recursive in (("paper_adjacent_max", False), ("paper_recursive_max", True)):
        pred, valid = paper_text_postprocess(data["shape"], data["maximum"], recursive)
        data[key], data[key + "_valid"] = pred, valid
        supplementary[key] = {}
        for s in ("train", "validation", "test"):
            idx = bundle.indices(s)
            invalid = bundle.ids[idx][~valid[idx]].tolist()
            supplementary[key][s] = {"invalid_position_ids": invalid,
                "metrics": metric(pred[idx], bundle.values[idx], bundle.ids[idx]) if not invalid else None}
    save_npz(folder / f"{checkpoint}_predictions.npz", samples=bundle.samples, ids=bundle.ids,
             splits=bundle.splits, times=bundle.times, reference_ams=bundle.values, **data)
    test = bundle.indices("test")
    rotations, rotated_raw, rotated_post = [], [], []
    for theta, phi in ROTATIONS:
        p = infer(model, bundle.samples[test], cfg, rotation_matrix(theta, phi))
        rotated_raw.append(p["raw"]); rotated_post.append(p["processed"])
        rotations.append({"theta": theta, "phi": phi,
                          **{k: metric(p[k], bundle.values[test], bundle.ids[test]) for k in ("raw", "processed")}})
    save_npz(folder / f"{checkpoint}_rotation_predictions.npz", ids=bundle.ids[test], samples=bundle.samples[test],
             times=bundle.times, reference_ams=bundle.values[test], angles=np.asarray(ROTATIONS),
             raw=np.stack(rotated_raw), processed=np.stack(rotated_post))
    radial = None
    if cfg["scenario"] == "concentric":
        samples = np.tile(np.asarray(bundle.geometry["rx_center"]), (len(test), 1))
        samples[:, 0] += np.linalg.norm(bundle.samples[test]-bundle.geometry["rx_center"], axis=1)
        mapped = infer(model, samples, cfg)
        radial = {k: metric(mapped[k], bundle.values[test], bundle.ids[test]) for k in ("raw", "processed")}
        save_npz(folder / f"{checkpoint}_radial_diagnostic.npz", ids=bundle.ids[test], samples=samples,
                 original_samples=bundle.samples[test], reference_ams=bundle.values[test], times=bundle.times, **mapped)
    return {"splits": result, "paper_text_diagnostics": supplementary, "rotations": rotations, "radial_diagnostic": radial}

def rng_to_json(state):
    return [state[0], state[1].tolist(), state[2], state[3], state[4]]

def rng_from_json(state):
    return (state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4])

def train_one(base, bundle, seed, budget, device_name):
    name = f"{base['scenario']}_seed{seed}"
    folder = OUT / "runs" / name / str(budget)
    result_path = folder / "results.json"
    cfg = author_config(base, seed, budget)
    if result_path.exists():
        result = read_json(result_path)
        if result["config"] != cfg or result["dataset_fingerprint"] != bundle.fingerprint:
            raise ValueError("Existing experiment differs from requested configuration")
        return result
    folder.mkdir(parents=True, exist_ok=True)
    initialize(seed, 1)
    device = choose_device(device_name)
    rng = np.random.RandomState(seed)
    model = MCvDTransformer(cfg).to(device)
    keras_initialize(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(.9, .999), eps=1e-7)
    ti, vi = bundle.indices("train"), bundle.indices("validation")
    # Test labels do not enter optimization or selection tensors.
    qt = torch.tensor(np.pad(bundle.values[ti], ((0, 0), (10, 0))), dtype=torch.float32, device=device)
    qv = torch.tensor(np.pad(bundle.values[vi], ((0, 0), (10, 0))), dtype=torch.float32, device=device)
    best, best_epoch, best_state = math.inf, 0, None
    history, steps, start_epoch, prior_wall = [], 0, 0, 0.
    resume = folder / "last.pt"
    if not resume.exists() and budget > 400:
        resume = OUT / "runs" / name / "400/last.pt"
    if resume.exists():
        cp = torch.load(resume, map_location="cpu", weights_only=True)
        previous = copy.deepcopy(cp["config"])
        previous["training"]["epochs"] = budget
        assert cp["framework"] == VERSION and previous == cfg and cp["dataset_fingerprint"] == bundle.fingerprint
        model.load_state_dict(cp["model_state"])
        optimizer.load_state_dict(cp["optimizer_state"])
        history, steps, start_epoch = cp["history"], cp["steps"], cp["epoch"]
        best, best_epoch, best_state = cp["best_metric"], cp["best_epoch"], cp["best_state"]
        rng.set_state(rng_from_json(cp["numpy_state"]))
        torch.set_rng_state(cp["torch_state"])
        if device.type == "mps":
            torch.mps.set_rng_state(cp["mps_state"])
        # Stage wall time includes only new training, excluding the 400 phase.
        prior_wall = cp["stage_wall_seconds"] if start_epoch > 400 or budget == 400 else 0.
    start = time.perf_counter()
    save_json(folder / "config.json", cfg)
    for e in range(start_epoch, budget):
        began = time.perf_counter()
        order = rng.permutation(len(ti))
        model.train()
        alpha, beta = weights_for_epoch(e)
        total, seen = 0., 0
        for indices in (order[i:i+64] for i in range(0, len(order), 64)):
            top, num = encode_scene(bundle.samples[ti[indices]], cfg, rng)
            optimizer.zero_grad(set_to_none=True)
            out = model(torch.as_tensor(top, device=device), torch.as_tensor(num, device=device))
            loss, _ = loss_values(out, qt[indices], alpha, beta, 1.)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss {name}, epoch {e+1}")
            loss.backward()
            steps += 1
            for group in optimizer.param_groups:
                group["lr"] = lr_for_epoch(e)
                group["eps"] = 1e-7/math.sqrt(1-.999**steps)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(indices)
            seen += len(indices)
        model.eval()
        top, num = encode_scene(bundle.samples[vi], cfg, rng)
        with torch.no_grad():
            out = model(torch.as_tensor(top, device=device), torch.as_tensor(num, device=device))
            vl, vs = loss_values(out, qv, alpha, beta, 1.)
        score = float(vs.cpu())
        if not math.isfinite(score):
            raise ValueError("Nonfinite validation score")
        history.append({"epoch": e+1, "steps": steps, "train_loss": total/seen,
                        "validation_loss": float(vl.cpu()), "validation_cir_max_loss": score,
                        "learning_rate": lr_for_epoch(e), "alpha": alpha, "beta": beta,
                        "epoch_seconds": time.perf_counter()-began})
        if score < best:
            best, best_epoch, best_state = score, e+1, snapshot(model)
        if (e+1) % 50 == 0 or e+1 == budget:
            cp = {"framework": VERSION, "config": cfg, "model_state": snapshot(model),
                  "optimizer_state": optimizer.state_dict(), "best_state": best_state,
                  "history": history, "steps": steps, "epoch": e+1, "best_epoch": best_epoch,
                  "best_metric": best, "dataset_fingerprint": bundle.fingerprint,
                  "numpy_state": rng_to_json(rng.get_state()), "torch_state": torch.get_rng_state(),
                  "mps_state": torch.mps.get_rng_state() if device.type == "mps" else None,
                  "stage_wall_seconds": prior_wall+time.perf_counter()-start,
                  "times": torch.from_numpy(bundle.times.copy())}
            atomic_torch_save(folder / "last.pt", cp)
            atomic_torch_save(folder / "best.pt", {"framework": VERSION, "config": cfg,
                "model_state": best_state, "epoch": best_epoch, "best_metric": best,
                "dataset_fingerprint": bundle.fingerprint, "times": torch.from_numpy(bundle.times.copy())})
            save_json(folder / "history.json", history)
            save_json(folder / "status.json", {"completed": e+1 == budget, "epoch": e+1,
                "best_epoch": best_epoch, "validation_best_MxNAE": best})
            print(f"{name}/{budget}: epoch {e+1}, val={score:.4g}, best={best:.4g}, stage={prior_wall+time.perf_counter()-start:.1f}s", flush=True)
    result = {"scene": base["scenario"], "seed": seed, "epochs": budget, "steps": steps,
              "best_epoch": best_epoch, "validation_best_MxNAE": best, "config": cfg,
              "dataset_fingerprint": bundle.fingerprint, "environment": environment(device),
              "stage_wall_seconds": prior_wall+time.perf_counter()-start,
              "parameters": sum(p.numel() for p in model.parameters())}
    result["last"] = evaluate(model, cfg, bundle, folder, "last")
    model.load_state_dict(best_state)
    result["best"] = evaluate(model, cfg, bundle, folder, "best")
    save_json(result_path, result)
    return result
