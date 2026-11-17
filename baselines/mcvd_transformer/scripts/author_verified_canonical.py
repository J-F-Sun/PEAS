#!/usr/bin/env python3
"""Audited canonical training routines; use scripts/train_mcvd_transformer.py at the project root."""
from __future__ import annotations
import copy, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mcvd_transformer.model import MCvDTransformer, KerasStyleMultiHeadSelfAttention
from mcvd_transformer.io import save_json, save_npz
from mcvd_transformer.runtime import choose_device, environment, initialize
from mcvd_transformer.evaluate import canonical_analytic as analytic
from mcvd_transformer.metrics import curve_metrics
VERIFIED = ROOT / "outputs/training/canonical"
COMMIT = "c580cca0ef956866e49fbcc84d38e331bcf24de0"

def metrics(pred, ref, ids):
    m = curve_metrics(pred, ref, ids)
    m["endpoint_relative_mae"] = float(np.mean(np.abs(pred[:, -1] - ref[:, -1]) / ref[:, -1]))
    m["q0_max_abs"] = float(np.max(np.abs(pred[:, 0])))
    m["negative_increment_fraction"] = float(np.mean(np.diff(pred, axis=1) < -1e-9))
    return m
def keras_initialize(model):
    """GlorotUniform including Keras EinsumDense 3-D attention kernels."""
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    for layer in model.modules():
        if isinstance(layer, KerasStyleMultiHeadSelfAttention):
            d, h, k = layer.embed_dim, layer.num_heads, layer.key_dim
            # Keras compute_fans uses product(shape[:-2]) for rank >=3.
            for projection in (layer.query, layer.key, layer.value):
                nn.init.uniform_(projection.weight, -math.sqrt(6 / (d*h+d*k)), math.sqrt(6 / (d*h+d*k)))
            nn.init.uniform_(layer.output.weight, -math.sqrt(6 / (h*k+h*d)), math.sqrt(6 / (h*k+h*d)))

def encode(samples, cfg, rng=None):
    """Author token [class,x,y,z,r,longitude,latitude,a,0,...], Tx origin."""
    center = np.asarray(cfg["geometry"]["rx_center"]) - np.asarray(samples)
    r = np.linalg.norm(center, axis=1)
    longitude = np.mod(np.arctan2(center[:, 1], center[:, 0]), 2*np.pi)
    latitude = np.arcsin(center[:, 2] / r)
    if rng is not None:
        # Author Point.rotate stores the unwrapped added spherical angles.
        longitude = longitude + rng.randint(0, 360, len(r))*np.pi/180
        latitude = latitude + rng.randint(-90, 90, len(r))*np.pi/180
        center = np.column_stack((r*np.cos(latitude)*np.cos(longitude),
                                  r*np.cos(latitude)*np.sin(longitude), r*np.sin(latitude)))
    topology = np.zeros((len(r), 15, 13), dtype=np.float32)
    topology[:, 1:, 0] = -1  # Padding class, NOT a transmitter entity.
    topology[:, 0, 1:4] = center
    topology[:, 0, 4:7] = np.column_stack((r, longitude, latitude))
    topology[:, 0, 7] = cfg["geometry"]["rx_radius"]
    if rng is not None:
        for row in topology:
            rng.shuffle(row[1:])  # Canonical has only identical padding rows.
    numerical = np.zeros((len(r), 7), dtype=np.float32)
    numerical[:, 0] = 1.0
    numerical[:, 1] = cfg["time"]["t_max"] / (cfg["time"]["n_samples"]-1)
    return topology, numerical

def lr_for_epoch(e):
    if e <= 25:
        return 1e-5 + (1e-3-1e-5)*e/25
    if e < 200:
        return 5e-5 + (1e-3-5e-5)*(1-(e-25)/175)
    return 5e-5

def weights_for_epoch(e):
    # AdaptiveLossWeight updates at epoch END (Keras epoch is zero based).
    previous = e-1
    alpha = 1.0 if previous <= 30 else max(1-(previous-30)/200, 0.0)
    return alpha, 1-alpha

def executed_recipe(cfg):
    """Author port overrides the unused legacy schedule fields in input cfg."""
    return {"version":"canonical_author_port_v1","author_commit":COMMIT,"epochs":cfg["training"]["epochs"],"batch_size":64,
            "optimizer":"Adam","betas":[.9,.999],"keras_adam_epsilon":1e-7,"gradient_clip":0.0,
            "learning_rate_schedule":{"index":"zero_based","warmup_epochs":25,"decay_epochs":175,
                "initial_lr":1e-5,"base_lr":1e-3,"min_lr":5e-5},
            "loss_weight_schedule":{"warmup_epochs":30,"decay_epochs":200,"update":"epoch_end"},
            "shape_monotonic_weight":.0001,"amplitude_epsilon":cfg["author_source"].get("amplitude_epsilon",1.0),
            "batch_pattern":([64,64,32] if cfg["author_source"].get("control")=="match_64_64_32_batches_to_n160" else "up to 64; see budget/stream record"),
            "selection_metric":"validation_cir_max_loss_raw","postprocess":"author_released",
            "config_json_role":"input adapter; nested legacy schedules ignored by this standalone trainer"}

def loss_values(output, q, alpha, beta, amplitude_epsilon=1.0):
    maximum = q[:, -1:]
    shape = q/maximum
    shape_loss = ((output["shape"]-shape)**2).mean() + .0001*torch.relu(output["shape"][:, :-1]-output["shape"][:, 1:]).square().mean()
    # metrics.py explicitly K.set_epsilon(1): preserve its MAPE denominator.
    max_loss = ((output["maximum"]-maximum)/maximum.abs().clamp_min(amplitude_epsilon)).square().mean()
    cir_squared = ((output["curve"]-q)/maximum).square().amax(dim=1).mean()
    cir_max = ((output["curve"]-q)/maximum).abs().amax(dim=1).mean()
    total = alpha*(shape_loss+max_loss)+beta*cir_squared
    return total, cir_max

def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def source_postprocess(shape, maximum):
    # No epsilon clamp in the author's released inference routine.
    pdf = torch.relu(shape[:, 1:]-shape[:, :-1])
    padded = torch.nn.functional.pad(pdf, (2, 0))
    smooth = sum(padded[:, j:j+pdf.shape[1]] for j in range(3))/3
    reconstructed = torch.cat((torch.zeros_like(shape[:, :1]), smooth.cumsum(dim=1)), dim=1)+shape[:, :1]
    reconstructed = reconstructed[:, 10:]
    return reconstructed/reconstructed[:, -1:]*maximum

def atomic_torch_save(path, obj):
    temporary = path.with_suffix(".tmp.pt")
    torch.save(obj, temporary)
    temporary.replace(path)

def evaluate(model, cfg, bundle, path):
    # Keras Functional.predict overrides the construction-time training=True.
    # The parity check confirms dropout is disabled during prediction.
    outputs = {}
    ref = analytic(bundle.samples, bundle.times, bundle.geometry)
    for mode in ("original_orientation", "author_rotated"):
        rng = np.random.RandomState(10) if mode == "author_rotated" else None
        initialize(10, 1)
        top, num = encode(bundle.samples, cfg, rng)
        model.eval()
        processed, raw = [], []
        with torch.no_grad():
            for i in range(0, len(top), 64):
                device = next(model.parameters()).device
                output = model(torch.as_tensor(top[i:i+64], device=device), torch.as_tensor(num[i:i+64], device=device))
                processed.append(source_postprocess(output["shape"], output["maximum"]).cpu().numpy())
                raw.append(output["curve"][:, 10:].cpu().numpy())
        p, raw = np.concatenate(processed).astype(np.float64), np.concatenate(raw).astype(np.float64)
        save_npz(path / f"predictions_{mode}.npz", samples=bundle.samples, ids=bundle.ids, splits=bundle.splits,
                 times=bundle.times, prediction=p, raw_prediction=raw, analytic_reference=ref, training_reference=bundle.values)
        outputs[mode] = {}
        for split in ("train", "validation", "test"):
            idx = bundle.indices(split)
            outputs[mode][split] = {}
            for label, pred, target in (("processed_analytic", p, ref), ("raw_analytic", raw, ref),
                                        ("processed_training_source", p, bundle.values)):
                m = metrics(pred[idx], target[idx], bundle.ids[idx])
                m["mean_max_normalized_absolute_error"] = float(np.mean(np.max(np.abs(pred[idx]-target[idx])/target[idx, -1:], axis=1)))
                m["mean_average_normalized_absolute_error"] = float(np.mean(np.abs(pred[idx]-target[idx])/target[idx, -1:]))
                outputs[mode][split][label] = m
    return outputs

def train_one(cfg, bundle, name, device_name, fixed_updates=False, initial_checkpoint=None, matched_examples=False):
    output = VERIFIED / "runs" / name
    result_file = output / "experiment_metrics.json"
    if result_file.exists():
        return json.loads(result_file.read_text())
    output.mkdir(parents=True, exist_ok=True)
    initialize(cfg["seed"], 1)
    device = choose_device(device_name)
    rng = np.random.RandomState(cfg["seed"])
    model = MCvDTransformer(cfg).to(device)
    keras_initialize(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(.9, .999), eps=1e-7)
    ti, vi = bundle.indices("train"), bundle.indices("validation")
    curve = np.pad(bundle.values, ((0, 0), (10, 0)))
    q = torch.tensor(curve, dtype=torch.float32, device=device)
    history, best, best_epoch, best_state, steps, start_epoch = [], math.inf, 0, None, 0, 0
    last = output / "last.pt"
    prior_wall, pending = 0.0, []
    resume_path = last if last.exists() else initial_checkpoint
    total_epochs = cfg["training"]["epochs"]
    if resume_path:
        cp = torch.load(resume_path, map_location="cpu", weights_only=False)
        previous_cfg = copy.deepcopy(cp["config"])
        previous_cfg["training"]["epochs"] = total_epochs
        assert cp["dataset_fingerprint"] == bundle.fingerprint and previous_cfg == cfg
        model.load_state_dict(cp["model_state"])
        optimizer.load_state_dict(cp["optimizer_state"])
        history, best, best_epoch, best_state, steps = cp["history"], cp["best_metric"], cp["best_epoch"], cp["best_state"], cp["steps"]
        start_epoch, prior_wall = cp["epoch"], cp["wall_seconds"]
        pending = cp["fixed_pending"] or []
        rng.set_state(cp["numpy_state"])
        torch.set_rng_state(cp["torch_state"])
        if device.type == "mps":
            torch.mps.set_rng_state(cp["mps_state"])
    start = time.perf_counter()
    save_json(output / "config.json", cfg)
    save_json(output / "executed_recipe.json", executed_recipe(cfg))
    for e in range(start_epoch, total_epochs):
        epoch_start = time.perf_counter()
        order = rng.permutation(ti)
        # Fixed control: three steps/epoch as in n160, cycles through n640.
        if fixed_updates:
            exposure = 160 if matched_examples else 192
            if len(pending) < exposure:
                pending.extend(list(order))
            selected = np.array(pending[:exposure])
            pending = pending[exposure:]
            batches = [selected[i:i+64] for i in range(0, len(selected), 64)]
        else:
            batches = [order[i:i+64] for i in range(0, len(order), 64)]
        model.train(True)
        alpha, beta = weights_for_epoch(e)
        epoch_loss, seen = 0.0, 0
        for indices in batches:
            top, num = encode(bundle.samples[indices], cfg, rng)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.as_tensor(top, device=device), torch.as_tensor(num, device=device))
            loss, _ = loss_values(prediction, q[indices], alpha, beta, cfg["author_source"].get("amplitude_epsilon", 1.0))
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss at {name} epoch {e+1}")
            loss.backward()
            steps += 1
            for group in optimizer.param_groups:
                group["lr"] = lr_for_epoch(e)
                # Keras Adam epsilon is added to uncorrected sqrt(v).
                group["eps"] = 1e-7/math.sqrt(1-.999**steps)
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())*len(indices)
            seen += len(indices)
        # Author validation rotates and shuffles, with inference-mode dropout.
        top, num = encode(bundle.samples[vi], cfg, rng)
        model.eval()
        with torch.no_grad():
            prediction = model(torch.as_tensor(top, device=device), torch.as_tensor(num, device=device))
            val_loss, score = loss_values(prediction, q[vi], alpha, beta, cfg["author_source"].get("amplitude_epsilon", 1.0))
            score = float(score.cpu())
        if not math.isfinite(score):
            raise ValueError("Nonfinite validation score")
        history.append({"epoch": e+1, "steps": steps, "train_loss": epoch_loss/seen,
                        "validation_loss": float(val_loss.cpu()), "validation_cir_max_loss": score,
                        "learning_rate": lr_for_epoch(e), "alpha": alpha, "beta": beta,
                        "epoch_seconds": time.perf_counter()-epoch_start})
        if score < best:
            best, best_epoch, best_state = score, e+1, snapshot(model)
        if (e+1) % 50 == 0 or e == total_epochs-1:
            cp = {"config": cfg, "model_state": snapshot(model), "optimizer_state": optimizer.state_dict(),
                  "best_state": best_state, "best_metric": best, "best_epoch": best_epoch, "history": history,
                  "epoch": e+1, "steps": steps, "dataset_fingerprint": bundle.fingerprint,
                  "numpy_state": rng.get_state(), "torch_state": torch.get_rng_state(),
                  "mps_state": torch.mps.get_rng_state() if device.type == "mps" else None,
                  "wall_seconds": prior_wall+time.perf_counter()-start,
                  "fixed_pending": pending if fixed_updates else None}
            atomic_torch_save(last, cp)
            atomic_torch_save(output / "best.pt", {"config": cfg, "model_state": best_state,
                "epoch": best_epoch, "dataset_fingerprint": bundle.fingerprint, "best_metric": best})
            save_json(output / "history.json", history)
            save_json(output / "status.json", {"completed": e == total_epochs-1, "epoch": e+1, "best_epoch": best_epoch, "best_metric": best})
            print(f"{name}: epoch {e+1}/{total_epochs}, val_MxNAE={score:.5g}, best={best:.5g}, elapsed={prior_wall+time.perf_counter()-start:.1f}s", flush=True)
    wall = prior_wall+time.perf_counter()-start
    result = {"kind": "MCvD-Transformer author port", "seed": cfg["seed"], "n_train": len(ti),
              "label_source": bundle.metadata["data_role"], "budget": "fixed_updates" if fixed_updates else "same_epochs",
              "steps": steps, "epochs": total_epochs, "best_epoch": best_epoch, "wall_seconds": wall,
              "environment": environment(device), "parameters": sum(x.numel() for x in model.parameters()), "author_commit": COMMIT}
    result["recipe"] = cfg["author_source"].get("recipe", "released_code")
    result["run_name"] = name
    result["additional_wall_seconds"] = time.perf_counter()-start
    if total_epochs > 400:
        result["budget"] = "extended_fixed_updates" if fixed_updates else "extended_epochs"
        result["initial_checkpoint"] = str(initial_checkpoint)
    if matched_examples:
        result["budget"] = "matched_updates_and_examples"
        result["training_examples_seen"] = total_epochs*160
    result["last"] = evaluate(model, cfg, bundle, output)
    # Save final predictions separately before evaluating best.
    for path in output.glob("predictions_*.npz"):
        path.rename(path.with_name("last_"+path.name))
    model.load_state_dict(best_state)
    result["best"] = evaluate(model, cfg, bundle, output)
    save_json(result_file, result)
    return result
