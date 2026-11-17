"""Independent artifacts, full Adam continuation, and the formal optimizer budgets."""
from __future__ import annotations

import copy
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from peas.ablations.config import code_hashes, contract, run_directory, sha256, smoke_config
from peas.ablations.model import AblationModel
from peas.common.checkpoints import capture_rng, portable_config, restore_rng
from peas.common.config import save_config
from peas.common.training import run_lbfgs, stages
from peas.common.utils import save_json, select_device, set_seed, weighted_total

FORMAT = "peas_ablation_v1"


def save_checkpoint(path, model, state, provenance):
    path = Path(path)
    payload = {"format": FORMAT, "config": portable_config(model.config, path.parent),
               "model_state_dict": model.transient.state_dict(),
               "training_state": state, "provenance": provenance}
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path, device="cpu"):
    path = Path(path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise ValueError("This is not an ablation checkpoint")
    c = copy.deepcopy(payload["config"])
    for section, key in (("paths", "output_root"), ("steady_state", "checkpoint"), ("evaluation", "reference")):
        value = c[section].get(key)
        if value is not None and not Path(value).is_absolute():
            c[section][key] = str((path.parent / value).resolve())
    c["runtime"]["device"] = str(device)
    device = select_device(c["runtime"])
    expected = payload["provenance"].get("steady_sha256")
    if expected and sha256(c["steady_state"]["checkpoint"]) != expected:
        raise ValueError("The run's steady asset checksum changed")
    model = AblationModel(c, device)
    model.transient.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload


def retain_shared_steady(c, output_root):
    """Copy a FEM asset once per experiment tree, not once per source model."""
    source = c["steady_state"].get("checkpoint")
    if source is None:
        return None
    digest = sha256(source)
    target = Path(output_root) / "assets" / f"{digest}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        shutil.copy2(source, tmp)
        tmp.replace(target)
    if sha256(target) != digest:
        raise ValueError("Shared FEM asset checksum mismatch")
    c["steady_state"]["checkpoint"] = str(target.resolve())
    return digest


def train(config, *, output_root=None, resume=False, max_steps=None, smoke=False):
    c = smoke_config(config) if smoke else copy.deepcopy(config)
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    root = Path(output_root or c["paths"]["output_root"]).resolve()
    if c["ablation"]["smoke"]:
        root = root / "_smoke"
    run = run_directory(c, root)
    device = select_device(c["runtime"])
    set_seed(c["seed"])
    software = code_hashes()
    optimizer = None
    stage_index = None
    epoch = 0
    history = []
    elapsed_before = 0.
    sphere = c["scenario"] in ("concentric", "eccentric")
    skipped = 0
    last_good = None
    if resume:
        model, payload = load_checkpoint(run / "last.pt", device)
        if contract(c) != contract(model.config):
            raise ValueError("Resume configuration differs from the saved run contract")
        if software != payload["provenance"]["code_sha256"]:
            raise ValueError("Python sources changed since this run; use a new --output-root")
        state = payload["training_state"]
        if state["phase"] == "complete":
            return json.loads((run / "summary.json").read_text())
        c = model.config
        provenance = payload["provenance"]
        epoch, stage_index = state["epoch"], state["stage"]
        history, elapsed_before = state["history"], state["elapsed_seconds"]
        skipped = state.get("skipped_nonfinite", 0)
        last_good = state.get("last_good")
        model.training.load_state_dict(state["physics"])
        if state["optimizer"] is not None:
            optimizer = torch.optim.Adam(model.transient.parameters(), lr=stages(c)[stage_index]["lr"])
            optimizer.load_state_dict(state["optimizer"])
            for values in optimizer.state.values():
                for key, value in values.items():
                    if torch.is_tensor(value):
                        values[key] = value.to(device)
        restore_rng(state["rng"])
    else:
        run.mkdir(parents=True, exist_ok=False)
        digest = retain_shared_steady(c, root)
        model = AblationModel(c, device)
        provenance = {"code_sha256": software, "steady_sha256": digest,
                      "baseline_config_sha256": c["ablation"]["baseline_config_sha256"],
                      "grid_sha256": c["ablation"]["grid_sha256"]}
        save_config(c, run / "config.json")
        save_json(run / "provenance.json", provenance)
        save_json(run / "environment.json", {"python": platform.python_version(),
                  "torch": str(torch.__version__), "numpy": np.__version__,
                  "device": str(device), "cuda": torch.version.cuda,
                  "parameters": sum(p.numel() for p in model.transient.parameters()),
                  "smoke": c["ablation"]["smoke"],
                  "source_q": model.source_q.item() if model.source_q is not None else None})
    start = time.perf_counter()
    tr = c["training"]
    model.transient.train()
    if sphere:
        last_good = ({k: v.to(device) for k, v in last_good.items()} if last_good else
                     {k: v.detach().clone() for k, v in model.transient.state_dict().items()})

    def save(phase):
        state = {"phase": phase, "epoch": epoch, "stage": stage_index,
                 "optimizer": optimizer.state_dict() if optimizer else None,
                 "rng": capture_rng(), "physics": model.training.state_dict(),
                 "history": history, "elapsed_seconds": elapsed_before + time.perf_counter() - start}
        if sphere:
            state.update(skipped_nonfinite=skipped, last_good=last_good)
        save_checkpoint(run / "last.pt", model, state, provenance)
        save_json(run / "history.json", history)
        save_json(run / "status.json", {"phase": phase, "epoch": epoch,
                                       "smoke": c["ablation"]["smoke"]})

    # Also allows continuation if the very first stage is interrupted.
    if not resume:
        save("adam")
    completed = 0
    offset = 0
    for i, stage in enumerate(stages(c)):
        end = offset + stage["epochs"]
        if epoch >= end:
            offset = end
            continue
        if stage_index != i or optimizer is None:
            if optimizer is None or tr.get("adam_reset_each_stage"):
                optimizer = torch.optim.Adam(model.transient.parameters(), lr=stage["lr"])
            else:
                for group in optimizer.param_groups:
                    group["lr"] = stage["lr"]
            stage_index = i
        while epoch < end:
            model.training.set_epoch(epoch)
            model.training.maybe_refresh_adaptive(model, device, epoch + 1)
            optimizer.zero_grad(set_to_none=True)
            parts = model.training.losses(model, model.training.make_batch(device))
            total = weighted_total(parts, stage["loss_weights"])
            success = bool(torch.isfinite(total))
            if success:
                total.backward()
                if tr.get("grad_clip_norm", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.transient.parameters(), tr["grad_clip_norm"])
                success = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.transient.parameters())
            if success:
                optimizer.step()
                success = all(bool(torch.isfinite(p).all()) for p in model.transient.parameters())
            if not success:
                if not sphere:
                    raise FloatingPointError("Nonfinite Adam loss, gradient or update; resume last.pt")
                skipped += 1
                if skipped > tr.get("max_nonfinite_skips", 100):
                    raise FloatingPointError("Too many nonfinite Adam steps; resume last.pt")
                model.transient.load_state_dict(last_good)
                lr = max(optimizer.param_groups[0]["lr"] * tr.get("nonfinite_lr_decay", .5), tr.get("min_lr", 1e-7))
                optimizer = torch.optim.Adam(model.transient.parameters(), lr=lr)
                model.training.adaptive_points = None
                print(f"[{c['scenario']}/{c['ablation']['variant']}] recovered nonfinite Adam step "
                      f"epoch={epoch + 1} skipped={skipped} lr={lr:.6g}", flush=True)
            elif sphere:
                last_good = {k: v.detach().clone() for k, v in model.transient.state_dict().items()}
            epoch += 1
            completed += 1
            if not success:
                history.append({"epoch": epoch, "phase": "adam", "skipped_nonfinite": True})
            if success and (epoch == 1 or epoch == tr["epochs"] or epoch % tr["log_every"] == 0):
                row = {"epoch": epoch, "phase": "adam", "stage": i,
                       "lr": optimizer.param_groups[0]["lr"], "total": float(total.detach().cpu()),
                       **{k: float(v.detach().cpu()) for k, v in parts.items()}}
                history.append(row)
                print(f"[{c['scenario']}/{c['ablation']['variant']}/seed_{c['seed']}"
                      f"/{c['ablation']['source_id'] or 'all_sources'}] "
                      f"adam={epoch}/{tr['epochs']} loss={row['total']:.6g}", flush=True)
            phase = "lbfgs_pending" if epoch == tr["epochs"] and tr["lbfgs"]["enabled"] else "adam"
            if epoch % tr.get("checkpoint_every", 250) == 0:
                save(phase)
            if max_steps is not None and completed >= max_steps and (epoch < tr["epochs"] or phase == "lbfgs_pending"):
                save(phase)
                return {"phase": phase, "epoch": epoch, "run_dir": str(run)}
        offset = end
    calls, best = 0, None
    if tr["lbfgs"]["enabled"]:
        # If interrupted here, resume restarts L-BFGS from this saved boundary.
        save("lbfgs_pending")
        model.training.set_epoch(epoch - 1)
        calls, best = run_lbfgs(model, device, history, epoch)
    save_checkpoint(run / "model.pt", model, {"phase": "complete", "epoch": epoch}, provenance)
    summary = {"phase": "complete", "epoch": epoch, "run_dir": str(run),
               "variant": c["ablation"]["variant"], "scene": c["scenario"], "seed": c["seed"],
               "source_id": c["ablation"]["source_id"], "smoke": c["ablation"]["smoke"],
               "elapsed_seconds": elapsed_before + time.perf_counter() - start,
               "lbfgs_calls": calls, "best_physics_validation_loss": best,
               "parameters": sum(p.numel() for p in model.transient.parameters())}
    if sphere:
        summary["skipped_nonfinite"] = skipped
    save_json(run / "summary.json", summary)
    save("complete")
    return summary

