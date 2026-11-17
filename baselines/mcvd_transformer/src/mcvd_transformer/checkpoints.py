from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def rng_state(generator: torch.Generator) -> dict[str, Any]:
    kind, keys, position, has_gauss, cached = np.random.get_state()
    result = {"python": random.getstate(), "numpy": [kind, keys.tolist(), position, has_gauss, cached],
              "torch": torch.get_rng_state(), "shuffle": generator.get_state()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch.mps, "get_rng_state") and torch.backends.mps.is_available():
        result["mps"] = torch.mps.get_rng_state()
    return result


def restore_rng(state: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(state["python"])
    kind, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((kind, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["shuffle"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        for i, rng in enumerate(state["cuda"]):
            if i < torch.cuda.device_count():
                torch.cuda.set_rng_state(rng.cpu(), i)
    if "mps" in state and hasattr(torch.mps, "set_rng_state") and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    # Files use only tensors and built-in containers; NumPy RNG arrays become lists.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != 1 or checkpoint.get("framework") != "mcvd_transformer_unified":
        raise ValueError("Not a unified MCvD Transformer checkpoint.")
    return checkpoint


def resume_contract(config: dict[str, Any]) -> dict[str, Any]:
    training = {k: v for k, v in config["training"].items() if k not in ("epochs", "log_interval")}
    return {k: config[k] for k in ("scenario", "geometry", "seed", "profile", "model", "time")} | {"training": training}
