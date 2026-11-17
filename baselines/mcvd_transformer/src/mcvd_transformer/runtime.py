from __future__ import annotations

import platform
import random
import sys
from typing import Any

import numpy as np
import torch


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and (not torch.cuda.is_available() or (device.index or 0) >= torch.cuda.device_count()):
        raise ValueError(f"CUDA device unavailable: {name}")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this Python/PyTorch environment.")
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("Supported devices: cpu, mps, cuda:N, auto.")
    return device


def initialize(seed: int, threads: int) -> None:
    torch.set_num_threads(max(1, int(threads)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment(device: torch.device) -> dict[str, Any]:
    return {"python": sys.version.split()[0], "numpy": np.__version__, "torch": str(torch.__version__),
            "platform": platform.platform(), "device": str(device), "threads": torch.get_num_threads()}
