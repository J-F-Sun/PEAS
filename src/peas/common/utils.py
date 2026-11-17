from __future__ import annotations
import json
import random
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

def timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def grad_scalar(y, x, *, create_graph=True):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=create_graph)[0]

def weighted_total(parts, weights):
    total = torch.zeros((), dtype=next(iter(parts.values())).dtype, device=next(iter(parts.values())).device)
    for key, value in parts.items():
        total = total + float(weights.get(key, 1.0)) * value
    return total

def select_device(runtime):
    torch.set_num_threads(int(runtime.get('threads', 1)))
    requested = runtime.get('device', 'auto')
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if str(requested).startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('Requested CUDA device is unavailable')
    return torch.device(requested)
