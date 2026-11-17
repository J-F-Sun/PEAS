from __future__ import annotations

from typing import Any

import torch
from peas.common.modules.features import sphere_input_scale


def rx_center(config: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(config["domain"]["rx_center"], dtype=torch.float32, device=device).reshape(1, 3)


def outer_center(config: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(config["domain"].get("outer_center", [0.0, 0.0, 0.0]), dtype=torch.float32, device=device).reshape(1, 3)


def rx_radius(config: dict[str, Any]) -> float:
    return float(config["domain"].get("rx_radius", 1.0))


def outer_radius(config: dict[str, Any]) -> float:
    return float(config["domain"]["outer_radius"])


def rand_unit(n: int, *, device: torch.device | None = None) -> torch.Tensor:
    v = torch.randn((n, 3), dtype=torch.float32, device=device)
    return v / v.norm(dim=1, keepdim=True).clamp_min(1.0e-8)


def sample_rx_ball(n: int, config: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    direction = rand_unit(n, device=device)
    radius = outer_radius(config) * torch.rand((n, 1), dtype=torch.float32, device=device).pow(1.0 / 3.0)
    return outer_center(config, device=device) + radius * direction


def sample_near_rx(n: int, config: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    eps = float(config["domain"].get("boundary_eps", 1.0e-4))
    width = float(config.get("sampling", {}).get("near_rx_width", 1.0))
    rr = rx_radius(config)
    max_radius = min(rr + max(width, eps), outer_radius(config) - eps)
    radius = rr + eps + (max_radius - rr - eps) * torch.rand((n, 1), dtype=torch.float32, device=device)
    return rx_center(config, device=device) + radius * rand_unit(n, device=device)


def sample_near_outer(n: int, config: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    eps = float(config["domain"].get("boundary_eps", 1.0e-4))
    width = float(config.get("sampling", {}).get("near_outer_width", 1.0))
    outer = outer_radius(config)
    rr = rx_radius(config)
    min_radius = max(rr + eps, outer - max(width, eps))
    radius = min_radius + (outer - eps - min_radius) * torch.rand((n, 1), dtype=torch.float32, device=device)
    return outer_center(config, device=device) + radius * rand_unit(n, device=device)


def sample_interior(n: int, config: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    eps = float(config["domain"].get("boundary_eps", 1.0e-4))
    rr = rx_radius(config)
    chunks: list[torch.Tensor] = []
    total = 0
    while total < n:
        candidates = sample_rx_ball(max(1024, 2 * (n - total)), config, device=device)
        keep = (candidates - rx_center(config, device=device)).norm(dim=1, keepdim=True) > rr + eps
        accepted = candidates[keep.squeeze(1)]
        if accepted.numel() == 0:
            continue
        chunks.append(accepted)
        total += accepted.shape[0]
    return torch.cat(chunks, dim=0)[:n]


def sample_mixed_interior(n: int, config: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    sampling = config.get("sampling", {})
    weights = sampling.get("interior_mixture")
    if not weights:
        return sample_interior(n, config, device=device)

    near_rx_fraction = float(weights.get("near_rx", 0.0))
    near_outer_fraction = float(weights.get("near_outer", 0.0))
    near_rx_count = max(0, min(n, int(round(n * near_rx_fraction))))
    near_outer_count = max(0, min(n - near_rx_count, int(round(n * near_outer_fraction))))
    volume_count = n - near_rx_count - near_outer_count
    parts = []
    if near_rx_count:
        parts.append(sample_near_rx(near_rx_count, config, device=device))
    if near_outer_count:
        parts.append(sample_near_outer(near_outer_count, config, device=device))
    if volume_count:
        parts.append(sample_interior(volume_count, config, device=device))
    samples = torch.cat(parts, dim=0)
    return samples[torch.randperm(samples.shape[0], device=device)]


def sample_rx_boundary(n: int, config: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    return rx_center(config, device=device) + rx_radius(config) * rand_unit(n, device=device)


def sample_outer_boundary(n: int, config: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    return outer_center(config, device=device) + outer_radius(config) * rand_unit(n, device=device)


def outer_boundary_normal(x: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    return (x - outer_center(config, device=x.device)) / outer_radius(config)


def sample_time(n: int, config: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    t_min = float(config["time"].get("t_min_pde", 1.0e-5))
    t_max = float(config["time"]["t_max"])
    time_sampling = config["time"].get("sampling", {})
    if time_sampling:
        split_t = float(time_sampling.get("split_t", t_max))
        early_fraction = float(time_sampling.get("early_fraction", 1.0))
        early_power = float(time_sampling.get("early_bias_power", 1.0))
        late_power = float(time_sampling.get("late_bias_power", 1.0))
        if t_max > split_t and t_min < split_t and early_fraction < 1.0:
            n_early = max(0, min(n, int(round(n * early_fraction))))
            n_late = n - n_early
            early_u = torch.rand((n_early, 1), dtype=torch.float32, device=device)
            late_u = torch.rand((n_late, 1), dtype=torch.float32, device=device)
            early = t_min + (split_t - t_min) * early_u.pow(early_power)
            late = split_t + (t_max - split_t) * late_u.pow(late_power)
            samples = torch.cat([early, late], dim=0)
            return samples[torch.randperm(samples.shape[0], device=device)]
        bias_power = early_power
    else:
        bias_power = float(config["time"].get("bias_power", 1.0))
    u = torch.rand((n, 1), dtype=torch.float32, device=device)
    return t_min + (t_max - t_min) * u.pow(bias_power)


def q_features(x: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    scale = sphere_input_scale(config)
    feats = [x / scale]
    if bool(config["features"].get("include_radii", True)):
        feats += [
            torch.sum((x - rx_center(config, device=x.device)) ** 2, dim=1, keepdim=True) / (scale**2),
            torch.sum((x - outer_center(config, device=x.device)) ** 2, dim=1, keepdim=True) / (scale**2),
        ]
    return torch.cat(feats, dim=1)


def u_features(t: torch.Tensor, x: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    scale = sphere_input_scale(config)
    t_scale = float(config["features"].get("time_scale", config["time"]["t_max"]))
    rho = (x - rx_center(config, device=x.device)).norm(dim=1, keepdim=True).clamp_min(1.0e-6)
    feats = [t / t_scale, x / scale]
    if bool(config["features"].get("include_tau", True)):
        feats.append(t / (rho**2))
    if bool(config["features"].get("include_radii", True)):
        feats += [
            torch.sum((x - rx_center(config, device=x.device)) ** 2, dim=1, keepdim=True) / (scale**2),
            torch.sum((x - outer_center(config, device=x.device)) ** 2, dim=1, keepdim=True) / (scale**2),
        ]
    if bool(config["features"].get("include_boundary_distances", False)):
        feats += [
            (rho - rx_radius(config)) / scale,
            (outer_radius(config) - (x - outer_center(config, device=x.device)).norm(dim=1, keepdim=True)) / scale,
        ]
    return torch.cat(feats, dim=1)


def grad_scalar(y: torch.Tensor, x: torch.Tensor, *, create_graph: bool = True) -> torch.Tensor:
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=create_graph)[0]


def laplacian(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    grad = grad_scalar(y, x, create_graph=True)
    terms: list[torch.Tensor] = []
    for i in range(x.shape[1]):
        terms.append(
            torch.autograd.grad(
                grad[:, i : i + 1],
                x,
                grad_outputs=torch.ones_like(grad[:, i : i + 1]),
                create_graph=True,
            )[0][:, i : i + 1]
        )
    return torch.stack(terms, dim=0).sum(dim=0)
