from __future__ import annotations
from typing import Any
import torch
from peas.common.modules.features import sphere_features as u_features
from peas.common.geometry import (grad_scalar, laplacian, outer_boundary_normal,
    sample_mixed_interior, sample_outer_boundary, sample_rx_boundary, sample_time)
def u_predict(
    model: torch.nn.Module, t: torch.Tensor, x: torch.Tensor, config: dict[str, Any]
) -> torch.Tensor:
    return model(u_features(t, x, config))

def make_batch(
    config: dict[str, Any],
    device: torch.device,
    adaptive_points: tuple[torch.Tensor, torch.Tensor] | None = None,
    batch_sizes: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    sizes = batch_sizes or config["training"]["batch_sizes"]
    n_pde = int(sizes["pde"])
    n_ic = int(sizes["ic"])
    n_rx = int(sizes["bc_rx"])
    n_outer = int(sizes["bc_outer"])
    pde_x = sample_mixed_interior(n_pde, config, device=device)
    pde_t = sample_time(n_pde, config, device=device)
    if adaptive_points is not None:
        adaptive_x, adaptive_t = adaptive_points
        adaptive_fraction = float(
            config.get("adaptive_sampling", {}).get("batch_fraction", 0.25)
        )
        n_adaptive = min(
            max(int(round(n_pde * adaptive_fraction)), 0), adaptive_x.shape[0]
        )
        if n_adaptive > 0:
            ids = torch.randint(adaptive_x.shape[0], (n_adaptive,), device=device)
            pde_x[:n_adaptive] = adaptive_x[ids]
            pde_t[:n_adaptive] = adaptive_t[ids]
            perm = torch.randperm(n_pde, device=device)
            pde_x = pde_x[perm]
            pde_t = pde_t[perm]
    return {
        "pde_x": pde_x,
        "pde_t": pde_t,
        "ic_x": sample_mixed_interior(n_ic, config, device=device),
        "ic_t": torch.zeros((n_ic, 1), dtype=torch.float32, device=device),
        "bc_rx_x": sample_rx_boundary(n_rx, config, device=device),
        "bc_rx_t": sample_time(n_rx, config, device=device),
        "bc_outer_x": sample_outer_boundary(n_outer, config, device=device),
        "bc_outer_t": sample_time(n_outer, config, device=device),
    }

def pde_residual(
    model: torch.nn.Module,
    q_provider: BisphericalFEMProvider,
    t: torch.Tensor,
    x: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u = u_predict(model, t, x, config)
    du_dt = grad_scalar(u, t, create_graph=True)
    du_dx = grad_scalar(u, x, create_graph=True)
    d2u = laplacian(u, x)
    q_inf, dq_inf = q_provider.value_and_grad(x)
    q_floor = float(config["q_fem"].get("clamp_min", 1.0e-6))
    q_inf = torch.nan_to_num(q_inf, nan=q_floor, posinf=1.0, neginf=q_floor).clamp_min(
        q_floor
    )
    dq_inf = torch.nan_to_num(dq_inf, nan=0.0, posinf=0.0, neginf=0.0)
    drift = 2.0 * dq_inf / q_inf
    drift_clip = float(config["q_fem"].get("drift_clip", 0.0))
    if drift_clip > 0.0:
        drift = drift.clamp(min=-drift_clip, max=drift_clip)
    residual = du_dt - d2u - (drift * du_dx).sum(dim=1, keepdim=True)
    residual_clip = float(config["q_fem"].get("residual_clip", 0.0))
    if residual_clip > 0.0:
        residual = torch.nan_to_num(
            residual, nan=0.0, posinf=residual_clip, neginf=-residual_clip
        )
        residual = residual.clamp(min=-residual_clip, max=residual_clip)
    return residual, u, q_inf

def losses(
    model: torch.nn.Module,
    q_provider: BisphericalFEMProvider,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    residual, u, q_inf = pde_residual(
        model, q_provider, batch["pde_t"], batch["pde_x"], config
    )
    u_ic = u_predict(model, batch["ic_t"], batch["ic_x"], config)
    u_rx = u_predict(model, batch["bc_rx_t"], batch["bc_rx_x"], config)
    x_outer = batch["bc_outer_x"].clone().requires_grad_(True)
    t_outer = batch["bc_outer_t"].clone().requires_grad_(True)
    u_outer = u_predict(model, t_outer, x_outer, config)
    du_dn = (
        grad_scalar(u_outer, x_outer) * outer_boundary_normal(x_outer, config)
    ).sum(dim=1, keepdim=True)
    q = q_inf * u
    return {
        "pde": torch.mean(residual**2),
        "ic": torch.mean(u_ic**2),
        "bc_rx": torch.mean((u_rx - 1.0) ** 2),
        "bc_outer": torch.mean(du_dn**2),
        "range": torch.mean(
            torch.relu(-u) ** 2
            + torch.relu(u - 1.0) ** 2
            + torch.relu(-q) ** 2
            + torch.relu(q - 1.0) ** 2
        ),
    }

def refresh_adaptive_points(
    model: torch.nn.Module,
    q_provider: BisphericalFEMProvider,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    cfg = config.get("adaptive_sampling", {})
    candidate_count = int(cfg.get("candidate_points", 8192))
    selected_count = min(int(cfg.get("selected_points", 1024)), candidate_count)
    chunk_size = int(cfg.get("chunk_size", 1024))
    xs: list[torch.Tensor] = []
    ts: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    was_training = model.training
    model.eval()
    for start in range(0, candidate_count, chunk_size):
        count = min(chunk_size, candidate_count - start)
        x = sample_mixed_interior(count, config, device=device)
        t = sample_time(count, config, device=device)
        residual, _, _ = pde_residual(model, q_provider, t, x, config)
        xs.append(x.detach())
        ts.append(t.detach())
        score_clip = float(
            config["adaptive_sampling"].get(
                "score_clip", config["q_fem"].get("residual_clip", 1.0e6)
            )
        )
        score = torch.abs(residual.detach()).reshape(-1)
        score = torch.nan_to_num(
            score, nan=0.0, posinf=score_clip, neginf=0.0
        ).clamp_max(score_clip)
        scores.append(score)
    if was_training:
        model.train()
    x_all = torch.cat(xs, dim=0)
    t_all = torch.cat(ts, dim=0)
    score_all = torch.cat(scores, dim=0)
    top_ids = torch.topk(score_all, k=selected_count, largest=True).indices
    return x_all[top_ids].detach(), t_all[top_ids].detach()
