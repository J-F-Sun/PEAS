from __future__ import annotations

from peas.common.modules.features import canonical_features
from typing import Any

import torch

from peas.scenarios.canonical.steady import QInfinityProvider, build_q_infinity_provider


def r_to_y(r: torch.Tensor) -> torch.Tensor:
    return 1.0 / r.clamp_min(1.0)


def _q_infty_mode(config: dict[str, Any]) -> str:
    return str(config.get("q_infty", {}).get("mode", "analytic")).lower()


def _fem_y_min(config: dict[str, Any]) -> float:
    outer_radius = float(config.get("q_infty", {}).get("outer_radius", 48.0))
    if outer_radius <= 1.0:
        raise ValueError("q_infty.outer_radius must exceed the unit RX radius.")
    return 1.0 / outer_radius


def _original_features(t: torch.Tensor, y: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    parts = [t, y]
    if bool(config.get("features", {}).get("include_tau", True)):
        parts.append(t * y**2)
    return torch.cat(parts, dim=1)


def predict_u_y(model: torch.nn.Module, t: torch.Tensor, y: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    return model(canonical_features(t, y, config))


def predict_q_y(
    model: torch.nn.Module,
    t: torch.Tensor,
    y: torch.Tensor,
    config: dict[str, Any],
    q_provider: QInfinityProvider | None = None,
) -> torch.Tensor:
    provider = q_provider or build_q_infinity_provider(config)
    q_infty, _ = provider.value_and_dy(y)
    return q_infty * predict_u_y(model, t, y, config)


def predict_q_r(
    model: torch.nn.Module,
    t: torch.Tensor,
    r: torch.Tensor,
    config: dict[str, Any],
    q_provider: QInfinityProvider | None = None,
) -> torch.Tensor:
    return predict_q_y(model, t, r_to_y(r), config, q_provider)


def analytic_q(t: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    safe_t = t.reshape(-1, 1).clamp_min(1.0e-8)
    safe_r = r.reshape(-1, 1).clamp_min(1.0)
    return torch.special.erfc((safe_r - 1.0) / torch.sqrt(4.0 * safe_t)) / safe_r


def _power_time(n: int, t_min: float, t_max: float, power: float, *, device: torch.device | None = None) -> torch.Tensor:
    if t_max <= t_min:
        return torch.full((n, 1), t_min, dtype=torch.float32, device=device)
    u = torch.rand((n, 1), dtype=torch.float32, device=device)
    return t_min + (t_max - t_min) * u.pow(power)


def sample_time(n: int, config: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    cfg = config["time"]
    sampling = cfg.get("sampling", {})
    t_min = float(cfg["t_min_pde"])
    t_max = float(cfg["t_max"])
    split_t = float(sampling.get("split_t", 2.0))
    early_fraction = float(sampling.get("early_fraction", 0.4))
    early_power = float(sampling.get("early_bias_power", 2.0))
    late_power = float(sampling.get("late_bias_power", 1.0))
    if t_max <= split_t:
        return _power_time(n, t_min, t_max, early_power, device=device)
    if split_t <= t_min:
        return _power_time(n, t_min, t_max, late_power, device=device)
    n_early = max(0, min(n, int(round(n * early_fraction))))
    early = _power_time(n_early, t_min, split_t, early_power, device=device)
    late = _power_time(n - n_early, split_t, t_max, late_power, device=device)
    out = torch.cat([early, late], dim=0)
    return out[torch.randperm(out.shape[0], device=device)]


def sample_pde(n: int, config: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    if _q_infty_mode(config) == "fem":
        y_min = _fem_y_min(config)
        y = y_min + (1.0 - y_min) * torch.rand((n, 1), dtype=torch.float32, device=device)
        y = y.clamp_(y_min + 1.0e-5, 1.0 - 1.0e-5)
    else:
        y = torch.rand((n, 1), dtype=torch.float32, device=device).clamp_(1.0e-5, 1.0 - 1.0e-5)
    return torch.cat([sample_time(n, config, device=device), y], dim=1)


def sample_ic(n: int, config: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    if _q_infty_mode(config) == "fem":
        y_min = _fem_y_min(config)
        y = y_min + (1.0 - y_min) * torch.rand((n, 1), dtype=torch.float32, device=device)
        y = y.clamp_(y_min + 1.0e-4, 1.0 - 1.0e-4)
    else:
        y = torch.rand((n, 1), dtype=torch.float32, device=device).clamp_(1.0e-4, 1.0 - 1.0e-4)
    return torch.cat([torch.zeros((n, 1), dtype=torch.float32, device=device), y], dim=1)


def sample_bc_rx(n: int, config: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    return torch.cat([sample_time(n, config, device=device), torch.ones((n, 1), dtype=torch.float32, device=device)], dim=1)


def sample_bc_far(n: int, config: dict[str, Any], *, device: torch.device | None = None) -> torch.Tensor:
    y_min = _fem_y_min(config) if _q_infty_mode(config) == "fem" else 0.0
    y = torch.full((n, 1), y_min, dtype=torch.float32, device=device)
    return torch.cat([sample_time(n, config, device=device), y], dim=1)


def _condition_residual(residual: torch.Tensor, y: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    """Keep the original far-field residual conditioning for both ansatz modes."""
    balance = config.get("loss_balance", {})
    eps = float(balance.get("eps", 5.0e-2))
    denom = y.clamp_min(0.0) ** 3 + eps
    mode = str(balance.get("mode", "full"))
    if mode == "full":
        return residual / denom
    if mode == "sqrt":
        return residual / torch.sqrt(denom)
    raise ValueError(f"Unsupported loss_balance.mode: {mode}")


def pde_residual(
    model: torch.nn.Module,
    t: torch.Tensor,
    y: torch.Tensor,
    config: dict[str, Any],
    q_provider: QInfinityProvider | None = None,
) -> torch.Tensor:
    t = t.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    u = predict_u_y(model, t, y, config)
    du_dt = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u_dy2 = torch.autograd.grad(du_dy, y, grad_outputs=torch.ones_like(du_dy), create_graph=True)[0]
    provider = q_provider or build_q_infinity_provider(config)
    if provider.mode == "analytic":
        residual = du_dt - y**3 * (2.0 * du_dy + y * d2u_dy2)
    else:
        q_infty, dq_infty_dy = provider.value_and_dy(y)
        q_floor = float(config.get("q_infty", {}).get("clamp_min", 1.0e-6))
        q_safe = q_infty.clamp_min(q_floor)
        # q_infty is harmonic.  This is the radial form of
        # u_t - Delta(u) - 2 grad(log q_infty) dot grad(u) = 0.
        residual = du_dt - y**4 * d2u_dy2 - 2.0 * y**4 * (dq_infty_dy / q_safe) * du_dy
    return _condition_residual(residual, y, config)


def make_batch(config: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    sizes = config["training"]["batch_sizes"]
    return {
        "pde": sample_pde(int(sizes["pde"]), config, device=device),
        "ic": sample_ic(int(sizes["ic"]), config, device=device),
        "bc_rx": sample_bc_rx(int(sizes["bc_rx"]), config, device=device),
        "bc_far": sample_bc_far(int(sizes["bc_far"]), config, device=device),
    }


def losses(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    q_provider: QInfinityProvider | None = None,
) -> dict[str, torch.Tensor]:
    pde = batch["pde"]
    ic = batch["ic"]
    bc_rx = batch["bc_rx"]
    bc_far = batch["bc_far"]
    provider = q_provider or build_q_infinity_provider(config)
    result = {
        "pde": torch.mean(pde_residual(model, pde[:, :1], pde[:, 1:], config, provider) ** 2),
        "ic": torch.mean(predict_u_y(model, ic[:, :1], ic[:, 1:], config) ** 2),
        "bc_rx": torch.mean((predict_u_y(model, bc_rx[:, :1], bc_rx[:, 1:], config) - 1.0) ** 2),
    }
    if provider.mode == "analytic":
        result["bc_far"] = torch.mean(predict_u_y(model, bc_far[:, :1], bc_far[:, 1:], config) ** 2)
    else:
        y_outer = bc_far[:, 1:].clone().requires_grad_(True)
        u_outer = predict_u_y(model, bc_far[:, :1], y_outer, config)
        du_dy_outer = torch.autograd.grad(u_outer, y_outer, grad_outputs=torch.ones_like(u_outer), create_graph=True)[0]
        result["bc_far"] = torch.mean(du_dy_outer**2)
    return result
