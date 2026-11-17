from __future__ import annotations
from typing import Any
import torch
from pathlib import Path
import json
import warnings
import numpy as np
from peas.scenarios.pipe.steady import Pipe3DGmshFEMProvider
from peas.common.modules.features import pipe_features
from peas.common.utils import grad_scalar, set_seed
def validate_qinf_checkpoint(checkpoint: Path, config: dict[str, Any]) -> None:
    payload = np.load(checkpoint)
    if "config" not in payload:
        raise ValueError(f"3D FEM q_infty checkpoint has no embedded config: {checkpoint}")
    fem_config = json.loads(str(payload["config"]))
    pinn_domain = config["domain"]
    fem_domain = fem_config["domain"]
    mismatches: list[str] = []

    def compare_scalar(label: str, pinn_value: Any, fem_value: Any) -> None:
        if not np.isclose(float(pinn_value), float(fem_value), rtol=1.0e-10, atol=1.0e-10):
            mismatches.append(f"{label}: PINN={pinn_value}, FEM={fem_value}")

    for key in ["pipe_radius", "z_min", "z_max", "rx_radius"]:
        compare_scalar(f"domain.{key}", pinn_domain[key], fem_domain[key])
    if not np.allclose(np.asarray(pinn_domain["rx_center"], dtype=float), np.asarray(fem_domain["rx_center"], dtype=float)):
        mismatches.append(f"domain.rx_center: PINN={pinn_domain['rx_center']}, FEM={fem_domain['rx_center']}")
    compare_scalar("boundary.kappa", config["boundary"]["kappa"], fem_config["boundary"]["kappa"])
    if mismatches:
        joined = "\n  - " + "\n  - ".join(mismatches)
        raise ValueError(
            "PINN domain does not match the reused FEM q_infty checkpoint. "
            "Update config/pipe.json, regenerate q_infty, and point steady_state.checkpoint "
            f"to the new file before training:{joined}"
        )

class TorchFEMProvider:
    def __init__(self, checkpoint: Path, device: torch.device, gradient_method: str, locator_candidates: int) -> None:
        self.provider = Pipe3DGmshFEMProvider(checkpoint, gradient_method=gradient_method)
        self.device = device
        self.locator_candidates = locator_candidates
        self.fallback_points = 0

    def value_and_grad(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, gradient, fallback = self.provider.value_and_grad(xyz.detach().cpu().numpy(), candidates=self.locator_candidates)
        if np.any(fallback):
            count = int(np.count_nonzero(fallback))
            self.fallback_points += count
            warnings.warn(
                "FEM locator used nearest-node interpolation for "
                f"{count} training point(s); cumulative fallback points={self.fallback_points}.",
                RuntimeWarning,
                stacklevel=2,
            )
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(gradient)):
            raise FloatingPointError("FEM provider returned non-finite q_infty values or recovered gradients.")
        return (
            torch.as_tensor(q, dtype=torch.float32, device=self.device),
            torch.as_tensor(gradient, dtype=torch.float32, device=self.device),
        )

def domain_values(config: dict[str, Any]) -> tuple[float, float, float, torch.Tensor, float]:
    domain = config["domain"]
    return (
        float(domain["pipe_radius"]),
        float(domain["z_min"]),
        float(domain["z_max"]),
        torch.tensor(domain["rx_center"], dtype=torch.float32),
        float(domain["rx_radius"]),
    )

def sample_time(n: int, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    cfg = config["time"]
    t_min = float(cfg.get("t_min_pde", 1.0e-5))
    split = float(cfg["sampling"]["split_t"])
    t_max = float(cfg["t_max"])
    if not 0.0 < t_min < split < t_max:
        raise ValueError("time sampling requires 0 < t_min_pde < split_t < t_max.")
    fraction = float(cfg["sampling"]["early_fraction"])
    early = torch.rand(n, device=device) < fraction
    result = torch.empty((n, 1), dtype=torch.float32, device=device)
    result[early] = t_min + (split - t_min) * torch.rand((int(early.sum()), 1), device=device).pow(
        float(cfg["sampling"]["early_bias_power"])
    )
    result[~early] = split + (t_max - split) * torch.rand((int((~early).sum()), 1), device=device).pow(
        float(cfg["sampling"]["late_bias_power"])
    )
    return result

def sample_cylinder(n: int, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    radius, z_min, z_max, _, _ = domain_values(config)
    inset = float(config["sampling"].get("fem_inset", 0.0))
    rho = (radius - inset) * torch.sqrt(torch.rand((n, 1), device=device))
    theta = 2.0 * torch.pi * torch.rand((n, 1), device=device)
    z = z_min + inset + (z_max - z_min - 2.0 * inset) * torch.rand((n, 1), device=device)
    return torch.cat([rho * torch.cos(theta), rho * torch.sin(theta), z], dim=1)

def outside_rx(xyz: torch.Tensor, config: dict[str, Any], *, margin: float = 0.0) -> torch.Tensor:
    _, _, _, center, rx_radius = domain_values(config)
    center = center.to(xyz.device)
    return torch.linalg.vector_norm(xyz - center, dim=1) > rx_radius + margin

def reject_rx(n: int, sampler: Any, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    eps = float(config["domain"].get("boundary_eps", 1.0e-4))
    chunks: list[torch.Tensor] = []
    total = 0
    while total < n:
        xyz = sampler(max(1024, 2 * (n - total)))
        keep = outside_rx(xyz, config, margin=eps)
        if keep.any():
            chunks.append(xyz[keep])
            total += int(keep.sum())
    return torch.cat(chunks, dim=0)[:n]

def sample_near_rx(
    n: int,
    config: dict[str, Any],
    device: torch.device,
    *,
    width: float | None = None,
    surface_margin: float | None = None,
) -> torch.Tensor:
    _, _, _, center, rx_radius = domain_values(config)
    width = float(config["sampling"]["near_rx_width"] if width is None else width)
    margin = float(config["domain"].get("boundary_eps", 1.0e-4) if surface_margin is None else surface_margin)
    if width <= 0.0 or margin <= 0.0:
        raise ValueError("RX neighborhood width and surface margin must be positive.")
    direction = torch.randn((n, 3), device=device)
    direction = direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    radius = rx_radius + margin + width * torch.rand((n, 1), device=device)
    return center.to(device) + radius * direction

def sample_near_wall(n: int, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    radius, z_min, z_max, _, _ = domain_values(config)
    width = float(config["sampling"]["near_wall_width"])
    inset = float(config["sampling"].get("fem_inset", 0.0))

    def sampler(m: int) -> torch.Tensor:
        rho = radius - inset - width * torch.rand((m, 1), device=device)
        theta = 2.0 * torch.pi * torch.rand((m, 1), device=device)
        z = z_min + (z_max - z_min) * torch.rand((m, 1), device=device)
        return torch.cat([rho * torch.cos(theta), rho * torch.sin(theta), z], dim=1)

    return reject_rx(n, sampler, config, device)

def sample_near_caps(n: int, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    radius, z_min, z_max, _, _ = domain_values(config)
    width = float(config["sampling"]["near_caps_width"])
    inset = float(config["sampling"].get("fem_inset", 0.0))
    rho = (radius - inset) * torch.sqrt(torch.rand((n, 1), device=device))
    theta = 2.0 * torch.pi * torch.rand((n, 1), device=device)
    side = torch.where(torch.rand((n, 1), device=device) < 0.5, -1.0, 1.0)
    z = torch.where(
        side < 0.0,
        z_min + inset + width * torch.rand((n, 1), device=device),
        z_max - inset - width * torch.rand((n, 1), device=device),
    )
    return torch.cat([rho * torch.cos(theta), rho * torch.sin(theta), z], dim=1)

def sample_interior(n: int, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    mixture = config["sampling"]["interior_mixture"]
    n_rx = int(round(n * float(mixture["near_rx"])))
    n_wall = int(round(n * float(mixture["near_wall"])))
    n_caps = int(round(n * float(mixture["near_caps"])))
    n_volume = n - n_rx - n_wall - n_caps
    if n_volume < 0:
        raise ValueError("sampling.interior_mixture fractions must sum to at most 1.")
    chunks = [
        reject_rx(n_volume, lambda m: sample_cylinder(m, config, device), config, device),
        sample_near_rx(n_rx, config, device),
        sample_near_wall(n_wall, config, device),
        sample_near_caps(n_caps, config, device),
    ]
    return torch.cat([chunk for chunk in chunks if chunk.shape[0]], dim=0)

def sample_ic_interior(n: int, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    cfg = config.get("ic_sampling", {})
    if not bool(cfg.get("enabled", False)):
        return sample_interior(n, config, device)
    fraction = float(cfg.get("near_rx_fraction", 0.0))
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"ic_sampling.near_rx_fraction must be in [0, 1], got {fraction}")
    n_rx = int(round(n * fraction))
    n_other = n - n_rx
    chunks = [
        reject_rx(n_other, lambda m: sample_cylinder(m, config, device), config, device),
        sample_near_rx(
            n_rx,
            config,
            device,
            width=float(cfg.get("near_rx_width", config["sampling"]["near_rx_width"])),
            surface_margin=float(cfg.get("rx_surface_margin", config["domain"].get("boundary_eps", 1.0e-4))),
        ),
    ]
    return torch.cat([chunk for chunk in chunks if chunk.shape[0]], dim=0)

def sample_rx(n: int, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    _, _, _, center, radius = domain_values(config)
    direction = torch.randn((n, 3), device=device)
    direction = direction / torch.linalg.vector_norm(direction, dim=1, keepdim=True)
    return center.to(device) + radius * direction

def sample_wall(n: int, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    radius, z_min, z_max, _, _ = domain_values(config)
    theta = 2.0 * torch.pi * torch.rand((n, 1), device=device)
    z = z_min + (z_max - z_min) * torch.rand((n, 1), device=device)
    return torch.cat([radius * torch.cos(theta), radius * torch.sin(theta), z], dim=1)

def sample_caps(n: int, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    radius, z_min, z_max, _, _ = domain_values(config)
    inset = float(config["sampling"].get("fem_inset", 0.0))
    rho = (radius - inset) * torch.sqrt(torch.rand((n, 1), device=device))
    theta = 2.0 * torch.pi * torch.rand((n, 1), device=device)
    z = torch.where(torch.rand((n, 1), device=device) < 0.5, z_min, z_max)
    return torch.cat([rho * torch.cos(theta), rho * torch.sin(theta), z], dim=1)

def replace_with_early_times(t: torch.Tensor, config: dict[str, Any]) -> None:
    cfg = config.get("early_time_sampling", {})
    if not bool(cfg.get("enabled", False)):
        return
    levels = cfg.get("levels")
    if levels is None:
        levels = [cfg]
    remaining = torch.randperm(t.shape[0], device=t.device)
    offset = 0
    for level in levels:
        fraction = float(level.get("fraction", 0.0))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"early_time_sampling fraction must be in [0, 1], got {fraction}")
        n = int(round(t.shape[0] * fraction))
        if offset + n > t.shape[0]:
            raise ValueError("early_time_sampling.levels fractions must sum to at most 1.")
        if n == 0:
            continue
        t_min = float(level.get("t_min", config["time"].get("t_min_pde", 1.0e-5)))
        t_max = float(level["t_max"])
        bias_power = float(level.get("bias_power", 1.0))
        if not t_min < t_max or bias_power <= 0.0:
            raise ValueError("Each early_time_sampling level requires t_min < t_max and bias_power > 0.")
        indices = remaining[offset : offset + n]
        t[indices] = t_min + (t_max - t_min) * torch.rand((n, 1), device=t.device).pow(bias_power)
        offset += n

def make_batch(config: dict[str, Any], device: torch.device, sizes: dict[str, Any]) -> dict[str, torch.Tensor]:
    n_pde, n_ic = int(sizes["pde"]), int(sizes["ic"])
    pde_t = sample_time(n_pde, config, device)
    replace_with_early_times(pde_t, config)
    batch = {
        "pde_t": pde_t,
        "pde_xyz": sample_interior(n_pde, config, device),
        "ic_t": torch.zeros((n_ic, 1), device=device),
        "ic_xyz": sample_ic_interior(n_ic, config, device),
        "rx_t": sample_time(int(sizes["rx"]), config, device),
        "rx_xyz": sample_rx(int(sizes["rx"]), config, device),
        "wall_t": sample_time(int(sizes["wall"]), config, device),
        "wall_xyz": sample_wall(int(sizes["wall"]), config, device),
        "caps_t": sample_time(int(sizes["caps"]), config, device),
        "caps_xyz": sample_caps(int(sizes["caps"]), config, device),
    }
    late_cfg = config.get("late_time_constraint", {})
    n_late = int(sizes.get("late", 0))
    if bool(late_cfg.get("enabled", False)) and n_late:
        radius, _, _, _, _ = domain_values(config)
        window = float(late_cfg["z_window"])

        def sampler(m: int) -> torch.Tensor:
            inset = float(config["sampling"].get("fem_inset", 0.0))
            rho = (radius - inset) * torch.sqrt(torch.rand((m, 1), device=device))
            theta = 2.0 * torch.pi * torch.rand((m, 1), device=device)
            z = -window + 2.0 * window * torch.rand((m, 1), device=device)
            return torch.cat([rho * torch.cos(theta), rho * torch.sin(theta), z], dim=1)

        batch["late_xyz"] = reject_rx(n_late, sampler, config, device)
        batch["late_t"] = float(late_cfg["t_min"]) + (float(late_cfg["t_max"]) - float(late_cfg["t_min"])) * torch.rand(
            (n_late, 1), device=device
        )
    return batch

def predict(model: torch.nn.Module, t: torch.Tensor, xyz: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    return model(pipe_features(t, xyz, config))

def laplacian(value: torch.Tensor, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gradient = grad_scalar(value, xyz, create_graph=True)
    result = torch.zeros_like(value)
    for index in range(3):
        result = result + grad_scalar(gradient[:, index : index + 1], xyz, create_graph=True)[:, index : index + 1]
    return result, gradient

def losses(
    model: torch.nn.Module,
    q_provider: TorchFEMProvider,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    t = batch["pde_t"].clone().requires_grad_(True)
    xyz = batch["pde_xyz"].clone().requires_grad_(True)
    u = predict(model, t, xyz, config)
    u_t = grad_scalar(u, t, create_graph=True)
    lap_u, grad_u = laplacian(u, xyz)
    q_inf, grad_q = q_provider.value_and_grad(xyz)
    residual = q_inf * (u_t - lap_u) - 2.0 * torch.sum(grad_q * grad_u, dim=1, keepdim=True)
    hybrid = config.get("hybrid_residual", {})
    if bool(hybrid.get("enabled", False)):
        epsilon_q = float(hybrid["epsilon_q"])
        residual = residual / (q_inf + epsilon_q)
    if not torch.isfinite(residual).all():
        diagnostics = {
            "q_inf": bool(torch.isfinite(q_inf).all()),
            "grad_q": bool(torch.isfinite(grad_q).all()),
            "u": bool(torch.isfinite(u).all()),
            "u_t": bool(torch.isfinite(u_t).all()),
            "lap_u": bool(torch.isfinite(lap_u).all()),
            "grad_u": bool(torch.isfinite(grad_u).all()),
            "xyz": bool(torch.isfinite(xyz).all()),
            "t": bool(torch.isfinite(t).all()),
        }
        raise FloatingPointError(f"Non-finite PDE residual components: {diagnostics}")

    u_ic = predict(model, batch["ic_t"], batch["ic_xyz"], config)
    u_rx = predict(model, batch["rx_t"], batch["rx_xyz"], config)
    wall_xyz = batch["wall_xyz"].clone().requires_grad_(True)
    u_wall = predict(model, batch["wall_t"], wall_xyz, config)
    grad_wall = grad_scalar(u_wall, wall_xyz, create_graph=True)
    wall_rho = torch.linalg.vector_norm(wall_xyz[:, :2], dim=1, keepdim=True)
    wall_normal = torch.cat([wall_xyz[:, :2] / wall_rho, torch.zeros_like(wall_rho)], dim=1)
    u_wall_normal = torch.sum(grad_wall * wall_normal, dim=1, keepdim=True)
    u_caps = predict(model, batch["caps_t"], batch["caps_xyz"], config)
    q_caps, _ = q_provider.value_and_grad(batch["caps_xyz"])
    q_hat = q_inf * u
    parts = {
        "pde": torch.mean(residual**2),
        "ic": torch.mean(u_ic**2),
        "rx": torch.mean((u_rx - 1.0) ** 2),
        "wall": torch.mean(u_wall_normal**2),
        "caps": torch.mean((q_caps * u_caps) ** 2),
        "range": torch.mean(torch.relu(-u) ** 2 + torch.relu(u - 1.2) ** 2 + torch.relu(-q_hat) ** 2 + torch.relu(q_hat - 1.0) ** 2),
    }
    if "late_t" in batch:
        parts["late"] = torch.mean((predict(model, batch["late_t"], batch["late_xyz"], config) - 1.0) ** 2)
    return parts

def make_seeded_batch(config: dict[str, Any], device: torch.device, sizes: dict[str, Any], seed: int) -> dict[str, torch.Tensor]:
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        set_seed(seed)
        return make_batch(config, device, sizes)
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
