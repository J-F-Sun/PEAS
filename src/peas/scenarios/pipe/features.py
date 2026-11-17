from __future__ import annotations
import math
import torch

from typing import Any
def drift_arrival_feature(t: torch.Tensor, gap: torch.Tensor, velocity: float,
                          floor: float) -> torch.Tensor:
    """Stable 1D drift hitting CDF, used only as an input feature.

    For h=t+floor, this solves F_h=F_gg-v*F_g. The erfcx identity
    avoids exp(v*g)*erfc((g+v*h)/sqrt(4*h)) overflow. It is an
    approximate propagation hint for the 3D pipe, not its exact solution.
    """
    if velocity < 0 or not math.isfinite(velocity) or floor <= 0:
        raise ValueError("Arrival velocity must be finite/nonnegative; floor positive")
    h = t + floor
    minus = (gap - velocity * h) / torch.sqrt(4.0 * h)
    plus = (gap + velocity * h) / torch.sqrt(4.0 * h)
    return 0.5 * (torch.erfc(minus) + torch.exp(-minus.square()) * torch.special.erfcx(plus))


def pipe_features(t: torch.Tensor, xyz: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    cfg, domain = config["features"], config["domain"]
    radius = float(domain["pipe_radius"])
    z_min, z_max = float(domain["z_min"]), float(domain["z_max"])
    # Preserve the float32 center construction used by the original predictor.
    center = torch.tensor(domain["rx_center"], dtype=torch.float32, device=xyz.device)
    rx_radius = float(domain["rx_radius"])
    eps = float(cfg.get("radial_smoothing_eps", 1.0e-3))
    if eps <= 0:
        raise ValueError("features.radial_smoothing_eps must be positive.")
    time_scale = float(cfg["time_scale"])
    if time_scale <= 0:
        raise ValueError("features.time_scale must be positive.")
    rho = torch.sqrt(torch.sum(xyz[:, :2] ** 2, dim=1, keepdim=True) + eps**2)
    distance = torch.linalg.vector_norm(xyz - center, dim=1, keepdim=True)
    mode = str(cfg.get("mode", "legacy"))
    axial_scale = float(cfg.get("axial_scale", (z_max - z_min) / 2))
    if axial_scale <= 0:
        raise ValueError("features.axial_scale must be positive.")
    # Use the exact legacy arithmetic unless a new axial scale is requested.
    axial = (2.0 * (xyz[:, 2:3] - z_min) / (z_max - z_min) - 1.0
             if "axial_scale" not in cfg else (xyz[:, 2:3] - (z_min + z_max) / 2) / axial_scale)
    base = [t / time_scale, xyz[:, 0:1] / radius, xyz[:, 1:2] / radius, axial]
    spatial = str(cfg.get("spatial_transform", "cartesian"))
    if spatial != "cartesian":
        # Both reflection planes are symmetries only for the fixed centered caps
        # and an RX in their intersection. Keep Cartesian PDE derivatives.
        if abs(float(domain["rx_center"][1])) > 1e-8 or abs(float(domain["rx_center"][2]) - (z_min + z_max) / 2) > 1e-8:
            raise ValueError("Mirror inputs require RX y=0 and RX z at the pipe midpoint")
        transverse_squared = xyz[:, 1:2].square() / radius**2
        axial_squared = ((xyz[:, 2:3] - (z_min + z_max) / 2) / axial_scale).square()
        if spatial == "mirror_square":
            base[2:] = [transverse_squared, axial_squared]
        elif spatial == "mirror_log":
            base[2:] = [torch.log1p(transverse_squared), torch.log1p(axial_squared)]
        else:
            raise ValueError(f"Unknown spatial transform: {spatial}")
    boundary = [(distance - rx_radius) / radius, (radius - rho) / radius,
                torch.minimum(xyz[:, 2:3] - z_min, z_max - xyz[:, 2:3]) / (z_max - z_min)]
    caps_transform = str(cfg.get("caps_transform", "distance"))
    if caps_transform == "mirror_square":
        half_length = (z_max - z_min) / 2
        relative_z = (xyz[:, 2:3] - (z_min + z_max) / 2) / half_length
        # Smooth, reflection-invariant cap hint: 1/2 at the midpoint, 0 at caps.
        boundary[2] = (1.0 - relative_z.square()) / 2.0
    elif caps_transform != "distance":
        raise ValueError("Unknown cap hint transform")
    envelope = str(cfg.get("geometry_envelope", "none"))
    if envelope != "none":
        floor = float(config["time"].get("t_min_pde", 1.0e-5))
        if envelope == "arrival":
            context = torch.erfc((distance - rx_radius) / torch.sqrt(4.0 * (t + floor)))
        elif envelope == "time":
            scale = float(cfg.get("geometry_time_scale", 1.0))
            if scale <= 0:
                raise ValueError("geometry_time_scale must be positive")
            context = t / (t + scale)
        else:
            raise ValueError("Unknown geometry envelope")
        base[1:] = [value * context for value in base[1:]]
        boundary = [value * context for value in boundary]
    if mode in {"legacy", "augmented"}:
        values = base + boundary
        if bool(cfg.get("include_tau", True)):
            floor = float(config["time"].get("t_min_pde", 1.0e-5))
            hint = str(cfg.get("time_hint", "sqrt"))
            if hint == "drift_arrival":
                values.append(drift_arrival_feature(t, distance - rx_radius,
                                                    float(cfg["arrival_velocity"]), floor))
            elif hint != "sqrt":
                raise ValueError("Unknown time hint")
            elif str(cfg.get("sqrt_time_transform", "clamp")) == "smooth":
                values.append((torch.sqrt(t + floor) - math.sqrt(floor)) / math.sqrt(time_scale))
            elif str(cfg.get("sqrt_time_transform", "clamp")) == "clamp":
                values.append(torch.sqrt(torch.clamp(t, min=floor) / time_scale))
            else:
                raise ValueError("Unknown sqrt time transform")
    elif mode == "compact":
        values = base
    elif mode == "compact_wall":
        values = base + [boundary[1]]
    elif mode == "compact_geometry":
        if spatial not in {"mirror_square", "mirror_log"}:
            raise ValueError("Geometry compression requires reflection-invariant coordinates")
        radial_squared = torch.sum(xyz[:, :2].square(), dim=1, keepdim=True) / radius**2
        radial_feature = torch.log1p(radial_squared) if spatial == "mirror_log" else radial_squared
        if envelope != "none":
            radial_feature = radial_feature * context
        # rho^2, z^2 and d uniquely determine x and y^2 for off-axis RX.
        values = [base[0], radial_feature, base[3]]
        compact_time = str(cfg.get("compact_time_hint", "linear"))
        if compact_time == "drift_arrival":
            floor = float(config["time"].get("t_min_pde", 1.0e-5))
            values[0] = drift_arrival_feature(t, distance - rx_radius,
                                             float(cfg["arrival_velocity"]), floor)
            # t is still recoverable from the center-distance eta and tau.
        elif compact_time != "linear":
            raise ValueError("Unknown compact time hint")
    else:
        raise ValueError(f"Unknown pipe feature mode: {mode}")
    if mode != "legacy":
        # Training points are outside the RX (distance >= rx_radius > 0).
        eta = 1.0 / distance
        diffusion_distance = str(cfg.get("diffusion_distance", "center"))
        if diffusion_distance == "center":
            diffusion_denominator = distance.square()
        elif diffusion_distance == "surface":
            gap_eps = float(cfg.get("diffusion_gap_eps", 1e-4))
            if gap_eps <= 0:
                raise ValueError("diffusion_gap_eps must be positive")
            diffusion_denominator = (distance - rx_radius).square() + gap_eps**2
        else:
            raise ValueError("Unknown diffusion distance")
        tau = t / diffusion_denominator
        tau_scale = float(cfg.get("tau_scale", 1.0))
        if not math.isfinite(tau_scale) or tau_scale <= 0:
            raise ValueError("features.tau_scale must be finite and positive.")
        transform = str(cfg.get("tau_transform", "linear"))
        if transform == "log1p":
            tau = torch.log1p(tau / tau_scale)
        elif transform == "rational":
            tau = (t / (t + tau_scale * diffusion_denominator)
                   if diffusion_distance == "surface" else tau / (tau_scale + tau))
        elif transform == "linear":
            tau = tau / tau_scale
        else:
            raise ValueError(f"Unknown tau transform: {transform}")
        values += [eta, tau]
    if bool(cfg.get("include_arrival", False)):
        # Canonical diffusion arrival shape as an INPUT only. The model still
        # has an unconstrained linear output and q=q_infty*u is unchanged.
        floor = float(config["time"].get("t_min_pde", 1.0e-5))
        arrival_type = str(cfg.get("arrival_type", "free"))
        if arrival_type == "drift":
            values.append(drift_arrival_feature(t, distance - rx_radius,
                                                float(cfg["arrival_velocity"]), floor))
        elif arrival_type == "free":
            values.append(torch.erfc((distance - rx_radius) / torch.sqrt(4.0 * (t + floor))))
        else:
            raise ValueError("Unknown arrival feature type")
    if bool(cfg.get("include_drift_arrival", False)):
        floor = float(config["time"].get("t_min_pde", 1.0e-5))
        values.append(drift_arrival_feature(t, distance - rx_radius,
                                            float(cfg["arrival_velocity"]), floor))
    if bool(cfg.get("include_gaussian", False)):
        floor = float(config["time"].get("t_min_pde", 1.0e-5))
        values.append(torch.exp(-(distance - rx_radius).square() / (4.0 * (t + floor))))
    result = torch.cat(values, dim=1)
    if result.shape[1] != int(cfg["input_dim"]):
        raise ValueError(f"features.input_dim={cfg['input_dim']}, but {mode} created {result.shape[1]} inputs.")
    return result


