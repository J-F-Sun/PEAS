from __future__ import annotations
import math
import torch

FEATURE_NAMES = (
    "t/T",
    "s/S",
    "rho2/S2",
    "t/d2",
    "d2/S2",
    "ro2/S2",
    "(d-a)/S",
    "(R-ro)/S",
    "1/d",
    "erfc((d-a)/sqrt(4*(t+eps)))",
)


def sphere_input_scale(config: dict) -> float:
    """S follows the physical outer radius when requested; numeric legacy S stays fixed.

    Keep the symbolic rule in saved configs so checkpoint inference applies
    the same geometry-dependent mapping as training.
    """
    configured = config["features"].get("input_scale", "outer_radius")
    if isinstance(configured, str):
        if configured != "outer_radius":
            raise ValueError("Sphere input_scale must be positive numeric or 'outer_radius'")
        configured = config["domain"]["outer_radius"]
    scale = float(configured)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Sphere input_scale must resolve to a finite positive value")
    return scale


def validate_sphere_config(config: dict) -> None:
    f = config["features"]
    scale = sphere_input_scale(config)
    if config["model"]["input_dim"] != 10:
        raise ValueError("axis10_arrival requires exactly 10 inputs")
    if f.get("mode") != "axis_augmented" or not f.get("include_arrival"):
        raise ValueError("Expected axis10_arrival (axis_augmented + include_arrival)")
    if f.get("tau_transform", "linear") != "linear" or f.get("tau_scale", 1.0) != 1.0:
        raise ValueError("axis10_arrival uses the original linear t/d² input")
    if f.get("arrival_prefactor", "none") != "none":
        raise ValueError("axis10_arrival uses the original unweighted erfc input")
    for key in (
        "include_rx_radius_squared",
        "include_outer_radius_squared",
        "include_rx_gap",
        "include_outer_gap",
    ):
        if not f.get(key, True):
            raise ValueError(f"axis10_arrival requires {key}")
    if (
        min(
            scale,
            float(f["time_scale"]),
            float(config["time"]["t_min_pde"]),
        )
        <= 0
    ):
        raise ValueError("Feature scales and time epsilon must be positive")
    if float(f.get("outer_eps", 0.0)) != 0.0:
        raise ValueError("Retained axis10_arrival uses outer_eps=0")
    if (
        f.get("axial_scale", scale) != scale
        or f.get("transverse_scale", scale) != scale
    ):
        raise ValueError("Retained axis10_arrival uses one spatial scale")


def sphere_features(t: torch.Tensor, x: torch.Tensor, config: dict) -> torch.Tensor:
    f, g = config["features"], config["domain"]
    c, origin = x.new_tensor(g["rx_center"]), x.new_tensor(g["outer_center"])
    d = torch.linalg.vector_norm(x - c, dim=1, keepdim=True)
    r2 = (x - origin).square().sum(dim=1, keepdim=True)
    axis = c - origin
    axis_norm = torch.linalg.vector_norm(axis)
    if axis_norm.item() == 0:
        axis = x.new_tensor(f["axis_direction"])
        if axis.shape != (3,) or not torch.isfinite(axis).all():
            raise ValueError("axis_direction must be a finite 3-vector")
        axis_norm = torch.linalg.vector_norm(axis)
        if axis_norm.item() == 0:
            raise ValueError("axis_direction must be nonzero")
    axis = axis / axis_norm
    axial = ((x - origin) * axis).sum(dim=1, keepdim=True)
    rho2 = ((x - origin) - axial * axis).square().sum(dim=1, keepdim=True)
    radius = torch.linalg.vector_norm(x - origin, dim=1, keepdim=True)
    return sphere_feature_values(t, d, radius, r2, axial, rho2, config)


def sphere_feature_values(t, d, radius, r2, axial, rho2, config):
    """Assemble the retained ten columns from the spatial descriptors."""
    f, g = config["features"], config["domain"]
    scale, ts = sphere_input_scale(config), float(f["time_scale"])
    arrival = torch.erfc(
        (d - float(g["rx_radius"]))
        / torch.sqrt(4 * (t + float(config["time"]["t_min_pde"])))
    )
    return torch.cat(
        [
            t / ts,
            axial / scale,
            rho2 / scale**2,
            t / d.square(),
            d.square() / scale**2,
            r2 / scale**2,
            (d - float(g["rx_radius"])) / scale,
            (float(g["outer_radius"]) - radius) / scale,
            1.0 / d,
            arrival,
        ],
        dim=1,
    )

