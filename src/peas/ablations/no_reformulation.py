"""Fixed-source FORWARD density, followed by unmodified signed flux integration.

Retain the FEM block via p(t,y|x0)=Q(x0)*v_theta(t,y). Q(x0) is constant in
Eulerian y, so the forward PDE has no backward drift term. Pipe Phi is even in
y,z; the IC is therefore averaged over the source's exact reflection orbit.
This changes the density representation but preserves the RX absorption curve.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid
from scipy.special import ndtr
from scipy.stats import ncx2

from peas.common.utils import grad_scalar
from peas.scenarios.canonical.physics_base import _condition_residual
from peas.scenarios.pipe import physics as pipe


def reflection_orbit(c):
    s = np.asarray(c["ablation"]["source"], dtype=float)
    mid = (c["problem"]["domain"]["z_min"] + c["problem"]["domain"]["z_max"]) / 2
    return np.unique([[s[0], sy * s[1], mid + sz * (s[2] - mid)]
                      for sy in (-1, 1) for sz in (-1, 1)], axis=0)


def gaussian_mass(c, sigma):
    if c["scenario"] in ("concentric", "eccentric"):
        from peas.ablations.spheres import gaussian_mass as sphere_mass
        return sphere_mass(c, sigma)
    s = np.asarray(c["ablation"]["source"])
    g = c["problem"]["domain"]
    if c["scenario"] == "canonical":
        r0 = np.linalg.norm(s)
        outer = c["steady_state"]["parameters"]["outer_radius"]
        return float(ndtr((outer - r0) / sigma) - ndtr((1 - r0) / sigma))
    outer = ncx2.cdf((g["pipe_radius"] / sigma)**2, 2, np.sum(s[:2]**2) / sigma**2)
    outer *= ndtr((g["z_max"] - s[2]) / sigma) - ndtr((g["z_min"] - s[2]) / sigma)
    inner = ncx2.cdf((g["rx_radius"] / sigma)**2, 3,
                    np.sum((s - g["rx_center"])**2) / sigma**2)
    mass = float(outer - inner)
    if not np.isfinite(mass) or mass <= 1e-8:
        raise ValueError("Gaussian has insufficient mass inside the fluid domain")
    return mass


def initial_density(coordinates, c, *, sigma=None):
    """Unit-mass radial-shell / reflection-averaged truncated Gaussian IC."""
    sigma = float(sigma or c["ablation"]["forward"]["sigma"])
    if c["scenario"] in ("concentric", "eccentric"):
        from peas.ablations.spheres import initial_density as sphere_density
        return sphere_density(coordinates, c, sigma)
    z = gaussian_mass(c, sigma)
    if c["scenario"] == "canonical":
        r = 1 / coordinates
        r0 = np.linalg.norm(c["ablation"]["source"])
        return torch.exp(-0.5 * ((r - r0) / sigma).square()) / (
            math.sqrt(2 * math.pi) * sigma * z * 4 * math.pi * r.square())
    centers = coordinates.new_tensor(reflection_orbit(c))
    squared = (coordinates[:, None, :] - centers[None, :, :]).square().sum(dim=2)
    return torch.exp(-squared / (2 * sigma**2)).mean(dim=1, keepdim=True) / (
        (2 * math.pi)**1.5 * sigma**3 * z)


def local_points(n, c, scale):
    if c["scenario"] in ("concentric", "eccentric"):
        from peas.ablations.spheres import sample_points
        return sample_points(n, c, local_scale=scale)
    if n == 0:
        return torch.empty((0, 1 if c["scenario"] == "canonical" else 3))
    chunks, total = [], 0
    while total < n:
        size = max(64, 2 * (n - total))
        if c["scenario"] == "canonical":
            r0 = np.linalg.norm(c["ablation"]["source"])
            r = r0 + scale * torch.randn(size, 1)
            outer = c["steady_state"]["parameters"]["outer_radius"]
            accepted = 1 / r[((r > 1) & (r < outer)).flatten()]
        else:
            centers = torch.tensor(reflection_orbit(c), dtype=torch.float32)
            xyz = centers[torch.randint(len(centers), (size,))] + scale * torch.randn(size, 3)
            g = c["problem"]["domain"]
            keep = ((torch.linalg.vector_norm(xyz[:, :2], dim=1) < g["pipe_radius"]) &
                    (xyz[:, 2] > g["z_min"]) & (xyz[:, 2] < g["z_max"]) &
                    (torch.linalg.vector_norm(xyz - xyz.new_tensor(g["rx_center"]), dim=1) > 1))
            accepted = xyz[keep]
        chunks.append(accepted)
        total += len(accepted)
    return torch.cat(chunks)[:n]


def augment_batch(batch, c):
    """Resolve the narrow IC; importance weights retain a uniform-volume MSE."""
    if c["scenario"] in ("concentric", "eccentric"):
        from peas.ablations.spheres import augment_batch as sphere_batch
        return sphere_batch(batch, c)
    f = c["ablation"]["forward"]
    canonical = c["scenario"] == "canonical"
    key = "ic" if canonical else "ic_xyz"
    n = len(batch[key])
    nl = int(round(n * f["ic_local_fraction"]))
    nu = n - nl
    frac = nl / n
    scale = f["sigma"] * f["ic_local_scale"]
    if canonical:
        outer = c["steady_state"]["parameters"]["outer_radius"]
        uniform = 1 / (1 + (outer - 1) * torch.rand(nu, 1))
        points = torch.cat((uniform, local_points(nl, c, scale)))
        # Desired uniform-r measure, not uniform 3D volume in the radial reduction.
        local_pdf = initial_density(points, c, sigma=scale) * (4 * math.pi / points.square())
        volume = outer - 1
        batch[key] = torch.cat((torch.zeros(n, 1), points), dim=1)
    else:
        # Exact full-volume proposal: the backward sampler's FEM inset would
        # invalidate these importance weights, and is unnecessary for density ICs.
        g = c["problem"]["domain"]
        chunks, total = [], 0
        while total < nu:
            m = max(64, 2 * (nu - total))
            rho = g["pipe_radius"] * torch.sqrt(torch.rand(m, 1))
            theta = 2 * math.pi * torch.rand(m, 1)
            z = g["z_min"] + (g["z_max"] - g["z_min"]) * torch.rand(m, 1)
            xyz = torch.cat((rho * theta.cos(), rho * theta.sin(), z), dim=1)
            keep = torch.linalg.vector_norm(xyz - xyz.new_tensor(g["rx_center"]), dim=1) > 1
            accepted = xyz[keep]
            chunks.append(accepted)
            total += len(accepted)
        uniform = torch.cat(chunks)[:nu] if nu else torch.empty((0, 3))
        points = torch.cat((uniform, local_points(nl, c, scale)))
        local_pdf = initial_density(points, c, sigma=scale)
        volume = math.pi * g["pipe_radius"]**2 * (g["z_max"] - g["z_min"]) - 4 * math.pi / 3
        batch[key] = points
    batch["ic_weights"] = 1 / ((1 - frac) + frac * volume * local_pdf)
    # A localized forward IC also needs PDE checks around its fixed source(s).
    pde_key = "pde" if canonical else "pde_xyz"
    count = int(round(len(batch[pde_key]) * f["pde_source_fraction"]))
    source_points = local_points(count, c, scale)
    if canonical:
        batch[pde_key][:count, 1:] = source_points
    else:
        batch[pde_key][:count] = source_points
    return batch


def canonical_parts(model, batch):
    t = batch["pde"][:, :1].clone().requires_grad_(True)
    y = batch["pde"][:, 1:].clone().requires_grad_(True)
    v = model.latent(t, y)
    v_y = grad_scalar(v, y)
    # Radial forward density in y=1/r: p_t=y^4 p_yy.
    residual = _condition_residual(grad_scalar(v, t) - y**4 * grad_scalar(v_y, y),
                                  y, model.training.native)
    ic = batch["ic"]
    v_ic = model.latent(ic[:, :1], ic[:, 1:])
    target = initial_density(ic[:, 1:], model.config) / model.source_q
    rx = batch["bc_rx"]
    vr = model.latent(rx[:, :1], rx[:, 1:])
    far = batch["bc_far"]
    yf = far[:, 1:].clone().requires_grad_(True)
    vf = model.latent(far[:, :1], yf)
    kappa = model.config["steady_state"]["parameters"]["robin_kappa"]
    rf = -yf.square() * grad_scalar(vf, yf) + kappa * vf
    return {"pde": residual, "ic": (v_ic - target) * batch["ic_weights"].sqrt(),
            "bc_rx": vr, "bc_far": rf}, v


def pipe_losses(model, batch):
    t = batch["pde_t"].clone().requires_grad_(True)
    xyz = batch["pde_xyz"].clone().requires_grad_(True)
    v = model.latent(t, xyz)
    lap, _ = pipe.laplacian(v, xyz)
    residual = grad_scalar(v, t) - lap
    eps = model.config["residual"]["hybrid_residual"]["epsilon_q"]
    residual = model.source_q * residual / (model.source_q + eps)
    vi = model.latent(batch["ic_t"], batch["ic_xyz"])
    target = initial_density(batch["ic_xyz"], model.config) / model.source_q
    vr = model.latent(batch["rx_t"], batch["rx_xyz"])
    wall = batch["wall_xyz"].clone().requires_grad_(True)
    vw = model.latent(batch["wall_t"], wall)
    rho = torch.linalg.vector_norm(wall[:, :2], dim=1, keepdim=True)
    normal = torch.cat((wall[:, :2] / rho, torch.zeros_like(rho)), dim=1)
    rw = (grad_scalar(vw, wall) * normal).sum(dim=1, keepdim=True)
    rw = rw + model.config["problem"]["boundary"]["kappa"] * vw
    vc = model.latent(batch["caps_t"], batch["caps_xyz"])
    result = {"pde": residual.square().mean(),
              "ic": ((vi - target).square() * batch["ic_weights"]).mean(),
              "rx": vr.square().mean(), "wall": rw.square().mean(),
              "caps": vc.square().mean(), "range": torch.relu(-v).square().mean()}
    if "late_t" in batch:
        # A forward density tends to zero, rather than the CDF factor's one.
        result["late"] = model.latent(batch["late_t"], batch["late_xyz"]).square().mean()
    return result


def sphere_quadrature(polar, azimuth):
    mu, w = np.polynomial.legendre.leggauss(polar)
    phi = 2 * math.pi * np.arange(azimuth) / azimuth
    x = np.sqrt(1 - mu[:, None]**2) * np.cos(phi[None, :])
    y = np.sqrt(1 - mu[:, None]**2) * np.sin(phi[None, :])
    z = np.repeat(mu[:, None], azimuth, axis=1)
    directions = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    weights = np.repeat(w, azimuth) * 2 * math.pi / azimuth
    return directions, weights


def integrate_flux(times, rate):
    times, rate = np.asarray(times), np.asarray(rate)
    if times.ndim != 1 or rate.shape != times.shape or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Flux integration requires a strictly increasing grid starting at zero")
    return cumulative_trapezoid(rate, times, initial=0)


def absorption_curve(model, times, *, batch_size=4096):
    """Differentiate density at RX and integrate; retain negative/raw results."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    c = model.config
    requested = np.asarray(times, dtype=float)
    if requested.ndim != 1 or not len(requested) or not np.isfinite(requested).all() or np.any(requested < 0):
        raise ValueError("Invalid evaluation times")
    f = c["ablation"]["forward"]
    end = float(requested.max())
    dense = np.array([0.]) if end == 0 else np.geomspace(min(1e-7, end), end, f["integration_points"])
    grid = np.unique(np.concatenate(([0.], requested, dense)))
    canonical = c["scenario"] == "canonical"
    if canonical:
        surface, weights = np.ones((1, 1)), np.array([4 * math.pi])
    else:
        directions, weights = sphere_quadrature(f["surface_polar"], f["surface_azimuth"])
        g = c["problem"]["domain"]
        surface = np.asarray(g["rx_center"]) + g["rx_radius"] * directions
        weights = weights * g["rx_radius"]**2
    n_surface = len(surface)
    rates = np.zeros(len(grid), dtype=np.float64)
    was_training = model.transient.training
    model.transient.eval()
    try:
        # Accumulate in float64; the network/derivatives retain formal float32.
        with torch.enable_grad():
            for start in range(0, len(grid) * n_surface, batch_size):
                flat = np.arange(start, min(start + batch_size, len(grid) * n_surface))
                ti, si = flat // n_surface, flat % n_surface
                t = torch.tensor(grid[ti, None], dtype=torch.float32, device=model.device)
                x = torch.tensor(surface[si], dtype=torch.float32, device=model.device, requires_grad=True)
                p = model.field(t, x)
                gradient = grad_scalar(p, x, create_graph=False)
                if canonical:
                    # d/dr = -y^2*d/dy; y=1 on unit RX.
                    flux = -gradient[:, 0].detach().cpu().numpy()
                else:
                    normal_rx_to_fluid = x.new_tensor(directions[si])
                    flux = (gradient * normal_rx_to_fluid).sum(dim=1).detach().cpu().numpy()
                np.add.at(rates, ti, flux.astype(float) * weights[si])
    finally:
        model.transient.train(was_training)
    if not np.isfinite(rates).all():
        raise FloatingPointError("Nonfinite RX flux")
    q = integrate_flux(grid, rates)
    return {"prediction": q[np.searchsorted(grid, requested)],
            "integration_times": grid, "rx_rate": rates, "integrated_q": q}

