"""Concentric/eccentric objectives and the axisymmetric forward source IC.

The retained ten features are invariant under rotations about the sphere axis.
Average a point Gaussian over that exact symmetry orbit, rather than asking an
axisymmetric MLP to represent a non-axisymmetric density. Every rotated source
has the same receiver absorption curve. No angular discretization is needed.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from scipy.stats import ncx2

from peas.common.geometry import grad_scalar, laplacian, outer_boundary_normal


def axis(c):
    g = c["problem"]["domain"]
    direction = np.asarray(g["rx_center"], dtype=float) - g["outer_center"]
    if np.linalg.norm(direction) < 1e-12:
        direction = np.asarray(c["features"]["axis_direction"], dtype=float)
    return direction / np.linalg.norm(direction)


def gaussian_mass(c, sigma):
    """Exact 3D isotropic Gaussian mass inside outer sphere minus RX ball."""
    g = c["problem"]["domain"]
    source = np.asarray(c["ablation"]["source"], dtype=float)
    mass = (ncx2.cdf((g["outer_radius"] / sigma)**2, 3,
                     np.sum((source - g["outer_center"])**2) / sigma**2)
            - ncx2.cdf((g["rx_radius"] / sigma)**2, 3,
                       np.sum((source - g["rx_center"])**2) / sigma**2))
    if not np.isfinite(mass) or mass <= 1e-8:
        raise ValueError("Gaussian has insufficient mass inside the spherical fluid domain")
    return float(mass)


def initial_density(x, c, sigma):
    """Analytic rotation average: stable exp(...) * I0e(rho*rho0/sigma²)."""
    g = c["problem"]["domain"]
    direction = x.new_tensor(axis(c))
    rel = x - x.new_tensor(g["outer_center"])
    source = x.new_tensor(c["ablation"]["source"]) - x.new_tensor(g["outer_center"])
    axial = (rel * direction).sum(dim=1, keepdim=True)
    axial0 = (source * direction).sum()
    rho = torch.linalg.vector_norm(rel - axial * direction, dim=1, keepdim=True)
    rho0 = torch.linalg.vector_norm(source - axial0 * direction)
    exponent = -((axial - axial0).square() + (rho - rho0).square()) / (2 * sigma**2)
    return (torch.exp(exponent) * torch.special.i0e(rho * rho0 / sigma**2)
            / ((2 * math.pi)**1.5 * sigma**3 * gaussian_mass(c, sigma)))


def inside(x, c):
    g = c["problem"]["domain"]
    return ((torch.linalg.vector_norm(x - x.new_tensor(g["outer_center"]), dim=1) < g["outer_radius"])
            & (torch.linalg.vector_norm(x - x.new_tensor(g["rx_center"]), dim=1) > g["rx_radius"]))


def sample_points(n, c, *, local_scale=None):
    """Exact full-volume or truncated rotation-averaged Gaussian proposal (CPU)."""
    if n == 0:
        return torch.empty((0, 3))
    g = c["problem"]["domain"]
    direction = torch.tensor(axis(c), dtype=torch.float32)
    origin = torch.tensor(g["outer_center"], dtype=torch.float32)
    rel = torch.tensor(c["ablation"]["source"], dtype=torch.float32) - origin
    axial = (rel * direction).sum() * direction
    transverse = rel - axial
    perpendicular = torch.linalg.cross(direction, transverse)
    chunks, total = [], 0
    while total < n:
        m = max(64, 2 * (n - total))
        if local_scale is None:
            unit = torch.randn(m, 3)
            unit /= torch.linalg.vector_norm(unit, dim=1, keepdim=True).clamp_min(1e-12)
            x = origin + g["outer_radius"] * torch.rand(m, 1).pow(1 / 3) * unit
        else:
            angle = 2 * math.pi * torch.rand(m, 1)
            centers = origin + axial + transverse * angle.cos() + perpendicular * angle.sin()
            x = centers + local_scale * torch.randn(m, 3)
        accepted = x[inside(x, c)]
        chunks.append(accepted)
        total += len(accepted)
    return torch.cat(chunks)[:n]


def augment_batch(batch, c):
    f, g = c["ablation"]["forward"], c["problem"]["domain"]
    n = len(batch["ic_x"])
    nl = int(round(n * f["ic_local_fraction"]))
    fraction = nl / n
    scale = f["sigma"] * f["ic_local_scale"]
    points = torch.cat((sample_points(n - nl, c), sample_points(nl, c, local_scale=scale)))
    volume = 4 * math.pi / 3 * (g["outer_radius"]**3 - g["rx_radius"]**3)
    batch["ic_x"] = points
    batch["ic_weights"] = 1 / ((1 - fraction) + fraction * volume * initial_density(points, c, scale))
    count = int(round(len(batch["pde_x"]) * f["pde_source_fraction"]))
    batch["pde_x"][:count] = sample_points(count, c, local_scale=scale)
    return batch


def residual(model, t, x):
    """Use the retained numerical clipping; remove Q drift only when appropriate."""
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    value = model.latent(t, x)
    r = grad_scalar(value, t) - laplacian(value, x)
    c = model.config
    settings = model.training.native.get("q_infty", model.training.native.get("q_fem", {}))
    q = None
    if model.variant == "no_features" or model.variant == "control":
        q, dq = model.steady.provider.value_and_grad(x)
        floor = float(settings.get("clamp_min", 1e-6))
        q = torch.nan_to_num(q, nan=floor, posinf=1., neginf=floor).clamp_min(floor)
        dq = torch.nan_to_num(dq, nan=0., posinf=0., neginf=0.)
        drift = 2 * dq / q
        limit = float(settings.get("drift_clip", 0))
        if limit > 0:
            drift = drift.clamp(-limit, limit)
        r = r - (drift * grad_scalar(value, x)).sum(dim=1, keepdim=True)
    limit = float(settings.get("residual_clip", 0))
    if limit > 0:
        r = torch.nan_to_num(r, nan=0., posinf=limit, neginf=-limit).clamp(-limit, limit)
    return r, value, q


def losses(model, batch):
    r, value, q = residual(model, batch["pde_t"], batch["pde_x"])
    vi = model.latent(batch["ic_t"], batch["ic_x"])
    vr = model.latent(batch["bc_rx_t"], batch["bc_rx_x"])
    outer = batch["bc_outer_x"].clone().requires_grad_(True)
    vo = model.latent(batch["bc_outer_t"], outer)
    normal = outer_boundary_normal(outer, model.training.native)
    ro = (grad_scalar(vo, outer) * normal).sum(dim=1, keepdim=True)
    if model.variant == "no_reformulation":
        target = initial_density(batch["ic_x"], model.config,
                                 model.config["ablation"]["forward"]["sigma"]) / model.source_q
        ic_loss = ((vi - target).square() * batch["ic_weights"]).mean()
        rx_loss = vr.square().mean()
        range_loss = torch.relu(-value).square().mean()
    else:
        ic_loss = vi.square().mean()
        rx_loss = (vr - 1).square().mean()
        range_loss = (torch.relu(-value).square() + torch.relu(value - 1).square()).mean()
        if q is not None:
            q_hat = q * value
            range_loss = range_loss + (torch.relu(-q_hat).square() + torch.relu(q_hat - 1).square()).mean()
    if model.variant in ("no_steady", "no_reformulation"):
        ro = ro + model.config["problem"]["boundary"]["kappa"] * vo
    return {"pde": r.square().mean(), "ic": ic_loss, "bc_rx": rx_loss,
            "bc_outer": ro.square().mean(), "range": range_loss}


def refresh_adaptive(model, device, generator):
    """Same candidate distribution/top-residual policy, on each ablated PDE."""
    from peas.common.geometry import sample_mixed_interior, sample_time
    c = model.config["training"]["adaptive_sampling"]
    count = int(c["candidate_points"])
    selected = min(int(c["selected_points"]), count)
    chunk = int(c["chunk_size"])
    xs, ts, scores = [], [], []
    previous = model.transient.training
    model.transient.eval()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(generator.get_state())
            for start in range(0, count, chunk):
                n = min(chunk, count - start)
                x = sample_mixed_interior(n, model.training.native, device=torch.device("cpu"))
                t = sample_time(n, model.training.native, device=torch.device("cpu"))
                r, _, _ = residual(model, t.to(device), x.to(device))
                limit = float(c.get("score_clip", 1e3))
                score = torch.nan_to_num(r.detach().abs().flatten(), nan=0., posinf=limit).clamp_max(limit)
                xs.append(x)
                ts.append(t)
                scores.append(score.cpu())
            generator.set_state(torch.get_rng_state())
        indices = torch.topk(torch.cat(scores), k=selected, largest=True).indices
        return torch.cat(xs)[indices].detach(), torch.cat(ts)[indices].detach()
    finally:
        model.transient.train(previous)
