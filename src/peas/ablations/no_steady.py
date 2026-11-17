"""Direct backward probability: no FEM field, multiplier, or Q-dependent prior."""
import torch

from peas.common.utils import grad_scalar
from peas.scenarios.canonical.physics_base import _condition_residual
from peas.scenarios.pipe.physics import laplacian


def canonical_parts(model, batch):
    c = model.config
    t = batch["pde"][:, :1].clone().requires_grad_(True)
    y = batch["pde"][:, 1:].clone().requires_grad_(True)
    q = model.latent(t, y)
    q_t = grad_scalar(q, t)
    q_y = grad_scalar(q, y)
    q_yy = grad_scalar(q_y, y)
    residual = _condition_residual(q_t - y**4 * q_yy, y, model.training.native)
    ic = batch["ic"]
    q_ic = model.latent(ic[:, :1], ic[:, 1:])
    rx = batch["bc_rx"]
    q_rx = model.latent(rx[:, :1], rx[:, 1:])
    far = batch["bc_far"]
    yf = far[:, 1:].clone().requires_grad_(True)
    q_far = model.latent(far[:, :1], yf)
    kappa = c["steady_state"]["parameters"]["robin_kappa"]
    # Original q_r + kappa*q = 0, not the factor's u_y = 0.
    far_residual = -yf.square() * grad_scalar(q_far, yf) + kappa * q_far
    return {"pde": residual, "ic": q_ic, "bc_rx": q_rx - 1,
            "bc_far": far_residual}, q


def pipe_losses(model, batch):
    t = batch["pde_t"].clone().requires_grad_(True)
    xyz = batch["pde_xyz"].clone().requires_grad_(True)
    q = model.latent(t, xyz)
    lap, _ = laplacian(q, xyz)
    residual = grad_scalar(q, t) - lap
    qi = model.latent(batch["ic_t"], batch["ic_xyz"])
    qr = model.latent(batch["rx_t"], batch["rx_xyz"])
    wall = batch["wall_xyz"].clone().requires_grad_(True)
    qw = model.latent(batch["wall_t"], wall)
    rho = torch.linalg.vector_norm(wall[:, :2], dim=1, keepdim=True)
    normal = torch.cat((wall[:, :2] / rho, torch.zeros_like(rho)), dim=1)
    wall_residual = (grad_scalar(qw, wall) * normal).sum(dim=1, keepdim=True)
    wall_residual = wall_residual + model.config["problem"]["boundary"]["kappa"] * qw
    qc = model.latent(batch["caps_t"], batch["caps_xyz"])
    # No Q-based residual normalization, upper envelope, or late q=Q target.
    return {"pde": residual.square().mean(), "ic": qi.square().mean(),
            "rx": (qr - 1).square().mean(), "wall": wall_residual.square().mean(),
            "caps": qc.square().mean(),
            "range": (torch.relu(-q).square() + torch.relu(q - 1).square()).mean()}

