from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from peas.common.utils import timestamp
@dataclass
class BisphericalGeometry:
    outer_center: np.ndarray
    axis: np.ndarray
    d: float
    rx_radius: float
    outer_radius: float
    z_shift: float
    a: float
    xi_outer: float
    xi_rx: float

def bispherical_geometry(config: dict[str, Any]) -> BisphericalGeometry:
    rx_center = np.asarray(config["domain"]["rx_center"], dtype=float)
    outer_center = np.asarray(
        config["domain"].get("outer_center", [0.0, 0.0, 0.0]), dtype=float
    )
    center_delta = rx_center - outer_center
    d = float(np.linalg.norm(center_delta))
    rx_radius = float(config["domain"].get("rx_radius", 1.0))
    outer_radius = float(config["domain"]["outer_radius"])
    if d <= 0.0:
        raise ValueError(
            "bispherical FEM requires nonconcentric geometry: domain.rx_center must be nonzero."
        )
    if d >= outer_radius - rx_radius:
        raise ValueError("The RX sphere must be strictly inside the outer sphere.")
    axis = center_delta / d
    z_shift = (d * d + outer_radius * outer_radius - rx_radius * rx_radius) / (2.0 * d)
    a2 = z_shift * z_shift - outer_radius * outer_radius
    if a2 <= 0.0:
        raise ValueError(
            "Invalid bispherical parameters; check radii and center distance."
        )
    a = float(np.sqrt(a2))
    xi_rx = -float(np.arcsinh(a / rx_radius))
    xi_outer = -float(np.arcsinh(a / outer_radius))
    return BisphericalGeometry(
        outer_center=outer_center,
        axis=axis,
        d=d,
        rx_radius=rx_radius,
        outer_radius=outer_radius,
        z_shift=float(z_shift),
        a=a,
        xi_outer=xi_outer,
        xi_rx=xi_rx,
    )

def node_id(i: int, j: int, n_eta: int) -> int:
    return i * n_eta + j

def shape_q1(s: float, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 0.25 * np.array(
        [(1 - s) * (1 - t), (1 + s) * (1 - t), (1 + s) * (1 + t), (1 - s) * (1 + t)]
    )
    dn_ds = 0.25 * np.array([-(1 - t), 1 - t, 1 + t, -(1 + t)])
    dn_dt = 0.25 * np.array([-(1 - s), -(1 + s), 1 + s, 1 - s])
    return n, dn_ds, dn_dt

def solve_q_infty_fem(config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    geom = bispherical_geometry(config)
    fem_cfg = config["q_fem"]
    n_xi = int(fem_cfg["n_xi"])
    n_eta = int(fem_cfg["n_eta"])
    xi = np.linspace(geom.xi_rx, geom.xi_outer, n_xi)
    eta = np.linspace(0.0, np.pi, n_eta)
    n_nodes = n_xi * n_eta
    matrix = lil_matrix((n_nodes, n_nodes), dtype=float)
    rhs = np.zeros(n_nodes, dtype=float)
    gauss = [(-1.0 / np.sqrt(3.0), 1.0), (1.0 / np.sqrt(3.0), 1.0)]

    for i in range(n_xi - 1):
        dxi = xi[i + 1] - xi[i]
        for j in range(n_eta - 1):
            deta = eta[j + 1] - eta[j]
            ids = [
                node_id(i, j, n_eta),
                node_id(i + 1, j, n_eta),
                node_id(i + 1, j + 1, n_eta),
                node_id(i, j + 1, n_eta),
            ]
            local = np.zeros((4, 4), dtype=float)
            for s, ws in gauss:
                xi_q = 0.5 * (xi[i] + xi[i + 1]) + 0.5 * dxi * s
                for t, wt in gauss:
                    eta_q = 0.5 * (eta[j] + eta[j + 1]) + 0.5 * deta * t
                    _, dn_ds, dn_dt = shape_q1(s, t)
                    dn_dxi = dn_ds * (2.0 / dxi)
                    dn_deta = dn_dt * (2.0 / deta)
                    h = geom.a / (np.cosh(xi_q) - np.cos(eta_q))
                    weight = (
                        h * np.sin(eta_q) * 0.25 * dxi * deta * ws * wt * 2.0 * np.pi
                    )
                    local += weight * (
                        np.outer(dn_dxi, dn_dxi) + np.outer(dn_deta, dn_deta)
                    )
            for a_local, row in enumerate(ids):
                for b_local, col in enumerate(ids):
                    matrix[row, col] += local[a_local, b_local]

    kappa = float(config["boundary"]["kappa"])
    i_outer = n_xi - 1
    for j in range(n_eta - 1):
        deta = eta[j + 1] - eta[j]
        ids = [node_id(i_outer, j, n_eta), node_id(i_outer, j + 1, n_eta)]
        for s, w in gauss:
            eta_q = 0.5 * (eta[j] + eta[j + 1]) + 0.5 * deta * s
            n = np.array([0.5 * (1 - s), 0.5 * (1 + s)])
            h = geom.a / (np.cosh(geom.xi_outer) - np.cos(eta_q))
            weight = kappa * h * h * np.sin(eta_q) * 0.5 * deta * w * 2.0 * np.pi
            for a_local, row in enumerate(ids):
                for b_local, col in enumerate(ids):
                    matrix[row, col] += weight * n[a_local] * n[b_local]

    dirichlet_nodes = [node_id(0, j, n_eta) for j in range(n_eta)]
    for idx in dirichlet_nodes:
        matrix.rows[idx] = [idx]
        matrix.data[idx] = [1.0]
        rhs[idx] = 1.0
    dirichlet_set = set(dirichlet_nodes)
    for row in range(n_nodes):
        if row in dirichlet_set:
            continue
        cols = matrix.rows[row]
        data = matrix.data[row]
        for k in range(len(cols) - 1, -1, -1):
            if cols[k] in dirichlet_set:
                rhs[row] -= data[k]
                del cols[k]
                del data[k]

    values = np.asarray(spsolve(matrix.tocsr(), rhs), dtype=float).reshape(n_xi, n_eta)
    values = np.clip(values, 0.0, 1.0)
    q_xi, q_eta = np.gradient(values, xi, eta, edge_order=2)

    stamp = timestamp()
    path = (
        Path(config["outputs"]["checkpoints_dir"])
        / f"q_infty_bispherical_fem_{stamp}.npz"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        xi=xi,
        eta=eta,
        q=values,
        q_xi=q_xi,
        q_eta=q_eta,
        axis=geom.axis,
        outer_center=geom.outer_center,
        d=geom.d,
        rx_radius=geom.rx_radius,
        outer_radius=geom.outer_radius,
        z_shift=geom.z_shift,
        a=geom.a,
        xi_outer=geom.xi_outer,
        xi_rx=geom.xi_rx,
        config=json.dumps(config),
    )
    summary = {
        "checkpoint": str(path),
        "q_min": float(values.min()),
        "q_max": float(values.max()),
        "nodes": int(n_nodes),
        "xi_outer": geom.xi_outer,
        "xi_rx": geom.xi_rx,
    }
    print(
        f"[bispherical-fem-qinf-pinn] q_fem={path} q_min={summary['q_min']:.6g} "
        f"q_max={summary['q_max']:.6g} nodes={n_nodes}",
        flush=True,
    )
    return summary, path

class BisphericalFEMProvider:
    def __init__(self, checkpoint: str | Path, device: torch.device) -> None:
        data = np.load(checkpoint, allow_pickle=False)
        self.xi = np.asarray(data["xi"], dtype=float)
        self.eta = np.asarray(data["eta"], dtype=float)
        q_grid = np.asarray(data["q"], dtype=float)
        q_xi_grid = np.asarray(data["q_xi"], dtype=float)
        q_eta_grid = np.asarray(data["q_eta"], dtype=float)
        q_grid = np.nan_to_num(q_grid, nan=0.0, posinf=1.0, neginf=0.0)
        q_xi_grid = np.nan_to_num(q_xi_grid, nan=0.0, posinf=0.0, neginf=0.0)
        q_eta_grid = np.nan_to_num(q_eta_grid, nan=0.0, posinf=0.0, neginf=0.0)
        self.q_interp = RegularGridInterpolator(
            (self.xi, self.eta), q_grid, bounds_error=False, fill_value=None
        )
        self.q_xi_interp = RegularGridInterpolator(
            (self.xi, self.eta), q_xi_grid, bounds_error=False, fill_value=None
        )
        self.q_eta_interp = RegularGridInterpolator(
            (self.xi, self.eta), q_eta_grid, bounds_error=False, fill_value=None
        )
        self.axis_np = np.asarray(data["axis"], dtype=float)
        self.outer_center_np = (
            np.asarray(data["outer_center"], dtype=float)
            if "outer_center" in data.files
            else np.zeros(3, dtype=float)
        )
        self.axis = torch.as_tensor(
            self.axis_np, dtype=torch.float32, device=device
        ).reshape(1, 3)
        self.outer_center = torch.as_tensor(
            self.outer_center_np, dtype=torch.float32, device=device
        ).reshape(1, 3)
        self.z_shift = float(data["z_shift"])
        self.a = float(data["a"])
        self.device = device

    def _coords_np(
        self, x_np: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y_np = x_np - self.outer_center_np.reshape(1, 3)
        z = y_np @ self.axis_np
        radial = y_np - z[:, None] * self.axis_np.reshape(1, 3)
        rho = np.linalg.norm(radial, axis=1)
        z_rel = z - self.z_shift
        a = self.a
        aa = np.maximum(rho * rho + (z_rel + a) ** 2, 1.0e-24)
        bb = np.maximum(rho * rho + (z_rel - a) ** 2, 1.0e-24)
        xi = 0.5 * np.log(aa / bb)
        eta = np.arctan2(2.0 * a * rho, rho * rho + z_rel * z_rel - a * a)
        eta = np.where(eta < 0.0, eta + np.pi, eta)
        xi = np.nan_to_num(xi, nan=self.xi[0], posinf=self.xi[-1], neginf=self.xi[0])
        eta = np.nan_to_num(eta, nan=0.0, posinf=self.eta[-1], neginf=self.eta[0])
        xi = np.clip(xi, self.xi[0], self.xi[-1])
        eta = np.clip(eta, self.eta[0], self.eta[-1])
        return xi, eta, rho, z_rel, radial

    def value_and_grad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_np = x.detach().cpu().numpy()
        xi, eta, rho, z_rel, radial = self._coords_np(x_np)
        points = np.stack([xi, eta], axis=1)
        q = np.asarray(self.q_interp(points), dtype=np.float32).reshape(-1, 1)
        q_xi = np.asarray(self.q_xi_interp(points), dtype=np.float32).reshape(-1, 1)
        q_eta = np.asarray(self.q_eta_interp(points), dtype=np.float32).reshape(-1, 1)
        q = np.nan_to_num(q, nan=0.0, posinf=1.0, neginf=0.0)
        q_xi = np.nan_to_num(q_xi, nan=0.0, posinf=0.0, neginf=0.0)
        q_eta = np.nan_to_num(q_eta, nan=0.0, posinf=0.0, neginf=0.0)

        a = self.a
        aa = np.maximum(rho * rho + (z_rel + a) ** 2, 1.0e-24)
        bb = np.maximum(rho * rho + (z_rel - a) ** 2, 1.0e-24)
        xi_rho = rho * (1.0 / aa - 1.0 / bb)
        xi_z = (z_rel + a) / aa - (z_rel - a) / bb
        n = 2.0 * a * rho
        m = rho * rho + z_rel * z_rel - a * a
        denom = np.maximum(m * m + n * n, 1.0e-24)
        eta_rho = (m * 2.0 * a - n * 2.0 * rho) / denom
        eta_z = -(n * 2.0 * z_rel) / denom
        rho_safe = np.maximum(rho, 1.0e-12)
        grad_rho = radial / rho_safe[:, None]
        grad_rho[rho < 1.0e-10] = 0.0
        axis = self.axis_np.reshape(1, 3)
        c_rho = q_xi * xi_rho.reshape(-1, 1) + q_eta * eta_rho.reshape(-1, 1)
        c_z = q_xi * xi_z.reshape(-1, 1) + q_eta * eta_z.reshape(-1, 1)
        grad = c_rho * grad_rho + c_z * axis
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        return (
            torch.as_tensor(q, dtype=torch.float32, device=self.device),
            torch.as_tensor(
                grad.astype(np.float32), dtype=torch.float32, device=self.device
            ),
        )
