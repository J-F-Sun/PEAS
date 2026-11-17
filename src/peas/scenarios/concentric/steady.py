from __future__ import annotations
from typing import Any
import json
from pathlib import Path
import numpy as np
import torch
def _center_array(config: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(config["domain"].get(key, [0.0, 0.0, 0.0]), dtype=float)

def center_distance(config: dict[str, Any]) -> float:
    return float(np.linalg.norm(_center_array(config, "rx_center") - _center_array(config, "outer_center")))

class CenteredFEMQInfinityProvider:
    """Frozen radial P1 FEM field for the physical concentric Robin shell.

    Solve -(r² q')'=0, q(a)=1, q'(R)+kappa*q(R)=0. Gradients are
    derivatives of the piecewise-linear FEM solution, never analytic values.
    """

    kind = "centered_radial_p1_fem"

    def __init__(self, config, device, parameters, checkpoint=None):
        self.config = config
        self.device = device
        self.parameters = dict(parameters)
        self.center = _center_array(config, "outer_center")
        self.inner_radius = float(config["domain"]["rx_radius"])
        self.outer_radius = float(config["domain"]["outer_radius"])
        self.kappa = float(config["boundary"]["kappa"])
        self.num_elements = int(parameters["num_elements"])
        self.mesh_power = float(parameters["mesh_power"])
        if center_distance(config) > 1e-10:
            raise ValueError("Concentric FEM requires coincident sphere centers")
        if not (0 < self.inner_radius < self.outer_radius) or self.kappa < 0:
            raise ValueError("Invalid concentric FEM geometry or Robin coefficient")
        if self.num_elements < 2 or not np.isfinite(self.mesh_power) or self.mesh_power <= 0:
            raise ValueError("Concentric FEM requires >=2 elements and positive mesh_power")
        if checkpoint is None:
            self.r, self.q = self._solve()
        else:
            with np.load(checkpoint, allow_pickle=False) as data:
                if str(data["kind"]) != self.kind or json.loads(str(data["parameters"])) != self.parameters:
                    raise ValueError("Concentric FEM method/mesh differs from saved asset")
                old = json.loads(str(data["config"]))
                for key in ("rx_center", "outer_center", "rx_radius", "outer_radius"):
                    if not np.allclose(old["domain"][key], config["domain"][key], atol=1e-10, rtol=0):
                        raise ValueError(f"Concentric FEM geometry mismatch: {key}")
                if not np.isclose(old["boundary"]["kappa"], self.kappa, atol=1e-10, rtol=0):
                    raise ValueError("Concentric FEM Robin coefficient mismatch")
                self.r, self.q = np.array(data["r"]), np.array(data["q"])
            expected = self.inner_radius + (self.outer_radius-self.inner_radius)*np.linspace(0, 1, self.num_elements+1)**self.mesh_power
            if (self.r.shape != expected.shape or self.q.shape != expected.shape
                    or not np.allclose(self.r, expected, atol=1e-12, rtol=1e-12)
                    or not np.isfinite(self.q).all() or not np.isclose(self.q[0], 1.)):
                raise ValueError("Invalid concentric FEM arrays")
        self.dq_dr = np.diff(self.q) / np.diff(self.r)

    def _solve(self):
        from scipy.linalg import solve_banded
        r = self.inner_radius + (self.outer_radius-self.inner_radius)*np.linspace(0, 1, self.num_elements+1)**self.mesh_power
        h = np.diff(r)
        integral = (r[1:]**3-r[:-1]**3)/(3*h**2)
        diagonal = np.zeros(len(r))
        diagonal[:-1] += integral
        diagonal[1:] += integral
        diagonal[-1] += self.outer_radius**2*self.kappa
        off = -integral
        rhs = np.zeros(self.num_elements)
        rhs[0] = -off[0]  # Eliminate the RX Dirichlet value q(a)=1.
        band = np.zeros((3, self.num_elements))
        band[1] = diagonal[1:]
        band[0, 1:] = off[1:]
        band[2, :-1] = off[1:]
        return r, np.r_[1., solve_banded((1, 1), band, rhs)]

    def _interpolate(self, radius):
        tol = 2e-6*self.outer_radius
        if (not np.isfinite(radius).all() or np.any(radius < self.inner_radius-tol)
                or np.any(radius > self.outer_radius+tol)):
            raise ValueError("Query outside concentric FEM shell")
        query = np.clip(radius, self.inner_radius, self.outer_radius)
        q = np.interp(query, self.r, self.q)
        index = np.clip(np.searchsorted(self.r, query, side="right")-1, 0, len(self.dq_dr)-1)
        return q, self.dq_dr[index]

    def value_and_radial_derivative(self, r):
        q, dq = self._interpolate(r.detach().cpu().numpy().astype(float))
        return (torch.as_tensor(q, dtype=r.dtype, device=r.device),
                torch.as_tensor(dq, dtype=r.dtype, device=r.device))

    def value_and_grad(self, x):
        y = x.detach().cpu().numpy().astype(float)-self.center
        radius = np.linalg.norm(y, axis=1)
        q, dq = self._interpolate(radius)
        grad = dq[:, None]*y/radius[:, None]
        return (torch.as_tensor(q[:, None], dtype=x.dtype, device=x.device),
                torch.as_tensor(grad, dtype=x.dtype, device=x.device))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, kind=self.kind, r=self.r, q=self.q,
                            config=json.dumps(self.config), parameters=json.dumps(self.parameters))

    def summary(self):
        return {"method": "radial_fem", "kind": self.kind,
                "num_elements": self.num_elements, "mesh_power": self.mesh_power,
                "inner_radius": self.inner_radius, "outer_radius": self.outer_radius,
                "robin_kappa": self.kappa, "q_min": float(self.q.min()), "q_max": float(self.q.max())}
