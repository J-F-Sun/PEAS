from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch


class QInfinityProvider(Protocol):
    """Frozen steady-state field used by the normalized PEAS ansatz."""

    mode: str
    y_min: float

    def value_and_dy(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return q_infty(y) and d q_infty / dy without autograd through FEM."""

    def summary(self) -> dict[str, float | int | str]:
        ...


@dataclass(frozen=True)
class AnalyticQInfinityProvider:
    mode: str = "analytic"
    y_min: float = 0.0

    def value_and_dy(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return y, torch.ones_like(y)

    def summary(self) -> dict[str, float | int | str]:
        return {"mode": self.mode, "description": "q_infty(y) = y = 1/r"}


class RadialFEMQInfinityProvider:
    """P1 FEM solution of the radial steady-state exterior absorption problem.

    The finite shell [1, R] uses ``dq/dr + q/R = 0`` at r=R.  This is the
    exact Robin far-field condition for the harmonic spherical tail C/r.
    """

    mode = "fem"

    def __init__(self, config: dict[str, Any]) -> None:
        cfg = config.get("q_infty", {})
        self.outer_radius = float(cfg.get("outer_radius", 48.0))
        self.num_elements = int(cfg.get("num_elements", 256))
        self.mesh_power = float(cfg.get("mesh_power", 2.0))
        self.kappa = float(cfg.get("robin_kappa", 1.0 / self.outer_radius))
        if self.outer_radius <= 1.0:
            raise ValueError("q_infty.outer_radius must exceed the unit RX radius.")
        if self.num_elements < 2:
            raise ValueError("q_infty.num_elements must be at least 2.")
        if self.kappa <= 0.0:
            raise ValueError("q_infty.robin_kappa must be positive.")
        if self.mesh_power <= 0.0:
            raise ValueError("q_infty.mesh_power must be positive.")
        self.y_min = 1.0 / self.outer_radius
        self._r, self._q = self._solve()
        self._dq_dr = np.diff(self._q) / np.diff(self._r)

    def _solve(self) -> tuple[np.ndarray, np.ndarray]:
        # Refine near the receiver, where the spherical tail has the largest
        # curvature, while retaining a coarse and inexpensive far-field mesh.
        coordinate = np.linspace(0.0, 1.0, self.num_elements + 1, dtype=float)
        r = 1.0 + (self.outer_radius - 1.0) * coordinate**self.mesh_power
        # Tridiagonal assembly preserves the original P1 weak formulation.
        # O(N) storage avoids allocating a dense 2049x2049 matrix on the laptop.
        from scipy.linalg import solve_banded
        diagonal = np.zeros(self.num_elements + 1, dtype=float)
        off = np.zeros(self.num_elements, dtype=float)
        for element in range(self.num_elements):
            left, right = r[element], r[element + 1]
            h = right - left
            integral = (right**3 - left**3) / (3.0 * h**2)
            diagonal[element] += integral
            diagonal[element+1] += integral
            off[element] = -integral
        diagonal[-1] += self.outer_radius**2 * self.kappa
        rhs = np.zeros(self.num_elements, dtype=float)
        rhs[0] = -off[0]
        band = np.zeros((3, self.num_elements), dtype=float)
        band[1] = diagonal[1:]
        band[0, 1:] = off[1:]
        band[2, :-1] = off[1:]
        values = np.empty(self.num_elements + 1, dtype=float)
        values[0] = 1.0
        values[1:] = solve_banded((1, 1), band, rhs)
        return r, values

    def value_and_dy(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y_np = y.detach().cpu().numpy().reshape(-1)
        r_query = 1.0 / np.clip(y_np, self.y_min, 1.0)
        q = np.interp(r_query, self._r, self._q)
        ids = np.searchsorted(self._r, r_query, side="right") - 1
        ids = np.clip(ids, 0, self._dq_dr.size - 1)
        # dy/dr = -1/r^2, hence dq/dy = -r^2 dq/dr.
        dq_dy = -(r_query**2) * self._dq_dr[ids]
        shape = y.shape
        return (
            torch.as_tensor(q.reshape(shape), dtype=y.dtype, device=y.device),
            torch.as_tensor(dq_dy.reshape(shape), dtype=y.dtype, device=y.device),
        )

    def summary(self) -> dict[str, float | int | str]:
        exact = 1.0 / self._r
        error = np.abs(self._q - exact)
        return {
            "mode": self.mode,
            "outer_radius": self.outer_radius,
            "num_elements": self.num_elements,
            "mesh_power": self.mesh_power,
            "robin_kappa": self.kappa,
            "q_infty_max_abs_error": float(error.max()),
            "q_infty_l2_error": float(np.sqrt(np.mean(error**2))),
        }


def build_q_infinity_provider(config: dict[str, Any]) -> QInfinityProvider:
    mode = str(config.get("q_infty", {}).get("mode", "analytic")).lower()
    if mode == "analytic":
        return AnalyticQInfinityProvider()
    if mode == "fem":
        return RadialFEMQInfinityProvider(config)
    raise ValueError(f"Unsupported q_infty.mode: {mode}")
