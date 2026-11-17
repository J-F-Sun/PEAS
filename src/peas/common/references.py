"""Transient FEM references for the fixed D=1, receiver-radius=1 scenes.

Radial P1, axisymmetric bispherical Q1, and tetrahedral P1 Galerkin systems
advance M dq/dt + K q = 0. Constant RX Dirichlet data are eliminated;
Robin walls enter K. Crank--Nicolson uses Rannacher start-up (two BE half
steps). This module never uses a surrogate prediction as a reference.

Quadrature/geometry conventions follow peas/fem/bispherical.py and the
historical peas_4_insphere/transient_fem.py; no old project imports are needed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from scipy.special import erfc


@dataclass(frozen=True)
class FEMSettings:
    dt: float = .01
    radial_elements: int = 1600
    n_xi: int = 96
    n_eta: int = 128

    def validate(self):
        if not np.isfinite(self.dt) or self.dt <= 0 or self.radial_elements < 8 or min(self.n_xi, self.n_eta) < 8:
            raise ValueError("FEM requires finite dt>0 and at least eight elements/nodes per direction")
        return self


def canonical_analytic(samples, times, geometry):
    radius = np.linalg.norm(samples-np.asarray(geometry["rx_center"]), axis=1)
    result = np.zeros((len(samples), len(times)))
    positive = times > 0
    result[:, positive] = erfc((radius[:, None]-1)/np.sqrt(4*times[None, positive]))/radius[:, None]
    return result


@lru_cache(maxsize=8)
def sphere_series_data(radius, kappa, terms):
    length, beta = radius-1., kappa-1./radius
    def equation(mu):
        return mu*np.cos(mu*length)+beta*np.sin(mu*length)
    grid = np.linspace(1e-8, (terms+8)*np.pi/length, max(2000, 300*terms))
    values = equation(grid)
    roots = [brentq(equation, left, right) for left, right, a, b in zip(grid[:-1], grid[1:], values[:-1], values[1:]) if a*b < 0][:terms]
    if len(roots) != terms:
        raise RuntimeError("Insufficient centered Robin eigenvalues")
    mu = np.array(roots)
    b = -kappa/(-1/radius**2+kappa*(1/radius-1))
    a = 1-b
    ml = mu*length
    coefficient = ((1-np.cos(ml))/mu+a*(-length*np.cos(ml)/mu+np.sin(ml)/mu**2))/(length/2-np.sin(2*ml)/(4*mu))
    return mu, coefficient, a, b


def concentric_analytic(samples, times, geometry, terms=320):
    """Robin eigenfunction series, exact interior q(0)=0, no probability clipping."""
    r = np.linalg.norm(samples-np.asarray(geometry["rx_center"]), axis=1)
    mu, coefficient, a, b = sphere_series_data(float(geometry["outer_radius"]), float(geometry["kappa"]), terms)
    result = a+b/r[:, None] - (np.sin((r[:, None]-1)*mu)*coefficient/r[:, None]) @ np.exp(-mu[:, None]**2*times[None, :])
    result[:, times == 0] = 0
    return result


def assemble_radial(nodes, *, spherical, outer_kappa=0.):
    n = len(nodes)
    cells = np.column_stack((np.arange(n-1), np.arange(1, n)))
    h = np.diff(nodes)
    mass = np.zeros((n-1, 2, 2))
    stiff = np.zeros_like(mass)
    gradient = np.column_stack((-1/h, 1/h))
    for s in (.5-.5/np.sqrt(3), .5+.5/np.sqrt(3)):
        shape = np.array([1-s, s])
        r = nodes[:-1]+h*s
        weight = .5*h*(r**2 if spherical else 1)
        mass += weight[:, None, None]*np.outer(shape, shape)
        stiff += weight[:, None, None]*gradient[:, :, None]*gradient[:, None, :]
    rows = np.repeat(cells, 2, axis=1).ravel()
    cols = np.tile(cells, (1, 2)).ravel()
    m = coo_matrix((mass.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    k = coo_matrix((stiff.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    if outer_kappa:
        k += coo_matrix(([outer_kappa*nodes[-1]**2], ([n-1], [n-1])), shape=(n, n)).tocsr()
    return m, k


def assemble_bispherical(geometry, settings):
    from peas.scenarios.eccentric.steady import bispherical_geometry, shape_q1
    geom = bispherical_geometry({"domain": geometry})
    xi = np.linspace(geom.xi_rx, geom.xi_outer, settings.n_xi)
    eta = np.linspace(0, np.pi, settings.n_eta)
    ii, jj = np.meshgrid(np.arange(len(xi)-1), np.arange(len(eta)-1), indexing="ij")
    i, j = ii.ravel(), jj.ravel()
    cells = np.column_stack((i*len(eta)+j, (i+1)*len(eta)+j, (i+1)*len(eta)+j+1, i*len(eta)+j+1))
    dx, de = xi[1]-xi[0], eta[1]-eta[0]
    mass, stiff = np.zeros((len(cells), 4, 4)), np.zeros((len(cells), 4, 4))
    for s in (-1/np.sqrt(3), 1/np.sqrt(3)):
        xq = .5*(xi[i]+xi[i+1])+.5*dx*s
        for t in (-1/np.sqrt(3), 1/np.sqrt(3)):
            eq = .5*(eta[j]+eta[j+1])+.5*de*t
            shape, ds, dt = shape_q1(s, t)
            h = geom.a/(np.cosh(xq)-np.cos(eq))
            measure = 2*np.pi*np.sin(eq)*.25*dx*de
            mass += (measure*h**3)[:, None, None]*np.outer(shape, shape)
            stiff += (measure*h)[:, None, None]*(np.outer(ds*2/dx, ds*2/dx)+np.outer(dt*2/de, dt*2/de))
    n = len(xi)*len(eta)
    rows, cols = np.repeat(cells, 4, axis=1).ravel(), np.tile(cells, (1, 4)).ravel()
    m = coo_matrix((mass.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    k = coo_matrix((stiff.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    wall = np.column_stack(((len(xi)-1)*len(eta)+np.arange(len(eta)-1), (len(xi)-1)*len(eta)+np.arange(1, len(eta))))
    local = np.zeros((len(wall), 2, 2))
    for s in (-1/np.sqrt(3), 1/np.sqrt(3)):
        eq = .5*(eta[:-1]+eta[1:])+.5*de*s
        shape = np.array([.5*(1-s), .5*(1+s)])
        h = geom.a/(np.cosh(geom.xi_outer)-np.cos(eq))
        weight = float(geometry["kappa"])*2*np.pi*h*h*np.sin(eq)*.5*de
        local += weight[:, None, None]*np.outer(shape, shape)
    k += coo_matrix((local.ravel(), (np.repeat(wall, 2, axis=1).ravel(), np.tile(wall, (1, 2)).ravel())), shape=(n, n)).tocsr()
    return geom, xi, eta, m, k


def bispherical_sampler(samples, geom, xi, eta):
    p = samples-geom.outer_center
    z = p @ geom.axis - geom.z_shift
    rho = np.linalg.norm(p - (p @ geom.axis)[:, None]*geom.axis, axis=1)
    x = .5*np.log((rho*rho+(z+geom.a)**2)/(rho*rho+(z-geom.a)**2))
    e = np.arctan2(2*geom.a*rho, rho*rho+z*z-geom.a**2)
    e = np.where(e < 0, e+np.pi, e)
    if np.any(x < xi[0]-1e-9) or np.any(x > xi[-1]+1e-9):
        raise ValueError("Query outside bispherical FEM domain")
    ix = np.clip(np.searchsorted(xi, x)-1, 0, len(xi)-2)
    ie = np.clip(np.searchsorted(eta, e)-1, 0, len(eta)-2)
    a, b = (x-xi[ix])/(xi[ix+1]-xi[ix]), (e-eta[ie])/(eta[ie+1]-eta[ie])
    cells = np.column_stack((ix*len(eta)+ie, (ix+1)*len(eta)+ie, (ix+1)*len(eta)+ie+1, ix*len(eta)+ie+1))
    weight = np.column_stack(((1-a)*(1-b), a*(1-b), a*b, (1-a)*b))
    return lambda values: np.sum(values[cells]*weight, axis=1)


def assemble_tetrahedral(points, tets, wall_faces, kappa):
    matrix = np.ones((len(tets), 4, 4))
    matrix[:, :, 1:] = points[tets]
    volume = np.abs(np.linalg.det(matrix))/6
    grad = np.linalg.inv(matrix)[:, 1:, :].transpose(0, 2, 1)
    stiff = volume[:, None, None]*(grad @ grad.transpose(0, 2, 1))
    mass = volume[:, None, None]/20*(np.ones((4, 4))+np.eye(4))
    n = len(points)
    rows, cols = np.repeat(tets, 4, axis=1).ravel(), np.tile(tets, (1, 4)).ravel()
    m = coo_matrix((mass.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    k = coo_matrix((stiff.ravel(), (rows, cols)), shape=(n, n)).tocsr()
    vertices = points[wall_faces]
    area = np.linalg.norm(np.cross(vertices[:, 1]-vertices[:, 0], vertices[:, 2]-vertices[:, 0]), axis=1)/2
    local = float(kappa)*area[:, None, None]/12*(np.ones((3, 3))+np.eye(3))
    k += coo_matrix((local.ravel(), (np.repeat(wall_faces, 3, axis=1).ravel(), np.tile(wall_faces, (1, 3)).ravel())), shape=(n, n)).tocsr()
    return m, k


def tetrahedral_sampler(points, tets, samples):
    vertices = points[tets]
    inverse = np.linalg.inv((vertices[:, 1:]-vertices[:, :1]).transpose(0, 2, 1))
    tree = cKDTree(vertices.mean(axis=1))
    cells, weights = [], []
    for point in samples:
        found = False
        for count in (32, 128, 512, 2048, len(tets)):
            _, indices = tree.query(point, k=min(count, len(tets)))
            indices = np.atleast_1d(indices)
            local = np.einsum("kij,kj->ki", inverse[indices], point-vertices[indices, 0])
            bary = np.column_stack((1-local.sum(axis=1), local))
            hits = np.flatnonzero(bary.min(axis=1) >= -1e-9)
            if len(hits):
                hit = hits[0]
                cells.append(tets[indices[hit]])
                weights.append(bary[hit])
                found = True
                break
        if not found:
            raise ValueError(f"Query is outside the tetrahedral FEM mesh: {point.tolist()}")
    cells, weights = np.array(cells), np.array(weights)
    return lambda values: np.sum(values[cells]*weights, axis=1)


def advance(mass, stiff, known, known_values, sample, times, max_dt):
    """Sample a consistent-mass CN solution without storing nodal trajectories."""
    n = mass.shape[0]
    free = np.setdiff1d(np.arange(n), known)
    m, k = mass[free][:, free].tocsc(), stiff[free][:, free].tocsc()
    force = -stiff[free][:, known] @ known_values
    steps = max(1, int(np.ceil(times[-1]/max_dt)))
    dt = times[-1]/steps
    full, state = np.zeros(n), np.zeros(len(free))
    full[known] = known_values
    previous = np.zeros(len(sample(full)))
    output = np.empty((len(previous), len(times)))
    cursor = 0
    while cursor < len(times) and times[cursor] == 0:
        output[:, cursor] = previous
        cursor += 1
    if dt == 0:
        return output, {"time_steps": 0, "dt": 0.}
    lhs = splu((m+.5*dt*k).tocsc())
    rhs = m-.5*dt*k
    for index in range(1, steps+1):
        if index == 1:
            for _ in range(2):
                state = lhs.solve(m @ state+.5*dt*force)
        else:
            state = lhs.solve(rhs @ state+dt*force)
        full[free] = state
        current = sample(full)
        left, right = (index-1)*dt, index*dt
        while cursor < len(times) and times[cursor] <= right+1e-11:
            alpha = np.clip((times[cursor]-left)/dt, 0, 1)
            output[:, cursor] = previous+alpha*(current-previous)
            cursor += 1
        previous = current
        if steps > 1000 and index % max(1, steps//4) == 0:
            print(f"  FEM time integration {index}/{steps}", flush=True)
    if cursor != len(times) or not np.isfinite(output).all():
        raise RuntimeError("Incomplete or nonfinite transient FEM result")
    return output, {"time_steps": steps, "dt": dt, "nodes": n, "free_nodes": len(free),
                    "method": "consistent mass FEM; Crank-Nicolson with two initial BE half steps",
                    "minimum": float(output.min()), "maximum": float(output.max())}


def solve_fem(scene, samples, times, geometry, settings=FEMSettings(), pipe_mesh: Path | None = None):
    settings.validate()
    if scene in ("canonical", "concentric"):
        radius = np.linalg.norm(samples-np.asarray(geometry["rx_center"]), axis=1)
        outer = float(radius.max()+max(8*np.sqrt(times[-1]), 2)) if scene == "canonical" else float(geometry["outer_radius"])
        nodes = 1+(outer-1)*np.linspace(0, 1, settings.radial_elements+1)**1.6
        spherical = scene == "concentric"
        m, k = assemble_radial(nodes, spherical=spherical, outer_kappa=float(geometry.get("kappa", 0)))
        known = np.array([0, len(nodes)-1]) if scene == "canonical" else np.array([0])
        values = np.array([1., 0.]) if scene == "canonical" else np.array([1.])
        def sample(v):
            q = np.interp(radius, nodes, v)
            return q if spherical else q/radius
        extra = {"elements": settings.radial_elements, "outer_radius": outer, "space": "radial P1"}
    elif scene == "eccentric":
        geom, xi, eta, m, k = assemble_bispherical(geometry, settings)
        sample = bispherical_sampler(samples, geom, xi, eta)
        known, values = np.arange(len(eta)), np.ones(len(eta))
        extra = {"n_xi": len(xi), "n_eta": len(eta), "space": "axisymmetric bispherical Q1", "mesh_generated_independently": True}
    elif scene == "pipe":
        if pipe_mesh is None:
            raise ValueError("Pipe transient FEM needs a tetrahedral mesh asset")
        import json
        with np.load(pipe_mesh, allow_pickle=False) as z:
            native = json.loads(str(z["config"]))
            for key in ("rx_center", "rx_radius", "pipe_radius", "z_min", "z_max"):
                if not np.allclose(native["domain"][key], geometry[key], atol=1e-10, rtol=0):
                    raise ValueError(f"Pipe FEM mesh geometry differs: {key}")
            if not np.isclose(native["boundary"].get("wall_kappa", native["boundary"].get("kappa")), geometry["wall_kappa"]):
                raise ValueError("Pipe FEM wall coefficient differs")
            points, tets = z["points"].copy(), z["tetrahedra"].copy()
            wall = z["wall_faces"].copy()
            rx, caps = np.unique(z["rx_faces"]), np.unique(z["cap_faces"])
        m, k = assemble_tetrahedral(points, tets, wall, geometry["wall_kappa"])
        sample = tetrahedral_sampler(points, tets, samples)
        known = np.union1d(rx, caps)
        values = np.isin(known, rx).astype(float)
        extra = {"elements": len(tets), "mesh_asset": str(pipe_mesh), "space": "tetrahedral P1", "stored_steady_q_used": False}
    else:
        raise ValueError(f"Unknown scene {scene}")
    q, diagnostics = advance(m, k, known, values, sample, times, settings.dt)
    return q, {**diagnostics, **extra, "settings": asdict(settings)}
