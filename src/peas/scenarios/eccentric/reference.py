"""Independent P1 finite-element reference for receiver first-hit probability.

Solve q_t = Delta q, q=1 on the receiver, d_n q + kappa q=0 on
reactive walls, and q=0 on pipe end caps. The latter truncate the
nominally infinite pipe and should be checked by extending the caps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree


def geometry(metadata: dict) -> dict:
    pipe = metadata.get("pipe_radius") is not None
    keys = (["rx_center", "rx_radius", "pipe_radius", "z_min", "z_max", "wall_kappa"] if pipe
            else ["rx_center", "rx_radius", "outer_center", "outer_radius", "kappa"])
    missing = [key for key in keys if metadata.get(key) is None]
    if missing:
        raise ValueError(f"Missing FEM geometry metadata: {missing}")
    if pipe and str(metadata.get("end_boundary", "absorbing")).lower() != "absorbing":
        raise ValueError("Pipe FEM requires absorbing end caps")
    result = {"kind": "pipe" if pipe else "shell", **{key: metadata[key] for key in keys}}
    if float(result["rx_radius"]) <= 0 or float(result["wall_kappa" if pipe else "kappa"]) < 0:
        raise ValueError("Receiver radius must be positive and wall reactivity nonnegative")
    if not pipe and (float(result["outer_radius"]) <= float(result["rx_radius"])):
        raise ValueError("Outer radius must exceed receiver radius")
    return result


def _radial_mesh(g: dict, elements: int) -> tuple[np.ndarray, csr_matrix, csr_matrix, np.ndarray, np.ndarray]:
    a, b = float(g["rx_radius"]), float(g["outer_radius"])
    # Quadratic grading resolves the sharp initial layer next to the receiver.
    r = a + (b - a) * np.linspace(0.0, 1.0, elements + 1) ** 2
    rows, cols, kvals, mvals = [], [], [], []
    xi = np.array([0.2113248654051871, 0.7886751345948129])
    for j in range(elements):
        h = r[j + 1] - r[j]
        kl = np.zeros((2, 2))
        ml = np.zeros((2, 2))
        for s in xi:
            radius = r[j] + h * s
            shape = np.array([1 - s, s])
            grad = np.array([-1 / h, 1 / h])
            kl += 0.5 * h * radius**2 * np.outer(grad, grad)
            ml += 0.5 * h * radius**2 * np.outer(shape, shape)
        for i in range(2):
            for k in range(2):
                rows.append(j + i)
                cols.append(j + k)
                kvals.append(kl[i, k])
                mvals.append(ml[i, k])
    kvals.append(float(g["kappa"]) * b**2)
    mvals.append(0.0)
    rows.append(elements)
    cols.append(elements)
    shape = (elements + 1, elements + 1)
    return r, coo_matrix((kvals, (rows, cols)), shape=shape).tocsr(), coo_matrix((mvals, (rows, cols)), shape=shape).tocsr(), np.array([0]), np.array([], dtype=int)


def _mesh_3d(g: dict, size: float):
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("absorption_fem")
        rx = np.asarray(g["rx_center"], dtype=float)
        if g["kind"] == "shell":
            center = np.asarray(g["outer_center"], dtype=float)
            if np.linalg.norm(rx - center) + float(g["rx_radius"]) >= float(g["outer_radius"]):
                raise ValueError("Receiver must be strictly inside the outer sphere")
            outer = gmsh.model.occ.addSphere(*center, float(g["outer_radius"]))
        else:
            radius, z0, z1 = float(g["pipe_radius"]), float(g["z_min"]), float(g["z_max"])
            if np.linalg.norm(rx[:2]) + float(g["rx_radius"]) >= radius or not z0 + g["rx_radius"] < rx[2] < z1 - g["rx_radius"]:
                raise ValueError("Receiver must be strictly inside the pipe")
            outer = gmsh.model.occ.addCylinder(0, 0, z0, 0, 0, z1-z0, radius)
        inner = gmsh.model.occ.addSphere(*rx, float(g["rx_radius"]))
        volume, _ = gmsh.model.occ.cut([(3, outer)], [(3, inner)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()
        if len(volume) != 1:
            raise RuntimeError("Gmsh did not produce one connected fluid volume")
        boundaries = gmsh.model.getBoundary(volume, oriented=False)
        groups = {"rx": [], "wall": [], "caps": []}
        for dim, tag in boundaries:
            if dim != 2:
                continue
            bb = gmsh.model.getBoundingBox(dim, tag)
            cm = np.array(gmsh.model.occ.getCenterOfMass(dim, tag))
            if g["kind"] == "pipe" and (abs(cm[2] - g["z_min"]) < 1e-6 or abs(cm[2] - g["z_max"]) < 1e-6):
                groups["caps"].append(tag)
            elif max(abs(bb[i] - (rx[i] - g["rx_radius"])) for i in range(3)) < 1e-5 and max(abs(bb[i+3] - (rx[i] + g["rx_radius"])) for i in range(3)) < 1e-5:
                groups["rx"].append(tag)
            else:
                groups["wall"].append(tag)
        if not groups["rx"] or not groups["wall"] or (g["kind"] == "pipe" and not groups["caps"]):
            raise RuntimeError(f"Could not classify mesh surfaces: {groups}")
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "FacesList", groups["rx"])
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", size / 4)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", size)
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", size / 2)
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", 3 * size)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size / 4)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.generate(3)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(xyz).reshape(-1, 3)
        index = {int(tag): i for i, tag in enumerate(tags)}
        def elements(dim, entities, code, width):
            chunks = []
            for entity in entities:
                types, _, nodes = gmsh.model.mesh.getElements(dim, entity)
                for typ, data in zip(types, nodes):
                    if typ == code:
                        chunks.append(np.fromiter((index[int(v)] for v in data), dtype=int).reshape(-1, width))
            return np.vstack(chunks) if chunks else np.empty((0, width), dtype=int)
        tets = elements(3, [volume[0][1]], 4, 4)
        faces = {name: elements(2, entities, 2, 3) for name, entities in groups.items()}
        return points, tets, faces
    finally:
        gmsh.finalize()


def _assemble_3d(points, tets, faces, kappa):
    rows, cols, kvals, mvals = [], [], [], []
    for tet in tets:
        matrix = np.ones((4, 4))
        matrix[:, 1:] = points[tet]
        volume = abs(np.linalg.det(matrix)) / 6
        grad = np.linalg.inv(matrix)[1:, :].T
        local_k = volume * grad @ grad.T
        local_m = volume / 20 * (np.ones((4, 4)) + np.eye(4))
        for i in range(4):
            for j in range(4):
                rows.append(tet[i])
                cols.append(tet[j])
                kvals.append(local_k[i,j])
                mvals.append(local_m[i,j])
    n = len(points)
    mass = coo_matrix((mvals, (rows, cols)), shape=(n,n)).tocsr()
    for face in faces["wall"]:
        v = points[face]
        area = np.linalg.norm(np.cross(v[1]-v[0], v[2]-v[0])) / 2
        local = kappa * area / 12 * (np.ones((3,3)) + np.eye(3))
        for i in range(3):
            for j in range(3):
                rows.append(face[i])
                cols.append(face[j])
                kvals.append(local[i,j])
    stiff = coo_matrix((kvals, (rows, cols)), shape=(n,n)).tocsr()
    return stiff, mass


def _sample_tets(points, tets, samples):
    verts = points[tets]
    tree = cKDTree(verts.mean(axis=1))
    result = []
    for point in samples:
        found = None
        for count in (32, 128, 512, 2048, len(tets)):
            _, indices = tree.query(point, k=min(count, len(tets)))
            for idx in np.atleast_1d(indices):
                v = verts[idx]
                bary = np.linalg.solve((v[1:] - v[0]).T, point - v[0])
                weights = np.r_[1-bary.sum(), bary]
                if weights.min() >= -1e-8:
                    found = (tets[idx], weights)
                    break
            if found is not None:
                break
        if found is None:
            raise ValueError(f"Sample lies outside FEM mesh: {point.tolist()}")
        result.append(found)
    return np.array([a for a, _ in result]), np.array([b for _, b in result])


def solve(g: dict, samples: np.ndarray, times: np.ndarray, *, mesh_size=0.5, dt=0.005, radial_elements=800) -> tuple[np.ndarray, dict]:
    samples = np.asarray(samples, dtype=float).reshape(-1, 3)
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or times[0] < 0 or np.any(np.diff(times) <= 0) or dt <= 0 or mesh_size <= 0 or radial_elements < 2:
        raise ValueError("Times must increase from nonnegative t; dt and mesh_size must be positive")
    radial = g["kind"] == "shell" and np.allclose(g["rx_center"], g["outer_center"])
    if radial:
        r, stiff, mass, rx_nodes, cap_nodes = _radial_mesh(g, radial_elements)
        radii = np.linalg.norm(samples - np.asarray(g["rx_center"]), axis=1)
        if np.any((radii < r[0]) | (radii > r[-1])):
            raise ValueError("Sample lies outside radial FEM domain")
        def sample(values):
            return np.interp(radii, r, values)
        node_count = len(r)
        element_count = radial_elements
    else:
        points, tets, faces = _mesh_3d(g, mesh_size)
        stiff, mass = _assemble_3d(points, tets, faces, float(g.get("kappa", g.get("wall_kappa"))))
        rx_nodes = np.unique(faces["rx"])
        cap_nodes = np.unique(faces["caps"])
        ids, weights = _sample_tets(points, tets, samples)
        def sample(values):
            return np.sum(values[ids] * weights, axis=1)
        node_count = len(points)
        element_count = len(tets)
    known = np.union1d(rx_nodes, cap_nodes)
    if np.intersect1d(rx_nodes, cap_nodes).size:
        raise ValueError("Receiver and cap nodes overlap")
    free = np.setdiff1d(np.arange(node_count), known)
    known_values = np.isin(known, rx_nodes).astype(float)
    forcing = -stiff[free][:, known] @ known_values
    mass = diags(np.asarray(mass.sum(axis=1)).ravel(), format="csr")
    m = mass[free][:, free].tocsc()
    k = stiff[free][:, free].tocsc()
    values = np.zeros(node_count)
    values[known] = known_values
    free_values = np.zeros(len(free))
    # The near-receiver first-hit layer evolves much faster than the late curve.
    # Fixed steps within each phase reuse one sparse factorization per phase.
    phases = ([(0.001, 2.5e-6), (0.01, 2.5e-5),
               (0.1, 2.5e-4), (1.0, 2.5e-3), (times[-1], dt)]
              if radial else [(times[-1], dt)])
    curve = np.empty((len(times), len(samples)))
    previous = np.zeros(len(samples))  # interior initial condition q(0, x0)=0
    current = 0.0
    target = 0
    while target < len(times) and times[target] == 0:
        curve[target] = previous
        target += 1
    smallest_step = float("inf")
    largest_step = 0.0
    step_count = 0
    for phase_end, preferred_step in phases:
        phase_end = min(float(phase_end), float(times[-1]))
        if phase_end <= current + 1e-14:
            continue
        count = int(np.ceil((phase_end - current) / min(dt, preferred_step)))
        step = (phase_end - current) / count
        advance = factorized((m + step*k).tocsc())
        phase_start = current
        for i in range(1, count + 1):
            free_values = advance(m @ free_values + step * forcing)
            values[free] = free_values
            next_values = sample(values)
            next_time = phase_start + i * step
            while target < len(times) and times[target] <= next_time + 1e-12:
                alpha = np.clip((times[target] - current) / step, 0, 1)
                curve[target] = (1-alpha)*previous + alpha*next_values
                target += 1
            previous = next_values
            current = next_time
        smallest_step = min(smallest_step, step)
        largest_step = max(largest_step, step)
        step_count += count
    diagnostics = {
        "nodes": node_count, "elements": element_count,
        "time_step": largest_step, "min_time_step": smallest_step,
        "time_steps": step_count,
        "mesh_size": None if radial else mesh_size,
        "radial_elements": radial_elements if radial else None,
        "min_probability": float(curve.min()), "max_probability": float(curve.max()),
        "min_temporal_increment": float(np.diff(curve, axis=0).min()),
    }
    return curve, diagnostics


def load_reference(path: Path, x0: np.ndarray, times: np.ndarray, metadata: dict | None = None) -> np.ndarray:
    payload = json.loads(Path(path).read_text())
    expected = geometry(metadata) if metadata is not None else None
    if expected is not None and payload["geometry"] != expected:
        raise ValueError(f"FEM geometry differs from comparison case: {path}")
    samples = np.asarray(payload["samples"], dtype=float)
    matching = np.flatnonzero(np.all(np.isclose(samples, x0, rtol=0, atol=1e-8), axis=1))
    if len(matching) != 1:
        raise ValueError(f"FEM result needs exactly one matching x0={x0.tolist()}: {path}")
    grid = np.asarray(payload["times"], dtype=float)
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
        raise ValueError("FEM output times must be finite and strictly increasing")
    if times.min() < grid[0]-1e-10 or times.max() > grid[-1]+1e-10:
        raise ValueError("Requested comparison times extend outside FEM output")
    return np.interp(times, grid, np.asarray(payload["q_fem"])[matching[0]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation-json", type=Path, action="append", default=[], help="Input case(s) with matching geometry; supplies sample x0 and output times")
    parser.add_argument("--simulation-dir", type=Path, help="Read all JSON case files in one geometry-matched directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float, help="Maximum time step; defaults to 0.01 for centered shells, 0.005 otherwise")
    parser.add_argument("--mesh-size", type=float, default=0.5)
    parser.add_argument("--radial-elements", type=int, default=800)
    parser.add_argument("--time-points", type=int, default=1001)
    parser.add_argument("--early-time", type=float, default=0.5, help="Dense output through this time")
    parser.add_argument("--early-dt", type=float, default=0.001, help="Maximum early-time output spacing")
    args = parser.parse_args()
    paths = [*args.simulation_json]
    if args.simulation_dir is not None:
        paths.extend(sorted(args.simulation_dir.glob("*.json")))
    if not paths:
        parser.error("provide --simulation-json or --simulation-dir")
    if args.time_points < 2:
        parser.error("--time-points must be at least 2")
    if args.early_time < 0 or args.early_dt <= 0:
        parser.error("--early-time must be nonnegative and --early-dt positive")
    samples_list = []
    tmax = 0.0
    g = None
    for path in paths:
        item = json.loads(path.read_text())
        current = geometry(item["metadata"])
        if g is None:
            g = current
        elif current != g:
            raise ValueError("All input cases must have identical geometry")
        samples_list.append(item["x0"])
        tmax = max(tmax, float(item["times"][-1]))
    samples = np.asarray(samples_list, dtype=float)
    assert g is not None
    uniform_times = np.linspace(0, tmax, args.time_points)
    early_end = min(tmax, args.early_time)
    early_times = np.linspace(0, early_end, max(2, int(np.ceil(early_end / args.early_dt)) + 1)) if early_end > 0 else np.array([0.0])
    times = np.unique(np.r_[uniform_times, early_times])
    centered = g["kind"] == "shell" and np.allclose(g["rx_center"], g["outer_center"])
    dt = args.dt if args.dt is not None else (0.01 if centered else 0.005)
    curves, diagnostics = solve(g, samples, times, mesh_size=args.mesh_size, dt=dt, radial_elements=args.radial_elements)
    result = {"method": "P1 FEM implicit Euler", "geometry": g, "samples": samples.tolist(), "times": times.tolist(), "q_fem": curves.T.tolist(), "diagnostics": diagnostics, "inputs": [str(p) for p in paths]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(f"[transient-fem] {args.output} {diagnostics}")


if __name__ == "__main__":
    main()
