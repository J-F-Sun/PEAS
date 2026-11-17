from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from peas.common.utils import save_json, timestamp


def ensure_output_dirs(config: dict[str, Any]) -> None:
    for value in config["outputs"].values():
        Path(value).mkdir(parents=True, exist_ok=True)


def _surface_groups(config: dict[str, Any], volume_tag: int) -> dict[str, list[int]]:
    radius = float(config["domain"]["pipe_radius"])
    z_min = float(config["domain"]["z_min"])
    z_max = float(config["domain"]["z_max"])
    tol = float(config["mesh"].get("surface_classification_tol", 1.0e-6))
    groups = {"rx": [], "wall": [], "caps": []}
    for dim, tag in gmsh.model.getBoundary([(3, volume_tag)], oriented=False, recursive=False):
        if dim != 2:
            continue
        center = np.asarray(gmsh.model.occ.getCenterOfMass(dim, tag), dtype=float)
        bbox = np.asarray(gmsh.model.getBoundingBox(dim, tag), dtype=float)
        spans_full_pipe = (
            bbox[0] <= -radius + tol
            and bbox[1] <= -radius + tol
            and bbox[3] >= radius - tol
            and bbox[4] >= radius - tol
            and bbox[2] <= z_min + tol
            and bbox[5] >= z_max - tol
        )
        if abs(center[2] - z_min) <= tol or abs(center[2] - z_max) <= tol:
            groups["caps"].append(tag)
        elif spans_full_pipe:
            groups["wall"].append(tag)
        else:
            groups["rx"].append(tag)
    if any(not tags for tags in groups.values()):
        raise RuntimeError(f"Failed to classify Gmsh surfaces: {groups}")
    return groups


def generate_gmsh_mesh(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], Path]:
    global gmsh
    import gmsh
    domain = config["domain"]
    mesh_cfg = config["mesh"]
    radius = float(domain["pipe_radius"])
    z_min = float(domain["z_min"])
    z_max = float(domain["z_max"])
    rx_center = [float(value) for value in domain["rx_center"]]
    rx_radius = float(domain["rx_radius"])
    if z_min >= z_max:
        raise ValueError("domain.z_min must be smaller than domain.z_max.")
    if np.hypot(rx_center[0], rx_center[1]) + rx_radius >= radius:
        raise ValueError("The RX sphere must lie strictly inside the pipe wall.")
    if not z_min + rx_radius < rx_center[2] < z_max - rx_radius:
        raise ValueError("The RX sphere must lie strictly between the pipe caps.")

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", float(bool(mesh_cfg.get("terminal_output", False))))
        gmsh.model.add("pipe_3d_gmsh_fem_qinf")
        pipe = gmsh.model.occ.addCylinder(0.0, 0.0, z_min, 0.0, 0.0, z_max - z_min, radius)
        receiver = gmsh.model.occ.addSphere(*rx_center, rx_radius)
        cut, _ = gmsh.model.occ.cut([(3, pipe)], [(3, receiver)], removeObject=True, removeTool=True)
        gmsh.model.occ.synchronize()
        if len(cut) != 1 or cut[0][0] != 3:
            raise RuntimeError(f"Expected one cut pipe volume, got {cut}")
        volume_tag = cut[0][1]
        groups = _surface_groups(config, volume_tag)
        for name, dim, tags in [
            ("volume", 3, [volume_tag]),
            ("rx", 2, groups["rx"]),
            ("wall", 2, groups["wall"]),
            ("caps", 2, groups["caps"]),
        ]:
            physical = gmsh.model.addPhysicalGroup(dim, tags)
            gmsh.model.setPhysicalName(dim, physical, name)

        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "FacesList", groups["rx"])
        threshold = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
        gmsh.model.mesh.field.setNumber(threshold, "SizeMin", float(mesh_cfg["size_min"]))
        gmsh.model.mesh.field.setNumber(threshold, "SizeMax", float(mesh_cfg["size_max"]))
        gmsh.model.mesh.field.setNumber(threshold, "DistMin", float(mesh_cfg["distance_min"]))
        gmsh.model.mesh.field.setNumber(threshold, "DistMax", float(mesh_cfg["distance_max"]))
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
        gmsh.option.setNumber("Mesh.MeshSizeMin", float(mesh_cfg["size_min"]))
        gmsh.option.setNumber("Mesh.MeshSizeMax", float(mesh_cfg["size_max"]))
        gmsh.option.setNumber("Mesh.ElementOrder", 1.0)
        gmsh.option.setNumber("Mesh.RandomSeed", float(config.get("project", {}).get("seed", 0)))
        gmsh.model.mesh.generate(3)

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        points = np.asarray(coordinates, dtype=float).reshape(-1, 3)
        tag_to_index = {int(tag): index for index, tag in enumerate(node_tags)}

        def elements(dim: int, tags: list[int], element_type: int, width: int) -> np.ndarray:
            chunks: list[np.ndarray] = []
            for entity_tag in tags:
                types, _, node_chunks = gmsh.model.mesh.getElements(dim, entity_tag)
                for current_type, current_nodes in zip(types, node_chunks, strict=True):
                    if int(current_type) == element_type:
                        mapped = np.fromiter((tag_to_index[int(tag)] for tag in current_nodes), dtype=int)
                        chunks.append(mapped.reshape(-1, width))
            if not chunks:
                raise RuntimeError(f"No Gmsh elements type={element_type} found for dim={dim}, tags={tags}")
            return np.vstack(chunks)

        tetrahedra = elements(3, [volume_tag], element_type=4, width=4)
        faces = {
            "rx": elements(2, groups["rx"], element_type=2, width=3),
            "wall": elements(2, groups["wall"], element_type=2, width=3),
            "caps": elements(2, groups["caps"], element_type=2, width=3),
        }
        mesh_path = Path(config["outputs"]["meshes_dir"]) / f"pipe_3d_gmsh_{timestamp()}.msh"
        gmsh.write(str(mesh_path))
        print(f"[pipe-3d-gmsh] mesh={mesh_path}", flush=True)
        return points, tetrahedra, faces, mesh_path
    finally:
        gmsh.finalize()


def tet_gradients(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    matrix = np.ones((4, 4), dtype=float)
    matrix[:, 1:] = vertices
    determinant = np.linalg.det(matrix)
    volume = abs(determinant) / 6.0
    return np.linalg.inv(matrix)[1:, :].T, volume


def assemble_system(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    faces: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[csr_matrix, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(points.shape[0], dtype=float)
    for tet in tetrahedra:
        gradients, volume = tet_gradients(points[tet])
        if volume <= 1.0e-15:
            continue
        local = volume * gradients @ gradients.T
        for local_i, row in enumerate(tet):
            for local_j, col in enumerate(tet):
                rows.append(int(row))
                cols.append(int(col))
                data.append(float(local[local_i, local_j]))

    kappa = float(config["boundary"]["kappa"])
    for face in faces["wall"]:
        vertices = points[face]
        area = 0.5 * np.linalg.norm(np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0]))
        local = kappa * area / 12.0 * (np.ones((3, 3), dtype=float) + np.eye(3))
        for local_i, row in enumerate(face):
            for local_j, col in enumerate(face):
                rows.append(int(row))
                cols.append(int(col))
                data.append(float(local[local_i, local_j]))
    matrix = coo_matrix((data, (rows, cols)), shape=(points.shape[0], points.shape[0])).tocsr()
    return matrix, rhs


def solve_dirichlet(
    matrix: csr_matrix,
    rhs: np.ndarray,
    rx_nodes: np.ndarray,
    cap_nodes: np.ndarray,
) -> tuple[np.ndarray, int]:
    constrained = np.zeros(matrix.shape[0], dtype=bool)
    constrained[rx_nodes] = True
    constrained[cap_nodes] = True
    free = np.flatnonzero(~constrained)
    known = np.flatnonzero(constrained)
    known_values = np.zeros(known.shape[0], dtype=float)
    known_values[np.isin(known, rx_nodes)] = 1.0
    reduced_rhs = rhs[free] - matrix[free][:, known] @ known_values
    solution = np.zeros(matrix.shape[0], dtype=float)
    solution[known] = known_values
    solution[free] = spsolve(matrix[free][:, free], reduced_rhs)
    return np.clip(solution, 0.0, 1.0), int(free.size)


def element_gradients(points: np.ndarray, tetrahedra: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.empty((tetrahedra.shape[0], 3), dtype=float)
    for index, tet in enumerate(tetrahedra):
        gradients, _ = tet_gradients(points[tet])
        result[index] = values[tet] @ gradients
    return result


def element_volumes(points: np.ndarray, tetrahedra: np.ndarray) -> np.ndarray:
    vertices = points[tetrahedra]
    edges = np.stack(
        [vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0], vertices[:, 3] - vertices[:, 0]],
        axis=2,
    )
    return np.abs(np.linalg.det(edges)) / 6.0


def recover_node_gradients(
    points: np.ndarray,
    tetrahedra: np.ndarray,
    gradients: np.ndarray,
    volumes: np.ndarray | None = None,
) -> np.ndarray:
    """Recover a continuous P1 gradient field by volume-weighted nodal averaging."""
    if volumes is None:
        volumes = element_volumes(points, tetrahedra)
    weighted_sum = np.zeros((points.shape[0], 3), dtype=float)
    total_weight = np.zeros(points.shape[0], dtype=float)
    for local_index in range(tetrahedra.shape[1]):
        nodes = tetrahedra[:, local_index]
        np.add.at(weighted_sum, nodes, gradients * volumes[:, None])
        np.add.at(total_weight, nodes, volumes)
    if np.any(total_weight <= 0.0):
        raise RuntimeError("Gradient recovery found nodes without adjacent tetrahedra.")
    return weighted_sum / total_weight[:, None]


class Pipe3DGmshFEMProvider:
    def __init__(self, checkpoint: str | Path, *, gradient_method: str = "volume_weighted_nodal") -> None:
        payload = np.load(checkpoint)
        self.points = np.asarray(payload["points"], dtype=float)
        self.tetrahedra = np.asarray(payload["tetrahedra"], dtype=int)
        self.q = np.asarray(payload["q"], dtype=float)
        self.element_gradients = np.asarray(payload["element_gradients"], dtype=float)
        self.element_volumes = (
            np.asarray(payload["element_volumes"], dtype=float)
            if "element_volumes" in payload
            else element_volumes(self.points, self.tetrahedra)
        )
        self.recovered_node_gradients = (
            np.asarray(payload["recovered_node_gradients"], dtype=float)
            if "recovered_node_gradients" in payload
            else recover_node_gradients(self.points, self.tetrahedra, self.element_gradients, self.element_volumes)
        )
        self.gradient_method = gradient_method
        self.centroids = self.points[self.tetrahedra].mean(axis=1)
        self.tree = cKDTree(self.centroids)
        vertices = self.points[self.tetrahedra]
        self.origins = vertices[:, 0]
        self.inverse_edges = np.linalg.inv(
            np.stack([vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0], vertices[:, 3] - vertices[:, 0]], axis=2)
        )
        self.point_tree = cKDTree(self.points)

    def value_and_grad(
        self,
        x: np.ndarray,
        *,
        candidates: int = 24,
        gradient_method: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        method = gradient_method or self.gradient_method
        if method not in {"raw_element", "volume_weighted_nodal"}:
            raise ValueError(f"Unsupported gradient recovery method: {method}")
        query = np.asarray(x, dtype=float).reshape(-1, 3)
        k = min(max(1, candidates), self.tetrahedra.shape[0])
        _, candidate_ids = self.tree.query(query, k=k)
        candidate_ids = np.asarray(candidate_ids, dtype=int).reshape(query.shape[0], -1)
        values = np.empty((query.shape[0], 1), dtype=float)
        gradients = np.empty((query.shape[0], 3), dtype=float)
        fallback = np.zeros(query.shape[0], dtype=bool)
        for row, point in enumerate(query):
            found = -1
            weights = None
            ids = candidate_ids[row]
            searched = k
            while True:
                for tet_id in ids:
                    local = self.inverse_edges[tet_id] @ (point - self.origins[tet_id])
                    barycentric = np.concatenate([[1.0 - local.sum()], local])
                    if np.min(barycentric) >= -1.0e-9 and np.max(barycentric) <= 1.0 + 1.0e-9:
                        found = int(tet_id)
                        weights = barycentric
                        break
                if found >= 0:
                    break
                # A containing tetrahedron need not have one of the nearest
                # centroids for elongated elements near a curved boundary.
                # Expand only failed queries, preserving the fast common path.
                if searched >= self.tetrahedra.shape[0]:
                    break
                searched = min(4 * searched, self.tetrahedra.shape[0])
                _, ids = self.tree.query(point, k=searched)
                ids = np.atleast_1d(np.asarray(ids, dtype=int))
            if found < 0:
                fallback[row] = True
                nearest = int(self.point_tree.query(point)[1])
                values[row, 0] = self.q[nearest]
                gradients[row] = self.recovered_node_gradients[nearest] if method == "volume_weighted_nodal" else 0.0
            else:
                values[row, 0] = float(weights @ self.q[self.tetrahedra[found]])
                gradients[row] = (
                    weights @ self.recovered_node_gradients[self.tetrahedra[found]]
                    if method == "volume_weighted_nodal"
                    else self.element_gradients[found]
                )
        return values, gradients, fallback



def solve(config: dict[str, Any]) -> dict[str, Any]:
    ensure_output_dirs(config)
    points, tetrahedra, faces, mesh_path = generate_gmsh_mesh(config)
    matrix, rhs = assemble_system(points, tetrahedra, faces, config)
    rx_nodes = np.unique(faces["rx"])
    cap_nodes = np.unique(faces["caps"])
    values, unknowns = solve_dirichlet(matrix, rhs, rx_nodes, cap_nodes)
    gradients = element_gradients(points, tetrahedra, values)
    volumes = element_volumes(points, tetrahedra)
    recovered_gradients = recover_node_gradients(points, tetrahedra, gradients, volumes)
    stamp = timestamp()
    checkpoint = Path(config["outputs"]["checkpoints_dir"]) / f"pipe_3d_gmsh_q_infty_fem_{stamp}.npz"
    np.savez_compressed(
        checkpoint,
        points=points,
        tetrahedra=tetrahedra,
        q=values,
        element_gradients=gradients,
        element_volumes=volumes,
        recovered_node_gradients=recovered_gradients,
        rx_faces=faces["rx"],
        wall_faces=faces["wall"],
        cap_faces=faces["caps"],
        config=json.dumps(config),
    )
    gradient_method = str(config.get("gradient_recovery", {}).get("method", "volume_weighted_nodal"))
    provider = Pipe3DGmshFEMProvider(checkpoint, gradient_method=gradient_method)
    samples = []
    for x0 in config.get("diagnostics", {}).get("sample_points", []):
        x = np.asarray([x0], dtype=float)
        q, grad, fallback = provider.value_and_grad(x)
        _, raw_grad, _ = provider.value_and_grad(x, gradient_method="raw_element")
        samples.append(
            {
                "x": x0,
                "q_infty": float(q[0, 0]),
                "gradient_method": gradient_method,
                "gradient": grad[0].tolist(),
                "raw_element_gradient": raw_grad[0].tolist(),
                "fallback": bool(fallback[0]),
            }
        )
    report = Path(config["outputs"]["reports_dir"]) / f"pipe_3d_gmsh_q_infty_fem_{stamp}.json"
    summary = {
        "checkpoint": str(checkpoint),
        "report": str(report),
        "mesh": str(mesh_path),
        "nodes": int(points.shape[0]),
        "tetrahedra": int(tetrahedra.shape[0]),
        "rx_faces": int(faces["rx"].shape[0]),
        "wall_faces": int(faces["wall"].shape[0]),
        "cap_faces": int(faces["caps"].shape[0]),
        "unknowns": unknowns,
        "q_min": float(values.min()),
        "q_max": float(values.max()),
        "gradient_method": gradient_method,
        "samples": samples,
    }
    save_json(report, summary)
    print(f"[pipe-3d-gmsh] checkpoint={checkpoint}", flush=True)
    print(f"[pipe-3d-gmsh] summary={summary}", flush=True)
    return summary
