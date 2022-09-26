import pathlib

import meshio
import numpy as np
import pytest

from neatmesh import Mesh2D

this_dir = pathlib.Path(__file__).resolve().parent
h5py = pytest.importorskip("h5py")


def test_single_quad_2d_mesh():
    mesh = Mesh2D(this_dir / "meshes" / "single_quad.med")
    assert np.all(mesh.owner_neighbor[:, 1] == -1)
    assert mesh.n_points == 4
    assert mesh.n_edges == 4
    assert mesh.n_boundary_edges == 4
    assert mesh.n_quad == 1
    assert mesh.n_faces == 1
    assert mesh.n_tri == 0

    normals = mesh.edge_normals
    assert np.dot(normals[0], normals[1]) == 0
    assert np.dot(normals[2], normals[3]) == 0
    assert np.isclose(np.sum(mesh.face_areas), 1.0)
    assert np.allclose(mesh.face_centers, np.array([[0.5, 0.5]]))

    assert not np.all(mesh.internal_edges_mask)
    assert np.all(mesh.boundary_edges_mask)


def test_two_triangles_2d_mesh(tmp_path):
    points = [
        [0, 0],  # 0
        [1, 0],  # 1
        [1, 1],  # 2
        [0, 1],  # 3
    ]

    cells = [[0, 1, 3], [1, 2, 3]]

    cell_blocks = [
        meshio.CellBlock(cell_type="triangle", data=cells),
    ]

    mesh = meshio.Mesh(points=points, cells=cell_blocks)
    meshio.write(tmp_path / "two_tri.su2", mesh)
    mesh = Mesh2D(tmp_path / "two_tri.su2")

    assert mesh.n_points == 4
    assert mesh.n_boundary_edges == 4
    assert mesh.n_edges == 5
    assert mesh.n_faces == 2
    assert mesh.n_quad == 0
    assert mesh.n_tri == 2
    assert np.isclose(np.sum(mesh.face_areas), 1.0)

    # only one shared edge
    assert np.count_nonzero(mesh.owner_neighbor[:, 1] != -1) == 1

    # get shared edge
    edge_index = np.argmax(mesh.owner_neighbor[:, 1])
    shared_edge_normal = mesh.edge_normals[edge_index]
    theoretical_normal = [0.70710678, 0.70710678]
    assert np.allclose(shared_edge_normal - theoretical_normal, [0.0, 0.0])
