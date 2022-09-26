import pathlib

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
    assert np.isclose(np.sum(mesh.face_areas), 1.)
    assert np.allclose(mesh.face_centers, np.array([[0.5, 0.5]]))
    
    assert not np.all(mesh.internal_edges_mask)
    assert np.all(mesh.boundary_edges_mask)
