import pathlib

import pytest

import neatmesh as nm

this_dir = pathlib.Path(__file__).resolve().parent
h5py = pytest.importorskip("h5py")


def test_dimensionality():
    mesh_2d = nm.read(this_dir / "meshes" / "disc_2d.med")
    mesh_3d = nm.read(this_dir / "meshes" / "coarse_cylinder.med")

    assert mesh_2d.dim == 2
    assert mesh_3d.dim == 3
