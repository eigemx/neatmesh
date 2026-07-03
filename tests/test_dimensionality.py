import pathlib

import neatmesh as nm

this_dir = pathlib.Path(__file__).resolve().parent


def test_dimensionality():
    mesh_2d = nm.read(this_dir / "meshes" / "disc_2d.vtk")
    mesh_3d = nm.read(this_dir / "meshes" / "coarse_cylinder.vtk")

    assert mesh_2d.dim == 2
    assert mesh_3d.dim == 3
