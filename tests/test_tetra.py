import pathlib

import numpy as np
import pytest

from neatmesh.analyzer import Analyzer3D
from neatmesh.geometry import tetra_data_from_tensor
from neatmesh.reader import assign_reader

h5py = pytest.importorskip("h5py")


def test_tetra_one_cell():
    points = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    cells = [3, 0, 1, 2]
    tetra_cell_tensor = np.array(
        [[points[i] for i in cells], [points[i] for i in cells]]
    )

    center, volume = tetra_data_from_tensor(tetra_cell_tensor)
    assert np.allclose(center[0, :], center[1, :])
    assert np.allclose(volume[0], volume[1])


def test_tetra_mesh():
    this_dir = pathlib.Path(__file__).resolve().parent
    mesh = assign_reader(this_dir / "meshes" / "fine_cylinder.med")
    q = Analyzer3D(mesh)
    q.count_cell_types()
    q.analyze_cells()
    assert np.allclose(np.sum(q.cells_volumes), [3.14061])
