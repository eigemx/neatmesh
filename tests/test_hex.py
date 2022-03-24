import pathlib

import numpy as np
import pytest

from neatmesh.analyzer import Analyzer3D
from neatmesh.reader import assign_reader

h5py = pytest.importorskip("h5py")


def test_hex_one_cell():
    this_dir = pathlib.Path(__file__).resolve().parent
    mesh = assign_reader(this_dir / "meshes" / "one_hex_cell.med")
    analyzer = Analyzer3D(mesh)
    analyzer.count_cell_types()
    analyzer.analyze_cells()
    analyzer.analyze_faces()

    assert np.isclose(np.sum(analyzer.cells_volumes), 1.0)
    assert np.allclose(analyzer.face_aspect_ratios, np.ones(shape=(6,)))
