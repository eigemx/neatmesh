import pathlib
import pytest
import meshio
import numpy as np

from neatmesh.analyzer import Analyzer2D
from neatmesh.reader import assign_reader

this_dir = pathlib.Path(__file__).resolve().parent


h5py = pytest.importorskip("h5py")

def test_total_area():
    reader = assign_reader(this_dir / "meshes" / "disc_2d.med")
    analyzer = Analyzer2D(reader)
    analyzer.analyze_faces()

    assert np.allclose([np.sum(analyzer.face_areas)], [3.14153])


def test_non_ortho(tmp_path):
    points = [
        [0, 0],  # 0
        [1, 0],  # 1
        [1, 1],  # 2
        [0, 1],  # 3
        [2, 0],  # 4
        [2, 1],  # 5
        [3, 0],  # 5
        [3, 1],  # 6
    ]

    cells = [[0, 1, 2, 3], [1, 4, 5, 2], [4, 7, 5], [4, 6, 7]]

    cell_blocks = [
        meshio.CellBlock(cell_type="quad", data=cells[0:2]),
        meshio.CellBlock(cell_type="triangle", data=cells[2:]),
    ]

    mesh = meshio.Mesh(points=points, cells=cell_blocks)
    meshio.write(tmp_path / "two_quads.su2", mesh)

    reader = assign_reader(tmp_path / "two_quads.su2")
    analyzer = Analyzer2D(reader)
    analyzer.analyze_faces()

    assert np.allclose([np.sum(analyzer.face_areas)], [3.0])

    analyzer.analyze_non_ortho()
    assert np.allclose(analyzer.non_ortho, [0.0, 11.3099, 0.0])

    analyzer.analyze_adjacents_area_ratio()
    assert np.allclose(analyzer.adj_ratio, [1.0, 2.0, 1.0])
