import pathlib
import numpy as np
from neatmesh.analyzer import Analyzer2D
from neatmesh.reader import assign_reader


this_dir = pathlib.Path(__file__).resolve().parent

def test_total_area():
    reader = assign_reader(this_dir / "meshes" / "disc_2d.med")
    analyzer = Analyzer2D(reader)
    analyzer.analyze_faces()
    
    assert np.allclose([np.sum(analyzer.face_areas)], [3.14153])
