import numpy as np
from neatmesh._geometry import tetra_data_from_tensor, tri_data_from_tensor


points = np.array(
        [
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 0., 0.]
        ]
    )    

def test_tetra():
    cells = [3, 0, 1, 2]
    tetra_cell_tensor = np.array([
        [points[i] for i in cells],
        [points[i] for i in cells]
    ])
        
    center, volume = tetra_data_from_tensor(tetra_cell_tensor)
    assert np.allclose(center[0, :], center[1, :])
    assert np.allclose(volume[0], volume[1])


def test_tri():
    faces = [
        [2, 0, 1],  # area = 0.866025 m2
        [3, 2, 1],  # area = 0.5 m2
        ]
    tri_faces_tensor = np.array([
        [points[i] for i in faces[0]],
        [points[i] for i in faces[1]],
    ])
    
    tri_centers, _, tri_areas, _ = tri_data_from_tensor(tri_faces_tensor)
    assert np.allclose(tri_areas, [0.866025, 0.5])
    assert np.allclose(
        tri_centers, 
        [
            [0.33333, 0.33333, 0.33333],
            [0.33333, 0.0, 0.33333]
        ]
    )
