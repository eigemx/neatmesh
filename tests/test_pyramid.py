import numpy as np
from neatmesh.geometry import pyramid_data_from_tensor


points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0.5, 0.5, 1]
        ]
    )    

def test_pyramid():
    cells = [0, 1, 2, 3, 4]
    pyr_cell_tensor = np.array([
        [points[i] for i in cells],
        [points[i] for i in cells]
    ]).reshape(2, 5, 3)
        
    center, volume = pyramid_data_from_tensor(pyr_cell_tensor)
    
    assert np.allclose(center, [[0.5, 0.5, 0.25], [0.5, 0.5, 0.25]])
    assert np.allclose(volume, [0.33333, 0.33333])
