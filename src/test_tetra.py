import numpy as np
from neatmesh._geometry import tetra_data_from_tensor, tri_data_from_tensor

def test_tetra():
    points = np.array(
        [
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 0., 0.]
        ]
    )    
    cells = [3, 0, 1, 2]
    tetra_cell_tensor = np.array([
        [points[i] for i in cells],
        [points[i] for i in cells]
    ])
        
    center, volume = tetra_data_from_tensor(tetra_cell_tensor)
    print(center, volume)

test_tetra()