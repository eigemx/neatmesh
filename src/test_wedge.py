import numpy as np
from neatmesh._geometry import wedge_data_from_tensor


points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
        ]
    )    

def test_wedge():
    cells = [0, 1, 2, 3, 4, 5]
    wedge_cell_tensor = np.array([
        [points[i] for i in cells],
        #[points[i] for i in cells],
    ]).reshape(1, 6, 3)
        
    center, volume = wedge_data_from_tensor(wedge_cell_tensor)
    
    print(center, volume)

test_wedge()