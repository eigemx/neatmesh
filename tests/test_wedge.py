import numpy as np

from neatmesh.geometry import wedge_data_from_tensor

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
    wedge_cell_tensor = np.array(
        [
            [points[i] for i in cells],
            [points[i] for i in cells],
        ]
    ).reshape(2, 6, 3)

    center, volume = wedge_data_from_tensor(wedge_cell_tensor)

    assert np.allclose(center, [[0.33333, 0.333333, 0.5], [0.33333, 0.333333, 0.5]])
    assert np.allclose(volume, [0.5, 0.5])
