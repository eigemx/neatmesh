import numpy as np
from typing import List, Dict


class Mesh3D:
    def __init__(
        self,
        points: np.ndarray,
        cells: List[np.ndarray],
        connectivity_map: Dict[int : List[int]] = None,
    ) -> None:
        self.points = points
        self.cells = cells
        self.connectivity_map = connectivity_map
