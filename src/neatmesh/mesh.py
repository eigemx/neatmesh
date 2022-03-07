import numpy as np
from typing import List, Dict


class Mesh3D:
    def __init__(
        self,
        points: np.ndarray,
        cells: List[np.ndarray],
        face_sharing_matrix: np.ndarray= None,
    ) -> None:
        # Points
        self.points = points
        self.n_points = self.points.shape[0]
        
        
        self.cells = cells
        self.face_sharing_matrix = face_sharing_matrix
