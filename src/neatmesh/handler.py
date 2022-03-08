from numpy.typing import NDArray
from typing import Tuple, List, Dict, FrozenSet
from meshio import CellBlock


class MeshHandler:
    points: NDArray
    n_point: int
    n_cells: int
    processed_faces: List[Tuple[int, ...]]
    face_to_faceid: Dict[FrozenSet, int]
    faceid_to_cellid: Dict[int, List[int]]

    def cells(self) -> Tuple[int, ...]:
        pass
    
