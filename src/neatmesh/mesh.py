from dataclasses import dataclass
from typing import Set, Tuple, Union, Generator, List

import numpy as np

from .analyzer import Analyzer2D, Analyzer3D
from .reader import assign_reader, MeshReader3D, MeshReader2D, cell_type_to_faces_fn_map


@dataclass
class Edge:
    edge_id: int
    points: Tuple[int, int]
    center: np.ndarray
    normal: np.ndarray
    length: float
    is_boundary: bool
    owner: int
    neighbor: Union[int, None]


@dataclass
class Face:
    face_id: int
    face_type: str
    vertices: Tuple[int, ...]
    center: np.ndarray
    normal: np.ndarray
    area: float
    is_boundary: bool
    owner: int
    neighbor: Union[int, None]


@dataclass
class Cell:
    cell_id: int
    cell_type: str
    adjacent_cells: Set
    vertices: Tuple[int, ...]
    faces: Tuple[Face, ...]
    volume: float
    center: np.ndarray


class Mesh3D:
    def __init__(self, mesh_file_path: str) -> None:
        self.reader: MeshReader3D = assign_reader(mesh_file_path)
        self.analyzer = Analyzer3D(self.reader)

        self.analyzer.count_cell_types()
        self.analyzer.analyze_faces()
        self.analyzer.analyze_cells()
        self.analyzer.analyze_non_ortho()

    def _face_from_id(self, face_id: int):
        face_vertices = self.analyzer.faces[face_id]
        face_center = self.analyzer.face_centers[face_id]
        face_normal = self.analyzer.face_normals[face_id]
        face_area = self.analyzer.face_areas[face_id]

        owner, neighbor = self.analyzer.owner_neighbor[face_id]
        neighbor = None if neighbor == -1 else neighbor

        return Face(
                face_id=face_id,
                face_type='quad' if face_vertices.shape[0] == 4 else 'triangle',
                vertices=face_vertices,
                center=face_center,
                normal=face_normal,
                area=face_area,
                is_boundary=neighbor is None,
                owner=owner,
                neighbor=neighbor,
            )

    def cells(self):
        for cell_block in self.reader.cell_blocks:
            adjacent_cells = set()
            for cell_id, cell_vertices in enumerate(cell_block.data):
                cell_type = cell_block.type
                volume = self.analyzer.cells_volumes[cell_id]
                center = self.analyzer.cells_centers[cell_id]
                faces: List[Face] = []

                get_faces_fn = cell_type_to_faces_fn_map[cell_type]
                for face in get_faces_fn(cell_vertices):
                    fface = frozenset(face)
                    face_id = self.reader.face_to_faceid[fface]
                    
                    face_obj = self._face_from_id(face_id)
                    faces.append(face_obj)

            yield Cell(
                cell_id=cell_id,
                cell_type=cell_type,
                adjacent_cells=adjacent_cells,
                vertices=cell_vertices,
                faces=faces,
                volume=volume,
                center=center,
            )
