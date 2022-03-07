from tkinter.tix import Tree
from typing import Callable, Dict, Final, List

import meshio
import numpy as np

from .exceptions import InvalidMeshException
from .trie import Trie


class MeshIOFaceType:
    """meshio face types"""

    Quad = "quad"
    Triangle = "triangle"


class MeshIOCellType:
    """meshio cell types"""

    Hex = "hexahedron"
    Tetra = "tetra"
    Wedge = "wedge"
    Pyramid = "pyramid"


meshio_3d: Final = {
    "tetra",
    "hexahedron",
    "wedge",
    "pyramid",
}

meshio_2d: Final = {
    "triangle",
    "quad",
}


class FromMeshio3D:
    def __init__(self, mesh_file_path: str) -> None:
        try:
            self.mesh_raw = meshio.read(mesh_file_path)
        except meshio.ReadError as exception:
            error = "Could not open mesh file.\n"
            error += f"{exception}"
            raise InvalidMeshException(error) from exception

        self.points = self.mesh_raw.points
        self.n_points = len(self.points)
        self.n_cells = 0

        # list of points labels of processed faces (all types)
        self.processed_faces = []
        
        self.face_trie = Trie()

        # maps sorted face points labels tuple to the face index in `processed_faces`
        self.face_to_faceid = {}

        # maps face id to a list of cells id. A face is shared by max. 2 cells.
        self.faceid_to_cellid = {}

        # keeps track of the face id to be processed.
        self.current_faceid = 0

        for cell_type, faces_fn in (
            (MeshIOCellType.Hex, self.hex_cell_faces),
            (MeshIOCellType.Tetra, self.tetra_cell_faces),
            (MeshIOCellType.Wedge, self.wedge_cell_faces),
            (MeshIOCellType.Pyramid, self.pyramid_cell_faces),
        ):
            self.process_cells(cell_type, faces_fn)


    def face_exists(self, face_labels: List) -> bool:
        """Checks if a list of face labels (aka a face) exists or not
        Args:
            face_labels (list): list of face points labels
        Returns:
            bool: True if face exists, False otherwise
        """
        return self.face_trie.search([
            face_labels[0], face_labels[3], face_labels[2], face_labels[1]
        ])

    def register_face(self, face: List) -> int:
        """Adds a face to list of processed faces and assign an id for it.
        Args:
            face (List): list of face points labels
        Returns:
            int: newly added face id
        """
        tface = tuple(face)
        self.face_to_faceid[tface] = self.current_faceid
        self.face_trie.insert(tface)

        # add face points labels to `processed_faces`
        self.processed_faces.append(face)

        self.current_faceid += 1
        return self.current_faceid - 1

    def link_face_to_cell(self, face: List, cellid: int) -> None:
        """Associates a face (list of points labels) to an owner or
            a neighbor cell given the cell id.
            Args:
                face (List): list of face points labels
                cellid (int): owner/neighbor cell id
            """
        face_id = self.face_to_faceid[tuple(face)]

        if face_id in self.faceid_to_cellid:
            self.faceid_to_cellid[face_id].append(cellid)
        else:
            self.faceid_to_cellid[face_id] = [cellid]

    def process_cells(self, cell_type: str, faces_list_fn: Callable) -> None:
        """Given a cell type and function for cell faces coordinates,
        loop over each cell, extract faces
        and construct owner-neighbor connectivity.
        Args:
            cell_type (str): CellType.Hex, CellType.Tetra or CellType.Wedge
            faces_list_fn (Callable): a function that returns a list of faces
                                      (list of points labels)for the type of cell given.
        """
        cell_block = list(
            filter(lambda cell_block: cell_block.type == cell_type, self.mesh_raw.cells)
        )

        if len(cell_block) == 0:
            # mesh has no cells with the given type, nothing to do here
            return

        cells = cell_block[0].data

        for cell_id, cell in enumerate(cells):
            self.n_cells += 1
            faces = faces_list_fn(cell)
            for face in faces:
                # have we met `face` before?
                '''if not self.face_exists(face):
                    self.register_face(face)'''
                if not self.face_trie.search(face):
                    self.register_face(face)

                # link the face to the cell who owns it
                self.link_face_to_cell(face, cell_id)

    @staticmethod
    def hex_cell_faces(cell_points: List) -> List[List]:
        """Returns coordinates of 6 faces of a hexahedron cell, using meshio nodes ordering
        Args:
            cell_points (List): list of points defining the cell
        Returns:
            List[List]: list of list of faces points labels
        """
        faces = [
            [cell_points[0], cell_points[3], cell_points[2], cell_points[1]],
            [cell_points[4], cell_points[5], cell_points[6], cell_points[7]],
            [cell_points[0], cell_points[1], cell_points[5], cell_points[4]],
            [cell_points[2], cell_points[3], cell_points[7], cell_points[6]],
            [cell_points[0], cell_points[4], cell_points[7], cell_points[3]],
            [cell_points[1], cell_points[2], cell_points[6], cell_points[5]],
        ]
        return faces

    @staticmethod
    def wedge_cell_faces(cell_points: List) -> List[List]:
        """Returns coordinates of 5 faces of a wedge cell, using meshio nodes ordering
        Args:
            cell_points (List): list of points defining the cell
        Returns:
            List[List]: list of list of faces points labels
        """
        faces = [
            [cell_points[0], cell_points[2], cell_points[1]],
            [cell_points[3], cell_points[4], cell_points[5]],
            [cell_points[3], cell_points[0], cell_points[1], cell_points[4]],
            [cell_points[0], cell_points[3], cell_points[5], cell_points[2]],
            [cell_points[1], cell_points[2], cell_points[5], cell_points[4]],
        ]
        return faces

    @staticmethod
    def tetra_cell_faces(cell_points: List) -> List[List]:
        """Returns coordinates of 4 faces of a tetrahedral cell, using meshio nodes ordering
        Args:
            cell_points (List): list of points defining the cell
        Returns:
            List[List]: list of list of faces points labels
        """
        faces = [
            [cell_points[0], cell_points[2], cell_points[1]],
            [cell_points[1], cell_points[2], cell_points[3]],
            [cell_points[0], cell_points[1], cell_points[3]],
            [cell_points[0], cell_points[3], cell_points[2]],
        ]
        return faces

    @staticmethod
    def pyramid_cell_faces(cell_points: List) -> List[List]:
        raise NotImplementedError("Pyramid cell is not implemented.")


def _raw_cell_type(cell_type: str) -> str:
    return "".join(ch for ch in cell_type if ch.isalpha())


def _is_3d(mesh: meshio.Mesh) -> bool:
    for cell_block in mesh.cells:
        if _raw_cell_type(cell_block.type) in meshio_3d:
            return True
    return False


def _is_2d(mesh: meshio.Mesh) -> bool:
    if _is_3d(mesh):
        return False

    for cell_block in mesh.cells:
        if _raw_cell_type(cell_block.type) in meshio_2d:
            return True
    return False
