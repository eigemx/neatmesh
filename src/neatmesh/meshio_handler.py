from typing import Callable, List, Set, FrozenSet, Dict

import meshio
from numpy.typing import NDArray

from .exceptions import InvalidMeshException
from .cell_faces import *


class MeshioProcessor3D:
    def __init__(self, mesh_file_path: str) -> None:
        #TODO: check if file exists outside of here,
        # raise meshio.ReadError only in case meshio cannot read or
        # support given mesh.
        try:
            self.mesh = meshio.read(mesh_file_path)
        except meshio.ReadError as exception:
            error = "Could not open mesh file.\n"
            error += f"{exception}"
            raise InvalidMeshException(error) from exception

        self.points: NDArray = self.mesh.points
        self.n_points: int = len(self.points)

        # list of points labels of processed faces (all types)
        self.processed_faces: List[Tuple[int, ...]] = []
        self.faces_set: Set = set()

        # maps sorted face points labels tuple to the face index in `processed_faces`
        self.face_to_faceid: Dict[FrozenSet, int] = {}

        # maps face id to a list of cells id. A face is shared by max. 2 cells.
        self.faceid_to_cellid: Dict[int, List[int]] = {}

        # keep track of the face id to be processed.
        self.current_faceid: int = 0

        # keep track of the cell id to be processed.
        self.current_cellid: int = 0

    def process_mesh(self) -> None:
        # TODO: Replace this with a dict[str, Callable],
        # and make process_cells call the key directly.
        # To avoid calling process_cells many times unnecessarily
        for cell_type, faces_fn in (
            (MeshIOCellType.Hex, hex_cell_faces),
            (MeshIOCellType.Hex20, hex20_cell_faces),
            (MeshIOCellType.Hex24, hex20_cell_faces),
            (MeshIOCellType.Hex27, hex20_cell_faces),
            (MeshIOCellType.Tetra, tetra_cell_faces),
            (MeshIOCellType.Tetra10, tetra10_cell_faces),
            (MeshIOCellType.Wedge, wedge_cell_faces),
            (MeshIOCellType.Wedge12, wedge12_cell_faces),
            (MeshIOCellType.Wedge15, wedge12_cell_faces),
            (MeshIOCellType.Pyramid, pyramid_cell_faces),
            (MeshIOCellType.Pyramid13, pyramid13_cell_faces),
            (MeshIOCellType.Pyramid14, pyramid13_cell_faces),
        ):
            self.process_cells(cell_type, faces_fn)

    def face_exists(self, face_labels: Tuple[int, ...]) -> bool:
        """Checks if a list of face labels (aka a face) exists or not
        Args:
            face_labels (list): list of face points labels
        Returns:
            bool: True if face exists, False otherwise
        """
        return frozenset(face_labels) in self.faces_set

    def add_face(self, face: Tuple[int, ...]) -> int:
        """Adds a face to list of processed faces and assign an id for it.
        Args:
            face (List): list of face points labels
        Returns:
            int: newly added face id
        """
        sface = frozenset(face)
        self.face_to_faceid[sface] = self.current_faceid
        self.faces_set.add(sface)

        # add face points labels to `processed_faces`
        self.processed_faces.append(face)

        self.current_faceid += 1
        return self.current_faceid - 1

    def link_face_to_cell(self, face: Tuple[int, ...], cellid: int) -> None:
        """Associates a face (list of points labels) to an owner or
        a neighbor cell given the cell id.
        Args:
            face (List): list of face points labels
            cellid (int): owner/neighbor cell id
        """
        face_id = self.face_to_faceid[frozenset(face)]

        if face_id in self.faceid_to_cellid:
            self.faceid_to_cellid[face_id][1] = cellid
        else:
            self.faceid_to_cellid[face_id] = [cellid, -1]

    def process_cells(
        self,
        cell_type: str,
        faces_list_fn: Callable[[List[int]], Tuple[Tuple[int, ...], ...]],
    ) -> None:
        """Given a cell type and function for cell faces coordinates,
        loop over each cell, extract faces
        and construct owner-neighbor connectivity.
        Args:
            cell_type (str): CellType.Hex, CellType.Tetra or CellType.Wedge
            faces_list_fn (Callable): a function that returns a list of faces
                                      (list of points labels)for the type of cell given.
        """
        cells = self.mesh.get_cells_type(cell_type)

        if cells.size == 0:
            # mesh has no cells with the given type, nothing to do here
            return

        for cell in cells:
            faces = faces_list_fn(cell)
            for face in faces:
                # have we met `face` before?
                if not self.face_exists(face):
                    self.add_face(face)

                # link the face to the cell who owns it
                self.link_face_to_cell(face, self.current_cellid)

            self.current_cellid += 1


def _alphabetic_cell_type(cell_type: str) -> str:
    """Return meshio cell type without numerical postfix"""
    return "".join(ch for ch in cell_type if ch.isalpha())


def _is_3d(mesh: meshio.Mesh) -> bool:
    """Check if mesh is 3-dimensional"""
    for cell_block in mesh.cells:
        if _alphabetic_cell_type(cell_block.type) in meshio_3d:
            return True
    return False


def _is_2d(mesh: meshio.Mesh) -> bool:
    """Check if mesh is 2-dimensional"""
    if _is_3d(mesh):
        return False

    for cell_block in mesh.cells:
        if _alphabetic_cell_type(cell_block.type) in meshio_2d:
            return True
    return False
