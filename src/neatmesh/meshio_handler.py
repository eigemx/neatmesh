from typing import Callable, FrozenSet, List, Set, Tuple, Final, Dict

from meshio import ReadError, read

from .exceptions import InvalidMeshException
from .meshio_common import *


# TODO: Check for unsupported cell types
class MeshioHandler3D:
    def __init__(self, mesh_file_path: str) -> None:
        # TODO: check if file exists outside of here,
        # raise meshio.ReadError only in case meshio cannot read or
        # support given mesh.
        try:
            self.mesh = read(mesh_file_path)
        except ReadError as exception:
            error = "Could not open mesh file.\n"
            error += f"{exception}"
            raise InvalidMeshException(error) from exception

        self.points = self.mesh.points
        self.n_points = len(self.points)

        self.n_cells = sum(
            [
                len(cell_block.data)
                for cell_block in self.mesh.cells
                if cell_block.type in meshio_3d
            ]
        )

        # list of points labels of processed faces (all types)
        self.faces: Tuple[int, ...] = []
        self.faces_set: Set[FrozenSet] = set()

        # map face points to face index in `faces`
        self.face_to_faceid: Dict[FrozenSet, int] = {}

        # maps face id to a list of cells id. A face is shared by max. 2 cells.
        self.faceid_to_cellid: Dict[int, int] = {}

        # keep track of the face id to be processed.
        self.current_faceid: int = 0

        # keep track of the cell id to be processed.
        self.current_cellid: int = 0

    def process_mesh(self) -> None:
        for cell_type, faces_fn in cell_type_handler_map.items():
            self.process_cells(cell_type, faces_fn)

    def cells(self) -> Tuple[Tuple[int, ...], MeshIOCellType]:
        for cell_block in self.mesh.cells:
            if cell_block.type not in cell_type_handler_map:
                continue
            for cell in cell_block.data:
                yield cell, cell_block.type

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

        # add face points labels to `faces`
        self.faces.append(face)

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
