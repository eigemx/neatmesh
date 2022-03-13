from typing import Callable, FrozenSet, List, Set, Tuple, Dict, Iterator
from attr import frozen

from meshio import ReadError, read

from ._exceptions import InvalidMeshException, NonSupportedElement
from ._common import *


class MeshReader3D:
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
        self.faces: List[Tuple[int, ...]] = []
        self.faces_set: Set[FrozenSet] = set()

        # map face points to face index in `faces`
        self.face_to_faceid: Dict[FrozenSet, int] = {}

        # maps face id to a list of cells id. A face is shared by max. 2 cells.
        self.faceid_to_cellid: Dict[int, List[int]] = {}

        # keep track of the face id to be processed.
        self.current_faceid: int = 0

        # keep track of the cell id to be processed.
        self.current_cellid: int = 0

        self._check_unsupported()
        self.process_mesh()

    def _check_unsupported(self):
        self.cell_blocks = []
        for cell_block in self.mesh.cells:
            if cell_block.type in meshio_3d and cell_block.data.size > 0:
                self.cell_blocks.append(cell_block)

            elif cell_block.type not in meshio_2d \
                and cell_block.type not in meshio_1d:
                raise NonSupportedElement(
                    f"neatmesh does not support element type: {cell_block.type}"
                )

    def process_mesh(self) -> None:
        for cell_block in self.cell_blocks:
            cells = cell_block.data

            for cell in cells:
                faces = cell_type_to_faces_func[meshio_3d_to_alpha[cell_block.type]](cell)
                for face in faces:
                    fface = frozenset(face)
                    # have we met `face` before?
                    if not fface in self.faces_set:
                        self.face_to_faceid[fface] = self.current_faceid
                        self.faces_set.add(fface)

                        # add face points labels to `faces`
                        self.faces.append(face)
                        self.faceid_to_cellid[self.current_faceid] = [self.current_cellid, -1]
                        self.current_faceid += 1
                    else:
                    # link the face to the cell who owns it
                        self.faceid_to_cellid[self.face_to_faceid[fface]][1] = self.current_cellid

                self.current_cellid += 1

    def cells(self) -> Iterator[Tuple[Tuple[int, ...], str]]:
        for cell_block in self.mesh.cells:
            if cell_block.type not in cell_type_to_faces_func:
                continue
            for cell in cell_block.data:
                yield cell, cell_block.type
