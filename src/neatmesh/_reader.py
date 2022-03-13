from typing import Dict, FrozenSet, List, Set, Tuple

from meshio import ReadError, read

from ._common import *
from ._exceptions import InvalidMeshException, NonSupportedElement


# TODO: check if mesh exists before calling MeshReader3D
class MeshReader3D:
    def __init__(self, mesh_file_path: str) -> None:
        try:
            self.mesh = read(mesh_file_path)
        except ReadError as exception:
            error = "Could not read mesh file (meshio error).\n"
            error += f"{exception}"
            raise InvalidMeshException(error) from exception

        self.points = self.mesh.points
        self.n_points = len(self.points)

        self._check_mesh()
        self.process_mesh()

    def _check_mesh(self):
        self.cell_blocks = []
        self.n_cells = 0

        for cell_block in self.mesh.cells:
            ctype = cell_block.type
            if ctype in meshio_type_to_alpha:
                ctype = meshio_type_to_alpha[ctype]

            if ctype in meshio_3d and cell_block.data.size > 0:
                self.cell_blocks.append(cell_block)
                self.n_cells += len(cell_block.data)

            elif ctype not in meshio_2d \
                and cell_block.type not in meshio_1d:
                raise NonSupportedElement(
                    f"neatmesh does not support element type: {cell_block.type}"
                )
        
        if not self.cell_blocks:
            raise InvalidMeshException("No 3D elements were found in mesh")

    def process_mesh(self) -> None:
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
        
        for cell_block in self.cell_blocks:
            cells = cell_block.data

            for cell in cells:
                cell_type = meshio_type_to_alpha[cell_block.type]
                faces_func = cell_type_to_faces_func[cell_type]
                faces = faces_func(cell)
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
                        face_id = self.face_to_faceid[fface]
                        self.faceid_to_cellid[face_id][1] = self.current_cellid

                self.current_cellid += 1
