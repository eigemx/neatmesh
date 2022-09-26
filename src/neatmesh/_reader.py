"""neatmesh 2D & 3D mesh readers"""
# pylint: disable=too-few-public-methods, too-many-instance-attributes
from typing import Callable, Dict, FrozenSet, List, Set, Tuple, Union

import meshio

from ._common import (
    is_2d_mesh,
    is_3d_mesh,
    meshio_1d_elements,
    meshio_2d_elements,
    meshio_3d_elements,
    meshio_type_to_alpha,
)
from ._exceptions import InvalidMeshError, NonSupportedElementError
from ._geometry import (
    hex_cell_faces,
    pyramid_cell_faces,
    quad_face_edges,
    tetra_cell_faces,
    tri_face_edges,
    wedge_cell_faces,
)


# pylint: disable=unused-private-member
class MeshReader:
    """Base class for readers, do not use directly"""

    def __init__(self, meshio_mesh: meshio.Mesh) -> None:
        self.meshio_mesh = meshio_mesh
        self.points = self.meshio_mesh.points
        self.n_points = len(self.points)

    def _check_mesh(self) -> None:
        return

    def _process_mesh(self) -> None:
        return


class MeshReader2D(MeshReader):
    """A 2D mesh reader"""

    def __init__(self, meshio_mesh: meshio.Mesh) -> None:
        super().__init__(meshio_mesh)

        # list of points labels of processed edges
        self.edges: List[Tuple] = []
        self.__edges_sets: Set[FrozenSet] = set()
        self.edge_to_edgeid: Dict[FrozenSet, int] = {}

        # maps edge id to a list of faces id. An edge is shared by max. 2 faces.
        self.edgeid_to_faceid: Dict[int, List[int]] = {}

        # keep track of the edge id to be processed.
        self.__current_edgeid: int = 0

        # keep track of the face id to be processed.
        self.__current_faceid: int = 0

        self._check_mesh()
        self._process_mesh()

    def _check_mesh(self) -> None:
        """Look for 2D cell types, and check if mesh contains unsupported types"""
        self.n_faces = 0
        self.cell_blocks = []

        for cell_block in self.meshio_mesh.cells:
            ctype = meshio_type_to_alpha.get(cell_block.type, "unsupported")

            if ctype in meshio_2d_elements and cell_block.data.size > 0:
                self.cell_blocks.append(cell_block)
                self.n_faces += len(cell_block.data)

            elif (
                ctype not in meshio_2d_elements
                and cell_block.type not in meshio_1d_elements
            ):
                raise NonSupportedElementError(
                    f"neatmesh does not support element type: {cell_block.type}"
                )

        if not self.cell_blocks:
            raise InvalidMeshError("No 2D elements were found in mesh")

    def _process_mesh(self) -> None:
        """Get edges of each 2D cell, and assign owner/neighbor of each edge"""
        for cell_block in self.cell_blocks:
            faces = cell_block.data
            face_type = meshio_type_to_alpha[cell_block.type]

            if face_type == "quad":
                edges_func = quad_face_edges
            else:
                edges_func = tri_face_edges

            for face in faces:
                edges = edges_func(face)
                for edge in edges:
                    f_edge = frozenset(edge)
                    # have we met `edge` before?
                    if f_edge not in self.__edges_sets:
                        self.edge_to_edgeid[f_edge] = self.__current_edgeid
                        self.edges.append(edge)
                        self.__edges_sets.add(f_edge)

                        self.edgeid_to_faceid[self.__current_edgeid] = [
                            self.__current_faceid,
                            -1,
                        ]
                        self.__current_edgeid += 1
                    else:
                        # link edge to face who owns it
                        edge_id = self.edge_to_edgeid[f_edge]
                        self.edgeid_to_faceid[edge_id][1] = self.__current_faceid

                self.__current_faceid += 1


class MeshReader3D(MeshReader):
    """A 3D mesh reader"""

    def __init__(self, mesh: meshio.Mesh) -> None:
        super().__init__(mesh)

        # list of points labels of processed faces (all types)
        self.faces: List[Tuple[int, ...]] = []

        self.__faces_set: Set[FrozenSet] = set()

        # map face points to face index in `faces`
        self.face_to_faceid: Dict[FrozenSet, int] = {}

        # maps face id to a list of cells id. A face is shared by max. 2 cells.
        self.faceid_to_cellid: Dict[int, List[int]] = {}

        # keep track of the face id to be processed.
        self.__current_faceid: int = 0

        # keep track of the cell id to be processed.
        self.__current_cellid: int = 0

        self._check_mesh()
        self._process_mesh()

    def _check_mesh(self) -> None:
        """Look for 2D cell types, and check if mesh contains unsupported types"""
        self.n_cells = 0
        self.cell_blocks = []

        for cell_block in self.meshio_mesh.cells:
            # look up cell_block type, and return "unsupported" if not found
            ctype = meshio_type_to_alpha.get(cell_block.type, "unsupported")

            if ctype in meshio_3d_elements and cell_block.data.size > 0:
                self.cell_blocks.append(cell_block)
                self.n_cells += len(cell_block.data)

            elif (
                ctype not in meshio_2d_elements
                and cell_block.type not in meshio_1d_elements
            ):
                raise NonSupportedElementError(
                    f"neatmesh does not support element type: {cell_block.type}"
                )

        if not self.cell_blocks:
            raise InvalidMeshError("No 3D elements were found in mesh")

    def _process_mesh(self) -> None:
        """Get faces of each 3D cell, and assign owner/neighbor of each face"""
        for cell_block in self.cell_blocks:
            cells = cell_block.data

            for cell in cells:
                cell_type = meshio_type_to_alpha[cell_block.type]
                get_faces_fn = cell_type_to_faces_fn_map[cell_type]
                faces = get_faces_fn(cell)

                for face in faces:
                    fface = frozenset(face)
                    # have we met current before?
                    if fface not in self.__faces_set:
                        self.face_to_faceid[fface] = self.__current_faceid
                        self.__faces_set.add(fface)

                        # add face points labels to `faces`
                        self.faces.append(face)
                        self.faceid_to_cellid[self.__current_faceid] = [
                            self.__current_cellid,
                            -1,
                        ]
                        self.__current_faceid += 1
                    else:
                        # link the face to the cell who owns it
                        face_id = self.face_to_faceid[fface]
                        self.faceid_to_cellid[face_id][1] = self.__current_cellid

                self.__current_cellid += 1


cell_type_to_faces_fn_map: Dict[str, Callable] = {
    "hexahedron": hex_cell_faces,
    "tetra": tetra_cell_faces,
    "wedge": wedge_cell_faces,
    "pyramid": pyramid_cell_faces,
}


def assign_reader(mesh_file_path: str) -> Union[MeshReader2D, MeshReader3D]:
    """Assign a 2D or 3D MeshReader, given the mesh file.

    Args:
        mesh_file_path (str): Path to mesh file

    Raises:
        InvalidMeshException: When assign_reader cannot decide mesh dimensions

    Returns:
        MeshReader: 2D or 3D Reader
    """
    try:
        mesh = meshio.read(mesh_file_path)
    except meshio.ReadError as exception:
        error = "Could not read mesh file (meshio error).\n"
        error += f"{exception}"
        raise InvalidMeshError(error) from exception

    if is_3d_mesh(mesh):
        return MeshReader3D(mesh)
    if is_2d_mesh(mesh):
        return MeshReader2D(mesh)

    raise InvalidMeshError("Couldn't decide on mesh dimensionality")
