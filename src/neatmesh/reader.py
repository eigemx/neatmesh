from typing import Dict, FrozenSet, List, Set, Tuple

import meshio

from .common import is_2d, is_3d, meshio_1d, meshio_2d, meshio_3d, meshio_type_to_alpha
from .exceptions import InvalidMeshException, NonSupportedElement


class MeshReader:
    cell_blocks: List[meshio.CellBlock] = []

    def __init__(self) -> None:
        pass

    def _check_mesh(self) -> None:
        pass

    def process_mesh(self) -> None:
        pass


class MeshReader2D(MeshReader):
    def __init__(self, mesh: meshio.Mesh) -> None:
        self.mesh = mesh
        self.points = self.mesh.points
        self.n_points = len(self.points)

        # list of points labels of processed edges
        self.edges: Set[FrozenSet] = set()
        self.edge_to_edgeid: Dict[FrozenSet, int] = {}

        # maps edge id to a list of faces id. An edge is shared by max. 2 faces.
        self.edgeid_to_faceid: Dict[int, List[int]] = {}

        # keep track of the edge id to be processed.
        self.current_edgeid: int = 0

        # keep track of the face id to be processed.
        self.current_faceid: int = 0

        self._check_mesh()
        self.process_mesh()

    def _check_mesh(self):
        self.n_faces = 0

        for cell_block in self.mesh.cells:
            ctype = meshio_type_to_alpha.get(cell_block.type, "unsupported")

            if ctype in meshio_2d and cell_block.data.size > 0:
                self.cell_blocks.append(cell_block)
                self.n_faces += len(cell_block.data)

            elif ctype not in meshio_2d and cell_block.type not in meshio_1d:
                raise NonSupportedElement(
                    f"neatmesh does not support element type: {cell_block.type}"
                )

        if not self.cell_blocks:
            raise InvalidMeshException("No 2D elements were found in mesh")

    def process_mesh(self) -> None:
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
                    if not f_edge in self.edges:
                        self.edge_to_edgeid[f_edge] = self.current_edgeid
                        self.edges.add(f_edge)

                        self.edgeid_to_faceid[self.current_edgeid] = [
                            self.current_faceid,
                            -1,
                        ]
                        self.current_edgeid += 1
                    else:
                        # link edge to face who owns it
                        edge_id = self.edge_to_edgeid[f_edge]
                        self.edgeid_to_faceid[edge_id][1] = self.current_faceid

                self.current_faceid += 1


class MeshReader3D(MeshReader):
    def __init__(self, mesh: meshio.Mesh) -> None:
        self.mesh = mesh
        self.points = self.mesh.points
        self.n_points = len(self.points)

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

        self._check_mesh()
        self.process_mesh()

    def _check_mesh(self):
        self.n_cells = 0

        for cell_block in self.mesh.cells:
            ctype = meshio_type_to_alpha.get(cell_block.type, "unsupported")

            if ctype in meshio_3d and cell_block.data.size > 0:
                self.cell_blocks.append(cell_block)
                self.n_cells += len(cell_block.data)

            elif ctype not in meshio_2d and cell_block.type not in meshio_1d:
                raise NonSupportedElement(
                    f"neatmesh does not support element type: {cell_block.type}"
                )

        if not self.cell_blocks:
            raise InvalidMeshException("No 3D elements were found in mesh")

    def process_mesh(self) -> None:
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
                        self.faceid_to_cellid[self.current_faceid] = [
                            self.current_cellid,
                            -1,
                        ]
                        self.current_faceid += 1
                    else:
                        # link the face to the cell who owns it
                        face_id = self.face_to_faceid[fface]
                        self.faceid_to_cellid[face_id][1] = self.current_cellid

                self.current_cellid += 1


def hex_cell_faces(cell: List) -> Tuple[Tuple[int, ...], ...]:
    """Returns coordinates of 6 faces of a hexahedron cell, using meshio nodes ordering
    Args:
        cell (List): list of points defining the cell
    Returns:
        List[List]: list of list of faces points labels
    """
    return (
        (cell[1], cell[2], cell[6], cell[5]),
        (cell[0], cell[4], cell[7], cell[3]),
        (cell[3], cell[7], cell[6], cell[2]),
        (cell[0], cell[1], cell[5], cell[4]),
        (cell[4], cell[5], cell[6], cell[7]),
        (cell[0], cell[3], cell[2], cell[1]),
    )


def wedge_cell_faces(cell: List) -> Tuple[Tuple[int, ...], ...]:
    """Returns coordinates of 5 faces of a wedge cell,
    using meshio nodes ordering for wedge
    Args:
        cell (List): list of points defining the cell
    Returns:
        List[List]: list of list of faces points labels
    """
    return (
        (cell[0], cell[2], cell[1], -1),
        (cell[3], cell[4], cell[5], -1),
        (cell[3], cell[0], cell[1], cell[4]),
        (cell[0], cell[3], cell[5], cell[2]),
        (cell[1], cell[2], cell[5], cell[4]),
    )


def tetra_cell_faces(cell: List) -> Tuple[Tuple[int, ...], ...]:
    """Returns coordinates of 4 faces of a tetrahedral cell,
    using meshio nodes ordering for tetra
    Args:
        cell (List): list of points defining the cell
    Returns:
        List[List]: list of list of faces points labels
    """
    return (
        (cell[0], cell[2], cell[1], -1),
        (cell[1], cell[2], cell[3], -1),
        (cell[0], cell[1], cell[3], -1),
        (cell[0], cell[3], cell[2], -1),
    )


def pyramid_cell_faces(cell: List) -> Tuple[Tuple[int, ...], ...]:
    """Returns coordinates of 4 faces of a tetrahedral cell,
    using meshio nodes ordering for pyramid
    Args:
        cell (List): list of points defining the cell
    Returns:
        List[List]: list of list of faces points labels
    """
    return (
        (cell[2], cell[1], cell[0], cell[3]),
        (cell[2], cell[3], cell[4], -1),
        (cell[1], cell[4], cell[0], -1),
        (cell[3], cell[0], cell[4], -1),
    )


def quad_face_edges(face: List) -> Tuple[Tuple[int, ...]]:
    return (
        (face[0], face[1]),
        (face[1], face[2]),
        (face[2], face[3]),
        (face[3], face[0]),
    )


def tri_face_edges(face: List) -> Tuple[Tuple[int, ...]]:
    return (
        (face[0], face[1]),
        (face[1], face[2]),
        (face[2], face[0]),
    )


cell_type_to_faces_func = {
    "hexahedron": hex_cell_faces,
    "tetra": tetra_cell_faces,
    "wedge": wedge_cell_faces,
    "pyramid": pyramid_cell_faces,
}


def assign_reader(mesh_file_path: str) -> MeshReader:
    try:
        mesh = meshio.read(mesh_file_path)
    except meshio.ReadError as exception:
        error = "Could not read mesh file (meshio error).\n"
        error += f"{exception}"
        raise InvalidMeshException(error) from exception

    if is_3d(mesh):
        return MeshReader3D(mesh)
    elif is_2d(mesh):
        return MeshReader2D(mesh)
    else:
        raise InvalidMeshException("Couldn't decide on mesh dimensionality")
