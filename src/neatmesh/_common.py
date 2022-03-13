from typing import List, Tuple, Final
from meshio import Mesh


def hex_cell_faces(cell: List) -> Tuple[Tuple[int, ...], ...]:
    """Returns coordinates of 6 faces of a hexahedron cell, using meshio nodes ordering
    Args:
        cell (List): list of points defining the cell
    Returns:
        List[List]: list of list of faces points labels
    """
    return (
        (cell[0], cell[3], cell[2], cell[1]),
        (cell[4], cell[5], cell[6], cell[7]),
        (cell[0], cell[1], cell[5], cell[4]),
        (cell[2], cell[3], cell[7], cell[6]),
        (cell[0], cell[4], cell[7], cell[3]),
        (cell[1], cell[2], cell[6], cell[5]),
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


class MeshIOCellType:
    """meshio cell types"""

    Hex = "hexahedron"
    Hex20 = "hexahedron20"
    Hex24 = "hexahedron24"
    Hex27 = "hexahedron27"
    Tetra = "tetra"
    Tetra10 = "tetra10"
    Wedge = "wedge"
    Wedge12 = "wedge12"
    Wedge15 = "wedge15"
    Pyramid = "pyramid"
    Pyramid13 = "pyramid13"
    Pyramid14 = "pyramid14"


meshio_3d: Final = {
    "tetra",
    "tetra10",
    "hexahedron",
    "hexahedron20",
    "hexahedron24",
    "hexahedron27",
    "wedge",
    "wedge12",
    "wedge15",
    "pyramid",
    "pyramid13",
    "pyramid14",
}

meshio_type_to_alpha: Final = {
    "vertex": "vertex",
    "line": "line",
    "line": "line3",
    "triangle": "triangle",
    "triangle6": "triangle",
    "triangle7": "triangle",
    "quad": "quad",
    "quad8": "quad",
    "quad9": "quad",
    "tetra": "tetra",
    "tetra10": "tetra",
    "hexahedron": "hexahedron",
    "hexahedron20":"hexahedron",
    "hexahedron24":"hexahedron",
    "hexahedron27":"hexahedron",
    "wedge": "wedge",    
    "wedge12":"wedge",
    "wedge15":"wedge",
    "pyramid":"pyramid",
    "pyramid13": "pyramid",
    "pyramid14":   "pyramid",
}

meshio_2d: Final = {
    "triangle",
    "quad",
}

meshio_1d: Final = {
    "vertex",
    "line",
    "line3",
}

cell_type_to_faces_func: Final = {
    MeshIOCellType.Hex: hex_cell_faces,
    MeshIOCellType.Tetra: tetra_cell_faces,
    MeshIOCellType.Wedge: wedge_cell_faces,
    MeshIOCellType.Pyramid: pyramid_cell_faces,
}

def is_3d(mesh: Mesh) -> bool:
    """Check if a meshio mesh is 3-dimensional"""
    for cell_block in mesh.cells:
        # first 3D element type is enough.
        if meshio_type_to_alpha[cell_block.type] in meshio_3d:
            return True
    return False


def is_2d(mesh: Mesh) -> bool:
    """Check ifa meshio mesh is 2-dimensional"""
    if is_3d(mesh):
        return False

    for cell_block in mesh.cells:
        if meshio_type_to_alpha[cell_block.type] in meshio_2d:
            return True
    return False
