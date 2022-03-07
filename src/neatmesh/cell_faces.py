from typing import Final, List

class MeshIOFaceType:
    """meshio face types"""

    Quad = "quad"
    Triangle = "triangle"


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
    Wedge15 ="wedge15"
    Pyramid = "pyramid"
    Pyramid13 = "pyramid13"
    Pyramid14 = "pyramid14"


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

def hex_cell_faces(cell: List) -> List[List]:
    """Returns coordinates of 6 faces of a hexahedron cell, using meshio nodes ordering
    Args:
        cell (List): list of points defining the cell
    Returns:
        List[List]: list of list of faces points labels
    """
    faces = (
        (cell[0], cell[3], cell[2], cell[1]),
        (cell[4], cell[5], cell[6], cell[7]),
        (cell[0], cell[1], cell[5], cell[4]),
        (cell[2], cell[3], cell[7], cell[6]),
        (cell[0], cell[4], cell[7], cell[3]),
        (cell[1], cell[2], cell[6], cell[5]),
    )
    return faces


def hex20_cell_faces(cell: List) -> List[List]:
    """Returns coordinates of 6 faces of a hexahedron cell, 
    using meshio nodes ordering for hexahedron20 and hexahedron24
    Args:
        cell (List): list of points defining the cell
    Returns:
        List[List]: list of list of faces points labels
    """
    faces = (
        (cell[0], cell[11], cell[3], cell[10], cell[2], cell[9], cell[1], cell[8]),
        (cell[4], cell[12], cell[5], cell[13], cell[6], cell[14], cell[7], cell[15]),
        (cell[0], cell[8], cell[1], cell[17], cell[5], cell[12], cell[4], cell[16]),
        (cell[2], cell[10], cell[3], cell[19], cell[7], cell[14], cell[6], cell[18]),
        (cell[0], cell[16], cell[4], cell[15], cell[7], cell[19], cell[3], cell[11]),
        (cell[1], cell[9], cell[2], cell[18], cell[6], cell[13], cell[5], cell[17]),
    )
    return faces


def wedge_cell_faces(cell: List) -> List[List]:
    """Returns coordinates of 5 faces of a wedge cell, 
    using meshio nodes ordering for wedge
    Args:
        cell (List): list of points defining the cell
    Returns:
        List[List]: list of list of faces points labels
    """
    faces =(
        (cell[0], cell[2], cell[1]),
        (cell[3], cell[4], cell[5]),
        (cell[3], cell[0], cell[1], cell[4]),
        (cell[0], cell[3], cell[5], cell[2]),
        (cell[1], cell[2], cell[5], cell[4]),
    )
    return faces


def wedge12_cell_faces(cell: List) -> List[List]:
    """Returns coordinates of 5 faces of a wedge cell, 
    using meshio nodes ordering for wedge12 and wedge15
    Args:
        cell (List): list of points defining the cell
    Returns:
        List[List]: list of list of faces points labels
    """
    faces =(
        (cell[0], cell[8], cell[2], cell[7], cell[1], cell[6]),
        (cell[3], cell[9], cell[4], cell[10], cell[5], cell[11]),
        (cell[3], cell[0], cell[6], cell[1], cell[4], cell[9]),
        (cell[0], cell[3], cell[11], cell[5], cell[2], cell[8]),
        (cell[1], cell[7], cell[2], cell[5], cell[10], cell[4]),
    )
    return faces


def tetra_cell_faces(cell: List) -> List[List]:
    """Returns coordinates of 4 faces of a tetrahedral cell, 
    using meshio nodes ordering for tetra
    Args:
        cell (List): list of points defining the cell
    Returns:
        List[List]: list of list of faces points labels
    """
    faces = (
        (cell[0], cell[2], cell[1]),
        (cell[1], cell[2], cell[3]),
        (cell[0], cell[1], cell[3]),
        (cell[0], cell[3], cell[2]),
    )
    return faces


def tetra10_cell_faces(cell: List) -> List[List]:
    """Returns coordinates of 4 faces of a tetrahedral cell, 
    using meshio nodes ordering for tetra10
    Args:
        cell (List): list of points defining the cell
    Returns:
        List[List]: list of list of faces points labels
    """
    faces = (
        (cell[0], cell[6], cell[2], cell[5], cell[1], cell[4]),
        (cell[1], cell[5], cell[2], cell[9], cell[3], cell[8]),
        (cell[0], cell[4], cell[1], cell[8], cell[3], cell[7]),
        (cell[0], cell[7], cell[3], cell[9], cell[2], cell[6],),
    )
    return faces


def pyramid_cell_faces(cell: List) -> List[List]:
    """Returns coordinates of 4 faces of a tetrahedral cell, 
    using meshio nodes ordering for pyramid
    Args:
        cell (List): list of points defining the cell
    Returns:
        List[List]: list of list of faces points labels
    """
    faces = (
        (cell[2], cell[1], cell[0], cell[3]),
        (cell[2], cell[3], cell[4]),
        (cell[1], cell[4], cell[0]),
        (cell[3], cell[0], cell[4]),
    )
    return faces


def pyramid13_cell_faces(cell: List) -> List[List]:
    """Returns coordinates of 4 faces of a tetrahedral cell, 
    using meshio nodes ordering for pyramid13 and pyramid14
    Args:
        cell (List): list of points defining the cell
    Returns:
        List[List]: list of list of faces points labels
    """
    faces = (
        (cell[2], cell[6], cell[1], cell[5], cell[0], cell[8], cell[3], cell[7]),
        (cell[2], cell[7], cell[3], cell[12], cell[4], cell[11]),
        (cell[1], cell[10], cell[4], cell[9], cell[0], cell[5]),
        (cell[3], cell[8], cell[0], cell[9], cell[4], cell[12],),
    )
    return faces
