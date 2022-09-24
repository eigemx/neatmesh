"""neatmesh common constants and functions"""

from meshio import Mesh

meshio_3d_elements = {
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

meshio_type_to_alpha = {
    "vertex": "vertex",
    "line": "line",
    "line3": "line",
    "triangle": "triangle",
    "triangle6": "triangle",
    "triangle7": "triangle",
    "quad": "quad",
    "quad8": "quad",
    "quad9": "quad",
    "tetra": "tetra",
    "tetra10": "tetra",
    "hexahedron": "hexahedron",
    "hexahedron20": "hexahedron",
    "hexahedron24": "hexahedron",
    "hexahedron27": "hexahedron",
    "wedge": "wedge",
    "wedge12": "wedge",
    "wedge15": "wedge",
    "pyramid": "pyramid",
    "pyramid13": "pyramid",
    "pyramid14": "pyramid",
}

meshio_2d_elements = {
    "triangle",
    "quad",
}

meshio_1d_elements = {
    "vertex",
    "line",
    "line3",
}


def is_3d_mesh(mesh: Mesh) -> bool:
    """Check if a meshio mesh is 3-dimensional"""
    for cell_block in mesh.cells:
        # first 3D element type is enough.
        if meshio_type_to_alpha[cell_block.type] in meshio_3d_elements:
            return True
    return False


def is_2d_mesh(mesh: Mesh) -> bool:
    """Check ifa meshio mesh is 2-dimensional"""
    if is_3d_mesh(mesh):
        return False

    for cell_block in mesh.cells:
        if meshio_type_to_alpha[cell_block.type] in meshio_2d_elements:
            return True
    return False
