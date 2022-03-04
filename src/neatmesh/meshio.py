import numpy as np
import meshio
from typing import Final


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


if __name__ == "__main__":
    mesh = meshio.read("fine_plane.med")
    print(_is_3d(mesh))
