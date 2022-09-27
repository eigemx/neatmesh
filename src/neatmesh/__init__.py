from typing import Union

from ._exceptions import (
    InputMeshDimensionError,
    InvalidMeshError,
    NonSupportedElementError,
)
from ._mesh import Mesh2D, Mesh3D


def read(mesh_file_path: str) -> Union[Mesh2D, Mesh3D]:
    from ._reader import assign_reader, MeshReader2D

    reader = assign_reader(mesh_file_path)

    if isinstance(reader, MeshReader2D):
        return Mesh2D(reader)
    else:
        return Mesh3D(reader)


__all__ = [
    "Mesh2D",
    "Mesh3D",
    "InputMeshDimensionError",
    "InvalidMeshError",
    "NonSupportedElementError",
    "read",
]
