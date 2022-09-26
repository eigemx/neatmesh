"""Neatmesh derived exceptions"""


class InvalidMeshError(Exception):
    """Raised by MeshReader in cases of not finding relevant cell types in
    given mesh or when encountering issues with mesh dimensionality"""


class NonSupportedElementError(Exception):
    """Raised by Mesh Reader in case of encountering unsupported cell type"""


class InputMeshDimensionError(Exception):
    """Raised by Mesh3D or Mesh2D in case either one is called in place of the other"""
