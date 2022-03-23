"""Neatmesh derived exceptions"""


class InvalidMeshException(Exception):
    """Raised by MeshReader in cases of not finding relevant cell types in
    given mesh or when encountering issues with mesh dimensionality"""

    ...


class NonSupportedElement(Exception):
    """Raised by Mesh Reader in case of encountering unsupported cell type"""

    ...
