"""neatmesh mesh 2D & 3D analyzers"""
# pylint: disable=too-many-instance-attributes
from typing import Tuple

import numpy as np

from ._common import meshio_type_to_alpha
from ._geometry import (
    dot,
    dot_normalize,
    hex_data_from_tensor,
    pyramid_data_from_tensor,
    quad_data_from_tensor,
    tetra_data_from_tensor,
    tri_data_from_tensor,
    wedge_data_from_tensor,
)
from ._reader import MeshReader2D, MeshReader3D


class Analyzer2D:
    """Analyze a 2D mesh and provide edges and faces count,
    faces areas, faces aspect ratios and adjacent faces volume ratios.

    Analyzer2D requires an instance of a MeshReader2D as an input.
    """

    def __init__(self, reader: MeshReader2D) -> None:
        self.reader = reader

        self.points = self.reader.points
        self.n_points = reader.n_points
        self.edges = self.reader.edges
        self.n_edges = len(reader.edges)
        self.n_faces = reader.n_faces

        self.n_quad = 0
        self.n_tri = 0

        self.face_centers: np.ndarray = np.array([]).reshape(0, 3)
        self.face_areas: np.ndarray = np.array([])
        self.face_aspect_ratios: np.ndarray = np.array([])
        self.owner_neighbor: np.ndarray = np.array([])

        # add 3rd dimension to points, to use geometry module 3D tri & quad functions
        if self.points.shape[1] == 2:
            self.__3d_points = np.concatenate(
                [self.points, np.zeros(shape=(self.n_points, 1))], axis=1
            )
        else:
            self.__3d_points = self.points

        # transform list of edge tuples, to array of point coordinates
        # output shape = (n_internal_edges, 2, 3)
        self.__edges_tensor = np.take(self.__3d_points, self.edges, axis=0)

        self.__interior_edges_shared_faces: np.ndarray = np.array([])
        self.n_boundary_edges: int = 0

        self.non_ortho: np.ndarray = np.array([])
        self.adj_ratio: np.ndarray = np.array([])

    def duplicate_nodes_count(self) -> int:
        """Count number of duplicate points/nodes"""
        return self.points.shape[0] - np.unique(self.points, axis=0).shape[0]

    def bounding_box(self) -> Tuple:
        """Return coordinates of bounding box of the mesh (min & max coords)"""
        x_min, y_min = np.min(self.points, axis=0)
        x_max, y_max = np.max(self.points, axis=0)

        return (
            (x_min, y_min),
            (x_max, y_max),
        )

    def count_face_types(self) -> None:
        """Count and update the number of quad and triangle faces"""
        for cell_block in self.reader.cell_blocks:
            face_type_str = meshio_type_to_alpha[cell_block.type]
            if face_type_str == "quad":
                self.n_quad += len(cell_block.data)

            elif face_type_str == "triangle":
                self.n_tri += len(cell_block.data)

    def analyze_faces(self) -> None:
        """Call appropriate methods for each face type and update the following:
        faces centers, faces areas and face aspect ratios.
        """
        for cell_block in self.reader.cell_blocks:
            face_type = meshio_type_to_alpha[cell_block.type]
            faces_tensor = np.take(self.__3d_points, cell_block.data, axis=0)

            if face_type == "quad":
                centers, _, areas, aspect_ratios = quad_data_from_tensor(faces_tensor)
            else:
                centers, _, areas, aspect_ratios = tri_data_from_tensor(faces_tensor)

            self.face_centers = np.concatenate([self.face_centers, centers])
            self.face_areas = np.concatenate([self.face_areas, areas])
            self.face_aspect_ratios = np.concatenate(
                [self.face_aspect_ratios, aspect_ratios]
            )

    def analyze_non_ortho(self) -> None:
        """For each internal edge, calculate the non-orthogonality between the faces.
        Non-orthogonality is defined as the angle (in degrees) between the vector
        connecting two neighbor faces centroids, and shared edge normal vector.
        """
        # owner_neighbor is 2D matrix of shape (n_interior_edges, 2)
        # first column is the index of the owner faces,
        # second column is the index of neighbor faces.
        # second column contains '-1' for boudnary edges (no neighbor faces).
        self.owner_neighbor = np.asarray(list(self.reader.edgeid_to_faceid.values()))

        # filter our boundary edges
        interior_edges_mask = self.owner_neighbor[:, 1] != -1

        # We will need interior_edges later in adjacent cells volume ratio
        self.__interior_edges_shared_faces = self.owner_neighbor[interior_edges_mask]

        self.n_boundary_edges = (
            self.n_edges - self.__interior_edges_shared_faces.shape[0]
        )

        owners, neighbors = (
            self.__interior_edges_shared_faces[:, 0],
            self.__interior_edges_shared_faces[:, 1],
        )
        owner_centers = np.take(self.face_centers, owners, axis=0)
        neighbor_centers = np.take(self.face_centers, neighbors, axis=0)

        # internal_edges_tensor has shape (n_internal_edges, 2, 3)
        # and represents the coordinates of first and second coordinates of
        # each edge.
        internal_edges_tensor = self.__edges_tensor[interior_edges_mask]

        internal_edge_vectors = (
            internal_edges_tensor[:, 1, :] - internal_edges_tensor[:, 0, :]
        )

        # array of vectors connecting adjacent faces centroids.
        neigbor_owner_vectors = neighbor_centers - owner_centers

        costheta = np.abs(dot_normalize(neigbor_owner_vectors, internal_edge_vectors))
        self.non_ortho = 90 - (np.arccos(costheta) * (180.0 / np.pi))

    def analyze_adjacents_area_ratio(self) -> None:
        """Calculate the area ratio for each two neighbor faces (max/min)"""
        adjacent_faces_areas = np.take(
            self.face_areas, self.__interior_edges_shared_faces, axis=0
        )
        self.adj_ratio = np.max(
            [
                adjacent_faces_areas[:, 0] / adjacent_faces_areas[:, 1],
                adjacent_faces_areas[:, 1] / adjacent_faces_areas[:, 0],
            ],
            axis=0,
        )


class Analyzer3D:
    """Analyze a 3D mesh and provide faces and cells count,
    faces areas, faces aspect ratios and adjacent cells volume ratios.

    Analyzer3D requires an instance of a MeshReader3D as an input.
    """

    def __init__(self, reader: MeshReader3D) -> None:
        self.reader = reader

        self.points = self.reader.points
        self.n_points = reader.n_points

        self.faces = np.asarray(self.reader.faces)
        self.n_faces = len(reader.faces)

        self.n_cells = reader.n_cells

        self.hex_count = 0
        self.tetra_count = 0
        self.wedge_count = 0
        self.pyramid_count = 0
        self.has_tri = False
        self.has_quad = False

        self.face_centers = np.zeros(shape=(self.n_faces, 3))
        self.face_normals = np.zeros(shape=(self.n_faces, 3))
        self.face_areas = np.zeros(shape=(self.n_faces,))
        self.face_aspect_ratios = np.zeros(shape=(self.n_faces,))
        self.owner_neighbor: np.ndarray = np.array([])

        self.n_tri = 0
        self.n_quad = 0

        self.cells_centers: np.ndarray = np.array([]).reshape(0, 3)
        self.cells_volumes: np.ndarray = np.array([])

        self.interior_faces: np.ndarray = np.array([])
        self.n_boundary_faces: int = 0

        self.non_ortho: np.ndarray = np.array([])
        self.adj_ratio: np.ndarray = np.array([])

    def duplicate_nodes_count(self) -> int:
        """Count number of duplicate points/nodes"""
        return self.points.shape[0] - np.unique(self.points, axis=0).shape[0]

    def bounding_box(self) -> Tuple:
        """Return coordinates of bounding box of the mesh (min & max coords)"""
        x_min, y_min, z_min = np.min(self.points, axis=0)
        x_max, y_max, z_max = np.max(self.points, axis=0)

        return (
            (x_min, y_min, z_min),
            (x_max, y_max, z_max),
        )

    def count_cell_types(self) -> None:
        """Count and update the number of cells for each cell type"""
        for cell_block in self.reader.cell_blocks:
            cell_type_str = meshio_type_to_alpha[cell_block.type]
            if cell_type_str == "hexahedron":
                self.hex_count += len(cell_block.data)
                self.has_quad = True

            elif cell_type_str == "tetra":
                self.tetra_count += len(cell_block.data)
                self.has_tri = True

            elif cell_type_str == "pyramid":
                self.pyramid_count += len(cell_block.data)
                self.has_quad = True
                self.has_tri = True

            elif cell_type_str == "wedge":
                self.wedge_count += len(cell_block.data)
                self.has_quad = True
                self.has_tri = True

    def analyze_faces(self):
        """Call appropriate methods for each face type and update the following:
        faces centers, faces areas, face normals and face aspect ratios.
        """
        if self.has_tri:
            tri_mask = self.faces[:, -1] == -1
            tri_faces = self.faces[tri_mask][:, :-1]
            tri_faces_tensor = np.take(self.points, tri_faces, axis=0)[:, 0:3, :]

            (
                self.face_centers[tri_mask],
                self.face_normals[tri_mask],
                self.face_areas[tri_mask],
                self.face_aspect_ratios[tri_mask],
            ) = tri_data_from_tensor(tri_faces_tensor)
            self.n_tri += len(tri_faces)

        if self.has_quad:
            quad_mask = self.faces[:, -1] != -1
            quad_faces = self.faces[quad_mask]

            quad_faces_tensor = np.take(self.points, quad_faces, axis=0)[:, 0:5, :]
            (
                self.face_centers[quad_mask],
                self.face_normals[quad_mask],
                self.face_areas[quad_mask],
                self.face_aspect_ratios[quad_mask],
            ) = quad_data_from_tensor(quad_faces_tensor)
            self.n_quad += len(quad_faces)

    def analyze_cells(self) -> None:
        """Call appropriate methods for each cell type and update cells centers
        and cells volumes.
        """
        cell_type_handler_fn_map = {
            "hexahedron": hex_data_from_tensor,
            "tetra": tetra_data_from_tensor,
            "wedge": wedge_data_from_tensor,
            "pyramid": pyramid_data_from_tensor,
        }

        for cell_block in self.reader.cell_blocks:
            cell_type_str = meshio_type_to_alpha[cell_block.type]
            cell_geometry_handler_fn = cell_type_handler_fn_map[cell_type_str]

            data_tensor = np.take(self.points, cell_block.data, axis=0)
            centers, vols = cell_geometry_handler_fn(data_tensor)
            self.cells_centers = np.concatenate([self.cells_centers, centers], axis=0)
            self.cells_volumes = np.concatenate([self.cells_volumes, vols], axis=0)

    def analyze_non_ortho(self) -> None:
        """For each internal face, calculate the non-orthogonality between the cells.
        Non-orthogonality is defined as the angle (in degrees) between the vector
        connecting two neighbor cells centroids, and shared face normal vector.
        """
        self.owner_neighbor = np.asarray(list(self.reader.faceid_to_cellid.values()))
        interior_faces_mask = self.owner_neighbor[:, 1] != -1

        # We will need interior_faces later in adjacent cells volume ratio
        self.interior_faces = self.owner_neighbor[interior_faces_mask]

        self.n_boundary_faces = self.n_faces - self.interior_faces.shape[0]

        owners, neighbors = self.interior_faces[:, 0], self.interior_faces[:, 1]
        owner_centers = np.take(self.cells_centers, owners, axis=0)
        neighbor_centers = np.take(self.cells_centers, neighbors, axis=0)

        interior_face_normals = self.face_normals[interior_faces_mask]
        nei_owner_vectors = neighbor_centers - owner_centers
        dot_product = dot(nei_owner_vectors, interior_face_normals)

        nei_owner_vectors[dot_product < 0] = -nei_owner_vectors[dot_product < 0]

        costheta = dot_normalize(nei_owner_vectors, interior_face_normals)
        self.non_ortho = np.arccos(costheta) * (180.0 / np.pi)

    def analyze_adjacents_volume_ratio(self) -> None:
        """Calculate the area ratio for each two neighbor cells (max/min)"""
        adjacent_cells_vol = np.take(self.cells_volumes, self.interior_faces, axis=0)
        self.adj_ratio = np.max(
            [
                adjacent_cells_vol[:, 0] / adjacent_cells_vol[:, 1],
                adjacent_cells_vol[:, 1] / adjacent_cells_vol[:, 0],
            ],
            axis=0,
        )
