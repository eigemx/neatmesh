from typing import Tuple

import numpy as np
from numpy.linalg import norm

from .common import meshio_type_to_alpha
from .geometry import (
    hex_data_from_tensor,
    pyramid_data_from_tensor,
    quad_data_from_tensor,
    tetra_data_from_tensor,
    tri_data_from_tensor,
    wedge_data_from_tensor,
)
from .reader import MeshReader2D, MeshReader3D


class Analyzer2D:
    def __init__(self, reader: MeshReader2D) -> None:
        self.reader = reader

        self.points = self.reader.points
        self.n_points = reader.n_points

        self.n_edges = len(reader.edges)
        self.n_faces = reader.n_faces

    def duplicate_nodes_count(self) -> int:
        return self.points.shape[0] - np.unique(self.points, axis=0).shape[0]

    def bounding_box(self) -> Tuple:
        x_min, y_min = np.min(self.points, axis=0)
        x_max, y_max = np.max(self.points, axis=0)

        return (
            (x_min, y_min),
            (x_max, y_max),
        )

    def count_face_types(self) -> None:
        self.n_quad = 0
        self.n_tri = 0

        for cell_block in self.reader.cell_blocks:
            alpha_face_type = meshio_type_to_alpha[cell_block.type]
            if alpha_face_type == "quad":
                self.n_quad += len(cell_block.data)

            elif alpha_face_type == "triangle":
                self.n_tri += len(cell_block.data)

    def analyze_faces(self) -> None:
        self.face_centers = np.array([]).reshape(0, 3)
        self.face_areas = np.array([])
        self.face_aspect_ratios = np.array([])

        # Add points 3rd dimension, to use same geometry module tri & quad functions
        self.__3d_points = np.concatenate(
            [self.points, np.zeros(shape=(self.n_points, 1))], axis=1
        )

        # This translates list of edge tuples, to point coordinates
        self.__edges_tensor = np.take(self.__3d_points, self.reader.edges, axis=0)

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
        # owner_neighbor is 2D matrix of shape (n_interior_edges, 2)
        # first column is the index of the owner faces,
        # second column is the index of neighbor faces.
        # second column contains '-1' for boudnary edges.
        owner_neighbor = np.asarray(list(self.reader.edgeid_to_faceid.values()))
        interior_edges_mask = owner_neighbor[:, 1] != -1

        # We will need interior_edges later in adjacent cells volume ratio
        self.interior_edges = owner_neighbor[interior_edges_mask]

        self.n_boundary_edges = self.n_edges - self.interior_edges.shape[0]

        owners, neighbors = self.interior_edges[:, 0], self.interior_edges[:, 1]
        owner_centers = np.take(self.face_centers, owners, axis=0)
        neighbor_centers = np.take(self.face_centers, neighbors, axis=0)

        internal_edges_tensor = self.__edges_tensor[interior_edges_mask]
        edges_vectors = internal_edges_tensor[:, 1, :] - internal_edges_tensor[:, 0, :]

        ef = neighbor_centers - owner_centers
        dot = lambda x, y: np.sum(x * y, axis=1) / (norm(x, axis=1) * norm(y, axis=1))

        costheta = np.abs(dot(ef, edges_vectors))
        self.non_ortho = 90 - (np.arccos(costheta) * (180.0 / np.pi))

    def analyze_adjacents_area_ratio(self) -> None:
        adjacent_faces_areas = np.take(self.face_areas, self.interior_edges, axis=0)
        self.adj_ratio = np.max(
            [
                adjacent_faces_areas[:, 0] / adjacent_faces_areas[:, 1],
                adjacent_faces_areas[:, 1] / adjacent_faces_areas[:, 0],
            ],
            axis=0,
        )


class Analyzer3D:
    def __init__(self, reader: MeshReader3D) -> None:
        self.reader = reader

        self.points = self.reader.points
        self.n_points = reader.n_points

        self.faces = np.asarray(self.reader.faces)
        self.n_faces = len(reader.faces)

        self.n_cells = reader.n_cells

    def duplicate_nodes_count(self) -> int:
        return self.points.shape[0] - np.unique(self.points, axis=0).shape[0]

    def bounding_box(self) -> Tuple:
        x_min, y_min, z_min = np.min(self.points, axis=0)
        x_max, y_max, z_max = np.max(self.points, axis=0)

        return (
            (x_min, y_min, z_min),
            (x_max, y_max, z_max),
        )

    def count_cell_types(self) -> None:
        self.hex_count = 0
        self.tetra_count = 0
        self.wedge_count = 0
        self.pyramid_count = 0
        self.has_tri = False
        self.has_quad = False

        for cell_block in self.reader.cell_blocks:
            alpha_cell_type = meshio_type_to_alpha[cell_block.type]
            if alpha_cell_type == "hexahedron":
                self.hex_count += len(cell_block.data)
                self.has_quad = True

            elif alpha_cell_type == "tetra":
                self.tetra_count += len(cell_block.data)
                self.has_tri = True

            elif alpha_cell_type == "pyramid":
                self.pyramid_count += len(cell_block.data)
                self.has_quad = True
                self.has_tri = True

            elif alpha_cell_type == "wedge":
                self.wedge_count += len(cell_block.data)
                self.has_quad = True
                self.has_tri = True

    def analyze_faces(self):
        self.face_centers = np.zeros(shape=(self.n_faces, 3))
        self.face_normals = np.zeros(shape=(self.n_faces, 3))
        self.face_areas = np.zeros(shape=(self.n_faces,))
        self.face_aspect_ratios = np.zeros(shape=(self.n_faces,))

        self.n_tri = 0
        self.n_quad = 0

        if self.has_tri:
            self.tri_mask = self.faces[:, -1] == -1
            tri_faces = self.faces[self.tri_mask][:, :-1]
            tri_faces_tensor = np.take(self.points, tri_faces, axis=0)[:, 0:3, :]

            (
                self.face_centers[self.tri_mask],
                self.face_normals[self.tri_mask],
                self.face_areas[self.tri_mask],
                self.face_aspect_ratios[self.tri_mask],
            ) = tri_data_from_tensor(tri_faces_tensor)
            self.n_tri += len(tri_faces)

        if self.has_quad:
            self.quad_mask = self.faces[:, -1] != -1
            quad_faces = self.faces[self.quad_mask]

            quad_faces_tensor = np.take(self.points, quad_faces, axis=0)[:, 0:5, :]
            (
                self.face_centers[self.quad_mask],
                self.face_normals[self.quad_mask],
                self.face_areas[self.quad_mask],
                self.face_aspect_ratios[self.quad_mask],
            ) = quad_data_from_tensor(quad_faces_tensor)
            self.n_quad += len(quad_faces)

    def analyze_cells(self) -> None:
        self.cells_centers: np.ndarray = np.array([]).reshape(0, 3)
        self.cells_volumes: np.ndarray = np.array([])

        cell_type_handler_map = {
            "hexahedron": self.analyze_hex_cells,
            "tetra": self.analyze_tetra_cells,
            "wedge": self.analyze_wedge_cells,
            "pyramid": self.analyze_pyramid_cells,
        }

        for cell_block in self.reader.cell_blocks:
            ctype = meshio_type_to_alpha[cell_block.type]
            handler = cell_type_handler_map[ctype]
            centers, vols = handler(cell_block.data)
            self.cells_centers = np.concatenate([self.cells_centers, centers], axis=0)
            self.cells_volumes = np.concatenate([self.cells_volumes, vols], axis=0)

    def analyze_tetra_cells(self, cells):
        tetra_cells_tensor = np.take(self.points, cells, axis=0)[:, 0:5, :]
        return tetra_data_from_tensor(tetra_cells_tensor)

    def analyze_hex_cells(self, cells):
        hex_cells_tensor = np.take(self.points, cells, axis=0)[:, 0:8, :]
        return hex_data_from_tensor(hex_cells_tensor)

    def analyze_wedge_cells(self, cells):
        wedge_cells_tensor = np.take(self.points, cells, axis=0)[:, 0:6, :]
        return wedge_data_from_tensor(wedge_cells_tensor)

    def analyze_pyramid_cells(self, cells):
        pyr_cells_tensor = np.take(self.points, cells, axis=0)[:, 0:5, :]
        return pyramid_data_from_tensor(pyr_cells_tensor)

    def analyze_non_ortho(self) -> None:
        owner_neighbor = np.asarray(list(self.reader.faceid_to_cellid.values()))
        interior_faces_mask = owner_neighbor[:, 1] != -1

        # We will need interior_faces later in adjacent cells volume ratio
        self.interior_faces = owner_neighbor[interior_faces_mask]

        self.n_boundary_faces = self.n_faces - self.interior_faces.shape[0]

        owners, neighbors = self.interior_faces[:, 0], self.interior_faces[:, 1]
        owner_centers = np.take(self.cells_centers, owners, axis=0)
        neighbor_centers = np.take(self.cells_centers, neighbors, axis=0)

        sf = self.face_normals[interior_faces_mask]
        ef = neighbor_centers - owner_centers

        dot = lambda x, y: np.sum(x * y, axis=1) / (norm(x, axis=1) * norm(y, axis=1))
        ef[dot(ef, sf) < 0] = -ef[dot(ef, sf) < 0]

        costheta = dot(ef, sf)
        self.non_ortho = np.arccos(costheta) * (180.0 / np.pi)

    def analyze_adjacents_volume_ratio(self) -> None:
        adjacent_cells_vol = np.take(self.cells_volumes, self.interior_faces, axis=0)
        self.adj_ratio = np.max(
            [
                adjacent_cells_vol[:, 0] / adjacent_cells_vol[:, 1],
                adjacent_cells_vol[:, 1] / adjacent_cells_vol[:, 0],
            ],
            axis=0,
        )
