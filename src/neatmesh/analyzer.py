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
from .reader import MeshReader3D


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
