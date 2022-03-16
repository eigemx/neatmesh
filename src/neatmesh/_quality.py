import numpy as np

from ._common import meshio_type_to_alpha
from ._reader import MeshReader3D
from ._geometry import *


class QualityInspector3D:
    def __init__(self, reader: MeshReader3D) -> None:
        self.reader = reader

        self.points = self.reader.points
        self.n_points = reader.n_points

        self.faces = np.asarray(self.reader.faces)
        self.n_faces = len(reader.faces)

        self.n_cells = reader.n_cells

    def count_cell_types(self) -> None:
        self.hex_count = 0
        self.tetra_count = 0
        self.wedge_count = 0
        self.pyramid_count = 0

        for cell_block in self.reader.cell_blocks:
            alpha_cell_type = meshio_type_to_alpha[cell_block.type]
            if alpha_cell_type == "hexahedron":
                self.hex_count += len(cell_block.data)
            elif alpha_cell_type == "tetra":
                self.tetra_count += len(cell_block.data)
            elif alpha_cell_type == "pyramid":
                self.pyramid_count += len(cell_block.data)
            elif alpha_cell_type == "wedge":
                self.wedge_count += len(cell_block.data)

    def _calc_face_data_tri(self):
        tri_faces = self.faces[self.faces[:, -1] == -1][:, :-1]
        tri_faces_tensor = np.take(self.points, tri_faces, axis=0)[:, 0:3, :]

        (
            self.tri_centers,
            self.tri_normals,
            self.tri_areas,
            self.tri_aspect_ratios,
        ) = tri_data_from_tensor(tri_faces_tensor)

    def _calc_face_data_quad(self):
        quad_faces = self.faces[self.faces[:, -1] != -1]

        quad_faces_tensor = np.take(self.points, quad_faces, axis=0)[:, 0:5, :]
        (
            self.quad_centroids,
            self.quad_normals,
            self.quad_areas,
            self.quad_aspect_ratios,
        ) = quad_data_from_tensor(quad_faces_tensor)

    def _calc_cell_data_tetra(self):
        for cell_block in self.reader.cell_blocks:
            if meshio_type_to_alpha[cell_block.type] == "tetra":
                cells = cell_block.data

        tetra_cells_tensor = np.take(self.points, cells, axis=0)[:, 0:5, :]
        self.tetra_centers, self.tetra_vols = tetra_data_from_tensor(tetra_cells_tensor)

    def _calc_cell_data_hex(self):
        for cell_block in self.reader.cell_blocks:
            if meshio_type_to_alpha[cell_block.type] == "hexahedron":
                cells = cell_block.data

        hex_cells_tensor = np.take(self.points, cells, axis=0)[:, 0:8, :]

        self.hex_centers, self.hex_vols = hex_data_from_tensor(hex_cells_tensor)

    def _calc_cell_data_wedge(self):
        wedge_cells_tensor = np.zeros(shape=(self.wedge_count, 6, 3))

        for cell_block in self.reader.cell_blocks:
            if meshio_type_to_alpha[cell_block.type] == "wedge":
                cells = cell_block.data

        wedge_cells_tensor = np.take(self.points, cells, axis=0)[:, 0:6, :]
        wedge_data_from_tensor(wedge_cells_tensor)

    def _calc_cell_data_pyramid(self):
        for cell_block in self.reader.cell_blocks:
            if meshio_type_to_alpha[cell_block.type] == "pyramid":
                cells = cell_block.data

        pyr_cells_tensor = np.take(self.points, cells, axis=0)[:, 0:5, :]

        (self.pyramids_centroids, self.pyramids_vol) = pyramid_data_from_tensor(
            pyr_cells_tensor
        )

    def calc_cells_data(self) -> None:
        self.cells_centers = np.zeros(shape=(self.n_cells, 3))
        self.cells_volumes = np.zeros(shape=(self.n_cells,))

        cell_type_data_handler_map = {
            "hexahedron": self._calc_cell_data_hex,
            "tetra": self._calc_cell_data_tetra,
            "wedge": self._calc_cell_data_wedge,
            "pyramid": self._calc_cell_data_pyramid,
        }

        for i, (cell, cell_type) in enumerate(self.reader.cells()):
            (
                self.cells_centers[i, :],
                self.cells_volumes[i],
            ) = cell_type_data_handler_map[cell_type](cell)

    def calc_faces_nonortho(self) -> None:
        self.non_ortho = np.zeros(self.faces_areas.shape)

        for i, face in enumerate(self.reader.faces_set):
            face_id = self.reader.face_to_faceid[face]
            owner_cell_id, neigbor_cell_id = self.reader.faceid_to_cellid[face_id]

            if neigbor_cell_id == -1:
                # Boundary face, nothing to do here
                self.non_ortho[i] = np.nan
                continue

            owner_center = self.cells_centers[owner_cell_id]
            neighbor_center = self.cells_centers[neigbor_cell_id]

            # face normal
            sf = self.faces_normals[face_id]

            # Line connecting current and adjacent cells centroids.
            ef = neighbor_center - owner_center

            if _dot(ef, sf) < 0:
                ef = -ef

            # Angle between ef and sf.
            costheta = _dot(ef, sf) / (_norm(ef) * _norm(sf))
            self.non_ortho[i] = acos(costheta) * (180.0 / pi)
