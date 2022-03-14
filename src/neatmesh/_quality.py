import numpy as np

from ._common import meshio_type_to_alpha
from ._reader import MeshIOCellType, MeshReader3D
from ._geometry import *

class QualityInspector3D:
    def __init__(self, mr: MeshReader3D) -> None:
        self.reader = mr

        self.points = self.reader.points
        self.n_points = mr.n_points

        self.faces = np.asarray(self.reader.faces)
        self.n_faces = len(mr.faces)

        self.n_cells = mr.n_cells

    def calc_cell_types_counts(self) -> None:
        self.hex_count = 0
        self.tetra_count = 0
        self.wedge_count = 0
        self.pyramid_count = 0

        for cell_block in self.reader.cell_blocks:
            alpha_cell_type = meshio_type_to_alpha[cell_block.type]
            if  alpha_cell_type == "hexahedron":
                self.hex_count += len(cell_block.data)
            elif alpha_cell_type == "tetra":
                self.tetra_count += len(cell_block.data)
            elif alpha_cell_type == "pyramid":
                self.pyramid_count += len(cell_block.data)
            elif alpha_cell_type == "wedge":
                self.wedge_count += len(cell_block.data)

    def _calc_face_data_tri(self):
        tri_faces = self.faces[self.faces[:,-1] == -1][:,:-1]
        n_faces = tri_faces.shape[0]

        tri_faces_tensor = np.zeros(shape=(n_faces, 3, 3))

        for i in range(n_faces):
            tri_faces_tensor[i] = [
                self.points[tri_faces[i][0]],
                self.points[tri_faces[i][1]],
                self.points[tri_faces[i][2]],
            ]

        (
            self.tri_centers,
            self.tri_normals,
            self.tri_areas,
            self.tri_aspect_ratios
        ) = tri_data_from_tensor(tri_faces_tensor)

    def _calc_face_data_quad(self):
        # Danger: this will mix up between hex quads and wedge/pyramid quads
        quad_faces = self.faces[self.faces[:,-1] != -1]
        n_faces = quad_faces.shape[0]

        quad_faces_tensor = np.zeros(shape=(n_faces, 4, 3))

        for i in range(n_faces):
            quad_faces_tensor[i] = [
                self.points[quad_faces[i][0]],
                self.points[quad_faces[i][1]],
                self.points[quad_faces[i][2]],
                self.points[quad_faces[i][3]],
            ]
        (
            self.quad_centroids,
            self.quad_normals,
            self.quad_areas,
            self.quad_aspect_ratios
        ) = quad_data_from_tensor(quad_faces_tensor)

    def _calc_cell_data_tetra(self):
        tetra_cells_tensor = np.zeros(shape=(self.tetra_count, 4, 3))
        for cell_block in self.reader.cell_blocks:
            if meshio_type_to_alpha[cell_block.type] == "tetra":
                cells = cell_block.data

        for i, cell in enumerate(cells):
            tetra_cells_tensor[i] = (
                self.points[cell[0]],
                self.points[cell[1]],
                self.points[cell[2]],
                self.points[cell[3]],
            )

        self.tetra_centers, self.tetra_vols = tetra_data_from_tensor(tetra_cells_tensor)

    def _calc_cell_data_hex(self):
        hex_cells_tensor = np.zeros(shape=(self.hex_count, 8, 3))

        for cell_block in self.reader.cell_blocks:
            if meshio_type_to_alpha[cell_block.type] == "hexahedron":
                cells = cell_block.data

        for i, cell in enumerate(cells):
            hex_cells_tensor[i] = (
                self.points[cell[0]],
                self.points[cell[1]],
                self.points[cell[2]],
                self.points[cell[3]],
                self.points[cell[4]],
                self.points[cell[5]],
                self.points[cell[6]],
                self.points[cell[7]],
            )

        self.hex_centers, self.hex_vols = hex_data_from_tensor(hex_cells_tensor)

    def _calc_cell_data_wedge(self):
        wedge_cells_tensor = np.zeros(shape=(self.wedge_count, 6, 3))

        for cell_block in self.reader.cell_blocks:
            if meshio_type_to_alpha[cell_block.type] == "pyramid":
                cells = cell_block.data

        for i, cell in enumerate(cells):
            wedge_cells_tensor[i] = (
                self.points[cell[0]],
                self.points[cell[1]],
                self.points[cell[2]],
                self.points[cell[3]],
                self.points[cell[4]],
                self.points[cell[5]],
            )

    def _calc_cell_data_pyramid(self):
        pyr_cells_tensor = np.zeros(shape=(self.pyramid_count, 5, 3))

        for cell_block in self.reader.cell_blocks:
            if meshio_type_to_alpha[cell_block.type] == "pyramid":
                cells = cell_block.data

        for i, cell in enumerate(cells):
            pyr_cells_tensor[i] = (
                self.points[cell[0]],
                self.points[cell[1]],
                self.points[cell[2]],
                self.points[cell[3]],
                self.points[cell[4]],
            )

        (
            self.pyramids_centroids,
            self.pyramids_vol
        ) = pyramid_data_from_tensor(pyr_cells_tensor)

    def calc_cells_data(self) -> None:
        self.cells_centers = np.zeros(shape=(self.n_cells, 3))
        self.cells_volumes = np.zeros(shape=(self.n_cells,))

        cell_type_data_handler_map = {
            MeshIOCellType.Hex: self._calc_cell_data_hex,
            MeshIOCellType.Hex20: self._calc_cell_data_hex,
            MeshIOCellType.Hex24: self._calc_cell_data_hex,
            MeshIOCellType.Hex27: self._calc_cell_data_hex,
            MeshIOCellType.Tetra: self._calc_cell_data_tetra,
            MeshIOCellType.Tetra10: self._calc_cell_data_tetra,
            MeshIOCellType.Wedge: self._calc_cell_data_wedge,
            MeshIOCellType.Wedge12: self._calc_cell_data_wedge,
            MeshIOCellType.Wedge15: self._calc_cell_data_wedge,
            MeshIOCellType.Pyramid: self._calc_cell_data_pyramid,
            MeshIOCellType.Pyramid13: self._calc_cell_data_pyramid,
            MeshIOCellType.Pyramid14: self._calc_cell_data_pyramid,
        }
        for i, (cell, cell_type) in enumerate(self.reader.cells()):
            self.cells_centers[i, :], self.cells_volumes[i] = cell_type_data_handler_map[
                cell_type
            ](cell)

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
