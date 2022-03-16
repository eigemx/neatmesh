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

        if self.has_quad:
            self.quad_mask = self.faces[:, -1] != -1
            quad_faces = self.faces[self.quad_mask]

            quad_faces_tensor = np.take(self.points, quad_faces, axis=0)[:, 0:5, :]
            (
                self.face_centroids[self.quad_mask],
                self.face_normals[self.quad_mask],
                self.face_areas[self.quad_mask],
                self.face_aspect_ratios[self.quad_mask],
            ) = quad_data_from_tensor(quad_faces_tensor)

    def analyze_cells(self) -> None:
        self.cells_centers = np.array([]).reshape(0, 3)
        self.cells_volumes = np.array([])

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
            self.cells_centers = np.concatenate(
                [self.cells_centers, centers],
                axis=0
            )
            self.cells_volumes = np.concatenate(
                [self.cells_volumes, vols],
                axis=0
            )

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
