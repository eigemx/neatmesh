from typing import Tuple, Union
from math import sqrt, pi, acos

import numpy as np

from ._reader import MeshReader3D, MeshIOCellType
from ._common import meshio_3d_to_alpha

class QualityInspector3D:
    def __init__(self, mr: MeshReader3D) -> None:
        self.reader = mr
        self.n_points = mr.n_points
        self.n_faces = len(mr.faces)
        self.n_cells = mr.n_cells

    def calc_cell_types_counts(self) -> None:
        self.hex_count = 0
        self.tetra_count = 0
        self.wedge_count = 0
        self.pyramid_count = 0

        for cell_block in self.reader.mesh.cells:
            if meshio_3d_to_alpha[cell_block.type] == "hexahedron":
                self.hex_count += len(cell_block.data)

            elif meshio_3d_to_alpha[cell_block.type] == "tetra":
                self.tetra_count += len(cell_block.data)

            elif meshio_3d_to_alpha[cell_block.type] == "pyramid":
                self.pyramid_count += len(cell_block.data)

            elif meshio_3d_to_alpha[cell_block.type] == "wedge":
                self.wedge_count += len(cell_block.data)

    def mesh_bounding_box(self) -> Tuple:
        x_min, y_min, z_min = np.min(self.reader.points, axis=0)
        x_max, y_max, z_max = np.max(self.reader.points, axis=0)
        return ((x_min, y_min, z_min), (x_max, y_max, z_max))

    def duplicate_nodes_count(self) -> int:
        return (
            self.reader.points.shape[0] - np.unique(self.reader.points, axis=0).shape[0]
        )

    def _calc_face_data_tri(self, face: Tuple[int, ...]) -> Tuple:
        # ugly but faster than a list comprehension
        p1, p2, p3 = (
            self.reader.points[face[0]],
            self.reader.points[face[1]],
            self.reader.points[face[2]],
        )

        face_center = _mean((p1, p2, p3))
        face_normal = _cross((p2 - p1), (p3 - p1))
        face_area = _norm(face_normal) / 2.0
        edges_norm = _norm(p1 - p2), _norm(p1 - p3), _norm(p2 - p3)
        aspect_ratio = max(edges_norm) / min(edges_norm)

        self.n_tri += 1

        return face_center, face_area, face_normal, aspect_ratio

    def _calc_face_data_quad(self, face: Tuple[int, ...]) -> Tuple:
        # Divide quad face into two triangles.
        tri1_data = self._calc_face_data_tri((face[0], face[1], face[2]))
        tri2_data = self._calc_face_data_tri((face[0], face[2], face[3]))

        # Calculate quad data from two triangle results
        quad_center = _mean((tri1_data[0], tri2_data[0]))
        quad_area = tri1_data[1] + tri2_data[1]
        quad_normal = _mean((tri1_data[2], tri2_data[2]))

        # Calculate face aspect ratio
        p1, p2, p3, p4 = (
            self.reader.points[face[0]],
            self.reader.points[face[1]],
            self.reader.points[face[2]],
            self.reader.points[face[3]],
        )

        edges_norm = _norm(p1 - p2), _norm(p2 - p3), _norm(p3 - p4), _norm(p4 - p1)
        aspect_ratio = max(edges_norm) / min(edges_norm)

        self.n_quad += 1
        self.n_tri -= 2

        return quad_center, quad_area, quad_normal, aspect_ratio

    def calc_faces_data(self) -> None:
        self.n_quad = 0
        self.n_tri = 0
        self.faces_areas = np.zeros(shape=(self.n_faces,))
        self.faces_centers = np.zeros(shape=(self.n_faces, 3))
        self.faces_normals = np.zeros(shape=(self.n_faces, 3))
        self.aspect_ratio = np.zeros(shape=(self.n_faces,))

        face_size_to_calc_func = {
            3: self._calc_face_data_tri,  # triangle
            6: self._calc_face_data_tri,  # triangle6
            7: self._calc_face_data_tri,  # triangle7
            4: self._calc_face_data_quad,  # quad
            8: self._calc_face_data_quad,  # quad8
            9: self._calc_face_data_quad,  # quad9
        }

        for i, face in enumerate(self.reader.faces):
            (
                self.faces_centers[i, :],
                self.faces_areas[i],
                self.faces_normals[i, :],
                self.aspect_ratio[i],
            ) = face_size_to_calc_func[len(face)](face)

    def _calc_cell_data_tetra(self, cell: Tuple[int, ...]) -> Tuple[np.ndarray, float]:
        points = (
            self.reader.points[cell[0]],
            self.reader.points[cell[1]],
            self.reader.points[cell[2]],
            self.reader.points[cell[3]],
        )

        return _mean(points), _tetra_vol(*points)

    def _calc_cell_data_hex(self, cell: Tuple[int, ...]) -> Tuple[np.ndarray, float]:
        points = (
            self.reader.points[cell[0]],
            self.reader.points[cell[1]],
            self.reader.points[cell[2]],
            self.reader.points[cell[3]],
            self.reader.points[cell[4]],
        )
        x = _norm(points[0] - points[1])
        y = _norm(points[0] - points[3])
        z = _norm(points[0] - points[4])
        volume = x * y * z

        return _mean(points), volume

    def _calc_cell_data_wedge(self, cell: Tuple[int, ...]) -> Tuple[np.ndarray, float]:
        points = tuple(self.reader.points[i] for i in cell)

        # get wedge lower triangle area
        area = self.faces_areas[
            self.reader.face_to_faceid[
                frozenset(
                    (cell[0], cell[1], cell[2])
                )
            ]
        ]

        volume = area * _norm(points[3] - points[0])
        return _mean(points), volume

    def _calc_cell_data_pyramid(self, cell: Tuple[int, ...]) -> Tuple[float, float]:
        apex = self.reader.points[cell[4]]

        # Pyramid base
        base_tri1_data = self._calc_face_data_tri((cell[0], cell[1], cell[2]))
        base_tri2_data = self._calc_face_data_tri((cell[0], cell[2], cell[3]))
        base_center = _mean((base_tri1_data[0], base_tri2_data[0]))
        base_area = base_tri1_data[1] + base_tri2_data[1]

        pyramid_centroid = (0.25 * apex) + _mult(base_center, 0.75)
        pyramid_volume = (1.0 / 3.0) * base_area * _norm(_sub(base_center, apex))

        return pyramid_centroid, pyramid_volume

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


# We implement basic vector operations, because for small size of np.arrays
# numpy equivalent of such functions are substantially slower.
def _cross(left: np.ndarray, right: np.ndarray) -> Tuple[float, float, float]:
    x = (left[1] * right[2]) - (left[2] * right[1])
    y = (left[2] * right[0]) - (left[0] * right[2])
    z = (left[0] * right[1]) - (left[1] * right[0])
    return (x, y, z)


def _mean(points: Tuple[np.ndarray, ...]):
    out = [0.0, 0.0, 0.0]
    for point in points:
        out[0] += point[0]
        out[1] += point[1]
        out[2] += point[2]

    n = len(points)
    out[0] = out[0] / n
    out[1] = out[1] / n
    out[2] = out[2] / n

    return out


def _det(m) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[1][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1])
        + m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1])
    )


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _tetra_vol(a, b, c, d):
    return abs(_det((_sub(a, b), _sub(b, c), _sub(c, d)))) / 6.0


def _norm(vec: Union[np.ndarray, Tuple]) -> float:
    return sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


def _dot(a, b):
    return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])


def _mult(vec, b):
    return (vec[0] * b, vec[1] * b, vec[2] * b)