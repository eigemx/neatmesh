from typing import Tuple, List
from math import sqrt

import numpy as np

from ._reader import MeshReader3D, MeshIOCellType


def _cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x = (left[1] * right[2]) - (left[2] * right[1])
    y = (left[2] * right[0]) - (left[0] * right[2])
    z = (left[0] * right[1]) - (left[1] * right[0])
    return [x, y, z]


def _sum(points: Tuple[np.ndarray, ...]) -> np.ndarray:
    out = [0, 0, 0]
    for point in points:
        out += point
    return out


def _mean(points: Tuple[np.ndarray, ...]) -> np.ndarray:
    out = [0, 0, 0]
    for point in points:
        out[0] += point[0]
        out[1] += point[1]
        out[2] += point[2]
    n = len(points)
    out[0] /= n
    out[1] /= n
    out[2] /= n
    return out


def _det(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[1][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1])
        + m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1])
    )


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def tetra_vol(a, b, c, d):
    return abs(_det((_sub(a, b), _sub(b, c),_sub(c, d)))) / 6.0


def _norm(vec: np.ndarray) -> float:
    return sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)


class QualityInspector3D:
    def __init__(self, mr: MeshReader3D) -> None:
        self.reader = mr

        self.n_points = mr.n_points

        self.n_faces = len(mr.faces)
        self.n_quad_faces = 0
        self.n_tri_faces = 0
        self.faces_areas = np.zeros(shape=(self.n_faces,))
        self.faces_centers = np.zeros(shape=(self.n_faces, 3))
        self.faces_normals = np.zeros(shape=(self.n_faces, 3))

        self.n_cells = mr.n_cells
        self.cells_centers = np.zeros(shape=(self.n_cells, 3))
        self.cells_volumes = np.zeros(shape=(self.n_cells,))

    def calc_cell_types_counts(self) -> None:
        self.hex_count = sum(
            [
                len(self.reader.mesh.get_cells_type(mtype).data)
                for mtype in (
                    MeshIOCellType.Hex,
                    MeshIOCellType.Hex20,
                    MeshIOCellType.Hex24,
                    MeshIOCellType.Hex27,
                )
            ]
        )

        self.tetra_count = sum(
            [
                len(self.reader.mesh.get_cells_type(mtype).data)
                for mtype in (
                    MeshIOCellType.Tetra,
                    MeshIOCellType.Tetra10,
                )
            ]
        )

        self.pyramid_count = sum(
            [
                len(self.reader.mesh.get_cells_type(mtype).data)
                for mtype in (
                    MeshIOCellType.Pyramid,
                    MeshIOCellType.Pyramid13,
                    MeshIOCellType.Pyramid14,
                )
            ]
        )

        self.wedge_count = sum(
            [
                len(self.reader.mesh.get_cells_type(mtype).data)
                for mtype in (
                    MeshIOCellType.Wedge,
                    # MeshIOCellType.Wedge12,
                    MeshIOCellType.Wedge15,
                )
            ]
        )

    def mesh_bounding_box(self) -> np.ndarray:
        x_min, y_min, z_min = np.min(self.reader.points, axis=0)
        x_max, y_max, z_max = np.max(self.reader.points, axis=0)
        return ((x_min, y_min, z_min), (x_max, y_max, z_max))

    def duplicate_nodes_count(self) -> int:
        return self.reader.points.shape[0] - np.unique(self.reader.points, axis=0).shape[0]

    def calc_face_data_tri(self, face: Tuple[int, ...]) -> Tuple[float, float, float]:
        p1, p2, p3 = [self.reader.points[i] for i in face]
        face_center = _mean([p1, p2, p3])
        face_normal = _cross((p2 - p1), (p3 - p1))
        face_area = _norm(face_normal) / 2.0
        return face_center, face_area, face_normal

    def calc_face_data_quad(self, face: Tuple[int, ...]) -> Tuple[float, float, float]:
        tri1_data = self.calc_face_data_tri((face[0], face[1], face[2]))
        tri2_data = self.calc_face_data_tri((face[0], face[2], face[3]))
        quad_center = _mean([tri1_data[0], tri2_data[0]])
        quad_area = tri1_data[1] + tri2_data[1]
        quad_normal = _mean([tri1_data[2], tri2_data[2]])
        return quad_center, quad_area, quad_normal

    def calc_faces_data(self) -> None:
        face_data_handler = {
            3: self.calc_face_data_tri,
            4: self.calc_face_data_quad,
        }

        for i, face in enumerate(self.reader.faces):
            (
                self.faces_centers[i, :],
                self.faces_areas[i],
                self.faces_normals[i, :],
            ) = face_data_handler[len(face)](face)

    def calc_cell_data_tetra(self, cell: Tuple[int, ...]) -> Tuple[float, float]:
        points = [
            self.reader.points[cell[0]],
            self.reader.points[cell[1]],
            self.reader.points[cell[2]],
            self.reader.points[cell[3]],
        ]
        return _mean(points), tetra_vol(*points)

    def calc_cell_data_hex(self, cell: Tuple[int, ...]) -> Tuple[float, float]:
        points = [self.reader.points[i] for i in cell]
        x = _norm(_sub(points[0], points[1]))
        y = _norm(_sub(points[0], points[3]))
        z = _norm(_sub(points[0], points[4]))
        volume = x * y * z

        return _mean(points), volume

    def calc_cell_data_wedge(self, cell: Tuple[int, ...]) -> Tuple[float, float]:
        points = [self.reader.points[i] for i in cell]
        face_id = self.reader.face_to_faceid[frozenset([cell[0], cell[1], cell[2]])]
        area = self.faces_areas[face_id]
        volume = area * _norm(_sub(points[3], points[0]))
        return _mean(points), volume

    def calc_cell_data_pyramid(self, cell: Tuple[int, ...]) -> Tuple[float, float]:
        points = [self.reader.points[i] for i in cell]
        raise NotImplemented("Not Implemented")

    def calc_cells_data(self) -> None:
        cell_type_data_handler = {
            MeshIOCellType.Hex: self.calc_cell_data_hex,
            MeshIOCellType.Hex20: self.calc_cell_data_hex,
            MeshIOCellType.Hex24: self.calc_cell_data_hex,
            MeshIOCellType.Hex27: self.calc_cell_data_hex,
            MeshIOCellType.Tetra: self.calc_cell_data_tetra,
            MeshIOCellType.Wedge: self.calc_cell_data_wedge,
            MeshIOCellType.Wedge12: self.calc_cell_data_wedge,
            MeshIOCellType.Wedge15: self.calc_cell_data_wedge,
        }
        for i, (cell, cell_type) in enumerate(self.reader.cells()):
            self.cells_centers[i, :], self.cells_volumes[i] = cell_type_data_handler[
                cell_type
            ](cell)
