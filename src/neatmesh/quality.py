from typing import Tuple

import numpy as np

from .meshio_handler import MeshioHandler3D, cell_type_handler_map, MeshIOCellType


class MeshQuality3D:
    def __init__(self, mh: MeshioHandler3D) -> None:
        self.mh = mh

        self.n_points = mh.n_points

        self.n_faces = len(mh.faces)
        self.n_quad_faces = 0
        self.n_tri_faces = 0
        self.faces_areas = np.zeros(shape=(self.n_faces,))
        self.faces_centers = np.zeros(shape=(self.n_faces, 3))
        self.faces_normals = np.zeros(shape=(self.n_faces, 3))

        self.n_cells = mh.n_cells
        self.cells_centers = np.zeros(shape=(self.n_cells, 3))
        self.cells_volumes = np.zeros(shape=(self.n_cells,))

        self.calc_cell_types_counts()

    def calc_cell_types_counts(self) -> None:
        self.hex_count = sum(
            [
                len(self.mh.mesh.get_cells_type(mtype).data)
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
                len(self.mh.mesh.get_cells_type(mtype).data)
                for mtype in (
                    MeshIOCellType.Tetra,
                    MeshIOCellType.Tetra10,
                )
            ]
        )

        self.pyramid_count = sum(
            [
                len(self.mh.mesh.get_cells_type(mtype).data)
                for mtype in (
                    MeshIOCellType.Pyramid,
                    MeshIOCellType.Pyramid13,
                    MeshIOCellType.Pyramid14,
                )
            ]
        )

        self.wedge_count = sum(
            [
                len(self.mh.mesh.get_cells_type(mtype).data)
                for mtype in (
                    MeshIOCellType.Wedge,
                    MeshIOCellType.Wedge12,
                    MeshIOCellType.Wedge15,
                )
            ]
        )


def face_data(face: Tuple[int, ...], mh: MeshioHandler3D) -> Tuple[float, float, float]:
    face_size = len(face)
    vertices = [mh.points[i] for i in face]
    geo_center = np.sum(vertices, axis=0) / face_size

    face_total_area = 0.0
    face_normal_vector = np.zeros((3,))
    face_centroid = np.zeros((3,))

    # form the traingular subfaces.
    # each face is constructed using an edge as the base of
    # the triangle and geometric center as its apex.
    for i in range(face_size):
        # Set the subface points
        p1 = vertices[i]
        p2 = vertices[(i + 1) % face_size]
        p3 = geo_center

        # calculate the subface geometric center
        subface_geo_center = np.sum([p1, p2, p3], axis=0) / 3.0

        # calculate the area and normal vector 'sf' for the subface
        sf = np.cross((p2 - p1), (p3 - p1))
        area = np.linalg.norm(sf) / 2.0

        face_normal_vector += sf
        face_total_area += area
        face_centroid = face_centroid + (area * subface_geo_center)

    face_centroid /= face_total_area
    face_normal_vector /= 2.0

    return (face_centroid, face_total_area, face_normal_vector)


def cell_data(
    cell: Tuple[int, ...], cell_type: MeshIOCellType, mh: MeshioHandler3D
) -> Tuple[float, float, float]:
    geo_centeroid = np.zeros((3,))

    cell_faces = cell_type_handler_map[cell_type](cell)
    for face in cell_faces:
        face_id = mh.face_to_faceid[frozenset(face)]
        geo_centeroid += face.center
    geo_centeroid /= len(self.faces)

    # construct a pyramid with each cell face as the base and geometric center as the apex.
    element_volume = 0.0
    element_centroid = np.zeros((3,))

    for face in self.cell_faces:
        face_centroid, face_area, _ = face.center_and_area()

        # pyramid centroid is located on 0.25 the distance between the base and the apex.
        pyramid_centroid = (0.75 * face_centroid) + (0.25 * geo_centeroid)
        pyramid_volume = (
            (1.0 / 3.0) * face_area * np.linalg.norm(face_centroid - geo_centeroid)
        )

        element_volume += pyramid_volume
        # element_centroid will be divided by total volume after the end of faces loop
        element_centroid += pyramid_volume * pyramid_centroid

    # Finally, calculate the volume weighted element centre.
    element_centroid = element_centroid / element_volume

    self.center, self.volume = element_centroid, element_volume

    return element_centroid, element_volume
