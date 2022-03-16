from typing import Tuple
import numpy as np
from numpy.linalg import norm, det


def bounding_box(points: np.ndarray) -> Tuple:
    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    return (
        (x_min, y_min, z_min),
        (x_max, y_max, z_max),
    )


def duplicate_nodes_count(points: np.ndarray) -> int:
    return points.shape[0] - np.unique(points, axis=0).shape[0]


def tri_data_from_tensor(tri_faces_tensor: np.ndarray):
    tri_centers = np.mean(tri_faces_tensor, axis=1)
    tri_normals = np.cross(
        tri_faces_tensor[:, 1, :] - tri_faces_tensor[:, 0, :],
        tri_faces_tensor[:, 2, :] - tri_faces_tensor[:, 0, :],
    )
    tri_areas = norm(tri_normals, axis=1) / 2.0
    tri_edges_norms = np.array(
        [
            norm(tri_faces_tensor[:, 0, :] - tri_faces_tensor[:, 1, :], axis=1),
            norm(tri_faces_tensor[:, 0, :] - tri_faces_tensor[:, 2, :], axis=1),
            norm(tri_faces_tensor[:, 1, :] - tri_faces_tensor[:, 2, :], axis=1),
        ]
    ).swapaxes(0, 1)

    tri_aspect_ratios = np.max(tri_edges_norms, axis=0) / np.min(
        tri_edges_norms, axis=0
    )

    return tri_centers, tri_normals, tri_areas, tri_aspect_ratios


def quad_data_from_tensor(faces_tensor: np.ndarray):
    # Quads geometrics centers
    gc = np.mean(faces_tensor, axis=1)

    # Quad sub-triangles
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))

    sub_triangles_centroids = [None, None, None, None]
    sub_triangles_normals = [None, None, None, None]

    for i, edge in enumerate(edges):
        sub_triangles_centroids[i] = np.mean(
            [gc, faces_tensor[:, edge[0], :], faces_tensor[:, edge[1], :]], axis=0
        )

        sub_triangles_normals[i] = np.cross(
            gc - faces_tensor[:, edge[0], :],
            faces_tensor[:, edge[1], :] - faces_tensor[:, edge[0], :],
            axis=1,
        )

    sub_triangles_centroids = np.swapaxes(sub_triangles_centroids, 0, 1)
    sub_triangles_normals = np.swapaxes(sub_triangles_normals, 0, 1)

    sub_triangles_areas = norm(sub_triangles_normals, axis=2) / 2.0
    area_weighted_centroids = (
        sub_triangles_areas[:, :, np.newaxis] * sub_triangles_centroids
    )
    quad_centroids = np.sum(area_weighted_centroids, axis=1)
    quad_areas = np.sum(sub_triangles_areas, axis=1)[:, np.newaxis]
    quad_centroids /= quad_areas

    quad_normals = np.sum(sub_triangles_normals, axis=1)

    quad_edges_norms = np.array(
        [
            norm(faces_tensor[:, 0, :] - faces_tensor[:, 1, :], axis=1),
            norm(faces_tensor[:, 1, :] - faces_tensor[:, 2, :], axis=1),
            norm(faces_tensor[:, 2, :] - faces_tensor[:, 3, :], axis=1),
        ]
    )
    quad_aspect_ratios = np.max(quad_edges_norms, axis=0) / np.min(
        quad_edges_norms, axis=0
    )

    return quad_centroids, quad_normals, quad_areas, quad_aspect_ratios


def tetra_data_from_tensor(tetra_cells_tensor: np.ndarray):
    # TODO: Shoudln't this be axis=1?
    tetra_centers = np.mean(tetra_cells_tensor, axis=1)
    tetra_vols = (
        np.abs(
            det(
                np.array(
                    (
                        tetra_cells_tensor[:, 0, :] - tetra_cells_tensor[:, 1, :],
                        tetra_cells_tensor[:, 1, :] - tetra_cells_tensor[:, 2, :],
                        tetra_cells_tensor[:, 2, :] - tetra_cells_tensor[:, 3, :],
                    )
                ).swapaxes(0, 1)
            )
        )
        / 6.0
    )
    return tetra_centers, tetra_vols


def pyramid_data_from_tensor(pyr_cells_tensor: np.ndarray):
    # Pyramid base area and centroid
    quad_base_tensor = pyr_cells_tensor[:, 0:-1, :]
    quad_centroids, quad_normals, quad_areas, _ = quad_data_from_tensor(quad_base_tensor)

    pyramids_apex = pyr_cells_tensor[:, -1, :]
    #pyramids_heights = norm(pyramids_apex - quad_centroids, axis=1)[:, np.newaxis]
    normals_unit_vecs = quad_normals / norm(quad_normals, axis=1)[:, np.newaxis]

    pyramids_heights = np.sum(
        (pyramids_apex - quad_centroids) * normals_unit_vecs,
        axis=1
    )
        
    pyramids_vol = (1.0 / 3.0) * quad_areas.flatten() * pyramids_heights
    pyramids_centroids = (0.75 * quad_centroids) + (0.25 * pyramids_apex)

    return pyramids_centroids, pyramids_vol


def wedge_data_from_tensor(wedge_cells_tensor: np.ndarray):
    gc = np.mean(wedge_cells_tensor, axis=1)[:, np.newaxis, :]
    n_wedge = wedge_cells_tensor.shape[0]
    wedges_vol = np.zeros(shape=(n_wedge, 1))

    # Divide wedge into 5 surfaces (two triangles and three quads)
    upper_tri_tensor = wedge_cells_tensor[:, [3, 4, 5], :]
    lower_tri_tensor = wedge_cells_tensor[:, [0, 1, 2], :]
    quad1_tensor = wedge_cells_tensor[:, [0, 1, 4, 3], :]
    quad2_tensor = wedge_cells_tensor[:, [1, 2, 5, 4], :]
    quad3_tensor = wedge_cells_tensor[:, [0, 3, 5, 2], :]

    # Upper & lower tetrahedrons
    upper_tetra_tensor = np.concatenate([upper_tri_tensor, gc], axis=1)
    lower_tetra_tensor = np.concatenate([lower_tri_tensor, gc], axis=1)
    tetras_tensor = np.concatenate([upper_tetra_tensor, lower_tetra_tensor], axis=0)

    tetras_data = tetra_data_from_tensor(tetras_tensor)

    upper_tetra_vol, lower_tetra_vol = (
        tetras_data[1][0:n_wedge],
        tetras_data[1][n_wedge:],
    )
    (upper_tetra_centers, lower_tetra_centers) = (
        tetras_data[0][0:n_wedge],
        tetras_data[0][n_wedge:],
    )

    wedges_vol = upper_tetra_vol[:, np.newaxis] + lower_tetra_vol[:, np.newaxis]

    # Three middle pyramids
    pyramid1_tensor = np.concatenate([quad1_tensor, gc], axis=1)
    pyramid2_tensor = np.concatenate([quad2_tensor, gc], axis=1)
    pyramid3_tensor = np.concatenate([quad3_tensor, gc], axis=1)
    pyramids_tensor = np.concatenate(
        [pyramid1_tensor, pyramid2_tensor, pyramid3_tensor], axis=0
    )
    

    pyramids_data = pyramid_data_from_tensor(pyramids_tensor)
    pyramid1_vol, pyramid2_vol, pyramid3_vol = (
        pyramids_data[1][0:n_wedge],
        pyramids_data[1][n_wedge : 2 * n_wedge],
        pyramids_data[1][2 * n_wedge :],
    )
    pyramid1_centers, pyramid2_centers, pyramid3_centers = (
        pyramids_data[0][0:n_wedge],
        pyramids_data[0][n_wedge : 2 * n_wedge],
        pyramids_data[0][2 * n_wedge :],
    )

    # Wedge volume
    wedges_vol += np.sum([pyramid1_vol, pyramid2_vol, pyramid3_vol], axis=0)[:, np.newaxis]
    
    # Wedge centroid
    wedges_center = upper_tetra_centers * upper_tetra_vol[:, np.newaxis]
    wedges_center += lower_tetra_centers * lower_tetra_vol[:, np.newaxis]
    wedges_center += pyramid1_centers * pyramid1_vol[:, np.newaxis]
    wedges_center += pyramid2_centers * pyramid2_vol[:, np.newaxis]
    wedges_center += pyramid3_centers * pyramid3_vol[:, np.newaxis]
    wedges_center /= wedges_vol
    
    return wedges_center, wedges_vol.flatten()

def hex_data_from_tensor(hex_cells_tensor: np.ndarray):
    # TODO: Check if non-regular hex might be allowed by meshio
    x = norm(hex_cells_tensor[:, 0, :] - hex_cells_tensor[:, 1, :], axis=1)
    y = norm(hex_cells_tensor[:, 0, :] - hex_cells_tensor[:, 3, :], axis=1)
    z = norm(hex_cells_tensor[:, 0, :] - hex_cells_tensor[:, 4, :], axis=1)

    hex_centers = np.mean(hex_cells_tensor, axis=1)
    hex_vols = x * y * z

    return hex_centers, hex_vols
