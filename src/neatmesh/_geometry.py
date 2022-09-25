"""geometry related functions for faces and cells"""
# pylint: disable=invalid-name, too-many-locals
from typing import List, Tuple

import numpy as np
from numpy.linalg import det, norm


def tri_data_from_tensor(tri_faces_tensor: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Calculate triangular faces centers, normal, area and aspect ratios.

    Args:
        tri_faces_tensor (np.ndarray): Triangle faces tensor
        shape of `tri_faces_tensor` is (n_faces, 3 (no. points), 3)

    Returns:
        Tuple[np.ndarray, ...]: centers, normals, areas and aspect ratios.
    """
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

    tri_aspect_ratios = np.max(tri_edges_norms, axis=1) / np.min(
        tri_edges_norms, axis=1
    )
    return tri_centers, tri_normals, tri_areas, tri_aspect_ratios


def quad_data_from_tensor(quad_faces_tensor: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Calculate quad faces centers, normal, area and aspect ratios.

    Args:
        quad_faces_tensor (np.ndarray): Quad faces tensor
        shape of `quad_faces_tensor` is (n_faces, 4 (no. points), 3)

    Returns:
        Tuple[np.ndarray, ...]: centers, normals, areas and aspect ratios.
    """
    # Quads geometrics centers
    gc = np.mean(quad_faces_tensor, axis=1)

    # Quad sub-triangles
    edges = ((0, 1), (1, 2), (2, 3), (3, 0))

    sub_triangles_centroids_unswapped = []
    sub_triangles_normals_unswapped = []

    for edge in edges:
        sub_triangles_centroids_unswapped.append(
            np.mean(
                [
                    gc,
                    quad_faces_tensor[:, edge[0], :],
                    quad_faces_tensor[:, edge[1], :],
                ],
                axis=0,
            )
        )

        sub_triangles_normals_unswapped.append(
            np.cross(
                gc - quad_faces_tensor[:, edge[0], :],
                quad_faces_tensor[:, edge[1], :] - quad_faces_tensor[:, edge[0], :],
                axis=1,
            )
        )

    sub_triangles_centroids = np.swapaxes(
        np.asarray(sub_triangles_centroids_unswapped), 0, 1
    )
    sub_triangles_normals = np.swapaxes(
        np.asarray(sub_triangles_normals_unswapped), 0, 1
    )

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
            norm(quad_faces_tensor[:, 0, :] - quad_faces_tensor[:, 1, :], axis=1),
            norm(quad_faces_tensor[:, 1, :] - quad_faces_tensor[:, 2, :], axis=1),
            norm(quad_faces_tensor[:, 2, :] - quad_faces_tensor[:, 3, :], axis=1),
        ]
    )
    quad_aspect_ratios = np.max(quad_edges_norms, axis=0) / np.min(
        quad_edges_norms, axis=0
    )

    return quad_centroids, quad_normals, quad_areas.flatten(), quad_aspect_ratios


def tetra_data_from_tensor(
    tetra_cells_tensor: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate tetrahedron cells centers and volumes

    Args:
        tetra_cells_tensor (np.ndarray): Tetrahedron cells tensor
        shape of `tetra_cells_tensor` is (n_cells, 4 (no. points), 3)

    Returns:
        Tuple[np.ndarray, ...]: centers and volumes
    """
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


def pyramid_data_from_tensor(pyr_cells_tensor: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Calculate pyramid cells centers and volumes

    Args:
        pyr_cells_tensor (np.ndarray): Pyramid cells tensor
        shape of `pyr_cells_tensor` is (n_cells, 4 (no. points), 3)

    Returns:
        Tuple[np.ndarray, ...]: centers and volumes
    """
    # Pyramid base area and centroid
    quad_base_tensor = pyr_cells_tensor[:, 0:-1, :]
    quad_centroids, quad_normals, quad_areas, _ = quad_data_from_tensor(
        quad_base_tensor
    )

    pyramids_apex = pyr_cells_tensor[:, -1, :]
    # pyramids_heights = norm(pyramids_apex - quad_centroids, axis=1)[:, np.newaxis]
    normals_unit_vecs = quad_normals / norm(quad_normals, axis=1)[:, np.newaxis]

    pyramids_heights = np.sum(
        (pyramids_apex - quad_centroids) * normals_unit_vecs, axis=1
    )

    pyramids_vol = (1.0 / 3.0) * quad_areas.flatten() * np.abs(pyramids_heights)
    pyramids_centroids = (0.75 * quad_centroids) + (0.25 * pyramids_apex)

    return pyramids_centroids, pyramids_vol


def wedge_data_from_tensor(wedge_cells_tensor: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Calculate wedge cells centers and volumes

    Args:
        wedge_data_from_tensor (np.ndarray): Wedge cells tensor
        shape of `wedge_data_from_tensor` is (n_cells, 6 (no. points), 3)

    Returns:
        Tuple[np.ndarray, ...]: centers and volumes
    """
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
    wedges_vol += np.sum([pyramid1_vol, pyramid2_vol, pyramid3_vol], axis=0)[
        :, np.newaxis
    ]

    # Wedge centroid
    wedges_center = upper_tetra_centers * upper_tetra_vol[:, np.newaxis]
    wedges_center += lower_tetra_centers * lower_tetra_vol[:, np.newaxis]
    wedges_center += pyramid1_centers * pyramid1_vol[:, np.newaxis]
    wedges_center += pyramid2_centers * pyramid2_vol[:, np.newaxis]
    wedges_center += pyramid3_centers * pyramid3_vol[:, np.newaxis]
    wedges_center /= wedges_vol

    return wedges_center, wedges_vol.flatten()


def hex_data_from_tensor(hex_cells_tensor: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Calculate hexahedron cells centers and volumes

    Args:
        hex_cells_tensor (np.ndarray): Hexahedron cells tensor
        shape of `hex_cells_tensor` is (n_cells, 8 (no. points), 3)

    Returns:
        Tuple[np.ndarray, ...]: centers and volumes
    """
    x = norm(hex_cells_tensor[:, 0, :] - hex_cells_tensor[:, 1, :], axis=1)
    y = norm(hex_cells_tensor[:, 0, :] - hex_cells_tensor[:, 3, :], axis=1)
    z = norm(hex_cells_tensor[:, 0, :] - hex_cells_tensor[:, 4, :], axis=1)

    hex_centers = np.mean(hex_cells_tensor, axis=1)
    hex_vols = x * y * z

    return hex_centers, hex_vols


def dot_normalize(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Element wise dot product with normalization"""
    return np.sum(x * y, axis=1) / ((norm(x, axis=1) * norm(y, axis=1)))


def dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Element wise dot product"""
    return np.sum(x * y, axis=1)


def hex_cell_faces(cell: List) -> Tuple[Tuple[int, ...], ...]:
    """Returns coordinates of 6 faces of a hexahedron cell, using meshio nodes ordering
    Args:
        cell (List): list of points defining the cell
    Returns:
        List[List]: list of list of faces points labels
    """
    return (
        (cell[1], cell[2], cell[6], cell[5]),
        (cell[0], cell[4], cell[7], cell[3]),
        (cell[3], cell[7], cell[6], cell[2]),
        (cell[0], cell[1], cell[5], cell[4]),
        (cell[4], cell[5], cell[6], cell[7]),
        (cell[0], cell[3], cell[2], cell[1]),
    )


def wedge_cell_faces(cell: List) -> Tuple[Tuple[int, ...], ...]:
    """Returns coordinates of 5 faces of a wedge cell,
    using meshio nodes ordering for wedge
    Args:
        cell (List): list of points defining the cell
    Returns:
        List[List]: list of list of faces points labels
    """
    return (
        (cell[0], cell[2], cell[1], -1),
        (cell[3], cell[4], cell[5], -1),
        (cell[3], cell[0], cell[1], cell[4]),
        (cell[0], cell[3], cell[5], cell[2]),
        (cell[1], cell[2], cell[5], cell[4]),
    )


def tetra_cell_faces(cell: List) -> Tuple[Tuple[int, ...], ...]:
    """Returns coordinates of 4 faces of a tetrahedral cell,
    using meshio nodes ordering for tetra
    Args:
        cell (List): list of points defining the cell
    Returns:
        List[List]: list of list of faces points labels
    """
    return (
        (cell[0], cell[2], cell[1], -1),
        (cell[1], cell[2], cell[3], -1),
        (cell[0], cell[1], cell[3], -1),
        (cell[0], cell[3], cell[2], -1),
    )


def pyramid_cell_faces(cell: List) -> Tuple[Tuple[int, ...], ...]:
    """Returns coordinates of 4 faces of a tetrahedral cell,
    using meshio nodes ordering for pyramid
    Args:
        cell (List): list of points defining the cell
    Returns:
        List[List]: list of list of faces points labels
    """
    return (
        (cell[2], cell[1], cell[0], cell[3]),
        (cell[2], cell[3], cell[4], -1),
        (cell[1], cell[4], cell[0], -1),
        (cell[3], cell[0], cell[4], -1),
    )


def quad_face_edges(face: List) -> Tuple[Tuple[int, ...], ...]:
    """Get quadilateral face edges labels"""
    return (
        (face[0], face[1]),
        (face[1], face[2]),
        (face[2], face[3]),
        (face[3], face[0]),
    )


def tri_face_edges(face: List) -> Tuple[Tuple[int, ...], ...]:
    """Get triangle face edges labels"""
    return (
        (face[0], face[1]),
        (face[1], face[2]),
        (face[2], face[0]),
    )
