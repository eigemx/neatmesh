import meshio
import numpy as np
from neatmesh._reader import MeshReader3D


def calc_tri_cells_data():
    mesh = MeshReader3D("./neatmesh/test_meshes/fine_cylinder.med")
    mesh.process_mesh()
    
    points = mesh.points
    tri_faces = mesh.faces
    
    n_faces = len(tri_faces)
    
    tri_faces_tensor = np.zeros(shape=(n_faces, 3, 3))

    for i in range(n_faces):
        tri_faces_tensor[i] = [
            points[tri_faces[i][0]],
            points[tri_faces[i][1]],
            points[tri_faces[i][2]],
        ]

    
    centers = np.mean(tri_faces_tensor, axis=2)
    normals = np.cross(
        tri_faces_tensor[:, 1, :] - tri_faces_tensor[:, 0, :],
        tri_faces_tensor[:, 2, :] - tri_faces_tensor[:, 0, :]
    )
    areas = np.linalg.norm(normals, axis=1)
    edges_norms = np.array([
        np.linalg.norm(tri_faces_tensor[:, 0, :] - tri_faces_tensor[:, 1, :], axis=1),
        np.linalg.norm(tri_faces_tensor[:, 0, :] - tri_faces_tensor[:, 2, :], axis=1),
        np.linalg.norm(tri_faces_tensor[:, 1, :] - tri_faces_tensor[:, 2, :], axis=1)
    ])
    aspect_ratios = np.max(edges_norms, axis=0) / np.min(edges_norms, axis=0)
    print(
        np.max(aspect_ratios),
        np.min(aspect_ratios),
        np.mean(aspect_ratios),
        np.std(aspect_ratios),
    )

calc_tri_cells_data()