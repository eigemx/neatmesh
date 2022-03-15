import meshio
import meshplex
import numpy as np
from neatmesh._reader import MeshReader3D

if __name__ == '__main__':
    m = meshio.read("./neatmesh/test_meshes/one_tetra.stl")
    print(m.points)
    mesh = meshio.Mesh(
        points=m.points,
        cells=[meshio.CellBlock("tetra", [3, 0, 1, 2])]
    )
    mesh = meshplex.Mesh(
        m.points,
        [[3, 0, 1, 2]]
    )
    print(mesh.cell_volumes, mesh.cell_centroids)