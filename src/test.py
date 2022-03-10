from neatmesh.meshio_handler import MeshioHandler3D
from neatmesh.quality import QualityInspector3D

if __name__ == "__main__":
    print("Reading mesh...")
    mesh = MeshioHandler3D("./neatmesh/test_meshes/fine_cylinder.med")
    mesh.process_mesh()

    q = QualityInspector3D(mesh)
    print("Getting mesh cell types...")
    q.calc_cell_types_counts()

    print("Calculating face centers, normals and areas...")
    q.calc_faces_data()
    
    print("Calculating cell centers and volumes...")
    q.calc_cells_data()
