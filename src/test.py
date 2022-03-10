from neatmesh._reader import MeshReader3D
from neatmesh._quality import QualityInspector3D

if __name__ == "__main__":
    print("Reading mesh...")
    mesh = MeshReader3D("./neatmesh/test_meshes/fine_hex_mesh.med")
    mesh.process_mesh()

    q = QualityInspector3D(mesh)
    print("Getting mesh cell types...")
    q.calc_cell_types_counts()

    print("Calculating face centers, normals and areas...")
    q.calc_faces_data()
    
    print("Calculating cell centers and volumes...")
    q.calc_cells_data()

    print('Mesh bounding box: ', q.mesh_bounding_box())
    print('Count of duplicate nodes: ', q.duplicate_nodes_count())