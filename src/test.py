from neatmesh._reader import MeshReader3D
from neatmesh._quality import QualityInspector3D
from rich.console import Console


if __name__ == "__main__":
    console = Console()
    print(f"""
                  _                      _     
 _ __   ___  __ _| |_ _ __ ___   ___ ___| |__  
| '_ \ / _ \/ _` | __| '_ ` _ \ / _ / __| '_ \ 
| | | |  __| (_| | |_| | | | | |  __\__ | | | |
|_| |_|\___|\__,_|\__|_| |_| |_|\___|___|_| |_|

version: 0.1b | license: MIT""")
    console.rule()
    print("Reading mesh...")
    mesh = MeshReader3D("./neatmesh/test_meshes/fine_hex_mesh.med")
    mesh.process_mesh()

    q = QualityInspector3D(mesh)
    print("Collecting cell types...")
    q.calc_cell_types_counts()

    print("Calculating face centers, normals and areas...")
    q.calc_faces_data()
    
    print("Calculating cell centers and volumes...")
    q.calc_cells_data()

    print('Mesh bounding box: ', q.mesh_bounding_box())
    print('Count of duplicate nodes: ', q.duplicate_nodes_count())
    console.rule()