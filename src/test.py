from neatmesh._reader import MeshReader3D
from neatmesh._quality import QualityInspector3D

import numpy as np

from rich import print as rprint


if __name__ == "__main__":
    print(f"""
                  _                      _     
 _ __   ___  __ _| |_ _ __ ___   ___ ___| |__  
| '_ \ / _ \/ _` | __| '_ ` _ \ / _ / __| '_ \ 
| | | |  __| (_| | |_| | | | | |  __\__ | | | |
|_| |_|\___|\__,_|\__|_| |_| |_|\___|___|_| |_|

version: 0.1b | license: MIT""")
    rprint("Reading mesh...")
    mesh = MeshReader3D("./neatmesh/test_meshes/fine_hex_mesh.med")
    mesh.process_mesh()

    q = QualityInspector3D(mesh)
    rprint("Collecting cell types...")
    q.calc_cell_types_counts()

    rprint("Calculating face centers, normals and areas...")
    q.calc_faces_data()
    
    rprint("Calculating cell centers and volumes...")
    q.calc_cells_data()
    
    print("Checking Non-Orthogonality...")
    q.calc_nonortho()
    rprint(
        f"non-orthogonality max. = {max(q.non_ortho):3f}"
        f", min. = {min(q.non_ortho):3f}," 
        f"mean = {np.mean(np.asarray(q.non_ortho)):3f}," 
        f"std = {np.std(np.asarray(q.non_ortho)):3f}"
    )
    

    rprint('Mesh bounding box: ', q.mesh_bounding_box())
    rprint('Count of duplicate nodes: ', q.duplicate_nodes_count())
    
