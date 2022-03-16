from typing import Tuple

from neatmesh._reader import MeshReader3D
from neatmesh._quality import QualityInspector3D

import numpy as np

from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich import box


def report_elements_count(console: Console, reader: MeshReader3D) -> None:
    cell_count = reader.n_cells
    face_count = len(reader.faces_set)
    point_count = reader.n_points

    stats_table = Table(title="", box=box.SIMPLE)
    stats_table.add_column("Element", justify="left", style="magenta")
    stats_table.add_column("Count", justify="left")

    stats_table.add_row("Points", str(point_count))
    stats_table.add_row("Faces", str(face_count))
    stats_table.add_row("Cells", str(cell_count))
    console.print(stats_table)


def report_face_types(console: Console, quad_count, tri_count) -> None:
    face_count_table = Table(title="Faces Count", box=box.SIMPLE)
    face_count_table.add_column("Type", justify="left", style="magenta")
    face_count_table.add_column("Count", justify="left")

    if quad_count > 0:
        face_count_table.add_row("Quadilateral", str(quad_count))
    elif tri_count > 0:
        face_count_table.add_row("Triangle", str(tri_count))

    console.print(face_count_table)


def report_cell_types(console: Console, q) -> None:
    cell_count_table = Table(title="", box=box.SIMPLE)
    cell_count_table.add_column("Type", justify="left", style="magenta")
    cell_count_table.add_column("Count", justify="left")

    for ctype, count in (
        ("Hexahedron", q.hex_count),
        ("Tetrahedron", q.tetra_count),
        ("Wedge", q.wedge_count),
        ("Pyramid", q.pyramid_count),
    ):
        if count == 0:
            continue
        cell_count_table.add_row(ctype, str(count))

    console.print(cell_count_table)


def stats_from_array(array: np.ndarray) -> Tuple[float, ...]:
    arr_max = np.nanmax(array)
    arr_min = np.nanmin(array)
    arr_mean = np.nanmean(array)
    arr_std = np.nanstd(array)
    
    return arr_max, arr_min, arr_mean, arr_std


def report_mesh_stats(console: Console, q: QualityInspector3D) -> None:
    stats = {
        "Face Area": q.face_areas,
        "Face Aspect Ratio": q.face_aspect_ratios,
        "Cell Volume": q.cells_volumes,
        #"Non-Orthogonality": q.non_ortho 
    }
    
    stats_table = Table(title="Mesh Statistics", box=box.SIMPLE)
    stats_table.add_column("", justify="left", style="magenta")
    stats_table.add_column("Max.", justify="right")
    stats_table.add_column("Min.", justify="right")
    stats_table.add_column("Mean.", justify="right")
    stats_table.add_column("Std.", justify="right")
    
    for stat, array in stats.items():
        _max, _min, _mean, _std = stats_from_array(array)
        stats_table.add_row(stat, f"{_max:.6f}", f"{_min:.6f}", f"{_mean:.6f}", f"{_std:.6f}")
    
    console.print(stats_table)
    
    

if __name__ == "__main__":
    print(
        f"""
                     __                      __  
   ____  ___  ____ _/ /_____ ___  ___  _____/ /_    |   
  / __ \/ _ \/ __ `/ __/ __ `__ \/ _ \/ ___/ __ \   |   Version: 0.1b
 / / / /  __/ /_/ / /_/ / / / / /  __(__  ) / / /   |   License: MIT
/_/ /_/\___/\__,_/\__/_/ /_/ /_/\___/____/_/ /_/    |   
                                                 
"""
    )
    console = Console()

    with console.status("Reading mesh..."):
        mesh = MeshReader3D("./neatmesh/test_meshes/fine_cylinder.med")
    report_elements_count(console, mesh)

    with console.status("Collecting cell types.."):
        q = QualityInspector3D(mesh)
        q.count_cell_types()

    report_cell_types(console, q)

    with console.status("Analyzing faces..."):
        q.analyze_faces()
    report_face_types(console, q.n_quad, q.n_tri)

    with console.status("Analyzing cells..."):
        q.analyze_cells()

    '''print("Checking non-orthogonality...\n")
    q.calc_faces_nonortho()'''

    report_mesh_stats(console, q)

    rprint("Count of duplicate nodes = ", q.duplicate_nodes_count())
    rprint("Mesh bounding box: ")
    for point in q.bounding_box():
        rprint(point)

'''reader = MeshReader3D("./neatmesh/test_meshes/fine_cylinder.med")
q = QualityInspector3D(reader)
print(q.hex_count, q.tetra_count, q.wedge_count, q.pyramid_count)
print('Aspect ratio: ', 
      np.max(q.face_aspect_ratios), np.mean(q.face_aspect_ratios), np.min(q.face_aspect_ratios))
print('Area: ', 
      np.max(q.face_areas), np.mean(q.face_areas), np.min(q.face_areas))

'''