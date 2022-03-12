from neatmesh._reader import MeshReader3D
from neatmesh._quality import QualityInspector3D

import time
import numpy as np

from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich import box


def report_main_mesh_stats(console: Console, reader: MeshReader3D):
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


def report_face_types(console: Console, quad_count, tri_count):
    face_count_table = Table(title="Faces Count", box=box.SIMPLE)
    face_count_table.add_column("Type", justify="left", style="magenta")
    face_count_table.add_column("Count", justify="left")

    if quad_count > 0:
        face_count_table.add_row("Quadilateral", str(quad_count))
    elif tri_count > 0:
        face_count_table.add_row("Triangle", str(tri_count))

    console.print(face_count_table)


def report_cell_types(console: Console):
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


if __name__ == "__main__":
    print(
        f"""
                  _                      _     
 _ __   ___  __ _| |_ _ __ ___   ___ ___| |__  
| '_ \ / _ \/ _` | __| '_ ` _ \ / _ / __| '_ \ 
| | | |  __| (_| | |_| | | | | |  __\__ | | | |
|_| |_|\___|\__,_|\__|_| |_| |_|\___|___|_| |_|
───────────────────────────────────────────────
version: 0.1b | license: MIT
"""
    )
    console = Console()

    print("Reading mesh...")
    mesh = MeshReader3D("./neatmesh/test_meshes/small_tetra.med")
    mesh.process_mesh()
    report_main_mesh_stats(console, mesh)

    print("Collecting cell types..")
    q = QualityInspector3D(mesh)
    q.calc_cell_types_counts()

    report_cell_types(console)

    print("Calculating face centers, normals, areas and aspect ratio...")
    q.calc_faces_data()
    report_face_types(console, q.n_quad, q.n_tri)

    print("Calculating cell centers and volumes...")
    q.calc_cells_data()
    rprint(
        f"Face aspect ratio max. = {np.max(q.aspect_ratio):.2f}"
        f", min. = {np.min(q.aspect_ratio):.2f}, "
        f"mean = {np.mean(q.aspect_ratio):.2f}, "
        f"std = {np.std(q.aspect_ratio):.2f}"
    )

    print("Checking non-orthogonality...")
    q.calc_faces_nonortho()
    rprint(
        f"non-orthogonality max. = {np.nanmax(q.non_ortho):.3f}"
        f", min. = {np.nanmin(q.non_ortho):.3f}, "
        f"mean = {np.nanmean(q.non_ortho):.3f}, "
        f"std = {np.nanstd(q.non_ortho):.3f}"
    )

    rprint("Mesh bounding box: ", q.mesh_bounding_box())
    rprint("Count of duplicate nodes: ", q.duplicate_nodes_count())

# cell volume: min, max, std, mean
# face area: min, max, std, mean
