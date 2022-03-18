import os
import sys
from typing import Tuple

import humanize
import numpy as np
from rich import box
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from neatmesh.quality import QualityInspector3D
from neatmesh.reader import MeshReader3D


def report_elements_count(
    console: Console, reader: MeshReader3D, q: QualityInspector3D
) -> None:
    cell_count = reader.n_cells
    face_count = len(reader.faces_set)
    point_count = reader.n_points

    rprint()
    tree = Tree("Mesh elements")

    tree.add(f"Points count = {point_count} point", highlight=True)
    faces_branch = tree.add(
        f"Faces count  = {face_count} face, with {q.n_boundary_faces} boundary faces"
    )

    if q.n_quad > 0:
        quad_pct = (q.n_quad / q.n_faces) * 100
        faces_branch.add(f"Quadilaterals: {q.n_quad} ({quad_pct:.1f}%)")
    if q.n_tri > 0:
        tri_pct = (q.n_tri / q.n_faces) * 100
        faces_branch.add(f"Triangles: {q.n_tri} ({tri_pct:.1f}%)")

    cells_branch = tree.add(f"Cells count = {cell_count} cell")
    for ctype, count in (
        ("Hexahedron", q.hex_count),
        ("Tetrahedron", q.tetra_count),
        ("Wedge", q.wedge_count),
        ("Pyramid", q.pyramid_count),
    ):
        if count == 0:
            continue

        pct = (count / q.n_cells) * 100
        cells_branch.add(f"{ctype}s: {count} ({pct:.1f}%)")

    rprint(tree)


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
        "Non-Orthogonality": q.non_ortho,
    }

    stats_table = Table(title="Mesh Statistics", box=box.SIMPLE)
    stats_table.add_column("", justify="left", style="magenta")
    stats_table.add_column("Max.", justify="right")
    stats_table.add_column("Min.", justify="right")
    stats_table.add_column("Mean.", justify="right")
    stats_table.add_column("Std.", justify="right")

    for stat, array in stats.items():
        _max, _min, _mean, _std = stats_from_array(array)
        stats_table.add_row(
            stat, f"{_max:.6f}", f"{_min:.6f}", f"{_mean:.6f}", f"{_std:.6f}"
        )

    console.print(stats_table)


def report_file_stats(filename: str):
    fsize = humanize.naturalsize(os.path.getsize(filename))
    rprint(f"Filename: {filename} ({fsize})")


def header_str(version: str):
    return f"""
                     __                      __  
   ____  ___  ____ _/ /_____ ___  ___  _____/ /_     
  / __ \/ _ \/ __ `/ __/ __ `__ \/ _ \/ ___/ __ \   Version: {version}
 / / / /  __/ /_/ / /_/ / / / / /  __(__  ) / / /   License: MIT
/_/ /_/\___/\__,_/\__/_/ /_/ /_/\___/____/_/ /_/                                                  
"""


def main():

    console = Console()
    filename = sys.argv[1]

    report_file_stats(filename)

    with console.status("Reading mesh..."):
        mesh = MeshReader3D(filename)

    with console.status("Collecting cell types.."):
        q = QualityInspector3D(mesh)
        q.count_cell_types()

    with console.status("Analyzing faces..."):
        q.analyze_faces()

    with console.status("Analyzing cells..."):
        q.analyze_cells()

    with console.status("Checking non-orthogonality...\n"):
        q.check_non_ortho()

    report_elements_count(console, mesh, q)

    report_mesh_stats(console, q)

    rprint("Count of duplicate nodes = ", q.duplicate_nodes_count())
    rprint("Mesh bounding box: ")
    for point in q.bounding_box():
        rprint(point)
