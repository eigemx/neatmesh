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

from neatmesh.analyzer import Analyzer3D
from neatmesh.reader import MeshReader3D


class Reporter:
    def __init__(self, analyzer: Analyzer3D, console: Console) -> None:
        self.analyzer = analyzer
        self.console = console

    def report_elements_count(self):
        cell_count = self.analyzer.n_cells
        face_count = self.analyzer.n_faces
        point_count = self.analyzer.n_points

        tree = Tree(label="[yellow bold]Elements Stats.")
        duplicate_nodes = self.analyzer.duplicate_nodes_count()

        points_branch = tree.add(f"Points count = {point_count}", highlight=True)
        duplicate_nodes_status = (
            "[green](Ok)" if duplicate_nodes == 0 else "[red](Error)"
        )
        points_branch.add(
            f"Duplicate nodes = {duplicate_nodes} {duplicate_nodes_status}"
        )
        faces_branch = tree.add(
            f"Faces count = {face_count}, with {self.analyzer.n_boundary_faces} boundary faces"
        )

        if self.analyzer.n_quad > 0:
            quad_pct = (self.analyzer.n_quad / self.analyzer.n_faces) * 100
            faces_branch.add(f"Quadilaterals: {self.analyzer.n_quad} ({quad_pct:.1f}%)")
        if self.analyzer.n_tri > 0:
            tri_pct = (self.analyzer.n_tri / self.analyzer.n_faces) * 100
            faces_branch.add(f"Triangles: {self.analyzer.n_tri} ({tri_pct:.1f}%)")

        cells_branch = tree.add(f"Cells count = {cell_count}")
        for ctype, count in (
            ("Hexahedron", self.analyzer.hex_count),
            ("Tetrahedron", self.analyzer.tetra_count),
            ("Wedge", self.analyzer.wedge_count),
            ("Pyramid", self.analyzer.pyramid_count),
        ):
            if count == 0:
                continue

            pct = (count / self.analyzer.n_cells) * 100
            cells_branch.add(f"{ctype}s: {count} ({pct:.1f}%)")

        self.console.print(tree)

    @staticmethod
    def stats_from_array(array: np.ndarray) -> Tuple[float, ...]:
        arr_max = np.nanmax(array)
        arr_min = np.nanmin(array)
        arr_mean = np.nanmean(array)
        arr_std = np.nanstd(array)

        return arr_max, arr_min, arr_mean, arr_std

    def report_mesh_stats(self) -> Panel:
        self.console.print()

        stats = {
            "Face Area": {
                "array": self.analyzer.face_areas,
                "sci_not": True,
            },
            "Face Aspect Ratio": {
                "array": self.analyzer.face_aspect_ratios,
                "sci_not": False,
                "max": 15.0,
            },
            "Cell Volume": {"array": self.analyzer.cells_volumes, "sci_not": True},
            "Non-Orthogonality": {
                "array": self.analyzer.non_ortho,
                "sci_not": False,
                "max": 50,
            },
            "Adjacent Cells Volume Ratio": {
                "array": self.analyzer.adj_ratio,
                "sci_not": False,
                "max": 10,
            },
        }

        stats_table = Table(box=box.SIMPLE)
        stats_table.add_column("", justify="left")
        stats_table.add_column("Max.", justify="right")
        stats_table.add_column("Min.", justify="right")
        stats_table.add_column("Mean.", justify="right")
        stats_table.add_column("Std.", justify="right")
        stats_table.add_column("", justify="right")

        for stat, stats_dict in stats.items():
            _max, _min, _mean, _std = self.stats_from_array(stats_dict["array"])
            status = ""

            if "max" in stats_dict:
                status = "[red]Not good" if _max > stats_dict["max"] else "[green]Ok"

            if _max > 1e-4 or not stats_dict["sci_not"]:
                stats_table.add_row(
                    stat,
                    f"{_max:.4f}",
                    f"{_min:.4f}",
                    f"{_mean:.4f}",
                    f"{_std:.4f}",
                    status,
                )
            else:
                stats_table.add_row(
                    stat, f"{_max:4e}", f"{_min:.4e}", f"{_mean:.4e}", f"{_std:.4e}"
                )

        panel = Panel(
            stats_table, title="[yellow bold]Quality Stats.", title_align="left", expand=False
        )

        self.console.print(panel)

    def report_file_size(self, filename: str):
        fsize = humanize.naturalsize(os.path.getsize(filename))
        self.console.print(f"Inspecting file [cyan]'{filename}'[/] (file size: {fsize})")
        self.console.print()

    def report_bounding_box(self):
        self.console.print("[yellow bold]Mesh bounding box: ")
        for point in self.analyzer.bounding_box():
            rprint(f"\t{point}")
        self.console.print()


def header_str(version: str):
    return f"""
                     __                      __
   ____  ___  ____ _/ /_____ ___  ___  _____/ /_
  / __ \/ _ \/ __ `/ __/ __ `__ \/ _ \/ ___/ __ \   Version: {version}
 / / / /  __/ /_/ / /_/ / / / / /  __(__  ) / / /   License: MIT
/_/ /_/\___/\__,_/\__/_/ /_/ /_/\___/____/_/ /_/

"""


def main():
    print(
        header_str("0.0.1b"),
    )

    filename = sys.argv[1]
    console = Console()

    with console.status("Reading mesh..."):
        mesh = MeshReader3D(filename)

    with console.status("Collecting cell types.."):
        analyzer = Analyzer3D(mesh)
        analyzer.count_cell_types()

    with console.status("Analyzing faces..."):
        analyzer.analyze_faces()

    with console.status("Analyzing cells..."):
        analyzer.analyze_cells()

    with console.status("Checking non-orthogonality..."):
        analyzer.check_non_ortho()

    with console.status("Checking adjacent cells volume ratio..."):
        analyzer.check_adjacents_volume_ratio()

    reporter = Reporter(analyzer, console)
    reporter.report_file_size(filename)
    reporter.report_bounding_box()
    reporter.report_elements_count()
    reporter.report_mesh_stats()
