import os
from typing import Dict, List, Tuple

import humanize
import numpy as np
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from neatmesh.analyzer import Analyzer2D, Analyzer3D
from neatmesh.reader import MeshReader


class Reporter:
    def __init__(self, console: Console, mesh: MeshReader, filename: str) -> None:
        self.console = console
        self.mesh = mesh
        self.filename = filename
        self.concerns: List[str] = []

    def report_elements_count(self):
        pass

    @staticmethod
    def stats_from_array(array: np.ndarray) -> Tuple[float, ...]:
        arr_max = np.nanmax(array)
        arr_min = np.nanmin(array)
        arr_mean = np.nanmean(array)
        arr_std = np.nanstd(array)

        return arr_max, arr_min, arr_mean, arr_std

    def report_mesh_stats(self, quality_metrics_dict):
        self.console.print()

        stats_table = Table(box=box.SIMPLE)
        stats_table.add_column("", justify="left")
        stats_table.add_column("Max.", justify="right")
        stats_table.add_column("Min.", justify="right")
        stats_table.add_column("Mean.", justify="right")
        stats_table.add_column("Std.", justify="right")
        stats_table.add_column("", justify="right")

        for metric_name, metric_dict in quality_metrics_dict.items():
            # Avoid calling stats_from_array in case input array is empty
            # this might occur in case of one cell mesh, where analyzer.adj_ratio is empty.
            if metric_dict["array"].shape[0] == 0:
                continue
            _max, _min, _mean, _std = self.stats_from_array(metric_dict["array"])
            status = ""

            if "max" in metric_dict:
                status = "[red]Not good" if _max > metric_dict["max"] else "[green]Ok"

                if _max > metric_dict["max"]:
                    n_failed = np.count_nonzero(
                        metric_dict["array"] > metric_dict["max"]
                    )
                    self.concerns.append(
                        f"* Found {n_failed} elements with '{metric_name}' greater than "
                        f" max. value {metric_dict['max']}."
                    )

            if _max > 1e-4 or not metric_dict["sci_not"]:
                stats_table.add_row(
                    metric_name,
                    f"{_max:.4f}",
                    f"{_min:.4f}",
                    f"{_mean:.4f}",
                    f"{_std:.4f}",
                    status,
                )
            else:
                stats_table.add_row(
                    metric_name,
                    f"{_max:4e}",
                    f"{_min:.4e}",
                    f"{_mean:.4e}",
                    f"{_std:.4e}",
                )

        panel = Panel(
            stats_table,
            title="[yellow bold]Quality Stats.",
            title_align="left",
            expand=False,
        )

        self.console.print(panel)

    def report_file_size(self, filename: str):
        fsize = humanize.naturalsize(os.path.getsize(filename))
        self.console.print(
            f"Inspecting file [cyan]'{filename}'[/] (file size: {fsize})"
        )
        self.console.print()

    def report_bounding_box(self):
        self.console.print("[yellow bold]Mesh bounding box: ")
        for point in self.analyzer.bounding_box():
            print(
                f"\t{point}",
            )
        self.console.print()

    def report_concerns(self):
        if self.concerns:
            concerns_table = Table(box=None)
            concerns_table.add_column("", justify="left")

            for concern in self.concerns:
                concerns_table.add_row(f"[red]{concern}")

            panel = Panel(
                concerns_table,
                expand=False,
                title="[yellow bold]Concerns",
                title_align="left",
            )
            self.console.print(panel)

    def report(self, rules: Dict[str, float]) -> None:
        pass


class Reporter2D(Reporter):
    def __init__(self, console: Console, mesh: MeshReader, filename: str) -> None:
        super().__init__(console, mesh, filename)

    def report_elements_count(self):
        face_count = self.analyzer.n_faces
        edge_count = self.analyzer.n_edges
        point_count = self.analyzer.n_points

        tree = Tree(label="[yellow bold]Elements Stats.")

        tree.add("Mesh is 2-Dimensional")
        points_branch = tree.add(f"Points count = {point_count}", highlight=True)

        duplicate_nodes = self.analyzer.duplicate_nodes_count()
        duplicate_nodes_branch = (
            "[green](Ok)" if duplicate_nodes == 0 else "[red](Not good)"
        )

        if duplicate_nodes > 0:
            self.concerns.append(f"* Found {duplicate_nodes} duplicate nodes.")

        points_branch.add(
            f"Duplicate nodes = {duplicate_nodes} {duplicate_nodes_branch}"
        )

        tree.add(
            f"Edges count = {edge_count}"
            f" (including {self.analyzer.n_boundary_edges} boundary edges)"
        )
        faces_branch = tree.add(f"Faces count = {face_count}")

        if self.analyzer.n_quad > 0:
            quad_pct = (self.analyzer.n_quad / self.analyzer.n_faces) * 100
            faces_branch.add(f"Quadilaterals: {self.analyzer.n_quad} ({quad_pct:.1f}%)")
        if self.analyzer.n_tri > 0:
            tri_pct = (self.analyzer.n_tri / self.analyzer.n_faces) * 100
            faces_branch.add(f"Triangles: {self.analyzer.n_tri} ({tri_pct:.1f}%)")

        self.console.print(tree)

    def report(self, rules: Dict[str, float]):
        with self.console.status("Collecting cell types.."):
            self.analyzer = Analyzer2D(self.mesh)  # type: ignore
            self.analyzer.count_face_types()

        with self.console.status("Analyzing faces..."):
            self.analyzer.analyze_faces()

        with self.console.status("Checking non-orthogonality..."):
            self.analyzer.analyze_non_ortho()

        with self.console.status("Checking adjacent cells volume ratio..."):
            self.analyzer.analyze_adjacents_area_ratio()

        self.report_file_size(self.filename)

        if not rules:
            self.console.print("No quality rules file found, using defaults.\n")
        else:
            self.console.print(f"Found quality rules file: '{rules['fname']}'\n")

        self.report_bounding_box()
        self.report_elements_count()

        quality_metric_dict = {
            "Face Area": {
                "array": self.analyzer.face_areas,
                "sci_not": True,
            },
            "Face Aspect Ratio": {
                "array": self.analyzer.face_aspect_ratios,
                "sci_not": False,
                "max": rules.get("max_face_aspect_ratio", 20),
            },
            "Non-Orthogonality": {
                "array": self.analyzer.non_ortho,
                "sci_not": False,
                "max": rules.get("max_non_orhto", 60),
            },
            "Adjacent Faces Area Ratio": {
                "array": self.analyzer.adj_ratio,
                "sci_not": False,
                "max": rules.get("max_neighbor_area_ratio", 15),
            },
        }

        self.report_mesh_stats(quality_metric_dict)
        self.report_concerns()


class Reporter3D(Reporter):
    def __init__(self, console: Console, mesh: MeshReader, filename: str) -> None:
        super().__init__(console, mesh, filename)

    def report_elements_count(self):
        cell_count = self.analyzer.n_cells
        face_count = self.analyzer.n_faces
        point_count = self.analyzer.n_points

        tree = Tree(label="[yellow bold]Elements Stats.")

        tree.add("Mesh is 3-Dimensional")
        points_branch = tree.add(f"Points count = {point_count}", highlight=True)

        duplicate_nodes = self.analyzer.duplicate_nodes_count()
        duplicate_nodes_status = (
            "[green](Ok)" if duplicate_nodes == 0 else "[red](Error)"
        )
        points_branch.add(
            f"Duplicate nodes = {duplicate_nodes} {duplicate_nodes_status}"
        )
        faces_branch = tree.add(
            f"Faces count = {face_count}"
            f" (including {self.analyzer.n_boundary_faces} boundary faces)"
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

    def report(self, rules: Dict[str, float]):
        with self.console.status("Collecting cell types.."):
            self.analyzer = Analyzer3D(self.mesh)  # type: ignore
            self.analyzer.count_cell_types()

        with self.console.status("Analyzing faces..."):
            self.analyzer.analyze_faces()

        with self.console.status("Analyzing cells..."):
            self.analyzer.analyze_cells()

        with self.console.status("Checking non-orthogonality..."):
            self.analyzer.analyze_non_ortho()

        with self.console.status("Checking adjacent cells volume ratio..."):
            self.analyzer.analyze_adjacents_volume_ratio()

        self.report_file_size(self.filename)

        if not rules:
            self.console.print("No quality rules file found, using defaults.\n")
        else:
            self.console.print(f"Found quality rules file: '{rules['fname']}'\n")

        self.report_bounding_box()
        self.report_elements_count()

        quality_metrics_dict = {
            "Face Area": {
                "array": self.analyzer.face_areas,
                "sci_not": True,
            },
            "Face Aspect Ratio": {
                "array": self.analyzer.face_aspect_ratios,
                "sci_not": False,
                "max": rules.get("max_face_aspect_ratio", 20),
            },
            "Cell Volume": {"array": self.analyzer.cells_volumes, "sci_not": True},
            "Non-Orthogonality": {
                "array": self.analyzer.non_ortho,
                "sci_not": False,
                "max": rules.get("max_non_orhto", 65),
            },
            "Adjacent Cells Volume Ratio": {
                "array": self.analyzer.adj_ratio,
                "sci_not": False,
                "max": rules.get("max_neighbor_volume_ratio", 15),
            },
        }

        self.report_mesh_stats(quality_metrics_dict)
        self.report_concerns()
