"""neatmesh cli entry point"""

import argparse
import os
import sys
from pathlib import Path

import toml
from rich.console import Console

from ._exceptions import InvalidMeshError
from ._reader import MeshReader3D, assign_reader
from ._reporter import Reporter2D, Reporter3D


# pylint: disable=import-outside-toplevel
def version() -> str:
    """get neatmesh version"""
    try:
        from importlib import metadata

        __version__ = metadata.version("neatmesh")
    except ImportError:
        __version__ = "0.0.0"

    return __version__


def header_str():
    """neatmesh header"""
    return f"neatmesh v{version()} — Mesh Quality Inspector | MIT License"


def error(msg: str) -> None:
    """display fatal error message and exit"""
    Console(stderr=True).print(f"[red][bold]Error:[/bold] {msg}[/red]")
    sys.exit(-1)


def main():
    """cli entry points"""
    print(header_str())

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file", metavar="input", type=str, nargs=1, help="Input mesh file name"
    )
    parser.add_argument(
        "--rules",
        "-r",
        type=str,
        default=None,
        help="Path to quality rules TOML file",
    )
    parser.add_argument(
        "--length-unit",
        "-u",
        type=str,
        default=None,
        help="Length unit for reporting (e.g., m, cm, mm)",
    )

    args = parser.parse_args()
    filename = args.input_file[0]

    if not os.path.isfile(filename):
        error(f"file {filename} does not exist!")

    if args.rules is not None and not os.path.isfile(args.rules):
        error(f"rules file {args.rules} does not exist!")

    length_unit = args.length_unit
    if length_unit is not None:
        length_unit = length_unit.strip()
        if not length_unit:
            length_unit = None
        elif len(length_unit) > 20:
            error(
                f"length unit '{length_unit}' exceeds maximum length of 20 characters"
            )

    console = Console()

    with console.status("Reading mesh..."):
        try:
            mesh = assign_reader(filename)
        except InvalidMeshError as mesh_error:
            error(f"{mesh_error}")

    if isinstance(mesh, MeshReader3D):
        reporter = Reporter3D(console, mesh, filename, length_unit)
    else:
        reporter = Reporter2D(console, mesh, filename, length_unit)

    # Check quality rules file, if exists
    rules_dict = {}
    if args.rules is not None:
        rules_dict = toml.load(args.rules)
        rules_dict["fname"] = args.rules
    else:
        fname_stripped = Path(filename).stem
        quality_rules_candidates = [
            f"{fname_stripped}.toml",
            "neatmesh.toml",
            "quality.toml",
        ]

        for fname in quality_rules_candidates:
            if os.path.isfile(fname):
                rules_dict = toml.load(fname)
                rules_dict["fname"] = fname
                break

    reporter.report(rules_dict)
