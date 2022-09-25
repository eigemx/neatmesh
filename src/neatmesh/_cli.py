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
    """neatmesh logo, version and license header"""
    return rf"""
                     __                      __
   ____  ___  ____ _/ /_____ ___  ___  _____/ /_
  / __ \/ _ \/ __ `/ __/ __ `__ \/ _ \/ ___/ __ \   Version: {version()}
 / / / /  __/ /_/ / /_/ / / / / /  __(__  ) / / /   License: MIT
/_/ /_/\___/\__,_/\__/_/ /_/ /_/\___/____/_/ /_/

"""


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

    args = parser.parse_args()
    filename = args.input_file[0]

    if not os.path.isfile(filename):
        error(f"file {filename} does not exist!")

    console = Console()

    with console.status("Reading mesh..."):
        try:
            mesh = assign_reader(filename)
        except InvalidMeshError as e:
            error(f"{e}")

    if isinstance(mesh, MeshReader3D):
        reporter = Reporter3D(console, mesh, filename)
    else:
        reporter = Reporter2D(console, mesh, filename)

    # Check quality rules file, if exists
    fname_stripped = Path(filename).stem
    quality_rules_candidates = [
        f"{fname_stripped}.toml",
        "neatmesh.toml",
        "quality.toml",
    ]

    rules_dict = {}
    for fname in quality_rules_candidates:
        if os.path.isfile(fname):
            rules_dict = toml.load(fname)
            rules_dict["fname"] = fname
            break

    reporter.report(rules_dict)
