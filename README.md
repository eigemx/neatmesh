# neatmesh
Mesh Quality Inspector

[![CI](https://github.com/eigenemara/neatmesh/actions/workflows/CI.yml/badge.svg)](https://github.com/eigenemara/neatmesh/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


neatmesh is a 2D/3D finite volume/element mesh quality inspector, neatmesh works with all formats supported by [meshio](https://github.com/nschloe/meshio).

neatmesh is in early active development stage, and all sort of contributions are welcome (specially if you're new to open source contribution! ☺️)

<p align="center">
    <img alt="neatmesh" src="https://media.githubusercontent.com/media/eigenemara/neatmesh/main/screenshots/cli.png" width="90%">
</p>


## Installation
Install with:

    pip install neatmesh

## Usage
Using neatmesh is simple:

    neatmesh my_awesome_mesh.su2

### Quality Rules
neatmesh will look for a quality rule file in current working directory, `neatmesh.toml` or `quality.toml` or `my_awesome_mesh.toml`. A quality rule file sets maximum values for quality metrics calculated by neatmesh, for example this is the content of a typical `neatmesh.toml` file:

    max_non_orhto = 50
    max_face_aspect_ratio = 10
    max_neighbor_volume_ratio = 3
    max_neighbor_area_ratio = 3

In case no quality rules file was present, neatmesh will use default max. values:

    max_non_orhto = 60
    max_face_aspect_ratio = 20
    max_neighbor_volume_ratio = 15
    max_neighbor_area_ratio = 15

