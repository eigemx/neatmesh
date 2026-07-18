# neatmesh
Mesh Quality Inspector

[![build](https://github.com/eigemx/neatmesh/actions/workflows/build.yml/badge.svg)](https://github.com/eigemx/neatmesh/actions/workflows/build.yml)[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


neatmesh is a 2D/3D mesh quality inspector and a python library that computes cells volumes, face areas, centers, normals and many more. neatmesh works with all formats supported by [meshio](https://github.com/nschloe/meshio).

neatmesh is in early active development stage, and all sort of contributions are welcome (specially if you're new to open source contribution! ☺️)

<p align="center">
    <img alt="neatmesh" src="https://media.githubusercontent.com/media/eigemx/neatmesh/main/screenshots/cli.png" width="90%">
</p>


## Installation
Install with:

    pip install neatmesh

## Usage
Using neatmesh is simple:

    neatmesh my_awesome_mesh.su2

or

    import neatmesh as nm
    
    mesh = nm.read('my_awesome_mesh.su2')
    # mesh.face_areas, mesh.face_centers, mesh.owner_neighbor,
    # mesh.n_points, mesh.n_edges, mesh.n_faces, ..... 
    
## What does neatmesh report?
neatmesh will report file size, mesh bounding box, mesh dimensionality, points count, duplicate points count, face counts and types, cell counts and types, and the following quality statistics
1. **Face Area**: face area in mesh units.
2. **Face Aspect Ratio**: ratio between max. and min. length of face bounding rectangle.
3. **Cell Volume**: volume of 3D cell, current supported cell types: hexahedron, tetrahedron, wedge and pyramid.
4. **Non-Orthogonality**: angle in degrees between (a) vector connecting two elements centers NP and (b) normal vector to the shared face or edge between the elements 

![Non-ortho example](https://media.githubusercontent.com/media/eigemx/neatmesh/main/screenshots/nonortho.png)

Source: [Introduction to Computational Fluid Dynamics: Governing Equations, Turbulence Modeling Introduction and Finite Volume Discretization Basics. - Joel Guerrero](https://www.researchgate.net/publication/280385280_Introduction_to_Computational_Fluid_Dynamics_Governing_Equations_Turbulence_Modeling_Introduction_and_Finite_Volume_Discretization_Basics)

5. **Neighbor Element Volume/Area Ratio**: ratio between the volume (for 3D mesh) or area (for 2D mesh) of all neighboring cells, defined as `max_vol / min_vol` or `max_area / min_area`, so it's always greater than or equal 1.

### Quality Rules
neatmesh will look for a quality rule file in current working directory: `neatmesh.toml`, `quality.toml`, or `<input_file_stem>.toml`. A quality rule file sets maximum values for quality metrics calculated by neatmesh, for example this is the content of a typical `neatmesh.toml` file:

You can also specify a custom rules file via `--rules` / `-r`:

    neatmesh my_awesome_mesh.su2 --rules /path/to/custom_rules.toml

When `--rules` is provided, it takes precedence over the automatic CWD search.

You can also specify a length unit for reporting via `--length-unit` / `-u`:

    neatmesh my_awesome_mesh.su2 --length-unit m

When provided, the unit is shown in place of the generic `L` (e.g., `[m^2]`, `[m^3]`).

    max_non_orhto = 50
    max_face_aspect_ratio = 10
    max_neighbor_volume_ratio = 3
    max_neighbor_area_ratio = 3
    duplicate_nodes_warning = false

In case no quality rules file was present, neatmesh will use default max. values:

    max_non_orhto = 60
    max_face_aspect_ratio = 20
    max_neighbor_volume_ratio = 15
    max_neighbor_area_ratio = 15
    duplicate_nodes_warning = true (default)

