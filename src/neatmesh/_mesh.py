from ._analyzer import Analyzer2D, Analyzer3D
from ._exceptions import InputMeshDimensionError
from ._reader import MeshReader2D, MeshReader3D, assign_reader


class Mesh2D:
    def __init__(self, mesh_file_path: str) -> None:
        self._reader: MeshReader2D = assign_reader(mesh_file_path)

        if isinstance(self._reader, MeshReader3D):
            raise InputMeshDimensionError(
                "Attempting to read a 3D mesh by a Mesh2D object, "
                "please use Mesh3D instead"
            )

        self._analyzer = Analyzer2D(self._reader)

        self._analyzer.count_face_types()
        self._analyzer.analyze_faces()
        self._analyzer.analyze_non_ortho()

        self.meshio_cell_blocks = self._reader.cell_blocks
        self.points = self._analyzer.points

        self.edges = self._analyzer.edges
        self.edge_normals = self._analyzer.edge_normals

        self.face_areas = self._analyzer.face_areas
        self.face_centers = self._analyzer.face_centers
        self.owner_neighbor = self._analyzer.owner_neighbor

        self.n_points = self._analyzer.n_points
        self.n_edges = self._analyzer.n_edges
        self.n_faces = self._analyzer.n_faces
        self.n_quad = self._analyzer.n_quad
        self.n_tri = self._analyzer.n_tri

        # boundary faces
        self.n_boundary_edges = self._analyzer.n_boundary_edges
        self.boundary_edges_mask = self.owner_neighbor[:, 1] == -1
        self.internal_edges_mask = self.owner_neighbor[:, 1] != -1


class Mesh3D:
    def __init__(self, mesh_file_path: str) -> None:
        self._reader: MeshReader3D = assign_reader(mesh_file_path)

        if isinstance(self._reader, MeshReader2D):
            raise InputMeshDimensionError(
                "Attempting to read a 2D mesh by a Mesh3D object, "
                "please use Mesh2D instead"
            )

        self._analyzer = Analyzer3D(self._reader)

        self._analyzer.count_cell_types()
        self._analyzer.analyze_faces()
        self._analyzer.analyze_cells()
        self._analyzer.analyze_non_ortho()

        self.meshio_cell_blocks = self._reader.cell_blocks
        self.points = self._analyzer.points

        # faces data
        self.faces = self._analyzer.faces
        self.face_normals = self._analyzer.face_normals
        self.face_centers = self._analyzer.face_centers
        self.face_areas = self._analyzer.face_areas
        self.face_non_ortho = self._analyzer.non_ortho
        self.owner_neighbor = self._analyzer.owner_neighbor

        # cells data
        self.cell_centers = self._analyzer.cells_centers
        self.cell_volumes = self._analyzer.cells_volumes

        # elements stats
        self.n_points = self._analyzer.n_points
        self.n_faces = self._analyzer.n_faces
        self.n_cells = self._analyzer.n_cells
        self.n_quad = self._analyzer.n_quad
        self.n_tri = self._analyzer.n_tri
        self.hex_count = self._analyzer.hex_count
        self.tetra_count = self._analyzer.tetra_count
        self.pyramid_count = self._analyzer.pyramid_count
        self.wedge_count = self._analyzer.wedge_count

        # boundary faces
        self.n_boundary_faces = self.n_boundary_faces
        self.boundary_faces_mask = self.owner_neighbor[:, 1] == -1
        self.internal_faces_mask = self.owner_neighbor[:, 1] != -1
