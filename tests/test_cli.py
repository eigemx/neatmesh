import pathlib
import sys
from unittest.mock import patch

import pytest

from neatmesh._cli import main

this_dir = pathlib.Path(__file__).resolve().parent
MESH_DIR = this_dir / "meshes"


def test_rules_valid_file(tmp_path):
    rules_file = tmp_path / "custom_rules.toml"
    rules_file.write_text("max_non_ortho = 30\n")
    mesh_file = MESH_DIR / "one_hex_cell.vtk"

    sys.argv = ["neatmesh", str(mesh_file), "--rules", str(rules_file)]

    with patch("neatmesh._cli.Reporter3D") as mock_reporter:
        mock_instance = mock_reporter.return_value
        main()
        rules_dict = mock_instance.report.call_args[0][0]
        assert rules_dict["fname"] == str(rules_file)
        assert rules_dict["max_non_ortho"] == 30


def test_rules_nonexistent_file(tmp_path):
    mesh_file = MESH_DIR / "one_hex_cell.vtk"
    sys.argv = [
        "neatmesh",
        str(mesh_file),
        "--rules",
        str(tmp_path / "nonexistent.toml"),
    ]

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == -1


def test_rules_short_form(tmp_path):
    rules_file = tmp_path / "short_rules.toml"
    rules_file.write_text("max_face_aspect_ratio = 5\n")
    mesh_file = MESH_DIR / "one_hex_cell.vtk"

    sys.argv = ["neatmesh", str(mesh_file), "-r", str(rules_file)]

    with patch("neatmesh._cli.Reporter3D") as mock_reporter:
        mock_instance = mock_reporter.return_value
        main()
        rules_dict = mock_instance.report.call_args[0][0]
        assert rules_dict["fname"] == str(rules_file)
        assert rules_dict["max_face_aspect_ratio"] == 5


def test_no_rules_fallback_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rules_file = tmp_path / "neatmesh.toml"
    rules_file.write_text("max_non_ortho = 45\n")

    mesh_file = MESH_DIR / "one_hex_cell.vtk"
    sys.argv = ["neatmesh", str(mesh_file)]

    with patch("neatmesh._cli.Reporter3D") as mock_reporter:
        mock_instance = mock_reporter.return_value
        main()
        rules_dict = mock_instance.report.call_args[0][0]
        assert rules_dict["fname"] == "neatmesh.toml"
        assert rules_dict["max_non_ortho"] == 45


def test_no_rules_fallback_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mesh_file = MESH_DIR / "one_hex_cell.vtk"
    sys.argv = ["neatmesh", str(mesh_file)]

    with patch("neatmesh._cli.Reporter3D") as mock_reporter:
        mock_instance = mock_reporter.return_value
        main()
        rules_dict = mock_instance.report.call_args[0][0]
        assert rules_dict == {}


def test_rules_precedes_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cwd_rules = tmp_path / "neatmesh.toml"
    cwd_rules.write_text("max_non_ortho = 10\n")
    custom_rules = tmp_path / "custom.toml"
    custom_rules.write_text("max_non_ortho = 99\n")

    mesh_file = MESH_DIR / "one_hex_cell.vtk"
    sys.argv = ["neatmesh", str(mesh_file), "--rules", str(custom_rules)]

    with patch("neatmesh._cli.Reporter3D") as mock_reporter:
        mock_instance = mock_reporter.return_value
        main()
        rules_dict = mock_instance.report.call_args[0][0]
        assert rules_dict["fname"] == str(custom_rules)
        assert rules_dict["max_non_ortho"] == 99


def test_length_unit_passed_to_reporter(tmp_path):
    mesh_file = MESH_DIR / "one_hex_cell.vtk"
    sys.argv = ["neatmesh", str(mesh_file), "--length-unit", "m"]

    with patch("neatmesh._cli.Reporter3D") as mock_reporter:
        main()
        args, _ = mock_reporter.call_args
        assert args[3] == "m"


def test_length_unit_short_form(tmp_path):
    mesh_file = MESH_DIR / "one_hex_cell.vtk"
    sys.argv = ["neatmesh", str(mesh_file), "-u", "cm"]

    with patch("neatmesh._cli.Reporter3D") as mock_reporter:
        main()
        args, _ = mock_reporter.call_args
        assert args[3] == "cm"


def test_length_unit_not_provided(tmp_path):
    mesh_file = MESH_DIR / "one_hex_cell.vtk"
    sys.argv = ["neatmesh", str(mesh_file)]

    with patch("neatmesh._cli.Reporter3D") as mock_reporter:
        main()
        args, _ = mock_reporter.call_args
        assert args[3] is None


def test_length_unit_too_long(tmp_path):
    mesh_file = MESH_DIR / "one_hex_cell.vtk"
    long_unit = "a" * 21
    sys.argv = ["neatmesh", str(mesh_file), "--length-unit", long_unit]

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == -1


def test_duplicate_nodes_warning_default(tmp_path):
    rules_file = tmp_path / "rules.toml"
    rules_file.write_text("")
    mesh_file = MESH_DIR / "duplicate_nodes.su2"

    sys.argv = ["neatmesh", str(mesh_file), "--rules", str(rules_file)]

    with patch("neatmesh._cli.Reporter2D") as mock_reporter:
        main()
        rules_dict = mock_reporter.return_value.report.call_args[0][0]
        assert "duplicate_nodes_warning" not in rules_dict


def test_duplicate_nodes_warning_disabled(tmp_path):
    rules_file = tmp_path / "rules.toml"
    rules_file.write_text("duplicate_nodes_warning = false\n")
    mesh_file = MESH_DIR / "duplicate_nodes.su2"

    sys.argv = ["neatmesh", str(mesh_file), "--rules", str(rules_file)]

    with patch("neatmesh._cli.Reporter2D") as mock_reporter:
        main()
        rules_dict = mock_reporter.return_value.report.call_args[0][0]
        assert rules_dict["duplicate_nodes_warning"] is False


def test_length_unit_empty_string(tmp_path):
    mesh_file = MESH_DIR / "one_hex_cell.vtk"
    sys.argv = ["neatmesh", str(mesh_file), "--length-unit", ""]

    with patch("neatmesh._cli.Reporter3D") as mock_reporter:
        main()
        args, _ = mock_reporter.call_args
        assert args[3] is None
