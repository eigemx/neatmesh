# Changelog

## v0.2.0

### Features
- Add `--rules` / `-r` CLI flag to specify a custom quality rules TOML file
- Add `--length-unit` / `-u` CLI flag to set a reporting length unit (e.g., m, cm, mm)
- Support `length_unit` in 2D area and 3D volume/surface area console output
- Add `duplicate_nodes_warning` rule (boolean, default `true`) to control duplicate node warnings
- Change duplicate nodes display from red `Not good` / `Error` to yellow `Warning`
- Improve concern percentage formatting for sub-1% and sub-0.001% cases
- Fix tab indentation in bounding box console output

### Tests
- Add `duplicate_nodes.su2` test mesh (single quad with 1 duplicate node)
- Add `test_duplicate_nodes` analyzer test
- Add CLI tests for `--rules`, `--length-unit`, and `duplicate_nodes_warning`

## v0.1.0

Initial release.
