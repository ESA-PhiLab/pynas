# Repository Audit

Date: 2026-04-24

## Modernization completed

- Packaging metadata now targets Python 3.11 through 3.14 and uses the `esa-pynas`
  distribution name while preserving the `pynas` import package.
- Dependency management has moved from PDM to `uv`, with runtime dependencies,
  development dependencies, and optional `data` and `export` extras declared in
  `pyproject.toml`.
- GitHub Actions workflows cover CI on Linux, Windows, and macOS, plus a tag-based
  PyPI publishing workflow using trusted publishing.
- The package now loads its bundled `config.ini` through `importlib.resources`
  instead of relying on the current working directory.
- Initial unit tests cover architecture parsing, losses, metrics, data utilities,
  blocks, and a lightweight U-Net forward pass.

## Remaining gaps

- PyPI trusted publishing must still be configured outside the repository for
  project `esa-pynas`, repository `ESA-PhiLab/pynas`, workflow `publish.yml`, and
  environment `pypi`.
- Repository governance files are still missing: `CONTRIBUTING.md`, `SECURITY.md`,
  `CHANGELOG.md`, and `CITATION.cff`.
- Example coverage is limited. The current `examples/` directory has a README but
  no executable minimal example that can run without the full research dataset.
- Dataset-dependent workflows assume the burned-area dataset layout exists locally.
  CI intentionally avoids downloading large datasets.
- OpenVINO and ONNX export support remains optional and is not exercised in CI.
- `GenericLightningNetwork_Custom` was made import-safe, but its public behavior
  should be reviewed against the original intended custom-network API before use.
- Training and evolutionary search workflows are still high-cost integration paths.
  They need separate smoke data or mocked datamodules before they can be enforced
  in normal CI.
