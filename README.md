<p align="center">
  <img src="https://raw.githubusercontent.com/ESA-PhiLab/pynas/main/docs/images/banner.png" alt="ESA Phi-lab and PyNAS banner" width="100%" />
</p>

<p align="center">
  <a href="https://github.com/ESA-PhiLab/pynas/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ESA-PhiLab/pynas/actions/workflows/ci.yml/badge.svg" /></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11+-blue.svg" /></a>
  <a href="https://github.com/ESA-PhiLab/pynas"><img alt="Version" src="https://img.shields.io/badge/version-0.1.0-informational.svg" /></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
  <a href="https://doi.org/10.5281/zenodo.15332053"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.15332053.svg" /></a>
</p>

# PyNAS

**PyNAS** is a Python framework for Neural Architecture Search (NAS) experiments focused on resource-constrained, edge-deployable deep learning models.
The project is developed by ESA Phi-lab with Little Place Lab and targets workflows where model accuracy, memory footprint, and deployment cost must be considered together.


## Why PyNAS

PyNAS provides building blocks for evolutionary NAS experiments, with an emphasis on compact neural networks for onboard and edge AI use cases.

| Area | What is included |
| --- | --- |
| Architecture generation | Utilities for generating, parsing, and rebuilding architecture codes |
| Search operators | Genetic mutation and single-point crossover primitives |
| Model blocks | Convolutional, pooling, activation, residual, classifier, and U-Net components |
| Training utilities | PyTorch Lightning modules, segmentation losses, metrics, and early stopping helpers |
| Data utilities | Hugging Face dataset download helper with retry handling |

## Project Status

| Item | Status |
| --- | --- |
| Package name | `esa-pynas` |
| Version | `0.1.0` |
| Python support | `>=3.11,<3.15` |
| Distribution | Local checkout; PyPI publication planned |
| License | Apache 2.0 |
| Development status | Alpha / research preview |

## Installation

PyNAS uses `uv` for reproducible dependency management. From a local checkout:

```bash
uv sync --locked
```

For development and verification:

```bash
uv sync --locked --dev
uv run pytest
uv run ruff format --check src scripts data tests
uv run ruff check src scripts data tests
```

The default lock configuration uses CPU PyTorch wheels for deterministic Windows, Linux, and macOS CI environments. For CUDA-enabled training workloads, install the PyTorch wheel index that matches your driver and hardware before running experiments.

## Quick Start

Generate and inspect a candidate architecture code:

```python
from pynas.core.architecture_builder import (
    generate_code_from_parsed_architecture,
    generate_random_architecture_code,
    parse_architecture_code,
)

architecture_code = generate_random_architecture_code()
parsed_architecture = parse_architecture_code(architecture_code)
round_tripped_code = generate_code_from_parsed_architecture(parsed_architecture)

print(architecture_code)
print(round_tripped_code)
```

Use the public package interface for training-related utilities:

```python
from pynas import (
    CategoricalCrossEntropyLoss,
    GenericLightningSegmentationNetwork,
    Individual,
    calculate_iou,
)
```

## Data

The burned-area segmentation dataset referenced by this project is hosted on Hugging Face:

- [ESA-PhiLab-Edge/LPL-Burned-Area-Seg](https://huggingface.co/datasets/ESA-PhiLab-Edge/LPL-Burned-Area-Seg)

Install the optional data dependencies and run the downloader:

```bash
uv sync --locked --extra data
uv run python data/download_hf_datasets.py
```

The downloader supports dataset and model repositories from the Hugging Face Hub, retry handling for transient network failures, progress reporting, and custom local output directories. Edit the `repo_ids` list in `data/download_hf_datasets.py` to change the default download targets.

## Repository Layout

```text
.
├── data/                 # Dataset download utilities and data notes
├── docs/                 # Documentation assets and static documentation files
├── examples/             # Usage examples and demos
├── notebooks/            # Experiment and walkthrough notebooks
├── papers/               # Research manuscript assets
├── scripts/              # Training and data-loading scripts
├── src/pynas/            # PyNAS Python package
│   ├── blocks/           # Neural network building blocks
│   ├── core/             # Architecture, population, configuration, and model logic
│   ├── opt/              # Evolutionary optimization operators
│   └── train/            # Losses, metrics, and training helpers
└── tests/                # Unit and integration tests
```

## Research Context

Spaceborne edge computing enables AI-capable CubeSats to process data onboard, reduce downlink pressure, and operate with greater autonomy. These systems face strict memory, power, and latency constraints, so model design must account for both predictive performance and deployment cost.

PyNAS explores evolutionary Neural Architecture Search for this setting. The framework is designed around compact segmentation architectures and hardware-aware optimization for CubeSat-class platforms such as NVIDIA Jetson AGX Orin and Intel Movidius Myriad X targets.

## Documentation

- Project documentation: [sirbastiano.github.io/pynas-docs](https://sirbastiano.github.io/pynas-docs/)
- Source repository: [github.com/ESA-PhiLab/pynas](https://github.com/ESA-PhiLab/pynas)
- ESA Phi-lab: [philab.esa.int](https://philab.esa.int)
- Little Place Lab: [littleplace.com](https://www.littleplace.com)

## Citation

If you use PyNAS in academic work, cite the Zenodo record below. A paper-specific citation will be added when the associated manuscript is available.

```bibtex
@software{pynas_zenodo_15332053,
  title = {PyNAS},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.15332053},
  url = {https://doi.org/10.5281/zenodo.15332053}
}
```

## Authors

- Roberto Del Prete ([Google Scholar](https://scholar.google.com/citations?user=Dwc8YxwAAAAJ))
- Parampuneet Thind ([Google Scholar](https://scholar.google.com/citations?user=Q71ynAkAAAAJ&hl=en&oi=sra))
- Andrea Mazzeo
- Lorenzo Papa ([Google Scholar](https://scholar.google.com/citations?user=P64hj-4AAAAJ))
- Matthew Whitley
- Gabriele Meoni ([Google Scholar](https://scholar.google.com/citations?user=vv34M9QAAAAJ))
- Nicolas Longepe ([Google Scholar](https://scholar.google.com/citations?user=YVVkIX8AAAAJ))

<p align="center">
  <img src="https://raw.githubusercontent.com/ESA-PhiLab/pynas/main/docs/images/lpl.jpg" alt="Little Place Lab logo" width="260" />
</p>

## License

This project is licensed under the **Apache License 2.0**. See [LICENSE](./LICENSE) for the full license text.
