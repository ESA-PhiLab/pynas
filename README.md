![GitHub stars](https://img.shields.io/github/stars/ESA-PhiLab/Repo-Template.svg)
![GitHub forks](https://img.shields.io/github/forks/ESA-PhiLab/Repo-Template.svg)
![GitHub issues](https://img.shields.io/github/issues/ESA-PhiLab/Repo-Template.svg)
![GitHub pull requests](https://img.shields.io/github/issues-pr/ESA-PhiLab/Repo-Template.svg)
![GitHub last commit](https://img.shields.io/github/last-commit/ESA-PhiLab/Repo-Template.svg)
![GitHub code size](https://img.shields.io/github/languages/code-size/ESA-PhiLab/Repo-Template.svg)
![GitHub top language](https://img.shields.io/github/languages/top/ESA-PhiLab/Repo-Template.svg)
![GitHub repo size](https://img.shields.io/github/repo-size/ESA-PhiLab/Repo-Template.svg)
![GitHub contributors](https://img.shields.io/github/contributors/ESA-PhiLab/Repo-Template.svg)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-green.svg)](https://github.com/ESA-PhiLab/Repo-Template/wiki)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://makeapullrequest.com)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15332053.svg)](https://doi.org/10.5281/zenodo.15332053)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

(For these shield to work, set proper name..)


<p align="right">
<img src="./docs/images/banner.png" alt="ESA-PhiLab Logo" width="950" style='margin-bottom: 0em;'/>
</p>




## Repository Template


Use this template to kickstart your ESA-PhiLab project with all the necessary structure and documentation. You'll get:
- A professional repository structure following best practices
- Automatic GitHub Pages website generation (example: https://esa-philab.github.io/Repo-Template/)
- Pre-configured badges and documentation templates
- Standardized license and citation formats

Getting started is easy - just follow the detailed instructions in `docs/readme.md` to customize the template for your specific project.

## 📌 Project Title
> Replace this with your project name.

## 👥 Authors
List all project contributors. Preferably include links to their professional profiles.

- Name One ([profile](https://example.com))  
- Name Two ([profile](https://example.com))

## 📖 Project Reference
Specify the type and context of the project.

- Example: ESA Φ-lab Research Fellowship — [Link to ESA page](https://philab.esa.int)

## 📝 Abstract
Provide a concise summary of the project goals, methodology, and outcomes.

## 🛠️ How to Use
Detailed instructions to set up and run the project.

### Requirements
This repository uses a modern Python environment standard:
- Python ≥ 3.9
- Dependencies are specified in the `pyproject.toml` file.

To install:
```bash
pip install .
```
Or using PDM:
```bash
pdm install
```

## 📚 Citations
Please cite the following works if you use this code:
- [Reference Paper 1]
- [Reference Paper 2]

## 📂 Repository Structure
```
├── LICENSE
├── README.md
├── pyproject.toml      # project metadata & dependencies
├── setup.cfg           # optional packaging config
├── environment.yml     # optional environment specification
├── src/                # source code & Python package
│   └── your_package/   # replace with your package name
├── notebooks/          # Jupyter notebooks for experiments or analysis
├── papers/             # manuscript sources (LaTeX, figures, assets)
├── data/               # raw/processed datasets or external links
├── scripts/            # utility scripts and entry points
├── tests/              # unit and integration tests
├── docs/               # documentation (Sphinx, MkDocs, GitHub Pages)
│   ├── index.html      # main documentation page
│   ├── images/         # documentation assets (images,..)
└── examples/           # usage examples and demos
```

The `docs/` directory contains the project's documentation website with comprehensive information about the project, including:
- Project overview and goals
- Installation and setup instructions
- Usage examples and tutorials
- API documentation
- Links to related resources and publications
- Team and contributor information


## 📄 License
This project is licensed under the **CC BY-NC-ND 4.0** license. See the [LICENSE](./LICENSE) file or read more at [creativecommons.org](https://creativecommons.org/licenses/by-nc-nd/4.0/).

## 📊 Dataset Hosting
Datasets are either included in `data/` or hosted externally:
- [Zenodo](https://zenodo.org)
- [Hugging Face Datasets](https://huggingface.co/datasets)

Ensure external links are functional and persistent.

### 📥 Data Download Utilities

This repository includes a utility script for downloading datasets or models from the Hugging Face Hub:

```python
# Install required dependencies
pip install huggingface_hub

# Run the download script
python data/download_hf_datasets.py
```

The script (`data/download_hf_datasets.py`) provides a robust way to fetch datasets with automatic retries and error handling. It uses the `huggingface_hub` library to download repositories with the following features:

- Downloads datasets or models from Hugging Face Hub
- Configurable retry mechanism for handling network issues
- Progress tracking and error reporting
- Supports custom local directories for downloads
- Uses HF's optimized transfer protocol

To customize which datasets to download, edit the `repo_ids` list in the script.

## 🌐 GitHub Pages
This repository supports GitHub Pages. Request support to enable and configure the page if needed.

---

*For internal use only: This repository conforms to the ESA-PhiLab development and archiving standards.*
