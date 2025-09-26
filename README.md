# TEMPL Pipeline

[![Live App](https://img.shields.io/badge/Live_App-templ.dyn.cloud.e--infra.cz-2ea44f?logo=google-chrome&logoColor=white)](https://templ.dyn.cloud.e-infra.cz/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/fulopjoz/templ-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/fulopjoz/templ-pipeline/actions/workflows/ci.yml)
[![Citation](https://github.com/fulopjoz/templ-pipeline/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/fulopjoz/templ-pipeline/actions/workflows/cffconvert.yml)

[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=fulopjoz_templ-pipeline&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=fulopjoz_templ-pipeline)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=fulopjoz_templ-pipeline&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=fulopjoz_templ-pipeline)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16890956.svg)](https://doi.org/10.5281/zenodo.16890956)
[![ChemRxiv](https://img.shields.io/badge/ChemRxiv-10.26434%2Fchemrxiv--2025--0v7b1-orange)](https://doi.org/10.26434/chemrxiv-2025-0v7b1)


Template-based protein–ligand pose prediction with command-line interface and web application.

---

## Overview

TEMPL is a template-based method for rapid protein–ligand pose prediction that leverages ligand similarity and template superposition. The method uses maximal common substructure (MCS) alignment and constrained conformer generation (ETKDG v3) for pose generation within known chemical space.

**Key Features:**

- Template-based pose prediction using ligand similarity
- Alignment driven by maximal common substructure (MCS)
- Shape and pharmacophore scoring for pose selection
- Built-in benchmarks (Polaris, time-split PDBbind)
- CPU/GPU adaptive

**⚠️ Scope:**

- Optimized for rapid pose prediction within known chemical space. Performance may be limited for novel scaffolds, allosteric sites, or targets with insufficient template coverage.

---

## Installation

### Installation (one-time)

```bash
git clone https://github.com/fulopjoz/templ-pipeline
cd templ-pipeline
source setup_templ_env.sh
```

The setup script automatically:

- Detects hardware configuration
- Creates the `.templ` virtual environment
- Installs dependencies with `uv`
- Downloads required datasets from Zenodo
- Verifies installation

For future sessions, activate the environment:

```bash
source .templ/bin/activate
```

---

## Data Requirements

TEMPL requires pre-computed embeddings and ligand structures that are automatically downloaded during setup from [Zenodo](https://doi.org/10.5281/zenodo.16890956):

- `templ_protein_embeddings_v1.0.0.npz` (~90MB) - Pre-computed ESM-2 protein embeddings for 18,902 PDBBind structures
- `templ_processed_ligands_v1.0.0.sdf.gz` (~10MB) - Processed ligand molecules

**Related Zenodo Datasets:**
- [TEMPL Pipeline Core Dataset](https://doi.org/10.5281/zenodo.15813500) - Essential data files for pipeline operation
- [TEMPL Pipeline Benchmark Results Dataset](https://doi.org/10.5281/zenodo.16875932) - PDBBind timesplit and Polaris benchmark results

See `data/README.md` for directory layout and the provided benchmark splits under `data/splits/`.

### PDBbind Dataset

For benchmarking, download the following **freely available v2020** subsets from the [official PDBbind website](https://www.pdbbind-plus.org.cn/download):

1. **Protein-ligand complexes: The general set minus refined set** (1.8 GB)
   → `PDBbind_v2020_other_PL`

2. **Protein-ligand complexes: The refined set** (658 MB)
   → `PDBbind_v2020_refined`

After downloading, extract both folders into `data/PDBBind/` using the standard directory structure.

---

## Project Structure

```text
.
├── setup_templ_env.sh        # One-shot environment setup
├── pyproject.toml            # Packaging and dependencies
├── data/                     # Embeddings, ligands, splits (see data/README.md)
├── templ_pipeline/           # Main Python package and CLI
├── scripts/                  # Helper entry points (UI launcher, tests, benchmarks)
├── output/                   # Pipeline run outputs (templ_run_...)
├── benchmarks/               # Benchmark workspaces and archives
├── tests/                    # Unit, integration, performance tests
├── deploy/                   # Docker and Kubernetes assets
├── docs/                     # Additional documentation
├── diagrams/                 # Architecture and flow diagrams
├── tools/                    # Dev configs (pytest, workspace)
└── README.md                 # This file
```

---

## Usage

### Core Workflow Documentation

- [`templ_pipeline/core/README.md`](templ_pipeline/core/README.md) – Module-by-module overview of the core scripts that implement the workflow in Figure 1.
- [`templ_pipeline/core/templ_demo.ipynb`](templ_pipeline/core/templ_demo.ipynb) – Executable notebook that reproduces the end-to-end Panel A → Panel B workflow using the bundled example data.


### Command Line Interface

```bash
# Basic pose prediction
templ run --protein-file protein.pdb --ligand-smiles "C1CC(=O)N(C1)CC(=O)N"

# Using PDB ID
templ run --protein-pdb-id 1iky --ligand-smiles "C1CC(=O)N(C1)CC(=O)N"

# Using SDF file
templ run --protein-file protein.pdb --ligand-file ligand.sdf

# Show available commands
templ --help
```

### Web Interface

Hosted app: [templ.dyn.cloud.e-infra.cz](https://templ.dyn.cloud.e-infra.cz/)

```bash
python scripts/run_streamlit_app.py
```

#### Notes

- The launcher picks the first free port starting at 8501. Override via PORT or TEMPL_PORT_START.
- It prints both Local and Network URLs; the app listens on 0.0.0.0 for LAN access.

### Benchmarking

```bash
# Full benchmark
templ benchmark polaris
templ benchmark time-split # PDBBind dataset

# Partial benchmark
templ benchmark time-split --test-only
```

#### Outputs

- Individual runs are written to output/ as templ_run_YYYYMMDD_HHMMSS_[pdbid]/
- Benchmark artifacts are saved under benchmarks/[suite]/...

---

## API Reference

| Command | Description |
|---------|-------------|
| `templ run` | Complete pipeline (recommended) |
| `templ embed` | Generate protein embeddings |
| `templ find-templates` | Find similar protein templates |
| `templ generate-poses` | Generate ligand poses |

For detailed command options, run `templ --help` or `templ <command> --help`.

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

For questions or discussions, please use [GitHub Discussions](https://github.com/fulopjoz/templ-pipeline/discussions).

---

## Citation

If you use TEMPL in your research, please cite the software:

```bibtex
@software{templ2025,
  title={TEMPL: Template-based protein-ligand pose prediction},
  author={J. Fülöp and M. Šícho and W. Dehaen},
  institution={University of Chemistry and Technology, Prague},
  url={https://github.com/fulopjoz/templ-pipeline},
  year={2025}
}
```

For the research paper describing the method, please cite:

```bibtex
@article{templ2025preprint,
  title={TEMPL: A template-based protein-ligand pose prediction baseline},
  author={J. Fülöp and M. Šícho and W. Dehaen},
  journal={ChemRxiv},
  year={2025},
  doi={10.26434/chemrxiv-2025-0v7b1},
  url={https://chemrxiv.org/engage/chemrxiv/article-details/68a48565a94eede1547e98fd},
  note={This content is a preprint and has not been peer-reviewed}
}
```

---

## Acknowledgement

J.F., M.Š. and W.D. were supported by the Ministry of Education, Youth and Sports of the Czech Republic – National Infrastructure for Chemical Biology (CZ-OPENSCREEN, LM2023052). W.D. was supported by the Ministry of Education, Youth and Sports of the Czech Republic by the project "New Technologies for Translational Research in Pharmaceutical Sciences/NETPHARM", project ID CZ.02.01.01/00/22_008/0004607, cofunded by the European Union.
Computational resources were provided by the e-INFRA CZ project (ID:90254), supported by the Ministry of Education, Youth and Sports of the Czech Republic.


---

## License

This project uses dual licensing:

- **Software Code**: Licensed under the [MIT License](LICENSE)
- **Data, Documentation, and Research Outputs**: Licensed under [CC BY 4.0](LICENSES/CC-BY-4.0.txt)
- **Third-party Data**: [Polaris benchmark datasets](https://polarishub.io/datasets/asap-discovery/antiviral-admet-2025-unblinded) are CC0-1.0 (public domain)

The software components use the permissive MIT License for maximum reusability. Research data and documentation are shared under Creative Commons Attribution 4.0 International License for proper academic attribution.
