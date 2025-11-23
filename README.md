# TEMPL Pipeline

[![Live App](https://img.shields.io/badge/Live_App-templ.dyn.cloud.e--infra.cz-2ea44f?logo=google-chrome&logoColor=white)](https://templ.dyn.cloud.e-infra.cz/)
[![JCIM](https://img.shields.io/badge/JCIM-10.1021%2Facs.jcim.5c01985-blue)](https://doi.org/10.1021/acs.jcim.5c01985)
[![Open Access](https://img.shields.io/badge/Open_Access-CC--BY_4.0-green?logo=open-access)](https://pubs.acs.org/doi/10.1021/acs.jcim.5c01985)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16890956.svg)](https://doi.org/10.5281/zenodo.16890956)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Citation](https://github.com/fulopjoz/templ-pipeline/actions/workflows/cffconvert.yml/badge.svg)](https://github.com/fulopjoz/templ-pipeline/actions/workflows/cffconvert.yml)


[![CI](https://github.com/fulopjoz/templ-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/fulopjoz/templ-pipeline/actions/workflows/ci.yml)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=fulopjoz_templ-pipeline&metric=ncloc)](https://sonarcloud.io/summary/new_code?id=fulopjoz_templ-pipeline)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=fulopjoz_templ-pipeline&metric=sqale_rating)](https://sonarcloud.io/summary/new_code?id=fulopjoz_templ-pipeline)

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
- **⚡ Ultra-fast setup with UV package manager (10-100x faster)**

**⚠️ Scope:**

- Optimized for rapid pose prediction within known chemical space. Performance may be limited for novel scaffolds, allosteric sites, or targets with insufficient template coverage.

---

## Installation

### Quick Start (Recommended) ⚡

```bash
git clone https://github.com/fulopjoz/templ-pipeline
cd templ-pipeline
source quick-start.sh
```

**That's it!** The quick-start script automatically:
- Installs UV package manager (if needed)
- Detects your hardware configuration
- Creates the `.templ` virtual environment
- Installs dependencies with ultra-fast UV
- Downloads required datasets from Zenodo
- Verifies the installation
- Activates the environment

**Setup time: ~30-60 seconds** (vs. 5-10 minutes with traditional pip)

### Manual Installation

```bash
# Auto-detect hardware and install optimally
source setup_templ_env.sh

# Or choose a specific profile:
source setup_templ_env.sh --cpu-only    # Lightweight
source setup_templ_env.sh --web         # With web UI
source setup_templ_env.sh --full        # All features
source setup_templ_env.sh --dev         # Development
```

### Future Sessions

```bash
# Activate existing environment
source .templ/bin/activate

# Or use quick-start (checks if setup needed)
source quick-start.sh
```

### What's New in v3.0

- **🚀 UV Package Manager**: 10-100x faster dependency installation
- **🛡️ RDKit Compatibility Layer**: Prevents common API errors
- **⚙️ Optimized Build System**: Switched to modern `hatchling`
- **📦 Smart Caching**: Pre-compiled wheels and parallel downloads
- **🔧 Better Error Handling**: Clear messages and automatic fixes

See [MODERNIZATION.md](MODERNIZATION.md) for detailed information.

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

1. **Protein-ligand complexes: The general set minus refined set** (1.8 GB)
   → `PDBbind_v2020_other_PL`

2. **Protein-ligand complexes: The refined set** (658 MB)
   → `PDBbind_v2020_refined`

After downloading, extract both folders into `data/PDBBind/` using the standard directory structure.

---

## Project Structure

```text
.
├── setup_templ_env.sh        # One-shot environment setup (UV-optimized)
├── quick-start.sh            # Ultra-fast quick start script
├── pyproject.toml            # Modern packaging with hatchling
├── uv.toml                   # UV configuration for speed
├── MODERNIZATION.md          # Modernization guide
├── data/                     # Embeddings, ligands, splits (see data/README.md)
├── templ_pipeline/           # Main Python package and CLI
│   └── utils/                # Utilities (including RDKit compatibility)
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

- [`templ_pipeline/core/README.md`](templ_pipeline/core/README.md) – Module-by-module overview of the core scripts that implement the workflow in Figure 1.
- [`templ_pipeline/core/templ_demo.ipynb`](templ_pipeline/core/templ_demo.ipynb) – Executable notebook that reproduces the end-to-end Panel A → Panel B workflow using the bundled example data.


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

## RDKit Compatibility

The pipeline includes a compatibility layer to prevent common RDKit API errors:

```python
from templ_pipeline.utils import get_morgan_generator, get_rdkit_fingerprint

# Automatic API version detection
gen = get_morgan_generator(radius=2, fp_size=2048)
fp = gen.GetFingerprint(mol)

# Or use convenience function
fp = get_rdkit_fingerprint(mol, radius=2, fp_size=2048)
```

This prevents errors like:
```
Boost.Python.ArgumentError: Python argument types in
    rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator()
did not match C++ signature
```

See [MODERNIZATION.md](MODERNIZATION.md) for details.

---

## Performance

### Setup Speed Comparison

| Method | Time | Speedup |
|--------|------|---------|
| Traditional (pip) | 6.7 min | 1x |
| **Modern (uv)** | **0.6 min** | **11x** ⚡ |

### Key Optimizations

- **Parallel downloads**: 10 concurrent connections
- **Pre-compiled wheels**: No compilation needed
- **Smart caching**: Reuse packages across projects
- **Native TLS**: Faster SSL operations
- **Optimized resolution**: Rust-based dependency solver

---

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

For questions or discussions, please use [GitHub Discussions](https://github.com/fulopjoz/templ-pipeline/discussions).

### Development Setup

```bash
# Install with development tools
source setup_templ_env.sh --dev

# Run tests
pytest tests/

# Run benchmarks
templ benchmark polaris
```

---

## Authors

**Jozef Fülöp** [![ORCID iD](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0003-2599-7997)
CZ-OPENSCREEN, Department of Informatics and Chemistry, Faculty of Chemical Technology
University of Chemistry and Technology, Prague

**Martin Šícho** [![ORCID iD](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0002-8771-1731)
CZ-OPENSCREEN, Department of Informatics and Chemistry, Faculty of Chemical Technology
University of Chemistry and Technology, Prague

**Wim Dehaen** [![ORCID iD](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0001-6979-5508)
CZ-OPENSCREEN, Department of Informatics and Chemistry & Department of Organic Chemistry
University of Chemistry and Technology, Prague
📧 [dehaenw@vscht.cz](mailto:dehaenw@vscht.cz)

---

## Citation


For the research paper describing the method, please cite:

```bibtex
@article{fulop2025templ,
  title={TEMPL: A Template-Based Protein--Ligand Pose Prediction Baseline},
  author={Fülöp, Jozef and Šícho, Martin and Dehaen, Wim},
  journal={Journal of Chemical Information and Modeling},
  year={2025},
  publisher={American Chemical Society},
  doi={10.1021/acs.jcim.5c01985},
  url={https://doi.org/10.1021/acs.jcim.5c01985},
  note={Published as part of the special issue "Open Science and Blind Data: The Antiviral Discovery Challenge"}
}
```

If you use TEMPL in your research, please cite the software:

```bibtex
@software{templ2025,
  title={TEMPL: Template-based protein-ligand pose prediction},
  author={Fülöp, Jozef and Šícho, Martin and Dehaen, Wim},
  institution={University of Chemistry and Technology, Prague},
  url={https://github.com/fulopjoz/templ-pipeline},
  doi={10.5281/zenodo.16890956},
  year={2025},
  version={1.0.0}
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
