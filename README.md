# TEMPL Pipeline

Template-based protein–ligand pose prediction for computational drug discovery.

## Overview

TEMPL leverages ligand similarity and template superposition to predict protein-ligand binding poses. The method uses maximal common substructure (MCS) alignment, constrained conformer generation, and shape/pharmacophore scoring to provide fast, accurate poses for familiar chemical space.

**Key Features:**
- Template-based pose prediction using ligand similarity
- Maximal common substructure (MCS) alignment
- Constrained conformer generation (ETKDG v3)
- Shape and pharmacophore scoring for pose selection
- Built-in benchmarks (Polaris, time-split PDBbind)
- Web interface and command-line tools
- CPU-optimized with optional GPU acceleration

## Installation

```bash
git clone https://github.com/fulopjoz/templ-pipeline
cd templ-pipeline
source setup_templ_env.sh
```

The setup script will automatically detect your hardware, install dependencies, and configure the environment.

## Quick Start

### Command Line Interface

```bash
# Activate the environment
source .templ/bin/activate

# Predict poses from SMILES
templ run \
  --protein-file examples/1a1c_protein.pdb \
  --ligand-smiles "CN(C)C(=O)Nc1cccc(c1)C2CCN(CC2)C" \
  --output poses.sdf

# View available commands
templ --help
```

### Web Interface

```bash
python run_streamlit_app.py
```

Access the web interface at `http://localhost:8501` for drag-and-drop pose prediction.

## Dataset Setup

TEMPL requires PDBbind dataset for template search:

1. Download PDBbind v2020 from the official website
2. Extract to `templ_pipeline/data/PDBBind/` with structure:
```
PDBBind/
├── PDBbind_v2020_refined/refined-set/<PDB>/
└── PDBbind_v2020_other_PL/v2020-other-PL/<PDB>/
```

## Core Commands

| Command | Description |
|---------|-------------|
| `templ run` | Complete pose prediction pipeline |
| `templ embed` | Generate protein embeddings |
| `templ find-templates` | Search for similar templates |
| `templ generate-poses` | Generate and rank conformers |
| `templ benchmark` | Run validation benchmarks |

## Benchmarking

Reproduce paper results or validate performance:

```bash
# Quick validation
templ benchmark polaris --quick

# Full benchmark
templ benchmark polaris --n-workers 8
templ benchmark time-split --n-workers 4
```

## Requirements

**Minimum:**
- Python 3.9+
- 4GB RAM
- 1GB disk space

**Recommended:**
- 8+ CPU cores
- 16GB+ RAM
- GPU with 4GB+ VRAM (optional)

## Citation

If you use TEMPL in your research, please cite:

```bibtex
@article{templ2024,
  title={TEMPL: A template-based protein ligand pose prediction baseline},
  author={J. Fülöp, M. Šícho, W. Dehaen},
  journal={Journal Name},
  year={2025}
}
```

## License

This project is licensed under the MIT License.

