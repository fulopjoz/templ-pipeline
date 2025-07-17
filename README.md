# TEMPL Pipeline

Template-based protein–ligand pose prediction with command-line interface and web application.

---

## Overview

TEMPL is a template-based method for rapid protein–ligand pose prediction that leverages ligand similarity and template superposition. The method uses maximal common substructure (MCS) alignment and constrained conformer generation (ETKDG v3) for pose generation within known chemical space.

**Key Features:**
- Template-based pose prediction using ligand similarity
- Alignment driven by maximal common substructure (MCS)
- Shape and pharmacophore scoring for pose selection
- Built-in benchmarks (Polaris, time-split PDBbind)
- CPU-only by default with optional GPU acceleration

**⚠️ Scope:** 
* Optimized for rapid pose prediction within known chemical space. Performance may be limited for novel scaffolds, allosteric sites, or targets with insufficient template coverage.

---

## Installation

### Standard Installation
```bash
git clone https://github.com/fulopjoz/templ-pipeline
cd templ-pipeline
source setup_templ_env.sh
```

### Development Installation
```bash
source setup_templ_env.sh --dev
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

TEMPL requires pre-computed embeddings and ligand structures that are automatically downloaded during setup from [Zenodo](https://doi.org/10.5281/zenodo.15813500):

- `templ_protein_embeddings_v1.0.0.npz` (~90MB) - Protein embeddings
- `templ_processed_ligands_v1.0.0.sdf.gz` (~10MB) - Processed ligand structures

### PDBbind Dataset

For benchmarking, download the following **freely available v2020** subsets from the [official PDBbind website](https://www.pdbbind-plus.org.cn/download):

1. **Protein-ligand complexes: The general set minus refined set** (1.8 GB)
   → `PDBbind_v2020_other_PL`

2. **Protein-ligand complexes: The refined set** (658 MB)
   → `PDBbind_v2020_refined`

After downloading, extract both folders into `data/PDBBind/` using the standard directory structure.

---

## Usage

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
```bash
python run_streamlit_app.py
```

### Benchmarking
```bash
# Full benchmark
templ benchmark polaris
templ benchmark time-split # PDBBind dataset

# Partial benchmark
templ benchmark time-split --test-only
```

---

## Pipeline Commands

| Command | Description |
|---------|-------------|
| `templ run` | Complete pipeline (recommended) |
| `templ embed` | Generate protein embeddings |
| `templ find-templates` | Find similar protein templates |
| `templ generate-poses` | Generate ligand poses |

---

## System Requirements

**Minimum:**
- Python 3.9+
- 4GB RAM
- 1GB disk space

**Recommended:**
- 8+ CPU cores
- 16GB+ RAM
- GPU with 4GB+ VRAM (optional)

---

## Citation

If you use TEMPL in your research, please cite:

```bibtex
@article{templ2025,
  title={TEMPL: Template-based Protein-Ligand Pose Prediction},
  author={},
  journal={},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

