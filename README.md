# TEMPL Pipeline

Template-based protein–ligand pose prediction with a single command-line interface **and** a Streamlit web app.

---

## Why TEMPL?
TEMPL leverages **ligand similarity** and **template superposition** instead of exhaustive docking or deep neural networks. For familiar chemical space it provides fast, accurate poses with minimal compute.

* Alignment driven by maximal common substructure (MCS)
* Constrained conformer generation (ETKDG v3)
* Shape / pharmacophore scoring for pose selection
* Built-in benchmarks (Polaris, time-split PDBbind)
* CPU-only by default; GPU optional for protein embeddings
* Python 3.9 + ⎯ pure-Python, no compiled C++ extensions beyond RDKit

---

## Installation

### One-Click Setup
After cloning this repository, just run:

```bash
git clone https://github.com/fulopjoz/templ-pipeline
cd templ-pipeline
bash setup_env_uv.sh    # One command - that's it!
```

This single script will:
1. Install `uv` if not already on your system
2. Create `.venv/` virtual environment 
3. Activate environment and install all dependencies
4. Drop you into a ready-to-use shell with TEMPL available

**Returning to the project later:**
```bash
source .venv/bin/activate   # whenever you return to the project
```

### Manual installation (if needed)

#### 1. One-click environment with **uv**
```bash
# Install uv once (10 s)
curl -Ls https://astral.sh/uv/install.sh | bash

# Create & activate virtual environment (adds .venv/)
uv venv .venv
source .venv/bin/activate

# Install in editable mode with all extras (streamlit, tests, docs)
uv pip install -e "./templ_pipeline[all]"
```

#### 2. Classic `pip`
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e "./templ_pipeline[all]"
```
If RDKit wheels are unavailable for your Python version, install via Conda:
```bash
conda create -n templ python=3.10 rdkit -c conda-forge
conda activate templ
pip install -e "./templ_pipeline[all]"
```

---

## Dataset Setup
TEMPL itself ships **no PDB structures or ligands**. Register and download datasets manually, then place them exactly as shown.

### PDBbind v2020
1. Create directory `templ_pipeline/data/PDBBind/` (case exact).
2. Unpack the following archives inside it so the layout becomes:
```
PDBBind/
├─ PDBbind_v2020_refined/refined-set/<PDB>/
└─ PDBbind_v2020_other_PL/v2020-other-PL/<PDB>/
```
TEMPL assumes this layout; if you prefer a different location pass `--data-root` to CLI commands.

### Polaris benchmark
Pre-processed SDFs plus metadata are already included under
`templ_pipeline/benchmark/polaris/data/`. If you deleted them run:
```bash
templ benchmark polaris --download
```
(the command will fetch and unpack ~6 MB of data.)

---

## Quick Start
```bash
# 1 line pose prediction
templ run \
  --protein-file examples/1a1c_protein.pdb \
  --ligand-smiles "CN(C)C(=O)Nc1cccc(c1)C2CCN(CC2)C" \
  --output poses.sdf
```

### Common CLI Commands
| Command | Purpose |
|---------|---------|
| `embed` | Generate or cache protein embeddings (ESM-2) |
| `find-templates` | K-NN template search in PDBbind |
| `generate-poses` | Constrained conformer generation & ranking |
| `run` | One-shot pipeline (`embed → search → pose`) |
| `benchmark` | Reproduce paper benchmarks (Polaris, time-split) |

Use `templ --help` or `templ <command> --help` for all options.

---

## Streamlit Web App
```bash
streamlit run templ_pipeline/ui/app.py
```
* drag-and-drop PDB + SMILES or SDF
* interactive 3-D viewer (py3Dmol)
* download best poses as SDF

---

## Benchmarking Examples
```bash
# Polaris challenge (CPU-only, 8 workers)
templ benchmark polaris --n-workers 8 --n-conformers 200

# PDBbind time-split (ensure PDBBind/ downloaded first)
# To avoid nested parallelism, keep internal workers = 1 when using many benchmark workers.
templ benchmark time-split --n-workers 8 --pipeline-workers 1
```
During long runs TEMPL prints Unicode RMSD tables; see docs on parsing these if integrating.

---

## Development & Tests
```bash
pip install -e ".[dev]"
pytest -q
```

---

## License
MIT – see `LICENSE` file. If you use TEMPL in academic work please cite the accompanying pre-print.
