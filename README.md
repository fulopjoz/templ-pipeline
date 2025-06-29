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

---

## Quick Installation

**Just run one command and you're ready to go:**

```bash
git clone https://github.com/fulopjoz/templ-pipeline
cd templ-pipeline
source setup_templ_env.sh
```

**That's it!** The script will:
- Detect your hardware (CPU cores, RAM, GPU)
- Install optimal dependencies for your system
- Create and activate the `.templ` environment  
- Verify everything works
- **Leave you ready to use `templ` immediately**

### Installation Options

```bash
# Default: Auto-detect and install optimally (recommended)
source setup_templ_env.sh

# Force lightweight CPU-only installation (~50MB)
source setup_templ_env.sh --cpu-only

# Force GPU installation (if auto-detection fails)
source setup_templ_env.sh --gpu-force

# Minimal server installation (no web interface)
source setup_templ_env.sh --minimal
```

### Using TEMPL Later

The installation creates a `.templ` environment. For future sessions:

```bash
# Activate the environment
source .templ/bin/activate
```

Once activated, just use `templ` commands directly!

---

## Quick Start

**After installation, you're immediately ready to use TEMPL:**

```bash
# Make sure you're in the TEMPL environment (should show (.templ) in prompt)
# If not, activate it: source .templ/bin/activate

# 1-line pose prediction
templ run \
  --protein-file examples/1a1c_protein.pdb \
  --ligand-smiles "CN(C)C(=O)Nc1cccc(c1)C2CCN(CC2)C" \
  --output poses.sdf

# Show all available commands  
templ --help

# Web interface
python run_streamlit_app.py
```

### Common CLI Commands
| Command | Purpose |
|---------|---------|
| `templ run` | One-shot pose prediction |
| `templ embed` | Generate protein embeddings (ESM-2) |
| `templ find-templates` | K-NN template search in PDBbind |
| `templ generate-poses` | Constrained conformer generation & ranking |
| `templ benchmark` | Reproduce paper benchmarks |

Use `templ <command> --help` for detailed options.

---

## Dataset Setup

TEMPL ships **no PDB structures or ligands**. Download datasets manually:

### PDBbind v2020
1. Register and download from PDBbind website
2. Place in `templ_pipeline/data/PDBBind/` with this structure:
```
PDBBind/
├─ PDBbind_v2020_refined/refined-set/<PDB>/
└─ PDBbind_v2020_other_PL/v2020-other-PL/<PDB>/
```

### Polaris benchmark
Pre-processed data is already included under `templ_pipeline/benchmark/data/polaris/`.

---

## Web Interface

Start the Streamlit app:
```python
python run_streamlit_app.py
```
* Drag-and-drop PDB + SMILES or SDF
* Download best poses as SDF

---

## Benchmarking Examples

```bash
# Polaris challenge (CPU-only, 8 workers)
templ benchmark polaris --n-workers 8 --n-conformers 200

# PDBbind time-split (ensure PDBBind/ downloaded first)
templ benchmark time-split --n-workers 8 --pipeline-workers 1
```

---

## Troubleshooting

### "Command not found" Error
If you get `templ: command not found`, make sure you're in the TEMPL environment:

```bash
# Check if environment is active (should show (.templ) in prompt)
# If not active, run:
source .templ/bin/activate

# If no environment exists, run setup:
source setup_templ_env.sh
```

### Environment Not Activating
Make sure to use `source` (not `./`) when running the setup:

```bash
# ✅ Correct - creates and activates environment
source setup_templ_env.sh

# ❌ Wrong - creates environment but doesn't activate it
./setup_templ_env.sh
```

### Get Help
```bash
templ --help getting-started    # Setup and basic usage
templ --help troubleshooting    # Common issues and solutions
templ --help examples           # Copy-paste examples
```

---

## Development & Testing

```bash
# Development installation
source setup_templ_env.sh --gpu-force  # or --cpu-only
pip install -e ".[dev]"                # Add dev dependencies
pytest -q                              # Run tests
```

---

## Hardware Requirements

**Minimum:**
- Python 3.9+
- 4GB RAM
- 1GB disk space

**Recommended:**
- 8+ CPU cores
- 16GB+ RAM  
- GPU with 4GB+ VRAM (optional, for faster embeddings)

The installer automatically detects your hardware and installs the optimal configuration!

---

