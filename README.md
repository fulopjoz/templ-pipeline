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
- Install `uv` for fast package management
- Create and activate the `.templ` virtual environment
- Install optimal dependencies for your system using `uv`
- Verify everything works
- **Leave you ready to use `templ` immediately**

> **Important:** Always use `source setup_templ_env.sh` (not manual `pip` commands) for the best experience.

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
# The setup script automatically activates the environment
# You should see (.templ) in your prompt

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

**For future sessions:**

```bash
# Activate the TEMPL environment
source .templ/bin/activate

# Then use templ commands as normal
templ --help
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

## Benchmarking

TEMPL includes comprehensive benchmarking capabilities for reproducing paper results and evaluating performance.

### Quick Benchmarks

```bash
# Quick Polaris benchmark (reduced dataset for testing)
templ benchmark polaris --quick

# Quick timesplit benchmark (test set only, 5 PDBs max)
templ benchmark time-split --test-only --max-pdbs 5
```

### Full Benchmarks

```bash
# Complete Polaris benchmark with hardware optimization
templ benchmark polaris --n-workers 8 --hardware-profile balanced

# Complete timesplit benchmark (requires PDBBind dataset)
templ benchmark time-split --n-workers 4 --max-ram 16.0
```

### Hardware Optimization

```bash
# Conservative profile (safe for shared systems)
templ benchmark polaris --hardware-profile conservative

# Aggressive profile (maximum performance)
templ benchmark time-split --hardware-profile aggressive --enable-hyperthreading

# Custom resource limits
templ benchmark polaris --cpu-limit 8 --memory-limit 12.0
```

### Benchmark Options

| Option | Description |
|--------|-------------|
| `--quick` | Run reduced dataset for quick validation |
| `--hardware-profile` | `conservative`, `balanced`, `aggressive`, or `auto` |
| `--n-workers` | Number of parallel workers |
| `--max-ram` | Memory limit in GB (timesplit only) |
| `--test-only` | Evaluate test set only (timesplit only) |
| `--max-pdbs` | Limit number of PDBs for testing |

Results are saved to structured workspace directories with CSV, JSON, and Markdown summaries.

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

### Wrong Installation Method
If you manually ran `pip install` instead of using the setup script:

```bash
# Wrong - using pip directly
pip install -e ".[dev]"

# Correct - use setup script  
source setup_templ_env.sh --dev

# OR manually with proper environment and uv
source .templ/bin/activate
uv pip install -e ".[dev]"
```

### Environment Not Activating
Make sure to use `source` (not `./`) when running the setup:

```bash
# Correct - creates and activates environment
source setup_templ_env.sh

# Wrong - creates environment but doesn't activate it
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

**Recommended approach - use the setup script:**

```bash
# Development installation (creates environment + installs dev dependencies)
source setup_templ_env.sh --dev

# Run tests
pytest -q
```

**Manual approach (only if needed):**

```bash
# Only if you need to manually add dev dependencies to existing environment
source .templ/bin/activate              # Activate TEMPL environment first
uv pip install -e ".[dev]"             # Add dev dependencies
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

