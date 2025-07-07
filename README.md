# TEMPL Pipeline

Template-based protein–ligand pose prediction with a single command-line interface **and** a Streamlit web app.

---

## Why TEMPL?

> ⚠️ **Method Scope**: TEMPL is a template-based baseline method optimized for rapid pose prediction within known chemical space. Performance may be limited for novel scaffolds, allosteric sites, or targets with insufficient template coverage. Consider validation or combination with physics-based methods.


TEMPL leverages **ligand similarity** and **template superposition** instead of exhaustive docking or deep neural networks. For familiar chemical space within the PDBBind training set, it provides fast, reasonable poses with minimal compute.

**Core Features:**
* Template-based pose prediction using ligand similarity
* Alignment driven by maximal common substructure (MCS)
* Constrained conformer generation (ETKDG v3)
* Shape / pharmacophore scoring for pose selection
* Built-in benchmarks (Polaris, time-split PDBbind)
* CPU-only by default; GPU optional for protein embeddings

**Best suited for:**
- Benchmarking and baseline comparisons
- Rapid pose generation for known chemical space
- Educational demonstrations of template-based methods
- Initial exploration before more sophisticated methods

**Not recommended for:**
- Novel drug discovery with unprecedented scaffolds
- Allosteric or cryptic binding sites
- Production pharmaceutical pipelines without validation

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

# Simple pose prediction
templ run --protein-file protein.pdb --ligand-smiles "CCO"

# Using PDB ID instead of file
templ run --protein-pdb-id 1iky --ligand-smiles "CCO"

# Using SDF file for ligand
templ run --protein-file protein.pdb --ligand-file ligand.sdf

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
| `templ run` | FULL: Complete pipeline (recommended for beginners) |
| `templ embed` | EMBED: Generate protein embeddings |
| `templ find-templates` | SEARCH: Find similar protein templates |
| `templ generate-poses` | GENERATE: Generate ligand poses |

Use `templ --help` for all commands or `templ --help examples` for example usage.

---

## Dataset Setup

TEMPL requires additional data files that are **automatically downloaded** from Zenodo during environment setup. The setup script will handle all data downloads for you!

### Automatic Download (Default)

The setup script automatically downloads required datasets from Zenodo:

```bash
# Data files are downloaded automatically during setup
source setup_templ_env.sh

# Files will be placed in:
# - data/embeddings/templ_protein_embeddings_v1.0.0.npz (~2GB)
# - data/ligands/templ_processed_ligands_v1.0.0.sdf.gz (~800MB)
```

### Manual Download (If Automatic Fails)

If automatic download fails, you can download manually from Zenodo:

1. **Download from ZENODO**: https://doi.org/10.5281/zenodo.15813500
   - `templ_protein_embeddings_v1.0.0.npz` (~2GB) - Pre-computed protein embeddings
   - `templ_processed_ligands_v1.0.0.sdf.gz` (~800MB) - Processed ligand structures

2. **Place files in correct locations**:
   ```bash
   # Create directories (if not exist)
   mkdir -p data/embeddings data/ligands
   
   # Move downloaded files
   mv templ_protein_embeddings_v1.0.0.npz data/embeddings/
   mv templ_processed_ligands_v1.0.0.sdf.gz data/ligands/
   ```

> **Note**: The setup script now automatically installs `zenodo_get` and downloads these files for you during installation.

### Using zenodo-get directly

You can also use zenodo-get directly to download the datasets:

```bash
# Install zenodo-get if not already installed
pip install zenodo-get

# Download TEMPL datasets
zenodo_get 10.5281/zenodo.15813500

# Move files to correct locations
mv templ_protein_embeddings_v1.0.0.npz data/embeddings/
mv templ_processed_ligands_v1.0.0.sdf.gz data/ligands/
```

### Option 3: PDBbind v2020 (For Full Dataset)

For complete benchmarking, also download PDBbind:

1. Register and download from [PDBbind website](https://www.pdbbind-plus.org.cn/download)
2. Place in `templ_pipeline/data/PDBBind/` with this structure:
```
PDBBind/
├─ PDBbind_v2020_refined/refined-set/<PDB>/
└─ PDBbind_v2020_other_PL/v2020-other-PL/<PDB>/
```

### Verify Installation

Check that all required files are present:

```bash
# Check essential files
templ --version                                              # Should show TEMPL version
ls data/embeddings/templ_protein_embeddings_v1.0.0.npz     # Should exist (~2GB)
ls data/ligands/templ_processed_ligands_v1.0.0.sdf.gz      # Should exist (~800MB)

# Test with a quick benchmark
templ benchmark polaris --quick
```

> **Note:** The data files (embeddings and ligands) are essential for TEMPL to work. If you encounter download issues during setup, you can download them manually from [Zenodo](https://doi.org/10.5281/zenodo.15813500) or open an issue on GitHub.

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

### Data File Issues
If you get errors about missing embeddings or ligand files:

```bash
# Check if data files exist
ls data/embeddings/templ_protein_embeddings_v1.0.0.npz
ls data/ligands/templ_processed_ligands_v1.0.0.sdf.gz

# If missing, re-run setup to download automatically
source setup_templ_env.sh

# Or download manually from Zenodo (see Dataset Setup section)
```

### GPU Detection Issues
If you have a GPU but it's not detected:

```bash
# Check GPU status
nvidia-smi

# Force GPU installation if auto-detection fails
source setup_templ_env.sh --gpu-force

# Use CPU-only if GPU issues persist
source setup_templ_env.sh --cpu-only
```

### Pose Alignment Issues  
If poses appear misaligned relative to protein binding sites:

- Ensure you have the PDBBind dataset for optimal template alignment
- Use PDB IDs from the database when possible (better than local files)
- Check that protein superalignment is enabled in your pipeline runs

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

