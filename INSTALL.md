# TEMPL Pipeline - Installation Guide

## Quick Start (2 options)

### Standard Installation (Recommended)
For most users who want the web interface and core functionality:

```bash
git clone <repository>
cd templ_pipeline
pip install -e ".[web]"
```

### Full Installation (Advanced Features)
For power users who want AI/ML features:

```bash
git clone <repository>
cd templ_pipeline
pip install -e ".[full]"
```

## Usage Options

### Web Interface
Launch the Streamlit web application:
```bash
python run_streamlit_app.py
```

The app will automatically:
- Check your dependencies
- Display all available URLs (Local, Network, External)
- Configure optimal settings
- Launch in your browser

### Command Line Interface
Use the templ CLI for programmatic access:

```bash
# Get help and see all available commands
templ --help

# One-shot pose prediction
templ run \
  --protein-file examples/1a1c_protein.pdb \
  --ligand-smiles "CN(C)C(=O)Nc1cccc(c1)C2CCN(CC2)C" \
  --output poses.sdf

# Generate protein embeddings
templ embed --protein-file protein.pdb --output embeddings.npz

# Find similar templates
templ find-templates --query-protein protein.pdb --top-k 10

# Run benchmarks
templ benchmark polaris --n-workers 8
```

### Available CLI Commands
| Command | Purpose |
|---------|---------|
| `templ run` | One-shot pose prediction |
| `templ embed` | Generate protein embeddings (ESM-2) |
| `templ find-templates` | K-NN template search in PDBbind |
| `templ generate-poses` | Constrained conformer generation & ranking |
| `templ benchmark` | Reproduce paper benchmarks |

Use `templ <command> --help` for detailed options on each command.

## What You Get

**Standard Installation ([web]):**
- Core pipeline functionality
- Streamlit web interface
- 3D molecular visualization
- CLI tools
- Lightweight (~40 dependencies)

**Full Installation ([full]):**
- Everything from Standard
- AI/ML embedding features
- GPU acceleration (if available)
- Advanced similarity search
- ~80 dependencies with auto-CUDA detection

## Troubleshooting

**Missing dependencies?**
The launcher will tell you exactly what to install.

**Command not found: templ**
Make sure you installed the package:
```bash
pip install -e ".[web]"  # or .[full]
```

**GPU not detected?**
Make sure you have CUDA drivers installed and use the "full" installation.

**Port conflicts?**
Set `PORT=8502` or any available port before running the web app.

## Development

For contributors:
```bash
pip install -e ".[dev]"
pytest -q  # Run tests
```
