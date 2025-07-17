# TEMPL Pipeline Web Interface

Simple, one-click web application for template-based protein ligand pose prediction.

## Quick Start

### Standard Launch
```bash
# From project root
python run_streamlit_app.py
```

### Troubleshooting White Page Issues
If you experience a white page when launching:

```bash
# Step 1: Run diagnostic checks
python run_streamlit_app.py --diagnostic

# Step 2: Use robust mode (recommended)
python run_streamlit_app.py
# Select option 2 (robust) or press Enter
```

See [TEMPL_TROUBLESHOOTING.md](../../../TEMPL_TROUBLESHOOTING.md) for detailed troubleshooting guide.

## Features

- One-click workflow: Input → Predict → Results
- Direct core module integration
- Progressive loading with immediate feedback
- Comprehensive error handling and fallback modes
- Simple, clean interface
- SMILES and PDB input validation
- Pose visualization and SDF download

## Usage

1. Launch the application using one of the available modes
2. Enter molecule SMILES (e.g., "CCO")
3. Enter PDB ID (e.g., "1a1e") or upload PDB file
4. Click "PREDICT POSES"
5. View results and download SDF file

## Available Launch Modes

### 1. Diagnostic Mode
```bash
python run_streamlit_app.py --diagnostic
```
- Runs comprehensive system checks
- Identifies missing dependencies
- Tests component loading
- Provides detailed error information

### 2. Robust Mode (Recommended)
```bash
python run_streamlit_app.py
# Select option 2 or press Enter
```
- Progressive loading with immediate feedback
- Graceful degradation on component failures
- Fallback UI if main components fail
- Better error handling

### 3. Main Mode
```bash
python run_streamlit_app.py
# Select option 3
```
- Full TEMPL Pipeline application
- Use only after diagnostic checks pass

## Common Issues

### RDKit Missing
Most common cause of white pages:
```bash
# Install RDKit
conda install -c conda-forge rdkit
# OR
pip install rdkit
```

### Hardware Detection Hangs
Use robust mode - it handles timeouts automatically.

### Import Errors
```bash
# Install missing dependencies
pip install numpy pandas biopython biotite streamlit
```

## Architecture

The application uses a modular architecture with:
- Progressive component loading
- Comprehensive error handling
- Fallback mechanisms
- Non-blocking initialization

Core modules:
- `templ_pipeline.core.embedding` - Protein embeddings and template search
- `templ_pipeline.core.mcs` - Conformer generation and MCS alignment
- `templ_pipeline.core.scoring` - Pose scoring and selection

## Getting Help

If issues persist:
1. Run diagnostic mode and save the output
2. Check the troubleshooting guide: [TEMPL_TROUBLESHOOTING.md](../../../TEMPL_TROUBLESHOOTING.md)
3. Verify all dependencies are installed
4. Check system resources (4GB+ RAM recommended) 