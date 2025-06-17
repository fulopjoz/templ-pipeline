# TEMPL Pipeline Web Interface

Simple, one-click web application for template-based protein ligand pose prediction.

## Quick Start

```bash
# From project root
python run_streamlit_app.py
```

## Features

- ✅ One-click workflow: Input → Predict → Results
- ✅ Direct core module integration
- ✅ Simple, clean interface
- ✅ SMILES and PDB input validation
- ✅ Pose visualization and SDF download

## Usage

1. Enter molecule SMILES (e.g., "CCO")
2. Enter PDB ID (e.g., "1a1e") or upload PDB file
3. Click "PREDICT POSES"
4. View results and download SDF file

## Architecture

Single-file application (`app.py`) that uses TEMPL core modules directly:
- `templ_pipeline.core.embedding` - Protein embeddings and template search
- `templ_pipeline.core.mcs` - Conformer generation and MCS alignment
- `templ_pipeline.core.scoring` - Pose scoring and selection 