# TEMPL Pipeline Project Brief

## Project Overview
TEMPL Pipeline is a template-based protein-ligand pose prediction tool that provides both CLI and Streamlit web interfaces.

## Core Capabilities
- Template-based ligand pose prediction
- Ligand similarity and template superposition
- Constrained conformer generation (ETKDG v3)
- Shape/pharmacophore scoring
- Built-in benchmarks (Polaris, time-split PDBbind)
- CPU-only by default, optional GPU support

## Technical Stack
- **Language**: Python 3.9+
- **Core Libraries**: RDKit (cheminformatics), ESM-2 (protein embeddings)
- **Web Framework**: Streamlit
- **CLI Framework**: Click
- **Environment**: Virtual environment (.templ)

## Project Structure
- `templ_pipeline/` - Main package directory
  - `core/` - Core functionality
  - `cli/` - Command-line interface
  - `ui/` - Web interface (Streamlit)
  - `fair/` - FAIR integration
  - `benchmark/` - Benchmarking tools

## Key Commands
- `templ run` - One-shot pose prediction
- `templ embed` - Generate protein embeddings
- `templ find-templates` - K-NN template search
- `templ generate-poses` - Conformer generation & ranking
- `templ benchmark` - Run benchmarks

## Current State
- Project is functional with CLI and web interfaces
- Awaiting specific task/enhancement requirements
