# TEMPL Pipeline - Project Brief

## Project Overview
Template-based protein–ligand pose prediction system with both CLI and web interface capabilities.

## Core Mission
TEMPL leverages **ligand similarity** and **template superposition** instead of exhaustive docking or deep neural networks. For familiar chemical space it provides fast, accurate poses with minimal compute.

## Key Features
- Alignment driven by maximal common substructure (MCS)
- Constrained conformer generation (ETKDG v3) 
- Shape / pharmacophore scoring for pose selection
- Built-in benchmarks (Polaris, time-split PDBbind)
- CPU-only by default; GPU optional for protein embeddings
- Streamlit web interface
- Single command-line interface

## Target Users
- Computational chemists
- Bioinformatics researchers
- Drug discovery teams
- Academic researchers in structural biology

## Technical Architecture
- Python 3.9+ with scientific computing stack
- RDKit for chemical informatics
- Biotite/BioPython for protein handling
- Optional GPU acceleration for embeddings
- Web interface via Streamlit

## Current Status
- Active development project
- Version 0.1.0
- Full CLI and web interface implemented
- Benchmarking capabilities included
- Hardware-adaptive installation system

## Hardware Support
- **Minimum:** Python 3.9+, 4GB RAM, 1GB disk
- **Recommended:** 8+ cores, 16GB+ RAM, GPU with 4GB+ VRAM

## Repository Structure
- Main pipeline code in `templ_pipeline/`
- Examples and benchmark data included
- Docker support available
- Comprehensive testing framework

## Installation Model
Single-command setup with hardware detection and optimal dependency installation.

## Date Created
2024 (Active)

## Current Focus Areas
- Performance optimization
- Benchmarking accuracy
- User experience improvements
- Hardware compatibility
