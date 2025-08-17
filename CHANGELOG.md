# Changelog

All notable changes to TEMPL Pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-08-17

### Added
- Initial public release of TEMPL Pipeline
- Template-based protein-ligand pose prediction system
- Maximum Common Substructure (MCS) alignment with ETKDG conformer generation
- Command-line interface with progressive user experience
- Streamlit web application for interactive use
- Built-in benchmarking capabilities (Polaris, time-split PDBbind)
- Protein embedding generation and template selection using ESM2
- Shape and pharmacophore-based scoring algorithms
- Comprehensive workspace and file management system
- Hardware detection and optimization features
- Multi-processing support for parallel pose generation
- Robust error handling and input validation
- Docker and Kubernetes deployment configurations
- FAIR compliance features including:
  - SPDX headers in all Python files
  - Citation.cff file for academic citation
  - CodeMeta.json for machine-readable metadata
  - CONTRIBUTING.md guidelines
  - CODE_OF_CONDUCT.md
  - MIT license with proper SPDX formatting

### Technical Features
- ESM2 protein embeddings for template selection
- RDKit-based molecular processing and manipulation
- Parallel conformer generation algorithms
- Intelligent caching and performance optimization
- Cross-platform compatibility (Linux, macOS, Windows)
- Memory-efficient processing of large datasets

### Data
- Pre-computed protein embeddings database (~90MB)
- Processed ligand structure library (~10MB)
- Benchmark datasets and validation splits
- Example protein-ligand complexes for testing
