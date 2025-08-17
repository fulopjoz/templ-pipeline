# Changelog

All notable changes to TEMPL Pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- FAIR compliance improvements
- SPDX headers to all Python files
- Citation.cff file for academic citation
- CodeMeta.json for machine-readable metadata
- AUTHORS file for contributor attribution
- CONTRIBUTING.md guidelines
- CODE_OF_CONDUCT.md
- LICENSES directory with SPDX-named files

## [1.0.0] - 2025-01-XX

### Added
- Initial release of TEMPL Pipeline
- Template-based protein-ligand pose prediction
- MCS alignment and ETKDG conformer generation
- Command-line interface with smart progressive UX
- Streamlit web application
- Built-in benchmarks (Polaris, time-split PDBbind)
- Protein embedding generation and template selection
- Shape and pharmacophore scoring
- Comprehensive workspace and file management
- Hardware detection and optimization
- Multi-processing support for pose generation
- Error handling and validation
- Documentation and examples

### Technical Features
- ESM2 protein embeddings
- RDKit-based molecular processing
- Parallel conformer generation
- Caching and optimization
- Cross-platform compatibility
- Docker and Kubernetes deployment support

### Data
- Pre-computed protein embeddings (~90MB)
- Processed ligand structures (~10MB)
- Benchmark datasets and splits
- Example protein-ligand complexes

## [0.9.0] - 2025-06-17

### Added
- Beta release with core functionality
- Basic pipeline implementation
- Initial CLI interface
- Web application prototype
- One-command installation script
- Git LFS tracking for large files

## [0.8.0] - 2025-06-17

### Added
- Alpha release
- Core MCS functionality
- Basic scoring algorithms
- Initial documentation
- First commit with project structure
