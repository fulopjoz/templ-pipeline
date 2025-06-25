# Memory Bank: Project Brief

## Project Overview
**TEMPL Pipeline** - Template-based protein–ligand pose prediction system with CLI and web interface

### Project Purpose
- Fast, accurate protein-ligand pose prediction using template-based methods
- Leverages ligand similarity and template superposition instead of exhaustive docking
- Provides both command-line interface and Streamlit web application
- Focuses on familiar chemical space with minimal computational requirements

### Current Architecture
- **CLI Interface**: Command-line tools for pose prediction, embedding, template search
- **Web Interface**: Streamlit-based web application for drag-and-drop functionality
- **Core Engine**: MCS-based alignment, constrained conformer generation, shape/pharmacophore scoring
- **Benchmarking**: Built-in Polaris and time-split PDBbind benchmarks
- **Hardware Flexibility**: CPU-only by default, optional GPU for protein embeddings

### Key Components
1. **CLI Commands**: `templ run`, `templ embed`, `templ find-templates`, `templ generate-poses`, `templ benchmark`
2. **Web App**: `run_streamlit_app.py` - Interactive interface
3. **Core Pipeline**: Template search, pose generation, scoring, output management
4. **Data Management**: PDBbind integration, Polaris benchmark data
5. **Setup System**: Smart installation with hardware detection

### Current User Pain Points
- CLI help system could be more user-friendly
- Output verbosity needs reduction for better UX
- File naming lacks important identifiers (e.g., PDBID)
- FAIR principles not implemented across the system
- UX/UI could follow better design practices

### Success Criteria for Enhancement
- Intuitive CLI interface following UX best practices
- Clear, understandable help documentation
- Appropriately verbose output with user control
- Standardized file naming with proper identifiers
- Full FAIR compliance (Findability, Accessibility, Interoperability, Reusability)
