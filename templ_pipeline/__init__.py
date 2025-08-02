"""
TEMPL Pipeline: Template-based Protein-Ligand pose prediction

TEMPL is a template-based approach for molecular pose prediction that uses:
1. Protein embeddings for finding similar binding pockets
2. Maximum common substructure (MCS) for ligand alignment
3. Shape and color scoring for pose selection

Main components:
- core: Core functionality (embedding, MCS, scoring)
- cli: Command-line interface
- benchmark: Benchmarking tools
- ui: Streamlit web application

Author: Cursor AI, 2025
"""

import importlib
from typing import Any

# Version information
__version__ = "1.0.0"

# Suppress RDKit warnings globally (including SCD/SED warnings)
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    RDLogger.DisableLog("rdkit.*")
except ImportError:
    # RDKit not available - warnings will be handled when RDKit is imported
    pass

__all__ = [
    # Package info
    "__version__",
    # Core functionality
    "EmbeddingManager",
    "select_templates",
    "get_protein_embedding",
    "find_mcs",
    "constrained_embed",
    "score_and_align",
    "select_best",
    "rmsd_raw",
    "DatasetSplits",
]

# Lazy loading for core functionality
_CORE_IMPORTS = {
    "EmbeddingManager",
    "select_templates",
    "get_protein_embedding",
    "find_mcs",
    "constrained_embed",
    "score_and_align",
    "select_best",
    "rmsd_raw",
    "DatasetSplits",
}


def __getattr__(name: str) -> Any:
    """Lazy loading of core functionality."""
    if name in _CORE_IMPORTS:
        from . import core

        return getattr(core, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
