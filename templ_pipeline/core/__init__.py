"""
TEMPL Pipeline Core Module

This package contains the core functionality for the TEMPL pipeline:
- Embedding generation and template selection
- MCS identification and conformer generation
- Scoring and pose selection

Import key functions directly from this module for easier access.
Uses lazy loading to avoid slow imports during help display.
"""

import importlib
from typing import Any

# Define what should be available for import
__all__ = [
    # Embedding
    "get_protein_sequence",
    "get_protein_embedding",
    "initialize_esm_model",
    "calculate_embedding",
    "EmbeddingManager",
    "select_templates",
    # MCS
    "find_mcs",
    "generate_conformers",
    "constrained_embed",
    "mmff_minimise_fixed_parallel",
    "mmff_minimise_fixed_sequential",
    "safe_name",
    "get_central_atom",
    "central_atom_embed",
    # Scoring
    "score_and_align",
    "select_best",
    "rmsd_raw",
    "generate_properties_for_sdf",
    "FixedMolecularProcessor",
    "ScoringFixer",
    "CoordinateMapper",
    # Datasets
    "DatasetSplits",
    # Diagnostics
    "PipelineErrorTracker",
    "ProteinAlignmentTracker",
    # Chemistry
    "detect_and_substitute_organometallic",
    "needs_uff_fallback",
    "has_rhenium_complex",
    "is_large_peptide",
    "validate_target_molecule",
    # Templates
    "filter_templates_by_ca_rmsd",
    "get_templates_with_progressive_fallback",
    "find_best_ca_rmsd_template",
    "load_uniprot_exclude",
    "get_uniprot_mapping",
    "load_pdb_filter",
    "standardize_atom_arrays",
    "DEFAULT_CA_RMSD_FALLBACK_THRESHOLDS",
    # Utils
    "find_pocket_chains",
    "find_pdbbind_paths",
]

# Module mapping for lazy loading
_MODULE_MAP = {
    # Embedding functions
    "get_protein_sequence": "embedding",
    "get_protein_embedding": "embedding",
    "initialize_esm_model": "embedding",
    "calculate_embedding": "embedding",
    "EmbeddingManager": "embedding",
    "select_templates": "embedding",
    # MCS functions
    "find_mcs": "mcs",
    "generate_conformers": "mcs",
    "constrained_embed": "mcs",
    "mmff_minimise_fixed_parallel": "mcs",
    "mmff_minimise_fixed_sequential": "mcs",
    "safe_name": "mcs",
    "get_central_atom": "mcs",
    "central_atom_embed": "mcs",
    # Scoring functions
    "score_and_align": "scoring",
    "select_best": "scoring",
    "rmsd_raw": "scoring",
    "generate_properties_for_sdf": "scoring",
    "FixedMolecularProcessor": "scoring",
    "ScoringFixer": "scoring",
    "CoordinateMapper": "scoring",
    # Diagnostics
    "PipelineErrorTracker": "diagnostics",
    "ProteinAlignmentTracker": "diagnostics",
    # Chemistry
    "detect_and_substitute_organometallic": "chemistry",
    "needs_uff_fallback": "chemistry",
    "has_rhenium_complex": "chemistry",
    "is_large_peptide": "chemistry",
    "validate_target_molecule": "chemistry",
    # Templates
    "filter_templates_by_ca_rmsd": "templates",
    "get_templates_with_progressive_fallback": "templates",
    "find_best_ca_rmsd_template": "templates",
    "load_uniprot_exclude": "templates",
    "get_uniprot_mapping": "templates",
    "load_pdb_filter": "templates",
    "standardize_atom_arrays": "templates",
    "DEFAULT_CA_RMSD_FALLBACK_THRESHOLDS": "templates",
    # Utils
    "find_pocket_chains": "utils",
    "find_pdbbind_paths": "utils",
    # Datasets
    "DatasetSplits": "datasets",
}

# Cache for loaded modules
_loaded_modules = {}


def __getattr__(name: str) -> Any:
    """Lazy loading of core functions and classes."""
    if name in _MODULE_MAP:
        module_name = _MODULE_MAP[name]

        # Load module if not cached
        if module_name not in _loaded_modules:
            full_module_name = f"templ_pipeline.core.{module_name}"
            _loaded_modules[module_name] = importlib.import_module(full_module_name)

        module = _loaded_modules[module_name]
        return getattr(module, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
