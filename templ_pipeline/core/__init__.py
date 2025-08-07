"""
TEMPL Pipeline Core Module

This package contains the core functionality for the TEMPL pipeline:
- Embedding generation and template selection
- MCS identification and conformer generation
- Scoring and pose selection
- Comprehensive workspace and file management
- Execution monitoring and error handling
- Hardware detection and optimization
- Data validation and processing

Import key functions directly from this module for easier access.
Uses lazy loading to avoid slow imports during help display.
"""

import importlib
from typing import Any, TYPE_CHECKING

# Type checking imports for better IDE support
if TYPE_CHECKING:
    from .embedding import (
        get_protein_sequence,
        get_protein_embedding,
        initialize_esm_model,
        calculate_embedding,
        calculate_embedding_single,
        EmbeddingManager,
        select_templates,
    )
    from .mcs import (
        find_mcs,
        find_best_ca_rmsd_template,
        constrained_embed,
        central_atom_embed,
        safe_name,
        get_central_atom,
        needs_uff_fallback,
        embed_with_uff_fallback,
        validate_molecular_connectivity,
        validate_molecular_geometry,
        simple_minimize_molecule,
        validate_mmff_parameters,
        log_coordinate_map,
        relax_close_constraints,
    )
    from .scoring import (
        score_and_align,
        select_best,
        rmsd_raw,
        generate_properties_for_sdf,
        FixedMolecularProcessor,
        ScoringFixer,
        CoordinateMapper,
    )
    from .chemistry import (
        detect_and_substitute_organometallic,
        has_rhenium_complex,
        is_large_peptide,
        validate_target_molecule,
    )
    from .templates import (
        load_reference_protein,
        load_target_data,
        transform_ligand,
        filter_templates_by_ca_rmsd,
        get_templates_with_progressive_fallback,
        pdb_path,
        ligand_path,
    )
    from .pipeline import (
        TEMPLPipeline,
        PipelineConfig,
        run_pipeline,
        run_pipeline_from_args,
        run_from_pdb_and_smiles,
    )
    from .workspace_manager import (
        WorkspaceManager,
        WorkspaceConfig,
        DirectoryManager,
        TempDirectoryManager,
        create_workspace_manager,
        emergency_cleanup,
    )
    from .file_manager import (
        FileManager,
        AdaptiveFileNamingEngine,
        PredictionContext,
        FileMetadata,
        create_file_manager,
        generate_adaptive_filename,
    )
    from .execution_manager import (
        ExecutionManager,
        SkipReason,
        MoleculeSkipException,
        get_safe_worker_count,
        skip_molecule,
        record_successful_processing,
        get_execution_summary,
    )
    from .hardware import (
        HardwareInfo,
        get_basic_hardware_info,
        get_hardware_info,
        get_optimized_worker_config,
        get_suggested_worker_config,
        detect_optimal_configuration,
    )
    from .data import (
        DatasetSplits,
        DatasetManager,
        load_benchmark_pdbs,
        validate_dataset_integrity,
    )
    from .validation import (
        SplitDataValidator,
        DatabaseValidator,
        MolecularValidationFramework,
        validate_pipeline_components,
        quick_validation_check,
    )


# Define what should be available for import
__all__ = [
    # Embedding
    "get_protein_sequence",
    "get_protein_embedding", 
    "initialize_esm_model",
    "calculate_embedding",
    "calculate_embedding_single",
    "EmbeddingManager",
    "select_templates",
    # MCS
    "find_mcs",
    "find_best_ca_rmsd_template",
    "constrained_embed",
    "central_atom_embed",
    "safe_name",
    "get_central_atom",
    "needs_uff_fallback",
    "embed_with_uff_fallback",
    "validate_molecular_connectivity",
    "validate_molecular_geometry",
    "simple_minimize_molecule",
    "validate_mmff_parameters",
    "log_coordinate_map",
    "relax_close_constraints",
    # Scoring
    "score_and_align",
    "select_best",
    "rmsd_raw",
    "generate_properties_for_sdf",
    "FixedMolecularProcessor",
    "ScoringFixer",
    "CoordinateMapper",
    # Chemistry
    "detect_and_substitute_organometallic",
    "has_rhenium_complex",
    "is_large_peptide",
    "validate_target_molecule",
    # Templates
    "load_reference_protein",
    "load_target_data",
    "transform_ligand", 
    "filter_templates_by_ca_rmsd",
    "get_templates_with_progressive_fallback",
    "pdb_path",
    "ligand_path",
    # Pipeline
    "TEMPLPipeline",
    "PipelineConfig",
    "run_pipeline",
    "run_pipeline_from_args",
    "run_from_pdb_and_smiles",
    # Workspace Management
    "WorkspaceManager",
    "WorkspaceConfig",
    "DirectoryManager",
    "TempDirectoryManager",
    "create_workspace_manager",
    "emergency_cleanup",
    # File Management
    "FileManager",
    "AdaptiveFileNamingEngine",
    "PredictionContext",
    "FileMetadata",
    "create_file_manager",
    "generate_adaptive_filename",
    # Execution Management
    "ExecutionManager",
    "SkipReason",
    "MoleculeSkipException",
    "get_safe_worker_count",
    "skip_molecule",
    "record_successful_processing",
    "get_execution_summary",
    # Hardware Detection
    "HardwareInfo",
    "get_basic_hardware_info",
    "get_hardware_info",
    "get_optimized_worker_config",
    "get_suggested_worker_config",
    "detect_optimal_configuration",
    # Data Management
    "DatasetSplits",
    "DatasetManager",
    "load_benchmark_pdbs",
    "validate_dataset_integrity",
    # Validation
    "SplitDataValidator",
    "DatabaseValidator",
    "MolecularValidationFramework",
    "validate_pipeline_components",
    "quick_validation_check",
]

# Module mapping for lazy loading
_MODULE_MAP = {
    # Embedding functions
    "get_protein_sequence": "embedding",
    "get_protein_embedding": "embedding",
    "initialize_esm_model": "embedding",
    "calculate_embedding": "embedding",
    "calculate_embedding_single": "embedding",
    "EmbeddingManager": "embedding",
    "select_templates": "embedding",
    # MCS functions
    "find_mcs": "mcs",
    "find_best_ca_rmsd_template": "mcs",
    "constrained_embed": "mcs",
    "central_atom_embed": "mcs",
    "safe_name": "mcs",
    "get_central_atom": "mcs",
    "needs_uff_fallback": "mcs",
    "embed_with_uff_fallback": "mcs",
    "validate_molecular_connectivity": "mcs",
    "validate_molecular_geometry": "mcs",
    "simple_minimize_molecule": "mcs",
    "validate_mmff_parameters": "mcs",
    "log_coordinate_map": "mcs",
    "relax_close_constraints": "mcs",
    # Scoring functions
    "score_and_align": "scoring",
    "select_best": "scoring",
    "rmsd_raw": "scoring",
    "generate_properties_for_sdf": "scoring",
    "FixedMolecularProcessor": "scoring",
    "ScoringFixer": "scoring",
    "CoordinateMapper": "scoring",
    # Chemistry functions
    "detect_and_substitute_organometallic": "chemistry",
    "has_rhenium_complex": "chemistry",
    "is_large_peptide": "chemistry",
    "validate_target_molecule": "chemistry",
    # Templates
    "load_reference_protein": "templates",
    "load_target_data": "templates",
    "transform_ligand": "templates",
    "filter_templates_by_ca_rmsd": "templates",
    "get_templates_with_progressive_fallback": "templates",
    "pdb_path": "templates",
    "ligand_path": "templates",
    # Pipeline
    "TEMPLPipeline": "pipeline",
    "PipelineConfig": "pipeline",
    "run_pipeline": "pipeline",
    "run_pipeline_from_args": "pipeline",
    "run_from_pdb_and_smiles": "pipeline",
    # Workspace Management
    "WorkspaceManager": "workspace_manager",
    "WorkspaceConfig": "workspace_manager",
    "DirectoryManager": "workspace_manager",
    "TempDirectoryManager": "workspace_manager",
    "create_workspace_manager": "workspace_manager",
    "emergency_cleanup": "workspace_manager",
    # File Management
    "FileManager": "file_manager",
    "AdaptiveFileNamingEngine": "file_manager",
    "PredictionContext": "file_manager",
    "FileMetadata": "file_manager",
    "create_file_manager": "file_manager",
    "generate_adaptive_filename": "file_manager",
    # Execution Management
    "ExecutionManager": "execution_manager",
    "SkipReason": "execution_manager",
    "MoleculeSkipException": "execution_manager",
    "get_safe_worker_count": "execution_manager",
    "skip_molecule": "execution_manager",
    "record_successful_processing": "execution_manager",
    "get_execution_summary": "execution_manager",
    # Hardware Detection
    "HardwareInfo": "hardware",
    "get_basic_hardware_info": "hardware",
    "get_hardware_info": "hardware",
    "get_optimized_worker_config": "hardware",
    "get_suggested_worker_config": "hardware",
    "detect_optimal_configuration": "hardware",
    # Data Management
    "DatasetSplits": "data",
    "DatasetManager": "data",
    "load_benchmark_pdbs": "data",
    "validate_dataset_integrity": "data",
    # Validation
    "SplitDataValidator": "validation",
    "DatabaseValidator": "validation",
    "MolecularValidationFramework": "validation",
    "validate_pipeline_components": "validation",
    "quick_validation_check": "validation",
}

# Cache for loaded modules
_loaded_modules = {}


def __getattr__(name: str) -> Any:  # type: ignore[misc]
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