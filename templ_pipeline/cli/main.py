#!/usr/bin/env python3
"""
TEMPL Pipeline Command Line Interface

Enhanced CLI with Smart Progressive Interface, contextual help, and adaptive UX.

This module provides a command-line interface for the TEMPL pipeline,
allowing users to:
1. Generate protein embeddings
2. Find similar protein templates
3. Generate poses for a query ligand based on templates
4. Score and select the best poses

Features:
- Smart Progressive Interface that adapts to user experience level
- Context-aware help and suggestions
- Adaptive verbosity and progress indication
- User preference learning and optimization

Usage examples:
  # Generate embedding for a protein
  templ embed --protein-file data/example/2hyy_protein.pdb

  # Find protein templates
  templ find-templates --protein-file data/example/2hyy_protein.pdb --embedding-file data/embeddings/templ_protein_embeddings_v1.0.0.npz

  # Generate poses with SMILES input
  templ generate-poses --protein-file data/example/2hyy_protein.pdb --ligand-smiles "Cc1cn(cn1)c2cc(NC(=O)c3ccc(C)c(Nc4nccc(n4)c5cccnc5)c3)cc(c2)C(F)(F)F" --template-pdb 5eqy

  # Generate poses with SDF input
  templ generate-poses --protein-file data/example/2hyy_protein.pdb --ligand-file data/example/2hyy_ligand.sdf --template-pdb 5eqy

  # Run full pipeline
  templ run --protein-file data/example/2hyy_protein.pdb --ligand-smiles "Cc1cn(cn1)c2cc(NC(=O)c3ccc(C)c(Nc4nccc(n4)c5cccnc5)c3)cc(c2)C(F)(F)F" --embedding-file data/embeddings/templ_protein_embeddings_v1.0.0.npz
"""

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Import UX enhancements
from .ux_config import (
    get_ux_config,
    configure_logging_for_verbosity,
    VerbosityLevel,
    ExperienceLevel,
)
from .help_system import create_enhanced_parser, handle_help_request
from .progress_indicators import progress_context, OperationType, simple_progress_wrapper

# Import version information
try:
    from templ_pipeline import __version__
except ImportError:
    __version__ = "unknown"

# Configure logging with UX-aware settings
ux_config = get_ux_config()
verbosity = ux_config.get_verbosity_level()
configure_logging_for_verbosity(verbosity, "templ-cli")
logger = logging.getLogger("templ-cli")


# Lazy hardware detection - only import when needed
def _get_hardware_config():
    """Lazy load hardware configuration to avoid heavy imports on startup"""
    try:
        from templ_pipeline.core.hardware import get_suggested_worker_config

        return get_suggested_worker_config()
    except ImportError:
        logger.warning("Hardware detection not available, using conservative defaults")
        return {"n_workers": 4, "internal_pipeline_workers": 1}


def _lazy_import_rdkit():
    """Lazy import RDKit and suppress logging."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.*")
        return Chem, AllChem
    except ImportError as e:
        logger.error(f"RDKit not available: {e}")
        sys.exit(1)


def _lazy_import_core():
    """Lazy import TEMPL core components."""
    try:
        from templ_pipeline.core import (
            EmbeddingManager,
            get_protein_embedding,
            get_protein_sequence,
            find_mcs,
            constrained_embed,
            select_best,
            rmsd_raw,
            generate_properties_for_sdf,
        )

        return (
            EmbeddingManager,
            get_protein_embedding,
            get_protein_sequence,
            find_mcs,
            constrained_embed,
            select_best,
            rmsd_raw,
            generate_properties_for_sdf,
        )
    except ImportError as e:
        logger.error(f"TEMPL core components not available: {e}")
        sys.exit(1)


def _lazy_import_numpy():
    """Lazy import numpy."""
    try:
        import numpy as np

        return np
    except ImportError as e:
        logger.error(f"NumPy not available: {e}")
        sys.exit(1)


def validate_smiles(smiles_string):
    """Validate SMILES string using RDKit."""
    try:
        Chem, _ = _lazy_import_rdkit()
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return False, "Invalid SMILES string"
        return True, "Valid SMILES"
    except Exception as e:
        return False, f"SMILES validation error: {str(e)}"


def setup_parser():
    """Set up the command-line argument parser with UX enhancements."""
    # Create enhanced parser with smart help system
    parser, help_system = create_enhanced_parser()

    # Add version argument
    parser.add_argument(
        "--version",
        action="version",
        version=f"TEMPL Pipeline {__version__}",
        help="Show version number and exit",
    )

    # Get user experience level for adaptive interface
    experience_level = ux_config.get_effective_experience_level()
    appropriate_args = ux_config.get_arguments_for_user_level(experience_level)

    # Show contextual welcome message for new users
    if (
        experience_level == ExperienceLevel.BEGINNER
        and ux_config.usage_patterns.total_commands < 3
    ):
        logger.info(
            "Welcome to TEMPL! For getting started help: templ --help getting-started"
        )

    # Add common arguments - log-level should always be available for debugging
    # Map verbosity levels to proper logging levels
    verbosity_to_log_level = {
        VerbosityLevel.MINIMAL: "WARNING",
        VerbosityLevel.NORMAL: "INFO",
        VerbosityLevel.DETAILED: "INFO",
        VerbosityLevel.DEBUG: "DEBUG",
    }
    default_log_level = verbosity_to_log_level.get(verbosity, "INFO")

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=default_log_level,
        help="Set logging level",
    )

    # Add other common arguments (adapt based on user level)
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Directory for output files"
    )

    # Add verbosity control
    parser.add_argument(
        "--verbosity",
        choices=["minimal", "normal", "detailed", "debug"],
        help="Output verbosity level",
    )
    
    # Add random seed for reproducible results
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results (default: 42)",
    )

    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Embedding generation command
    embed_parser = subparsers.add_parser(
        "embed", help="Generate embedding for a protein"
    )
    embed_parser.add_argument(
        "--protein-file", type=str, required=True, help="Path to protein PDB file"
    )
    embed_parser.add_argument(
        "--chain",
        type=str,
        default=None,
        help="Specific chain to use (default: first chain)",
    )
    embed_parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save embedding (default: based on protein filename)",
    )

    # Template finding command
    find_templates_parser = subparsers.add_parser(
        "find-templates", help="Find similar protein templates"
    )
    find_templates_parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query protein (.pdb) or pre-computed embedding (.npz)",
    )
    find_templates_parser.add_argument(
        "--embedding-file",
        type=str,
        required=True,
        help="Path to pre-computed embeddings file (.npz)",
    )
    find_templates_parser.add_argument(
        "--num-templates", type=int, default=10, help="Number of templates to return"
    )
    find_templates_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Minimum similarity threshold (overrides --num-templates if provided)",
    )
    find_templates_parser.add_argument(
        "--exclude-uniprot-file",
        type=str,
        default=None,
        help="File containing UniProt IDs to exclude (one per line)",
    )

    # Pose generation command
    generate_poses_parser = subparsers.add_parser(
        "generate-poses", help="Generate poses for a query ligand based on templates"
    )
    generate_poses_parser.add_argument(
        "--protein-file", type=str, required=True, help="Path to query protein PDB file"
    )
    generate_poses_parser.add_argument(
        "--ligand-smiles",
        type=str,
        default=None,
        help="SMILES string for query ligand (e.g., 'COc1ccc(C(C)=O)c(O)c1[C@H]1C[C@H]1NC(=S)Nc1ccc(C#N)cn1')",
    )
    generate_poses_parser.add_argument(
        "--ligand-file",
        type=str,
        default=None,
        help="SDF file containing query ligand (alternative to --ligand-smiles)",
    )
    generate_poses_parser.add_argument(
        "--template-pdb", type=str, required=True, help="PDB ID of template to use"
    )
    generate_poses_parser.add_argument(
        "--template-ligand-file",
        type=str,
        default=None,
        help="SDF file containing template ligand (optional, auto-loaded from database if not provided)",
    )
    generate_poses_parser.add_argument(
        "--num-conformers",
        type=int,
        default=100,
        help="Number of conformers to generate",
    )
    generate_poses_parser.add_argument(
        "--workers",
        type=int,
        default=None,  # Will be set in command handler
        help="Number of parallel workers to use (auto-detected based on hardware)",
    )
    generate_poses_parser.add_argument(
        "--no-realign",
        action="store_true",
        help="Use raw conformers (no shape alignment)",
    )
    generate_poses_parser.add_argument(
        "--unconstrained",
        action="store_true",
        help="Skip MCS and constrained embedding for unconstrained conformer generation",
    )
    generate_poses_parser.add_argument(
        "--align-metric",
        choices=["shape", "color", "combo"],
        default="combo",
        help="Shape alignment metric for pose scoring (default: combo)",
    )
    generate_poses_parser.add_argument(
        "--enable-optimization",
        action="store_true",
        help="Enable force field optimization (optimization disabled by default)",
    )

    # Full pipeline command
    run_parser = subparsers.add_parser("run", help="Run full TEMPL pipeline")

    # Protein input (either file or PDB ID)
    protein_group = run_parser.add_mutually_exclusive_group(required=True)
    protein_group.add_argument(
        "--protein-file", type=str, help="Path to query protein PDB file"
    )
    protein_group.add_argument(
        "--protein-pdb-id",
        type=str,
        help="PDB ID for query protein (alternative to --protein-file)",
    )

    # Ligand input (either SMILES or file)
    ligand_group = run_parser.add_mutually_exclusive_group(required=True)
    ligand_group.add_argument(
        "--ligand-smiles",
        type=str,
        help="SMILES string for query ligand (e.g., 'COc1ccc(C(C)=O)c(O)c1[C@H]1C[C@H]1NC(=S)Nc1ccc(C#N)cn1')",
    )
    ligand_group.add_argument(
        "--ligand-file",
        type=str,
        help="SDF file containing query ligand (alternative to --ligand-smiles)",
    )

    run_parser.add_argument(
        "--embedding-file",
        type=str,
        default=None,
        help="Path to pre-computed embeddings file (.npz) - uses default if not specified",
    )
    run_parser.add_argument(
        "--num-templates", type=int, default=100, help="Number of templates to consider"
    )
    run_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Minimum similarity threshold (overrides --num-templates if provided)",
    )
    run_parser.add_argument(
        "--num-conformers",
        type=int,
        default=100,
        help="Number of conformers to generate",
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        default=None,  # Will be set in command handler
        help="Number of parallel workers to use (auto-detected based on hardware)",
    )
    run_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run identifier (default: timestamp)",
    )
    run_parser.add_argument(
        "--no-realign",
        action="store_true",
        help="Use raw conformers (no shape alignment)",
    )
    run_parser.add_argument(
        "--enable-optimization",
        action="store_true",
        help="Enable force field optimization (optimization disabled by default)",
    )
    run_parser.add_argument(
        "--unconstrained",
        action="store_true",
        help="Skip MCS and constrained embedding for unconstrained conformer generation",
    )
    run_parser.add_argument(
        "--align-metric",
        choices=["shape", "color", "combo"],
        default="combo",
        help="Shape alignment metric for pose scoring (default: combo)",
    )

    # Benchmark command ---------------------------------------------------
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run built-in benchmark suites"
    )
    benchmark_parser.add_argument(
        "suite", choices=["polaris", "time-split"], help="Benchmark suite to execute"
    )
    benchmark_parser.add_argument(
        "--n-workers",
        "--workers",
        dest="n_workers",
        type=int,
        default=None,  # Will be set in command handler
        help="Number of CPU workers to utilise (auto-detected based on hardware)",
    )
    benchmark_parser.add_argument(
        "--n-conformers",
        type=int,
        default=200,
        help="Number of conformers to generate per molecule",
    )
    benchmark_parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced quick subset for smoke testing",
    )
    benchmark_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug output (otherwise minimal progress bars)",
    )
    benchmark_parser.add_argument(
        "--save-poses",
        action="store_true",
        help="Save predicted poses as SDF files for backtesting analysis",
    )
    benchmark_parser.add_argument(
        "--poses-dir",
        type=str,
        default=None,
        help="Directory to save predicted poses (default: benchmark_poses_<timestamp>)",
    )
    # Time-split specific arguments
    benchmark_parser.add_argument(
        "--template-knn",
        type=int,
        default=100,
        help="Number of nearest neighbors for template selection (time-split only)",
    )
    benchmark_parser.add_argument(
        "--max-pdbs",
        type=int,
        default=None,
        help="Maximum number of PDBs to evaluate (for testing, time-split only)",
    )
    benchmark_parser.add_argument(
        "--val-only",
        action="store_true",
        help="Only evaluate validation set (time-split only)",
    )
    benchmark_parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only evaluate test set (time-split only)",
    )
    benchmark_parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only evaluate training set (time-split only)",
    )
    benchmark_parser.add_argument(
        "--pipeline-timeout",
        type=int,
        default=1800,
        help="Timeout in seconds for each individual PDB processing (time-split only)",
    )
    benchmark_parser.add_argument(
        "--max-ram",
        dest="max_ram_gb",
        type=float,
        default=None,
        help="Maximum RAM (GiB) before throttling worker submission (time-split only)",
    )
    benchmark_parser.add_argument(
        "--per-worker-ram-gb",
        type=float,
        default=4.0,
        help="Maximum RAM (GiB) per worker process (prevents memory explosion, default: 4.0)",
    )
    # Advanced hardware optimization arguments
    benchmark_parser.add_argument(
        "--hardware-profile",
        choices=["auto", "conservative", "balanced", "aggressive"],
        default="auto",
        help="Hardware utilization profile: auto (detect), conservative (safe), balanced (default), aggressive (max performance)",
    )
    benchmark_parser.add_argument(
        "--cpu-limit",
        type=int,
        default=None,
        help="Maximum number of CPU cores to use (overrides auto-detection)",
    )
    benchmark_parser.add_argument(
        "--memory-limit",
        type=float,
        default=None,
        help="Maximum system memory to use in GiB (overrides auto-detection)",
    )
    benchmark_parser.add_argument(
        "--worker-strategy", 
        choices=["auto", "io-bound", "cpu-bound", "memory-bound"],
        default="auto",
        help="Worker allocation strategy based on workload characteristics",
    )
    benchmark_parser.add_argument(
        "--enable-hyperthreading",
        action="store_true",
        help="Enable hyperthreading utilization (may improve or hurt performance)",
    )
    benchmark_parser.add_argument(
        "--disable-auto-scaling",
        action="store_true", 
        help="Disable automatic worker scaling based on system load",
    )
    benchmark_parser.add_argument(
        "--peptide-threshold",
        type=int,
        default=8,
        help="Maximum number of amino acid residues before considering a molecule a large peptide to skip (default: 8)",
    )
    # Ablation study flags
    benchmark_parser.add_argument(
        "--unconstrained",
        action="store_true",
        help="Skip MCS and constrained embedding for unconstrained conformer generation",
    )
    benchmark_parser.add_argument(
        "--align-metric",
        choices=["shape", "color", "combo"],
        default="combo",
        help="Shape alignment metric for pose scoring (default: combo)",
    )
    benchmark_parser.add_argument(
        "--enable-optimization",
        action="store_true",
        help="Enable force field optimization (optimization disabled by default)",
    )
    benchmark_parser.add_argument(
        "--no-realign",
        action="store_true",
        help="Disable pose realignment for scoring-only mode",
    )
    benchmark_parser.set_defaults(func=benchmark_command)

    return parser, help_system


def set_log_level(level_name):
    """Set the logging level based on the provided name."""
    level = getattr(logging, level_name)
    logger.setLevel(level)
    # Also set for root logger
    logging.getLogger().setLevel(level)


def detect_query_type(query_path):
    """Detect input type based on file extension."""
    ext = os.path.splitext(query_path)[1].lower()
    if ext == ".pdb":
        return "protein"
    elif ext == ".npz":
        return "embedding"
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdb or .npz")


def load_query_embedding(query_path):
    """Load embedding from either PDB or NPZ file."""
    np = _lazy_import_numpy()
    (
        EmbeddingManager,
        get_protein_embedding,
        get_protein_sequence,
        find_mcs,
        constrained_embed,
        select_best,
        rmsd_raw,
        generate_properties_for_sdf,
    ) = _lazy_import_core()

    query_type = detect_query_type(query_path)

    if query_type == "protein":
        # Generate embedding from PDB
        logger.info(f"Generating embedding from PDB file: {query_path}")
        embedding, chains = get_protein_embedding(query_path)
        query_id = os.path.splitext(os.path.basename(query_path))[0]
        if query_id.endswith("_protein"):
            query_id = query_id[:-8]  # Remove '_protein' suffix
        return embedding, query_id

    elif query_type == "embedding":
        # Load pre-computed embedding
        logger.info(f"Loading pre-computed embedding from: {query_path}")
        try:
            data = np.load(query_path)

            # Validate structure
            if "embeddings" not in data or "pdb_ids" not in data:
                raise ValueError(
                    "Invalid NPZ structure: missing 'embeddings' or 'pdb_ids'"
                )

            if data["embeddings"].shape[1] != 1280:
                raise ValueError(
                    f"Embedding dimension mismatch: expected 1280, got {data['embeddings'].shape[1]}"
                )

            embedding = data["embeddings"][0]  # Shape: (1280,)
            query_id = str(data["pdb_ids"][0])
            return embedding, query_id

        except Exception as e:
            raise ValueError(f"Failed to load embedding: {str(e)}")


def embed_command(args):
    """Generate embedding for a protein."""
    # Lazy import dependencies
    (
        EmbeddingManager,
        get_protein_embedding,
        get_protein_sequence,
        find_mcs,
        constrained_embed,
        select_best,
        rmsd_raw,
        generate_properties_for_sdf,
    ) = _lazy_import_core()
    np = _lazy_import_numpy()

    protein_file = args.protein_file
    chain = args.chain
    output_file = args.output_file

    if not os.path.exists(protein_file):
        logger.error(f"Protein file not found: {protein_file}")
        return 1

    # Generate output filename if not provided
    if not output_file:
        base_name = os.path.splitext(os.path.basename(protein_file))[0]
        output_file = os.path.join(args.output_dir, f"{base_name}_embedding.npz")
    else:
        # If output_file is just a filename, put it in the output directory
        if not os.path.dirname(output_file):
            output_file = os.path.join(args.output_dir, output_file)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Extract sequence from protein
    logger.info(f"Extracting sequence from {protein_file}")
    sequence, chains = get_protein_sequence(protein_file, chain)

    if not sequence:
        logger.error("Failed to extract sequence from protein file")
        return 1

    logger.info(
        f"Extracted sequence of length {len(sequence)} from chains {', '.join(chains)}"
    )

    # Generate embedding
    logger.info("Generating embedding")
    embedding, chains = get_protein_embedding(protein_file, chain)

    if embedding is None:
        logger.error("Failed to generate embedding")
        return 1

    # Save embedding in npz format compatible with EmbeddingManager
    logger.info(f"Saving embedding to {output_file}")

    # Extract PDB ID from filename for consistent format
    pdb_id = os.path.splitext(os.path.basename(protein_file))[0]
    if pdb_id.endswith("_protein"):
        pdb_id = pdb_id[:-8]  # Remove '_protein' suffix

    # Save in format compatible with EmbeddingManager
    np.savez_compressed(
        output_file,
        pdb_ids=np.array([pdb_id]),
        embeddings=np.array([embedding]),
        chain_ids=np.array([",".join(chains)]),
    )

    logger.info(f"Embedding saved successfully (shape: {embedding.shape})")
    return 0


def find_templates_command(args):
    """Find similar protein templates."""
    # Lazy import dependencies
    (
        EmbeddingManager,
        get_protein_embedding,
        get_protein_sequence,
        find_mcs,
        constrained_embed,
        select_best,
        rmsd_raw,
        generate_properties_for_sdf,
    ) = _lazy_import_core()

    query_path = args.query
    embedding_file = args.embedding_file
    num_templates = args.num_templates
    similarity_threshold = args.similarity_threshold
    exclude_uniprot_file = args.exclude_uniprot_file

    # Validate files exist
    if not os.path.exists(query_path):
        logger.error(f"Query file not found: {query_path}")
        return 1

    if not os.path.exists(embedding_file):
        logger.error(f"Embedding file not found: {embedding_file}")
        return 1

    # Load UniProt IDs to exclude if provided
    exclude_uniprot = set()
    if exclude_uniprot_file and os.path.exists(exclude_uniprot_file):
        with open(exclude_uniprot_file) as f:
            exclude_uniprot = {line.strip() for line in f if line.strip()}
        logger.info(f"Loaded {len(exclude_uniprot)} UniProt IDs to exclude")

    # Initialize embedding manager
    logger.info(f"Initializing embedding manager with {embedding_file}")
    manager = EmbeddingManager(embedding_file)

    # Auto-detect and load query embedding
    try:
        query_embedding, query_id = load_query_embedding(query_path)
    except Exception as e:
        logger.error(f"Failed to load query: {str(e)}")
        return 1

    if query_embedding is None:
        logger.error("Failed to load query embedding")
        return 1

    # Check if the embedding database was loaded properly
    if not manager.embedding_db:
        logger.warning(
            "Embedding database is empty - this embedding file may be a single protein embedding"
        )
        logger.info(
            "For template search, you need a database of multiple protein embeddings"
        )

    # Find templates
    logger.info("Finding similar templates")
    templates = manager.find_neighbors(
        query_id,
        query_embedding=query_embedding,
        exclude_uniprot_ids=exclude_uniprot,
        k=num_templates if similarity_threshold is None else None,
        similarity_threshold=similarity_threshold,
        return_similarities=True,
    )

    if not templates:
        logger.error("No templates found")
        return 1

    # Output templates
    logger.info(f"Found {len(templates)} templates")
    for i, (pdb_id, similarity) in enumerate(templates):
        logger.info(f"{i+1}. {pdb_id} (similarity: {similarity:.4f})")

    # Save results to file
    output_file = os.path.join(args.output_dir, "templates.txt")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("PDB_ID\tSimilarity\n")
        for pdb_id, similarity in templates:
            f.write(f"{pdb_id}\t{similarity:.4f}\n")

    logger.info(f"Template list saved to {output_file}")
    return 0


def generate_poses_command(args):
    """Generate poses using TEMPLPipeline."""
    from templ_pipeline.core.pipeline import TEMPLPipeline

    try:
        # Set workers if not provided
        if args.workers is None:
            hardware_config = _get_hardware_config()
            args.workers = hardware_config["n_workers"]

        # Initialize pipeline
        pipeline = TEMPLPipeline(
            embedding_path=getattr(args, "embedding_file", None),
            output_dir=args.output_dir,
        )

        # Load template molecules
        template_mols = pipeline.load_template_molecules([args.template_pdb])
        if not template_mols:
            logger.error(f"Template ligand not found for PDB {args.template_pdb}")
            return 1

        # Prepare query molecule
        query_mol = pipeline.prepare_query_molecule(
            ligand_smiles=args.ligand_smiles, ligand_file=args.ligand_file
        )

        # Generate poses
        results = pipeline.generate_poses(
            query_mol=query_mol,
            template_mols=template_mols,
            num_conformers=args.num_conformers,
            n_workers=args.workers,
        )

        # Save results - extract poses from results dict
        poses = (
            results["poses"]
            if isinstance(results, dict) and "poses" in results
            else results
        )
        output_file = pipeline.save_results(poses, args.template_pdb)
        logger.info(f"Results saved to: {output_file}")

        return 0

    except Exception as e:
        logger.error(f"Pose generation failed: {str(e)}")
        return 1


def run_command(args):
    """Run the full TEMPL pipeline."""
    try:
        from templ_pipeline.core.pipeline import TEMPLPipeline
    except ImportError as e:
        logger.error(f"Failed to import pipeline: {e}")
        return 1

    print("Starting TEMPL pipeline")

    try:
        # Set workers if not provided
        if not hasattr(args, "workers") or args.workers is None:
            hardware_config = _get_hardware_config()
            args.workers = hardware_config["n_workers"]

        # Initialize pipeline
        pipeline = TEMPLPipeline(
            embedding_path=getattr(args, "embedding_file", None),
            output_dir=args.output_dir,
            run_id=getattr(args, "run_id", None),
        )

        # Run pipeline with progress indication
        def run_pipeline():
            return pipeline.run_full_pipeline(
                protein_file=getattr(args, "protein_file", None),
                protein_pdb_id=getattr(args, "protein_pdb_id", None),
                ligand_smiles=getattr(args, "ligand_smiles", None),
                ligand_file=getattr(args, "ligand_file", None),
                num_templates=getattr(args, "num_templates", 100),
                num_conformers=getattr(args, "num_conformers", 100),
                n_workers=args.workers,
                similarity_threshold=getattr(args, "similarity_threshold", None),
                no_realign=getattr(args, "no_realign", False),
                enable_optimization=getattr(args, "enable_optimization", False),
                unconstrained=getattr(args, "unconstrained", False),
                align_metric=getattr(args, "align_metric", "combo"),
            )

        results = simple_progress_wrapper("Running TEMPL pipeline", run_pipeline)

        # Report results
        print(f"Pipeline completed successfully!")
        print(f"Found {len(results.get('templates', []))} templates")
        print(f"Generated {len(results.get('poses', {}))} poses")
        print(f"Results saved to: {results.get('output_file', 'unknown')}")

        return 0

    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return 1


def load_template_molecules_from_sdf(template_pdbs):
    """Load template molecules using pipeline method."""
    from templ_pipeline.core.pipeline import TEMPLPipeline

    try:
        pipeline = TEMPLPipeline()
        return pipeline.load_template_molecules(template_pdbs)

    except Exception as e:
        logger.error(f"Failed to load template molecules: {e}")
        raise


def _generate_unified_summary(workspace_dir, benchmark_type):
    """Generate unified summary files for benchmark results."""
    try:
        from templ_pipeline.benchmark.summary_generator import BenchmarkSummaryGenerator
        import json
        from pathlib import Path
        
        generator = BenchmarkSummaryGenerator()
        raw_results_dir = workspace_dir / "raw_results"
        summaries_dir = workspace_dir / "summaries"
        
        # Find result files based on benchmark type
        result_files = []
        if benchmark_type == "polaris":
            # Look for Polaris JSON results in workspace directory first, then fallback to old location
            polaris_workspace_dir = raw_results_dir / "polaris"
            polaris_fallback_dir = Path("templ_benchmark_results_polaris")
            
            logger.info(f"Searching for Polaris results in: {polaris_workspace_dir}")
            if polaris_workspace_dir.exists():
                json_files = list(polaris_workspace_dir.glob("*.json"))
                result_files.extend(json_files)
                logger.info(f"Found {len(json_files)} JSON files in workspace directory")
            else:
                logger.info(f"Workspace directory {polaris_workspace_dir} does not exist")
                
            if not result_files and polaris_fallback_dir.exists():
                fallback_files = list(polaris_fallback_dir.glob("*.json"))
                result_files.extend(fallback_files)
                logger.info(f"Found {len(fallback_files)} JSON files in fallback directory: {polaris_fallback_dir}")
            
            # Additional search in raw_results_dir root
            if not result_files and raw_results_dir.exists():
                root_files = list(raw_results_dir.glob("*.json"))
                result_files.extend(root_files)
                logger.info(f"Found {len(root_files)} JSON files in raw results root directory")
        elif benchmark_type == "timesplit":
            # Look for Timesplit JSONL results - check multiple locations
            jsonl_locations = [
                raw_results_dir / "timesplit" / "results_stream.jsonl",
                raw_results_dir / "results_stream.jsonl", 
                workspace_dir / "raw_results" / "timesplit" / "results_stream.jsonl",
                workspace_dir / "timesplit_stream_results" / "results_stream.jsonl"
            ]
            
            logger.info(f"Searching for Timesplit results in {len(jsonl_locations)} locations")
            for jsonl_path in jsonl_locations:
                logger.info(f"Checking: {jsonl_path}")
                if jsonl_path.exists():
                    result_files.append(jsonl_path)
                    logger.info(f"Found timesplit results: {jsonl_path}")
                    break
            
            # Also look for any other JSON/JSONL files as fallback
            if not result_files and raw_results_dir.exists():
                jsonl_files = list(raw_results_dir.glob("**/*.jsonl"))
                json_files = list(raw_results_dir.glob("**/*.json"))
                result_files.extend(jsonl_files)
                result_files.extend(json_files)
                logger.info(f"Fallback search found {len(jsonl_files)} JSONL and {len(json_files)} JSON files")
        
        if not result_files:
            logger.warning(f"No result files found for {benchmark_type} benchmark summary")
            logger.warning(f"Searched in: {raw_results_dir}")
            if raw_results_dir.exists():
                all_files = list(raw_results_dir.rglob("*"))
                logger.warning(f"Files found in results dir: {all_files}")
            return
        
        # Load and combine results
        all_results = {}
        for file_path in result_files:
            try:
                logger.info(f"Loading results from: {file_path}")
                if file_path.suffix == ".jsonl":
                    # JSONL format
                    results = []
                    with open(file_path, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            if line.strip():
                                try:
                                    results.append(json.loads(line))
                                except json.JSONDecodeError as je:
                                    logger.warning(f"Invalid JSON on line {line_num} in {file_path}: {je}")
                    logger.info(f"Loaded {len(results)} results from {file_path}")
                    all_results[file_path.stem] = results
                elif file_path.suffix == ".json":
                    # JSON format
                    with open(file_path, 'r') as f:
                        results = json.load(f)
                    logger.info(f"Loaded JSON results from {file_path}")
                    all_results[file_path.stem] = results
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        if not all_results:
            logger.warning("No valid results loaded for summary generation")
            return
        
        # Generate summary
        if len(all_results) == 1:
            results_data = list(all_results.values())[0]
        else:
            results_data = all_results
            
        summary = generator.generate_unified_summary(results_data, benchmark_type)
        
        # Save summary files
        saved_files = generator.save_summary_files(
            summary, 
            summaries_dir,
            f"{benchmark_type}_benchmark_summary"
        )
        
        if saved_files:
            logger.info(f"Generated summary files:")
            for fmt, path in saved_files.items():
                logger.info(f"  {fmt.upper()}: {path}")
        
    except Exception as e:
        logger.warning(f"Failed to generate unified summary: {e}")


def _optimize_hardware_config(args):
    """Simplified hardware config: decide n_workers and memory based on args and system."""
    try:
        import psutil
        total_mem_gb = psutil.virtual_memory().total / (1024 ** 3)
        physical_cpus = psutil.cpu_count(logical=False) or 4
        logical_cpus = psutil.cpu_count(logical=True) or physical_cpus
    except ImportError:
        total_mem_gb = 8.0
        physical_cpus = logical_cpus = 4

    # Defaults
    profile = getattr(args, "hardware_profile", "auto")
    suite = getattr(args, "suite", "polaris")
    use_hyper = getattr(args, "enable_hyperthreading", False)
    cpu_limit = getattr(args, "cpu_limit", None)
    mem_limit = getattr(args, "memory_limit", None)
    per_worker_ram = getattr(args, "per_worker_ram_gb", 4.0)

    cpus = logical_cpus if use_hyper else physical_cpus
    if cpu_limit:
        cpus = min(cpus, cpu_limit)
    mem = min(total_mem_gb, mem_limit) if mem_limit else total_mem_gb

    # Use all CPUs, but cap by RAM
    n_workers = cpus
    n_workers = min(n_workers, int(mem // per_worker_ram))
    n_workers = max(1, n_workers)

    # Special case for time-split
    if suite == "time-split":
        n_workers = min(n_workers, 22)  # Cap at 22 for safety
        max_ram_gb = mem * 0.8
    else:
        max_ram_gb = mem * 0.9

    # User override
    if getattr(args, "n_workers", None):
        n_workers = args.n_workers

    return {
        "n_workers": n_workers,
        "max_ram_gb": max_ram_gb,
        "total_memory_gb": total_mem_gb,
        "profile": profile,
        "strategy": "auto",
        "per_worker_ram_gb": per_worker_ram,
    }


def benchmark_command(args):
    """Execute selected benchmark suite with enhanced workspace organization."""
    # Apply hardware optimization based on CLI arguments
    hardware_config = _optimize_hardware_config(args)
    args.n_workers = hardware_config["n_workers"]
    args.per_worker_ram_gb = hardware_config["per_worker_ram_gb"]
    
    # Log hardware optimization decisions
    logger.info(f"Hardware optimization: {hardware_config['profile']} profile")
    logger.info(f"Workers: {args.n_workers}, Strategy: {hardware_config['strategy']}")
    if hardware_config.get('memory_optimized'):
        logger.info(f"Memory optimized: {hardware_config['total_memory_gb']:.1f}GB available")

    # Create organized workspace directory for benchmark results
    from datetime import datetime
    from pathlib import Path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace_dir = Path(f"benchmark_workspace_{args.suite}_{timestamp}")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for organization
    (workspace_dir / "raw_results").mkdir(exist_ok=True)
    (workspace_dir / "summaries").mkdir(exist_ok=True)
    logs_dir = workspace_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Setup structured log files
    main_log_file = logs_dir / "benchmark.log"
    error_log_file = logs_dir / "errors.log"

    if args.suite == "polaris":
        print("DEBUG: Polaris benchmark requested")
        try:
            from templ_pipeline.benchmark.polaris.benchmark import (
                main as benchmark_main,
            )

            # Setup file-only logging for clean progress bar display
            from templ_pipeline.core.benchmark_logging import benchmark_logging_context
            
            # Determine log level based on verbosity
            if hasattr(args, "verbose") and args.verbose:
                log_level = "DEBUG"
            else:
                log_level = "INFO"
                
            # Suppress warnings before starting benchmark
            
            # Use benchmark logging context for clean terminal output
            with benchmark_logging_context(
                workspace_dir=workspace_dir,
                benchmark_name="polaris",
                log_level=log_level,
                suppress_console=False  # Allow progress bars to show
            ) as log_info:
                logger.info(f"Starting Polaris benchmark with {args.n_workers} workers")
                logger.info(f"Workspace directory: {workspace_dir}/")
                logger.info(f"Logs will be written to: {logs_dir}")

                # Setup quiet mode for terminal output only
                # For clean progress bars, we want quiet=False so progress bars show up
                # but all logging goes to files due to the logging context
                quiet_mode = False

                # Convert CLI args to benchmark args with workspace integration
                benchmark_args = []
                
                # Set output directory to workspace raw_results
                polaris_output_dir = workspace_dir / "raw_results" / "polaris"
                polaris_output_dir.mkdir(parents=True, exist_ok=True)
                benchmark_args.extend(["--output-dir", str(polaris_output_dir)])
                
                # Add workspace directory for logging integration
                benchmark_args.extend(["--workspace-dir", str(workspace_dir)])
                
                if hasattr(args, "n_workers") and args.n_workers:
                    benchmark_args.extend(["--n-workers", str(args.n_workers)])
                if hasattr(args, "n_conformers") and args.n_conformers:
                    benchmark_args.extend(["--n-conformers", str(args.n_conformers)])
                if hasattr(args, "quick") and args.quick:
                    benchmark_args.append("--quick")
                if hasattr(args, "verbose") and args.verbose:
                    benchmark_args.extend(["--log-level", "DEBUG"])
                else:
                    benchmark_args.extend(["--log-level", "INFO"])
                
                # Add pose saving arguments
                if hasattr(args, "save_poses") and args.save_poses:
                    benchmark_args.append("--save-poses")
                    if hasattr(args, "poses_dir") and args.poses_dir:
                        benchmark_args.extend(["--poses-dir", args.poses_dir])
                    else:
                        # Use workspace subdirectory for poses
                        default_poses_dir = workspace_dir / "predicted_poses"
                        default_poses_dir.mkdir(exist_ok=True)
                        benchmark_args.extend(["--poses-dir", str(default_poses_dir)])
                
                # Add ablation study flags
                if hasattr(args, "unconstrained") and args.unconstrained:
                    benchmark_args.append("--unconstrained")
                if hasattr(args, "align_metric") and args.align_metric:
                    benchmark_args.extend(["--align-metric", args.align_metric])
                if hasattr(args, "enable_optimization") and args.enable_optimization:
                    benchmark_args.append("--enable-optimization")
                if hasattr(args, "no_realign") and args.no_realign:
                    benchmark_args.append("--no-realign")

                result = benchmark_main(benchmark_args)

                # Generate unified summary for Polaris results
                _generate_unified_summary(workspace_dir, "polaris")

                logger.info(f"Polaris benchmark completed. Workspace: {workspace_dir}")
                return result
        except ImportError as e:
            logger.error(f"Polaris benchmark module not available: {e}")
            return 1

    elif args.suite == "time-split":
        try:
            from templ_pipeline.benchmark.timesplit import run_timesplit_benchmark

            # Determine which splits to run
            splits_to_run = []
            if hasattr(args, "train_only") and args.train_only:
                splits_to_run = ["train"]
            elif hasattr(args, "val_only") and args.val_only:
                splits_to_run = ["val"]
            elif hasattr(args, "test_only") and args.test_only:
                splits_to_run = ["test"]
            else:
                splits_to_run = ["train", "val", "test"]

            # Setup file-only logging for clean progress bar display
            from templ_pipeline.core.benchmark_logging import benchmark_logging_context
            
            # Determine log level based on verbosity
            if hasattr(args, "verbose") and args.verbose:
                log_level = "DEBUG"
            else:
                log_level = "INFO"

            # Use benchmark logging context for clean terminal output
            with benchmark_logging_context(
                workspace_dir=workspace_dir,
                benchmark_name="timesplit",
                log_level=log_level,
                suppress_console=False  # Allow progress bars to show
            ) as log_info:
                logger.info(f"Starting Timesplit benchmark for splits: {', '.join(splits_to_run)}")
                logger.info(f"Using {args.n_workers} workers, {args.n_conformers} conformers")
                logger.info(f"Workspace directory: {workspace_dir}/")
                logger.info(f"Logs will be written to: {logs_dir}")

                # Use workspace subdirectory for timesplit results
                timesplit_results_dir = workspace_dir / "raw_results" / "timesplit"
                timesplit_results_dir.mkdir(parents=True, exist_ok=True)
                
                # Setup poses directory
                poses_dir = None
                if hasattr(args, "save_poses") and args.save_poses:
                    poses_dir = str(workspace_dir / "raw_results" / "timesplit" / "poses")
                
                # Discover data directory dynamically
                import os
                from pathlib import Path as PathlibPath
                
                # Try multiple potential data directory locations
                potential_data_dirs = [
                    PathlibPath(__file__).resolve().parent.parent / "data",
                    PathlibPath.cwd() / "data",
                    PathlibPath.cwd() / "templ_pipeline" / "data",
                    PathlibPath("data"),
                    PathlibPath("..") / "data",
                ]

                data_dir = None
                for candidate_path in potential_data_dirs:
                    if (
                        candidate_path.exists()
                        and (candidate_path / "ligands" / "templ_processed_ligands_v1.0.0.sdf.gz").exists()
                    ):
                        data_dir = str(candidate_path)
                        break

                # Fallback to PDBBind-style directories if TEMPL data not found
                if data_dir is None:
                    env_data_dir = os.environ.get("PDBBIND_DATA_DIR")
                    if env_data_dir and PathlibPath(env_data_dir).exists():
                        data_dir = env_data_dir
                    else:
                        data_dir = "/data/pdbbind"  # Final fallback
                
                logger.info(f"Using data directory: {data_dir}")
                
                # Run the new timesplit benchmark
                result = run_timesplit_benchmark(
                    splits_to_run=splits_to_run,
                    n_workers=args.n_workers,
                    n_conformers=args.n_conformers,
                    template_knn=getattr(args, "template_knn", 100),
                    max_pdbs=getattr(args, "max_pdbs", None),
                    data_dir=data_dir,
                    results_dir=str(timesplit_results_dir),
                    poses_output_dir=poses_dir,
                    timeout=getattr(args, "timeout", 180),
                    quiet=False,  # Let progress bars show
                    unconstrained=getattr(args, "unconstrained", False),
                    align_metric=getattr(args, "align_metric", "combo"),
                    enable_optimization=getattr(args, "enable_optimization", False),
                    no_realign=getattr(args, "no_realign", False),
                    per_worker_ram_gb=getattr(args, "per_worker_ram_gb", 4.0),
                )

                # Generate unified summary for Timesplit results
                if result.get("success", False):
                    _generate_unified_summary(workspace_dir, "timesplit")
                    logger.info(f"Timesplit benchmark completed successfully!")
                    logger.info(f"Workspace directory: {workspace_dir}")
                    return 0
                else:
                    logger.error(f"Timesplit benchmark failed: {result.get('error', 'Unknown error')}")
                    return 1
        except ImportError as e:
            logger.error(f"Time-split benchmark module not available: {e}")
            return 1

    else:
        logger.error(f"Unknown benchmark suite: {args.suite}")
        return 1


def main():
    """Main entry point for the CLI with enhanced UX."""
    try:
        # Quick check for help arguments to use enhanced help system
        import sys

        if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
            # Determine help type from additional arguments
            help_type = None
            if len(sys.argv) > 2:
                help_type = sys.argv[2]

            # Use enhanced help system
            from .help_system import TEMPLHelpSystem

            help_system = TEMPLHelpSystem()
            handle_help_request(help_type, help_system)
            return 0

        parser, help_system = setup_parser()
        args = parser.parse_args()

        # Handle custom help system (backup for other help patterns)
        if hasattr(args, "help") and args.help is not None:
            handle_help_request(args.help, help_system)
            return 0

        # Update verbosity if specified
        if hasattr(args, "verbosity") and args.verbosity:
            verbosity = VerbosityLevel(args.verbosity)
            configure_logging_for_verbosity(verbosity, "templ-cli")
            ux_config.update_preferences(default_verbosity=verbosity)
        
        # Set random seed for reproducible results
        if hasattr(args, "seed") and args.seed is not None:
            try:
                from templ_pipeline.core.utils import set_global_random_seed
                set_global_random_seed(args.seed)
                logger.info(f"Random seed set to {args.seed} for reproducible results")
            except ImportError:
                logger.warning("Could not set random seed - utils module not available")
        else:
            # Set default seed if none specified
            try:
                from templ_pipeline.core.utils import ensure_reproducible_environment
                ensure_reproducible_environment()
            except ImportError:
                logger.warning("Could not ensure reproducible environment")

        # Validate arguments
        if hasattr(args, "ligand_smiles") and args.ligand_smiles:
            valid, msg = validate_smiles(args.ligand_smiles)
            if not valid:
                logger.error(f"Invalid SMILES: {msg}")
                return 2

        if hasattr(args, "num_conformers") and args.num_conformers is not None:
            if args.num_conformers <= 0:
                logger.error("Number of conformers must be positive")
                return 2

        if hasattr(args, "workers") and args.workers is not None:
            if args.workers <= 0:
                logger.error("Number of workers must be positive")
                return 2

        if (
            hasattr(args, "similarity_threshold")
            and args.similarity_threshold is not None
        ):
            if not (0.0 <= args.similarity_threshold <= 1.0):
                logger.error("Similarity threshold must be between 0.0 and 1.0")
                return 2

        # Check file existence
        if hasattr(args, "protein_file") and args.protein_file:
            if not os.path.exists(args.protein_file):
                logger.error(f"Protein file not found: {args.protein_file}")
                return 2

        if hasattr(args, "query") and args.query:
            if not os.path.exists(args.query):
                logger.error(f"Query file not found: {args.query}")
                return 2
            # Validate file type
            try:
                detect_query_type(args.query)
            except ValueError as e:
                logger.error(str(e))
                return 2

        if hasattr(args, "ligand_file") and args.ligand_file:
            if not os.path.exists(args.ligand_file):
                logger.error(f"Ligand file not found: {args.ligand_file}")
                return 2

        # Set up logging
        set_log_level(args.log_level)

        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)

        # Apply smart defaults
        smart_defaults = ux_config.get_smart_defaults(args.command or "run", vars(args))
        for key, value in smart_defaults.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

        # Show contextual help hints
        help_system.show_contextual_help(args.command or "run", vars(args))

        # Track command usage and execute
        start_time = time.time()
        result = 1

        try:
            # Handle commands with usage tracking
            if args.command == "embed":
                result = embed_command(args)
            elif args.command == "find-templates":
                result = find_templates_command(args)
            elif args.command == "generate-poses":
                result = generate_poses_command(args)
            elif args.command == "run":
                result = run_command(args)
            elif args.command == "benchmark":
                result = benchmark_command(args)
            else:
                handle_help_request("main", help_system)
                result = 1

            # Track successful command completion
            ux_config.track_command_usage(
                args.command or "help", vars(args), success=(result == 0)
            )

            return result

        except Exception as e:
            # Track error for learning
            error_context = f"{args.command or 'unknown'}:{str(e)[:100]}"
            ux_config.track_error(error_context)
            raise

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
