#!/usr/bin/env python3
"""
TEMPL Pipeline Command Line Interface

This module provides a command-line interface for the TEMPL pipeline,
allowing users to:
1. Generate protein embeddings
2. Find similar protein templates
3. Generate poses for a query ligand based on templates
4. Score and select the best poses

Usage examples:
  # Generate embedding for a protein
  templ embed --protein-file data/example/1iky_protein.pdb
  
  # Find protein templates
  templ find-templates --protein-file data/example/1iky_protein.pdb --embedding-file data/embeddings/protein_embeddings_base.npz
  
  # Generate poses with SMILES input
  templ generate-poses --protein-file data/example/1iky_protein.pdb --ligand-smiles "COc1ccc(C(C)=O)c(O)c1[C@H]1C[C@H]1NC(=S)Nc1ccc(C#N)cn1" --template-pdb 5eqy
  
  # Generate poses with SDF input
  templ generate-poses --protein-file data/example/1iky_protein.pdb --ligand-file data/example/1iky_ligand.sdf --template-pdb 5eqy
  
  # Run full pipeline
  templ run --protein-file data/example/1iky_protein.pdb --ligand-smiles "COc1ccc(C(C)=O)c(O)c1[C@H]1C[C@H]1NC(=S)Nc1ccc(C#N)cn1" --embedding-file data/embeddings/protein_embeddings_base.npz
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("templ-cli")

# Import hardware detection
try:
    from templ_pipeline.core.hardware_utils import get_suggested_worker_config
    hardware_config = get_suggested_worker_config()
except ImportError:
    logger.warning("Hardware detection not available, using conservative defaults")
    hardware_config = {"n_workers": 4, "internal_pipeline_workers": 1}

def _lazy_import_rdkit():
    """Lazy import RDKit and suppress logging."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        RDLogger.DisableLog('rdApp.*')
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
            generate_properties_for_sdf
        )
        return (EmbeddingManager, get_protein_embedding, get_protein_sequence,
                find_mcs, constrained_embed, select_best, rmsd_raw, 
                generate_properties_for_sdf)
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
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="TEMPL Pipeline: Template-based Protein-Ligand Pose Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False  # Disable default help to use custom help system
    )
    
    # Add custom help arguments
    parser.add_argument(
        "-h", "--help",
        nargs="?",
        const="main",
        help="Show help message. Options: simple, examples, performance, <command>"
    )
    
    # Add common arguments
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output files"
    )
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Embedding generation command
    embed_parser = subparsers.add_parser(
        "embed",
        help="Generate embedding for a protein"
    )
    embed_parser.add_argument(
        "--protein-file",
        type=str,
        required=True,
        help="Path to protein PDB file"
    )
    embed_parser.add_argument(
        "--chain",
        type=str,
        default=None,
        help="Specific chain to use (default: first chain)"
    )
    embed_parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save embedding (default: based on protein filename)"
    )
    
    # Template finding command
    find_templates_parser = subparsers.add_parser(
        "find-templates",
        help="Find similar protein templates"
    )
    find_templates_parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query protein (.pdb) or pre-computed embedding (.npz)"
    )
    find_templates_parser.add_argument(
        "--embedding-file",
        type=str,
        required=True,
        help="Path to pre-computed embeddings file (.npz)"
    )
    find_templates_parser.add_argument(
        "--num-templates",
        type=int,
        default=10,
        help="Number of templates to return"
    )
    find_templates_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Minimum similarity threshold (overrides --num-templates if provided)"
    )
    find_templates_parser.add_argument(
        "--exclude-uniprot-file",
        type=str,
        default=None,
        help="File containing UniProt IDs to exclude (one per line)"
    )
    
    # Pose generation command
    generate_poses_parser = subparsers.add_parser(
        "generate-poses",
        help="Generate poses for a query ligand based on templates"
    )
    generate_poses_parser.add_argument(
        "--protein-file",
        type=str,
        required=True,
        help="Path to query protein PDB file"
    )
    generate_poses_parser.add_argument(
        "--ligand-smiles",
        type=str,
        default=None,
        help="SMILES string for query ligand (e.g., 'COc1ccc(C(C)=O)c(O)c1[C@H]1C[C@H]1NC(=S)Nc1ccc(C#N)cn1')"
    )
    generate_poses_parser.add_argument(
        "--ligand-file",
        type=str,
        default=None,
        help="SDF file containing query ligand (alternative to --ligand-smiles)"
    )
    generate_poses_parser.add_argument(
        "--template-pdb",
        type=str,
        required=True,
        help="PDB ID of template to use"
    )
    generate_poses_parser.add_argument(
        "--template-ligand-file",
        type=str,
        default=None,
        help="SDF file containing template ligand (optional, auto-loaded from database if not provided)"
    )
    generate_poses_parser.add_argument(
        "--num-conformers",
        type=int,
        default=100,
        help="Number of conformers to generate"
    )
    generate_poses_parser.add_argument(
        "--workers",
        type=int,
        default=hardware_config["n_workers"],
        help=f"Number of parallel workers to use (auto-detected: {hardware_config['n_workers']})"
    )
    generate_poses_parser.add_argument(
        "--no-realign",
        action="store_true",
        help="Use raw conformers (no shape alignment)"
    )
    
    # Full pipeline command
    run_parser = subparsers.add_parser(
        "run",
        help="Run full TEMPL pipeline"
    )
    
    # Protein input (either file or PDB ID)
    protein_group = run_parser.add_mutually_exclusive_group(required=True)
    protein_group.add_argument(
        "--protein-file",
        type=str,
        help="Path to query protein PDB file"
    )
    protein_group.add_argument(
        "--protein-pdb-id",
        type=str,
        help="PDB ID for query protein (alternative to --protein-file)"
    )
    
    # Ligand input (either SMILES or file)
    ligand_group = run_parser.add_mutually_exclusive_group(required=True)
    ligand_group.add_argument(
        "--ligand-smiles",
        type=str,
        help="SMILES string for query ligand (e.g., 'COc1ccc(C(C)=O)c(O)c1[C@H]1C[C@H]1NC(=S)Nc1ccc(C#N)cn1')"
    )
    ligand_group.add_argument(
        "--ligand-file",
        type=str,
        help="SDF file containing query ligand (alternative to --ligand-smiles)"
    )
    
    run_parser.add_argument(
        "--embedding-file",
        type=str,
        default=None,
        help="Path to pre-computed embeddings file (.npz) - uses default if not specified"
    )
    run_parser.add_argument(
        "--num-templates",
        type=int,
        default=100,
        help="Number of templates to consider"
    )
    run_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Minimum similarity threshold (overrides --num-templates if provided)"
    )
    run_parser.add_argument(
        "--num-conformers",
        type=int,
        default=100,
        help="Number of conformers to generate"
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        default=hardware_config["n_workers"],
        help=f"Number of parallel workers to use (auto-detected: {hardware_config['n_workers']})"
    )
    run_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom run identifier (default: timestamp)"
    )
    
    # Benchmark command ---------------------------------------------------
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run built-in benchmark suites"
    )
    benchmark_parser.add_argument(
        "suite",
        choices=["polaris", "time-split"],
        help="Benchmark suite to execute"
    )
    benchmark_parser.add_argument(
        "--n-workers",
        type=int,
        default=hardware_config.get("n_workers", 4),
        help=f"Number of CPU workers to utilise (auto-detected: {hardware_config.get('n_workers', 4)})"
    )
    benchmark_parser.add_argument(
        "--n-conformers",
        type=int,
        default=200,
        help="Number of conformers to generate per molecule"
    )
    benchmark_parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reduced quick subset for smoke testing"
    )
    benchmark_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug output (otherwise minimal progress bars)"
    )
    # Time-split specific arguments
    benchmark_parser.add_argument(
        "--template-knn",
        type=int,
        default=100,
        help="Number of nearest neighbors for template selection (time-split only)"
    )
    benchmark_parser.add_argument(
        "--max-pdbs",
        type=int,
        default=None,
        help="Maximum number of PDBs to evaluate (for testing, time-split only)"
    )
    benchmark_parser.add_argument(
        "--val-only",
        action="store_true",
        help="Only evaluate validation set (time-split only)"
    )
    benchmark_parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only evaluate test set (time-split only)"
    )
    benchmark_parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only evaluate training set (time-split only)"
    )
    benchmark_parser.add_argument(
        "--pipeline-timeout",
        type=int,
        default=1800,
        help="Timeout in seconds for each individual PDB processing (time-split only)"
    )
    benchmark_parser.set_defaults(func=benchmark_command)
    
    return parser

def set_log_level(level_name):
    """Set the logging level based on the provided name."""
    level = getattr(logging, level_name)
    logger.setLevel(level)
    # Also set for root logger
    logging.getLogger().setLevel(level)

def detect_query_type(query_path):
    """Detect input type based on file extension."""
    ext = os.path.splitext(query_path)[1].lower()
    if ext == '.pdb':
        return 'protein'
    elif ext == '.npz':
        return 'embedding'
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdb or .npz")

def load_query_embedding(query_path):
    """Load embedding from either PDB or NPZ file."""
    np = _lazy_import_numpy()
    (EmbeddingManager, get_protein_embedding, get_protein_sequence,
     find_mcs, constrained_embed, select_best, rmsd_raw, 
     generate_properties_for_sdf) = _lazy_import_core()
    
    query_type = detect_query_type(query_path)
    
    if query_type == 'protein':
        # Generate embedding from PDB
        logger.info(f"Generating embedding from PDB file: {query_path}")
        embedding, chains = get_protein_embedding(query_path)
        query_id = os.path.splitext(os.path.basename(query_path))[0]
        if query_id.endswith('_protein'):
            query_id = query_id[:-8]  # Remove '_protein' suffix
        return embedding, query_id
    
    elif query_type == 'embedding':
        # Load pre-computed embedding
        logger.info(f"Loading pre-computed embedding from: {query_path}")
        try:
            data = np.load(query_path)
            
            # Validate structure
            if 'embeddings' not in data or 'pdb_ids' not in data:
                raise ValueError("Invalid NPZ structure: missing 'embeddings' or 'pdb_ids'")
            
            if data['embeddings'].shape[1] != 1280:
                raise ValueError(f"Embedding dimension mismatch: expected 1280, got {data['embeddings'].shape[1]}")
            
            embedding = data['embeddings'][0]  # Shape: (1280,)
            query_id = str(data['pdb_ids'][0])
            return embedding, query_id
            
        except Exception as e:
            raise ValueError(f"Failed to load embedding: {str(e)}")

def embed_command(args):
    """Generate embedding for a protein."""
    # Lazy import dependencies
    (EmbeddingManager, get_protein_embedding, get_protein_sequence,
     find_mcs, constrained_embed, select_best, rmsd_raw, 
     generate_properties_for_sdf) = _lazy_import_core()
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
    
    logger.info(f"Extracted sequence of length {len(sequence)} from chains {', '.join(chains)}")
    
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
    if pdb_id.endswith('_protein'):
        pdb_id = pdb_id[:-8]  # Remove '_protein' suffix
    
    # Save in format compatible with EmbeddingManager
    np.savez_compressed(
        output_file,
        pdb_ids=np.array([pdb_id]),
        embeddings=np.array([embedding]),
        chain_ids=np.array([','.join(chains)])
    )
    
    logger.info(f"Embedding saved successfully (shape: {embedding.shape})")
    return 0

def find_templates_command(args):
    """Find similar protein templates."""
    # Lazy import dependencies
    (EmbeddingManager, get_protein_embedding, get_protein_sequence,
     find_mcs, constrained_embed, select_best, rmsd_raw, 
     generate_properties_for_sdf) = _lazy_import_core()
    
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
        logger.warning("Embedding database is empty - this embedding file may be a single protein embedding")
        logger.info("For template search, you need a database of multiple protein embeddings")
    
    # Find templates
    logger.info("Finding similar templates")
    templates = manager.find_neighbors(
        query_id,
        query_embedding=query_embedding,
        exclude_uniprot_ids=exclude_uniprot,
        k=num_templates if similarity_threshold is None else None,
        similarity_threshold=similarity_threshold,
        return_similarities=True
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
        # Initialize pipeline
        pipeline = TEMPLPipeline(
            embedding_path=getattr(args, 'embedding_file', None),
            output_dir=args.output_dir
        )
        
        # Load template molecules
        template_mols = pipeline.load_template_molecules([args.template_pdb])
        if not template_mols:
            logger.error(f"Template ligand not found for PDB {args.template_pdb}")
            return 1
        
        # Prepare query molecule
        query_mol = pipeline.prepare_query_molecule(
            ligand_smiles=args.ligand_smiles,
            ligand_file=args.ligand_file
        )
        
        # Generate poses
        results = pipeline.generate_poses(
            query_mol=query_mol,
            template_mols=template_mols,
            num_conformers=args.num_conformers,
            n_workers=args.workers
        )
        
        # Save results
        output_file = pipeline.save_results(results, args.template_pdb)
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
    
    logger.info("Running full TEMPL pipeline")
    
    try:
        # Initialize pipeline
        pipeline = TEMPLPipeline(
            embedding_path=getattr(args, 'embedding_file', None),
            output_dir=args.output_dir,
            run_id=getattr(args, 'run_id', None)
        )
        
        # Run pipeline
        results = pipeline.run_full_pipeline(
            protein_file=getattr(args, 'protein_file', None),
            protein_pdb_id=getattr(args, 'protein_pdb_id', None),
            ligand_smiles=getattr(args, 'ligand_smiles', None),
            ligand_file=getattr(args, 'ligand_file', None),
            num_templates=getattr(args, 'num_templates', 100),
            num_conformers=getattr(args, 'num_conformers', 100),
            n_workers=getattr(args, 'workers', 4),
            similarity_threshold=getattr(args, 'similarity_threshold', None)
        )
        
        # Report results
        logger.info("Pipeline completed successfully!")
        logger.info(f"Found {len(results.get('templates', []))} templates")
        logger.info(f"Generated {len(results.get('poses', {}))} poses")
        logger.info(f"Results saved to: {results.get('output_file', 'unknown')}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
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

def benchmark_command(args):
    """Execute selected benchmark suite with minimal elegant progress display."""
    if args.suite == "polaris":
        try:
            from templ_pipeline.benchmark.polaris.benchmark import main as benchmark_main
            
            # Setup quiet logging for benchmark unless verbose requested
            if not hasattr(args, 'verbose') or not args.verbose:
                # Use WARNING level to minimize output, benchmark has its own progress system
                original_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.WARNING)
            
            # Convert CLI args to benchmark args
            benchmark_args = []
            if hasattr(args, 'n_workers') and args.n_workers:
                benchmark_args.extend(["--n-workers", str(args.n_workers)])
            if hasattr(args, 'n_conformers') and args.n_conformers:
                benchmark_args.extend(["--n-conformers", str(args.n_conformers)])
            if hasattr(args, 'quick') and args.quick:
                benchmark_args.append("--quick")
            if hasattr(args, 'verbose') and args.verbose:
                benchmark_args.extend(["--log-level", "DEBUG"])
            else:
                benchmark_args.extend(["--log-level", "WARNING"])
            
            result = benchmark_main(benchmark_args)
            
            # Restore original logging level
            if not hasattr(args, 'verbose') or not args.verbose:
                logging.getLogger().setLevel(original_level)
                
            return result
        except ImportError as e:
            logger.error(f"Polaris benchmark module not available: {e}")
            return 1
    
    elif args.suite == "time-split":
        try:
            from templ_pipeline.benchmark.timesplit import run_timesplit_benchmark
            
            # Determine which splits to run
            splits_to_run = []
            if hasattr(args, 'train_only') and args.train_only:
                splits_to_run = ["train"]
            elif hasattr(args, 'val_only') and args.val_only:
                splits_to_run = ["val"]
            elif hasattr(args, 'test_only') and args.test_only:
                splits_to_run = ["test"]
            else:
                splits_to_run = ["train", "val", "test"]
            
            # Setup quiet logging for benchmark unless verbose requested
            quiet_mode = not (hasattr(args, 'verbose') and args.verbose)
            if quiet_mode:
                # Use WARNING level to minimize output, benchmark has its own progress system
                original_level = logging.getLogger().level
                logging.getLogger().setLevel(logging.WARNING)
            
            # Convert CLI args to function kwargs
            kwargs = {
                'splits_to_run': splits_to_run,
                'quiet': quiet_mode
            }
            
            if hasattr(args, 'n_workers') and args.n_workers:
                kwargs['n_workers'] = args.n_workers
            if hasattr(args, 'n_conformers') and args.n_conformers:
                kwargs['n_conformers'] = args.n_conformers
            if hasattr(args, 'template_knn') and args.template_knn:
                kwargs['template_knn'] = args.template_knn
            if hasattr(args, 'max_pdbs') and args.max_pdbs:
                kwargs['max_pdbs'] = args.max_pdbs
            
            result = run_timesplit_benchmark(**kwargs)
            
            # Restore original logging level
            if quiet_mode:
                logging.getLogger().setLevel(original_level)
                
            return 0 if result else 1
        except ImportError as e:
            logger.error(f"Time-split benchmark module not available: {e}")
            return 1
    
    else:
        logger.error(f"Unknown benchmark suite: {args.suite}")
        return 1

def main():
    """Main entry point for the CLI."""
    try:
        parser = setup_parser()
        args = parser.parse_args()
        
        # Handle custom help system
        if hasattr(args, 'help') and args.help is not None:
            try:
                from .help_system import show_enhanced_help
            except ImportError:
                # Fallback for direct execution - import directly from file
                import importlib.util
                current_dir = os.path.dirname(__file__)
                help_file = os.path.join(current_dir, 'help_system.py')
                
                # Load help_system module directly
                spec = importlib.util.spec_from_file_location("help_system", help_file)
                help_system = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(help_system)
                show_enhanced_help = help_system.show_enhanced_help
            
            show_enhanced_help(args.help)
            return 0
        
        # Validate arguments
        if hasattr(args, 'ligand_smiles') and args.ligand_smiles:
            valid, msg = validate_smiles(args.ligand_smiles)
            if not valid:
                logger.error(f"Invalid SMILES: {msg}")
                return 2
        
        if hasattr(args, 'num_conformers') and args.num_conformers is not None:
            if args.num_conformers <= 0:
                logger.error("Number of conformers must be positive")
                return 2
        
        if hasattr(args, 'workers') and args.workers is not None:
            if args.workers <= 0:
                logger.error("Number of workers must be positive")
                return 2
        
        if hasattr(args, 'similarity_threshold') and args.similarity_threshold is not None:
            if not (0.0 <= args.similarity_threshold <= 1.0):
                logger.error("Similarity threshold must be between 0.0 and 1.0")
                return 2
        
        # Check file existence
        if hasattr(args, 'protein_file') and args.protein_file:
            if not os.path.exists(args.protein_file):
                logger.error(f"Protein file not found: {args.protein_file}")
                return 2
        
        if hasattr(args, 'query') and args.query:
            if not os.path.exists(args.query):
                logger.error(f"Query file not found: {args.query}")
                return 2
            # Validate file type
            try:
                detect_query_type(args.query)
            except ValueError as e:
                logger.error(str(e))
                return 2
        
        if hasattr(args, 'ligand_file') and args.ligand_file:
            if not os.path.exists(args.ligand_file):
                logger.error(f"Ligand file not found: {args.ligand_file}")
                return 2
        
        # Set up logging
        set_log_level(args.log_level)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
    
            # Handle commands
        if args.command == "embed":
            return embed_command(args)
        elif args.command == "find-templates":
            return find_templates_command(args)
        elif args.command == "generate-poses":
            return generate_poses_command(args)
        elif args.command == "run":
            return run_command(args)
        elif args.command == "benchmark":
            return benchmark_command(args)
        else:
            try:
                from .help_system import show_enhanced_help
            except ImportError:
                # Fallback for direct execution - import directly from file
                import importlib.util
                current_dir = os.path.dirname(__file__)
                help_file = os.path.join(current_dir, 'help_system.py')
                
                # Load help_system module directly
                spec = importlib.util.spec_from_file_location("help_system", help_file)
                help_system = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(help_system)
                show_enhanced_help = help_system.show_enhanced_help
            
            show_enhanced_help("main")
            return 1
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
