#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
TEMPL Pipeline - Main entry point

This script serves as the main entry point for the TEMPL pipeline, providing
a simplified interface for running the pipeline with common options.
"""

import argparse
import logging
import sys
from pathlib import Path

# Import pipeline modules
from templ_pipeline.cli.main import main as cli_main


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TEMPL (Template-based Protein Ligand) pose prediction pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Main inputs
    parser.add_argument(
        "--target-pdb",
        type=str,
        required=True,
        help="PDB ID or path to target protein structure",
    )
    parser.add_argument(
        "--ligand-smiles",
        type=str,
        required=True,
        help="SMILES string of the query ligand",
    )

    # Optional parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--template-knn",
        type=int,
        default=100,
        help="Number of nearest neighbors for template selection",
    )
    parser.add_argument(
        "--n-conformers", type=int, default=200, help="Number of conformers to generate"
    )
    parser.add_argument(
        "--scoring-method",
        type=str,
        default="combo",
        choices=["shape", "color", "combo"],
        help="Scoring method for pose selection",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Maximum number of workers for parallel processing (0 = auto)",
    )

    # Advanced options
    parser.add_argument(
        "--data-dir", type=str, default=None, help="Base directory for data files"
    )
    parser.add_argument(
        "--embeddings-file",
        type=str,
        default=None,
        help="Path to pre-computed embeddings file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    return parser.parse_args()


def setup_logging(log_level):
    """Set up logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Main entry point for the TEMPL pipeline."""
    args = parse_args()
    setup_logging(args.log_level)

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Forward to CLI main by modifying sys.argv
    # The CLI main function parses arguments from sys.argv directly
    import sys
    
    # Build CLI arguments from parsed args
    cli_args = ["templ", "run"]
    
    if args.target_pdb.endswith('.pdb'):
        cli_args.extend(["--protein-file", args.target_pdb])
    else:
        cli_args.extend(["--protein-pdb-id", args.target_pdb])
    
    cli_args.extend(["--ligand-smiles", args.ligand_smiles])
    cli_args.extend(["--output-dir", args.output_dir])
    cli_args.extend(["--num-templates", str(args.template_knn)])
    cli_args.extend(["--num-conformers", str(args.n_conformers)])
    cli_args.extend(["--align-metric", args.scoring_method])
    
    if args.max_workers > 0:
        cli_args.extend(["--workers", str(args.max_workers)])
    
    if args.embeddings_file:
        cli_args.extend(["--embedding-file", args.embeddings_file])
    
    cli_args.extend(["--log-level", args.log_level])
    
    # Replace sys.argv temporarily
    original_argv = sys.argv[:]
    sys.argv = cli_args
    
    try:
        return cli_main()
    finally:
        # Restore original sys.argv
        sys.argv = original_argv


if __name__ == "__main__":
    sys.exit(main())
