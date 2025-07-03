#!/usr/bin/env python
"""
Time-Split Example for TEMPL Pipeline

This script demonstrates how to use the time-split datasets for template selection in the TEMPL pipeline.
It shows:
1. Loading the time-split datasets
2. Finding templates for a test protein using only the training set
3. Visualizing the similarity distribution of templates
"""

import os
import sys
import logging
import argparse
import random
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

from templ_pipeline.core import EmbeddingManager, DatasetSplits, get_embedding

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("time-split-example")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Time-split example for TEMPL pipeline."
    )

    parser.add_argument(
        "--embedding-path",
        type=str,
        help="Path to pre-computed embeddings (NPZ file)",
    )

    parser.add_argument(
        "--splits-dir",
        type=str,
        help="Directory containing time-split files",
    )

    parser.add_argument(
        "--test-pdb",
        type=str,
        help="Specific test PDB ID to use (randomly selected if not provided)",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=50,
        help="Number of templates to find (default: 50)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="time_split_results",
        help="Directory to save results (default: time_split_results)",
    )

    return parser.parse_args()


def find_templates_for_test_protein(
    test_pdb: str,
    dataset_splits: DatasetSplits,
    embedding_manager: EmbeddingManager,
    k: int = 50,
    output_dir: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """Find templates for a test protein using only the training set.

    Args:
        test_pdb: PDB ID of the test protein
        dataset_splits: DatasetSplits object
        embedding_manager: EmbeddingManager object
        k: Number of templates to find
        output_dir: Directory to save results

    Returns:
        List of (template_pdb, similarity) tuples
    """
    # Verify the PDB is in the test set
    if not dataset_splits.is_in_split(test_pdb, "test"):
        logger.warning(
            f"PDB {test_pdb} is not in the test set. Results may not be meaningful."
        )

    # Get the embedding for the test protein
    test_embedding, test_chains = embedding_manager.get_embedding(test_pdb)
    if test_embedding is None:
        logger.error(f"Could not get embedding for test PDB {test_pdb}")
        return []

    # Find templates using only the training set
    templates = embedding_manager.find_neighbors_in_split(
        query_pdb=test_pdb,
        split_name="train",
        dataset_splits=dataset_splits,
        k=k,
        return_similarities=True,
    )

    logger.info(f"Found {len(templates)} templates for {test_pdb} from training set")

    # Save results if output directory is provided
    if output_dir and templates:
        os.makedirs(output_dir, exist_ok=True)

        # Save list of templates with similarities
        with open(os.path.join(output_dir, f"{test_pdb}_templates.tsv"), "w") as f:
            f.write("template_pdb\tsimilarity\n")
            for template_pdb, similarity in templates:
                f.write(f"{template_pdb}\t{similarity:.4f}\n")

        # Create a histogram of similarities
        similarities = [sim for _, sim in templates]
        plt.figure(figsize=(10, 6))
        plt.hist(similarities, bins=20, alpha=0.7)
        plt.xlabel("Embedding Similarity")
        plt.ylabel("Number of Templates")
        plt.title(f"Template Similarity Distribution for {test_pdb}")
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, f"{test_pdb}_similarity_dist.png"))
        plt.close()

        logger.info(f"Saved results to {output_dir}")

    return templates


def run_time_split_example(args: argparse.Namespace) -> None:
    """Run the time-split example."""
    # Initialize dataset splits
    dataset_splits = DatasetSplits(args.splits_dir)

    # Print dataset statistics
    stats = dataset_splits.get_statistics()
    logger.info(f"Dataset statistics:")
    logger.info(f"  Training set: {stats['train']} PDBs")
    logger.info(f"  Validation set: {stats['val']} PDBs")
    logger.info(f"  Test set: {stats['test']} PDBs")

    # Find embeddings path if not provided
    embedding_path = args.embedding_path
    if not embedding_path:
        potential_paths = [
            "data/embeddings/protein_embeddings_base.npz",
            "templ_pipeline/data/embeddings/protein_embeddings_base.npz",
            "mcs_bench/data/protein_embeddings_base.npz",
        ]
        for path in potential_paths:
            if os.path.exists(path):
                embedding_path = path
                break

    if not embedding_path or not os.path.exists(embedding_path):
        logger.error("Could not find embedding file. Please provide --embedding-path.")
        return

    logger.info(f"Using embedding file: {embedding_path}")

    # Initialize embedding manager
    embedding_manager = EmbeddingManager(embedding_path)

    # Select test PDB
    test_pdb = args.test_pdb
    if not test_pdb:
        # Select a random PDB from the test set
        test_pdbs = list(dataset_splits.test_pdbs)
        if not test_pdbs:
            logger.error("No test PDBs found in the dataset splits.")
            return

        test_pdb = random.choice(test_pdbs)
        logger.info(f"Randomly selected test PDB: {test_pdb}")

    # Find templates for the test protein
    templates = find_templates_for_test_protein(
        test_pdb=test_pdb,
        dataset_splits=dataset_splits,
        embedding_manager=embedding_manager,
        k=args.k,
        output_dir=args.output_dir,
    )

    # Print top templates
    if templates:
        logger.info(f"Top {min(10, len(templates))} templates for {test_pdb}:")
        for i, (template_pdb, similarity) in enumerate(templates[:10], 1):
            logger.info(f"  {i}. {template_pdb}: {similarity:.4f}")
    else:
        logger.warning(f"No templates found for {test_pdb}")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        run_time_split_example(args)
        return 0
    except Exception as e:
        logger.exception(f"Error running time-split example: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
