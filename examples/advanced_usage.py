# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Advanced usage examples for the TEMPL pipeline.

This script demonstrates more advanced usage of the TEMPL pipeline,
including a complete workflow for finding protein templates based on
structural similarity and applying various filtering criteria.
"""

import os
import argparse
import logging
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("templ-advanced-example")

# Import TEMPL pipeline modules
from templ_pipeline.core.embedding import (
    EmbeddingManager,
    get_protein_sequence,
    calculate_embedding,
    get_embedding,
    select_templates,
)


def load_uniprot_mapping(mapping_file: str) -> Dict[str, str]:
    """Load PDB to UniProt ID mapping from JSON file.

    Args:
        mapping_file: Path to JSON file with mapping data

    Returns:
        Dictionary mapping PDB IDs to UniProt IDs
    """
    if not os.path.exists(mapping_file):
        logger.warning(f"UniProt mapping file {mapping_file} not found")
        return {}

    pdb_to_uniprot = {}
    try:
        with open(mapping_file) as f:
            data = json.load(f)
            for pdb_id, info in data.items():
                if isinstance(info, dict) and "uniprot" in info:
                    pdb_to_uniprot[pdb_id] = info["uniprot"]
    except Exception as e:
        logger.error(f"Error loading UniProt mapping: {e}")
        return {}

    logger.info(f"Loaded {len(pdb_to_uniprot)} UniProt mappings from {mapping_file}")
    return pdb_to_uniprot


def load_exclude_uniprot(exclude_file: str) -> Set[str]:
    """Load list of UniProt IDs to exclude.

    Args:
        exclude_file: Path to file with UniProt IDs to exclude

    Returns:
        Set of UniProt IDs to exclude
    """
    if not os.path.exists(exclude_file):
        logger.warning(f"Exclude UniProt file {exclude_file} not found")
        return set()

    try:
        with open(exclude_file) as f:
            return {line.strip() for line in f if line.strip()}
    except Exception as e:
        logger.error(f"Error loading UniProt exclusion list: {e}")
        return set()


def find_templates_for_target(
    target_pdb: str,
    embedding_path: str,
    cache_dir: Optional[str] = None,
    pdb_to_uniprot_file: Optional[str] = None,
    exclude_uniprot_file: Optional[str] = None,
    similarity_threshold: float = 0.90,
    max_templates: int = 50,
    output_dir: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """Find suitable templates for a target protein.

    Args:
        target_pdb: Target PDB ID or path to PDB file
        embedding_path: Path to pre-computed embeddings
        cache_dir: Directory for caching embeddings
        pdb_to_uniprot_file: Path to JSON file with PDB to UniProt mapping
        exclude_uniprot_file: Path to file with UniProt IDs to exclude
        similarity_threshold: Minimum similarity threshold
        max_templates: Maximum number of templates to return
        output_dir: Directory to save visualizations

    Returns:
        List of (PDB ID, similarity) tuples for selected templates
    """
    # Initialize EmbeddingManager
    logger.info(f"Initializing EmbeddingManager with {embedding_path}")
    manager = EmbeddingManager(embedding_path, use_cache=True, cache_dir=cache_dir)

    # Load UniProt mapping if provided
    pdb_to_uniprot = {}
    exclude_uniprot = set()

    if pdb_to_uniprot_file:
        pdb_to_uniprot = load_uniprot_mapping(pdb_to_uniprot_file)
        manager.set_uniprot_mapping(pdb_to_uniprot)

        # Load UniProt exclusions if provided
        if exclude_uniprot_file:
            exclude_uniprot = load_exclude_uniprot(exclude_uniprot_file)
            logger.info(f"Loaded {len(exclude_uniprot)} UniProt IDs to exclude")

    # Check if target is a PDB ID or file path
    target_is_file = os.path.exists(target_pdb)
    target_id = Path(target_pdb).stem if target_is_file else target_pdb

    # Get target embedding
    logger.info(f"Getting embedding for target {target_id}")
    if target_is_file:
        target_embedding, _ = manager.get_embedding(target_id, target_pdb)
    else:
        target_embedding, _ = manager.get_embedding(target_id)

    if target_embedding is None:
        logger.error(f"Failed to get embedding for target {target_id}")
        return []

    # Find neighbors
    logger.info(f"Finding template proteins with similarity >= {similarity_threshold}")
    neighbors = manager.find_neighbors(
        target_id,
        target_embedding,
        exclude_uniprot_ids=exclude_uniprot,
        similarity_threshold=similarity_threshold,
        return_similarities=True,
    )

    # Limit number of templates
    if max_templates and len(neighbors) > max_templates:
        logger.info(f"Limiting to top {max_templates} templates by similarity")
        neighbors = neighbors[:max_templates]

    logger.info(f"Found {len(neighbors)} suitable templates")

    # Save visualization if output directory provided
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            visualize_templates(
                target_id, target_embedding, neighbors, manager, output_dir
            )
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")

    return neighbors


def visualize_templates(
    target_id: str,
    target_embedding: np.ndarray,
    neighbors: List[Tuple[str, float]],
    manager: EmbeddingManager,
    output_dir: str,
) -> None:
    """Create visualizations for template selection.

    Args:
        target_id: Target PDB ID
        target_embedding: Target embedding
        neighbors: List of (PDB ID, similarity) tuples
        manager: EmbeddingManager instance
        output_dir: Directory to save visualizations
    """
    # Create similarity distribution plot
    similarities = [sim for _, sim in neighbors]

    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=20, alpha=0.7)
    plt.axvline(
        x=min(similarities),
        color="r",
        linestyle="--",
        label=f"Min similarity: {min(similarities):.3f}",
    )
    plt.xlabel("Similarity Score")
    plt.ylabel("Number of Templates")
    plt.title(f"Template Similarity Distribution for {target_id}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, f"{target_id}_similarity_dist.png"))

    # Create PCA visualization of embeddings
    try:
        # Collect embeddings
        template_ids = [pid for pid, _ in neighbors]
        embeddings = []

        # Add target embedding
        embeddings.append(target_embedding)

        # Add template embeddings
        for pid in template_ids:
            emb, _ = manager.get_embedding(pid)
            if emb is not None:
                embeddings.append(emb)

        if len(embeddings) > 2:  # Need at least 3 points for meaningful visualization
            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(np.vstack(embeddings))

            # Create scatter plot
            plt.figure(figsize=(10, 8))

            # Plot templates
            plt.scatter(
                embeddings_2d[1:, 0],
                embeddings_2d[1:, 1],
                c=similarities,
                cmap="viridis",
                alpha=0.7,
                s=50,
            )

            # Highlight target
            plt.scatter(
                embeddings_2d[0, 0],
                embeddings_2d[0, 1],
                c="red",
                marker="*",
                s=200,
                label=f"Target: {target_id}",
            )

            plt.colorbar(label="Similarity Score")
            plt.title(f"PCA Visualization of Template Embeddings for {target_id}")
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"{target_id}_pca_visualization.png"))

            logger.info(f"Saved visualization to {output_dir}")
    except Exception as e:
        logger.error(f"Error creating PCA visualization: {e}")


def generate_report(
    target_id: str, templates: List[Tuple[str, float]], output_dir: str
) -> None:
    """Generate a report with template selection results.

    Args:
        target_id: Target PDB ID
        templates: List of (PDB ID, similarity) tuples
        output_dir: Directory to save report
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{target_id}_template_report.txt")

    with open(report_path, "w") as f:
        f.write(f"Template Selection Report for {target_id}\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Number of templates found: {len(templates)}\n")
        if templates:
            f.write(
                f"Similarity range: {templates[-1][1]:.4f} - {templates[0][1]:.4f}\n\n"
            )

            f.write("Top 10 templates:\n")
            for i, (pid, sim) in enumerate(templates[:10]):
                f.write(f"{i+1}. {pid} (similarity: {sim:.4f})\n")

            f.write("\nAll templates:\n")
            for pid, sim in templates:
                f.write(f"{pid}\t{sim:.4f}\n")

    logger.info(f"Saved template report to {report_path}")


def main():
    """Run the advanced template selection example."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="TEMPL Pipeline Advanced Example")

    parser.add_argument(
        "--target", required=True, help="Target PDB ID or path to PDB file"
    )
    parser.add_argument(
        "--embeddings", default=None, help="Path to pre-computed embeddings (.npz file)"
    )
    parser.add_argument(
        "--cache-dir", default=None, help="Directory for caching embeddings"
    )
    parser.add_argument(
        "--uniprot-mapping",
        default=None,
        help="Path to JSON file with PDB to UniProt mapping",
    )
    parser.add_argument(
        "--exclude-uniprot",
        default=None,
        help="Path to file with UniProt IDs to exclude",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.90,
        help="Minimum similarity threshold",
    )
    parser.add_argument(
        "--max-templates",
        type=int,
        default=50,
        help="Maximum number of templates to return",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to save results and visualizations",
    )

    args = parser.parse_args()

    # Find embedding path if not provided
    embedding_path = args.embeddings
    if embedding_path is None:
        potential_paths = [
            "data/esm2_embeddings/embeddings.npz",
            "data/embeddings/templ_protein_embeddings_v1.0.0.npz",
            os.path.expanduser("~/.cache/templ/embeddings/embeddings.npz"),
        ]
        for path in potential_paths:
            if os.path.exists(path):
                embedding_path = path
                logger.info(f"Using embedding file: {embedding_path}")
                break
        else:
            logger.error("No embedding file found. Please provide --embeddings")
            return

    # Find templates for target
    templates = find_templates_for_target(
        args.target,
        embedding_path,
        args.cache_dir,
        args.uniprot_mapping,
        args.exclude_uniprot,
        args.similarity_threshold,
        args.max_templates,
        args.output_dir,
    )

    # Generate report
    if templates:
        generate_report(
            Path(args.target).stem if os.path.exists(args.target) else args.target,
            templates,
            args.output_dir,
        )

        logger.info(f"Template selection complete. Results saved to {args.output_dir}")
    else:
        logger.error("No templates found for the target.")


if __name__ == "__main__":
    main()
