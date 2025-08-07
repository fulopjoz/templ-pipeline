"""
Dataset management utilities for TEMPL pipeline.

This module provides functionality for:
- Loading time-split datasets
- Managing dataset splits (train/validation/test)
- Filtering templates by dataset membership
- Supporting benchmarking operations
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class DatasetSplits:
    """
    Handles dataset splits for training, validation, and testing.

    This class loads and manages time-split dataset files, which contain
    PDB IDs separated into training, validation, and test sets based on
    deposition dates.
    """

    def __init__(self, splits_dir: Optional[str] = None):
        """
        Initialize dataset splits from files.

        Args:
            splits_dir: Directory containing the split files.
                If None, uses default location (data/splits)
        """
        # Determine splits directory with proper path resolution
        if splits_dir is None:
            # Try multiple potential locations
            potential_dirs = [
                os.path.join(os.getcwd(), "data", "splits"),
                os.path.join(os.getcwd(), "templ_pipeline", "data", "splits"),
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                    "data",
                    "splits",
                ),
            ]

            for potential_dir in potential_dirs:
                if os.path.exists(potential_dir):
                    splits_dir = potential_dir
                    break

            if splits_dir is None:
                splits_dir = potential_dirs[0]  # Use first as default
                logger.warning(
                    f"Splits directory not found, using default: {splits_dir}"
                )

        self.splits_dir = Path(splits_dir)
        self.train_pdbs = set()
        self.val_pdbs = set()
        self.test_pdbs = set()

        # Load splits
        self._load_splits()

    def _load_splits(self) -> None:
        """Load dataset splits from files."""
        split_files = {
            "train": "train_pdbs.txt",
            "val": "val_pdbs.txt",
            "test": "test_pdbs.txt"
        }

        for split_name, filename in split_files.items():
            filepath = self.splits_dir / filename
            pdbs = self._load_pdb_list(filepath)

            if split_name == "train":
                self.train_pdbs = pdbs
            elif split_name == "val":
                self.val_pdbs = pdbs
            elif split_name == "test":
                self.test_pdbs = pdbs

            logger.info(f"Loaded {len(pdbs)} PDB IDs for {split_name} split")

    def _load_pdb_list(self, filepath: Path) -> Set[str]:
        """Load PDB IDs from a text file."""
        if not filepath.exists():
            logger.warning(f"Split file not found: {filepath}")
            return set()

        try:
            with open(filepath, 'r') as f:
                pdbs = {line.strip().lower() for line in f if line.strip()}
            return pdbs
        except Exception as e:
            logger.error(f"Failed to load PDB list from {filepath}: {e}")
            return set()

    def get_split_pdbs(self, split_name: str) -> Set[str]:
        """
        Get PDB IDs for a specific split.

        Args:
            split_name: One of 'train', 'val', 'test'

        Returns:
            Set of PDB IDs for the specified split
        """
        if split_name == "train":
            return self.train_pdbs
        elif split_name in ["val", "validation"]:
            return self.val_pdbs
        elif split_name == "test":
            return self.test_pdbs
        else:
            raise ValueError(f"Unknown split name: {split_name}")

    def get_split_statistics(self) -> Dict[str, int]:
        """Get statistics about dataset splits."""
        return {
            "train_count": len(self.train_pdbs),
            "val_count": len(self.val_pdbs),
            "test_count": len(self.test_pdbs),
            "total_count": (
                len(self.train_pdbs) + len(self.val_pdbs) + len(self.test_pdbs)
            ),
        }

    def validate_splits(self) -> Dict[str, Any]:
        """Validate dataset splits for completeness and non-overlap."""
        # Check for overlaps
        train_val_overlap = self.train_pdbs.intersection(self.val_pdbs)
        train_test_overlap = self.train_pdbs.intersection(self.test_pdbs)
        val_test_overlap = self.val_pdbs.intersection(self.test_pdbs)

        has_overlaps = (
            len(train_val_overlap) > 0 or
            len(train_test_overlap) > 0 or
            len(val_test_overlap) > 0
        )

        # Check for empty splits
        empty_splits = []
        if len(self.train_pdbs) == 0:
            empty_splits.append("train")
        if len(self.val_pdbs) == 0:
            empty_splits.append("val")
        if len(self.test_pdbs) == 0:
            empty_splits.append("test")

        return {
            "valid": not has_overlaps and len(empty_splits) == 0,
            "has_overlaps": has_overlaps,
            "empty_splits": empty_splits,
            "overlaps": {
                "train_val": list(train_val_overlap),
                "train_test": list(train_test_overlap),
                "val_test": list(val_test_overlap),
            },
            "statistics": self.get_split_statistics(),
        }


class DatasetManager:
    """
    Comprehensive dataset management for TEMPL pipeline.

    Handles loading, validation, and filtering of datasets for benchmarking
    and training purposes.
    """

    def __init__(
        self,
        data_dir: str = "data",
        splits_dir: Optional[str] = None,
        embedding_path: Optional[str] = None,
        ligands_path: Optional[str] = None
    ):
        """
        Initialize dataset manager.

        Args:
            data_dir: Base data directory
            splits_dir: Directory containing split files
            embedding_path: Path to embeddings file
            ligands_path: Path to ligands file
        """
        self.data_dir = Path(data_dir)

        # Set default paths
        if embedding_path is None:
            embedding_path = str(
                self.data_dir / "embeddings" /
                "templ_protein_embeddings_v1.0.0.npz"
            )
        if ligands_path is None:
            ligands_path = str(
                self.data_dir / "ligands" /
                "templ_processed_ligands_v1.0.0.sdf.gz"
            )

        self.embedding_path = Path(embedding_path)
        self.ligands_path = Path(ligands_path)

        # Initialize splits if directory provided
        self.splits = DatasetSplits(splits_dir) if splits_dir else None

        # Cached data
        self._available_pdbs = None

    def get_available_pdbs(self) -> Set[str]:
        """Get set of PDB IDs that have embeddings available."""
        if self._available_pdbs is None:
            try:
                if self.embedding_path.exists():
                    data = np.load(self.embedding_path, allow_pickle=True)
                    self._available_pdbs = set(data["pdb_ids"])
                    logger.info(
                        f"Found {len(self._available_pdbs)} available PDB embeddings"
                    )
                else:
                    logger.warning(f"Embedding file not found: {self.embedding_path}")
                    self._available_pdbs = set()
            except Exception as e:
                logger.error(f"Failed to load available PDB IDs: {e}")
                self._available_pdbs = set()

        return self._available_pdbs

    def filter_pdbs_by_availability(self, pdb_ids: List[str]) -> List[str]:
        """Filter PDB IDs to only include those with available embeddings."""
        available_pdbs = self.get_available_pdbs()
        return [pid for pid in pdb_ids if pid.lower() in available_pdbs]

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        stats = {
            "total_available_pdbs": len(self.get_available_pdbs()),
            "files": {
                "embeddings_exists": self.embedding_path.exists(),
                "ligands_exists": self.ligands_path.exists(),
            }
        }

        if self.embedding_path.exists():
            stats["files"]["embeddings_size_mb"] = (
                self.embedding_path.stat().st_size / (1024 * 1024)
            )

        if self.ligands_path.exists():
            stats["files"]["ligands_size_mb"] = (
                self.ligands_path.stat().st_size / (1024 * 1024)
            )

        if self.splits:
            split_stats = self.splits.get_split_statistics()
            validation_result = self.splits.validate_splits()
            stats["splits"] = {
                **split_stats,
                "validation": validation_result
            }

        return stats


def load_benchmark_pdbs(
    benchmark_name: str,
    data_dir: str = "data",
    max_pdbs: Optional[int] = None
) -> Set[str]:
    """
    Load PDB IDs for a specific benchmark.

    Args:
        benchmark_name: Name of benchmark ("polaris", "time_split", "all")
        data_dir: Base data directory
        max_pdbs: Optional limit on number of PDB IDs

    Returns:
        Set of PDB IDs for the benchmark
    """
    manager = DatasetManager(data_dir)

    if benchmark_name == "polaris":
        # For Polaris benchmark, use test split if available
        if manager.splits:
            pdbs = manager.splits.get_split_pdbs("test")
        else:
            pdbs = manager.get_available_pdbs()
    elif benchmark_name == "time_split":
        # For time-split benchmark, use test split
        if manager.splits:
            pdbs = manager.splits.get_split_pdbs("test")
        else:
            logger.warning("No splits available for time-split benchmark")
            pdbs = manager.get_available_pdbs()
    elif benchmark_name == "all":
        pdbs = manager.get_available_pdbs()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    # Filter to only available PDbs
    available_pdbs = manager.filter_pdbs_by_availability(list(pdbs))

    if max_pdbs and len(available_pdbs) > max_pdbs:
        # Random sampling for subset
        np.random.seed(42)  # For reproducibility
        sampled_pdbs = np.random.choice(
            available_pdbs,
            size=max_pdbs,
            replace=False
        )
        return set(sampled_pdbs)

    return set(available_pdbs)


def validate_dataset_integrity(data_dir: str = "data") -> bool:
    """
    Quick validation of dataset integrity.

    Args:
        data_dir: Base data directory

    Returns:
        True if dataset passes basic integrity checks
    """
    manager = DatasetManager(data_dir)

    # Check file existence
    if not manager.embedding_path.exists():
        logger.error(f"Embedding file not found: {manager.embedding_path}")
        return False

    if not manager.ligands_path.exists():
        logger.error(f"Ligands file not found: {manager.ligands_path}")
        return False

    # Check if we can load embeddings
    try:
        available_pdbs = manager.get_available_pdbs()
        if len(available_pdbs) == 0:
            logger.error("No PDB IDs found in embeddings")
            return False
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return False

    # Check splits if available
    if manager.splits:
        split_validation = manager.splits.validate_splits()
        if not split_validation["valid"]:
            logger.warning("Dataset splits validation failed")
            # Don't return False here as splits are optional

    logger.info("Dataset integrity check passed")
    return True