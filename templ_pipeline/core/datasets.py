"""
TEMPL Pipeline Dataset Utilities

This module provides utilities for handling dataset splits and filtering:
1. Loading time-split datasets (train, validation, test)
2. Filtering templates by dataset split
3. Supporting benchmarking with time-based train/test separation
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class DatasetSplits:
    """Handles dataset splits for training, validation, and testing.

    This class loads and manages time-split dataset files, which contain
    PDB IDs separated into training, validation, and test sets based on
    deposition dates.

    Attributes:
        splits_dir: Directory containing the split files
        train_pdbs: Set of PDB IDs in the training set
        val_pdbs: Set of PDB IDs in the validation set
        test_pdbs: Set of PDB IDs in the test set
    """

    def __init__(self, splits_dir: Optional[str] = None):
        """Initialize dataset splits from files.

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

            for dir_path in potential_dirs:
                if os.path.exists(dir_path):
                    splits_dir = dir_path
                    break

            if splits_dir is None:
                logger.warning(
                    "Could not find splits directory. Using default path."
                )
                splits_dir = os.path.join(os.getcwd(), "data", "splits")

        self.splits_dir = splits_dir
        logger.debug(f"Using splits directory: {self.splits_dir}")

        # Load all splits
        self.train_pdbs = self._load_split("timesplit_train")
        self.val_pdbs = self._load_split("timesplit_val")
        self.test_pdbs = self._load_split("timesplit_test")

        # Log split sizes
        logger.info(
            f"Loaded dataset splits: train={len(self.train_pdbs)}, "
            f"val={len(self.val_pdbs)}, test={len(self.test_pdbs)}"
        )

    def _load_split(self, filename: str) -> Set[str]:
        """Load PDB IDs from a split file.

        Args:
            filename: Name of the split file

        Returns:
            Set of PDB IDs (lowercase)
        """
        path = os.path.join(self.splits_dir, filename)
        if not os.path.exists(path):
            logger.warning(f"Split file not found: {path}")
            return set()

        with open(path) as f:
            # Strip whitespace and convert to lowercase for consistent matching
            return {line.strip().lower() for line in f if line.strip()}

    def get_split(self, split_name: str) -> Set[str]:
        """Get PDB IDs for a specific split.

        Args:
            split_name: Name of the split ('train', 'val', or 'test')

        Returns:
            Set of PDB IDs in the split

        Raises:
            ValueError: If split_name is not recognized
        """
        split_name = split_name.lower()
        if split_name == "train":
            return self.train_pdbs
        elif split_name in ["val", "validation"]:
            return self.val_pdbs
        elif split_name == "test":
            return self.test_pdbs
        else:
            raise ValueError(
                f"Unknown split name: {split_name}. "
                f"Use 'train', 'val', or 'test'."
            )

    def is_in_split(self, pdb_id: str, split_name: str) -> bool:
        """Check if a PDB ID is in a specific split.

        Args:
            pdb_id: PDB ID to check
            split_name: Name of the split ('train', 'val', or 'test')

        Returns:
            True if the PDB ID is in the specified split, False otherwise
        """
        return pdb_id.lower() in self.get_split(split_name)

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the dataset splits.

        Returns:
            Dictionary with counts for each split
        """
        return {
            "train": len(self.train_pdbs),
            "val": len(self.val_pdbs),
            "test": len(self.test_pdbs),
            "total": (
                len(self.train_pdbs) + len(self.val_pdbs) + len(self.test_pdbs)
            ),
        }
