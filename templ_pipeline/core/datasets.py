"""Dataset utilities for time-split management in the TEMPL pipeline.

This module provides utilities for handling dataset splits and filtering:
- Loading time-split datasets (train, validation, test)
- Filtering templates by dataset split
- Supporting benchmarking with time-based train/test separation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Set, Union


logger = logging.getLogger(__name__)


class DatasetSplits:
    """Handle dataset splits for training, validation, and testing.

    This class loads and manages time-split dataset files, which contain PDB
    IDs separated into training, validation, and test sets based on deposition
    dates.

    Attributes:
        splits_dir: Directory containing the split files
        train_pdbs: Set of PDB IDs in the training set
        val_pdbs: Set of PDB IDs in the validation set
        test_pdbs: Set of PDB IDs in the test set
    """

    def __init__(self, splits_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize dataset splits from files.

        Args:
            splits_dir: Directory containing the split files. If None, multiple
                common locations are checked and the first that exists is used.
        """
        resolved_dir = self._resolve_splits_dir(splits_dir)
        self.splits_dir: Path = resolved_dir
        logger.debug(f"Using splits directory: {self.splits_dir}")

        # Load all splits
        self.train_pdbs: Set[str] = self._load_split("timesplit_train")
        self.val_pdbs: Set[str] = self._load_split("timesplit_val")
        self.test_pdbs: Set[str] = self._load_split("timesplit_test")

        logger.info(
            "Loaded dataset splits: train=%d, val=%d, test=%d",
            len(self.train_pdbs),
            len(self.val_pdbs),
            len(self.test_pdbs),
        )

    def _resolve_splits_dir(
        self, splits_dir: Optional[Union[str, Path]]
    ) -> Path:
        """Resolve the splits directory using sensible defaults.

        Preference order when `splits_dir` is not provided:
        1) <cwd>/data/splits
        2) <cwd>/templ_pipeline/data/splits
        3) <project_root>/data/splits (derived from this file's location)
        """
        if splits_dir is not None:
            return Path(splits_dir)

        potential_dirs = [
            Path.cwd() / "data" / "splits",
            Path.cwd() / "templ_pipeline" / "data" / "splits",
            Path(__file__).resolve().parents[2] / "data" / "splits",
        ]

        for candidate in potential_dirs:
            if candidate.exists():
                return candidate

        logger.warning(
            "Could not find splits directory in any common location. "
            "Falling back to default path: %s",
            potential_dirs[0],
        )
        return potential_dirs[0]

    def _load_split(self, filename: str) -> Set[str]:
        """Load PDB IDs from a split file.

        Args:
            filename: Name of the split file

        Returns:
            Set of PDB IDs (lowercase)
        """
        path = self.splits_dir / filename
        if not path.exists():
            logger.warning("Split file not found: %s", path)
            return set()

        try:
            with path.open("r", encoding="utf-8") as file:
                return {line.strip().lower() for line in file if line.strip()}
        except OSError as exc:
            logger.error("Failed to read split file %s: %s", path, exc)
            return set()

    def get_split(self, split_name: str) -> Set[str]:
        """Get PDB IDs for a specific split.

        Args:
            split_name: Name of the split ("train", "val", or "test"). The
                alias "validation" is also accepted.

        Returns:
            Set of PDB IDs in the split.

        Raises:
            ValueError: If `split_name` is not recognized.
        """
        name = split_name.lower()
        if name == "train":
            return self.train_pdbs
        if name in {"val", "validation"}:
            return self.val_pdbs
        if name == "test":
            return self.test_pdbs

        raise ValueError("Unknown split name: %s. Use 'train', 'val', or 'test'." % split_name)

    def is_in_split(self, pdb_id: str, split_name: str) -> bool:
        """Check if a PDB ID is in a specific split."""
        return pdb_id.lower() in self.get_split(split_name)

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about the dataset splits."""
        return {
            "train": len(self.train_pdbs),
            "val": len(self.val_pdbs),
            "test": len(self.test_pdbs),
            "total": len(self.train_pdbs) + len(self.val_pdbs) + len(self.test_pdbs),
        }
