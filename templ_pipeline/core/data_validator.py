"""
Data validation utilities for ensuring consistency between splits and embeddings.
"""

import os
import logging
from typing import Set, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class SplitDataValidator:
    """Validates split data against available embeddings."""

    def __init__(self, embedding_path: str, splits_dir: str):
        self.embedding_path = embedding_path
        self.splits_dir = splits_dir
        self._available_pdbs = None
        self._validated_splits = None

    def _load_available_pdbs(self) -> Set[str]:
        """Load available PDB IDs from embedding database."""
        if self._available_pdbs is None:
            try:
                data = np.load(self.embedding_path, allow_pickle=True)
                self._available_pdbs = set(data["pdb_ids"])
                logger.info(
                    f"Loaded {len(self._available_pdbs)} available PDB embeddings"
                )
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
                self._available_pdbs = set()
        return self._available_pdbs

    def _load_split_file(self, filename: str) -> Set[str]:
        """Load PDB IDs from split file."""
        filepath = os.path.join(self.splits_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"Split file not found: {filepath}")
            return set()

        with open(filepath) as f:
            return {line.strip().lower() for line in f if line.strip()}

    def get_validated_splits(self) -> Dict[str, Set[str]]:
        """Get validated splits with only available PDB IDs."""
        if self._validated_splits is None:
            available_pdbs = self._load_available_pdbs()

            splits = {
                "train": self._load_split_file("timesplit_train"),
                "val": self._load_split_file("timesplit_val"),
                "test": self._load_split_file("timesplit_test"),
            }

            self._validated_splits = {}
            for split_name, split_pdbs in splits.items():
                validated = split_pdbs.intersection(available_pdbs)
                coverage = len(validated) / len(split_pdbs) * 100 if split_pdbs else 0

                self._validated_splits[split_name] = validated
                logger.info(
                    f"{split_name}: {len(validated)}/{len(split_pdbs)} PDBs ({coverage:.1f}% coverage)"
                )

        return self._validated_splits

    def get_first_available_test_pdb(self) -> Optional[str]:
        """Get first available test PDB ID."""
        splits = self.get_validated_splits()
        test_pdbs = list(splits.get("test", set()))
        return test_pdbs[0] if test_pdbs else None

    def filter_pdbs_by_availability(self, pdb_ids: Set[str]) -> Set[str]:
        """Filter PDB IDs to only include those with available embeddings."""
        available_pdbs = self._load_available_pdbs()
        return pdb_ids.intersection(available_pdbs)
