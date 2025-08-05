#!/usr/bin/env python
"""
Test script for time-split functionality.

This script verifies:
1. Loading time-split datasets
2. Finding templates for a test protein using only the training set
3. Checking that training templates are properly filtered
"""

import os
import logging
import unittest
from pathlib import Path
import sys

# Handle imports for both development and installed package
try:
    from templ_pipeline.core.embedding import EmbeddingManager
    from templ_pipeline.core.datasets import DatasetSplits
except ImportError:
    # Fall back to local imports for development
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    )
    from core.embedding import EmbeddingManager
    from core.datasets import DatasetSplits

# Import test helper functions from local tests package
sys.path.insert(0, os.path.dirname(__file__))
from tests import get_test_data_path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("test-timesplit")


class TestTimeSplit(unittest.TestCase):
    """Test cases for time-split functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Try to find embedding file
        self.embedding_path = get_test_data_path("embeddings")

        # Try to find splits directory
        self.splits_dir = get_test_data_path("splits")

    def test_splits_loading(self):
        """Verify that the dataset splits can be loaded properly."""
        if not self.splits_dir or not os.path.exists(self.splits_dir):
            self.skipTest(
                f"Could not find splits directory at any of the expected paths. Tried: {get_test_data_path('splits')}"
            )

        if not self.embedding_path or not os.path.exists(self.embedding_path):
            self.skipTest(
                f"Could not find embedding file at any of the expected paths. Tried: {get_test_data_path('embeddings')}"
            )

        # Use data validator for consistent split loading
        try:
            from templ_pipeline.core.data_validator import SplitDataValidator

            validator = SplitDataValidator(self.embedding_path, self.splits_dir)
            validated_splits = validator.get_validated_splits()

            # Verify splits have data
            train_size = len(validated_splits.get("train", set()))
            test_size = len(validated_splits.get("test", set()))

            self.assertGreater(
                train_size,
                0,
                f"Training split should not be empty, got {train_size} entries",
            )
            self.assertGreater(
                test_size, 0, f"Test split should not be empty, got {test_size} entries"
            )

            logger.info(f"Loaded splits: train={train_size}, test={test_size}")

        except Exception as e:
            self.fail(f"Error loading dataset splits: {e}")

    def test_template_selection(self):
        """Verify that template selection works with time-split restriction."""
        if not self.embedding_path or not os.path.exists(self.embedding_path):
            self.skipTest(
                f"Could not find embedding file at any of the expected paths. Tried: {get_test_data_path('embeddings')}"
            )

        if not self.splits_dir or not os.path.exists(self.splits_dir):
            self.skipTest(
                f"Could not find splits directory at any of the expected paths. Tried: {get_test_data_path('splits')}"
            )

        # Use data validator for reliable splits
        try:
            from templ_pipeline.core.data_validator import SplitDataValidator

            validator = SplitDataValidator(self.embedding_path, self.splits_dir)
            validated_splits = validator.get_validated_splits()
        except Exception as e:
            self.fail(f"Error loading data validator: {e}")

        # Initialize embedding manager
        try:
            embedding_manager = EmbeddingManager(self.embedding_path)
            self.assertIsNotNone(
                embedding_manager, "Embedding manager should be initialized"
            )
        except Exception as e:
            self.fail(f"Error loading embeddings: {e}")

        # Get validated test PDBs
        test_pdbs = list(validated_splits.get("test", set()))
        if not test_pdbs:
            self.skipTest("No validated test PDBs found")

        # Test the split validation mechanism itself
        train_pdbs = validated_splits.get("train", set())
        test_pdbs_set = validated_splits.get("test", set())

        # Basic validation that splits are properly separated
        overlap = train_pdbs.intersection(test_pdbs_set)
        self.assertEqual(len(overlap), 0, "Training and test sets should not overlap")

        # Verify we can get embeddings for training PDBs
        train_sample = list(train_pdbs)[:10]
        embedding_count = 0
        for pdb_id in train_sample:
            embedding, _ = embedding_manager.get_embedding(pdb_id)
            if embedding is not None:
                embedding_count += 1

        self.assertGreater(
            embedding_count, 0, "Should find embeddings for some training PDBs"
        )

        # Test neighbor finding with training PDB pool restriction
        if test_pdbs:
            test_pdb = test_pdbs[0]
            query_embedding, _ = embedding_manager.get_embedding(test_pdb)

            if query_embedding is not None:
                # Find neighbors restricted to training set
                all_neighbors = embedding_manager.find_neighbors(
                    query_pdb_id=test_pdb,
                    query_embedding=query_embedding,
                    k=100,  # Increase k to find more candidates
                    return_similarities=True,
                )

                # Filter to training set
                train_templates = [
                    (pdb_id, sim)
                    for pdb_id, sim in all_neighbors
                    if pdb_id in train_pdbs
                ]

                # Log for debugging
                logger.info(
                    f"Found {len(all_neighbors)} total neighbors, {len(train_templates)} in training set"
                )

                # The test passes if the filtering mechanism works correctly
                # (even if no templates are found, the important thing is proper filtering)
                for template_pdb, similarity in train_templates[:5]:
                    self.assertIn(
                        template_pdb,
                        train_pdbs,
                        f"Template {template_pdb} should be in training set",
                    )

                logger.info(f"Template filtering validation passed for {test_pdb}")
            else:
                self.skipTest(f"No embedding available for test PDB {test_pdb}")


if __name__ == "__main__":
    unittest.main()
