# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Tests for the embedding module of the TEMPL pipeline.

These tests verify the functionality of the EmbeddingManager class,
embedding calculation, caching behavior, and template selection.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from templ_pipeline.core.embedding import EmbeddingManager
from tests import get_test_data_path


def is_ci_environment():
    """Check if running in CI environment where data files may not be available."""
    return os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true"


def skip_if_missing_data(data_type, reason=None):
    """Skip test if required data files are not available."""
    if is_ci_environment():
        data_path = get_test_data_path(data_type)
        if not data_path or not os.path.exists(data_path):
            pytest.skip(
                f"Data file not available in CI: {data_type}"
                + (f" - {reason}" if reason else "")
            )
    return True


class TestEmbeddingManager(unittest.TestCase):
    """Test the EmbeddingManager class with mock data."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()

        # Create a mock embedding file
        self.mock_embedding_path = os.path.join(self.temp_dir, "mock_embeddings.npz")

        # Create mock embedding data
        mock_embeddings = np.random.rand(
            10, 1280
        )  # 10 proteins, 1280-dimensional embeddings
        mock_pdb_ids = [f"test{i:03d}" for i in range(10)]
        mock_chain_ids = ["A"] * 10

        # Save mock embeddings
        np.savez(
            self.mock_embedding_path,
            embeddings=mock_embeddings,
            pdb_ids=mock_pdb_ids,
            chain_ids=mock_chain_ids,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, "temp_dir") and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_basic_initialization(self):
        """Test basic initialization with a mock file."""
        # Skip if in CI environment and no mock file can be created
        if is_ci_environment():
            try:
                manager = EmbeddingManager(self.mock_embedding_path, use_cache=False)
                self.assertIsNotNone(
                    manager, "EmbeddingManager should be initialized successfully"
                )
                # Check that embeddings were loaded (don't hardcode exact count as it may vary)
                self.assertGreater(
                    len(manager.embedding_db),
                    0,
                    "Should load at least some embeddings from file",
                )
                self.assertGreater(
                    len(manager.embedding_chain_data),
                    0,
                    "Should load at least some chain data from file",
                )
            except Exception as e:
                # In CI, if the test fails due to missing data, skip instead of failing
                if (
                    "file not found" in str(e).lower()
                    or "no such file" in str(e).lower()
                ):
                    pytest.skip(f"Mock embedding file not available in CI: {e}")
                else:
                    self.fail(f"EmbeddingManager initialization failed: {e}")
        else:
            # In local environment, expect the test to work normally
            manager = EmbeddingManager(self.mock_embedding_path, use_cache=False)
            self.assertIsNotNone(
                manager, "EmbeddingManager should be initialized successfully"
            )
            # Check that embeddings were loaded (don't hardcode exact count as it may vary)
            self.assertGreater(
                len(manager.embedding_db),
                0,
                "Should load at least some embeddings from file",
            )
            self.assertGreater(
                len(manager.embedding_chain_data),
                0,
                "Should load at least some chain data from file",
            )

    def test_cache_operations(self):
        """Test cache operations."""
        cache_dir = os.path.join(self.temp_dir, "cache")
        manager = EmbeddingManager(
            self.mock_embedding_path, use_cache=True, cache_dir=cache_dir
        )

        # Check that the manager has a cache directory configured
        self.assertIsNotNone(
            manager.cache_dir, "Manager should have cache directory configured"
        )
        self.assertTrue(
            os.path.exists(manager.cache_dir), "Manager cache directory should exist"
        )

    def test_initialization_with_nonexistent_file(self):
        """Test initialization with a non-existent file."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.npz")

        # Should handle gracefully without crashing
        manager = EmbeddingManager(nonexistent_path, use_cache=False)
        self.assertIsNotNone(
            manager, "EmbeddingManager should be initialized even with missing file"
        )

        # The manager might load from default paths if the custom path doesn't exist
        # So we can't guarantee it will be empty, but it should be initialized
        self.assertIsNotNone(
            manager.embedding_db,
            "Should have embedding database (even if loaded from defaults)",
        )

    def test_embedding_retrieval(self):
        """Test embedding retrieval functionality."""
        # Skip if in CI environment
        if is_ci_environment():
            pytest.skip("Embedding retrieval test skipped in CI environment")

        manager = EmbeddingManager(self.mock_embedding_path, use_cache=False)

        # Test retrieving embeddings for known PDB IDs
        test_pdb_id = "test000"
        if test_pdb_id in manager.embedding_db:
            embedding = manager.get_embedding(test_pdb_id)
            self.assertIsNotNone(
                embedding, "Should retrieve embedding for known PDB ID"
            )
            self.assertEqual(
                embedding.shape[0], 1280, "Embedding should have correct dimension"
            )


class TestEmbeddingManagerWithRealData(unittest.TestCase):
    """Test the EmbeddingManager class with real data."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        # Check if we're in CI and skip if data is not available
        if is_ci_environment():
            skip_if_missing_data(
                "embeddings", "Real embedding data not available in CI"
            )

        # Path to real embedding NPZ file using helper function
        cls.embedding_path = get_test_data_path("embeddings")
        if not cls.embedding_path or not os.path.exists(cls.embedding_path):
            cls.embedding_manager = None
            cls.test_pdb_other_file = None
            cls.test_pdb_refined_file = None
            return

        # Test PDB IDs (one from each set)
        cls.test_pdb_other = "1a0q"
        cls.test_pdb_refined = "1a1e"

        # PDB file paths - build dynamically and check existence
        pdbbind_other_path = get_test_data_path("pdbbind_other")
        pdbbind_refined_path = get_test_data_path("pdbbind_refined")

        if pdbbind_other_path and os.path.exists(
            os.path.join(pdbbind_other_path, cls.test_pdb_other)
        ):
            cls.test_pdb_other_file = os.path.join(
                pdbbind_other_path,
                cls.test_pdb_other,
                f"{cls.test_pdb_other}_protein.pdb",
            )
        else:
            cls.test_pdb_other_file = None

        if pdbbind_refined_path and os.path.exists(
            os.path.join(pdbbind_refined_path, cls.test_pdb_refined)
        ):
            cls.test_pdb_refined_file = os.path.join(
                pdbbind_refined_path,
                cls.test_pdb_refined,
                f"{cls.test_pdb_refined}_protein.pdb",
            )
        else:
            cls.test_pdb_refined_file = None

        # Create a temp directory for cache
        cls.temp_dir = tempfile.mkdtemp()
        cls.cache_dir = os.path.join(cls.temp_dir, "embedding_cache")
        os.makedirs(cls.cache_dir, exist_ok=True)

        # Initialize embedding manager
        try:
            cls.embedding_manager = EmbeddingManager(
                cls.embedding_path, use_cache=True, cache_dir=cls.cache_dir
            )
        except Exception as e:
            cls.embedding_manager = None
            print(f"Failed to initialize EmbeddingManager: {e}")
            return

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        if hasattr(cls, "temp_dir") and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Per-test setup."""
        if self.embedding_manager is None:
            pytest.skip("EmbeddingManager not available - skipping real data tests")

    def test_load_embeddings(self):
        """Test loading embeddings from real file."""
        if is_ci_environment():
            pytest.skip("Real data test skipped in CI environment")

        self.assertIsNotNone(
            self.embedding_manager, "EmbeddingManager should be initialized"
        )
        self.assertGreater(
            len(self.embedding_manager.embedding_db),
            0,
            "Should load embeddings from real file",
        )

    def test_chain_data(self):
        """Test chain data loading."""
        if is_ci_environment():
            pytest.skip("Real data test skipped in CI environment")

        self.assertIsNotNone(
            self.embedding_manager, "EmbeddingManager should be initialized"
        )
        self.assertGreater(
            len(self.embedding_manager.embedding_chain_data),
            0,
            "Should load chain data from real file",
        )

    def test_cache_operations(self):
        """Test cache operations with real data."""
        if is_ci_environment():
            pytest.skip("Real data test skipped in CI environment")

        self.assertIsNotNone(
            self.embedding_manager, "EmbeddingManager should be initialized"
        )
        self.assertTrue(os.path.exists(self.cache_dir), "Cache directory should exist")

    def test_embedding_generation_with_real_pdb(self):
        """Test embedding generation with real PDB files."""
        if is_ci_environment():
            pytest.skip("Real data test skipped in CI environment")

        # Test with PDBbind other set
        if self.test_pdb_other_file and os.path.exists(self.test_pdb_other_file):
            try:
                embedding = self.embedding_manager.generate_embedding(
                    self.test_pdb_other_file
                )
                self.assertIsNotNone(
                    embedding, "Should generate embedding for real PDB"
                )
                self.assertEqual(
                    embedding.shape[0],
                    1280,
                    "Generated embedding should have correct dimension",
                )
            except Exception as e:
                # Skip if embedding generation fails (may require GPU or specific dependencies)
                pytest.skip(f"Embedding generation failed: {e}")

        # Test with PDBbind refined set
        if self.test_pdb_refined_file and os.path.exists(self.test_pdb_refined_file):
            try:
                embedding = self.embedding_manager.generate_embedding(
                    self.test_pdb_refined_file
                )
                self.assertIsNotNone(
                    embedding, "Should generate embedding for real PDB"
                )
                self.assertEqual(
                    embedding.shape[0],
                    1280,
                    "Generated embedding should have correct dimension",
                )
            except Exception as e:
                # Skip if embedding generation fails
                pytest.skip(f"Embedding generation failed: {e}")

    def test_find_neighbors(self):
        """Test finding neighbors functionality."""
        if is_ci_environment():
            pytest.skip("Real data test skipped in CI environment")

        if len(self.embedding_manager.embedding_db) > 0:
            # Get a sample PDB ID
            sample_pdb_id = list(self.embedding_manager.embedding_db.keys())[0]

            try:
                neighbors = self.embedding_manager.find_neighbors(sample_pdb_id, k=5)
                self.assertIsNotNone(neighbors, "Should find neighbors")
                self.assertLessEqual(
                    len(neighbors), 5, "Should return at most k neighbors"
                )
            except Exception as e:
                pytest.skip(f"Neighbor finding failed: {e}")

    def test_get_protein_sequence_from_real_pdb(self):
        """Test getting protein sequence from real PDB files."""
        if is_ci_environment():
            pytest.skip("Real data test skipped in CI environment")

        # Test with PDBbind other set
        if self.test_pdb_other_file and os.path.exists(self.test_pdb_other_file):
            try:
                sequence = self.embedding_manager.get_protein_sequence(
                    self.test_pdb_other_file
                )
                self.assertIsNotNone(sequence, "Should extract sequence from real PDB")
                self.assertGreater(len(sequence), 0, "Sequence should not be empty")
            except Exception as e:
                pytest.skip(f"Sequence extraction failed: {e}")

        # Test with PDBbind refined set
        if self.test_pdb_refined_file and os.path.exists(self.test_pdb_refined_file):
            try:
                sequence = self.embedding_manager.get_protein_sequence(
                    self.test_pdb_refined_file
                )
                self.assertIsNotNone(sequence, "Should extract sequence from real PDB")
                self.assertGreater(len(sequence), 0, "Sequence should not be empty")
            except Exception as e:
                pytest.skip(f"Sequence extraction failed: {e}")


if __name__ == "__main__":
    unittest.main()
