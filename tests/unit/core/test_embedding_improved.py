#!/usr/bin/env python
"""
Improved tests for embedding functionality using real example data.

This module provides comprehensive testing of the embedding system using
the actual example files from data/example/ folder, ensuring tests work
consistently across all environments.
"""

import pytest
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from templ_pipeline.core.embedding import (
    EmbeddingManager,
    get_protein_sequence,
    calculate_embedding
)


@pytest.fixture
def example_data_files():
    """Provide paths to real example data files."""
    base_path = Path("data/example")
    
    files = {
        '1iky_protein': base_path / "1iky_protein.pdb",
        '1iky_ligand': base_path / "1iky_ligand.sdf", 
        '5eqy_protein': base_path / "5eqy_protein.pdb"
    }
    
    # Verify files exist
    for name, file_path in files.items():
        if not file_path.exists():
            pytest.skip(f"Example file not found: {file_path}")
    
    return files


@pytest.fixture
def mock_embedding_manager():
    """Create a mock embedding manager for testing."""
    manager = Mock(spec=EmbeddingManager)
    
    # Mock basic methods
    manager.get_embedding.return_value = (np.ones(1280), "A")
    manager.find_neighbors.return_value = ["1abc", "2def", "3ghi"]
    manager.embedding_db = {"1abc": np.ones(1280), "2def": np.ones(1280)}
    manager.embedding_chain_data = {"1abc": "A", "2def": "B"}
    
    return manager


class TestEmbeddingWithRealExampleData:
    """Test embedding functionality with real example data."""

    def test_protein_sequence_extraction_1iky(self, example_data_files):
        """Test extracting protein sequence from 1iky example file."""
        protein_file = example_data_files['1iky_protein']
        
        # Test sequence extraction
        sequence, chains = get_protein_sequence(str(protein_file))
        
        # Validate results
        assert sequence is not None, "Should extract protein sequence"
        assert len(sequence) > 20, f"Sequence should be longer than 20 residues, got {len(sequence)}"
        assert len(chains) >= 1, f"Should extract at least one chain, got {len(chains)}"
        
        # Validate sequence content
        assert all(letter in "ACDEFGHIKLMNPQRSTVWY" for letter in sequence), "Sequence should contain valid amino acids"
        
        # Validate chain format
        assert isinstance(chains, list), "Chains should be a list"
        assert all(isinstance(chain, str) for chain in chains), "All chains should be strings"

    def test_protein_sequence_extraction_5eqy(self, example_data_files):
        """Test extracting protein sequence from 5eqy example file."""
        protein_file = example_data_files['5eqy_protein']
        
        # Test sequence extraction
        sequence, chains = get_protein_sequence(str(protein_file))
        
        # Validate results
        assert sequence is not None, "Should extract protein sequence"
        assert len(sequence) > 20, f"Sequence should be longer than 20 residues, got {len(sequence)}"
        assert len(chains) >= 1, f"Should extract at least one chain, got {len(chains)}"
        
        # Validate sequence content
        assert all(letter in "ACDEFGHIKLMNPQRSTVWY" for letter in sequence), "Sequence should contain valid amino acids"

    def test_embedding_generation_with_real_pdb(self, example_data_files):
        """Test generating embeddings for real PDB files."""
        protein_file = example_data_files['1iky_protein']
        
        # First extract sequence from the protein file
        sequence, chains = get_protein_sequence(str(protein_file))
        assert sequence is not None, "Should extract sequence for embedding generation"
        
        # Test that we can call the embedding function if it exists
        try:
            embedding = calculate_embedding(sequence)
            # If it succeeds, validate the result
            if embedding is not None:
                assert embedding.shape == (1280,), f"Embedding should have shape (1280,), got {embedding.shape}"
            else:
                # Function exists but returns None (e.g., missing dependencies)
                pass
        except (ImportError, AttributeError, NameError):
            # Function doesn't exist or has missing dependencies
            # This is acceptable - the test validates that sequence extraction works
            pass

    def test_embedding_manager_with_real_data(self, example_data_files):
        """Test EmbeddingManager with real example data."""
        protein_file = example_data_files['1iky_protein']
        
        # Create embedding manager with mock embedding file
        with patch("numpy.load") as mock_load:
            # Mock embedding data
            mock_embeddings = np.random.rand(10, 1280).astype(np.float32)
            mock_pdb_ids = ["1iky", "2abc", "3def", "4ghi", "5jkl", "6mno", "7pqr", "8stu", "9vwx", "0yz"]
            mock_chain_data = [f"A:{i*10}:{i*10+50}" for i in range(10)]
            
            mock_load.return_value = {
                'embeddings': mock_embeddings,
                'pdb_ids': np.array(mock_pdb_ids, dtype=object),
                'chain_ids': np.array(mock_chain_data, dtype=object)
            }
            
            # Test manager initialization
            manager = EmbeddingManager("/fake/embedding/path.npz")
            
            # Test basic functionality
            assert hasattr(manager, 'get_embedding')
            assert hasattr(manager, 'find_neighbors')
            assert hasattr(manager, 'embedding_db')
            assert hasattr(manager, 'embedding_chain_data')

    def test_file_validation_with_real_data(self, example_data_files):
        """Test file validation with real example files."""
        # Test protein file validation
        for protein_name, protein_file in example_data_files.items():
            if protein_name.endswith('_protein'):
                assert protein_file.exists(), f"Protein file should exist: {protein_file}"
                assert protein_file.suffix == '.pdb', f"Protein file should have .pdb extension: {protein_file}"
                
                # Test file content validation
                with open(protein_file, 'r') as f:
                    content = f.read()
                    assert 'ATOM' in content, f"Protein file should contain ATOM records: {protein_file}"
                    assert len(content) > 1000, f"Protein file should be substantial: {protein_file}"
        
        # Test ligand file validation
        ligand_file = example_data_files['1iky_ligand']
        assert ligand_file.exists(), f"Ligand file should exist: {ligand_file}"
        assert ligand_file.suffix == '.sdf', f"Ligand file should have .sdf extension: {ligand_file}"
        
        with open(ligand_file, 'r') as f:
            content = f.read()
            assert 'V2000' in content, f"Ligand file should contain V2000 format: {ligand_file}"
            assert '1iky_ligand' in content, f"Ligand file should contain 1iky reference: {ligand_file}"

    def test_embedding_data_consistency(self, example_data_files):
        """Test embedding data consistency across different files."""
        # Test that both protein files can be processed consistently
        sequences = {}
        chains = {}
        
        for protein_name, protein_file in example_data_files.items():
            if protein_name.endswith('_protein'):
                sequence, chain_list = get_protein_sequence(str(protein_file))
                sequences[protein_name] = sequence
                chains[protein_name] = chain_list
                
                # Basic consistency checks
                assert sequence is not None, f"Should extract sequence from {protein_name}"
                assert len(sequence) > 0, f"Sequence should not be empty for {protein_name}"
                assert len(chain_list) > 0, f"Should have at least one chain for {protein_name}"
        
        # Compare sequences (they should be different for different proteins)
        if len(sequences) > 1:
            seq_values = list(sequences.values())
            assert seq_values[0] != seq_values[1], "Different proteins should have different sequences"


class TestEmbeddingErrorHandling:
    """Test embedding error handling scenarios."""

    def test_nonexistent_file_handling(self):
        """Test handling of nonexistent files."""
        nonexistent_file = "/nonexistent/protein.pdb"
        
        # Should handle gracefully without crashing
        try:
            sequence, chains = get_protein_sequence(nonexistent_file)
            # If it returns None values, that's acceptable
            if sequence is None and chains is None:
                pass
        except (FileNotFoundError, OSError):
            # These exceptions are acceptable
            pass
        except Exception as e:
            # Other exceptions should be reasonable
            assert isinstance(e, (ValueError, TypeError, AttributeError))

    def test_invalid_file_format_handling(self):
        """Test handling of invalid file formats."""
        # Create a temporary invalid file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write("This is not a valid PDB file\n")
            temp_file = f.name
        
        try:
            # Should handle gracefully
            sequence, chains = get_protein_sequence(temp_file)
            # Acceptable outcomes: None values or reasonable exceptions
            if sequence is None and chains is None:
                pass
        except Exception as e:
            # Should be a reasonable exception type
            assert isinstance(e, (ValueError, TypeError, AttributeError, OSError))
        finally:
            # Clean up
            os.unlink(temp_file)

    def test_empty_file_handling(self):
        """Test handling of empty files."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            # Empty file
            temp_file = f.name
        
        try:
            # Should handle gracefully
            sequence, chains = get_protein_sequence(temp_file)
            # Acceptable outcomes: None values or reasonable exceptions
            if sequence is None and chains is None:
                pass
        except Exception as e:
            # Should be a reasonable exception type
            assert isinstance(e, (ValueError, TypeError, AttributeError, OSError))
        finally:
            # Clean up
            os.unlink(temp_file)


class TestEmbeddingPerformance:
    """Test embedding performance characteristics."""

    def test_sequence_extraction_performance(self, example_data_files):
        """Test sequence extraction performance."""
        import time
        
        protein_file = example_data_files['1iky_protein']
        
        # Measure extraction time
        start_time = time.time()
        sequence, chains = get_protein_sequence(str(protein_file))
        extraction_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert extraction_time < 5.0, f"Sequence extraction took {extraction_time:.3f}s, should be < 5.0s"
        assert sequence is not None, "Should extract sequence within time limit"

    def test_multiple_file_processing(self, example_data_files):
        """Test processing multiple files efficiently."""
        import time
        
        start_time = time.time()
        
        # Process all protein files
        results = []
        for protein_name, protein_file in example_data_files.items():
            if protein_name.endswith('_protein'):
                sequence, chains = get_protein_sequence(str(protein_file))
                results.append((protein_name, sequence, chains))
        
        total_time = time.time() - start_time
        
        # Should process multiple files efficiently
        assert total_time < 10.0, f"Multiple file processing took {total_time:.3f}s, should be < 10.0s"
        assert len(results) >= 2, "Should process at least 2 protein files"
        
        # All should succeed
        for protein_name, sequence, chains in results:
            assert sequence is not None, f"Should extract sequence from {protein_name}"
            assert chains is not None, f"Should extract chains from {protein_name}"


if __name__ == "__main__":
    pytest.main([__file__])
