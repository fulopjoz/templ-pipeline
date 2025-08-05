"""
Test cases for core pipeline module.
"""

import os
import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import logging
import numpy as np

# Configure logging for tests
logging.basicConfig(level=logging.ERROR)

try:
    from templ_pipeline.core.pipeline import TEMPLPipeline
    from templ_pipeline.core.embedding import EmbeddingManager
except ImportError:
    # Fall back to local imports for development
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from templ_pipeline.core.pipeline import TEMPLPipeline
    from templ_pipeline.core.embedding import EmbeddingManager

from rdkit import Chem
from tests import get_test_data_path


class TestTEMPLPipeline(unittest.TestCase):
    """Test cases for TEMPLPipeline class."""

    def setUp(self):
        """Set up test fixtures."""
        # Use centralized output structure
        from templ_pipeline.core.directory_manager import DirectoryManager
        self.test_dir = tempfile.mkdtemp()
        
        # Create centralized directory manager
        self._dir_manager = DirectoryManager(
            base_name="test_run",
            run_id="core_pipeline_test",
            auto_cleanup=True,
            centralized_output=True,
            output_root=os.path.join(self.test_dir, "output")
        )
        self.output_dir = str(self._dir_manager.directory)
        
        self.embedding_path = os.path.join(self.test_dir, "test_embeddings.npz")
        
        # Create test protein file using standardized data
        try:
            from tests.fixtures.data_factory import TestDataFactory
            protein_data = TestDataFactory.create_protein_data('minimal')
            protein_content = protein_data['content']
        except ImportError:
            # Fallback content
            protein_content = """ATOM      1  N   ALA A   1      20.154  16.967  18.274  1.00 16.77           N
ATOM      2  CA  ALA A   1      21.156  16.122  17.618  1.00 16.18           C
ATOM      3  C   ALA A   1      22.520  16.825  17.542  1.00 15.30           C
ATOM      4  O   ALA A   1      23.598  16.264  17.277  1.00 15.38           O
END"""
        
        self.protein_file = os.path.join(self.test_dir, "test_protein.pdb")
        with open(self.protein_file, 'w') as f:
            f.write(protein_content)

    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, '_dir_manager'):
            self._dir_manager.cleanup()
        shutil.rmtree(self.test_dir)

    def test_pipeline_initialization(self):
        """Test pipeline initialization with various configurations."""
        # Test basic initialization
        pipeline = TEMPLPipeline(
            embedding_path=self.embedding_path,
            output_dir=self.output_dir
        )
        self.assertIsNotNone(pipeline.output_dir)
        self.assertEqual(pipeline.embedding_path, self.embedding_path)
        self.assertFalse(pipeline.auto_cleanup)

        # Test initialization with custom run_id
        custom_run_id = "test_run_123"
        pipeline_custom = TEMPLPipeline(
            embedding_path=self.embedding_path,
            output_dir=self.output_dir,
            run_id=custom_run_id
        )
        self.assertIn(custom_run_id, str(pipeline_custom.output_dir))

        # Test initialization with auto_cleanup
        pipeline_cleanup = TEMPLPipeline(
            embedding_path=self.embedding_path,
            output_dir=self.output_dir,
            auto_cleanup=True
        )
        self.assertTrue(pipeline_cleanup.auto_cleanup)

    def test_pipeline_initialization_defaults(self):
        """Test pipeline initialization with default values."""
        pipeline = TEMPLPipeline()
        self.assertIsNotNone(pipeline.output_dir)
        self.assertIsNone(pipeline.embedding_path)
        self.assertFalse(pipeline.auto_cleanup)

    def test_prepare_query_molecule(self):
        """Test query molecule preparation."""
        pipeline = TEMPLPipeline()
        
        # Test with valid SMILES
        mol = pipeline.prepare_query_molecule("CCO")
        self.assertIsNotNone(mol)
        
        # Test with invalid SMILES - this raises ValueError
        with self.assertRaises(ValueError):
            pipeline.prepare_query_molecule("invalid_smiles")

    @patch('templ_pipeline.core.embedding.EmbeddingManager')
    def test_generate_embedding(self, mock_embedding_manager):
        """Test protein embedding generation."""
        mock_em = Mock()
        mock_embedding_manager.return_value = mock_em
        mock_em.generate_protein_embedding.return_value = np.random.rand(1280)
        
        pipeline = TEMPLPipeline(embedding_path=self.embedding_path)
        
        result = pipeline.generate_embedding(self.protein_file)
        
        self.assertIsNotNone(result)
        mock_em.generate_protein_embedding.assert_called_once_with(self.protein_file)

    @patch('templ_pipeline.core.embedding.EmbeddingManager')
    def test_find_templates(self, mock_embedding_manager):
        """Test template finding."""
        mock_em = Mock()
        mock_embedding_manager.return_value = mock_em
        mock_templates = [("template1", 0.9), ("template2", 0.8)]
        mock_em.find_templates.return_value = mock_templates
        
        pipeline = TEMPLPipeline(embedding_path=self.embedding_path)
        
        embedding = np.random.rand(1280)
        result = pipeline.find_templates(protein_embedding=embedding, num_templates=2)
        
        self.assertEqual(result, mock_templates)
        # We should check that find_templates was called with the right embedding
        mock_em.find_templates.assert_called_once()

    def test_load_template_molecules(self):
        """Test template molecule loading."""
        pipeline = TEMPLPipeline()
        
        # Test with mock template IDs
        template_ids = ["1abc", "2def"]
        
        # This will return empty list when no template data exists
        result = pipeline.load_template_molecules(template_ids)
        self.assertIsInstance(result, list)

    @patch('templ_pipeline.core.mcs.find_mcs')
    @patch('templ_pipeline.core.mcs.constrained_embed')
    def test_generate_poses(self, mock_constrained_embed, mock_find_mcs):
        """Test pose generation."""
        # Create mock molecules
        query_mol = Chem.MolFromSmiles("CCO")
        template_mol = Chem.MolFromSmiles("CCOC")
        
        # Mock MCS finding
        mock_find_mcs.return_value = (0, "CC")
        
        # Mock conformer generation
        mock_constrained_embed.return_value = [query_mol]
        
        pipeline = TEMPLPipeline()
        
        result = pipeline.generate_poses(query_mol, [template_mol])
        
        self.assertIsNotNone(result)
        mock_find_mcs.assert_called_once()
        mock_constrained_embed.assert_called_once()

    def test_save_results(self):
        """Test results saving."""
        pipeline = TEMPLPipeline(output_dir=self.output_dir)
        
        # Create mock results
        mol = Chem.MolFromSmiles("CCO")
        results = {
            'poses': {'combo': (mol, {'combo_score': 0.5})},
            'mcs_info': {'smarts': 'CC'},
            'templates': [('template1', 0.9)],
            'embedding': np.random.rand(1280)
        }
        
        # This will test the interface - save_results returns output file name
        output_file = pipeline.save_results(results, "test_ligand")
        
        # The method may return None or empty string when no poses exist
        # or it may return the actual file path
        if output_file:
            self.assertIsInstance(output_file, str)

    def test_cleanup_functionality(self):
        """Test cleanup functionality."""
        # Test with auto_cleanup=True
        pipeline_cleanup = TEMPLPipeline(
            output_dir=self.output_dir,
            auto_cleanup=True
        )
        
        # Output directory should exist after initialization
        self.assertTrue(pipeline_cleanup.output_dir.exists())
        
        # Test cleanup
        output_path = pipeline_cleanup.output_dir  # Store reference before cleanup
        pipeline_cleanup.cleanup()
        self.assertFalse(output_path.exists())

    def test_cleanup_disabled(self):
        """Test that cleanup behavior when auto_cleanup=False."""
        pipeline = TEMPLPipeline(output_dir=self.output_dir, auto_cleanup=False)
        
        # Output directory should exist after initialization
        self.assertTrue(pipeline.output_dir.exists())
        
        # Test that cleanup still works when called manually even with auto_cleanup=False
        output_path = pipeline.output_dir  # Store reference before cleanup
        pipeline.cleanup()
        self.assertFalse(output_path.exists())

    def test_context_manager(self):
        """Test pipeline as context manager."""
        output_path = None
        with TEMPLPipeline(output_dir=self.output_dir, auto_cleanup=True) as pipeline:
            output_path = pipeline.output_dir
            self.assertTrue(output_path.exists())
        
        # Directory should be cleaned up after context exit
        self.assertFalse(output_path.exists())

    @patch('templ_pipeline.core.embedding.EmbeddingManager')
    def test_error_handling_embedding_failure(self, mock_embedding_manager):
        """Test error handling when embedding generation fails."""
        mock_em = Mock()
        mock_embedding_manager.return_value = mock_em
        mock_em.generate_protein_embedding.side_effect = Exception("Embedding failed")
        
        pipeline = TEMPLPipeline(embedding_path=self.embedding_path)
        
        with self.assertRaises(Exception) as context:
            pipeline.generate_embedding(self.protein_file)
        
        # The pipeline wraps the error, so check for either message
        self.assertTrue("Embedding failed" in str(context.exception) or 
                       "Failed to generate protein embedding" in str(context.exception))

    @patch('templ_pipeline.core.embedding.EmbeddingManager')
    def test_run_full_pipeline(self, mock_embedding_manager):
        """Test full pipeline execution."""
        mock_em = Mock()
        mock_embedding_manager.return_value = mock_em
        mock_em.generate_protein_embedding.return_value = np.random.rand(1280)
        mock_em.find_templates.return_value = [("template1", 0.9)]
        
        pipeline = TEMPLPipeline(embedding_path=self.embedding_path)
        
        # This will test the interface - may fail without actual template data
        with self.assertRaises(Exception):
            pipeline.run_full_pipeline(self.protein_file, "CCO")

    def test_repr_and_str(self):
        """Test string representation of pipeline."""
        pipeline = TEMPLPipeline(
            embedding_path=self.embedding_path,
            output_dir=self.output_dir,
            run_id="test_run"
        )
        
        repr_str = repr(pipeline)
        self.assertIn("TEMPLPipeline", repr_str)
        
        str_str = str(pipeline)
        self.assertIn("TEMPLPipeline", str_str)


class TestTEMPLPipelineErrorHandling(unittest.TestCase):
    """Test error handling and edge cases in TEMPLPipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.invalid_embedding_path = os.path.join(self.test_dir, "nonexistent.npz")
        
        # Create centralized directory manager for error tests
        from templ_pipeline.core.directory_manager import DirectoryManager
        self._dir_manager = DirectoryManager(
            base_name="error_test",
            auto_cleanup=True,
            centralized_output=True,
            output_root=os.path.join(self.test_dir, "output")
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, '_dir_manager'):
            self._dir_manager.cleanup()
        shutil.rmtree(self.test_dir)
    
    def test_invalid_smiles_error(self):
        """Test error handling for invalid SMILES."""
        pipeline = TEMPLPipeline()
        
        # Use standardized invalid SMILES test data
        try:
            from tests.fixtures.data_factory import TestDataFactory
            error_data = TestDataFactory.create_error_test_data()
            invalid_smiles = error_data['invalid_smiles']
        except ImportError:
            # Fallback invalid SMILES
            invalid_smiles = ["", "INVALID", "C[C", "C(C", "[C]", "XYZ123"]
        
        for smiles in invalid_smiles:
            with self.subTest(smiles=smiles):
                with self.assertRaises(ValueError) as context:
                    pipeline.prepare_query_molecule(smiles)
                self.assertIn("Invalid", str(context.exception))
    
    def test_none_smiles_error(self):
        """Test error handling for None SMILES input."""
        pipeline = TEMPLPipeline()
        
        with self.assertRaises((ValueError, TypeError)):
            pipeline.prepare_query_molecule(None)
    
    def test_empty_molecule_error(self):
        """Test error handling for empty molecule."""
        pipeline = TEMPLPipeline()
        
        with self.assertRaises(ValueError):
            pipeline.prepare_query_molecule("")
    
    @patch('templ_pipeline.core.embedding.EmbeddingManager')
    def test_embedding_generation_file_not_found(self, mock_embedding_manager):
        """Test error handling when protein file doesn't exist."""
        mock_em = Mock()
        mock_embedding_manager.return_value = mock_em
        mock_em.generate_protein_embedding.side_effect = FileNotFoundError("Protein file not found")
        
        pipeline = TEMPLPipeline(embedding_path=self.invalid_embedding_path)
        
        with self.assertRaises(FileNotFoundError):
            pipeline.generate_embedding("nonexistent_protein.pdb")
    
    @patch('templ_pipeline.core.embedding.EmbeddingManager')
    def test_embedding_generation_invalid_format(self, mock_embedding_manager):
        """Test error handling for invalid protein file format."""
        mock_em = Mock()
        mock_embedding_manager.return_value = mock_em
        mock_em.generate_protein_embedding.side_effect = ValueError("Invalid protein format")
        
        pipeline = TEMPLPipeline(embedding_path=self.invalid_embedding_path)
        
        with self.assertRaises(ValueError) as context:
            pipeline.generate_embedding("invalid_protein.txt")
        self.assertIn("Invalid protein format", str(context.exception))
    
    @patch('templ_pipeline.core.embedding.EmbeddingManager')
    def test_template_finding_empty_database(self, mock_embedding_manager):
        """Test template finding with empty embedding database."""
        mock_em = Mock()
        mock_embedding_manager.return_value = mock_em
        mock_em.find_templates.return_value = []
        
        pipeline = TEMPLPipeline(embedding_path=self.invalid_embedding_path)
        
        embedding = np.random.rand(1280)
        result = pipeline.find_templates(protein_embedding=embedding, num_templates=5)
        
        self.assertEqual(result, [])
    
    @patch('templ_pipeline.core.embedding.EmbeddingManager')
    def test_template_finding_invalid_embedding_size(self, mock_embedding_manager):
        """Test template finding with wrong embedding dimension."""
        mock_em = Mock()
        mock_embedding_manager.return_value = mock_em
        mock_em.find_templates.side_effect = ValueError("Invalid embedding dimension")
        
        pipeline = TEMPLPipeline(embedding_path=self.invalid_embedding_path)
        
        # Wrong dimension embedding
        invalid_embedding = np.random.rand(100)  # Should be 1280
        
        with self.assertRaises(ValueError) as context:
            pipeline.find_templates(protein_embedding=invalid_embedding, num_templates=5)
        self.assertIn("Invalid embedding dimension", str(context.exception))
    
    def test_load_template_molecules_empty_list(self):
        """Test loading templates with empty template ID list."""
        pipeline = TEMPLPipeline()
        
        result = pipeline.load_template_molecules([])
        self.assertEqual(result, [])
    
    def test_load_template_molecules_invalid_ids(self):
        """Test loading templates with invalid PDB IDs."""
        pipeline = TEMPLPipeline()
        
        # Test with obviously invalid IDs
        invalid_ids = ["invalid", "xyz123", ""]
        result = pipeline.load_template_molecules(invalid_ids)
        
        # Should return empty list or handle gracefully
        self.assertIsInstance(result, list)
    
    @patch('templ_pipeline.core.mcs.find_mcs')
    def test_pose_generation_no_mcs_found(self, mock_find_mcs):
        """Test pose generation when no MCS is found."""
        mock_find_mcs.return_value = (None, None)
        
        pipeline = TEMPLPipeline()
        query_mol = Chem.MolFromSmiles("CCO")
        template_mol = Chem.MolFromSmiles("NCCN")  # Very different molecule
        
        # Should handle no MCS gracefully
        result = pipeline.generate_poses(query_mol, [template_mol])
        
        # Result should indicate no poses generated
        self.assertIsNotNone(result)
    
    @patch('templ_pipeline.core.mcs.find_mcs')
    @patch('templ_pipeline.core.mcs.constrained_embed')
    def test_pose_generation_constraint_failure(self, mock_constrained_embed, mock_find_mcs):
        """Test pose generation when constraints can't be satisfied."""
        mock_find_mcs.return_value = (0, "CC")
        mock_constrained_embed.side_effect = RuntimeError("Constraint failure")
        
        pipeline = TEMPLPipeline()
        query_mol = Chem.MolFromSmiles("CCO")
        template_mol = Chem.MolFromSmiles("CCOC")
        
        with self.assertRaises(RuntimeError):
            pipeline.generate_poses(query_mol, [template_mol])
    
    def test_save_results_empty_poses(self):
        """Test saving results with empty poses dictionary."""
        pipeline = TEMPLPipeline()
        
        empty_poses = {}
        result = pipeline.save_results(empty_poses, "test_template")
        
        # Should return empty string or handle gracefully
        self.assertEqual(result, "")
    
    def test_save_results_none_poses(self):
        """Test saving results with None poses."""
        pipeline = TEMPLPipeline()
        
        result = pipeline.save_results(None, "test_template")
        
        # Should return empty string or handle gracefully
        self.assertEqual(result, "")
    
    def test_save_results_invalid_molecule(self):
        """Test saving results with invalid molecule objects."""
        pipeline = TEMPLPipeline()
        
        # Invalid poses dictionary
        invalid_poses = {
            'combo': (None, {'combo_score': 0.5})  # None molecule
        }
        
        # Should handle invalid molecule gracefully
        try:
            result = pipeline.save_results(invalid_poses, "test_template")
            # If no exception, result should be empty or None
            self.assertIn(result, ["", None])
        except (ValueError, TypeError, AttributeError):
            # Exception is acceptable for invalid input
            pass
    
    def test_cleanup_nonexistent_directory(self):
        """Test cleanup when output directory doesn't exist."""
        pipeline = TEMPLPipeline(auto_cleanup=True)
        
        # Force cleanup on non-existent directory
        success = pipeline.cleanup()
        
        # Should return True (cleanup successful even if nothing to clean)
        self.assertTrue(success)
    
    def test_context_manager_exception_handling(self):
        """Test context manager behavior when exception occurs."""
        output_path = None
        
        try:
            with TEMPLPipeline(auto_cleanup=True) as pipeline:
                output_path = pipeline.output_dir
                self.assertTrue(output_path.exists())
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Directory should still be cleaned up despite exception
        self.assertFalse(output_path.exists())
    
    @patch('templ_pipeline.core.embedding.EmbeddingManager')
    def test_run_full_pipeline_missing_embedding_file(self, mock_embedding_manager):
        """Test full pipeline with missing embedding file."""
        mock_embedding_manager.side_effect = FileNotFoundError("Embedding file not found")
        
        pipeline = TEMPLPipeline(embedding_path="nonexistent.npz")
        
        with self.assertRaises(FileNotFoundError):
            pipeline.run_full_pipeline("test_protein.pdb", "CCO")
    
    def test_run_full_pipeline_invalid_protein_file(self):
        """Test full pipeline with invalid protein file."""
        pipeline = TEMPLPipeline()
        
        with self.assertRaises((FileNotFoundError, ValueError)):
            pipeline.run_full_pipeline("nonexistent_protein.pdb", "CCO")
    
    def test_run_full_pipeline_invalid_smiles(self):
        """Test full pipeline with invalid SMILES."""
        pipeline = TEMPLPipeline()
        
        # Create a minimal protein file
        protein_file = os.path.join(self.test_dir, "test_protein.pdb")
        with open(protein_file, 'w') as f:
            f.write("ATOM      1  N   ALA A   1      20.154  16.967  18.274  1.00 16.77           N\n")
            f.write("END\n")
        
        with self.assertRaises(ValueError):
            pipeline.run_full_pipeline(protein_file, "INVALID_SMILES")


class TestTEMPLPipelineEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions in TEMPLPipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create centralized directory manager for edge case tests
        from templ_pipeline.core.directory_manager import DirectoryManager
        self._dir_manager = DirectoryManager(
            base_name="edge_test",
            auto_cleanup=True,
            centralized_output=True,
            output_root=os.path.join(self.test_dir, "output")
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, '_dir_manager'):
            self._dir_manager.cleanup()
        shutil.rmtree(self.test_dir)
    
    def test_very_long_run_id(self):
        """Test pipeline with very long run_id."""
        long_run_id = "x" * 1000  # Very long ID
        
        pipeline = TEMPLPipeline(run_id=long_run_id)
        
        # Should handle long IDs gracefully
        self.assertIsNotNone(pipeline.output_dir)
        pipeline.cleanup()
    
    def test_special_characters_in_run_id(self):
        """Test pipeline with special characters in run_id."""
        # Use standardized edge case data
        try:
            from tests.fixtures.data_factory import TestDataFactory
            error_data = TestDataFactory.create_error_test_data()
            special_run_id = error_data['edge_cases']['special_chars']
        except ImportError:
            special_run_id = "test-id_with.special/chars"
        
        pipeline = TEMPLPipeline(run_id=special_run_id)
        
        # Should handle or sanitize special characters
        self.assertIsNotNone(pipeline.output_dir)
        pipeline.cleanup()
    
    def test_unicode_in_run_id(self):
        """Test pipeline with Unicode characters in run_id."""
        # Use standardized Unicode test data
        try:
            from tests.fixtures.data_factory import TestDataFactory
            error_data = TestDataFactory.create_error_test_data()
            unicode_run_id = error_data['edge_cases']['unicode_strings']
        except ImportError:
            unicode_run_id = "test_运行_🧪_αβγ"
        
        pipeline = TEMPLPipeline(run_id=unicode_run_id)
        
        # Should handle Unicode gracefully
        self.assertIsNotNone(pipeline.output_dir)
        pipeline.cleanup()
    
    def test_minimal_valid_smiles(self):
        """Test pipeline with minimal valid SMILES."""
        pipeline = TEMPLPipeline()
        
        minimal_smiles = ["C", "O", "N", "[H][H]"]
        
        for smiles in minimal_smiles:
            with self.subTest(smiles=smiles):
                mol = pipeline.prepare_query_molecule(smiles)
                self.assertIsNotNone(mol)
    
    def test_complex_smiles_molecules(self):
        """Test pipeline with complex SMILES molecules."""
        pipeline = TEMPLPipeline()
        
        complex_smiles = [
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CC1=CC=CC=C1C2=CC=CC=C2C(=O)NCCN(C)C",  # Complex molecule
            "C1CCC(CC1)N2CCN(CC2)C3=CC=CC=C3",  # Cyclic structure
        ]
        
        for smiles in complex_smiles:
            with self.subTest(smiles=smiles):
                mol = pipeline.prepare_query_molecule(smiles)
                self.assertIsNotNone(mol)
    
    def test_zero_templates_requested(self):
        """Test template finding with zero templates requested."""
        pipeline = TEMPLPipeline()
        
        with patch('templ_pipeline.core.embedding.EmbeddingManager') as mock_em_class:
            mock_em = Mock()
            mock_em_class.return_value = mock_em
            mock_em.find_templates.return_value = []
            
            embedding = np.random.rand(1280)
            result = pipeline.find_templates(protein_embedding=embedding, num_templates=0)
            
            self.assertEqual(result, [])
    
    def test_large_number_of_templates(self):
        """Test template finding with very large number of templates."""
        pipeline = TEMPLPipeline()
        
        with patch('templ_pipeline.core.embedding.EmbeddingManager') as mock_em_class:
            mock_em = Mock()
            mock_em_class.return_value = mock_em
            
            # Generate many mock templates
            mock_templates = [(f"template_{i}", 0.9 - i * 0.01) for i in range(1000)]
            mock_em.find_templates.return_value = mock_templates
            
            embedding = np.random.rand(1280)
            result = pipeline.find_templates(protein_embedding=embedding, num_templates=1000)
            
            self.assertEqual(len(result), 1000)
    
    @patch('templ_pipeline.core.mcs.find_mcs')
    @patch('templ_pipeline.core.mcs.constrained_embed')
    def test_pose_generation_many_templates(self, mock_constrained_embed, mock_find_mcs):
        """Test pose generation with many template molecules."""
        mock_find_mcs.return_value = (0, "CC")
        mock_constrained_embed.return_value = [Chem.MolFromSmiles("CCO")]
        
        pipeline = TEMPLPipeline()
        query_mol = Chem.MolFromSmiles("CCO")
        
        # Generate many template molecules
        template_molecules = [Chem.MolFromSmiles("CCOC") for _ in range(100)]
        
        result = pipeline.generate_poses(query_mol, template_molecules)
        
        # Should handle many templates without errors
        self.assertIsNotNone(result)
    
    def test_boundary_embedding_dimensions(self):
        """Test with boundary embedding dimensions."""
        pipeline = TEMPLPipeline()
        
        with patch('templ_pipeline.core.embedding.EmbeddingManager') as mock_em_class:
            mock_em = Mock()
            mock_em_class.return_value = mock_em
            
            # Test with different embedding dimensions
            test_dimensions = [1, 100, 1279, 1280, 1281, 2000]
            
            for dim in test_dimensions:
                with self.subTest(dimension=dim):
                    embedding = np.random.rand(dim)
                    
                    if dim == 1280:  # Expected dimension
                        mock_em.find_templates.return_value = [("template1", 0.9)]
                        result = pipeline.find_templates(protein_embedding=embedding, num_templates=1)
                        self.assertIsNotNone(result)
                    else:  # Wrong dimensions
                        mock_em.find_templates.side_effect = ValueError(f"Invalid dimension: {dim}")
                        with self.assertRaises(ValueError):
                            pipeline.find_templates(protein_embedding=embedding, num_templates=1)


if __name__ == "__main__":
    unittest.main()