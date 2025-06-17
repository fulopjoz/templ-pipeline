"""
Unit tests for the TEMPL Pipeline Scoring module.

These tests verify the functionality of the Scoring module, including
shape-based scoring, color scoring, combo scoring, and pose selection.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import logging
import tempfile
import numpy as np
import sys

from rdkit import Chem
from rdkit.Chem import AllChem

# Handle imports for both development and installed package
try:
    from templ_pipeline.core.scoring import (
        score_and_align,
        select_best,
        rmsd_raw,
        generate_properties_for_sdf
    )
except ImportError:
    # Fall back to local imports for development
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from core.scoring import (
        score_and_align,
        select_best,
        rmsd_raw,
        generate_properties_for_sdf
    )

# Configure logging for tests
logging.basicConfig(level=logging.ERROR)

class TestScoring(unittest.TestCase):
    """Test cases for Scoring module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test molecules
        self.mol1 = Chem.MolFromSmiles('CCO')  # Ethanol
        self.mol2 = Chem.MolFromSmiles('CCOC')  # Methyl ethyl ether
        
        # Generate 3D conformers
        for mol in [self.mol1, self.mol2]:
            if mol is not None:
                mol.SetProp("_Name", "test_mol")
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
                
        # Create a molecule with multiple conformers for testing select_best
        self.multi_conf_mol = Chem.MolFromSmiles('CCCCO')  # 1-butanol
        if self.multi_conf_mol is not None:
            self.multi_conf_mol.SetProp("_Name", "multi_conf_test")
            # Generate multiple conformers
            AllChem.EmbedMultipleConfs(self.multi_conf_mol, numConfs=5)
            AllChem.AddHs(self.multi_conf_mol)
            # Optimize all conformers
            for conf_id in range(self.multi_conf_mol.GetNumConformers()):
                AllChem.MMFFOptimizeMolecule(self.multi_conf_mol, confId=conf_id)
    
    def test_score_and_align_basic(self):
        """Test basic scoring and alignment between similar molecules."""
        # Ensure molecules exist
        self.assertIsNotNone(self.mol1)
        self.assertIsNotNone(self.mol2)
        
        # Run scoring
        scores, aligned_mol = score_and_align(self.mol1, self.mol2)
        
        # Check that scores exist and are reasonable
        self.assertIn("shape", scores)
        self.assertIn("color", scores)
        self.assertIn("combo", scores)
        
        # Shape score should be between 0 and 1
        self.assertGreaterEqual(scores["shape"], 0.0)
        self.assertLessEqual(scores["shape"], 1.0)
        
        # Color score should be between 0 and 1
        self.assertGreaterEqual(scores["color"], 0.0)
        self.assertLessEqual(scores["color"], 1.0)
        
        # Combo score should be average of shape and color
        self.assertAlmostEqual(scores["combo"], 0.5 * (scores["shape"] + scores["color"]))
        
        # Verify aligned molecule was returned
        self.assertIsNotNone(aligned_mol)
        self.assertEqual(aligned_mol.GetNumAtoms(), self.mol1.GetNumAtoms())
    
    def test_score_and_align_error_handling(self):
        """Test error handling in score_and_align."""
        # Create an invalid molecule (missing coordinates)
        invalid_mol = Chem.MolFromSmiles('CCO')
        
        # Run scoring with invalid molecule
        scores, aligned_mol = score_and_align(invalid_mol, self.mol2)
        
        # Should return default scores on error
        self.assertEqual(scores["shape"], -1.0)
        self.assertEqual(scores["color"], -1.0)
        self.assertEqual(scores["combo"], -1.0)
    
    @patch('templ_pipeline.core.scoring.rmsdwrapper')
    def test_rmsd_raw(self, mock_rmsdwrapper):
        """Test RMSD calculation between molecules."""
        # Mock the rmsdwrapper to return a fixed value
        mock_rmsdwrapper.return_value = [2.5]
        
        # Calculate RMSD
        rmsd = rmsd_raw(self.mol1, self.mol2)
        
        # Verify the expected RMSD value
        self.assertEqual(rmsd, 2.5)
        
        # Verify rmsdwrapper was called with correct arguments
        mock_rmsdwrapper.assert_called_once()
    
    def test_rmsd_raw_error_handling(self):
        """Test error handling in RMSD calculation."""
        # Create a molecule with invalid coordinates for RMSD calculation
        invalid_mol = Chem.MolFromSmiles('CCO')
        
        # Calculate RMSD with invalid input
        rmsd = rmsd_raw(invalid_mol, self.mol2)
        
        # Should return NaN on error
        self.assertTrue(np.isnan(rmsd))
    
    @patch('templ_pipeline.core.scoring.ProcessPoolExecutor')
    def test_select_best_parallel(self, mock_executor):
        """Test parallel execution of select_best."""
        # Ensure multi-conformer molecule exists
        self.assertIsNotNone(self.multi_conf_mol)
        
        # Mock results for parallel execution
        mock_results = [
            (0, {"shape": 0.8, "color": 0.6, "combo": 0.7}, self.mol1),
            (1, {"shape": 0.7, "color": 0.9, "combo": 0.8}, self.mol2)
        ]
        mock_executor.return_value.__enter__.return_value.map.return_value = mock_results
        
        # Run select_best with parallel processing
        best = select_best(self.multi_conf_mol, self.mol2, n_workers=2)
        
        # Should return dictionary with shape, color, combo keys
        self.assertIn("shape", best)
        self.assertIn("color", best)
        self.assertIn("combo", best)
    
    def test_select_best_sequential(self):
        """Test sequential execution of select_best."""
        # Ensure multi-conformer molecule exists and has conformers
        self.assertIsNotNone(self.multi_conf_mol)
        self.assertGreater(self.multi_conf_mol.GetNumConformers(), 0)
        
        # Run select_best sequentially
        best = select_best(self.multi_conf_mol, self.mol2, n_workers=1)
        
        # Should return dictionary with shape, color, combo keys
        self.assertIn("shape", best)
        self.assertIn("color", best)
        self.assertIn("combo", best)
        
        # Each entry should have molecule and scores
        for metric in ["shape", "color", "combo"]:
            self.assertIsInstance(best[metric], tuple)
            self.assertEqual(len(best[metric]), 2)
            
            # First item should be a molecule
            mol, scores = best[metric]
            if mol is not None:  # Some scores might be negative, resulting in None
                self.assertIsInstance(mol, Chem.Mol)
            
            # Second item should be a dictionary with scores
            self.assertIsInstance(scores, dict)
            self.assertIn("shape", scores)
            self.assertIn("color", scores)
            self.assertIn("combo", scores)
    
    def test_select_best_no_conformers(self):
        """Test select_best with molecule having no conformers."""
        # Create molecule with no conformers
        mol_no_confs = Chem.MolFromSmiles('CCO')
        mol_no_confs.RemoveAllConformers()
        
        # Run select_best
        best = select_best(mol_no_confs, self.mol2)
        
        # Should return default values
        for metric in ["shape", "color", "combo"]:
            self.assertIn(metric, best)
            mol, scores = best[metric]
            self.assertIsNone(mol)
            self.assertEqual(scores["shape"], -1.0)
            self.assertEqual(scores["color"], -1.0)
            self.assertEqual(scores["combo"], -1.0)
    
    def test_generate_properties_for_sdf(self):
        """Test property generation for SDF output."""
        # Test molecule
        mol = Chem.MolFromSmiles('CCO')
        mol.SetProp("_Name", "test_mol")
        
        # Template info
        template_info = {
            "embedding_similarity": "0.85",
            "ref_chains": "A",
            "mob_chains": "B"
        }
        
        # Generate properties
        result = generate_properties_for_sdf(mol, "shape", 0.75, "1abc", template_info)
        
        # Verify properties were set
        self.assertEqual(result.GetProp("_Name"), "1abc_shape_pose")
        self.assertEqual(result.GetProp("metric"), "shape")
        self.assertEqual(result.GetProp("metric_score"), "0.750")
        self.assertEqual(result.GetProp("template_pid"), "1abc")
        
        # Verify template info properties
        self.assertEqual(result.GetProp("template_embedding_similarity"), "0.85")
        self.assertEqual(result.GetProp("template_ref_chains"), "A")
        self.assertEqual(result.GetProp("template_mob_chains"), "B")
        
        # Verify original molecule was not modified
        self.assertEqual(mol.GetProp("_Name"), "test_mol")
        self.assertFalse(mol.HasProp("template_pid"))

if __name__ == '__main__':
    unittest.main()
