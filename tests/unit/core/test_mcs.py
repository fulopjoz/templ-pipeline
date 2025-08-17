# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Unit tests for the TEMPL Pipeline MCS module.

These tests verify the functionality of the MCS (Maximum Common Substructure)
module, including MCS identification and constrained embedding.
"""

import unittest
from unittest.mock import patch
import os
import logging
import tempfile
import shutil
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path

# Handle imports for both development and installed package
try:
    from templ_pipeline.core.mcs import (
        find_mcs,
        constrained_embed,
        simple_minimize_molecule,
        safe_name,
        # transform_ligand,
    )
    from templ_pipeline.core.embedding import EmbeddingManager
except ImportError:
    # Fall back to local imports for development
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    )
    from core.mcs import (
        find_mcs,
        constrained_embed,
        simple_minimize_molecule,
        safe_name,
        # transform_ligand,
    )
    from core.embedding import EmbeddingManager

# Import test helper functions from local tests package
sys.path.insert(0, os.path.dirname(__file__))
from tests import get_test_data_path

# Configure logging for tests
logging.basicConfig(level=logging.ERROR)


class TestMCS(unittest.TestCase):
    """Test cases for MCS module."""

    def setUp(self):
        """Set up test fixtures."""
        # Use standardized test molecules
        try:
            from tests.fixtures.data_factory import TestDataFactory
            
            # Create standard test molecules
            ethanol_data = TestDataFactory.create_molecule_data('simple_alkane')
            aromatic_data = TestDataFactory.create_molecule_data('aromatic')
            self.mol1 = ethanol_data['mol']  # Ethanol
            self.mol4 = aromatic_data['mol']  # Benzene
            
            # Create additional molecules for MCS testing
            self.mol2 = Chem.MolFromSmiles("CCOC")  # Methyl ethyl ether
            self.mol3 = Chem.MolFromSmiles("CCCC")  # Butane
            
        except ImportError:
            # Fallback to original molecule creation
            self.mol1 = Chem.MolFromSmiles("CCO")  # Ethanol
            self.mol2 = Chem.MolFromSmiles("CCOC")  # Methyl ethyl ether
            self.mol3 = Chem.MolFromSmiles("CCCC")  # Butane
            self.mol4 = Chem.MolFromSmiles("c1ccccc1")  # Benzene

        # Ensure test molecules were created successfully
        if any(mol is None for mol in [self.mol1, self.mol2, self.mol3, self.mol4]):
            self.fail("Failed to create one or more test molecules")

        # Ensure molecules have 3D coordinates for constrained embedding
        for mol in [self.mol1, self.mol2, self.mol3, self.mol4]:
            mol.SetProp("_Name", mol.GetNumAtoms() * "X")  # Set some name for testing
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)

    def test_safe_name(self):
        """Test the safe_name function."""
        # Test with molecule that has a name
        self.mol1.SetProp("_Name", "TestMol")
        self.assertEqual(
            safe_name(self.mol1, "Default"),
            "TestMol",
            "safe_name should return existing name when available",
        )

        # Test with molecule that has no name
        self.mol2.ClearProp("_Name")
        self.assertEqual(
            safe_name(self.mol2, "Default"),
            "Default", 
            "safe_name should return default name when molecule has no name",
        )
        
        # Test with string input
        self.assertEqual(
            safe_name("test_name", "Default"),
            "test_name",
            "safe_name should return cleaned string name",
        )

    def test_find_mcs_simple(self):
        """Test MCS finding between simple molecules."""
        # Ethanol and Methyl ethyl ether should share a CC substructure
        idx, smarts = find_mcs(self.mol1, [self.mol2])
        self.assertEqual(
            idx, 0, "MCS index should be 0 for the first reference molecule"
        )
        self.assertIsNotNone(smarts, "MCS SMARTS pattern should not be None")

        # Verify the SMARTS can match in both molecules
        patt = Chem.MolFromSmarts(smarts)
        self.assertIsNotNone(
            patt, "Failed to create RDKit molecule from SMARTS pattern"
        )
        self.assertTrue(
            self.mol1.HasSubstructMatch(patt),
            "Query molecule should match the MCS pattern",
        )
        self.assertTrue(
            self.mol2.HasSubstructMatch(patt),
            "Reference molecule should match the MCS pattern",
        )

    def test_find_mcs_multiple_refs(self):
        """Test MCS finding with multiple reference molecules."""
        idx, smarts = find_mcs(self.mol1, [self.mol3, self.mol2])
        self.assertIsNotNone(idx, "MCS index should not be None")
        self.assertIsNotNone(smarts, "MCS SMARTS pattern should not be None")

        # Natural MCS selection: algorithm selects template with highest MCS score
        # mol3 (butane) has 2 atoms matched, mol2 (methyl ethyl ether) has 3 atoms matched
        # Template 1 (mol2) is selected because it has the better MCS score (3 > 2)
        self.assertEqual(
            idx, 1, "Template 1 is selected due to natural MCS scoring (highest score wins)"
        )

    def test_find_mcs_no_match(self):
        """Test MCS finding with no possible match."""
        # Ethanol and Benzene have no meaningful common substructure
        idx, smarts = find_mcs(self.mol1, [self.mol4])
        # This should return None, None or a minimal match
        if idx is not None:
            # If a match is found, ensure it's minimal
            patt = Chem.MolFromSmarts(smarts)
            matches1 = self.mol1.GetSubstructMatches(patt)
            matches4 = self.mol4.GetSubstructMatches(patt)
            # The match should be small (1-2 atoms)
            self.assertLessEqual(
                len(matches1[0]) if matches1 else 0,
                2,
                "Match in query molecule should be minimal (1-2 atoms)",
            )
            self.assertLessEqual(
                len(matches4[0]) if matches4 else 0,
                2,
                "Match in reference molecule should be minimal (1-2 atoms)",
            )

    def test_simple_minimize_molecule_mmff(self):
        """Test simple MMFF minimization on a standard molecule."""
        # Create a simple molecule with a conformer
        mol = Chem.MolFromSmiles("CCO")
        if mol is None:
            self.fail("Failed to create test molecule for MMFF minimization")

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)

        # Get initial energy (if available)
        result = simple_minimize_molecule(mol)
        
        # Check that minimization reported success
        self.assertTrue(result, "Simple minimization should succeed for standard molecule")

    def test_simple_minimize_molecule_no_conformer(self):
        """Test simple minimization behavior with no conformers."""
        # Create a molecule without conformers
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        
        # Should return False for molecules without conformers
        result = simple_minimize_molecule(mol)
        self.assertFalse(result, "Simple minimization should fail for molecule without conformers")

    def test_simple_minimize_molecule_multiple_conformers(self):
        """Test simple minimization with multiple conformers."""
        # Create a molecule with multiple conformers
        mol = Chem.MolFromSmiles("CCO")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, 3)
        
        # Should handle multiple conformers
        result = simple_minimize_molecule(mol)
        self.assertTrue(result, "Simple minimization should succeed for molecule with multiple conformers")

    def test_constrained_embed_valid_mcs(self):
        """Test constrained embedding with a valid MCS."""
        # Create molecules with a common substructure
        ref = Chem.MolFromSmiles("CCO")
        tgt = Chem.MolFromSmiles("CCOC")
        if ref is None or tgt is None:
            self.fail("Failed to create test molecules for constrained embedding")

        AllChem.EmbedMolecule(ref)

        # Find MCS
        _, smarts = find_mcs(tgt, [ref])
        self.assertIsNotNone(smarts, "MCS pattern should be found")

        # Perform constrained embedding with correct parameter name n_conformers
        result = constrained_embed(tgt, ref, smarts, n_conformers=5)

        # Verify result has conformers
        self.assertGreater(
            result.GetNumConformers(),
            0,
            "Constrained embedding should generate conformers",
        )

        # Verify atoms are positioned (not at origin)
        conf = result.GetConformer(0)
        positions = [conf.GetAtomPosition(i) for i in range(result.GetNumAtoms())]
        non_origin = any(
            abs(pos.x) > 0.1 or abs(pos.y) > 0.1 or abs(pos.z) > 0.1
            for pos in positions
        )
        self.assertTrue(non_origin, "At least some atoms should be away from origin")

    def test_constrained_embed_invalid_mcs(self):
        """Test constrained embedding with an invalid MCS."""
        # Use benzene and ethanol which have no meaningful common substructure
        ref = self.mol4  # Benzene
        tgt = self.mol1  # Ethanol

        # Create an intentionally invalid SMARTS
        invalid_smarts = "C"  # Just a single carbon atom

        # Perform constrained embedding with invalid MCS - use correct parameter name
        result = constrained_embed(tgt, ref, invalid_smarts, n_conformers=3)

        # It should fall back to unconstrained embedding and still generate conformers
        self.assertGreater(
            result.GetNumConformers(),
            0,
            "Even with invalid MCS, embedding should generate conformers",
        )

    def test_constrained_embed_with_parallel(self):
        """Test constrained embedding with parallel processing."""
        ref = Chem.MolFromSmiles("CCO")
        tgt = Chem.MolFromSmiles("CCOS")  # Added a sulfur atom
        if ref is None or tgt is None:
            self.fail(
                "Failed to create test molecules for parallel constrained embedding"
            )

        AllChem.EmbedMolecule(ref)

        # Find MCS
        _, smarts = find_mcs(tgt, [ref])

        # Perform constrained embedding with parallel processing - use correct parameter name
        result = constrained_embed(tgt, ref, smarts, n_conformers=5, n_workers_pipeline=2)

        # Verify result has conformers
        self.assertGreater(
            result.GetNumConformers(),
            0,
            "Constrained embedding with parallel should generate conformers",
        )


class TestMCSWithRealData(unittest.TestCase):
    """Test the MCS pipeline with real PDBbind data."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Use helper function for path resolution
        cls.embedding_path = get_test_data_path("embeddings")

        # Test PDB IDs (one from each set)
        cls.test_pdb_other = "1a0q"
        cls.test_pdb_refined = "1a1e"

        # Get paths using helper function
        pdbbind_other_path = get_test_data_path("pdbbind_other")
        pdbbind_refined_path = get_test_data_path("pdbbind_refined")

        # Initialize with None in case paths don't exist
        cls.test_pdb_other_file = None
        cls.test_pdb_refined_file = None
        cls.test_ligand_other_file = None
        cls.test_ligand_refined_file = None

        # Set paths if they exist
        if pdbbind_other_path and os.path.exists(
            os.path.join(pdbbind_other_path, cls.test_pdb_other)
        ):
            cls.test_pdb_other_file = os.path.join(
                pdbbind_other_path,
                cls.test_pdb_other,
                f"{cls.test_pdb_other}_protein.pdb",
            )
            cls.test_ligand_other_file = os.path.join(
                pdbbind_other_path,
                cls.test_pdb_other,
                f"{cls.test_pdb_other}_ligand.sdf",
            )

        if pdbbind_refined_path and os.path.exists(
            os.path.join(pdbbind_refined_path, cls.test_pdb_refined)
        ):
            cls.test_pdb_refined_file = os.path.join(
                pdbbind_refined_path,
                cls.test_pdb_refined,
                f"{cls.test_pdb_refined}_protein.pdb",
            )
            cls.test_ligand_refined_file = os.path.join(
                pdbbind_refined_path,
                cls.test_pdb_refined,
                f"{cls.test_pdb_refined}_ligand.sdf",
            )

        # Skip if essential files are missing
        required_files = [
            cls.embedding_path,
            cls.test_pdb_other_file,
            cls.test_pdb_refined_file,
            cls.test_ligand_other_file,
            cls.test_ligand_refined_file,
        ]

        if any(not f or not os.path.exists(f) for f in required_files):
            # Don't set up the rest if files are missing
            cls.embedding_manager = None
            cls.test_ligand_other = None
            cls.test_ligand_refined = None
            return

        # Create a temp directory for cache
        cls.temp_dir = tempfile.mkdtemp()
        cls.cache_dir = os.path.join(cls.temp_dir, "embedding_cache")
        os.makedirs(cls.cache_dir, exist_ok=True)

        # Initialize embedding manager
        try:
            cls.embedding_manager = EmbeddingManager(
                cls.embedding_path,
                use_cache=True,
                cache_dir=cls.cache_dir,
                enable_batching=True,
                max_batch_size=2,
            )
        except Exception as e:
            cls.embedding_manager = None
            print(f"Failed to initialize EmbeddingManager: {e}")
            return

        # Load test ligands
        try:
            cls.test_ligand_other = next(Chem.SDMolSupplier(cls.test_ligand_other_file))
            cls.test_ligand_refined = next(
                Chem.SDMolSupplier(cls.test_ligand_refined_file)
            )

            # Ensure ligands have 3D coordinates for MCS tests
            if cls.test_ligand_other.GetNumConformers() == 0:
                cls.test_ligand_other = Chem.AddHs(cls.test_ligand_other)
                AllChem.EmbedMolecule(cls.test_ligand_other)
                AllChem.MMFFOptimizeMolecule(cls.test_ligand_other)

            if cls.test_ligand_refined.GetNumConformers() == 0:
                cls.test_ligand_refined = Chem.AddHs(cls.test_ligand_refined)
                AllChem.EmbedMolecule(cls.test_ligand_refined)
                AllChem.MMFFOptimizeMolecule(cls.test_ligand_refined)
        except Exception as e:
            cls.test_ligand_other = None
            cls.test_ligand_refined = None
            print(f"Failed to load test ligands: {e}")

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        if hasattr(cls, "temp_dir") and os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def setUp(self):
        """Per-test setup - replaced with improved version in test_mcs_improved.py."""
        pass

    def test_find_mcs_with_real_ligands(self):
        """Test finding MCS between real ligands."""
        # Replaced with improved version in test_mcs_improved.py
        pass

    def test_constrained_embed(self):
        """Test constrained embedding with real molecules."""
        # Replaced with improved version in test_mcs_improved.py
        pass


class TestMCSErrorHandling(unittest.TestCase):
    """Test error handling and edge cases in MCS functionality."""
    
    def test_find_mcs_invalid_molecules(self):
        """Test MCS finding with invalid molecules."""
        # Test with None molecules
        try:
            result = find_mcs(None, [None])
            self.assertIsNone(result[0] if result else None)
        except (AttributeError, TypeError):
            # Acceptable to raise error for None input
            pass
        
        # Test with empty molecule list
        mol1 = Chem.MolFromSmiles("CCO") 
        try:
            result = find_mcs(mol1, [])
            # Should handle empty list gracefully
            self.assertIsNotNone(result)
        except (AttributeError, TypeError, IndexError, ValueError):
            # Acceptable to raise error for empty list (ValueError from max() on empty sequence)
            pass
        
        # Test with invalid molecule objects  
        try:
            result = find_mcs("invalid", [mol1])
            self.fail("Should raise error for invalid target molecule")
        except (AttributeError, TypeError):
            # Expected to raise error for invalid input
            pass
    
    def test_find_mcs_completely_different_molecules(self):
        """Test MCS finding with completely different molecules."""
        # Very different molecules
        mol1 = Chem.MolFromSmiles("CCCCCCCCCC")  # Alkane
        mol2 = Chem.MolFromSmiles("c1ccccc1")   # Benzene
        
        idx, smarts = find_mcs(mol1, [mol2])
        
        # Should find minimal common structure or return None
        if smarts:
            # Should be very simple pattern like single carbon
            self.assertLessEqual(len(smarts), 10)
    
    def test_find_mcs_single_atom_molecules(self):
        """Test MCS finding with single atom molecules."""
        mol1 = Chem.MolFromSmiles("C")
        mol2 = Chem.MolFromSmiles("C")
        
        idx, smarts = find_mcs(mol1, [mol2])
        
        # Should handle single atom molecules gracefully
        # May return "*" as fallback or "[#6]" as specific pattern
        if smarts:
            self.assertIn(smarts, ["*", "[#6]"])  # Accept either fallback or specific pattern
    
    def test_find_mcs_empty_molecules(self):
        """Test MCS finding with molecules with no atoms."""
        # Create empty molecule
        empty_mol = Chem.Mol()
        normal_mol = Chem.MolFromSmiles("CCO")
        
        try:
            result = find_mcs(empty_mol, [normal_mol])
            # Function should handle empty molecules gracefully 
            # May return (0, "*") for fallback case
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            idx, smarts = result
            self.assertIn(smarts, ["*", None])  # Fallback or None
        except (AttributeError, TypeError):
            # Acceptable to raise error for empty molecule
            pass
    
    def test_constrained_embed_invalid_molecule(self):
        """Test constrained embedding with invalid molecule."""
        ref_mol = Chem.MolFromSmiles("CC")
        AllChem.EmbedMolecule(ref_mol)  # Give it 3D coords
        smarts = "[#6][#6]"  # Simple carbon-carbon pattern
        
        # Test with None target molecule
        try:
            result = constrained_embed(None, ref_mol, smarts)
            self.assertIsNone(result)
        except (AttributeError, TypeError):
            # Acceptable to raise error for None input
            pass
        
        # Test with invalid molecule object
        try:
            result = constrained_embed("invalid", ref_mol, smarts)
            self.fail("Should raise error for invalid target molecule")
        except (AttributeError, TypeError):
            # Expected to raise error for invalid input
            pass
    
    def test_constrained_embed_invalid_pattern(self):
        """Test constrained embedding with invalid SMARTS pattern."""
        tgt_mol = Chem.MolFromSmiles("CCOC")
        ref_mol = Chem.MolFromSmiles("CCO")
        AllChem.EmbedMolecule(ref_mol)  # Give it 3D coords
        
        # Test with invalid SMARTS pattern
        try:
            result = constrained_embed(tgt_mol, ref_mol, "INVALID_SMARTS")
            # Should handle gracefully - might return None or use fallback
            self.assertTrue(result is None or isinstance(result, Chem.Mol))
        except (AttributeError, TypeError):
            # Acceptable to raise error for invalid pattern
            pass
    
    def test_constrained_embed_incompatible_pattern(self):
        """Test constrained embedding with incompatible pattern."""
        mol = Chem.MolFromSmiles("CCO")
        incompatible_pattern = Chem.MolFromSmiles("c1ccccc1")  # Benzene vs alkane
        
        # First find MCS to get smarts pattern (should be very small or none)
        _, smarts = find_mcs(mol, [incompatible_pattern])
        
        if smarts and smarts != "*":
            result = constrained_embed(mol, incompatible_pattern, smarts)
            # Should handle incompatible patterns gracefully
            self.assertIsInstance(result, (list, type(None)))
        else:
            # No meaningful MCS found - this is expected for incompatible patterns
            self.assertIn(smarts, [None, "*"])  # Should be None or fallback pattern
    
    def test_constrained_embed_zero_conformers(self):
        """Test constrained embedding with zero conformers requested."""
        mol = Chem.MolFromSmiles("CCOC")
        pattern = Chem.MolFromSmiles("CC")
        
        # First find MCS to get smarts pattern
        _, smarts = find_mcs(mol, [pattern])
        self.assertIsNotNone(smarts, "Should find MCS between CCOC and CC")
        
        result = constrained_embed(mol, pattern, smarts, n_conformers=0)
        
        # Should return empty list, None, or single Mol for zero conformers (fallback case)
        self.assertIsInstance(result, (list, type(None), Chem.Mol))
    
    def test_constrained_embed_large_number_conformers(self):
        """Test constrained embedding with very large number of conformers."""
        mol = Chem.MolFromSmiles("CCOC")
        pattern = Chem.MolFromSmiles("CC")
        
        # First find MCS to get smarts pattern
        _, smarts = find_mcs(mol, [pattern])
        self.assertIsNotNone(smarts, "Should find MCS between CCOC and CC")
        
        # Request many conformers (use smaller number for testing)
        result = constrained_embed(mol, pattern, smarts, n_conformers=1000)
        
        # Should handle large requests gracefully (may return fewer than requested or single Mol for fallback)
        self.assertIsInstance(result, (list, type(None), Chem.Mol))
        if isinstance(result, list):
            self.assertLessEqual(len(result), 1000)
    

class TestMCSParametrized(unittest.TestCase):
    """Test MCS functionality with parametrized inputs."""
    
    def test_find_mcs_various_molecule_pairs(self):
        """Test MCS finding with various molecule pairs."""
        # Use standardized MCS test pairs
        try:
            from tests.fixtures.data_factory import TestDataFactory
            test_pairs = TestDataFactory.create_mcs_test_pairs()
        except ImportError:
            # Fallback test pairs
            test_pairs = [
                ("CCO", "CCC", "Similar alkanes"),
                ("CCO", "CCOC", "Alcohol vs ether"),
                ("c1ccccc1", "c1ccccc1O", "Benzene vs phenol"),
                ("CCCC", "C", "Chain vs single carbon"),
                ("CCO", "O", "Ethanol vs water"),
                ("C1CCCCC1", "c1ccccc1", "Cyclohexane vs benzene"),
                ("CC(C)C", "CCC", "Branched vs linear"),
            ]
        
        for smiles1, smiles2, description in test_pairs:
            with self.subTest(smiles1=smiles1, smiles2=smiles2, desc=description):
                mol1 = Chem.MolFromSmiles(smiles1)
                mol2 = Chem.MolFromSmiles(smiles2)
                
                idx, smarts = find_mcs(mol1, [mol2])
                
                # All should return some result (even if no MCS found])
                self.assertIsInstance(idx, (int, type(None)))
                self.assertIsInstance(smarts, (str, type(None)))
    
    def test_constrained_embed_various_patterns(self):
        """Test constrained embedding with various pattern types."""
        base_molecule = "CCCCCCCC"  # Octane
        test_patterns = [
            ("C", "Single carbon"),
            ("CC", "Two carbons"),
            ("CCC", "Three carbons"),
            ("CCCC", "Four carbons"),
            ("C=C", "Double bond"),
            ("C#C", "Triple bond"),
        ]
        
        mol = Chem.MolFromSmiles(base_molecule)
        
        for pattern_smiles, description in test_patterns:
            with self.subTest(pattern=pattern_smiles, desc=description):
                pattern = Chem.MolFromSmiles(pattern_smiles)
                
                # First find MCS to get smarts pattern
                _, smarts = find_mcs(mol, [pattern])
                
                if smarts:  # Only test if MCS found
                    result = constrained_embed(mol, pattern, smarts, n_conformers=5)
                    
                    # Should return a list, None, or single Mol (fallback case)
                    self.assertIsInstance(result, (list, type(None), Chem.Mol))
                    
                    # If conformers generated, they should be valid
                    if result:
                        if isinstance(result, list):
                            for conf_mol in result:
                                self.assertIsNotNone(conf_mol)
                                self.assertGreater(conf_mol.GetNumAtoms(), 0)
                        elif isinstance(result, Chem.Mol):
                            self.assertGreater(result.GetNumAtoms(), 0)
    
    def test_constrained_embed_different_conformer_counts(self):
        """Test constrained embedding with different conformer counts."""
        mol = Chem.MolFromSmiles("CCCCCCCC")
        pattern = Chem.MolFromSmiles("CC")
        
        # First find MCS to get smarts pattern
        _, smarts = find_mcs(mol, [pattern])
        self.assertIsNotNone(smarts, "Should find MCS between CCCCCCCC and CC")
        
        conformer_counts = [1, 5, 10, 50, 100]
        
        for num_conf in conformer_counts:
            with self.subTest(n_conformers=num_conf):
                result = constrained_embed(mol, pattern, smarts, n_conformers=num_conf)
                
                self.assertIsInstance(result, (list, type(None), Chem.Mol))
                # May return fewer conformers than requested (if list) or single Mol (fallback)
                if isinstance(result, list):
                    self.assertLessEqual(len(result), num_conf)
    
    def test_mcs_similarity_thresholds(self):
        """Test MCS behavior with different similarity thresholds."""
        mol1 = Chem.MolFromSmiles("CCCCCCCC")
        mol2 = Chem.MolFromSmiles("CCCCC")
        
        # Test different approaches to MCS finding
        idx, smarts = find_mcs(mol1, [mol2])
        
        if smarts:
            # Pattern should be reasonable
            pattern_mol = Chem.MolFromSmarts(smarts)
            self.assertIsNotNone(pattern_mol)
            
            # Pattern should be simpler than either input molecule
            self.assertLessEqual(pattern_mol.GetNumAtoms(), mol1.GetNumAtoms())
            self.assertLessEqual(pattern_mol.GetNumAtoms(), mol2.GetNumAtoms())


class TestMCSBoundaryConditions(unittest.TestCase):
    """Test boundary conditions and edge cases in MCS functionality."""
    
    def test_find_mcs_identical_molecules(self):
        """Test MCS finding with identical molecules."""
        mol = Chem.MolFromSmiles("CCCCCCCC")
        
        idx, smarts = find_mcs(mol, [mol])
        
        # Should find the entire molecule as MCS
        if smarts:
            pattern_mol = Chem.MolFromSmarts(smarts)
            # Pattern should match the original molecule size
            self.assertEqual(pattern_mol.GetNumAtoms(), mol.GetNumAtoms())
    
    def test_find_mcs_very_large_molecules(self):
        """Test MCS finding with very large molecules."""
        # Generate large molecules
        large_smiles1 = "C" + "C" * 100  # 101 carbons
        large_smiles2 = "C" + "C" * 95   # 96 carbons
        
        mol1 = Chem.MolFromSmiles(large_smiles1)
        mol2 = Chem.MolFromSmiles(large_smiles2)
        
        if mol1 and mol2:
            idx, smarts = find_mcs(mol1, [mol2])
            
            # Should handle large molecules without crashing
            self.assertIsInstance(idx, (int, type(None)))
            self.assertIsInstance(smarts, (str, type(None)))
    
    def test_constrained_embed_complex_molecule(self):
        """Test constrained embedding with complex molecule structures."""
        complex_molecules = [
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "C1CCC(CC1)N2CCN(CC2)C3=CC=CC=C3",  # Complex cyclic
            "CC1=CC=CC=C1C2=CC=CC=C2C(=O)NCCN(C)C",  # Multi-ring
        ]
        
        simple_pattern = Chem.MolFromSmiles("CC")
        
        for smiles in complex_molecules:
            with self.subTest(molecule=smiles):
                mol = Chem.MolFromSmiles(smiles)
                
                if mol:
                    # First find MCS to get smarts pattern
                    _, smarts = find_mcs(mol, [simple_pattern])
                    
                    if smarts:  # Only test if MCS found
                        result = constrained_embed(mol, simple_pattern, smarts, n_conformers=3)
                        
                        # Should handle complex molecules gracefully
                        self.assertIsInstance(result, (list, type(None), Chem.Mol))
    
    def test_constrained_embed_with_stereochemistry(self):
        """Test constrained embedding with stereochemical molecules."""
        # Molecules with stereocenters
        stereo_molecules = [
            "C[C@H](O)C",     # Chiral center
            "C[C@@H](O)C",    # Opposite chirality
            "C/C=C/C",        # E alkene
            "C/C=C\\C",       # Z alkene
        ]
        
        pattern = Chem.MolFromSmiles("CC")
        
        for smiles in stereo_molecules:
            with self.subTest(molecule=smiles):
                mol = Chem.MolFromSmiles(smiles)
                
                if mol:
                    # First find MCS to get smarts pattern
                    _, smarts = find_mcs(mol, [pattern])
                    
                    if smarts:  # Only test if MCS found
                        result = constrained_embed(mol, pattern, smarts, n_conformers=3)
                        
                        # Should handle stereochemistry appropriately
                        self.assertIsInstance(result, (list, type(None), Chem.Mol))
    
    def test_memory_intensive_operations(self):
        """Test MCS operations that might be memory intensive."""
        # Create moderately complex molecules for memory testing
        mol1 = Chem.MolFromSmiles("CCCCCCCCCCCCCCCCCCCC")  # 20 carbons
        mol2 = Chem.MolFromSmiles("CCCCCCCCCCCCCCCCC")     # 17 carbons
        
        if mol1 and mol2:
            # Test multiple MCS operations
            for i in range(10):
                with self.subTest(iteration=i):
                    idx, smarts = find_mcs(mol1, [mol2])
                    
                    # Should complete without memory issues
                    self.assertIsInstance(idx, (int, type(None)))
                    self.assertIsInstance(smarts, (str, type(None)))
    
    def test_concurrent_mcs_operations(self):
        """Test MCS operations that might run concurrently."""
        import threading
        import time
        
        results = []
        errors = []
        
        def mcs_worker():
            try:
                mol1 = Chem.MolFromSmiles("CCCCCCCC")
                mol2 = Chem.MolFromSmiles("CCCCC")
                
                idx, smarts = find_mcs(mol1, [mol2])
                results.append((idx, smarts))
            except Exception as e:
                errors.append(e)
        
        # Run multiple MCS operations concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=mcs_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Check that operations completed successfully
        self.assertEqual(len(errors), 0, f"Concurrent MCS operations failed: {errors}")
        self.assertEqual(len(results), 5, "Not all concurrent operations completed")


if __name__ == "__main__":
    unittest.main()
