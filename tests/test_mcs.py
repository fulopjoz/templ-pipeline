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
        mmff_minimise_fixed_parallel,
        mmff_minimise_fixed_sequential,
        safe_name,
        transform_ligand,
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
        mmff_minimise_fixed_parallel,
        mmff_minimise_fixed_sequential,
        safe_name,
        transform_ligand,
    )
    from core.embedding import EmbeddingManager

# Import test helper functions from local tests package
sys.path.insert(0, os.path.dirname(__file__))
from . import get_test_data_path

# Configure logging for tests
logging.basicConfig(level=logging.ERROR)


class TestMCS(unittest.TestCase):
    """Test cases for MCS module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create simple test molecules for MCS testing
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
        # Verify the property was set
        self.assertEqual(
            self.mol2.GetProp("_Name"),
            "Default",
            "safe_name should set the name property on the molecule",
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

        # The best match should be with mol2 (Methyl ethyl ether)
        self.assertEqual(
            idx, 1, "Best MCS match should be with the second reference molecule"
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

    def test_mmff_minimise_fixed_sequential(self):
        """Test sequential MMFF minimization with fixed atoms."""
        # Create a simple molecule with a conformer
        mol = Chem.MolFromSmiles("CCO")
        if mol is None:
            self.fail("Failed to create test molecule for MMFF minimization")

        AllChem.EmbedMultipleConfs(mol, 2)

        # Get initial coordinates
        init_pos = [mol.GetConformer(0).GetAtomPosition(0)]
        init_pos_x = init_pos[0].x

        # Fix the first atom and minimize
        mmff_minimise_fixed_sequential(mol, [0], [0])

        # Check that the fixed atom didn't move
        final_pos = mol.GetConformer(0).GetAtomPosition(0)
        self.assertAlmostEqual(
            final_pos.x,
            init_pos_x,
            places=4,
            msg="Fixed atom position should not change during minimization",
        )

    @patch("templ_pipeline.core.mcs.ProcessPoolExecutor")
    def test_mmff_minimise_fixed_parallel(self, mock_executor):
        """Test parallel MMFF minimization with fixed atoms."""
        # Setup mock executor to simulate parallel execution
        mock_executor.return_value.__enter__.return_value.map.return_value = [
            (0, [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            (1, [[-1.1, 0.1, 0.1], [0.1, 0.1, 0.1], [1.1, 0.1, 0.1]]),
        ]

        # Create a test molecule with conformers
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        if mol is None:
            self.fail("Failed to create test molecule for parallel MMFF minimization")

        conf_ids = AllChem.EmbedMultipleConfs(mol, 2)

        # Run the function with mocked parallel execution
        mmff_minimise_fixed_parallel(mol, conf_ids, fixed_idx=[0], n_workers=2)

        # Check that map was called appropriately
        self.assertTrue(
            mock_executor.return_value.__enter__.return_value.map.called,
            "ProcessPoolExecutor.map should be called in parallel minimization",
        )

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
        result = constrained_embed(tgt, ref, smarts, n_conformers=5, n_workers=2)

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
        """Per-test setup to check if we should skip tests."""
        if not self.embedding_path or not os.path.exists(self.embedding_path):
            self.skipTest(
                f"Embedding file not found at any of the expected paths. Tried: {get_test_data_path('embeddings')}"
            )
        if not hasattr(self, "embedding_manager") or self.embedding_manager is None:
            self.skipTest(
                "Embedding manager could not be initialized - check if embedding file exists and has the correct format"
            )
        if not self.test_ligand_other or not self.test_ligand_refined:
            self.skipTest(
                "Test ligands could not be loaded - check if ligand files exist and can be parsed"
            )

    def test_find_mcs_with_real_ligands(self):
        """Test finding MCS between real ligands."""
        # Remove Hs for MCS search
        mol1 = Chem.RemoveHs(self.test_ligand_other)
        mol2 = Chem.RemoveHs(self.test_ligand_refined)

        # Find MCS
        best_idx, smarts = find_mcs(mol1, [mol2])

        # Check results
        self.assertIsNotNone(best_idx, "MCS index should not be None")
        self.assertIsNotNone(smarts, "MCS SMARTS pattern should not be None")
        self.assertEqual(best_idx, 0, "Should match the only template")

        # Check that SMARTS is valid
        patt = Chem.MolFromSmarts(smarts)
        self.assertIsNotNone(patt, "MCS pattern should be a valid SMARTS")

        # Check that both molecules match the pattern
        matches1 = mol1.GetSubstructMatches(patt)
        matches2 = mol2.GetSubstructMatches(patt)
        self.assertGreater(
            len(matches1), 0, "Query molecule should match the MCS pattern"
        )
        self.assertGreater(
            len(matches2), 0, "Template molecule should match the MCS pattern"
        )

    def test_constrained_embed(self):
        """Test constrained embedding with real molecules."""
        # Remove Hs for MCS search
        mol1 = Chem.RemoveHs(self.test_ligand_other)
        mol2 = Chem.RemoveHs(self.test_ligand_refined)

        # Find MCS
        best_idx, smarts = find_mcs(mol1, [mol2])

        # Generate conformers with constrained embedding
        if smarts:
            # Use small number of conformers for fast testing - correct parameter name
            n_conformers = 3
            confs = constrained_embed(mol1, mol2, smarts, n_conformers=n_conformers)

            # Check that conformers were generated
            self.assertIsNotNone(
                confs, "constrained_embed should return a molecule with conformers"
            )
            self.assertGreaterEqual(
                confs.GetNumConformers(),
                1,
                f"Should generate at least 1 conformer, got {confs.GetNumConformers()}",
            )

            # Check that conformers have 3D coordinates
            for i in range(confs.GetNumConformers()):
                conf = confs.GetConformer(i)
                pos = conf.GetAtomPosition(0)
                self.assertTrue(
                    hasattr(pos, "x"), "Conformer positions should have x coordinate"
                )
                self.assertTrue(
                    hasattr(pos, "y"), "Conformer positions should have y coordinate"
                )
                self.assertTrue(
                    hasattr(pos, "z"), "Conformer positions should have z coordinate"
                )
        else:
            self.skipTest("Could not generate MCS pattern for constrained embedding")


if __name__ == "__main__":
    unittest.main()
