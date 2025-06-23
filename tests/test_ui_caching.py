"""
Tests for UI caching functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Mock streamlit before importing app
mock_st = MagicMock()

# Create a flexible cache decorator that handles both @decorator and @decorator() syntax
def mock_cache_decorator(func=None, **kwargs):
    """Mock cache decorator that works with or without parentheses"""
    def decorator(f):
        return f
    
    # If called without parentheses (@cache_data), func will be the decorated function
    if func is not None and callable(func) and not kwargs:
        return func
    
    # If called with parentheses (@cache_data()), return the decorator
    return decorator

mock_st.cache_data = mock_cache_decorator
mock_st.cache_resource = mock_cache_decorator

sys.modules['streamlit'] = mock_st
sys.modules['py3Dmol'] = MagicMock()
sys.modules['stmol'] = MagicMock()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from templ_pipeline.ui.app import validate_smiles_input, generate_molecule_image


class TestUICaching:
    """Test caching mechanisms in the UI"""
    
    def test_smiles_validation_caching(self):
        """Test that SMILES validation results are cached"""
        # Test with valid SMILES
        smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
        
        with patch('templ_pipeline.ui.app.get_rdkit_modules') as mock_rdkit:
            # Mock RDKit modules
            mock_chem = MagicMock()
            mock_mol = MagicMock()
            mock_mol.GetNumAtoms.return_value = 17
            mock_mol.ToBinary.return_value = b"mock_binary_data"
            mock_chem.MolFromSmiles.return_value = mock_mol
            
            mock_rdkit.return_value = (mock_chem, MagicMock(), MagicMock())
            
            # First call should compute
            valid1, msg1, mol_data1 = validate_smiles_input(smiles)
            assert valid1 is True
            assert "Valid molecule (17 atoms)" in msg1
            assert mol_data1 == b"mock_binary_data"
            
            # Second call would be cached in real app
            valid2, msg2, mol_data2 = validate_smiles_input(smiles)
            assert valid1 == valid2
            assert msg1 == msg2
            assert mol_data1 == mol_data2
    
    def test_smiles_validation_edge_cases(self):
        """Test SMILES validation with edge cases"""
        with patch('templ_pipeline.ui.app.get_rdkit_modules') as mock_rdkit:
            mock_chem = MagicMock()
            mock_rdkit.return_value = (mock_chem, MagicMock(), MagicMock())
            
            # Empty SMILES
            valid, msg, mol_data = validate_smiles_input("")
            assert valid is False
            assert "Please enter a SMILES string" in msg
            assert mol_data is None
            
            # Invalid SMILES
            mock_chem.MolFromSmiles.return_value = None
            valid, msg, mol_data = validate_smiles_input("INVALID")
            assert valid is False
            assert "Invalid SMILES string format" in msg
            assert mol_data is None
            
            # Too small molecule
            mock_mol = MagicMock()
            mock_mol.GetNumAtoms.return_value = 2
            mock_chem.MolFromSmiles.return_value = mock_mol
            valid, msg, mol_data = validate_smiles_input("CC")
            assert valid is False
            assert "Molecule too small" in msg
            assert mol_data is None
    
    def test_molecule_image_caching(self):
        """Test that molecule images are cached"""
        # Create a mock molecule binary
        mock_mol_binary = b"test_molecule_binary_data"
        
        with patch('templ_pipeline.ui.app.get_rdkit_modules') as mock_rdkit:
            # Mock RDKit modules
            mock_chem = MagicMock()
            mock_allchem = MagicMock()
            mock_draw = MagicMock()
            mock_rdkit.return_value = (mock_chem, mock_allchem, mock_draw)
            
            # Mock molecule operations
            mock_mol = MagicMock()
            mock_mol_copy = MagicMock()
            mock_mol_copy.GetNumConformers.return_value = 0
            
            mock_chem.Mol.return_value = mock_mol
            mock_chem.RemoveHs.return_value = mock_mol_copy
            
            # Mock image generation
            expected_img = MagicMock()
            mock_draw.MolToImage.return_value = expected_img
            
            # First call
            img1 = generate_molecule_image(mock_mol_binary, 400, 300)
            assert img1 is expected_img
            
            # Verify the drawing was called correctly
            mock_draw.MolToImage.assert_called_once_with(
                mock_mol_copy, 
                size=(400, 300)
            )
    
    def test_molecule_image_with_highlights(self):
        """Test molecule image generation with atom highlighting"""
        mock_mol_binary = b"test_molecule_binary_data"
        highlight_atoms = (0, 1, 2)  # Tuple for hashability
        
        with patch('templ_pipeline.ui.app.get_rdkit_modules') as mock_rdkit:
            # Mock RDKit modules
            mock_chem = MagicMock()
            mock_allchem = MagicMock()
            mock_draw = MagicMock()
            mock_rdkit.return_value = (mock_chem, mock_allchem, mock_draw)
            
            # Mock molecule operations
            mock_mol = MagicMock()
            mock_mol_copy = MagicMock()
            mock_mol_copy.GetNumConformers.return_value = 1
            
            mock_chem.Mol.return_value = mock_mol
            mock_chem.RemoveHs.return_value = mock_mol_copy
            
            # Mock image generation
            expected_img = MagicMock()
            mock_draw.MolToImage.return_value = expected_img
            
            # Generate with highlights
            img = generate_molecule_image(mock_mol_binary, 400, 300, highlight_atoms)
            assert img is expected_img
            
            # Verify MolToImage was called with highlightAtoms
            mock_draw.MolToImage.assert_called_with(
                mock_mol_copy, 
                size=(400, 300), 
                highlightAtoms=highlight_atoms
            )
    
    def test_cache_key_differentiation(self):
        """Test that different inputs create different cache keys"""
        with patch('templ_pipeline.ui.app.get_rdkit_modules') as mock_rdkit:
            # Mock RDKit modules
            mock_chem = MagicMock()
            
            # Mock different molecules
            mock_mol1 = MagicMock()
            mock_mol1.GetNumAtoms.return_value = 3
            mock_mol1.ToBinary.return_value = b"mol1_binary"
            
            mock_mol2 = MagicMock()
            mock_mol2.GetNumAtoms.return_value = 4
            mock_mol2.ToBinary.return_value = b"mol2_binary"
            
            def mol_from_smiles(smiles):
                if smiles == "CCO":
                    return mock_mol1
                elif smiles == "CC(C)O":
                    return mock_mol2
                return None
            
            mock_chem.MolFromSmiles.side_effect = mol_from_smiles
            mock_rdkit.return_value = (mock_chem, MagicMock(), MagicMock())
            
            # Different SMILES should have different results
            smiles1 = "CCO"
            smiles2 = "CC(C)O" 
            
            valid1, msg1, data1 = validate_smiles_input(smiles1)
            valid2, msg2, data2 = validate_smiles_input(smiles2)
            
            # Both should be valid but different
            assert valid1 is True
            assert valid2 is True
            assert msg1 == "Valid molecule (3 atoms)"
            assert msg2 == "Valid molecule (4 atoms)"
            assert data1 != data2


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 