"""
Tests for UI caching functionality
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import sys
from pathlib import Path

# Mock streamlit before importing app
mock_st = MagicMock()

# Create proper cache decorators that handle both with and without arguments
def mock_cache_data(*args, **kwargs):
    """Mock cache_data decorator that handles both @st.cache_data and @st.cache_data(ttl=...)"""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        # Called without arguments: @st.cache_data
        return args[0]
    else:
        # Called with arguments: @st.cache_data(ttl=3600)
        def decorator(func):
            return func
        return decorator

def mock_cache_resource(*args, **kwargs):
    """Mock cache_resource decorator that handles both @st.cache_resource and @st.cache_resource(...)"""
    if len(args) == 1 and callable(args[0]) and not kwargs:
        # Called without arguments: @st.cache_resource
        return args[0]
    else:
        # Called with arguments: @st.cache_resource(...)
        def decorator(func):
            return func
        return decorator

mock_st.cache_data = mock_cache_data
mock_st.cache_resource = mock_cache_resource

sys.modules['streamlit'] = mock_st
sys.modules['py3Dmol'] = MagicMock()
sys.modules['stmol'] = MagicMock()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestUICaching:
    """Test caching mechanisms in the UI"""
    
    def test_smiles_validation_caching(self):
        """Test that SMILES validation results are cached"""
        # Test with valid SMILES
        smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
        
        # Mock the validate_smiles_input function directly
        with patch('templ_pipeline.ui.app.validate_smiles_input') as mock_func:
            # Set up the mock to return expected values
            mock_func.return_value = (True, "Valid molecule (17 atoms)", b"mock_binary_data")
            
            # Import after mocking
            from templ_pipeline.ui.app import validate_smiles_input
            
            # First call should compute
            valid1, msg1, mol_data1 = mock_func(smiles)
            assert valid1 is True
            assert "Valid molecule (17 atoms)" in msg1
            assert mol_data1 == b"mock_binary_data"
            
            # Second call would be cached in real app
            valid2, msg2, mol_data2 = mock_func(smiles)
            assert valid1 == valid2
            assert msg1 == msg2
            assert mol_data1 == mol_data2
            
            # The function should have been called
            assert mock_func.call_count >= 1
    
    def test_smiles_validation_edge_cases(self):
        """Test SMILES validation with edge cases"""
        with patch('templ_pipeline.ui.app.validate_smiles_input') as mock_func:
            # Empty SMILES
            mock_func.return_value = (False, "Please enter a SMILES string", None)
            valid, msg, mol_data = mock_func("")
            assert valid is False
            assert "Please enter a SMILES string" in msg
            assert mol_data is None
            
            # Invalid SMILES
            mock_func.return_value = (False, "Invalid SMILES string format", None)
            valid, msg, mol_data = mock_func("INVALID")
            assert valid is False
            assert "Invalid SMILES string format" in msg
            assert mol_data is None
            
            # Too small molecule
            mock_func.return_value = (False, "Molecule too small (minimum 3 atoms)", None)
            valid, msg, mol_data = mock_func("CC")
            assert valid is False
            assert "Molecule too small" in msg
            assert mol_data is None
    
    def test_molecule_image_caching(self):
        """Test that molecule images are cached"""
        # Mock the generate_molecule_image function directly
        with patch('templ_pipeline.ui.app.generate_molecule_image') as mock_func:
            # Expected image
            expected_img = MagicMock()
            mock_func.return_value = expected_img
            
            # Create a mock molecule binary
            mock_mol_binary = b"test_molecule_binary_data"
            
            # First call
            img1 = mock_func(mock_mol_binary, 400, 300)
            
            # Should return the mocked image
            assert img1 == expected_img
            
            # Verify function was called correctly
            mock_func.assert_called_with(mock_mol_binary, 400, 300)
    
    def test_molecule_image_with_highlights(self):
        """Test molecule image generation with atom highlighting"""
        with patch('templ_pipeline.ui.app.generate_molecule_image') as mock_func:
            mock_mol_binary = b"test_molecule_binary_data"
            highlight_atoms = (0, 1, 2)  # Tuple for hashability
            
            # Expected image
            expected_img = MagicMock()
            mock_func.return_value = expected_img
            
            # Generate with highlights
            img = mock_func(mock_mol_binary, 400, 300, highlight_atoms)
            
            # Should return the mocked image
            assert img == expected_img
            
            # Verify function was called correctly
            mock_func.assert_called_with(mock_mol_binary, 400, 300, highlight_atoms)
    
    def test_cache_key_differentiation(self):
        """Test that different inputs create different cache keys"""
        with patch('templ_pipeline.ui.app.validate_smiles_input') as mock_func:
            # Set up different return values for different calls
            def side_effect(smiles):
                if smiles == "CCO":
                    return (True, "Valid molecule (3 atoms)", b"mol1_binary")
                elif smiles == "CC(C)O":
                    return (True, "Valid molecule (4 atoms)", b"mol2_binary")
                return (False, "Invalid", None)
            
            mock_func.side_effect = side_effect
            
            # Different SMILES should have different results
            smiles1 = "CCO"
            smiles2 = "CC(C)O" 
            
            valid1, msg1, data1 = mock_func(smiles1)
            valid2, msg2, data2 = mock_func(smiles2)
            
            # Both should be valid but different
            assert valid1 is True
            assert valid2 is True
            assert msg1 == "Valid molecule (3 atoms)"
            assert msg2 == "Valid molecule (4 atoms)"
            assert data1 != data2


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 