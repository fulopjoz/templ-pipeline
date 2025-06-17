"""
Tests for templ_pipeline.core.chemistry module.
"""
import pytest
import sys
import os
from rdkit import Chem

# Handle imports for both development and installed package
try:
    from templ_pipeline.core.chemistry import (
        detect_and_substitute_organometallic,
        needs_uff_fallback,
        has_rhenium_complex,
        is_large_peptide,
        validate_target_molecule
    )
except ImportError:
    # Fall back to local imports for development
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from core.chemistry import (
        detect_and_substitute_organometallic,
        needs_uff_fallback,
        has_rhenium_complex,
        is_large_peptide,
        validate_target_molecule
    )


def test_detect_and_substitute_organometallic():
    """Test organometallic detection and substitution."""
    # Test with None molecule
    mol, modified, log = detect_and_substitute_organometallic(None)
    assert mol is None
    assert not modified
    assert "Input molecule is None" in log
    
    # Test with simple organic molecule (no metals)
    benzene = Chem.MolFromSmiles("c1ccccc1")
    mol, modified, log = detect_and_substitute_organometallic(benzene)
    assert mol is not None
    assert not modified
    assert "No organometallic atoms detected" in log
    
    # Test with copper complex (simplified)
    copper_mol = Chem.MolFromSmiles("[Cu]")
    if copper_mol:
        mol, modified, log = detect_and_substitute_organometallic(copper_mol, "test_cu")
        assert modified
        assert any("Found Cu" in entry for entry in log)
        assert any("Substituted Cu with C" in entry for entry in log)


def test_needs_uff_fallback():
    """Test UFF fallback detection."""
    # Test with None
    assert needs_uff_fallback(None) is True
    
    # Test with simple organic molecule
    benzene = Chem.MolFromSmiles("c1ccccc1")
    result = needs_uff_fallback(benzene)
    # Should return False for simple organics (MMFF should work)
    assert result is False
    
    # Test with metal (simplified)
    copper_mol = Chem.MolFromSmiles("[Cu]")
    if copper_mol:
        assert needs_uff_fallback(copper_mol) is True


def test_has_rhenium_complex():
    """Test rhenium complex detection with 3rj7 special handling."""
    # Test with None
    has_re, msg = has_rhenium_complex(None)
    assert not has_re
    assert msg == ""
    
    # Test with simple organic molecule
    benzene = Chem.MolFromSmiles("c1ccccc1")
    has_re, msg = has_rhenium_complex(benzene)
    assert not has_re
    assert msg == ""
    
    # Test with rhenium (general case)
    re_mol = Chem.MolFromSmiles("[Re]")
    if re_mol:
        has_re, msg = has_rhenium_complex(re_mol, "test_re")
        assert has_re
        assert "rhenium complex" in msg.lower()
    
    # Test 3rj7 special case
    if re_mol:
        has_re, msg = has_rhenium_complex(re_mol, "3rj7")
        assert not has_re  # Should allow 3rj7
        assert msg == ""
        
        # Test case insensitive
        has_re, msg = has_rhenium_complex(re_mol, "3RJ7")
        assert not has_re
        assert msg == ""


def test_is_large_peptide():
    """Test large peptide detection."""
    # Test with None
    is_peptide, msg = is_large_peptide(None)
    assert not is_peptide
    assert msg == ""
    
    # Test with simple molecule
    benzene = Chem.MolFromSmiles("c1ccccc1")
    is_peptide, msg = is_large_peptide(benzene)
    assert not is_peptide
    assert msg == ""
    
    # Test with small peptide (should pass)
    dipeptide = Chem.MolFromSmiles("CC(N)C(=O)NC(C)C(=O)O")  # Ala-Ala
    if dipeptide:
        is_peptide, msg = is_large_peptide(dipeptide, residue_threshold=8)
        assert not is_peptide
        assert msg == ""


def test_validate_target_molecule():
    """Test overall molecule validation."""
    # Test with None
    valid, msg = validate_target_molecule(None, "test_none")
    assert not valid
    assert "Invalid molecule object" in msg
    
    # Test with simple valid molecule
    benzene = Chem.MolFromSmiles("c1ccccc1")
    valid, msg = validate_target_molecule(benzene, "benzene")
    assert valid
    assert msg == ""
    
    # Test with rhenium (general case)
    re_mol = Chem.MolFromSmiles("[Re]")
    if re_mol:
        valid, msg = validate_target_molecule(re_mol, "re_test", "test_re")
        assert not valid
        assert "rhenium complex" in msg.lower()
    
    # Test 3rj7 special case
    if re_mol:
        valid, msg = validate_target_molecule(re_mol, "3rj7_test", "3rj7")
        assert valid  # Should allow 3rj7
        assert msg == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 