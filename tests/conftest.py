import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock
import streamlit as st
from rdkit import Chem

@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing"""
    return [
        "CCO",  # ethanol
        "c1ccccc1",  # benzene
        "CC(=O)O",  # acetic acid
    ]

@pytest.fixture
def sample_mol():
    """Sample RDKit molecule"""
    return Chem.MolFromSmiles("CCO")

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit session state for UI tests"""
    if 'session_state' not in st.__dict__:
        st.session_state = {}
    yield st.session_state
    st.session_state.clear()

@pytest.fixture
def output_dir(temp_dir):
    """Output directory for test results"""
    output = temp_dir / "output"
    output.mkdir(exist_ok=True)
    return output

@pytest.fixture
def test_data_dir():
    """Test data directory"""
    return Path(__file__).parent.parent / "data" 