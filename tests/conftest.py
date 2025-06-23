import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock
from rdkit import Chem

# Only import streamlit for UI tests - make it conditional
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    st = None
    STREAMLIT_AVAILABLE = False

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
    if not STREAMLIT_AVAILABLE:
        # Create a mock session state when streamlit is not available
        mock_state = {}
        yield mock_state
        mock_state.clear()
        return
    
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

def pytest_configure(config):
    """Configure pytest with custom markers and skip logic"""
    config.addinivalue_line("markers", "ui: mark test as requiring UI dependencies")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "fast: mark test as fast running")

def pytest_collection_modifyitems(config, items):
    """Skip UI tests if streamlit is not available"""
    if STREAMLIT_AVAILABLE:
        return
    
    skip_ui = pytest.mark.skip(reason="streamlit not available")
    for item in items:
        if "ui" in item.keywords:
            item.add_marker(skip_ui) 