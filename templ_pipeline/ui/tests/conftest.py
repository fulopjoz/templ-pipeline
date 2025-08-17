# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Pytest configuration and fixtures for TEMPL Pipeline UI tests.
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from rdkit import Chem


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_pdb_file(temp_dir):
    """Create a test PDB file."""
    pdb_content = """HEADER    TEST PROTEIN
ATOM      1  N   ALA A   1      20.154  16.967  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      19.030  16.067  10.000  1.00 20.00           C
ATOM      3  C   ALA A   1      17.710  16.820  10.000  1.00 20.00           C
ATOM      4  O   ALA A   1      17.710  18.050  10.000  1.00 20.00           O
ATOM      5  CB  ALA A   1      19.030  15.300   8.700  1.00 20.00           C
ATOM      6  N   GLY A   2      16.590  16.100  10.000  1.00 20.00           N
ATOM      7  CA  GLY A   2      15.270  16.853  10.000  1.00 20.00           C
ATOM      8  C   GLY A   2      13.950  16.100  10.000  1.00 20.00           C
ATOM      9  O   GLY A   2      13.950  14.870  10.000  1.00 20.00           O
END
"""

    pdb_file = os.path.join(temp_dir, "test_protein.pdb")
    with open(pdb_file, "w") as f:
        f.write(pdb_content)

    return pdb_file


@pytest.fixture
def test_sdf_file(temp_dir):
    """Create a test SDF file."""
    sdf_content = """Test Ligand
  -I-interpret- 

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
M  END
$$$$
"""

    sdf_file = os.path.join(temp_dir, "test_ligand.sdf")
    with open(sdf_file, "w") as f:
        f.write(sdf_content)

    return sdf_file


@pytest.fixture
def test_molecule():
    """Create a test RDKit molecule."""
    mol = Chem.MolFromSmiles("CCO")
    mol.SetProp("_Name", "ethanol")
    return mol


@pytest.fixture
def mock_pipeline_results():
    """Create mock pipeline results."""
    return {
        "poses": {
            "shape": (
                Chem.MolFromSmiles("CCO"),
                {"shape": 0.8, "color": 0.6, "combo": 0.7},
            ),
            "color": (
                Chem.MolFromSmiles("CCO"),
                {"shape": 0.7, "color": 0.9, "combo": 0.8},
            ),
            "combo": (
                Chem.MolFromSmiles("CCO"),
                {"shape": 0.75, "color": 0.75, "combo": 0.75},
            ),
        },
        "mcs_info": {
            "mcs_smiles": "CCO",
            "mcs_size": 3,
            "query_match": [0, 1, 2],
            "template_match": [0, 1, 2],
        },
        "templates": [("1ABC", 0.95), ("2DEF", 0.88), ("3GHI", 0.82)],
        "embedding": np.random.rand(1280),
        "output_file": "/tmp/test_poses.sdf",
    }


@pytest.fixture
def mock_pipeline_service():
    """Create a mock pipeline service."""
    mock_service = Mock()
    mock_service.run_pipeline.return_value = {
        "poses": {"shape": (Chem.MolFromSmiles("CCO"), {"shape": 0.8, "color": 0.6})},
        "mcs_info": {"mcs_smiles": "CCO"},
        "templates": [("1ABC", 0.95)],
        "embedding": np.random.rand(1280),
        "output_file": "/tmp/test_output.sdf",
    }
    return mock_service


@pytest.fixture
def streamlit_app_env():
    """Setup environment for Streamlit app testing."""
    # Mock Streamlit's session state
    with patch("streamlit.session_state") as mock_session_state:
        mock_session_state.configure_mock(
            **{
                "pipeline_results": None,
                "current_step": "input",
                "error_message": None,
                "processing": False,
                "uploaded_files": [],
            }
        )
        yield mock_session_state


@pytest.fixture
def mock_hardware_manager():
    """Create a mock hardware manager."""
    mock_manager = Mock()
    mock_manager.get_optimal_workers.return_value = 4
    mock_manager.has_gpu.return_value = True
    mock_manager.get_memory_info.return_value = {
        "total": 16 * 1024 * 1024 * 1024,  # 16GB
        "available": 8 * 1024 * 1024 * 1024,  # 8GB
        "percent": 50.0,
    }
    return mock_manager


@pytest.fixture
def mock_embedding_manager():
    """Create a mock embedding manager."""
    mock_manager = Mock()
    mock_manager.load_database.return_value = True
    mock_manager.generate_embedding.return_value = np.random.rand(1280)
    mock_manager.find_similar_templates.return_value = [
        ("1ABC", 0.95),
        ("2DEF", 0.88),
        ("3GHI", 0.82),
    ]
    return mock_manager


@pytest.fixture(scope="session")
def playwright_browser():
    """Create a Playwright browser instance for the session."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            yield browser
            browser.close()
    except ImportError:
        pytest.skip("Playwright not available")


@pytest.fixture
def browser_page(playwright_browser):
    """Create a browser page for testing."""
    context = playwright_browser.new_context()
    page = context.new_page()
    yield page
    context.close()


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "ui: marks tests as UI tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add markers based on test file names
        if "e2e" in item.fspath.basename:
            item.add_marker(pytest.mark.e2e)
        if "integration" in item.fspath.basename:
            item.add_marker(pytest.mark.integration)
        if "ui" in item.fspath.basename or "streamlit" in item.fspath.basename:
            item.add_marker(pytest.mark.ui)


# Skip tests if dependencies not available
def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    # Skip E2E tests if Playwright not available
    if item.get_closest_marker("e2e"):
        try:
            import playwright
        except ImportError:
            pytest.skip("Playwright not available for E2E tests")

    # Skip UI tests if Streamlit not available
    if item.get_closest_marker("ui"):
        try:
            import streamlit
        except ImportError:
            pytest.skip("Streamlit not available for UI tests")
