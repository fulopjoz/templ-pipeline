# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Pipeline-specific test fixtures.

This module provides fixtures and test data specifically for
TEMPL pipeline testing scenarios.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from .data_factory import TestDataFactory


@pytest.fixture(scope="session")
def standard_test_molecules():
    """
    Session-wide fixture for standard test molecules.

    Returns:
        Dict mapping molecule types to molecule data
    """
    molecules = {}
    for mol_type in TestDataFactory.STANDARD_MOLECULES.keys():
        if mol_type != "invalid":  # Skip invalid for session fixture
            molecules[mol_type] = TestDataFactory.create_molecule_data(mol_type)
    return molecules


@pytest.fixture(scope="session")
def standard_test_proteins():
    """
    Session-wide fixture for standard test proteins.

    Returns:
        Dict mapping protein types to protein data
    """
    proteins = {}
    for protein_type in TestDataFactory.STANDARD_PROTEINS.keys():
        proteins[protein_type] = TestDataFactory.create_protein_data(protein_type)
    return proteins


@pytest.fixture(scope="session")
def mock_embeddings():
    """
    Session-wide fixture for mock embedding data.

    Returns:
        Numpy array of mock embeddings
    """
    return TestDataFactory.create_embedding_data(size=1280, num_proteins=10)


@pytest.fixture(scope="session")
def mcs_test_pairs():
    """
    Session-wide fixture for MCS test molecule pairs.

    Returns:
        List of (mol1_smiles, mol2_smiles, description) tuples
    """
    return TestDataFactory.create_mcs_test_pairs()


@pytest.fixture(scope="function")
def mock_pipeline_results():
    """
    Function-scoped fixture for mock pipeline results.

    Returns:
        Function that creates mock results based on success parameter
    """

    def _create_results(success: bool = True) -> Dict[str, Any]:
        return TestDataFactory.create_mock_pipeline_results(success)

    return _create_results


@pytest.fixture(scope="function")
def mock_embedding_manager():
    """
    Function-scoped mock embedding manager.

    Returns:
        Mock EmbeddingManager with realistic behavior
    """
    mock_manager = Mock()

    # Configure standard embeddings
    embeddings = TestDataFactory.create_embedding_data()
    mock_manager.generate_protein_embedding.return_value = embeddings[0]

    # Configure template finding
    mock_manager.find_templates.return_value = [
        ("template1", 0.9),
        ("template2", 0.8),
        ("template3", 0.7),
    ]

    # Configure database access
    mock_manager.embedding_db = {
        f"test{i:03d}": embeddings[i] for i in range(len(embeddings))
    }

    return mock_manager


@pytest.fixture(scope="function")
def test_pipeline_config():
    """
    Function-scoped pipeline configuration for testing.

    Returns:
        Dict containing test pipeline configuration
    """
    return {
        "embedding_path": None,  # Will use mock
        "output_dir": "output",
        "run_id": "test_run",
        "auto_cleanup": True,
        "num_workers": 2,
        "timeout": 60,
    }


@pytest.fixture(scope="function")
def pipeline_test_data(tmp_path):
    """
    Function-scoped fixture creating complete pipeline test environment.

    Args:
        tmp_path: pytest temporary path fixture

    Returns:
        Dict containing all necessary test files and data
    """
    # Create test files
    test_files = TestDataFactory.create_test_files(tmp_path)

    # Create molecules
    molecules = {
        "simple": TestDataFactory.create_molecule_data("simple_alkane"),
        "complex": TestDataFactory.create_molecule_data("complex_drug"),
    }

    # Create proteins
    proteins = {
        "minimal": TestDataFactory.create_protein_data("minimal"),
        "multi_chain": TestDataFactory.create_protein_data("multi_chain"),
    }

    return {
        "files": test_files,
        "molecules": molecules,
        "proteins": proteins,
        "temp_dir": tmp_path,
        "embeddings": TestDataFactory.create_embedding_data(size=1280, num_proteins=5),
    }


@pytest.fixture(scope="function")
def error_test_scenarios():
    """
    Function-scoped fixture for error testing scenarios.

    Returns:
        Dict containing various error test cases
    """
    return TestDataFactory.create_error_test_data()


def create_mock_rdkit_molecules(smiles_list: List[str]) -> List[Any]:
    """
    Helper function to create mock RDKit molecules from SMILES.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of mock RDKit molecule objects
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecules = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Add 3D coordinates for testing
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
            molecules.append(mol)
        else:
            molecules.append(None)

    return molecules


def create_mock_pipeline_execution(success: bool = True) -> MagicMock:
    """
    Helper function to create mock pipeline execution.

    Args:
        success: Whether the execution should be successful

    Returns:
        MagicMock configured for pipeline testing
    """
    mock_pipeline = MagicMock()

    if success:
        mock_pipeline.run_full_pipeline.return_value = (
            TestDataFactory.create_mock_pipeline_results(True)
        )
        mock_pipeline.generate_embedding.return_value = (
            TestDataFactory.create_embedding_data(size=1280, num_proteins=1)[0],
            "A",
        )
        mock_pipeline.find_templates.return_value = [("template1", 0.9)]
        mock_pipeline.generate_poses.return_value = {
            "combo": (Mock(), {"combo_score": 0.8})
        }
    else:
        mock_pipeline.run_full_pipeline.side_effect = RuntimeError(
            "Mock pipeline failure"
        )
        mock_pipeline.generate_embedding.side_effect = ValueError(
            "Mock embedding failure"
        )

    return mock_pipeline
