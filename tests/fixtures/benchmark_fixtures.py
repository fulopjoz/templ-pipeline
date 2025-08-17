# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Benchmark-specific test fixtures.

This module provides fixtures and test data specifically for
TEMPL benchmark testing scenarios.
"""

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest

from .data_factory import TestDataFactory


@pytest.fixture(scope="session")
def benchmark_target_data():
    """
    Session-wide fixture for benchmark target data.

    Returns:
        List of benchmark target dictionaries
    """
    return TestDataFactory.create_benchmark_target_data()


@pytest.fixture(scope="function")
def mock_benchmark_config():
    """
    Function-scoped benchmark configuration for testing.

    Returns:
        Dict containing test benchmark configuration
    """
    return {
        "name": "test_benchmark",
        "description": "Test benchmark for unit testing",
        "num_workers": 2,
        "timeout": 60,
        "output_format": "json",
        "batch_size": 10,
        "max_retries": 3,
    }


@pytest.fixture(scope="function")
def benchmark_test_workspace(tmp_path):
    """
    Function-scoped benchmark workspace setup.

    Args:
        tmp_path: pytest temporary path fixture

    Returns:
        Dict containing benchmark workspace structure
    """
    # Create workspace structure
    workspace = {
        "root": tmp_path,
        "output": tmp_path / "benchmark_output",
        "logs": tmp_path / "logs",
        "results": tmp_path / "results",
        "temp": tmp_path / "temp",
    }

    # Create directories
    for dir_path in workspace.values():
        if isinstance(dir_path, Path):
            dir_path.mkdir(exist_ok=True)

    # Create test target files
    target_data = TestDataFactory.create_benchmark_target_data()
    target_files = {}

    for target in target_data:
        pdb_id = target["pdb_id"]

        # Create protein file
        protein_file = workspace["temp"] / target["protein_file"]
        protein_data = TestDataFactory.create_protein_data("minimal")
        protein_file.write_text(protein_data["content"])
        target_files[f"{pdb_id}_protein"] = protein_file

        # Update target with actual file paths
        target["protein_file"] = str(protein_file)

    workspace["target_data"] = target_data
    workspace["target_files"] = target_files

    return workspace


@pytest.fixture(scope="function")
def mock_benchmark_results():
    """
    Function-scoped fixture for mock benchmark results.

    Returns:
        Function that creates benchmark results based on parameters
    """

    def _create_results(
        num_targets: int = 3, success_rate: float = 0.8, include_errors: bool = True
    ) -> List[Dict[str, Any]]:

        results = []
        target_data = TestDataFactory.create_benchmark_target_data()

        for i in range(num_targets):
            target = target_data[i % len(target_data)]

            # Determine if this target should succeed
            success = (i / num_targets) < success_rate

            if success:
                result = {
                    "success": True,
                    "pdb_id": target["pdb_id"],
                    "runtime": 30.0 + (i * 5),
                    "poses": {"combo": (Mock(), {"combo_score": 0.8 - (i * 0.1)})},
                    "mcs_info": {"smarts": "CCO"},
                    "templates": [("template1", 0.9 - (i * 0.05))],
                    "rmsd_values": {"combo": {"rmsd": 1.2 + (i * 0.2)}},
                }
            else:
                error_types = [
                    "Pipeline failed",
                    "Invalid SMILES",
                    "Protein file not found",
                    "Timeout error",
                    "Memory error",
                ]

                result = {
                    "success": False,
                    "pdb_id": target["pdb_id"],
                    "runtime": 5.0 + (i * 2),
                    "error": error_types[i % len(error_types)],
                }

            results.append(result)

        return results

    return _create_results


@pytest.fixture(scope="function")
def benchmark_summary_data():
    """
    Function-scoped fixture for benchmark summary data.

    Returns:
        Dict containing benchmark summary information
    """
    return {
        "summary": {
            "total_targets": 100,
            "successful_runs": 85,
            "failed_runs": 15,
            "average_runtime": 45.2,
            "total_runtime": 4520.0,
            "success_rate": 0.85,
            "average_rmsd": 1.8,
            "best_rmsd": 0.5,
            "worst_rmsd": 4.2,
        },
        "detailed_results": TestDataFactory.create_benchmark_target_data(),
        "error_analysis": {
            "Pipeline failed": 8,
            "Invalid SMILES": 3,
            "Timeout error": 2,
            "Memory error": 2,
        },
        "performance_metrics": {
            "targets_per_minute": 2.2,
            "memory_usage_mb": 512,
            "cpu_usage_percent": 75.5,
        },
    }


@pytest.fixture(scope="function")
def mock_benchmark_pipeline():
    """
    Function-scoped mock benchmark pipeline.

    Returns:
        Mock pipeline configured for benchmark testing
    """
    mock_pipeline = Mock()

    # Configure successful pipeline execution
    mock_pipeline.run_full_pipeline.return_value = (
        TestDataFactory.create_mock_pipeline_results(True)
    )

    # Configure embedding generation
    embeddings = TestDataFactory.create_embedding_data(size=1280, num_proteins=1)
    mock_pipeline.generate_embedding.return_value = (embeddings[0], "A")

    # Configure template finding
    mock_pipeline.find_templates.return_value = [("template1", 0.9)]

    return mock_pipeline


@pytest.fixture(scope="function")
def benchmark_error_scenarios():
    """
    Function-scoped fixture for benchmark error scenarios.

    Returns:
        Dict containing various benchmark error cases
    """
    error_data = TestDataFactory.create_error_test_data()

    # Add benchmark-specific error scenarios
    error_data["benchmark_errors"] = {
        "timeout_scenarios": [
            {"timeout": 1, "expected_error": "TimeoutError"},
            {"timeout": 0, "expected_error": "ValueError"},
        ],
        "resource_exhaustion": [
            {"num_workers": -1, "expected_error": "ValueError"},
            {"num_workers": 0, "expected_error": "ValueError"},
            {"batch_size": 0, "expected_error": "ValueError"},
        ],
        "data_corruption": [
            {"missing_protein_file": True},
            {"invalid_target_format": True},
            {"corrupted_config": True},
        ],
    }

    return error_data


def create_benchmark_test_data(
    num_targets: int = 10, include_invalid: bool = True
) -> List[Dict[str, Any]]:
    """
    Helper function to create benchmark test data.

    Args:
        num_targets: Number of test targets to create
        include_invalid: Whether to include invalid test cases

    Returns:
        List of benchmark target dictionaries
    """
    base_targets = TestDataFactory.create_benchmark_target_data()
    targets = []

    for i in range(num_targets):
        base_target = base_targets[i % len(base_targets)]
        target = base_target.copy()
        target["pdb_id"] = f"test{i+1:03d}"
        target["protein_file"] = f"test{i+1:03d}_protein.pdb"
        targets.append(target)

    if include_invalid:
        # Add some invalid targets for error testing
        invalid_targets = [
            {
                "pdb_id": "invalid001",
                "protein_file": "nonexistent.pdb",
                "ligand_smiles": "CCO",
            },
            {
                "pdb_id": "invalid002",
                "protein_file": "test001_protein.pdb",
                "ligand_smiles": "INVALID_SMILES",
            },
        ]
        targets.extend(invalid_targets)

    return targets


def create_mock_benchmark_execution(
    success_rate: float = 0.8, runtime_range: tuple = (10, 60)
) -> MagicMock:
    """
    Helper function to create mock benchmark execution.

    Args:
        success_rate: Fraction of targets that should succeed
        runtime_range: (min, max) runtime in seconds

    Returns:
        MagicMock configured for benchmark testing
    """
    import random

    random.seed(42)  # For reproducible testing

    mock_benchmark = MagicMock()

    def mock_run_single_target(target_data):
        success = random.random() < success_rate
        runtime = random.uniform(*runtime_range)

        if success:
            return {
                "success": True,
                "pdb_id": target_data["pdb_id"],
                "runtime": runtime,
                "poses": {"combo": (Mock(), {"combo_score": random.uniform(0.5, 0.9)})},
                "rmsd_values": {"combo": {"rmsd": random.uniform(0.5, 3.0)}},
            }
        else:
            return {
                "success": False,
                "pdb_id": target_data["pdb_id"],
                "runtime": runtime * 0.3,  # Failed runs are faster
                "error": "Mock benchmark failure",
            }

    mock_benchmark.run_single_target.side_effect = mock_run_single_target

    # Reset random seed
    random.seed()

    return mock_benchmark
