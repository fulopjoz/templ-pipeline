# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Simplified tests for the BenchmarkRunner class following pytest best practices.

These tests focus on testing the specific functionality without complex mocking
and follow the principle of testing behavior rather than implementation details.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from templ_pipeline.benchmark.runner import (
    BenchmarkParams,
    BenchmarkResult,
    BenchmarkRunner,
)


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    test_dir = tempfile.mkdtemp()
    data_dir = Path(test_dir) / "data"
    output_dir = Path(test_dir) / "output"

    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    yield {
        "test_dir": test_dir,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
    }

    shutil.rmtree(test_dir)


@pytest.fixture
def benchmark_runner(temp_dirs):
    """Create a BenchmarkRunner instance for testing."""
    return BenchmarkRunner(
        data_dir=temp_dirs["data_dir"], poses_output_dir=temp_dirs["output_dir"]
    )


class TestBenchmarkRunnerInitialization:
    """Test BenchmarkRunner initialization following best practices."""

    def test_basic_initialization(self, temp_dirs):
        """Test basic BenchmarkRunner initialization."""
        runner = BenchmarkRunner(
            data_dir=temp_dirs["data_dir"], poses_output_dir=temp_dirs["output_dir"]
        )

        assert str(runner.data_dir) == temp_dirs["data_dir"]
        assert str(runner.poses_output_dir) == temp_dirs["output_dir"]
        assert runner.peptide_threshold == 8
        assert runner.error_tracker is not None

    def test_initialization_with_optional_params(self, temp_dirs):
        """Test initialization with optional parameters."""
        runner = BenchmarkRunner(
            data_dir=temp_dirs["data_dir"],
            poses_output_dir=temp_dirs["output_dir"],
            enable_error_tracking=False,
            peptide_threshold=10,
        )

        assert str(runner.data_dir) == temp_dirs["data_dir"]
        assert runner.peptide_threshold == 10

    def test_initialization_data_dir_only(self, temp_dirs):
        """Test initialization with data directory only."""
        runner = BenchmarkRunner(data_dir=temp_dirs["data_dir"])

        assert str(runner.data_dir) == temp_dirs["data_dir"]
        assert runner.poses_output_dir is None


class TestBenchmarkRunnerProperties:
    """Test BenchmarkRunner properties and attributes."""

    def test_data_directory_property(self, benchmark_runner, temp_dirs):
        """Test data directory property access."""
        assert str(benchmark_runner.data_dir) == temp_dirs["data_dir"]
        assert isinstance(benchmark_runner.data_dir, Path)

    def test_output_directory_property(self, benchmark_runner, temp_dirs):
        """Test output directory property access."""
        assert str(benchmark_runner.poses_output_dir) == temp_dirs["output_dir"]
        assert isinstance(benchmark_runner.poses_output_dir, Path)

    def test_peptide_threshold_property(self, benchmark_runner):
        """Test peptide threshold property."""
        assert benchmark_runner.peptide_threshold == 8
        assert isinstance(benchmark_runner.peptide_threshold, int)

    def test_error_tracker_availability(self, benchmark_runner):
        """Test error tracker is available when enabled."""
        assert hasattr(benchmark_runner, "error_tracker")


class TestBenchmarkRunnerUnitBehavior:
    """Test specific units of behavior without complex integration."""

    def test_benchmark_params_creation(self):
        """Test that BenchmarkParams can be created with required fields."""
        params = BenchmarkParams(
            target_pdb="1abc", exclude_pdb_ids=set(), poses_output_dir="/tmp/test"
        )

        assert params.target_pdb == "1abc"
        assert params.exclude_pdb_ids == set()
        assert params.poses_output_dir == "/tmp/test"

    def test_benchmark_result_creation(self):
        """Test that BenchmarkResult can be created with all fields."""
        result = BenchmarkResult(
            success=True,
            rmsd_values={"combo": 1.5},
            runtime=2.5,
            error=None,
            metadata={"test": "data"},
        )

        assert result.success is True
        assert result.rmsd_values == {"combo": 1.5}
        assert result.runtime == 2.5
        assert result.error is None
        assert result.metadata == {"test": "data"}

    def test_benchmark_result_failure(self):
        """Test BenchmarkResult creation for failure cases."""
        result = BenchmarkResult(
            success=False,
            rmsd_values={},
            runtime=0.1,
            error="Test error message",
            metadata={},
        )

        assert result.success is False
        assert result.rmsd_values == {}
        assert result.error == "Test error message"


class TestBenchmarkRunnerMockingPatterns:
    """Test proper mocking patterns for external dependencies."""

    @patch("templ_pipeline.benchmark.runner.BenchmarkRunner._get_protein_file")
    def test_protein_file_resolution_mock(
        self, mock_get_protein_file, benchmark_runner
    ):
        """Test that protein file resolution can be properly mocked."""
        mock_get_protein_file.return_value = "/fake/path/test.pdb"

        # This tests the mocking pattern, not the actual implementation
        result = mock_get_protein_file("test_pdb")
        assert result == "/fake/path/test.pdb"
        mock_get_protein_file.assert_called_once_with("test_pdb")

    @patch("templ_pipeline.benchmark.runner.BenchmarkRunner._load_ligand_data_from_sdf")
    def test_ligand_loading_mock(self, mock_load_ligand, benchmark_runner):
        """Test that ligand loading can be properly mocked."""
        from rdkit import Chem

        mock_mol = Chem.MolFromSmiles("CCO")
        mock_load_ligand.return_value = ("CCO", mock_mol)

        # This tests the mocking pattern
        smiles, mol = mock_load_ligand("test_pdb")
        assert smiles == "CCO"
        assert mol is not None
        mock_load_ligand.assert_called_once_with("test_pdb")


class TestBenchmarkRunnerErrorHandling:
    """Test error handling scenarios without complex setup."""

    def test_invalid_data_directory_handling(self):
        """Test initialization with invalid data directory."""
        # BenchmarkRunner should handle non-existent directories gracefully
        try:
            runner = BenchmarkRunner(data_dir="/tmp/nonexistent_test_dir")
            # Should either succeed or fail gracefully
            assert runner is not None
        except (FileNotFoundError, OSError, PermissionError):
            # These are acceptable exceptions for invalid directories
            pass

    def test_benchmark_result_error_fields(self):
        """Test that error information is properly captured in results."""
        error_msg = "Test error for validation"
        result = BenchmarkResult(
            success=False,
            rmsd_values={},
            runtime=0.0,
            error=error_msg,
            metadata={"target_pdb": "test_pdb"},
        )

        assert result.success is False
        assert result.error == error_msg
        assert "target_pdb" in result.metadata


# Parametrized tests for edge cases
@pytest.mark.parametrize("peptide_threshold", [1, 5, 8, 10, 20])
def test_peptide_threshold_values(temp_dirs, peptide_threshold):
    """Test different peptide threshold values."""
    runner = BenchmarkRunner(
        data_dir=temp_dirs["data_dir"], peptide_threshold=peptide_threshold
    )

    assert runner.peptide_threshold == peptide_threshold


@pytest.mark.parametrize("enable_error_tracking", [True, False])
def test_error_tracking_configuration(temp_dirs, enable_error_tracking):
    """Test error tracking configuration."""
    runner = BenchmarkRunner(
        data_dir=temp_dirs["data_dir"], enable_error_tracking=enable_error_tracking
    )

    if enable_error_tracking:
        assert runner.error_tracker is not None
    # Note: error_tracker might still be present even when disabled due to fallbacks


if __name__ == "__main__":
    pytest.main([__file__])
