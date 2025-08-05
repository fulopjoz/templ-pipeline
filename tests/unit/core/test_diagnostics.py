"""
Tests for the diagnostics module.
"""

import pytest
import tempfile
import json
from pathlib import Path
from templ_pipeline.core.diagnostics import (
    PipelineErrorTracker,
    ProteinAlignmentTracker,
)


class TestPipelineErrorTracker:
    """Test PipelineErrorTracker functionality."""

    def setup_method(self):
        """Clear errors before each test."""
        PipelineErrorTracker.clear_errors()

    def test_context_manager_no_error(self):
        """Test context manager when no error occurs."""
        with PipelineErrorTracker("test_pdb", "test_stage"):
            pass  # No error

        summary = PipelineErrorTracker.get_error_summary()
        assert summary["total_errors"] == 0
        assert summary["failed_pdbs"] == 0

    def test_context_manager_with_error(self):
        """Test context manager when error occurs."""
        with pytest.raises(ValueError):
            with PipelineErrorTracker("test_pdb", "test_stage"):
                raise ValueError("Test error")

        summary = PipelineErrorTracker.get_error_summary()
        assert summary["total_errors"] == 1
        assert summary["failed_pdbs"] == 1
        assert "test_pdb" in summary["detailed_errors"]

        error = summary["detailed_errors"]["test_pdb"][0]
        assert error["stage"] == "test_stage"
        assert error["error_type"] == "ValueError"
        assert error["error_message"] == "Test error"
        assert "timestamp" in error
        assert "duration" in error

    def test_multiple_errors_same_pdb(self):
        """Test multiple errors for the same PDB."""
        with pytest.raises(ValueError):
            with PipelineErrorTracker("test_pdb", "stage1"):
                raise ValueError("Error 1")

        with pytest.raises(RuntimeError):
            with PipelineErrorTracker("test_pdb", "stage2"):
                raise RuntimeError("Error 2")

        summary = PipelineErrorTracker.get_error_summary()
        assert summary["total_errors"] == 2
        assert summary["failed_pdbs"] == 1
        assert len(summary["detailed_errors"]["test_pdb"]) == 2

    def test_multiple_pdbs_with_errors(self):
        """Test errors across multiple PDBs."""
        with pytest.raises(ValueError):
            with PipelineErrorTracker("pdb1", "stage1"):
                raise ValueError("Error 1")

        with pytest.raises(RuntimeError):
            with PipelineErrorTracker("pdb2", "stage2"):
                raise RuntimeError("Error 2")

        summary = PipelineErrorTracker.get_error_summary()
        assert summary["total_errors"] == 2
        assert summary["failed_pdbs"] == 2
        assert "pdb1" in summary["detailed_errors"]
        assert "pdb2" in summary["detailed_errors"]

    def test_error_summary_aggregation(self):
        """Test error summary aggregation by stage and type."""
        with pytest.raises(ValueError):
            with PipelineErrorTracker("pdb1", "embedding"):
                raise ValueError("Error 1")

        with pytest.raises(ValueError):
            with PipelineErrorTracker("pdb2", "embedding"):
                raise ValueError("Error 2")

        with pytest.raises(RuntimeError):
            with PipelineErrorTracker("pdb3", "mcs"):
                raise RuntimeError("Error 3")

        summary = PipelineErrorTracker.get_error_summary()
        assert summary["errors_by_stage"]["embedding"] == 2
        assert summary["errors_by_stage"]["mcs"] == 1
        assert summary["errors_by_type"]["ValueError"] == 2
        assert summary["errors_by_type"]["RuntimeError"] == 1

    def test_save_error_report(self):
        """Test saving error report to JSON file."""
        with pytest.raises(ValueError):
            with PipelineErrorTracker("test_pdb", "test_stage"):
                raise ValueError("Test error")

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = PipelineErrorTracker.save_error_report(temp_dir, "target_pdb")

            assert report_path is not None
            assert Path(report_path).exists()

            with open(report_path, "r") as f:
                report_data = json.load(f)

            assert report_data["total_errors"] == 1
            assert report_data["failed_pdbs"] == 1

    def test_save_error_report_no_errors(self):
        """Test saving error report when no errors exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = PipelineErrorTracker.save_error_report(temp_dir, "target_pdb")
            assert report_path is None

    def test_clear_errors(self):
        """Test clearing all errors."""
        with pytest.raises(ValueError):
            with PipelineErrorTracker("test_pdb", "test_stage"):
                raise ValueError("Test error")

        assert PipelineErrorTracker.get_error_summary()["total_errors"] == 1

        PipelineErrorTracker.clear_errors()
        assert PipelineErrorTracker.get_error_summary()["total_errors"] == 0


class TestProteinAlignmentTracker:
    """Test ProteinAlignmentTracker functionality."""

    def setup_method(self):
        """Clear logs before each test."""
        ProteinAlignmentTracker.clear_logs()

    def test_track_successful_alignment(self):
        """Test tracking successful alignment."""
        details = {"rmsd": 1.5, "atoms_aligned": 100}
        ProteinAlignmentTracker.track_alignment_attempt(
            "pdb1", "initial", True, details
        )

        summary = ProteinAlignmentTracker.get_alignment_summary()
        assert summary["total_attempts"] == 1
        assert summary["successful"] == 1
        assert summary["failed"] == 0
        assert summary["success_rate"] == 1.0
        assert summary["success_by_stage"]["initial"] == 1

    def test_track_failed_alignment(self):
        """Test tracking failed alignment."""
        details = {"error": "No common atoms found"}
        ProteinAlignmentTracker.track_alignment_attempt(
            "pdb1", "initial", False, details
        )

        summary = ProteinAlignmentTracker.get_alignment_summary()
        assert summary["total_attempts"] == 1
        assert summary["successful"] == 0
        assert summary["failed"] == 1
        assert summary["success_rate"] == 0.0
        assert summary["failure_by_stage"]["initial"] == 1

    def test_multiple_alignment_attempts(self):
        """Test tracking multiple alignment attempts."""
        ProteinAlignmentTracker.track_alignment_attempt("pdb1", "initial", True, {})
        ProteinAlignmentTracker.track_alignment_attempt("pdb2", "initial", False, {})
        ProteinAlignmentTracker.track_alignment_attempt("pdb3", "refinement", True, {})

        summary = ProteinAlignmentTracker.get_alignment_summary()
        assert summary["total_attempts"] == 3
        assert summary["successful"] == 2
        assert summary["failed"] == 1
        assert summary["success_rate"] == 2 / 3
        assert summary["success_by_stage"]["initial"] == 1
        assert summary["success_by_stage"]["refinement"] == 1
        assert summary["failure_by_stage"]["initial"] == 1

    def test_save_alignment_report(self):
        """Test saving alignment report to JSON file."""
        ProteinAlignmentTracker.track_alignment_attempt("pdb1", "initial", True, {})

        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = ProteinAlignmentTracker.save_alignment_report(
                temp_dir, "target_pdb"
            )

            assert report_path is not None
            assert Path(report_path).exists()

            with open(report_path, "r") as f:
                report_data = json.load(f)

            assert report_data["total_attempts"] == 1
            assert report_data["successful"] == 1

    def test_save_alignment_report_no_logs(self):
        """Test saving alignment report when no logs exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = ProteinAlignmentTracker.save_alignment_report(
                temp_dir, "target_pdb"
            )
            assert report_path is None

    def test_clear_logs(self):
        """Test clearing all alignment logs."""
        ProteinAlignmentTracker.track_alignment_attempt("pdb1", "initial", True, {})
        assert ProteinAlignmentTracker.get_alignment_summary()["total_attempts"] == 1

        ProteinAlignmentTracker.clear_logs()
        assert ProteinAlignmentTracker.get_alignment_summary()["total_attempts"] == 0

    def test_detailed_logs_structure(self):
        """Test structure of detailed logs."""
        details = {"rmsd": 1.5, "method": "kabsch"}
        ProteinAlignmentTracker.track_alignment_attempt(
            "pdb1", "initial", True, details
        )

        summary = ProteinAlignmentTracker.get_alignment_summary()
        log_entry = summary["detailed_logs"][0]

        assert log_entry["pdb_id"] == "pdb1"
        assert log_entry["stage"] == "initial"
        assert log_entry["success"] is True
        assert log_entry["details"] == details
        assert "timestamp" in log_entry
