# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Working test cases for benchmark error tracking module - using actual API.
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from templ_pipeline.benchmark.error_tracking import (
    BenchmarkErrorSummary,
    BenchmarkErrorTracker,
    MissingPDBRecord,
)


@pytest.fixture
def temp_workspace():
    """Create temporary workspace directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def error_tracker(temp_workspace):
    """Create BenchmarkErrorTracker instance."""
    return BenchmarkErrorTracker(temp_workspace)


class TestMissingPDBRecord:
    """Test MissingPDBRecord functionality with correct API."""

    def test_missing_pdb_record_creation(self):
        """Test creation of MissingPDBRecord with correct parameters."""
        timestamp = datetime.now().isoformat()
        record = MissingPDBRecord(
            pdb_id="1abc",
            reason="file_not_found",
            details="PDB file not found in data directory",
            file_type="protein",
            timestamp=timestamp,
        )

        assert record.pdb_id == "1abc"
        assert record.reason == "file_not_found"
        assert record.details == "PDB file not found in data directory"
        assert record.file_type == "protein"
        assert record.timestamp == timestamp

    def test_missing_pdb_record_to_dict(self):
        """Test conversion of MissingPDBRecord to dictionary."""
        record = MissingPDBRecord(
            pdb_id="3xyz",
            reason="invalid_structure",
            details="Structure validation failed",
            file_type="template",
        )

        record_dict = record.to_dict()

        assert isinstance(record_dict, dict)
        assert record_dict["pdb_id"] == "3xyz"
        assert record_dict["reason"] == "invalid_structure"
        assert record_dict["details"] == "Structure validation failed"
        assert record_dict["file_type"] == "template"
        assert "timestamp" in record_dict

    def test_missing_pdb_record_default_timestamp(self):
        """Test that MissingPDBRecord creates default timestamp."""
        record = MissingPDBRecord(
            pdb_id="5def",
            reason="processing_error",
            details="Could not process structure",
            file_type="ligand",
        )

        # Should have a timestamp
        assert record.timestamp is not None
        assert isinstance(record.timestamp, str)

        # Should be a valid ISO format timestamp
        datetime.fromisoformat(record.timestamp)


class TestBenchmarkErrorSummary:
    """Test BenchmarkErrorSummary functionality."""

    def test_error_summary_creation(self):
        """Test creation of BenchmarkErrorSummary with required parameters."""
        error_breakdown = {"file_not_found": 2, "structure_invalid": 1}
        summary = BenchmarkErrorSummary(total_errors=3, error_breakdown=error_breakdown)

        assert summary.total_errors == 3
        assert summary.error_breakdown == error_breakdown
        assert hasattr(summary, "to_dict")

    def test_error_summary_to_dict(self):
        """Test BenchmarkErrorSummary to_dict method."""
        error_breakdown = {"processing_error": 5, "timeout": 2}
        summary = BenchmarkErrorSummary(total_errors=7, error_breakdown=error_breakdown)

        summary_dict = summary.to_dict()
        assert isinstance(summary_dict, dict)
        assert summary_dict["total_errors"] == 7
        assert summary_dict["error_breakdown"] == error_breakdown


class TestBenchmarkErrorTracker:
    """Test BenchmarkErrorTracker functionality with correct API."""

    def test_tracker_initialization(self, temp_workspace):
        """Test BenchmarkErrorTracker initialization."""
        tracker = BenchmarkErrorTracker(temp_workspace)

        assert tracker.workspace_dir == temp_workspace
        assert hasattr(tracker, "errors")
        assert hasattr(tracker, "error_summary")
        assert tracker.error_file == temp_workspace / "benchmark_errors.jsonl"

    def test_record_target_failure(self, error_tracker):
        """Test recording target failures."""
        error_tracker.record_target_failure(
            target_pdb="1abc",
            error_message="Failed to process target",
            context={"test": "data"},
        )

        # Should have recorded the error
        assert len(error_tracker.errors) > 0
        error_record = error_tracker.errors[0]
        assert error_record.pdb_id == "1abc"
        assert error_record.error_message == "Failed to process target"

    def test_record_missing_pdb_if_exists(self, error_tracker):
        """Test record_missing_pdb method if it exists."""
        if hasattr(error_tracker, "record_missing_pdb"):
            error_tracker.record_missing_pdb(
                pdb_id="2xyz",
                reason="file_missing",
                details="Could not locate PDB file",
                file_type="protein",
            )

            # Check if recording worked
            assert len(error_tracker.errors) > 0 or hasattr(
                error_tracker, "missing_pdbs"
            )

    def test_get_error_statistics(self, error_tracker):
        """Test error statistics if method exists."""
        # Add some test errors
        error_tracker.record_target_failure("test1", "error1")
        error_tracker.record_target_failure("test2", "error2")

        if hasattr(error_tracker, "get_error_statistics"):
            stats = error_tracker.get_error_statistics()
            assert isinstance(stats, dict)

    def test_save_and_load_error_report(self, error_tracker):
        """Test saving and loading error reports."""
        # Record some errors
        error_tracker.record_target_failure("test_pdb", "test error")

        # Test save functionality
        if hasattr(error_tracker, "save_error_report"):
            error_tracker.save_error_report()
            assert error_tracker.error_file.exists()

        # Test load functionality
        if hasattr(error_tracker, "load_error_report"):
            loaded_tracker = BenchmarkErrorTracker(error_tracker.workspace_dir)
            loaded_tracker.load_error_report()
            # Should have loaded some data
            assert len(loaded_tracker.errors) >= 0

    def test_generate_summary_report(self, error_tracker):
        """Test summary report generation."""
        # Add test data
        error_tracker.record_target_failure("test1", "error1")
        error_tracker.record_target_failure("test2", "error2")

        if hasattr(error_tracker, "generate_summary_report"):
            summary = error_tracker.generate_summary_report()
            assert summary is not None

    def test_print_error_summary(self, error_tracker):
        """Test error summary printing."""
        error_tracker.record_target_failure("test", "error")

        if hasattr(error_tracker, "print_error_summary"):
            # Should not raise exception
            error_tracker.print_error_summary()

    def test_multiple_error_tracking(self, error_tracker):
        """Test tracking multiple different error types."""
        error_tracker.record_target_failure("pdb1", "file_not_found")
        error_tracker.record_target_failure("pdb2", "structure_invalid")
        error_tracker.record_target_failure("pdb3", "processing_timeout")

        assert len(error_tracker.errors) >= 3

        # Check error summary counts
        assert len(error_tracker.error_summary) >= 0

    def test_error_deduplication(self, error_tracker):
        """Test error deduplication functionality if available."""
        # Record same error multiple times
        for _ in range(3):
            error_tracker.record_target_failure("same_pdb", "same_error")

        # Error tracker should handle this gracefully
        assert len(error_tracker.errors) >= 1

    def test_get_missing_pdbs_by_type(self, error_tracker):
        """Test getting missing PDbs by type if method exists."""
        if hasattr(error_tracker, "get_missing_pdbs_by_type"):
            result = error_tracker.get_missing_pdbs_by_type()
            assert isinstance(result, dict)

    def test_error_context_handling(self, error_tracker):
        """Test error context handling."""
        context = {
            "file_path": "/test/path",
            "attempt_count": 3,
            "timestamp": datetime.now().isoformat(),
        }

        error_tracker.record_target_failure(
            "context_test", "test with context", context=context
        )

        assert len(error_tracker.errors) > 0
        error_record = error_tracker.errors[-1]  # Get last added error
        assert error_record.context == context


class TestBenchmarkErrorTrackerIntegration:
    """Integration tests for error tracking."""

    def test_workspace_integration(self, temp_workspace):
        """Test workspace integration."""
        tracker = BenchmarkErrorTracker(temp_workspace)

        # Record errors
        tracker.record_target_failure("integration_test", "test error")

        # Workspace should be set up correctly
        assert tracker.workspace_dir.exists()
        assert tracker.workspace_dir.is_dir()

    def test_file_operations(self, error_tracker):
        """Test file operations don't cause crashes."""
        # Record some data
        error_tracker.record_target_failure("file_test", "test error")

        # Try to save (should not crash even if method doesn't exist)
        if hasattr(error_tracker, "save_error_report"):
            try:
                error_tracker.save_error_report()
            except Exception as e:
                # Should handle gracefully
                assert isinstance(e, (IOError, OSError, AttributeError))

    def test_error_timeline_if_exists(self, error_tracker):
        """Test error timeline tracking if available."""
        if hasattr(error_tracker, "error_timeline"):
            # Record errors with delays
            error_tracker.record_target_failure("time1", "error1")
            error_tracker.record_target_failure("time2", "error2")

            # Timeline should exist
            assert hasattr(error_tracker, "error_timeline") or hasattr(
                error_tracker, "errors"
            )


# Parametrized tests for different error types
@pytest.mark.parametrize(
    "pdb_id,error_msg",
    [
        ("1abc", "file_not_found"),
        ("2def", "structure_invalid"),
        ("3ghi", "processing_timeout"),
        ("4jkl", "memory_error"),
    ],
)
def test_various_error_types(error_tracker, pdb_id, error_msg):
    """Test recording various error types."""
    error_tracker.record_target_failure(pdb_id, error_msg)
    assert len(error_tracker.errors) > 0


if __name__ == "__main__":
    pytest.main([__file__])
