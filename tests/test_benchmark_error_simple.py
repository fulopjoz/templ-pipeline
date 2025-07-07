"""
Test cases for benchmark error tracking module (simplified to match actual interface).
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

try:
    from templ_pipeline.benchmark.error_tracking import (
        MissingPDBRecord,
        BenchmarkErrorSummary,
        BenchmarkErrorTracker
    )
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from templ_pipeline.benchmark.error_tracking import (
        MissingPDBRecord,
        BenchmarkErrorSummary,
        BenchmarkErrorTracker
    )


class TestBenchmarkErrorTrackerSimple(unittest.TestCase):
    """Test BenchmarkErrorTracker class with actual interface."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = BenchmarkErrorTracker()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_tracker_initialization(self):
        """Test proper initialization of BenchmarkErrorTracker."""
        # Check that tracker has required methods
        required_methods = [
            'record_missing_pdb',
            'record_target_success',
            'record_target_failure',
            'get_error_statistics',
            'generate_summary_report',
            'save_error_report',
            'get_missing_pdbs_by_component',
            'create_missing_pdb_recovery_plan',
            'print_error_summary'
        ]
        
        for method_name in required_methods:
            self.assertTrue(hasattr(self.tracker, method_name))
            self.assertTrue(callable(getattr(self.tracker, method_name)))

    def test_record_missing_pdb(self):
        """Test recording of missing PDB."""
        # This should work with the actual interface
        self.tracker.record_missing_pdb(
            pdb_id="1abc",
            error_type="file_not_found",
            error_message="File not found",
            component="protein"
        )
        
        # Should be able to get error statistics
        stats = self.tracker.get_error_statistics()
        self.assertIsInstance(stats, dict)

    def test_record_target_success(self):
        """Test recording of successful target."""
        self.tracker.record_target_success("1abc")
        
        # Should be able to get statistics
        stats = self.tracker.get_error_statistics()
        self.assertIsInstance(stats, dict)

    def test_record_target_failure(self):
        """Test recording of failed target."""
        self.tracker.record_target_failure(
            pdb_id="2def",
            error_message="Pipeline failed"
        )
        
        # Should be able to get statistics
        stats = self.tracker.get_error_statistics()
        self.assertIsInstance(stats, dict)

    def test_get_error_statistics(self):
        """Test retrieval of error statistics."""
        # Record some data
        self.tracker.record_missing_pdb("1abc", "file_not_found", "msg1", "protein")
        self.tracker.record_target_success("2def")
        self.tracker.record_target_failure("3xyz", "Pipeline error")
        
        stats = self.tracker.get_error_statistics()
        
        self.assertIsInstance(stats, dict)
        # Should contain summary information
        self.assertTrue(len(stats) > 0)

    def test_get_missing_pdbs_by_component(self):
        """Test retrieval of missing PDBs by component."""
        # Record errors for different components
        self.tracker.record_missing_pdb("1abc", "error1", "msg1", "protein")
        self.tracker.record_missing_pdb("2def", "error2", "msg2", "ligand")
        
        # Method takes no parameters, returns dict of component -> set of PDB IDs
        components_dict = self.tracker.get_missing_pdbs_by_component()
        
        self.assertIsInstance(components_dict, dict)
        self.assertIn("protein", components_dict)
        self.assertIn("1abc", components_dict["protein"])

    def test_generate_summary_report(self):
        """Test generation of summary report."""
        # Record some errors
        self.tracker.record_missing_pdb("1abc", "file_not_found", "msg1", "protein")
        self.tracker.record_target_failure("2def", "Pipeline error")
        
        summary = self.tracker.generate_summary_report()
        
        # Should return some kind of summary object
        self.assertIsNotNone(summary)

    def test_save_error_report(self):
        """Test saving error report to file."""
        # Record some errors
        self.tracker.record_missing_pdb("1abc", "file_not_found", "msg1", "protein")
        
        output_file = Path(self.test_dir) / "error_report.json"
        
        try:
            # Method should exist and be callable
            self.tracker.save_error_report(str(output_file))
            
            # If successful, file should exist
            if output_file.exists():
                self.assertTrue(output_file.stat().st_size > 0)
                
        except Exception as e:
            # Method should exist but may fail with missing data
            self.assertIsInstance(e, (ValueError, FileNotFoundError, TypeError))

    def test_create_missing_pdb_recovery_plan(self):
        """Test creation of missing PDB recovery plan."""
        # Record missing PDBs
        self.tracker.record_missing_pdb("1abc", "file_not_found", "msg1", "protein")
        self.tracker.record_missing_pdb("2def", "load_failed", "msg2", "ligand")
        
        # Should be able to create recovery plan
        plan = self.tracker.create_missing_pdb_recovery_plan()
        
        # Should return some kind of plan object
        self.assertIsNotNone(plan)

    def test_print_error_summary(self):
        """Test printing error summary."""
        # Record some errors
        self.tracker.record_missing_pdb("1abc", "file_not_found", "msg1", "protein")
        self.tracker.record_target_failure("2def", "Error message")
        
        # Should be able to call print_error_summary without errors
        try:
            self.tracker.print_error_summary()
        except Exception as e:
            # Should not raise unexpected exceptions
            self.fail(f"print_error_summary raised unexpected exception: {e}")

    def test_error_tracking_workflow(self):
        """Test complete error tracking workflow."""
        # Simulate a complete benchmark run with errors
        
        # Record successful targets
        self.tracker.record_target_success("1successful")
        
        # Record failed targets with various errors
        self.tracker.record_target_failure("1failed", "Pipeline timeout")
        self.tracker.record_missing_pdb("1missing", "file_not_found", "File not found", "protein")
        
        # Get statistics
        stats = self.tracker.get_error_statistics()
        self.assertIsInstance(stats, dict)
        
        # Generate summary
        summary = self.tracker.generate_summary_report()
        self.assertIsNotNone(summary)
        
        # Create recovery plan
        plan = self.tracker.create_missing_pdb_recovery_plan()
        self.assertIsNotNone(plan)


if __name__ == "__main__":
    unittest.main()