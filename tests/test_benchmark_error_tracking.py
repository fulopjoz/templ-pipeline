"""
Test cases for benchmark error tracking module.
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


class TestMissingPDBRecord(unittest.TestCase):
    """Test MissingPDBRecord dataclass."""

    def test_missing_pdb_record_creation(self):
        """Test creation of MissingPDBRecord."""
        timestamp = datetime.now().isoformat()
        record = MissingPDBRecord(
            pdb_id="1abc",
            error_type="file_not_found",
            error_message="PDB file not found in data directory",
            component="protein",
            timestamp=timestamp
        )
        
        self.assertEqual(record.pdb_id, "1abc")
        self.assertEqual(record.error_type, "file_not_found")
        self.assertEqual(record.error_message, "PDB file not found in data directory")
        self.assertEqual(record.component, "protein")
        self.assertEqual(record.timestamp, timestamp)
        self.assertIsNone(record.context)

    def test_missing_pdb_record_with_context(self):
        """Test creation of MissingPDBRecord with context."""
        timestamp = datetime.now().isoformat()
        context = {"attempted_paths": ["/path1", "/path2"], "search_depth": 3}
        
        record = MissingPDBRecord(
            pdb_id="2def",
            error_type="load_failed",
            error_message="Failed to load ligand structure",
            component="ligand",
            timestamp=timestamp,
            context=context
        )
        
        self.assertEqual(record.pdb_id, "2def")
        self.assertEqual(record.error_type, "load_failed")
        self.assertEqual(record.component, "ligand")
        self.assertEqual(record.context, context)

    def test_missing_pdb_record_to_dict(self):
        """Test conversion of MissingPDBRecord to dictionary."""
        timestamp = datetime.now().isoformat()
        context = {"test_key": "test_value"}
        
        record = MissingPDBRecord(
            pdb_id="3xyz",
            error_type="invalid_structure",
            error_message="Structure validation failed",
            component="template",
            timestamp=timestamp,
            context=context
        )
        
        record_dict = record.to_dict()
        
        self.assertIsInstance(record_dict, dict)
        self.assertEqual(record_dict["pdb_id"], "3xyz")
        self.assertEqual(record_dict["error_type"], "invalid_structure")
        self.assertEqual(record_dict["error_message"], "Structure validation failed")
        self.assertEqual(record_dict["component"], "template")
        self.assertEqual(record_dict["timestamp"], timestamp)
        self.assertEqual(record_dict["context"], context)


class TestBenchmarkErrorSummary(unittest.TestCase):
    """Test BenchmarkErrorSummary dataclass."""

    def test_error_summary_creation(self):
        """Test creation of BenchmarkErrorSummary."""
        timestamp = datetime.now().isoformat()
        
        missing_record = MissingPDBRecord(
            pdb_id="1abc",
            error_type="file_not_found",
            error_message="File not found",
            component="protein",
            timestamp=timestamp
        )
        
        summary = BenchmarkErrorSummary(
            total_targets=100,
            successful_targets=85,
            failed_targets=15,
            missing_pdbs={"1abc": [missing_record]},
            error_categories={"file_not_found": 10, "load_failed": 5},
            error_timeline=[(timestamp, "file_not_found")]
        )
        
        self.assertEqual(summary.total_targets, 100)
        self.assertEqual(summary.successful_targets, 85)
        self.assertEqual(summary.failed_targets, 15)
        self.assertEqual(len(summary.missing_pdbs), 1)
        self.assertIn("1abc", summary.missing_pdbs)
        self.assertEqual(summary.error_categories["file_not_found"], 10)
        self.assertEqual(len(summary.error_timeline), 1)

    def test_error_summary_to_dict(self):
        """Test conversion of BenchmarkErrorSummary to dictionary."""
        timestamp = datetime.now().isoformat()
        
        missing_record = MissingPDBRecord(
            pdb_id="2def",
            error_type="load_failed",
            error_message="Load failed",
            component="ligand",
            timestamp=timestamp
        )
        
        summary = BenchmarkErrorSummary(
            total_targets=50,
            successful_targets=40,
            failed_targets=10,
            missing_pdbs={"2def": [missing_record]},
            error_categories={"load_failed": 10},
            error_timeline=[(timestamp, "load_failed")]
        )
        
        summary_dict = summary.to_dict()
        
        self.assertIsInstance(summary_dict, dict)
        self.assertEqual(summary_dict["total_targets"], 50)
        self.assertEqual(summary_dict["successful_targets"], 40)
        self.assertEqual(summary_dict["failed_targets"], 10)
        self.assertIn("missing_pdbs", summary_dict)
        self.assertIn("error_categories", summary_dict)
        self.assertIn("error_timeline", summary_dict)
        
        # Check that missing PDB records are properly converted
        self.assertIn("2def", summary_dict["missing_pdbs"])
        self.assertEqual(len(summary_dict["missing_pdbs"]["2def"]), 1)
        self.assertIsInstance(summary_dict["missing_pdbs"]["2def"][0], dict)


class TestBenchmarkErrorTracker(unittest.TestCase):
    """Test BenchmarkErrorTracker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = BenchmarkErrorTracker()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_tracker_initialization(self):
        """Test proper initialization of BenchmarkErrorTracker."""
        self.assertIsInstance(self.tracker.missing_pdbs, dict)
        self.assertIsInstance(self.tracker.error_counts, dict)
        self.assertIsInstance(self.tracker.error_timeline, list)
        self.assertEqual(len(self.tracker.missing_pdbs), 0)
        self.assertEqual(len(self.tracker.error_counts), 0)
        self.assertEqual(len(self.tracker.error_timeline), 0)

    def test_record_missing_pdb(self):
        """Test recording of missing PDB."""
        self.tracker.record_missing_pdb(
            pdb_id="1abc",
            error_type="file_not_found",
            error_message="File not found",
            component="protein"
        )
        
        self.assertIn("1abc", self.tracker.missing_pdbs)
        self.assertEqual(len(self.tracker.missing_pdbs["1abc"]), 1)
        
        record = self.tracker.missing_pdbs["1abc"][0]
        self.assertEqual(record.pdb_id, "1abc")
        self.assertEqual(record.error_type, "file_not_found")
        self.assertEqual(record.component, "protein")
        
        # Check error counts
        self.assertEqual(self.tracker.error_counts["file_not_found"], 1)
        self.assertEqual(len(self.tracker.error_timeline), 1)

    def test_record_multiple_missing_pdbs(self):
        """Test recording multiple missing PDBs."""
        # Record first PDB
        self.tracker.record_missing_pdb(
            pdb_id="1abc",
            error_type="file_not_found",
            error_message="File not found",
            component="protein"
        )
        
        # Record second PDB
        self.tracker.record_missing_pdb(
            pdb_id="2def",
            error_type="load_failed",
            error_message="Load failed",
            component="ligand"
        )
        
        # Record another error for first PDB
        self.tracker.record_missing_pdb(
            pdb_id="1abc",
            error_type="invalid_structure",
            error_message="Invalid structure",
            component="template"
        )
        
        self.assertEqual(len(self.tracker.missing_pdbs), 2)
        self.assertEqual(len(self.tracker.missing_pdbs["1abc"]), 2)
        self.assertEqual(len(self.tracker.missing_pdbs["2def"]), 1)
        
        # Check error counts
        self.assertEqual(self.tracker.error_counts["file_not_found"], 1)
        self.assertEqual(self.tracker.error_counts["load_failed"], 1)
        self.assertEqual(self.tracker.error_counts["invalid_structure"], 1)
        self.assertEqual(len(self.tracker.error_timeline), 3)

    def test_record_missing_pdb_with_context(self):
        """Test recording missing PDB with context information."""
        context = {"search_paths": ["/path1", "/path2"], "timeout": 30}
        
        self.tracker.record_missing_pdb(
            pdb_id="3xyz",
            error_type="timeout",
            error_message="Operation timed out",
            component="embedding",
            context=context
        )
        
        record = self.tracker.missing_pdbs["3xyz"][0]
        self.assertEqual(record.context, context)

    def test_get_missing_pdbs_by_type(self):
        """Test retrieval of missing PDBs by error type."""
        # Record different types of errors
        self.tracker.record_missing_pdb("1abc", "file_not_found", "msg1", "protein")
        self.tracker.record_missing_pdb("2def", "file_not_found", "msg2", "ligand")
        self.tracker.record_missing_pdb("3xyz", "load_failed", "msg3", "template")
        
        file_not_found_pdbs = self.tracker.get_missing_pdbs_by_type("file_not_found")
        load_failed_pdbs = self.tracker.get_missing_pdbs_by_type("load_failed")
        
        self.assertEqual(len(file_not_found_pdbs), 2)
        self.assertEqual(len(load_failed_pdbs), 1)
        self.assertIn("1abc", file_not_found_pdbs)
        self.assertIn("2def", file_not_found_pdbs)
        self.assertIn("3xyz", load_failed_pdbs)

    def test_get_missing_pdbs_by_component(self):
        """Test retrieval of missing PDBs by component."""
        # Record errors for different components
        self.tracker.record_missing_pdb("1abc", "error1", "msg1", "protein")
        self.tracker.record_missing_pdb("2def", "error2", "msg2", "protein")
        self.tracker.record_missing_pdb("3xyz", "error3", "msg3", "ligand")
        
        protein_pdbs = self.tracker.get_missing_pdbs_by_component("protein")
        ligand_pdbs = self.tracker.get_missing_pdbs_by_component("ligand")
        
        self.assertEqual(len(protein_pdbs), 2)
        self.assertEqual(len(ligand_pdbs), 1)
        self.assertIn("1abc", protein_pdbs)
        self.assertIn("2def", protein_pdbs)
        self.assertIn("3xyz", ligand_pdbs)

    def test_get_error_statistics(self):
        """Test retrieval of error statistics."""
        # Record various errors
        self.tracker.record_missing_pdb("1abc", "file_not_found", "msg1", "protein")
        self.tracker.record_missing_pdb("2def", "file_not_found", "msg2", "ligand")
        self.tracker.record_missing_pdb("3xyz", "load_failed", "msg3", "template")
        
        stats = self.tracker.get_error_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats["total_errors"], 3)
        self.assertEqual(stats["unique_pdbs"], 3)
        self.assertEqual(stats["error_types"]["file_not_found"], 2)
        self.assertEqual(stats["error_types"]["load_failed"], 1)
        self.assertEqual(stats["components"]["protein"], 1)
        self.assertEqual(stats["components"]["ligand"], 1)
        self.assertEqual(stats["components"]["template"], 1)

    def test_generate_summary_report(self):
        """Test generation of error summary report."""
        # Record some targets and errors
        self.tracker.record_target_success("1successful")
        self.tracker.record_target_failure("1failed", "Pipeline error")
        self.tracker.record_missing_pdb("1abc", "file_not_found", "msg1", "protein")
        self.tracker.record_missing_pdb("2def", "load_failed", "msg2", "ligand")
        
        # Method takes no parameters, calculates totals from internal state
        summary = self.tracker.generate_summary_report()
        
        self.assertIsInstance(summary, BenchmarkErrorSummary)
        self.assertEqual(summary.total_targets, 2)  # 1 successful + 1 failed
        self.assertEqual(summary.successful_targets, 1)
        self.assertEqual(summary.failed_targets, 1)
        self.assertEqual(len(summary.missing_pdbs), 3)  # 2 explicit + 1 from failed target
        self.assertIn("file_not_found", summary.error_categories)
        self.assertEqual(summary.error_categories["load_failed"], 1)

    def test_save_error_report(self):
        """Test saving error report to file."""
        # Record some errors
        self.tracker.record_missing_pdb("1abc", "file_not_found", "msg1", "protein")
        self.tracker.record_missing_pdb("2def", "load_failed", "msg2", "ligand")
        
        summary = self.tracker.generate_error_summary(50, 48, 2)
        output_file = Path(self.test_dir) / "error_report.json"
        
        self.tracker.save_error_report(summary, str(output_file))
        
        self.assertTrue(output_file.exists())
        
        with open(output_file, 'r') as f:
            report_data = json.load(f)
        
        self.assertIn("summary", report_data)
        self.assertIn("detailed_errors", report_data)
        self.assertIn("metadata", report_data)
        self.assertEqual(report_data["summary"]["total_targets"], 50)

    def test_load_error_report(self):
        """Test loading error report from file."""
        # Create a test error report
        test_data = {
            "summary": {
                "total_targets": 100,
                "successful_targets": 95,
                "failed_targets": 5
            },
            "detailed_errors": {
                "1abc": [{
                    "error_type": "file_not_found",
                    "error_message": "File not found",
                    "component": "protein",
                    "timestamp": datetime.now().isoformat()
                }]
            },
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        
        report_file = Path(self.test_dir) / "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(test_data, f)
        
        loaded_data = self.tracker.load_error_report(str(report_file))
        
        self.assertEqual(loaded_data["summary"]["total_targets"], 100)
        self.assertIn("1abc", loaded_data["detailed_errors"])

    def test_print_error_summary(self):
        """Test printing error summary to console."""
        # Record some targets and errors
        self.tracker.record_target_success("1successful")
        self.tracker.record_target_failure("1failed", "Pipeline error")
        self.tracker.record_missing_pdb("1abc", "file_not_found", "msg1", "protein")
        
        # Should not raise exceptions
        try:
            self.tracker.print_error_summary()
        except Exception as e:
            self.fail(f"print_error_summary raised unexpected exception: {e}")
        
        # Test with skip information disabled
        try:
            self.tracker.print_error_summary(include_skips=False)
        except Exception as e:
            self.fail(f"print_error_summary raised unexpected exception: {e}")

    def test_multiple_error_tracking(self):
        """Test tracking multiple errors for same PDB."""
        # Record multiple errors for same PDB
        self.tracker.record_missing_pdb("1abc", "file_not_found", "msg1", "protein")
        self.tracker.record_missing_pdb("1abc", "load_failed", "msg2", "protein")
        
        # Verify multiple records for same PDB
        self.assertEqual(len(self.tracker.missing_pdbs["1abc"]), 2)
        
        # Verify error counts
        self.assertEqual(self.tracker.error_counts["file_not_found"], 1)
        self.assertEqual(self.tracker.error_counts["load_failed"], 1)
        
        # Verify timeline
        self.assertEqual(len(self.tracker.error_timeline), 2)

    def test_error_timeline_tracking(self):
        """Test error timeline tracking functionality."""
        # Record errors over time
        self.tracker.record_missing_pdb("1abc", "error1", "msg1", "protein")
        self.tracker.record_missing_pdb("2def", "error2", "msg2", "ligand")
        
        # Verify timeline has entries
        self.assertEqual(len(self.tracker.error_timeline), 2)
        
        # Verify timeline structure (timestamp, error_type)
        for timestamp, error_type in self.tracker.error_timeline:
            self.assertIsInstance(timestamp, str)
            self.assertIsInstance(error_type, str)
            self.assertIn(error_type, ["error1", "error2"])

    def test_missing_pdb_deduplication(self):
        """Test that missing PDB lists handle deduplication correctly."""
        # Record errors for different PDBs including duplicates
        self.tracker.record_missing_pdb("1abc", "error1", "msg1", "protein")
        self.tracker.record_missing_pdb("2def", "error2", "msg2", "ligand")
        self.tracker.record_missing_pdb("1abc", "error3", "msg3", "template")  # Duplicate PDB
        
        # Get unique missing PDB list from missing_pdbs keys
        missing_list = list(self.tracker.missing_pdbs.keys())
        
        self.assertIsInstance(missing_list, list)
        self.assertEqual(len(missing_list), 2)  # Should deduplicate
        self.assertIn("1abc", missing_list)
        self.assertIn("2def", missing_list)

    def test_get_missing_pdbs_by_component_detailed(self):
        """Test detailed missing PDB retrieval by component."""
        # Record errors for different components
        self.tracker.record_missing_pdb("1abc", "error1", "msg1", "protein")
        self.tracker.record_missing_pdb("2def", "error2", "msg2", "ligand")
        self.tracker.record_missing_pdb("3xyz", "error3", "msg3", "protein")
        
        # Use the actual method that exists
        by_component = self.tracker.get_missing_pdbs_by_component()
        
        self.assertEqual(len(by_component["protein"]), 2)
        self.assertEqual(len(by_component["ligand"]), 1)
        self.assertIn("1abc", by_component["protein"])
        self.assertIn("3xyz", by_component["protein"])
        self.assertIn("2def", by_component["ligand"])


if __name__ == "__main__":
    unittest.main()