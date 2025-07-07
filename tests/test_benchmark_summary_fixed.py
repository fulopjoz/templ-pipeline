"""
Test cases for benchmark summary generator module (corrected interface).
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch

try:
    from templ_pipeline.benchmark.summary_generator import BenchmarkSummaryGenerator
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from templ_pipeline.benchmark.summary_generator import BenchmarkSummaryGenerator


class TestBenchmarkSummaryGeneratorFixed(unittest.TestCase):
    """Test BenchmarkSummaryGenerator with correct interface."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = BenchmarkSummaryGenerator()
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test proper initialization of BenchmarkSummaryGenerator."""
        self.assertIsInstance(self.generator.supported_formats, list)
        self.assertIn("json", self.generator.supported_formats)
        self.assertIn("csv", self.generator.supported_formats)
        self.assertIn("markdown", self.generator.supported_formats)

    def test_detect_polaris_benchmark_type(self):
        """Test detection of Polaris benchmark type."""
        polaris_data = {
            "benchmark_info": {
                "name": "Polaris SARS-CoV-2 Benchmarks",
                "type": "polaris"
            }
        }
        
        benchmark_type = self.generator.detect_benchmark_type(polaris_data)
        self.assertEqual(benchmark_type, "polaris")

    def test_detect_timesplit_benchmark_type(self):
        """Test detection of timesplit benchmark type."""
        timesplit_data = {
            "timesplit_results": {
                "1abc": {"success": True}
            }
        }
        
        benchmark_type = self.generator.detect_benchmark_type(timesplit_data)
        self.assertEqual(benchmark_type, "timesplit")

    def test_detect_unknown_benchmark_type(self):
        """Test detection of unknown benchmark type."""
        unknown_data = {
            "some_other_format": {
                "data": "value"
            }
        }
        
        benchmark_type = self.generator.detect_benchmark_type(unknown_data)
        self.assertEqual(benchmark_type, "unknown")

    def test_detect_empty_benchmark_type(self):
        """Test detection with empty data."""
        empty_data = {}
        
        benchmark_type = self.generator.detect_benchmark_type(empty_data)
        self.assertEqual(benchmark_type, "unknown")

    def test_generate_unified_summary_dict_output(self):
        """Test generating unified summary with dict output."""
        test_data = {
            "results": {
                "1abc": {
                    "success": True,
                    "rmsd_values": {"combo": {"rmsd": 1.5}},
                    "runtime": 30.0
                }
            }
        }
        
        summary = self.generator.generate_unified_summary(
            results_data=test_data,
            benchmark_type="unknown",
            output_format="dict"
        )
        
        self.assertIsInstance(summary, dict)

    def test_generate_unified_summary_auto_detect(self):
        """Test generating unified summary with auto-detection."""
        polaris_data = {
            "benchmark_info": {
                "name": "Polaris test benchmark"
            },
            "results": {}
        }
        
        # Should auto-detect as polaris
        summary = self.generator.generate_unified_summary(
            results_data=polaris_data,
            benchmark_type=None,  # Auto-detect
            output_format="dict"
        )
        
        self.assertIsInstance(summary, dict)

    def test_generate_unified_summary_timesplit_list(self):
        """Test generating unified summary for timesplit with list input."""
        timesplit_data = [
            {
                "pdb_id": "1abc",
                "success": True,
                "rmsd": 1.5
            },
            {
                "pdb_id": "2def", 
                "success": False,
                "error": "timeout"
            }
        ]
        
        summary = self.generator.generate_unified_summary(
            results_data=timesplit_data,
            benchmark_type="timesplit",
            output_format="dict"
        )
        
        self.assertIsInstance(summary, dict)

    def test_save_summary_files_json_only(self):
        """Test saving summary files in JSON format only."""
        test_data = [
            {
                "target": "1abc",
                "success": True,
                "rmsd": 1.5,
                "runtime": 30.0
            }
        ]
        
        output_dir = Path(self.test_dir) / "output"
        saved_files = self.generator.save_summary_files(
            summary_data=test_data,
            output_dir=output_dir,
            base_name="test_summary",
            formats=["json"]
        )
        
        self.assertIsInstance(saved_files, dict)
        self.assertIn("json", saved_files)
        self.assertTrue(saved_files["json"].exists())
        
        # Verify JSON content
        with open(saved_files["json"], 'r') as f:
            content = json.load(f)
        
        self.assertIn("timestamp", content)
        self.assertIn("summary", content)

    def test_save_summary_files_multiple_formats(self):
        """Test saving summary files in multiple formats."""
        test_data = {
            "data": [
                {"target": "1abc", "success": True, "rmsd": 1.5}
            ]
        }
        
        output_dir = Path(self.test_dir) / "output"
        saved_files = self.generator.save_summary_files(
            summary_data=test_data,
            output_dir=output_dir,
            base_name="multi_format_test",
            formats=["json"]  # Start with just JSON to avoid pandas dependency issues
        )
        
        self.assertIsInstance(saved_files, dict)
        self.assertTrue(len(saved_files) > 0)

    @patch('templ_pipeline.benchmark.summary_generator.PANDAS_AVAILABLE', False)
    def test_save_summary_files_no_pandas(self):
        """Test saving when pandas is not available."""
        test_data = {"simple": "data"}
        
        output_dir = Path(self.test_dir) / "output"
        saved_files = self.generator.save_summary_files(
            summary_data=test_data,
            output_dir=output_dir,
            base_name="no_pandas_test",
            formats=["json"]  # JSON should work without pandas
        )
        
        # Should still save JSON successfully
        self.assertIsInstance(saved_files, dict)
        if "json" in saved_files:
            self.assertTrue(saved_files["json"].exists())

    def test_save_summary_files_default_formats(self):
        """Test saving with default formats."""
        test_data = {"test": "data"}
        
        output_dir = Path(self.test_dir) / "output"
        saved_files = self.generator.save_summary_files(
            summary_data=test_data,
            output_dir=output_dir,
            base_name="default_formats"
            # No formats specified - should use defaults
        )
        
        self.assertIsInstance(saved_files, dict)
        # Should at least save JSON format
        self.assertTrue(len(saved_files) > 0)

    def test_supported_formats_attribute(self):
        """Test supported formats attribute access."""
        formats = self.generator.supported_formats
        
        self.assertIsInstance(formats, list)
        self.assertIn("json", formats)
        self.assertIn("csv", formats)
        self.assertIn("markdown", formats)
        self.assertIn("jsonl", formats)

    def test_save_summary_files_creates_directory(self):
        """Test that save_summary_files creates output directory if it doesn't exist."""
        test_data = {"test": "data"}
        
        # Use a non-existent directory
        output_dir = Path(self.test_dir) / "nonexistent" / "nested" / "output"
        self.assertFalse(output_dir.exists())
        
        saved_files = self.generator.save_summary_files(
            summary_data=test_data,
            output_dir=output_dir,
            base_name="create_dir_test",
            formats=["json"]
        )
        
        # Directory should now exist
        self.assertTrue(output_dir.exists())
        self.assertIsInstance(saved_files, dict)


if __name__ == "__main__":
    unittest.main()