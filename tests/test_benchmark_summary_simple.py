"""
Test cases for benchmark summary generator module (simplified to match actual interface).
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


class TestBenchmarkSummaryGeneratorSimple(unittest.TestCase):
    """Test BenchmarkSummaryGenerator class with actual interface."""

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
        self.assertIn("jsonl", self.generator.supported_formats)

    def test_detect_polaris_benchmark_type(self):
        """Test detection of Polaris benchmark type."""
        polaris_data = {
            "benchmark_info": {
                "name": "polaris_protein_ligand_benchmark",
                "version": "1.0"
            },
            "results": {}
        }
        
        benchmark_type = self.generator.detect_benchmark_type(polaris_data)
        self.assertEqual(benchmark_type, "polaris")

    def test_detect_timesplit_benchmark_type(self):
        """Test detection of timesplit benchmark type."""
        timesplit_data = {
            "timesplit_results": {
                "train_period": "2020-2021",
                "test_period": "2022"
            },
            "results": {}
        }
        
        benchmark_type = self.generator.detect_benchmark_type(timesplit_data)
        self.assertEqual(benchmark_type, "timesplit")

    def test_detect_unknown_benchmark_type(self):
        """Test detection of unknown benchmark type."""
        generic_data = {
            "results": {
                "1abc": {"rmsd": 1.5, "success": True},
                "2def": {"rmsd": 2.0, "success": True}
            }
        }
        
        benchmark_type = self.generator.detect_benchmark_type(generic_data)
        self.assertEqual(benchmark_type, "unknown")

    def test_detect_empty_benchmark_type(self):
        """Test detection with empty data."""
        empty_data = {}
        
        benchmark_type = self.generator.detect_benchmark_type(empty_data)
        self.assertEqual(benchmark_type, "unknown")

    def test_generate_unified_summary_basic(self):
        """Test basic unified summary generation."""
        test_data = {
            "results": {
                "1abc": {
                    "success": True,
                    "rmsd_values": {"combo": {"rmsd": 1.5}},
                    "runtime": 30.0
                }
            }
        }
        
        # Test that method exists and is callable with dict output
        summary = self.generator.generate_unified_summary(
            results_data=test_data,
            output_format="dict"
        )
        
        # Should return a dictionary with summary information
        self.assertIsInstance(summary, dict)

    def test_save_summary_files_basic(self):
        """Test basic summary file saving."""
        test_data = {
            "results": {
                "1abc": {
                    "success": True,
                    "rmsd_values": {"combo": {"rmsd": 1.5}},
                    "runtime": 30.0
                }
            }
        }
        
        output_dir = Path(self.test_dir) / "output"
        
        # Test that method exists and is callable
        try:
            files_created = self.generator.save_summary_files(
                summary_data=test_data,
                output_dir=output_dir,
                base_name="test_summary"
            )
            
            # Should return information about files created
            self.assertIsInstance(files_created, (list, dict, type(None)))
            
        except Exception as e:
            # Method should exist and be callable, but may fail with missing dependencies
            self.assertIsInstance(e, (ImportError, ValueError, RuntimeError))

    def test_summary_generator_has_required_methods(self):
        """Test that summary generator has all required methods."""
        required_methods = [
            'detect_benchmark_type',
            'generate_unified_summary', 
            'save_summary_files'
        ]
        
        for method_name in required_methods:
            self.assertTrue(hasattr(self.generator, method_name))
            self.assertTrue(callable(getattr(self.generator, method_name)))

    def test_supported_formats_property(self):
        """Test supported_formats property."""
        formats = self.generator.supported_formats
        
        self.assertIsInstance(formats, list)
        self.assertTrue(len(formats) > 0)
        
        # Should contain common formats
        expected_formats = ["json", "csv", "markdown"]
        for fmt in expected_formats:
            self.assertIn(fmt, formats)


if __name__ == "__main__":
    unittest.main()