"""
Test cases for benchmark summary generator module.
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


class TestBenchmarkSummaryGenerator(unittest.TestCase):
    """Test BenchmarkSummaryGenerator class."""

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

    def test_detect_generic_benchmark_type(self):
        """Test detection of generic benchmark type."""
        generic_data = {
            "results": {
                "1abc": {"rmsd": 1.5, "success": True},
                "2def": {"rmsd": 2.0, "success": True}
            }
        }
        
        benchmark_type = self.generator.detect_benchmark_type(generic_data)
        self.assertEqual(benchmark_type, "generic")

    def test_detect_empty_benchmark_type(self):
        """Test detection with empty data."""
        empty_data = {}
        
        benchmark_type = self.generator.detect_benchmark_type(empty_data)
        self.assertEqual(benchmark_type, "generic")

    def test_extract_polaris_metrics(self):
        """Test extraction of metrics from Polaris benchmark data."""
        polaris_data = {
            "benchmark_info": {
                "name": "polaris_test",
                "description": "Test benchmark"
            },
            "results": {
                "1abc": {
                    "success": True,
                    "rmsd_values": {"combo": {"rmsd": 1.2}},
                    "runtime": 45.5
                },
                "2def": {
                    "success": False,
                    "error": "Pipeline failed",
                    "runtime": 10.0
                }
            }
        }
        
        metrics = self.generator.extract_benchmark_metrics(polaris_data)
        
        self.assertIsInstance(metrics, list)
        self.assertEqual(len(metrics), 2)
        
        # Check first result
        result1 = metrics[0]
        self.assertEqual(result1["pdb_id"], "1abc")
        self.assertTrue(result1["success"])
        self.assertEqual(result1["runtime"], 45.5)
        self.assertEqual(result1["combo_rmsd"], 1.2)
        
        # Check second result
        result2 = metrics[1]
        self.assertEqual(result2["pdb_id"], "2def")
        self.assertFalse(result2["success"])
        self.assertEqual(result2["runtime"], 10.0)
        self.assertIsNone(result2["combo_rmsd"])

    def test_extract_timesplit_metrics(self):
        """Test extraction of metrics from timesplit benchmark data."""
        timesplit_data = {
            "timesplit_info": {
                "train_cutoff": "2021-01-01",
                "test_period": "2022"
            },
            "results": {
                "test_set_1": {
                    "1xyz": {
                        "success": True,
                        "rmsd_values": {"shape": {"rmsd": 0.8}},
                        "runtime": 30.2
                    }
                }
            }
        }
        
        metrics = self.generator.extract_benchmark_metrics(timesplit_data)
        
        self.assertIsInstance(metrics, list)
        self.assertEqual(len(metrics), 1)
        
        result = metrics[0]
        self.assertEqual(result["pdb_id"], "1xyz")
        self.assertTrue(result["success"])
        self.assertEqual(result["runtime"], 30.2)
        self.assertEqual(result["shape_rmsd"], 0.8)

    def test_extract_generic_metrics(self):
        """Test extraction of metrics from generic benchmark data."""
        generic_data = {
            "results": {
                "3abc": {
                    "success": True,
                    "rmsd_values": {
                        "combo": {"rmsd": 1.5},
                        "pharmacophore": {"rmsd": 2.0}
                    },
                    "runtime": 60.0
                }
            }
        }
        
        metrics = self.generator.extract_benchmark_metrics(generic_data)
        
        self.assertIsInstance(metrics, list)
        self.assertEqual(len(metrics), 1)
        
        result = metrics[0]
        self.assertEqual(result["pdb_id"], "3abc")
        self.assertTrue(result["success"])
        self.assertEqual(result["runtime"], 60.0)
        self.assertEqual(result["combo_rmsd"], 1.5)
        self.assertEqual(result["pharmacophore_rmsd"], 2.0)

    def test_calculate_summary_statistics(self):
        """Test calculation of summary statistics."""
        metrics = [
            {"pdb_id": "1abc", "success": True, "runtime": 30.0, "combo_rmsd": 1.2},
            {"pdb_id": "2def", "success": True, "runtime": 45.0, "combo_rmsd": 1.8},
            {"pdb_id": "3xyz", "success": False, "runtime": 10.0, "combo_rmsd": None},
            {"pdb_id": "4ghi", "success": True, "runtime": 25.0, "combo_rmsd": 0.9}
        ]
        
        stats = self.generator.calculate_summary_statistics(metrics)
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats["total_targets"], 4)
        self.assertEqual(stats["successful_runs"], 3)
        self.assertEqual(stats["failed_runs"], 1)
        self.assertAlmostEqual(stats["success_rate"], 0.75, places=2)
        
        # Check runtime statistics
        self.assertAlmostEqual(stats["avg_runtime"], 27.5, places=1)
        self.assertEqual(stats["min_runtime"], 10.0)
        self.assertEqual(stats["max_runtime"], 45.0)
        
        # Check RMSD statistics (only successful runs)
        self.assertAlmostEqual(stats["avg_combo_rmsd"], 1.3, places=1)
        self.assertEqual(stats["min_combo_rmsd"], 0.9)
        self.assertEqual(stats["max_combo_rmsd"], 1.8)

    def test_calculate_summary_statistics_empty(self):
        """Test calculation of summary statistics with empty data."""
        stats = self.generator.calculate_summary_statistics([])
        
        self.assertEqual(stats["total_targets"], 0)
        self.assertEqual(stats["successful_runs"], 0)
        self.assertEqual(stats["failed_runs"], 0)
        self.assertEqual(stats["success_rate"], 0.0)

    def test_calculate_summary_statistics_all_failed(self):
        """Test calculation of summary statistics with all failed runs."""
        metrics = [
            {"pdb_id": "1abc", "success": False, "runtime": 10.0, "combo_rmsd": None},
            {"pdb_id": "2def", "success": False, "runtime": 15.0, "combo_rmsd": None}
        ]
        
        stats = self.generator.calculate_summary_statistics(metrics)
        
        self.assertEqual(stats["total_targets"], 2)
        self.assertEqual(stats["successful_runs"], 0)
        self.assertEqual(stats["failed_runs"], 2)
        self.assertEqual(stats["success_rate"], 0.0)
        self.assertIsNone(stats["avg_combo_rmsd"])

    def test_generate_unified_summary_json(self):
        """Test generation of unified summary in JSON format."""
        test_data = {
            "results": {
                "1abc": {
                    "success": True,
                    "rmsd_values": {"combo": {"rmsd": 1.5}},
                    "runtime": 30.0
                }
            }
        }
        
        # Test JSON output format
        summary = self.generator.generate_unified_summary(
            results_data=test_data,
            benchmark_type="unknown",
            output_format="dict"
        )
        
        self.assertIsInstance(summary, dict)
        # The summary should contain some processed data
        self.assertTrue(len(summary) > 0)

    def test_save_summary_files_basic(self):
        """Test saving summary files to multiple formats."""
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

    @patch('templ_pipeline.benchmark.summary_generator.PANDAS_AVAILABLE', False)
    def test_save_summary_files_no_pandas(self):
        """Test saving summary files when pandas is not available."""
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
        # Should still work for JSON format even without pandas
        saved_files = self.generator.save_summary_files(
            summary_data=test_data,
            output_dir=output_dir,
            base_name="test_summary",
            formats=["json"]
        )
        
        self.assertIsInstance(saved_files, dict)
        if "json" in saved_files:
            self.assertTrue(saved_files["json"].exists())
            self.generator.generate_summary_table(
                results_data=test_data,
                output_file=str(output_file),
                format_type="csv"
            )
        
        self.assertIn("pandas", str(context.exception))

    def test_generate_summary_table_markdown(self):
        """Test generation of summary table in Markdown format."""
        test_data = {
            "results": {
                "1abc": {
                    "success": True,
                    "rmsd_values": {"combo": {"rmsd": 1.5}},
                    "runtime": 30.0
                },
                "2def": {
                    "success": False,
                    "error": "Failed",
                    "runtime": 10.0
                }
            }
        }
        
        output_file = Path(self.test_dir) / "summary.md"
        
        self.generator.generate_summary_table(
            results_data=test_data,
            output_file=str(output_file),
            format_type="markdown"
        )
        
        self.assertTrue(output_file.exists())
        
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Check for markdown formatting
        self.assertIn("# Benchmark Summary", content)
        self.assertIn("## Summary Statistics", content)
        self.assertIn("## Individual Results", content)
        self.assertIn("|", content)  # Table formatting
        self.assertIn("1abc", content)
        self.assertIn("2def", content)

    def test_generate_summary_table_jsonl(self):
        """Test generation of summary table in JSONL format."""
        test_data = {
            "results": {
                "1abc": {
                    "success": True,
                    "rmsd_values": {"combo": {"rmsd": 1.5}},
                    "runtime": 30.0
                },
                "2def": {
                    "success": True,
                    "rmsd_values": {"shape": {"rmsd": 2.0}},
                    "runtime": 25.0
                }
            }
        }
        
        output_file = Path(self.test_dir) / "summary.jsonl"
        
        self.generator.generate_summary_table(
            results_data=test_data,
            output_file=str(output_file),
            format_type="jsonl"
        )
        
        self.assertTrue(output_file.exists())
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        # Should have one line per result
        self.assertEqual(len(lines), 2)
        
        # Each line should be valid JSON
        for line in lines:
            data = json.loads(line.strip())
            self.assertIn("pdb_id", data)
            self.assertIn("success", data)

    def test_generate_summary_table_invalid_format(self):
        """Test generation with invalid format type."""
        test_data = {"results": {}}
        output_file = Path(self.test_dir) / "summary.txt"
        
        with self.assertRaises(ValueError) as context:
            self.generator.generate_summary_table(
                results_data=test_data,
                output_file=str(output_file),
                format_type="invalid_format"
            )
        
        self.assertIn("Unsupported format", str(context.exception))

    def test_load_benchmark_results_valid_file(self):
        """Test loading valid benchmark results file."""
        test_data = {
            "results": {
                "1abc": {"success": True, "runtime": 30.0}
            }
        }
        
        results_file = Path(self.test_dir) / "results.json"
        with open(results_file, 'w') as f:
            json.dump(test_data, f)
        
        loaded_data = self.generator.load_benchmark_results(str(results_file))
        
        self.assertEqual(loaded_data, test_data)

    def test_load_benchmark_results_nonexistent_file(self):
        """Test loading nonexistent benchmark results file."""
        nonexistent_file = Path(self.test_dir) / "nonexistent.json"
        
        with self.assertRaises(FileNotFoundError):
            self.generator.load_benchmark_results(str(nonexistent_file))

    def test_load_benchmark_results_invalid_json(self):
        """Test loading invalid JSON file."""
        invalid_file = Path(self.test_dir) / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json content {")
        
        with self.assertRaises(json.JSONDecodeError):
            self.generator.load_benchmark_results(str(invalid_file))

    def test_format_runtime_seconds(self):
        """Test runtime formatting in seconds."""
        formatted = self.generator.format_runtime(45.5)
        self.assertEqual(formatted, "45.5s")

    def test_format_runtime_minutes(self):
        """Test runtime formatting in minutes."""
        formatted = self.generator.format_runtime(125.0)
        self.assertEqual(formatted, "2m 5s")

    def test_format_runtime_hours(self):
        """Test runtime formatting in hours."""
        formatted = self.generator.format_runtime(3665.0)
        self.assertEqual(formatted, "1h 1m 5s")

    def test_format_rmsd_value(self):
        """Test RMSD value formatting."""
        formatted = self.generator.format_rmsd(1.23456)
        self.assertEqual(formatted, "1.23")
        
        formatted_none = self.generator.format_rmsd(None)
        self.assertEqual(formatted_none, "N/A")

    def test_supported_formats_attribute(self):
        """Test supported formats attribute."""
        formats = self.generator.supported_formats
        
        self.assertIsInstance(formats, list)
        self.assertIn("json", formats)
        self.assertIn("csv", formats)
        self.assertIn("markdown", formats)
        self.assertIn("jsonl", formats)


if __name__ == "__main__":
    unittest.main()