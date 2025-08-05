"""
Test cases for benchmark runner module.
"""

import unittest
import tempfile
import shutil
import json
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

try:
    from templ_pipeline.benchmark.runner import (
        BenchmarkParams,
        BenchmarkResult,
        monitor_memory_usage,
        run_templ_pipeline_for_benchmark
    )
    from templ_pipeline.core.pipeline import TEMPLPipeline
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from templ_pipeline.benchmark.runner import (
        BenchmarkParams,
        BenchmarkResult,
        monitor_memory_usage,
        run_templ_pipeline_for_benchmark
    )
    from templ_pipeline.core.pipeline import TEMPLPipeline

from rdkit import Chem


class TestBenchmarkParams(unittest.TestCase):
    """Test BenchmarkParams dataclass."""

    def test_benchmark_params_creation(self):
        """Test BenchmarkParams object creation with required fields."""
        exclude_set = {"1abc", "2def"}
        params = BenchmarkParams(
            target_pdb="3xyz",
            exclude_pdb_ids=exclude_set
        )
        
        self.assertEqual(params.target_pdb, "3xyz")
        self.assertEqual(params.exclude_pdb_ids, exclude_set)
        self.assertIsNone(params.poses_output_dir)
        self.assertEqual(params.n_conformers, 200)
        self.assertEqual(params.template_knn, 100)
        self.assertIsNone(params.similarity_threshold)
        self.assertEqual(params.internal_workers, 1)
        self.assertEqual(params.timeout, 300)

    def test_benchmark_params_with_optional_fields(self):
        """Test BenchmarkParams with all optional fields set."""
        exclude_set = {"1abc"}
        params = BenchmarkParams(
            target_pdb="4test",
            exclude_pdb_ids=exclude_set,
            poses_output_dir="/tmp/poses",
            n_conformers=50,
            template_knn=20,
            similarity_threshold=0.8,
            internal_workers=4,
            timeout=3300
        )
        
        self.assertEqual(params.target_pdb, "4test")
        self.assertEqual(params.exclude_pdb_ids, exclude_set)
        self.assertEqual(params.poses_output_dir, "/tmp/poses")
        self.assertEqual(params.n_conformers, 50)
        self.assertEqual(params.template_knn, 20)
        self.assertEqual(params.similarity_threshold, 0.8)
        self.assertEqual(params.internal_workers, 4)
        self.assertEqual(params.timeout, 600)

    def test_benchmark_params_empty_exclude_set(self):
        """Test BenchmarkParams with empty exclude set."""
        params = BenchmarkParams(
            target_pdb="1test",
            exclude_pdb_ids=set()
        )
        
        self.assertEqual(params.target_pdb, "1test")
        self.assertEqual(len(params.exclude_pdb_ids), 0)
        self.assertIsInstance(params.exclude_pdb_ids, set)


class TestBenchmarkResult(unittest.TestCase):
    """Test BenchmarkResult dataclass."""

    def test_successful_benchmark_result(self):
        """Test creation of successful benchmark result."""
        rmsd_data = {
            "combo": {"rmsd": 1.5, "score": 0.8},
            "shape": {"rmsd": 2.0, "score": 0.7}
        }
        
        result = BenchmarkResult(
            success=True,
            rmsd_values=rmsd_data,
            runtime=120.5,
            error=None,
            metadata={"test_info": "value"}
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.rmsd_values, rmsd_data)
        self.assertEqual(result.runtime, 120.5)
        self.assertIsNone(result.error)
        self.assertEqual(result.metadata["test_info"], "value")

    def test_failed_benchmark_result(self):
        """Test creation of failed benchmark result."""
        result = BenchmarkResult(
            success=False,
            rmsd_values={},
            runtime=30.0,
            error="Pipeline execution failed"
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.rmsd_values, {})
        self.assertEqual(result.runtime, 30.0)
        self.assertEqual(result.error, "Pipeline execution failed")
        self.assertIsNone(result.metadata)

    def test_benchmark_result_to_dict_success(self):
        """Test conversion of successful result to dictionary."""
        rmsd_data = {"combo": {"rmsd": 1.2}}
        result = BenchmarkResult(
            success=True,
            rmsd_values=rmsd_data,
            runtime=85.3,
            error=None
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertTrue(result_dict["success"])
        self.assertEqual(result_dict["rmsd_values"], rmsd_data)
        self.assertEqual(result_dict["runtime"], 85.3)
        self.assertIsNone(result_dict["error"])

    def test_benchmark_result_to_dict_failure(self):
        """Test conversion of failed result to dictionary."""
        result = BenchmarkResult(
            success=False,
            rmsd_values=None,
            runtime=10.0,
            error="Test error message"
        )
        
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertFalse(result_dict["success"])
        self.assertEqual(result_dict["rmsd_values"], {})
        self.assertEqual(result_dict["runtime"], 10.0)
        self.assertEqual(result_dict["error"], "Test error message")

    def test_benchmark_result_to_dict_serialization_error(self):
        """Test to_dict method handles serialization errors gracefully."""
        # Create a result with non-serializable data
        class NonSerializable:
            def __str__(self):
                raise Exception("Cannot serialize")
        
        result = BenchmarkResult(
            success=True,
            rmsd_values={"test": NonSerializable()},
            runtime=60.0,
            error=None
        )
        
        # The to_dict method should handle this gracefully
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertFalse(result_dict["success"])
        self.assertIn("Serialization failed", result_dict["error"])


class TestMemoryMonitoring(unittest.TestCase):
    """Test memory monitoring functionality."""

    @patch('templ_pipeline.benchmark.runner.PSUTIL_AVAILABLE', False)
    def test_monitor_memory_usage_no_psutil(self):
        """Test memory monitoring when psutil is not available."""
        result = monitor_memory_usage()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["memory_gb"], 0.0)
        self.assertFalse(result["warning"])
        self.assertFalse(result["critical"])

    @patch('templ_pipeline.benchmark.runner.PSUTIL_AVAILABLE', True)
    @patch('templ_pipeline.benchmark.runner.psutil')
    def test_monitor_memory_usage_normal(self, mock_psutil):
        """Test memory monitoring with normal memory usage."""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 2 * 1024**3  # 2GB
        mock_psutil.Process.return_value = mock_process
        
        result = monitor_memory_usage()
        
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(result["memory_gb"], 2.0, places=1)
        self.assertFalse(result["warning"])
        self.assertFalse(result["critical"])

    @patch('templ_pipeline.benchmark.runner.PSUTIL_AVAILABLE', True)
    @patch('templ_pipeline.benchmark.runner.psutil')
    def test_monitor_memory_usage_warning(self, mock_psutil):
        """Test memory monitoring with warning level usage."""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 7 * 1024**3  # 7GB
        mock_psutil.Process.return_value = mock_process
        
        result = monitor_memory_usage()
        
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(result["memory_gb"], 7.0, places=1)
        self.assertTrue(result["warning"])
        self.assertFalse(result["critical"])

    @patch('templ_pipeline.benchmark.runner.PSUTIL_AVAILABLE', True)
    @patch('templ_pipeline.benchmark.runner.psutil')
    def test_monitor_memory_usage_critical(self, mock_psutil):
        """Test memory monitoring with critical level usage."""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 9 * 1024**3  # 9GB
        mock_psutil.Process.return_value = mock_process
        
        result = monitor_memory_usage()
        
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(result["memory_gb"], 9.0, places=1)
        self.assertTrue(result["warning"])
        self.assertTrue(result["critical"])

    @patch('templ_pipeline.benchmark.runner.PSUTIL_AVAILABLE', True)
    @patch('templ_pipeline.benchmark.runner.psutil')
    def test_monitor_memory_usage_error(self, mock_psutil):
        """Test memory monitoring handles errors gracefully."""
        mock_psutil.Process.side_effect = Exception("Process error")
        
        result = monitor_memory_usage()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["memory_gb"], 0.0)
        self.assertFalse(result["warning"])
        self.assertFalse(result["critical"])


class TestBenchmarkRunner(unittest.TestCase):
    """Test benchmark runner functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.protein_file = Path(self.test_dir) / "test_protein.pdb"
        
        # Create minimal PDB file
        with open(self.protein_file, 'w') as f:
            f.write("ATOM      1  N   ALA A   1      20.154  16.967  18.274  1.00 16.77           N\n")
            f.write("ATOM      2  CA  ALA A   1      21.156  16.122  17.618  1.00 16.18           C\n")
            f.write("END\n")

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    @patch('templ_pipeline.benchmark.runner.TEMPLPipeline')
    @patch('templ_pipeline.benchmark.runner.get_protein_file_paths')
    @patch('templ_pipeline.benchmark.runner.find_ligand_by_pdb_id')
    def test_run_templ_pipeline_successful(self, mock_find_ligand, mock_get_protein, mock_pipeline_class):
        """Test successful benchmark run."""
        # Setup mocks
        mock_get_protein.return_value = [str(self.protein_file)]
        mock_ligand = Chem.MolFromSmiles("CCO")
        mock_find_ligand.return_value = mock_ligand
        
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_full_pipeline.return_value = {
            'poses': {'combo': (mock_ligand, {'combo_score': 0.8})},
            'output_file': 'test_output.sdf'
        }
        
        # Run benchmark with correct signature
        result = run_templ_pipeline_for_benchmark(
            target_pdb="1test",
            exclude_pdb_ids={"1abc"},
            timeout=60
        )
        
        # Verify result is a dictionary (not BenchmarkResult)
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertIn('runtime', result)

    @patch('templ_pipeline.benchmark.runner.get_protein_file_paths')
    def test_run_templ_pipeline_no_protein_file(self, mock_get_protein):
        """Test benchmark run when protein file is not found."""
        mock_get_protein.return_value = []
        
        result = run_templ_pipeline_for_benchmark(
            target_pdb="nonexistent",
            exclude_pdb_ids=set()
        )
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result.get('success', True))
        self.assertIn('error', result)

    @patch('templ_pipeline.benchmark.runner.get_protein_file_paths')
    @patch('templ_pipeline.benchmark.runner.find_ligand_by_pdb_id')
    def test_run_templ_pipeline_no_ligand(self, mock_find_ligand, mock_get_protein):
        """Test benchmark run when ligand is not found."""
        mock_get_protein.return_value = [str(self.protein_file)]
        mock_find_ligand.return_value = None
        
        result = run_templ_pipeline_for_benchmark(
            target_pdb="1test",
            exclude_pdb_ids=set()
        )
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result.get('success', True))
        self.assertIn('error', result)

    @patch('templ_pipeline.benchmark.runner.TEMPLPipeline')
    @patch('templ_pipeline.benchmark.runner.get_protein_file_paths')
    @patch('templ_pipeline.benchmark.runner.find_ligand_by_pdb_id')
    def test_run_templ_pipeline_execution_error(self, mock_find_ligand, mock_get_protein, mock_pipeline_class):
        """Test benchmark run when pipeline execution fails."""
        # Setup mocks
        mock_get_protein.return_value = [str(self.protein_file)]
        mock_ligand = Chem.MolFromSmiles("CCO")
        mock_find_ligand.return_value = mock_ligand
        
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_full_pipeline.side_effect = Exception("Pipeline failed")
        
        result = run_templ_pipeline_for_benchmark(
            target_pdb="1test",
            exclude_pdb_ids=set()
        )
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result.get('success', True))
        self.assertIn('error', result)

    @patch('templ_pipeline.benchmark.runner.TEMPLPipeline')
    @patch('templ_pipeline.benchmark.runner.get_protein_file_paths')
    @patch('templ_pipeline.benchmark.runner.find_ligand_by_pdb_id')
    @patch('templ_pipeline.benchmark.runner.calculate_rmsd')
    def test_run_templ_pipeline_with_rmsd_calculation(self, mock_calc_rmsd, mock_find_ligand, mock_get_protein, mock_pipeline_class):
        """Test benchmark run with RMSD calculation."""
        # Setup mocks
        mock_get_protein.return_value = [str(self.protein_file)]
        mock_ligand = Chem.MolFromSmiles("CCO")
        mock_find_ligand.return_value = mock_ligand
        
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        
        # Mock pipeline results with poses
        mock_pose_mol = Chem.MolFromSmiles("CCO")
        mock_pipeline.run_full_pipeline.return_value = {
            'poses': {
                'combo': (mock_pose_mol, {'combo_score': 0.8}),
                'shape': (mock_pose_mol, {'shape_score': 0.7})
            },
            'output_file': 'test_output.sdf'
        }
        
        # Mock RMSD calculation
        mock_calc_rmsd.return_value = 1.5
        
        result = run_templ_pipeline_for_benchmark(
            target_pdb="1test",
            exclude_pdb_ids=set()
        )
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get('success', False))
        self.assertIn('rmsd_values', result)

    def test_benchmark_function_interface(self):
        """Test that benchmark function has correct interface."""
        # Test with minimal required parameters
        try:
            result = run_templ_pipeline_for_benchmark(
                target_pdb="1test",
                exclude_pdb_ids=set()
            )
            # Should return a dictionary even if it fails
            self.assertIsInstance(result, dict)
        except Exception as e:
            # Function should exist and be callable
            self.assertIsInstance(e, (FileNotFoundError, ValueError, RuntimeError))

    def test_benchmark_result_runtime_tracking(self):
        """Test that benchmark result properly tracks runtime."""
        start_time = time.time()
        
        result = BenchmarkResult(
            success=True,
            rmsd_values={},
            runtime=0.0,
            error=None
        )
        
        # Simulate some work
        time.sleep(0.01)
        
        result.runtime = time.time() - start_time
        
        self.assertGreater(result.runtime, 0)
        self.assertLess(result.runtime, 1.0)  # Should be very quick


if __name__ == "__main__":
    unittest.main()