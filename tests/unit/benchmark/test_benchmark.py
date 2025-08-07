"""
Comprehensive tests for the benchmark module.

Tests cover benchmark execution, error handling, progress tracking,
and edge cases for the TEMPL benchmark system.
"""

import unittest
import tempfile
import shutil
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import logging
from datetime import datetime

try:
    from templ_pipeline.benchmark.runner import BenchmarkRunner
    from templ_pipeline.benchmark.error_tracking import BenchmarkErrorTracker
    from templ_pipeline.benchmark.summary_generator import BenchmarkSummaryGenerator
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from templ_pipeline.benchmark.runner import BenchmarkRunner
    from templ_pipeline.benchmark.error_tracking import BenchmarkErrorTracker  
    from templ_pipeline.benchmark.summary_generator import BenchmarkSummaryGenerator

# Alias for backward compatibility
Benchmark = BenchmarkRunner


class TestBenchmark(unittest.TestCase):
    """Test main Benchmark class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Use centralized output structure
        from templ_pipeline.core.workspace_manager import DirectoryManager
        self._dir_manager = DirectoryManager(
            base_name="benchmark_test",
            auto_cleanup=True,
            centralized_output=True,
            output_root=str(Path(self.test_dir) / "output")
        )
        self.output_dir = self._dir_manager.directory
        
        # Use standardized benchmark configuration
        try:
            from tests.fixtures.benchmark_fixtures import create_benchmark_test_data
            self.config = {
                'name': 'test_benchmark',
                'description': 'Test benchmark for unit testing',
                'num_workers': 2,
                'timeout': 60,
                'output_format': 'json'
            }
        except ImportError:
            # Fallback configuration
            self.config = {
                'name': 'test_benchmark',
                'description': 'Test benchmark for unit testing',
                'num_workers': 2,
                'timeout': 60,
                'output_format': 'json'
            }
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, '_dir_manager'):
            self._dir_manager.cleanup()
        shutil.rmtree(self.test_dir)
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization with various configurations."""
        # Test basic initialization
        benchmark = Benchmark(
            name="test_benchmark",
            output_dir=str(self.output_dir)
        )
        
        self.assertEqual(benchmark.name, "test_benchmark")
        self.assertIsNotNone(benchmark.output_dir)
        self.assertIsInstance(benchmark.error_tracker, BenchmarkErrorTracker)
    
    def test_benchmark_initialization_with_config(self):
        """Test benchmark initialization with full configuration."""
        benchmark = Benchmark(
            name="configured_benchmark",
            output_dir=str(self.output_dir),
            config=self.config
        )
        
        self.assertEqual(benchmark.name, "configured_benchmark")
        self.assertEqual(benchmark.config['num_workers'], 2)
        self.assertEqual(benchmark.config['timeout'], 60)
    
    def test_benchmark_initialization_defaults(self):
        """Test benchmark initialization with default values."""
        benchmark = Benchmark(name="default_benchmark")
        
        self.assertEqual(benchmark.name, "default_benchmark")
        self.assertIsNotNone(benchmark.output_dir)
        self.assertIsNotNone(benchmark.config)
    
    @patch('templ_pipeline.core.pipeline.TEMPLPipeline')
    def test_run_single_target_success(self, mock_pipeline_class):
        """Test successful execution of single benchmark target."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_full_pipeline.return_value = {
            'poses': {'combo': (Mock(), {'combo_score': 0.8})},
            'mcs_info': {'smarts': 'CCO'},
            'templates': [('1abc', 0.9)],
            'output_file': 'test_output.sdf'
        }
        
        benchmark = Benchmark(
            name="single_target_test",
            output_dir=str(self.output_dir)
        )
        
        target_data = {
            'pdb_id': '1abc',
            'protein_file': 'protein.pdb',
            'ligand_smiles': 'CCO'
        }
        
        result = benchmark.run_single_target(target_data)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['pdb_id'], '1abc')
        self.assertIn('runtime', result)
        self.assertIn('poses', result)
        mock_pipeline.run_full_pipeline.assert_called_once()
    
    @patch('templ_pipeline.core.pipeline.TEMPLPipeline')
    def test_run_single_target_pipeline_failure(self, mock_pipeline_class):
        """Test handling of pipeline failures in single target execution."""
        # Setup mock pipeline to raise exception
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_full_pipeline.side_effect = ValueError("Pipeline failed")
        
        benchmark = Benchmark(
            name="failure_test",
            output_dir=str(self.output_dir)
        )
        
        target_data = {
            'pdb_id': '1abc',
            'protein_file': 'protein.pdb',
            'ligand_smiles': 'CCO'
        }
        
        result = benchmark.run_single_target(target_data)
        
        self.assertFalse(result['success'])
        self.assertEqual(result['pdb_id'], '1abc')
        self.assertIn('error', result)
        self.assertIn('Pipeline failed', result['error'])
    
    @patch('templ_pipeline.core.pipeline.TEMPLPipeline')
    def test_run_single_target_timeout(self, mock_pipeline_class):
        """Test timeout handling in single target execution."""
        # Setup mock pipeline to simulate timeout
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_full_pipeline.side_effect = TimeoutError("Execution timed out")
        
        benchmark = Benchmark(
            name="timeout_test",
            output_dir=str(self.output_dir),
            config={'timeout': 1}
        )
        
        target_data = {
            'pdb_id': '1abc',
            'protein_file': 'protein.pdb',
            'ligand_smiles': 'CCO'
        }
        
        result = benchmark.run_single_target(target_data)
        
        self.assertFalse(result['success'])
        self.assertIn('timeout', result['error'].lower())
    
    def test_process_benchmark_results_success(self):
        """Test processing of successful benchmark results."""
        benchmark = Benchmark(
            name="results_test",
            output_dir=str(self.output_dir)
        )
        
        raw_results = [
            {
                'success': True,
                'pdb_id': '1abc',
                'runtime': 30.5,
                'poses': {'combo': (Mock(), {'combo_score': 0.8})},
                'rmsd_values': {'combo': {'rmsd': 1.2}}
            },
            {
                'success': False,
                'pdb_id': '2def',
                'error': 'Pipeline failed',
                'runtime': 5.0
            }
        ]
        
        processed = benchmark.process_results(raw_results)
        
        self.assertIn('summary', processed)
        self.assertIn('detailed_results', processed)
        self.assertEqual(processed['summary']['total_targets'], 2)
        self.assertEqual(processed['summary']['successful_runs'], 1)
        self.assertEqual(processed['summary']['failed_runs'], 1)
    
    def test_process_benchmark_results_empty(self):
        """Test processing of empty benchmark results."""
        benchmark = Benchmark(
            name="empty_results_test",
            output_dir=str(self.output_dir)
        )
        
        processed = benchmark.process_results([])
        
        self.assertIn('summary', processed)
        self.assertEqual(processed['summary']['total_targets'], 0)
        self.assertEqual(processed['summary']['successful_runs'], 0)
        self.assertEqual(processed['summary']['failed_runs'], 0)
    
    def test_save_benchmark_results(self):
        """Test saving benchmark results to file."""
        benchmark = Benchmark(
            name="save_test",
            output_dir=str(self.output_dir)
        )
        
        results_data = {
            'summary': {
                'total_targets': 1,
                'successful_runs': 1,
                'failed_runs': 0
            },
            'detailed_results': [
                {
                    'pdb_id': '1abc',
                    'success': True,
                    'runtime': 30.0
                }
            ]
        }
        
        output_file = benchmark.save_results(results_data)
        
        self.assertIsNotNone(output_file)
        self.assertTrue(Path(output_file).exists())
        
        # Verify saved content
        with open(output_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data['summary']['total_targets'], 1)
        self.assertEqual(len(saved_data['detailed_results']), 1)


class TestBenchmarkErrorHandling(unittest.TestCase):
    """Test error handling and edge cases in benchmark execution."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Use centralized output structure for error tests
        from templ_pipeline.core.workspace_manager import DirectoryManager
        self._dir_manager = DirectoryManager(
            base_name="error_test",
            auto_cleanup=True,
            centralized_output=True,
            output_root=str(Path(self.test_dir) / "output")
        )
        self.output_dir = self._dir_manager.directory
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, '_dir_manager'):
            self._dir_manager.cleanup()
        shutil.rmtree(self.test_dir)
    
    def test_benchmark_invalid_configuration(self):
        """Test benchmark initialization with invalid configuration."""
        # Test with invalid num_workers
        with self.assertRaises((ValueError, TypeError)):
            Benchmark(
                name="invalid_config",
                config={'num_workers': -1}
            )
    
    def test_benchmark_nonexistent_output_directory(self):
        """Test benchmark with non-existent output directory."""
        nonexistent_dir = self.test_dir / "nonexistent"
        
        # Should create directory or handle gracefully
        benchmark = Benchmark(
            name="nonexistent_dir_test",
            output_dir=str(nonexistent_dir)
        )
        
        self.assertIsNotNone(benchmark.output_dir)
    
    @patch('templ_pipeline.core.pipeline.TEMPLPipeline')
    def test_run_single_target_missing_protein_file(self, mock_pipeline_class):
        """Test single target execution with missing protein file."""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_full_pipeline.side_effect = FileNotFoundError("Protein file not found")
        
        benchmark = Benchmark(
            name="missing_file_test",
            output_dir=str(self.output_dir)
        )
        
        target_data = {
            'pdb_id': '1abc',
            'protein_file': 'nonexistent.pdb',
            'ligand_smiles': 'CCO'
        }
        
        result = benchmark.run_single_target(target_data)
        
        self.assertFalse(result['success'])
        self.assertIn('not found', result['error'].lower())
    
    @patch('templ_pipeline.core.pipeline.TEMPLPipeline')
    def test_run_single_target_invalid_smiles(self, mock_pipeline_class):
        """Test single target execution with invalid SMILES."""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_full_pipeline.side_effect = ValueError("Invalid SMILES")
        
        benchmark = Benchmark(
            name="invalid_smiles_test",
            output_dir=str(self.output_dir)
        )
        
        target_data = {
            'pdb_id': '1abc',
            'protein_file': 'protein.pdb',
            'ligand_smiles': 'INVALID_SMILES'
        }
        
        result = benchmark.run_single_target(target_data)
        
        self.assertFalse(result['success'])
        self.assertIn('invalid', result['error'].lower())
    
    def test_run_single_target_missing_data_fields(self):
        """Test single target execution with missing required data fields."""
        benchmark = Benchmark(
            name="missing_fields_test",
            output_dir=str(self.output_dir)
        )
        
        # Missing protein_file
        incomplete_data = {
            'pdb_id': '1abc',
            'ligand_smiles': 'CCO'
        }
        
        result = benchmark.run_single_target(incomplete_data)
        
        self.assertFalse(result['success'])
        self.assertIn('error', result)
    
    def test_run_single_target_none_data(self):
        """Test single target execution with None data."""
        benchmark = Benchmark(
            name="none_data_test",
            output_dir=str(self.output_dir)
        )
        
        result = benchmark.run_single_target(None)
        
        self.assertFalse(result['success'])
        self.assertIn('error', result)
    
    def test_run_single_target_empty_data(self):
        """Test single target execution with empty data."""
        benchmark = Benchmark(
            name="empty_data_test",
            output_dir=str(self.output_dir)
        )
        
        result = benchmark.run_single_target({})
        
        self.assertFalse(result['success'])
        self.assertIn('error', result)
    
    def test_save_results_invalid_data(self):
        """Test saving results with invalid data."""
        benchmark = Benchmark(
            name="invalid_save_test",
            output_dir=str(self.output_dir)
        )
        
        # Test with None data
        result = benchmark.save_results(None)
        self.assertIsNone(result)
        
        # Test with empty data
        result = benchmark.save_results({})
        # Should handle gracefully or return None
        self.assertIsInstance(result, (str, type(None)))
    
    def test_save_results_permission_error(self):
        """Test saving results with permission error."""
        benchmark = Benchmark(
            name="permission_test",
            output_dir="/root/forbidden"  # Directory with no write permission
        )
        
        results_data = {
            'summary': {'total_targets': 1},
            'detailed_results': []
        }
        
        # Should handle permission error gracefully
        with self.assertRaises((PermissionError, OSError)):
            benchmark.save_results(results_data)


class TestBenchmarkEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions in benchmark execution."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Use centralized output structure for edge case tests
        from templ_pipeline.core.workspace_manager import DirectoryManager
        self._dir_manager = DirectoryManager(
            base_name="edge_case_test",
            auto_cleanup=True,
            centralized_output=True,
            output_root=str(Path(self.test_dir) / "output")
        )
        self.output_dir = self._dir_manager.directory
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, '_dir_manager'):
            self._dir_manager.cleanup()
        shutil.rmtree(self.test_dir)
    
    def test_benchmark_very_long_name(self):
        """Test benchmark with very long name."""
        long_name = "benchmark_" + "x" * 1000
        
        benchmark = Benchmark(
            name=long_name,
            output_dir=str(self.output_dir)
        )
        
        # Should handle long names gracefully
        self.assertIsNotNone(benchmark.name)
    
    def test_benchmark_special_characters_in_name(self):
        """Test benchmark with special characters in name."""
        special_name = "benchmark-test_v1.0/2024"
        
        benchmark = Benchmark(
            name=special_name,
            output_dir=str(self.output_dir)
        )
        
        # Should handle or sanitize special characters
        self.assertIsNotNone(benchmark.name)
    
    def test_benchmark_unicode_name(self):
        """Test benchmark with Unicode characters in name."""
        unicode_name = "基准测试_🧪_αβγ"
        
        benchmark = Benchmark(
            name=unicode_name,
            output_dir=str(self.output_dir)
        )
        
        # Should handle Unicode gracefully
        self.assertIsNotNone(benchmark.name)
    
    @patch('templ_pipeline.core.pipeline.TEMPLPipeline')
    def test_run_single_target_zero_runtime(self, mock_pipeline_class):
        """Test single target with instantaneous execution."""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_full_pipeline.return_value = {
            'poses': {},
            'mcs_info': {},
            'templates': [],
            'output_file': None
        }
        
        benchmark = Benchmark(
            name="zero_runtime_test",
            output_dir=str(self.output_dir)
        )
        
        target_data = {
            'pdb_id': '1abc',
            'protein_file': 'protein.pdb',
            'ligand_smiles': 'C'  # Minimal molecule
        }
        
        result = benchmark.run_single_target(target_data)
        
        self.assertTrue(result['success'])
        self.assertGreaterEqual(result['runtime'], 0)
    
    def test_process_results_very_large_dataset(self):
        """Test processing results with very large dataset."""
        benchmark = Benchmark(
            name="large_dataset_test",
            output_dir=str(self.output_dir)
        )
        
        # Generate large result set
        large_results = []
        for i in range(10000):
            large_results.append({
                'success': i % 10 != 0,  # 90% success rate
                'pdb_id': f'pdb_{i:04d}',
                'runtime': float(i % 100),
                'error': 'Test error' if i % 10 == 0 else None
            })
        
        processed = benchmark.process_results(large_results)
        
        self.assertEqual(processed['summary']['total_targets'], 10000)
        self.assertEqual(processed['summary']['successful_runs'], 9000)
        self.assertEqual(processed['summary']['failed_runs'], 1000)
    
    @patch('templ_pipeline.core.pipeline.TEMPLPipeline')
    def test_run_single_target_extremely_long_smiles(self, mock_pipeline_class):
        """Test single target with extremely long SMILES string."""
        mock_pipeline = Mock()
        mock_pipeline_class.return_value = mock_pipeline
        mock_pipeline.run_full_pipeline.return_value = {
            'poses': {'combo': (Mock(), {'combo_score': 0.5})},
            'mcs_info': {'smarts': 'C'},
            'templates': [],
            'output_file': 'output.sdf'
        }
        
        benchmark = Benchmark(
            name="long_smiles_test",
            output_dir=str(self.output_dir)
        )
        
        # Generate very long SMILES
        long_smiles = "C" + "C" * 10000
        
        target_data = {
            'pdb_id': '1abc',
            'protein_file': 'protein.pdb',
            'ligand_smiles': long_smiles
        }
        
        result = benchmark.run_single_target(target_data)
        
        # Should handle long SMILES appropriately
        self.assertIsNotNone(result)
    
    def test_benchmark_zero_worker_config(self):
        """Test benchmark with zero workers configuration."""
        config = {'num_workers': 0}
        
        # Should handle zero workers gracefully or raise appropriate error
        try:
            benchmark = Benchmark(
                name="zero_workers_test",
                output_dir=str(self.output_dir),
                config=config
            )
            # If no exception, workers should be set to a reasonable default
            self.assertGreater(benchmark.config.get('num_workers', 1), 0)
        except (ValueError, AssertionError):
            # Appropriate error for invalid configuration
            pass
    
    def test_benchmark_huge_timeout_config(self):
        """Test benchmark with extremely large timeout."""
        config = {'timeout': 999999999}  # Very large timeout
        
        benchmark = Benchmark(
            name="huge_timeout_test",
            output_dir=str(self.output_dir),
            config=config
        )
        
        # Should handle large timeout values
        self.assertEqual(benchmark.config['timeout'], 999999999)
    
    def test_save_results_with_complex_data_types(self):
        """Test saving results with complex data types."""
        benchmark = Benchmark(
            name="complex_data_test",
            output_dir=str(self.output_dir)
        )
        
        complex_results = {
            'summary': {
                'total_targets': 1,
                'timestamp': datetime.now(),  # Complex type
                'nested_dict': {'deep': {'nesting': True}},
                'list_of_dicts': [{'a': 1}, {'b': 2}]
            },
            'detailed_results': [
                {
                    'pdb_id': '1abc',
                    'success': True,
                    'complex_data': {
                        'poses': [Mock(), Mock()],  # Non-serializable objects
                        'arrays': [1, 2, 3, 4, 5] * 1000  # Large array
                    }
                }
            ]
        }
        
        # Should handle complex data gracefully (may serialize or skip non-serializable)
        try:
            output_file = benchmark.save_results(complex_results)
            if output_file:
                self.assertTrue(Path(output_file).exists())
        except (TypeError, ValueError):
            # Acceptable if non-serializable data causes error
            pass


if __name__ == "__main__":
    unittest.main()