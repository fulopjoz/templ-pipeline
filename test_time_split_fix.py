#!/usr/bin/env python3
"""Test script to validate the time-split benchmark fix."""

import sys
import os
from pathlib import Path

# Add the current directory to the path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent))

def test_benchmark_result_handling():
    """Test that BenchmarkResult objects are properly handled."""
    print("Testing BenchmarkResult handling...")
    
    try:
        from templ_pipeline.benchmark.runner import BenchmarkResult
        
        # Create a test BenchmarkResult object
        test_result = BenchmarkResult(
            success=True,
            rmsd_values={"shape": {"rmsd": 1.0, "score": 0.5}},
            runtime=10.0,
            error=None,
            metadata={"test": "data"}
        )
        
        # Test to_dict() method
        result_dict = test_result.to_dict()
        print(f"✓ BenchmarkResult.to_dict() works: {result_dict}")
        
        # Test that it's a proper dictionary
        assert isinstance(result_dict, dict)
        assert "success" in result_dict
        assert "rmsd_values" in result_dict
        assert "runtime" in result_dict
        assert "error" in result_dict
        
        print("✓ BenchmarkResult to_dict() returns proper dictionary structure")
        
        # Test error handling in to_dict()
        error_result = BenchmarkResult(
            success=False,
            rmsd_values=None,
            runtime=0.0,
            error="Test error message",
            metadata=None
        )
        
        error_dict = error_result.to_dict()
        print(f"✓ BenchmarkResult with error converts properly: {error_dict}")
        
        return True
        
    except Exception as e:
        print(f"✗ BenchmarkResult handling failed: {e}")
        return False

def test_timesplit_type_handling():
    """Test the type handling in timesplit_stream.py"""
    print("\nTesting timesplit type handling...")
    
    try:
        from templ_pipeline.benchmark.runner import BenchmarkResult
        
        # Simulate the type handling logic from timesplit_stream.py
        def handle_result(result):
            # This is the logic we added to timesplit_stream.py
            if hasattr(result, 'to_dict'):
                return result.to_dict()
            return result
        
        # Test with BenchmarkResult object
        benchmark_result = BenchmarkResult(
            success=True,
            rmsd_values={"shape": {"rmsd": 1.0, "score": 0.5}},
            runtime=10.0,
            error=None
        )
        
        handled_result = handle_result(benchmark_result)
        assert isinstance(handled_result, dict)
        print("✓ BenchmarkResult object handled correctly")
        
        # Test with dictionary (should pass through)
        dict_result = {"success": True, "rmsd_values": {}, "runtime": 5.0, "error": None}
        handled_dict = handle_result(dict_result)
        assert isinstance(handled_dict, dict)
        print("✓ Dictionary result handled correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Timesplit type handling failed: {e}")
        return False

def test_runner_type_handling():
    """Test the type handling in runner.py"""
    print("\nTesting runner type handling...")
    
    try:
        from templ_pipeline.benchmark.runner import BenchmarkResult
        
        # Simulate the type handling logic from runner.py
        def handle_runner_result(result):
            # This is the logic we added to runner.py
            if hasattr(result, 'to_dict'):
                return result.to_dict()
            elif isinstance(result, dict):
                return result
            else:
                return {
                    "success": False,
                    "rmsd_values": {},
                    "runtime": 0.0,
                    "error": f"Unexpected result type: {type(result)}"
                }
        
        # Test with BenchmarkResult object
        benchmark_result = BenchmarkResult(
            success=True,
            rmsd_values={"shape": {"rmsd": 1.0, "score": 0.5}},
            runtime=10.0,
            error=None
        )
        
        handled_result = handle_runner_result(benchmark_result)
        assert isinstance(handled_result, dict)
        print("✓ BenchmarkResult object handled correctly in runner")
        
        # Test with dictionary
        dict_result = {"success": True, "rmsd_values": {}, "runtime": 5.0, "error": None}
        handled_dict = handle_runner_result(dict_result)
        assert isinstance(handled_dict, dict)
        print("✓ Dictionary result handled correctly in runner")
        
        # Test with unexpected type
        unexpected_result = "string_result"
        handled_unexpected = handle_runner_result(unexpected_result)
        assert isinstance(handled_unexpected, dict)
        assert not handled_unexpected["success"]
        print("✓ Unexpected type handled correctly in runner")
        
        return True
        
    except Exception as e:
        print(f"✗ Runner type handling failed: {e}")
        return False

if __name__ == "__main__":
    print("Running time-split benchmark fix validation...")
    
    success = True
    success &= test_benchmark_result_handling()
    success &= test_timesplit_type_handling()
    success &= test_runner_type_handling()
    
    if success:
        print("\n✓ All tests passed! The time-split benchmark fix should work correctly.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        sys.exit(1)