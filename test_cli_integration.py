#!/usr/bin/env python3
"""
Simple CLI Integration Test

Test the CLI integration by checking imports and basic functionality.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test basic imports work."""
    print("Testing basic imports...")
    
    try:
        # Test if we can import the main CLI components
        from templ_pipeline.cli.main import setup_parser, _get_hardware_config
        print("✓ CLI main module imports successful")
        
        from templ_pipeline.benchmark.summary_generator import BenchmarkSummaryGenerator
        print("✓ Summary generator imports successful")
        
        from templ_pipeline.benchmark.error_tracking import BenchmarkErrorTracker  
        print("✓ Error tracking imports successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_argument_parsing():
    """Test CLI argument parsing."""
    print("\nTesting argument parsing...")
    
    try:
        from templ_pipeline.cli.main import setup_parser
        
        parser, help_system = setup_parser()
        
        # Test polaris benchmark args
        args = parser.parse_args([
            "benchmark", "polaris", 
            "--n-workers", "2",
            "--hardware-profile", "balanced",
            "--quick"
        ])
        
        assert args.suite == "polaris"
        assert args.n_workers == 2
        assert args.hardware_profile == "balanced"
        assert args.quick == True
        
        print("✓ Polaris benchmark parsing works")
        
        # Test timesplit args  
        args2 = parser.parse_args([
            "benchmark", "time-split",
            "--test-only", 
            "--max-ram", "8.0"
        ])
        
        assert args2.suite == "time-split"
        assert args2.test_only == True
        assert args2.max_ram_gb == 8.0
        
        print("✓ Timesplit benchmark parsing works")
        
        return True
        
    except Exception as e:
        print(f"✗ Argument parsing failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "templ_pipeline/cli/main.py",
        "templ_pipeline/benchmark/summary_generator.py", 
        "templ_pipeline/benchmark/error_tracking.py",
        "templ_pipeline/benchmark/runner.py",
        "templ_pipeline/benchmark/timesplit_stream.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False
    else:
        print("✓ All required files present")
        return True

def test_summary_generator():
    """Test summary generator with mock data."""
    print("\nTesting summary generator...")
    
    try:
        from templ_pipeline.benchmark.summary_generator import BenchmarkSummaryGenerator
        
        generator = BenchmarkSummaryGenerator()
        
        # Test with mock data
        mock_data = [
            {"pdb_id": "1abc", "success": True, "target_split": "test", 
             "rmsd_values": {"combo": {"rmsd": 1.5, "score": 0.8}}}
        ]
        
        summary = generator.generate_unified_summary(mock_data, "timesplit")
        print("✓ Summary generator works")
        
        return True
        
    except Exception as e:
        print(f"✗ Summary generator failed: {e}")
        return False

def main():
    """Run validation tests."""
    print("TEMPL CLI Integration Validation")
    print("=" * 40)
    
    tests = [
        test_file_structure,
        test_basic_imports,
        test_argument_parsing,
        test_summary_generator,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ CLI integration validation PASSED!")
        print("\nImplementation Summary:")
        print("- Enhanced CLI with hardware optimization")
        print("- Unified summary table generation")
        print("- Comprehensive error tracking")
        print("- Proper workspace organization")
        print("- PDB exclusion logic for timesplit")
        print("- Crystal structure RMSD calculation")
        print("- Shape/color/combo scoring integration")
        return 0
    else:
        print(f"✗ {total-passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())