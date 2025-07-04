#!/usr/bin/env python3
"""Test path resolution for benchmark data directories."""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_polaris_path_resolution():
    """Test that polaris data can be found."""
    print("Testing Polaris path resolution...")
    
    try:
        # Import the polaris benchmark module
        from templ_pipeline.benchmark.polaris.benchmark import resolve_dataset_paths
        
        # Test the path resolution logic from the polaris benchmark
        potential_paths = [
            # From benchmark file location: go up to project root then to data
            Path(__file__).resolve().parent / "data" / "polaris",
            # From current working directory
            Path.cwd() / "data" / "polaris", 
            # Relative to templ_pipeline directory
            Path.cwd() / "templ_pipeline" / "data" / "polaris",
            # If running from project root
            Path("data") / "polaris",
            # If running from templ_pipeline subdirectory
            Path("..") / "data" / "polaris",
        ]
        
        found_path = None
        for path in potential_paths:
            if path.exists() and (path / "train_sarsmols.sdf").exists():
                found_path = path
                print(f"✓ Found polaris data at: {found_path}")
                break
        
        if found_path is None:
            print("✗ Polaris data not found in any expected location")
            for path in potential_paths:
                print(f"  Tried: {path} - exists: {path.exists()}")
            return False
        
        # Test that all required files exist
        required_files = [
            "train_sarsmols.sdf",
            "train_mersmols.sdf", 
            "train_sarsmols_aligned_to_mers.sdf",
            "test_poses_with_properties.sdf"
        ]
        
        missing_files = []
        for file_name in required_files:
            if not (found_path / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"✗ Missing polaris files: {missing_files}")
            return False
        else:
            print("✓ All polaris files found")
            return True
            
    except Exception as e:
        print(f"✗ Polaris path resolution failed: {e}")
        return False

def test_timesplit_path_resolution():
    """Test that timesplit data can be found."""
    print("\nTesting Timesplit path resolution...")
    
    try:
        # Test the data directory resolution
        potential_data_dirs = [
            # From current file location: go up to find data directory
            Path(__file__).resolve().parent / "data",
            # From current working directory
            Path.cwd() / "data",
            # If running from templ_pipeline subdirectory
            Path.cwd() / "templ_pipeline" / "data",
            # Relative paths
            Path("data"),
            Path("..") / "data",
        ]
        
        data_dir = None
        for candidate_path in potential_data_dirs:
            # Check if this looks like a TEMPL data directory by looking for key files
            if (candidate_path.exists() and 
                (candidate_path / "ligands" / "processed_ligands_new.sdf.gz").exists()):
                data_dir = candidate_path
                print(f"✓ Found TEMPL data directory at: {data_dir}")
                break
        
        if data_dir is None:
            print("✗ TEMPL data directory not found")
            for path in potential_data_dirs:
                ligand_file = path / "ligands" / "processed_ligands_new.sdf.gz"
                print(f"  Tried: {path} - exists: {path.exists()}, ligand file: {ligand_file.exists()}")
            return False
        
        # Test split files
        split_names = ["train", "val", "test"]
        potential_split_paths = [
            data_dir / "splits",
            # From current working directory
            Path.cwd() / "data" / "splits",
            # Relative paths
            Path("data") / "splits",
        ]
        
        splits_found = 0
        for split_name in split_names:
            for split_dir in potential_split_paths:
                split_file = split_dir / f"timesplit_{split_name}"
                if split_file.exists():
                    print(f"✓ Found split file: {split_file}")
                    splits_found += 1
                    break
        
        if splits_found == len(split_names):
            print("✓ All timesplit files found")
            return True
        else:
            print(f"✗ Only found {splits_found}/{len(split_names)} split files")
            return False
            
    except Exception as e:
        print(f"✗ Timesplit path resolution failed: {e}")
        return False

def main():
    """Run path resolution tests."""
    print("TEMPL Benchmark Path Resolution Test")
    print("=" * 40)
    
    tests = [
        test_polaris_path_resolution,
        test_timesplit_path_resolution,
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
    
    print(f"Path resolution tests: {passed}/{total} passed")
    
    if passed == total:
        print("✓ Path resolution is working correctly!")
        print("\nYou can now run benchmarks with:")
        print("  python -m templ_pipeline.cli.main benchmark polaris --quick")
        print("  python -m templ_pipeline.cli.main benchmark time-split --test-only --max-pdbs 5")
        return 0
    else:
        print(f"✗ {total-passed} path resolution tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())