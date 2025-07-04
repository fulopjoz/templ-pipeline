#!/usr/bin/env python3
"""Simple test of data directory path resolution."""

from pathlib import Path

def test_polaris_paths():
    """Test polaris data directory discovery."""
    print("Testing Polaris data paths...")
    
    potential_paths = [
        Path(__file__).resolve().parent / "data" / "polaris",
        Path.cwd() / "data" / "polaris", 
        Path("data") / "polaris",
    ]
    
    for path in potential_paths:
        if path.exists() and (path / "train_sarsmols.sdf").exists():
            print(f"✓ Found polaris data at: {path}")
            return True
    
    print("✗ Polaris data not found")
    return False

def test_timesplit_paths():
    """Test timesplit data directory discovery.""" 
    print("Testing Timesplit data paths...")
    
    potential_data_dirs = [
        Path(__file__).resolve().parent / "data",
        Path.cwd() / "data",
        Path("data"),
    ]
    
    for candidate_path in potential_data_dirs:
        if (candidate_path.exists() and 
            (candidate_path / "ligands" / "processed_ligands_new.sdf.gz").exists()):
            print(f"✓ Found TEMPL data directory at: {candidate_path}")
            
            # Check split files
            splits_dir = candidate_path / "splits"
            if splits_dir.exists():
                split_files = list(splits_dir.glob("timesplit_*"))
                print(f"✓ Found {len(split_files)} split files")
                return True
    
    print("✗ Timesplit data not found")
    return False

if __name__ == "__main__":
    print("Simple Path Resolution Test")
    print("=" * 30)
    
    polaris_ok = test_polaris_paths()
    timesplit_ok = test_timesplit_paths()
    
    print("\n" + "=" * 30)
    if polaris_ok and timesplit_ok:
        print("✓ All paths resolved successfully!")
        print("\nThe benchmark path resolution should now work correctly.")
    else:
        if not polaris_ok:
            print("✗ Polaris path resolution failed") 
        if not timesplit_ok:
            print("✗ Timesplit path resolution failed")