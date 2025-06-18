#!/usr/bin/env python3
"""Test script to verify max_ram_gb parameter works correctly."""

from templ_pipeline.benchmark.timesplit import run_timesplit_benchmark

print('Testing max_ram_gb parameter...')

# This should not raise an error
try:
    result = run_timesplit_benchmark(
        n_workers=1,  
        max_pdbs=0,  # Process 0 PDBs to test function signature only
        max_ram_gb=5.0,  # Test the parameter
        splits_to_run=["train"],
        quiet=True
    )
    print('✓ max_ram_gb parameter accepted successfully')
    print(f'  Result type: {type(result)}')
    print(f'  Keys: {list(result.keys()) if isinstance(result, dict) else "Not a dict"}')
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc() 