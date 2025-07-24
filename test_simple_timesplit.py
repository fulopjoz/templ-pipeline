#!/usr/bin/env python3
"""
Test script for the simplified timesplit runner.
Tests the approach with a few sample PDBs to verify it works correctly.
"""

import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from templ_pipeline.benchmark.timesplit.simple_runner import SimpleTimeSplitRunner

def test_simple_runner():
    """Test the simplified timesplit runner with a few samples."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Testing SimpleTimeSplitRunner...")
    
    # Initialize runner
    runner = SimpleTimeSplitRunner(
        data_dir="data",
        results_dir="test_simple_timesplit_results"
    )
    
    # Test with a few PDBs from each split
    test_pdbs = {
        "test": ["6qqw", "6d08"],  # First 2 from test split
        "val": ["4lp9", "1me7"],   # First 2 from val split  
        "train": ["3dpf", "2zy1"]  # First 2 from train split
    }
    
    logger.info("Testing individual target processing...")
    
    for split_name, pdbs in test_pdbs.items():
        logger.info(f"\nTesting {split_name} split...")
        
        for pdb_id in pdbs:
            logger.info(f"Processing {pdb_id}...")
            
            # Get allowed templates for this split and PDB
            allowed_templates = runner.get_allowed_templates_for_split(split_name, pdb_id)
            logger.info(f"  Allowed templates: {len(allowed_templates)}")
            
            # Test subprocess call (with reduced conformers for speed)
            result = runner.run_single_target_subprocess(
                target_pdb=pdb_id,
                allowed_templates=allowed_templates,
                n_conformers=10,  # Reduced for testing
                timeout=300       # 5 minute timeout for testing
            )
            
            logger.info(f"  Result: {'SUCCESS' if result.get('success') else 'FAILED'}")
            if not result.get('success'):
                logger.error(f"    Error: {result.get('error', 'Unknown error')}")
                if 'stderr' in result:
                    logger.error(f"    Stderr: {result['stderr'][:200]}...")
            else:
                logger.info(f"    Runtime: {result.get('runtime_total', 0):.1f}s")
            
            # Test just one PDB per split for initial validation
            break
    
    logger.info("\nTesting split benchmark with limited PDBs...")
    
    # Test running a small benchmark
    summary = runner.run_split_benchmark(
        split_name="test",
        n_workers=2,
        n_conformers=10,
        max_pdbs=2,  # Test with just 2 PDBs
        timeout=300
    )
    
    logger.info(f"Benchmark summary:")
    logger.info(f"  Total targets: {summary['total_targets']}")
    logger.info(f"  Processed: {summary['processed']}")
    logger.info(f"  Success rate: {summary['success_rate']:.1f}%")
    logger.info(f"  Results file: {summary['results_file']}")
    
    logger.info("\nTest completed!")

if __name__ == "__main__":
    test_simple_runner()