#!/usr/bin/env python3
"""
Basic test for the simplified timesplit runner - just test the core functionality.
"""

import logging
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

from templ_pipeline.benchmark.timesplit.simple_runner import SimpleTimeSplitRunner

def test_basic_functionality():
    """Test basic functionality of the simplified runner."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Testing basic SimpleTimeSplitRunner functionality...")
    
    # Initialize runner
    runner = SimpleTimeSplitRunner(
        data_dir="data",
        results_dir="test_basic_simple_timesplit_results"
    )
    
    # Test split loading worked
    logger.info(f"Loaded splits successfully:")
    logger.info(f"  Train: {len(runner.train_pdbs)} PDBs")
    logger.info(f"  Val: {len(runner.val_pdbs)} PDBs") 
    logger.info(f"  Test: {len(runner.test_pdbs)} PDBs")
    
    # Test a few individual functions
    test_pdbs = ["6qqw", "4lp9", "3dpf"]  # One from each split
    
    for pdb_id in test_pdbs:
        try:
            # Test split determination
            split = runner.determine_split(pdb_id)
            logger.info(f"{pdb_id} is in {split} split")
            
            # Test allowed templates calculation
            allowed = runner.get_allowed_templates_for_split(split, pdb_id)
            logger.info(f"  {pdb_id} ({split}): {len(allowed)} allowed templates")
            
            # Test ligand loading (this may take time but is important)
            logger.info(f"  Testing ligand loading for {pdb_id}...")
            ligand_smiles, crystal_mol = runner.molecule_loader.get_ligand_data(pdb_id)
            
            if ligand_smiles:
                logger.info(f"  ✓ Found ligand SMILES for {pdb_id}: {ligand_smiles[:50]}...")
            else:
                logger.warning(f"  ✗ No ligand SMILES found for {pdb_id}")
                
        except Exception as e:
            logger.error(f"Error testing {pdb_id}: {e}")
    
    logger.info("Basic functionality test completed!")

if __name__ == "__main__":
    test_basic_functionality()