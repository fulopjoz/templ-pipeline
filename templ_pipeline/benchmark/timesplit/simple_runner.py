#!/usr/bin/env python3
"""
Simplified time-split benchmark runner for TEMPL pipeline.

This module implements a streamlined time-split benchmarking approach that:
- Uses subprocess calls to `templ run` for memory isolation
- Leverages existing DatasetSplits class for split management
- Follows the proven pattern from run_custom_split_benchmark.py
- Maintains proper time-split data hygiene
"""

import json
import logging
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set

from tqdm import tqdm

from templ_pipeline.benchmark.runner import LazyMoleculeLoader

logger = logging.getLogger(__name__)


class SimpleTimeSplitRunner:
    """
    Simplified time-split benchmark runner using subprocess calls to CLI.
    
    This approach provides better memory isolation and follows the proven
    pattern from run_custom_split_benchmark.py while leveraging the existing
    CLI infrastructure.
    """
    
    def __init__(self, data_dir: Optional[str] = None, results_dir: Optional[str] = None):
        """
        Initialize the simple time-split runner.
        
        Args:
            data_dir: Directory containing benchmark data and splits
            results_dir: Directory for benchmark results
        """
        self.data_dir = Path(data_dir) if data_dir else self._find_data_directory()
        self.results_dir = Path(results_dir) if results_dir else Path("simple_timesplit_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize splits directory and molecule loader
        self.splits_dir = self.data_dir / "splits"
        self.molecule_loader = LazyMoleculeLoader(str(self.data_dir))
        
        # Load split PDB IDs
        self._load_splits()
        
        # Set timestamp for compatibility
        from datetime import datetime
        self.benchmark_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Simple TimeSplit runner initialized:")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Results directory: {self.results_dir}")

    def _find_data_directory(self) -> Path:
        """Find the data directory with timesplit files."""
        potential_paths = [
            Path(__file__).resolve().parent.parent.parent / "data",
            Path.cwd() / "data",
            Path("data"),
        ]
        
        for path in potential_paths:
            if path.exists() and (path / "splits").exists():
                return path
        
        return Path("data")  # Fallback

    def _load_splits(self):
        """Load dataset splits from timesplit files."""
        self.split_files = {
            "train": self.splits_dir / "timesplit_train",
            "val": self.splits_dir / "timesplit_val", 
            "test": self.splits_dir / "timesplit_test"
        }
        
        # Load split PDB IDs
        self.train_pdbs = self._load_pdb_list(self.split_files["train"])
        self.val_pdbs = self._load_pdb_list(self.split_files["val"])
        self.test_pdbs = self._load_pdb_list(self.split_files["test"])
        
        logger.info(f"Loaded splits:")
        logger.info(f"  Train: {len(self.train_pdbs)} PDBs")
        logger.info(f"  Val: {len(self.val_pdbs)} PDBs")
        logger.info(f"  Test: {len(self.test_pdbs)} PDBs")

    def _load_pdb_list(self, filepath: Path) -> Set[str]:
        """Load PDB IDs from a split file."""
        pdb_ids = set()
        if not filepath.exists():
            logger.warning(f"Split file not found: {filepath}")
            return pdb_ids
            
        with open(filepath, 'r') as f:
            for line in f:
                pdb_id = line.strip().lower()
                if pdb_id:
                    pdb_ids.add(pdb_id)
        
        return pdb_ids

    def get_split_pdbs(self, split_name: str) -> Set[str]:
        """Get PDB IDs for a specific split."""
        if split_name == "train":
            return self.train_pdbs
        elif split_name == "val":
            return self.val_pdbs
        elif split_name == "test":
            return self.test_pdbs
        else:
            raise ValueError(f"Unknown split: {split_name}")

    def determine_split(self, pdb_id: str) -> str:
        """Determine which split a PDB ID belongs to."""
        pdb_id = pdb_id.lower()
        
        if pdb_id in self.train_pdbs:
            return "train"
        elif pdb_id in self.val_pdbs:
            return "val"
        elif pdb_id in self.test_pdbs:
            return "test"
        else:
            raise ValueError(f"PDB {pdb_id} not found in any split")

    def get_allowed_templates_for_split(self, target_split: str, target_pdb: str) -> Set[str]:
        """
        Get allowed template PDB IDs for a target split based on time-split rules.
        
        Time-split rules:
        - Test: can use train + val templates (no future information)
        - Val: can only use train templates (no future information)  
        - Train: uses leave-one-out (other train molecules as templates)
        
        Args:
            target_split: Split of the target molecule ('train', 'val', 'test')
            target_pdb: PDB ID of the target (for leave-one-out exclusion)
            
        Returns:
            Set of PDB IDs allowed as templates
        """
        allowed_templates = set()
        
        if target_split == "test":
            # Test can use train + val templates
            allowed_templates.update(self.get_split_pdbs("train"))
            allowed_templates.update(self.get_split_pdbs("val"))
        elif target_split == "val":
            # Val can only use train templates
            allowed_templates.update(self.get_split_pdbs("train"))
        elif target_split == "train":
            # Train uses leave-one-out (other train molecules as templates)
            allowed_templates.update(self.get_split_pdbs("train"))
        else:
            raise ValueError(f"Unknown target split: {target_split}")
        
        # Always exclude target itself (leave-one-out principle)
        allowed_templates.discard(target_pdb.upper())
        
        # Convert to uppercase for consistency
        allowed_templates = {pdb_id.upper() for pdb_id in allowed_templates}
        
        return allowed_templates

    def validate_cli_command(self, cmd: List[str]) -> bool:
        """
        Validate CLI command structure before execution.
        
        Args:
            cmd: Command list to validate
            
        Returns:
            True if command structure is valid
        """
        try:
            # Check basic structure: ["templ", "--output-dir", "path", "run", ...]
            if len(cmd) < 4:
                logger.error(f"Command too short: {cmd}")
                return False
                
            if cmd[0] != "templ":
                logger.error(f"Command should start with 'templ', got: {cmd[0]}")
                return False
                
            # Find the subcommand position
            subcommand_pos = None
            for i, arg in enumerate(cmd):
                if arg in ["run", "embed", "find-templates", "generate-poses", "benchmark"]:
                    subcommand_pos = i
                    break
                    
            if subcommand_pos is None:
                logger.error(f"No valid subcommand found in: {cmd}")
                return False
                
            # Validate that global parameters (like --output-dir) come before subcommand
            global_params = ["--output-dir", "--log-level", "--verbosity", "--seed"]
            for i in range(1, subcommand_pos):
                arg = cmd[i]
                if arg.startswith("--") and arg not in global_params:
                    logger.warning(f"Parameter {arg} might be a subcommand parameter placed before subcommand")
                    
            logger.debug(f"CLI command validation passed: {' '.join(cmd[:6])}...")
            return True
            
        except Exception as e:
            logger.error(f"CLI command validation failed: {e}")
            return False

    def run_single_target_subprocess(self, 
                                   target_pdb: str,
                                   allowed_templates: Set[str],
                                   n_conformers: int = 200,
                                   timeout: int = 600) -> Dict:
        """
        Run benchmark for a single target using subprocess call to CLI.
        
        Args:
            target_pdb: Target PDB ID
            allowed_templates: Set of allowed template PDB IDs
            n_conformers: Number of conformers to generate
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with benchmark results
        """
        start_time = time.time()
        
        try:
            # Load ligand SMILES for this PDB
            ligand_smiles, _ = self.molecule_loader.get_ligand_data(target_pdb)
            if not ligand_smiles:
                return {
                    "success": False,
                    "target_pdb": target_pdb,
                    "error": f"Could not load ligand SMILES for {target_pdb}",
                    "runtime_total": time.time() - start_time,
                    "rmsd_values": {},  # Empty RMSD values for SMILES loading error
                }
            
            # Build CLI command - --output-dir must come before subcommand
            cmd = [
                "templ", 
                "--output-dir", str(self.results_dir / "poses" / target_pdb.lower()),
                "run",
                "--protein-pdb-id", target_pdb,
                "--ligand-smiles", ligand_smiles,
                "--num-conformers", str(n_conformers),
                "--allowed-pdb-ids", ",".join(sorted(allowed_templates)),
                "--exclude-pdb-ids", target_pdb.upper(),  # Ensure target is excluded
            ]
            
            # Validate CLI command structure
            if not self.validate_cli_command(cmd):
                return {
                    "success": False,
                    "target_pdb": target_pdb,
                    "error": f"CLI command validation failed for {target_pdb}",
                    "runtime_total": time.time() - start_time,
                    "rmsd_values": {},  # Empty RMSD values for validation error
                }
            
            logger.info(f"Running: {' '.join(cmd[:6])} ... (with {len(allowed_templates)} allowed templates)")
            
            # Run subprocess with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.data_dir.parent)  # Run from project root
            )
            
            runtime_total = time.time() - start_time
            
            if result.returncode == 0:
                # Parse JSON output from CLI stdout for RMSD data
                rmsd_values = {}
                try:
                    # Look for the JSON result line in stdout
                    for line in result.stdout.split('\n'):
                        if line.startswith("TEMPL_JSON_RESULT:"):
                            json_str = line[len("TEMPL_JSON_RESULT:"):]
                            json_data = json.loads(json_str)
                            rmsd_values = json_data.get("rmsd_values", {})
                            break
                except Exception as e:
                    logger.debug(f"Failed to parse JSON output for {target_pdb}: {e}")
                
                return {
                    "success": True,
                    "target_pdb": target_pdb,
                    "allowed_templates_count": len(allowed_templates),
                    "runtime_total": runtime_total,
                    "stdout": result.stdout,
                    "rmsd_values": rmsd_values,  # Add RMSD data for summary generation
                }
            else:
                return {
                    "success": False,
                    "target_pdb": target_pdb,
                    "error": f"CLI returned {result.returncode}",
                    "stderr": result.stderr,
                    "runtime_total": runtime_total,
                    "rmsd_values": {},  # Empty RMSD values for failed runs
                }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "target_pdb": target_pdb,
                "error": f"Timeout after {timeout}s",
                "runtime_total": time.time() - start_time,
                "rmsd_values": {},  # Empty RMSD values for timeout
            }
        except Exception as e:
            return {
                "success": False,
                "target_pdb": target_pdb,
                "error": str(e),
                "runtime_total": time.time() - start_time,
                "rmsd_values": {},  # Empty RMSD values for exceptions
            }

    def run_split_benchmark(self,
                           split_name: str,
                           n_workers: int = 4,
                           n_conformers: int = 200,
                           max_pdbs: Optional[int] = None,
                           timeout: int = 600) -> Dict:
        """
        Run benchmark for a complete split using subprocess parallelization.
        
        Args:
            split_name: Name of split to run ('train', 'val', 'test')
            n_workers: Number of parallel workers
            n_conformers: Number of conformers per molecule
            max_pdbs: Maximum number of PDBs to process (for testing)
            timeout: Timeout per molecule in seconds
            
        Returns:
            Dictionary with complete benchmark results
        """
        # Load target PDBs for this split
        target_pdbs = list(self.get_split_pdbs(split_name))
        
        if max_pdbs and max_pdbs > 0:
            target_pdbs = target_pdbs[:max_pdbs]
            logger.info(f"Limited to {len(target_pdbs)} PDBs for testing")
        
        logger.info(f"Running {split_name} split with {len(target_pdbs)} targets using {n_workers} workers")
        
        # Create output file for streaming results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_jsonl = self.results_dir / f"results_{split_name}_{timestamp}.jsonl"
        
        # Initialize result tracking
        processed_count = 0
        success_count = 0
        failed_count = 0
        
        # Process targets with parallel execution
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all jobs
            future_to_pdb = {}
            
            for target_pdb in target_pdbs:
                # Determine target split and get allowed templates
                target_split = self.determine_split(target_pdb)
                allowed_templates = self.get_allowed_templates_for_split(target_split, target_pdb)
                
                logger.debug(f"Submitting {target_pdb} ({target_split}) with {len(allowed_templates)} templates")
                
                future = executor.submit(
                    self.run_single_target_subprocess,
                    target_pdb=target_pdb,
                    allowed_templates=allowed_templates,
                    n_conformers=n_conformers,
                    timeout=timeout
                )
                future_to_pdb[future] = target_pdb
            
            # Collect results with progress bar
            desc = f"{split_name.title()} Split"
            progress_bar = tqdm(total=len(future_to_pdb), desc=desc, ncols=100)
            
            for future in as_completed(future_to_pdb):
                target_pdb = future_to_pdb[future]
                
                try:
                    result = future.result(timeout=timeout + 30)
                    
                    # Stream result to file
                    with open(output_jsonl, 'a') as f:
                        json.dump(result, f)
                        f.write('\n')
                    
                    # Update counters
                    processed_count += 1
                    if result.get("success"):
                        success_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Future failed for {target_pdb}: {e}")
                    error_result = {
                        "success": False,
                        "target_pdb": target_pdb,
                        "error": f"Future execution failed: {str(e)}",
                    }
                    
                    with open(output_jsonl, 'a') as f:
                        json.dump(error_result, f)
                        f.write('\n')
                    
                    processed_count += 1
                    failed_count += 1
                
                # Update progress
                success_rate = (success_count / processed_count * 100) if processed_count > 0 else 0
                progress_bar.set_postfix({
                    "success": f"{success_rate:.1f}%",
                    "errors": failed_count
                })
                progress_bar.update(1)
            
            progress_bar.close()
        
        # Generate summary
        summary = {
            "split": split_name,
            "total_targets": len(target_pdbs),
            "processed": processed_count,
            "successful": success_count,
            "failed": failed_count,
            "success_rate": (success_count / processed_count * 100) if processed_count > 0 else 0,
            "results_file": str(output_jsonl),
            "timestamp": timestamp,
            "benchmark_info": {
                "name": "simple_timesplit_benchmark",
                "split": split_name,
                "n_conformers": n_conformers,
                "n_workers": n_workers,
                "timeout": timeout,
            }
        }
        
        # Save summary
        summary_file = self.results_dir / f"summary_{split_name}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Split {split_name} completed:")
        logger.info(f"  Processed: {processed_count}/{len(target_pdbs)}")
        logger.info(f"  Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"  Results: {output_jsonl}")
        
        return summary


def main():
    """CLI entry point for simple timesplit runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple TimeSplit Benchmark Runner")
    parser.add_argument("split", choices=["train", "val", "test"], help="Split to run")
    parser.add_argument("--data-dir", type=str, help="Data directory")
    parser.add_argument("--results-dir", type=str, help="Results directory")
    parser.add_argument("--n-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--n-conformers", type=int, default=200, help="Number of conformers")
    parser.add_argument("--max-pdbs", type=int, help="Max PDBs to process (for testing)")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per target")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create runner and run benchmark
    runner = SimpleTimeSplitRunner(
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )
    
    summary = runner.run_split_benchmark(
        split_name=args.split,
        n_workers=args.n_workers,
        n_conformers=args.n_conformers,
        max_pdbs=args.max_pdbs,
        timeout=args.timeout
    )
    
    print(f"Benchmark completed with {summary['success_rate']:.1f}% success rate")


if __name__ == "__main__":
    main()