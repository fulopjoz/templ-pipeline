#!/usr/bin/env python3
"""
Time-split benchmark runner for TEMPL pipeline.

This module implements time-split benchmarking with proper data hygiene,
ensuring test sets only use templates from earlier time periods.

Based on the proven approach from run_custom_split_benchmark.py with
improvements for modularity and integration with the unified benchmark infrastructure.
"""

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
from tqdm import tqdm

# Import hardware detection and configuration
from templ_pipeline.core.hardware import get_suggested_worker_config

# Import unified benchmark runner
from templ_pipeline.benchmark.runner import run_templ_pipeline_for_benchmark

logger = logging.getLogger(__name__)


def load_split_pdb_ids_standalone(split_file_path: Path) -> Set[str]:
    """
    Load PDB IDs for a specific split (standalone version for multiprocessing).
    
    Args:
        split_file_path: Path to split file
        
    Returns:
        Set of PDB IDs in the split
    """
    if not split_file_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_file_path}")
    
    pdb_ids = set()
    with open(split_file_path, 'r') as f:
        for line in f:
            pdb_id = line.strip().lower()
            if pdb_id:
                pdb_ids.add(pdb_id)
    
    return pdb_ids


def determine_target_split_standalone(target_pdb: str, split_files: Dict[str, Path]) -> str:
    """
    Determine which split a target PDB belongs to (standalone version for multiprocessing).
    
    Args:
        target_pdb: PDB ID to classify
        split_files: Dictionary mapping split names to file paths
        
    Returns:
        Split name ('train', 'val', 'test')
    """
    target_pdb = target_pdb.lower()
    
    for split_name in ["train", "val", "test"]:
        split_file = split_files[split_name]
        split_pdb_ids = load_split_pdb_ids_standalone(split_file)
        if target_pdb in split_pdb_ids:
            return split_name
    
    raise ValueError(f"PDB {target_pdb} not found in any split")


def get_allowed_templates_for_split_standalone(target_split: str, split_files: Dict[str, Path]) -> Set[str]:
    """
    Get allowed template PDB IDs for a target split based on time-split rules (standalone version for multiprocessing).
    
    Args:
        target_split: Split of the target molecule ('train', 'val', 'test')
        split_files: Dictionary mapping split names to file paths
        
    Returns:
        Set of PDB IDs allowed as templates (excludes target itself)
    """
    allowed_templates = set()
    
    if target_split == "test":
        # Test can use train + val templates (no future information)
        allowed_templates.update(load_split_pdb_ids_standalone(split_files["train"]))
        allowed_templates.update(load_split_pdb_ids_standalone(split_files["val"]))
    elif target_split == "val":
        # Val can only use train templates (no future information)
        allowed_templates.update(load_split_pdb_ids_standalone(split_files["train"]))
    elif target_split == "train":
        # Train uses leave-one-out (other train molecules as templates)
        allowed_templates.update(load_split_pdb_ids_standalone(split_files["train"]))
    else:
        raise ValueError(f"Unknown target split: {target_split}")
    
    # Convert to uppercase for compatibility
    allowed_templates = {pdb_id.upper() for pdb_id in allowed_templates}
    
    return allowed_templates


def run_timesplit_single_target(target_pdb: str,
                               split_files: Dict[str, Path],
                               data_dir: str,
                               poses_output_dir: Optional[str] = None,
                               n_conformers: int = 200,
                               template_knn: int = 100,
                               similarity_threshold: Optional[float] = None,
                               timeout: int = 300,
                               unconstrained: bool = False,
                               align_metric: str = "combo",
                               enable_optimization: bool = False,
                               no_realign: bool = False) -> Dict:
    """
    Run benchmark for a single target PDB with time-split constraints (standalone function for multiprocessing).
    
    Args:
        target_pdb: Target PDB ID
        split_files: Dictionary mapping split names ('train', 'val', 'test') to file paths
        data_dir: Data directory path
        poses_output_dir: Directory for poses output (optional)
        n_conformers: Number of conformers to generate
        template_knn: Number of template neighbors
        similarity_threshold: Similarity threshold (overrides KNN)
        timeout: Timeout in seconds
        unconstrained: Skip MCS and constrained embedding
        align_metric: Alignment metric ('shape', 'color', 'combo')
        enable_optimization: Enable force field optimization
        no_realign: Disable pose realignment
        
    Returns:
        Dictionary with benchmark results
    """
    start_time = time.time()
    
    try:
        # Determine target split and allowed templates
        target_split = determine_target_split_standalone(target_pdb, split_files)
        allowed_templates = get_allowed_templates_for_split_standalone(target_split, split_files)
        
        # Always exclude target itself (leave-one-out principle)
        exclude_pdb_ids = {target_pdb.upper()}
        allowed_templates.discard(target_pdb.upper())
        
        logger.info(f"Processing {target_pdb} ({target_split} split)")
        logger.info(f"Using {len(allowed_templates)} allowed templates")
        
        # Template filtering detailed logging
        logger.info(f"TEMPLATE_FILTERING: Template pool for {target_pdb}:")
        logger.info(f"TEMPLATE_FILTERING:   Split type: {target_split}")
        logger.info(f"TEMPLATE_FILTERING:   Initial allowed templates: {len(allowed_templates)}")
        logger.info(f"TEMPLATE_FILTERING:   Excluded targets: {len(exclude_pdb_ids)} (including self)")
        
        # Log template pool composition for debugging
        if len(allowed_templates) > 0:
            sample_templates = sorted(list(allowed_templates))[:10]
            logger.info(f"TEMPLATE_FILTERING:   Sample templates: {', '.join(sample_templates)}...")
        else:
            logger.warning(f"TEMPLATE_FILTERING:   WARNING: Empty template pool for {target_pdb}")
        
        # Performance Comparison Logging - Timesplit Direct Method  
        logger.info(f"BENCHMARK_METHOD: Direct class method call (timesplit)")
        logger.info(f"TEMPLATE_FILTERING: In-memory allowed_pdb_ids parameter")
        logger.info(f"EXECUTION_CONTEXT: Python ProcessPoolExecutor worker")
        
        # Run pipeline with template restrictions
        result = run_templ_pipeline_for_benchmark(
            target_pdb=target_pdb,
            exclude_pdb_ids=exclude_pdb_ids,
            allowed_pdb_ids=allowed_templates,
            n_conformers=n_conformers,
            template_knn=template_knn,
            similarity_threshold=similarity_threshold,
            internal_workers=1,  # Always 1 for parallel execution
            timeout=timeout,
            data_dir=data_dir,
            poses_output_dir=poses_output_dir,
            unconstrained=unconstrained,
            align_metric=align_metric,
            enable_optimization=enable_optimization,
            no_realign=no_realign,
        )
        
        # Add time-split metadata
        result.update({
            "target_pdb": target_pdb,
            "target_split": target_split,
            "allowed_templates_count": len(allowed_templates),
            "exclusions_count": len(exclude_pdb_ids),
            "runtime_total": time.time() - start_time,
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to process {target_pdb}: {e}")
        return {
            "success": False,
            "target_pdb": target_pdb,
            "error": str(e),
            "runtime_total": time.time() - start_time,
        }


class TimeSplitBenchmarkRunner:
    """
    Time-split benchmark runner with proper data hygiene and template restrictions.
    
    This class manages the execution of time-split benchmarks, ensuring that:
    - Test sets only use templates from training sets (no future information)
    - Validation sets only use templates from training sets
    - Training sets use leave-one-out validation
    """
    
    def __init__(self, 
                 data_dir: Optional[str] = None,
                 results_dir: Optional[str] = None,
                 poses_output_dir: Optional[str] = None,
                 memory_threshold_gb: float = 10.0,
                 enable_memory_monitoring: bool = True,
                 enable_process_recovery: bool = True):
        """
        Initialize the time-split benchmark runner.
        
        Args:
            data_dir: Directory containing benchmark data and splits
            results_dir: Directory for benchmark results
            poses_output_dir: Directory for generated poses (optional)
            memory_threshold_gb: Memory threshold for worker termination (GB)
            enable_memory_monitoring: Whether to enable memory monitoring
            enable_process_recovery: Whether to enable process recovery
        """
        self.data_dir = Path(data_dir) if data_dir else self._find_data_directory()
        self.results_dir = Path(results_dir) if results_dir else Path("timesplit_results")
        self.poses_output_dir = Path(poses_output_dir) if poses_output_dir else None
        
        # Memory management configuration
        self.memory_threshold_gb = memory_threshold_gb
        self.enable_memory_monitoring = enable_memory_monitoring
        self.enable_process_recovery = enable_process_recovery
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up split file paths
        self._setup_split_paths()
        
        # Configuration
        self.hardware_config = get_suggested_worker_config()
        self.benchmark_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize memory management components
        self._initialize_memory_management()
        
        logger.info(f"TimeSplit runner initialized:")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Results directory: {self.results_dir}")
        logger.info(f"  Hardware config: {self.hardware_config['n_workers']} workers")
        logger.info(f"  Memory monitoring: {enable_memory_monitoring} (threshold: {memory_threshold_gb:.1f}GB)")
        logger.info(f"  Process recovery: {enable_process_recovery}")

    def _find_data_directory(self) -> Path:
        """Find the data directory with timesplit files."""
        potential_paths = [
            Path(__file__).resolve().parent.parent.parent / "data",
            Path.cwd() / "data",
            Path.cwd() / "templ_pipeline" / "data",
            Path("data"),
            Path("..") / "data",
        ]
        
        for path in potential_paths:
            if (path.exists() and 
                (path / "ligands" / "templ_processed_ligands_v1.0.0.sdf.gz").exists()):
                return path
        
        # Fallback
        return Path("/data/pdbbind")

    def _setup_split_paths(self):
        """Set up paths to split files."""
        splits_dir = self.data_dir / "splits"
        if not splits_dir.exists():
            # Try alternative locations
            potential_splits = [
                self.data_dir / "time_splits_ghrepo",
                Path("mcs_bench/data/time_splits_ghrepo"),
                Path("data/time_splits_ghrepo"),
            ]
            for split_path in potential_splits:
                if split_path.exists():
                    splits_dir = split_path
                    break
        
        self.split_files = {
            "train": splits_dir / "timesplit_train",
            "val": splits_dir / "timesplit_val", 
            "test": splits_dir / "timesplit_test"
        }
        
        logger.info(f"Split files directory: {splits_dir}")

    def load_split_pdb_ids(self, split_name: str) -> Set[str]:
        """
        Load PDB IDs for a specific split.
        
        Args:
            split_name: Name of split ('train', 'val', 'test')
            
        Returns:
            Set of PDB IDs in the split
        """
        if split_name not in self.split_files:
            raise ValueError(f"Unknown split: {split_name}")
        
        split_file = self.split_files[split_name]
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        pdb_ids = set()
        with open(split_file, 'r') as f:
            for line in f:
                pdb_id = line.strip().lower()
                if pdb_id:
                    pdb_ids.add(pdb_id)
        
        logger.info(f"Loaded {len(pdb_ids)} PDB IDs for {split_name} split")
        return pdb_ids

    def _initialize_memory_management(self):
        """Initialize memory management components."""
        # Simplified initialization without optional dependencies
        self.memory_diagnostics = None
        self.recovery_manager = None
        logger.info("Memory management components initialized (simplified mode)")

    def get_memory_management_status(self) -> Dict:
        """Get comprehensive memory management and recovery status."""
        status = {
            "memory_monitoring_enabled": self.enable_memory_monitoring,
            "process_recovery_enabled": self.enable_process_recovery,
            "memory_threshold_gb": self.memory_threshold_gb
        }
        
        return status

    def get_allowed_templates_for_split(self, target_split: str) -> Set[str]:
        """
        Get allowed template PDB IDs for a target split based on time-split rules.
        
        Args:
            target_split: Split of the target molecule ('train', 'val', 'test')
            
        Returns:
            Set of PDB IDs allowed as templates (excludes target itself)
        """
        allowed_templates = set()
        
        if target_split == "test":
            # Test can use train + val templates (no future information)
            allowed_templates.update(self.load_split_pdb_ids("train"))
            allowed_templates.update(self.load_split_pdb_ids("val"))
        elif target_split == "val":
            # Val can only use train templates (no future information)
            allowed_templates.update(self.load_split_pdb_ids("train"))
        elif target_split == "train":
            # Train uses leave-one-out (other train molecules as templates)
            allowed_templates.update(self.load_split_pdb_ids("train"))
        else:
            raise ValueError(f"Unknown target split: {target_split}")
        
        # Convert to uppercase for compatibility
        allowed_templates = {pdb_id.upper() for pdb_id in allowed_templates}
        
        logger.debug(f"Split {target_split}: {len(allowed_templates)} allowed templates")
        return allowed_templates

    def determine_target_split(self, target_pdb: str) -> str:
        """
        Determine which split a target PDB belongs to.
        
        Args:
            target_pdb: PDB ID to classify
            
        Returns:
            Split name ('train', 'val', 'test')
        """
        target_pdb = target_pdb.lower()
        
        for split_name in ["train", "val", "test"]:
            split_pdb_ids = self.load_split_pdb_ids(split_name)
            if target_pdb in split_pdb_ids:
                return split_name
        
        raise ValueError(f"PDB {target_pdb} not found in any split")

    def run_single_target(self,
                         target_pdb: str,
                         n_conformers: int = 200,
                         template_knn: int = 100,
                         similarity_threshold: Optional[float] = None,
                         timeout: int = 300,
                         unconstrained: bool = False,
                         align_metric: str = "combo",
                         enable_optimization: bool = False,
                         no_realign: bool = False) -> Dict:
        """
        Run benchmark for a single target PDB with time-split constraints.
        
        Args:
            target_pdb: Target PDB ID
            n_conformers: Number of conformers to generate
            template_knn: Number of template neighbors
            similarity_threshold: Similarity threshold (overrides KNN)
            timeout: Timeout in seconds
            unconstrained: Skip MCS and constrained embedding
            align_metric: Alignment metric ('shape', 'color', 'combo')
            enable_optimization: Enable force field optimization
            no_realign: Disable pose realignment
            
        Returns:
            Dictionary with benchmark results
        """
        start_time = time.time()
        
        try:
            # Determine target split and allowed templates
            target_split = self.determine_target_split(target_pdb)
            allowed_templates = self.get_allowed_templates_for_split(target_split)
            
            # Always exclude target itself (leave-one-out principle)
            exclude_pdb_ids = {target_pdb.upper()}
            allowed_templates.discard(target_pdb.upper())
            
            logger.info(f"Processing {target_pdb} ({target_split} split)")
            logger.info(f"Using {len(allowed_templates)} allowed templates")
            
            # Template filtering detailed logging for class method version
            logger.info(f"TEMPLATE_FILTERING: Template pool for {target_pdb} (class method):")
            logger.info(f"TEMPLATE_FILTERING:   Split type: {target_split}")
            logger.info(f"TEMPLATE_FILTERING:   Initial allowed templates: {len(allowed_templates)}")
            logger.info(f"TEMPLATE_FILTERING:   Excluded targets: {len(exclude_pdb_ids)} (including self)")
            
            # Log template pool composition for debugging
            if len(allowed_templates) > 0:
                sample_templates = sorted(list(allowed_templates))[:10]
                logger.info(f"TEMPLATE_FILTERING:   Sample templates: {', '.join(sample_templates)}...")
            else:
                logger.warning(f"TEMPLATE_FILTERING:   WARNING: Empty template pool for {target_pdb}")
            
            # Performance Comparison Logging - Timesplit Direct Method
            logger.info(f"BENCHMARK_METHOD: Direct class method call (timesplit)")
            logger.info(f"TEMPLATE_FILTERING: In-memory allowed_pdb_ids parameter")
            logger.info(f"EXECUTION_CONTEXT: TimeSplitBenchmarkRunner class method")
            
            # Run pipeline with template restrictions
            result = run_templ_pipeline_for_benchmark(
                target_pdb=target_pdb,
                exclude_pdb_ids=exclude_pdb_ids,
                allowed_pdb_ids=allowed_templates,
                n_conformers=n_conformers,
                template_knn=template_knn,
                similarity_threshold=similarity_threshold,
                internal_workers=1,  # Always 1 for parallel execution
                timeout=timeout,
                data_dir=str(self.data_dir),
                poses_output_dir=str(self.poses_output_dir) if self.poses_output_dir else None,
                unconstrained=unconstrained,
                align_metric=align_metric,
                enable_optimization=enable_optimization,
                no_realign=no_realign,
            )
            
            # Add time-split metadata
            result.update({
                "target_pdb": target_pdb,
                "target_split": target_split,
                "allowed_templates_count": len(allowed_templates),
                "exclusions_count": len(exclude_pdb_ids),
                "runtime_total": time.time() - start_time,
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process {target_pdb}: {e}")
            return {
                "success": False,
                "target_pdb": target_pdb,
                "error": str(e),
                "runtime_total": time.time() - start_time,
            }

    def run_split_benchmark(self,
                           split_name: str,
                           n_workers: int = None,
                           n_conformers: int = 200,
                           template_knn: int = 100,
                           max_pdbs: Optional[int] = None,
                           similarity_threshold: Optional[float] = None,
                           timeout: int = 300,
                           unconstrained: bool = False,
                           align_metric: str = "combo", 
                           enable_optimization: bool = False,
                           no_realign: bool = False,
                           quiet: bool = False) -> Dict:
        """
        Run benchmark for a complete split.
        
        Args:
            split_name: Name of split to run ('train', 'val', 'test')
            n_workers: Number of parallel workers
            n_conformers: Number of conformers per molecule
            template_knn: Number of template neighbors
            max_pdbs: Maximum number of PDBs to process (for testing)
            similarity_threshold: Similarity threshold
            timeout: Timeout per molecule in seconds
            unconstrained: Skip MCS and constrained embedding
            align_metric: Alignment metric ('shape', 'color', 'combo')
            enable_optimization: Enable force field optimization
            no_realign: Disable pose realignment
            quiet: Suppress progress output
            
        Returns:
            Dictionary with complete benchmark results
        """
        if n_workers is None:
            n_workers = self.hardware_config["n_workers"]
        
        # Load target PDBs for this split
        target_pdbs = list(self.load_split_pdb_ids(split_name))
        
        # Apply max_pdbs limit if specified
        if max_pdbs and max_pdbs > 0:
            target_pdbs = target_pdbs[:max_pdbs]
            logger.info(f"Limited to {len(target_pdbs)} PDBs for testing")
        
        logger.info(f"Running {split_name} split with {len(target_pdbs)} targets")
        
        # Create output file for streaming results
        output_jsonl = self.results_dir / f"results_{split_name}_{self.benchmark_timestamp}.jsonl"
        
        # Initialize result tracking
        processed_count = 0
        success_count = 0
        failed_count = 0
        
        # Process targets with parallel execution
        if n_workers > 1 and len(target_pdbs) > 1:
            logger.info(f"Using {n_workers} workers for parallel processing")
            
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit all jobs
                future_to_pdb = {}
                
                for target_pdb in target_pdbs:
                    # Submit job
                    future = executor.submit(
                        run_timesplit_single_target,
                        target_pdb=target_pdb,
                        split_files=self.split_files,
                        data_dir=str(self.data_dir),
                        poses_output_dir=str(self.poses_output_dir) if self.poses_output_dir else None,
                        n_conformers=n_conformers,
                        template_knn=template_knn,
                        similarity_threshold=similarity_threshold,
                        timeout=timeout,
                        unconstrained=unconstrained,
                        align_metric=align_metric,
                        enable_optimization=enable_optimization,
                        no_realign=no_realign
                    )
                    future_to_pdb[future] = target_pdb
                
                # Collect results with progress bar
                desc = f"{split_name.title()} Split"
                if not quiet:
                    progress_bar = tqdm(total=len(future_to_pdb), desc=desc, ncols=100)
                
                for future in as_completed(future_to_pdb):
                    target_pdb = future_to_pdb[future]
                    
                    try:
                        result = future.result(timeout=timeout + 30)  # Extra time for cleanup
                        
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
                            "timeout": True,
                        }
                        
                        with open(output_jsonl, 'a') as f:
                            json.dump(error_result, f)
                            f.write('\n')
                        
                        processed_count += 1
                        failed_count += 1
                    
                    # Update progress
                    if not quiet:
                        success_rate = (success_count / processed_count * 100) if processed_count > 0 else 0
                        progress_bar.set_postfix({
                            "success": f"{success_rate:.1f}%",
                            "errors": failed_count
                        })
                        progress_bar.update(1)
                
                if not quiet:
                    progress_bar.close()
        
        else:
            # Sequential processing
            logger.info("Running sequentially")
            desc = f"{split_name.title()} Split (sequential)"
            
            if not quiet:
                progress_bar = tqdm(target_pdbs, desc=desc, ncols=100)
            else:
                progress_bar = target_pdbs
            
            for target_pdb in progress_bar:
                result = self.run_single_target(
                    target_pdb=target_pdb,
                    n_conformers=n_conformers,
                    template_knn=template_knn,
                    similarity_threshold=similarity_threshold,
                    timeout=timeout,
                    unconstrained=unconstrained,
                    align_metric=align_metric,
                    enable_optimization=enable_optimization,
                    no_realign=no_realign,
                )
                
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
                
                # Update progress
                if not quiet and hasattr(progress_bar, 'set_postfix'):
                    success_rate = (success_count / processed_count * 100) if processed_count > 0 else 0
                    progress_bar.set_postfix({
                        "success": f"{success_rate:.1f}%",
                        "errors": failed_count
                    })
        
        # Generate summary
        summary = {
            "split": split_name,
            "total_targets": len(target_pdbs),
            "processed": processed_count,
            "successful": success_count,
            "failed": failed_count,
            "success_rate": (success_count / processed_count * 100) if processed_count > 0 else 0,
            "results_file": str(output_jsonl),
            "timestamp": self.benchmark_timestamp,
            "benchmark_info": {
                "name": "templ_timesplit_benchmark",
                "split": split_name,
                "n_conformers": n_conformers,
                "template_knn": template_knn,
                "n_workers": n_workers,
                "timeout": timeout,
                "unconstrained": unconstrained,
                "align_metric": align_metric,
                "enable_optimization": enable_optimization,
                "no_realign": no_realign,
            }
        }
        
        # Save summary
        summary_file = self.results_dir / f"summary_{split_name}_{self.benchmark_timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Split {split_name} completed:")
        logger.info(f"  Processed: {processed_count}/{len(target_pdbs)}")
        logger.info(f"  Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"  Results: {output_jsonl}")
        
        return summary

    def create_experiment_directory(self, split_name: str) -> Path:
        """Create organized experiment directory for outputs."""
        experiment_dir = self.results_dir / split_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir