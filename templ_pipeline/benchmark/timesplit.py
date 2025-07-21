# -----------------------------------------------------------------------------
# Time-Split Benchmark for TEMPL Pipeline
# -----------------------------------------------------------------------------
# This script implements a time-split benchmark for the TEMPL pipeline, using
# efficient hardware resource management and parallel processing. It is designed
# to evaluate the pipeline on different data splits (train/val/test) and report
# performance metrics. The script is inspired by the Polaris benchmarking style.
#
# Key features:
# - Hardware auto-detection and scaling for optimal worker count
# - Parallel processing using ProcessPoolExecutor or pebble
# - Memory optimization via shared cache
# - Progress bars and logging for monitoring
# - Flexible CLI for custom runs
# -----------------------------------------------------------------------------

import argparse
import gc
import json
import logging
import multiprocessing as mp
import sys
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
    TimeoutError as FutureTimeoutError,
)
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm
import os
import psutil
import resource

# -----------------------------
# Imports and Hardware Detection
# -----------------------------
# Import hardware detection - optimized for maximum performance
try:
    from templ_pipeline.core.hardware import get_suggested_worker_config, get_basic_hardware_info
    HARDWARE_CONFIG = get_suggested_worker_config()
    
    # Override conservative limits for maximum performance
    # Since user wants all CPUs utilized and not be strict on memory
    HARDWARE_INFO = get_basic_hardware_info()
    TOTAL_CPUS = HARDWARE_INFO.cpu_count
    
    # Use conservative hardware detection like Polaris (max 16 workers for system stability)
    DEFAULT_WORKERS = HARDWARE_CONFIG["n_workers"]
    
    # Use module-specific logger instead of root logger to avoid CLI pollution
    _logger = logging.getLogger(__name__)
    _logger.debug(f"Timesplit hardware optimization: {DEFAULT_WORKERS}/{TOTAL_CPUS} workers")
    
except ImportError:
    _logger = logging.getLogger(__name__)
    _logger.warning("Hardware detection not available, using conservative default")
    DEFAULT_WORKERS = 8

# -----------------------------
# Benchmark Infrastructure Import
# -----------------------------
# Import benchmark infrastructure
from templ_pipeline.benchmark.runner import run_templ_pipeline_for_benchmark

# -----------------------------
# Optional: Pebble for Advanced Timeout Handling
# -----------------------------
# Try to import pebble for proper timeout handling (same as Polaris)
try:
    import pebble
    from pebble import ProcessPool
    PEBBLE_AVAILABLE = True
except ImportError:
    PEBBLE_AVAILABLE = False
    ProcessPool = None

# -----------------------------
# Configuration Constants
# -----------------------------
# Configuration - use same pattern as Polaris
MOLECULE_TIMEOUT = 180  # 3 minutes like Polaris

# Progress Bar Configuration (copied from Polaris)
class ProgressConfig:
    """Configuration for progress bars"""

    @staticmethod
    def get_bar_format():
        return "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

    @staticmethod
    def get_postfix_format(success_rate: float, errors: int = 0):
        return {"success": f"{success_rate:.1f}%", "errors": errors}
    
    @staticmethod
    def get_tqdm_config(desc: str = None):
        """Get complete tqdm configuration for benchmark progress bars"""
        config = {
            'bar_format': ProgressConfig.get_bar_format(),
            'ncols': 100,
            'leave': True,
            'file': sys.stdout,
            'disable': False
        }
        if desc:
            config['desc'] = desc
        return config


# -----------------------------
# Logging Setup
# -----------------------------
def setup_benchmark_logging(log_level: str = "INFO"):
    """Configure logging for benchmark with reduced verbosity (same as Polaris)"""
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Reduce verbosity for specific modules during benchmark
    mcs_logger = logging.getLogger('templ_pipeline.core.mcs')
    scoring_logger = logging.getLogger('templ_pipeline.core.scoring')
    
    # Set MCS and scoring to WARNING to reduce verbose output
    mcs_logger.setLevel(logging.WARNING)
    scoring_logger.setLevel(logging.WARNING)
    
    # Keep benchmark logger at INFO level
    benchmark_logger = logging.getLogger(__name__)
    benchmark_logger.setLevel(getattr(logging, log_level))
    
    return benchmark_logger


# -----------------------------------------------------------------------------
# Split Loading Utilities
# -----------------------------------------------------------------------------
# These functions load the lists of PDB IDs for each split (train/val/test) and
# determine which PDBs should be excluded from template search for a given target.

def load_timesplit_pdb_list(split_name: str) -> List[str]:
    """Load PDB IDs for the requested split using timesplit files."""
    
    split_name = split_name.lower()
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split name: {split_name}")

    # Locate split files with multiple fallback paths
    potential_paths = [
        Path(__file__).resolve().parent.parent / "data" / "splits" / f"timesplit_{split_name}",
        Path.cwd() / "data" / "splits" / f"timesplit_{split_name}",
        Path.cwd() / "templ_pipeline" / "data" / "splits" / f"timesplit_{split_name}",
        Path("data") / "splits" / f"timesplit_{split_name}",
        Path("..") / "data" / "splits" / f"timesplit_{split_name}",
    ]

    split_file = None
    for path in potential_paths:
        if path.exists():
            split_file = path
            break

    if split_file is None:
        raise FileNotFoundError(
            f"Time-split file 'timesplit_{split_name}' not found. Tried locations:\n"
            + "\n".join(f"  - {p}" for p in potential_paths)
        )

    with split_file.open("r", encoding="utf-8") as fh:
        pdbs = [ln.strip().lower() for ln in fh if ln.strip()]

    return pdbs


def get_timesplit_template_exclusions(target_pdb: str, target_split: str) -> Set[str]:
    """Get PDB IDs to exclude from template search based on time split rules."""
    
    exclusions = {target_pdb}  # Always exclude the target itself

    try:
        if target_split == "train":
            # Train: exclude val and test sets (leave-one-out within train)
            exclusions.update(load_timesplit_pdb_list("val"))
            exclusions.update(load_timesplit_pdb_list("test"))
        elif target_split == "val":
            # Val: exclude val and test sets (use only train templates)
            exclusions.update(load_timesplit_pdb_list("val"))
            exclusions.update(load_timesplit_pdb_list("test"))
        elif target_split == "test":
            # Test: exclude test set only (use train + val templates)
            exclusions.update(load_timesplit_pdb_list("test"))
    except Exception as e:
        logging.warning(f"Could not load split files for exclusions: {e}")

    return exclusions


def get_timesplit_allowed_templates(target_pdb: str, target_split: str) -> Set[str]:
    """Get PDB IDs allowed as templates based on time split rules.
    
    This function implements proper data hygiene for time-split benchmarking
    by restricting the template search space to only appropriate splits.
    
    Args:
        target_pdb: PDB ID of the target molecule
        target_split: Split the target belongs to ("train", "val", "test")
        
    Returns:
        Set of PDB IDs that are allowed to be used as templates
    """
    allowed_templates = set()
    
    try:
        if target_split == "test":
            # Test: can use train + val templates (no future information)
            allowed_templates.update(load_timesplit_pdb_list("train"))
            allowed_templates.update(load_timesplit_pdb_list("val"))
            logging.debug(f"Test target {target_pdb}: allowing train+val templates")
        elif target_split == "val":
            # Val: can only use train templates (no future information)
            allowed_templates.update(load_timesplit_pdb_list("train"))
            logging.debug(f"Val target {target_pdb}: allowing train templates only")
        elif target_split == "train":
            # Train: can use train templates (leave-one-out)
            allowed_templates.update(load_timesplit_pdb_list("train"))
            logging.debug(f"Train target {target_pdb}: allowing train templates (LOO)")
        else:
            logging.warning(f"Unknown split '{target_split}' for {target_pdb}")
            return set()
    except Exception as e:
        logging.error(f"Could not load split files for allowed templates: {e}")
        return set()
    
    # Normalize all PDB IDs to uppercase for compatibility with embedding_db
    allowed_templates = {pid.upper() for pid in allowed_templates}
    # CRITICAL: Remove target itself to prevent self-templating data leak
    target_pdb_upper = target_pdb.upper()
    if target_pdb_upper in allowed_templates:
        allowed_templates.remove(target_pdb_upper)
        logging.debug(f"Removed target {target_pdb_upper} from its own allowed templates")
    
    logging.info(f"Target {target_pdb} ({target_split}): {len(allowed_templates)} allowed templates")
    
    return allowed_templates


# -----------------------------------------------------------------------------
# Core Worker Function
# -----------------------------------------------------------------------------
# This function runs the benchmark pipeline for a single target PDB in a separate
# process. It determines the split, computes exclusions, and calls the unified
# benchmark runner. Results and errors are returned as dictionaries.

def _process_single_target(args: Tuple, per_worker_ram_gb: float = 4.0) -> Dict:
    """Process a single target PDB in an isolated subprocess.
    
    Uses the unified benchmark runner (same as Polaris) for reliable execution.
    """
    # Set a per-process memory limit (user-configurable, default 4GB)
    try:
        max_bytes = int(per_worker_ram_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
    except Exception as e:
        logging.warning(f"Could not set memory limit: {e}")

    import psutil
    import os
    target_pdb, config_dict = args
    start_time = time.time()

    # --- EARLY LOGGING: molecule stats before processing ---
    try:
        # Try to load molecule info if possible (requires access to data_dir and ligand file)
        data_dir = config_dict.get("data_dir")
        ligand_path = None
        if data_dir:
            from templ_pipeline.core.templates import ligand_path as get_ligand_path
            ligand_path = get_ligand_path(target_pdb, data_dir)
        atom_count = None
        conf_count = None
        if ligand_path:
            from rdkit import Chem
            suppl = Chem.SDMolSupplier(ligand_path)
            mols = [m for m in suppl if m is not None]
            if mols:
                mol = mols[0]
                atom_count = mol.GetNumAtoms()
                conf_count = mol.GetNumConformers()
        proc = psutil.Process(os.getpid())
        mem_before = proc.memory_info().rss / 1e9
        logging.info(f"[EARLY] {target_pdb}: atoms={atom_count}, conformers={conf_count}, memory={mem_before:.2f} GB")
    except Exception as e:
        logging.warning(f"[EARLY] Could not log molecule stats for {target_pdb}: {e}")
    # --- END EARLY LOGGING ---

    # Determine target split
    target_split = None
    split_load_errors = []  # Collect errors for diagnostics
    for split_name in ["train", "val", "test"]:
        try:
            split_pdbs = load_timesplit_pdb_list(split_name)
            if target_pdb.lower() in [pdb.lower() for pdb in split_pdbs]:
                target_split = split_name
                break
        except FileNotFoundError as e:
            logging.error(f"Split file not found for {split_name}: {e}")
            split_load_errors.append(f"{split_name}: {e}")
            continue
        except Exception as e:
            logging.error(f"Unexpected error loading {split_name}: {e}")
            split_load_errors.append(f"{split_name}: {e}")
            # Optionally, re-raise or continue depending on desired strictness
            continue

    if target_split is None:
        error_msg = (
            f"Could not determine split for {target_pdb}. "
            f"Split file load errors: {split_load_errors if split_load_errors else 'No split files found.'}"
        )
        logging.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "pdb_id": target_pdb,
            "runtime_total": time.time() - start_time,
        }
    
    # Get template restrictions based on time split rules
    exclusions = get_timesplit_template_exclusions(target_pdb, target_split)
    allowed_templates = get_timesplit_allowed_templates(target_pdb, target_split)
    
    # Validation: ensure target is not in allowed templates (data leak prevention)
    if target_pdb.lower() in allowed_templates:
        logging.error(f"CRITICAL: Target {target_pdb} found in its own allowed templates - this would cause data leak!")
        return {
            "success": False,
            "error": f"Data leak prevented: target {target_pdb} was in its own allowed templates",
            "pdb_id": target_pdb,
            "target_split": target_split,
            "runtime_total": time.time() - start_time,
        }
    
    # Validation: ensure allowed templates is not empty
    if not allowed_templates:
        logging.warning(f"No allowed templates found for {target_pdb} in {target_split} split")
    
    # Use unified benchmark runner with template restrictions (same approach as Polaris)
    try:
        result = run_templ_pipeline_for_benchmark(
            target_pdb=target_pdb,
            exclude_pdb_ids=exclusions,
            allowed_pdb_ids=allowed_templates,  # NEW: restrict search space to allowed templates
            n_conformers=config_dict.get("n_conformers", 200),
            template_knn=config_dict.get("template_knn", 100),
            similarity_threshold=config_dict.get("similarity_threshold", 0.9),
            internal_workers=1,  # Always 1 to prevent nested parallelization
            timeout=config_dict.get("timeout", MOLECULE_TIMEOUT),
            data_dir=config_dict.get("data_dir"),
            poses_output_dir=config_dict.get("poses_output_dir"),
            shared_cache_file=config_dict.get("shared_cache_file"),
            unconstrained=config_dict.get("unconstrained", False),
            align_metric=config_dict.get("align_metric", "combo"),
            enable_optimization=config_dict.get("enable_optimization", False),
            no_realign=config_dict.get("no_realign", False),
        )
        
        # Add target metadata including template restrictions
        result["pdb_id"] = target_pdb
        result["target_split"] = target_split
        result["exclusions_count"] = len(exclusions)
        result["allowed_templates_count"] = len(allowed_templates)
        
        # Log memory usage after processing
        try:
            proc = psutil.Process(os.getpid())
            mem_after = proc.memory_info().rss / 1e9
            logging.info(f"[MEMORY] After processing {target_pdb}: {mem_after:.2f} GB")
        except Exception as e:
            logging.warning(f"Could not log memory after: {e}")

        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Pipeline execution failed: {str(e)}",
            "pdb_id": target_pdb,
            "target_split": target_split,
            "runtime_total": time.time() - start_time,
        }


# -----------------------------------------------------------------------------
# Evaluation Function (Parallel Processing)
# -----------------------------------------------------------------------------
# This function manages the parallel execution of the benchmark across all targets.
# It uses either pebble or ProcessPoolExecutor to distribute work, handles timeouts,
# collects results, and updates progress bars. Results are streamed to a JSONL file.

def evaluate_timesplit_targets(
    target_pdbs: List[str],
    config_dict: Dict,
    n_workers: int,
    output_jsonl: Path,
    quiet: bool = False,
    per_worker_ram_gb: float = 4.0,
) -> Tuple[int, int, int]:
    """Evaluate targets using Polaris-style ProcessPoolExecutor pattern.
    
    Returns:
        Tuple of (processed_count, success_count, failed_count)
    """
    
    processed_count = 0
    success_count = 0
    failed_count = 0
    
    # Process targets with timeout handling using Polaris pattern
    if PEBBLE_AVAILABLE:
        with ProcessPool(max_workers=n_workers) as pool:
            futures = []
            for target_pdb in target_pdbs:
                future = pool.schedule(
                    _process_single_target,
                    args=[(target_pdb, config_dict), per_worker_ram_gb],
                    timeout=config_dict.get("timeout", MOLECULE_TIMEOUT)
                )
                futures.append((future, target_pdb))
            
            # Collect results with progress bar (same as Polaris)
            desc = f"Timesplit Benchmark"
            
            if not quiet:
                tqdm_config = ProgressConfig.get_tqdm_config(desc)
                tqdm_config['total'] = len(futures)
                progress_bar = tqdm(**tqdm_config)
            else:
                progress_bar = None
            
            for future, pdb_id in futures:
                try:
                    result = future.result()
                    
                    # Stream result to file immediately
                    with output_jsonl.open("a", encoding="utf-8") as fh:
                        json.dump(result, fh)
                        fh.write("\n")
                    
                    # Update counters
                    processed_count += 1
                    if result.get("success"):
                        success_count += 1
                    else:
                        failed_count += 1
                        
                except pebble.ProcessExpired:
                    error_result = {
                        "success": False,
                        "error": f"Process timeout after {config_dict.get('timeout', MOLECULE_TIMEOUT)}s",
                        "pdb_id": pdb_id,
                        "runtime_total": config_dict.get("timeout", MOLECULE_TIMEOUT),
                        "timeout": True,
                    }
                    with output_jsonl.open("a", encoding="utf-8") as fh:
                        json.dump(error_result, fh)
                        fh.write("\n")
                    
                    processed_count += 1
                    failed_count += 1
                    
                except Exception as e:
                    error_result = {
                        "success": False,
                        "error": f"Pipeline error: {str(e)}",
                        "pdb_id": pdb_id,
                        "runtime_total": 0,
                    }
                    with output_jsonl.open("a", encoding="utf-8") as fh:
                        json.dump(error_result, fh)
                        fh.write("\n")
                    
                    processed_count += 1
                    failed_count += 1
                
                # Update progress bar (same as Polaris)
                if progress_bar:
                    success_rate = (success_count / processed_count * 100) if processed_count > 0 else 0
                    postfix = ProgressConfig.get_postfix_format(success_rate, failed_count)
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(1)
            
            if progress_bar:
                progress_bar.close()
    else:
        # Fallback to ProcessPoolExecutor (same as Polaris)
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
            future_to_pdb = {}
            for target_pdb in target_pdbs:
                future = executor.submit(_process_single_target, (target_pdb, config_dict), per_worker_ram_gb)
                future_to_pdb[future] = target_pdb
            
            # Collect results with progress bar (same as Polaris)
            desc = f"Timesplit Benchmark"
            
            if not quiet:
                tqdm_config = ProgressConfig.get_tqdm_config(desc)
                tqdm_config['total'] = len(future_to_pdb)
                progress_bar = tqdm(**tqdm_config)
            else:
                progress_bar = None
            
            for future in as_completed(future_to_pdb):
                pdb_id = future_to_pdb[future]
                
                try:
                    result = future.result(timeout=config_dict.get("timeout", MOLECULE_TIMEOUT))
                    
                    # Stream result to file immediately
                    with output_jsonl.open("a", encoding="utf-8") as fh:
                        json.dump(result, fh)
                        fh.write("\n")
                    
                    # Update counters
                    processed_count += 1
                    if result.get("success"):
                        success_count += 1
                    else:
                        failed_count += 1
                        
                except FutureTimeoutError:
                    error_result = {
                        "success": False,
                        "error": f"Molecule timeout after {config_dict.get('timeout', MOLECULE_TIMEOUT)}s",
                        "pdb_id": pdb_id,
                        "runtime_total": config_dict.get("timeout", MOLECULE_TIMEOUT),
                        "timeout": True,
                    }
                    with output_jsonl.open("a", encoding="utf-8") as fh:
                        json.dump(error_result, fh)
                        fh.write("\n")
                    
                    processed_count += 1
                    failed_count += 1
                    
                except Exception as e:
                    error_result = {
                        "success": False,
                        "error": f"Pipeline error: {str(e)}",
                        "pdb_id": pdb_id,
                        "runtime_total": 0,
                    }
                    with output_jsonl.open("a", encoding="utf-8") as fh:
                        json.dump(error_result, fh)
                        fh.write("\n")
                    
                    processed_count += 1
                    failed_count += 1
                
                # Update progress bar (same as Polaris)
                if progress_bar:
                    success_rate = (success_count / processed_count * 100) if processed_count > 0 else 0
                    postfix = ProgressConfig.get_postfix_format(success_rate, failed_count)
                    progress_bar.set_postfix(postfix)
                    progress_bar.update(1)
            
            if progress_bar:
                progress_bar.close()
    
    # Simple memory cleanup (similar to Polaris)
    gc.collect()
    
    return processed_count, success_count, failed_count


# -----------------------------------------------------------------------------
# Main Benchmark Function
# -----------------------------------------------------------------------------
# This function orchestrates the full benchmark run: loading targets, setting up
# directories, preparing configuration, launching parallel evaluation, and saving
# summary statistics. It is the main entry point for programmatic use.

def run_timesplit_benchmark(
    splits_to_run: List[str] = None,
    n_workers: int = None,
    n_conformers: int = 200,
    template_knn: int = 100,
    max_pdbs: Optional[int] = None,
    data_dir: str = None,
    results_dir: str = None,
    poses_output_dir: str = None,
    similarity_threshold: float = 0.9,
    timeout: int = MOLECULE_TIMEOUT,
    quiet: bool = False,
    ca_rmsd_threshold: float = 10.0,
    # Missing CLI arguments that need to be supported
    unconstrained: bool = False,
    align_metric: str = "combo",
    enable_optimization: bool = False,
    no_realign: bool = False,
    # Memory optimization parameter
    shared_cache_file: str = None,
    per_worker_ram_gb: float = 4.0,
) -> Dict:
    """Run time-split benchmark using Polaris ProcessPoolExecutor pattern."""
    
    if splits_to_run is None:
        splits_to_run = ["test"]  # Default to test split only
    
    # Use maximum performance configuration (all available CPUs)
    n_workers = n_workers or DEFAULT_WORKERS
    
    # Enable shared cache by default for memory optimization
    if shared_cache_file is None:
        shared_cache_file = "shared_ligands.cache"
    
    # Default data directory
    if data_dir is None:
        potential_data_dirs = [
            Path(__file__).resolve().parent.parent / "data",
            Path.cwd() / "data",
            Path.cwd() / "templ_pipeline" / "data",
            Path("data"),
            Path("..") / "data",
        ]
        
        for candidate_path in potential_data_dirs:
            if (candidate_path.exists() and 
                (candidate_path / "ligands" / "templ_processed_ligands_v1.0.0.sdf.gz").exists()):
                data_dir = str(candidate_path)
                break
        
        if data_dir is None:
            data_dir = "/data/pdbbind"  # Fallback
    
    # Default results directory (will be created in workspace by CLI)
    if results_dir is None:
        results_dir = "timesplit_results"
    
    # Setup poses output directory
    if poses_output_dir is None:
        poses_output_dir = Path(results_dir) / "poses"
        Path(poses_output_dir).mkdir(parents=True, exist_ok=True)
    else:
        Path(poses_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load target PDBs from split files
    target_pdbs = []
    for split in splits_to_run:
        try:
            split_pdbs = load_timesplit_pdb_list(split)
            target_pdbs.extend(split_pdbs)
        except Exception as e:
            logging.error(f"Failed to load '{split}' split: {e}")
            return {"success": False, "error": str(e)}
    
    # Remove duplicates while preserving order
    seen = set()
    unique_pdbs = []
    for pdb in target_pdbs:
        if pdb not in seen:
            seen.add(pdb)
            unique_pdbs.append(pdb)
    
    target_pdbs = unique_pdbs
    
    # Apply max_pdbs limit if specified
    if max_pdbs:
        target_pdbs = target_pdbs[:max_pdbs]
    
    if not target_pdbs:
        return {"success": False, "error": "No target PDBs found"}
    
    # Create results directory
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare configuration for workers
    config_dict = {
        "n_conformers": n_conformers,
        "template_knn": template_knn,
        "similarity_threshold": similarity_threshold,
        "timeout": timeout,
        "data_dir": data_dir,
        "poses_output_dir": poses_output_dir,
        "ca_rmsd_threshold": ca_rmsd_threshold,
        # Additional CLI arguments from Polaris parity
        "unconstrained": unconstrained,
        "align_metric": align_metric,
        "enable_optimization": enable_optimization,
        "no_realign": no_realign,
        # Memory optimization
        "shared_cache_file": shared_cache_file,
    }
    
    # Initialize benchmark info
    benchmark_info = {
        "name": "templ_timesplit_benchmark",
        "splits": splits_to_run,
        "timestamp": datetime.now().isoformat(),
        "total_targets": len(target_pdbs),
        "n_conformers": n_conformers,
        "template_knn": template_knn,
        "similarity_threshold": similarity_threshold,
        "ca_rmsd_threshold": ca_rmsd_threshold,
        "n_workers": n_workers,
        "timeout": timeout,
    }
    
    output_jsonl = Path(results_dir) / "results_stream.jsonl"
    
    # Clear previous results
    if output_jsonl.exists():
        output_jsonl.unlink()
    
    # Process targets using Polaris-style direct processing (no chunking)
    if not quiet:
        print(f"Processing {len(target_pdbs)} targets with {n_workers} workers")
    
    processed_count, success_count, failed_count = evaluate_timesplit_targets(
        target_pdbs, config_dict, n_workers, output_jsonl, quiet, per_worker_ram_gb
    )
    
    # Calculate summary metrics
    summary = {
        "total_targets": len(target_pdbs),
        "processed": processed_count,
        "successful": success_count,
        "failed": failed_count,
        "success_rate": (success_count / processed_count * 100) if processed_count > 0 else 0,
        "benchmark_info": benchmark_info,
    }
    
    # Save summary
    summary_file = Path(results_dir) / "benchmark_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    if not quiet:
        print(f"\nBenchmark completed:")
        print(f"  Processed: {processed_count}/{len(target_pdbs)} targets")
        print(f"  Success rate: {summary['success_rate']:.1f}%")
        print(f"  Results saved to: {results_dir}")
    
    return {
        "success": True,
        "summary": summary,
        "results_file": str(output_jsonl),
        "summary_file": str(summary_file),
    }


# -----------------------------------------------------------------------------
# CLI / Main Entrypoint
# -----------------------------------------------------------------------------
# The following functions define the command-line interface, parse arguments,
# and invoke the main benchmark logic. This allows the script to be run as a CLI
# tool with flexible options for splits, worker count, quick mode, etc.

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Time-split benchmark using efficient Polaris-style processing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["test"],
        help="Which splits to evaluate",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (max performance: {DEFAULT_WORKERS})",
    )
    p.add_argument(
        "--n-conformers",
        type=int,
        default=200,
        help="Conformers per query molecule",
    )
    p.add_argument(
        "--template-knn",
        type=int,
        default=100,
        help="KNN for template selection",
    )
    p.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.9,
        help="Similarity threshold (overrides KNN)",
    )
    p.add_argument(
        "--ca-rmsd-threshold", 
        type=float,
        default=10.0,
        help="CA RMSD threshold in Angstroms",
    )
    p.add_argument(
        "--max-pdbs",
        type=int,
        help="Maximum number of PDBs to process (for testing)",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        help="Data directory path",
    )
    p.add_argument(
        "--results-dir",
        type=str,
        help="Results output directory",
    )
    p.add_argument(
        "--poses-dir",
        type=str,
        help="Poses output directory",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=MOLECULE_TIMEOUT,
        help="Timeout per molecule in seconds",
    )
    p.add_argument(
        "--train-only",
        action="store_true",
        help="Evaluate only training set",
    )
    p.add_argument(
        "--val-only",
        action="store_true", 
        help="Evaluate only validation set",
    )
    p.add_argument(
        "--test-only",
        action="store_true",
        help="Evaluate only test set",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 10 conformers, 10 templates, first 10 molecules",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    p.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )
    p.add_argument(
        "--per-worker-ram-gb",
        type=float,
        default=4.0,
        help="Maximum RAM (GiB) per worker process (prevents memory explosion, default: 4.0)",
    )
    
    return p


def main(argv: List[str] = None) -> int:
    """Main entry point for timesplit benchmark."""
    parser = build_parser()
    args = parser.parse_args(argv or sys.argv[1:])
    
    # Set up logging
    setup_benchmark_logging(args.log_level)
    
    # Determine splits to run
    if args.train_only:
        splits_to_run = ["train"]
    elif args.val_only:
        splits_to_run = ["val"]
    elif args.test_only:
        splits_to_run = ["test"]
    else:
        splits_to_run = args.splits
    
    # Apply quick mode overrides
    if args.quick:
        args.n_conformers = 10
        args.template_knn = 10
        args.max_pdbs = 10
        args.n_workers = min(4, args.n_workers)
        print("Quick mode: 10 conformers, 10 templates, 10 molecules, 4 workers")
    
    try:
        result = run_timesplit_benchmark(
            splits_to_run=splits_to_run,
            n_workers=args.n_workers,
            n_conformers=args.n_conformers,
            template_knn=args.template_knn,
            max_pdbs=args.max_pdbs,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            poses_output_dir=args.poses_dir,
            similarity_threshold=args.similarity_threshold,
            timeout=args.timeout,
            quiet=args.quiet,
            ca_rmsd_threshold=args.ca_rmsd_threshold,
            unconstrained=args.unconstrained,
            align_metric=args.align_metric,
            enable_optimization=args.enable_optimization,
            no_realign=args.no_realign,
            per_worker_ram_gb=getattr(args, "per_worker_ram_gb", 4.0),
        )

        # Auto-generate summary files after run
        from pathlib import Path
        from templ_pipeline.benchmark.summary_generator import BenchmarkSummaryGenerator
        
        if result.get("results_file"):
            results_file = Path(result["results_file"])
            if results_file.exists():
                workspace_dir = results_file.parent
                summaries_dir = workspace_dir / "summaries"
                
                try:
                    with open(results_file) as f:
                        results = [json.loads(line) for line in f if line.strip()]
                    generator = BenchmarkSummaryGenerator()
                    summary = generator.generate_unified_summary(results, benchmark_type="timesplit")
                    generator.save_summary_files(summary, summaries_dir)
                    print(f"✓ Summary files written to: {summaries_dir}")
                except Exception as e:
                    print(f"Warning: Failed to generate summary files: {e}")

        return 0 if result["success"] else 1
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())