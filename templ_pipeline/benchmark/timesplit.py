"""
Time-Split Benchmark for TEMPL Pipeline

This module provides time-split benchmarking functionality following the successful
polaris benchmark pattern. It uses ProcessPoolExecutor with spawn context for 
reliable memory management and prevents OOM issues.

Key features:
- Simple 2-file architecture (timesplit.py + runner.py)
- ProcessPoolExecutor with spawn context for process isolation
- Conservative resource limits to prevent OOM
- Time-based exclusions for train/val/test splits
- Clean CLI integration with workspace organization
- Hardware-aware configuration
"""

import argparse
import gc
import json
import logging
import multiprocessing as mp
import psutil
import sys
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

# Import hardware detection
try:
    from templ_pipeline.core.hardware import get_suggested_worker_config
    HARDWARE_CONFIG = get_suggested_worker_config()
    DEFAULT_WORKERS = min(8, HARDWARE_CONFIG["n_workers"])  # Conservative cap
except ImportError:
    logging.warning("Hardware detection not available, using conservative default")
    DEFAULT_WORKERS = 8

# Import benchmark infrastructure
from templ_pipeline.benchmark.runner import run_templ_pipeline_for_benchmark

# Configuration - use conservative defaults like polaris
MOLECULE_TIMEOUT = 180  # 3 minutes like polaris
OUTPUT_DIR = "timesplit_benchmark_results"

# Memory management configuration
MEMORY_CLEANUP_INTERVAL = 10  # Force cleanup every N targets
MEMORY_WARNING_THRESHOLD = 0.85  # Warning when memory usage > 85%
MEMORY_CRITICAL_THRESHOLD = 0.95  # Critical when memory usage > 95%
AGGRESSIVE_CLEANUP_INTERVAL = 5  # Aggressive cleanup when memory is high
CACHE_SIZE_LIMIT_MB = 1024  # Maximum cache size in MB (1GB)

# Chunked processing configuration
CHUNK_SIZE = 50  # Number of targets to process in each chunk
MEMORY_BARRIER_THRESHOLD = 0.80  # Memory threshold to trigger barrier

# Progress Bar Configuration (copied from polaris)
class ProgressConfig:
    """Configuration for progress bars"""

    @staticmethod
    def get_bar_format():
        return "{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

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


def setup_benchmark_logging(log_level: str = "INFO", workspace_dir: Optional[Path] = None):
    """Configure logging for benchmark with file-only output and clean progress bars"""
    from templ_pipeline.core.benchmark_logging import create_benchmark_logger
    
    if workspace_dir:
        return create_benchmark_logger("timesplit")
    else:
        # Fallback to original behavior for compatibility
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level))
        
        # Keep benchmark logger at INFO level
        benchmark_logger = logging.getLogger(__name__)
        benchmark_logger.setLevel(getattr(logging, log_level))
        
        return benchmark_logger


# -----------------------------------------------------------------------------
# Split loading utilities
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Core worker function (simplified like polaris)
# -----------------------------------------------------------------------------

def _process_single_target(args: Tuple) -> Dict:
    """Process a single target PDB in an isolated subprocess.
    
    Uses the same pattern as polaris benchmark for reliable memory management.
    Includes explicit memory cleanup and monitoring to prevent accumulation.
    """
    target_pdb, config_dict = args
    
    start_time = time.time()
    
    # Monitor initial memory state
    process = psutil.Process()
    initial_memory = process.memory_info().rss / (1024**3)  # GB
    
    # Suppress worker logging to prevent console pollution
    from templ_pipeline.core.benchmark_logging import suppress_worker_logging
    suppress_worker_logging()
    
    # Initialize variables for cleanup
    target_split = None
    exclusions = set()
    result = None
    
    try:
        # Determine target split
        for split_name in ["train", "val", "test"]:
            try:
                split_pdbs = load_timesplit_pdb_list(split_name)
                if target_pdb.lower() in [pdb.lower() for pdb in split_pdbs]:
                    target_split = split_name
                    break
                # Explicit cleanup of split_pdbs
                del split_pdbs
                gc.collect()  # Force cleanup
            except Exception:
                continue
        
        if not target_split:
            target_split = "unknown"
        
        # Get time-based exclusions
        exclusions = get_timesplit_template_exclusions(target_pdb, target_split)
        
        # Monitor memory before pipeline execution
        pre_pipeline_memory = process.memory_info().rss / (1024**3)
        if pre_pipeline_memory > 4.0:  # Warning threshold
            logging.warning(f"High memory usage before pipeline: {pre_pipeline_memory:.1f}GB for {target_pdb}")
        
        # Run pipeline using benchmark infrastructure
        result = run_templ_pipeline_for_benchmark(
            target_pdb=target_pdb,
            exclude_pdb_ids=exclusions,
            n_conformers=config_dict.get("n_conformers", 200),
            template_knn=config_dict.get("template_knn", 100),
            similarity_threshold=config_dict.get("similarity_threshold"),
            internal_workers=1,  # Always 1 to prevent nested parallelization
            timeout=config_dict.get("timeout", MOLECULE_TIMEOUT),
            data_dir=config_dict.get("data_dir"),
            poses_output_dir=config_dict.get("poses_output_dir"),
            unconstrained=config_dict.get("unconstrained", False),
            align_metric=config_dict.get("align_metric", "combo"),
            enable_optimization=config_dict.get("enable_optimization", False),
            no_realign=config_dict.get("no_realign", False),
        )
        
        # Ensure result is a dictionary
        if hasattr(result, "to_dict"):
            result = result.to_dict()
        
        # Add metadata
        result["pdb_id"] = target_pdb
        result["target_split"] = target_split
        result["exclusions_count"] = len(exclusions)
        result["runtime_total"] = time.time() - start_time
        
        # Monitor final memory state
        final_memory = process.memory_info().rss / (1024**3)
        result["memory_usage"] = {
            "initial_gb": initial_memory,
            "final_gb": final_memory,
            "peak_delta_gb": final_memory - initial_memory
        }
        
        # Explicit cleanup before returning
        del exclusions
        gc.collect()  # Force garbage collection in worker
        
        return result
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "pdb_id": target_pdb,
            "target_split": target_split,
            "runtime_total": time.time() - start_time,
            "memory_usage": {
                "initial_gb": initial_memory,
                "final_gb": process.memory_info().rss / (1024**3),
                "peak_delta_gb": process.memory_info().rss / (1024**3) - initial_memory
            }
        }
        
        # Cleanup on error
        if exclusions:
            del exclusions
        if result:
            del result
        gc.collect()  # Force garbage collection on error
        
        return error_result
    
    finally:
        # Final cleanup - clear any remaining references
        try:
            # Clear global caches that might accumulate
            _clear_worker_caches()
        except Exception:
            pass  # Don't let cleanup errors crash the worker


def _clear_worker_caches():
    """Clear worker-specific caches to prevent memory accumulation."""
    try:
        # Clear RDKit caches if available
        import rdkit
        if hasattr(rdkit, 'Chem'):
            # Clear any RDKit molecule caches
            pass
    except ImportError:
        pass
    
    # Force garbage collection
    gc.collect()


def _get_memory_info() -> Dict[str, float]:
    """Get detailed memory information for monitoring."""
    memory = psutil.virtual_memory()
    return {
        "total_gb": memory.total / (1024**3),
        "available_gb": memory.available / (1024**3),
        "used_gb": memory.used / (1024**3),
        "percent": memory.percent,
        "free_gb": memory.free / (1024**3),
    }


def _should_trigger_cleanup(processed_count: int, memory_percent: float) -> bool:
    """Determine if cleanup should be triggered based on memory usage and interval."""
    # Standard cleanup interval
    if processed_count % MEMORY_CLEANUP_INTERVAL == 0:
        return True
    
    # Aggressive cleanup when memory is high
    if memory_percent > MEMORY_WARNING_THRESHOLD * 100:
        return processed_count % AGGRESSIVE_CLEANUP_INTERVAL == 0
    
    # Emergency cleanup when memory is critical
    if memory_percent > MEMORY_CRITICAL_THRESHOLD * 100:
        return True
    
    return False


def _clear_worker_caches():
    """Clear any global caches that might accumulate in worker processes."""
    cleared_components = []
    
    try:
        # Clear RDKit molecule cache if it exists
        from templ_pipeline.core.utils import clear_global_molecule_cache
        clear_global_molecule_cache()
        cleared_components.append("molecule_cache")
    except ImportError:
        pass
    
    try:
        # Clear embedding caches if they exist
        from templ_pipeline.core.embedding import clear_embedding_cache
        # Clear all cache types with size limits
        clear_embedding_cache(clear_model_cache=True, clear_disk_cache=False, clear_memory_cache=True)
        cleared_components.append("embedding_cache")
    except ImportError:
        pass
    
    # Force garbage collection (multiple passes for stubborn objects)
    gc.collect()
    gc.collect()
    
    return cleared_components


def _aggressive_memory_cleanup():
    """Perform aggressive memory cleanup when memory usage is critical."""
    logging.warning("Performing aggressive memory cleanup due to high memory usage")
    
    # Clear all caches
    _clear_worker_caches()
    
    # Multiple garbage collection passes
    for _ in range(3):
        gc.collect()
    
    # Try to clear disk caches as well if memory is critically low
    try:
        from templ_pipeline.core.embedding import clear_embedding_cache
        clear_embedding_cache(clear_model_cache=True, clear_disk_cache=True, clear_memory_cache=True)
    except ImportError:
        pass
    
    # Final garbage collection
    gc.collect()


class ScopedResourceManager:
    """Context manager for scoped resource management during benchmarks."""
    
    def __init__(self, cleanup_interval: int = MEMORY_CLEANUP_INTERVAL):
        self.cleanup_interval = cleanup_interval
        self.initial_memory = None
        self.resources_initialized = False
    
    def __enter__(self):
        """Initialize scoped resources."""
        self.initial_memory = psutil.virtual_memory().percent
        logging.info(f"Initializing scoped resources (initial memory: {self.initial_memory:.1f}%)")
        
        # Initialize resources with scoped management
        self._initialize_scoped_resources()
        self.resources_initialized = True
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup all scoped resources."""
        if self.resources_initialized:
            logging.info("Cleaning up scoped resources")
            self._cleanup_scoped_resources()
            
            final_memory = psutil.virtual_memory().percent
            logging.info(f"Scoped cleanup complete (final memory: {final_memory:.1f}%)")
    
    def _initialize_scoped_resources(self):
        """Initialize resources with scoped management."""
        try:
            # Pre-initialize caches to avoid repeated initialization
            from templ_pipeline.core.utils import initialize_global_molecule_cache
            initialize_global_molecule_cache()
            
            # Pre-initialize embedding manager if needed
            try:
                from templ_pipeline.core.embedding import EmbeddingManager  # noqa: F401
                # Don't create instance, just ensure class is loaded
            except ImportError:
                pass
            
        except ImportError:
            logging.warning("Could not initialize some scoped resources")
    
    def _cleanup_scoped_resources(self):
        """Cleanup all scoped resources."""
        try:
            # Clear all caches
            _clear_worker_caches()
            
            # Force singleton cleanup
            self._cleanup_singletons()
            
            # Multiple garbage collection passes
            for _ in range(2):
                gc.collect()
                
        except Exception as e:
            logging.warning(f"Error during scoped resource cleanup: {e}")
    
    def _cleanup_singletons(self):
        """Force cleanup of singleton instances."""
        try:
            from templ_pipeline.core.embedding import EmbeddingManager
            
            # Reset singleton if it exists
            if hasattr(EmbeddingManager, '_instance') and EmbeddingManager._instance is not None:
                EmbeddingManager._instance = None
                EmbeddingManager._initialized = False
                logging.info("Reset EmbeddingManager singleton")
                
        except ImportError:
            pass
        except Exception as e:
            logging.warning(f"Error cleaning up singletons: {e}")
    
    def periodic_cleanup(self, processed_count: int):
        """Perform periodic cleanup based on processed count."""
        if processed_count % self.cleanup_interval == 0:
            current_memory = psutil.virtual_memory().percent
            
            if current_memory > MEMORY_WARNING_THRESHOLD * 100:
                logging.info(f"Periodic cleanup triggered (memory: {current_memory:.1f}%)")
                self._cleanup_scoped_resources()
                
                # Re-initialize if needed
                self._initialize_scoped_resources()
                
                new_memory = psutil.virtual_memory().percent
                logging.info(f"Periodic cleanup complete (memory: {new_memory:.1f}%)")


def _process_targets_in_chunks(
    target_pdbs: List[str], 
    config_dict: Dict, 
    n_workers: int,
    output_jsonl: Path,
    quiet: bool = False
) -> Tuple[int, int, int]:
    """Process targets in chunks with memory barriers between chunks."""
    
    total_targets = len(target_pdbs)
    total_processed = 0
    total_success = 0
    total_failed = 0
    
    # Split targets into chunks
    chunks = [target_pdbs[i:i + CHUNK_SIZE] for i in range(0, len(target_pdbs), CHUNK_SIZE)]
    
    if not quiet:
        print(f"Processing {total_targets} targets in {len(chunks)} chunks of {CHUNK_SIZE} targets each")
    
    # Process each chunk
    for chunk_idx, chunk_pdbs in enumerate(chunks):
        if not quiet:
            print(f"\n--- Processing chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk_pdbs)} targets) ---")
        
        # Memory barrier before chunk
        _memory_barrier(f"Before chunk {chunk_idx + 1}")
        
        # Process chunk
        processed, success, failed = _process_chunk(
            chunk_pdbs, config_dict, n_workers, output_jsonl, quiet, chunk_idx + 1
        )
        
        # Update totals
        total_processed += processed
        total_success += success
        total_failed += failed
        
        # Memory barrier after chunk
        _memory_barrier(f"After chunk {chunk_idx + 1}")
        
        # Progress update
        if not quiet:
            success_rate = (success / processed * 100) if processed > 0 else 0
            print(f"Chunk {chunk_idx + 1} complete: {success}/{processed} ({success_rate:.1f}% success)")
            
            overall_success_rate = (total_success / total_processed * 100) if total_processed > 0 else 0
            print(f"Overall progress: {total_processed}/{total_targets} ({overall_success_rate:.1f}% success)")
    
    return total_processed, total_success, total_failed


def _memory_barrier(context: str):
    """Create a memory barrier - force cleanup and wait for memory to stabilize."""
    memory_before = psutil.virtual_memory().percent
    
    # Force aggressive cleanup
    _aggressive_memory_cleanup()
    
    # Additional cleanup if memory is still high
    if memory_before > MEMORY_BARRIER_THRESHOLD * 100:
        logging.warning(f"Memory barrier triggered - high memory usage: {memory_before:.1f}%")
        
        # Multiple cleanup passes
        for _ in range(3):
            _clear_worker_caches()
            gc.collect()
            time.sleep(0.1)  # Brief pause to allow cleanup
        
        memory_after = psutil.virtual_memory().percent
        logging.info(f"Memory barrier complete ({context}): {memory_before:.1f}% → {memory_after:.1f}%")
    else:
        logging.info(f"Memory barrier ({context}): {memory_before:.1f}% memory usage")


def _process_chunk(
    chunk_pdbs: List[str], 
    config_dict: Dict, 
    n_workers: int,
    output_jsonl: Path,
    quiet: bool,
    chunk_number: int
) -> Tuple[int, int, int]:
    """Process a single chunk of targets."""
    
    mp_context = mp.get_context("spawn")
    
    # Add worker recycling to prevent memory accumulation
    recycle_interval = max(10, len(chunk_pdbs) // 4)  # Recycle workers every 25% of chunk
    current_batch = 0
    
    with ScopedResourceManager() as resource_manager:
        
        # Process in batches with worker recycling
        batch_size = min(recycle_interval, len(chunk_pdbs))
        for batch_start in range(0, len(chunk_pdbs), batch_size):
            batch_end = min(batch_start + batch_size, len(chunk_pdbs))
            batch_pdbs = chunk_pdbs[batch_start:batch_end]
            current_batch += 1
            
            logging.info(f"Processing batch {current_batch} with {len(batch_pdbs)} targets (worker recycling)")
            
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
        
                # Submit all tasks for this batch
                future_to_pdb = {}
                for pdb_id in batch_pdbs:
                    future = executor.submit(_process_single_target, (pdb_id, config_dict))
                    future_to_pdb[future] = pdb_id
                
                # Initialize progress tracking for this batch
                batch_processed = 0
                batch_success = 0
                batch_failed = 0
                
                if not quiet:
                    desc = f"Chunk {chunk_number} - Batch {current_batch}"
                    tqdm_config = ProgressConfig.get_tqdm_config(desc)
                    tqdm_config['total'] = len(future_to_pdb)
                    progress_bar = tqdm(**tqdm_config)
                else:
                    progress_bar = None
                
                # Collect results for this batch
                for future in as_completed(future_to_pdb):
                    pdb_id = future_to_pdb[future]
                    
                    try:
                        result = future.result(timeout=config_dict.get("timeout", MOLECULE_TIMEOUT))
                        
                    except Exception as e:
                        result = {
                            "success": False,
                            "error": f"Pipeline error: {str(e)}",
                            "pdb_id": pdb_id,
                            "runtime_total": 0,
                            "timeout": False,
                        }
                    
                    # Stream result to file immediately
                    with output_jsonl.open("a", encoding="utf-8") as fh:
                        json.dump(result, fh)
                        fh.write("\n")
                    
                    # Update batch counters
                    batch_processed += 1
                    if result.get("success"):
                        batch_success += 1
                    else:
                        batch_failed += 1
                    
                    # Memory management
                    del result
                    
                    # Enhanced memory monitoring and cleanup
                    memory_info = _get_memory_info()
                    resource_manager.periodic_cleanup(batch_processed)
                    
                    # Update progress bar
                    if progress_bar:
                        success_rate = (batch_success / batch_processed * 100) if batch_processed > 0 else 0
                        postfix = ProgressConfig.get_postfix_format(success_rate, batch_failed)
                        postfix["mem"] = f"{memory_info['percent']:.1f}%"
                        postfix["free"] = f"{memory_info['available_gb']:.1f}GB"
                        progress_bar.set_postfix(postfix)
                        progress_bar.update(1)
                
                if progress_bar:
                    progress_bar.close()
                
                # Log batch completion and force garbage collection
                logging.info(f"Batch {current_batch} completed: {batch_success}/{batch_processed} successful")
                gc.collect()
            
            # End of executor context - workers are recycled here
    
    # Return total counts across all batches
    return batch_processed, batch_success, batch_failed


# -----------------------------------------------------------------------------
# Main benchmark function using polaris pattern
# -----------------------------------------------------------------------------

def run_timesplit_benchmark(
    splits_to_run: List[str] = None,
    n_workers: int = None,
    n_conformers: int = 200,
    template_knn: int = 100,
    max_pdbs: Optional[int] = None,
    data_dir: str = None,
    results_dir: str = None,
    poses_output_dir: str = None,
    similarity_threshold: float = None,
    timeout: int = MOLECULE_TIMEOUT,
    quiet: bool = False,
    use_chunked_processing: bool = True,
    unconstrained: bool = False,
    align_metric: str = "combo",
    enable_optimization: bool = False,
    no_realign: bool = False,
) -> Dict:
    """Run time-split benchmark using polaris ProcessPoolExecutor pattern."""
    
    if splits_to_run is None:
        splits_to_run = ["test"]  # Default to test split only
    
    # Apply conservative worker limits (like polaris)
    n_workers = min(n_workers or DEFAULT_WORKERS, 16)  # Cap at 16 workers
    
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
    
    # Default results directory
    if results_dir is None:
        results_dir = OUTPUT_DIR
    
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
    
    # Setup poses output directory
    if poses_output_dir:
        Path(poses_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare configuration for workers
    config_dict = {
        "n_conformers": n_conformers,
        "template_knn": template_knn,
        "similarity_threshold": similarity_threshold,
        "timeout": timeout,
        "data_dir": data_dir,
        "poses_output_dir": poses_output_dir,
        "unconstrained": unconstrained,
        "align_metric": align_metric,
        "enable_optimization": enable_optimization,
        "no_realign": no_realign,
    }
    
    # Initialize benchmark info (no results accumulation)
    benchmark_info = {
        "name": "templ_timesplit_benchmark",
        "splits": splits_to_run,
        "timestamp": datetime.now().isoformat(),
        "total_targets": len(target_pdbs),
        "n_conformers": n_conformers,
        "template_knn": template_knn,
        "n_workers": n_workers,
        "timeout": timeout,
    }
    
    # Initialize memory monitoring
    initial_memory = psutil.virtual_memory().percent
    logging.info(f"Initial memory usage: {initial_memory:.1f}%")
    
    output_jsonl = Path(results_dir) / "results_stream.jsonl"
    
    # Clear previous results
    if output_jsonl.exists():
        output_jsonl.unlink()
    
    # Process targets using chunked processing or standard processing
    if use_chunked_processing and len(target_pdbs) > CHUNK_SIZE:
        if not quiet:
            print(f"Using chunked processing with {CHUNK_SIZE} targets per chunk")
        
        processed_count, success_count, failed_count = _process_targets_in_chunks(
            target_pdbs, config_dict, n_workers, output_jsonl, quiet
        )
    else:
        # Standard processing for smaller datasets
        if not quiet:
            print(f"Using standard processing for {len(target_pdbs)} targets")
        
        processed_count, success_count, failed_count = _process_chunk(
            target_pdbs, config_dict, n_workers, output_jsonl, quiet, 1
        )
    
    # Calculate summary metrics (no results accumulation)
    final_memory = psutil.virtual_memory().percent
    summary = {
        "total_targets": len(target_pdbs),
        "processed": processed_count,
        "successful": success_count,
        "failed": failed_count,
        "completion_rate": processed_count / len(target_pdbs) * 100,
        "success_rate": success_count / processed_count * 100 if processed_count > 0 else 0,
        "memory_usage": {
            "initial_percent": initial_memory,
            "final_percent": final_memory,
            "memory_increase": final_memory - initial_memory,
        }
    }
    
    # Save final results (lightweight - no accumulated results)
    results_file = Path(results_dir) / f"timesplit_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    final_results = {
        "benchmark_info": benchmark_info,
        "summary": summary,
        "note": "Individual results are streamed to results_stream.jsonl to prevent memory accumulation"
    }
    
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    
    if not quiet:
        print(f"\nTimesplit benchmark completed:")
        print(f"  Processed: {processed_count}/{len(target_pdbs)} targets")
        print(f"  Success rate: {success_count}/{processed_count} ({success_count/processed_count*100:.1f}%)")
        print(f"  Memory usage: {initial_memory:.1f}% → {final_memory:.1f}% (Δ{final_memory-initial_memory:+.1f}%)")
        print(f"  Results streamed to: {output_jsonl}")
        print(f"  Summary saved to: {results_file}")
    
    return {"success": True, "results_dir": results_dir, "memory_info": summary["memory_usage"]}


# -----------------------------------------------------------------------------
# CLI interface
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for timesplit benchmark."""
    parser = argparse.ArgumentParser(
        description="Time-split benchmark for TEMPL pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["test"],
        help="Which splits to run",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--n-conformers",
        type=int,
        default=200,
        help="Number of conformers per molecule",
    )
    parser.add_argument(
        "--template-knn",
        type=int,
        default=100,
        help="Number of template neighbors to consider",
    )
    parser.add_argument(
        "--max-pdbs",
        type=int,
        default=None,
        help="Maximum number of PDBs to process (for testing)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing TEMPL data files",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to save results",
    )
    parser.add_argument(
        "--poses-dir",
        type=str,
        default=None,
        help="Directory to save predicted poses",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=MOLECULE_TIMEOUT,
        help="Timeout per molecule in seconds",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 10 conformers, 10 templates, first 10 molecules",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--chunked-processing",
        action="store_true",
        default=True,
        help="Use chunked processing with memory barriers (default: True)",
    )
    parser.add_argument(
        "--no-chunked-processing",
        action="store_false",
        dest="chunked_processing",
        help="Disable chunked processing",
    )
    
    # Ablation study arguments
    parser.add_argument(
        "--unconstrained",
        action="store_true",
        help="Skip MCS and constrained embedding (unconstrained conformer generation)",
    )
    parser.add_argument(
        "--align-metric",
        choices=["shape", "color", "combo"],
        default="combo",
        help="Shape alignment metric (default: combo)",
    )
    parser.add_argument(
        "--enable-optimization",
        action="store_true",
        help="Enable force field optimization (MMFF/UFF)",
    )
    parser.add_argument(
        "--no-realign",
        action="store_true",
        help="Use AlignMol scores only for ranking, disable pose realignment",
    )
    
    return parser


def main(argv: List[str] = None) -> int:
    """Main entry point for timesplit benchmark."""
    parser = build_parser()
    args = parser.parse_args(argv or sys.argv[1:])
    
    # Apply quick mode overrides
    if args.quick:
        args.n_conformers = 10
        args.template_knn = 10
        args.max_pdbs = 10
        args.n_workers = min(2, args.n_workers)
        print("Quick mode: 10 conformers, 10 templates, 10 molecules, 2 workers")
    
    try:
        result = run_timesplit_benchmark(
            splits_to_run=args.splits,
            n_workers=args.n_workers,
            n_conformers=args.n_conformers,
            template_knn=args.template_knn,
            max_pdbs=args.max_pdbs,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            poses_output_dir=args.poses_dir,
            timeout=args.timeout,
            quiet=args.quiet,
            use_chunked_processing=args.chunked_processing,
            unconstrained=args.unconstrained,
            align_metric=args.align_metric,
            enable_optimization=args.enable_optimization,
            no_realign=args.no_realign,
        )
        
        return 0 if result["success"] else 1
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())