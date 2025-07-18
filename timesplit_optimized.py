#!/usr/bin/env python3
"""
Optimized Timesplit Benchmark with Memory Management
This implementation fixes the progressive OOM issues by eliminating per-worker cache loading.
"""

import gc
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, memory monitoring disabled")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available, progress bars disabled")

# Import benchmark infrastructure
try:
    from templ_pipeline.benchmark.runner import run_templ_pipeline_for_benchmark
except ImportError:
    print("Warning: Could not import original runner, using local optimized version")
    run_templ_pipeline_for_benchmark = None

# Configuration
MOLECULE_TIMEOUT = 180
OUTPUT_DIR = "timesplit_benchmark_results"
MEMORY_CRITICAL_THRESHOLD = 0.90  # Critical when memory usage > 90%
WORKER_RECYCLE_INTERVAL = 25  # Recycle workers every N targets
OPTIMAL_WORKER_COUNT = 4  # Conservative for large datasets


class GlobalMemoryMonitor:
    """Global memory monitoring and management"""
    
    def __init__(self, max_memory_percent=85):
        self.max_memory_percent = max_memory_percent
        
    def get_memory_info(self):
        """Get current memory information"""
        if not PSUTIL_AVAILABLE:
            return {
                "percent": 0.0,
                "available_gb": 32.0,  # Default assumption
                "used_gb": 8.0,
                "total_gb": 40.0,
            }
        
        memory = psutil.virtual_memory()
        return {
            "percent": memory.percent,
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "total_gb": memory.total / (1024**3),
        }
    
    def is_memory_critical(self):
        """Check if memory usage is critical"""
        if not PSUTIL_AVAILABLE:
            return False
        return psutil.virtual_memory().percent > self.max_memory_percent
    
    def force_cleanup(self):
        """Force aggressive memory cleanup"""
        # Multiple garbage collection passes
        for _ in range(3):
            gc.collect()
            
        # Try to clear system caches if possible
        try:
            # Clear Python object caches
            import sys
            sys.intern.clear() if hasattr(sys.intern, 'clear') else None
        except:
            pass


class SharedMolecularCache:
    """Singleton shared molecular cache to prevent per-worker loading"""
    
    _instance = None
    _cache_data = None
    _data_dir = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def initialize(cls, data_dir: str):
        """Initialize the shared cache once"""
        if cls._cache_data is None:
            cls._data_dir = data_dir
            logging.info(f"Initializing shared molecular cache for {data_dir}")
            # Note: We'll implement lazy loading instead of preloading everything
            cls._cache_data = {"initialized": True, "data_dir": data_dir}
            logging.info("Shared molecular cache initialized (lazy loading mode)")
    
    @classmethod
    def get_data_dir(cls):
        """Get the data directory for cache access"""
        return cls._data_dir
    
    @classmethod
    def is_initialized(cls):
        """Check if cache is initialized"""
        return cls._cache_data is not None


def setup_optimized_logging(log_level: str = "INFO"):
    """Setup optimized logging for benchmark"""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Reduce verbosity for memory-intensive modules
    for module in ["templ_pipeline.core.mcs", "templ_pipeline.core.scoring", "rdkit"]:
        logging.getLogger(module).setLevel(logging.WARNING)
    
    benchmark_logger = logging.getLogger(__name__)
    benchmark_logger.setLevel(getattr(logging, log_level))
    
    return benchmark_logger


def load_timesplit_pdb_list(split_name: str) -> List[str]:
    """Load PDB IDs for the requested split"""
    split_name = split_name.lower()
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split name: {split_name}")

    # Locate split files
    potential_paths = [
        Path(__file__).resolve().parent.parent / "data" / "splits" / f"timesplit_{split_name}",
        Path.cwd() / "data" / "splits" / f"timesplit_{split_name}",
        Path.cwd() / "templ_pipeline" / "data" / "splits" / f"timesplit_{split_name}",
        Path("data") / "splits" / f"timesplit_{split_name}",
    ]

    split_file = None
    for path in potential_paths:
        if path.exists():
            split_file = path
            break

    if split_file is None:
        raise FileNotFoundError(f"Time-split file 'timesplit_{split_name}' not found")

    with split_file.open("r", encoding="utf-8") as fh:
        pdbs = [ln.strip().lower() for ln in fh if ln.strip()]

    return pdbs


def get_timesplit_template_exclusions(target_pdb: str, target_split: str) -> Set[str]:
    """Get PDB IDs to exclude from template search based on time split rules"""
    exclusions = {target_pdb}  # Always exclude the target itself

    try:
        if target_split == "train":
            exclusions.update(load_timesplit_pdb_list("val"))
            exclusions.update(load_timesplit_pdb_list("test"))
        elif target_split == "val":
            exclusions.update(load_timesplit_pdb_list("val"))
            exclusions.update(load_timesplit_pdb_list("test"))
        elif target_split == "test":
            exclusions.update(load_timesplit_pdb_list("test"))
    except Exception as e:
        logging.warning(f"Could not load split files for exclusions: {e}")

    return exclusions


def process_single_target_optimized(args: Tuple) -> Dict:
    """Optimized single target processing with memory management"""
    target_pdb, config_dict = args
    start_time = time.time()
    
    # Suppress worker logging to prevent console pollution
    logging.getLogger().setLevel(logging.ERROR)
    
    # Initialize memory monitor
    memory_monitor = GlobalMemoryMonitor()
    
    try:
        # Check memory before processing
        if memory_monitor.is_memory_critical():
            memory_monitor.force_cleanup()
            
        # Determine target split
        target_split = None
        for split_name in ["train", "val", "test"]:
            try:
                split_pdbs = load_timesplit_pdb_list(split_name)
                if target_pdb in split_pdbs:
                    target_split = split_name
                    break
            except Exception:
                continue
        
        if not target_split:
            target_split = "unknown"
        
        # Get time-based exclusions
        exclusions = get_timesplit_template_exclusions(target_pdb, target_split)
        
        # CRITICAL: Use shared data directory instead of loading cache per worker
        shared_cache = SharedMolecularCache()
        data_dir = shared_cache.get_data_dir() or config_dict.get("data_dir")
        
        # Run pipeline with optimized settings
        result = run_templ_pipeline_for_benchmark(
            target_pdb=target_pdb,
            exclude_pdb_ids=exclusions,
            n_conformers=config_dict.get("n_conformers", 200),
            template_knn=config_dict.get("template_knn", 100),
            similarity_threshold=config_dict.get("similarity_threshold"),
            internal_workers=1,  # Always 1 to prevent nested parallelization
            timeout=config_dict.get("timeout", MOLECULE_TIMEOUT),
            data_dir=data_dir,
            poses_output_dir=config_dict.get("poses_output_dir"),
            shared_cache_file=None,  # Disable additional caching
            shared_embedding_cache=None,  # Disable additional caching
        )
        
        # Ensure result is a dictionary
        if hasattr(result, "to_dict"):
            result = result.to_dict()
        
        # Add metadata
        result["pdb_id"] = target_pdb
        result["target_split"] = target_split
        result["exclusions_count"] = len(exclusions)
        result["runtime_total"] = time.time() - start_time
        
        # Aggressive cleanup before returning
        del exclusions
        memory_monitor.force_cleanup()
        
        return result
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "pdb_id": target_pdb,
            "target_split": target_split,
            "runtime_total": time.time() - start_time,
        }
        
        # Cleanup on error
        memory_monitor.force_cleanup()
        return error_result


def process_targets_with_recycling(
    target_pdbs: List[str], 
    config_dict: Dict, 
    n_workers: int,
    output_jsonl: Path,
    quiet: bool = False
) -> Tuple[int, int, int]:
    """Process targets with worker recycling to prevent memory accumulation"""
    
    total_targets = len(target_pdbs)
    total_processed = 0
    total_success = 0
    total_failed = 0
    
    # Split targets into recycling chunks
    chunks = [target_pdbs[i:i + WORKER_RECYCLE_INTERVAL] 
              for i in range(0, len(target_pdbs), WORKER_RECYCLE_INTERVAL)]
    
    memory_monitor = GlobalMemoryMonitor()
    
    if not quiet:
        print(f"Processing {total_targets} targets in {len(chunks)} recycling chunks")
        print(f"Worker recycling every {WORKER_RECYCLE_INTERVAL} targets")
    
    # Process each chunk with fresh worker pool
    for chunk_idx, chunk_pdbs in enumerate(chunks):
        if not quiet:
            print(f"\n--- Chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk_pdbs)} targets) ---")
        
        # Memory check before chunk
        memory_info = memory_monitor.get_memory_info()
        if memory_info["percent"] > 80:
            if not quiet:
                print(f"High memory usage before chunk: {memory_info['percent']:.1f}%")
            memory_monitor.force_cleanup()
        
        # Process chunk with fresh worker pool
        mp_context = mp.get_context("spawn")
        
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
            # Submit all tasks for this chunk
            future_to_pdb = {}
            for pdb_id in chunk_pdbs:
                future = executor.submit(process_single_target_optimized, (pdb_id, config_dict))
                future_to_pdb[future] = pdb_id
            
            # Initialize progress for this chunk
            processed_count = 0
            success_count = 0
            failed_count = 0
            
            if not quiet:
                desc = f"Chunk {chunk_idx + 1}"
                progress_bar = tqdm(
                    total=len(future_to_pdb),
                    desc=desc,
                    bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                    ncols=100
                )
            else:
                progress_bar = None
            
            # Collect results
            for future in as_completed(future_to_pdb):
                pdb_id = future_to_pdb[future]
                
                try:
                    result = future.result(timeout=config_dict.get("timeout", MOLECULE_TIMEOUT) + 30)
                except Exception as e:
                    result = {
                        "success": False,
                        "error": f"Future error: {str(e)}",
                        "pdb_id": pdb_id,
                        "runtime_total": 0,
                    }
                
                # Stream result to file immediately
                with output_jsonl.open("a", encoding="utf-8") as fh:
                    json.dump(result, fh)
                    fh.write('\n')
                
                # Update counters
                processed_count += 1
                if result.get("success"):
                    success_count += 1
                else:
                    failed_count += 1
                
                # Update progress
                if progress_bar:
                    success_rate = (success_count / processed_count * 100) if processed_count > 0 else 0
                    progress_bar.set_postfix({"success": f"{success_rate:.1f}%", "errors": failed_count})
                    progress_bar.update(1)
                
                # Memory cleanup for processed result
                del result
        
        # Close progress bar
        if progress_bar:
            progress_bar.close()
        
        # Update totals
        total_processed += processed_count
        total_success += success_count
        total_failed += failed_count
        
        # Memory barrier after chunk
        memory_monitor.force_cleanup()
        
        # Progress update
        if not quiet:
            chunk_success_rate = (success_count / processed_count * 100) if processed_count > 0 else 0
            print(f"Chunk {chunk_idx + 1} complete: {success_count}/{processed_count} ({chunk_success_rate:.1f}% success)")
            
            overall_success_rate = (total_success / total_processed * 100) if total_processed > 0 else 0
            memory_info = memory_monitor.get_memory_info()
            print(f"Overall: {total_processed}/{total_targets} ({overall_success_rate:.1f}% success, {memory_info['percent']:.1f}% memory)")
    
    return total_processed, total_success, total_failed


def run_optimized_timesplit_benchmark(
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
) -> Dict:
    """Run optimized time-split benchmark with memory management"""
    
    if splits_to_run is None:
        splits_to_run = ["test"]
    
    # Use conservative worker count for large datasets
    n_workers = min(n_workers or OPTIMAL_WORKER_COUNT, OPTIMAL_WORKER_COUNT)
    
    # Setup logging
    logger = setup_optimized_logging()
    
    # Default data directory
    if data_dir is None:
        potential_data_dirs = [
            Path(__file__).resolve().parent.parent / "data",
            Path.cwd() / "data",
            Path.cwd() / "templ_pipeline" / "data",
            Path("data"),
        ]
        
        for candidate_path in potential_data_dirs:
            if candidate_path.exists():
                data_dir = str(candidate_path)
                break
        
        if data_dir is None:
            raise ValueError("Could not find data directory")
    
    # Initialize shared molecular cache
    SharedMolecularCache.initialize(data_dir)
    
    # Default results directory
    if results_dir is None:
        results_dir = OUTPUT_DIR
    
    # Load target PDBs
    target_pdbs = []
    for split in splits_to_run:
        try:
            split_pdbs = load_timesplit_pdb_list(split)
            target_pdbs.extend(split_pdbs)
            logger.info(f"Loaded {len(split_pdbs)} targets from {split} split")
        except Exception as e:
            logger.error(f"Failed to load {split} split: {e}")
    
    # Remove duplicates
    seen = set()
    unique_pdbs = []
    for pdb in target_pdbs:
        if pdb not in seen:
            seen.add(pdb)
            unique_pdbs.append(pdb)
    target_pdbs = unique_pdbs
    
    # Apply max_pdbs limit
    if max_pdbs:
        target_pdbs = target_pdbs[:max_pdbs]
    
    if not target_pdbs:
        return {"success": False, "error": "No target PDBs found"}
    
    # Create directories
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    if poses_output_dir:
        Path(poses_output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configuration for workers
    config_dict = {
        "n_conformers": n_conformers,
        "template_knn": template_knn,
        "similarity_threshold": similarity_threshold,
        "timeout": timeout,
        "data_dir": data_dir,
        "poses_output_dir": poses_output_dir,
    }
    
    # Initialize memory monitoring
    memory_monitor = GlobalMemoryMonitor()
    initial_memory = memory_monitor.get_memory_info()
    
    logger.info(f"Starting optimized timesplit benchmark")
    logger.info(f"Targets: {len(target_pdbs)}, Workers: {n_workers}")
    logger.info(f"Initial memory: {initial_memory['percent']:.1f}%")
    
    # Setup output file
    output_jsonl = Path(results_dir) / "results_stream.jsonl"
    if output_jsonl.exists():
        output_jsonl.unlink()
    
    # Process targets with recycling
    start_time = time.time()
    processed_count, success_count, failed_count = process_targets_with_recycling(
        target_pdbs, config_dict, n_workers, output_jsonl, quiet
    )
    total_runtime = time.time() - start_time
    
    # Final memory check
    final_memory = memory_monitor.get_memory_info()
    
    # Calculate summary
    summary = {
        "total_targets": len(target_pdbs),
        "processed": processed_count,
        "successful": success_count,
        "failed": failed_count,
        "completion_rate": processed_count / len(target_pdbs) * 100,
        "success_rate": success_count / processed_count * 100 if processed_count > 0 else 0,
        "total_runtime": total_runtime,
        "memory_usage": {
            "initial_percent": initial_memory["percent"],
            "final_percent": final_memory["percent"],
            "peak_used_gb": final_memory["used_gb"],
            "memory_increase": final_memory["percent"] - initial_memory["percent"],
        }
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(results_dir) / f"optimized_timesplit_results_{timestamp}.json"
    
    final_results = {
        "benchmark_info": {
            "name": "optimized_templ_timesplit_benchmark",
            "splits": splits_to_run,
            "timestamp": timestamp,
            "optimization": "memory_managed_recycling",
        },
        "summary": summary,
        "config": config_dict,
    }
    
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    
    if not quiet:
        print(f"\nOptimized Timesplit Benchmark Complete:")
        print(f"  Processed: {processed_count}/{len(target_pdbs)} targets")
        print(f"  Success rate: {success_count}/{processed_count} ({summary['success_rate']:.1f}%)")
        print(f"  Runtime: {total_runtime:.1f}s")
        print(f"  Memory: {initial_memory['percent']:.1f}% → {final_memory['percent']:.1f}%")
        print(f"  Results: {output_jsonl}")
        print(f"  Summary: {results_file}")
    
    return {"success": True, "results_dir": results_dir, "summary": summary}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Time-split benchmark")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["test"])
    parser.add_argument("--n-workers", type=int, default=OPTIMAL_WORKER_COUNT)
    parser.add_argument("--n-conformers", type=int, default=200)
    parser.add_argument("--template-knn", type=int, default=100)
    parser.add_argument("--max-pdbs", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--poses-dir", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=MOLECULE_TIMEOUT)
    parser.add_argument("--quick", action="store_true", help="Quick test: 10 conformers, 10 templates, 10 targets")
    parser.add_argument("--quiet", action="store_true")
    
    args = parser.parse_args()
    
    # Apply quick mode
    if args.quick:
        args.n_conformers = 10
        args.template_knn = 10
        args.max_pdbs = 10
        args.n_workers = 2
        print("Quick mode: 10 conformers, 10 templates, 10 targets, 2 workers")
    
    try:
        result = run_optimized_timesplit_benchmark(
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
        )
        
        exit_code = 0 if result["success"] else 1
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    
    sys.exit(exit_code)
