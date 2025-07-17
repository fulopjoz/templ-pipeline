"""Streaming version of the timesplit benchmark.

Each target PDB is processed in a short-lived worker process (one task per
process).  This guarantees that memory used by RDKit objects and any module-
level caches is released when the process exits, preventing accumulation of
resident set size (RSS) across the full run.

The coordinator enqueues targets, consumes results lazily, and appends them to
newline-delimited JSON (jsonlines) and an optional CSV summary.  No large data
structures are retained in memory.

Usage (programmatic):

>>> from templ_pipeline.benchmark.timesplit_stream import run_timesplit_streaming
>>> run_timesplit_streaming(target_pdbs, data_dir="/data/pdbbind")

A CLI helper can later wrap this function.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

# Optional – used only if available
try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover – psutil is optional
    psutil = None  # type: ignore

from templ_pipeline.benchmark.runner import run_templ_pipeline_for_benchmark
from templ_pipeline.core.hardware import get_optimized_worker_config

# Public symbols for 'from ... import *' users
__all__: List[str] = []

###############################################################################
# Chemical validation for data leakage detection
###############################################################################

def validate_template_similarity(rmsd_values: Dict[str, Dict[str, float]], 
                                pdb_id: str, 
                                template_info: Dict = None) -> Dict[str, any]:
    """Validate template similarity scores to detect potential data leakage.
    
    Checks for suspiciously high similarity scores and low RMSD values that
    might indicate the same chemical structure appearing in query and template.
    
    Parameters
    ----------
    rmsd_values : Dict[str, Dict[str, float]]
        RMSD results from pipeline with format {metric: {rmsd: float, score: float}}
    pdb_id : str
        The target PDB ID being processed
    template_info : Dict, optional
        Additional template information if available
        
    Returns
    -------
    Dict[str, any]
        Validation results with warnings and flags
    """
    validation_result = {
        "has_warnings": False,
        "warnings": [],
        "potential_data_leakage": False,
        "perfect_matches": []
    }
    
    for metric, values in rmsd_values.items():
        rmsd = values.get("rmsd", float('inf'))
        score = values.get("score", 0.0)
        
        # Flag extremely low RMSD values (< 0.05Å suggests identical structures)
        if rmsd < 0.05:
            validation_result["potential_data_leakage"] = True
            validation_result["perfect_matches"].append({
                "metric": metric,
                "rmsd": rmsd,
                "score": score,
                "reason": "sub_angstrom_rmsd"
            })
            warning = f"POTENTIAL DATA LEAKAGE: {pdb_id} {metric} RMSD={rmsd:.4f}Å (< 0.05Å threshold)"
            validation_result["warnings"].append(warning)
            
        # Flag extremely high similarity scores (> 0.999 suggests identical structures)
        if score > 0.999:
            validation_result["potential_data_leakage"] = True
            validation_result["perfect_matches"].append({
                "metric": metric,
                "rmsd": rmsd,
                "score": score,
                "reason": "perfect_similarity"
            })
            warning = f"POTENTIAL DATA LEAKAGE: {pdb_id} {metric} similarity={score:.6f} (> 0.999 threshold)"
            validation_result["warnings"].append(warning)
    
    if validation_result["warnings"]:
        validation_result["has_warnings"] = True
        
    return validation_result

###############################################################################
# Configuration objects
###############################################################################


@dataclass(slots=True)
class TimesplitConfig:
    """User-visible configuration for streaming benchmark."""

    data_dir: str
    results_dir: str
    target_pdbs: Sequence[str]
    exclude_pdb_ids: Set[str] | None = None
    n_conformers: int = 200
    template_knn: int = 100
    similarity_threshold: float | None = None
    internal_workers: int = 1
    timeout: int = 1800  # seconds per individual target
    estimated_time_per_target: float = 3.0  # seconds (for timeout calculation)
    timeout_safety_buffer: float = 1.5  # multiplier for safety margin
    max_workers: int | None = None  # concurrent processes
    max_ram_gb: float | None = None  # adaptive throttling
    # Rough estimate of how much RSS a single worker consumes (GiB). Used
    # to auto-scale the process pool so that total memory stays within the
    # available system RAM. Can be tweaked by advanced users via CLI flag.
    memory_per_worker_gb: float = 1.5
    # Hard cap for each worker process (address space limit). If a worker
    # exceeds this amount the kernel will OOM-kill it or raise MemoryError.
    per_worker_ram_gb: float = 4.0
    # Shared cache files for workers
    shared_cache_file: Optional[str] = None
    shared_embedding_cache: Optional[str] = None
    # Peptide filtering threshold
    peptide_threshold: int = 8

    def ensure_dirs(self) -> None:
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)


###############################################################################
# Worker logic
###############################################################################


def _worker_task(args: tuple) -> Dict:
    """Process a single target PDB in an isolated subprocess.

    Parameters
    ----------
    args
        Tuple of positional arguments required by the worker. This is necessary
        because ``multiprocessing.Pool`` pickles a single object.
    Returns
    -------
    Dict
        A lightweight dictionary with success flag, runtime, RMSD metrics, and
        error message (if any).
    """

    (
        target_pdb,
        cfg_dict,
    ) = args

    # Worker only processes one target, so create minimal config
    cfg = TimesplitConfig(target_pdbs=[target_pdb], **cfg_dict)
    start = time.perf_counter()

    # Fix worker environment: ensure proper working directory
    import os
    original_cwd = os.getcwd()
    
    # Change to the project root directory to ensure relative paths work
    project_root = "/home/ubuntu/mcs/templ_pipeline"
    if os.path.exists(project_root) and original_cwd != project_root:
        os.chdir(project_root)

    # Suppress worker logging to prevent console pollution
    from templ_pipeline.core.benchmark_logging import suppress_worker_logging
    suppress_worker_logging()
    
    # Set up timeout signal handler (Unix only)
    import signal
    # Use maximum timeout of 5 minutes (300s) to prevent extreme outliers
    # while keeping the configured timeout for most targets
    max_timeout = 300  # 5 minutes maximum per target
    timeout_seconds = min(cfg.timeout, max_timeout)
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Worker task for {target_pdb} exceeded {timeout_seconds}s timeout (max: {max_timeout}s)")
    
    # Set signal handler for timeout
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
    except (AttributeError, ValueError):
        # signal.alarm not available on Windows or in some environments
        pass

    # --------------------------------------------------------------
    # Apply per-worker memory cap (POSIX only). We use RLIMIT_AS so it
    # covers heap + mmap allocations. If unsupported we silently skip.
    # --------------------------------------------------------------
    try:
        import math
        import resource  # POSIX only

        per_worker_limit = cfg_dict.get("per_worker_ram_gb", 4.0)
        bytes_limit = int(math.ceil(per_worker_limit * 1024**3))
        
        # Set memory limit with some buffer for system overhead
        # Use soft limit = 90% of requested, hard limit = 100%
        soft_limit = int(bytes_limit * 0.9)
        resource.setrlimit(resource.RLIMIT_AS, (soft_limit, bytes_limit))

        if per_worker_limit < 3.0:
            import warnings

            warnings.warn(
                f"per_worker_ram_gb={per_worker_limit} GiB may be too low; large SDF/GZ loading can fail.",
                RuntimeWarning,
            )
    except Exception:  # pragma: no cover – fallback for non-POSIX
        pass

    try:
        # Determine which split this target belongs to and compute exclusions
        target_split = None
        timesplit_exclusions = set()

        # Try to determine target's split by checking which split file contains it
        for split_name in ["train", "val", "test"]:
            try:
                split_pdbs = load_timesplit_pdb_list(split_name)
                if target_pdb.lower() in [pdb.lower() for pdb in split_pdbs]:
                    target_split = split_name
                    break
            except Exception:
                continue

        if target_split:
            # Get time-based exclusions for this target's split
            timesplit_exclusions = get_timesplit_template_exclusions(
                target_pdb, target_split
            )
        else:
            # Fallback: just exclude the target itself
            timesplit_exclusions = {target_pdb}

        # Combine original exclusions with time-based exclusions
        effective_exclusions = (cfg.exclude_pdb_ids or set()) | timesplit_exclusions

        # Ensure poses output directory exists
        poses_output_dir = os.path.join(cfg.results_dir, "poses")
        os.makedirs(poses_output_dir, exist_ok=True)

        result = run_templ_pipeline_for_benchmark(
            target_pdb=target_pdb,
            exclude_pdb_ids=effective_exclusions,
            n_conformers=cfg.n_conformers,
            template_knn=cfg.template_knn,
            similarity_threshold=cfg.similarity_threshold,
            internal_workers=cfg.internal_workers,
            timeout=cfg.timeout,
            data_dir=cfg.data_dir,
            poses_output_dir=poses_output_dir,
            shared_cache_file=cfg_dict.get("shared_cache_file"),
            shared_embedding_cache=cfg_dict.get("shared_embedding_cache"),
        )

        # Ensure result is a dictionary (handle case where BenchmarkResult object is returned)
        if hasattr(result, "to_dict"):
            result = result.to_dict()

        result["pdb_id"] = target_pdb
        result["target_split"] = target_split
        result["exclusions_count"] = len(effective_exclusions)
        result["runtime_total"] = time.perf_counter() - start
        result["is_retry"] = cfg_dict.get("is_retry", False)
        
        # Validate template similarity to detect potential data leakage
        if result.get("success") and result.get("rmsd_values"):
            validation = validate_template_similarity(
                rmsd_values=result["rmsd_values"], 
                pdb_id=target_pdb,
                template_info=result.get("template_info")
            )
            result["data_leakage_validation"] = validation
            
            # Log warnings for potential data leakage
            if validation["has_warnings"]:
                import logging
                logger = logging.getLogger(__name__)
                for warning in validation["warnings"]:
                    logger.warning(warning)
        
        # Clear timeout alarm on successful completion
        try:
            signal.alarm(0)
        except (AttributeError, ValueError):
            pass
        
        # Force garbage collection to prevent memory leaks
        import gc
        gc.collect()
        
        return result

    except MemoryError as exc:  # Handle memory allocation errors specifically
        import psutil
        try:
            mem_info = psutil.virtual_memory()
            mem_msg = f"Memory allocation failed. Available: {mem_info.available/1024**3:.1f}GB, Used: {mem_info.percent:.1f}%"
        except:
            mem_msg = "Memory allocation failed"
        
        # Force garbage collection after memory error
        import gc
        gc.collect()
        
        return {
            "success": False,
            "error": f"MemoryError: {mem_msg}. Original error: {str(exc)}",
            "error_type": "memory_error",
            "pdb_id": target_pdb,
            "runtime_total": time.perf_counter() - start,
            "is_retry": cfg_dict.get("is_retry", False),
        }
    except TimeoutError as exc:  # Handle timeout errors specifically
        # Clear timeout alarm
        try:
            signal.alarm(0)
        except (AttributeError, ValueError):
            pass
        
        # Force garbage collection after timeout
        import gc
        gc.collect()
        
        return {
            "success": False,
            "error": f"TimeoutError: {str(exc)}",
            "error_type": "timeout_error",
            "pdb_id": target_pdb,
            "runtime_total": time.perf_counter() - start,
            "is_retry": cfg_dict.get("is_retry", False),
        }
    except Exception as exc:  # pragma: no cover – wide net to ensure robustness
        # Force garbage collection after any error
        import gc
        gc.collect()
        
        return {
            "success": False,
            "error": str(exc),
            "pdb_id": target_pdb,
            "runtime_total": time.perf_counter() - start,
            "is_retry": cfg_dict.get("is_retry", False),
        }


###############################################################################
# Coordinator
###############################################################################


def run_timesplit_streaming(
    target_pdbs: Sequence[str],
    *,
    data_dir: str,
    results_dir: str | None = None,
    exclude_pdb_ids: Set[str] | None = None,
    n_conformers: int = 200,
    template_knn: int = 100,
    similarity_threshold: float | None = None,
    internal_workers: int = 1,
    timeout: int = 1800,
    max_workers: int | None = None,
    max_ram_gb: float | None = None,
    memory_per_worker_gb: float = 1.5,
    per_worker_ram_gb: float = 4.0,
    peptide_threshold: int = 8,
    quiet: bool = False,
) -> None:
    """Run streaming benchmark over ``target_pdbs``.

    Results are appended to ``results_stream.jsonl`` in *results_dir*.
    A CSV summary ``summary.csv`` is also generated if *pandas* is present.
    """

    if not target_pdbs:
        raise ValueError("No target PDBs provided")

    results_dir = results_dir or os.path.join(os.getcwd(), "timesplit_stream_results")

    # CRITICAL FIX: Use conservative settings instead of hardware detection
    # Hardware detection was causing resource exhaustion by using too many workers
    # Force conservative settings to prevent "cannot allocate memory for thread-local data" errors
    
    # Override any hardware detection - use conservative values
    # This prevents "cannot allocate memory for thread-local data" errors
    
    # ABSOLUTE SAFETY CAP: Allow up to 20 workers for 24-core systems
    ABSOLUTE_MAX_WORKERS = 20
    effective_max_workers = min(ABSOLUTE_MAX_WORKERS, max_workers or 12)
    internal_workers = 1  # Always use 1 internal worker to prevent thread explosion (was 8)
    
    if not quiet:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Using conservative worker settings: max_workers={effective_max_workers}, internal_workers={internal_workers}")
        logger.info("Hardware detection bypassed to prevent resource exhaustion")
        
        # Log if CLI settings were overridden by safety cap
        if max_workers and max_workers > ABSOLUTE_MAX_WORKERS:
            logger.warning(f"CLI requested {max_workers} workers but capped to {ABSOLUTE_MAX_WORKERS} for system safety")
        
        logger.info(f"Worker configuration: {effective_max_workers} concurrent processes, {internal_workers} internal workers each")

    cfg = TimesplitConfig(
        data_dir=data_dir,
        results_dir=results_dir,
        target_pdbs=target_pdbs,
        exclude_pdb_ids=exclude_pdb_ids or set(),
        n_conformers=n_conformers,
        template_knn=template_knn,
        similarity_threshold=similarity_threshold,
        internal_workers=internal_workers,
        timeout=timeout,
        max_workers=effective_max_workers,
        max_ram_gb=max_ram_gb,
        memory_per_worker_gb=memory_per_worker_gb,
        per_worker_ram_gb=per_worker_ram_gb,
        peptide_threshold=peptide_threshold,
    )
    cfg.ensure_dirs()

    output_jsonl = Path(cfg.results_dir) / "results_stream.jsonl"
    progress_jsonl = Path(cfg.results_dir) / "progress.jsonl" 
    # Truncate previous files if they exist to avoid mixing runs
    if output_jsonl.exists():
        output_jsonl.unlink()
    if progress_jsonl.exists():
        progress_jsonl.unlink()

    # Pandas is optional
    try:
        import pandas as pd  # type: ignore

        pandas_available = True
        summary_csv = Path(cfg.results_dir) / "summary.csv"
        if summary_csv.exists():
            summary_csv.unlink()
    except ImportError:  # pragma: no cover
        pandas_available = False

    # ------------------------------------------------------------------
    # Dynamically determine a safe pool size based on available system RAM
    # and a user-tunable per-worker memory estimate. Falls back to the
    # user-requested value if psutil is unavailable or the user explicitly
    # set max_ram_gb (legacy behaviour).
    # ------------------------------------------------------------------

    def _compute_pool_size(cfg_: TimesplitConfig) -> int:  # local helper
        # honour legacy behaviour if user supplied explicit max_ram_gb or
        # psutil is missing
        if cfg_.max_ram_gb is not None or psutil is None:
            return cfg_.max_workers

        try:
            avail_gb = psutil.virtual_memory().available / 1_073_741_824
        except Exception:  # pragma: no cover – very unlikely
            return cfg_.max_workers

        ram_limited = max(1, int(avail_gb // cfg_.memory_per_worker_gb))
        
        # Add memory pressure detection - if less than 2GB available, be more conservative
        if avail_gb < 2.0:
            ram_limited = max(1, ram_limited // 2)  # Halve workers if memory pressure
        
        # CRITICAL FIX: Cap maximum workers to prevent system resource exhaustion
        # The "cannot allocate memory for thread-local data" error indicates thread/process
        # limits being hit, not memory exhaustion. Add conservative hard cap.
        SYSTEM_SAFE_MAX_WORKERS = 20  # Allow up to 20 workers for 24-core systems (matches ABSOLUTE_MAX_WORKERS)
        
        computed_workers = max(1, min(cfg_.max_workers, os.cpu_count() or 2, ram_limited))
        
        # Apply system-safe cap
        safe_workers = min(computed_workers, SYSTEM_SAFE_MAX_WORKERS)
        
        # Always log worker limiting for debugging (not dependent on quiet parameter)
        if safe_workers < computed_workers:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Limiting workers from {computed_workers} to {safe_workers} to prevent system resource exhaustion")
        
        return safe_workers

    pool_size = _compute_pool_size(cfg)

    # Pre-load shared caches for workers
    if not quiet:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Using {pool_size} workers for {len(cfg.target_pdbs)} targets")
        logger.info(f"Pre-loading caches to share across {pool_size} workers...")
    shared_cache_file = None
    shared_embedding_cache = None

    # 1. Pre-load molecule cache
    try:
        from templ_pipeline.core.utils import (
            cleanup_shared_cache, create_shared_embedding_cache,
            create_shared_molecule_cache, load_molecules_with_shared_cache)

        # Load molecules once
        data_path = Path(data_dir)
        molecules = load_molecules_with_shared_cache(data_path)
        if molecules:
            # Create shared cache file for workers
            shared_cache_file = create_shared_molecule_cache(
                molecules, Path(cfg.results_dir)
            )
            if not quiet:
                logger.info(
                    f"Created shared molecule cache with {len(molecules)} molecules: {shared_cache_file}"
                )
        else:
            if not quiet:
                logger.warning("Failed to pre-load molecule cache")
    except Exception as e:
        if not quiet:
            logger.warning(f"Failed to pre-load molecule cache: {e}")

    # 2. Pre-load embedding cache
    try:
        from templ_pipeline.core.embedding import EmbeddingManager

        # Create a temporary embedding manager to load embeddings
        temp_embedding_manager = EmbeddingManager()
        if temp_embedding_manager.embedding_db:
            # Create shared embedding cache
            embedding_data = {
                "embedding_db": temp_embedding_manager.embedding_db,
                "embedding_chain_data": temp_embedding_manager.embedding_chain_data,
            }
            shared_embedding_cache = create_shared_embedding_cache(
                embedding_data, Path(cfg.results_dir)
            )
            if not quiet:
                logger.info(
                    f"Created shared embedding cache with {len(embedding_data['embedding_db'])} embeddings: {shared_embedding_cache}"
                )
        else:
            if not quiet:
                logger.warning("Failed to load embeddings for shared cache")
    except Exception as e:
        if not quiet:
            logger.warning(f"Failed to pre-load embedding cache: {e}")

    # Add cleanup handler
    import atexit

    if shared_cache_file:
        atexit.register(cleanup_shared_cache)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=pool_size, maxtasksperchild=1) as pool:
        # Add shared cache files to config for workers
        cfg_dict = asdict(cfg)
        cfg_dict["shared_cache_file"] = shared_cache_file
        cfg_dict["shared_embedding_cache"] = shared_embedding_cache
        # Remove target_pdbs from worker config as workers only process one target
        cfg_dict.pop("target_pdbs", None)
        tasks_iter = ((pdb_id, cfg_dict) for pdb_id in cfg.target_pdbs)
        # Initialize progress tracking
        total_targets = len(cfg.target_pdbs)
        processed_count = 0
        success_count = 0
        failed_count = 0
        skip_statistics = {}
        start_time = time.time()

        if not quiet:
            # Log initial status (will go to file)
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Processing {total_targets} targets with {pool_size} workers...")

            # Try to use tqdm for progress bar with clean format
            try:
                from tqdm import tqdm
                from templ_pipeline.core.benchmark_logging import get_progress_bar_config

                # Get clean progress bar configuration
                progress_config = get_progress_bar_config('timesplit')
                progress_bar = tqdm(
                    total=total_targets,
                    desc="Processing targets",
                    unit="targets",
                    **progress_config
                )
                use_progress_bar = True
            except ImportError:
                # Fallback to simple progress updates
                use_progress_bar = False
                progress_bar = None
        else:
            use_progress_bar = False
            progress_bar = None

        # Create initial tasks list and retry tasks list
        initial_tasks = [(pdb_id, cfg_dict) for pdb_id in cfg.target_pdbs]
        retry_tasks = []
        
        # Process all initial tasks with timeout handling
        result_iter = pool.imap_unordered(_worker_task, initial_tasks)
        expected_results = len(initial_tasks)
        collected_results = 0
        no_result_timeout = 30  # seconds to wait when no results are coming
        last_result_time = time.time()
        # Dynamic timeout based on dataset size and expected processing time
        estimated_time_per_target = cfg.estimated_time_per_target
        safety_buffer = cfg.timeout_safety_buffer
        min_timeout = 3600  # At least 1 hour
        max_timeout = 86400  # At most 24 hours
        
        estimated_total_time = len(initial_tasks) * estimated_time_per_target * safety_buffer
        absolute_timeout = max(min_timeout, min(max_timeout, estimated_total_time))
        
        if not quiet:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Dynamic timeout calculation:")
            logger.info(f"  - Targets to process: {len(initial_tasks)}")
            logger.info(f"  - Estimated time per target: {estimated_time_per_target}s")
            logger.info(f"  - Safety buffer: {safety_buffer}x")
            logger.info(f"  - Calculated timeout: {absolute_timeout/3600:.1f} hours ({absolute_timeout}s)")
            
        start_collection_time = time.time()
        
        # Track which PDB IDs have been processed
        processed_pdb_ids = set()
        
        while collected_results < expected_results:
            try:
                # Check for absolute timeout
                if time.time() - start_collection_time > absolute_timeout:
                    if not quiet:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Absolute timeout reached after {absolute_timeout}s")
                        logger.warning(f"Completing benchmark with {collected_results}/{expected_results} results")
                    break
                
                # Try to get next result with timeout
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("No result received within timeout")
                
                # Set timeout for getting next result
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(no_result_timeout)
                except (AttributeError, ValueError):
                    # signal.alarm not available on Windows
                    pass
                
                try:
                    result = next(result_iter)
                    try:
                        signal.alarm(0)  # Cancel timeout
                    except (AttributeError, ValueError):
                        pass
                    collected_results += 1
                    last_result_time = time.time()
                    
                    # Add to processed PDB IDs
                    if "pdb_id" in result:
                        processed_pdb_ids.add(result["pdb_id"])
                    
                except StopIteration:
                    try:
                        signal.alarm(0)
                    except (AttributeError, ValueError):
                        pass
                    break  # No more results
                except TimeoutError:
                    try:
                        signal.alarm(0)
                    except (AttributeError, ValueError):
                        pass
                    
                    # Check if we've got most results and should complete
                    completion_rate = collected_results / expected_results
                    time_since_last = time.time() - last_result_time
                    
                    # CRITICAL FIX: Only terminate early if we have 100% completion or extreme timeout
                    # This ensures all 363 targets are attempted, not just 288/363
                    if completion_rate >= 1.0 or time_since_last > 600:  # 100% completion or no results for 10 minutes
                        if not quiet:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.warning(f"Completing benchmark early: {collected_results}/{expected_results} results collected ({completion_rate:.1%})")
                            logger.warning(f"No results received for {time_since_last:.1f}s (timeout: 600s)")
                            logger.warning(f"Remaining {expected_results - collected_results} targets appear to have failed")
                            
                        # Find which PDB IDs are missing by comparing with original targets
                        missing_pdbs = []
                        for task in initial_tasks:
                            pdb_id = task[0]  # First element of task tuple is PDB ID
                            if pdb_id not in processed_pdb_ids:
                                missing_pdbs.append(pdb_id)
                        
                        if not quiet:
                            logger.warning(f"Missing PDB IDs: {missing_pdbs[:10]}{'...' if len(missing_pdbs) > 10 else ''}")
                            logger.warning(f"Worker pool status: {pool._state if hasattr(pool, '_state') else 'unknown'}")
                        
                        # Create failure records for missing PDBs
                        for pdb_id in missing_pdbs:
                            failed_result = {
                                "success": False,
                                "error": f"Worker process failed to return result after {time_since_last:.1f}s timeout (likely memory allocation failure or deadlock)",
                                "error_type": "worker_timeout",
                                "pdb_id": pdb_id,
                                "runtime_total": time_since_last,
                                "target_split": None,
                                "exclusions_count": 0,
                                "rmsd_values": {}
                            }
                            
                            # Append to JSONL
                            with output_jsonl.open("a", encoding="utf-8") as fh:
                                json.dump(failed_result, fh)
                                fh.write("\n")
                            
                            # Update counters
                            failed_count += 1
                            processed_count += 1
                            
                            # Update progress tracking
                            skip_statistics["worker_timeout"] = skip_statistics.get("worker_timeout", 0) + 1
                            
                            if not quiet and use_progress_bar and progress_bar:
                                progress_bar.update(1)
                        
                        break
                    else:
                        # Not enough results yet, continue waiting
                        continue
                
            except Exception as e:
                if not quiet:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error collecting results: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                break
            
            # Process the result we got
            # Check for memory errors and queue for retry with lower memory
            if not result.get("success") and result.get("error_type") == "memory_error":
                pdb_id = result.get("pdb_id")
                if pdb_id and not result.get("is_retry", False):  # Only retry once
                    # Create retry config with 50% memory limit
                    retry_cfg_dict = cfg_dict.copy()
                    retry_cfg_dict["per_worker_ram_gb"] = cfg_dict.get("per_worker_ram_gb", 4.0) * 0.5
                    retry_cfg_dict["is_retry"] = True
                    retry_tasks.append((pdb_id, retry_cfg_dict))
                    
                    if not quiet:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.info(f"Memory error for {pdb_id} - will retry with {retry_cfg_dict['per_worker_ram_gb']:.1f}GB limit")
                else:
                    if not quiet:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Memory error for {result.get('pdb_id', 'unknown')} - retry failed, skipping")
            
            # Check for timeout errors and track problematic targets
            if not result.get("success") and result.get("error_type") == "timeout_error":
                pdb_id = result.get("pdb_id")
                if not quiet:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Timeout error for {pdb_id}: {result.get('error', 'Unknown error')}")
                    logger.warning("Consider reducing --timeout to avoid hanging tasks")
                
                # Track problematic targets that consistently timeout
                skip_statistics["timeout_targets"] = skip_statistics.get("timeout_targets", 0) + 1
            
            # Append to JSONL
            with output_jsonl.open("a", encoding="utf-8") as fh:
                json.dump(result, fh)
                fh.write("\n")

            # Append to CSV if pandas available and result successful
            if pandas_available and result.get("success"):
                import pandas as pd  # type: ignore  # reimport safe under pool

                df = pd.DataFrame([result])
                df.to_csv(
                    summary_csv, mode="a", header=not summary_csv.exists(), index=False
                )

            # Update progress tracking
            processed_count += 1
            if result.get("success"):
                success_count += 1
            else:
                failed_count += 1
                # Track skip reasons for better statistics
                error_msg = result.get("error", "Unknown error")
                skip_reason = "general_error"
                if "No poses generated" in error_msg:
                    skip_reason = "no_poses_generated"
                elif "RMSD calculation failed" in error_msg:
                    skip_reason = "rmsd_calculation_failed"
                elif "zip() argument" in error_msg:
                    skip_reason = "index_mismatch"
                elif "EmbedMultipleConfs failed" in error_msg:
                    skip_reason = "conformer_generation_failed"
                elif "timeout" in error_msg.lower():
                    skip_reason = "timeout"
                
                skip_statistics[skip_reason] = skip_statistics.get(skip_reason, 0) + 1

            # Write progress to file
            current_time = time.time()
            progress_data = {
                "timestamp": current_time,
                "elapsed_time": current_time - start_time,
                "processed": processed_count,
                "total": total_targets,
                "success": success_count,
                "failed": failed_count,
                "percentage": (processed_count / total_targets) * 100,
                "rate": processed_count / (current_time - start_time) if current_time > start_time else 0,
                "skip_statistics": skip_statistics
            }
            
            with progress_jsonl.open("a", encoding="utf-8") as fh:
                json.dump(progress_data, fh)
                fh.write("\n")

            # Update progress display
            if not quiet:
                if use_progress_bar and progress_bar:
                    progress_bar.update(1)
                else:
                    # Simple progress update every 10% or every 5 targets
                    if (
                        processed_count % max(1, total_targets // 10) == 0
                        or processed_count % 5 == 0
                    ):
                        progress_pct = (processed_count / total_targets) * 100
                        remaining_targets = total_targets - processed_count
                        logger.info(
                            f"Progress: {processed_count}/{total_targets} ({progress_pct:.1f}%) - "
                            f"{success_count} successful, {failed_count} failed, {remaining_targets} remaining"
                        )

            _maybe_throttle(pool, cfg)

        # Process retry tasks if any
        if retry_tasks:
            if not quiet:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Processing {len(retry_tasks)} failed targets with reduced memory limits...")
            
            # Update progress bar total if needed
            if use_progress_bar and progress_bar:
                progress_bar.total += len(retry_tasks)
                progress_bar.refresh()
            
            # Process retry tasks with timeout handling
            retry_result_iter = pool.imap_unordered(_worker_task, retry_tasks)
            expected_retry_results = len(retry_tasks)
            collected_retry_results = 0
            retry_start_time = time.time()
            # Dynamic retry timeout based on number of retry tasks
            retry_estimated_time = len(retry_tasks) * estimated_time_per_target * safety_buffer
            retry_min_timeout = 300  # At least 5 minutes for retries
            retry_max_timeout = 7200  # At most 2 hours for retries
            retry_absolute_timeout = max(retry_min_timeout, min(retry_max_timeout, retry_estimated_time))
            
            while collected_retry_results < expected_retry_results:
                # Check for absolute timeout
                if time.time() - retry_start_time > retry_absolute_timeout:
                    if not quiet:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Retry absolute timeout reached after {retry_absolute_timeout}s")
                        logger.warning(f"Completing benchmark with {collected_retry_results}/{expected_retry_results} retry results")
                    break
                    
                try:
                    # Set timeout for retry results too
                    try:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(no_result_timeout)
                    except (AttributeError, ValueError):
                        # signal.alarm not available on Windows
                        pass
                    
                    try:
                        result = next(retry_result_iter)
                        try:
                            signal.alarm(0)
                        except (AttributeError, ValueError):
                            pass
                        collected_retry_results += 1
                    except StopIteration:
                        try:
                            signal.alarm(0)
                        except (AttributeError, ValueError):
                            pass
                        break
                    except TimeoutError:
                        try:
                            signal.alarm(0)
                        except (AttributeError, ValueError):
                            pass
                        if not quiet:
                            logger.warning(f"Timeout waiting for retry results: {collected_retry_results}/{expected_retry_results} collected")
                        break
                        
                except Exception as e:
                    if not quiet:
                        logger.error(f"Error collecting retry results: {e}")
                    break
                processed_count += 1
                if result.get("success"):
                    success_count += 1
                else:
                    failed_count += 1
                
                # Append retry results to JSONL
                with output_jsonl.open("a", encoding="utf-8") as fh:
                    json.dump(result, fh)
                    fh.write("\n")
                
                # Update progress display for retries
                if not quiet:
                    if use_progress_bar and progress_bar:
                        progress_bar.update(1)
                
                # Log final retry failures
                if not result.get("success"):
                    if not quiet:
                        logger.warning(f"Retry failed for {result.get('pdb_id', 'unknown')}: {result.get('error', 'Unknown error')}")

        # Close progress bar and show summary
        if not quiet:
            if use_progress_bar and progress_bar:
                progress_bar.close()

            # Show comprehensive final summary
            total_time = time.time() - start_time
            total_targets = len(cfg.target_pdbs)
            completion_rate = processed_count / total_targets if total_targets > 0 else 0
            
            logger.info("="*60)
            logger.info("BENCHMARK COMPLETION SUMMARY")
            logger.info("="*60)
            logger.info(f"Total targets in test split: {total_targets}")
            logger.info(f"Targets submitted for processing: {processed_count}")
            logger.info(f"Targets never submitted: {total_targets - processed_count}")
            logger.info(f"Successful targets: {success_count}")
            logger.info(f"Failed targets: {failed_count}")
            logger.info(f"Overall completion rate: {completion_rate:.1%}")
            logger.info(f"Success rate of submitted targets: {success_count/processed_count*100:.1f}%" if processed_count > 0 else "Success rate: N/A")
            logger.info(f"Total benchmark time: {total_time:.1f}s")
            
            if failed_count > 0 and skip_statistics:
                logger.info("\nFailure breakdown:")
                for reason, count in sorted(skip_statistics.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"  {reason}: {count} targets ({count/failed_count*100:.1f}%)")
            
            # Show missing targets warning if any
            if processed_count < total_targets:
                missing_count = total_targets - processed_count
                logger.warning(f"\nWARNING: {missing_count} targets were never submitted for processing")
                logger.warning("This indicates worker failure or early termination")
                logger.warning("Check worker logs for memory allocation or timeout issues")
            
            logger.info(f"\nResults saved to: {cfg.results_dir}")
            logger.info(f"Progress log: {progress_jsonl}")
            
            if processed_count == total_targets and failed_count == 0:
                logger.info("✅ ALL TARGETS PROCESSED SUCCESSFULLY!")
            elif processed_count == total_targets:
                logger.info("✅ All targets submitted for processing")
            else:
                logger.warning("❌ Incomplete benchmark - some targets not processed")

    # Generate error summary report at the end (only if not quiet)
    if not quiet:
        _generate_error_summary_report(cfg, quiet=False)
    else:
        _generate_error_summary_report(cfg, quiet=True)


def _generate_error_summary_report(cfg: TimesplitConfig, quiet: bool = False) -> None:
    """Generate comprehensive error summary report from individual error reports."""
    try:
        import json
        from collections import defaultdict
        from pathlib import Path

        from .error_tracking import BenchmarkErrorTracker

        results_dir = Path(cfg.results_dir)
        error_tracking_dir = results_dir / "error_tracking"

        if not error_tracking_dir.exists():
            return  # No error tracking data available

        # Initialize combined error tracker
        combined_tracker = BenchmarkErrorTracker(results_dir)

        # Load and combine individual worker error reports
        error_report_files = list(error_tracking_dir.glob("worker_error_report_*.json"))

        if error_report_files:
            if not quiet:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    f"Combining error reports from {len(error_report_files)} workers..."
                )

            for error_file in error_report_files:
                try:
                    with open(error_file, "r") as f:
                        worker_report = json.load(f)

                    # Add worker data to combined tracker
                    if "missing_pdbs" in worker_report:
                        for pdb_id, records in worker_report["missing_pdbs"].items():
                            for record in records:
                                combined_tracker.record_missing_pdb(
                                    record["pdb_id"],
                                    record["error_type"],
                                    record["error_message"],
                                    record["component"],
                                    record.get("context", {}),
                                )

                    # Track successful/failed targets
                    if worker_report.get("successful_targets", 0) > 0:
                        # Extract PDB ID from filename
                        pdb_id = error_file.stem.replace("worker_error_report_", "")
                        combined_tracker.record_target_success(pdb_id)
                    elif worker_report.get("failed_targets", 0) > 0:
                        pdb_id = error_file.stem.replace("worker_error_report_", "")
                        combined_tracker.record_target_failure(
                            pdb_id, "Worker processing failed"
                        )

                except Exception as e:
                    if not quiet:
                        logger.warning(
                            f"Failed to process error report {error_file}: {e}"
                        )

        # Also parse the main results JSONL for additional error information
        results_jsonl = results_dir / "results_stream.jsonl"
        if results_jsonl.exists():
            try:
                with open(results_jsonl, "r") as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line)
                            pdb_id = result.get("pdb_id", "unknown")

                            if result.get("success", False):
                                combined_tracker.record_target_success(pdb_id)
                            else:
                                error_msg = result.get("error", "Unknown error")
                                combined_tracker.record_target_failure(
                                    pdb_id, error_msg
                                )
            except Exception as e:
                if not quiet:
                    logger.warning(f"Failed to parse results JSONL: {e}")

        # Generate and save final combined error report
        final_report_path = combined_tracker.save_error_report(
            "timesplit_combined_error_report.json"
        )

        # Print summary to console (only if not quiet)
        if not quiet:
            combined_tracker.print_error_summary()

        # Generate recovery plan
        recovery_plan = combined_tracker.create_missing_pdb_recovery_plan()
        if recovery_plan:
            recovery_plan_path = error_tracking_dir / "recovery_plan.json"
            with open(recovery_plan_path, "w") as f:
                json.dump(recovery_plan, f, indent=2)
            if not quiet:
                logger.info(f"Recovery plan saved to: {recovery_plan_path}")
                logger.info("Recovery Plan Summary:")
                for category, pdbs in recovery_plan.items():
                    logger.info(f"  {category}: {len(pdbs)} PDBs")

        if not quiet:
            logger.info(f"Detailed error report: {final_report_path}")

    except ImportError:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Error tracking module not available - skipping error summary")
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to generate error summary: {e}")


###############################################################################
# Helper utilities
###############################################################################


def _calculate_optimal_internal_workers(n_workers: int) -> int:
    """Calculate optimal internal workers for scoring performance.

    Strategy: Maximize parallel processing while maintaining system stability.
    Optimized for maximum hardware utilization.

    Args:
        n_workers: Number of benchmark workers (concurrent PDB targets)

    Returns:
        Optimal number of internal workers for scoring
    """
    # Aggressive optimization - use more internal workers for better performance
    if n_workers >= 32:
        return 4  # High worker count: moderate internal workers to prevent oversubscription
    elif n_workers >= 16:
        return 4  # Increased from 3 to 4 for better utilization
    elif n_workers >= 8:
        return 3  # Increased from 2 to 3 for better utilization
    elif n_workers >= 4:
        return 3  # Increased from 2 to 3 for better utilization
    else:
        return max(
            1, min(4, 8 // n_workers)
        )  # For low worker counts, allow even more internal workers


def _maybe_throttle(pool: mp.pool.Pool, cfg: TimesplitConfig) -> None:
    """Pause pool submission while total system RAM usage exceeds limit.

    We monitor *system-wide* used memory rather than the parent process RSS
    because heavy allocations happen inside the worker processes.  Using
    ``psutil.virtual_memory().used`` reliably captures that aggregate usage.
    """

    if cfg.max_ram_gb is None or psutil is None:  # pragma: no cover – safeguard
        return  # Throttling disabled or psutil unavailable.

    while True:
        mem_used_gb = psutil.virtual_memory().used / 1_073_741_824  # bytes → GiB
        if mem_used_gb < cfg.max_ram_gb:
            break  # Enough free RAM, resume processing.

        # Allow current workers to finish and free memory.
        time.sleep(1)

        # Exit if the pool has completed all tasks or is closed.
        if pool._state != mp.pool.RUN or not pool._cache:  # type: ignore[attr-defined]
            break


###############################################################################
# Split loading helper
###############################################################################

# We expose this as public (via __all__) so that test suites can verify correct
# behaviour without resorting to private attribute access.


def load_timesplit_pdb_list(split_name: str) -> List[str]:
    """Return list of PDB IDs for the requested *split_name*.

    The time-split lists are stored in the repository under
    ``common_pdbids_<split>.txt`` files, which contain only PDB IDs
    that exist in both the temporal splits AND the template database.
    This prevents data leakage while ensuring templates are available.

    Parameters
    ----------
    split_name
        One of "train", "val", "test" (case-insensitive).

    Returns
    -------
    List[str]
        Ordered list of PDB IDs (as lowercase strings).
    """

    split_name = split_name.lower()
    if split_name not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split name: {split_name}")

    # Locate the split file with multiple fallback paths for robustness
    # Use common_pdbids_* files which contain filtered PDB IDs that exist in template database
    potential_paths = [
        # From current working directory (main project directory)
        Path.cwd() / f"common_pdbids_{split_name}.txt",
        # From package root (templ_pipeline/)
        Path(__file__).resolve().parent.parent.parent / f"common_pdbids_{split_name}.txt",
        # From project root
        Path.cwd() / "templ_pipeline" / f"common_pdbids_{split_name}.txt",
        # Relative paths
        Path(f"common_pdbids_{split_name}.txt"),
        Path("..") / f"common_pdbids_{split_name}.txt",
        # Fallback to original timesplit files if common_pdbids not found
        Path(__file__).resolve().parent.parent
        / "data"
        / "splits"
        / f"timesplit_{split_name}",
        Path.cwd() / "data" / "splits" / f"timesplit_{split_name}",
    ]

    split_file = None
    for path in potential_paths:
        if path.exists():
            split_file = path
            break

    if split_file is None:
        raise FileNotFoundError(
            f"Time-split file 'common_pdbids_{split_name}.txt' not found. Tried locations:\n"
            + "\n".join(f"  - {p}" for p in potential_paths)
        )

    with split_file.open("r", encoding="utf-8") as fh:
        pdbs = [ln.strip().lower() for ln in fh if ln.strip()]

    return pdbs


def get_timesplit_template_exclusions(target_pdb: str, target_split: str) -> Set[str]:
    """Get PDB IDs to exclude from template search based on time split rules.

    Time split rules:
    - Train (<2018): Use leave-one-out within train (exclude target + val + test)
    - Val (2018-2019): Use only train templates (exclude target + val + test)
    - Test (>2019): Use train + val templates (exclude target + test)

    Parameters
    ----------
    target_pdb
        The target PDB ID being processed
    target_split
        The split this target belongs to ("train", "val", "test")

    Returns
    -------
    Set[str]
        Set of PDB IDs to exclude from template search
    """
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
        # If we can't load splits, at least exclude the target
        import logging

        logging.warning(f"Could not load split files for exclusions: {e}")

    return exclusions


# Expose helper for external use (e.g., tests)
__all__.append("load_timesplit_pdb_list")

###############################################################################
# Compatibility wrapper for old API
###############################################################################


def run_timesplit_benchmark(
    n_workers: int = None,
    n_conformers: int = 200,
    template_knn: int = 100,
    max_pdbs: Optional[int] = None,
    splits_to_run: List[str] = None,
    quiet: bool = False,
    streaming_output_dir: Optional[str] = None,
    max_ram_gb: Optional[float] = None,
    memory_per_worker_gb: float = 1.5,
    per_worker_ram_gb: float = 4.0,
    peptide_threshold: int = 8,
) -> Dict:
    """Compatibility wrapper for the old timesplit API using streaming implementation."""

    # Import the time splits data
    import json
    # Default data directory - discover TEMPL pipeline data location dynamically
    import os
    from pathlib import Path

    # Try multiple potential data directory locations
    potential_data_dirs = [
        # From current file location: go up to find data directory
        Path(__file__).resolve().parent.parent / "data",
        # From current working directory
        Path.cwd() / "data",
        # If running from templ_pipeline subdirectory
        Path.cwd() / "templ_pipeline" / "data",
        # Relative paths
        Path("data"),
        Path("..") / "data",
        Path("../../data"),  # In case we're deeper in the directory structure
    ]

    data_dir = None
    for candidate_path in potential_data_dirs:
        # Check if this looks like a TEMPL data directory by looking for key files
        if (
            candidate_path.exists()
            and (candidate_path / "ligands" / "templ_processed_ligands_v1.0.0.sdf.gz").exists()
        ):
            data_dir = str(candidate_path)
            break

    # Fallback to PDBBind-style directories if TEMPL data not found
    if data_dir is None:
        # Check environment variable first
        env_data_dir = os.environ.get("PDBBIND_DATA_DIR")
        if env_data_dir and Path(env_data_dir).exists():
            data_dir = env_data_dir
        else:
            # Try to find PDBBind data directories dynamically
            potential_pdbbind_dirs = [
                # From discovered data directory
                *(
                    candidate_path / "PDBBind"
                    for candidate_path in potential_data_dirs
                    if candidate_path.exists()
                ),
                # Generic fallback locations
                Path.cwd() / "data" / "PDBBind",
                Path("data") / "PDBBind",
                Path("..") / "data" / "PDBBind",
            ]

            for candidate in potential_pdbbind_dirs:
                if candidate.exists():
                    data_dir = str(candidate)
                    break

            # Final fallback
            if data_dir is None:
                data_dir = "/data/pdbbind"

    # ------------------------------------------------------------------
    # Load selected splits from packaged lists
    # ------------------------------------------------------------------

    if splits_to_run is None:
        splits_to_run = ["test"]

    target_pdbs: List[str] = []
    for split in splits_to_run:
        try:
            target_pdbs.extend(load_timesplit_pdb_list(split))
        except Exception as exc:
            if not quiet:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to load '{split}' split: {exc}")
            return {"success": False, "error": str(exc)}

    # Keep order deterministic while removing duplicates (train/val/test sets
    # are mutually exclusive, but we guard for safety).
    seen: Set[str] = set()
    target_pdbs_unique: List[str] = []
    for pdb in target_pdbs:
        if pdb not in seen:
            seen.add(pdb)
            target_pdbs_unique.append(pdb)

    if max_pdbs:
        target_pdbs_unique = target_pdbs_unique[:max_pdbs]

    if not target_pdbs_unique:
        return {"success": False, "error": "No PDB IDs selected after filtering"}

    results_dir = streaming_output_dir or "timesplit_stream_results"

    try:
        run_timesplit_streaming(
            target_pdbs=target_pdbs_unique,
            data_dir=data_dir,
            results_dir=results_dir,
            n_conformers=n_conformers,
            template_knn=template_knn,
            max_workers=n_workers or 2,
            max_ram_gb=max_ram_gb,
            memory_per_worker_gb=memory_per_worker_gb,
            per_worker_ram_gb=per_worker_ram_gb,
            internal_workers=_calculate_optimal_internal_workers(
                n_workers or 2
            ),  # Balanced workers for scoring performance
            peptide_threshold=peptide_threshold,
            quiet=quiet,
        )

        # Return success indicator for compatibility
        return {"success": True, "results_dir": results_dir}

    except Exception as e:
        if not quiet:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Streaming benchmark failed: {e}")
        return {"success": False, "error": str(e)}


###############################################################################
# CLI entry-point (optional)
###############################################################################

if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="Streaming timesplit benchmark")
    ap.add_argument("targets_file", help="Path to file with one PDB ID per line")
    ap.add_argument("--data-dir", required=True, help="Directory with dataset files")
    ap.add_argument("--results-dir", default=None, help="Where to write outputs")
    ap.add_argument(
        "--workers", type=int, default=None, help="Max concurrent processes"
    )
    ap.add_argument(
        "--max-ram", type=float, default=None, help="Max RAM in GiB before throttling"
    )
    ap.add_argument(
        "--mem-per-worker",
        type=float,
        default=1.5,
        help="Estimated GiB used per worker (default 1.5)",
    )
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    with open(args.targets_file, "r", encoding="utf-8") as fh:
        targets = [ln.strip() for ln in fh if ln.strip()]

    run_timesplit_streaming(
        targets,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        max_workers=args.workers,
        max_ram_gb=args.max_ram,
        memory_per_worker_gb=args.mem_per_worker,
        timeout=args.timeout,
    )
