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
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import multiprocessing as mp

# Optional – used only if available
try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover – psutil is optional
    psutil = None  # type: ignore

from templ_pipeline.benchmark.runner import run_templ_pipeline_for_benchmark

# Public symbols for 'from ... import *' users
__all__: List[str] = []

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
    timeout: int = 1800  # seconds
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

    cfg = TimesplitConfig(**cfg_dict)
    start = time.perf_counter()

    # --------------------------------------------------------------
    # Apply per-worker memory cap (POSIX only). We use RLIMIT_AS so it
    # covers heap + mmap allocations. If unsupported we silently skip.
    # --------------------------------------------------------------
    try:
        import resource, math  # POSIX only

        per_worker_limit = cfg_dict.get("per_worker_ram_gb", 4.0)
        bytes_limit = int(math.ceil(per_worker_limit * 1024**3))
        resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))

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
            timesplit_exclusions = get_timesplit_template_exclusions(target_pdb, target_split)
        else:
            # Fallback: just exclude the target itself
            timesplit_exclusions = {target_pdb}
        
        # Combine original exclusions with time-based exclusions
        effective_exclusions = (cfg.exclude_pdb_ids or set()) | timesplit_exclusions

        result = run_templ_pipeline_for_benchmark(
            target_pdb=target_pdb,
            exclude_pdb_ids=effective_exclusions,
            n_conformers=cfg.n_conformers,
            template_knn=cfg.template_knn,
            similarity_threshold=cfg.similarity_threshold,
            internal_workers=cfg.internal_workers,
            timeout=cfg.timeout,
            data_dir=cfg.data_dir,
            poses_output_dir=os.path.join(cfg.results_dir, "poses"),
            shared_cache_file=cfg_dict.get('shared_cache_file'),
            shared_embedding_cache=cfg_dict.get('shared_embedding_cache'),
        )
        
        # Ensure result is a dictionary (handle case where BenchmarkResult object is returned)
        if hasattr(result, 'to_dict'):
            result = result.to_dict()
        
        result["pdb_id"] = target_pdb
        result["target_split"] = target_split
        result["exclusions_count"] = len(effective_exclusions)
        result["runtime_total"] = time.perf_counter() - start
        return result

    except Exception as exc:  # pragma: no cover – wide net to ensure robustness
        return {
            "success": False,
            "error": str(exc),
            "pdb_id": target_pdb,
            "runtime_total": time.perf_counter() - start,
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
) -> None:
    """Run streaming benchmark over ``target_pdbs``.

    Results are appended to ``results_stream.jsonl`` in *results_dir*.
    A CSV summary ``summary.csv`` is also generated if *pandas* is present.
    """

    if not target_pdbs:
        raise ValueError("No target PDBs provided")

    results_dir = results_dir or os.path.join(os.getcwd(), "timesplit_stream_results")
    
    # Calculate optimal internal workers if not specified or if default value
    effective_max_workers = max_workers or os.cpu_count() or 2
    if internal_workers == 1:  # Default value, optimize it
        internal_workers = _calculate_optimal_internal_workers(effective_max_workers)
    
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
    )
    cfg.ensure_dirs()

    output_jsonl = Path(cfg.results_dir) / "results_stream.jsonl"
    # Truncate previous file if exists to avoid mixing runs
    if output_jsonl.exists():
        output_jsonl.unlink()

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
        return max(1, min(cfg_.max_workers, os.cpu_count() or 2, ram_limited))

    pool_size = _compute_pool_size(cfg)

    # Pre-load shared caches for workers
    print(f"Pre-loading caches to share across {pool_size} workers...")
    shared_cache_file = None
    shared_embedding_cache = None
    
    # 1. Pre-load molecule cache
    try:
        from templ_pipeline.core.utils import (
            load_molecules_with_shared_cache, 
            create_shared_molecule_cache,
            create_shared_embedding_cache,
            cleanup_shared_cache
        )
        
        # Load molecules once
        data_path = Path(data_dir)
        molecules = load_molecules_with_shared_cache(data_path)
        if molecules:
            # Create shared cache file for workers
            shared_cache_file = create_shared_molecule_cache(molecules, Path(cfg.results_dir))
            print(f"Created shared molecule cache with {len(molecules)} molecules: {shared_cache_file}")
        else:
            print("Warning: Failed to pre-load molecule cache")
    except Exception as e:
        print(f"Warning: Failed to pre-load molecule cache: {e}")

    # 2. Pre-load embedding cache
    try:
        from templ_pipeline.core.embedding import EmbeddingManager
        
        # Create a temporary embedding manager to load embeddings
        temp_embedding_manager = EmbeddingManager()
        if temp_embedding_manager.embedding_db:
            # Create shared embedding cache
            embedding_data = {
                'embedding_db': temp_embedding_manager.embedding_db,
                'embedding_chain_data': temp_embedding_manager.embedding_chain_data
            }
            shared_embedding_cache = create_shared_embedding_cache(embedding_data, Path(cfg.results_dir))
            print(f"Created shared embedding cache with {len(embedding_data['embedding_db'])} embeddings: {shared_embedding_cache}")
        else:
            print("Warning: Failed to load embeddings for shared cache")
    except Exception as e:
        print(f"Warning: Failed to pre-load embedding cache: {e}")

    # Add cleanup handler
    import atexit
    if shared_cache_file:
        atexit.register(cleanup_shared_cache)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=pool_size, maxtasksperchild=1) as pool:
        # Add shared cache files to config for workers
        cfg_dict = asdict(cfg)
        cfg_dict['shared_cache_file'] = shared_cache_file
        cfg_dict['shared_embedding_cache'] = shared_embedding_cache
        tasks_iter = ((pdb_id, cfg_dict) for pdb_id in cfg.target_pdbs)
        for result in pool.imap_unordered(_worker_task, tasks_iter):
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

            _maybe_throttle(pool, cfg)
    
    # Generate error summary report at the end
    _generate_error_summary_report(cfg)


def _generate_error_summary_report(cfg: TimesplitConfig) -> None:
    """Generate comprehensive error summary report from individual error reports."""
    try:
        from .error_tracking import BenchmarkErrorTracker
        import json
        from pathlib import Path
        from collections import defaultdict
        
        results_dir = Path(cfg.results_dir)
        error_tracking_dir = results_dir / "error_tracking"
        
        if not error_tracking_dir.exists():
            return  # No error tracking data available
            
        # Initialize combined error tracker
        combined_tracker = BenchmarkErrorTracker(results_dir)
        
        # Load and combine individual worker error reports
        error_report_files = list(error_tracking_dir.glob("worker_error_report_*.json"))
        
        if error_report_files:
            print(f"Combining error reports from {len(error_report_files)} workers...")
            
            for error_file in error_report_files:
                try:
                    with open(error_file, 'r') as f:
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
                                    record.get("context", {})
                                )
                    
                    # Track successful/failed targets
                    if worker_report.get("successful_targets", 0) > 0:
                        # Extract PDB ID from filename
                        pdb_id = error_file.stem.replace("worker_error_report_", "")
                        combined_tracker.record_target_success(pdb_id)
                    elif worker_report.get("failed_targets", 0) > 0:
                        pdb_id = error_file.stem.replace("worker_error_report_", "") 
                        combined_tracker.record_target_failure(pdb_id, "Worker processing failed")
                        
                except Exception as e:
                    print(f"Warning: Failed to process error report {error_file}: {e}")
        
        # Also parse the main results JSONL for additional error information
        results_jsonl = results_dir / "results_stream.jsonl"
        if results_jsonl.exists():
            try:
                with open(results_jsonl, 'r') as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line)
                            pdb_id = result.get("pdb_id", "unknown")
                            
                            if result.get("success", False):
                                combined_tracker.record_target_success(pdb_id)
                            else:
                                error_msg = result.get("error", "Unknown error")
                                combined_tracker.record_target_failure(pdb_id, error_msg)
            except Exception as e:
                print(f"Warning: Failed to parse results JSONL: {e}")
        
        # Generate and save final combined error report
        final_report_path = combined_tracker.save_error_report("timesplit_combined_error_report.json")
        
        # Print summary to console
        combined_tracker.print_error_summary()
        
        # Generate recovery plan
        recovery_plan = combined_tracker.create_missing_pdb_recovery_plan()
        if recovery_plan:
            recovery_plan_path = error_tracking_dir / "recovery_plan.json"
            with open(recovery_plan_path, 'w') as f:
                json.dump(recovery_plan, f, indent=2)
            print(f"\nRecovery plan saved to: {recovery_plan_path}")
            print("Recovery Plan Summary:")
            for category, pdbs in recovery_plan.items():
                print(f"  {category}: {len(pdbs)} PDBs")
        
        print(f"\nDetailed error report: {final_report_path}")
        
    except ImportError:
        print("Error tracking module not available - skipping error summary")
    except Exception as e:
        print(f"Warning: Failed to generate error summary: {e}")


###############################################################################
# Helper utilities
###############################################################################


def _calculate_optimal_internal_workers(n_workers: int) -> int:
    """Calculate optimal internal workers for scoring performance.
    
    Strategy: Use fewer benchmark workers with more internal workers each
    to optimize scoring performance while preventing CPU oversubscription.
    
    Args:
        n_workers: Number of benchmark workers (concurrent PDB targets)
        
    Returns:
        Optimal number of internal workers for scoring
    """
    # For high worker counts, reduce benchmark workers but increase internal workers
    if n_workers >= 16:
        return 3  # 16 benchmark workers × 3 internal = 48 total (may oversubscribe but scoring is I/O bound)
    elif n_workers >= 8:
        return 2  # 8 benchmark workers × 2 internal = 16 total (balanced)
    elif n_workers >= 4:
        return 2  # 4 benchmark workers × 2 internal = 8 total (conservative)
    else:
        return max(1, min(3, 6 // n_workers))  # For low worker counts, allow more internal workers


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
    ``templ_pipeline/data/splits/timesplit_<split>`` (no file extension).
    This helper keeps the path logic in one place and fails with a clear
    error message if the file cannot be located.

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
    potential_paths = [
        # From package root (templ_pipeline/)
        Path(__file__).resolve().parent.parent / "data" / "splits" / f"timesplit_{split_name}",
        # From current working directory
        Path.cwd() / "data" / "splits" / f"timesplit_{split_name}",
        # From project root
        Path.cwd() / "templ_pipeline" / "data" / "splits" / f"timesplit_{split_name}",
        # Relative paths
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
            f"Time-split file 'timesplit_{split_name}' not found. Tried locations:\n" +
            "\n".join(f"  - {p}" for p in potential_paths)
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
) -> Dict:
    """Compatibility wrapper for the old timesplit API using streaming implementation."""

    # Import the time splits data
    from pathlib import Path
    import json

    # Default data directory - discover TEMPL pipeline data location dynamically
    import os
    
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
        if (candidate_path.exists() and 
            (candidate_path / "ligands" / "processed_ligands_new.sdf.gz").exists()):
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
                *(candidate_path / "PDBBind" for candidate_path in potential_data_dirs if candidate_path.exists()),
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
                print(f"Failed to load '{split}' split: {exc}")
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
            internal_workers=_calculate_optimal_internal_workers(n_workers or 2),  # Balanced workers for scoring performance
        )

        # Return success indicator for compatibility
        return {"success": True, "results_dir": results_dir}

    except Exception as e:
        if not quiet:
            print(f"Streaming benchmark failed: {e}")
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
