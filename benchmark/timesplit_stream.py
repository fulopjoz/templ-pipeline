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
        bytes_limit = int(math.ceil(per_worker_limit * 1024 ** 3))
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
        result = run_templ_pipeline_for_benchmark(
            target_pdb=target_pdb,
            exclude_pdb_ids=cfg.exclude_pdb_ids or set(),
            n_conformers=cfg.n_conformers,
            template_knn=cfg.template_knn,
            similarity_threshold=cfg.similarity_threshold,
            internal_workers=cfg.internal_workers,
            timeout=cfg.timeout,
            data_dir=cfg.data_dir,
            poses_output_dir=os.path.join(cfg.results_dir, "poses"),
        )
        result["pdb_id"] = target_pdb
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
        max_workers=max_workers or os.cpu_count() or 2,
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

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=pool_size, maxtasksperchild=1) as pool:
        tasks_iter = ((pdb_id, asdict(cfg)) for pdb_id in cfg.target_pdbs)
        for result in pool.imap_unordered(_worker_task, tasks_iter):
            # Append to JSONL
            with output_jsonl.open("a", encoding="utf-8") as fh:
                json.dump(result, fh)
                fh.write("\n")

            # Append to CSV if pandas available and result successful
            if pandas_available and result.get("success"):
                import pandas as pd  # type: ignore  # reimport safe under pool

                df = pd.DataFrame([result])
                df.to_csv(summary_csv, mode="a", header=not summary_csv.exists(), index=False)

            _maybe_throttle(pool, cfg)


###############################################################################
# Helper utilities
###############################################################################

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

    # Locate the split file relative to the package root so that it works in
    # both editable installs and packaged wheels.
    package_root = Path(__file__).resolve().parent.parent  # templ_pipeline/
    split_file = package_root / "data" / "splits" / f"timesplit_{split_name}"

    if not split_file.exists():
        raise FileNotFoundError(f"Time-split file not found: {split_file}")

    with split_file.open("r", encoding="utf-8") as fh:
        pdbs = [ln.strip().lower() for ln in fh if ln.strip()]

    return pdbs

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
    
    # Default data directory - try to find actual data location
    import os
    data_dir = os.environ.get("PDBBIND_DATA_DIR", "/data/pdbbind")
    
    # Try common locations if default doesn't exist
    if not os.path.exists(data_dir):
        possible_dirs = [
            "/home/ubuntu/mcs/templ_pipeline/data",
            "/home/ubuntu/mcs/templ_pipeline/data/PDBBind",
            "/home/ubuntu/mcs/data",
            "/home/ubuntu/data",
            "./templ_pipeline/data",
            "./templ_pipeline/data/PDBBind",
            "./data",
            "../data"
        ]
        for candidate in possible_dirs:
            if os.path.exists(candidate):
                data_dir = candidate
                break
    
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
            internal_workers=1  # Force single internal worker to prevent nested parallelization
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
    ap.add_argument("--workers", type=int, default=None, help="Max concurrent processes")
    ap.add_argument("--max-ram", type=float, default=None, help="Max RAM in GiB before throttling")
    ap.add_argument("--mem-per-worker", type=float, default=1.5, help="Estimated GiB used per worker (default 1.5)")
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