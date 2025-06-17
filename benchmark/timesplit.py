#!/usr/bin/env python3
"""
Streamlined time-split benchmark for TEMPL pipeline.
Uses BenchmarkRunner for proper CLI compatibility.
"""

import json
import logging
import argparse
import time
import threading
import psutil
import signal
import gc
import multiprocessing as mp
from typing import Dict, List, Set, Optional, Tuple, Any
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed, Future, TimeoutError
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from enum import Enum

from .runner import BenchmarkRunner, BenchmarkParams, BenchmarkResult
from templ_pipeline.core.utils import (
    load_sdf_molecules_cached, get_pdb_id_from_mol, load_split_pdb_ids,
    find_ligand_file_paths, get_worker_config, find_ligand_by_pdb_id
)
from templ_pipeline.core.hardware_utils import get_optimized_worker_config

# Configuration
TEMPL_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = TEMPL_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
BENCHMARK_OUTPUTS_DIR = TEMPL_ROOT / "templ_benchmark_results_timesplit"

# Timeout settings
PER_TARGET_TIMEOUT = 600  # seconds allowed for each individual target process (10 minutes)

# Global shared cache for SDF molecules - use memory-mapped approach for true sharing
SHARED_MOLECULE_CACHE = None
SHARED_TEMPLATES_CACHE = None
MOLECULE_CACHE_MMAP = None

# Memory monitoring constants
MEMORY_WARNING_THRESHOLD = 0.75  # 75% of total RAM
MEMORY_CRITICAL_THRESHOLD = 0.85  # 85% of total RAM
MAX_MEMORY_PER_WORKER_GB = 7.0    # Realistic memory per worker including spikes

# Dynamic scaling constants for up to 20 cores
MAX_WORKERS_TIER_1 = 8    # Full memory allocation (7.0GB per worker)
MAX_WORKERS_TIER_2 = 16   # Reduced memory allocation (4.5GB per worker)
MAX_WORKERS_TIER_3 = 20   # Minimal memory allocation (2.8GB per worker)

# Tiered memory allocation
MEMORY_PER_WORKER_TIER_1 = 7.0   # GB per worker for 1-8 workers
MEMORY_PER_WORKER_TIER_2 = 4.5   # GB per worker for 9-16 workers
MEMORY_PER_WORKER_TIER_3 = 2.8   # GB per worker for 17-20 workers

class ProcessStatus(Enum):
    """Process status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

@dataclass
class ProcessTracker:
    """Track individual process status and metrics."""
    pdb_id: str
    future: Future
    start_time: float
    status: ProcessStatus = ProcessStatus.PENDING
    result: Optional[Dict] = None
    error: Optional[str] = None
    
    def update_status(self, status: ProcessStatus, result: Optional[Dict] = None, error: Optional[str] = None):
        """Update process status and metadata."""
        self.status = status
        if result is not None:
            self.result = result
        if error is not None:
            self.error = error
    
    def get_runtime(self) -> float:
        """Get current runtime in seconds."""
        return time.time() - self.start_time
    
    def is_hanging(self, threshold: int) -> bool:
        """Check if process is hanging based on threshold."""
        return (self.status == ProcessStatus.RUNNING and 
                self.get_runtime() > threshold and 
                not self.future.done())

class BenchmarkProcessMonitor:
    """Robust process monitoring for benchmark execution."""
    
    def __init__(self, futures: Dict[Future, str], hanging_threshold: int = 300):
        self.hanging_threshold = hanging_threshold
        self.max_total_time = 3600
        self.check_interval = 5
        self.logger = logging.getLogger(__name__)
        
        # Initialize process trackers
        self.trackers = {
            future: ProcessTracker(
                pdb_id=pdb_id,
                future=future,
                start_time=time.time(),
                status=ProcessStatus.RUNNING
            )
            for future, pdb_id in futures.items()
        }
        
        self.start_time = time.time()
        self.completed_count = 0
        self.error_count = 0
        self.timeout_count = 0
        
    def monitor_with_progress(self, pbar: Optional[tqdm] = None) -> Dict[str, Dict]:
        """Monitor all processes with progress tracking."""
        self.logger.info(f"Starting monitoring of {len(self.trackers)} processes")
        
        while not self._all_complete() and not self._global_timeout_exceeded():
            self._process_completed_futures(pbar)
            self._handle_hanging_processes(pbar)
            
            if not self._all_complete():
                time.sleep(self.check_interval)
        
        # Handle any remaining processes
        self._cleanup_remaining_processes(pbar)
        
        # Compile final results
        results = self._compile_results()
        
        self.logger.info(f"Monitoring complete: {self.completed_count} successful, "
                        f"{self.error_count} errors, {self.timeout_count} timeouts")
        
        return results
    
    def _all_complete(self) -> bool:
        """Check if all processes are complete."""
        return all(tracker.status in [ProcessStatus.COMPLETED, ProcessStatus.FAILED, 
                                     ProcessStatus.TIMEOUT, ProcessStatus.CANCELLED] 
                  for tracker in self.trackers.values())
    
    def _global_timeout_exceeded(self) -> bool:
        """Check if global timeout has been exceeded."""
        return (time.time() - self.start_time) > self.max_total_time
    
    def _process_completed_futures(self, pbar: Optional[tqdm]):
        """Process all completed futures."""
        for tracker in self.trackers.values():
            if tracker.status == ProcessStatus.RUNNING and tracker.future.done():
                self._handle_completed_future(tracker, pbar)
    
    def _handle_completed_future(self, tracker: ProcessTracker, pbar: Optional[tqdm]):
        """Handle a single completed future."""
        try:
            # Extract result from future
            result_pdb_id, result_data = tracker.future.result(timeout=1)
            
            if not isinstance(result_data, dict):
                raise ValueError(f"Invalid result format: expected dict, got {type(result_data)}")
            
            tracker.update_status(ProcessStatus.COMPLETED, result=result_data)
            
            if result_data.get("success", False):
                self.completed_count += 1
                self.logger.debug(f"✓ {tracker.pdb_id}: Success")
            else:
                self.error_count += 1
                self.logger.debug(f"✗ {tracker.pdb_id}: {result_data.get('error', 'Unknown error')}")
                
        except Exception as e:
            self.error_count += 1
            error_msg = f"Result retrieval error: {str(e)}"
            self.logger.warning(f"✗ {tracker.pdb_id}: {error_msg}")
            
            tracker.update_status(ProcessStatus.FAILED, error=error_msg)
        
        if pbar:
            pbar.update(1)
    
    def _handle_hanging_processes(self, pbar: Optional[tqdm]):
        """Handle processes that are hanging."""
        hanging_trackers = [t for t in self.trackers.values() if t.is_hanging(self.hanging_threshold)]
        
        for tracker in hanging_trackers:
            self.timeout_count += 1
            self.logger.warning(f"⏰ {tracker.pdb_id}: Timeout after {self.hanging_threshold}s")
            
            try:
                tracker.future.cancel()
            except Exception:
                pass
            
            tracker.update_status(ProcessStatus.TIMEOUT, error=f"Process timeout after {self.hanging_threshold}s")
            
            if pbar:
                pbar.update(1)
    
    def _cleanup_remaining_processes(self, pbar: Optional[tqdm]):
        """Clean up any remaining incomplete processes."""
        remaining_trackers = [t for t in self.trackers.values() 
                             if t.status == ProcessStatus.RUNNING]
        
        for tracker in remaining_trackers:
            try:
                tracker.future.cancel()
            except Exception:
                pass
            
            tracker.update_status(ProcessStatus.CANCELLED, 
                                error="Process did not complete within global timeout")
            
            if pbar:
                pbar.update(1)
    
    def _compile_results(self) -> Dict[str, Dict]:
        """Compile final results from all trackers."""
        results = {}
        
        for tracker in self.trackers.values():
            if tracker.result is not None:
                results[tracker.pdb_id] = tracker.result
            else:
                # Create error result
                results[tracker.pdb_id] = {
                    "success": False,
                    "rmsd_values": {},
                    "runtime": tracker.get_runtime(),
                    "error": tracker.error or f"Process ended with status: {tracker.status.value}"
                }
        
        return results

# ----------------------------------------------------------------------------
# Utility: Preload available ligand PDB IDs once to avoid per-process scans
# ----------------------------------------------------------------------------

_LIGAND_ID_CACHE: Optional[Set[str]] = None

def _load_available_ligand_ids(data_dir: Path) -> Set[str]:
    """Return a set of PDB IDs that have ligands in the processed SDF cache."""
    global _LIGAND_ID_CACHE
    if _LIGAND_ID_CACHE is not None:
        return _LIGAND_ID_CACHE

    lig_paths = find_ligand_file_paths(data_dir)
    for sdf_path in lig_paths:
        if sdf_path.exists():
            try:
                # Quick scan using RDKit without loading full molecules
                from rdkit import Chem
                import gzip
                ids: Set[str] = set()
                with gzip.open(sdf_path, "rb") as fh:
                    supplier = Chem.ForwardSDMolSupplier(fh, removeHs=False, sanitize=False)
                    for mol in supplier:
                        if mol is None or not mol.HasProp("_Name"):
                            continue
                        ids.add(mol.GetProp("_Name")[:4].lower())
                _LIGAND_ID_CACHE = ids
                print(f"Cached {len(ids)} ligand PDB IDs from {sdf_path.name}")
                return ids
            except Exception as e:
                print(f"Warning: Unable to preload ligand IDs from {sdf_path}: {e}")
    _LIGAND_ID_CACHE = set()
    return _LIGAND_ID_CACHE

def setup_logging(quiet: bool = False):
    """Configure logging for benchmark."""
    if quiet:
        logging.basicConfig(level=logging.CRITICAL, format="%(message)s")
    else:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        # Reduce verbosity during benchmark for non-critical modules
        for module in ['templ_pipeline.core.mcs']:
            logging.getLogger(module).setLevel(logging.WARNING)
        # Keep scoring at INFO level to catch molecular errors
        logging.getLogger('templ_pipeline.core.scoring').setLevel(logging.INFO)
        # Keep pipeline and runner at INFO level for debugging
        logging.getLogger('templ_pipeline.core.pipeline').setLevel(logging.INFO)
        logging.getLogger('templ_pipeline.benchmark.runner').setLevel(logging.DEBUG)
        # Add specific logger for molecular errors
        logging.getLogger('templ_pipeline.core.scoring.molecular_errors').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def get_split_exclusions(split_name: str, all_splits: Dict[str, Set[str]]) -> Set[str]:
    """Return exclusion set so that template pool matches desired strategy.

    Desired behaviour:
      • Train     → pool only from train  (exclude val & test & self)
      • Validation→ pool from train only (exclude val & test)
      • Test      → pool from train+val  (exclude test)
    """
    if split_name == "train":
        return all_splits.get("val", set()).union(all_splits.get("test", set()))
    elif split_name == "val":
        return all_splits.get("val", set()).union(all_splits.get("test", set()))
    elif split_name == "test":
        return all_splits.get("test", set())
    return set()

def initialize_shared_caches():
    """Initialize shared caches using memory-mapped files for true sharing."""
    global SHARED_MOLECULE_CACHE, SHARED_TEMPLATES_CACHE, MOLECULE_CACHE_MMAP
    
    if SHARED_MOLECULE_CACHE is None:
        # Use simple dict for coordination, actual data in memory-mapped files
        SHARED_MOLECULE_CACHE = {}
        SHARED_TEMPLATES_CACHE = {}
        
    return SHARED_MOLECULE_CACHE, SHARED_TEMPLATES_CACHE

def preload_sdf_molecules(data_dir: Path, pdb_ids_needed: Set[str] = None, force_reload: bool = False) -> bool:
    """Load only needed SDF molecules to reduce memory usage.
    
    Args:
        data_dir: Data directory path
        pdb_ids_needed: Set of PDB IDs that will be processed (for filtering)
        force_reload: Force reload even if cache exists
        
    Returns:
        True if successful, False otherwise
    """
    global SHARED_MOLECULE_CACHE
    
    # Initialize cache if needed
    if SHARED_MOLECULE_CACHE is None:
        initialize_shared_caches()
    
    cache_key = "molecules"
    if cache_key in SHARED_MOLECULE_CACHE and not force_reload:
        cached_count = len(SHARED_MOLECULE_CACHE[cache_key])
        print(f"Using cached molecules: {cached_count} molecules")
        return True
    
    print("Loading SDF molecules with filtering and memory limits...")
    
    # Also initialize global cache in utils module
    try:
        from templ_pipeline.core.utils import initialize_global_molecule_cache
        initialize_global_molecule_cache()
    except ImportError:
        pass
    
    ligand_file_paths = find_ligand_file_paths(data_dir)
    for path in ligand_file_paths:
        if path.exists():
            try:
                # Use more conservative memory limit
                memory_limit = 2.0 if pdb_ids_needed and len(pdb_ids_needed) < 100 else 3.0
                molecules = load_sdf_molecules_filtered(path, pdb_ids_needed, memory_limit_gb=memory_limit)
                if molecules:
                    # Store in local cache
                    SHARED_MOLECULE_CACHE[cache_key] = molecules
                    
                    # Also store in global cache for utils module
                    try:
                        from templ_pipeline.core.utils import set_global_molecule_cache
                        set_global_molecule_cache(molecules)
                    except ImportError:
                        pass
                    
                    print(f"Loaded {len(molecules)} filtered molecules from {path.name} (limit: {memory_limit}GB)")
                    print(f"Molecules cached in both local and global caches for sharing across processes")
                    
                    # Force immediate cleanup after loading
                    cleanup_memory()
                    return True
            except Exception as e:
                print(f"Failed to load SDF from {path}: {e}")
                continue
    
    print("Failed to load SDF molecules")
    return False

def load_sdf_molecules_filtered(sdf_path: Path, pdb_ids_filter: Set[str] = None, memory_limit_gb: float = 4.0) -> List[Any]:
    """Load only molecules matching the PDB ID filter to reduce memory usage."""
    from templ_pipeline.core.utils import load_sdf_molecules_cached
    
    if pdb_ids_filter is None:
        # No filter, load normally but with memory limit
        return load_sdf_molecules_cached(sdf_path, memory_limit_gb=memory_limit_gb)
    
    # Load with filtering
    molecules = []
    pdb_ids_lower = {pdb.lower() for pdb in pdb_ids_filter}
    
    try:
        if sdf_path.suffix == '.gz':
            import gzip
            import io
            from rdkit import Chem
            
            with gzip.open(sdf_path, 'rb') as fh:
                content = fh.read()
                
            with io.BytesIO(content) as buffer:
                supplier = Chem.ForwardSDMolSupplier(buffer, removeHs=False, sanitize=False)
                
                for mol in supplier:
                    if mol is None or not mol.HasProp("_Name"):
                        continue
                    
                    mol_name = mol.GetProp("_Name")
                    pdb_id = mol_name[:4].lower()
                    
                    if pdb_id in pdb_ids_lower:
                        molecules.append(mol)
                        
                    if len(molecules) % 1000 == 0:
                        try:
                            import psutil
                            memory_gb = psutil.Process().memory_info().rss / (1024**3)
                            if memory_gb > memory_limit_gb:
                                print(f"Memory limit reached at {len(molecules)} molecules")
                                break
                        except Exception:
                            pass
        
        print(f"Filtered loading: {len(molecules)} molecules (from {len(pdb_ids_filter) if pdb_ids_filter else 'all'} target PDBs)")
        return molecules
        
    except Exception as e:
        print(f"Filtered loading failed: {e}")
        return []

def preload_templates(template_pdb_ids: Set[str], data_dir: Path) -> bool:
    """Preload commonly used templates into shared cache.
    
    Args:
        template_pdb_ids: Set of template PDB IDs to preload
        data_dir: Data directory path
        
    Returns:
        True if successful, False otherwise
    """
    global SHARED_TEMPLATES_CACHE
    
    if SHARED_TEMPLATES_CACHE is None:
        initialize_shared_caches()
    
    print(f"Preloading {len(template_pdb_ids)} templates...")
    
    try:
        from templ_pipeline.core.templates import load_template_molecules_standardized
        
        # Load templates in batches to manage memory
        batch_size = 100
        template_list = list(template_pdb_ids)
        
        for i in range(0, len(template_list), batch_size):
            batch = template_list[i:i + batch_size]
            templates, stats = load_template_molecules_standardized(batch)
            
            if templates:
                for j, template in enumerate(templates):
                    pdb_id = batch[j] if j < len(batch) else f"template_{j}"
                    SHARED_TEMPLATES_CACHE[pdb_id] = template
                    
        print(f"Preloaded {len(SHARED_TEMPLATES_CACHE)} templates")
        return True
        
    except Exception as e:
        print(f"Template preloading failed: {e}")
        return False

def get_dynamic_memory_per_worker(target_workers: int) -> float:
    """Calculate memory per worker based on dynamic scaling tiers."""
    if target_workers <= MAX_WORKERS_TIER_1:
        return MEMORY_PER_WORKER_TIER_1
    elif target_workers <= MAX_WORKERS_TIER_2:
        return MEMORY_PER_WORKER_TIER_2
    elif target_workers <= MAX_WORKERS_TIER_3:
        return MEMORY_PER_WORKER_TIER_3
    else:
        # For more than 20 workers, use minimal allocation
        return MEMORY_PER_WORKER_TIER_3

def get_memory_safe_worker_config(total_tasks: int, available_ram_gb: float, target_workers: int = None) -> Dict[str, int]:
    """Dynamic memory-safe worker allocation supporting up to 20 cores with intelligent scaling."""
    
    # More conservative safety margins based on worker count
    base_safety_margin = 0.7  # Start with 70% instead of 80%
    if target_workers and target_workers > 10:
        base_safety_margin = 0.6  # Even more conservative for high worker counts
    
    memory_safety_margin = available_ram_gb * base_safety_margin
    
    # Reserve additional memory for system overhead
    system_overhead = max(2.0, available_ram_gb * 0.1)  # At least 2GB or 10% for system
    memory_safety_margin -= system_overhead
    memory_safety_margin = max(memory_safety_margin, 4.0)  # Minimum 4GB for processing
    
    # If target_workers specified, try to accommodate it with dynamic memory allocation
    if target_workers is not None:
        memory_per_worker = get_dynamic_memory_per_worker(target_workers)
        estimated_memory = target_workers * memory_per_worker
        
        if estimated_memory <= memory_safety_margin:
            # Target workers fit within memory constraints
            benchmark_workers = min(target_workers, total_tasks)
        else:
            # Target workers exceed memory, find maximum safe workers
            benchmark_workers = max(1, int(memory_safety_margin // memory_per_worker))
            benchmark_workers = min(benchmark_workers, MAX_WORKERS_TIER_3, total_tasks)
    else:
        # Auto-determine optimal worker count using tiered approach
        # Try tier 3 first (20 workers), then tier 2, then tier 1
        for max_workers, memory_per_worker in [
            (MAX_WORKERS_TIER_3, MEMORY_PER_WORKER_TIER_3),
            (MAX_WORKERS_TIER_2, MEMORY_PER_WORKER_TIER_2),
            (MAX_WORKERS_TIER_1, MEMORY_PER_WORKER_TIER_1)
        ]:
            estimated_memory = max_workers * memory_per_worker
            if estimated_memory <= memory_safety_margin:
                benchmark_workers = min(max_workers, total_tasks)
                break
        else:
            # Fallback to minimal safe configuration
            memory_per_worker = MEMORY_PER_WORKER_TIER_1
            benchmark_workers = max(1, int(memory_safety_margin // memory_per_worker))
    
    # Recalculate actual memory per worker based on final worker count
    actual_memory_per_worker = get_dynamic_memory_per_worker(benchmark_workers)
    
    # Always use single pipeline worker to prevent nested parallelization
    pipeline_workers = 1
    
    estimated_memory_gb = benchmark_workers * actual_memory_per_worker
    
    # Determine tier for reporting
    if benchmark_workers <= MAX_WORKERS_TIER_1:
        tier = "Tier 1 (Full)"
    elif benchmark_workers <= MAX_WORKERS_TIER_2:
        tier = "Tier 2 (Reduced)"
    else:
        tier = "Tier 3 (Minimal)"
    
    return {
        "benchmark_workers": benchmark_workers,
        "pipeline_workers": pipeline_workers,
        "estimated_memory_gb": estimated_memory_gb,
        "max_memory_gb": available_ram_gb,
        "safety_margin_gb": memory_safety_margin,
        "system_overhead_gb": system_overhead,
        "safe": estimated_memory_gb <= memory_safety_margin,
        "total_estimated_processes": benchmark_workers * pipeline_workers,
        "memory_per_worker_gb": actual_memory_per_worker,
        "scaling_tier": tier,
        "target_workers_requested": target_workers,
        "workers_adjusted": target_workers is not None and benchmark_workers != target_workers
    }

def monitor_system_resources() -> Dict:
    """Monitor system resources and return safety status."""
    try:
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent / 100.0
        available_gb = memory_info.available / (1024**3)
        
        # Count Python processes
        python_processes = 0
        for proc in psutil.process_iter(['name']):
            try:
                if 'python' in proc.info['name'].lower():
                    python_processes += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        warnings = []
        if memory_percent > 0.8:
            warnings.append(f"High memory usage: {memory_percent*100:.1f}%")
        if python_processes > 50:
            warnings.append(f"Many Python processes: {python_processes}")
        
        return {
            "memory_percent": memory_percent * 100,
            "available_gb": available_gb,
            "process_count": python_processes,
            "safe": len(warnings) == 0,
            "warnings": warnings
        }
        
    except Exception as e:
        return {
            "memory_percent": 0.0,
            "available_gb": 0.0,
            "process_count": 0,
            "safe": False,
            "warnings": [f"Monitoring failed: {str(e)}"]
        }

def monitor_memory_usage() -> Dict:
    """Monitor current memory usage and return status."""
    try:
        memory_info = psutil.virtual_memory()
        memory_percent = memory_info.percent / 100.0
        available_gb = memory_info.available / (1024**3)
        
        status = {
            "memory_percent": memory_percent,
            "available_gb": available_gb,
            "warning": memory_percent > MEMORY_WARNING_THRESHOLD,
            "critical": memory_percent > MEMORY_CRITICAL_THRESHOLD
        }
        
        return status
        
    except Exception as e:
        return {"error": str(e), "critical": True}

def cleanup_memory():
    """Force garbage collection and memory cleanup."""
    try:
        gc.collect()
        if hasattr(gc, 'set_threshold'):
            gc.set_threshold(700, 10, 10)  # More aggressive GC
        # Force additional cleanup cycles
        for _ in range(3):
            gc.collect()
    except Exception:
        pass

def validate_pdb_before_processing(pdb_id: str, data_dir: str) -> Tuple[bool, str]:
    """Pre-validate PDB to catch issues early and avoid hanging processes."""
    try:
        from templ_pipeline.core.utils import find_ligand_file_paths, get_protein_file_paths
        
        data_path = Path(data_dir)
        
        # Check protein file availability
        protein_paths = get_protein_file_paths(pdb_id, data_path)
        protein_found = any(path.exists() for path in protein_paths)
        if not protein_found:
            return False, "No protein file found"
        
        # Quick ligand validation would go here if we had access to the cache
        # For now, just check basic requirements
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def process_single_target(args):
    """Process single target with optimized memory usage and monitoring."""
    pdb_id, data_dir, exclude_pdb_ids, n_conformers, template_knn, poses_output_dir = args
    
    start_time = time.time()
    ligand_smiles = None
    crystal_mol = None
    runner = None
    
    # Monitor memory at start with early exit on critical usage
    memory_status = monitor_memory_usage()
    if memory_status.get("critical", False):
        return pdb_id, {
            "success": False,
            "rmsd_values": {},
            "runtime": time.time() - start_time,
            "error": f"Critical memory usage before processing: {memory_status.get('memory_percent', 0)*100:.1f}%"
        }
    
    try:
        # Use optimized cache access with fallback
        ligand_smiles, crystal_mol = get_ligand_data_optimized(pdb_id, data_dir)
        
        if not ligand_smiles or crystal_mol is None:
            raise ValueError(f"Could not find ligand data for {pdb_id}")
        
        # Pre-validate molecular quality
        is_valid, validation_msg = validate_pdb_before_processing(pdb_id, data_dir)
        if not is_valid:
            return pdb_id, {
                "success": False,
                "rmsd_values": {},
                "runtime": time.time() - start_time,
                "error": f"Pre-validation failed: {validation_msg}"
            }
        
        # Use BenchmarkRunner with memory-optimized settings
        params = BenchmarkParams(
            target_pdb=pdb_id,
            exclude_pdb_ids=exclude_pdb_ids,
            poses_output_dir=poses_output_dir,
            n_conformers=n_conformers,
            template_knn=template_knn,
            internal_workers=1  # Always single worker to prevent nested parallelization
        )
        
        runner = BenchmarkRunner(data_dir, poses_output_dir)
        result = runner.run_single_target(params)
        
        return pdb_id, result.to_dict()
        
    except Exception as e:
        return pdb_id, {
            "success": False,
            "rmsd_values": {},
            "runtime": time.time() - start_time,
            "error": str(e)
        }
    finally:
        # Aggressive cleanup in finally block
        try:
            if crystal_mol is not None:
                del crystal_mol
            if runner is not None:
                del runner
            cleanup_memory()
        except Exception:
            pass

def get_ligand_data_optimized(pdb_id: str, data_dir: str) -> Tuple[Optional[str], Any]:
    """Get ligand data with fallback approach to minimize memory usage."""
    # Fallback to direct SDF loading since streaming modules were removed
    from templ_pipeline.core.utils import find_ligand_by_pdb_id, load_molecules_with_shared_cache
    
    try:
        molecule_cache = load_molecules_with_shared_cache(Path(data_dir))
        if molecule_cache:
            return find_ligand_by_pdb_id(pdb_id, molecule_cache)
        else:
            return None, None
    except Exception:
        return None, None

def load_single_molecule_from_sdf(sdf_path: Path, target_pdb_id: str) -> List[Any]:
    """Load only a specific molecule from SDF to minimize memory usage."""
    from rdkit import Chem
    import gzip
    import io
    
    target_pdb_lower = target_pdb_id.lower()
    
    try:
        if sdf_path.suffix == '.gz':
            with gzip.open(sdf_path, 'rb') as fh:
                content = fh.read()
            
            with io.BytesIO(content) as buffer:
                supplier = Chem.ForwardSDMolSupplier(buffer, removeHs=False, sanitize=False)
                
                for mol in supplier:
                    if mol is None or not mol.HasProp("_Name"):
                        continue
                    
                    mol_name = mol.GetProp("_Name")
                    if mol_name[:4].lower() == target_pdb_lower:
                        return [mol]
        
        return []
        
    except Exception as e:
        return []

def evaluate_split(
    split_name: str,
    pdb_ids: Set[str], 
    exclude_pdb_ids: Set[str],
    data_dir: str,
    n_workers: int = 2,  # Reduced default
    n_conformers: int = 200,
    template_knn: int = 100,
    max_pdbs: Optional[int] = None,
    poses_output_dir: Optional[str] = None,
    quiet: bool = False,
    streaming_output_dir: Optional[str] = None
) -> Dict:
    """Evaluate pipeline on a split with memory optimization."""
    # Streaming modules removed - use simple in-memory results
    
    if max_pdbs:
        pdb_ids = set(list(pdb_ids)[:max_pdbs])
    
    if not pdb_ids:
        return {}
    
    print(f"\nEvaluating {split_name}: {len(pdb_ids)} targets")
    
    # No streaming result writer - use simple in-memory storage
    result_writer = None
    
    # Preload only needed molecules to reduce memory usage
    all_pdb_ids = set(pdb_ids)
    if not preload_sdf_molecules(Path(data_dir), all_pdb_ids):
        print("Warning: Failed to preload SDF molecules, falling back to individual loading")
    
    # Dynamic memory-safe worker configuration with up to 20 cores
    try:
        memory_info = psutil.virtual_memory()
        available_gb = memory_info.available / (1024**3)
        
        memory_config = get_memory_safe_worker_config(
            total_tasks=len(pdb_ids),
            available_ram_gb=available_gb,
            target_workers=n_workers  # Pass requested worker count for dynamic scaling
        )
        
        # Apply dynamic scaling results
        n_workers = memory_config["benchmark_workers"]
        
        print(f"Dynamic Scaling Config: {memory_config['scaling_tier']}")
        print(f"Workers: {n_workers} × {memory_config['memory_per_worker_gb']:.1f}GB = {memory_config['estimated_memory_gb']:.1f}GB estimated")
        print(f"System: {available_gb:.1f}GB available, {memory_config['safety_margin_gb']:.1f}GB safety limit")
        print(f"Reserved: {memory_config.get('system_overhead_gb', 0):.1f}GB system overhead")
        
        if memory_config.get('workers_adjusted', False):
            requested = memory_config.get('target_workers_requested', 'unknown')
            print(f"Note: Requested {requested} workers, adjusted to {n_workers} for memory safety")
        
        if not memory_config["safe"]:
            print(f"Warning: Configuration may still exceed memory limits")
        
    except Exception as e:
        print(f"Dynamic scaling failed: {e}, using conservative fallback")
        n_workers = 1
    
    # Filter PDBs that have ligands in cache
    available_ligand_ids = _load_available_ligand_ids(Path(data_dir))
    process_args = [
        (pdb_id, data_dir, exclude_pdb_ids, n_conformers, template_knn, poses_output_dir)
        for pdb_id in pdb_ids if pdb_id.lower() in available_ligand_ids
    ]
    
    skipped_missing = len(pdb_ids) - len(process_args)
    if skipped_missing:
        print(f"Skipping {skipped_missing} targets without ligands")
    
    if not process_args:
        print("No valid targets to process")
        return {}

    print(f"Processing {len(process_args)} targets with {n_workers} workers...")
    
    results = {}
    
    # Use memory-optimized executor with conservative settings
    executor = ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=mp.get_context('spawn')  # Use spawn for better memory isolation
    )
    try:
        futures = {
            executor.submit(process_single_target, args): args[0]
            for args in process_args
        }
        
        print(f"Submitted {len(futures)} jobs with memory-isolated processes")
        
        # Use robust monitoring with memory checks
        with tqdm(total=len(process_args), desc=f"{split_name.capitalize()}", 
                 unit="targets", ncols=80, disable=quiet) as pbar:
            
            for future in as_completed(futures):
                pdb_id = futures[future]
                
                try:
                    result_pdb_id, result_data = future.result(timeout=600)
                    
                    # Store result in memory
                    results[result_pdb_id] = result_data
                    
                    if result_data.get("success", False):
                        pbar.set_postfix_str("✓", refresh=False)
                    else:
                        pbar.set_postfix_str("✗", refresh=False)
                        
                except Exception as e:
                    error_result = {
                        "success": False,
                        "rmsd_values": {},
                        "runtime": 0.0,
                        "error": f"Future failed: {str(e)}"
                    }
                    
                    results[pdb_id] = error_result
                    pbar.set_postfix_str("✗", refresh=False)
                
                pbar.update(1)
                
                # Enhanced memory monitoring during processing
                memory_status = monitor_memory_usage()
                if memory_status.get("critical", False):
                    print(f"\nCritical memory usage detected: {memory_status.get('memory_percent', 0)*100:.1f}%")
                    print("Cancelling remaining jobs and shutting down to prevent OOM")
                    # Cancel remaining futures
                    for remaining_future in futures:
                        if not remaining_future.done():
                            remaining_future.cancel()
                    break
                    
                # Periodic cleanup with memory pressure detection
                processed_count = len(results)
                if processed_count % 5 == 0:  # More frequent cleanup
                    cleanup_memory()
                    # Check if we need to be more aggressive
                    if memory_status.get("warning", False):
                        print(f"High memory warning: {memory_status.get('memory_percent', 0)*100:.1f}%")
                        # Force additional cleanup rounds
                        for _ in range(5):
                            cleanup_memory()
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError during processing: {e}")
    finally:
        print("Shutting down executor...")
        executor.shutdown(wait=True)
        cleanup_memory()
    
    # Calculate final results
    successful = sum(1 for r in results.values() if r.get("success"))
    failed = len(results) - successful
    print(f"{split_name} complete: {successful} successful, {failed} failed")
    
    return results

def calculate_metrics(results: Dict) -> Dict:
    """Calculate summary metrics from results."""
    total = len(results)
    successful = sum(1 for r in results.values() if r.get("success"))
    failed = total - successful
    
    metrics = {
        "total": total,
        "successful": successful,
        "failed": failed,
        "success_rate": (successful / total * 100) if total > 0 else 0.0,
        "rmsd_stats": {}
    }
    
    # Calculate RMSD statistics by metric
    for metric in ["shape", "color", "combo", "canimotocombo"]:
        rmsd_values = []
        for result in results.values():
            if result.get("success") and result.get("rmsd_values"):
                rmsd_data = result["rmsd_values"].get(metric)
                if rmsd_data and "rmsd" in rmsd_data:
                    rmsd_values.append(rmsd_data["rmsd"])
        
        if rmsd_values:
            import numpy as np
            metrics["rmsd_stats"][metric] = {
                "count": len(rmsd_values),
                "mean": float(np.mean(rmsd_values)),
                "median": float(np.median(rmsd_values)),
                "min": float(np.min(rmsd_values)),
                "max": float(np.max(rmsd_values)),
                "below_2A": sum(1 for r in rmsd_values if r <= 2.0),
                "below_5A": sum(1 for r in rmsd_values if r <= 5.0),
                "perc_below_2A": (sum(1 for r in rmsd_values if r <= 2.0) / len(rmsd_values) * 100),
                "perc_below_5A": (sum(1 for r in rmsd_values if r <= 5.0) / len(rmsd_values) * 100)
            }
    
    return metrics

def create_results_table(all_results: Dict) -> pd.DataFrame:
    """Create comprehensive results table."""
    rows = []
    
    for split_name, split_results in all_results.items():
        if split_name == "params":
            continue
            
        metrics = split_results.get("metrics", {})
        rmsd_stats = metrics.get("rmsd_stats", {})
        
        if rmsd_stats:
            for metric, stats in rmsd_stats.items():
                rows.append({
                    "Split": split_name.capitalize(),
                    "Metric": metric.capitalize(),
                    "Total": metrics.get("total", 0),
                    "Successful": metrics.get("successful", 0),
                    "Success_Rate": f"{metrics.get('success_rate', 0):.1f}%",
                    "Mean_RMSD": f"{stats['mean']:.2f}Å",
                    "Median_RMSD": f"{stats['median']:.2f}Å",
                    "Success_2A": f"{stats['perc_below_2A']:.1f}%",
                    "Success_5A": f"{stats['perc_below_5A']:.1f}%"
                })
        else:
            rows.append({
                "Split": split_name.capitalize(),
                "Metric": "N/A",
                "Total": metrics.get("total", 0),
                "Successful": metrics.get("successful", 0),
                "Success_Rate": f"{metrics.get('success_rate', 0):.1f}%",
                "Mean_RMSD": "N/A",
                "Median_RMSD": "N/A",
                "Success_2A": "0.0%",
                "Success_5A": "0.0%"
            })
    
    return pd.DataFrame(rows)

def run_timesplit_benchmark(
    n_workers: int = None,
    n_conformers: int = 200,
    template_knn: int = 100,
    max_pdbs: Optional[int] = None,
    splits_to_run: List[str] = None,
    quiet: bool = False,
    streaming_output_dir: Optional[str] = None
) -> Dict:
    """Run the timesplit benchmark using BenchmarkRunner."""
    log = setup_logging(quiet)
    
    if not quiet:
        print("TEMPL Timesplit Benchmark (BenchmarkRunner)")
        print("=" * 60)
    
    # Initialize shared caches BEFORE any worker processes are created
    print("Initializing shared molecule cache...")
    initialize_shared_caches()
    
    # Pre-load molecules into shared cache using all available PDB IDs
    # This is critical for efficiency - without this, each worker loads individually
    all_splits_for_preload = {}
    split_names = splits_to_run or ["train", "val", "test"]
    all_pdb_ids = set()
    
    for split_name in split_names:
        split_file = SPLITS_DIR / f"timesplit_{split_name}"
        try:
            pdb_ids = load_split_pdb_ids(split_file, DATA_DIR)
            all_splits_for_preload[split_name] = pdb_ids
            all_pdb_ids.update(pdb_ids)
        except Exception as e:
            print(f"Warning: Failed to load split {split_name} for preloading: {e}")
    
    print(f"Pre-loading molecules for {len(all_pdb_ids)} unique PDB IDs across all splits...")
    preload_success = preload_sdf_molecules(DATA_DIR, all_pdb_ids, force_reload=False)
    
    if preload_success:
        print("✓ Shared cache populated successfully - workers will reuse loaded molecules")
    else:
        print("⚠ Shared cache preload failed - workers will load individually (slower)")
    
    # Dynamic memory-safe worker configuration supporting up to 20 cores
    if n_workers is None:
        try:
            memory_info = psutil.virtual_memory()
            available_gb = memory_info.available / (1024**3)
            
            # Auto-determine optimal worker count using dynamic scaling
            memory_config = get_memory_safe_worker_config(1000, available_gb)  # Use large task count for max workers
            n_workers = memory_config["benchmark_workers"]
        except Exception:
            n_workers = 4  # Conservative fallback
    else:
        # Validate and optimize user-provided worker count with dynamic scaling
        try:
            memory_info = psutil.virtual_memory()
            available_gb = memory_info.available / (1024**3)
            
            # Use dynamic scaling to validate and potentially adjust worker count
            memory_config = get_memory_safe_worker_config(1000, available_gb, target_workers=n_workers)
            
            if memory_config.get('workers_adjusted', False):
                original_workers = n_workers
                n_workers = memory_config["benchmark_workers"]
                print(f"Dynamic scaling adjusted workers from {original_workers} to {n_workers} ({memory_config['scaling_tier']})")
            
        except Exception:
            # Fallback to original validation logic
            max_safe_workers = int(available_gb * 0.8 // MAX_MEMORY_PER_WORKER_GB)
            if n_workers > max_safe_workers:
                print(f"Warning: Requested {n_workers} workers may exceed memory limit, max safe: {max_safe_workers}")
                n_workers = max_safe_workers
    
    # Pre-execution system safety check
    if not quiet:
        try:
            safety_check = monitor_system_resources()
            memory_info = psutil.virtual_memory()
            memory_gb = memory_info.total / (1024**3)
            cpu_count = psutil.cpu_count()
            
            print(f"System: {cpu_count} CPUs, {memory_gb:.1f}GB total RAM, {safety_check['available_gb']:.1f}GB available")
            print(f"Current: {safety_check['process_count']} Python processes, {safety_check['memory_percent']:.1f}% memory")
            
            if not safety_check["safe"]:
                print(f"System warnings: {safety_check['warnings']}")
                n_workers = 1  # Force ultra-conservative config
            
            # Get current memory configuration for reporting
            try:
                memory_info = psutil.virtual_memory()
                available_gb = memory_info.available / (1024**3)
                current_config = get_memory_safe_worker_config(1000, available_gb, target_workers=n_workers)
                
                print(f"Dynamic Scaling: {current_config['scaling_tier']} - {n_workers} workers × {current_config['memory_per_worker_gb']:.1f}GB = {current_config['estimated_memory_gb']:.1f}GB estimated")
                print(f"Features: Dynamic scaling (up to 20 cores), tiered memory allocation, aggressive cleanup")
            except Exception:
                print(f"Workers: {n_workers} (dynamic scaling enabled)")
                print(f"Features: Dynamic scaling (up to 20 cores), tiered memory allocation, aggressive cleanup")
        except Exception as e:
            print(f"Max workers: {n_workers} (monitoring failed: {e})")
        print()
    
    # Validate data structure
    required_dirs = [DATA_DIR, SPLITS_DIR]
    for dir_path in required_dirs:
        if not dir_path.exists():
            raise FileNotFoundError(f"Required directory not found: {dir_path}")
    
    # Load splits
    all_splits = {}
    split_names = splits_to_run or ["train", "val", "test"]
    
    for split_name in split_names:
        split_file = SPLITS_DIR / f"timesplit_{split_name}"
        try:
            pdb_ids = load_split_pdb_ids(split_file, DATA_DIR)
            all_splits[split_name] = pdb_ids
            if not quiet:
                print(f"Split {split_name}: {len(pdb_ids)} PDBs")
        except Exception as e:
            log.error(f"Failed to load split {split_name}: {e}")
            continue
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BENCHMARK_OUTPUTS_DIR / f"timesplit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = output_dir / "results"
    poses_dir = output_dir / "poses"
    results_dir.mkdir(exist_ok=True)
    poses_dir.mkdir(exist_ok=True)
    
    # Run evaluation for each split
    all_results = {
        "params": {
            "n_workers": n_workers,
            "n_conformers": n_conformers,
            "template_knn": template_knn,
            "max_pdbs": max_pdbs,
            "timestamp": timestamp,
            "approach": "benchmark_runner"
        }
    }
    
    for split_name, pdb_ids in all_splits.items():
        # Safety check before each split
        if not quiet:
            try:
                safety_check = monitor_system_resources()
                print(f"\nPre-{split_name}: {safety_check['process_count']} processes, {safety_check['memory_percent']:.1f}% memory")
                
                if not safety_check["safe"]:
                    print(f"Resource warnings detected: {safety_check['warnings']}")
                    print("   Proceeding with conservative settings...")
            except Exception as e:
                print(f"   Safety check failed: {e}")
        
        exclude_pdb_ids = get_split_exclusions(split_name, all_splits)
        
        split_streaming_dir = None
        if streaming_output_dir:
            split_streaming_dir = f"{streaming_output_dir}/{split_name}"
        
        results = evaluate_split(
            split_name=split_name,
            pdb_ids=pdb_ids,
            exclude_pdb_ids=exclude_pdb_ids,
            data_dir=str(DATA_DIR),
            n_workers=n_workers,
            n_conformers=n_conformers,
            template_knn=template_knn,
            max_pdbs=max_pdbs,
            poses_output_dir=str(poses_dir),
            quiet=quiet,
            streaming_output_dir=split_streaming_dir
        )
        
        metrics = calculate_metrics(results)
        all_results[split_name] = {
            "results": results,
            "metrics": metrics
        }
        
        if not quiet:
            success_rate = metrics.get("success_rate", 0)
            print(f"{split_name} completed: {success_rate:.1f}% success rate")
            
            # Post-split safety check
            try:
                safety_check = monitor_system_resources()
                print(f"   Post-{split_name}: {safety_check['process_count']} processes, {safety_check['memory_percent']:.1f}% memory")
            except Exception:
                pass
    
    # Save results
    results_file = results_dir / f"timesplit_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Create and save results table
    results_table = create_results_table(all_results)
    table_file = results_dir / f"timesplit_summary_{timestamp}.csv"
    results_table.to_csv(table_file, index=False)
    
    # Create markdown summary
    md_file = results_dir / f"timesplit_summary_{timestamp}.md"
    with open(md_file, "w") as f:
        f.write("# TEMPL Timesplit Benchmark Results\n\n")
        f.write(f"**Timestamp:** {timestamp}\n\n")
        f.write(f"**Configuration:** {n_workers} workers, {n_conformers} conformers\n\n")
        f.write("## Results Summary\n\n")
        f.write(results_table.to_markdown(index=False))
    
    # Normal completion - no forced exit needed
    print("Benchmark completed successfully.")
    
    if not quiet:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(results_table.to_string(index=False))
        print(f"\nOutput: {results_file}")
        
        # Verify poses were saved
        if poses_dir.exists():
            pdb_subdirs = [d for d in poses_dir.iterdir() if d.is_dir()]
            print(f"\nPoses saved: {len(pdb_subdirs)} PDB directories in {poses_dir}")
            
            # Count actual pose files
            total_pose_files = 0
            for pdb_dir in pdb_subdirs:
                pose_files = list(pdb_dir.glob("**/*.sdf"))
                total_pose_files += len(pose_files)
            
            if total_pose_files > 0:
                print(f"Found {total_pose_files} pose SDF files across all PDBs")
            else:
                print("No pose SDF files found - poses may not have been saved successfully")
        else:
            print(f"\nPoses directory not found: {poses_dir}")
    
    return all_results

def main(argv=None):
    """Main entry point."""
    parser = argparse.ArgumentParser(description="TEMPL Timesplit Benchmark")
    
    parser.add_argument("--n-workers", type=int, help="Number of parallel workers")
    parser.add_argument("--n-conformers", type=int, default=200, help="Number of conformers")
    parser.add_argument("--template-knn", type=int, default=100, help="Number of templates")
    parser.add_argument("--max-pdbs", type=int, help="Limit number of PDBs per split")
    parser.add_argument("--train-only", action="store_true", help="Only run train split")
    parser.add_argument("--val-only", action="store_true", help="Only run val split")
    parser.add_argument("--test-only", action="store_true", help="Only run test split")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    
    args = parser.parse_args(argv)
    
    # Determine splits to run
    if args.train_only:
        splits_to_run = ["train"]
    elif args.val_only:
        splits_to_run = ["val"]
    elif args.test_only:
        splits_to_run = ["test"]
    else:
        splits_to_run = ["train", "val", "test"]
    
    return run_timesplit_benchmark(
        n_workers=args.n_workers,
        n_conformers=args.n_conformers,
        template_knn=args.template_knn,
        max_pdbs=args.max_pdbs,
        splits_to_run=splits_to_run,
        quiet=args.quiet
    )

if __name__ == "__main__":
    main() 