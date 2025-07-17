"""
Simplified streaming timesplit benchmark based on working Polaris pattern.

This version uses ProcessPoolExecutor with proper timeout handling and memory management,
avoiding the hanging and memory issues of the original mp.Pool implementation.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set
from tqdm import tqdm

from templ_pipeline.benchmark.runner import run_templ_pipeline_for_benchmark
from templ_pipeline.core.hardware import get_optimized_worker_config


@dataclass(slots=True)
class SimplifiedTimesplitConfig:
    """Simplified configuration for streaming benchmark."""
    data_dir: str
    results_dir: str
    target_pdbs: Sequence[str]
    exclude_pdb_ids: Set[str] | None = None
    n_conformers: int = 200
    template_knn: int = 100
    similarity_threshold: float | None = None
    internal_workers: int = 1
    timeout: int = 180  # 3 minutes per target
    max_workers: int = 8  # Conservative default
    shared_cache_file: Optional[str] = None
    shared_embedding_cache: Optional[str] = None
    peptide_threshold: int = 8

    def ensure_dirs(self) -> None:
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)


def simple_worker_task(args: tuple) -> Dict:
    """Process a single target PDB in a clean subprocess."""
    (
        target_pdb,
        cfg_dict,
        exclude_pdb_ids,
        data_dir,
        shared_cache_file,
        shared_embedding_cache,
        peptide_threshold,
    ) = args
    
    start_time = time.time()
    
    # Suppress worker logging to prevent console pollution
    from templ_pipeline.core.benchmark_logging import suppress_worker_logging
    suppress_worker_logging()
    
    try:
        # Early complexity check
        try:
            from templ_pipeline.core.utils import find_ligand_by_pdb_id
            from templ_pipeline.core.utils import load_molecules_from_cache
            
            molecules = load_molecules_from_cache(shared_cache_file)
            if molecules:
                ligand_smiles, ligand_mol = find_ligand_by_pdb_id(target_pdb, molecules)
                if ligand_mol:
                    num_atoms = ligand_mol.GetNumAtoms()
                    num_bonds = ligand_mol.GetNumBonds()
                    
                    # Skip very complex molecules
                    if num_atoms > 100 or num_bonds > 150:
                        return {
                            "success": False,
                            "error": f"Molecule too complex: {num_atoms} atoms, {num_bonds} bonds - skipping",
                            "pdb_id": target_pdb,
                            "runtime": time.time() - start_time,
                            "rmsd_values": {},
                        }
        except Exception:
            pass  # Continue if complexity check fails
        
        # Run the pipeline
        result = run_templ_pipeline_for_benchmark(
            target_pdb=target_pdb,
            exclude_pdb_ids=exclude_pdb_ids or set(),
            n_conformers=cfg_dict.get("n_conformers", 200),
            template_knn=cfg_dict.get("template_knn", 100),
            similarity_threshold=cfg_dict.get("similarity_threshold"),
            internal_workers=cfg_dict.get("internal_workers", 1),
            timeout=cfg_dict.get("timeout", 180),
            data_dir=data_dir,
            poses_output_dir=os.path.join(cfg_dict.get("results_dir", ""), "poses"),
            shared_cache_file=shared_cache_file,
            shared_embedding_cache=shared_embedding_cache,
        )
        
        # Ensure result is a dictionary
        if hasattr(result, "to_dict"):
            result = result.to_dict()
        
        # Add metadata
        result["pdb_id"] = target_pdb
        result["runtime"] = time.time() - start_time
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "pdb_id": target_pdb,
            "runtime": time.time() - start_time,
            "rmsd_values": {},
        }


def run_simple_timesplit_streaming(
    target_pdbs: Sequence[str],
    data_dir: str,
    results_dir: str,
    exclude_pdb_ids: Set[str] | None = None,
    n_conformers: int = 200,
    template_knn: int = 100,
    similarity_threshold: float | None = None,
    internal_workers: int = 1,
    timeout: int = 180,
    max_workers: int | None = None,
    shared_cache_file: Optional[str] = None,
    shared_embedding_cache: Optional[str] = None,
    peptide_threshold: int = 8,
    quiet: bool = False,
) -> Dict[str, any]:
    """
    Run simplified timesplit benchmark using ProcessPoolExecutor pattern.
    
    This follows the same pattern as the working Polaris benchmark to avoid
    memory issues and hanging processes.
    """
    
    # Create configuration
    cfg = SimplifiedTimesplitConfig(
        data_dir=data_dir,
        results_dir=results_dir,
        target_pdbs=target_pdbs,
        exclude_pdb_ids=exclude_pdb_ids,
        n_conformers=n_conformers,
        template_knn=template_knn,
        similarity_threshold=similarity_threshold,
        internal_workers=internal_workers,
        timeout=timeout,
        max_workers=max_workers or 8,
        shared_cache_file=shared_cache_file,
        shared_embedding_cache=shared_embedding_cache,
        peptide_threshold=peptide_threshold,
    )
    cfg.ensure_dirs()
    
    # Determine worker count
    if max_workers is None:
        worker_config = get_optimized_worker_config(
            workload_type="cpu_intensive",
            dataset_size=len(target_pdbs)
        )
        max_workers = min(worker_config["n_workers"], 8)  # Cap at 8 for safety
    
    if not quiet:
        print(f"Processing {len(target_pdbs)} targets with {max_workers} workers (timeout: {timeout}s)")
    
    # Setup output files
    output_jsonl = Path(cfg.results_dir) / "results_stream.jsonl"
    if output_jsonl.exists():
        output_jsonl.unlink()
    
    start_time = time.time()
    results = []
    
    # Use ProcessPoolExecutor like Polaris benchmark
    mp_context = mp.get_context("spawn")
    
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
        # Submit all tasks
        future_to_pdb = {}
        
        for target_pdb in target_pdbs:
            args = (
                target_pdb,
                asdict(cfg),
                exclude_pdb_ids,
                data_dir,
                shared_cache_file,
                shared_embedding_cache,
                peptide_threshold,
            )
            future = executor.submit(simple_worker_task, args)
            future_to_pdb[future] = target_pdb
        
        # Process results with progress bar
        successful = 0
        failed = 0
        
        if not quiet:
            pbar = tqdm(
                total=len(target_pdbs),
                desc="Processing targets",
                unit="targets",
                bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        
        for future in as_completed(future_to_pdb):
            target_pdb = future_to_pdb[future]
            
            try:
                result = future.result(timeout=timeout)
                
                if result.get("success"):
                    successful += 1
                else:
                    failed += 1
                
                results.append(result)
                
                # Write to JSONL
                with output_jsonl.open("a", encoding="utf-8") as fh:
                    json.dump(result, fh)
                    fh.write("\n")
                
            except FutureTimeoutError:
                failed += 1
                timeout_result = {
                    "success": False,
                    "error": f"Task timeout after {timeout}s",
                    "pdb_id": target_pdb,
                    "runtime": timeout,
                    "rmsd_values": {},
                }
                results.append(timeout_result)
                
                with output_jsonl.open("a", encoding="utf-8") as fh:
                    json.dump(timeout_result, fh)
                    fh.write("\n")
                
            except Exception as e:
                failed += 1
                error_result = {
                    "success": False,
                    "error": f"Task failed: {str(e)}",
                    "pdb_id": target_pdb,
                    "runtime": 0,
                    "rmsd_values": {},
                }
                results.append(error_result)
                
                with output_jsonl.open("a", encoding="utf-8") as fh:
                    json.dump(error_result, fh)
                    fh.write("\n")
            
            if not quiet:
                pbar.update(1)
        
        if not quiet:
            pbar.close()
    
    total_time = time.time() - start_time
    
    if not quiet:
        print(f"\nCompleted {len(results)} targets in {total_time:.1f}s")
        print(f"Success: {successful}, Failed: {failed}")
        print(f"Success rate: {successful/len(results)*100:.1f}%")
    
    return {
        "results": results,
        "summary": {
            "total_targets": len(target_pdbs),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(target_pdbs) * 100 if target_pdbs else 0,
            "total_runtime": total_time,
        }
    }


if __name__ == "__main__":
    # Simple test
    test_pdbs = ["1abc", "2def", "3ghi"]
    result = run_simple_timesplit_streaming(
        target_pdbs=test_pdbs,
        data_dir="/tmp/test_data",
        results_dir="/tmp/test_results",
        max_workers=2,
        timeout=60,
    )
    print(json.dumps(result, indent=2))