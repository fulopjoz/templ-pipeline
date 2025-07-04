#!/usr/bin/env python3
"""
Internal benchmark runner for TEMPL pipeline.
Provides benchmark-specific interface to core TEMPL functionality.
"""

import os
import time
import logging
import gc
from typing import Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from templ_pipeline.core.pipeline import TEMPLPipeline
from templ_pipeline.core.utils import (
    load_molecules_with_shared_cache,
    find_ligand_by_pdb_id,
    calculate_rmsd,
    get_protein_file_paths,
    find_ligand_file_paths,
    get_worker_config,
    get_global_molecule_cache,
)

# Memory monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from rdkit import Chem
except ImportError as e:
    raise RuntimeError(
        "Required dependencies not available - please check templ_pipeline/requirements.txt"
    ) from e

# Memory thresholds - updated for realistic usage
MEMORY_WARNING_GB = 6.0  # Warn when process uses >6GB
MEMORY_CRITICAL_GB = 8.0  # Critical when process uses >8GB


@dataclass
class BenchmarkParams:
    """Parameters for benchmark execution."""

    target_pdb: str
    exclude_pdb_ids: Set[str]
    poses_output_dir: Optional[str] = None
    n_conformers: int = 200
    template_knn: int = 100
    similarity_threshold: Optional[float] = None
    internal_workers: int = 1
    timeout: int = 1800


@dataclass
class BenchmarkResult:
    """Structured benchmark result."""

    success: bool
    rmsd_values: Dict[str, Dict[str, float]]
    runtime: float
    error: Optional[str]
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary format compatible with existing benchmark code."""
        try:
            return {
                "success": self.success,
                "rmsd_values": self.rmsd_values or {},
                "runtime": self.runtime,
                "error": str(self.error) if self.error is not None else None,
            }
        except Exception as e:
            # Fallback if serialization fails
            return {
                "success": False,
                "rmsd_values": {},
                "runtime": 0.0,
                "error": f"Serialization failed: {str(e)}"
            }


def monitor_memory_usage() -> Dict[str, float]:
    """Monitor current process memory usage."""
    if not PSUTIL_AVAILABLE:
        return {"memory_gb": 0.0, "warning": False, "critical": False}

    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024**3)

        return {
            "memory_gb": memory_gb,
            "warning": memory_gb > MEMORY_WARNING_GB,
            "critical": memory_gb > MEMORY_CRITICAL_GB,
        }
    except Exception:
        return {"memory_gb": 0.0, "warning": False, "critical": False}


def cleanup_memory():
    """Aggressive memory cleanup."""
    try:
        gc.collect()
        if hasattr(gc, "set_threshold"):
            gc.set_threshold(700, 10, 10)
    except Exception:
        pass


class BenchmarkRunner:
    """Memory-optimized TEMPL pipeline runner for benchmarks."""

    def __init__(self, data_dir: str, poses_output_dir: Optional[str] = None, enable_error_tracking: bool = True, shared_cache_file: Optional[str] = None, shared_embedding_cache: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.poses_output_dir = Path(poses_output_dir) if poses_output_dir else None
        self.pipeline = None
        self.log = logging.getLogger(__name__)
        self._molecule_cache = None  # Will use shared cache
        self._shared_cache_file = shared_cache_file
        self._shared_embedding_cache = shared_embedding_cache
        
        # Initialize error tracking if enabled
        self.error_tracker = None
        if enable_error_tracking:
            try:
                from .error_tracking import BenchmarkErrorTracker
                workspace_dir = self.poses_output_dir.parent if self.poses_output_dir else Path.cwd()
                self.error_tracker = BenchmarkErrorTracker(workspace_dir)
                self.log.info("Error tracking enabled")
            except ImportError:
                self.log.warning("Error tracking module not available")
        
        self._setup_components()

    def _setup_components(self):
        """Initialize TEMPL components with memory monitoring."""
        memory_status = monitor_memory_usage()
        if memory_status["critical"]:
            self.log.error(
                f"Critical memory usage at startup: {memory_status['memory_gb']:.1f}GB"
            )
            self.pipeline = None
            return

        try:
            # Use correct subdirectory structure for embeddings
            embedding_path = (
                self.data_dir / "embeddings" / "protein_embeddings_base.npz"
            )

            # Set output directory
            if self.poses_output_dir:
                output_dir = self.poses_output_dir
            else:
                output_dir = self.data_dir / "benchmark_output"

            self.log.debug(
                f"Initializing TEMPL pipeline with data_dir: {self.data_dir}"
            )
            self.log.debug(f"Memory status: {memory_status['memory_gb']:.1f}GB")

            if embedding_path.exists():
                self.pipeline = TEMPLPipeline(
                    embedding_path=str(embedding_path), 
                    output_dir=str(output_dir),
                    shared_embedding_cache=self._shared_embedding_cache
                )
            else:
                # Check legacy location as fallback
                alt_embedding_path = self.data_dir / "protein_embeddings_base.npz"

                if alt_embedding_path.exists():
                    self.log.info(f"Using legacy embedding path: {alt_embedding_path}")
                    self.pipeline = TEMPLPipeline(
                        embedding_path=str(alt_embedding_path),
                        output_dir=str(output_dir),
                        shared_embedding_cache=self._shared_embedding_cache
                    )
                else:
                    self.log.warning(
                        "No embeddings found, initializing pipeline without embeddings"
                    )
                    self.pipeline = TEMPLPipeline(
                        output_dir=str(output_dir),
                        shared_embedding_cache=self._shared_embedding_cache
                    )

            self.log.info(
                f"TEMPL pipeline initialized with {memory_status['memory_gb']:.1f}GB memory usage"
            )

        except Exception as e:
            self.log.error(f"Pipeline initialization failed: {e}")
            self.pipeline = None

    def _load_ligand_data_from_sdf(
        self, pdb_id: str
    ) -> Tuple[Optional[str], Optional["Chem.Mol"]]:
        """Load ligand SMILES and crystal molecule using shared cache."""

        # Use shared cache if available, fallback to optimized loading
        if self._molecule_cache is None:
            try:
                # First try to get from global cache in utils
                global_cache = get_global_molecule_cache()
                if global_cache and "molecules" in global_cache:
                    self._molecule_cache = global_cache["molecules"]
                    self.log.info(
                        f"Using global molecule cache: {len(self._molecule_cache)} molecules"
                    )
                else:
                    # Check for shared cache file from multiprocessing
                    shared_cache_file = getattr(self, '_shared_cache_file', None)
                    # Fallback to original loading method
                    self._molecule_cache = load_molecules_with_shared_cache(
                        self.data_dir, shared_cache_file=shared_cache_file
                    )
                    if not self._molecule_cache:
                        self.log.error("Failed to load molecule cache")
                        return None, None
                    else:
                        self.log.info(
                            f"Loaded {len(self._molecule_cache)} molecules into cache"
                        )
            except Exception as e:
                self.log.error(f"Failed to load molecules: {e}")
                return None, None

        return find_ligand_by_pdb_id(pdb_id, self._molecule_cache)

    def _calculate_rmsd_to_crystal(
        self, pose_mol: "Chem.Mol", crystal_mol: "Chem.Mol"
    ) -> float:
        """Calculate RMSD between pose and crystal ligand using shared utility."""
        return calculate_rmsd(pose_mol, crystal_mol)

    def run_single_target(self, params: BenchmarkParams) -> BenchmarkResult:
        """Run TEMPL pipeline for single target with memory monitoring."""
        if self.pipeline is None:
            return BenchmarkResult(
                success=False,
                rmsd_values={},
                runtime=0.0,
                error="Pipeline not initialized",
            )

        start_time = time.time()

        # Initialize variables to prevent reference errors
        rmsd_values = {}
        pipeline_result = None
        crystal_mol = None
        effective_exclusions = set()

        # Initialize error tracking if available
        error_tracker = getattr(self, 'error_tracker', None)

        # Initial memory check
        memory_status = monitor_memory_usage()
        if memory_status["critical"]:
            if error_tracker:
                error_tracker.record_target_failure(
                    params.target_pdb, 
                    f"Critical memory usage: {memory_status['memory_gb']:.1f}GB"
                )
            return BenchmarkResult(
                success=False,
                rmsd_values={},
                runtime=0.0,
                error=f"Critical memory usage before processing: {memory_status['memory_gb']:.1f}GB",
            )

        if memory_status["warning"]:
            self.log.warning(
                f"High memory usage for {params.target_pdb}: {memory_status['memory_gb']:.1f}GB"
            )

        try:
            # Load protein file for target PDB with graceful error handling
            try:
                protein_file = self._get_protein_file(params.target_pdb)
            except FileNotFoundError as e:
                if error_tracker:
                    error_tracker.record_missing_pdb(
                        params.target_pdb,
                        "protein_file_not_found",
                        str(e),
                        "protein"
                    )
                raise ValueError(f"Protein file not found for {params.target_pdb}: {e}")

            # Load ligand data using optimized shared cache with graceful error handling
            try:
                ligand_smiles, crystal_mol = self._load_ligand_data_from_sdf(
                    params.target_pdb
                )
                if not ligand_smiles or crystal_mol is None:
                    if error_tracker:
                        error_tracker.record_missing_pdb(
                            params.target_pdb,
                            "ligand_not_found",
                            "Ligand data not found in database",
                            "ligand"
                        )
                    raise ValueError(f"Could not load ligand data for {params.target_pdb}")
            except Exception as e:
                if error_tracker:
                    error_tracker.record_missing_pdb(
                        params.target_pdb,
                        "ligand_load_failed",
                        str(e),
                        "ligand"
                    )
                raise ValueError(f"Could not load ligand data for {params.target_pdb}: {e}")

            # Early molecular quality validation
            try:
                from templ_pipeline.core.scoring import validate_molecule_quality

                is_valid, msg = validate_molecule_quality(crystal_mol)
                if not is_valid:
                    self.log.debug(
                        f"Skipping {params.target_pdb} due to crystal quality: {msg}"
                    )
                    raise ValueError(f"Poor quality crystal structure: {msg}")
            except ImportError:
                # Continue if validation not available
                pass

            self.log.info(f"Running TEMPL pipeline for {params.target_pdb}")

            # For LOO approach, exclude target PDB from template pool
            effective_exclusions = params.exclude_pdb_ids.copy()
            effective_exclusions.add(params.target_pdb)

            # Input validation
            if not os.path.exists(protein_file):
                raise FileNotFoundError(f"Protein file not found: {protein_file}")

            if len(ligand_smiles.strip()) == 0:
                raise ValueError(f"Empty ligand SMILES for {params.target_pdb}")

            # Memory check before pipeline execution
            memory_status = monitor_memory_usage()
            if memory_status["critical"]:
                raise RuntimeError(
                    f"Critical memory usage before pipeline: {memory_status['memory_gb']:.1f}GB"
                )

            # Run TEMPL pipeline with memory-safe settings
            try:
                self.log.info(f"Starting pipeline execution for {params.target_pdb}")
                self.log.debug(
                    f"Pipeline params: templates={params.template_knn}, conformers={params.n_conformers}, workers={params.internal_workers}"
                )
                self.log.debug(f"Exclusions: {len(effective_exclusions)} PDBs")

                # Use benchmark-specific output directory to prevent root directory pollution
                benchmark_output_dir = str(self.poses_output_dir) if self.poses_output_dir else None
                
                pipeline_result = self.pipeline.run_full_pipeline(
                    protein_file=protein_file,
                    protein_pdb_id=params.target_pdb,
                    ligand_smiles=ligand_smiles,
                    num_templates=params.template_knn,
                    num_conformers=params.n_conformers,
                    n_workers=params.internal_workers,  # Always 1 to prevent nested parallelization
                    similarity_threshold=params.similarity_threshold,
                    exclude_pdb_ids=effective_exclusions,
                    output_dir=benchmark_output_dir,
                )

                self.log.debug(
                    f"Pipeline result keys: {list(pipeline_result.keys()) if isinstance(pipeline_result, dict) else 'Not a dict'}"
                )

            except Exception as pipeline_error:
                # Handle skip exceptions gracefully
                from ..core.skip_manager import MoleculeSkipException, skip_molecule, SkipReason
                
                if isinstance(pipeline_error, MoleculeSkipException):
                    # Record the skip and return gracefully
                    skip_molecule(
                        params.target_pdb, 
                        pipeline_error.reason, 
                        pipeline_error.message, 
                        pipeline_error.details
                    )
                    self.log.info(f"Molecule {params.target_pdb} skipped: {pipeline_error.message}")
                    
                    # Return a special skip result that can be handled by benchmark
                    return BenchmarkResult(
                        success=False,
                        rmsd_values={},
                        runtime=time.time() - start_time,
                        error=f"SKIPPED ({pipeline_error.reason.value}): {pipeline_error.message}",
                        metadata={
                            "target_pdb": params.target_pdb,
                            "status": "skipped",
                            "skip_reason": pipeline_error.reason.value,
                            "skip_message": pipeline_error.message,
                        }
                    )
                else:
                    # Handle regular errors as before
                    self.log.error(
                        f"Pipeline execution failed for {params.target_pdb}: {pipeline_error}"
                    )
                    self.log.error(f"Error type: {type(pipeline_error).__name__}")
                    import traceback

                    self.log.error(f"Traceback: {traceback.format_exc()}")
                    raise RuntimeError(f"Pipeline execution failed: {str(pipeline_error)}")

            # Validate pipeline result structure
            if not isinstance(pipeline_result, dict):
                raise ValueError(
                    f"Pipeline returned invalid result type: {type(pipeline_result)}"
                )

            if "poses" not in pipeline_result:
                self.log.warning(f"No poses in pipeline result for {params.target_pdb}")
                pipeline_result["poses"] = {}

            # Extract and calculate RMSD values
            rmsd_values = self._extract_and_calculate_rmsd(pipeline_result, crystal_mol)

            self.log.debug(
                f"RMSD calculation result for {params.target_pdb}: {len(rmsd_values)} metrics calculated"
            )
            if rmsd_values:
                for metric, values in rmsd_values.items():
                    self.log.debug(
                        f"  {metric}: RMSD={values.get('rmsd', 'N/A'):.3f}, Score={values.get('score', 'N/A'):.3f}"
                    )
            else:
                self.log.warning(
                    f"No RMSD values calculated for {params.target_pdb} - checking pipeline result structure"
                )
                self.log.debug(f"Pipeline result type: {type(pipeline_result)}")
                if isinstance(pipeline_result, dict):
                    self.log.debug(
                        f"Pipeline result keys: {list(pipeline_result.keys())}"
                    )
                    if "poses" in pipeline_result:
                        poses = pipeline_result["poses"]
                        self.log.debug(
                            f"Poses type: {type(poses)}, length: {len(poses) if poses else 0}"
                        )
                        if poses:
                            for k, v in poses.items():
                                self.log.debug(
                                    f"  Pose {k}: type={type(v)}, valid={v is not None}"
                                )

            runtime = time.time() - start_time

            # Aggressive memory cleanup and monitoring
            cleanup_memory()

            final_memory = monitor_memory_usage()
            if final_memory["warning"]:
                self.log.warning(
                    f"High memory after processing {params.target_pdb}: {final_memory['memory_gb']:.1f}GB"
                )

            # Additional cleanup for molecular objects
            try:
                if pipeline_result is not None:
                    del pipeline_result
                if crystal_mol is not None:
                    del crystal_mol
                import gc

                gc.collect()
            except Exception:
                pass

            if rmsd_values:
                # Record successful target processing
                if self.error_tracker:
                    self.error_tracker.record_target_success(params.target_pdb)
                    
                self.log.info(
                    f"TEMPL completed for {params.target_pdb} in {runtime:.1f}s"
                )
                return BenchmarkResult(
                    success=True,
                    rmsd_values=rmsd_values,
                    runtime=runtime,
                    error=None,
                    metadata={
                        "target_pdb": params.target_pdb,
                        "excluded_count": len(effective_exclusions),
                        "final_memory_gb": final_memory["memory_gb"],
                    },
                )
            else:
                # Record failed target processing
                error_msg = "No poses generated or RMSD calculation failed"
                if self.error_tracker:
                    self.error_tracker.record_target_failure(params.target_pdb, error_msg)
                    
                self.log.warning(
                    f"TEMPL completed for {params.target_pdb} but no RMSD values"
                )
                return BenchmarkResult(
                    success=False,
                    rmsd_values={},
                    runtime=runtime,
                    error=error_msg,
                    metadata={
                        "target_pdb": params.target_pdb,
                        "excluded_count": len(effective_exclusions),
                        "final_memory_gb": final_memory["memory_gb"],
                    },
                )

        except FileNotFoundError as e:
            runtime = time.time() - start_time
            if self.error_tracker:
                self.error_tracker.record_target_failure(params.target_pdb, f"File not found: {str(e)}")
            self.log.error(f"File not found for {params.target_pdb}: {e}")
            cleanup_memory()
            return BenchmarkResult(
                success=False,
                rmsd_values={},
                runtime=runtime,
                error=f"File not found: {str(e)}",
            )
        except ValueError as e:
            runtime = time.time() - start_time
            if self.error_tracker:
                self.error_tracker.record_target_failure(params.target_pdb, f"Invalid input data: {str(e)}")
            self.log.error(f"Invalid input data for {params.target_pdb}: {e}")
            cleanup_memory()
            return BenchmarkResult(
                success=False,
                rmsd_values={},
                runtime=runtime,
                error=f"Invalid input data: {str(e)}",
            )
        except Exception as e:
            runtime = time.time() - start_time
            if self.error_tracker:
                self.error_tracker.record_target_failure(params.target_pdb, str(e))
            self.log.error(f"TEMPL pipeline failed for {params.target_pdb}: {e}")
            cleanup_memory()
            return BenchmarkResult(
                success=False, rmsd_values={}, runtime=runtime, error=str(e)
            )

    def _get_protein_file(self, pdb_id: str) -> str:
        """Get protein file path for PDB ID using shared utilities."""
        search_paths = get_protein_file_paths(pdb_id, self.data_dir)

        for protein_file in search_paths:
            if protein_file.exists():
                return str(protein_file)

        raise FileNotFoundError(
            f"Protein file not found for {pdb_id} in any of the search paths"
        )

    def _extract_and_calculate_rmsd(
        self, pipeline_result: Dict, crystal_mol: Chem.Mol
    ) -> Dict[str, Dict[str, float]]:
        """Extract poses from TEMPL CLI pipeline result and calculate RMSD to crystal ligand.

        Args:
            pipeline_result: Result dictionary from TEMPLPipeline.run_full_pipeline
            crystal_mol: Crystal ligand molecule for RMSD reference

        Returns:
            Dictionary with metric -> {"rmsd": float, "score": float} mapping
        """
        rmsd_values = {}

        try:
            # Validate pipeline result structure
            if not isinstance(pipeline_result, dict):
                self.log.error("Pipeline result is not a dictionary")
                return rmsd_values

            if "poses" not in pipeline_result:
                self.log.warning("No poses found in pipeline result")
                return rmsd_values

            poses = pipeline_result["poses"]
            if not poses:
                self.log.warning("Empty poses dictionary in pipeline result")
                return rmsd_values

            # Validate crystal molecule
            if crystal_mol is None:
                self.log.error("Crystal molecule is None")
                return rmsd_values

            if crystal_mol.GetNumConformers() == 0:
                self.log.error("Crystal molecule has no conformers")
                return rmsd_values

            # Prepare crystal ligand for RMSD calculation (remove hydrogens for consistency)
            try:
                crystal_ref = Chem.RemoveHs(Chem.Mol(crystal_mol))
                if crystal_ref.GetNumAtoms() == 0:
                    self.log.error("Crystal molecule has no heavy atoms")
                    return rmsd_values
            except Exception as e:
                self.log.error(f"Failed to process crystal molecule: {e}")
                return rmsd_values

            # Process each pose metric (shape, color, combo)
            for metric, pose_data in poses.items():
                try:
                    # Validate pose data structure
                    if not isinstance(pose_data, tuple) or len(pose_data) != 2:
                        self.log.warning(
                            f"Invalid pose data format for metric {metric}: expected tuple of length 2, got {type(pose_data)}"
                        )
                        continue

                    pose_mol, scores_dict = pose_data

                    if pose_mol is None:
                        self.log.warning(f"No pose molecule for metric {metric}")
                        continue

                    # Validate pose molecule
                    if pose_mol.GetNumConformers() == 0:
                        self.log.warning(
                            f"Pose molecule for {metric} has no conformers"
                        )
                        continue

                    # Calculate RMSD to crystal ligand
                    try:
                        pose_no_h = Chem.RemoveHs(Chem.Mol(pose_mol))
                        if pose_no_h.GetNumAtoms() == 0:
                            self.log.warning(
                                f"Pose molecule for {metric} has no heavy atoms"
                            )
                            continue

                        rmsd = self._calculate_rmsd_to_crystal(pose_no_h, crystal_ref)

                        if rmsd is None or np.isnan(rmsd):
                            self.log.warning(
                                f"RMSD calculation returned invalid value for {metric}"
                            )
                            continue

                    except Exception as e:
                        self.log.warning(f"RMSD calculation failed for {metric}: {e}")
                        continue

                    # Extract score for this metric with validation
                    score = 0.0
                    try:
                        if isinstance(scores_dict, dict):
                            if metric in scores_dict:
                                score = float(scores_dict[metric])
                            elif "score" in scores_dict:
                                score = float(scores_dict["score"])
                            elif "similarity_score" in scores_dict:
                                score = float(scores_dict["similarity_score"])
                            else:
                                self.log.warning(
                                    f"No score found for metric {metric} in scores dict: {scores_dict.keys()}"
                                )
                        else:
                            self.log.warning(
                                f"Scores dict for {metric} is not a dictionary: {type(scores_dict)}"
                            )
                    except (ValueError, TypeError) as e:
                        self.log.warning(f"Failed to extract score for {metric}: {e}")
                        score = 0.0

                    # Store valid results
                    key_name = "canimotocombo" if metric == "combo" else metric
                    rmsd_values[key_name] = {"rmsd": float(rmsd), "score": float(score)}
                    self.log.debug(
                        f"Calculated RMSD for {metric}: {rmsd:.3f}Å, score: {score:.3f}"
                    )

                except Exception as e:
                    self.log.warning(f"Failed to process pose for metric {metric}: {e}")
                    continue

            if rmsd_values:
                self.log.info(
                    f"Successfully calculated RMSD for {len(rmsd_values)} metrics"
                )
            else:
                self.log.warning("No valid RMSD values calculated for any metric")

        except Exception as e:
            self.log.error(f"Error in RMSD extraction and calculation: {e}")

        return rmsd_values


def run_templ_pipeline_for_benchmark(
    target_pdb: str,
    exclude_pdb_ids: Set[str],
    n_conformers: int = 200,
    template_knn: int = 100,
    similarity_threshold: Optional[float] = None,
    internal_workers: int = 1,
    timeout: int = 1800,
    data_dir: str = None,
    poses_output_dir: str = None,
    shared_cache_file: str = None,
    shared_embedding_cache: str = None,
) -> Dict:
    """Main entry point for benchmark pipeline execution."""

    if data_dir is None:
        # Default to templ_pipeline/data
        current_dir = Path(__file__).parent.parent
        data_dir = str(current_dir / "data")

    params = BenchmarkParams(
        target_pdb=target_pdb,
        exclude_pdb_ids=exclude_pdb_ids,
        poses_output_dir=poses_output_dir,
        n_conformers=n_conformers,
        template_knn=template_knn,
        similarity_threshold=similarity_threshold,
        internal_workers=internal_workers,
        timeout=timeout,
    )

    runner = BenchmarkRunner(
        data_dir, 
        poses_output_dir, 
        shared_cache_file=shared_cache_file,
        shared_embedding_cache=shared_embedding_cache
    )
    result = runner.run_single_target(params)
    
    # Ensure result is properly converted to dictionary
    if hasattr(result, 'to_dict'):
        return result.to_dict()
    elif isinstance(result, dict):
        return result
    else:
        # Fallback for unexpected return types
        return {
            "success": False,
            "rmsd_values": {},
            "runtime": 0.0,
            "error": f"Unexpected result type: {type(result)}"
        }
