#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Internal benchmark runner for TEMPL pipeline.
Provides benchmark-specific interface to core TEMPL functionality.
"""

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np

# Lazy imports to speed up module loading
# from templ_pipeline.core.pipeline import TEMPLPipeline  # Moved to lazy import
# from templ_pipeline.core.utils import (...)  # Moved to lazy import

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

logger = logging.getLogger(__name__)

# Memory thresholds - updated for realistic usage
MEMORY_WARNING_GB = 6.0  # Warn when process uses >6GB
MEMORY_CRITICAL_GB = 8.0  # Critical when process uses >8GB


class SharedMolecularCache:
    """Singleton shared molecular cache to prevent per-worker loading."""

    _instance = None
    _cache_data = None
    _data_dir = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def initialize(cls, data_dir: str):
        """Initialize the shared cache once."""
        if cls._cache_data is None:
            cls._data_dir = data_dir
            # Use lazy loading strategy instead of preloading massive cache
            cls._cache_data = {"initialized": True, "data_dir": data_dir}

    @classmethod
    def get_data_dir(cls):
        """Get the data directory for cache access."""
        return cls._data_dir

    @classmethod
    def is_initialized(cls):
        """Check if cache is initialized."""
        return cls._cache_data is not None


class LazyMoleculeLoader:
    """Truly lazy molecule loader that only loads specific molecules on demand."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.log = logging.getLogger(__name__)
        self._sdf_path = None
        self._molecule_cache = {}  # Cache for individual molecules

    def _find_sdf_file(self):
        """Find the SDF file without loading it."""
        if self._sdf_path is not None:
            return self._sdf_path

        try:
            from templ_pipeline.core.utils import find_ligand_file_paths

            ligand_file_paths = find_ligand_file_paths(self.data_dir)
            for ligand_path in ligand_file_paths:
                if ligand_path.exists():
                    self._sdf_path = ligand_path
                    self.log.info(f"Found SDF file: {ligand_path.name}")
                    return self._sdf_path
        except Exception as e:
            self.log.warning(f"Failed to find SDF file: {e}")

        return None

    def get_ligand_data(
        self, pdb_id: str
    ) -> Tuple[Optional[str], Optional["Chem.Mol"]]:
        """Load specific ligand data on demand using efficient lookup."""
        try:
            # Check cache first
            if pdb_id in self._molecule_cache:
                return self._molecule_cache[pdb_id]

            # Find SDF file
            sdf_path = self._find_sdf_file()
            if not sdf_path:
                self.log.error(f"No SDF file found for ligand search")
                return None, None

            # Load specific molecule by PDB ID
            molecule, smiles = self._load_specific_molecule(sdf_path, pdb_id)

            # Cache the result
            self._molecule_cache[pdb_id] = (smiles, molecule)

            return smiles, molecule

        except Exception as e:
            self.log.error(f"Failed to load ligand data for {pdb_id}: {e}")
            return None, None

    def _load_specific_molecule(
        self, sdf_path: Path, target_pdb_id: str
    ) -> Tuple[Optional["Chem.Mol"], Optional[str]]:
        """Load a specific molecule by PDB ID without loading the entire file."""
        try:
            if sdf_path.suffix == ".gz":
                import gzip
                import io

                # Read compressed file in chunks
                with gzip.open(sdf_path, "rb") as fh:
                    content = fh.read()

                # Process from memory buffer
                with io.BytesIO(content) as buffer:
                    return self._search_molecule_in_supplier(buffer, target_pdb_id)
            else:
                # Handle uncompressed SDF files
                with open(sdf_path, "rb") as fh:
                    return self._search_molecule_in_supplier(fh, target_pdb_id)

        except Exception as e:
            self.log.error(f"Failed to load specific molecule {target_pdb_id}: {e}")
            return None, None

    def _search_molecule_in_supplier(
        self, file_handle, target_pdb_id: str
    ) -> Tuple[Optional["Chem.Mol"], Optional[str]]:
        """Search for a specific molecule in an SDMolSupplier."""
        try:
            supplier = Chem.ForwardSDMolSupplier(
                file_handle, removeHs=False, sanitize=False
            )

            for idx, mol in enumerate(supplier):
                try:
                    if mol is None or not mol.HasProp("_Name"):
                        continue

                    mol_name = mol.GetProp("_Name")

                    # Check if this is the molecule we're looking for
                    if mol_name.lower() == target_pdb_id.lower():
                        mol.SetProp("original_name", mol_name)
                        mol.SetProp("molecule_index", str(idx))

                        # Extract SMILES
                        smiles = Chem.MolToSmiles(mol) if mol else None

                        self.log.debug(f"Found molecule {target_pdb_id} at index {idx}")
                        return mol, smiles

                except Exception as mol_err:
                    continue

            self.log.warning(f"Molecule {target_pdb_id} not found in SDF file")
            return None, None

        except Exception as e:
            self.log.error(f"Error searching for molecule {target_pdb_id}: {e}")
            return None, None

    def _load_specific_molecule_fallback(
        self, sdf_path: Path, target_pdb_id: str
    ) -> Tuple[Optional["Chem.Mol"], Optional[str]]:
        """Fallback method using memory-efficient loading with limits."""
        try:
            from templ_pipeline.core.utils import load_sdf_molecules_cached

            # Load with very low memory limit to prevent explosion
            molecules = load_sdf_molecules_cached(
                sdf_path, cache={}, memory_limit_gb=2.0
            )

            if not molecules:
                return None, None

            # Search in loaded molecules
            from templ_pipeline.core.utils import find_ligand_by_pdb_id

            smiles, molecule = find_ligand_by_pdb_id(target_pdb_id, molecules)

            # Clear the molecules list to free memory immediately
            molecules.clear()
            del molecules

            return molecule, smiles

        except Exception as e:
            self.log.error(f"Fallback loading failed for {target_pdb_id}: {e}")
            return None, None


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
    timeout: int = 300
    unconstrained: bool = False
    align_metric: str = "combo"
    enable_optimization: bool = False
    no_realign: bool = False
    allowed_pdb_ids: Optional[Set[str]] = None  # NEW: restrict template search


@dataclass
class BenchmarkResult:
    """Structured benchmark result."""

    success: bool
    rmsd_values: Dict[str, Dict[str, float]]
    runtime: float
    error: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

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
                "error": f"Serialization failed: {str(e)}",
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

    def __init__(
        self,
        data_dir: str,
        poses_output_dir: Optional[str] = None,
        enable_error_tracking: bool = True,
        shared_cache_file: Optional[str] = None,
        peptide_threshold: int = 8,
    ):
        self.data_dir = Path(data_dir)
        self.poses_output_dir = Path(poses_output_dir) if poses_output_dir else None
        self.pipeline = None
        self.log = logging.getLogger(__name__)
        self._molecule_cache = None  # Will use shared cache
        self._shared_cache_file = shared_cache_file
        self.peptide_threshold = peptide_threshold

        # Initialize error tracking if enabled
        self.error_tracker = None
        if enable_error_tracking:
            try:
                from .error_tracking import BenchmarkErrorTracker

                workspace_dir = (
                    self.poses_output_dir.parent
                    if self.poses_output_dir
                    else Path.cwd()
                )
                self.error_tracker = BenchmarkErrorTracker(workspace_dir)
                self.log.info("Error tracking enabled")
            except ImportError:
                self.log.warning("Error tracking module not available")

        # Initialize skip tracking
        self.skip_tracker = None
        try:
            from .skip_tracker import BenchmarkSkipTracker

            workspace_dir = (
                self.poses_output_dir.parent if self.poses_output_dir else Path.cwd()
            )
            self.skip_tracker = BenchmarkSkipTracker(workspace_dir)
            self.skip_tracker.load_existing_skips()  # Load any existing skip records
            self.log.info("Skip tracking enabled")
        except ImportError:
            self.log.warning("Skip tracking module not available")

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
                self.data_dir / "embeddings" / "templ_protein_embeddings_v1.0.0.npz"
            )

            # Set output directory
            if self.poses_output_dir:
                output_dir = self.poses_output_dir
            else:
                # Default to a subdirectory in the data_dir
                output_dir = self.data_dir / "benchmark_outputs"
                output_dir.mkdir(parents=True, exist_ok=True)

            self.log.debug(f"Using output_dir for TEMPLPipeline: {output_dir}")

            self.log.debug(
                f"Initializing TEMPL pipeline with data_dir: {self.data_dir}"
            )
            self.log.debug(f"Memory status: {memory_status['memory_gb']:.1f}GB")

            # Lazy import to avoid slow import at module level
            from templ_pipeline.core.pipeline import TEMPLPipeline

            if embedding_path.exists():
                self.pipeline = TEMPLPipeline(
                    embedding_path=str(embedding_path), output_dir=str(output_dir)
                )
            else:
                self.log.warning(
                    f"Embedding file not found at {embedding_path}, initializing pipeline without embeddings"
                )
                self.pipeline = TEMPLPipeline(output_dir=str(output_dir))

            self.log.info(
                f"TEMPL pipeline initialized with {memory_status['memory_gb']:.1f}GB memory usage"
            )

        except Exception as e:
            self.log.error(f"Pipeline initialization failed: {e}")
            self.pipeline = None

    def _load_ligand_data_from_sdf(
        self, pdb_id: str
    ) -> Tuple[Optional[str], Optional["Chem.Mol"]]:
        """Load ligand SMILES and crystal molecule using lazy loading to prevent memory explosion."""

        # Use lazy loading approach to prevent per-worker 6GB cache loading
        if not hasattr(self, "molecule_loader") or self.molecule_loader is None:
            # Initialize lazy loader if not already done
            if not SharedMolecularCache.is_initialized():
                SharedMolecularCache.initialize(str(self.data_dir))

            self.molecule_loader = LazyMoleculeLoader(str(self.data_dir))
            self.log.info(
                "Initialized lazy molecule loader to prevent memory explosion"
            )

        # Load specific ligand data on demand
        return self.molecule_loader.get_ligand_data(pdb_id)

    def _calculate_rmsd_to_crystal(
        self, pose_mol: "Chem.Mol", crystal_mol: "Chem.Mol"
    ) -> float:
        """Calculate RMSD between pose and crystal ligand using shared utility."""
        # Lazy import to avoid slow import at module level
        from templ_pipeline.core.utils import calculate_rmsd

        # Skip alignment for pose prediction benchmarking - measure original prediction accuracy
        return calculate_rmsd(pose_mol, crystal_mol, skip_alignment=True)

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

        # Log memory usage before processing
        try:
            proc = psutil.Process(os.getpid())
            mem_before = proc.memory_info().rss / 1e9
            self.log.info(
                f"[MEMORY] Before pipeline for {params.target_pdb}: {mem_before:.2f} GB"
            )
        except Exception as e:
            self.log.warning(f"Could not log memory before: {e}")

        # Initialize variables to prevent reference errors
        rmsd_values = {}
        pipeline_result = None
        crystal_mol = None
        effective_exclusions = set()

        # Initialize error tracking if available
        error_tracker = getattr(self, "error_tracker", None)

        # Initial memory check
        memory_status = monitor_memory_usage()
        if memory_status["critical"]:
            if error_tracker:
                error_tracker.record_target_failure(
                    params.target_pdb,
                    f"Critical memory usage: {memory_status['memory_gb']:.1f}GB",
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
                        params.target_pdb, "protein_file_not_found", str(e), "protein"
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
                            "ligand",
                        )
                    self.log.warning(
                        f"Could not load ligand data for {params.target_pdb}"
                    )
                    raise ValueError(
                        f"Could not load ligand data for {params.target_pdb}"
                    )
            except Exception as e:
                if error_tracker:
                    error_tracker.record_missing_pdb(
                        params.target_pdb, "ligand_load_failed", str(e), "ligand"
                    )
                raise ValueError(
                    f"Could not load ligand data for {params.target_pdb}: {e}"
                )

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

            # Log template restriction information
            if params.allowed_pdb_ids is not None:
                self.log.info(
                    f"Template search restricted to {len(params.allowed_pdb_ids)} allowed PDBs for {params.target_pdb}"
                )
                if params.target_pdb.lower() in params.allowed_pdb_ids:
                    self.log.error(
                        f"CRITICAL: Target {params.target_pdb} found in allowed templates - data leak risk!"
                    )
            else:
                self.log.debug(f"No template restrictions for {params.target_pdb}")

            self.log.info(
                f"Excluding {len(effective_exclusions)} PDBs from template search for {params.target_pdb}"
            )

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
                benchmark_output_dir = (
                    str(self.poses_output_dir) if self.poses_output_dir else None
                )

                if params.similarity_threshold is not None:
                    pipeline_result = self.pipeline.run_full_pipeline(
                        protein_file=protein_file,
                        protein_pdb_id=params.target_pdb,
                        ligand_smiles=ligand_smiles,
                        num_templates=params.template_knn,
                        num_conformers=params.n_conformers,
                        n_workers=params.internal_workers,  # Always 1 to prevent nested parallelization
                        similarity_threshold=float(params.similarity_threshold),
                        exclude_pdb_ids=effective_exclusions,
                        allowed_pdb_ids=params.allowed_pdb_ids,  # NEW: restrict template search space
                        output_dir=benchmark_output_dir,
                        unconstrained=params.unconstrained,
                        align_metric=params.align_metric,
                        enable_optimization=params.enable_optimization,
                        no_realign=params.no_realign,
                    )
                else:
                    pipeline_result = self.pipeline.run_full_pipeline(
                        protein_file=protein_file,
                        protein_pdb_id=params.target_pdb,
                        ligand_smiles=ligand_smiles,
                        num_templates=params.template_knn,
                        num_conformers=params.n_conformers,
                        n_workers=params.internal_workers,  # Always 1 to prevent nested parallelization
                        exclude_pdb_ids=effective_exclusions,
                        allowed_pdb_ids=params.allowed_pdb_ids,  # NEW: restrict template search space
                        output_dir=benchmark_output_dir,
                        unconstrained=params.unconstrained,
                        align_metric=params.align_metric,
                        enable_optimization=params.enable_optimization,
                        no_realign=params.no_realign,
                    )

                self.log.debug(
                    f"Pipeline result keys: {list(pipeline_result.keys()) if isinstance(pipeline_result, dict) else 'Not a dict'}"
                )

            except Exception as pipeline_error:
                # Handle molecule validation exceptions and other skip cases
                from ..core.pipeline import MoleculeValidationException

                # Check for molecule validation exception
                if hasattr(pipeline_error, "reason") and hasattr(
                    pipeline_error, "details"
                ):
                    # This is a validation exception
                    if self.skip_tracker:
                        self.skip_tracker.track_skip(
                            params.target_pdb,
                            getattr(pipeline_error, "reason", "validation_failed"),
                            str(pipeline_error),
                            getattr(pipeline_error, "molecule_info", None),
                        )

                    self.log.info(
                        f"Molecule {params.target_pdb} skipped: {pipeline_error}"
                    )

                    # Return a special skip result that can be handled by benchmark
                    return BenchmarkResult(
                        success=False,
                        rmsd_values={},
                        runtime=time.time() - start_time,
                        error=f"SKIPPED ({getattr(pipeline_error, 'reason', 'validation_failed')}): {pipeline_error}",
                        metadata={
                            "target_pdb": params.target_pdb,
                            "status": "skipped",
                            "skip_reason": getattr(
                                pipeline_error, "reason", "validation_failed"
                            ),
                            "skip_message": str(pipeline_error),
                        },
                    )
                else:
                    # Enhanced error handling for pipeline failures
                    error_msg = str(pipeline_error)
                    error_type = type(pipeline_error).__name__

                    # Categorize different types of pipeline failures
                    if (
                        "No poses generated" in error_msg
                        or "RMSD calculation failed" in error_msg
                    ):
                        error_category = "pose_generation_failed"
                        detailed_msg = (
                            f"Pose generation or RMSD calculation failed: {error_msg}"
                        )
                    elif "EmbedMultipleConfs failed" in error_msg:
                        error_category = "conformer_generation_failed"
                        detailed_msg = f"RDKit conformer generation failed: {error_msg}"
                    elif "AlignMol failed" in error_msg:
                        error_category = "alignment_failed"
                        detailed_msg = f"RDKit molecular alignment failed: {error_msg}"
                    elif (
                        "zip() argument" in error_msg
                        and "is longer than argument" in error_msg
                    ):
                        error_category = "index_mismatch_error"
                        detailed_msg = (
                            f"MCS index length mismatch during alignment: {error_msg}"
                        )
                    elif "MCS" in error_msg or "match" in error_msg.lower():
                        error_category = "mcs_calculation_failed"
                        detailed_msg = f"Maximum Common Substructure calculation failed: {error_msg}"
                    elif "embedding" in error_msg.lower():
                        error_category = "embedding_failed"
                        detailed_msg = f"Molecular embedding failed: {error_msg}"
                    else:
                        error_category = "pipeline_general_error"
                        detailed_msg = f"General pipeline error: {error_msg}"

                    self.log.error(
                        f"Pipeline execution failed for {params.target_pdb}: {detailed_msg}"
                    )
                    self.log.error(
                        f"Error type: {error_type}, Category: {error_category}"
                    )

                    # Log additional context for debugging
                    import traceback

                    self.log.error(f"Full traceback: {traceback.format_exc()}")

                    # Record error in tracker if available
                    if error_tracker:
                        error_tracker.record_target_failure(
                            params.target_pdb,
                            detailed_msg,
                            context={
                                "error_type": error_type,
                                "error_category": error_category,
                                "target_pdb": params.target_pdb,
                                "ligand_smiles": (
                                    ligand_smiles[:100] if ligand_smiles else None
                                ),
                            },
                        )

                    raise RuntimeError(
                        f"Pipeline execution failed ({error_category}): {detailed_msg}"
                    )

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
                # Enhanced logging for cases where no RMSD values were calculated
                self.log.error(
                    f"No RMSD values calculated for {params.target_pdb} - indicates pose generation or scoring failure"
                )
                self.log.error(f"Pipeline result type: {type(pipeline_result)}")

                if isinstance(pipeline_result, dict):
                    self.log.error(
                        f"Pipeline result keys: {list(pipeline_result.keys())}"
                    )
                    if "poses" in pipeline_result:
                        poses = pipeline_result["poses"]
                        self.log.error(
                            f"Poses type: {type(poses)}, length: {len(poses) if poses else 0}"
                        )
                        if poses:
                            for k, v in poses.items():
                                self.log.error(
                                    f"  Pose {k}: type={type(v)}, valid={v is not None}"
                                )
                        else:
                            self.log.error(
                                "Poses dictionary is empty - pose generation failed"
                            )
                    else:
                        self.log.error(
                            "No 'poses' key in pipeline result - pipeline structure error"
                        )

                # Record this as an error in tracking
                if error_tracker:
                    error_tracker.record_target_failure(
                        params.target_pdb,
                        "No poses generated or RMSD calculation failed",
                        context={
                            "error_type": "target_processing_failed",
                            "component": "pipeline",
                            "target_pdb": params.target_pdb,
                            "pipeline_result_type": str(type(pipeline_result)),
                            "has_poses_key": (
                                "poses" in pipeline_result
                                if isinstance(pipeline_result, dict)
                                else False
                            ),
                        },
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
                        "allowed_templates_count": (
                            len(params.allowed_pdb_ids)
                            if params.allowed_pdb_ids
                            else None
                        ),
                        "final_memory_gb": final_memory["memory_gb"],
                    },
                )
            else:
                # Record failed target processing
                error_msg = "No poses generated or RMSD calculation failed"
                if self.error_tracker:
                    self.error_tracker.record_target_failure(
                        params.target_pdb, error_msg
                    )

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
                        "allowed_templates_count": (
                            len(params.allowed_pdb_ids)
                            if params.allowed_pdb_ids
                            else None
                        ),
                        "final_memory_gb": final_memory["memory_gb"],
                    },
                )

        except FileNotFoundError as e:
            runtime = time.time() - start_time
            if self.error_tracker:
                self.error_tracker.record_target_failure(
                    params.target_pdb, f"File not found: {str(e)}"
                )
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
                self.error_tracker.record_target_failure(
                    params.target_pdb, f"Invalid input data: {str(e)}"
                )
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

        # After all processing, before return:
        try:
            proc = psutil.Process(os.getpid())
            mem_after = proc.memory_info().rss / 1e9
            self.log.info(
                f"[MEMORY] After pipeline for {params.target_pdb}: {mem_after:.2f} GB"
            )
        except Exception as e:
            self.log.warning(f"Could not log memory after: {e}")
        # Aggressive memory cleanup
        try:
            import gc

            gc.collect()
        except Exception:
            pass

    def _get_protein_file(self, pdb_id: str) -> str:
        """Get protein file path for PDB ID using shared utilities."""
        # Lazy import to avoid slow import at module level
        from templ_pipeline.core.utils import get_protein_file_paths

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
    timeout: int = 300,
    data_dir: Optional[str] = None,
    poses_output_dir: Optional[str] = None,
    shared_cache_file: Optional[str] = None,
    unconstrained: bool = False,
    align_metric: str = "combo",
    enable_optimization: bool = False,
    no_realign: bool = False,
    allowed_pdb_ids: Optional[
        Set[str]
    ] = None,  # NEW: restrict template search to these PDB IDs
) -> Dict:
    """Main entry point for benchmark pipeline execution."""

    # Lazy imports to speed up module loading
    from templ_pipeline.core.pipeline import TEMPLPipeline
    from templ_pipeline.core.utils import (
        calculate_rmsd,
        find_ligand_by_pdb_id,
        find_ligand_file_paths,
        get_protein_file_paths,
        load_sdf_molecules_cached,
    )

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
        unconstrained=unconstrained,
        align_metric=align_metric,
        enable_optimization=enable_optimization,
        no_realign=no_realign,
        allowed_pdb_ids=allowed_pdb_ids,  # NEW: pass allowed templates to pipeline
    )

    # Pipeline execution detailed logging
    logger.info(f"PIPELINE_EXEC: Starting benchmark execution for {target_pdb}:")
    logger.info(f"PIPELINE_EXEC:   Data directory: {data_dir}")
    logger.info(
        f"PIPELINE_EXEC:   Template restrictions: {len(allowed_pdb_ids) if allowed_pdb_ids else 0} allowed"
    )
    logger.info(f"PIPELINE_EXEC:   Excluded PDB IDs: {len(exclude_pdb_ids)}")
    logger.info(f"PIPELINE_EXEC:   Conformers requested: {n_conformers}")
    logger.info(f"PIPELINE_EXEC:   Template KNN: {template_knn}")
    logger.info(f"PIPELINE_EXEC:   Internal workers: {internal_workers}")
    logger.info(f"PIPELINE_EXEC:   Align metric: {align_metric}")

    runner = BenchmarkRunner(
        data_dir, poses_output_dir, shared_cache_file=shared_cache_file
    )

    logger.info(f"PIPELINE_EXEC: BenchmarkRunner initialized, executing pipeline...")
    result = runner.run_single_target(params)

    # Log pipeline execution results
    logger.info(f"PIPELINE_EXEC: Pipeline execution completed for {target_pdb}:")
    if isinstance(result, dict):
        logger.info(f"PIPELINE_EXEC:   Success: {result.get('success', False)}")
        logger.info(f"PIPELINE_EXEC:   Runtime: {result.get('runtime', 0):.2f}s")
        rmsd_values = result.get("rmsd_values", {})
        logger.info(
            f"PIPELINE_EXEC:   RMSD metrics available: {list(rmsd_values.keys())}"
        )

        # Log RMSD values from pipeline execution
        for metric, values in rmsd_values.items():
            rmsd_val = values.get("rmsd") if isinstance(values, dict) else None
            score_val = values.get("score") if isinstance(values, dict) else None
            logger.info(
                f"PIPELINE_EXEC:   {metric}: RMSD={rmsd_val}, Score={score_val}"
            )
    else:
        logger.warning(f"PIPELINE_EXEC:   Unexpected result type: {type(result)}")

    # Ensure result is properly converted to dictionary
    if hasattr(result, "to_dict"):
        final_result = result.to_dict()
        logger.info(f"PIPELINE_EXEC: Result converted to dict via to_dict() method")
        return final_result
    elif isinstance(result, dict):
        logger.info(f"PIPELINE_EXEC: Result already in dict format")
        return result
    else:
        # Fallback for unexpected return types
        logger.error(
            f"PIPELINE_EXEC: Unexpected result type {type(result)}, returning error dict"
        )
        return {
            "success": False,
            "rmsd_values": {},
            "runtime": 0.0,
            "error": f"Unexpected result type: {type(result)}",
        }
