import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
    TimeoutError as FutureTimeoutError,
)
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

# Import hardware detection
try:
    from templ_pipeline.core.hardware import get_suggested_worker_config

    HARDWARE_CONFIG = get_suggested_worker_config()
    DEFAULT_WORKERS = HARDWARE_CONFIG["n_workers"]
except ImportError:
    logging.warning("Hardware detection not available, using conservative default")
    DEFAULT_WORKERS = 8

# Disable RDKit noisy logging
RDLogger.DisableLog("rdApp.*")
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

# Legacy core TEMPL components (for backwards compatibility only)
# TODO: Remove these imports once legacy run_templ_pipeline_single() is deprecated
from templ_pipeline.core.mcs import find_mcs, constrained_embed, safe_name
from templ_pipeline.core.scoring import select_best

# Unified pipeline infrastructure (preferred approach)
from templ_pipeline.benchmark.runner import run_templ_pipeline_for_benchmark
from templ_pipeline.core.pipeline import TEMPLPipeline
import tempfile
import os

try:
    from spyrmsd.molecule import Molecule
    from spyrmsd.rmsd import rmsdwrapper
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "spyrmsd must be installed – please check templ_pipeline/requirements.txt"
    ) from e

# Try to import pebble for proper timeout handling
try:
    import pebble
    from pebble import ProcessPool

    PEBBLE_AVAILABLE = True
except ImportError:
    PEBBLE_AVAILABLE = False
    ProcessPool = None

# Configuration
MOLECULE_TIMEOUT = 180
OUTPUT_DIR = "templ_benchmark_results_polaris"

# Global flag for graceful shutdown
shutdown_requested = False


# Progress Bar Configuration
class ProgressConfig:
    """Configuration for progress bars"""

    @staticmethod
    def get_bar_format():
        # Clean format matching user's request: 53%|████████████████████████████████████████████████████| 409/770 [03:13<02:49, 2.12it/s]
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
    # Import the new benchmark logging system
    from templ_pipeline.core.benchmark_logging import (
        benchmark_logging_context, 
        suppress_worker_logging,
        create_benchmark_logger
    )
    
    # If workspace directory is provided, set up file-only logging
    if workspace_dir:
        # The context manager will be used by the calling function
        # This function now just returns the logger
        return create_benchmark_logger("polaris")
    else:
        # Fallback to original behavior for compatibility
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level))

        # Reduce verbosity for specific modules during benchmark
        mcs_logger = logging.getLogger("templ_pipeline.core.mcs")
        scoring_logger = logging.getLogger("templ_pipeline.core.scoring")

        # Set MCS and scoring to WARNING to reduce verbose output
        mcs_logger.setLevel(logging.WARNING)
        scoring_logger.setLevel(logging.WARNING)

        # Keep benchmark logger at INFO level
        benchmark_logger = logging.getLogger(__name__)
        benchmark_logger.setLevel(getattr(logging, log_level))

        return benchmark_logger


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def load_sdf_molecules(sdf_path: Path) -> List[Chem.Mol]:
    """Load molecules from an SDF file with minimal sanitisation.

    Each molecule gets a unique name `<original>_idx_<n>` to ensure uniqueness.
    """
    molecules: List[Chem.Mol] = []
    if not sdf_path.exists():
        logging.error(f"SDF file not found: {sdf_path}")
        return molecules

    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    for idx, mol in enumerate(suppl):
        if mol is None:
            continue
        original_name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{idx}"
        unique_name = f"{original_name}_idx_{idx:04d}"
        mol.SetProp("_Name", unique_name)
        mol.SetProp("original_name", original_name)
        mol.SetProp("molecule_index", str(idx))
        molecules.append(mol)
    logging.info(f"Loaded {len(molecules)} molecules from {sdf_path.name}")
    return molecules


def get_virus_type(mol: Chem.Mol) -> str:
    """Return virus class for a molecule based on the `Protein_Label` property."""
    if mol.HasProp("Protein_Label"):
        label = mol.GetProp("Protein_Label").lower()
        if "mers" in label:
            return "MERS"
        if "sars" in label:
            return "SARS"
    # Fallback using molecule name
    name = mol.GetProp("_Name").lower() if mol.HasProp("_Name") else ""
    if "mers" in name:
        return "MERS"
    if "sars" in name:
        return "SARS"
    return "UNKNOWN"


def rmsd_raw(a: Chem.Mol, b: Chem.Mol) -> float:
    """Heavy‐atom symmetry‐aware RMSD using sPyRMSD."""
    try:
        return rmsdwrapper(
            Molecule.from_rdkit(a),
            Molecule.from_rdkit(b),
            minimize=False,
            strip=True,
            symmetry=True,
        )[0]
    except AssertionError:
        return float("nan")


# -----------------------------------------------------------------------------
# Template management functions
# -----------------------------------------------------------------------------


def get_training_templates(
    virus_type: str,
    template_source: str,
    train_sars: List[Chem.Mol],
    train_mers: List[Chem.Mol],
    train_sars_aligned: List[Chem.Mol],
) -> Tuple[List[Chem.Mol], Dict[str, int]]:
    """Load appropriate training molecules based on virus type and template source."""
    template_counts = {}

    if virus_type == "SARS":
        # SARS always uses native SARS templates
        templates = train_sars
        template_counts["SARS_native"] = len(templates)
        return templates, template_counts

    elif virus_type == "MERS":
        if template_source == "native":
            # Native MERS templates only
            templates = train_mers
            template_counts["MERS_native"] = len(templates)
            return templates, template_counts
        elif template_source == "cross_aligned":
            # Combined template pool: native MERS + SARS-aligned templates
            combined_templates = train_mers + train_sars_aligned

            # Track counts for reporting
            template_counts["MERS_native"] = len(train_mers)
            template_counts["SARS_aligned"] = len(train_sars_aligned)
            template_counts["total_combined"] = len(combined_templates)

            logging.info(
                f"Combined template pool: {len(train_mers)} MERS + {len(train_sars_aligned)} SARS-aligned = {len(combined_templates)} total"
            )
            return combined_templates, template_counts

    raise ValueError(f"Invalid combination: {virus_type}, {template_source}")


# -----------------------------------------------------------------------------
# Core single‐pose runner
# -----------------------------------------------------------------------------


def run_templ_pipeline_single_unified(
    query_mol: Chem.Mol,
    reference_mol: Chem.Mol,
    n_conformers: int = 200,
    n_workers: int = 1,
    save_poses: bool = False,
    poses_output_dir: Optional[str] = None,
) -> Dict:
    """Run TEMPL pipeline for a single molecule using unified TEMPLPipeline infrastructure.
    
    This function bridges Polaris benchmark with unified architecture by:
    1. Converting molecules to SMILES for TEMPLPipeline
    2. Using TEMPLPipeline.run_full_pipeline() as execution engine
    3. Converting results back to Polaris format
    """
    result = {
        "success": False,
        "rmsd_values": {},
        "n_conformers_generated": 0,
        "template_used": None,
        "error": None,
        "processing_time": 0,
        "timeout": False,
        "molecule_name": safe_name(query_mol, "unknown"),
        "filter_reason": None,
    }
    
    start_time = time.time()
    
    try:
        # Convert query molecule to SMILES for TEMPLPipeline
        query_smiles = Chem.MolToSmiles(query_mol)
        if not query_smiles:
            result["error"] = "Failed to convert query molecule to SMILES"
            return result
        
        # Create temporary directory for pipeline execution
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize TEMPLPipeline with temporary output directory
            pipeline = TEMPLPipeline(output_dir=temp_dir)
            
            # Run the full pipeline using unified infrastructure
            # Note: We pass a dummy protein_pdb_id since Polaris doesn't use protein-specific templates
            pipeline_result = pipeline.run_full_pipeline(
                protein_pdb_id="1iky",  # Dummy PDB ID for Polaris
                ligand_smiles=query_smiles,
                num_conformers=n_conformers,
                n_workers=n_workers,
            )
            
            if pipeline_result.get("success", False) and "poses" in pipeline_result and pipeline_result["poses"]:
                result["success"] = True
                result["n_conformers_generated"] = pipeline_result.get("template_info", {}).get("num_conformers_generated", n_conformers)
                
                # Extract and calculate RMSD values from TEMPLPipeline poses
                if reference_mol is not None and pipeline_result["poses"]:
                    reference_noH = Chem.RemoveHs(Chem.Mol(reference_mol))
                    
                    # Process each pose metric from TEMPLPipeline
                    for metric, pose_data in pipeline_result["poses"].items():
                        if isinstance(pose_data, tuple) and len(pose_data) == 2:
                            pose_mol, scores_dict = pose_data
                            
                            if pose_mol is not None and pose_mol.GetNumConformers() > 0:
                                # Calculate RMSD between pose and reference
                                try:
                                    pose_noH = Chem.RemoveHs(Chem.Mol(pose_mol))
                                    rmsd = rmsd_raw(pose_noH, reference_noH)
                                    
                                    # Extract score (prioritize shape/combo metrics for Polaris compatibility)
                                    score = 0.0
                                    if isinstance(scores_dict, dict):
                                        if metric in scores_dict:
                                            score = float(scores_dict[metric])
                                        elif "score" in scores_dict:
                                            score = float(scores_dict["score"])
                                        elif "similarity_score" in scores_dict:
                                            score = float(scores_dict["similarity_score"])
                                    
                                    # Store results using Polaris format
                                    metric_key = "canimotocombo" if metric == "combo" else metric
                                    result["rmsd_values"][metric_key] = {
                                        "rmsd": float(rmsd),
                                        "score": float(score)
                                    }
                                    
                                except Exception as e:
                                    logging.warning(f"Failed to calculate RMSD for metric {metric}: {e}")
                                    continue
                
                # Extract template information if available
                if "template_info" in pipeline_result:
                    result["template_used"] = pipeline_result["template_info"]
                    
            else:
                result["error"] = pipeline_result.get("error", "TEMPLPipeline execution failed")
                
    except Exception as e:
        result["error"] = f"Unified pipeline execution failed: {str(e)}"
    
    result["processing_time"] = time.time() - start_time
    return result


def run_templ_pipeline_single(
    query_mol: Chem.Mol,
    templates: List[Chem.Mol],
    reference_mol: Chem.Mol,
    exclude_mol: Optional[Chem.Mol] = None,
    n_conformers: int = 200,
    n_workers: int = 1,
    save_poses: bool = False,
    poses_output_dir: Optional[str] = None,
) -> Dict:
    """Run TEMPL pipeline for a single molecule with comprehensive result tracking.
    
    This function maintains backwards compatibility with Polaris-specific interface
    while delegating to unified TEMPLPipeline infrastructure for actual execution.
    
    Legacy Polaris Interface Wrapper:
    - Handles template filtering and exclusion logic
    - Delegates to run_templ_pipeline_single_unified() for execution
    - Maintains original result format for compatibility
    """
    result = {
        "success": False,
        "rmsd_values": {},
        "n_conformers_generated": 0,
        "template_used": None,
        "error": None,
        "processing_time": 0,
        "timeout": False,
        "molecule_name": safe_name(query_mol, "unknown"),
        "filter_reason": None,
    }

    start_time = time.time()

    try:
        # Filter out the exclude molecule if provided (for leave-one-out)
        filtered_templates = []
        for template in templates:
            if exclude_mol is not None:
                # Compare SMILES to avoid issues with object identity
                query_smiles = Chem.MolToSmiles(exclude_mol)
                template_smiles = Chem.MolToSmiles(template)
                if query_smiles == template_smiles:
                    continue
            filtered_templates.append(template)

        if not filtered_templates:
            result["error"] = "No templates available after filtering"
            result["filter_reason"] = "no_templates_after_filtering"
            return result

        # Prepare query molecule
        query_noH = Chem.RemoveHs(Chem.Mol(query_mol))

        # Find MCS against templates (using same algorithm as TEMPLPipeline)
        idx, smarts = find_mcs(query_noH, filtered_templates)
        if idx is None or smarts is None:
            result["error"] = "MCS search failed - no common substructure found"
            result["filter_reason"] = "mcs_failed"
            return result

        template_mol = filtered_templates[idx]
        result["template_used"] = safe_name(template_mol, f"template_{idx}")

        # Generate constrained conformers (using same algorithm as TEMPLPipeline)
        confs = constrained_embed(
            query_noH, template_mol, smarts, n_conformers, n_workers
        )
        result["n_conformers_generated"] = confs.GetNumConformers()

        if confs.GetNumConformers() == 0:
            result["error"] = "No conformers generated"
            result["filter_reason"] = "conformer_generation_failed"
            return result

        # Select best poses (using same algorithm as TEMPLPipeline)
        best_poses = select_best(
            confs, template_mol, no_realign=False, n_workers=n_workers
        )

        # Calculate RMSD to reference
        reference_noH = Chem.RemoveHs(Chem.Mol(reference_mol))

        for metric, (pose, scores) in best_poses.items():
            if pose is not None:
                try:
                    rmsd = rmsd_raw(pose, reference_noH)
                    result["rmsd_values"][metric] = {
                        "rmsd": rmsd,
                        "score": scores[metric],
                    }
                    
                    # Save pose if requested
                    if save_poses and poses_output_dir:
                        try:
                            pose_path = _save_pose(
                                pose, 
                                result["molecule_name"], 
                                metric, 
                                poses_output_dir,
                                {
                                    "rmsd": rmsd,
                                    "score": scores[metric],
                                    "template_used": result["template_used"],
                                    "n_conformers_generated": result["n_conformers_generated"],
                                }
                            )
                            if "pose_files" not in result:
                                result["pose_files"] = {}
                            result["pose_files"][metric] = pose_path
                        except Exception as e:
                            logging.warning(f"Failed to save pose for {metric}: {str(e)}")
                            
                except Exception as e:
                    logging.warning(
                        f"RMSD calculation failed for {metric} on {result['molecule_name']}: {str(e)}"
                    )

        if result["rmsd_values"]:
            result["success"] = True
        else:
            result["error"] = "No valid poses generated"
            result["filter_reason"] = "no_valid_poses"

    except Exception as e:
        result["error"] = f"Pipeline error: {str(e)}"
        result["filter_reason"] = f"exception: {str(e)}"
        logging.error(
            f"Error in TEMPL pipeline for {result['molecule_name']}: {str(e)}"
        )
    finally:
        result["processing_time"] = time.time() - start_time

    return result


# -----------------------------------------------------------------------------
# Evaluation strategies
# -----------------------------------------------------------------------------


def evaluate_with_leave_one_out(
    query_mols: List[Chem.Mol],
    template_pool: List[Chem.Mol],
    virus_type: str,
    template_source: str,
    template_counts: Dict[str, int],
    n_workers: int = 1,
    n_conformers: int = 200,
    save_poses: bool = False,
    poses_output_dir: Optional[str] = None,
) -> Dict:
    """Enhanced leave-one-out evaluation."""

    logging.info(
        f"Starting {virus_type} {template_source} leave-one-out evaluation with {len(query_mols)} molecules"
    )

    # Create experiment-specific output directory only if needed
    experiment_name = f"{virus_type}_train_{template_source}"
    output_subdir = Path(OUTPUT_DIR) / experiment_name

    # Prepare results structure
    results = {
        "benchmark_info": {
            "name": "templ_polaris_benchmark",
            "split": f"{virus_type.lower()}_{template_source}_train_loo",
            "virus_type": virus_type,
            "template_source": template_source,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_molecules": len(query_mols),
            "n_conformers": n_conformers,
            "n_workers": n_workers,
        },
        "results": {},
        "query_count": len(query_mols),
        "template_counts": template_counts,
        "evaluation_type": "leave_one_out",
        "summary": {},
        "errors": [],
    }

    # Process molecules with timeout handling and progress bar
    if PEBBLE_AVAILABLE:
        with ProcessPool(max_workers=n_workers) as pool:
            futures = []
            for i, query_mol in enumerate(query_mols):
                mol_name = safe_name(query_mol, f"mol_{i}")

                future = pool.schedule(
                    run_templ_pipeline_single,
                    args=[
                        query_mol,
                        template_pool,
                        query_mol,
                        query_mol,
                        n_conformers,
                        1,
                        save_poses,
                        poses_output_dir,
                    ],
                    timeout=MOLECULE_TIMEOUT,
                )
                futures.append((future, mol_name, query_mol))

            # Collect results with progress bar
            desc = f"{virus_type} Train ({template_source})"
            successes = 0
            errors = 0

            tqdm_config = ProgressConfig.get_tqdm_config(desc)
            tqdm_config['total'] = len(futures)
            with tqdm(**tqdm_config) as pbar:

                for future, mol_name, query_mol in futures:
                    try:
                        result = future.result()
                        results["results"][mol_name] = result

                        if result["success"]:
                            successes += 1
                        else:
                            errors += 1

                    except pebble.ProcessExpired:
                        error_msg = f"Process timeout after {MOLECULE_TIMEOUT}s"
                        results["results"][mol_name] = {
                            "success": False,
                            "error": error_msg,
                            "molecule_name": mol_name,
                            "rmsd_values": {},
                            "processing_time": MOLECULE_TIMEOUT,
                            "timeout": True,
                            "filter_reason": "timeout",
                        }
                        results["errors"].append(f"{mol_name}: {error_msg}")
                        errors += 1
                    except Exception as e:
                        error_msg = f"Pipeline error: {str(e)}"
                        results["results"][mol_name] = {
                            "success": False,
                            "error": error_msg,
                            "molecule_name": mol_name,
                            "rmsd_values": {},
                            "processing_time": 0,
                            "timeout": False,
                            "filter_reason": "pipeline_error",
                        }
                        results["errors"].append(f"{mol_name}: {error_msg}")
                        errors += 1

                    # Update progress bar
                    success_rate = (
                        (successes / (successes + errors) * 100)
                        if (successes + errors) > 0
                        else 0
                    )
                    pbar.set_postfix(
                        ProgressConfig.get_postfix_format(success_rate, errors)
                    )
                    pbar.update(1)
    else:
        # Fallback to ProcessPoolExecutor
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp_context
        ) as executor:
            future_to_mol = {}
            for i, query_mol in enumerate(query_mols):
                mol_name = safe_name(query_mol, f"mol_{i}")

                future = executor.submit(
                    run_templ_pipeline_single,
                    query_mol,
                    template_pool,
                    query_mol,
                    query_mol,
                    n_conformers,
                    1,
                    save_poses,
                    poses_output_dir,
                )
                future_to_mol[future] = (mol_name, query_mol)

            # Collect results with progress bar
            desc = f"{virus_type} Train ({template_source})"
            successes = 0
            errors = 0

            tqdm_config = ProgressConfig.get_tqdm_config(desc)
            tqdm_config['total'] = len(future_to_mol)
            with tqdm(**tqdm_config) as pbar:

                for future in as_completed(future_to_mol):
                    mol_name, query_mol = future_to_mol[future]

                    try:
                        result = future.result(timeout=MOLECULE_TIMEOUT)
                        results["results"][mol_name] = result

                        if result["success"]:
                            successes += 1
                        else:
                            errors += 1

                    except FutureTimeoutError:
                        error_msg = f"Molecule timeout after {MOLECULE_TIMEOUT}s"
                        results["results"][mol_name] = {
                            "success": False,
                            "error": error_msg,
                            "molecule_name": mol_name,
                            "rmsd_values": {},
                            "processing_time": MOLECULE_TIMEOUT,
                            "timeout": True,
                            "filter_reason": "timeout",
                        }
                        results["errors"].append(f"{mol_name}: {error_msg}")
                        errors += 1
                    except Exception as e:
                        error_msg = f"Pipeline error: {str(e)}"
                        results["results"][mol_name] = {
                            "success": False,
                            "error": error_msg,
                            "molecule_name": mol_name,
                            "rmsd_values": {},
                            "processing_time": 0,
                            "timeout": False,
                            "filter_reason": "pipeline_error",
                        }
                        results["errors"].append(f"{mol_name}: {error_msg}")
                        errors += 1

                    # Update progress bar
                    success_rate = (
                        (successes / (successes + errors) * 100)
                        if (successes + errors) > 0
                        else 0
                    )
                    pbar.set_postfix(
                        ProgressConfig.get_postfix_format(success_rate, errors)
                    )
                    pbar.update(1)

    # Calculate summary metrics
    results["summary"] = calculate_success_rates(results)

    return results


def evaluate_with_templates(
    query_mols: List[Chem.Mol],
    template_mols: List[Chem.Mol],
    virus_type: str,
    template_source: str,
    template_counts: Dict[str, int],
    n_workers: int = 1,
    n_conformers: int = 200,
    save_poses: bool = False,
    poses_output_dir: Optional[str] = None,
) -> Dict:
    """Enhanced evaluation with templates."""

    logging.info(
        f"Starting {virus_type} {template_source} evaluation with {len(query_mols)} queries and {len(template_mols)} templates"
    )

    # Create experiment-specific output directory only if needed
    experiment_name = f"{virus_type}_test_{template_source}"
    output_subdir = Path(OUTPUT_DIR) / experiment_name

    # Prepare results structure
    results = {
        "benchmark_info": {
            "name": "templ_polaris_benchmark",
            "split": f"{virus_type.lower()}_{template_source}_test",
            "virus_type": virus_type,
            "template_source": template_source,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_molecules": len(query_mols),
            "n_conformers": n_conformers,
            "n_workers": n_workers,
        },
        "results": {},
        "query_count": len(query_mols),
        "template_counts": template_counts,
        "evaluation_type": "template_based",
        "summary": {},
        "errors": [],
    }

    # Process molecules with timeout handling and progress bar
    if PEBBLE_AVAILABLE:
        with ProcessPool(max_workers=n_workers) as pool:
            futures = []
            for i, query_mol in enumerate(query_mols):
                mol_name = safe_name(query_mol, f"mol_{i}")

                future = pool.schedule(
                    run_templ_pipeline_single,
                    args=[query_mol, template_mols, query_mol, None, n_conformers, 1, save_poses, poses_output_dir],
                    timeout=MOLECULE_TIMEOUT,
                )
                futures.append((future, mol_name, query_mol))

            # Collect results with progress bar
            desc = f"{virus_type} Test ({template_source})"
            successes = 0
            errors = 0

            tqdm_config = ProgressConfig.get_tqdm_config(desc)
            tqdm_config['total'] = len(futures)
            with tqdm(**tqdm_config) as pbar:

                for future, mol_name, query_mol in futures:
                    try:
                        result = future.result()
                        results["results"][mol_name] = result

                        if result["success"]:
                            successes += 1
                        else:
                            errors += 1

                    except pebble.ProcessExpired:
                        error_msg = f"Process timeout after {MOLECULE_TIMEOUT}s"
                        results["results"][mol_name] = {
                            "success": False,
                            "error": error_msg,
                            "molecule_name": mol_name,
                            "rmsd_values": {},
                            "processing_time": MOLECULE_TIMEOUT,
                            "timeout": True,
                            "filter_reason": "timeout",
                        }
                        results["errors"].append(f"{mol_name}: {error_msg}")
                        errors += 1
                    except Exception as e:
                        error_msg = f"Pipeline error: {str(e)}"
                        results["results"][mol_name] = {
                            "success": False,
                            "error": error_msg,
                            "molecule_name": mol_name,
                            "rmsd_values": {},
                            "processing_time": 0,
                            "timeout": False,
                            "filter_reason": "pipeline_error",
                        }
                        results["errors"].append(f"{mol_name}: {error_msg}")
                        errors += 1

                    # Update progress bar
                    success_rate = (
                        (successes / (successes + errors) * 100)
                        if (successes + errors) > 0
                        else 0
                    )
                    pbar.set_postfix(
                        ProgressConfig.get_postfix_format(success_rate, errors)
                    )
                    pbar.update(1)
    else:
        # Fallback to ProcessPoolExecutor
        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp_context
        ) as executor:
            future_to_mol = {}
            for i, query_mol in enumerate(query_mols):
                mol_name = safe_name(query_mol, f"mol_{i}")

                future = executor.submit(
                    run_templ_pipeline_single,
                    query_mol,
                    template_mols,
                    query_mol,
                    None,
                    n_conformers,
                    1,
                    save_poses,
                    poses_output_dir,
                )
                future_to_mol[future] = (mol_name, query_mol)

            # Collect results with progress bar
            desc = f"{virus_type} Test ({template_source})"
            successes = 0
            errors = 0

            tqdm_config = ProgressConfig.get_tqdm_config(desc)
            tqdm_config['total'] = len(future_to_mol)
            with tqdm(**tqdm_config) as pbar:

                for future in as_completed(future_to_mol):
                    mol_name, query_mol = future_to_mol[future]

                    try:
                        result = future.result(timeout=MOLECULE_TIMEOUT)
                        results["results"][mol_name] = result

                        if result["success"]:
                            successes += 1
                        else:
                            errors += 1

                    except FutureTimeoutError:
                        error_msg = f"Molecule timeout after {MOLECULE_TIMEOUT}s"
                        results["results"][mol_name] = {
                            "success": False,
                            "error": error_msg,
                            "molecule_name": mol_name,
                            "rmsd_values": {},
                            "processing_time": MOLECULE_TIMEOUT,
                            "timeout": True,
                            "filter_reason": "timeout",
                        }
                        results["errors"].append(f"{mol_name}: {error_msg}")
                        errors += 1
                    except Exception as e:
                        error_msg = f"Pipeline error: {str(e)}"
                        results["results"][mol_name] = {
                            "success": False,
                            "error": error_msg,
                            "molecule_name": mol_name,
                            "rmsd_values": {},
                            "processing_time": 0,
                            "timeout": False,
                            "filter_reason": "pipeline_error",
                        }
                        results["errors"].append(f"{mol_name}: {error_msg}")
                        errors += 1

                    # Update progress bar
                    success_rate = (
                        (successes / (successes + errors) * 100)
                        if (successes + errors) > 0
                        else 0
                    )
                    pbar.set_postfix(
                        ProgressConfig.get_postfix_format(success_rate, errors)
                    )
                    pbar.update(1)

    # Calculate summary metrics
    results["summary"] = calculate_success_rates(results)

    return results


# -----------------------------------------------------------------------------
# Results analysis
# -----------------------------------------------------------------------------


def calculate_success_rates(results_data: Dict) -> Dict:
    """Calculate success rates at 2Å and 5Å RMSD thresholds."""
    individual_results = results_data.get("results", {})
    if not individual_results:
        return {
            "total": 0,
            "successful": 0,
            "success_rate_2A": 0.0,
            "success_rate_5A": 0.0,
        }

    # Count different types of results
    total_molecules = len(individual_results)
    successful_molecules = sum(
        1 for r in individual_results.values() if r.get("success")
    )
    timeout_molecules = sum(1 for r in individual_results.values() if r.get("timeout"))
    failed_molecules = total_molecules - successful_molecules - timeout_molecules

    # Track filtering reasons
    filter_reasons = defaultdict(int)

    for result in individual_results.values():
        if not result.get("success") and result.get("filter_reason"):
            reason = result["filter_reason"]
            filter_reasons[reason] += 1

    metrics = {
        "total": total_molecules,
        "successful": successful_molecules,
        "timeout": timeout_molecules,
        "failed": failed_molecules,
        "processable": total_molecules,
        "filter_reasons": dict(filter_reasons),
        "rmsd_counts_2A": defaultdict(int),
        "rmsd_counts_5A": defaultdict(int),
        "all_rmsds": defaultdict(list),
        "all_scores": defaultdict(list),
    }

    for result in individual_results.values():
        if result.get("success") and result.get("rmsd_values"):
            for metric_key, values_dict in result["rmsd_values"].items():
                rmsd = values_dict.get("rmsd")
                score = values_dict.get("score")
                if rmsd is not None and not np.isnan(rmsd):
                    metrics["all_rmsds"][metric_key].append(rmsd)
                    if rmsd <= 2.0:
                        metrics["rmsd_counts_2A"][metric_key] += 1
                    if rmsd <= 5.0:
                        metrics["rmsd_counts_5A"][metric_key] += 1
                if score is not None:
                    metrics["all_scores"][metric_key].append(score)

    # Calculate success rates for each metric
    metrics["success_rates"] = {}
    processable_count = metrics["processable"]

    for metric_key in metrics["all_rmsds"]:
        n_results = len(metrics["all_rmsds"][metric_key])
        if n_results > 0 and processable_count > 0:
            metrics["success_rates"][metric_key] = {
                "count": n_results,
                "rate_2A": metrics["rmsd_counts_2A"][metric_key]
                / processable_count
                * 100,
                "rate_5A": metrics["rmsd_counts_5A"][metric_key]
                / processable_count
                * 100,
                "rate_2A_of_successful": metrics["rmsd_counts_2A"][metric_key]
                / n_results
                * 100,
                "rate_5A_of_successful": metrics["rmsd_counts_5A"][metric_key]
                / n_results
                * 100,
                "mean_rmsd": np.mean(metrics["all_rmsds"][metric_key]),
                "median_rmsd": np.median(metrics["all_rmsds"][metric_key]),
            }

    return metrics


def generate_comprehensive_summary_table(all_results: Dict) -> pd.DataFrame:
    """Generate the enhanced summary table with cross-virus evaluation and template tracking."""
    table_data = []

    # Define the expected result keys and their display information
    result_configs = [
        ("SARS_train_native", "SARS", "Train", "SARS"),
        ("SARS_test_native", "SARS", "Test", "SARS"),
        ("MERS_train_native", "MERS", "Train", "MERS"),
        ("MERS_test_native", "MERS", "Test", "MERS"),
        ("MERS_train_cross", "MERS", "Train", "MERS+SARS-aligned^T"),
        ("MERS_test_cross", "MERS", "Test", "MERS+SARS-aligned^T"),
    ]

    for result_key, virus_type, dataset, template_source in result_configs:
        if result_key in all_results:
            result_data = all_results[result_key]
            metrics = calculate_success_rates(result_data)

            # Get query count and template information
            query_count = result_data.get(
                "query_count", len(result_data.get("results", {}))
            )
            template_counts = result_data.get("template_counts", {})

            # Format template count description
            if "total_combined" in template_counts:
                # Combined template pool (MERS + SARS-aligned)
                mers_count = template_counts.get("MERS_native", 0)
                sars_count = template_counts.get("SARS_aligned", 0)
                total_count = template_counts.get(
                    "total_combined", mers_count + sars_count
                )
                template_desc = f"{total_count} ({mers_count}+{sars_count})"
            elif dataset == "Train" and "leave_one_out" in result_data.get(
                "evaluation_type", ""
            ):
                # Leave-one-out: template count is query_count - 1
                template_desc = f"{query_count-1} (LOO)"
            else:
                # Single template pool
                total_templates = (
                    sum(template_counts.values()) if template_counts else 0
                )
                template_desc = str(total_templates)

            # Use 'combo' metric as the primary metric for the table
            if "combo" in metrics.get("success_rates", {}):
                combo_stats = metrics["success_rates"]["combo"]
                table_data.append(
                    [
                        virus_type,
                        dataset,
                        template_source,
                        query_count,
                        template_desc,
                        f"{combo_stats['rate_2A']:.1f}%",
                        f"{combo_stats['rate_5A']:.1f}%",
                    ]
                )
            else:
                # Fallback if no combo results
                table_data.append(
                    [
                        virus_type,
                        dataset,
                        template_source,
                        query_count,
                        template_desc,
                        "0.0%",
                        "0.0%",
                    ]
                )
        else:
            # Missing result - determine expected query count from config
            if result_key == "SARS_train_native":
                expected_queries = "770"
                expected_templates = "769 (LOO)"
            elif result_key == "SARS_test_native":
                expected_queries = "119"
                expected_templates = "770"
            elif result_key == "MERS_train_native":
                expected_queries = "17"
                expected_templates = "16 (LOO)"
            elif result_key == "MERS_test_native":
                expected_queries = "76"
                expected_templates = "17"
            elif result_key in ["MERS_train_cross", "MERS_test_cross"]:
                expected_queries = "17" if "train" in result_key else "76"
                expected_templates = "~787 (17+770)"
            else:
                expected_queries = "0"
                expected_templates = "0"

            table_data.append(
                [
                    virus_type,
                    dataset,
                    template_source,
                    expected_queries,
                    expected_templates,
                    "N/A",
                    "N/A",
                ]
            )

    return pd.DataFrame(
        table_data,
        columns=[
            "Virus Type",
            "Dataset",
            "Template Source",
            "Queries",
            "Templates",
            "Success Rate (<2Å)",
            "Success Rate (<5Å)",
        ],
    )


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Comprehensive Polaris benchmark using the TEMPL pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Directory containing Polaris SDF files",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of parallel workers (auto-detected: {DEFAULT_WORKERS})",
    )
    p.add_argument(
        "--n-conformers", type=int, default=200, help="Conformers per query molecule"
    )
    p.add_argument(
        "--train-only", action="store_true", help="Evaluate only training sets"
    )
    p.add_argument("--test-only", action="store_true", help="Evaluate only test sets")
    p.add_argument(
        "--quick",
        action="store_true",
        help="Quick debug mode: 20 conformers, first 50 molecules, 4 workers",
    )
    p.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )
    p.add_argument(
        "--save-poses",
        action="store_true",
        help="Save predicted poses as SDF files for backtesting analysis",
    )
    p.add_argument(
        "--poses-dir",
        type=str,
        default=None,
        help="Directory to save predicted poses (default: benchmark_poses_<timestamp>)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to save benchmark results (default: %(default)s)",
    )
    p.add_argument(
        "--workspace-dir",
        type=str,
        default=None,
        help="Workspace directory for organized logging and file management",
    )
    return p


def resolve_dataset_paths(dataset_dir: Path) -> Dict[str, Path]:
    paths = {
        "train_sars": dataset_dir / "train_sarsmols.sdf",
        "train_mers": dataset_dir / "train_mersmols.sdf",
        "train_sars_aligned": dataset_dir / "train_sarsmols_aligned_to_mers.sdf",
        "test": dataset_dir / "test_poses_with_properties.sdf",
    }
    return paths


def load_datasets(
    dataset_dir: Path,
) -> Tuple[List[Chem.Mol], List[Chem.Mol], List[Chem.Mol], List[Chem.Mol]]:
    paths = resolve_dataset_paths(dataset_dir)
    train_sars = load_sdf_molecules(paths["train_sars"])
    train_mers = load_sdf_molecules(paths["train_mers"])
    train_sars_aligned = load_sdf_molecules(paths["train_sars_aligned"])
    test_set = load_sdf_molecules(paths["test"])
    return train_sars, train_mers, train_sars_aligned, test_set


def validate_data_files(dataset_dir: Path):
    """Validate that all required data files exist."""
    paths = resolve_dataset_paths(dataset_dir)

    for name, file_path in paths.items():
        if not file_path.exists():
            logging.error(f"Required data file not found: {file_path}")
            return False
        else:
            logging.info(f"Found data file: {file_path}")

    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    logging.info(f"Created output directory: {OUTPUT_DIR}")

    return True


def main(argv: List[str] | None = None):
    global OUTPUT_DIR
    argv = argv or sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    # Override OUTPUT_DIR if specified
    if args.output_dir != OUTPUT_DIR:
        OUTPUT_DIR = args.output_dir
        
    # Set up benchmark logging with reduced verbosity
    # Check if workspace directory is provided (when called from CLI)
    workspace_dir = getattr(args, 'workspace_dir', None)
    
    if workspace_dir:
        # Use new benchmark logging system with file-only output
        from templ_pipeline.core.benchmark_logging import benchmark_logging_context
        
        # The context will be managed in the CLI layer
        logger = setup_benchmark_logging(args.log_level, workspace_dir)
    else:
        # Fallback to original logging for direct script execution
        logger = setup_benchmark_logging(args.log_level)

    # Log hardware detection results
    try:
        from templ_pipeline.core.hardware import get_hardware_info

        hardware_info = get_hardware_info()
        logger.info(
            f"Hardware auto-detection: {DEFAULT_WORKERS} workers suggested for {hardware_info['cpu_count']} CPU cores"
        )
    except ImportError:
        logger.info(f"Using fallback default: {DEFAULT_WORKERS} workers")

    # Apply quick-mode overrides
    if args.quick:
        args.n_conformers = 10  # Minimal conformers for speed
        args.n_workers = min(2, os.cpu_count() or 1)  # Minimal workers
        args.train_only = False  # Skip training (LOO is too slow)
        args.test_only = True  # Only run test evaluation
        logger.info(
            f"Quick mode: SARS test only, 10 molecules, 10 templates, 10 conformers, {args.n_workers} workers"
        )
    else:
        logger.info(
            f"Full benchmark: {args.n_conformers} conformers, {args.n_workers} workers"
        )

    # Dataset directory resolution with ZENODO-compatible centralized path management
    if args.dataset_dir:
        data_dir = Path(args.dataset_dir)
    else:
        # First try new centralized path management for ZENODO-compatible paths
        data_dir = None
        
        # Polaris data stays in GitHub repository (not ZENODO)
        logger.debug("Using direct path resolution for Polaris data")
        
        # Fallback to legacy path resolution if centralized management didn't work
        if data_dir is None:
            potential_paths = [
                # ZENODO format first
                Path(__file__).resolve().parent.parent.parent / "zenodo" / "data" / "polaris",
                # Legacy paths
                Path(__file__).resolve().parent.parent.parent / "data" / "polaris",
                Path.cwd() / "data" / "polaris", 
                Path.cwd() / "templ_pipeline" / "data" / "polaris",
                Path("data") / "polaris",
                Path("..") / "data" / "polaris",
            ]
            
            for path in potential_paths:
                if path.exists() and (path / "train_sarsmols.sdf").exists():
                    data_dir = path
                    logger.info(f"Found polaris data at: {data_dir}")
                    break
        
        if data_dir is None:
            raise FileNotFoundError(
                f"Polaris dataset directory not found. Tried locations:\n" + 
                "\n".join(f"  - {p}" for p in potential_paths) +
                "\n\nPolaris data is stored in the GitHub repository, not ZENODO. "
                "Please ensure polaris data files are in one of these locations or use --dataset-dir"
            )

    # Validate data files and load datasets with progress
    if not validate_data_files(data_dir):
        logger.error("Data validation failed. Exiting.")
        return 1

    logger.info("Loading datasets...")
    tqdm_config = ProgressConfig.get_tqdm_config("Dataset Loading")
    tqdm_config['total'] = 4
    with tqdm(**tqdm_config) as pbar:
        train_sars = load_sdf_molecules(data_dir / "train_sarsmols.sdf")
        pbar.update(1)
        train_mers = load_sdf_molecules(data_dir / "train_mersmols.sdf")
        pbar.update(1)
        train_sars_aligned = load_sdf_molecules(
            data_dir / "train_sarsmols_aligned_to_mers.sdf"
        )
        pbar.update(1)
        test_set = load_sdf_molecules(data_dir / "test_poses_with_properties.sdf")
        pbar.update(1)

    logger.info(
        f"✓ Loaded datasets: {len(train_sars)} SARS train, {len(train_mers)} MERS train, {len(test_set)} test molecules"
    )

    all_results = {}

    # Determine what to run
    run_train = not args.test_only
    run_test = not args.train_only

    try:
        # Run training set evaluations
        if run_train:
            print("\n=== TRAINING SET EVALUATION ===")

            # SARS training evaluation (leave-one-out)
            sars_train_mols = train_sars[:50] if args.quick else train_sars
            if sars_train_mols:
                sars_template_counts = {"SARS_native": len(sars_train_mols)}
                all_results["SARS_train_native"] = evaluate_with_leave_one_out(
                    sars_train_mols,
                    sars_train_mols,
                    "SARS",
                    "native",
                    sars_template_counts,
                    args.n_workers,
                    args.n_conformers,
                    save_poses=getattr(args, 'save_poses', False),
                    poses_output_dir=getattr(args, 'poses_dir', None),
                )

            # MERS training evaluation with native templates (leave-one-out)
            mers_train_mols = train_mers[:50] if args.quick else train_mers
            if mers_train_mols:
                mers_template_counts = {"MERS_native": len(mers_train_mols)}
                all_results["MERS_train_native"] = evaluate_with_leave_one_out(
                    mers_train_mols,
                    mers_train_mols,
                    "MERS",
                    "native",
                    mers_template_counts,
                    args.n_workers,
                    args.n_conformers,
                    save_poses=getattr(args, 'save_poses', False),
                    poses_output_dir=getattr(args, 'poses_dir', None),
                )

            # MERS training evaluation with combined SARS-aligned + MERS templates
            if mers_train_mols:
                # Get combined template pool
                combined_templates, cross_template_counts = get_training_templates(
                    "MERS", "cross_aligned", train_sars, train_mers, train_sars_aligned
                )
                all_results["MERS_train_cross"] = evaluate_with_leave_one_out(
                    mers_train_mols,
                    combined_templates,
                    "MERS",
                    "cross_aligned",
                    cross_template_counts,
                    args.n_workers,
                    args.n_conformers,
                    save_poses=getattr(args, 'save_poses', False),
                    poses_output_dir=getattr(args, 'poses_dir', None),
                )

        # Run test set evaluations
        if run_test:
            print("\n=== TEST SET EVALUATION ===")

            if test_set:
                # Separate test molecules by virus type
                sars_test_mols = [m for m in test_set if get_virus_type(m) == "SARS"]
                mers_test_mols = [m for m in test_set if get_virus_type(m) == "MERS"]

                # 1. SARS test evaluation
                if sars_test_mols:
                    sars_templates, sars_template_counts = get_training_templates(
                        "SARS", "native", train_sars, train_mers, train_sars_aligned
                    )
                    if sars_templates:
                        # Apply quick mode limits
                        if args.quick:
                            sars_test_mols = sars_test_mols[
                                :10
                            ]  # Limit to 10 test molecules
                            sars_templates = sars_templates[
                                :10
                            ]  # Limit to 10 templates
                            sars_template_counts = {"SARS_native": len(sars_templates)}

                        all_results["SARS_test_native"] = evaluate_with_templates(
                            sars_test_mols,
                            sars_templates,
                            "SARS",
                            "native",
                            sars_template_counts,
                            args.n_workers,
                            args.n_conformers,
                            save_poses=args.save_poses,
                            poses_output_dir=args.poses_dir,
                        )

                # 2. MERS test evaluation with native templates
                if mers_test_mols and not args.quick:
                    mers_templates, mers_template_counts = get_training_templates(
                        "MERS", "native", train_sars, train_mers, train_sars_aligned
                    )
                    if mers_templates:
                        all_results["MERS_test_native"] = evaluate_with_templates(
                            mers_test_mols,
                            mers_templates,
                            "MERS",
                            "native",
                            mers_template_counts,
                            args.n_workers,
                            args.n_conformers,
                            save_poses=args.save_poses,
                            poses_output_dir=args.poses_dir,
                        )

                # 3. MERS test evaluation with combined MERS + SARS-aligned templates
                if mers_test_mols and not args.quick:
                    combined_templates, mers_cross_template_counts = (
                        get_training_templates(
                            "MERS",
                            "cross_aligned",
                            train_sars,
                            train_mers,
                            train_sars_aligned,
                        )
                    )
                    if combined_templates:
                        all_results["MERS_test_cross"] = evaluate_with_templates(
                            mers_test_mols,
                            combined_templates,
                            "MERS",
                            "cross_aligned",
                            mers_cross_template_counts,
                            args.n_workers,
                            args.n_conformers,
                            save_poses=args.save_poses,
                            poses_output_dir=args.poses_dir,
                        )

    except KeyboardInterrupt:
        print("\nWARNING: Benchmark interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

    # Save final results
    if all_results:
        print("\n=== SAVING RESULTS ===")

        # Generate summary table
        summary_table = generate_comprehensive_summary_table(all_results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(OUTPUT_DIR)
        results_dir.mkdir(exist_ok=True)

        # Save detailed results as JSON
        results_file = results_dir / f"templ_polaris_benchmark_results_{timestamp}.json"
        with open(results_file, "w") as f:
            import json

            json.dump(
                {
                    "timestamp": timestamp,
                    "parameters": vars(args),
                    "results": all_results,
                },
                f,
                indent=2,
                default=str,
            )

        # Save summary table as CSV
        csv_file = results_dir / f"templ_polaris_summary_table_{timestamp}.csv"
        summary_table.to_csv(csv_file, index=False)

        # Save summary table as Markdown
        md_file = results_dir / f"templ_polaris_summary_table_{timestamp}.md"
        with open(md_file, "w") as f:
            f.write(f"# TEMPL Polaris Benchmark Summary ({timestamp})\n\n")
            f.write("## Parameters\n")
            f.write(f"- Workers: {args.n_workers}\n")
            f.write(f"- Conformers: {args.n_conformers}\n")
            f.write(f"- Molecule timeout: {MOLECULE_TIMEOUT}s\n")
            f.write(f"- Log level: {args.log_level}\n")
            f.write("\n## Results\n\n")
            f.write(summary_table.to_markdown(index=False))
            f.write("\n\n## Notes\n")
            f.write(
                "- ^T indicates cross-virus templates (SARS molecules aligned to MERS binding site)\n"
            )
            f.write(
                "- Success rates are based on the 'combo' metric (combination of shape and color scores)\n"
            )
            f.write("- Training sets use leave-one-out evaluation\n")
            f.write(
                "- Test sets use training templates of the specified type (no test-set leakage)\n"
            )

        # Print summary table
        print("\n" + "=" * 80)
        print("TEMPL POLARIS BENCHMARK SUMMARY")
        print("=" * 80)
        print(summary_table.to_string(index=False))
        print("=" * 80)
        print(f"\n✓ Results saved to {results_file.name}")
        print(f"✓ Summary saved to {csv_file.name}")
        print("✓ Benchmark completed successfully!")
    else:
        logging.warning("No results to save.")

    return 0


def _save_pose(
    pose_mol: Chem.Mol,
    molecule_name: str,
    metric: str,
    poses_output_dir: str,
    metadata: Dict,
) -> str:
    """Save a pose molecule to SDF file with metadata.
    
    Args:
        pose_mol: RDKit molecule object with 3D coordinates
        molecule_name: Name of the query molecule  
        metric: Scoring metric used (shape, color, combo)
        poses_output_dir: Base directory for pose output
        metadata: Dictionary containing RMSD, score, template info
        
    Returns:
        Path to saved SDF file
    """
    # Create organized directory structure
    mol_dir = Path(poses_output_dir) / "poses" / molecule_name
    mol_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename: {molecule}_{metric}_{timestamp}.sdf
    filename = f"{molecule_name}_{metric}_{timestamp}.sdf"
    pose_path = mol_dir / filename
    
    # Add metadata as properties to molecule
    pose_mol.SetProp("_Name", molecule_name)
    pose_mol.SetProp("metric", metric)
    pose_mol.SetProp("rmsd", str(metadata.get("rmsd", "N/A")))
    pose_mol.SetProp("score", str(metadata.get("score", "N/A")))
    pose_mol.SetProp("template_used", str(metadata.get("template_used", "N/A")))
    pose_mol.SetProp("n_conformers_generated", str(metadata.get("n_conformers_generated", "N/A")))
    pose_mol.SetProp("timestamp", timestamp)
    
    # Write to SDF file
    writer = Chem.SDWriter(str(pose_path))
    writer.write(pose_mol)
    writer.close()
    
    return str(pose_path)


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main(_sys.argv[1:]))
