#!/usr/bin/env python3
# run_custom_split_benchmark.py - Run benchmarking using custom, pre-defined splits

import argparse
import json
import logging
import multiprocessing as mp

# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
import os
import subprocess

# Add the parent directory to the path to import from mcs_bench
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rdkit import Chem, RDLogger
from rdkit.Chem import rdRascalMCES

# Import the hardware utility
from templ_pipeline.core.hardware import get_suggested_worker_config

RDLogger.DisableLog("rdApp.*")

# ─── collision detection function ───────────────────────────────────────


def get_unique_filename(base_dir: str, base_name: str, extension: str) -> str:
    """Generate unique filename with collision detection.

    Args:
        base_dir: Directory where file will be saved
        base_name: Base filename without extension
        extension: File extension (e.g., '.sdf')

    Returns:
        Unique filename (with path) that doesn't exist
    """
    os.makedirs(base_dir, exist_ok=True)

    # Try base name first
    filename = f"{base_name}{extension}"
    full_path = os.path.join(base_dir, filename)

    if not os.path.exists(full_path):
        return full_path

    # If collision detected, add incremental suffix
    counter = 2
    while True:
        filename = f"{base_name}_v{counter}{extension}"
        full_path = os.path.join(base_dir, filename)
        if not os.path.exists(full_path):
            return full_path
        counter += 1


def create_experiment_directory(split_name: str) -> str:
    """Create simple experiment directory structure

    Args:
        split_name: Name of the split (val, test)

    Returns:
        Path to the experiment output directory
    """
    experiment_name = f"custom_{split_name}_time_split"
    experiment_output_dir = os.path.join(OUTPUT_DIR_CUSTOM, experiment_name, split_name)
    os.makedirs(experiment_output_dir, exist_ok=True)
    return experiment_output_dir


# ─── configuration ──────────────────────────────────────────────────────
DATA_DIR = "mcs_bench/data"  # Fixed: Use correct relative path from mcs/ directory
CUSTOM_SPLITS_DIR = (
    f"{DATA_DIR}/time_splits_ghrepo"  # Directory for custom PDB ID lists
)
ALLOWED_PDBIDS_FILE = f"{DATA_DIR}/templ_pdbids.txt"
# PDBBIND_DATES_FILE = f"{DATA_DIR}/pdbbind_dates.json" # No longer needed for UniProt exclusion

# Output directories for this custom benchmark
OUTPUT_DIR_CUSTOM = (
    "mcs_bench/output_custom"  # Fixed: Use correct relative path from mcs/ directory
)
RESULTS_DIR_CUSTOM = "mcs_bench/benchmark_results_custom"  # Fixed: Use correct relative path from mcs/ directory

# Input PDB ID Files for custom splits
TRAIN_PDB_FILE = os.path.join(CUSTOM_SPLITS_DIR, "timesplit_no_lig_overlap_train")
VAL_PDB_FILE = os.path.join(CUSTOM_SPLITS_DIR, "timesplit_no_lig_overlap_val")
TEST_PDB_FILE = os.path.join(CUSTOM_SPLITS_DIR, "timesplit_test")

# Create timestamp for this benchmark run
BENCHMARK_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# ─── logging setup ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s – %(message)s"
)
log = logging.getLogger("custom-split-benchmark")

# ─── enhanced MCS analysis functions ─────────────────────────────────────


def extract_mcs_info_from_sdf(sdf_file: str) -> Dict:
    """Extract MCS information from the first molecule in SDF."""
    try:
        suppl = Chem.SDMolSupplier(sdf_file)
        mol = next(suppl, None)
        if mol and mol.HasProp("mcs_smarts"):
            return {
                "smarts": mol.GetProp("mcs_smarts"),
                "atom_count": (
                    int(mol.GetProp("mcs_atom_count"))
                    if mol.HasProp("mcs_atom_count")
                    else 0
                ),
                "bond_count": (
                    int(mol.GetProp("mcs_bond_count"))
                    if mol.HasProp("mcs_bond_count")
                    else 0
                ),
                "similarity_score": (
                    float(mol.GetProp("mcs_similarity_score"))
                    if mol.HasProp("mcs_similarity_score")
                    else 0.0
                ),
                "query_atoms": (
                    mol.GetProp("mcs_query_atoms")
                    if mol.HasProp("mcs_query_atoms")
                    else ""
                ),
                "template_atoms": (
                    mol.GetProp("mcs_template_atoms")
                    if mol.HasProp("mcs_template_atoms")
                    else ""
                ),
            }
    except Exception as e:
        log.warning(f"Could not extract MCS info from {sdf_file}: {e}")

    return {}


def analyze_mcs_patterns(results_data: Dict) -> Dict:
    """Analyze MCS patterns across all results."""
    mcs_analysis = {
        "unique_smarts_patterns": set(),
        "atom_count_distribution": [],
        "bond_count_distribution": [],
        "similarity_score_distribution": [],
        "pattern_frequency": defaultdict(int),
    }

    for result in results_data.get("results", {}).values():
        if result.get("success") and result.get("mcs_info"):
            mcs_info = result["mcs_info"]

            smarts = mcs_info.get("smarts", "")
            if smarts:
                mcs_analysis["unique_smarts_patterns"].add(smarts)
                mcs_analysis["pattern_frequency"][smarts] += 1

            mcs_analysis["atom_count_distribution"].append(
                mcs_info.get("atom_count", 0)
            )
            mcs_analysis["bond_count_distribution"].append(
                mcs_info.get("bond_count", 0)
            )
            mcs_analysis["similarity_score_distribution"].append(
                mcs_info.get("similarity_score", 0.0)
            )

    # Calculate statistics
    mcs_analysis["statistics"] = {
        "total_unique_patterns": len(mcs_analysis["unique_smarts_patterns"]),
        "avg_atom_count": (
            np.mean(mcs_analysis["atom_count_distribution"])
            if mcs_analysis["atom_count_distribution"]
            else 0
        ),
        "avg_bond_count": (
            np.mean(mcs_analysis["bond_count_distribution"])
            if mcs_analysis["bond_count_distribution"]
            else 0
        ),
        "avg_similarity_score": (
            np.mean(mcs_analysis["similarity_score_distribution"])
            if mcs_analysis["similarity_score_distribution"]
            else 0
        ),
        "most_common_patterns": dict(
            sorted(
                mcs_analysis["pattern_frequency"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
        ),
    }

    return mcs_analysis


def save_enhanced_results(
    results_data: Dict,
    output_file: str,
    benchmark_name: str,
    split_name: str,
    organized_dirs: Optional[Dict[str, str]] = None,
) -> None:
    """Enhanced results saving with MCS analysis and organized output structure aligned with polaris."""

    # Perform MCS analysis
    mcs_analysis = analyze_mcs_patterns(results_data)
    results_data["mcs_analysis"] = mcs_analysis

    # Use provided organized directories or create basic structure
    if organized_dirs:
        results_data["organized_directories"] = organized_dirs

        # Save MCS analysis to organized location
        mcs_analysis_file = os.path.join(
            organized_dirs["mcs_patterns"], f"{split_name}_mcs_analysis.json"
        )
        with open(mcs_analysis_file, "w") as f:
            # Convert sets to lists for JSON serialization
            mcs_analysis_serializable = dict(mcs_analysis)
            mcs_analysis_serializable["unique_smarts_patterns"] = list(
                mcs_analysis["unique_smarts_patterns"]
            )
            json.dump(mcs_analysis_serializable, f, indent=2)

        log.info(f"Saved MCS analysis to {mcs_analysis_file}")

        # Save main results to organized location
        if "experiment_dir" in organized_dirs:
            # save at experiment directory level
            experiment_results_file = os.path.join(
                organized_dirs["results"],
                f"results_{split_name}_{BENCHMARK_TIMESTAMP}.json",
            )
        else:
            # Fallback to organized results subdirectory
            experiment_results_file = os.path.join(
                organized_dirs["results"], f"{split_name}_detailed_results.json"
            )

        with open(experiment_results_file, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        log.info(f"Organized results saved to {experiment_results_file}")

        # Also save to the original location for backward compatibility if it's different
        if experiment_results_file != output_file:
            with open(output_file, "w") as f:
                json.dump(results_data, f, indent=2, default=str)
            log.info(f"Results also saved to {output_file} for backward compatibility")

    else:
        # Fallback to original behavior
        base_dir = os.path.dirname(output_file)

        organized_dirs_fallback = {
            "poses": os.path.join(base_dir, "poses", benchmark_name, split_name),
            "analysis": os.path.join(base_dir, "analysis", benchmark_name),
            "mcs_patterns": os.path.join(base_dir, "mcs_patterns", benchmark_name),
        }

        for dir_path in organized_dirs_fallback.values():
            os.makedirs(dir_path, exist_ok=True)

        results_data["organized_directories"] = organized_dirs_fallback

        # Save MCS analysis separately
        mcs_analysis_file = os.path.join(
            organized_dirs_fallback["mcs_patterns"], f"{split_name}_mcs_analysis.json"
        )
        with open(mcs_analysis_file, "w") as f:
            # Convert sets to lists for JSON serialization
            mcs_analysis_serializable = dict(mcs_analysis)
            mcs_analysis_serializable["unique_smarts_patterns"] = list(
                mcs_analysis["unique_smarts_patterns"]
            )
            json.dump(mcs_analysis_serializable, f, indent=2)

        log.info(f"Saved MCS analysis to {mcs_analysis_file}")

        # Save main results
        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        log.info(f"Results saved to {output_file}")

    # Print summary statistics
    total_molecules = len(results_data.get("results", {}))
    successful = sum(
        1 for r in results_data.get("results", {}).values() if r.get("success")
    )
    mcs_found = sum(
        1 for r in results_data.get("results", {}).values() if r.get("mcs_found")
    )

    log.info(f"Benchmark Summary for {split_name}:")
    log.info(f"  Total molecules: {total_molecules}")
    log.info(f"  Successful: {successful} ({successful/total_molecules*100:.1f}%)")
    log.info(f"  MCS found: {mcs_found} ({mcs_found/total_molecules*100:.1f}%)")

    if mcs_analysis["statistics"]:
        stats = mcs_analysis["statistics"]
        log.info(f"  Unique MCS patterns: {stats['total_unique_patterns']}")
        log.info(f"  Avg MCS atoms: {stats['avg_atom_count']:.1f}")
        log.info(f"  Avg similarity score: {stats['avg_similarity_score']:.3f}")


def validate_paths():
    """Validate that necessary paths exist."""
    required_files = [
        ALLOWED_PDBIDS_FILE,
        # PDBBIND_DATES_FILE, # No longer needed
        TRAIN_PDB_FILE,
        VAL_PDB_FILE,
        TEST_PDB_FILE,
        f"{DATA_DIR}/processed_ligands_new.sdf.gz",
    ]
    for f_path in required_files:
        if not os.path.exists(f_path):
            log.error(f"Required file not found: {f_path}")
            return False

    pdb_refined_dir = f"{DATA_DIR}/PDBbind_v2020_refined/refined-set"
    pdb_other_dir = f"{DATA_DIR}/PDBbind_v2020_other_PL/v2020-other-PL"
    if not os.path.exists(pdb_refined_dir) and not os.path.exists(pdb_other_dir):
        log.error("Neither PDBbind refined nor other-PL directories found.")
        return False

    for directory in [RESULTS_DIR_CUSTOM, OUTPUT_DIR_CUSTOM]:
        if not os.path.exists(directory):
            log.info(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)

    log.info("All required paths for custom benchmark validated successfully.")
    return True


def load_pdb_ids_from_file(filepath: str) -> Set[str]:
    """Load PDB IDs from a text file (one PDB ID per line)."""
    pdbs = set()
    if os.path.exists(filepath):
        with open(filepath) as f:
            for line in f:
                pdb_id = line.strip().lower()  # Ensure consistent casing
                if pdb_id:
                    pdbs.add(pdb_id)
    else:
        log.warning(f"PDB ID file not found: {filepath}")
    return pdbs


# def load_pdb_to_uniprot_mapping(json_filepath: str) -> Dict[str, str]:
#     """Load PDB to UniProt ID mapping from pdbbind_dates.json file."""
#     pdb_to_uniprot = {}
#     if os.path.exists(json_filepath):
#         with open(json_filepath) as f:
#             data = json.load(f)
#             for pdb_id, info in data.items():
#                 # Ensure consistent casing for lookup
#                 pdb_id_lower = pdb_id.lower()
#                 if isinstance(info, dict) and "uniprot" in info:
#                     pdb_to_uniprot[pdb_id_lower] = info["uniprot"]
#     else:
#         log.error(f"UniProt mapping file not found: {json_filepath}")
#     return pdb_to_uniprot


def extract_rmsd_from_json_structured(json_file: str) -> Dict[str, Dict[str, float]]:
    """Extract RMSD data from structured pipeline results JSON."""
    rmsd_data = {}

    try:
        import json

        with open(json_file, "r") as f:
            data = json.load(f)

        # Extract RMSD data from poses section
        poses = data.get("poses", {})
        for metric, pose_data in poses.items():
            if pose_data.get("rmsd_to_crystal") is not None:
                rmsd = float(pose_data["rmsd_to_crystal"])
                scores = pose_data.get("scores", {})
                score = scores.get(metric, 0.0)
                rmsd_data[metric] = {"score": score, "rmsd": rmsd}

    except Exception as e:
        log.debug(f"Error reading structured JSON for RMSD: {e}")

    return rmsd_data


def extract_rmsd_from_stdout(stdout_text: str) -> Dict[str, Dict[str, float]]:
    """Extract RMSD values from pipeline stdout using Unicode table parsing.

    Expected table format:
    ┌────────┬────────┬────────┬──────────┐
    │ metric │ score  │ RMSD   │ Template │
    ├────────┼────────┼────────┼──────────┤
    │ shape  │  0.513 │ 28.089 │ 1bkm     │
    │ color  │  0.076 │ 27.529 │ 1bkm     │
    │ combo  │  0.281 │ 28.089 │ 1bkm     │
    └────────┴────────┴────────┴──────────┘
    """
    rmsd_results = {}

    if not stdout_text:
        return rmsd_results

    lines = stdout_text.split("\n")
    table_start = False

    for line in lines:
        if "Final RMSD to crystal" in line:
            table_start = True
            continue

        if table_start and "│" in line:
            parts = line.split("│")
            # Updated: Now expecting 6 parts for 4-column table (empty + 4 cols + empty)
            if len(parts) >= 6:
                try:
                    metric = parts[1].strip()
                    score = float(parts[2].strip())
                    rmsd = float(parts[3].strip())
                    # parts[4] would be the template name (not needed for RMSD data)

                    if metric and metric not in ["metric", "────────"]:
                        rmsd_results[metric] = {"score": score, "rmsd": rmsd}
                except (ValueError, IndexError) as e:
                    # Skip header row and malformed lines silently
                    continue

        if table_start and "└" in line:
            break

    return rmsd_results


def find_output_files(output_dir: str, target_pdb: str) -> Dict[str, Optional[str]]:
    """Find output files generated by the pipeline."""
    output_files = {
        "poses_sdf": None,
        "all_poses_sdf": None,
        "structured_json": None,
        "template_sdf": None,
        "error_report": None,
        "alignment_report": None,
    }

    if not os.path.exists(output_dir):
        return output_files

    patterns = {
        "poses_sdf": f"{target_pdb}_poses_multi*.sdf",
        "all_poses_sdf": f"{target_pdb}_all_poses_ranked*.sdf",
        "structured_json": f"{target_pdb}_pipeline_results*.json",
        "template_sdf": f"target_{target_pdb}_mces_winner_template_*.sdf",
        "error_report": f"{target_pdb}_pipeline_errors*.json",
        "alignment_report": f"{target_pdb}_alignment_tracking*.json",
    }

    for file_type, pattern in patterns.items():
        import glob

        matches = glob.glob(os.path.join(output_dir, pattern))
        if matches:
            # Sort to get the most recent file (collision detection creates _v2, _v3, etc.)
            output_files[file_type] = sorted(matches)[-1]

    return output_files


def extract_rmsd_from_sdf(sdf_file: str) -> Dict[str, Dict[str, float]]:
    """Extract RMSD data from poses SDF file as fallback."""
    rmsd_data = {}

    try:
        from rdkit import Chem

        supplier = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False)
        for mol in supplier:
            if mol is None:
                continue

            if mol.HasProp("metric") and mol.HasProp("rmsd_to_crystal"):
                metric = mol.GetProp("metric")
                try:
                    rmsd = float(mol.GetProp("rmsd_to_crystal"))
                    score = 0.0

                    if mol.HasProp("metric_score"):
                        score = float(mol.GetProp("metric_score"))
                    elif mol.HasProp(f"tanimoto_{metric}_score"):
                        score = float(mol.GetProp(f"tanimoto_{metric}_score"))

                    rmsd_data[metric] = {"score": score, "rmsd": rmsd}
                except (ValueError, TypeError):
                    continue

    except Exception as e:
        print(f"Warning: Failed to extract RMSD from SDF {sdf_file}: {e}")

    return rmsd_data


def extract_ca_rmsd_data(
    experiment_output_dir: str, target_pdb: str, output_files: Dict[str, Optional[str]]
) -> Dict[str, Any]:
    """Extract CA RMSD data from multiple sources."""
    ca_rmsd_data = {}

    # Try to extract from SDF files first (template properties)
    try:
        if output_files.get("poses_sdf"):
            ca_data_sdf = extract_ca_rmsd_from_sdf(output_files["poses_sdf"])
            if ca_data_sdf:
                ca_rmsd_data.update(ca_data_sdf)
    except Exception as e:
        log.debug(f"Failed to extract CA RMSD from SDF: {e}")

    # Try to extract from structured JSON (more reliable)
    try:
        if output_files.get("structured_json"):
            ca_data_json = extract_ca_rmsd_from_json(output_files["structured_json"])
            if ca_data_json:
                ca_rmsd_data.update(ca_data_json)
    except Exception as e:
        log.debug(f"Failed to extract CA RMSD from JSON: {e}")

    # Try to extract from alignment tracking report (most detailed)
    try:
        if output_files.get("alignment_report"):
            ca_data_tracking = extract_ca_rmsd_from_alignment_report(
                output_files["alignment_report"]
            )
            if ca_data_tracking:
                ca_rmsd_data.update(ca_data_tracking)
    except Exception as e:
        log.debug(f"Failed to extract CA RMSD from alignment report: {e}")

    return ca_rmsd_data


def extract_ca_rmsd_from_sdf(sdf_file: str) -> Dict[str, Any]:
    """Extract CA RMSD data from template properties in SDF file."""
    ca_data = {}

    try:
        from rdkit import Chem

        supplier = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False)
        for mol in supplier:
            if mol is None:
                continue

            # Look for template CA RMSD properties
            if mol.HasProp("ca_rmsd"):
                try:
                    ca_rmsd = float(mol.GetProp("ca_rmsd"))
                    ca_data["ca_rmsd"] = ca_rmsd

                    # Extract related alignment metrics
                    if mol.HasProp("aligned_residues_count"):
                        ca_data["matched_residues"] = int(
                            mol.GetProp("aligned_residues_count")
                        )

                    if mol.HasProp("aligned_percentage"):
                        coverage_str = mol.GetProp("aligned_percentage").replace(
                            "%", ""
                        )
                        ca_data["alignment_coverage"] = float(coverage_str)

                    if mol.HasProp("embedding_similarity"):
                        ca_data["embedding_similarity"] = float(
                            mol.GetProp("embedding_similarity")
                        )

                    # Found CA RMSD data, break after first valid entry
                    break

                except (ValueError, TypeError) as e:
                    log.debug(f"Error parsing CA RMSD properties: {e}")
                    continue

    except Exception as e:
        log.debug(f"Error reading SDF file for CA RMSD: {e}")

    return ca_data


def extract_ca_rmsd_from_json(json_file: str) -> Dict[str, Any]:
    """Extract CA RMSD data from structured JSON results."""
    ca_data = {}

    try:
        import json

        with open(json_file, "r") as f:
            data = json.load(f)

        # Extract from template_info section
        template_info = data.get("template_info", {})
        if template_info.get("ca_rmsd"):
            try:
                ca_rmsd = float(template_info["ca_rmsd"])
                ca_data["ca_rmsd"] = ca_rmsd

                if template_info.get("aligned_residues_count"):
                    ca_data["matched_residues"] = int(
                        template_info["aligned_residues_count"]
                    )

                if template_info.get("aligned_percentage"):
                    coverage_str = str(template_info["aligned_percentage"]).replace(
                        "%", ""
                    )
                    ca_data["alignment_coverage"] = float(coverage_str)

                if template_info.get("embedding_similarity"):
                    ca_data["embedding_similarity"] = float(
                        template_info["embedding_similarity"]
                    )

            except (ValueError, TypeError) as e:
                log.debug(f"Error parsing CA RMSD from JSON: {e}")

    except Exception as e:
        log.debug(f"Error reading JSON file for CA RMSD: {e}")

    return ca_data


def extract_ca_rmsd_from_alignment_report(alignment_file: str) -> Dict[str, Any]:
    """Extract CA RMSD data from alignment tracking report."""
    ca_data = {}

    try:
        import json

        with open(alignment_file, "r") as f:
            data = json.load(f)

        # Look for successful alignment attempts with CA RMSD data
        detailed_logs = data.get("detailed_logs", [])
        for log_entry in detailed_logs:
            if (
                log_entry.get("success")
                and log_entry.get("stage") == "superimpose_homologs"
                and log_entry.get("ca_rmsd") is not None
            ):

                try:
                    ca_rmsd = float(log_entry["ca_rmsd"])
                    ca_data["ca_rmsd"] = ca_rmsd

                    if log_entry.get("aligned_residues"):
                        ca_data["matched_residues"] = int(log_entry["aligned_residues"])

                    if log_entry.get("alignment_coverage_ref"):
                        ca_data["alignment_coverage"] = float(
                            log_entry["alignment_coverage_ref"]
                        )

                    # Found CA RMSD data, break after first valid entry
                    break

                except (ValueError, TypeError) as e:
                    log.debug(f"Error parsing CA RMSD from alignment report: {e}")
                    continue

    except Exception as e:
        log.debug(f"Error reading alignment report for CA RMSD: {e}")

    return ca_data


def run_pipeline(
    target_pdb: str,
    experiment_output_dir: str,  # NEW: Direct output directory for this experiment
    # exclude_uniprots_file: str, # No longer used by this script
    n_conformers: int = 200,
    template_knn: int = 100,
    similarity_threshold: Optional[float] = None,
    pipeline_timeout: int = 600,  # Consistent 600 seconds (10 minutes)
    multiprocessing_pipeline: bool = True,  # New flag for true_mcs_pipeline.py
    template_pool_pdb_ids_file: Optional[str] = None,  # New argument
    log_level: str = "INFO",  # Add log level parameter
    use_embedding_cache: bool = True,  # New parameter for embedding cache
    embedding_cache_dir: Optional[
        str
    ] = None,  # New parameter for custom cache directory
    batch_embedding: bool = True,  # New parameter for batch embedding
    max_batch_size: int = 8,  # New parameter for batch size
    internal_pipeline_workers: int = 1,  # NEW: always set to 1 if running in parallel
    ca_rmsd_threshold: float = 10.0,  # NEW: CA RMSD threshold parameter
    save_enhanced_output: bool = True,  # NEW: Enable enhanced output features
) -> Dict:
    """Run the MCS pipeline for a single target PDB with timeout handling."""

    # Phase 2: Dynamic timeout scaling based on conformers
    # Base timeout (1200s for 200 conformers) + additional time for higher conformer counts
    base_conformers = 200
    additional_seconds_per_conformer = 3  # 3 seconds per additional conformer

    if n_conformers > base_conformers:
        additional_time = (
            n_conformers - base_conformers
        ) * additional_seconds_per_conformer
        adjusted_timeout = pipeline_timeout + additional_time
        log.debug(
            f"Scaling timeout for {n_conformers} conformers: {pipeline_timeout}s + {additional_time}s = {adjusted_timeout}s"
        )
    else:
        adjusted_timeout = pipeline_timeout

    # Track start time
    start_time = time.time()

    # Build command
    cmd = [
        sys.executable,
        "mcs_bench/true_mcs_pipeline.py",  # Fixed: Use correct relative path from mcs/ directory
        "--target-pdb",
        target_pdb,
        "--n-conformers",
        str(n_conformers),
        "--template-knn",
        str(template_knn),
        "--output-dir",
        experiment_output_dir,
        "--log-level",
        log_level,
        "--ca-rmsd-threshold",
        str(ca_rmsd_threshold),
    ]

    # Add internal pipeline workers control
    cmd.extend(["--internal-pipeline-workers", str(internal_pipeline_workers)])

    # Add multiprocessing flag
    if multiprocessing_pipeline:
        cmd.append("--multiprocessing")
    else:
        cmd.append("--no-multiprocessing")

    # Add embedding cache parameters
    if use_embedding_cache:
        cmd.append("--use-embedding-cache")
    else:
        cmd.append("--no-use-embedding-cache")

    if embedding_cache_dir:
        cmd.extend(["--embedding-cache-dir", embedding_cache_dir])

    # Add batch embedding parameters
    if batch_embedding:
        cmd.append("--batch-embedding")
    else:
        cmd.append("--no-batch-embedding")

    cmd.extend(["--max-batch-size", str(max_batch_size)])

    # Add enhanced output features
    if save_enhanced_output:
        cmd.extend(["--save-all-poses", "--save-mcs-info"])

    # Add skip target validation to allow peptides
    cmd.append("--skip-target-validation")

    if similarity_threshold is not None:
        cmd.extend(["--similarity-threshold", str(similarity_threshold)])

    # Add template pool filtering if provided
    if template_pool_pdb_ids_file:
        cmd.extend(
            [
                "--template-pdb-ids-file",
                template_pool_pdb_ids_file,
                "--enable-pdb-filtering",
            ]
        )

    log.debug(f"Running command with {adjusted_timeout}s timeout: {' '.join(cmd)}")

    result_data = {
        "target_pdb": target_pdb,
        "success": False,
        "runtime_seconds": 0.0,
        "timeout_used": adjusted_timeout,
        "conformers_requested": n_conformers,
        "error_message": None,
        "rmsd_data": {},
        "mcs_info": {},
        "template_info": {},
        "pose_count": 0,
        "exit_code": None,
    }

    try:
        # Run without changing working directory to preserve relative paths
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=adjusted_timeout,  # Use the dynamically adjusted timeout
        )

        runtime = time.time() - start_time
        result_data["runtime_seconds"] = runtime
        result_data["exit_code"] = result.returncode

        if result.returncode == 0:
            result_data["success"] = True
            log.info(
                f"Pipeline completed successfully for {target_pdb} in {runtime:.1f}s (timeout was {adjusted_timeout}s)"
            )

            # Extract RMSD data from multiple sources (prioritize JSON)
            try:
                output_files = find_output_files(experiment_output_dir, target_pdb)
                rmsd_results = {}

                # 1. Try JSON first (most reliable)
                if output_files.get("structured_json"):
                    rmsd_results = extract_rmsd_from_json_structured(
                        output_files["structured_json"]
                    )
                    if rmsd_results:
                        result_data["rmsd_data"] = rmsd_results
                        log.debug(
                            f"Extracted RMSD data from JSON for {target_pdb}: {rmsd_results}"
                        )

                # 2. Fall back to stdout parsing if JSON failed
                if not rmsd_results:
                    rmsd_results = extract_rmsd_from_stdout(result.stdout)
                    if rmsd_results:
                        result_data["rmsd_data"] = rmsd_results
                        log.debug(
                            f"Extracted RMSD data from stdout for {target_pdb}: {rmsd_results}"
                        )
                    else:
                        log.warning(
                            f"No RMSD values parsed from stdout for {target_pdb}. Trying SDF fallback."
                        )

                        # 3. Fall back to SDF parsing if both JSON and stdout failed
                        if output_files.get("poses_sdf"):
                            rmsd_results = extract_rmsd_from_sdf(
                                output_files["poses_sdf"]
                            )
                            if rmsd_results:
                                result_data["rmsd_data"] = rmsd_results
                                log.info(
                                    f"Successfully extracted RMSD from SDF for {target_pdb}"
                                )
                            else:
                                log.warning(
                                    f"No RMSD found in SDF either for {target_pdb}"
                                )
                        else:
                            log.warning(f"No poses SDF found for {target_pdb}")

            except Exception as e:
                log.warning(f"Failed to extract RMSD data for {target_pdb}: {e}")

            # Extract MCS info from output files
            try:
                output_files = find_output_files(experiment_output_dir, target_pdb)
                if output_files.get("all_poses_sdf"):
                    mcs_info = extract_mcs_info_from_sdf(output_files["all_poses_sdf"])
                    result_data["mcs_info"] = mcs_info
                    result_data["pose_count"] = mcs_info.get("total_poses", 0)
                    log.debug(
                        f"Extracted MCS info for {target_pdb}: poses={result_data['pose_count']}"
                    )
            except Exception as e:
                log.warning(f"Failed to extract MCS info for {target_pdb}: {e}")

            # Extract CA RMSD data from multiple sources
            try:
                ca_rmsd_data = extract_ca_rmsd_data(
                    experiment_output_dir, target_pdb, output_files
                )
                if ca_rmsd_data:
                    result_data["ca_rmsd_data"] = ca_rmsd_data
                    ca_rmsd_value = ca_rmsd_data.get("ca_rmsd", "N/A")
                    if isinstance(ca_rmsd_value, (int, float)):
                        log.info(
                            f"Successfully extracted CA RMSD data for {target_pdb}: CA RMSD={ca_rmsd_value:.3f}Å"
                        )
                    else:
                        log.info(
                            f"Successfully extracted CA RMSD data for {target_pdb}: CA RMSD={ca_rmsd_value}"
                        )
                else:
                    log.debug(f"No CA RMSD data found for {target_pdb} from any source")
                    # Try to extract from template SDF files as last resort
                    template_file = output_files.get("template_sdf")
                    if template_file and os.path.exists(template_file):
                        ca_data_template = extract_ca_rmsd_from_sdf(template_file)
                        if ca_data_template:
                            result_data["ca_rmsd_data"] = ca_data_template
                            ca_rmsd_value = ca_data_template.get("ca_rmsd", "N/A")
                            if isinstance(ca_rmsd_value, (int, float)):
                                log.info(
                                    f"Extracted CA RMSD from template SDF for {target_pdb}: {ca_rmsd_value:.3f}Å"
                                )
                            else:
                                log.info(
                                    f"Extracted CA RMSD from template SDF for {target_pdb}: {ca_rmsd_value}"
                                )
            except Exception as e:
                log.warning(f"Failed to extract CA RMSD data for {target_pdb}: {e}")
        else:
            log.error(
                f"Pipeline failed for {target_pdb} with exit code {result.returncode}"
            )
            result_data["error_message"] = (
                f"Exit code {result.returncode}: {result.stderr.strip()}"
            )
            if result.stderr:
                log.error(f"STDERR for {target_pdb}: {result.stderr.strip()}")

    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        result_data["runtime_seconds"] = runtime
        result_data["error_message"] = (
            f"Pipeline timed out after {adjusted_timeout} seconds"
        )
        log.error(
            f"Pipeline timeout for {target_pdb} after {adjusted_timeout}s (requested {n_conformers} conformers)"
        )

    except Exception as e:
        runtime = time.time() - start_time
        result_data["runtime_seconds"] = runtime
        result_data["error_message"] = f"Pipeline error: {str(e)}"
        log.error(f"Pipeline error for {target_pdb}: {str(e)}")

    return result_data


def evaluate_split(
    split_name_key: str,
    pdbs_in_split: Set[str],
    template_pool_file_for_split: Optional[str],
    n_workers: int,
    n_conformers: int,
    template_knn: int,
    similarity_threshold: Optional[float],
    pipeline_timeout: int,
    multiprocessing_pipeline: bool,
    max_pdbs_to_process: Optional[int] = None,
    log_level: str = "INFO",
    use_embedding_cache: bool = True,
    embedding_cache_dir: Optional[str] = None,
    batch_embedding: bool = True,
    max_batch_size: int = 8,
    ca_rmsd_threshold: float = 10.0,  # Add the CA RMSD threshold parameter
    benchmark_timestamp: Optional[str] = None,  # NEW: Add benchmark timestamp parameter
) -> Dict:
    """Evaluate pipeline on a custom split with simple output structure."""
    if not pdbs_in_split:
        log.warning(f"No PDBs to evaluate for {split_name_key}!")
        return {"results": {}, "name": split_name_key}

    # Create simple experiment directory
    experiment_output_dir = create_experiment_directory(split_name_key)
    log.info(
        f"Created experiment directory for {split_name_key} split: {experiment_output_dir}"
    )

    pdbs_to_run = sorted(list(pdbs_in_split))
    if max_pdbs_to_process and max_pdbs_to_process > 0:
        pdbs_to_run = pdbs_to_run[:max_pdbs_to_process]
        log.info(
            f"Limiting {split_name_key} evaluation to {len(pdbs_to_run)} PDBs for testing."
        )

    log.info(f"Starting evaluation on {len(pdbs_to_run)} PDBs for {split_name_key}.")

    split_run_results = {}
    # --- CRITICAL: Always set internal pipeline workers to 1 if running in parallel ---
    internal_pipeline_workers = 1 if n_workers > 1 else (os.cpu_count() or 1)
    if n_workers > 1 and internal_pipeline_workers > 1:
        log.warning(
            f"Both benchmark n_workers ({n_workers}) and pipeline internal workers ({internal_pipeline_workers}) > 1. Setting pipeline internal workers to 1 to prevent oversubscription."
        )

    if n_workers > 1 and len(pdbs_to_run) > 1:
        # Use spawn context to prevent fork-related deadlocks
        import multiprocessing as mp

        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=mp_context
        ) as executor:
            futures = {
                executor.submit(
                    run_pipeline,
                    pdb_id,
                    experiment_output_dir,  # Pass the experiment output directory
                    n_conformers,
                    template_knn,
                    similarity_threshold,
                    pipeline_timeout,
                    multiprocessing_pipeline,
                    template_pool_file_for_split,
                    log_level,
                    use_embedding_cache,
                    embedding_cache_dir,
                    batch_embedding,
                    max_batch_size,
                    internal_pipeline_workers,  # Pass the coordinated value
                    ca_rmsd_threshold,  # Pass the CA RMSD threshold
                    True,  # Pass the save_enhanced_output flag
                ): pdb_id
                for pdb_id in pdbs_to_run
            }

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Evaluating {split_name_key}",
            ):
                pdb_id = futures[fut]
                try:
                    split_run_results[pdb_id] = fut.result()
                except Exception as e_future:
                    log.error(f"Exception from future for {pdb_id}: {str(e_future)}")
                    split_run_results[pdb_id] = {
                        "success": False,
                        "error": str(e_future),
                        "rmsd_values": {},
                    }
    else:
        log.info(
            f"Running {split_name_key} sequentially (n_workers={n_workers}, pdbs_to_run={len(pdbs_to_run)})..."
        )
        for pdb_id in tqdm(
            pdbs_to_run, desc=f"Evaluating {split_name_key} (sequential)"
        ):
            split_run_results[pdb_id] = run_pipeline(
                pdb_id,
                experiment_output_dir,
                n_conformers,
                template_knn,
                similarity_threshold,
                pipeline_timeout,
                multiprocessing_pipeline,
                template_pool_file_for_split,
                log_level,
                use_embedding_cache,
                embedding_cache_dir,
                batch_embedding,
                max_batch_size,
                os.cpu_count()
                or 1,  # Use all CPUs for pipeline if running sequentially
                ca_rmsd_threshold,  # Pass the CA RMSD threshold
                True,  # Pass the save_enhanced_output flag
            )

    # Simple result structure like polaris
    result = {
        "results": split_run_results,
        "name": split_name_key,
        "experiment_output_dir": experiment_output_dir,
        "total_molecules": len(pdbs_to_run),
    }
    return result


def calculate_metrics(split_results_data: Dict) -> Dict:
    pdb_results = split_results_data.get("results", {})
    if not pdb_results:
        return {"total": 0, "successful": 0}

    metrics = {
        "total": len(pdb_results),
        "successful": sum(
            1 for r_data in pdb_results.values() if r_data.get("success")
        ),
        "failed": sum(
            1 for r_data in pdb_results.values() if not r_data.get("success")
        ),
        "rmsd_counts_2A": defaultdict(int),
        "rmsd_counts_5A": defaultdict(int),
        "all_rmsds": defaultdict(list),
        "all_scores": defaultdict(list),
        "runtimes": [],
        # New fields for CA RMSD metrics
        "ca_rmsd_values": [],
        "ca_rmsd_matched_residues": [],
        "ca_rmsd_coverage": [],
        "ca_rmsd_filtered_count": 0,  # Count of templates filtered by CA RMSD threshold
        # New fields for enhanced RMSD statistics
        "min_rmsd": defaultdict(lambda: float("inf")),
        "max_rmsd": defaultdict(lambda: float("-inf")),
        "outliers_50A": defaultdict(int),  # Count >50Å RMSD
    }

    for pdb_id, r_data in pdb_results.items():
        if r_data.get("success"):
            # FIX: Use "runtime_seconds" instead of "runtime"
            if r_data.get("runtime_seconds") is not None:
                metrics["runtimes"].append(r_data["runtime_seconds"])

            # Process CA RMSD data
            if r_data.get("ca_rmsd_data"):
                ca_data = r_data["ca_rmsd_data"]
                if "ca_rmsd" in ca_data:
                    metrics["ca_rmsd_values"].append(ca_data["ca_rmsd"])
                if "matched_residues" in ca_data:
                    metrics["ca_rmsd_matched_residues"].append(
                        ca_data["matched_residues"]
                    )
                if "alignment_coverage" in ca_data:
                    metrics["ca_rmsd_coverage"].append(ca_data["alignment_coverage"])

            # FIX: Use "rmsd_data" instead of "rmsd_values"
            if r_data.get("rmsd_data"):
                for metric_key, values_dict in r_data["rmsd_data"].items():
                    rmsd = values_dict.get("rmsd")
                    score = values_dict.get("score")
                    if rmsd is not None:
                        metrics["all_rmsds"][metric_key].append(rmsd)
                        # Update min/max tracking
                        metrics["min_rmsd"][metric_key] = min(
                            metrics["min_rmsd"][metric_key], rmsd
                        )
                        metrics["max_rmsd"][metric_key] = max(
                            metrics["max_rmsd"][metric_key], rmsd
                        )
                        # Count outliers and thresholds
                        if rmsd <= 2.0:
                            metrics["rmsd_counts_2A"][metric_key] += 1
                        if rmsd <= 5.0:
                            metrics["rmsd_counts_5A"][metric_key] += 1
                        if rmsd > 50.0:
                            metrics["outliers_50A"][metric_key] += 1
                    if score is not None:
                        metrics["all_scores"][metric_key].append(score)

    metrics["mean_runtime"] = (
        np.mean(metrics["runtimes"]) if metrics["runtimes"] else None
    )

    # Calculate CA RMSD statistics
    if metrics["ca_rmsd_values"]:
        metrics["ca_rmsd_stats"] = {
            "count": len(metrics["ca_rmsd_values"]),
            "mean": np.mean(metrics["ca_rmsd_values"]),
            "median": np.median(metrics["ca_rmsd_values"]),
            "min": np.min(metrics["ca_rmsd_values"]),
            "max": np.max(metrics["ca_rmsd_values"]),
        }

        # Calculate average matched residues and coverage
        if metrics["ca_rmsd_matched_residues"]:
            metrics["ca_rmsd_stats"]["avg_matched_residues"] = np.mean(
                metrics["ca_rmsd_matched_residues"]
            )
        if metrics["ca_rmsd_coverage"]:
            metrics["ca_rmsd_stats"]["avg_coverage"] = np.mean(
                metrics["ca_rmsd_coverage"]
            )

        # Calculate correlation between CA RMSD and ligand RMSD if available
        for metric_key, rmsd_list in metrics["all_rmsds"].items():
            if len(rmsd_list) > 5 and len(metrics["ca_rmsd_values"]) >= len(rmsd_list):
                # Truncate CA RMSD list to match length if needed
                ca_rmsd_subset = metrics["ca_rmsd_values"][: len(rmsd_list)]
                try:
                    correlation = np.corrcoef(ca_rmsd_subset, rmsd_list)[0, 1]
                    metrics["ca_rmsd_stats"][
                        f"correlation_with_{metric_key}"
                    ] = correlation
                except Exception as e:
                    log.warning(
                        f"Could not calculate correlation between CA RMSD and {metric_key} RMSD: {e}"
                    )

    # Continue with per-method statistics
    metrics_summary_per_method = {}
    for metric_key in metrics["all_rmsds"]:
        rmsd_list = metrics["all_rmsds"][metric_key]
        score_list = metrics["all_scores"][metric_key]
        num_successful_for_metric = len(rmsd_list)

        metrics_summary_per_method[metric_key] = {
            "count": num_successful_for_metric,
            "mean_rmsd": np.mean(rmsd_list) if rmsd_list else None,
            "median_rmsd": np.median(rmsd_list) if rmsd_list else None,
            "min_rmsd": (
                metrics["min_rmsd"][metric_key]
                if metrics["min_rmsd"][metric_key] != float("inf")
                else None
            ),
            "max_rmsd": (
                metrics["max_rmsd"][metric_key]
                if metrics["max_rmsd"][metric_key] != float("-inf")
                else None
            ),
            "mean_score": np.mean(score_list) if score_list else None,
            "perc_below_2A": (
                (
                    metrics["rmsd_counts_2A"][metric_key]
                    / num_successful_for_metric
                    * 100
                )
                if num_successful_for_metric > 0
                else 0
            ),
            "perc_below_5A": (
                (
                    metrics["rmsd_counts_5A"][metric_key]
                    / num_successful_for_metric
                    * 100
                )
                if num_successful_for_metric > 0
                else 0
            ),
            "outliers_above_50A": metrics["outliers_50A"][metric_key],
        }
    metrics["per_method_stats"] = metrics_summary_per_method

    return metrics


def plot_rmsd_distribution(
    split_results_data: Dict,
    split_name_key: str,
    output_plot_dir: str,
    histogram_bins: int = 50,
) -> None:
    pdb_results = split_results_data.get("results", {})
    if not pdb_results:
        return
    rmsds_by_metric = defaultdict(list)
    for r_data in pdb_results.values():
        if r_data.get("success") and r_data.get("rmsd_data"):
            for metric_key, values_dict in r_data["rmsd_data"].items():
                if values_dict.get("rmsd") is not None:
                    rmsds_by_metric[metric_key].append(values_dict["rmsd"])
    if not rmsds_by_metric:
        return

    # Use the results directory for plots
    plot_dir = output_plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    metric_colors = {"shape": "#2ecc71", "color": "#e74c3c", "combo": "#3498db"}
    for metric_key, values in rmsds_by_metric.items():
        if not values:
            continue
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(10, 6))
        color = metric_colors.get(metric_key, "#95a5a6")
        plt.hist(
            values,
            bins=histogram_bins,
            alpha=0.8,
            color=color,
            edgecolor="black",
            linewidth=1.2,
            zorder=2,
        )
        plt.axvline(
            x=2.0, color="#e67e22", linestyle="--", linewidth=2.5, label="2Å", zorder=1
        )
        plt.axvline(
            x=5.0, color="#9b59b6", linestyle="-.", linewidth=2.5, label="5Å", zorder=1
        )
        plt.xlabel("RMSD (Å)", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.title(
            f"RMSD Distribution for {split_name_key} ({metric_key.capitalize()})",
            fontsize=16,
            fontweight="bold",
        )
        mean_val, median_val = np.mean(values), np.median(values)
        min_val, max_val = np.min(values), np.max(values)
        outliers_50 = sum(1 for v in values if v > 50.0)
        plt.text(
            0.95,
            0.95,
            f"Mean: {mean_val:.2f}Å\nMedian: {median_val:.2f}Å\nMin: {min_val:.2f}Å\nMax: {max_val:.2f}Å\n>50Å: {outliers_50}",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
            fontsize=12,
        )
        s2A, s5A = (
            sum(1 for v in values if v <= t) / len(values) * 100 for t in [2.0, 5.0]
        )
        plt.text(
            0.05,
            0.95,
            f"≤2Å: {s2A:.1f}%\n≤5Å: {s5A:.1f}%",
            transform=plt.gca().transAxes,
            ha="left",
            va="top",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
            fontsize=12,
        )
        plt.xlim(left=0, right=max(10, min(max(values) + 1 if values else 10, 20)))
        plt.legend(
            loc="center right", frameon=True, fontsize=12, bbox_to_anchor=(0.98, 0.5)
        )
        plt.grid(True, linestyle="--", alpha=0.7, zorder=0)
        plt.gca().set_axisbelow(True)
        plt.gca().set_facecolor("#f9f9f9")
        plt.tight_layout()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(
            plot_dir, f"rmsd_dist_{split_name_key}_{metric_key}_{ts}.png"
        )
        plt.savefig(plot_file, dpi=300)
        log.info(f"Saved RMSD plot to {plot_file}")
        plt.close()


def plot_ca_rmsd_distribution(
    split_results_data: Dict,
    split_name_key: str,
    output_plot_dir: str,
    histogram_bins: int = 50,
) -> None:
    """Create visualization of CA RMSD distribution."""
    pdb_results = split_results_data.get("results", {})
    if not pdb_results:
        return

    # Use the results directory for plots
    plot_dir = output_plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    # Collect CA RMSD values
    ca_rmsd_values = []
    for r_data in pdb_results.values():
        if (
            r_data.get("success")
            and r_data.get("ca_rmsd_data")
            and "ca_rmsd" in r_data["ca_rmsd_data"]
        ):
            ca_rmsd_values.append(r_data["ca_rmsd_data"]["ca_rmsd"])

    if not ca_rmsd_values:
        log.warning(
            f"No CA RMSD values found for {split_name_key}, skipping CA RMSD plot"
        )
        return

    # Plot CA RMSD distribution
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(10, 6))
    color = "#3498db"  # Blue color for CA RMSD
    plt.hist(
        ca_rmsd_values,
        bins=histogram_bins,
        alpha=0.8,
        color=color,
        edgecolor="black",
        linewidth=1.2,
        zorder=2,
    )

    # Add a vertical line at the threshold value (assume 10.0 if not otherwise known)
    threshold = 10.0  # Default threshold
    plt.axvline(
        x=threshold,
        color="#e74c3c",
        linestyle="--",
        linewidth=2.5,
        label=f"{threshold}Å Threshold",
        zorder=1,
    )

    plt.xlabel("C-alpha RMSD (Å)", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title(
        f"C-alpha RMSD Distribution for {split_name_key}",
        fontsize=16,
        fontweight="bold",
    )

    # Add statistics with min/max/outliers
    mean_val, median_val = np.mean(ca_rmsd_values), np.median(ca_rmsd_values)
    min_val, max_val = np.min(ca_rmsd_values), np.max(ca_rmsd_values)
    outliers_50 = sum(1 for v in ca_rmsd_values if v > 50.0)
    plt.text(
        0.95,
        0.95,
        f"Mean: {mean_val:.2f}Å\nMedian: {median_val:.2f}Å\nMin: {min_val:.2f}Å\nMax: {max_val:.2f}Å\n>50Å: {outliers_50}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
        fontsize=12,
    )

    # Calculate percentage below threshold
    pct_below = (
        sum(1 for v in ca_rmsd_values if v <= threshold) / len(ca_rmsd_values) * 100
    )
    plt.text(
        0.05,
        0.95,
        f"≤{threshold}Å: {pct_below:.1f}%",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
        fontsize=12,
    )

    plt.xlim(
        left=0,
        right=max(
            threshold * 1.5,
            min(
                max(ca_rmsd_values) + 1 if ca_rmsd_values else threshold * 1.5,
                threshold * 2,
            ),
        ),
    )
    plt.legend(
        loc="center right", frameon=True, fontsize=12, bbox_to_anchor=(0.98, 0.5)
    )
    plt.grid(True, linestyle="--", alpha=0.7, zorder=0)
    plt.gca().set_axisbelow(True)
    plt.gca().set_facecolor("#f9f9f9")
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = os.path.join(plot_dir, f"ca_rmsd_dist_{split_name_key}_{ts}.png")
    plt.savefig(plot_file, dpi=300)
    log.info(f"Saved CA RMSD plot to {plot_file}")
    plt.close()


def plot_ca_rmsd_vs_ligand_rmsd(
    split_results_data: Dict, split_name_key: str, output_plot_dir: str
) -> None:
    """Create scatter plot showing correlation between CA RMSD and ligand RMSD."""
    pdb_results = split_results_data.get("results", {})
    if not pdb_results:
        return

    # Use the results directory for plots
    plot_dir = output_plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    # Collect data: we need pairs of (CA RMSD, ligand RMSD) values
    data_pairs = defaultdict(
        list
    )  # Dict mapping metric -> list of (ca_rmsd, ligand_rmsd) pairs

    for r_data in pdb_results.values():
        if (
            not r_data.get("success")
            or not r_data.get("ca_rmsd_data")
            or "ca_rmsd" not in r_data["ca_rmsd_data"]
        ):
            continue

        ca_rmsd = r_data["ca_rmsd_data"]["ca_rmsd"]

        if not r_data.get("rmsd_data"):
            continue

        for metric_key, values_dict in r_data["rmsd_data"].items():
            if "rmsd" in values_dict:
                ligand_rmsd = values_dict["rmsd"]
                data_pairs[metric_key].append((ca_rmsd, ligand_rmsd))

    if not data_pairs:
        log.warning(
            f"No CA RMSD vs ligand RMSD pairs found for {split_name_key}, skipping correlation plot"
        )
        return

    # Create one plot per metric
    metric_colors = {"shape": "#2ecc71", "color": "#e74c3c", "combo": "#3498db"}

    for metric_key, pairs in data_pairs.items():
        if len(pairs) < 5:  # Need at least a few points for meaningful correlation
            continue

        x_vals = [p[0] for p in pairs]  # CA RMSD
        y_vals = [p[1] for p in pairs]  # Ligand RMSD

        plt.style.use("seaborn-v0_8-whitegrid")
        plt.figure(figsize=(10, 8))
        color = metric_colors.get(metric_key, "#95a5a6")

        # Create scatter plot
        plt.scatter(
            x_vals, y_vals, alpha=0.7, color=color, edgecolor="black", s=70, zorder=3
        )

        # Add trend line
        try:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            plt.plot(
                sorted(x_vals),
                p(sorted(x_vals)),
                "r--",
                alpha=0.8,
                linewidth=2,
                zorder=2,
            )

            # Calculate correlation
            correlation = np.corrcoef(x_vals, y_vals)[0, 1]
        except Exception as e:
            log.warning(f"Error calculating trend line for {metric_key}: {e}")
            correlation = float("nan")

        plt.xlabel("C-alpha RMSD (Å)", fontsize=14)
        plt.ylabel("Ligand RMSD (Å)", fontsize=14)
        plt.title(
            f"C-alpha RMSD vs Ligand RMSD ({metric_key.capitalize()}) for {split_name_key}",
            fontsize=16,
            fontweight="bold",
        )

        # Add correlation info
        if not np.isnan(correlation):
            plt.text(
                0.05,
                0.95,
                f"Correlation: {correlation:.3f}",
                transform=plt.gca().transAxes,
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
                fontsize=12,
            )

        # Add horizontal lines at common RMSD thresholds
        plt.axhline(
            y=2.0,
            color="#e67e22",
            linestyle="--",
            linewidth=1.5,
            label="2Å (ligand)",
            alpha=0.7,
            zorder=1,
        )
        plt.axhline(
            y=5.0,
            color="#9b59b6",
            linestyle="-.",
            linewidth=1.5,
            label="5Å (ligand)",
            alpha=0.7,
            zorder=1,
        )

        # Add diagonal line (x=y) to show where points would fall if CA RMSD = ligand RMSD
        max_val = max(max(x_vals), max(y_vals)) + 1
        plt.plot([0, max_val], [0, max_val], "k-", alpha=0.3, linewidth=1, zorder=1)

        plt.grid(True, linestyle="--", alpha=0.7, zorder=0)
        plt.gca().set_axisbelow(True)
        plt.gca().set_facecolor("#f9f9f9")
        plt.legend(
            loc="lower right", frameon=True, fontsize=11, bbox_to_anchor=(0.98, 0.02)
        )
        plt.tight_layout()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(
            plot_dir, f"ca_rmsd_vs_ligand_rmsd_{split_name_key}_{metric_key}_{ts}.png"
        )
        plt.savefig(plot_file, dpi=300)
        log.info(f"Saved CA RMSD vs ligand RMSD correlation plot to {plot_file}")
        plt.close()


def save_comparison_table(all_benchmark_metrics: Dict, output_results_dir: str) -> None:
    # Update the header to include CA RMSD statistics
    header = [
        "Split",
        "N_PDBs",
        "TotalSuccess(%)",
        "Method",
        "MeanRMSD(Å)",
        "MedianRMSD(Å)",
        "%≤2Å",
        "%≤5Å",
        "MeanScore",
        "CA_RMSD_Count",
        "MeanCA_RMSD(Å)",
        "MedianCA_RMSD(Å)",
        "AvgMatchedResidues",
        "AvgCoverage(%)",
    ]
    table_rows = [header]

    for split_key, metrics_data in all_benchmark_metrics.items():
        if not metrics_data or not metrics_data.get("per_method_stats"):
            continue

        total_pdbs = metrics_data.get("total", 0)
        successful_runs = metrics_data.get("successful", 0)
        overall_success_rate = (
            (successful_runs / total_pdbs * 100) if total_pdbs > 0 else 0
        )

        # Get CA RMSD stats
        ca_rmsd_stats = metrics_data.get("ca_rmsd_stats", {})
        ca_rmsd_count = ca_rmsd_stats.get("count", 0)
        ca_rmsd_mean = ca_rmsd_stats.get("mean", None)
        ca_rmsd_median = ca_rmsd_stats.get("median", None)
        ca_rmsd_avg_residues = ca_rmsd_stats.get("avg_matched_residues", None)
        ca_rmsd_avg_coverage = ca_rmsd_stats.get("avg_coverage", None)

        for method_name, stats in metrics_data["per_method_stats"].items():
            row = [
                split_key,
                total_pdbs,
                f"{overall_success_rate:.1f}",
                method_name.capitalize(),
                (
                    f"{stats.get('mean_rmsd', 'N/A'):.2f}"
                    if isinstance(stats.get("mean_rmsd"), float)
                    else "N/A"
                ),
                (
                    f"{stats.get('median_rmsd', 'N/A'):.2f}"
                    if isinstance(stats.get("median_rmsd"), float)
                    else "N/A"
                ),
                (
                    f"{stats.get('perc_below_2A', 'N/A'):.1f}"
                    if isinstance(stats.get("perc_below_2A"), float)
                    else "N/A"
                ),
                (
                    f"{stats.get('perc_below_5A', 'N/A'):.1f}"
                    if isinstance(stats.get("perc_below_5A"), float)
                    else "N/A"
                ),
                (
                    f"{stats.get('mean_score', 'N/A'):.3f}"
                    if isinstance(stats.get("mean_score"), float)
                    else "N/A"
                ),
                # Add CA RMSD stats
                ca_rmsd_count,
                f"{ca_rmsd_mean:.2f}" if isinstance(ca_rmsd_mean, float) else "N/A",
                f"{ca_rmsd_median:.2f}" if isinstance(ca_rmsd_median, float) else "N/A",
                (
                    f"{ca_rmsd_avg_residues:.1f}"
                    if isinstance(ca_rmsd_avg_residues, float)
                    else "N/A"
                ),
                (
                    f"{ca_rmsd_avg_coverage:.1f}"
                    if isinstance(ca_rmsd_avg_coverage, float)
                    else "N/A"
                ),
            ]
            table_rows.append(row)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(table_rows[1:], columns=table_rows[0])

    # Use the results directory
    os.makedirs(output_results_dir, exist_ok=True)

    csv_file = os.path.join(output_results_dir, f"custom_benchmark_summary_{ts}.csv")
    df.to_csv(csv_file, index=False)
    log.info(f"Saved summary table to {csv_file}")

    md_file = os.path.join(output_results_dir, f"custom_benchmark_summary_{ts}.md")
    with open(md_file, "w") as f:
        f.write(f"# Custom Benchmark Summary ({ts})\n\n")
        f.write(df.to_markdown(index=False))

        # Add CA RMSD correlation section if available
        for split_key, metrics_data in all_benchmark_metrics.items():
            ca_rmsd_stats = metrics_data.get("ca_rmsd_stats", {})
            correlation_metrics = [
                k for k in ca_rmsd_stats.keys() if k.startswith("correlation_with_")
            ]

            if correlation_metrics:
                f.write(f"\n\n## CA RMSD Correlations for {split_key}\n\n")
                f.write("| Method | Correlation with Ligand RMSD |\n")
                f.write("|--------|-----------------------------|\n")

                for metric in correlation_metrics:
                    method = metric.replace("correlation_with_", "").capitalize()
                    correlation = ca_rmsd_stats[metric]
                    f.write(f"| {method} | {correlation:.3f} |\n")

    log.info(f"Saved summary table to {md_file}")
    print("\nCustom Benchmark Summary:")
    print(df.to_string(index=False))


def analyze_alignment_tracking_logs(split_results_data: Dict) -> Dict:
    """Analyze alignment tracking logs from pipeline runs to identify missing CA RMSD patterns.

    Args:
        split_results_data: Results data with pipeline outputs

    Returns:
        Analysis summary of alignment tracking patterns
    """
    alignment_analysis = {
        "total_pdbs_processed": 0,
        "pdbs_with_alignment_logs": 0,
        "common_failure_stages": {},
        "missing_ca_rmsd_patterns": [],
        "validation_issues": [],
    }

    # Extract experiment output directory for file searches
    experiment_output_dir = split_results_data.get("experiment_output_dir", "")

    for pdb_id, pdb_data in split_results_data.get("results", {}).items():
        if not isinstance(pdb_data, dict):
            continue

        alignment_analysis["total_pdbs_processed"] += 1

        # Look for alignment tracking report using find_output_files
        output_files = find_output_files(experiment_output_dir, pdb_id)
        alignment_report_path = output_files.get("alignment_report")

        if alignment_report_path and os.path.exists(alignment_report_path):
            alignment_analysis["pdbs_with_alignment_logs"] += 1

            try:
                with open(alignment_report_path, "r") as f:
                    alignment_data = json.load(f)

                # Analyze failure stages
                if "detailed_logs" in alignment_data:
                    for log_entry in alignment_data["detailed_logs"]:
                        if not log_entry.get("success", True):
                            stage = log_entry.get("stage", "unknown")
                            alignment_analysis["common_failure_stages"][stage] = (
                                alignment_analysis["common_failure_stages"].get(
                                    stage, 0
                                )
                                + 1
                            )

                # Check for missing CA RMSD in successful cases
                if "summary" in alignment_data:
                    summary = alignment_data["summary"]
                    if summary.get("missing_ca_rmsd_count", 0) > 0:
                        alignment_analysis["missing_ca_rmsd_patterns"].append(
                            {
                                "pdb_id": pdb_id,
                                "missing_count": summary["missing_ca_rmsd_count"],
                                "total_attempts": summary.get("total_attempts", 0),
                                "success_rate": summary.get("success_rate", 0.0),
                            }
                        )

            except Exception as e:
                alignment_analysis["validation_issues"].append(
                    f"Failed to parse {alignment_report_path}: {e}"
                )

        # Additional validation: Check if pipeline succeeded but no CA RMSD data extracted
        if pdb_data.get("success", False):
            ca_rmsd_data = pdb_data.get("ca_rmsd_data", {})
            if not ca_rmsd_data.get("ca_rmsd"):
                alignment_analysis["validation_issues"].append(
                    f"{pdb_id}: Pipeline succeeded but no CA RMSD data extracted"
                )

    return alignment_analysis


def save_alignment_analysis_report(
    alignment_analysis: Dict, split_name: str, output_dir: str
) -> Optional[str]:
    """Save alignment analysis report for investigation.

    Args:
        alignment_analysis: Analysis results from analyze_alignment_tracking_logs
        split_name: Name of the split being analyzed
        output_dir: Output directory for the report

    Returns:
        Path to saved report file or None if failed
    """
    try:
        report_file = os.path.join(output_dir, f"alignment_analysis_{split_name}.json")

        # Add summary statistics
        analysis_with_summary = alignment_analysis.copy()
        analysis_with_summary["analysis_summary"] = {
            "alignment_log_coverage": f"{alignment_analysis['pdbs_with_alignment_logs']}/{alignment_analysis['total_pdbs_processed']}",
            "most_common_failure_stage": (
                max(
                    alignment_analysis["common_failure_stages"].items(),
                    key=lambda x: x[1],
                )[0]
                if alignment_analysis["common_failure_stages"]
                else "None"
            ),
            "pdbs_with_missing_ca_rmsd": len(
                alignment_analysis["missing_ca_rmsd_patterns"]
            ),
            "total_validation_issues": len(alignment_analysis["validation_issues"]),
        }

        with open(report_file, "w") as f:
            json.dump(analysis_with_summary, f, indent=2)

        print(f"Alignment analysis saved to: {report_file}")
        return report_file

    except Exception as e:
        print(f"Failed to save alignment analysis: {e}")
        return None


def main():
    suggested_config = get_suggested_worker_config()
    parser = argparse.ArgumentParser(description="Run custom time-split benchmark")
    parser.add_argument(
        "--n-workers",
        type=int,
        default=suggested_config["n_workers"],
        help=f"Parallel PDB processing workers (auto: {suggested_config['n_workers']}). If >1, pipeline internal workers will be set to 1 to avoid oversubscription.",
    )
    parser.add_argument(
        "--n-conformers", type=int, default=200, help="Conformers per ligand"
    )
    parser.add_argument(
        "--template-knn", type=int, default=100, help="KNN for template selection"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Similarity threshold (overrides KNN)",
    )
    parser.add_argument(
        "--max-pdbs", type=int, default=None, help="Max PDBs per split (for testing)"
    )
    parser.add_argument(
        "--multiprocessing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable multiprocessing within true_mcs_pipeli6py (default: enabled)",
    )
    parser.add_argument(
        "--pipeline-timeout",
        type=int,
        default=600,
        help="Timeout per PDB in seconds (default: 10min)",
    )
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--val-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set the log level for the true_mcs_pipeline.py subprocess (default: INFO)",
    )
    parser.add_argument(
        "--ca-rmsd-threshold",
        type=float,
        default=10.0,
        help="Maximum C-alpha RMSD in Angstroms for protein filtering (default: 10.0)",
    )

    # Add new embedding cache and batch processing arguments
    parser.add_argument(
        "--use-embedding-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable embedding disk cache (default: enabled)",
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=str,
        default=None,
        help="Custom directory for embedding cache",
    )
    parser.add_argument(
        "--clear-embedding-cache",
        action="store_true",
        help="Clear the embedding cache before running",
    )
    parser.add_argument(
        "--batch-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable batch embedding processing (default: enabled)",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Maximum batch size for embedding processing (default: 8)",
    )

    # Add histogram enhancement arguments
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=50,
        help="Number of histogram bins for RMSD distribution plots (default: 50)",
    )

    args = parser.parse_args()

    if not validate_paths():
        log.error("Path validation failed. Exiting.")
        return
    allowed_pdbids_set = load_pdb_ids_from_file(ALLOWED_PDBIDS_FILE)
    if not allowed_pdbids_set:
        log.error(
            f"Allowed PDB IDs file {ALLOWED_PDBIDS_FILE} is empty. Cannot proceed."
        )
        return
    log.info(f"Loaded {len(allowed_pdbids_set)} allowed PDB IDs.")

    # Handle embedding cache clearing if requested
    if args.clear_embedding_cache:
        if args.embedding_cache_dir:
            cache_dir = args.embedding_cache_dir
        else:
            cache_dir = os.path.expanduser(
                "~/.cache/templ/embeddings/esm2_t33_650M_UR50D"
            )

        if os.path.exists(cache_dir):
            try:
                cleared_files = 0
                for filename in os.listdir(cache_dir):
                    if filename.endswith(".npz"):
                        os.remove(os.path.join(cache_dir, filename))
                        cleared_files += 1
                log.info(
                    f"Cleared {cleared_files} files from embedding cache: {cache_dir}"
                )
            except Exception as e:
                log.error(f"Failed to clear embedding cache: {str(e)}")
        else:
            log.info(f"No embedding cache directory found at: {cache_dir}")

    splits_data = {}
    for split_key, file_path in zip(
        ["train", "val", "test"], [TRAIN_PDB_FILE, VAL_PDB_FILE, TEST_PDB_FILE]
    ):
        raw_pdbs = load_pdb_ids_from_file(file_path)
        filtered_pdbs = raw_pdbs.intersection(allowed_pdbids_set)
        splits_data[split_key] = {"raw": raw_pdbs, "filtered": filtered_pdbs}
        log.info(
            f"{split_key.capitalize()}: {len(raw_pdbs)} raw -> {len(filtered_pdbs)} filtered PDBs"
        )

    # pdb_to_uniprot = load_pdb_to_uniprot_mapping(PDBBIND_DATES_FILE) # No longer needed
    # if not pdb_to_uniprot: log.warning("Failed to load UniProt mapping. Proceeding without strict UniProt-based template exclusion.")

    # uniprot_sets = {key: {pdb_to_uniprot[pdb] for pdb in data["filtered"] if pdb in pdb_to_uniprot} for key, data in splits_data.items()}
    # uniprots_to_exclude_from_templates = uniprot_sets.get("val", set()).union(uniprot_sets.get("test", set()))
    # for key, uset in uniprot_sets.items(): log.info(f"Derived {len(uset)} UniProt IDs for filtered {key} set.")
    # log.info(f"{len(uniprots_to_exclude_from_templates)} UniProt IDs will be excluded from template pools (those in val or test).")
    # No UniProt exclusion logic at this script level anymore.
    log.info(
        "UniProt-based template exclusion between splits is NOT applied by this script."
    )
    log.info(
        "Instead, PDB-based template pool restriction is used: validation and test sets can only use templates from the training set."
    )
    log.info(
        "This is enforced by passing the training set PDB IDs file to true_mcs_pipeline.py with --enable-pdb-filtering."
    )

    all_benchmark_results = {
        "params": vars(args),
        "timestamp": BENCHMARK_TIMESTAMP,
        "split_info": {},
        "results_data": {},
        "metrics_summary": {},
    }

    # Create simple results directory like polaris
    os.makedirs(RESULTS_DIR_CUSTOM, exist_ok=True)
    log.info(f"Created results directory: {RESULTS_DIR_CUSTOM}")

    # Prepare the template pool file from the training set
    train_pdbs_for_template_pool = splits_data["train"]["filtered"]
    template_pool_file_path = None
    if train_pdbs_for_template_pool:
        template_pool_file_path = os.path.join(
            RESULTS_DIR_CUSTOM, "train_set_template_pool.txt"
        )
        with open(template_pool_file_path, "w") as f_template_pool:
            for pdb_id in sorted(list(train_pdbs_for_template_pool)):
                f_template_pool.write(f"{pdb_id}\n")
        log.info(
            f"Created template pool file from training set PDBs at: {template_pool_file_path}"
        )
    else:
        log.warning(
            "Training set is empty. Validation and Test set evaluations will not have a template pool from train."
        )

    for key, data in splits_data.items():
        all_benchmark_results["split_info"][f"custom_{key}"] = {
            "count_raw": len(data["raw"]),
            "count_filtered": len(data["filtered"]),
            # "uniprot_count": len(uniprot_sets[key]) # No uniprot_sets anymore
        }

    run_flags = {"train": args.train_only, "val": args.val_only, "test": args.test_only}
    # If no specific flags are set, run all. If any is set, run only those.
    should_run_all = not any(run_flags.values())

    for split_key in ["train", "val", "test"]:
        if should_run_all or run_flags[split_key]:
            log.info(f"\n--- Evaluating Custom {split_key.capitalize()} Set ---")

            current_template_pool_file = None
            if split_key == "train":
                # For training set, we are skipping evaluation as per the revised plan.
                # If you wanted to run train with leave-one-out, you'd pass None or a file with all train PDBs.
                # true_mcs_pipeline.py handles its own TARGET_PDB exclusion from the provided pool.
                log.info(
                    f"Skipping evaluation for '{split_key}' split as per revised plan."
                )
                # Add placeholder to results if needed or just skip
                all_benchmark_results["results_data"][f"custom_{split_key}"] = {
                    "results": {},
                    "name": f"custom_{split_key}",
                    "status": "skipped",
                }
                all_benchmark_results["metrics_summary"][f"custom_{split_key}"] = {
                    "total": 0,
                    "successful": 0,
                    "status": "skipped",
                }
                continue  # Skip to the next split
            elif split_key in ["val", "test"]:
                # For val and test, use ONLY the template pool derived from the training set
                # This ensures proper time-split validation where test data templates come only from training data
                if template_pool_file_path:
                    current_template_pool_file = template_pool_file_path
                else:
                    log.warning(
                        f"Cannot run {split_key} evaluation as training set template pool file is not available."
                    )
                    all_benchmark_results["results_data"][f"custom_{split_key}"] = {
                        "results": {},
                        "name": f"custom_{split_key}",
                        "status": "skipped_no_template_pool",
                    }
                    all_benchmark_results["metrics_summary"][f"custom_{split_key}"] = {
                        "total": 0,
                        "successful": 0,
                        "status": "skipped_no_template_pool",
                    }
                    continue

            split_run_data = evaluate_split(
                split_key,
                splits_data[split_key]["filtered"],
                current_template_pool_file,  # Pass the path to the train set PDB ID list file
                args.n_workers,
                args.n_conformers,
                args.template_knn,
                args.similarity_threshold,
                args.pipeline_timeout,
                args.multiprocessing,
                args.max_pdbs,
                args.log_level,  # Pass the configured log level
                args.use_embedding_cache,
                args.embedding_cache_dir,  # Pass embedding cache settings
                args.batch_embedding,
                args.max_batch_size,  # Pass batch embedding settings
                args.ca_rmsd_threshold,  # Pass the CA RMSD threshold parameter
                BENCHMARK_TIMESTAMP,  # Pass the global benchmark timestamp
            )
            all_benchmark_results["results_data"][
                f"custom_{split_key}"
            ] = split_run_data
            metrics = calculate_metrics(split_run_data)
            all_benchmark_results["metrics_summary"][f"custom_{split_key}"] = metrics

            # Generate plots in results directory like polaris
            plot_rmsd_distribution(
                split_run_data,
                f"custom_{split_key}",
                RESULTS_DIR_CUSTOM,
                args.histogram_bins,
            )
            plot_ca_rmsd_distribution(
                split_run_data,
                f"custom_{split_key}",
                RESULTS_DIR_CUSTOM,
                args.histogram_bins,
            )
            plot_ca_rmsd_vs_ligand_rmsd(
                split_run_data, f"custom_{split_key}", RESULTS_DIR_CUSTOM
            )

            # Analyze protein alignment tracking logs for missing CA RMSD investigation
            print(
                f"\n=== Analyzing Protein Alignment Patterns for {split_key.upper()} ==="
            )
            alignment_analysis = analyze_alignment_tracking_logs(split_run_data)

            if alignment_analysis["total_pdbs_processed"] > 0:
                print(
                    f"Alignment tracking coverage: {alignment_analysis['pdbs_with_alignment_logs']}/{alignment_analysis['total_pdbs_processed']} PDbs"
                )

                if alignment_analysis["missing_ca_rmsd_patterns"]:
                    print(
                        f"⚠️  Found {len(alignment_analysis['missing_ca_rmsd_patterns'])} PDbs with missing CA RMSD values"
                    )

                if alignment_analysis["validation_issues"]:
                    print(
                        f"⚠️  {len(alignment_analysis['validation_issues'])} validation issues detected"
                    )

                # Save detailed analysis report
                save_alignment_analysis_report(
                    alignment_analysis, split_key, RESULTS_DIR_CUSTOM
                )
            else:
                print(
                    "No alignment tracking data found - consider running with enhanced tracking enabled"
                )

    # Save overall results like polaris
    overall_results_file = os.path.join(
        RESULTS_DIR_CUSTOM, f"custom_benchmark_results_{BENCHMARK_TIMESTAMP}.json"
    )
    with open(overall_results_file, "w") as f:
        json.dump(all_benchmark_results, f, indent=2, cls=NpEncoder)
    log.info(f"Overall custom benchmark results saved to {overall_results_file}")

    # Save comparison table to results directory
    save_comparison_table(all_benchmark_results["metrics_summary"], RESULTS_DIR_CUSTOM)

    # Clean up the temporary template pool file
    if template_pool_file_path and os.path.exists(template_pool_file_path):
        try:
            os.remove(template_pool_file_path)
            log.info(f"Removed temporary template pool file: {template_pool_file_path}")
        except OSError as e:
            log.warning(
                f"Error removing temporary template pool file {template_pool_file_path}: {e}"
            )

    log.info(
        "Benchmarking complete. Val and test splits were properly restricted to using only training set templates for KNN search."
    )

    # Print final stats about embedding cache if enabled
    if args.use_embedding_cache:
        try:
            cache_dir = args.embedding_cache_dir or os.path.expanduser(
                "~/.cache/templ/embeddings/esm2_t33_650M_UR50D"
            )
            if os.path.exists(cache_dir):
                files = [f for f in os.listdir(cache_dir) if f.endswith(".npz")]
                total_size = sum(
                    os.path.getsize(os.path.join(cache_dir, f)) for f in files
                )
                log.info(
                    f"Final embedding cache stats: {len(files)} entries ({total_size / (1024 * 1024):.2f} MB)"
                )
        except Exception as e:
            log.warning(f"Error getting final cache stats: {str(e)}")

    log.info("Custom benchmark completed.")

    # Print final summary of output structure
    log.info(f"\n=== OUTPUT STRUCTURE ===")
    log.info(f"Output directory: {OUTPUT_DIR_CUSTOM}")
    log.info(f"Results directory: {RESULTS_DIR_CUSTOM}")

    # List experiment-specific directories that were created
    for split_key in ["val", "test"]:
        if f"custom_{split_key}" in all_benchmark_results["results_data"]:
            split_data = all_benchmark_results["results_data"][f"custom_{split_key}"]
            if "experiment_output_dir" in split_data:
                log.info(
                    f"  {split_key.capitalize()} poses: {split_data['experiment_output_dir']}"
                )

    log.info("=== END OUTPUT STRUCTURE ===\n")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)  # Convert sets to lists
        return super(NpEncoder, self).default(obj)


if __name__ == "__main__":
    main()
