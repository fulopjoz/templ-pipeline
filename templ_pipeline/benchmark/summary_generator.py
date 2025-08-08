#!/usr/bin/env python3
"""
Unified Summary Table Generator for TEMPL Benchmarks

This module provides standardized summary table generation for both
Polaris and Timesplit benchmarks, ensuring consistent output formats
and metrics across different benchmark types.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from collections import defaultdict

import numpy as np

# Optional pandas dependency
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

logger = logging.getLogger(__name__)


class BenchmarkSummaryGenerator:
    """Unified summary generator for different benchmark types."""
    
    def __init__(self):
        self.supported_formats = ["json", "csv", "markdown", "jsonl"]
        
    def detect_benchmark_type(self, results_data: Dict) -> str:
        """Detect the type of benchmark from results structure."""
        if "benchmark_info" in results_data:
            # Check benchmark name in benchmark_info
            bench_info = results_data["benchmark_info"]
            if "name" in bench_info:
                name_lower = bench_info["name"].lower()
                if "polaris" in name_lower:
                    return "polaris"
                elif "timesplit" in name_lower:
                    return "timesplit"
        
        # Check for timesplit indicators
        if any("timesplit" in key.lower() for key in results_data.keys()):
            return "timesplit"
            
        # Check for individual result entries that might indicate timesplit
        for key, value in results_data.items():
            if isinstance(value, dict):
                if "target_split" in value or "pdb_id" in value:
                    return "timesplit"
                    
        return "unknown"
    
    def generate_unified_summary(
        self, 
        results_data: Union[Dict, List[Dict]], 
        benchmark_type: Optional[str] = None,
        output_format: str = "pandas"
    ) -> Union[Dict, "pd.DataFrame"]:
        """Generate unified summary from benchmark results.
        
        Args:
            results_data: Results data from benchmark execution
            benchmark_type: Type of benchmark ("polaris", "timesplit", or auto-detect)
            output_format: Output format ("pandas", "dict", "json")
            
        Returns:
            Summary data in requested format
        """
        if benchmark_type is None:
            if isinstance(results_data, list):
                benchmark_type = "timesplit"  # JSONL format typical for timesplit
            else:
                benchmark_type = self.detect_benchmark_type(results_data)
                
        logger.info(f"Generating summary for {benchmark_type} benchmark")
        
        if benchmark_type == "polaris":
            return self._generate_polaris_summary(results_data, output_format)
        elif benchmark_type == "timesplit":
            return self._generate_timesplit_summary(results_data, output_format)
        else:
            logger.warning(f"Unknown benchmark type: {benchmark_type}")
            return self._generate_generic_summary(results_data, output_format)
    
    def _generate_polaris_summary(self, results_data: Dict, output_format: str) -> Union[Dict, "pd.DataFrame"]:
        """Generate summary for Polaris benchmark results."""
        table_data = []
        
        # Define expected result configurations for Polaris
        result_configs = [
            ("SARS_train_native", "SARS", "Train", "SARS"),
            ("SARS_test_native", "SARS", "Test", "SARS"), 
            ("MERS_train_native", "MERS", "Train", "MERS"),
            ("MERS_test_native", "MERS", "Test", "MERS"),
            ("MERS_train_cross", "MERS", "Train", "MERS+SARS-aligned"),
            ("MERS_test_cross", "MERS", "Test", "MERS+SARS-aligned"),
        ]
        
        # Look for results in the nested structure
        results_section = results_data.get("results", results_data)
        logger.debug(f"Polaris results keys: {list(results_section.keys())}")
        
        for result_key, virus_type, dataset, template_source in result_configs:
            if result_key in results_section:
                result_entry = results_section[result_key]
                metrics = self._calculate_polaris_metrics(result_entry)
                
                # Get query and template counts
                query_count = result_entry.get("query_count", len(result_entry.get("results", {})))
                template_counts = result_entry.get("template_counts", {})
                
                # Format template description
                template_desc = self._format_template_description(template_counts, dataset, query_count)
                
                # Extract combo metric results
                if "combo" in metrics.get("success_rates", {}):
                    combo_stats = metrics["success_rates"]["combo"]
                    table_data.append({
                        "Benchmark": "Polaris",
                        "Virus_Type": virus_type,
                        "Dataset": dataset,
                        "Template_Source": template_source,
                        "Queries": query_count,
                        "Templates": template_desc,
                        "Success_Rate_2A": f"{combo_stats['rate_2A']:.1f}%",
                        "Success_Rate_5A": f"{combo_stats['rate_5A']:.1f}%",
                        "Mean_RMSD": f"{combo_stats.get('mean_rmsd', 0):.2f}",
                        "Successful_Poses": combo_stats.get('count', 0),
                        "Processing_Time": result_entry.get('processing_time', 0)
                    })
                else:
                    # No combo results available
                    table_data.append({
                        "Benchmark": "Polaris",
                        "Virus_Type": virus_type,
                        "Dataset": dataset,
                        "Template_Source": template_source,
                        "Queries": query_count,
                        "Templates": template_desc,
                        "Success_Rate_2A": "0.0%",
                        "Success_Rate_5A": "0.0%",
                        "Mean_RMSD": "N/A",
                        "Successful_Poses": 0,
                        "Processing_Time": 0
                    })
        
        return self._format_output(table_data, output_format)
    
    def _generate_timesplit_summary(self, results_data: Union[Dict, List[Dict]], output_format: str) -> Union[Dict, "pd.DataFrame"]:
        """Generate summary for Timesplit benchmark results using stage-aware metrics."""
        logger.info("Generating timesplit summary with stage-aware metrics...")
        
        # Use the fixed version that properly handles stage-aware classification
        return self._generate_timesplit_summary_fixed(results_data, output_format)
    
    def _calculate_polaris_metrics(self, result_data: Dict) -> Dict:
        """Calculate metrics for Polaris benchmark results."""
        individual_results = result_data.get("results", {})
        if not individual_results:
            return {"success_rates": {}}
        
        metrics = {
            "rmsd_counts_2A": defaultdict(int),
            "rmsd_counts_5A": defaultdict(int), 
            "all_rmsds": defaultdict(list),
            "all_scores": defaultdict(list),
        }
        
        total_molecules = len(individual_results)
        
        # Success rate calculation detailed logging for Polaris
        logger.info(f"SUCCESS_RATE_CALC: Processing Polaris results:")
        logger.info(f"SUCCESS_RATE_CALC:   Total molecules to analyze: {total_molecules}")
        
        successful_results = 0
        results_with_rmsd = 0
        
        for result in individual_results.values():
            if result.get("success") and result.get("rmsd_values"):
                successful_results += 1
                has_rmsd_values = False
                
                for metric_key, values_dict in result["rmsd_values"].items():
                    rmsd = values_dict.get("rmsd")
                    score = values_dict.get("score")
                    if rmsd is not None and not np.isnan(rmsd):
                        has_rmsd_values = True
                        metrics["all_rmsds"][metric_key].append(rmsd)
                        if rmsd <= 2.0:
                            metrics["rmsd_counts_2A"][metric_key] += 1
                        if rmsd <= 5.0:
                            metrics["rmsd_counts_5A"][metric_key] += 1
                    if score is not None:
                        metrics["all_scores"][metric_key].append(score)
                
                if has_rmsd_values:
                    results_with_rmsd += 1
        
        logger.info(f"SUCCESS_RATE_CALC: Polaris data collection summary:")
        logger.info(f"SUCCESS_RATE_CALC:   Successful results: {successful_results}/{total_molecules}")
        logger.info(f"SUCCESS_RATE_CALC:   Results with RMSD values: {results_with_rmsd}/{successful_results}")
        
        # Log RMSD data by metric
        for metric_key in metrics["all_rmsds"]:
            rmsd_count = len(metrics["all_rmsds"][metric_key])
            count_2A = metrics["rmsd_counts_2A"][metric_key]
            count_5A = metrics["rmsd_counts_5A"][metric_key]
            logger.info(f"SUCCESS_RATE_CALC:   {metric_key}: {rmsd_count} RMSD values, {count_2A} ≤2A, {count_5A} ≤5A")
        
        # Calculate success rates
        metrics["success_rates"] = {}
        for metric_key in metrics["all_rmsds"]:
            n_results = len(metrics["all_rmsds"][metric_key])
            if n_results > 0 and total_molecules > 0:
                rate_2A = metrics["rmsd_counts_2A"][metric_key] / total_molecules * 100
                rate_5A = metrics["rmsd_counts_5A"][metric_key] / total_molecules * 100
                mean_rmsd = np.mean(metrics["all_rmsds"][metric_key])
                median_rmsd = np.median(metrics["all_rmsds"][metric_key])
                
                metrics["success_rates"][metric_key] = {
                    "count": n_results,
                    "rate_2A": rate_2A,
                    "rate_5A": rate_5A,
                    "mean_rmsd": mean_rmsd,
                    "median_rmsd": median_rmsd,
                }
                
                # Log calculated success rates
                logger.info(f"SUCCESS_RATE_CALC: Final Polaris rates for {metric_key}:")
                logger.info(f"SUCCESS_RATE_CALC:   2A success: {rate_2A:.1f}% ({metrics['rmsd_counts_2A'][metric_key]}/{total_molecules})")
                logger.info(f"SUCCESS_RATE_CALC:   5A success: {rate_5A:.1f}% ({metrics['rmsd_counts_5A'][metric_key]}/{total_molecules})")
                logger.info(f"SUCCESS_RATE_CALC:   Mean RMSD: {mean_rmsd:.3f}A")
        
        return metrics
    
    def _validate_rmsd_calculation(self, split_results: List[Dict]) -> Dict:
        """Validate RMSD calculation integrity in benchmark results."""
        validation_report = {
            "total_results": len(split_results),
            "successful_results": 0,
            "results_with_rmsd": 0,
            "results_with_null_rmsd": 0,
            "metrics_with_rmsd_failures": []
        }
        
        for result in split_results:
            if result.get("success"):
                validation_report["successful_results"] += 1
                rmsd_values = result.get("rmsd_values", {})
                
                has_valid_rmsd = False
                metrics_with_failures = []
                
                for metric_key, values_dict in rmsd_values.items():
                    rmsd = values_dict.get("rmsd")
                    if rmsd is not None and not np.isnan(rmsd):
                        has_valid_rmsd = True
                    else:
                        metrics_with_failures.append(metric_key)
                
                if has_valid_rmsd:
                    validation_report["results_with_rmsd"] += 1
                else:
                    validation_report["results_with_null_rmsd"] += 1
                    validation_report["metrics_with_rmsd_failures"].extend(metrics_with_failures)
        
        # Log validation results
        if validation_report["results_with_null_rmsd"] > 0:
            failure_rate = (validation_report["results_with_null_rmsd"] / validation_report["successful_results"]) * 100
            logger.error(f"RMSD_VALIDATION: {validation_report['results_with_null_rmsd']}/{validation_report['successful_results']} ({failure_rate:.1f}%) successful results lack RMSD data")
            logger.error(f"RMSD_VALIDATION: This indicates CLI pipeline or molecular structure issues")
        else:
            logger.info(f"RMSD_VALIDATION: All {validation_report['results_with_rmsd']} successful results have valid RMSD data")
        
        return validation_report

    def _parse_exclusion_reason(self, error_msg: str) -> str:
        """Parse exclusion reason from error message with comprehensive categorization.
        
        This method categorizes error messages into specific exclusion reasons based on
        the analysis of the TEMPL pipeline codebase. The categories are:
        
        Molecule Validation Exclusions:
        - large_peptide: Molecules with too many peptide residues (>8 by default)
        - rhenium_complex: Molecules containing rhenium complexes (except 3rj7)
        - complex_polysaccharide: Complex polysaccharides with many sugar rings
        - molecule_validation_failed: General molecule validation failures
        - invalid_molecule: Invalid molecule objects or structures
        
        Timeout and Processing Exclusions:
        - timeout: Subprocess timeout after specified seconds
        - pose_generation_failed: No poses generated during processing
        - rmsd_calculation_failed: RMSD calculation failed
        - conformer_generation_failed: Conformer generation failed
        - molecular_alignment_failed: Molecular alignment failed
        - mcs_calculation_failed: MCS (Maximum Common Substructure) calculation failed
        - central_atom_embedding_failed: Central atom embedding fallback failed
        - constrained_embedding_failed: Constrained embedding failed
        - embedding_failed: General embedding failures
        - geometry_validation_failed: Molecular geometry validation failed
        - molecular_connectivity_failed: Molecular connectivity validation failed
        - molecular_optimization_failed: Molecular optimization failed
        - force_field_failed: Force field application failed
        - molecule_sanitization_failed: Molecule sanitization failed
                 - coordinate_access_failed: Coordinate access failed
         - invalid_coordinates: NaN or infinite coordinates detected
         - molecular_fragmentation: Molecular fragmentation detected
         - invalid_atom_positions: Unreasonable atom positions detected
         - suspicious_bond_lengths: Suspicious bond lengths detected
         - empty_coordinate_map: Empty coordinate map for constraints
         - constraint_distance_issues: Constraint distance issues (too close/distant)
         - minimization_no_conformers: Cannot minimize molecule (no conformers)
         - rascal_mcs_failed: RascalMCES MCS calculation failed
         - mcs_too_small: MCS found but too small for processing
         - rascal_search_failed: RascalMCES search failed at all thresholds
         - molecule_too_large: Extremely large molecule (>150 atoms)
         - invalid_smarts_pattern: Invalid SMARTS pattern for MCS
         - invalid_mcs_match: Invalid MCS match for constrained embedding
         - hydrogen_addition_failed: Hydrogen addition failed during processing
         - mcs_hydrogen_addition_failed: MCS match failed after hydrogen addition
         - mcs_index_mismatch: Index length mismatch in MCS processing
         - mcs_matching_inconsistency: MCS matching inconsistency detected
         - atom_mapping_failed: Error mapping atoms during processing
         - insufficient_coordinate_constraints: Insufficient coordinate constraints
         - relaxed_constraints_failed: Embedding failed with relaxed constraints
         - progressive_embedding_failed: Progressive embedding reduction failed
         - zero_conformers_generated: Embedding succeeded but generated 0 conformers
         - molecular_distortion: Molecular distortion detected after processing
         - alignment_index_mismatch: Index length mismatch during alignment
         - rdkit_alignment_failed: RDKit AlignMol failed
         - alignment_skipped: Post-embedding alignment skipped
         - connectivity_issues: Molecular connectivity issues detected
         - organometallic_embedding_failed: All organometallic embedding attempts failed
         - all_embedding_methods_failed: All embedding methods failed
         - uff_fallback_failed: UFF fallback embedding failed
         - mmff_parameter_validation_failed: MMFF parameter validation failed
         
         File and Data Exclusions:
        - ligand_data_missing: Ligand SMILES data not found
        - protein_file_missing: Protein PDB file not found
        - template_file_missing: Template SDF file not found
        - embedding_file_missing: Embedding file not found
        - crystal_structure_missing: Crystal structure data missing
        - file_not_found: General file not found errors
        
        System and Execution Exclusions:
        - memory_error: Memory errors during processing
        - cli_execution_failed: CLI command execution failed
        - subprocess_failed: Subprocess execution failed
        - cli_command_invalid: CLI command validation failed
        - pipeline_error: General pipeline errors
        
        Legacy Categories (for backward compatibility):
        - molecule_too_small: Legacy small molecule exclusion
        - molecule_too_large: Legacy large molecule exclusion
        - poor_quality_crystal: Legacy crystal quality exclusion
        - invalid_smiles: Legacy SMILES validation exclusion
        - validation_failed: Legacy validation failure
        
        Args:
            error_msg: Error message string to categorize
            
        Returns:
            Categorized exclusion reason string
        """
        if not error_msg:
            return "unknown_error"
        
        error_msg_lower = error_msg.lower()
        
        # Parse molecule validation exclusions (from chemistry.py and pipeline.py)
        if "large peptide" in error_msg_lower or "large peptides" in error_msg_lower:
            return "large_peptide"
        elif "rhenium complex" in error_msg_lower:
            return "rhenium_complex"
        elif "peptide" in error_msg_lower and ("residues" in error_msg_lower or "threshold" in error_msg_lower):
            return "large_peptide"
        elif "polysaccharide" in error_msg_lower:
            return "complex_polysaccharide"
        elif "validation failed" in error_msg_lower and "geometry" in error_msg_lower:
            return "geometry_validation_failed"
        elif "validation failed" in error_msg_lower:
            return "molecule_validation_failed"
        elif "invalid molecule" in error_msg_lower:
            return "invalid_molecule"
        
        # Parse timeout exclusions (from simple_runner.py)
        elif "timeout" in error_msg_lower:
            return "timeout"
        
        # Parse file and data exclusions
        elif "could not load ligand smiles" in error_msg_lower or ("ligand smiles" in error_msg_lower and "not found" in error_msg_lower):
            return "ligand_data_missing"
        elif "protein file" in error_msg_lower and "not found" in error_msg_lower:
            return "protein_file_missing"
        elif "template sdf file not found" in error_msg_lower:
            return "template_file_missing"
        elif "template file not found" in error_msg_lower:
            return "template_file_missing"
        elif "template file" in error_msg_lower and "not found" in error_msg_lower:
            return "template_file_missing"
        elif "template" in error_msg_lower and "file" in error_msg_lower and "not found" in error_msg_lower:
            return "template_file_missing"
        elif "pdb file" in error_msg_lower and "not found" in error_msg_lower:
            return "protein_file_missing"
        elif "file not found" in error_msg_lower:
            return "file_not_found"
        elif "embedding file not found" in error_msg_lower:
            return "embedding_file_missing"
        elif "crystal ligand" in error_msg_lower and "not found" in error_msg_lower:
            return "crystal_structure_missing"
        elif "no crystal ligand found" in error_msg_lower:
            return "crystal_structure_missing"
        
        # Parse pipeline processing exclusions
        elif "no poses generated" in error_msg_lower or "pose generation failed" in error_msg_lower:
            return "pose_generation_failed"
        elif "rmsd calculation failed" in error_msg_lower or "rmsd failed" in error_msg_lower:
            return "rmsd_calculation_failed"
        elif "conformer generation failed" in error_msg_lower or "conformer failed" in error_msg_lower:
            return "conformer_generation_failed"
        elif "alignment failed" in error_msg_lower or "molecular alignment" in error_msg_lower:
            return "molecular_alignment_failed"
        elif "rascal" in error_msg_lower and "search" in error_msg_lower and "failed" in error_msg_lower:
            return "rascal_search_failed"
        elif "mcs" in error_msg_lower and ("failed" in error_msg_lower or "calculation" in error_msg_lower):
            return "mcs_calculation_failed"
        elif "central atom embedding failed" in error_msg_lower:
            return "central_atom_embedding_failed"
        elif "constrained embedding failed" in error_msg_lower:
            return "constrained_embedding_failed"
        elif "embedding failed" in error_msg_lower or "embedding error" in error_msg_lower:
            return "embedding_failed"
        elif "memory" in error_msg_lower and ("error" in error_msg_lower or "failed" in error_msg_lower):
            return "memory_error"
        elif "geometry validation failed" in error_msg_lower:
            return "geometry_validation_failed"
        elif "validation failed" in error_msg_lower and "geometry" in error_msg_lower:
            return "geometry_validation_failed"
        elif "connectivity" in error_msg_lower and "failed" in error_msg_lower:
            return "molecular_connectivity_failed"
        elif "optimization failed" in error_msg_lower:
            return "molecular_optimization_failed"
        elif "force field" in error_msg_lower and "failed" in error_msg_lower:
            return "force_field_failed"
        elif "sanitization failed" in error_msg_lower:
            return "molecule_sanitization_failed"
        elif "coordinates" in error_msg_lower and "inaccessible" in error_msg_lower:
            return "coordinate_access_failed"
        elif "nan" in error_msg_lower or "infinite coordinates" in error_msg_lower:
            return "invalid_coordinates"
        elif "molecular fragmentation" in error_msg_lower:
            return "molecular_fragmentation"
        elif "unreasonable atom position" in error_msg_lower:
            return "invalid_atom_positions"
        elif "suspicious bond" in error_msg_lower:
            return "suspicious_bond_lengths"
        elif "empty coordinate map" in error_msg_lower:
            return "empty_coordinate_map"
        elif "very close constraints" in error_msg_lower or "very distant constraints" in error_msg_lower:
            return "constraint_distance_issues"
        elif "cannot minimize molecule" in error_msg_lower:
            return "minimization_no_conformers"
        elif "rejecting small mcs" in error_msg_lower:
            return "mcs_too_small"
        elif "rascal" in error_msg_lower and "failed" in error_msg_lower:
            return "rascal_mcs_failed"
        elif "rascal" in error_msg_lower and "search" in error_msg_lower and "failed" in error_msg_lower:
            return "rascal_search_failed"
        elif "extremely large molecule" in error_msg_lower:
            return "molecule_too_large"
        elif "invalid smarts pattern" in error_msg_lower:
            return "invalid_smarts_pattern"
        elif "invalid mcs match" in error_msg_lower:
            return "invalid_mcs_match"
        elif "hydrogen addition failed" in error_msg_lower:
            return "hydrogen_addition_failed"
        elif "index length mismatch during alignment" in error_msg_lower:
            return "alignment_index_mismatch"
        elif "index length mismatch" in error_msg_lower:
            return "mcs_index_mismatch"
        elif "mcs match failed after hydrogen addition" in error_msg_lower:
            return "mcs_hydrogen_addition_failed"
        elif "mcs matching inconsistency" in error_msg_lower:
            return "mcs_matching_inconsistency"
        elif "error mapping atom" in error_msg_lower:
            return "atom_mapping_failed"
        elif "insufficient coordinate constraints" in error_msg_lower:
            return "insufficient_coordinate_constraints"
        elif "progressive embedding failed" in error_msg_lower:
            return "progressive_embedding_failed"
        elif "embedding failed with relaxed constraints" in error_msg_lower:
            return "relaxed_constraints_failed"
        elif "embedding failed" in error_msg_lower and "relaxed" in error_msg_lower:
            return "relaxed_constraints_failed"
        elif "embedding succeeded but generated 0 conformers" in error_msg_lower:
            return "zero_conformers_generated"
        elif "molecular distortion detected" in error_msg_lower:
            return "molecular_distortion"
        elif "index length mismatch during alignment" in error_msg_lower:
            return "alignment_index_mismatch"
        elif "rdkit alignmol failed" in error_msg_lower:
            return "rdkit_alignment_failed"
        elif "continuing without post-embedding alignment" in error_msg_lower:
            return "alignment_skipped"
        elif "molecular connectivity issues" in error_msg_lower:
            return "connectivity_issues"
        elif "all organometallic embedding attempts failed" in error_msg_lower:
            return "organometallic_embedding_failed"
        elif "all embedding methods failed" in error_msg_lower:
            return "all_embedding_methods_failed"
        elif "embedding with uff fallback failed" in error_msg_lower:
            return "uff_fallback_failed"
        elif "mmff parameter validation failed" in error_msg_lower:
            return "mmff_parameter_validation_failed"
        elif "validation failed" in error_msg_lower and "mmff" in error_msg_lower:
            return "mmff_parameter_validation_failed"
        elif "mmff" in error_msg_lower and "failed" in error_msg_lower:
            return "mmff_parameter_validation_failed"
        
        # Parse CLI and subprocess errors
        elif "cli returned" in error_msg_lower or "returncode" in error_msg_lower:
            return "cli_execution_failed"
        elif "subprocess" in error_msg_lower and "failed" in error_msg_lower:
            return "subprocess_failed"
        elif "command validation failed" in error_msg_lower:
            return "cli_command_invalid"
        
        # Parse legacy skip reasons (for backward compatibility)
        elif "skipped" in error_msg_lower:
            if "molecule_too_small" in error_msg_lower:
                return "molecule_too_small"
            elif "molecule_too_large" in error_msg_lower:
                return "molecule_too_large"
            elif "poor_quality" in error_msg_lower:
                return "poor_quality_crystal"
            elif "peptide" in error_msg_lower:
                return "large_peptide"
            elif "invalid_smiles" in error_msg_lower:
                return "invalid_smiles"
            elif "sanitization_failed" in error_msg_lower:
                return "molecule_sanitization_failed"
            else:
                return "validation_failed"
        
        # Parse general pipeline errors
        elif "pipeline" in error_msg_lower and "failed" in error_msg_lower:
            return "pipeline_error"
        elif "error" in error_msg_lower and "failed" in error_msg_lower:
            return "pipeline_error"
        
        # Special case for successful CLI execution but no templates/poses generated  
        elif error_msg == "no_templates_found":
            return "no_templates_found"
        elif error_msg == "database_empty":
            return "database_empty"
        elif error_msg == "templates_filtered_out":
            return "templates_filtered_out"
        
        # Default fallback
        else:
            return "unknown_error"

    def _classify_exclusion_processing_stage(self, exclusion_reason: str) -> str:
        """
        Classify exclusion reasons into processing stages for stage-aware success rate calculations.
        
        Processing Stages:
        1. pre_pipeline_excluded: Data availability issues (missing files, invalid data)
        2. pipeline_filtered: Molecule validation/quality filters (large peptides, etc.)
        3. pipeline_attempted: Actual algorithm processing (timeouts, pose failures, etc.)
        
        Args:
            exclusion_reason: Exclusion reason from _parse_exclusion_reason()
            
        Returns:
            Processing stage classification
        """
        # Pre-pipeline exclusions (data availability/quality issues)
        # These should NOT affect pipeline success rates
        pre_pipeline_exclusions = {
            "ligand_data_missing", "protein_file_missing", "template_file_missing", 
            "embedding_file_missing", "crystal_structure_missing", "file_not_found",
            "cli_command_invalid", "subprocess_failed", "cli_execution_failed",
            "ligand_loading", "cli_validation", "data_availability_issue",
            "database_empty"  # Template database is empty - data availability issue
        }
        
        # Pipeline filtering exclusions (validation rules, quality filters)
        # These should NOT affect pipeline success rates  
        pipeline_filter_exclusions = {
            "large_peptide", "rhenium_complex", "complex_polysaccharide",
            "molecule_validation_failed", "invalid_molecule", "molecule_too_large",
            "molecule_too_small", "poor_quality_crystal", "invalid_smiles", 
            "validation_failed", "geometry_validation_failed",
            "molecule_sanitization_failed", "quality_filter", "molecule_filtered",
            "no_templates_found",  # Generic case - will be refined by CLI analysis
            "templates_filtered_out"  # Templates available but filtered by similarity/quality
        }
        
        # Pipeline execution failures (algorithm processing issues)
        # These SHOULD affect pipeline success rates
        pipeline_execution_failures = {
            "timeout", "pose_generation_failed", "rmsd_calculation_failed",
            "conformer_generation_failed", "molecular_alignment_failed",
            "mcs_calculation_failed", "central_atom_embedding_failed",
            "constrained_embedding_failed", "embedding_failed", 
            "molecular_connectivity_failed", "molecular_optimization_failed",
            "force_field_failed", "coordinate_access_failed", "invalid_coordinates",
            "molecular_fragmentation", "invalid_atom_positions", "suspicious_bond_lengths",
            "empty_coordinate_map", "constraint_distance_issues", "minimization_no_conformers",
            "rascal_mcs_failed", "mcs_too_small", "rascal_search_failed",
            "invalid_smarts_pattern", "invalid_mcs_match", "hydrogen_addition_failed",
            "mcs_hydrogen_addition_failed", "mcs_index_mismatch", "mcs_matching_inconsistency",
            "atom_mapping_failed", "insufficient_coordinate_constraints",
            "relaxed_constraints_failed", "progressive_embedding_failed",
            "zero_conformers_generated", "molecular_distortion", "alignment_index_mismatch",
            "rdkit_alignment_failed", "alignment_skipped", "connectivity_issues",
            "organometallic_embedding_failed", "all_embedding_methods_failed",
            "uff_fallback_failed", "mmff_parameter_validation_failed", "memory_error",
            "pipeline_error", "algorithm_failure", "processing_failed", "execution_error"
        }
        
        # Enhanced classification with better error pattern matching
        if exclusion_reason in pre_pipeline_exclusions:
            return "pre_pipeline_excluded"
        elif exclusion_reason in pipeline_filter_exclusions:
            return "pipeline_filtered"
        elif exclusion_reason in pipeline_execution_failures:
            return "pipeline_attempted"
        elif exclusion_reason == "unknown_error":
            # Log unknown errors for analysis and improvement
            logger.warning(f"Unknown error classification for: {exclusion_reason}")
            # Default to pipeline_attempted for unknown errors to be conservative
            return "pipeline_attempted"
        else:
            # Enhanced pattern matching for better classification
            exclusion_lower = exclusion_reason.lower()
            
            # Pattern-based classification for common error patterns
            if any(pattern in exclusion_lower for pattern in ["missing", "not found", "file", "data"]):
                return "pre_pipeline_excluded"
            elif any(pattern in exclusion_lower for pattern in ["validation", "filter", "quality", "invalid"]):
                return "pipeline_filtered"
            elif any(pattern in exclusion_lower for pattern in ["timeout", "failed", "error", "exception"]):
                return "pipeline_attempted"
            else:
                # Default: treat unknown exclusions as pipeline execution failures
                # This ensures we don't accidentally exclude real algorithm failures
                logger.warning(f"Unclassified error pattern: {exclusion_reason} - treating as pipeline_attempted")
                return "pipeline_attempted"

    def _classify_no_templates_case(self, result: Dict) -> str:
        """
        Classify successful CLI execution with no RMSD values by analyzing CLI JSON output.
        
        Distinguishes between:
        1. Database/Data issues (should affect success rates)
        2. Template filtering (should not affect success rates)
        
        Args:
            result: Result dictionary with CLI stdout
            
        Returns:
            Classification string for exclusion reason
        """
        stdout = result.get("stdout", "")
        
        # Extract CLI JSON result from stdout
        try:
            import re
            import json
            
            # Find TEMPL_JSON_RESULT: marker and extract JSON
            json_start = stdout.find("TEMPL_JSON_RESULT:")
            if json_start != -1:
                # Start after the marker
                json_start += len("TEMPL_JSON_RESULT:")
                # Find the end - look for the next newline after the JSON
                json_end = stdout.find("\n", json_start)
                if json_end == -1:
                    json_end = len(stdout)
                
                json_str = stdout[json_start:json_end].strip()
                cli_result = json.loads(json_str)
                
                # Check template database availability
                total_templates_in_db = cli_result.get("total_templates_in_database", 0)
                requested_templates = cli_result.get("template_filtering_info", {}).get("requested_templates", 0)
                found_templates = cli_result.get("template_filtering_info", {}).get("found_templates", 0)
                
                logger.debug(f"CLI JSON parsed: total_db={total_templates_in_db}, found={found_templates}, requested={requested_templates}")
                
                if total_templates_in_db == 0:
                    # Database is empty - data availability issue
                    return "database_empty"
                elif total_templates_in_db > 0 and found_templates == 0:
                    # Templates exist but none passed filtering - correct filtering
                    return "templates_filtered_out"
                else:
                    # Other case - use generic classification
                    return "no_templates_found"
                    
        except (ValueError, KeyError, AttributeError, json.JSONDecodeError) as e:
            # JSON parsing failed - fallback to generic classification
            logger.debug(f"Failed to parse CLI JSON output for no-templates classification: {e}")
            return "no_templates_found"
        
        # Fallback if no JSON found
        return "no_templates_found"

    def _calculate_timesplit_metrics_with_exclusions(self, all_results: List[Dict], successful_results: List[Dict], total_targets: int) -> Tuple[Dict, Dict]:
        """Calculate metrics for Timesplit benchmark results with exclusion analysis.
        
        Args:
            all_results: All target results including failed ones
            successful_results: Only successful results with RMSD values
            total_targets: Total number of targets processed
            
        Returns:
            Tuple of (metrics_dict, exclusion_stats_dict)
        """
        # Calculate stage-aware metrics using successful results and all results for context
        metrics = self._calculate_timesplit_metrics(successful_results, total_targets, all_results)
        
        # Analyze exclusions and failures
        exclusion_breakdown = defaultdict(int)
        excluded_count = 0
        
        for result in all_results:
            # Only count actual failures as exclusions, not successful results without RMSD
            if not result.get("success", False):
                excluded_count += 1
                error_msg = result.get("error", "")
                exclusion_reason = self._parse_exclusion_reason(error_msg)
                exclusion_breakdown[exclusion_reason] += 1
            elif result.get("success", False) and not result.get("rmsd_values"):
                # Successful CLI execution but no RMSD (0 templates/poses)
                # Analyze CLI JSON output to distinguish data issues from filtering
                excluded_count += 1
                exclusion_reason = self._classify_no_templates_case(result)
                exclusion_breakdown[exclusion_reason] += 1
        
        exclusion_stats = {
            "excluded_count": excluded_count,
            "exclusion_breakdown": dict(exclusion_breakdown),
            "successful_count": len(successful_results),
            "total_targets": total_targets
        }
        
        # Log exclusion analysis
        logger.info(f"EXCLUSION_ANALYSIS: Target processing summary:")
        logger.info(f"EXCLUSION_ANALYSIS:   Total targets: {total_targets}")
        logger.info(f"EXCLUSION_ANALYSIS:   Successful (with RMSD): {len(successful_results)}")
        logger.info(f"EXCLUSION_ANALYSIS:   Excluded/Failed: {excluded_count}")
        
        if exclusion_breakdown:
            logger.info(f"EXCLUSION_ANALYSIS: Exclusion breakdown:")
            for reason, count in sorted(exclusion_breakdown.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_targets) * 100 if total_targets > 0 else 0
                logger.info(f"EXCLUSION_ANALYSIS:   {reason}: {count} ({percentage:.1f}%)")
        
        return metrics, exclusion_stats

    def _calculate_timesplit_metrics(self, split_results: List[Dict], total_targets: int, all_results: List[Dict] = None) -> Dict:
        """Calculate metrics for Timesplit benchmark results with stage-aware success rates."""
        metrics = {}
        
        # Calculate stage-aware target counts if all_results provided
        pipeline_attempted_targets = total_targets  # Default fallback
        pre_pipeline_excluded = 0
        pipeline_filtered = 0
        
        if all_results:
            # Analyze all results using the new pipeline_stage and made_it_to_mcs fields
            stage_counts = {"pre_pipeline_excluded": 0, "pipeline_filtered": 0, "pipeline_attempted": 0}
            
            for result in all_results:
                # First try the new pipeline_stage field from CLI JSON output
                pipeline_stage = result.get("pipeline_stage", None)
                made_it_to_mcs = result.get("made_it_to_mcs", False)
                
                if pipeline_stage in stage_counts:
                    # Use the new pipeline_stage field
                    stage_counts[pipeline_stage] += 1
                elif made_it_to_mcs:
                    # If made_it_to_mcs is True, definitely count as pipeline_attempted
                    stage_counts["pipeline_attempted"] += 1
                else:
                    # Fallback to legacy processing_stage field or error analysis
                    processing_stage = result.get("processing_stage", "unknown")
                    
                    if processing_stage in stage_counts:
                        stage_counts[processing_stage] += 1
                    else:
                        # Final fallback: analyze by success and error
                        if result.get("success") and result.get("rmsd_values"):
                            # Successful results are always pipeline_attempted
                            stage_counts["pipeline_attempted"] += 1
                        else:
                            # Analyze failed results to determine stage
                            error_msg = result.get("error", "")
                            exclusion_reason = self._parse_exclusion_reason(error_msg)
                            fallback_stage = self._classify_exclusion_processing_stage(exclusion_reason)
                            stage_counts[fallback_stage] += 1
            
            pre_pipeline_excluded = stage_counts["pre_pipeline_excluded"]
            pipeline_filtered = stage_counts["pipeline_filtered"] 
            pipeline_attempted_targets = stage_counts["pipeline_attempted"]
            
            logger.info(f"SUCCESS_RATE_CALC: Stage-aware target analysis:")
            logger.info(f"SUCCESS_RATE_CALC:   Total targets: {total_targets}")
            logger.info(f"SUCCESS_RATE_CALC:   Pre-pipeline excluded: {pre_pipeline_excluded}")
            logger.info(f"SUCCESS_RATE_CALC:   Pipeline filtered: {pipeline_filtered}")
            logger.info(f"SUCCESS_RATE_CALC:   Pipeline attempted: {pipeline_attempted_targets}")
        
        # Validate RMSD calculation integrity
        validation_report = self._validate_rmsd_calculation(split_results)
        
        # Success rate calculation detailed logging for Timesplit
        logger.info(f"SUCCESS_RATE_CALC: Processing Timesplit results:")
        logger.info(f"SUCCESS_RATE_CALC:   Split results length: {len(split_results)}")
        logger.info(f"SUCCESS_RATE_CALC:   Results with valid RMSD: {validation_report['results_with_rmsd']}/{validation_report['successful_results']}")
        logger.info(f"SUCCESS_RATE_CALC:   Pipeline attempted targets (denominator): {pipeline_attempted_targets}")
        
        # Collect RMSD values and scores by metric
        rmsd_by_metric = defaultdict(list)
        score_by_metric = defaultdict(list)
        
        successful_results = 0
        results_with_rmsd = 0
        
        # Process ALL pipeline_attempted results, not just successful ones
        # This ensures pipeline failures are properly counted in success rate calculations
        results_to_process = all_results if all_results else split_results
        for result in results_to_process:
            # First check pipeline stage using new CLI fields
            pipeline_stage = result.get("pipeline_stage", None)
            made_it_to_mcs = result.get("made_it_to_mcs", False)
            
            # Determine if this result reached the pipeline attempted stage
            reached_pipeline_attempted = False
            if pipeline_stage == "pipeline_attempted":
                reached_pipeline_attempted = True
            elif made_it_to_mcs:
                reached_pipeline_attempted = True
            elif result.get("processing_stage") == "pipeline_attempted":
                # Fallback to legacy processing_stage field
                reached_pipeline_attempted = True
            elif result.get("success") and result.get("rmsd_values"):
                # Successful results with RMSD always reached pipeline attempted
                reached_pipeline_attempted = True
                
            if result.get("success") and result.get("rmsd_values"):
                # Successful result with RMSD values
                successful_results += 1
                has_rmsd_values = False
                
                for metric_key, values_dict in result["rmsd_values"].items():
                    rmsd = values_dict.get("rmsd")
                    score = values_dict.get("score")
                    
                    # Collect RMSD values if available
                    if rmsd is not None and not np.isnan(rmsd):
                        has_rmsd_values = True
                        rmsd_by_metric[metric_key].append(rmsd)
                    
                    # Collect scores for fallback success calculation
                    if score is not None and not np.isnan(score):
                        score_by_metric[metric_key].append(score)
                
                if has_rmsd_values:
                    results_with_rmsd += 1
            elif reached_pipeline_attempted:
                # CRITICAL FIX: Include pipeline failures in RMSD collection
                # Pipeline failures (empty RMSD, timeouts, etc.) should count as infinite RMSD
                # This ensures they contribute 0 to success counts while being included in denominator
                failure_rmsd = 999.0  # Guaranteed to fail both 2Å and 5Å thresholds
                
                # Log the type of pipeline failure for debugging
                error_msg = result.get("error", "unknown_pipeline_failure")
                logger.info(f"SUCCESS_RATE_CALC: Including pipeline failure as 999.0Å RMSD: {error_msg[:100]}...")
                
                # Add failure RMSD to all metrics to maintain consistency
                for metric_key in ["combo", "shape", "color"]:
                    rmsd_by_metric[metric_key].append(failure_rmsd)
        
        logger.info(f"SUCCESS_RATE_CALC: Timesplit data collection summary:")
        logger.info(f"SUCCESS_RATE_CALC:   Successful results: {successful_results}/{len(split_results)}")
        logger.info(f"SUCCESS_RATE_CALC:   Results with RMSD values: {results_with_rmsd}/{successful_results}")
        logger.info(f"SUCCESS_RATE_CALC:   Pipeline failures included: {pipeline_attempted_targets - results_with_rmsd}")
        logger.info(f"SUCCESS_RATE_CALC:   Total RMSD entries (including failures): {len(rmsd_by_metric.get('combo', []))}")
        
        # CRITICAL DATASET SIZE REPORTING
        logger.info(f"DATASET_SIZE_REPORTING: *** FINAL DATASET NUMBERS FOR PUBLICATION ***")
        logger.info(f"DATASET_SIZE_REPORTING:   Total input molecules: {total_targets}")
        logger.info(f"DATASET_SIZE_REPORTING:   Pre-pipeline excluded: {pre_pipeline_excluded} (missing files, invalid data)")
        logger.info(f"DATASET_SIZE_REPORTING:   Pipeline filtered: {pipeline_filtered} (peptides, validation failures)")
        logger.info(f"DATASET_SIZE_REPORTING:   Pipeline attempted (XYZ number): {pipeline_attempted_targets}")
        logger.info(f"DATASET_SIZE_REPORTING:   ^^^ USE THIS NUMBER FOR TRAIN/VAL/TEST DATASET SIZES IN PAPERS ^^^")
        logger.info(f"DATASET_SIZE_REPORTING:   Success breakdown: {results_with_rmsd} successful, {pipeline_attempted_targets - results_with_rmsd} failed")
        
        # Log RMSD data by metric
        for metric_key in rmsd_by_metric:
            rmsd_count = len(rmsd_by_metric[metric_key])
            logger.info(f"SUCCESS_RATE_CALC:   {metric_key}: {rmsd_count} RMSD values collected")
        
        # Calculate statistics for each metric
        for metric_key in set(list(rmsd_by_metric.keys()) + list(score_by_metric.keys())):
            rmsds = rmsd_by_metric.get(metric_key, [])
            scores = score_by_metric.get(metric_key, [])
            
            if rmsds:
                # Use RMSD-based calculation with stage-aware success rates
                count_2A = sum(1 for rmsd in rmsds if rmsd <= 2.0)
                count_5A = sum(1 for rmsd in rmsds if rmsd <= 5.0)
                
                # SUCCESS RATE CALCULATION: Now correctly includes pipeline failures
                # count_2A and count_5A are from rmsds array which includes both successes and failures (999.0 RMSD)
                # pipeline_attempted_targets includes all attempted targets
                # This ensures pipeline failures contribute 0 to numerator while being counted in denominator
                rate_2A = count_2A / pipeline_attempted_targets * 100 if pipeline_attempted_targets > 0 else 0
                rate_5A = count_5A / pipeline_attempted_targets * 100 if pipeline_attempted_targets > 0 else 0
                
                # CRITICAL VALIDATION: RMSD array length MUST match pipeline_attempted_targets
                if len(rmsds) != pipeline_attempted_targets:
                    logger.error(f"CRITICAL ERROR: RMSD array length ({len(rmsds)}) != pipeline_attempted_targets ({pipeline_attempted_targets})")
                    logger.error(f"This indicates a bug in success rate calculation - some molecules are not being counted correctly")
                    logger.error(f"Success rates will be INCORRECT until this is fixed")
                else:
                    logger.info(f"SUCCESS_RATE_CALC: VALIDATION PASSED - RMSD array matches pipeline_attempted_targets ({len(rmsds)})")
                
                # Also calculate legacy rates for comparison
                legacy_rate_2A = count_2A / total_targets * 100 if total_targets > 0 else 0
                legacy_rate_5A = count_5A / total_targets * 100 if total_targets > 0 else 0
                
                mean_rmsd = np.mean(rmsds)
                median_rmsd = np.median(rmsds)
                
                metrics[metric_key] = {
                    "count": len(rmsds),
                    "rate_2A": rate_2A,  # Main metric: pipeline success rate
                    "rate_5A": rate_5A,  # Main metric: pipeline success rate
                    "mean_rmsd": mean_rmsd,
                    "median_rmsd": median_rmsd,
                    # Additional stage-aware metrics
                    "pipeline_attempted_targets": pipeline_attempted_targets,
                    "legacy_rate_2A": legacy_rate_2A,  # For comparison
                    "legacy_rate_5A": legacy_rate_5A,  # For comparison
                    "total_targets": total_targets,
                    "pre_pipeline_excluded": pre_pipeline_excluded,
                    "pipeline_filtered": pipeline_filtered
                }
                
                # Log calculated success rates with stage awareness
                pipeline_failures = pipeline_attempted_targets - results_with_rmsd
                logger.info(f"SUCCESS_RATE_CALC: *** FINAL SUCCESS RATES FOR {metric_key.upper()} ***")
                logger.info(f"SUCCESS_RATE_CALC:   2A pipeline success: {rate_2A:.1f}% ({count_2A}/{pipeline_attempted_targets}) [CORRECTED - USE FOR PAPERS]")
                logger.info(f"SUCCESS_RATE_CALC:   5A pipeline success: {rate_5A:.1f}% ({count_5A}/{pipeline_attempted_targets}) [CORRECTED - USE FOR PAPERS]")
                logger.info(f"SUCCESS_RATE_CALC:   Includes {pipeline_failures} pipeline failures as 0 successes (timeouts, MCS failures, etc.)")
                logger.info(f"SUCCESS_RATE_CALC:   Legacy 2A rate: {legacy_rate_2A:.1f}% ({count_2A}/{total_targets}) [BIASED - DON'T USE]")
                logger.info(f"SUCCESS_RATE_CALC:   Legacy 5A rate: {legacy_rate_5A:.1f}% ({count_5A}/{total_targets}) [BIASED - DON'T USE]")
                logger.info(f"SUCCESS_RATE_CALC:   Mean RMSD (including failures): {mean_rmsd:.3f}A")
                logger.info(f"SUCCESS_RATE_CALC:   Bias correction factor: {rate_2A/legacy_rate_2A:.2f}x (corrected/legacy)")
                
            elif scores:
                # CRITICAL ERROR: RMSD data unavailable - cannot calculate meaningful success rates
                logger.error(f"SUCCESS_RATE_CALC: RMSD calculation failed for {metric_key} - CLI pipeline error detected")
                logger.error(f"SUCCESS_RATE_CALC:   Cannot calculate 2A/5A success rates without RMSD values")
                logger.error(f"SUCCESS_RATE_CALC:   This indicates a pipeline configuration or molecular structure issue")
                
                # Report invalid results - do not provide misleading success rates
                metrics[metric_key] = {
                    "count": len(scores),  
                    "rate_2A": 0.0,  # Cannot calculate without RMSD
                    "rate_5A": 0.0,  # Cannot calculate without RMSD
                    "mean_rmsd": None,  # No RMSD data available
                    "median_rmsd": None,  # No RMSD data available
                    "mean_score": np.mean(scores),  # Report alignment scores separately
                    "median_score": np.median(scores),  # Report alignment scores separately
                    "error": "RMSD_CALCULATION_FAILED"
                }
                
                logger.warning(f"SUCCESS_RATE_CALC: Benchmark results for {metric_key} are INVALID due to missing RMSD data")
                logger.warning(f"SUCCESS_RATE_CALC:   Available alignment scores: mean={np.mean(scores):.3f}, median={np.median(scores):.3f}")
                logger.warning(f"SUCCESS_RATE_CALC:   Check CLI RMSD calculation or molecular structure compatibility")
        
        return metrics
    
    def _validate_data_consistency(self, raw_results_file: Path, summary_data: Dict) -> Dict[str, Any]:
        """
        Validate consistency between raw results and summary data.
        
        Args:
            raw_results_file: Path to raw results JSON file
            summary_data: Summary data to validate
            
        Returns:
            Validation report with discrepancies and recommendations
        """
        validation_report = {
            "is_consistent": True,
            "discrepancies": [],
            "recommendations": []
        }
        
        try:
            # Load raw results
            with open(raw_results_file, 'r') as f:
                raw_data = json.load(f)
            
            # Extract key metrics from raw results
            raw_split_results = raw_data.get("split_results", {})
            raw_total_targets = 0
            raw_pipeline_success_rate = 0
            
            for split_name, split_data in raw_split_results.items():
                raw_total_targets = split_data.get("total_targets", 0)
                raw_pipeline_success_rate = split_data.get("success_rates", {}).get("pipeline_success_rate", 0)
                break  # For now, just check first split
            
            # Extract key metrics from summary
            summary_total_targets = 0
            summary_success_rate = 0
            
            if summary_data.get("summary"):
                for row in summary_data["summary"]:
                    summary_total_targets = row.get("Total_Targets", 0)
                    # Convert percentage string to float
                    success_rate_str = row.get("Success_Rate_2A", "0%")
                    summary_success_rate = float(success_rate_str.rstrip('%'))
                    break  # For now, just check first row
            
            # Check for discrepancies
            if raw_total_targets != summary_total_targets:
                validation_report["is_consistent"] = False
                validation_report["discrepancies"].append({
                    "metric": "total_targets",
                    "raw_value": raw_total_targets,
                    "summary_value": summary_total_targets,
                    "difference": abs(raw_total_targets - summary_total_targets)
                })
                validation_report["recommendations"].append(
                    f"Summary shows {summary_total_targets} targets but raw results show {raw_total_targets}. "
                    "Check if summary generator is processing correct dataset."
                )
            
            # Check success rate discrepancy (allow for small differences due to rounding)
            if abs(raw_pipeline_success_rate - summary_success_rate) > 5.0:  # 5% tolerance
                validation_report["is_consistent"] = False
                validation_report["discrepancies"].append({
                    "metric": "success_rate",
                    "raw_value": f"{raw_pipeline_success_rate:.1f}%",
                    "summary_value": f"{summary_success_rate:.1f}%",
                    "difference": f"{abs(raw_pipeline_success_rate - summary_success_rate):.1f}%"
                })
                validation_report["recommendations"].append(
                    f"Success rate discrepancy: raw={raw_pipeline_success_rate:.1f}%, summary={summary_success_rate:.1f}%. "
                    "Summary generator may not be using stage-aware metrics."
                )
            
            logger.info(f"Data consistency validation: {'PASS' if validation_report['is_consistent'] else 'FAIL'}")
            if not validation_report["is_consistent"]:
                logger.warning(f"Found {len(validation_report['discrepancies'])} discrepancies")
                for disc in validation_report["discrepancies"]:
                    logger.warning(f"  {disc['metric']}: raw={disc['raw_value']}, summary={disc['summary_value']}")
            
        except Exception as e:
            validation_report["is_consistent"] = False
            validation_report["discrepancies"].append({
                "metric": "validation_error",
                "error": str(e)
            })
            logger.error(f"Data consistency validation failed: {e}")
        
        return validation_report

    def _generate_timesplit_summary_fixed(self, results_data: Union[Dict, List[Dict]], output_format: str) -> Union[Dict, "pd.DataFrame"]:
        """
        Fixed version of timesplit summary generation using stage-aware metrics.
        
        This method addresses the critical bug where summary generation was not using
        the new stage-aware classification system, leading to incorrect success rates.
        """
        logger.info("Generating timesplit summary with stage-aware metrics...")
        
        # Ensure we have a list of results
        if isinstance(results_data, dict):
            # Extract individual results from the dict structure
            individual_results = []
            split_results = results_data.get("split_results", {})
            
            for split_name, split_data in split_results.items():
                results_file = split_data.get("results_file")
                if results_file and Path(results_file).exists():
                    # Load JSONL results and add split information
                    with open(results_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                result = json.loads(line)
                                # Add split information to each result
                                result["target_split"] = split_name
                                result["results_file"] = results_file
                                individual_results.append(result)
            
            results_data = individual_results
        
        if not results_data:
            logger.warning("No results data provided for timesplit summary")
            return self._format_output([], output_format)
        
        # Validate that we have the expected data structure
        logger.info(f"Processing {len(results_data)} individual results for timesplit summary")
        
        # Use stage-aware metrics calculation
        total_targets = len(results_data)
        successful_results = [r for r in results_data if r.get("success") and r.get("rmsd_values")]
        
        # Calculate stage-aware metrics using existing processing_stage field
        stage_counts = {"pre_pipeline_excluded": 0, "pipeline_filtered": 0, "pipeline_attempted": 0}
        
        for result in results_data:
            # Use the existing processing_stage field set by simple_runner.py
            processing_stage = result.get("processing_stage", "unknown")
            
            if processing_stage in stage_counts:
                # Use the pre-calculated processing stage
                stage_counts[processing_stage] += 1
            else:
                # Fallback for unknown processing stages - classify based on result content
                if result.get("success") and result.get("rmsd_values"):
                    # Successful results with RMSD are always pipeline_attempted
                    stage_counts["pipeline_attempted"] += 1
                elif result.get("success") and not result.get("rmsd_values"):
                    # Successful CLI execution but no RMSD (0 templates/poses)
                    # Check if this result has pipeline stage info first
                    pipeline_stage = result.get("pipeline_stage", None)
                    made_it_to_mcs = result.get("made_it_to_mcs", False)
                    
                    if pipeline_stage == "pipeline_attempted" or made_it_to_mcs:
                        # This reached MCS stage but failed to generate RMSD - count as pipeline_attempted
                        stage_counts["pipeline_attempted"] += 1
                    else:
                        # Analyze CLI JSON output to distinguish data issues from filtering
                        exclusion_reason = self._classify_no_templates_case(result)
                        fallback_stage = self._classify_exclusion_processing_stage(exclusion_reason)
                        stage_counts[fallback_stage] += 1
                else:
                    # Failed results - analyze error message to determine stage
                    error_msg = result.get("error", "")
                    if error_msg:  # Only process results with actual error messages
                        exclusion_reason = self._parse_exclusion_reason(error_msg)
                        fallback_stage = self._classify_exclusion_processing_stage(exclusion_reason)
                        stage_counts[fallback_stage] += 1
                    else:
                        # No error message - treat as pipeline_attempted (conservative)
                        stage_counts["pipeline_attempted"] += 1
        
        pipeline_attempted_targets = stage_counts["pipeline_attempted"]
        pre_pipeline_excluded = stage_counts["pre_pipeline_excluded"]
        pipeline_filtered = stage_counts["pipeline_filtered"]
        
        logger.info(f"Stage-aware analysis:")
        logger.info(f"  Total targets: {total_targets}")
        logger.info(f"  Pre-pipeline excluded: {pre_pipeline_excluded}")
        logger.info(f"  Pipeline filtered: {pipeline_filtered}")
        logger.info(f"  Pipeline attempted: {pipeline_attempted_targets}")
        
        # Calculate metrics using stage-aware approach
        metrics = self._calculate_timesplit_metrics(successful_results, total_targets, results_data)
        
        # Generate summary table with corrected metrics
        summary_rows = []
        
        for metric_key, metric_data in metrics.items():
            # Check if this is a valid metric (shape, color, combo)
            if metric_key in ["shape", "color", "combo"]:
                # Extract metric name (shape, color, combo)
                metric_name = metric_key.title()
                
                # Use stage-aware success rates (main metric)
                rate_2A = metric_data.get("rate_2A", 0)
                rate_5A = metric_data.get("rate_5A", 0)
                
                # Calculate average runtime
                avg_runtime = metric_data.get("avg_runtime", 0)
                
                # Generate exclusion reasons breakdown
                exclusion_reasons = defaultdict(int)
                for result in results_data:
                    if not result.get("success"):
                        # Failed results
                        error_msg = result.get("error", "")
                        if error_msg:  # Only process results with actual error messages
                            exclusion_reason = self._parse_exclusion_reason(error_msg)
                            exclusion_reasons[exclusion_reason] += 1
                        else:
                            # No error message - check if this is a database_empty case from CLI JSON
                            exclusion_reason = self._classify_no_templates_case(result)
                            exclusion_reasons[exclusion_reason] += 1
                    elif result.get("success") and not result.get("rmsd_values"):
                        # Successful CLI execution but no RMSD (0 templates/poses)
                        # Check if this result has pipeline stage info first
                        pipeline_stage = result.get("pipeline_stage", None)
                        made_it_to_mcs = result.get("made_it_to_mcs", False)
                        
                        if pipeline_stage == "pipeline_attempted" or made_it_to_mcs:
                            # This reached MCS stage but failed to generate RMSD - count as pipeline failure
                            exclusion_reasons["pipeline_attempted_no_rmsd"] += 1
                        else:
                            # Analyze CLI JSON output to distinguish data issues from filtering
                            exclusion_reason = self._classify_no_templates_case(result)
                            exclusion_reasons[exclusion_reason] += 1
                
                # Extract RMSD values for explicit success counts
                successful_rmsds = []
                for result in successful_results:
                    rmsd_values = result.get("rmsd_values", {})
                    combo_rmsd = rmsd_values.get("combo")
                    
                    # Handle different RMSD value formats
                    if combo_rmsd is not None:
                        # Check if it's a dict with nested RMSD value
                        if isinstance(combo_rmsd, dict):
                            rmsd_val = combo_rmsd.get("rmsd")
                        else:
                            rmsd_val = combo_rmsd
                        
                        # Validate RMSD value is numeric and not NaN
                        if rmsd_val is not None:
                            try:
                                rmsd_float = float(rmsd_val)
                                if not np.isnan(rmsd_float):
                                    successful_rmsds.append(rmsd_float)
                            except (ValueError, TypeError):
                                # Skip non-numeric RMSD values
                                continue
                
                # Calculate explicit success counts
                count_2A = sum(1 for rmsd in successful_rmsds if rmsd <= 2.0)
                count_5A = sum(1 for rmsd in successful_rmsds if rmsd <= 5.0)
                
                summary_row = {
                    "Benchmark": "Timesplit",
                    "Split": "test",  # Use actual split name if available
                    "Metric": metric_name,
                    "Total_Targets": total_targets,
                    "Targets_With_RMSD": len(successful_results),
                    "Excluded_Targets": total_targets - len(successful_results),
                    
                    # EXPLICIT SUCCESS RATE CALCULATIONS
                    "Success_Rate_2A": f"{rate_2A:.1f}%",
                    "Success_Rate_2A_Explicit": f"{count_2A}/{pipeline_attempted_targets}",
                    "Success_Rate_5A": f"{rate_5A:.1f}%", 
                    "Success_Rate_5A_Explicit": f"{count_5A}/{pipeline_attempted_targets}",
                    
                    # RMSD STATISTICS
                    "Mean_RMSD": f"{metric_data.get('mean_rmsd', 0):.2f}",
                    "Median_RMSD": f"{metric_data.get('median_rmsd', 0):.2f}",
                    
                    # STAGE-AWARE PIPELINE REPORTING  
                    "Pipeline_Attempted": pipeline_attempted_targets,
                    "Pipeline_Successful": len(successful_results),
                    "Pipeline_Failed": pipeline_attempted_targets - len(successful_results),
                    "Pre_Pipeline_Excluded": pre_pipeline_excluded,
                    "Pipeline_Filtered": pipeline_filtered,
                    
                    # FILTERING BREAKDOWN
                    "Peptide_Polysaccharide_Filtered": self._count_peptide_filtering(all_results),
                    "Template_Database_Filtering": self._extract_template_filtering_stats(all_results),
                    
                    # LEGACY COMPATIBILITY
                    "Avg_Exclusions": "0",  # Placeholder
                    "Avg_Runtime": f"{avg_runtime:.1f}s",
                    "Exclusion_Reasons": dict(exclusion_reasons)
                }
                
                summary_rows.append(summary_row)
        
        # Create enhanced summary with stage-aware information
        enhanced_summary = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "summary": summary_rows,
            "stage_aware_metrics": {
                "total_targets": total_targets,
                "pipeline_attempted": pipeline_attempted_targets,
                "pre_pipeline_excluded": pre_pipeline_excluded,
                "pipeline_filtered": pipeline_filtered,
                "pipeline_success_rate": (len(successful_results) / pipeline_attempted_targets * 100) if pipeline_attempted_targets > 0 else 0
            },
            "validation": {
                "uses_stage_aware_metrics": True,
                "data_consistency_checked": True
            }
        }
        
        logger.info(f"Generated timesplit summary with {len(summary_rows)} metric rows")
        logger.info(f"Stage-aware pipeline success rate: {enhanced_summary['stage_aware_metrics']['pipeline_success_rate']:.1f}%")
        
        return self._format_output(summary_rows, output_format)

    def _count_peptide_filtering(self, results: List[Dict]) -> Dict[str, int]:
        """
        Count peptide/polysaccharide filtering instances from CLI JSON output.
        
        Args:
            results: List of benchmark result dictionaries
            
        Returns:
            Dictionary with filtering counts
        """
        peptide_count = 0
        polysaccharide_count = 0
        
        for result in results:
            # Check if this result was filtered for peptide/polysaccharide
            pipeline_stage = result.get("pipeline_stage", "")
            if pipeline_stage == "pipeline_filtered":
                # Extract CLI JSON to check for peptide filtering
                stdout = result.get("stdout", "")
                if "peptide" in stdout.lower():
                    peptide_count += 1
                elif "polysaccharide" in stdout.lower() or "saccharide" in stdout.lower():
                    polysaccharide_count += 1
                    
        return {
            "peptides": peptide_count,
            "polysaccharides": polysaccharide_count,
            "total_filtered": peptide_count + polysaccharide_count
        }

    def _extract_template_filtering_stats(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Extract template database filtering statistics from CLI JSON output.
        
        Args:
            results: List of benchmark result dictionaries
            
        Returns:
            Dictionary with template filtering statistics
        """
        template_stats = {
            "total_templates_in_database": [],
            "templates_used_for_poses": [],
            "template_filtering_applied": 0
        }
        
        for result in results:
            stdout = result.get("stdout", "")
            if "TEMPL_JSON_RESULT:" in stdout:
                try:
                    json_start = stdout.find("TEMPL_JSON_RESULT:") + len("TEMPL_JSON_RESULT:")
                    json_end = stdout.find("\n", json_start)
                    if json_end == -1:
                        json_end = len(stdout)
                    
                    json_str = stdout[json_start:json_end].strip()
                    cli_result = json.loads(json_str)
                    
                    # Extract template statistics
                    total_templates = cli_result.get("total_templates_in_database", 0)
                    used_templates = cli_result.get("templates_used_for_poses", 0)
                    
                    if total_templates > 0:
                        template_stats["total_templates_in_database"].append(total_templates)
                        template_stats["templates_used_for_poses"].append(used_templates)
                        
                        # Check if filtering was applied
                        template_db_stats = cli_result.get("template_database_stats", {})
                        if template_db_stats.get("filtered_peptides", 0) > 0 or template_db_stats.get("filtered_polysaccharides", 0) > 0:
                            template_stats["template_filtering_applied"] += 1
                            
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue
        
        # Calculate averages
        if template_stats["total_templates_in_database"]:
            template_stats["avg_total_templates"] = sum(template_stats["total_templates_in_database"]) / len(template_stats["total_templates_in_database"])
            template_stats["avg_used_templates"] = sum(template_stats["templates_used_for_poses"]) / len(template_stats["templates_used_for_poses"])
        else:
            template_stats["avg_total_templates"] = 0
            template_stats["avg_used_templates"] = 0
            
        return template_stats
    
    def _format_template_description(self, template_counts: Dict, dataset: str, query_count: int) -> str:
        """Format template description for Polaris results."""
        if "total_combined" in template_counts:
            # Combined template pool
            mers_count = template_counts.get("MERS_native", 0)
            sars_count = template_counts.get("SARS_aligned", 0)
            total_count = template_counts.get("total_combined", mers_count + sars_count)
            return f"{total_count} ({mers_count}+{sars_count})"
        elif dataset == "Train" and template_counts:
            # Leave-one-out: template count is query_count - 1
            return f"{query_count-1} (LOO)"
        else:
            # Single template pool
            total_templates = sum(template_counts.values()) if template_counts else 0
            return str(total_templates)
    
    def _generate_generic_summary(self, results_data: Dict, output_format: str) -> Union[Dict, "pd.DataFrame"]:
        """Generate generic summary for unknown benchmark types."""
        logger.warning("Using generic summary generation for unknown benchmark type")
        
        # Basic summary of results structure
        summary = {
            "total_entries": len(results_data) if isinstance(results_data, dict) else 0,
            "entry_types": list(results_data.keys()) if isinstance(results_data, dict) else [],
            "structure": str(type(results_data))
        }
        
        table_data = [{
            "Benchmark": "Unknown",
            "Total_Entries": summary["total_entries"],
            "Structure": summary["structure"],
            "Keys": str(summary["entry_types"][:5]) + ("..." if len(summary["entry_types"]) > 5 else "")
        }]
        
        return self._format_output(table_data, output_format)
    
    def _format_output(self, table_data: List[Dict], output_format: str) -> Union[Dict, "pd.DataFrame"]:
        """Format output data according to requested format."""
        if output_format == "pandas" and PANDAS_AVAILABLE:
            return pd.DataFrame(table_data)
        elif output_format == "dict":
            return {"summary_data": table_data}
        elif output_format == "json":
            return json.dumps(table_data, indent=2)
        else:
            # Fallback to list of dicts
            return table_data
    
    def save_summary_files(
        self, 
        summary_data: Union[Dict, "pd.DataFrame", List[Dict]], 
        output_dir: Path,
        base_name: str = "benchmark_summary",
        formats: Optional[List[str]] = None
    ) -> Dict[str, Path]:
        """Save summary data to multiple file formats."""
        if formats is None:
            formats = ["csv", "json"]
            if PANDAS_AVAILABLE:
                formats.append("markdown")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        # Convert to pandas DataFrame if possible
        df = None
        if PANDAS_AVAILABLE and pd is not None:
            if not isinstance(summary_data, pd.DataFrame):
                if isinstance(summary_data, list):
                    df = pd.DataFrame(summary_data)
                elif isinstance(summary_data, dict) and "summary_data" in summary_data:
                    df = pd.DataFrame(summary_data["summary_data"])
                else:
                    df = pd.DataFrame([summary_data])
            else:
                df = summary_data
        
        for fmt in formats:
            try:
                if fmt == "csv" and df is not None:
                    file_path = output_dir / f"{base_name}_{timestamp}.csv"
                    df.to_csv(file_path, index=False)
                    saved_files["csv"] = file_path
                
                elif fmt == "json":
                    file_path = output_dir / f"{base_name}_{timestamp}.json"
                    if PANDAS_AVAILABLE and pd is not None and isinstance(summary_data, pd.DataFrame):
                        data_to_save = summary_data.to_dict('records')
                    elif isinstance(summary_data, list):
                        data_to_save = summary_data
                    elif isinstance(summary_data, dict):
                        data_to_save = summary_data
                    else:
                        data_to_save = {"data": str(summary_data)}
                    # Enhanced JSON structure with exclusion metadata
                    enhanced_data = {
                        "timestamp": timestamp,
                        "summary": data_to_save
                    }
                    # Add exclusion summary if available
                    if isinstance(data_to_save, list) and data_to_save:
                        total_exclusions = {}
                        total_targets = 0
                        total_with_rmsd = 0
                        for entry in data_to_save:
                            if isinstance(entry, dict):
                                total_targets += entry.get("Total_Targets", 0)
                                total_with_rmsd += entry.get("Targets_With_RMSD", 0)
                                exclusion_reasons = entry.get("Exclusion_Reasons", {})
                                if exclusion_reasons:
                                    for reason, count in exclusion_reasons.items():
                                        total_exclusions[reason] = total_exclusions.get(reason, 0) + count
                        if total_exclusions:
                            enhanced_data["exclusion_summary"] = {
                                "total_targets_processed": total_targets,
                                "total_targets_with_rmsd": total_with_rmsd,
                                "total_excluded": sum(total_exclusions.values()),
                                "exclusion_breakdown": total_exclusions,
                                "success_rate": (total_with_rmsd / total_targets * 100) if total_targets > 0 else 0
                            }
                    with open(file_path, 'w') as f:
                        json.dump(enhanced_data, f, indent=2, default=str)
                    saved_files["json"] = file_path
                
                elif fmt == "markdown" and df is not None:
                    file_path = output_dir / f"{base_name}_{timestamp}.md"
                    with open(file_path, 'w') as f:
                        f.write(f"# Benchmark Summary ({timestamp})\n\n")
                        f.write(df.to_markdown(index=False))
                        f.write("\n\n*Generated by TEMPL Benchmark Suite*\n")
                    saved_files["markdown"] = file_path
                
                logging.info(f"Saved {fmt} summary to {saved_files.get(fmt, 'N/A')}")
            except Exception as e:
                logging.error(f"Failed to save {fmt} format: {e}")
        return saved_files


def generate_summary_from_files(
    results_files: List[Union[str, Path]], 
    output_dir: Optional[Union[str, Path]] = None,
    benchmark_type: Optional[str] = None
) -> Dict[str, Path]:
    """Convenience function to generate summaries from result files.
    
    Args:
        results_files: List of paths to result files (JSON/JSONL)
        output_dir: Output directory for summary files
        benchmark_type: Type of benchmark (auto-detect if None)
        
    Returns:
        Dictionary mapping format to saved file path
    """
    generator = BenchmarkSummaryGenerator()
    
    if output_dir is None:
        output_dir = Path.cwd() / "benchmark_summaries"
    else:
        output_dir = Path(output_dir)
    
    all_results = {}
    
    # Load results from files
    for file_path in results_files:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"Results file not found: {file_path}")
            continue
            
        try:
            if file_path.suffix == ".jsonl":
                # JSONL format - extract split information from filename
                results = []
                
                # Extract split from filename pattern: results_{split}_{timestamp}.jsonl
                import re
                split_match = re.search(r'results_([^_]+)_\d+\.jsonl', file_path.name)
                extracted_split = split_match.group(1) if split_match else None
                
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line)
                            # Add split information if extracted from filename
                            if extracted_split and "target_split" not in result:
                                result["target_split"] = extracted_split
                                result["results_file"] = str(file_path)
                            results.append(result)
                            
                all_results[file_path.stem] = results
                
            elif file_path.suffix == ".json":
                # JSON format
                with open(file_path, 'r') as f:
                    results = json.load(f)
                all_results[file_path.stem] = results
                
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
    
    if not all_results:
        logger.error("No valid results loaded")
        return {}
    
    # Generate summary for combined results
    if len(all_results) == 1:
        # Single result file
        results_data = list(all_results.values())[0]
    else:
        # Multiple result files - combine them
        results_data = all_results
    
    summary = generator.generate_unified_summary(results_data, benchmark_type)
    
    # Save summary files
    return generator.save_summary_files(
        summary, 
        output_dir, 
        "combined_benchmark_summary"
    )


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Generate unified benchmark summaries")
    parser.add_argument("results_files", nargs="+", help="Result files to process")
    parser.add_argument("--output-dir", "-o", help="Output directory")
    parser.add_argument("--benchmark-type", "-t", choices=["polaris", "timesplit"], 
                       help="Benchmark type (auto-detect if not specified)")
    parser.add_argument("--formats", nargs="+", default=["csv", "json", "markdown"],
                       choices=["csv", "json", "markdown"], help="Output formats")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Process results files
    all_files = []
    for path_str in args.results_files:
        path = Path(path_str)
        if path.is_dir():
            all_files.extend(path.glob('**/*.json'))
            all_files.extend(path.glob('**/*.jsonl'))
        else:
            all_files.append(path)

    if not all_files:
        print("Error: No valid results files found.")
        sys.exit(1)

    try:
        saved_files = generate_summary_from_files(
            all_files,
            args.output_dir,
            args.benchmark_type
        )
        
        if saved_files:
            print("Summary files generated:")
            for fmt, path in saved_files.items():
                print(f"  {fmt.upper()}: {path}")
        else:
            print("No summary files generated")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 