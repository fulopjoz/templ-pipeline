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
            # Polaris benchmark format
            bench_info = results_data["benchmark_info"]
            if "name" in bench_info and "polaris" in bench_info["name"].lower():
                return "polaris"
        
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
        """Generate summary for Timesplit benchmark results."""
        
        logger.debug(f"Timesplit data type: {type(results_data)}")
        if isinstance(results_data, list):
            logger.debug(f"List length: {len(results_data)}")
        elif isinstance(results_data, dict):
            logger.debug(f"Dict keys: {list(results_data.keys())}")
        
        # Handle both dictionary and list formats
        if isinstance(results_data, list):
            # JSONL format - list of individual results
            individual_results = results_data
        elif isinstance(results_data, dict):
            # Dictionary format - extract individual results
            individual_results = []
            for key, value in results_data.items():
                if isinstance(value, dict) and "pdb_id" in value:
                    individual_results.append(value)
                elif isinstance(value, list):
                    # Handle case where results are in a list within the dict
                    individual_results.extend(value)
        else:
            logger.error(f"Unsupported results format: {type(results_data)}")
            return self._format_output([], output_format)
        
        logger.debug(f"Found {len(individual_results)} individual results")
        
        # Handle empty results gracefully
        if not individual_results:
            logger.warning("No individual results found for timesplit summary")
            return self._format_output([], output_format)
        
        # Group results by split (include both successful and failed for statistics)
        split_groups = defaultdict(list)
        successful_groups = defaultdict(list)
        
        for result in individual_results:
            # Handle missing target_split by extracting from context or defaulting
            split = result.get("target_split")
            if split is None:
                # Extract split from filename or default to "test"
                split = "test"  # Default for timesplit benchmarks
            
            split_groups[split].append(result)
            
            # Consider a result successful if it has success=True AND generated poses (non-empty rmsd_values)
            has_poses = result.get("rmsd_values") and bool(result["rmsd_values"])
            if result.get("success", False) and has_poses:
                successful_groups[split].append(result)
        
        table_data = []
        
        # Generate summary for each split
        for split_name, split_results in split_groups.items():
            if not split_results:
                continue
                
            successful_results = successful_groups.get(split_name, [])
            logger.debug(f"Split {split_name}: {len(successful_results)} successful out of {len(split_results)} total")
                
            # Use successful results for metrics calculation and get exclusion analysis
            total_processed = len(split_results)
            metrics, exclusion_stats = self._calculate_timesplit_metrics_with_exclusions(split_results, successful_results, total_processed)
            
            # Count successful results
            successful_count = len(successful_results)
            
            # Calculate average exclusions and runtime
            avg_exclusions = np.mean([r.get("exclusions_count", 0) for r in split_results])
            avg_runtime = np.mean([r.get("runtime_total", 0) for r in split_results])
            
            # Add entry for each scoring metric or summary if no successful results
            if metrics and successful_count > 0:
                for metric in ["shape", "color", "combo"]:
                    if metric in metrics:
                        metric_data = metrics[metric]
                        table_data.append({
                            "Benchmark": "Timesplit",
                            "Split": split_name.title(),
                            "Metric": metric.title(),
                            "Total_Targets": total_processed,
                            "Targets_With_RMSD": successful_count,
                            "Excluded_Targets": exclusion_stats.get("excluded_count", 0),
                            "Success_Rate_2A": f"{metric_data.get('rate_2A', 0):.1f}%",
                            "Success_Rate_5A": f"{metric_data.get('rate_5A', 0):.1f}%",
                            "Mean_RMSD": f"{metric_data.get('mean_rmsd', 0):.2f}",
                            "Median_RMSD": f"{metric_data.get('median_rmsd', 0):.2f}",
                            "Avg_Exclusions": f"{avg_exclusions:.0f}",
                            "Avg_Runtime": f"{avg_runtime:.1f}s",
                            "Exclusion_Reasons": exclusion_stats.get("exclusion_breakdown", {})
                        })
            else:
                # No successful results - add summary entry
                table_data.append({
                    "Benchmark": "Timesplit",
                    "Split": split_name.title(),
                    "Metric": "Summary",
                    "Total_Targets": total_processed,
                    "Targets_With_RMSD": successful_count,
                    "Excluded_Targets": exclusion_stats.get("excluded_count", 0),
                    "Success_Rate_2A": "0.0%",
                    "Success_Rate_5A": "0.0%",
                    "Mean_RMSD": "N/A",
                    "Median_RMSD": "N/A",
                    "Avg_Exclusions": f"{avg_exclusions:.0f}",
                    "Avg_Runtime": f"{avg_runtime:.1f}s",
                    "Exclusion_Reasons": exclusion_stats.get("exclusion_breakdown", {})
                })
        
        return self._format_output(table_data, output_format)
    
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
        """Parse exclusion reason from error message."""
        if not error_msg:
            return "unknown_error"
        
        error_msg_lower = error_msg.lower()
        
        # Parse skip reasons
        if "skipped" in error_msg_lower:
            if "molecule_too_small" in error_msg_lower:
                return "molecule_too_small"
            elif "molecule_too_large" in error_msg_lower:
                return "molecule_too_large"
            elif "poor_quality" in error_msg_lower:
                return "poor_quality_crystal"
            elif "peptide" in error_msg_lower:
                return "peptide_excluded"
            elif "invalid_smiles" in error_msg_lower:
                return "invalid_smiles"
            elif "sanitization_failed" in error_msg_lower:
                return "molecule_sanitization_failed"
            else:
                return "validation_failed"
        
        # Parse pipeline errors
        elif "file not found" in error_msg_lower:
            if "protein" in error_msg_lower:
                return "protein_file_missing"
            elif "ligand" in error_msg_lower:
                return "ligand_data_missing"
            else:
                return "file_not_found"
        elif "pose generation" in error_msg_lower or "no poses generated" in error_msg_lower:
            return "pose_generation_failed"
        elif "rmsd calculation failed" in error_msg_lower:
            return "rmsd_calculation_failed"
        elif "conformer generation failed" in error_msg_lower:
            return "conformer_generation_failed"
        elif "alignment failed" in error_msg_lower:
            return "molecular_alignment_failed"
        elif "mcs" in error_msg_lower:
            return "mcs_calculation_failed"
        elif "embedding failed" in error_msg_lower:
            return "embedding_failed"
        elif "timeout" in error_msg_lower:
            return "timeout"
        elif "memory" in error_msg_lower:
            return "memory_error"
        else:
            return "pipeline_error"

    def _calculate_timesplit_metrics_with_exclusions(self, all_results: List[Dict], successful_results: List[Dict], total_targets: int) -> Tuple[Dict, Dict]:
        """Calculate metrics for Timesplit benchmark results with exclusion analysis.
        
        Args:
            all_results: All target results including failed ones
            successful_results: Only successful results with RMSD values
            total_targets: Total number of targets processed
            
        Returns:
            Tuple of (metrics_dict, exclusion_stats_dict)
        """
        # Calculate standard metrics using successful results
        metrics = self._calculate_timesplit_metrics(successful_results, total_targets)
        
        # Analyze exclusions and failures
        exclusion_breakdown = defaultdict(int)
        excluded_count = 0
        
        for result in all_results:
            if not result.get("success", False) or not result.get("rmsd_values"):
                excluded_count += 1
                error_msg = result.get("error", "")
                exclusion_reason = self._parse_exclusion_reason(error_msg)
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

    def _calculate_timesplit_metrics(self, split_results: List[Dict], total_targets: int) -> Dict:
        """Calculate metrics for Timesplit benchmark results."""
        metrics = {}
        
        # Validate RMSD calculation integrity
        validation_report = self._validate_rmsd_calculation(split_results)
        
        # Success rate calculation detailed logging for Timesplit
        logger.info(f"SUCCESS_RATE_CALC: Processing Timesplit results:")
        logger.info(f"SUCCESS_RATE_CALC:   Total targets to analyze: {total_targets}")
        logger.info(f"SUCCESS_RATE_CALC:   Split results length: {len(split_results)}")
        logger.info(f"SUCCESS_RATE_CALC:   Results with valid RMSD: {validation_report['results_with_rmsd']}/{validation_report['successful_results']}")
        
        # Collect RMSD values and scores by metric
        rmsd_by_metric = defaultdict(list)
        score_by_metric = defaultdict(list)
        
        successful_results = 0
        results_with_rmsd = 0
        
        for result in split_results:
            if result.get("success") and result.get("rmsd_values"):
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
        
        logger.info(f"SUCCESS_RATE_CALC: Timesplit data collection summary:")
        logger.info(f"SUCCESS_RATE_CALC:   Successful results: {successful_results}/{len(split_results)}")
        logger.info(f"SUCCESS_RATE_CALC:   Results with RMSD values: {results_with_rmsd}/{successful_results}")
        
        # Log RMSD data by metric
        for metric_key in rmsd_by_metric:
            rmsd_count = len(rmsd_by_metric[metric_key])
            logger.info(f"SUCCESS_RATE_CALC:   {metric_key}: {rmsd_count} RMSD values collected")
        
        # Calculate statistics for each metric
        for metric_key in set(list(rmsd_by_metric.keys()) + list(score_by_metric.keys())):
            rmsds = rmsd_by_metric.get(metric_key, [])
            scores = score_by_metric.get(metric_key, [])
            
            if rmsds:
                # Use RMSD-based calculation when available
                count_2A = sum(1 for rmsd in rmsds if rmsd <= 2.0)
                count_5A = sum(1 for rmsd in rmsds if rmsd <= 5.0)
                rate_2A = count_2A / total_targets * 100 if total_targets > 0 else 0
                rate_5A = count_5A / total_targets * 100 if total_targets > 0 else 0
                mean_rmsd = np.mean(rmsds)
                median_rmsd = np.median(rmsds)
                
                metrics[metric_key] = {
                    "count": len(rmsds),
                    "rate_2A": rate_2A,
                    "rate_5A": rate_5A,
                    "mean_rmsd": mean_rmsd,
                    "median_rmsd": median_rmsd,
                }
                
                # Log calculated success rates
                logger.info(f"SUCCESS_RATE_CALC: Final Timesplit rates for {metric_key} (RMSD-based):")
                logger.info(f"SUCCESS_RATE_CALC:   2A success: {rate_2A:.1f}% ({count_2A}/{total_targets})")
                logger.info(f"SUCCESS_RATE_CALC:   5A success: {rate_5A:.1f}% ({count_5A}/{total_targets})")
                logger.info(f"SUCCESS_RATE_CALC:   Mean RMSD: {mean_rmsd:.3f}A")
                
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
        """Save summary data to multiple file formats.
        
        Args:
            summary_data: Summary data to save
            output_dir: Output directory
            base_name: Base name for output files
            formats: List of formats to save ("csv", "json", "markdown")
            
        Returns:
            Dictionary mapping format to saved file path
        """
        if formats is None:
            formats = ["csv", "json"]
            if PANDAS_AVAILABLE:
                formats.append("markdown")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        # Convert to pandas DataFrame if possible
        if PANDAS_AVAILABLE and not isinstance(summary_data, pd.DataFrame):
            if isinstance(summary_data, list):
                df = pd.DataFrame(summary_data)
            elif isinstance(summary_data, dict) and "summary_data" in summary_data:
                df = pd.DataFrame(summary_data["summary_data"])
            else:
                df = pd.DataFrame([summary_data])
        elif isinstance(summary_data, pd.DataFrame):
            df = summary_data
        else:
            df = None
        
        for fmt in formats:
            try:
                if fmt == "csv" and df is not None:
                    file_path = output_dir / f"{base_name}_{timestamp}.csv"
                    df.to_csv(file_path, index=False)
                    saved_files["csv"] = file_path
                    
                elif fmt == "json":
                    file_path = output_dir / f"{base_name}_{timestamp}.json"
                    if isinstance(summary_data, pd.DataFrame):
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
                    
                logger.info(f"Saved {fmt} summary to {saved_files.get(fmt, 'N/A')}")
                
            except Exception as e:
                logger.error(f"Failed to save {fmt} format: {e}")
        
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
                # JSONL format
                results = []
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            results.append(json.loads(line))
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