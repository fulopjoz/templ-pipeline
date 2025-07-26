#!/usr/bin/env python3
"""
Standalone Benchmark Results Processor

This script processes existing benchmark results JSON files to generate:
- Summary tables (CSV/Markdown)
- RMSD distribution plots  
- Metrics analysis
- MCS pattern analysis

Usage:
    python process_benchmark_results.py <results_file.json>
    python process_benchmark_results.py <results_file.json> --output-dir custom_output/
"""

import json
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Set
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the parent directory to sys.path to import from mcs_bench
sys.path.append(str(Path(__file__).parent.parent))

def calculate_metrics_fixed(split_results_data: Dict) -> Dict:
    """Fixed version of calculate_metrics that works with actual data structure."""
    pdb_results = split_results_data.get("results", {})
    if not pdb_results: 
        return {"total": 0, "successful": 0}

    metrics = {
        "total": len(pdb_results),
        "successful": sum(1 for r_data in pdb_results.values() if r_data.get("success")),
        "failed": sum(1 for r_data in pdb_results.values() if not r_data.get("success")),
        "rmsd_counts_2A": defaultdict(int),
        "rmsd_counts_5A": defaultdict(int),
        "all_rmsds": defaultdict(list),
        "all_scores": defaultdict(list),
        "runtimes": [],
        # CA RMSD fields (will be empty with current data)
        "ca_rmsd_values": [],
        "ca_rmsd_matched_residues": [],
        "ca_rmsd_coverage": [],
        "ca_rmsd_filtered_count": 0,
        # Enhanced RMSD statistics
        "min_rmsd": defaultdict(lambda: float('inf')),
        "max_rmsd": defaultdict(lambda: float('-inf')),
        "outliers_50A": defaultdict(int)
    }
    
    for pdb_id, r_data in pdb_results.items():
        if r_data.get("success"):
            # Fix: Use "runtime_seconds" instead of "runtime"
            if r_data.get("runtime_seconds") is not None: 
                metrics["runtimes"].append(r_data["runtime_seconds"])
            
            # Fix: Use "rmsd_data" instead of "rmsd_values"
            if r_data.get("rmsd_data"):
                for metric_key, values_dict in r_data["rmsd_data"].items():
                    rmsd = values_dict.get("rmsd")
                    score = values_dict.get("score")
                    if rmsd is not None: 
                        metrics["all_rmsds"][metric_key].append(rmsd)
                        # Update min/max tracking
                        metrics["min_rmsd"][metric_key] = min(metrics["min_rmsd"][metric_key], rmsd)
                        metrics["max_rmsd"][metric_key] = max(metrics["max_rmsd"][metric_key], rmsd)
                        # Count outliers and thresholds
                        if rmsd <= 2.0: metrics["rmsd_counts_2A"][metric_key] += 1
                        if rmsd <= 5.0: metrics["rmsd_counts_5A"][metric_key] += 1
                        if rmsd > 50.0: metrics["outliers_50A"][metric_key] += 1
                    if score is not None: 
                        metrics["all_scores"][metric_key].append(score)
    
    metrics["mean_runtime"] = np.mean(metrics["runtimes"]) if metrics["runtimes"] else None
    
    # Calculate per-method statistics
    metrics_summary_per_method = {}
    for metric_key in metrics["all_rmsds"]:
        rmsd_list = metrics["all_rmsds"][metric_key]
        score_list = metrics["all_scores"][metric_key]
        num_successful_for_metric = len(rmsd_list)
        
        metrics_summary_per_method[metric_key] = {
            "count": num_successful_for_metric,
            "mean_rmsd": np.mean(rmsd_list) if rmsd_list else None,
            "median_rmsd": np.median(rmsd_list) if rmsd_list else None,
            "min_rmsd": metrics["min_rmsd"][metric_key] if metrics["min_rmsd"][metric_key] != float('inf') else None,
            "max_rmsd": metrics["max_rmsd"][metric_key] if metrics["max_rmsd"][metric_key] != float('-inf') else None,
            "mean_score": np.mean(score_list) if score_list else None,
            "perc_below_2A": (metrics["rmsd_counts_2A"][metric_key] / num_successful_for_metric * 100) if num_successful_for_metric > 0 else 0,
            "perc_below_5A": (metrics["rmsd_counts_5A"][metric_key] / num_successful_for_metric * 100) if num_successful_for_metric > 0 else 0,
            "outliers_above_50A": metrics["outliers_50A"][metric_key]
        }
    
    metrics["per_method_stats"] = metrics_summary_per_method
    return metrics

def plot_rmsd_distribution_fixed(split_results_data: Dict, split_name_key: str, output_plot_dir: str, histogram_bins: int = 50) -> None:
    """Fixed version of RMSD plotting that works with actual data structure."""
    pdb_results = split_results_data.get("results", {})
    if not pdb_results: 
        return
        
    rmsds_by_metric = defaultdict(list)
    for r_data in pdb_results.values():
        # Fix: Use "rmsd_data" instead of "rmsd_values"
        if r_data.get("success") and r_data.get("rmsd_data"):
            for metric_key, values_dict in r_data["rmsd_data"].items():
                if values_dict.get("rmsd") is not None: 
                    rmsds_by_metric[metric_key].append(values_dict["rmsd"])
    
    if not rmsds_by_metric: 
        print(f"No RMSD data found for {split_name_key}")
        return

    os.makedirs(output_plot_dir, exist_ok=True)

    metric_colors = {"shape": "#2ecc71", "color": "#e74c3c", "combo": "#3498db"}
    for metric_key, values in rmsds_by_metric.items():
        if not values: continue
        
        plt.style.use('default')  # Use default instead of seaborn for better compatibility
        plt.figure(figsize=(10, 6))
        color = metric_colors.get(metric_key, "#95a5a6")
        plt.hist(values, bins=histogram_bins, alpha=0.8, color=color, edgecolor='black', linewidth=1.2)
        plt.axvline(x=2.0, color='#e67e22', linestyle='--', linewidth=2.5, label='2Å')
        plt.axvline(x=5.0, color='#9b59b6', linestyle='-.', linewidth=2.5, label='5Å')
        plt.xlabel('RMSD (Å)', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.title(f'RMSD Distribution for {split_name_key} ({metric_key.capitalize()})', fontsize=16, fontweight='bold')
        
        # Statistics
        mean_val, median_val = np.mean(values), np.median(values)
        min_val, max_val = np.min(values), np.max(values)
        outliers_50 = sum(1 for v in values if v > 50.0)
        plt.text(0.95, 0.95, f'Mean: {mean_val:.2f}Å\nMedian: {median_val:.2f}Å\nMin: {min_val:.2f}Å\nMax: {max_val:.2f}Å\n>50Å: {outliers_50}', 
                transform=plt.gca().transAxes, ha='right', va='top', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'), fontsize=12)
        
        s2A = sum(1 for v in values if v <= 2.0) / len(values) * 100
        s5A = sum(1 for v in values if v <= 5.0) / len(values) * 100
        plt.text(0.05, 0.95, f'≤2Å: {s2A:.1f}%\n≤5Å: {s5A:.1f}%', 
                transform=plt.gca().transAxes, ha='left', va='top', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'), fontsize=12)
        
        plt.xlim(left=0, right=max(10, min(max(values) + 1 if values else 10, 20)))
        plt.legend(loc='center right', frameon=True, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(output_plot_dir, f"rmsd_dist_{split_name_key}_{metric_key}_{ts}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Saved RMSD plot: {plot_file}")
        plt.close()

def save_comparison_table_fixed(all_benchmark_metrics: Dict, output_results_dir: str) -> None:
    """Fixed version of comparison table generation."""
    header = [
        "Split", "N_PDBs", "TotalSuccess(%)", 
        "Method", "MeanRMSD(Å)", "MedianRMSD(Å)", "%≤2Å", "%≤5Å", "MeanScore"
    ]
    table_rows = [header]
    
    for split_key, metrics_data in all_benchmark_metrics.items():
        if not metrics_data or not metrics_data.get("per_method_stats"): 
            continue
        
        total_pdbs = metrics_data.get("total", 0)
        successful_runs = metrics_data.get("successful", 0)
        overall_success_rate = (successful_runs / total_pdbs * 100) if total_pdbs > 0 else 0
        
        for method_name, stats in metrics_data["per_method_stats"].items():
            row = [
                split_key,
                total_pdbs,
                f"{overall_success_rate:.1f}",
                method_name.capitalize(),
                f"{stats.get('mean_rmsd', 'N/A'):.2f}" if isinstance(stats.get('mean_rmsd'), float) else 'N/A',
                f"{stats.get('median_rmsd', 'N/A'):.2f}" if isinstance(stats.get('median_rmsd'), float) else 'N/A',
                f"{stats.get('perc_below_2A', 'N/A'):.1f}" if isinstance(stats.get('perc_below_2A'), float) else 'N/A',
                f"{stats.get('perc_below_5A', 'N/A'):.1f}" if isinstance(stats.get('perc_below_5A'), float) else 'N/A',
                f"{stats.get('mean_score', 'N/A'):.3f}" if isinstance(stats.get('mean_score'), float) else 'N/A'
            ]
            table_rows.append(row)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(table_rows[1:], columns=table_rows[0])
    
    os.makedirs(output_results_dir, exist_ok=True)
    
    csv_file = os.path.join(output_results_dir, f"benchmark_summary_{ts}.csv")
    df.to_csv(csv_file, index=False)
    print(f"Saved summary table: {csv_file}")
    
    md_file = os.path.join(output_results_dir, f"benchmark_summary_{ts}.md")
    with open(md_file, "w") as f:
        f.write(f"# Benchmark Summary ({ts})\n\n")
        f.write(df.to_markdown(index=False))
    
    print(f"Saved summary table: {md_file}")
    print("\nBenchmark Summary:")
    print(df.to_string(index=False))

def analyze_mcs_patterns(results_data: Dict) -> Dict:
    """Analyze MCS patterns from the results data."""
    mcs_pattern_analysis = {
        "total_successful": 0,
        "smarts_patterns": defaultdict(int),
        "atom_count_distribution": defaultdict(int),
        "bond_count_distribution": defaultdict(int),
        "similarity_score_distribution": defaultdict(int),
        "pattern_statistics": {}
    }
    
    for split_name, split_data in results_data.items():
        if "results" not in split_data:
            continue
            
        for pdb_id, pdb_data in split_data["results"].items():
            if not pdb_data.get("success") or not pdb_data.get("mcs_info"):
                continue
                
            mcs_pattern_analysis["total_successful"] += 1
            mcs_info = pdb_data["mcs_info"]
            
            # Count SMARTS patterns
            smarts = mcs_info.get("smarts", "")
            if smarts:
                mcs_pattern_analysis["smarts_patterns"][smarts] += 1
            
            # Distribute atom counts
            atom_count = mcs_info.get("atom_count", 0)
            if atom_count > 0:
                mcs_pattern_analysis["atom_count_distribution"][atom_count] += 1
            
            # Distribute bond counts
            bond_count = mcs_info.get("bond_count", 0)
            if bond_count > 0:
                mcs_pattern_analysis["bond_count_distribution"][bond_count] += 1
            
            # Distribute similarity scores
            similarity_score = mcs_info.get("similarity_score", 0.0)
            if similarity_score > 0:
                score_bucket = round(similarity_score, 1)  # Round to nearest 0.1
                mcs_pattern_analysis["similarity_score_distribution"][score_bucket] += 1
    
    # Calculate statistics
    if mcs_pattern_analysis["total_successful"] > 0:
        atom_counts = list(mcs_pattern_analysis["atom_count_distribution"].keys())
        bond_counts = list(mcs_pattern_analysis["bond_count_distribution"].keys())
        
        mcs_pattern_analysis["pattern_statistics"] = {
            "mean_atom_count": np.mean(atom_counts) if atom_counts else 0,
            "median_atom_count": np.median(atom_counts) if atom_counts else 0,
            "mean_bond_count": np.mean(bond_counts) if bond_counts else 0,
            "median_bond_count": np.median(bond_counts) if bond_counts else 0,
            "most_common_pattern": max(mcs_pattern_analysis["smarts_patterns"].items(), key=lambda x: x[1])[0] if mcs_pattern_analysis["smarts_patterns"] else "None",
            "unique_patterns": len(mcs_pattern_analysis["smarts_patterns"])
        }
    
    return mcs_pattern_analysis

def process_benchmark_results(results_file: str, output_dir: Optional[str] = None) -> None:
    """Process benchmark results and generate all analysis outputs."""
    
    # Load results data
    print(f"Loading results from: {results_file}")
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Create output directory
    if output_dir is None:
        results_dir = Path(results_file).parent
        output_dir = results_dir / "processed"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Extract results data
    results_data = data.get("results_data", {})
    if not results_data:
        print("ERROR: No results_data found in the JSON file!")
        return
    
    print(f"Found data for splits: {list(results_data.keys())}")
    
    # Calculate metrics for each split
    all_benchmark_metrics = {}
    
    for split_name, split_data in results_data.items():
        print(f"\nProcessing {split_name}...")
        
        # Calculate metrics
        metrics = calculate_metrics_fixed(split_data)
        all_benchmark_metrics[split_name] = metrics
        
        print(f"  Total PDBs: {metrics['total']}")
        print(f"  Successful: {metrics['successful']}")
        print(f"  Failed: {metrics['failed']}")
        print(f"  Methods found: {list(metrics.get('per_method_stats', {}).keys())}")
        
        # Generate plots
        plot_rmsd_distribution_fixed(split_data, split_name, str(plots_dir))
    
    # Generate summary table
    print(f"\nGenerating summary table...")
    save_comparison_table_fixed(all_benchmark_metrics, str(output_dir))
    
    # Analyze MCS patterns
    print(f"\nAnalyzing MCS patterns...")
    mcs_analysis = analyze_mcs_patterns(results_data)
    
    # Save MCS analysis
    mcs_file = output_dir / f"mcs_pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(mcs_file, 'w') as f:
        json.dump(dict(mcs_analysis), f, indent=2, default=str)
    print(f"Saved MCS analysis: {mcs_file}")
    
    # Print MCS summary
    print(f"\nMCS Pattern Summary:")
    print(f"  Total successful molecules: {mcs_analysis['total_successful']}")
    print(f"  Unique SMARTS patterns: {mcs_analysis['pattern_statistics'].get('unique_patterns', 0)}")
    print(f"  Mean atom count: {mcs_analysis['pattern_statistics'].get('mean_atom_count', 0):.1f}")
    print(f"  Most common pattern: {mcs_analysis['pattern_statistics'].get('most_common_pattern', 'None')}")
    
    print(f"\n✅ Processing complete! Results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Process benchmark results JSON file")
    parser.add_argument("results_file", help="Path to benchmark results JSON file")
    parser.add_argument("--output-dir", "-o", help="Output directory for processed results")
    parser.add_argument("--plots-only", action="store_true", help="Generate only plots")
    parser.add_argument("--summary-only", action="store_true", help="Generate only summary tables")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"ERROR: Results file not found: {args.results_file}")
        sys.exit(1)
    
    try:
        process_benchmark_results(args.results_file, args.output_dir)
    except Exception as e:
        print(f"ERROR: Failed to process results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 