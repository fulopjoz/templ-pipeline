#!/bin/bash

# Polaris Benchmark Ablation Study Script
# Generates all variants needed for the ablation study table
# Usage: ./run_polaris_ablation_study.sh [N_WORKERS]

set -e

# Default configuration
N_WORKERS=""
WORKERS_ARG=""

# Parse command line arguments
if [ $# -eq 1 ]; then
    N_WORKERS="$1"
    WORKERS_ARG="--n-workers $1"
    echo "Using $N_WORKERS workers for all benchmark runs"
elif [ $# -gt 1 ]; then
    echo "Usage: $0 [N_WORKERS]"
    echo "Example: $0 8"
    exit 1
fi

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="$SCRIPT_DIR/../templ_pipeline/benchmark/polaris/benchmark.py"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_DIR="benchmarks/polaris"
RESULTS_DIR="$BASE_DIR/ablation_study_results_$TIMESTAMP"
RAW_RESULTS_DIR="$RESULTS_DIR/raw_results"
LOGS_DIR="$RESULTS_DIR/logs"
POSES_DIR="$RESULTS_DIR/poses"
SUMMARY_FILE="ablation_study_summary_$TIMESTAMP"

# Create results directory structure
mkdir -p "$RESULTS_DIR"
mkdir -p "$RAW_RESULTS_DIR"
mkdir -p "$LOGS_DIR"
mkdir -p "$POSES_DIR"

echo "=== Polaris Benchmark Ablation Study ==="
echo "Results will be saved to: $RESULTS_DIR"
echo "Summary will be saved to: $SUMMARY_FILE"
if [ -n "$N_WORKERS" ]; then
    echo "Using $N_WORKERS workers"
fi
echo "Started at: $(date)"
echo ""

# Function to run benchmark with specific parameters
run_benchmark() {
    local name="$1"
    local args="$2"
    local output_file="$RAW_RESULTS_DIR/${name}_${TIMESTAMP}.json"
    local log_file="$LOGS_DIR/${name}.log"
    local poses_subdir="$POSES_DIR/${name}"
    
    # Create poses subdirectory for this experiment
    mkdir -p "$poses_subdir"
    
    echo "Running: $name"
    echo "Command: python $BENCHMARK_SCRIPT $args $WORKERS_ARG --output-dir $RAW_RESULTS_DIR --poses-dir $poses_subdir"
    echo "Output: $output_file"
    echo "Log: $log_file"
    echo "Poses: $poses_subdir"
    
    python "$BENCHMARK_SCRIPT" $args $WORKERS_ARG --output-dir "$RAW_RESULTS_DIR" --poses-dir "$poses_subdir" > "$log_file" 2>&1
    
    # Find and copy the most recent results file
    latest_result=$(find "$RAW_RESULTS_DIR" -name "templ_polaris_benchmark_results_*.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$latest_result" ]; then
        cp "$latest_result" "$output_file"
        # Remove the original benchmark output file to keep directory clean
        rm "$latest_result"
        echo "Results saved to: $output_file"
    else
        echo "Warning: No results file found for $name"
    fi
    echo ""
}

# 1. DEFAULT: MCS + 200 conformers + ComboTanimoto (cross-aligned templates)
echo "=== 1. DEFAULT CONFIGURATION ==="
run_benchmark "default" "--test-only --n-conformers 200 --align-metric combo --template-source cross_aligned"

# 2. MERS PDB reference (native templates only for MERS)
echo "=== 2. MERS PDB REFERENCE ==="
run_benchmark "mers_native_only" "--test-only --n-conformers 200 --align-metric combo --template-source native"

# 3. Different alignment metrics
echo "=== 3. ALIGNMENT METRICS ==="
run_benchmark "shape_metric" "--test-only --n-conformers 200 --align-metric shape"
run_benchmark "color_metric" "--test-only --n-conformers 200 --align-metric color"

# 4. Unconstrained embedding
echo "=== 4. UNCONSTRAINED EMBEDDING ==="
run_benchmark "unconstrained" "--test-only --n-conformers 200 --align-metric combo --unconstrained"
run_benchmark "unconstrained_shape" "--test-only --n-conformers 200 --align-metric shape --unconstrained"
run_benchmark "unconstrained_color" "--test-only --n-conformers 200 --align-metric color --unconstrained"

# 5. No realignment
echo "=== 5. NO REALIGNMENT ==="
run_benchmark "no_realign" "--test-only --n-conformers 200 --align-metric combo --no-realign"

# 6. MMFF94x force field optimization
echo "=== 6. MMFF94X OPTIMIZATION ==="
run_benchmark "mmff94x" "--test-only --n-conformers 200 --align-metric combo --enable-optimization"

# 7. Conformer count variations
echo "=== 7. CONFORMER COUNT STUDIES ==="
for n_confs in 1 5 10 20 50 100; do
    run_benchmark "confs_${n_confs}" "--test-only --n-conformers $n_confs --align-metric combo"
done

echo "=== All benchmark runs completed at $(date) ==="
echo ""

# Generate summary
echo "=== GENERATING SUMMARY ==="
python3 << EOF
import json
import glob
import os
from pathlib import Path

def extract_success_rate(filepath, virus_type, metric="combo", template_source="auto"):
    """Extract success rate from JSON results file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Determine which result key to use based on template source
        if virus_type == "MERS":
            if template_source == "native":
                target_key = "MERS_test_native"
            elif template_source == "cross_aligned":
                target_key = "MERS_test_cross"
            else:  # auto - prefer cross_aligned for better performance comparison
                target_key = "MERS_test_cross" if "MERS_test_cross" in data.get("results", {}) else "MERS_test_native"
        else:
            target_key = f"{virus_type}_test_native"
        
        # Look for specific test results
        result = data.get("results", {}).get(target_key)
        if result:
            summary = result.get("summary", {})
            success_rates = summary.get("success_rates", {})
            if metric in success_rates:
                return success_rates[metric].get("rate_2A", 0.0)
        return 0.0
    except:
        return 0.0

def extract_combined_rate(filepath, metric="combo", template_source="auto"):
    """Extract combined MERS+SARS success rate"""
    mers_rate = extract_success_rate(filepath, "MERS", metric, template_source)
    sars_rate = extract_success_rate(filepath, "SARS", metric, template_source)
    
    # Weighted average based on dataset sizes (MERS: 97, SARS: 98)
    total_molecules = 97 + 98
    combined_rate = (mers_rate * 97 + sars_rate * 98) / total_molecules
    return combined_rate

# Results mapping
experiments = {
    "MCS+200 confs+ComboTanimoto (default)": "default_*.json",
    "MERS PDB reference": "mers_native_only_*.json", 
    "ShapeTanimoto": "shape_metric_*.json",
    "ColorTanimoto": "color_metric_*.json",
    "Unconstrained embedding": "unconstrained_*.json",
    "Unconstrained embedding + ShapeTanimoto": "unconstrained_shape_*.json",
    "Unconstrained embedding + ColorTanimoto": "unconstrained_color_*.json",
    "No realignment": "no_realign_*.json",
    "MMFF94x": "mmff94x_*.json",
    "100 confs": "confs_100_*.json",
    "50 confs": "confs_50_*.json", 
    "20 confs": "confs_20_*.json",
    "10 confs": "confs_10_*.json",
    "5 confs": "confs_5_*.json",
    "1 conf": "confs_1_*.json"
}

# Prepare output files
import csv
from datetime import datetime
import os

# Create results directory if it doesn't exist
results_dir = "$RESULTS_DIR"
os.makedirs(results_dir, exist_ok=True)

summary_base = os.path.join(results_dir, "$SUMMARY_FILE")
txt_file = f"{summary_base}.txt"
csv_file = f"{summary_base}.csv"
md_file = f"{summary_base}.md"
json_file = f"{summary_base}.json"

# Collect all results first
results_data = []
header = ["Settings", "MERS < 2 Å", "SARS < 2 Å", "MERS+SARS < 2 Å"]

for exp_name, pattern in experiments.items():
    files = glob.glob(os.path.join("$RAW_RESULTS_DIR", pattern))
    if files:
        filepath = files[0]  # Take first match
        
        # Determine metric based on experiment
        metric = "combo"  # default
        if "Shape" in exp_name:
            metric = "shape"
        elif "Color" in exp_name:
            metric = "color"
            
        # Determine template source based on experiment
        template_source = "auto"  # default - uses cross_aligned if available
        if "MERS PDB reference" in exp_name:
            template_source = "native"
            
        mers_rate = extract_success_rate(filepath, "MERS", metric, template_source)
        sars_rate = extract_success_rate(filepath, "SARS", metric, template_source)
        combined_rate = extract_combined_rate(filepath, metric, template_source)
        
        results_data.append([exp_name, f"{mers_rate:.1f}", f"{sars_rate:.1f}", f"{combined_rate:.1f}"])
    else:
        results_data.append([exp_name, "N/A", "N/A", "N/A"])

# Generate timestamp
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 1. Console output
print("=== ABLATION STUDY RESULTS ===")
print(f"{'Settings':<45} {'MERS < 2 Å':<12} {'SARS < 2 Å':<12} {'MERS+SARS < 2 Å':<15}")
print("-" * 85)
for row in results_data:
    print(f"{row[0]:<45} {row[1]:<12} {row[2]:<12} {row[3]:<15}")
print("-" * 85)
print(f"Generated at: {os.getcwd()}")

# 2. Text file output
with open(txt_file, 'w') as f:
    f.write("TEMPL Polaris Benchmark - Ablation Study Results\\n")
    f.write(f"Generated: {timestamp}\\n")
    f.write(f"Location: {os.getcwd()}\\n")
    f.write("=" * 85 + "\\n\\n")
    f.write(f"{'Settings':<45} {'MERS < 2 Å':<12} {'SARS < 2 Å':<12} {'MERS+SARS < 2 Å':<15}\\n")
    f.write("-" * 85 + "\\n")
    for row in results_data:
        f.write(f"{row[0]:<45} {row[1]:<12} {row[2]:<12} {row[3]:<15}\\n")
    f.write("-" * 85 + "\\n")
    f.write(f"\\nTotal configurations tested: {len(results_data)}\\n")

# 3. CSV file output
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["# TEMPL Polaris Benchmark - Ablation Study Results"])
    writer.writerow([f"# Generated: {timestamp}"])
    writer.writerow([f"# Location: {os.getcwd()}"])
    writer.writerow([])  # Empty row
    writer.writerow(header)
    writer.writerows(results_data)

# 4. Markdown file output
with open(md_file, 'w') as f:
    f.write("# TEMPL Polaris Benchmark - Ablation Study Results\\n\\n")
    f.write(f"**Generated:** {timestamp}\\n\\n")
    f.write(f"**Location:** `{os.getcwd()}`\\n\\n")
    f.write("## Results Table\\n\\n")
    f.write("| Settings | MERS < 2 Å | SARS < 2 Å | MERS+SARS < 2 Å |\\n")
    f.write("|----------|------------|------------|------------------|\\n")
    for row in results_data:
        f.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |\\n")
    f.write(f"\\n**Total configurations tested:** {len(results_data)}\\n")

# 5. JSON file output
summary_json = {
    "metadata": {
        "generated": timestamp,
        "location": os.getcwd(),
        "total_configurations": len(results_data)
    },
    "results": []
}

for row in results_data:
    summary_json["results"].append({
        "settings": row[0],
        "mers_success_rate": row[1],
        "sars_success_rate": row[2], 
        "combined_success_rate": row[3]
    })

with open(json_file, 'w') as f:
    json.dump(summary_json, f, indent=2)

print(f"\\nResults saved to:")
print(f"   JSON format: {json_file}")
print(f"   Text format: {txt_file}")
print(f"   CSV format:  {csv_file}")
print(f"   Markdown:    {md_file}")
EOF

echo ""
echo "Summary generation completed!"

# Cleanup any remaining temporary files in raw_results
echo "Cleaning up temporary files..."
find "$RAW_RESULTS_DIR" -name "templ_polaris_benchmark_results_*.json" -delete 2>/dev/null || true

# Clean up any stray experiment subdirectories that might be created
find "$RAW_RESULTS_DIR" -type d -name "*_test_*" -exec rm -rf {} + 2>/dev/null || true
find "$RAW_RESULTS_DIR" -type d -name "*_train_*" -exec rm -rf {} + 2>/dev/null || true

# Clean up any stray benchmark_poses directories in root or other locations
find . -maxdepth 1 -type d -name "benchmark_poses_polaris_*" -exec rm -rf {} + 2>/dev/null || true

echo ""
echo "=== RESULTS ORGANIZATION ==="
echo "Main results directory: $RESULTS_DIR"
echo "   ├── Raw results (JSON): $RAW_RESULTS_DIR/"
echo "   ├── Log files: $LOGS_DIR/"
echo "   ├── Poses (SDF files): $POSES_DIR/"
echo "   └── Summary files:"
echo "       ├── ${SUMMARY_FILE}.json  (Structured JSON format)"
echo "       ├── ${SUMMARY_FILE}.txt   (Human-readable text)"
echo "       ├── ${SUMMARY_FILE}.csv   (Excel/CSV format)"  
echo "       └── ${SUMMARY_FILE}.md    (Markdown format)"
echo ""
echo "Total files generated:"
echo "   $(find "$RAW_RESULTS_DIR" -name "*.json" | wc -l) raw result files"
echo "   $(find "$LOGS_DIR" -name "*.log" | wc -l) log files"
echo "   $(find "$POSES_DIR" -name "*.sdf" 2>/dev/null | wc -l) pose files"
echo "   4 summary files (JSON, TXT, CSV, MD)"
echo ""
echo "Ablation study complete!"