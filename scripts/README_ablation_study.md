# Polaris Ablation Study Script

## Overview

The `run_polaris_ablation_study.sh` script performs a comprehensive ablation study on the TEMPL Polaris benchmark, testing various configurations and parameters to understand their impact on pose prediction performance.

## Location

The script is located in the `scripts/` directory and outputs all results to `benchmarks/polaris/`.

## Usage

```bash
# Run with default settings (uses all available CPU cores)
./scripts/run_polaris_ablation_study.sh

# Run with specific number of workers
./scripts/run_polaris_ablation_study.sh 8
```

## What it tests

The script runs the following ablation studies:

1. **Default Configuration**: MCS + 200 conformers + ComboTanimoto (cross-aligned templates)
2. **MERS PDB Reference**: Native templates only for MERS
3. **Alignment Metrics**: ShapeTanimoto and ColorTanimoto variants
4. **Unconstrained Embedding**: Without MCS constraints
5. **No Realignment**: Using AlignMol scores only
6. **MMFF94x Optimization**: Force field optimization
7. **Conformer Count Studies**: 1, 5, 10, 20, 50, 100 conformers

## Output

All results are saved to `benchmarks/polaris/`:

- Individual benchmark results: `templ_polaris_benchmark_results_*.json`
- Summary files: `ablation_study_summary_*.txt`, `*.csv`, `*.md`
- Log files: Individual logs for each experiment

## Requirements

- TEMPL Pipeline environment activated
- Polaris dataset available
- Sufficient disk space for results

## Notes

- The script creates the `benchmarks/polaris/` directory if it doesn't exist
- Each experiment runs independently and can be interrupted/resumed
- Summary files provide easy-to-read tables of all results 