# TEMPL Benchmark Results Guide

This README explains the structure and meaning of TEMPL benchmark results and JSON files.

## Directory Structure

```
benchmarks/
├── pdbbind/
│   ├── benchmark_timesplit_YYYYMMDD_HHMMSS/    # Individual benchmark run
│   │   ├── logs/                                # Execution logs
│   │   ├── raw_results/timesplit/              # Raw pipeline outputs
│   │   │   ├── results_test_YYYYMMDD_HHMMSS.jsonl    # Individual molecule results
│   │   │   ├── complete_results_YYYYMMDD_HHMMSS.json # Aggregated statistics
│   │   │   └── poses/                          # Generated molecular poses
│   │   └── summaries/                          # Processed summary files
│   │       ├── timesplit_benchmark_summary_*.json   # JSON summary
│   │       ├── timesplit_benchmark_summary_*.csv    # CSV summary
│   │       └── timesplit_benchmark_summary_*.md     # Markdown summary
│   └── polaris/                                # Polaris benchmark results
└── README.md                                   # This file
```

## Summary JSON Fields

### Core Metrics
- **`Benchmark`**: Type of benchmark ("Timesplit", "Polaris")
- **`Split`**: Dataset split ("test", "val", "train")
- **`Metric`**: Scoring method ("Combo", "Shape", "Color")
- **`Total_Targets`**: Total number of input molecules

### Success Rate Calculations
- **`Success_Rate_2A`**: Percentage of molecules with RMSD ≤ 2.0Å
- **`Success_Rate_2A_Explicit`**: Explicit calculation (e.g., "77/340")
- **`Success_Rate_5A`**: Percentage of molecules with RMSD ≤ 5.0Å  
- **`Success_Rate_5A_Explicit`**: Explicit calculation (e.g., "137/340")

```
Success Rate = Successful molecules / Pipeline attempted molecules
```
This includes pipeline failures (timeouts, MCS failures) in the denominator to avoid survivor bias.

### Pipeline Stage Classification

#### **`Pre_Pipeline_Excluded`**
Molecules that failed before entering the algorithm:
- Missing protein/ligand files
- Invalid SMILES/PDB structures  
- Data loading errors
- Network/file system issues

#### **`Pipeline_Filtered`** 
Molecules filtered during validation:
- **Peptides**: Large peptides (>8 residues)
- **Polysaccharides**: Complex sugars (>3 sugar rings)
- Molecule validation failures
- Invalid chemical structures

#### **`Pipeline_Attempted`**
Molecules that entered the main algorithm:
- **Pipeline_Successful**: Generated valid RMSD values
- **Pipeline_Failed**: Reached MCS stage but failed (timeouts, MCS failures, conformer errors)

### Filtering Statistics
- **`Peptide_Polysaccharide_Filtered`**: Count of filtered peptides/polysaccharides
- **`Template_Database_Filtering`**: Template database filtering statistics
- **`Filtering_Details`**: Breakdown of molecular filtering reasons

### RMSD Statistics
- **`Mean_RMSD`**: Average RMSD including pipeline failures (as 999.0Å)
- **`Median_RMSD`**: Median RMSD of successful molecules
- **`Targets_With_RMSD`**: Count of molecules with valid RMSD values

## 🔍 Exclusion Reasons

### **`database_empty`**
No templates available for the target protein:
- Template database contains 0 matching structures
- Usually indicates data availability issues
- **Stage**: `pre_pipeline_excluded`

### **`ligand_data_missing`** 
Target ligand data unavailable:
- Missing or corrupted SDF files
- SMILES parsing failures
- **Stage**: `pre_pipeline_excluded`

### **`timeout`**
Pipeline execution exceeded time limit:
- Default timeout: 300 seconds
- Indicates computational complexity
- **Stage**: `pipeline_attempted` (counts toward success rates)

### **`large_peptide`**
Target molecule is a large peptide:
- >8 amino acid residues
- Requires specialized conformational methods
- **Stage**: `pipeline_filtered`

### **`large_polysaccharide`**
Target molecule is a complex sugar:
- >3 sugar ring structures  
- Requires specialized carbohydrate methods
- **Stage**: `pipeline_filtered`

## Dataset Size Reporting (XYZ Numbers)

For scientific publications, use **`Pipeline_Attempted`** numbers:

```
Train: X molecules (pipeline_attempted in training set)
Val:   Y molecules (pipeline_attempted in validation set)  
Test:  Z molecules (pipeline_attempted in test set)
```

These represent molecules that actually attempted the algorithm, providing scientifically valid dataset sizes.

## Success Rate Formula

**Correct (Unbiased) Formula:**
```
Success Rate = Successful RMSD molecules / Pipeline attempted molecules
             = count(RMSD ≤ threshold) / (successes + pipeline_failures)
```

**Incorrect (Biased) Formula:**
```  
Success Rate = Successful RMSD molecules / Total input molecules
```

The formula includes pipeline failures in the denominator.

## CLI JSON Output Fields

Individual molecule results contain CLI JSON with these fields:

- **`success`**: CLI execution success (boolean)
- **`pipeline_stage`**: Stage classification ("pre_pipeline_excluded", "pipeline_filtered", "pipeline_attempted")  
- **`made_it_to_mcs`**: Reached MCS processing stage (boolean)
- **`total_templates_in_database`**: Available templates count
- **`templates_used_for_poses`**: Templates actually used
- **`rmsd_values`**: RMSD results by metric {combo: {rmsd: 1.2, score: 0.95}}
- **`template_database_stats`**: Template filtering statistics
- **`poses_count`**: Number of generated poses

## Example Interpretation

```json
{
  "Metric": "Combo",
  "Total_Targets": 363,
  "Success_Rate_2A": "23.8%", 
  "Success_Rate_2A_Explicit": "81/340",
  "Pipeline_Attempted": 340,
  "Pre_Pipeline_Excluded": 21,  
  "Pipeline_Filtered": 2
}
```

**Interpretation:**
- 363 molecules total in test set
- 21 excluded due to data issues (missing files, etc.)
- 2 filtered as peptides/polysaccharides  
- 340 molecules attempted the algorithm
- 81 of these 340 achieved ≤2.0Å RMSD
- Success rate: 81/340 = 23.8%

259 molecules (340-81) attempted the algorithm but failed due to timeouts, MCS failures, etc. These failures are included in the denominator.
