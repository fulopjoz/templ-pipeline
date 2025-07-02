# TEMPL Polaris Benchmark Summary (20250617_032212)

## Parameters
- Workers: 20
- Conformers: 200
- Molecule timeout: 180s
- Log level: WARNING

## Results

| Virus Type   | Dataset   | Template Source     |   Queries | Templates    | Success Rate (<2Å)   | Success Rate (<5Å)   |
|:-------------|:----------|:--------------------|----------:|:-------------|:---------------------|:---------------------|
| SARS         | Train     | SARS                |       770 | 769 (LOO)    | 72.5%                | 87.1%                |
| SARS         | Test      | SARS                |       119 | 770          | 64.7%                | 77.3%                |
| MERS         | Train     | MERS                |        17 | 16 (LOO)     | 35.3%                | 88.2%                |
| MERS         | Test      | MERS                |        76 | 17           | 15.8%                | 48.7%                |
| MERS         | Train     | MERS+SARS-aligned^T |        17 | 787 (17+770) | 47.1%                | 82.4%                |
| MERS         | Test      | MERS+SARS-aligned^T |        76 | 787 (17+770) | 55.3%                | 90.8%                |

## Notes
- ^T indicates cross-virus templates (SARS molecules aligned to MERS binding site)
- Success rates are based on the 'combo' metric (combination of shape and color scores)
- Training sets use leave-one-out evaluation
- Test sets use training templates of the specified type
