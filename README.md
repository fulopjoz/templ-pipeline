# TEMPL Pipeline

Template-based protein-ligand pose prediction using similarity search and conformer generation.

## Installation

**Prerequisites:** Python 3.9+, RDKit

```bash
git clone <repository-url>
cd templ_pipeline
pip install -e .
```

## Quick Start

**CLI:**
```bash
templ run --protein-file protein.pdb --ligand-smiles "SMILES_STRING"
```

**Web Interface:**
```bash
streamlit run templ_pipeline/ui/app.py
```

## CLI Commands

| Command | Purpose |
|---------|---------|
| `embed` | Generate protein embeddings |
| `find-templates` | Find similar protein templates |
| `generate-poses` | Generate poses using specific templates |
| `run` | Complete pipeline with template discovery |
| `benchmark` | Run built-in benchmark suites |

Use `templ --help` or `templ <command> --help` for detailed options.

## Examples

**Generate embedding:**
```bash
templ embed --protein-file protein.pdb
```

**Find templates:**
```bash
templ find-templates --query protein.pdb --embedding-file embeddings.npz --num-templates 10
```

**Generate poses with specific template:**
```bash
templ generate-poses --protein-file protein.pdb --ligand-smiles "SMILES" --template-pdb 1a1e
```

**Full pipeline:**
```bash
templ run --protein-file protein.pdb --ligand-smiles "SMILES" --num-templates 100 --num-conformers 200
```

**Run benchmark:**
```bash
templ benchmark polaris --n-workers 8 --n-conformers 200
```

## Benchmarking

Built-in benchmarks available: `polaris`, `time-split`. Use `--quick` for fast testing.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT License
