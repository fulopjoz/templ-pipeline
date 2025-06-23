# TEMPL Pipeline

Template-based protein–ligand pose prediction with a single command-line interface **and** a Streamlit web app.

---

## Why TEMPL?
TEMPL leverages **ligand similarity** and **template superposition** instead of exhaustive docking or deep neural networks. For familiar chemical space it provides fast, accurate poses with minimal compute.

* Alignment driven by maximal common substructure (MCS)
* Constrained conformer generation (ETKDG v3)
* Shape / pharmacophore scoring for pose selection
* Built-in benchmarks (Polaris, time-split PDBbind)
* CPU-only by default; GPU optional for protein embeddings

---

## 🚀 Smart One-Click Installation

After cloning this repository, run **one command**:

```bash
git clone https://github.com/fulopjoz/templ-pipeline
cd templ-pipeline
./setup_env_smart.sh    # That's it! 🎉
```

### 🎯 **What the Smart Installer Does:**

1. **🔍 Detects your hardware** - CPU cores, RAM, GPU availability
2. **🎛️ Recommends optimal configuration** - CPU-minimal, CPU-optimized, or GPU-enabled
3. **📦 Installs tailored dependencies** - No bloat! (PyTorch GPU libs = 5GB+ vs 500MB CPU-only)
4. **✅ Verifies everything works** - Tests CLI and imports
5. **🚀 Ready to use immediately** - `templ --help` works right away

### 🔧 **Installation Options:**

```bash
# Auto-detect hardware and install optimally (recommended)
./setup_env_smart.sh

# Force lightweight CPU-only installation (~500MB)
./setup_env_smart.sh --cpu-only

# Force GPU installation (if auto-detection fails)
./setup_env_smart.sh --gpu-force

# Minimal server installation (no web interface)
./setup_env_smart.sh --minimal

# Benchmark performance after installation
./setup_env_smart.sh --benchmark
```

**Returning later?** Simply activate the environment:
```bash
source .venv/bin/activate
```

### 💡 **Why Smart Installation?**

- **🏃‍♂️ Efficient**: CPU-minimal (~500MB) vs full AI stack (8GB+)
- **⚡ Hardware-aware**: Only installs GPU deps if GPU detected
- **🔧 Optimized**: Right-sized models (ESM-2 150M for CPU, larger for GPU)
- **🚫 No bloat**: Skip unnecessary ML libraries for basic docking

### Alternative Installation (Advanced Users)

<details>
<summary>Click to expand manual installation options</summary>

#### Using uv manually:
```bash
curl -Ls https://astral.sh/uv/install.sh | bash
uv venv .venv && source .venv/bin/activate
uv pip install -e "."                     # CPU-minimal
uv pip install -e ".[ai-cpu,web]"         # CPU-optimized + web
uv pip install -e ".[ai-gpu,web]"         # GPU-enabled
```

#### Using pip:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e "."
```

#### Using conda (if RDKit issues):
```bash
conda create -n templ python=3.10 rdkit -c conda-forge
conda activate templ
pip install -e "."
```
</details>

---

## Dataset Setup
TEMPL itself ships **no PDB structures or ligands**. Register and download datasets manually, then place them exactly as shown.

### PDBbind v2020
1. In this directory `templ_pipeline/data/PDBBind/` (case exact).
2. Unpack the following archives inside it so the layout becomes:
```
PDBBind/
├─ PDBbind_v2020_refined/refined-set/<PDB>/
└─ PDBbind_v2020_other_PL/v2020-other-PL/<PDB>/
```
TEMPL assumes this layout; if you prefer a different location pass `--data-root` to CLI commands.

### Polaris benchmark
Pre-processed SDFs plus metadata are already included under
`templ_pipeline/benchmark/data/polaris/`.

---

## Quick Start
```bash
# 1 line pose prediction
templ run \
  --protein-file examples/1a1c_protein.pdb \
  --ligand-smiles "CN(C)C(=O)Nc1cccc(c1)C2CCN(CC2)C" \
  --output poses.sdf
```

### Common CLI Commands
| Command | Purpose |
|---------|---------|
| `embed` | Generate or cache protein embeddings (ESM-2) |
| `find-templates` | K-NN template search in PDBbind |
| `generate-poses` | Constrained conformer generation & ranking |
| `run` | One-shot pipeline (`embed → search → pose`) |
| `benchmark` | Reproduce paper benchmarks (Polaris, time-split) |

Use `templ --help` or `templ <command> --help` for all options.

---

## Streamlit Web App
```python
python run_streamlit_app.py
```
* drag-and-drop PDB + SMILES or SDF
* download best poses as SDF

---

## Benchmarking Examples
```bash
# Polaris challenge (CPU-only, 8 workers)
templ benchmark polaris --n-workers 8 --n-conformers 200

# PDBbind time-split (ensure PDBBind/ downloaded first)
# To avoid nested parallelism, keep internal workers = 1 when using many benchmark workers.
templ benchmark time-split --n-workers 8 --pipeline-workers 1
```
---

## Development & Tests
```bash
./setup_env_uv.sh --dev  # Install with dev dependencies
pytest -q
```

---

