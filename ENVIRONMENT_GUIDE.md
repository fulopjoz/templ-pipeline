# TEMPL Pipeline Environment Management Guide

Complete guide for setting up, managing, and troubleshooting TEMPL Pipeline environments.

## Quick Start

**One-command setup:**
```bash
source setup_templ_env.sh
```

That's it! The script auto-detects your hardware and installs the optimal configuration.

## Installation Modes

### Auto Detection (Recommended)
```bash
source setup_templ_env.sh --auto  # Default behavior
```
- Detects CPU cores, RAM, GPU availability
- Chooses optimal installation profile
- Balances features vs resource usage

### Specific Modes

| Mode | Command | Use Case | Dependencies |
|------|---------|----------|--------------|
| **CPU-only** | `--cpu-only` | Lightweight, servers | Core only (~50MB) |
| **Web** | `--web` | Standard usage | Core + Streamlit UI |
| **Full** | `--full` | Power users | Core + Web + Embedding acceleration |
| **GPU Force** | `--gpu-force` | Force GPU install | Full with GPU acceleration |
| **Minimal** | `--minimal` | Containers, CI/CD | Core libraries only |
| **Dev** | `--dev` | Contributors | Everything + dev tools |

### Examples
```bash
# Lightweight server installation
source setup_templ_env.sh --cpu-only

# Force GPU installation (if auto-detection fails)
source setup_templ_env.sh --gpu-force

# Development environment
source setup_templ_env.sh --dev

# Use pinned versions from requirements.txt
source setup_templ_env.sh --web --use-requirements
```

## Dependency Approaches

### 1. PyProject.toml (Recommended)
- **Flexible versions** - gets latest compatible versions
- **Modular extras** - install only what you need
- **Fast updates** - always gets bug fixes

```bash
# Uses pyproject.toml by default
source setup_templ_env.sh --web
```

**Dependency groups:**
```toml
[project]
dependencies = [...]  # Core always installed

[project.optional-dependencies]
web = ["streamlit", "stmol", ...]           # Web interface
full = ["torch", "transformers", ...]       # Embedding acceleration  
dev = ["pytest", "black", ...]             # Development tools
```

### 2. Requirements.txt (Stable)
- **Pinned versions** - exact reproducible builds
- **Tested combinations** - known to work together
- **Slower updates** - more stable, less frequent changes

```bash
# Force use of requirements.txt
source setup_templ_env.sh --web --use-requirements
```

### When to use which?

| Use pyproject.toml when: | Use requirements.txt when: |
|--------------------------|----------------------------|
| Development work | Production deployments |
| Want latest features | Need reproducible builds |
| Personal installations | Docker containers |
| Contributing to project | CI/CD pipelines |

## Environment Management

### Activation
```bash
# After setup, activate for new sessions:
source .templ/bin/activate

# Check if activated (should show (.templ) in prompt)
which python  # Should point to .templ/bin/python
```

### Deactivation
```bash
deactivate
```

### Environment Information
```bash
# Show environment details
python -c "import templ_pipeline; print(f'Version: {templ_pipeline.__version__}')"
pip list | grep templ
python -c "import templ_pipeline; print(templ_pipeline.__file__)"

# Show installed extras
pip show templ-pipeline
```

### Updating
```bash
# Update to latest versions (pyproject.toml approach)
source .templ/bin/activate
pip install -e .[web] --upgrade

# Update from requirements.txt
pip install -r requirements.txt --upgrade
```

### Rebuilding
```bash
# Complete rebuild
rm -rf .templ
source setup_templ_env.sh
```

## Hardware Optimization

### CPU-Only Systems
```bash
source setup_templ_env.sh --cpu-only
```
- Minimal memory usage (~2GB)
- No GPU dependencies
- Optimized for CPU inference
- Good for: servers, containers, laptops

### GPU Systems
```bash
source setup_templ_env.sh --full  # Auto-detects GPU
# or force if detection fails:
source setup_templ_env.sh --gpu-force
```
- Includes PyTorch with CUDA
- GPU-accelerated protein embeddings
- Faster template search
- Good for: workstations, cloud instances

### Memory-Constrained Systems
```bash
source setup_templ_env.sh --minimal
```
- Core functionality only
- No web interface
- CLI-only usage
- Good for: CI/CD, edge computing

## Troubleshooting

### Common Issues

#### "templ: command not found"
```bash
# Ensure environment is activated
source .templ/bin/activate

# Check if CLI is installed
pip show templ-pipeline | grep Scripts

# Reinstall CLI
pip install -e .
```

#### "Import Error" for modules
```bash
# Check which extras are installed
pip show templ-pipeline

# Reinstall with needed extras
pip install -e .[web]  # For web interface
pip install -e .[full] # For embedding features
```

#### GPU not detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Force GPU installation
source setup_templ_env.sh --gpu-force

# Test GPU in Python
python -c "import torch; print(torch.cuda.is_available())"
```

#### Python version issues
```bash
# Check Python version
python3 --version  # Must be 3.9+

# Use specific Python version
python3.10 -m venv .templ
source .templ/bin/activate
```

#### Memory errors during installation
```bash
# Use pre-compiled wheels
pip install --only-binary=all -e .[web]

# Or use requirements.txt (smaller install)
source setup_templ_env.sh --use-requirements
```

### Environment Diagnosis
```bash
# Run built-in diagnostics
python verify_environment.py

# Manual checks
python -c "
import sys
print(f'Python: {sys.version}')
print(f'Platform: {sys.platform}')

try:
    import templ_pipeline
    print('✓ TEMPL Pipeline installed')
    print(f'✓ Version: {templ_pipeline.__version__}')
except ImportError:
    print('✗ TEMPL Pipeline not found')

try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
    print(f'✓ CUDA available: {torch.cuda.is_available()}')
except ImportError:
    print('✗ PyTorch not installed')

try:
    import streamlit
    print(f'✓ Streamlit: {streamlit.__version__}')
except ImportError:
    print('✗ Streamlit not installed')
"
```

## Docker Usage

### Simple Container
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Setup TEMPL environment
RUN source setup_templ_env.sh --minimal --use-requirements

# Run application
CMD ["templ", "run", "--help"]
```

### Web Interface Container
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Setup web environment
RUN source setup_templ_env.sh --web --use-requirements

EXPOSE 8501
CMD ["python", "run_streamlit_app.py"]
```

## Performance Tips

### Speed Up Installation
```bash
# Use binary wheels only (faster)
pip install --only-binary=all -e .[web]

# Use conda for RDKit (much faster)
conda install -c conda-forge rdkit
pip install -e . --no-deps
```

### Optimize Runtime
```bash
# Set optimal number of workers
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

# For GPU systems
export CUDA_VISIBLE_DEVICES=0
```

### Monitor Resource Usage
```bash
# During installation
htop  # Watch CPU/memory usage

# During runtime  
nvidia-smi  # GPU usage
templ benchmark polaris --n-workers 4  # Test performance
```

## CI/CD Integration

### GitHub Actions
```yaml
- name: Setup TEMPL Environment
  run: |
    source setup_templ_env.sh --minimal --use-requirements
    source .templ/bin/activate
    python -c "import templ_pipeline; print(f'Version: {templ_pipeline.__version__}')"

- name: Run Tests
  run: |
    source .templ/bin/activate
    pytest tests/
```

### GitLab CI
```yaml
setup_environment:
  script:
    - source setup_templ_env.sh --minimal --use-requirements
    - source .templ/bin/activate
    - python -c "import templ_pipeline; print(f'Version: {templ_pipeline.__version__}')"
```

## Best Practices

### For Developers
- Use `--dev` mode for contribution work
- Always test with `pytest` before committing
- Use `pyproject.toml` approach for latest features

### For Production
- Use `--use-requirements` for reproducible builds
- Pin to specific versions in containers
- Test thoroughly before deployment

### For Users
- Start with `--auto` mode (default)
- Upgrade only when needed
- Use `--web` for regular usage

### For Servers
- Use `--minimal` or `--cpu-only` modes
- Disable GUI features
- Monitor resource usage

Remember: **Always use `source setup_templ_env.sh`** (not `./setup_templ_env.sh`) to properly activate the environment!
