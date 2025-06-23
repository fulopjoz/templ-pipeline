#!/usr/bin/env bash
# setup_env_smart.sh – Intelligent TEMPL Pipeline installation with hardware detection
# Usage: ./setup_env_smart.sh [--cpu-only] [--gpu-force] [--minimal] [--benchmark]
# 
# This script will:
#   1. Detect hardware capabilities (CPU, GPU, memory)
#   2. Recommend optimal installation configuration
#   3. Install dependencies tailored to your hardware
#   4. Benchmark performance (optional)
#   5. Verify installation and provide usage instructions

set -euo pipefail
IFS=$'\n\t'

#------------- helper logging -------------------------------------------------
CLR_RESET="\033[0m"
CLR_GREEN="\033[32m"
CLR_CYAN="\033[36m"
CLR_RED="\033[31m"
CLR_YELLOW="\033[33m"
CLR_BLUE="\033[34m"
CLR_MAGENTA="\033[35m"

info()  { echo -e "${CLR_CYAN}[INFO]${CLR_RESET} $*"; }
ok()    { echo -e "${CLR_GREEN}[ OK ]${CLR_RESET} $*"; }
warn()  { echo -e "${CLR_YELLOW}[WARN]${CLR_RESET} $*"; }
err()   { echo -e "${CLR_RED}[ERR]${CLR_RESET} $*" >&2; }
highlight() { echo -e "${CLR_MAGENTA}[HIGHLIGHT]${CLR_RESET} $*"; }

abort() { 
    err "$1"
    if [[ -d ".venv" && "${CREATED_VENV:-}" == "true" ]]; then
        warn "Cleaning up failed installation..."
        rm -rf .venv
    fi
    exit 1
}

#------------- parse command line arguments -----------------------------------
FORCE_CPU_ONLY=false
FORCE_GPU=false
MINIMAL_INSTALL=false
RUN_BENCHMARK=false
INTERACTIVE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-only)
            FORCE_CPU_ONLY=true
            shift
            ;;
        --gpu-force)
            FORCE_GPU=true
            shift
            ;;
        --minimal)
            MINIMAL_INSTALL=true
            shift
            ;;
        --benchmark)
            RUN_BENCHMARK=true
            shift
            ;;
        --non-interactive)
            INTERACTIVE=false
            shift
            ;;
        --help|-h)
            cat << EOF
TEMPL Pipeline Smart Installation Script

Usage: $0 [OPTIONS]

OPTIONS:
    --cpu-only          Force CPU-only installation (no GPU dependencies)
    --gpu-force         Force GPU installation even if GPU not detected
    --minimal           Install minimal dependencies only
    --benchmark         Run performance benchmarks after installation
    --non-interactive   Skip user prompts and use recommended settings
    --help, -h          Show this help message

EXAMPLES:
    $0                          # Auto-detect hardware and install optimally
    $0 --cpu-only              # Force lightweight CPU-only installation
    $0 --gpu-force --benchmark  # Force GPU setup and benchmark performance
    $0 --minimal               # Minimal installation for servers

EOF
            exit 0
            ;;
        *)
            warn "Unknown option: $1"
            warn "Use --help for usage information"
            shift
            ;;
    esac
done

#------------- banner ---------------------------------------------------------
cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        🏛️  TEMPL Pipeline Smart Setup                        ║
║                                                                               ║
║   Intelligent installation with hardware detection and dependency optimization ║
╚═══════════════════════════════════════════════════════════════════════════════╝

EOF

#------------- python version check -------------------------------------------
PY_MIN=3.9
if ! command -v python3 >/dev/null 2>&1; then
    abort "Python 3 not found. Please install Python >=${PY_MIN}."
fi

PY_VER=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null || echo "0.0")
if [[ -z "$PY_VER" ]] || ! python3 -c "
import sys
major, minor = sys.version_info[:2]
required = tuple(map(int, '${PY_MIN}'.split('.')))
current = (major, minor)
exit(0 if current >= required else 1)
" 2>/dev/null; then
    abort "Python ${PY_MIN}+ required (found ${PY_VER}). Please upgrade Python."
fi

ok "Python ${PY_VER} found"

#------------- uv installation -------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    info "Installing uv package manager..."
    if ! curl -fsSL https://astral.sh/uv/install.sh | bash; then
        abort "Failed to install uv. Please check your internet connection and try again."
    fi
    
    # uv can install to different locations - check both
    UV_PATHS=("$HOME/.cargo/bin" "$HOME/.local/bin")
    UV_FOUND=false
    
    for uv_path in "${UV_PATHS[@]}"; do
        if [[ -f "$uv_path/uv" ]]; then
            export PATH="$uv_path:$PATH"
            UV_FOUND=true
            break
        fi
    done
    
    if [[ "$UV_FOUND" == "false" ]] || ! command -v uv >/dev/null 2>&1; then
        abort "uv installation failed. Please install manually: https://docs.astral.sh/uv/getting-started/installation/"
    fi
    ok "uv installed successfully"
else
    ok "uv already available"
fi

#------------- hardware detection ---------------------------------------------
info "🔍 Detecting hardware configuration..."

# Install minimal dependencies for hardware detection
ENV_DIR=".venv"
if [[ ! -d "$ENV_DIR" ]]; then
    info "Creating temporary virtual environment for hardware detection..."
    uv venv "$ENV_DIR" || abort "Failed to create virtual environment"
    CREATED_VENV=true
fi

# Activate environment
if [[ -f "$ENV_DIR/bin/activate" ]]; then
    source "$ENV_DIR/bin/activate"
else
    abort "Virtual environment activation script not found"
fi

# Install minimal dependencies for hardware detection
if ! python -c "import psutil" >/dev/null 2>&1; then
    info "Installing basic dependencies for hardware detection..."
    uv pip install psutil >/dev/null 2>&1 || abort "Failed to install basic dependencies"
fi

# Run hardware detection
HARDWARE_DETECTION_OUTPUT=$(python3 -c "
import json
import sys
import os
sys.path.insert(0, os.getcwd())

try:
    from templ_pipeline.core.hardware_detection import get_hardware_recommendation
    recommendation = get_hardware_recommendation()
    print(json.dumps(recommendation, indent=2))
except Exception as e:
    # Fallback basic detection
    import psutil
    import subprocess
    
    # Basic GPU detection
    gpu_available = False
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        gpu_available = result.returncode == 0
    except:
        pass
    
    cpu_count = psutil.cpu_count()
    ram_gb = psutil.virtual_memory().total / (1024**3)
    
    if gpu_available and ram_gb >= 16:
        config = 'gpu-medium'
    elif ram_gb >= 16 and cpu_count >= 8:
        config = 'cpu-optimized'
    else:
        config = 'cpu-minimal'
    
    fallback = {
        'hardware_info': {
            'cpu_count': cpu_count,
            'total_ram_gb': ram_gb,
            'gpu_available': gpu_available,
            'recommended_config': config
        },
        'recommended_installation': {
            'command': f'uv pip install -e .[{\"ai-cpu,web\" if config != \"cpu-minimal\" else \"\"}]',
            'description': f'{config} installation'
        }
    }
    print(json.dumps(fallback, indent=2))
")

# Parse hardware detection results
CPU_COUNT=$(echo "$HARDWARE_DETECTION_OUTPUT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['hardware_info']['cpu_count'])")
RAM_GB=$(echo "$HARDWARE_DETECTION_OUTPUT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"{data['hardware_info']['total_ram_gb']:.1f}\")")
GPU_AVAILABLE=$(echo "$HARDWARE_DETECTION_OUTPUT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(str(data['hardware_info']['gpu_available']).lower())")
RECOMMENDED_CONFIG=$(echo "$HARDWARE_DETECTION_OUTPUT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['hardware_info']['recommended_config'])")
RECOMMENDED_COMMAND=$(echo "$HARDWARE_DETECTION_OUTPUT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['recommended_installation']['command'])")
RECOMMENDED_DESC=$(echo "$HARDWARE_DETECTION_OUTPUT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['recommended_installation']['description'])")

# Display hardware detection results
echo
highlight "🖥️  Hardware Detection Results:"
echo "   💻 CPU Cores: $CPU_COUNT"
echo "   🧠 RAM: ${RAM_GB} GB"
echo "   🎮 GPU Available: $GPU_AVAILABLE"
echo "   📊 Recommended Config: $RECOMMENDED_CONFIG"
echo

#------------- determine installation configuration ---------------------------
if [[ "$FORCE_CPU_ONLY" == "true" ]]; then
    INSTALL_CONFIG="cpu-minimal"
    INSTALL_COMMAND="uv pip install -e ."
    INSTALL_DESC="Forced CPU-only installation"
    highlight "🔧 Using forced CPU-only configuration"
elif [[ "$FORCE_GPU" == "true" ]]; then
    INSTALL_CONFIG="gpu-medium"
    INSTALL_COMMAND="uv pip install -e .[ai-gpu,web]"
    INSTALL_DESC="Forced GPU installation"
    highlight "🔧 Using forced GPU configuration"
elif [[ "$MINIMAL_INSTALL" == "true" ]]; then
    INSTALL_CONFIG="cpu-minimal"
    INSTALL_COMMAND="uv pip install -e ."
    INSTALL_DESC="Minimal installation"
    highlight "🔧 Using minimal installation"
else
    INSTALL_CONFIG="$RECOMMENDED_CONFIG"
    INSTALL_COMMAND="$RECOMMENDED_COMMAND"
    INSTALL_DESC="$RECOMMENDED_DESC"
    highlight "🎯 Using recommended configuration: $INSTALL_CONFIG"
fi

# Show user what will be installed
echo
info "📦 Installation Plan:"
echo "   Configuration: $INSTALL_CONFIG"
echo "   Description: $INSTALL_DESC"
echo "   Command: $INSTALL_COMMAND"

# Calculate estimated download size and time
case $INSTALL_CONFIG in
    "cpu-minimal")
        ESTIMATED_SIZE="~500MB"
        ESTIMATED_TIME="2-5 minutes"
        ;;
    "cpu-optimized")
        ESTIMATED_SIZE="~1.5GB"
        ESTIMATED_TIME="5-10 minutes"
        ;;
    "gpu-"*)
        ESTIMATED_SIZE="~3-8GB"
        ESTIMATED_TIME="10-20 minutes"
        ;;
    *)
        ESTIMATED_SIZE="~1GB"
        ESTIMATED_TIME="5-10 minutes"
        ;;
esac

echo "   Estimated download: $ESTIMATED_SIZE"
echo "   Estimated time: $ESTIMATED_TIME"

# Ask for user confirmation (unless non-interactive)
if [[ "$INTERACTIVE" == "true" ]]; then
    echo
    read -p "🤔 Proceed with this installation? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        info "Installation cancelled by user"
        exit 0
    fi
fi

#------------- dependency installation ----------------------------------------
echo
info "📥 Installing dependencies (this may take several minutes)..."

# Extract the installation part after "uv pip install"
INSTALL_ARGS="${INSTALL_COMMAND#uv pip install }"

if ! uv pip install $INSTALL_ARGS; then
    abort "Failed to install dependencies. Check the error messages above."
fi

ok "Dependencies installed successfully"

#------------- verification ---------------------------------------------------
info "🔍 Verifying installation..."

# Test core imports
python3 -c "
import sys
errors = []

# Test core dependencies
try:
    import templ_pipeline
    print('✓ templ_pipeline imported')
except ImportError as e:
    errors.append(f'templ_pipeline: {e}')

try:
    import numpy, pandas, rdkit
    print('✓ Core scientific libraries imported')
except ImportError as e:
    errors.append(f'Scientific libraries: {e}')

# Test AI dependencies if installed
if '$INSTALL_CONFIG' != 'cpu-minimal':
    try:
        import torch
        print('✓ PyTorch imported')
        if torch.cuda.is_available():
            print(f'✓ CUDA available: {torch.cuda.device_count()} GPU(s)')
        else:
            print('✓ PyTorch CPU-only mode')
    except ImportError as e:
        errors.append(f'PyTorch: {e}')

# Test CLI entry point
try:
    from templ_pipeline.cli.main import main
    print('✓ CLI entry point found')
except ImportError as e:
    errors.append(f'CLI: {e}')

if errors:
    print('\\n❌ Import errors found:')
    for error in errors:
        print(f'   - {error}')
    sys.exit(1)
else:
    print('\\n✅ All components imported successfully')
"

# Test CLI functionality
CLI_SCRIPT="$ENV_DIR/bin/templ"
if [[ -f "$CLI_SCRIPT" ]]; then
    if "$CLI_SCRIPT" --help >/dev/null 2>&1; then
        ok "CLI functionality verified"
    else
        warn "CLI script found but --help failed"
        info "Try: python -m templ_pipeline.cli.main --help"
    fi
else
    warn "CLI script not found at expected location"
    info "Try: python -m templ_pipeline.cli.main --help"
fi

#------------- performance benchmark (optional) ------------------------------
if [[ "$RUN_BENCHMARK" == "true" ]]; then
    echo
    info "🧪 Running performance benchmarks..."
    
    python3 -c "
import time
from templ_pipeline.core.hardware_detection import ProteinEmbeddingBenchmark

benchmark = ProteinEmbeddingBenchmark()
results = benchmark.benchmark_cpu_vs_gpu(['150M'])

print('\\n📊 Performance Benchmark Results:')
for hardware, bench_results in results.items():
    if bench_results:
        result = bench_results[0]
        print(f'   {hardware.upper()}: {result.sequences_per_second:.1f} seq/sec')
"
fi

#------------- success message ------------------------------------------------
echo
cat << EOF
${CLR_GREEN}🎉 TEMPL Pipeline installed successfully!${CLR_RESET}

${CLR_CYAN}🚀 Quick Start:${CLR_RESET}
  templ --help                    # Show all commands
  templ run --help                # One-shot pose prediction
  templ benchmark --help          # Run benchmarks

${CLR_CYAN}💡 Example Usage:${CLR_RESET}
  templ run \\
    --protein-file protein.pdb \\
    --ligand-smiles "CCO" \\
    --output poses.sdf

${CLR_CYAN}🌐 Web Interface:${CLR_RESET}
  python run_streamlit_app.py     # Start web interface

${CLR_CYAN}⚙️ Environment:${CLR_RESET}
  Activate: source ${ENV_DIR}/bin/activate
  Deactivate: deactivate

${CLR_CYAN}📝 Configuration Installed:${CLR_RESET}
  Profile: $INSTALL_CONFIG
  Description: $INSTALL_DESC
  
${CLR_GREEN}✨ Ready to use! Try: templ --help${CLR_RESET}

EOF

# Show performance tips based on configuration
case $INSTALL_CONFIG in
    "cpu-minimal"|"cpu-optimized")
        echo "${CLR_YELLOW}💡 Performance Tips:${CLR_RESET}"
        echo "   • Use smaller protein models (ESM-2 150M-650M) for CPU deployment"
        echo "   • CPU embedding takes ~2-10x longer than GPU but uses less memory"
        echo "   • For production, consider enabling 'server' mode to reduce memory usage"
        ;;
    "gpu-"*)
        echo "${CLR_YELLOW}💡 Performance Tips:${CLR_RESET}"
        echo "   • GPU acceleration is 5-10x faster for protein embedding"
        echo "   • Monitor GPU memory usage with: nvidia-smi"
        echo "   • Use --benchmark flag to test optimal model sizes for your hardware"
        ;;
esac

echo 