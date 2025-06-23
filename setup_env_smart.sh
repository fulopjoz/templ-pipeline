#!/usr/bin/env bash
# setup_env_smart.sh – TEMPL Pipeline Installation Script
# Usage: ./setup_env_smart.sh [OPTIONS]
#    OR: source setup_env_smart.sh [OPTIONS]  (for automatic activation)

set -euo pipefail
IFS=$'\n\t'

# Detect if script is being sourced (allows automatic activation)
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    SCRIPT_SOURCED=true
else
    SCRIPT_SOURCED=false
fi

#------------- logging functions ----------------------------------------------
info()  { echo "[INFO]  $*"; }
ok()    { echo "[OK]    $*"; }
warn()  { echo "[WARN]  $*"; }
err()   { echo "[ERROR] $*" >&2; }

abort() { 
    err "$1"
    [[ -d ".templ" && "${CREATED_VENV:-}" == "true" ]] && rm -rf .templ
    if [[ "$SCRIPT_SOURCED" == "true" ]]; then
        return 1
    else
        exit 1
    fi
}

#------------- parse arguments ------------------------------------------------
FORCE_CPU_ONLY=false
FORCE_GPU=false
MINIMAL_INSTALL=false
RUN_BENCHMARK=false
INTERACTIVE=false  # Default to non-interactive for automated installation

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-only) FORCE_CPU_ONLY=true; shift ;;
        --gpu-force) FORCE_GPU=true; shift ;;
        --minimal) MINIMAL_INSTALL=true; shift ;;
        --benchmark) RUN_BENCHMARK=true; shift ;;
        --interactive) INTERACTIVE=true; shift ;;
        --help|-h)
            cat << EOF
TEMPL Pipeline Installation Script

USAGE:
    $0 [OPTIONS]                 # Standard installation  
    source $0 [OPTIONS]          # Install + auto-activate environment

OPTIONS:
    --cpu-only          Force CPU-only installation
    --gpu-force         Force GPU installation  
    --minimal           Install minimal dependencies only
    --benchmark         Run performance benchmarks after installation
    --interactive       Prompt for user confirmation
    --help, -h          Show this help message

EXAMPLES:
    $0                  # Standard install (requires manual activation)
    source $0           # Install + auto-activate (recommended)
    $0 --cpu-only       # Force CPU-only installation
    $0 --minimal        # Minimal installation for resource-constrained environments

EOF
            if [[ "$SCRIPT_SOURCED" == "true" ]]; then
                return 0
            else
                exit 0
            fi
            ;;
        *) warn "Unknown option: $1. Use --help for usage information."; shift ;;
    esac
done

#------------- header ---------------------------------------------------------
info "Installing TEMPL Pipeline..."

#------------- python check ---------------------------------------------------
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

#------------- uv installation ------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
    info "Installing uv package manager..."
    if ! curl -fsSL https://astral.sh/uv/install.sh | bash >/dev/null 2>&1; then
        abort "Failed to install uv package manager. Please check internet connection."
    fi
    
    # Add uv to PATH
    UV_PATHS=("$HOME/.cargo/bin" "$HOME/.local/bin")
    for uv_path in "${UV_PATHS[@]}"; do
        if [[ -f "$uv_path/uv" ]]; then
            export PATH="$uv_path:$PATH"
            break
        fi
    done
    
    if ! command -v uv >/dev/null 2>&1; then
        abort "uv package manager installation failed"
    fi
fi

#------------- hardware detection ---------------------------------------------
ENV_DIR=".templ"
if [[ ! -d "$ENV_DIR" ]]; then
    uv venv "$ENV_DIR" >/dev/null 2>&1 || abort "Failed to create .templ virtual environment"
    CREATED_VENV=true
fi

source "$ENV_DIR/bin/activate" || abort "Failed to activate .templ virtual environment"

# Install minimal dependencies for hardware detection
if ! python -c "import psutil" >/dev/null 2>&1; then
    uv pip install psutil >/dev/null 2>&1 || abort "Failed to install system detection dependencies"
fi

# Hardware detection with fallback
HARDWARE_DETECTION_OUTPUT=$(python3 -c "
import json, sys, os
sys.path.insert(0, os.getcwd())

try:
    from templ_pipeline.core.hardware_detection import get_hardware_recommendation
    print(json.dumps(get_hardware_recommendation(), indent=2))
except Exception:
    import psutil, subprocess
    
    gpu_available = False
    try:
        subprocess.run(['nvidia-smi'], capture_output=True, timeout=5, check=True)
        gpu_available = True
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

#------------- determine installation configuration --------------------------- 
if [[ "$FORCE_CPU_ONLY" == "true" ]]; then
    INSTALL_CONFIG="cpu-minimal"
    INSTALL_COMMAND="uv pip install -e ."
    INSTALL_DESC="CPU-only installation"
elif [[ "$FORCE_GPU" == "true" ]]; then
    INSTALL_CONFIG="gpu-medium"
    INSTALL_COMMAND="uv pip install -e .[ai-gpu,web]"
    INSTALL_DESC="GPU-accelerated installation"
elif [[ "$MINIMAL_INSTALL" == "true" ]]; then
    INSTALL_CONFIG="cpu-minimal"
    INSTALL_COMMAND="uv pip install -e ."
    INSTALL_DESC="Minimal installation"
else
    INSTALL_CONFIG="$RECOMMENDED_CONFIG"
    INSTALL_COMMAND="$RECOMMENDED_COMMAND"
    INSTALL_DESC="$RECOMMENDED_DESC"
fi

info "System: ${CPU_COUNT} CPUs, ${RAM_GB}GB RAM, GPU: ${GPU_AVAILABLE} → ${INSTALL_CONFIG}"

# User confirmation for interactive mode
if [[ "$INTERACTIVE" == "true" ]]; then
    read -p "Proceed with installation? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        info "Installation cancelled by user"
        if [[ "$SCRIPT_SOURCED" == "true" ]]; then
            return 0
        else
            exit 0
        fi
    fi
fi

#------------- dependency installation ----------------------------------------
info "Installing dependencies..."

INSTALL_ARGS="${INSTALL_COMMAND#uv pip install }"
INSTALL_START_TIME=$(date +%s)

if ! uv pip install $INSTALL_ARGS >/dev/null 2>&1; then
    abort "Dependency installation failed. Please check package requirements."
fi

INSTALL_END_TIME=$(date +%s)
INSTALL_DURATION=$((INSTALL_END_TIME - INSTALL_START_TIME))

#------------- installation verification --------------------------------------
python3 -c "
import sys
errors = []

try:
    import templ_pipeline
except ImportError as e:
    errors.append(f'templ_pipeline: {e}')

try:
    import numpy, pandas, rdkit
except ImportError as e:
    errors.append(f'Scientific libraries: {e}')

if '$INSTALL_CONFIG' != 'cpu-minimal':
    try:
        import torch
        if torch.cuda.is_available():
            pass  # Silent success
    except ImportError as e:
        errors.append(f'PyTorch: {e}')

try:
    from templ_pipeline.cli.main import main
except ImportError as e:
    errors.append(f'CLI interface: {e}')

if errors:
    for error in errors:
        print(f'ERROR: {error}')
    sys.exit(1)
" 2>/dev/null || abort "Installation verification failed"

# CLI functionality verification
CLI_SCRIPT="$ENV_DIR/bin/templ"
if [[ ! -f "$CLI_SCRIPT" ]] || ! "$CLI_SCRIPT" --help >/dev/null 2>&1; then
    warn "CLI script not available. Use: python -m templ_pipeline.cli.main"
fi

#------------- optional benchmark ---------------------------------------------
if [[ "$RUN_BENCHMARK" == "true" ]]; then
    info "Running performance benchmark..."
    python3 -c "
try:
    from templ_pipeline.core.hardware_detection import ProteinEmbeddingBenchmark
    benchmark = ProteinEmbeddingBenchmark()
    results = benchmark.benchmark_cpu_vs_gpu(['150M'])
    for hardware, bench_results in results.items():
        if bench_results:
            result = bench_results[0]
            print('Performance {}: {:.1f} sequences/second'.format(hardware.upper(), result.sequences_per_second))
except Exception as e:
    print('Benchmark execution failed: {}'.format(e))
"
fi

#------------- environment activation setup -----------------------------------
# Enhance activation script with custom prompt
ACTIVATE_SCRIPT=".templ/bin/activate"
if [[ -f "$ACTIVATE_SCRIPT" ]]; then
    if ! grep -q "TEMPL Pipeline" "$ACTIVATE_SCRIPT"; then
        cat >> "$ACTIVATE_SCRIPT" << 'EOF'

# TEMPL Pipeline environment customization
export TEMPL_ENV_ACTIVE=1
if [[ -z "${TEMPL_PROMPT_BACKUP:-}" ]]; then
    export TEMPL_PROMPT_BACKUP="$PS1"
    export PS1="(templ) $PS1"
fi
EOF
    fi
fi

# Create convenience activation script for future sessions
cat > "activate_templ.sh" << EOF
#!/bin/bash
# TEMPL Pipeline Environment Activation Script
# Usage: source activate_templ.sh

if [[ -f ".templ/bin/activate" ]]; then
    source .templ/bin/activate
    echo "TEMPL Pipeline environment activated"
    echo "Available commands: templ --help"
else
    echo "Error: .templ environment not found"
    echo "Please run: ./setup_env_smart.sh"
    exit 1
fi
EOF

chmod +x activate_templ.sh

#------------- installation summary -------------------------------------------
ok "TEMPL Pipeline installed successfully (${INSTALL_DURATION}s)"

# Check if .templ environment is active in current shell
if [[ "${VIRTUAL_ENV:-}" == *".templ"* ]]; then
    ENV_STATUS="Active"
    ACTIVATION_NEEDED=false
else
    ENV_STATUS="Ready"
    ACTIVATION_NEEDED=true
fi

echo "Environment: $ENV_STATUS | Config: $INSTALL_CONFIG"

if [[ "$ACTIVATION_NEEDED" == "true" ]]; then
    echo "Activate: source activate_templ.sh"
fi

echo "Usage: templ --help | python run_streamlit_app.py"

# Automatic activation when sourced
if [[ "$SCRIPT_SOURCED" == "true" ]]; then
    source "$ENV_DIR/bin/activate"
    ok "Environment activated! Try: templ --help"
fi 