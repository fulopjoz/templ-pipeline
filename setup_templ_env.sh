#!/usr/bin/env bash
# TEMPL Pipeline - Complete Setup & Activation Script
# 
# USAGE (RECOMMENDED):
#   source setup_templ_env.sh [OPTIONS]    # Creates env + installs + activates immediately
#
# ALTERNATIVE:
#   ./setup_templ_env.sh [OPTIONS]         # Creates env + installs (manual activation needed)

set -euo pipefail
IFS=$'\n\t'

# Detect if script is being sourced (enables automatic activation)
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
INTERACTIVE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-only) FORCE_CPU_ONLY=true; shift ;;
        --gpu-force) FORCE_GPU=true; shift ;;
        --minimal) MINIMAL_INSTALL=true; shift ;;
        --benchmark) RUN_BENCHMARK=true; shift ;;
        --interactive) INTERACTIVE=true; shift ;;
        --help|-h)
            cat << EOF
TEMPL Pipeline - Complete Setup & Activation Script

RECOMMENDED USAGE:
    source $0 [OPTIONS]         # Creates environment + installs dependencies + activates immediately
                               # This puts you directly into the TEMPL environment ready to use!

ALTERNATIVE USAGE:
    ./$0 [OPTIONS]             # Creates environment + installs dependencies (requires manual activation)

OPTIONS:
    --cpu-only          Force CPU-only installation
    --gpu-force         Force GPU installation  
    --minimal           Install minimal dependencies only
    --benchmark         Run performance benchmarks after installation
    --interactive       Prompt for user confirmation
    --help, -h          Show this help message

EXAMPLES:
    source $0               # Recommended: Complete setup with immediate activation
    source $0 --cpu-only    # Force CPU-only installation with activation
    source $0 --minimal     # Minimal installation with activation

AFTER SETUP:
    templ --help           # Check available commands
    python run_streamlit_app.py  # Run the web interface

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
if [[ "$SCRIPT_SOURCED" == "true" ]]; then
    info "Setting up TEMPL Pipeline with automatic activation..."
else
    info "Setting up TEMPL Pipeline..."
    warn "For automatic activation, use: source $0"
fi

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
    
    # Add uv to PATH for current session
    UV_PATHS=("$HOME/.cargo/bin" "$HOME/.local/bin")
    for uv_path in "${UV_PATHS[@]}"; do
        if [[ -f "$uv_path/uv" ]]; then
            export PATH="$uv_path:$PATH"
            break
        fi
    done
    
    # Verify uv is now available
    if ! command -v uv >/dev/null 2>&1; then
        abort "uv package manager installation failed - not found in PATH"
    fi
    
    ok "uv package manager installed successfully"
fi

# Verify uv version
UV_VERSION=$(uv --version 2>/dev/null || echo "unknown")
info "Using uv version: $UV_VERSION"

#------------- environment creation -------------------------------------------
ENV_DIR=".templ"
if [[ ! -d "$ENV_DIR" ]]; then
    info "Creating virtual environment with uv..."
    if ! uv venv "$ENV_DIR" --python python3 2>/dev/null; then
        abort "Failed to create .templ virtual environment with uv"
    fi
    CREATED_VENV=true
    ok "Virtual environment created successfully"
else
    info "Using existing virtual environment"
fi

# Verify environment structure
if [[ ! -f "$ENV_DIR/bin/activate" ]]; then
    abort "Virtual environment activation script not found"
fi

#------------- environment activation -----------------------------------------
info "Activating virtual environment..."
source "$ENV_DIR/bin/activate" || abort "Failed to activate .templ virtual environment"

# Verify we're in the right environment
if [[ "${VIRTUAL_ENV:-}" != *".templ"* ]]; then
    abort "Environment activation failed - not in .templ environment"
fi

ok "Virtual environment activated: $VIRTUAL_ENV"

# Verify uv works in the activated environment
if ! uv --version >/dev/null 2>&1; then
    abort "uv not available in activated environment"
fi

#------------- install minimal dependencies for hardware detection ------------
info "Installing system detection dependencies..."
if ! uv pip install psutil --quiet 2>/dev/null; then
    abort "Failed to install system detection dependencies"
fi

#------------- hardware detection ---------------------------------------------
info "Detecting hardware configuration..."
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
            'extras': f'ai-cpu,web' if config != 'cpu-minimal' else '',
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
RECOMMENDED_EXTRAS=$(echo "$HARDWARE_DETECTION_OUTPUT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['recommended_installation'].get('extras', ''))")
RECOMMENDED_DESC=$(echo "$HARDWARE_DETECTION_OUTPUT" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['recommended_installation']['description'])")

#------------- determine installation configuration ---------------------------
if [[ "$FORCE_CPU_ONLY" == "true" ]]; then
    INSTALL_CONFIG="cpu-minimal"
    INSTALL_EXTRAS=""
    INSTALL_DESC="CPU-only installation"
elif [[ "$FORCE_GPU" == "true" ]]; then
    INSTALL_CONFIG="gpu-medium"
    INSTALL_EXTRAS="ai-gpu,web"
    INSTALL_DESC="GPU-accelerated installation"
elif [[ "$MINIMAL_INSTALL" == "true" ]]; then
    INSTALL_CONFIG="cpu-minimal"
    INSTALL_EXTRAS=""
    INSTALL_DESC="Minimal installation"
else
    INSTALL_CONFIG="$RECOMMENDED_CONFIG"
    INSTALL_EXTRAS="$RECOMMENDED_EXTRAS"
    INSTALL_DESC="$RECOMMENDED_DESC"
fi

info "System: ${CPU_COUNT} CPUs, ${RAM_GB}GB RAM, GPU: ${GPU_AVAILABLE} → ${INSTALL_CONFIG}"

# User confirmation for interactive mode
if [[ "$INTERACTIVE" == "true" ]]; then
    read -p "Proceed with $INSTALL_DESC? [Y/n] " -n 1 -r
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
info "Installing dependencies using uv..."
info "Configuration: $INSTALL_DESC"

INSTALL_START_TIME=$(date +%s)

# Build installation command
if [[ -n "$INSTALL_EXTRAS" ]]; then
    INSTALL_ARGS="-e .[${INSTALL_EXTRAS}]"
else
    INSTALL_ARGS="-e ."
fi

info "Running: uv pip install $INSTALL_ARGS"

if ! uv pip install $INSTALL_ARGS --quiet 2>/dev/null; then
    err "Dependency installation failed"
    err "Trying without quiet mode for debugging..."
    uv pip install $INSTALL_ARGS || abort "Dependency installation failed completely"
fi

# Also install from requirements.txt to ensure all dependencies are covered
if [[ -f "requirements.txt" ]]; then
    info "Installing additional dependencies from requirements.txt..."
    if ! uv pip install -r requirements.txt --quiet 2>/dev/null; then
        warn "Some dependencies from requirements.txt failed to install"
    fi
fi

INSTALL_END_TIME=$(date +%s)
INSTALL_DURATION=$((INSTALL_END_TIME - INSTALL_START_TIME))

#------------- installation verification --------------------------------------
info "Verifying installation..."
python3 -c "
import sys
errors = []

try:
    import templ_pipeline
    print('✓ templ_pipeline imported successfully')
except ImportError as e:
    errors.append(f'templ_pipeline: {e}')

try:
    import numpy, pandas, rdkit
    print('✓ Scientific libraries (numpy, pandas, rdkit) imported successfully')
except ImportError as e:
    errors.append(f'Scientific libraries: {e}')

if '$INSTALL_CONFIG' != 'cpu-minimal':
    try:
        import torch
        print('✓ PyTorch imported successfully')
        if torch.cuda.is_available():
            print('✓ CUDA available')
        else:
            print('✓ PyTorch CPU-only mode')
    except ImportError as e:
        errors.append(f'PyTorch: {e}')

try:
    from templ_pipeline.cli.main import main
    print('✓ CLI interface available')
except ImportError as e:
    errors.append(f'CLI interface: {e}')

if errors:
    print('\\nERRORS FOUND:')
    for error in errors:
        print(f'  {error}')
    sys.exit(1)
else:
    print('\\n All core components verified successfully')
" || abort "Installation verification failed"

# Verify CLI functionality
if command -v templ >/dev/null 2>&1; then
    if templ --help >/dev/null 2>&1; then
        ok "CLI command 'templ' is working"
    else
        warn "CLI command 'templ' exists but not working properly"
    fi
else
    warn "CLI command 'templ' not available. Use: python -m templ_pipeline.cli.main"
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

#------------- environment customization --------------------------------------
# Enhance activation script with custom prompt
ACTIVATE_SCRIPT=".templ/bin/activate"
if [[ -f "$ACTIVATE_SCRIPT" ]]; then
    if ! grep -q "TEMPL Pipeline" "$ACTIVATE_SCRIPT"; then
        cat >> "$ACTIVATE_SCRIPT" << 'EOF'

# TEMPL Pipeline environment customization
export TEMPL_ENV_ACTIVE=1
if [[ -z "${TEMPL_PROMPT_BACKUP:-}" ]]; then
    export TEMPL_PROMPT_BACKUP="$PS1"
    export PS1="(.templ) $PS1"
fi
EOF
    fi
fi

#------------- completion summary ---------------------------------------------
ok "TEMPL Pipeline installed successfully (${INSTALL_DURATION}s)"
ok "Configuration: $INSTALL_CONFIG"
ok "Using uv for package management"

# Final verification
if python3 -c "import rdkit; print(f'rdkit {rdkit.__version__} available')" 2>/dev/null; then
    ok "rdkit is properly installed and accessible"
else
    warn "rdkit verification failed"
fi

# Check actual environment status in current shell
if [[ "$SCRIPT_SOURCED" == "true" ]]; then
    # When sourced, the environment should be active in the current shell
    if [[ "${VIRTUAL_ENV:-}" == *".templ"* ]]; then
        ok "Environment is ACTIVE and ready to use!"
        echo ""
        echo "   You're now in the TEMPL environment! Try these commands:"
        echo "   templ --help                    # View available commands"
        echo "   python run_streamlit_app.py    # Launch web interface"
        echo "   python -c \"import rdkit; print('rdkit version:', rdkit.__version__)\"  # Test rdkit"
        echo ""
    else
        warn "Environment was created but activation failed."
        echo "   Manual activation: source .templ/bin/activate"
    fi
else
    echo ""
    echo "Installation completed successfully!"
    echo ""
    echo "To activate the TEMPL environment, run:"
    echo "   source .templ/bin/activate"
    echo ""
fi 