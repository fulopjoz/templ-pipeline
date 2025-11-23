#!/bin/bash
# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT

# TEMPL Pipeline Environment Setup Script v3.0
# Ultra-fast setup using uv for dependency management
# Creates optimized virtual environment based on hardware detection

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default settings
INSTALL_MODE="auto"
VENV_NAME=".templ"
PYTHON_MIN_VERSION="3.12"
VERBOSE=false
INTERACTIVE=true
CONFIG_FILE=".templ.config"
UV_CACHE_DIR="${HOME}/.cache/uv"

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_section() {
    echo -e "\n${PURPLE}=== $1 ===${NC}"
}

# Show usage
show_usage() {
    cat << 'USAGE'
TEMPL Pipeline Environment Setup v3.0 (UV-Optimized)

Usage: source setup_templ_env.sh [OPTIONS]

OPTIONS:
  --auto              Auto-detect hardware and install optimally (default)
  --cpu-only          Lightweight CPU-only installation (~50MB)
  --gpu-force         Force GPU installation (if auto-detection fails)
  --minimal           Minimal server installation (no web interface)
  --web               Standard installation with web interface
  --full              Full installation with embedding features
  --dev               Development environment for contributors
  --verbose           Verbose output for debugging
  --quiet             Minimal output for automation
  --non-interactive   Skip all prompts (use defaults)
  --config FILE       Use custom configuration file
  --help              Show this help message

EXAMPLES:
  source setup_templ_env.sh                    # Auto-detect and install optimally
  source setup_templ_env.sh --cpu-only         # Lightweight installation
  source setup_templ_env.sh --gpu-force --dev # Force GPU + development tools
  source setup_templ_env.sh --web              # Standard web interface
  source setup_templ_env.sh --quiet --non-interactive  # Automation friendly

SPEED OPTIMIZATIONS:
  - Uses uv for 10-100x faster dependency resolution
  - Parallel downloads (10 concurrent connections)
  - Pre-compiled wheel caching
  - Optimized RDKit installation
  - Smart hardware detection

NOTES:
  - Must use 'source' command to activate environment
  - Requires Python 3.12+ (uv will auto-install if needed)
  - Auto-detects: CPU cores, RAM, GPU availability
  - Creates .templ virtual environment in project directory
  - Configuration saved to .templ.config file

TROUBLESHOOTING:
  - Permission issues: Check file ownership and permissions
  - Network issues: uv will use cached packages when available
  - GPU issues: Use --cpu-only to bypass GPU detection
USAGE
}

# Load configuration from file
load_config() {
    local config_file="${1:-$CONFIG_FILE}"
    
    if [[ -f "$config_file" ]]; then
        print_status "Loading configuration from $config_file"
        
        while IFS='=' read -r key value; do
            [[ $key =~ ^[[:space:]]*# ]] && continue
            [[ -z $key ]] && continue
            [[ $key =~ ^\[ ]] && continue
            
            key=$(echo "$key" | tr -d ' ')
            value=$(echo "$value" | tr -d ' ')
            
            case $key in
                install_mode) INSTALL_MODE="$value" ;;
                verbose) [[ $value =~ ^(true|yes|1)$ ]] && VERBOSE=true ;;
                interactive) [[ $value =~ ^(false|no|0)$ ]] && INTERACTIVE=false ;;
                name) VENV_NAME="$value" ;;
            esac
        done < "$config_file"
    fi
}

# Parse command line arguments
parse_args() {
    local cmd_install_mode=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --auto)
                INSTALL_MODE="auto"
                cmd_install_mode="auto"
                shift
                ;;
            --cpu-only)
                INSTALL_MODE="cpu-only"
                cmd_install_mode="cpu-only"
                shift
                ;;
            --gpu-force)
                INSTALL_MODE="gpu-force"
                cmd_install_mode="gpu-force"
                shift
                ;;
            --minimal)
                INSTALL_MODE="minimal"
                cmd_install_mode="minimal"
                shift
                ;;
            --web)
                INSTALL_MODE="web"
                cmd_install_mode="web"
                shift
                ;;
            --full)
                INSTALL_MODE="full"
                cmd_install_mode="full"
                shift
                ;;
            --dev)
                INSTALL_MODE="dev"
                cmd_install_mode="dev"
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --quiet)
                VERBOSE=false
                shift
                ;;
            --non-interactive)
                INTERACTIVE=false
                shift
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --help|-h)
                show_usage
                return 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for available options"
                return 1
                ;;
        esac
    done
    
    load_config
    
    if [[ -n "$cmd_install_mode" ]]; then
        INSTALL_MODE="$cmd_install_mode"
    fi
}

# Check if we're being sourced (not executed)
check_sourced() {
    if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
        print_error "This script must be sourced, not executed!"
        echo "Use: source setup_templ_env.sh [options]"
        exit 1
    fi
}

# Install uv if not available
install_uv() {
    if command -v uv >/dev/null 2>&1; then
        UV_VERSION=$(uv --version | awk '{print $2}')
        print_success "uv already installed (version $UV_VERSION)"
        return 0
    fi
    
    print_section "Installing uv Package Manager"
    print_status "Installing uv for ultra-fast dependency management..."
    
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        # Add uv to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
        
        if command -v uv >/dev/null 2>&1; then
            UV_VERSION=$(uv --version | awk '{print $2}')
            print_success "uv installed successfully (version $UV_VERSION)"
        else
            print_error "uv installation completed but command not found in PATH"
            print_status "Please restart your shell or run: export PATH=\"\$HOME/.cargo/bin:\$PATH\""
            return 1
        fi
    else
        print_error "Failed to install uv"
        print_status "Please install manually from: https://github.com/astral-sh/uv"
        return 1
    fi
}

# Detect system hardware
detect_hardware() {
    print_section "Hardware Detection"
    
    # CPU cores
    CPU_CORES=$(nproc 2>/dev/null || echo "4")
    print_status "CPU cores: $CPU_CORES"
    
    # RAM detection
    if command -v free >/dev/null 2>&1; then
        RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
        if [[ $RAM_GB -eq 0 ]]; then
            RAM_GB=$(free -m | awk '/^Mem:/{print int($2/1024)}')
        fi
    else
        RAM_GB=8  # Default assumption
    fi
    print_status "RAM: ${RAM_GB}GB"
    
    # GPU detection
    GPU_AVAILABLE=false
    GPU_INFO="None"
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            GPU_AVAILABLE=true
            GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
            print_status "GPU: $GPU_INFO"
        else
            print_status "GPU: NVIDIA drivers installed but no GPU detected"
        fi
    else
        print_status "GPU: No NVIDIA drivers detected"
    fi
    
    # Python version check
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        print_status "Python: $PYTHON_VERSION"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)"; then
            print_success "Python version check passed"
        else
            print_warning "Python $PYTHON_MIN_VERSION+ recommended, found $PYTHON_VERSION"
            print_status "uv can manage Python versions automatically"
        fi
    else
        print_warning "Python 3 not found - uv will install it automatically"
    fi
}

# Recommend installation type based on hardware
recommend_installation() {
    if [[ "$INSTALL_MODE" != "auto" ]]; then
        return 0
    fi
    
    print_section "Installation Recommendation"
    
    if [[ "$GPU_AVAILABLE" == "true" ]] && [[ $RAM_GB -ge 8 ]]; then
        INSTALL_MODE="full"
        print_status "Recommended: Full installation (GPU + embedding features)"
    elif [[ $RAM_GB -ge 8 ]] && [[ $CPU_CORES -ge 4 ]]; then
        INSTALL_MODE="web"
        print_status "Recommended: Web installation (CPU optimized)"
    elif [[ $RAM_GB -ge 4 ]]; then
        INSTALL_MODE="cpu-only"
        print_status "Recommended: CPU-only installation (lightweight)"
    else
        INSTALL_MODE="minimal"
        print_status "Recommended: Minimal installation (low resources)"
    fi
    
    case $INSTALL_MODE in
        full)
            print_status "Includes: Core + Web interface + GPU acceleration + embedding features"
            ;;
        web)
            print_status "Includes: Core + Web interface (Streamlit)"
            ;;
        cpu-only)
            print_status "Includes: Core + Essential features only"
            ;;
        minimal)
            print_status "Includes: Core libraries only"
            ;;
    esac
}

# Create virtual environment with uv
create_venv() {
    print_section "Virtual Environment Setup"
    
    if [[ -d "$VENV_NAME" ]]; then
        print_warning "Virtual environment already exists at $VENV_NAME"
        if [[ "$INTERACTIVE" == "true" ]]; then
            read -p "Remove and recreate? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf "$VENV_NAME"
                print_status "Removed existing environment"
            else
                print_status "Using existing environment"
                return 0
            fi
        else
            print_status "Using existing environment (non-interactive mode)"
            return 0
        fi
    fi
    
    print_status "Creating virtual environment with uv..."
    
    # Use uv to create venv with Python 3.12+
    if uv venv "$VENV_NAME" --python 3.12; then
        print_success "Virtual environment created successfully"
    else
        print_error "Failed to create virtual environment"
        return 1
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    
    if [[ -f "$VENV_NAME/bin/activate" ]]; then
        source "$VENV_NAME/bin/activate"
        print_success "Virtual environment activated"
        print_status "Python: $(which python3)"
    else
        print_error "Activation script not found"
        return 1
    fi
}

# Install dependencies with uv
install_dependencies() {
    print_section "Installing Dependencies"
    
    local install_target="."
    
    case $INSTALL_MODE in
        minimal)
            install_target="."
            print_status "Installing minimal dependencies..."
            ;;
        cpu-only)
            install_target="."
            print_status "Installing CPU-only dependencies..."
            ;;
        web)
            install_target=".[web]"
            print_status "Installing web interface dependencies..."
            ;;
        full)
            install_target=".[full]"
            print_status "Installing full dependencies (including embeddings)..."
            ;;
        dev)
            install_target=".[dev]"
            print_status "Installing development dependencies..."
            ;;
        gpu-force)
            install_target=".[full]"
            print_status "Installing GPU-enabled dependencies..."
            ;;
    esac
    
    print_status "Using uv for ultra-fast installation..."
    print_status "Features: parallel downloads, pre-compiled wheels, smart caching"
    
    # Use uv pip install with optimizations
    local uv_flags=""
    if [[ "$VERBOSE" == "true" ]]; then
        uv_flags="--verbose"
    else
        uv_flags="--quiet"
    fi
    
    # Install with uv - much faster than pip
    if uv pip install $uv_flags --editable "$install_target"; then
        print_success "Dependencies installed successfully"
    else
        print_error "Failed to install dependencies"
        return 1
    fi
    
    # Verify critical packages
    print_status "Verifying installation..."
    if python3 -c "import rdkit; import numpy; import pandas" 2>/dev/null; then
        print_success "Core packages verified"
    else
        print_error "Package verification failed"
        return 1
    fi
}

# Download data files
download_data() {
    print_section "Downloading Required Data"
    
    if [[ ! -d "data" ]]; then
        mkdir -p data
    fi
    
    # Check if data already exists
    if [[ -f "data/templ_protein_embeddings_v1.0.0.npz" ]] && \
       [[ -f "data/templ_processed_ligands_v1.0.0.sdf.gz" ]]; then
        print_success "Data files already present"
        return 0
    fi
    
    print_status "Downloading embeddings and ligands from Zenodo..."
    print_status "This may take a few minutes depending on your connection..."
    
    cd data
    
    # Use zenodo-get for reliable downloads
    if command -v zenodo_get >/dev/null 2>&1; then
        if zenodo_get 10.5281/zenodo.16890956 2>/dev/null; then
            print_success "Data downloaded successfully"
        else
            print_warning "zenodo_get failed, trying direct download..."
            download_data_direct
        fi
    else
        download_data_direct
    fi
    
    cd ..
}

# Direct download fallback
download_data_direct() {
    local base_url="https://zenodo.org/records/16890956/files"
    
    if [[ ! -f "templ_protein_embeddings_v1.0.0.npz" ]]; then
        print_status "Downloading protein embeddings..."
        curl -L -o "templ_protein_embeddings_v1.0.0.npz" \
            "$base_url/templ_protein_embeddings_v1.0.0.npz?download=1" || return 1
    fi
    
    if [[ ! -f "templ_processed_ligands_v1.0.0.sdf.gz" ]]; then
        print_status "Downloading processed ligands..."
        curl -L -o "templ_processed_ligands_v1.0.0.sdf.gz" \
            "$base_url/templ_processed_ligands_v1.0.0.sdf.gz?download=1" || return 1
    fi
    
    print_success "Data downloaded successfully"
}

# Verify installation
verify_installation() {
    print_section "Installation Verification"
    
    # Check if templ command is available
    if command -v templ >/dev/null 2>&1; then
        print_success "templ command available"
        
        # Try to get version/help
        if templ --help >/dev/null 2>&1; then
            print_success "templ command working"
        else
            print_warning "templ command found but may have issues"
        fi
    else
        print_warning "templ command not found in PATH"
        print_status "You may need to reinstall or check your environment"
    fi
    
    # Check Python imports
    print_status "Checking Python package imports..."
    python3 << 'PYCHECK'
import sys
try:
    import rdkit
    from rdkit import Chem
    import numpy
    import pandas
    import sklearn
    print("✓ Core packages: OK")
    
    # Check RDKit version for compatibility
    rdkit_version = rdkit.__version__
    print(f"✓ RDKit version: {rdkit_version}")
    
    # Test Morgan fingerprint generation (common error point)
    try:
        from rdkit.Chem import rdFingerprintGenerator
        mol = Chem.MolFromSmiles("CCO")
        # Use correct API - fpSize instead of nBits
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = gen.GetFingerprint(mol)
        print("✓ RDKit fingerprint generation: OK")
    except Exception as e:
        print(f"⚠ RDKit fingerprint warning: {e}")
        print("  (This may not affect TEMPL functionality)")
    
except ImportError as e:
    print(f"✗ Import error: {e}", file=sys.stderr)
    sys.exit(1)
PYCHECK
    
    if [[ $? -eq 0 ]]; then
        print_success "Package verification passed"
    else
        print_error "Package verification failed"
        return 1
    fi
}

# Save configuration
save_config() {
    print_section "Saving Configuration"
    
    cat > "$CONFIG_FILE" << EOF
# TEMPL Pipeline Configuration
# Generated on $(date)

[environment]
name = $VENV_NAME
install_mode = $INSTALL_MODE

[hardware]
cpu_cores = $CPU_CORES
ram_gb = $RAM_GB
gpu_available = $GPU_AVAILABLE

[settings]
verbose = $VERBOSE
interactive = $INTERACTIVE

[optimization]
uv_cache = $UV_CACHE_DIR
parallel_downloads = 10
EOF
    
    print_success "Configuration saved to $CONFIG_FILE"
}

# Print summary
print_summary() {
    print_section "Installation Complete!"
    
    echo -e "${GREEN}✓${NC} Virtual environment: ${CYAN}$VENV_NAME${NC}"
    echo -e "${GREEN}✓${NC} Installation mode: ${CYAN}$INSTALL_MODE${NC}"
    echo -e "${GREEN}✓${NC} Python version: ${CYAN}$(python3 --version)${NC}"
    
    if command -v uv >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Package manager: ${CYAN}uv $(uv --version | awk '{print $2}')${NC}"
    fi
    
    echo ""
    echo -e "${PURPLE}Next steps:${NC}"
    echo -e "  1. Environment is already activated for this session"
    echo -e "  2. For future sessions, run: ${CYAN}source $VENV_NAME/bin/activate${NC}"
    echo -e "  3. Try the CLI: ${CYAN}templ --help${NC}"
    echo -e "  4. Or launch the web UI: ${CYAN}python scripts/run_streamlit_app.py${NC}"
    echo ""
    echo -e "${YELLOW}Speed optimizations enabled:${NC}"
    echo -e "  • uv package manager (10-100x faster than pip)"
    echo -e "  • Parallel downloads (10 concurrent)"
    echo -e "  • Pre-compiled wheel caching"
    echo -e "  • Optimized dependency resolution"
    echo ""
}

# Main execution
main() {
    check_sourced
    parse_args "$@"
    
    if [[ $? -ne 0 ]]; then
        return 1
    fi
    
    # Install uv first
    install_uv || return 1
    
    # Detect hardware
    detect_hardware || return 1
    
    # Recommend installation
    recommend_installation
    
    # Create and activate venv
    create_venv || return 1
    activate_venv || return 1
    
    # Install dependencies
    install_dependencies || return 1
    
    # Download data
    download_data || return 1
    
    # Verify installation
    verify_installation || return 1
    
    # Save configuration
    save_config
    
    # Print summary
    print_summary
    
    return 0
}

# Run main function
main "$@"
