#!/bin/bash

# TEMPL Pipeline Environment Setup Script
# Creates optimized virtual environment based on hardware detection
# Supports multiple installation profiles and dependency approaches

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
PYTHON_MIN_VERSION="3.9"
USE_REQUIREMENTS_TXT=false
VERBOSE=false
INTERACTIVE=true
CONFIG_FILE=".templ.config"

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
TEMPL Pipeline Environment Setup v2.1

Usage: source setup_templ_env.sh [OPTIONS]

OPTIONS:
  --auto              Auto-detect hardware and install optimally (default)
  --cpu-only          Lightweight CPU-only installation (~50MB)
  --gpu-force         Force GPU installation (if auto-detection fails)
  --minimal           Minimal server installation (no web interface)
  --web               Standard installation with web interface
  --full              Full installation with embedding features
  --dev               Development environment for contributors
  --use-requirements  Use requirements.txt instead of pyproject.toml
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

NOTES:
  - Must use 'source' command to activate environment
  - Requires Python 3.9+ and pip
  - Auto-detects: CPU cores, RAM, GPU availability
  - Creates .templ virtual environment in project directory
  - Configuration saved to .templ.config file
  - Use './manage_environment.sh status' to check environment after setup

CONFIGURATION:
  Edit .templ.config to customize behavior, or use --config FILE
  
TROUBLESHOOTING:
  - Permission issues: Check file ownership and permissions
  - Network issues: Try --use-requirements for offline installation
  - GPU issues: Use --cpu-only to bypass GPU detection
  - For more help: './manage_environment.sh doctor'
USAGE
}

# Load configuration from file
load_config() {
    local config_file="${1:-$CONFIG_FILE}"
    
    if [[ -f "$config_file" ]]; then
        print_status "Loading configuration from $config_file"
        
        # Parse simple key=value format (ignore sections for now)
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            [[ $key =~ ^[[:space:]]*# ]] && continue
            [[ -z $key ]] && continue
            [[ $key =~ ^\[ ]] && continue
            
            # Remove whitespace
            key=$(echo "$key" | tr -d ' ')
            value=$(echo "$value" | tr -d ' ')
            
            case $key in
                install_mode) INSTALL_MODE="$value" ;;
                verbose) [[ $value =~ ^(true|yes|1)$ ]] && VERBOSE=true ;;
                interactive) [[ $value =~ ^(false|no|0)$ ]] && INTERACTIVE=false ;;
                use_requirements_txt) [[ $value =~ ^(true|yes|1)$ ]] && USE_REQUIREMENTS_TXT=true ;;
                name) VENV_NAME="$value" ;;
            esac
        done < "$config_file"
    elif [[ ! -f "$CONFIG_FILE" ]] && [[ -f ".templ.config.template" ]]; then
        print_status "Creating default configuration file"
        cp ".templ.config.template" "$CONFIG_FILE"
    fi
}

# Parse command line arguments
parse_args() {
    # Store command line arguments to check for overrides later
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
            --use-requirements)
                USE_REQUIREMENTS_TXT=true
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
    
    # Load configuration first
    load_config
    
    # Command line arguments override config file settings
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
        
        # Version comparison
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
            print_success "Python version check passed"
        else
            print_error "Python $PYTHON_MIN_VERSION+ required, found $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python 3 not found"
        return 1
    fi
}

# Recommend installation type based on hardware
recommend_installation() {
    if [[ "$INSTALL_MODE" != "auto" ]]; then
        return 0
    fi
    
    print_section "Installation Recommendation"
    
    # Decision logic
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
    
    # Show what this includes
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

# Create virtual environment
create_venv() {
    print_section "Virtual Environment Setup"
    
    # Install uv if not available
    if ! command -v uv >/dev/null 2>&1; then
        print_status "Installing uv package manager..."
        if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
            print_error "Failed to install uv. Please install manually:"
            echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
            echo "  Or visit: https://github.com/astral-sh/uv"
            return 1
        fi
        # Source the shell configuration to make uv available
        source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
        
        # Verify uv is available
        if ! command -v uv >/dev/null 2>&1; then
            print_error "uv installation failed. Please restart your shell and try again."
            return 1
        fi
    fi
    
    if [[ -d "$VENV_NAME" ]]; then
        print_warning "Virtual environment $VENV_NAME already exists"
        
        if [[ "$INTERACTIVE" == "true" ]]; then
            read -p "Remove and recreate? [y/N]: " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf "$VENV_NAME"
                print_status "Removed existing environment"
            else
                print_status "Using existing environment"
                return 0
            fi
        else
            print_status "Non-interactive mode: Using existing environment"
            return 0
        fi
    fi
    
    print_status "Creating virtual environment: $VENV_NAME"
    uv venv "$VENV_NAME"
    
    print_status "Activating virtual environment"
    source "$VENV_NAME/bin/activate"
    
    # Verify we're in the correct environment
    if [[ "$VIRTUAL_ENV" != "$(pwd)/$VENV_NAME" ]]; then
        print_error "Failed to activate virtual environment properly"
        return 1
    fi

    # Verify Python is from the virtual environment
    if ! which python | grep -q "$VENV_NAME"; then
        print_error "Python is not from the virtual environment"
        return 1
    fi
    
    # Upgrade pip (this installs pip in the virtual environment)
    print_status "Upgrading pip"
    uv pip install --upgrade pip
    
    # Now verify pip is properly installed
    if ! which pip | grep -q "$VENV_NAME"; then
        print_warning "pip not found in virtual environment, but uv pip is available"
    fi
    
    print_success "Virtual environment created and activated"
}

# Install dependencies
install_dependencies() {
    print_section "Installing Dependencies"
    
    if [[ "$USE_REQUIREMENTS_TXT" == "true" ]]; then
        install_from_requirements
    else
        install_from_pyproject
    fi
}

# Install from requirements.txt (pinned versions)
install_from_requirements() {
    print_status "Installing from requirements.txt (pinned versions)"
    
    if [[ ! -f "requirements.txt" ]]; then
        print_error "requirements.txt not found"
        return 1
    fi
    
    # Install base requirements
    uv pip install -r requirements.txt
    
    # Install current package in development mode
    uv pip install -e .
    
    print_success "Installed from requirements.txt"
}

# Install from pyproject.toml (flexible versions)
install_from_pyproject() {
    print_status "Installing from pyproject.toml (flexible versions)"
    
    if [[ ! -f "pyproject.toml" ]]; then
        print_error "pyproject.toml not found"
        return 1
    fi
    
    # Determine extras to install
    local extras=""
    case $INSTALL_MODE in
        minimal)
            extras=""
            print_status "Installing: Core dependencies only"
            ;;
        cpu-only)
            extras=""
            print_status "Installing: Core dependencies (CPU-optimized)"
            ;;
        web)
            extras="[web]"
            print_status "Installing: Core + Web interface"
            ;;
        full|gpu-force)
            extras="[full]"
            print_status "Installing: Core + Web + embedding features"
            ;;
        dev)
            extras="[dev]"
            print_status "Installing: Development environment"
            ;;
    esac
    
    # Install package with appropriate extras
    if [[ -n "$extras" ]]; then
        uv pip install -e ".$extras"
    else
        uv pip install -e .
    fi
    
    print_success "Installed from pyproject.toml"
}

# Download and setup data files from Zenodo
setup_data_files() {
    print_section "Data Files Setup"
    
    # Create data directory structure - only missing directories
    local dirs=("data" "data/embeddings" "data/ligands" "data/PDBBind")
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        fi
    done
    
    # Check if data files already exist
    if [[ -f "data/embeddings/templ_protein_embeddings_v1.0.0.npz" ]] && [[ -f "data/ligands/templ_processed_ligands_v1.0.0.sdf.gz" ]]; then
        print_success "Data files already exist, skipping download"
        return 0
    fi
    
    print_status "Downloading TEMPL datasets from Zenodo..."
    
    # Install zenodo_get if not available
    if ! command -v zenodo_get >/dev/null 2>&1; then
        print_status "Installing zenodo_get for data download..."
        uv pip install zenodo_get || {
            print_warning "Failed to install zenodo_get, will provide manual download instructions"
        }
    fi
    
    # Download from Zenodo using zenodo-get
    # DOI: https://doi.org/10.5281/zenodo.15813500
    
    if command -v zenodo_get >/dev/null 2>&1; then
        print_status "Using zenodo_get to download datasets..."
        
        # Create temporary download directory
        TEMP_DOWNLOAD_DIR=$(mktemp -d)
        cd "$TEMP_DOWNLOAD_DIR"
        
        # Download from Zenodo record
        print_status "Downloading from Zenodo record 15813500..."
        zenodo_get 10.5281/zenodo.15813500 || {
            print_error "Failed to download from Zenodo using zenodo_get"
            cd - > /dev/null
            rm -rf "$TEMP_DOWNLOAD_DIR"
            return 1
        }
        
        # Move files to appropriate locations
        cd - > /dev/null
        
        if [[ -f "$TEMP_DOWNLOAD_DIR/templ_protein_embeddings_v1.0.0.npz" ]]; then
            mv "$TEMP_DOWNLOAD_DIR/templ_protein_embeddings_v1.0.0.npz" data/embeddings/
            print_success "Protein embeddings downloaded and moved to data/embeddings/"
        fi
        
        if [[ -f "$TEMP_DOWNLOAD_DIR/templ_processed_ligands_v1.0.0.sdf.gz" ]]; then
            mv "$TEMP_DOWNLOAD_DIR/templ_processed_ligands_v1.0.0.sdf.gz" data/ligands/
            print_success "Processed ligands downloaded and moved to data/ligands/"
        fi
        
        # Clean up temporary directory
        rm -rf "$TEMP_DOWNLOAD_DIR"
        
    else
        # Fallback: Use wget with direct URLs (if available)
        print_warning "zenodo_get not available, attempting direct download with wget..."
        
        # Note: These URLs would need to be updated with actual Zenodo file URLs
        # For now, we'll provide instructions to the user
        print_error "Direct download URLs not available. Please manually download:"
        echo ""
        echo "1. Visit: https://doi.org/10.5281/zenodo.15813500"
        echo "2. Download the following files:"
        echo "   - templ_protein_embeddings_v1.0.0.npz"
        echo "   - templ_processed_ligands_v1.0.0.sdf.gz"
        echo "3. Place them in the appropriate directories:"
        echo "   - templ_protein_embeddings_v1.0.0.npz → data/embeddings/"
        echo "   - templ_processed_ligands_v1.0.0.sdf.gz → data/ligands/"
        echo ""
        print_warning "Setup will continue, but you'll need to download data files manually"
        return 0
    fi
    
    # Verify downloads
    if [[ -f "data/embeddings/templ_protein_embeddings_v1.0.0.npz" ]] && [[ -f "data/ligands/templ_processed_ligands_v1.0.0.sdf.gz" ]]; then
        print_success "All data files downloaded successfully!"
        
        # Display file sizes
        print_status "Downloaded file information:"
        ls -lh data/embeddings/templ_protein_embeddings_v1.0.0.npz 2>/dev/null || true
        ls -lh data/ligands/templ_processed_ligands_v1.0.0.sdf.gz 2>/dev/null || true
    else
        print_warning "Some data files may not have downloaded correctly"
        print_status "You can download them manually from: https://doi.org/10.5281/zenodo.15813500"
    fi
}

# Verify installation
verify_installation() {
    print_section "Installation Verification"
    
    # Test core imports
    print_status "Testing core imports..."
    python3 -c "
import templ_pipeline
import numpy
import pandas
import rdkit
print('✓ Core modules imported successfully')
" || {
        print_error "Core module import failed"
        return 1
    }
    
    # Test web components if installed
    if [[ "$INSTALL_MODE" == "web" ]] || [[ "$INSTALL_MODE" == "full" ]] || [[ "$INSTALL_MODE" == "dev" ]]; then
        print_status "Testing web components..."
        python3 -c "
import streamlit
print('✓ Web components imported successfully')
" || {
            print_warning "Web components import failed"
        }
    fi
    
    # Test Embedding components if installed
    if [[ "$INSTALL_MODE" == "full" ]] || [[ "$INSTALL_MODE" == "dev" ]]; then
        print_status "Testing Embedding components..."
        python3 -c "
import torch
import transformers
print('✓ Embedding components imported successfully')
" || {
            print_warning "Embedding components import failed"
        }
    fi
    
    # Test CLI command
    print_status "Testing CLI command..."
    if command -v templ >/dev/null 2>&1; then
        templ --version >/dev/null 2>&1 && print_success "✓ CLI command works" || print_warning "CLI command found but version check failed"
    else
        print_warning "CLI command not found (may need to restart shell)"
    fi
    
    print_success "Installation verification completed"
}

# Show final instructions
show_final_instructions() {
    print_section "Setup Complete!"
    
    echo -e "${GREEN}✓ TEMPL Pipeline environment created successfully!${NC}"
    echo
    echo -e "${CYAN}Environment Details:${NC}"
    echo "  Location: $(pwd)/$VENV_NAME"
    echo "  Mode: $INSTALL_MODE"
    echo "  Python: $(python3 --version)"
    echo "  Hardware: $CPU_CORES cores, ${RAM_GB}GB RAM, GPU: $GPU_AVAILABLE"
    echo "  Data Files: Automatically downloaded from Zenodo"
    echo
    
    echo -e "${CYAN}Usage:${NC}"
    if [[ "$INSTALL_MODE" == "web" ]] || [[ "$INSTALL_MODE" == "full" ]] || [[ "$INSTALL_MODE" == "dev" ]]; then
        echo "  # Start web interface"
        echo "  python run_streamlit_app.py"
        echo
    fi
    
    echo "  # CLI usage"
    echo "  templ --help"
    echo "  templ run --protein-file examples/1a1c_protein.pdb --ligand-smiles 'CCO' --output poses.sdf"
    echo
    
    echo -e "${CYAN}For future sessions:${NC}"
    echo "  source $VENV_NAME/bin/activate"
    echo "  "
    echo -e "${CYAN}Quick status check:${NC}"
    echo "  ./manage_environment.sh status"
    echo "  "
    echo -e "${CYAN}Get help:${NC}"
    echo "  ./manage_environment.sh help"
    echo "  templ --help"
    echo
    
    if [[ "$INSTALL_MODE" == "dev" ]]; then
        echo -e "${CYAN}Development commands:${NC}"
        echo "  pytest                    # Run tests"
        echo "  templ benchmark polaris   # Run benchmarks"
        echo
    fi
    
    echo -e "${YELLOW}Note: The environment is now active for this session.${NC}"
}

# Main execution function
main() {
    # Check if we're being sourced
    check_sourced
    
    # Parse arguments (this will also load config)
    parse_args "$@"
    
    # Show header (suppress if quiet mode)
    if [[ "$VERBOSE" != "false" ]]; then
    cat << 'HEADER'

 ████████╗███████╗███╗   ███╗██████╗ ██╗     
 ╚══██╔══╝██╔════╝████╗ ████║██╔══██╗██║     
    ██║   █████╗  ██╔████╔██║██████╔╝██║     
    ██║   ██╔══╝  ██║╚██╔╝██║██╔═══╝ ██║     
    ██║   ███████╗██║ ╚═╝ ██║██║     ███████╗
    ╚═╝   ╚══════╝╚═╝     ╚═╝╚═╝     ╚══════╝
                                            
Template-based Protein-Ligand Pose Prediction
Environment Setup Script v2.0

HEADER
    
        echo -e "${CYAN}Setting up TEMPL Pipeline environment...${NC}"
    fi
    
    # Main setup steps
    detect_hardware
    recommend_installation
    create_venv
    install_dependencies
    setup_data_files
    verify_installation
    
    # Show final instructions (suppress if quiet mode)
    if [[ "$VERBOSE" != "false" ]]; then
        show_final_instructions
    else
        print_success "TEMPL environment setup complete!"
        echo "Run 'source $VENV_NAME/bin/activate' to activate"
    fi
    
    # Return success
    return 0
}

# Execute main function with all arguments
main "$@"
