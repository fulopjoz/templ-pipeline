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
TEMPL Pipeline Environment Setup

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
  --help              Show this help message

EXAMPLES:
  source setup_templ_env.sh                    # Auto-detect and install optimally
  source setup_templ_env.sh --cpu-only         # Lightweight installation
  source setup_templ_env.sh --gpu-force --dev # Force GPU + development tools
  source setup_templ_env.sh --web              # Standard web interface

NOTES:
  - Must use 'source' command to activate environment
  - Requires Python 3.9+ and pip
  - Auto-detects: CPU cores, RAM, GPU availability
  - Creates .templ virtual environment in project directory
USAGE
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --auto)
                INSTALL_MODE="auto"
                shift
                ;;
            --cpu-only)
                INSTALL_MODE="cpu-only"
                shift
                ;;
            --gpu-force)
                INSTALL_MODE="gpu-force"
                shift
                ;;
            --minimal)
                INSTALL_MODE="minimal"
                shift
                ;;
            --web)
                INSTALL_MODE="web"
                shift
                ;;
            --full)
                INSTALL_MODE="full"
                shift
                ;;
            --dev)
                INSTALL_MODE="dev"
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
            --help|-h)
                show_usage
                return 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                return 1
                ;;
        esac
    done
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
        print_status "Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Source the shell configuration to make uv available
        source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
    fi
    
    if [[ -d "$VENV_NAME" ]]; then
        print_warning "Virtual environment $VENV_NAME already exists"
        read -p "Remove and recreate? [y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_NAME"
            print_status "Removed existing environment"
        else
            print_status "Using existing environment"
            return 0
        fi
    fi
    
    print_status "Creating virtual environment: $VENV_NAME"
    uv venv "$VENV_NAME"
    
    print_status "Activating virtual environment"
    source "$VENV_NAME/bin/activate"
    
    # Upgrade pip
    print_status "Upgrading pip"
    uv pip install --upgrade pip
    
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
    
    # Parse arguments
    parse_args "$@"
    
    # Show header
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
    
    # Main setup steps
    detect_hardware
    recommend_installation
    create_venv
    install_dependencies
    verify_installation
    show_final_instructions
    
    # Return success
    return 0
}

# Execute main function with all arguments
main "$@"
