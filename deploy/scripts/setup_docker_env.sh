#!/bin/bash
# TEMPL Pipeline Docker Environment Setup Script
# Docker-compatible version of setup_templ_env.sh
# Creates optimized virtual environment for container deployment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default settings for Docker
INSTALL_MODE="full"  # Changed from "web" to "full" to include PyTorch/transformers
VENV_NAME=".templ"
PYTHON_MIN_VERSION="3.9"
USE_REQUIREMENTS_TXT=false
VERBOSE=false
INTERACTIVE=false
CONFIG_FILE=".templ.config"
DOWNLOAD_DATA=true  # Changed to true to ensure data files are downloaded

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

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --web)
                INSTALL_MODE="web"
                shift
                ;;
            --full)
                INSTALL_MODE="full"
                shift
                ;;
            --cpu-only)
                INSTALL_MODE="cpu-only"
                shift
                ;;
            --minimal)
                INSTALL_MODE="minimal"
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
            --quiet)
                VERBOSE=false
                shift
                ;;
            --non-interactive)
                INTERACTIVE=false
                shift
                ;;
            --download-data)
                DOWNLOAD_DATA=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Show usage
show_usage() {
    cat << 'USAGE'
TEMPL Pipeline Docker Environment Setup

Usage: bash setup_docker_env.sh [OPTIONS]

OPTIONS:
  --web               Standard installation with web interface (default)
  --full              Full installation with embedding features
  --cpu-only          Lightweight CPU-only installation
  --minimal           Minimal server installation
  --dev               Development environment
  --use-requirements  Use requirements.txt instead of pyproject.toml
  --verbose           Verbose output for debugging
  --quiet             Minimal output for automation
  --non-interactive   Skip all prompts (use defaults)
  --download-data     Download data files during build
  --help              Show this help message

EXAMPLES:
  bash setup_docker_env.sh --web                    # Standard web interface
  bash setup_docker_env.sh --full --download-data   # Full with data download
  bash setup_docker_env.sh --cpu-only --quiet       # Lightweight, quiet

NOTES:
  - Designed for Docker container environments
  - Creates .templ virtual environment
  - Non-interactive by default
  - Optimized for container deployment
USAGE
}

# Detect system hardware (simplified for Docker)
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
        RAM_GB=8  # Default assumption for containers
    fi
    print_status "RAM: ${RAM_GB}GB"
    
    # GPU detection (simplified for containers)
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

# Create virtual environment
create_venv() {
    print_section "Creating Virtual Environment"
    
    # Check if uv is available
    if ! command -v uv >/dev/null 2>&1; then
        print_status "Installing uv package manager..."
        pip install uv
    fi
    
    # Create virtual environment
    if [[ -d "$VENV_NAME" ]]; then
        print_warning "Virtual environment already exists, removing..."
        rm -rf "$VENV_NAME"
    fi
    
    print_status "Creating virtual environment: $VENV_NAME"
    uv venv "$VENV_NAME"
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source "$VENV_NAME/bin/activate"
    
    print_success "Virtual environment created and activated"
}

# Install dependencies
install_dependencies() {
    print_section "Installing Dependencies"
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    if [[ "$USE_REQUIREMENTS_TXT" == "true" ]]; then
        install_from_requirements
    else
        install_from_pyproject
    fi
}

# Install from requirements.txt
install_from_requirements() {
    print_status "Installing from requirements.txt"
    
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
            print_status "Installing: Complete TEMPL Pipeline (Core + Web + Embeddings + Performance)"
            ;;
        dev)
            extras="[dev]"
            print_status "Installing: Development environment (Full + Dev tools)"
            ;;
    esac
    
    # Install package with appropriate extras
    if [[ -n "$extras" ]]; then
        print_status "Installing with extras: $extras"
        uv pip install -e ".$extras"
    else
        print_status "Installing core package only"
        uv pip install -e .
    fi
    
    print_success "Installed from pyproject.toml"
}

# Setup data files (optional for Docker)
setup_data_files() {
    print_section "Data Files Setup"
    
    # Create data directory structure
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
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
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
    print_section "Verifying Installation"
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
    
    # Test Python imports
    print_status "Testing Python imports..."
    python3 -c "
import sys
import os
required_modules = ['streamlit', 'rdkit', 'numpy', 'pandas'] if os.environ.get('TEMPL_SKIP_IMPORT_TEST') == '1' else ['streamlit', 'templ_pipeline', 'rdkit', 'numpy', 'pandas']
missing = []
for module in required_modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError:
        missing.append(module)
        print(f'❌ {module}')

if missing:
    print(f'Missing modules: {missing}')
    sys.exit(1)
else:
    print('✅ All critical dependencies available')
"
    
    if [[ $? -ne 0 ]]; then
        print_error "Dependency verification failed"
        return 1
    fi
    
    # Test embedding components (for full installation)
    if [[ "$INSTALL_MODE" == "full" ]] || [[ "$INSTALL_MODE" == "dev" ]]; then
        print_status "Testing embedding components..."
        python3 -c "
import sys
embedding_modules = ['torch', 'transformers']
missing = []
for module in embedding_modules:
    try:
        __import__(module)
        print(f'✅ {module}')
    except ImportError:
        missing.append(module)
        print(f'❌ {module}')

if missing:
    print(f'Missing embedding modules: {missing}')
    print('Embedding features will be disabled')
else:
    print('✅ All embedding dependencies available')
"
    fi
    
    # Test CLI command
    if command -v templ >/dev/null 2>&1; then
        print_success "CLI command available"
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
    
    echo -e "${CYAN}For container usage:${NC}"
    echo "  source $VENV_NAME/bin/activate"
    echo "  python run_streamlit_app.py"
    echo
}

# Main execution function
main() {
    # Parse arguments
    parse_args "$@"
    
    # Show header
    if [[ "$VERBOSE" == "true" ]]; then
        cat << 'HEADER'

 ████████╗███████╗███╗   ███╗██████╗ ██╗     
 ╚══██╔══╝██╔════╝████╗ ████║██╔══██╗██║     
    ██║   █████╗  ██╔████╔██║██████╔╝██║     
    ██║   ██╔══╝  ██║╚██╔╝██║██╔═══╝ ██║     
    ██║   ███████╗██║ ╚═╝ ██║██║     ███████╗
    ╚═╝   ╚══════╝╚═╝     ╚═╝╚═╝     ╚══════╝
                                            
Template-based Protein-Ligand Pose Prediction
Docker Environment Setup Script v1.0

HEADER
        echo -e "${CYAN}Setting up TEMPL Pipeline environment for Docker...${NC}"
    fi
    
    # Main setup steps
    detect_hardware
    create_venv
    install_dependencies
    setup_data_files
    verify_installation
    
    # Show final instructions
    if [[ "$VERBOSE" == "true" ]]; then
        show_final_instructions
    else
        print_success "TEMPL environment setup complete!"
        echo "Virtual environment: $VENV_NAME"
    fi
    
    # Return success
    return 0
}

# Execute main function with all arguments
main "$@"