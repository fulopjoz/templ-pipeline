#!/bin/bash
set -e

# TEMPL Pipeline Data Initialization Script
# Downloads and sets up all required datasets in persistent storage

echo "======================================================="
echo "TEMPL Pipeline Data Initialization"
echo "======================================================="

# Configuration
DATA_DIR="/app/data"
EMBEDDINGS_DIR="$DATA_DIR/embeddings"
LIGANDS_DIR="$DATA_DIR/ligands"
PDBIND_DIR="$DATA_DIR/PDBBind"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${BLUE}[INFO] $1${NC}"; }
success() { echo -e "${GREEN}[SUCCESS] $1${NC}"; }
warning() { echo -e "${YELLOW}[WARNING] $1${NC}"; }
error() { echo -e "${RED}[ERROR] $1${NC}"; }

# Create directory structure
create_directories() {
    info "Creating directory structure..."
    mkdir -p "$EMBEDDINGS_DIR" "$LIGANDS_DIR" "$PDBIND_DIR"
    success "Directory structure created"
}

# Check if data already exists
check_existing_data() {
    info "Checking for existing data..."
    
    local has_embeddings=false
    local has_ligands=false
    local has_pdbind=false
    
    # Check for Zenodo data
    if [[ -f "$EMBEDDINGS_DIR/templ_protein_embeddings_v1.0.0.npz" ]]; then
        info "Protein embeddings found"
        has_embeddings=true
    fi
    
    if [[ -f "$LIGANDS_DIR/templ_processed_ligands_v1.0.0.sdf.gz" ]]; then
        info "Processed ligands found"
        has_ligands=true
    fi
    
    # Check for PDBBind data
    if [[ -d "$PDBIND_DIR/PDBbind_v2020_refined" ]] || [[ -d "$PDBIND_DIR/PDBbind_v2020_other_PL" ]]; then
        info "PDBBind data found"
        has_pdbind=true
    fi
    
    # Export status for later use
    export HAS_EMBEDDINGS=$has_embeddings
    export HAS_LIGANDS=$has_ligands
    export HAS_PDBIND=$has_pdbind
    
    if $has_embeddings && $has_ligands && $has_pdbind; then
        success "All required datasets are already present"
        return 0
    else
        info "Some datasets missing, will download:"
        $has_embeddings || info "  - Protein embeddings (Zenodo)"
        $has_ligands || info "  - Processed ligands (Zenodo)"
        $has_pdbind || info "  - PDBBind data (manual setup required)"
        return 1
    fi
}

# Download Zenodo datasets
download_zenodo_data() {
    if $HAS_EMBEDDINGS && $HAS_LIGANDS; then
        info "Zenodo datasets already present, skipping download"
        return 0
    fi
    
    info "Downloading TEMPL datasets from Zenodo..."
    
    # Create temporary download directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    # Download from Zenodo record using zenodo_get
    info "Downloading from Zenodo record 15813500..."
    zenodo_get 10.5281/zenodo.15813500 || {
        error "Failed to download from Zenodo"
        cd /data-setup
        rm -rf "$TEMP_DIR"
        return 1
    }
    
    # Move files to appropriate locations
    cd /data-setup
    
    if [[ -f "$TEMP_DIR/templ_protein_embeddings_v1.0.0.npz" ]] && ! $HAS_EMBEDDINGS; then
        mv "$TEMP_DIR/templ_protein_embeddings_v1.0.0.npz" "$EMBEDDINGS_DIR/"
        success "Protein embeddings moved to $EMBEDDINGS_DIR/"
    fi
    
    if [[ -f "$TEMP_DIR/templ_processed_ligands_v1.0.0.sdf.gz" ]] && ! $HAS_LIGANDS; then
        mv "$TEMP_DIR/templ_processed_ligands_v1.0.0.sdf.gz" "$LIGANDS_DIR/"
        success "Processed ligands moved to $LIGANDS_DIR/"
    fi
    
    # Clean up
    rm -rf "$TEMP_DIR"
    
    success "Zenodo datasets downloaded successfully"
}

# Download PDBBind data using existing Python script
download_pdbind_data() {
    if $HAS_PDBIND; then
        info "PDBBind data already present, skipping download"
        return 0
    fi
    
    info "PDBBind dataset setup has been removed for security"
    warning "Please configure PDBBind data manually using authorized access"
    warning "Refer to config/datasets.yaml.template for setup instructions"
    
    return 0
}

# Verify all data is present
verify_data() {
    info "Verifying all datasets..."
    
    local all_present=true
    
    # Check Zenodo data
    if [[ ! -f "$EMBEDDINGS_DIR/templ_protein_embeddings_v1.0.0.npz" ]]; then
        error "Missing: templ_protein_embeddings_v1.0.0.npz"
        all_present=false
    else
        success "Found: Protein embeddings"
    fi
    
    if [[ ! -f "$LIGANDS_DIR/templ_processed_ligands_v1.0.0.sdf.gz" ]]; then
        error "Missing: templ_processed_ligands_v1.0.0.sdf.gz"
        all_present=false
    else
        success "Found: Processed ligands"
    fi
    
    # Check PDBBind data
    if [[ ! -d "$PDBIND_DIR/PDBbind_v2020_refined" ]] && [[ ! -d "$PDBIND_DIR/PDBbind_v2020_other_PL" ]]; then
        error "Missing: PDBBind data directories"
        all_present=false
    else
        success "Found: PDBBind data"
    fi
    
    if $all_present; then
        success "All required datasets verified successfully"
        
        # Show disk usage summary
        info "Disk usage summary:"
        du -sh "$DATA_DIR"/* 2>/dev/null | sort -hr || true
        
        return 0
    else
        error "Data verification failed - some datasets are missing"
        return 1
    fi
}

# Main execution
main() {
    echo "Starting data initialization process..."
    echo "Target directory: $DATA_DIR"
    echo ""
    
    create_directories
    
    # Check what's already there
    if check_existing_data; then
        info "All data already present, verification only"
        verify_data
        success "Data initialization completed (no downloads needed)"
        return 0
    fi
    
    # Download missing data
    info "Downloading missing datasets..."
    
    # Download Zenodo data
    download_zenodo_data || {
        error "Failed to download Zenodo datasets"
        return 1
    }
    
    # Download PDBBind data  
    download_pdbind_data || {
        error "Failed to download PDBBind data"
        return 1
    }
    
    # Final verification
    verify_data || {
        error "Data verification failed after download"
        return 1
    }
    
    success "Data initialization completed successfully!"
    return 0
}

# Execute main function
main "$@"