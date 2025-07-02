#!/bin/bash

# TEMPL Environment Management Utility
# Quick commands for common environment tasks

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

VENV_NAME=".templ"

show_usage() {
    cat << 'USAGE'
TEMPL Environment Management

Usage: ./manage_environment.sh <command>

Commands:
    setup [mode]     Setup new environment (same as setup_templ_env.sh)
    activate         Show activation command
    status           Show environment status
    verify           Run environment verification
    update           Update dependencies
    clean            Remove environment
    info             Show detailed environment info
    help             Show this help

Examples:
    ./manage_environment.sh setup --web
    ./manage_environment.sh status
    ./manage_environment.sh verify
    ./manage_environment.sh clean

USAGE
}

check_venv_exists() {
    if [[ ! -d "$VENV_NAME" ]]; then
        echo -e "${YELLOW}TEMPL environment not found.${NC}"
        echo "Run: source setup_templ_env.sh"
        return 1
    fi
    return 0
}

cmd_setup() {
    echo -e "${BLUE}Setting up TEMPL environment...${NC}"
    source setup_templ_env.sh "$@"
}

cmd_activate() {
    echo -e "${BLUE}To activate TEMPL environment:${NC}"
    echo "source $VENV_NAME/bin/activate"
    
    if check_venv_exists; then
        echo -e "${GREEN}Environment ready!${NC}"
    fi
}

cmd_status() {
    echo -e "${BLUE}Environment Status:${NC}"
    
    if [[ ! -d "$VENV_NAME" ]]; then
        echo "❌ Environment: Not created"
        echo "   Run: source setup_templ_env.sh"
        return 1
    else
        echo "✅ Environment: Created"
    fi
    
    # Check if currently activated
    if [[ "$VIRTUAL_ENV" == *".templ"* ]]; then
        echo "✅ Status: Active"
        echo "📍 Location: $VIRTUAL_ENV"
        
        # Show package info
        if command -v templ >/dev/null 2>&1; then
            echo "✅ CLI: Available"
            # Get version from Python module instead of CLI
            python -c "import templ_pipeline; print(f'   Version: {templ_pipeline.__version__}')" 2>/dev/null || echo "⚠️  CLI: Found but version check failed"
        else
            echo "❌ CLI: Not found"
        fi
        
        # Check key packages
        python -c "
import sys
try:
    import templ_pipeline
    print('✅ Core: Installed')
except ImportError:
    print('❌ Core: Missing')

try:
    import streamlit
    print('✅ Web: Available')
except ImportError:
    print('⚪ Web: Not installed')
    
try:
    import torch
    print('✅ Embedding: Available')
except ImportError:
    print('⚪ Embedding: Not installed')
" 2>/dev/null
        
    else
        echo "⚪ Status: Inactive"
        echo "   Run: source $VENV_NAME/bin/activate"
    fi
}

cmd_verify() {
    echo -e "${BLUE}Running environment verification...${NC}"
    
    if check_venv_exists; then
        if [[ "$VIRTUAL_ENV" == *".templ"* ]]; then
            python verify_environment.py
        else
            echo "Activating environment for verification..."
            source "$VENV_NAME/bin/activate" && python verify_environment.py
        fi
    fi
}

cmd_update() {
    echo -e "${BLUE}Updating environment...${NC}"
    
    if ! check_venv_exists; then
        return 1
    fi
    
    echo "Activating environment..."
    source "$VENV_NAME/bin/activate"
    
    echo "Updating pip..."
    pip install --upgrade pip
    
    echo "Updating TEMPL Pipeline..."
    if [[ -f "requirements.txt" ]]; then
        echo "Using requirements.txt..."
        pip install -r requirements.txt --upgrade
    else
        echo "Using pyproject.toml..."
        pip install -e .[web] --upgrade
    fi
    
    echo -e "${GREEN}Update complete!${NC}"
}

cmd_clean() {
    echo -e "${YELLOW}This will remove the TEMPL environment.${NC}"
    read -p "Are you sure? [y/N]: " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [[ -d "$VENV_NAME" ]]; then
            rm -rf "$VENV_NAME"
            echo -e "${GREEN}Environment removed.${NC}"
        else
            echo "Environment was not found."
        fi
    else
        echo "Cancelled."
    fi
}

cmd_info() {
    echo -e "${BLUE}Environment Information:${NC}"
    
    if ! check_venv_exists; then
        return 1
    fi
    
    # Basic info
    echo "Location: $(pwd)/$VENV_NAME"
    echo "Python: $($VENV_NAME/bin/python --version)"
    
    # Disk usage
    if command -v du >/dev/null 2>&1; then
        size=$(du -sh "$VENV_NAME" 2>/dev/null | cut -f1)
        echo "Size: $size"
    fi
    
    # Package count
    package_count=$($VENV_NAME/bin/pip list 2>/dev/null | wc -l)
    echo "Packages: $((package_count - 2))"  # Subtract header lines
    
    # Key packages with versions
    echo
    echo "Key Packages:"
    $VENV_NAME/bin/pip show templ-pipeline 2>/dev/null | grep Version || echo "  templ-pipeline: Not found"
    $VENV_NAME/bin/pip show streamlit 2>/dev/null | grep Version || echo "  streamlit: Not installed"
    $VENV_NAME/bin/pip show torch 2>/dev/null | grep Version || echo "  torch: Not installed"
    $VENV_NAME/bin/pip show rdkit 2>/dev/null | grep Version || echo "  rdkit: Not found"
}

# Main command dispatcher
case ${1:-help} in
    setup)
        shift
        cmd_setup "$@"
        ;;
    activate)
        cmd_activate
        ;;
    status)
        cmd_status
        ;;
    verify)
        cmd_verify
        ;;
    update)
        cmd_update
        ;;
    clean)
        cmd_clean
        ;;
    info)
        cmd_info
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac
