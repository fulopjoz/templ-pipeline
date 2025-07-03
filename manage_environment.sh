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
TEMPL Environment Management v2.1

Usage: ./manage_environment.sh <command> [options]

Commands:
    setup [mode]     Setup new environment (same as setup_templ_env.sh)
    activate         Show activation command
    status           Show environment status
    verify           Run environment verification
    update           Update dependencies
    clean            Remove environment
    info             Show detailed environment info
    doctor           Diagnose common issues
    config           Show configuration
    help             Show this help

Examples:
    ./manage_environment.sh setup --web
    ./manage_environment.sh status
    ./manage_environment.sh verify
    ./manage_environment.sh clean
    ./manage_environment.sh doctor

Configuration:
    Edit .templ.config to customize behavior
    
Troubleshooting:
    Use 'doctor' command for automatic issue detection
    Check logs in .templ/logs/ if available

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
            # Already activated, run verification
            python -c "
import sys, os
try:
    import templ_pipeline
    print('✅ Core: templ_pipeline imported successfully')
except ImportError as e:
    print(f'❌ Core: {e}')
    sys.exit(1)

try:
    import numpy, pandas, rdkit
    print('✅ Dependencies: numpy, pandas, rdkit available')
except ImportError as e:
    print(f'❌ Dependencies: {e}')
    sys.exit(1)

print('✅ Environment verification passed')
"
        else
            echo "Activating environment for verification..."
            source "$VENV_NAME/bin/activate" && cmd_verify
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

# New commands
cmd_doctor() {
    echo -e "${BLUE}Running environment diagnostics...${NC}"
    
    local issues=0
    
    # Check Python version
    if command -v python3 >/dev/null 2>&1; then
        python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
            echo "✅ Python: $python_version (OK)"
        else
            echo "❌ Python: $python_version (Need 3.9+)"
            ((issues++))
        fi
    else
        echo "❌ Python: Not found"
        ((issues++))
    fi
    
    # Check virtual environment
    if [[ -d "$VENV_NAME" ]]; then
        echo "✅ Virtual environment: Exists"
        
        # Check if it's working
        if "$VENV_NAME/bin/python" -c "import sys; print('Python OK')" >/dev/null 2>&1; then
            echo "✅ Virtual environment: Functional"
        else
            echo "❌ Virtual environment: Corrupted"
            ((issues++))
        fi
    else
        echo "❌ Virtual environment: Not found"
        echo "   Run: source setup_templ_env.sh"
        ((issues++))
    fi
    
    # Check disk space
    if command -v df >/dev/null 2>&1; then
        available_space=$(df . | tail -1 | awk '{print $4}')
        if [[ $available_space -gt 1000000 ]]; then  # 1GB in KB
            echo "✅ Disk space: Available"
        else
            echo "⚠️  Disk space: Low ($(df -h . | tail -1 | awk '{print $4}'))"
        fi
    fi
    
    # Check network connectivity
    if ping -c 1 pypi.org >/dev/null 2>&1; then
        echo "✅ Network: PyPI accessible"
    else
        echo "⚠️  Network: PyPI not reachable"
    fi
    
    # Summary
    echo
    if [[ $issues -eq 0 ]]; then
        echo -e "${GREEN}✅ No issues found!${NC}"
    else
        echo -e "${YELLOW}⚠️  Found $issues issue(s) that need attention${NC}"
        echo "Run the suggested commands above to fix them."
    fi
}

cmd_config() {
    echo -e "${BLUE}Environment Configuration:${NC}"
    
    if [[ -f ".templ.config" ]]; then
        echo "Configuration file: .templ.config"
        echo
        cat .templ.config
    else
        echo "No configuration file found."
        echo "Run setup to create default configuration."
    fi
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
    doctor)
        cmd_doctor
        ;;
    config)
        cmd_config
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use 'help' to see available commands."
        exit 1
        ;;
esac
