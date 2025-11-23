#!/bin/bash
# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT

# Quick Start Script for TEMPL Pipeline
# Ultra-fast one-command setup and activation

set -e

echo "🚀 TEMPL Pipeline Quick Start"
echo "=============================="
echo ""

# Check if already in virtual environment
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "✓ Already in virtual environment: $VIRTUAL_ENV"
    echo ""
    echo "Available commands:"
    echo "  templ --help              # Show CLI help"
    echo "  templ run --help          # Run pose prediction"
    echo "  python scripts/run_streamlit_app.py  # Launch web UI"
    exit 0
fi

# Check if .templ exists
if [[ -d ".templ" ]]; then
    echo "✓ Virtual environment found"
    echo "  Activating..."
    source .templ/bin/activate
    echo ""
    echo "✓ Environment activated!"
    echo ""
    echo "Available commands:"
    echo "  templ --help              # Show CLI help"
    echo "  templ run --help          # Run pose prediction"
    echo "  python scripts/run_streamlit_app.py  # Launch web UI"
else
    echo "⚙ Virtual environment not found"
    echo "  Running setup (this will take a few minutes)..."
    echo ""
    
    # Run setup with auto mode
    source setup_templ_env.sh --auto --non-interactive
    
    if [[ $? -eq 0 ]]; then
        echo ""
        echo "✓ Setup complete and environment activated!"
        echo ""
        echo "Available commands:"
        echo "  templ --help              # Show CLI help"
        echo "  templ run --help          # Run pose prediction"
        echo "  python scripts/run_streamlit_app.py  # Launch web UI"
    else
        echo ""
        echo "✗ Setup failed. Please check the error messages above."
        exit 1
    fi
fi
