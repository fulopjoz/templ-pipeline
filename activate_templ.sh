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
