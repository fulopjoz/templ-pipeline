#!/bin/bash
set -e

# Activate conda environment
source /opt/conda/bin/activate templ

# Set deployment mode
MODE=${1:-production}

echo "Starting TEMPL Pipeline in $MODE mode..."

# Data directory setup
if [ ! -d "/app/data/embeddings" ] || [ ! -d "/app/data/ligands" ]; then
    echo "Required data files not found. Checking for existing data..."
    
    # Check if data directory is mounted with existing data
    if [ "$(ls -A /app/data 2>/dev/null)" ]; then
        echo "Data directory contains files, assuming data is pre-loaded."
    else
        echo "Warning: No data found. The application may have limited functionality."
        echo "Please ensure data files are mounted or copied to /app/data/"
    fi
fi

# Environment-specific configuration
case "$MODE" in
    "development")
        export STREAMLIT_GLOBAL_LOG_LEVEL=debug
        export TEMPL_WORKERS=1
        echo "Development mode: Debug logging enabled, single worker"
        ;;
    "production")
        export STREAMLIT_GLOBAL_LOG_LEVEL=info
        export TEMPL_WORKERS=auto
        echo "Production mode: Info logging, auto workers"
        ;;
    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac

# Ensure temp directory exists and is writable
mkdir -p /app/temp
chmod 755 /app/temp

# Start application
echo "Starting Streamlit application..."
echo "Available at: http://0.0.0.0:8501"
exec python run_streamlit_app.py
