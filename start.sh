#!/bin/bash
set -e

echo "🚀 Starting TEMPL Pipeline Application"
echo "📍 Working directory: $(pwd)"
echo "🔧 Environment variables:"
echo "   PORT: ${PORT:-'not set'}"
echo "   STREAMLIT_SERVER_PORT: ${STREAMLIT_SERVER_PORT:-'not set'}"

# Set Streamlit port from DigitalOcean's PORT environment variable
export STREAMLIT_SERVER_PORT="${PORT:-8080}"
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
export STREAMLIT_SERVER_HEADLESS="true"
export STREAMLIT_SERVER_ENABLE_CORS="false"
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION="false"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"

echo "✅ Streamlit configuration:"
echo "   STREAMLIT_SERVER_PORT: $STREAMLIT_SERVER_PORT"
echo "   STREAMLIT_SERVER_ADDRESS: $STREAMLIT_SERVER_ADDRESS"

# Verify essential files exist
echo "🔍 Verifying essential files..."
if [ ! -f "templ_pipeline/ui/app.py" ]; then
    echo "❌ Error: templ_pipeline/ui/app.py not found"
    exit 1
fi

if [ ! -d "data-minimal" ]; then
    echo "❌ Error: data-minimal directory not found"
    exit 1
fi

echo "✅ All essential files verified"

# Start Streamlit application
echo "🎯 Starting Streamlit on port $STREAMLIT_SERVER_PORT..."
exec streamlit run templ_pipeline/ui/app.py
