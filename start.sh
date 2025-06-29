#!/bin/bash
set -e

echo "🚀 Starting TEMPL Pipeline Application"
echo "📍 Working directory: $(pwd)"
echo "🕐 Timestamp: $(date)"

# Debug: Show all environment variables to identify any issues
echo "🔍 ALL environment variables (for debugging):"
env | sort

echo "🔧 Environment variables related to PORT/STREAMLIT:"
env | grep -E "(PORT|STREAMLIT)" | sort || echo "No PORT/STREAMLIT vars found"

# CRITICAL: Unset any conflicting environment variables first
echo "🧹 Clearing any existing STREAMLIT environment variables..."
unset STREAMLIT_SERVER_PORT

# DEFENSIVE APPROACH: Always use port 8080 to match DigitalOcean configuration
# This avoids any issues with PORT environment variable handling
echo "🔌 Configuring port settings..."
echo "🔍 PORT variable check: '${PORT:-UNSET}'"

# Always use 8080 since that's what DigitalOcean expects (http_port: 8080)
export STREAMLIT_SERVER_PORT="8080"
echo "✅ Using fixed port 8080 (matches DigitalOcean http_port configuration)"
echo "✅ STREAMLIT_SERVER_PORT set to: $STREAMLIT_SERVER_PORT"

# Set all Streamlit configuration explicitly
export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
export STREAMLIT_SERVER_HEADLESS="true"
export STREAMLIT_SERVER_ENABLE_CORS="false"
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION="false"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"

echo "✅ Final Streamlit configuration:"
echo "   STREAMLIT_SERVER_PORT: '$STREAMLIT_SERVER_PORT'"
echo "   STREAMLIT_SERVER_ADDRESS: '$STREAMLIT_SERVER_ADDRESS'"
echo "   STREAMLIT_SERVER_HEADLESS: '$STREAMLIT_SERVER_HEADLESS'"

# Verify essential files exist
echo "🔍 Verifying essential files..."
if [ ! -f "templ_pipeline/ui/app.py" ]; then
    echo "❌ Error: templ_pipeline/ui/app.py not found"
    echo "📁 Current directory contents:"
    ls -la
    exit 1
fi

if [ ! -d "data-minimal" ]; then
    echo "❌ Error: data-minimal directory not found"
    echo "📁 Current directory contents:"
    ls -la
    exit 1
fi

echo "✅ All essential files verified"

# Final environment check before starting
echo "🔍 Final environment check:"
echo "   PORT (from DigitalOcean): ${PORT:-'NOT SET'}"
echo "   STREAMLIT_SERVER_PORT (our setting): ${STREAMLIT_SERVER_PORT}"

# Start Streamlit application with explicit configuration
echo "🎯 Starting Streamlit on port $STREAMLIT_SERVER_PORT..."
echo "🎯 Command: streamlit run templ_pipeline/ui/app.py"

# Use exec to replace the shell process - ONLY environment variables, no CLI args
exec streamlit run templ_pipeline/ui/app.py
