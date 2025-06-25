#!/bin/bash
set -e

echo "🚀 Starting TEMPL Pipeline Application"
echo "📍 Working directory: $(pwd)"
echo "🕐 Timestamp: $(date)"
echo "🔧 Raw environment variables:"
env | grep -E "(PORT|STREAMLIT)" | sort || echo "No PORT/STREAMLIT vars found"

# CRITICAL: Unset any conflicting environment variables first
echo "🧹 Clearing any existing STREAMLIT environment variables..."
unset STREAMLIT_SERVER_PORT

# Set port from DigitalOcean's PORT, with extensive logging
echo "🔌 Configuring port settings..."
if [ -n "$PORT" ]; then
    export STREAMLIT_SERVER_PORT="$PORT"
    echo "✅ Using DigitalOcean PORT: $PORT"
    echo "✅ STREAMLIT_SERVER_PORT set to: $STREAMLIT_SERVER_PORT"
else
    export STREAMLIT_SERVER_PORT="8080"
    echo "⚠️  PORT environment variable not set by DigitalOcean"
    echo "✅ Using default port: 8080"
    echo "✅ STREAMLIT_SERVER_PORT set to: $STREAMLIT_SERVER_PORT"
fi

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
    echo "�� Current directory contents:"
    ls -la
    exit 1
fi

echo "✅ All essential files verified"

# Final environment check before starting
echo "🔍 Final environment check:"
echo "   PORT (from DigitalOcean): ${PORT:-'NOT SET'}"
echo "   STREAMLIT_SERVER_PORT (our setting): ${STREAMLIT_SERVER_PORT:-'NOT SET'}"

# Start Streamlit application with explicit configuration
echo "🎯 Starting Streamlit on port $STREAMLIT_SERVER_PORT..."
echo "🎯 Command: streamlit run templ_pipeline/ui/app.py"

# Use exec to replace the shell process
exec streamlit run templ_pipeline/ui/app.py
