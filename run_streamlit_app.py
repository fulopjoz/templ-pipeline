#!/usr/bin/env python
"""
Helper script to run the TEMPL Pipeline Web Application.
Runs the redesigned, modern interface with one-click workflow.
Updated with working configuration that resolves WebSocket connection issues.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import streamlit.web.cli as stcli
except ImportError:
    print("Error: Streamlit is not installed. Please install it with:")
    print("pip install streamlit")
    sys.exit(1)

def main():
    """Run the Streamlit app with the working configuration."""
    streamlit_app_path = project_root / "templ_pipeline" / "ui" / "app.py"
    
    if not streamlit_app_path.exists():
        print(f"Error: Streamlit app not found at: {streamlit_app_path}")
        sys.exit(1)
    
    # Configure sys.argv for Streamlit with WORKING settings
    # These settings resolve the WebSocket connection issues
    sys.argv = [
        "streamlit", 
        "run", 
        str(streamlit_app_path),
        "--server.port", "8502",                    # Use working port
        "--server.address", "127.0.0.1",           # Use localhost binding (fixes WebSocket)
        "--server.headless", "false",               # Enable browser integration
        "--server.enableCORS", "true",              # Enable CORS for remote access
        "--server.enableXsrfProtection", "false",   # Disable XSRF for development
        "--browser.gatherUsageStats", "false"       # Disable usage stats
    ]
    
    # Check if PYTHONPATH is set, if not, set it
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = str(project_root)
    
    print("🚀 Starting TEMPL Pipeline with working configuration...")
    print("📍 URL: http://127.0.0.1:8502")
    print("⚙️  Configuration: localhost binding, CORS enabled, headless=false")
    
    # Run the Streamlit app
    sys.exit(stcli.main())

if __name__ == "__main__":
    main() 