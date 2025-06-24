#!/usr/bin/env python
"""
Helper script to run the TEMPL Pipeline Web Application.
Runs the redesigned, modern interface with one-click workflow.
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
    """Run the Streamlit app with the correct path configuration."""
    streamlit_app_path = project_root / "templ_pipeline" / "ui" / "app.py"
    
    if not streamlit_app_path.exists():
        print(f"Error: Streamlit app not found at: {streamlit_app_path}")
        sys.exit(1)
    
    # Pass arguments to Streamlit
    sys.argv = ["streamlit", "run", str(streamlit_app_path), "--", "--demo", "True"]
    
    # Check if PYTHONPATH is set, if not, set it
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = str(project_root)
    
    # Run the Streamlit app
    sys.exit(stcli.main())

if __name__ == "__main__":
    main() 