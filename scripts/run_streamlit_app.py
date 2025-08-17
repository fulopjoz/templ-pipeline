#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
TEMPL Pipeline - Streamlit Web Application Launcher

Clean launcher for the TEMPL Pipeline Streamlit application.
"""

import os
import socket
import sys
import time
from pathlib import Path


def is_port_available(port, host="localhost"):
    """Check if a port is available for binding"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.settimeout(1)
            s.bind((host, port))
            return True
    except (OSError, socket.timeout):
        return False


def find_available_port(start_port=8501, max_attempts=10, host="localhost"):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port, host):
            return port

    raise RuntimeError(
        f"No available ports found in range {start_port}-{start_port + max_attempts - 1}"
    )


def get_network_urls(port=8501):
    """Get all available URLs for the Streamlit app"""
    urls = {"local": f"http://localhost:{port}", "network": None}

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        urls["network"] = f"http://{local_ip}:{port}"
    except Exception:
        pass

    return urls


def main():
    """Launch TEMPL Pipeline Streamlit application"""

    print("TEMPL Pipeline Launcher")
    print("=" * 50)

    # Check if main app exists
    app_path = Path("templ_pipeline/ui/app.py")
    if not app_path.exists():
        print(f"Error: Main application file not found: {app_path}")
        print("Please ensure you're running from the project root directory.")
        sys.exit(1)

    # Check basic dependencies
    try:
        import streamlit
    except ImportError:
        print("Error: Streamlit not installed. Please install: pip install streamlit")
        sys.exit(1)

    # Select port
    try:
        start_port = int(os.getenv("TEMPL_PORT_START", "8501"))
    except ValueError:
        start_port = 8501

    # Check for explicit port
    explicit_port = os.getenv("PORT")
    if explicit_port:
        try:
            port = int(explicit_port)
            if not is_port_available(port):
                print(
                    f"Warning: Port {port} not available, searching for alternative..."
                )
                port = find_available_port(start_port)
        except ValueError:
            print(f"Warning: Invalid PORT value '{explicit_port}', using default")
            port = find_available_port(start_port)
    else:
        port = find_available_port(start_port)

    print(f"Using port: {port}")

    # Get URLs
    urls = get_network_urls(port)
    print(f"\nAccess URLs:")
    print(f"  Local:    {urls['local']}")
    if urls["network"]:
        print(f"  Network:  {urls['network']}")

    print(f"\nStarting TEMPL Pipeline server...")

    # Configure environment
    os.environ.update(
        {
            "STREAMLIT_SERVER_PORT": str(port),
            "STREAMLIT_SERVER_ADDRESS": "0.0.0.0",
            "STREAMLIT_SERVER_HEADLESS": "true",
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
            "STREAMLIT_GLOBAL_LOG_LEVEL": "info",
            "STREAMLIT_SERVER_MAX_UPLOAD_SIZE": "200",
        }
    )

    # Add project to Python path
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        # Launch app
        import streamlit.web.cli as stcli

        print("Application starting...")
        print("\nTROUBLESHOOTING:")
        print("- If you see issues, check the browser console (F12)")
        print("- Try refreshing the page or clearing browser cache")

        sys.argv = ["streamlit", "run", str(app_path), "--server.port", str(port)]
        stcli.main()

    except KeyboardInterrupt:
        print("\nServer stopped by user")

    except Exception as e:
        print(f"\nError launching application: {e}")
        print("Please check that all dependencies are installed and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
