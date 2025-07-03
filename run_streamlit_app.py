#!/usr/bin/env python
"""
TEMPL Pipeline - Streamlit Web Application Launcher

Advanced launcher with dynamic port selection, automatic dependency checking,
and intelligent fallback mechanisms for robust deployment.

Features:
- Dynamic port selection with automatic fallback
- Comprehensive dependency validation
- Network URL detection for multi-device access
- Environment variable configuration
- Detailed user feedback and error handling

Environment Variables:
- PORT: Explicit port number (overrides default)
- STREAMLIT_SERVER_PORT: Alternative port specification
- TEMPL_PORT_START: Starting port for auto-selection (default: 8501)
- TEMPL_PORT_RANGE: Number of ports to try (default: 10)

Usage:
    python run_streamlit_app.py
    PORT=9000 python run_streamlit_app.py
    TEMPL_PORT_START=8500 TEMPL_PORT_RANGE=20 python run_streamlit_app.py
"""

import os
import socket
import sys
import time
from pathlib import Path


<<<<<<< HEAD
def is_port_available(port, host='localhost'):
    """Check if a port is available for binding
    
    Args:
        port: Port number to check
        host: Host address to bind to (default: localhost)
        
    Returns:
        bool: True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.settimeout(1)  # 1 second timeout
            s.bind((host, port))
            return True
    except (OSError, socket.timeout):
        return False


def find_available_port(start_port=8501, max_attempts=10, host='localhost'):
    """Find an available port starting from start_port
    
    Args:
        start_port: Starting port number (default: 8501)
        max_attempts: Maximum number of ports to try (default: 10)
        host: Host address to check (default: localhost)
        
    Returns:
        int: Available port number
        
    Raises:
        RuntimeError: If no available ports found in range
    """
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port, host):
            return port
    
    raise RuntimeError(
        f"No available ports found in range {start_port}-{start_port + max_attempts - 1}. "
        f"Please free up a port or set a custom port using PORT environment variable."
    )


def get_network_urls(port=8501):
    """Get all available URLs for the Streamlit app
    
    Args:
        port: Port number for the URLs
        
    Returns:
        dict: Dictionary with local, network, and external URLs
    """
    urls = {"local": f"http://localhost:{port}", "network": None, "external": None}

    try:
        # Get local network IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        urls["network"] = f"http://{local_ip}:{port}"
        urls["external"] = f"http://{local_ip}:{port}"
    except Exception:
        pass

    return urls


def check_dependencies():
    """Check if required dependencies are installed"""
    # Core dependencies that must be present (import_name: description)
    core_deps = {
        "streamlit": "Web interface",
        "rdkit": "Molecular processing",
        "numpy": "Numerical computing",
        "pandas": "Data handling",
        "Bio": "Protein processing (biopython)",
        "biotite": "Structural biology",
        "templ_pipeline": "Pipeline core",
    }

    missing = []
    for dep, description in core_deps.items():
        try:
            __import__(dep)
        except ImportError:
            missing.append(f"{dep} ({description})")

    if missing:
        print("Missing core dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print()
        print("Install with:")
        print('  uv pip install -e ".[web]"     # Standard installation')
        print(
            '  uv pip install -e ".[full]"    # Full installation with Embedding features'
        )
        print()
        print("Or install individual dependencies:")
        if any("rdkit" in dep for dep in missing):
            print("  conda install -c conda-forge rdkit")
        if any("Bio" in dep for dep in missing):
            print(" uv pip install biopython")
        return False

    return True


def select_port():
    """Smart port selection with fallback and user feedback
    
    Returns:
        tuple: (port, selection_message)
    """
    # Configuration from environment variables
    start_port = int(os.getenv("TEMPL_PORT_START", "8501"))
    port_range = int(os.getenv("TEMPL_PORT_RANGE", "10"))
    
    # Check for explicit port configuration
    explicit_port = None
    if os.getenv("PORT"):
        explicit_port = int(os.getenv("PORT"))
        source = "PORT environment variable"
    elif os.getenv("STREAMLIT_SERVER_PORT"):
        explicit_port = int(os.getenv("STREAMLIT_SERVER_PORT"))
        source = "STREAMLIT_SERVER_PORT environment variable"
    
    if explicit_port:
        # User specified a port explicitly
        if is_port_available(explicit_port):
            return explicit_port, f"Using port {explicit_port} from {source}"
        else:
            print(f"Warning: Port {explicit_port} (from {source}) is not available")
            print(f"Searching for alternative port starting from {start_port}...")
            try:
                port = find_available_port(start_port, port_range)
                return port, f"Using port {port} (fallback from {explicit_port})"
            except RuntimeError as e:
                raise RuntimeError(f"Explicit port {explicit_port} unavailable and {str(e)}")
    
    # No explicit port - use smart detection
    if is_port_available(start_port):
        return start_port, f"Using default port {start_port}"
    else:
        print(f"Port {start_port} in use, searching for alternative...")
        try:
            port = find_available_port(start_port + 1, port_range - 1)  # Skip already checked port
            return port, f"Port {start_port} in use, using {port} instead"
        except RuntimeError as e:
            raise RuntimeError(str(e))


def main():
    """Launch TEMPL Pipeline with smart configuration and dynamic port selection"""

    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)

    # Check for app file
    app_path = Path("templ_pipeline/ui/app.py")
    if not app_path.exists():
        print(f"Error: App file not found: {app_path}")
        print("Make sure you're running from the project root directory.")
        sys.exit(1)

    print("Starting TEMPL Pipeline...")
    print()
    
    # Smart port selection with user feedback
    try:
        port, port_message = select_port()
        print(port_message)
    except RuntimeError as e:
        print(f"Error: Port selection failed: {e}")
        print()
        print("Solutions:")
        print("   - Free up some ports (kill other web servers)")
        print("   - Set a custom port: export PORT=9000")
        print("   - Change port range: export TEMPL_PORT_RANGE=20")
        sys.exit(1)
    
    # Get URLs
    urls = get_network_urls(port)
    
    print()
    print("Access URLs:")
    print(f"   Local:    {urls['local']}")
    if urls["network"]:
        print(f"   Network:  {urls['network']}")
    if urls["external"] and urls["external"] != urls["network"]:
        print(f"   External: {urls['external']}")
    print()
    print("Starting server... (this may take a moment)")
    print()

    # Configure environment with selected port
    os.environ.update(
        {
            "STREAMLIT_SERVER_PORT": str(port),
            "STREAMLIT_SERVER_ADDRESS": "0.0.0.0",
            "STREAMLIT_SERVER_HEADLESS": "true",
            "STREAMLIT_SERVER_ENABLE_CORS": "true",
            "STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION": "false",
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        }
    )

    # Add project to Python path
    try:
        project_root = Path(__file__).resolve().parent
    except NameError:
        # Handle case when running as script/test
        project_root = Path.cwd()
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Launch app
    import streamlit.web.cli as stcli

    sys.argv = ["streamlit", "run", str(app_path), "--server.port", str(port)]
    stcli.main()


if __name__ == "__main__":
    main()
