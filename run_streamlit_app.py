#!/usr/bin/env python
"""
TEMPL Pipeline - Streamlit Web Application Launcher
Simple, user-friendly startup with automatic URL display and comprehensive dependency checking.
"""

import os
import sys
import socket
from pathlib import Path

def get_network_urls(port=8501):
    """Get all available URLs for the Streamlit app"""
    urls = {
        'local': f'http://localhost:{port}',
        'network': None,
        'external': None
    }
    
    try:
        # Get local network IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        urls['network'] = f'http://{local_ip}:{port}'
        urls['external'] = f'http://{local_ip}:{port}'
    except:
        pass
    
    return urls

def check_dependencies():
    """Check if required dependencies are installed"""
    # Core dependencies that must be present (import_name: description)
    core_deps = {
        'streamlit': 'Web interface',
        'rdkit': 'Molecular processing', 
        'numpy': 'Numerical computing',
        'pandas': 'Data handling',
        'Bio': 'Protein processing (biopython)',
        'biotite': 'Structural biology',
        'templ_pipeline': 'Pipeline core'
    }
    
    missing = []
    for dep, description in core_deps.items():
        try:
            __import__(dep)
        except ImportError:
            missing.append(f'{dep} ({description})')
    
    if missing:
        print("Missing core dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print()
        print("Install with:")
        print("  pip install -e \".[web]\"     # Standard installation")
        print("  pip install -e \".[full]\"    # Full installation with AI features")
        print()
        print("Or install individual dependencies:")
        if any('rdkit' in dep for dep in missing):
            print("  conda install -c conda-forge rdkit")
        if any('Bio' in dep for dep in missing):
            print("  pip install biopython")
        return False
    
    return True

def main():
    """Launch TEMPL Pipeline with smart configuration"""
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Check for app file
    app_path = Path("templ_pipeline/ui/app.py")
    if not app_path.exists():
        print(f"App file not found: {app_path}")
        sys.exit(1)
    
    # Smart port selection
    port = 8501
    if os.getenv('PORT'):
        port = int(os.getenv('PORT'))
    elif os.getenv('STREAMLIT_SERVER_PORT'):
        port = int(os.getenv('STREAMLIT_SERVER_PORT'))
    
    # Get URLs
    urls = get_network_urls(port)
    
    # Display startup info
    print("Starting TEMPL Pipeline")
    print(f"Local:    {urls['local']}")
    if urls['network']:
        print(f"Network:  {urls['network']}")
    if urls['external'] and urls['external'] != urls['network']:
        print(f"External: {urls['external']}")
    print()
    
    # Configure environment
    os.environ.update({
        'STREAMLIT_SERVER_PORT': str(port),
        'STREAMLIT_SERVER_ADDRESS': '0.0.0.0',
        'STREAMLIT_SERVER_HEADLESS': 'true',
        'STREAMLIT_SERVER_ENABLE_CORS': 'true',
        'STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION': 'false',
        'STREAMLIT_BROWSER_GATHER_USAGE_STATS': 'false'
    })
    
    # Add project to Python path
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Launch app
    import streamlit.web.cli as stcli
    sys.argv = ['streamlit', 'run', str(app_path), '--server.port', str(port)]
    stcli.main()

if __name__ == "__main__":
    main()
