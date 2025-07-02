#!/usr/bin/env python3
"""
TEMPL Pipeline Environment Verification Script

Quick diagnostic tool to verify installation and identify issues.
"""

import sys
import subprocess
import importlib
import platform
from pathlib import Path

def print_status(message, status="INFO"):
    colors = {
        "INFO": "\033[0;34m[INFO]\033[0m",
        "SUCCESS": "\033[0;32m[SUCCESS]\033[0m", 
        "WARNING": "\033[1;33m[WARNING]\033[0m",
        "ERROR": "\033[0;31m[ERROR]\033[0m"
    }
    print(f"{colors.get(status, '')} {message}")

def check_python_version():
    """Check Python version compatibility"""
    print("\n=== Python Environment ===")
    version = sys.version_info
    print_status(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 9):
        print_status("Python version check passed", "SUCCESS")
        return True
    else:
        print_status(f"Python 3.9+ required, found {version.major}.{version.minor}", "ERROR")
        return False

def check_virtual_env():
    """Check if we're in a virtual environment"""
    print("\n=== Virtual Environment ===")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        venv_path = Path(sys.prefix)
        print_status(f"Virtual environment active: {venv_path}", "SUCCESS")
        
        if '.templ' in str(venv_path):
            print_status("Using TEMPL environment", "SUCCESS")
        else:
            print_status("Not using .templ environment", "WARNING")
        return True
    else:
        print_status("No virtual environment detected", "WARNING")
        return False

def check_module_import(module_name, display_name=None, required=True):
    """Check if a module can be imported"""
    if display_name is None:
        display_name = module_name
        
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        print_status(f"✓ {display_name}: {version}", "SUCCESS")
        return True
    except ImportError:
        status = "ERROR" if required else "WARNING"
        symbol = "✗" if required else "○"
        print_status(f"{symbol} {display_name}: Not installed", status)
        return False

def check_core_dependencies():
    """Check core TEMPL dependencies"""
    print("\n=== Core Dependencies ===")
    
    core_modules = [
        ("templ_pipeline", "TEMPL Pipeline", True),
        ("numpy", "NumPy", True),
        ("pandas", "Pandas", True),
        ("rdkit", "RDKit", True),
        ("Bio", "BioPython", True),
        ("biotite", "Biotite", True),
        ("spyrmsd", "spyRMSD", True),
        ("sklearn", "Scikit-learn", True),
    ]
    
    all_good = True
    for module, name, required in core_modules:
        if not check_module_import(module, name, required):
            all_good = False
    
    return all_good

def check_web_dependencies():
    """Check web interface dependencies"""
    print("\n=== Web Interface ===")
    
    web_modules = [
        ("streamlit", "Streamlit", False),
        ("stmol", "stmol", False),
    ]
    
    for module, name, required in web_modules:
        check_module_import(module, name, required)

def check_embedding_dependencies():
    """Check embedding/GPU acceleration dependencies"""
    print("\n=== Embedding Features ===")
    
    embedding_modules = [
        ("torch", "PyTorch", False),
        ("transformers", "Transformers", False),
    ]
    
    for module, name, required in embedding_modules:
        check_module_import(module, name, required)

def check_gpu_support():
    """Check GPU availability"""
    print("\n=== GPU Support ===")
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print_status(f"✓ GPU available: {gpu_name} ({gpu_count} devices)", "SUCCESS")
        else:
            print_status("○ No GPU available (CPU-only mode)", "INFO")
    except ImportError:
        print_status("○ PyTorch not installed (cannot check GPU)", "INFO")

def check_cli_command():
    """Check CLI command availability"""
    print("\n=== CLI Command ===")
    
    try:
        # Test if CLI exists and basic help works
        result = subprocess.run(['templ', '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print_status("✓ CLI command available", "SUCCESS")
            
            # Get version from Python module instead of CLI
            try:
                import templ_pipeline
                version = getattr(templ_pipeline, '__version__', 'unknown')
                print_status(f"✓ Version: {version}", "SUCCESS")
            except ImportError:
                print_status("⚠ Could not determine version", "WARNING")
            
            return True
        else:
            print_status("✗ CLI command failed", "ERROR")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_status("✗ CLI command not found", "ERROR")
        return False

def check_file_structure():
    """Check project file structure"""
    print("\n=== Project Structure ===")
    
    required_files = [
        "pyproject.toml",
        "requirements.txt", 
        "run_streamlit_app.py",
        "templ_pipeline/__init__.py",
    ]
    
    all_present = True
    for file_path in required_files:
        if Path(file_path).exists():
            print_status(f"✓ {file_path}", "SUCCESS")
        else:
            print_status(f"✗ {file_path} missing", "ERROR")
            all_present = False
    
    return all_present

def run_basic_functionality_test():
    """Test basic functionality"""
    print("\n=== Functionality Test ===")
    
    try:
        # Test basic import and usage
        from templ_pipeline import __version__
        print_status(f"✓ TEMPL Pipeline version: {__version__}", "SUCCESS")
        
        # Test RDKit molecule creation
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        if mol:
            print_status("✓ RDKit molecule creation works", "SUCCESS")
        else:
            print_status("✗ RDKit molecule creation failed", "ERROR")
            return False
            
        return True
        
    except Exception as e:
        print_status(f"✗ Functionality test failed: {e}", "ERROR")
        return False

def generate_report():
    """Generate summary report"""
    print("\n" + "="*50)
    print("ENVIRONMENT VERIFICATION SUMMARY")
    print("="*50)
    
    # System info
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version()),
        ("Virtual Environment", check_virtual_env()),
        ("Core Dependencies", check_core_dependencies()),
        ("Project Structure", check_file_structure()),
        ("CLI Command", check_cli_command()),
        ("Basic Functionality", run_basic_functionality_test()),
    ]
    
    # Optional checks
    check_web_dependencies()
    check_embedding_dependencies() 
    check_gpu_support()
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    for check_name, result in checks:
        status_symbol = "✓" if result else "✗"
        print(f"{status_symbol} {check_name}")
    
    print(f"\nPassed: {passed}/{total} checks")
    
    if passed == total:
        print_status("Environment verification PASSED", "SUCCESS")
        print("\nYour TEMPL Pipeline installation is working correctly!")
        return True
    else:
        print_status("Environment verification FAILED", "ERROR")
        print("\nSome issues were found. Please check the errors above.")
        return False

def show_usage():
    """Show usage information"""
    print("""
TEMPL Pipeline Environment Verification

This script checks your TEMPL Pipeline installation and identifies common issues.

Usage:
    python verify_environment.py

The script will:
- Check Python version compatibility
- Verify virtual environment setup
- Test core dependencies
- Check optional features (web, embeddings)
- Test basic functionality
- Generate a summary report

If issues are found, refer to ENVIRONMENT_GUIDE.md for troubleshooting steps.
""")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        show_usage()
        sys.exit(0)
    
    print("TEMPL Pipeline Environment Verification")
    print("=" * 40)
    
    success = generate_report()
    
    if not success:
        print("\nFor troubleshooting help, see:")
        print("- ENVIRONMENT_GUIDE.md")
        print("- templ --help")
        sys.exit(1)
    else:
        print("\nTo get started:")
        print("- templ --help")
        print("- python run_streamlit_app.py")
        sys.exit(0)
