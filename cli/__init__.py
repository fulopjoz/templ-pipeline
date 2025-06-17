"""
TEMPL Pipeline CLI Module

This module provides the command-line interface for the TEMPL pipeline.
It includes the main entry point for CLI usage.
"""

def main():
    """Lazy import and call main function."""
    from templ_pipeline.cli.main import main as _main
    return _main()

__all__ = ["main"]
