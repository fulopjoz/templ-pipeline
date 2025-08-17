#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Test runner for TEMPL Pipeline UI tests.

This script provides a convenient way to run different types of tests
with appropriate configuration and reporting.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path


def run_command(cmd, cwd=None):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


def install_dependencies():
    """Install required dependencies for testing."""
    print("Installing test dependencies...")

    dependencies = [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-html>=3.1.0",
        "pytest-cov>=4.0.0",
        "playwright>=1.30.0",
        "requests>=2.28.0",
    ]

    for dep in dependencies:
        success = run_command([sys.executable, "-m", "pip", "install", dep])
        if not success:
            print(f"Failed to install {dep}")
            return False

    # Install Playwright browsers
    print("Installing Playwright browsers...")
    success = run_command([sys.executable, "-m", "playwright", "install", "chromium"])
    if not success:
        print("Failed to install Playwright browsers")
        return False

    return True


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests for Streamlit components."""
    print("Running unit tests...")

    cmd = [sys.executable, "-m", "pytest", "test_app.py"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=../", "--cov-report=html", "--cov-report=term"])

    cmd.extend(["-m", "not e2e"])  # Exclude E2E tests

    return run_command(cmd)


def run_e2e_tests(verbose=False, headless=True):
    """Run end-to-end tests."""
    print("Running end-to-end tests...")

    cmd = [sys.executable, "-m", "pytest", "test_e2e.py"]

    if verbose:
        cmd.append("-v")

    cmd.extend(["-m", "e2e"])

    if not headless:
        # Set environment variable for non-headless testing
        os.environ["PYTEST_HEADLESS"] = "false"

    return run_command(cmd)


def run_all_tests(verbose=False, coverage=False):
    """Run all tests."""
    print("Running all tests...")

    cmd = [sys.executable, "-m", "pytest", "."]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend(["--cov=../", "--cov-report=html", "--cov-report=term"])

    # Generate HTML report
    cmd.extend(["--html=test_report.html", "--self-contained-html"])

    return run_command(cmd)


def run_performance_tests(verbose=False):
    """Run performance tests."""
    print("Running performance tests...")

    cmd = [sys.executable, "-m", "pytest", "-k", "performance", "."]

    if verbose:
        cmd.append("-v")

    return run_command(cmd)


def run_accessibility_tests(verbose=False):
    """Run accessibility tests."""
    print("Running accessibility tests...")

    cmd = [sys.executable, "-m", "pytest", "-k", "accessibility", "."]

    if verbose:
        cmd.append("-v")

    return run_command(cmd)


def check_streamlit_app():
    """Check if Streamlit app can be imported."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import app

        print("Streamlit app import successful")
        return True
    except Exception as e:
        print(f"Failed to import Streamlit app: {e}")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="TEMPL Pipeline UI Test Runner")
    parser.add_argument(
        "--install-deps", action="store_true", help="Install test dependencies"
    )
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--e2e", action="store_true", help="Run E2E tests only")
    parser.add_argument(
        "--performance", action="store_true", help="Run performance tests"
    )
    parser.add_argument(
        "--accessibility", action="store_true", help="Run accessibility tests"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )
    parser.add_argument(
        "--no-headless", action="store_true", help="Run browser tests with GUI"
    )
    parser.add_argument(
        "--check-app",
        action="store_true",
        help="Check if Streamlit app can be imported",
    )

    args = parser.parse_args()

    # Change to test directory
    test_dir = Path(__file__).parent
    os.chdir(test_dir)

    success = True

    if args.install_deps:
        success &= install_dependencies()

    if args.check_app:
        success &= check_streamlit_app()

    if args.unit:
        success &= run_unit_tests(verbose=args.verbose, coverage=args.coverage)
    elif args.e2e:
        success &= run_e2e_tests(verbose=args.verbose, headless=not args.no_headless)
    elif args.performance:
        success &= run_performance_tests(verbose=args.verbose)
    elif args.accessibility:
        success &= run_accessibility_tests(verbose=args.verbose)
    elif args.all:
        success &= run_all_tests(verbose=args.verbose, coverage=args.coverage)
    else:
        # Default: run unit tests
        success &= run_unit_tests(verbose=args.verbose, coverage=args.coverage)

    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
