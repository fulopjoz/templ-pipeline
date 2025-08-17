#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Smart test execution script with automatic configuration selection.
Provides optimized testing configurations for different scenarios.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


class TestRunner:
    """Intelligent test runner with performance optimization."""

    def __init__(self):
        # Use repository root (two levels up from this script)
        self.project_root = Path(__file__).resolve().parent.parent
        self.configs = {
            "default": "pytest.ini",
            "performance": "pytest-performance.ini",  # Keep specialized performance config
        }
        self.presets = {
            "quick": ["-m", "fast", "--tb=line"],
            "medium": ["-m", "fast or medium", "--tb=short"],
            "full": ["--tb=short"],
            "critical": ["-m", "critical", "--tb=long"],
            "integration": ["-m", "integration", "--tb=long"],
            "ui": ["-m", "ui", "--tb=short", "--no-cov"],
            "performance": ["-m", "performance", "--tb=short"],
            "flaky": ["-m", "flaky", "--tb=long", "--repeat=3"],
            "ci": [
                "--tb=short",
                "--cov-report=xml",
                "--durations=20",
                "--timeout=600",
                "--strict-markers",
                "--strict-config",
            ],
            "parallel": [
                "-n",
                "auto",
                "--dist=load",
                "--max-worker-restart=2",
                "--durations=20",
            ],
        }

    def detect_environment(self) -> str:
        """Auto-detect the testing environment."""
        if os.getenv("PYTEST_PERFORMANCE"):
            return "performance"
        else:
            return "default"

    def detect_preset(self) -> Optional[str]:
        """Auto-detect the testing preset based on environment."""
        if os.getenv("CI"):
            return "ci"
        elif os.getenv("PYTEST_PARALLEL"):
            return "parallel"
        else:
            return None

    def get_cpu_count(self) -> int:
        """Get optimal number of test workers."""
        try:
            import multiprocessing

            cpu_count = multiprocessing.cpu_count()
            # Use 75% of available CPUs for testing
            return max(1, int(cpu_count * 0.75))
        except (ImportError, OSError):
            return 2

    def build_command(
        self,
        config: str,
        preset: Optional[str] = None,
        parallel: bool = False,
        args: List[str] = None,
    ) -> List[str]:
        """Build the pytest command with appropriate options."""
        cmd = ["python", "-m", "pytest"]

        # Add configuration file (stored under tools/)
        config_file = self.project_root / "tools" / self.configs[config]
        if config_file.exists():
            cmd.extend(["-c", str(config_file)])

        # Add preset options
        if preset and preset in self.presets:
            cmd.extend(self.presets[preset])

        # Add parallel execution (handled by preset now)
        if (
            parallel and preset != "parallel"
        ):  # Don't duplicate if preset already includes parallel
            cpu_count = self.get_cpu_count()
            cmd.extend(["-n", str(cpu_count)])

        # Add custom arguments
        if args:
            cmd.extend(args)

        return cmd

    def run_tests(
        self,
        config: str,
        preset: Optional[str] = None,
        parallel: bool = False,
        args: List[str] = None,
    ) -> int:
        """Execute tests with the specified configuration."""
        cmd = self.build_command(config, preset, parallel, args)

        print(f"Running tests with configuration: {config}")
        if preset:
            print(f"Using preset: {preset}")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 80)

        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            return result.returncode
        except KeyboardInterrupt:
            print("\nTest execution interrupted by user.")
            return 1
        except Exception as e:
            print(f"Error running tests: {e}")
            return 1


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Smart test execution with automatic configuration selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Auto-detect environment
  python run_tests.py --config ci        # Use CI configuration
  python run_tests.py --preset quick     # Run only fast tests
  python run_tests.py --parallel         # Enable parallel execution
  python run_tests.py --preset critical --parallel  # Critical tests in parallel
  python run_tests.py --config performance --preset performance  # Performance tests
        """,
    )

    runner = TestRunner()

    parser.add_argument(
        "--config",
        choices=runner.configs.keys(),
        help="Test configuration to use (auto-detected if not specified)",
    )
    parser.add_argument(
        "--preset", choices=runner.presets.keys(), help="Test preset to use"
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Enable parallel test execution"
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available configurations and presets",
    )

    args, pytest_args = parser.parse_known_args()

    if args.list_configs:
        print("Available configurations:")
        for name, file in runner.configs.items():
            print(f"  {name}: {file}")
        print("\nAvailable presets:")
        for name, options in runner.presets.items():
            print(f"  {name}: {' '.join(options)}")
        return 0

    # Auto-detect configuration and preset if not specified
    config = args.config or runner.detect_environment()
    preset = args.preset or runner.detect_preset()

    return runner.run_tests(
        config=config, preset=preset, parallel=args.parallel, args=pytest_args
    )


if __name__ == "__main__":
    sys.exit(main())
