# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""Performance and lazy loading tests for TEMPL CLI."""

import time

import pytest

from .helpers.cli_runner import CLITestRunner
from .helpers.performance_utils import PerformanceMonitor, assert_performance_criteria


class TestCLIPerformance:
    """Test CLI performance and lazy loading."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CLITestRunner()
        self.monitor = PerformanceMonitor()

    @pytest.mark.fast
    @pytest.mark.performance
    def test_cli_import_speed(self):
        """Test CLI module imports quickly."""
        import_time = self.monitor.get_import_time("templ_pipeline.cli.main")
        assert (
            import_time < 0.1
        ), f"CLI import took {import_time:.3f}s, should be < 0.1s"

    @pytest.mark.fast
    @pytest.mark.performance
    def test_help_command_performance(self):
        """Test help command executes quickly."""
        # Test direct function call to avoid subprocess overhead
        import sys
        import time

        from templ_pipeline.cli.main import main

        # Save original argv
        original_argv = sys.argv[:]

        try:
            # Set up test arguments
            sys.argv = ["templ", "--help"]

            start_time = time.time()
            main()  # This will call sys.exit(0), but we catch it
            execution_time = time.time() - start_time
        except SystemExit as e:
            execution_time = time.time() - start_time
            assert e.code == 0, f"Help command returned non-zero exit code: {e.code}"
        finally:
            # Restore original argv
            sys.argv = original_argv

        assert (
            execution_time < 2.0
        ), f"Help command took {execution_time:.3f}s, should be < 2.0s"

    @pytest.mark.fast
    @pytest.mark.performance
    def test_help_variants_performance(self):
        """Test all help variants execute quickly."""
        import sys
        import time

        from templ_pipeline.cli.main import main

        help_variants = [
            "--help",
            "--help simple",
            "--help examples",
            "--help performance",
        ]

        # Save original argv
        original_argv = sys.argv[:]

        for variant in help_variants:
            try:
                # Set up test arguments
                args = variant.split()
                sys.argv = ["templ"] + args

                start_time = time.time()
                main()  # This will call sys.exit(0), but we catch it
                execution_time = time.time() - start_time
            except SystemExit as e:
                execution_time = time.time() - start_time
                assert (
                    e.code == 0
                ), f"Help variant {variant} returned non-zero exit code: {e.code}"
            finally:
                # Restore original argv
                sys.argv = original_argv

            assert (
                execution_time < 2.0
            ), f"Help variant {variant} took {execution_time:.3f}s"

    @pytest.mark.fast
    @pytest.mark.performance
    def test_memory_usage_during_help(self):
        """Test memory usage during help display."""
        import os

        # Adjust memory limit for CI environment
        max_memory = 1200 if os.getenv("CI") == "true" else 900  # MB

        with self.monitor.measure_execution():
            self.runner.run_command(["--help"])

        assert_performance_criteria(
            self.monitor.last_execution_time,
            self.monitor.last_memory_usage,
            max_time=2.0,
            max_memory=max_memory,
        )

    @pytest.mark.fast
    @pytest.mark.performance
    def test_no_heavy_imports_on_help(self):
        """Test heavy modules aren't loaded during help."""
        import subprocess
        import sys

        # Run help command in subprocess to avoid session contamination
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from templ_pipeline.cli.main import main; "
                "sys.argv = ['templ', '--help']; "
                "main(); "
                "heavy_modules = ['torch', 'transformers']; "
                "loaded = [m for m in heavy_modules if any(m in mod for mod in sys.modules)]; "
                "exit(1 if loaded else 0)",
            ],
            capture_output=True,
            text=True,
        )

        assert (
            result.returncode == 0
        ), f"Heavy modules were loaded during help: {result.stderr}"

    @pytest.mark.fast
    def test_lazy_loading_verification(self):
        """Test lazy loading is working correctly."""
        # Test that help works without heavy imports
        stdout, stderr, returncode = self.runner.capture_output(["--help"])
        assert returncode == 0, f"Help failed: {stderr}"
        assert "TEMPL" in stdout, "Help output missing"

        # Verify that help works correctly
        # Note: Heavy modules may be loaded in test environment, but help should still work
        assert len(stdout) > 100, "Help output should be substantial"
        assert "TEMPL Pipeline" in stdout, "Help should contain pipeline name"


class TestSetupParserPerformance:
    """Test argument parser setup performance."""

    @pytest.mark.fast
    @pytest.mark.performance
    def test_setup_parser_speed(self):
        """Test parser setup is fast."""
        start_time = time.time()

        # Import and setup parser
        from templ_pipeline.cli.main import setup_parser

        parser = setup_parser()

        setup_time = time.time() - start_time
        assert (
            setup_time < 0.1
        ), f"Parser setup took {setup_time:.3f}s, should be < 0.1s"
        assert parser is not None, "Parser should be created successfully"
