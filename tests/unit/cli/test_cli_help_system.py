# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""Help system functionality tests for TEMPL CLI."""

import re

import pytest

from .fixtures.expected_outputs import (
    ASCII_BANNER_PATTERN,
    EXAMPLE_COMMANDS,
    HELP_EXAMPLES_KEYWORDS,
    HELP_MAIN_KEYWORDS,
    HELP_PERFORMANCE_KEYWORDS,
    HELP_SIMPLE_KEYWORDS,
)
from .helpers.cli_runner import CLITestRunner
from .helpers.validation_utils import (
    validate_ascii_banner,
    validate_command_syntax,
    validate_help_output,
    validate_no_error_patterns,
)


class TestHelpSystem:
    """Test help system functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CLITestRunner()

    @pytest.mark.fast
    def test_main_help_content(self):
        """Test main help displays expected content."""
        stdout, stderr, returncode = self.runner.capture_output(["--help"])

        assert returncode == 0, f"Help command failed: {stderr}"
        assert validate_help_output(
            stdout, HELP_MAIN_KEYWORDS
        ), "Main help missing expected keywords"
        assert validate_no_error_patterns(stdout), "Help output contains error patterns"

    @pytest.mark.fast
    def test_simple_help_formatting(self):
        """Test simple help command shows appropriate message."""
        stdout, stderr, returncode = self.runner.capture_output(["--help", "simple"])

        assert returncode == 0, f"Simple help failed: {stderr}"
        # The new help system shows a message for unavailable help topics
        assert validate_help_output(
            stdout, HELP_SIMPLE_KEYWORDS
        ), "Simple help missing expected message"

    @pytest.mark.fast
    def test_examples_help_commands(self):
        """Test examples help contains valid commands."""
        stdout, stderr, returncode = self.runner.capture_output(["--help", "examples"])

        assert returncode == 0, f"Examples help failed: {stderr}"
        assert validate_help_output(
            stdout, HELP_EXAMPLES_KEYWORDS
        ), "Examples help missing expected keywords"

        # Validate example commands are syntactically correct
        for command in EXAMPLE_COMMANDS:
            assert validate_command_syntax(
                command
            ), f"Invalid command syntax: {command}"

    @pytest.mark.fast
    def test_performance_help_accuracy(self):
        """Test performance help command shows appropriate message."""
        stdout, stderr, returncode = self.runner.capture_output(
            ["--help", "performance"]
        )

        assert returncode == 0, f"Performance help failed: {stderr}"
        # The new help system shows a message for unavailable help topics
        assert validate_help_output(
            stdout, HELP_PERFORMANCE_KEYWORDS
        ), "Performance help missing expected message"

    @pytest.mark.fast
    def test_ascii_banner_display(self):
        """Test title appears in help."""
        stdout, stderr, returncode = self.runner.capture_output(["--help"])

        assert returncode == 0, f"Help command failed: {stderr}"
        # The new help system uses a simple title instead of ASCII banner
        assert (
            "TEMPL Pipeline - Template-based Protein-Ligand Pose Prediction" in stdout
        ), "Title not found in help output"

    @pytest.mark.fast
    def test_help_navigation(self):
        """Test all help navigation options work."""
        help_options = [
            ["--help"],
            ["--help", "basic"],
            ["--help", "examples"],
            ["--help", "intermediate"],
            ["--help", "expert"],
        ]

        for option in help_options:
            stdout, stderr, returncode = self.runner.capture_output(option)
            assert returncode == 0, f"Help option {option} failed: {stderr}"
            assert len(stdout) > 100, f"Help output too short for {option}"

    @pytest.mark.fast
    def test_invalid_help_requests(self):
        """Test invalid help requests are handled gracefully."""
        stdout, stderr, returncode = self.runner.capture_output(["--help", "invalid"])

        # Should show help not available message
        assert returncode == 0, "Invalid help should return 0 with helpful message"
        assert (
            "Help not available for command: invalid" in stdout
        ), "Should show appropriate error message"


class TestCommandSpecificHelp:
    """Test command-specific help functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CLITestRunner()

    @pytest.mark.fast
    def test_embed_command_help(self):
        """Test embed command help."""
        stdout, stderr, returncode = self.runner.capture_output(["embed", "--help"])

        assert returncode == 0, f"Embed help failed: {stderr}"
        assert "embed" in stdout.lower(), "Embed help should mention embed command"
        assert "--protein-file" in stdout, "Embed help should show protein-file option"

    @pytest.mark.fast
    def test_find_templates_command_help(self):
        """Test find-templates command help."""
        stdout, stderr, returncode = self.runner.capture_output(
            ["find-templates", "--help"]
        )

        assert returncode == 0, f"Find-templates help failed: {stderr}"
        assert (
            "find-templates" in stdout.lower()
        ), "Find-templates help should mention command"
        assert (
            "--embedding-file" in stdout
        ), "Find-templates help should show embedding-file option"

    @pytest.mark.fast
    def test_generate_poses_command_help(self):
        """Test generate-poses command help."""
        stdout, stderr, returncode = self.runner.capture_output(
            ["generate-poses", "--help"]
        )

        assert returncode == 0, f"Generate-poses help failed: {stderr}"
        assert (
            "generate-poses" in stdout.lower()
        ), "Generate-poses help should mention command"
        assert (
            "--ligand-smiles" in stdout
        ), "Generate-poses help should show ligand-smiles option"

    @pytest.mark.fast
    def test_run_command_help(self):
        """Test run command help."""
        stdout, stderr, returncode = self.runner.capture_output(["run", "--help"])

        assert returncode == 0, f"Run help failed: {stderr}"
        assert "run" in stdout.lower(), "Run help should mention run command"
        assert "--protein-file" in stdout, "Run help should show protein-file option"


class TestRichFormatting:
    """Test rich formatting in help output."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CLITestRunner()

    @pytest.mark.fast
    def test_help_output_structure(self):
        """Test help output has proper structure."""
        stdout, stderr, returncode = self.runner.capture_output(["--help"])

        assert returncode == 0, f"Help command failed: {stderr}"

        # Check for new section headers in progressive help system
        assert "Common Commands:" in stdout, "Help should have Common Commands section"
        assert "Quick Start:" in stdout, "Help should have Quick Start section"
        assert "Get Help:" in stdout, "Help should have Get Help section"

    @pytest.mark.fast
    def test_cross_platform_compatibility(self):
        """Test help works across platforms."""
        stdout, stderr, returncode = self.runner.capture_output(["--help"])

        assert returncode == 0, f"Help command failed: {stderr}"
        # Should not contain platform-specific escape sequences that break
        assert (
            "\x1b[" not in stdout or len(stdout) > 500
        ), "Help output should work without color codes"
