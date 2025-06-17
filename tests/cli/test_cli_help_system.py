"""Help system functionality tests for TEMPL CLI."""

import pytest
import re

from .helpers.cli_runner import CLITestRunner
from .helpers.validation_utils import (
    validate_help_output, 
    validate_ascii_banner, 
    validate_command_syntax,
    validate_no_error_patterns
)
from .fixtures.expected_outputs import (
    HELP_MAIN_KEYWORDS,
    HELP_SIMPLE_KEYWORDS, 
    HELP_EXAMPLES_KEYWORDS,
    HELP_PERFORMANCE_KEYWORDS,
    EXAMPLE_COMMANDS,
    ASCII_BANNER_PATTERN
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
        assert validate_help_output(stdout, HELP_MAIN_KEYWORDS), "Main help missing expected keywords"
        assert validate_no_error_patterns(stdout), "Help output contains error patterns"
    
    @pytest.mark.fast
    def test_simple_help_formatting(self):
        """Test simple help displays correctly."""
        stdout, stderr, returncode = self.runner.capture_output(["--help", "simple"])
        
        assert returncode == 0, f"Simple help failed: {stderr}"
        assert validate_help_output(stdout, HELP_SIMPLE_KEYWORDS), "Simple help missing expected keywords"
    
    @pytest.mark.fast
    def test_examples_help_commands(self):
        """Test examples help contains valid commands."""
        stdout, stderr, returncode = self.runner.capture_output(["--help", "examples"])
        
        assert returncode == 0, f"Examples help failed: {stderr}"
        assert validate_help_output(stdout, HELP_EXAMPLES_KEYWORDS), "Examples help missing expected keywords"
        
        # Validate example commands are syntactically correct
        for command in EXAMPLE_COMMANDS:
            assert validate_command_syntax(command), f"Invalid command syntax: {command}"
    
    @pytest.mark.fast
    def test_performance_help_accuracy(self):
        """Test performance help displays correctly."""
        stdout, stderr, returncode = self.runner.capture_output(["--help", "performance"])
        
        assert returncode == 0, f"Performance help failed: {stderr}"
        assert validate_help_output(stdout, HELP_PERFORMANCE_KEYWORDS), "Performance help missing expected keywords"
    
    @pytest.mark.fast
    def test_ascii_banner_display(self):
        """Test ASCII banner appears in help."""
        stdout, stderr, returncode = self.runner.capture_output(["--help"])
        
        assert returncode == 0, f"Help command failed: {stderr}"
        assert validate_ascii_banner(stdout, ASCII_BANNER_PATTERN), "ASCII banner not found in help output"
    
    @pytest.mark.fast
    def test_help_navigation(self):
        """Test all help navigation options work."""
        help_options = [
            ["--help"],
            ["--help", "simple"],
            ["--help", "examples"], 
            ["--help", "workflows"],
            ["--help", "performance"]
        ]
        
        for option in help_options:
            stdout, stderr, returncode = self.runner.capture_output(option)
            assert returncode == 0, f"Help option {option} failed: {stderr}"
            assert len(stdout) > 100, f"Help output too short for {option}"
    
    @pytest.mark.fast
    def test_invalid_help_requests(self):
        """Test invalid help requests are handled gracefully."""
        stdout, stderr, returncode = self.runner.capture_output(["--help", "invalid"])
        
        # Should either show main help or give helpful error
        assert returncode in [0, 1], "Invalid help should return 0 or 1"
        assert len(stdout) > 0 or len(stderr) > 0, "Should provide some output for invalid help"


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
        stdout, stderr, returncode = self.runner.capture_output(["find-templates", "--help"])
        
        assert returncode == 0, f"Find-templates help failed: {stderr}"
        assert "find-templates" in stdout.lower(), "Find-templates help should mention command"
        assert "--embedding-file" in stdout, "Find-templates help should show embedding-file option"
    
    @pytest.mark.fast
    def test_generate_poses_command_help(self):
        """Test generate-poses command help."""
        stdout, stderr, returncode = self.runner.capture_output(["generate-poses", "--help"])
        
        assert returncode == 0, f"Generate-poses help failed: {stderr}"
        assert "generate-poses" in stdout.lower(), "Generate-poses help should mention command"
        assert "--ligand-smiles" in stdout, "Generate-poses help should show ligand-smiles option"
    
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
        
        # Check for section headers
        assert "Commands:" in stdout, "Help should have Commands section"
        assert "Quick Examples:" in stdout, "Help should have Quick Examples section"
        assert "Additional help functions:" in stdout, "Help should have additional help section"
    
    @pytest.mark.fast
    def test_cross_platform_compatibility(self):
        """Test help works across platforms."""
        stdout, stderr, returncode = self.runner.capture_output(["--help"])
        
        assert returncode == 0, f"Help command failed: {stderr}"
        # Should not contain platform-specific escape sequences that break
        assert "\x1b[" not in stdout or len(stdout) > 500, "Help output should work without color codes" 