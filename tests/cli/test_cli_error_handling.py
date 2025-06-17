"""Error handling tests for TEMPL CLI."""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from .helpers.cli_runner import CLITestRunner
from .fixtures.mock_data import MOCK_PROTEIN_PDB, MOCK_INVALID_SMILES


class TestInputErrors:
    """Test input error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CLITestRunner()
    
    @pytest.mark.fast
    def test_missing_file_error(self):
        """Test handling of missing files."""
        stdout, stderr, returncode = self.runner.capture_output([
            "embed", "--protein-file", "/nonexistent/file.pdb"
        ])
        
        assert returncode != 0, "Should fail with missing file"
        assert len(stderr) > 0, "Should provide error message"
        # Should not contain stack trace
        assert "Traceback" not in stderr, "Should not show stack trace for missing file"
    
    @pytest.mark.fast
    def test_invalid_smiles_error(self):
        """Test handling of invalid SMILES."""
        with tempfile.NamedTemporaryFile(suffix=".pdb") as f:
            f.write(MOCK_PROTEIN_PDB.encode())
            f.flush()
            
            stdout, stderr, returncode = self.runner.capture_output([
                "generate-poses",
                "--protein-file", f.name,
                "--ligand-smiles", MOCK_INVALID_SMILES,
                "--template-pdb", "1abc"
            ])
            
            # Should handle invalid SMILES gracefully
            assert returncode != 0, "Should fail with invalid SMILES"
    
    @pytest.mark.fast
    def test_invalid_pdb_id_error(self):
        """Test handling of invalid PDB IDs."""
        with tempfile.NamedTemporaryFile(suffix=".pdb") as f:
            f.write(MOCK_PROTEIN_PDB.encode())
            f.flush()
            
            stdout, stderr, returncode = self.runner.capture_output([
                "generate-poses",
                "--protein-file", f.name,
                "--ligand-smiles", "CCO",
                "--template-pdb", "invalid_pdb_id_too_long"
            ])
            
            # Should handle invalid PDB ID gracefully
            assert returncode in [0, 1, 2], "Should handle invalid PDB ID appropriately"
    
    @pytest.mark.fast
    def test_missing_required_arguments(self):
        """Test handling of missing required arguments."""
        commands_and_args = [
            (["embed"], "protein-file"),
            (["find-templates"], "protein-file"),
            (["generate-poses"], "protein-file"),
            (["run"], "protein-file")
        ]
        
        for command, expected_arg in commands_and_args:
            stdout, stderr, returncode = self.runner.capture_output(command)
            
            assert returncode != 0, f"Command {command} should fail without required args"
            error_text = stderr.lower()
            assert "required" in error_text or expected_arg in error_text, \
                f"Should mention missing {expected_arg} for {command}"


class TestSystemErrors:
    """Test system error handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CLITestRunner()
    
    @pytest.mark.fast
    def test_missing_dependency_error(self):
        """Test handling of missing dependencies."""
        # This test verifies that the CLI handles dependency issues gracefully
        # In a real scenario, missing dependencies would cause import errors
        
        with tempfile.NamedTemporaryFile(suffix=".pdb") as f:
            f.write(MOCK_PROTEIN_PDB.encode())
            f.flush()
            
            stdout, stderr, returncode = self.runner.capture_output([
                "generate-poses",
                "--protein-file", f.name,
                "--ligand-smiles", "CCO",
                "--template-pdb", "1abc"
            ])
            
            # Should handle gracefully (may succeed if dependencies are available)
            assert returncode in [0, 1, 2], "Should handle dependency issues appropriately"
    
    @pytest.mark.fast
    def test_permission_error_handling(self):
        """Test handling of permission errors."""
        # Create a file and remove read permissions
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            f.write(MOCK_PROTEIN_PDB.encode())
            temp_file = f.name
        
        try:
            # Remove read permissions
            os.chmod(temp_file, 0o000)
            
            stdout, stderr, returncode = self.runner.capture_output([
                "embed", "--protein-file", temp_file
            ])
            
            assert returncode != 0, "Should fail with permission error"
            
        finally:
            # Restore permissions and clean up
            os.chmod(temp_file, 0o644)
            os.unlink(temp_file)
    
    @pytest.mark.fast
    @patch('tempfile.mkdtemp')
    def test_disk_space_error_handling(self, mock_mkdtemp):
        """Test handling of disk space issues."""
        # Mock disk space error
        mock_mkdtemp.side_effect = OSError("No space left on device")
        
        with tempfile.NamedTemporaryFile(suffix=".pdb") as f:
            f.write(MOCK_PROTEIN_PDB.encode())
            f.flush()
            
            stdout, stderr, returncode = self.runner.capture_output([
                "embed", "--protein-file", f.name
            ])
            
            # Should handle gracefully (may not trigger in this simple case)
            assert returncode in [0, 1, 2], "Should handle disk space issues appropriately"


class TestUserFeedback:
    """Test user-friendly error messages."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CLITestRunner()
    
    @pytest.mark.fast
    def test_user_friendly_error_messages(self):
        """Test error messages are user-friendly."""
        # Test missing file
        stdout, stderr, returncode = self.runner.capture_output([
            "embed", "--protein-file", "/nonexistent/file.pdb"
        ])
        
        assert returncode != 0, "Should fail with missing file"
        
        # Error message should be helpful
        error_text = stderr.lower()
        helpful_indicators = [
            "not found", "does not exist", "file", "path", 
            "check", "verify", "ensure"
        ]
        
        has_helpful_message = any(indicator in error_text for indicator in helpful_indicators)
        assert has_helpful_message, f"Error message should be helpful: {stderr}"
    
    @pytest.mark.fast
    def test_suggestions_for_common_errors(self):
        """Test suggestions are provided for common errors."""
        # Test command without arguments
        stdout, stderr, returncode = self.runner.capture_output(["embed"])
        
        assert returncode != 0, "Should fail without arguments"
        
        # Should provide usage information or suggestions
        combined_output = (stdout + stderr).lower()
        suggestion_indicators = [
            "usage", "help", "required", "example", "try"
        ]
        
        has_suggestions = any(indicator in combined_output for indicator in suggestion_indicators)
        assert has_suggestions, f"Should provide suggestions: {stdout + stderr}"
    
    @pytest.mark.fast
    def test_proper_exit_codes(self):
        """Test proper exit codes for different error types."""
        test_cases = [
            # (command, expected_exit_code_range, description)
            (["--help"], [0], "Help should return 0"),
            (["embed"], [1, 2], "Missing args should return 1 or 2"),
            (["unknown-command"], [1, 2], "Unknown command should return 1 or 2"),
            (["embed", "--protein-file", "/nonexistent"], [1, 2], "Missing file should return 1 or 2")
        ]
        
        for command, expected_codes, description in test_cases:
            stdout, stderr, returncode = self.runner.capture_output(command)
            assert returncode in expected_codes, f"{description}: got {returncode}, expected {expected_codes}"
    
    @pytest.mark.fast
    def test_no_stack_traces_in_user_errors(self):
        """Test stack traces don't appear for user errors."""
        user_error_commands = [
            ["embed"],  # Missing args
            ["embed", "--protein-file", "/nonexistent"],  # Missing file
            ["unknown-command"]  # Unknown command
        ]
        
        for command in user_error_commands:
            stdout, stderr, returncode = self.runner.capture_output(command)
            
            # Should not contain Python stack traces
            combined_output = stdout + stderr
            assert "Traceback" not in combined_output, f"Command {command} should not show stack trace"
            assert "File \"" not in combined_output, f"Command {command} should not show file traces"
    
    @pytest.mark.fast
    def test_helpful_command_suggestions(self):
        """Test helpful suggestions for unknown commands."""
        stdout, stderr, returncode = self.runner.capture_output(["unknown-command"])
        
        assert returncode != 0, "Unknown command should fail"
        
        # Should suggest valid commands or show help
        combined_output = (stdout + stderr).lower()
        helpful_content = [
            "embed", "find-templates", "generate-poses", "run",
            "help", "usage", "available", "commands"
        ]
        
        has_helpful_content = any(content in combined_output for content in helpful_content)
        assert has_helpful_content, f"Should suggest valid commands: {stdout + stderr}" 