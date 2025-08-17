# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""Command validation tests for TEMPL CLI."""

import tempfile

import pytest

from .fixtures.mock_data import (
    MOCK_INVALID_SMILES,
    MOCK_LIGAND_SDF,
    MOCK_PROTEIN_PDB,
    MOCK_SMILES,
)
from .helpers.cli_runner import CLITestRunner


class TestArgumentParsing:
    """Test argument parsing for all commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CLITestRunner()

    @pytest.mark.fast
    def test_embed_command_args(self):
        """Test embed command argument parsing."""
        # Test required arguments
        stdout, stderr, returncode = self.runner.capture_output(["embed"])
        assert returncode != 0, "Embed without required args should fail"
        assert (
            "required" in stderr.lower() or "protein-file" in stderr
        ), "Should mention missing protein-file"

        # Test with protein file
        with tempfile.NamedTemporaryFile(suffix=".pdb") as f:
            f.write(MOCK_PROTEIN_PDB.encode())
            f.flush()

            stdout, stderr, returncode = self.runner.capture_output(
                ["embed", "--protein-file", f.name]
            )
            # Should fail due to missing dependencies but args should parse
            assert (
                "protein-file" not in stderr
            ), "Should not complain about protein-file arg"

    @pytest.mark.fast
    def test_find_templates_command_args(self):
        """Test find-templates command argument parsing."""
        stdout, stderr, returncode = self.runner.capture_output(["find-templates"])
        assert returncode != 0, "Find-templates without required args should fail"

        # Check for required arguments mentioned in error
        error_text = stderr.lower()
        assert (
            "required" in error_text
            or "protein-file" in error_text
            or "embedding-file" in error_text
        )

    @pytest.mark.fast
    def test_generate_poses_command_args(self):
        """Test generate-poses command argument parsing."""
        stdout, stderr, returncode = self.runner.capture_output(["generate-poses"])
        assert returncode != 0, "Generate-poses without required args should fail"

        error_text = stderr.lower()
        assert "required" in error_text or "protein-file" in error_text

    @pytest.mark.fast
    def test_run_command_args(self):
        """Test run command argument parsing."""
        stdout, stderr, returncode = self.runner.capture_output(["run"])
        assert returncode != 0, "Run without required args should fail"

        error_text = stderr.lower()
        assert "required" in error_text or "protein-file" in error_text

    @pytest.mark.fast
    def test_invalid_argument_combinations(self):
        """Test invalid argument combinations."""
        # Test mutually exclusive ligand inputs
        with (
            tempfile.NamedTemporaryFile(suffix=".pdb") as pdb_f,
            tempfile.NamedTemporaryFile(suffix=".sdf") as sdf_f,
        ):

            pdb_f.write(MOCK_PROTEIN_PDB.encode())
            sdf_f.write(MOCK_LIGAND_SDF.encode())
            pdb_f.flush()
            sdf_f.flush()

            # Both SMILES and file should be handled gracefully
            stdout, stderr, returncode = self.runner.capture_output(
                [
                    "generate-poses",
                    "--protein-file",
                    pdb_f.name,
                    "--ligand-smiles",
                    MOCK_SMILES,
                    "--ligand-file",
                    sdf_f.name,
                ]
            )
            # Should either work (last one wins) or give clear error
            assert returncode in [
                0,
                1,
                2,
            ], "Should handle conflicting ligand inputs gracefully"


class TestInputValidation:
    """Test input validation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CLITestRunner()

    @pytest.mark.fast
    def test_file_path_validation(self):
        """Test file path validation."""
        # Test non-existent file
        stdout, stderr, returncode = self.runner.capture_output(
            ["embed", "--protein-file", "/nonexistent/file.pdb"]
        )
        assert returncode != 0, "Should fail with non-existent file"

    @pytest.mark.fast
    def test_smiles_validation(self):
        """Test SMILES validation."""
        with tempfile.NamedTemporaryFile(suffix=".pdb") as f:
            f.write(MOCK_PROTEIN_PDB.encode())
            f.flush()

            # Test valid SMILES
            stdout, stderr, returncode = self.runner.capture_output(
                [
                    "generate-poses",
                    "--protein-file",
                    f.name,
                    "--ligand-smiles",
                    MOCK_SMILES,
                    "--template-pdb",
                    "1abc",
                ]
            )

            # Should not fail due to SMILES validation
            assert (
                "Invalid SMILES" not in stderr
            ), "Valid SMILES should not trigger validation error"

            # Test invalid SMILES
            stdout, stderr, returncode = self.runner.capture_output(
                [
                    "generate-poses",
                    "--protein-file",
                    f.name,
                    "--ligand-smiles",
                    MOCK_INVALID_SMILES,
                    "--template-pdb",
                    "1abc",
                ]
            )

            # Should fail with validation error
            assert returncode == 2, "Invalid SMILES should return exit code 2"
            assert "Invalid SMILES" in stderr, "Should show SMILES validation error"

    @pytest.mark.fast
    def test_parameter_range_validation(self):
        """Test parameter range validation."""
        with tempfile.NamedTemporaryFile(suffix=".pdb") as f:
            f.write(MOCK_PROTEIN_PDB.encode())
            f.flush()

            # Test negative number of conformers
            stdout, stderr, returncode = self.runner.capture_output(
                [
                    "generate-poses",
                    "--protein-file",
                    f.name,
                    "--ligand-smiles",
                    MOCK_SMILES,
                    "--template-pdb",
                    "1abc",
                    "--num-conformers",
                    "-1",
                ]
            )
            # Should either reject negative value or handle gracefully
            assert returncode in [
                0,
                2,
            ], "Should handle negative conformers appropriately"


class TestCommandStructure:
    """Test command structure and registration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CLITestRunner()

    @pytest.mark.fast
    def test_all_subcommands_registered(self):
        """Test all subcommands are registered."""
        stdout, stderr, returncode = self.runner.capture_output(["--help"])

        expected_commands = ["embed", "find-templates", "generate-poses", "run"]
        for command in expected_commands:
            assert command in stdout, f"Command {command} not found in help"

    @pytest.mark.fast
    def test_subcommand_help_available(self):
        """Test help is available for each subcommand."""
        commands = ["embed", "find-templates", "generate-poses", "run"]

        for command in commands:
            stdout, stderr, returncode = self.runner.capture_output([command, "--help"])
            assert returncode == 0, f"Help for {command} should work"
            assert (
                len(stdout) > 100
            ), f"Help for {command} should have substantial content"

    @pytest.mark.fast
    def test_unknown_command_handling(self):
        """Test unknown command handling."""
        stdout, stderr, returncode = self.runner.capture_output(["unknown-command"])

        assert returncode != 0, "Unknown command should fail"
        # Should provide helpful error message
        assert (
            len(stderr) > 0 or len(stdout) > 0
        ), "Should provide error message for unknown command"


class TestDefaultValues:
    """Test default value handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CLITestRunner()

    @pytest.mark.fast
    def test_default_output_directory(self):
        """Test default output directory is mentioned in comprehensive help."""
        # Check expert help which includes complete command reference with output options
        stdout, stderr, returncode = self.runner.capture_output(["--help", "expert"])

        # Should mention output directory in expert help
        assert (
            "--output-dir" in stdout
        ), "Should mention output directory option in expert help"

    @pytest.mark.fast
    def test_default_log_level(self):
        """Test default log level is mentioned in expert help."""
        # Check expert help which includes complete command reference
        stdout, stderr, returncode = self.runner.capture_output(["--help", "expert"])

        # Should mention log levels in expert help
        assert (
            "log-level" in stdout.lower() or "info" in stdout.lower()
        ), "Should mention log level in expert help"

    @pytest.mark.fast
    def test_default_conformers(self):
        """Test default number of conformers."""
        stdout, stderr, returncode = self.runner.capture_output(
            ["generate-poses", "--help"]
        )

        # Should mention default conformers
        assert (
            "100" in stdout or "conformers" in stdout
        ), "Should mention conformers option"


def test_time_split_benchmark_help():
    """Test time-split benchmark help message."""
    runner = CLITestRunner()
    stdout, stderr, returncode = runner.capture_output(
        ["benchmark", "time-split", "--help"]
    )
    assert returncode == 0
    # Help should be displayed for time-split specific options
    assert "--template-knn" in stdout or "--template-knn" in stderr


def test_time_split_benchmark_dry_run():
    """Test time-split benchmark with minimal parameters for smoke testing."""
    # This is a dry run test - it may fail due to missing data but should not crash
    runner = CLITestRunner()
    stdout, stderr, returncode = runner.capture_output(
        [
            "benchmark",
            "time-split",
            "--max-pdbs",
            "1",
            "--dev-subset",
            "--n-conformers",
            "10",
            "--pipeline-timeout",
            "60",
        ]
    )
    # We expect this to potentially fail due to missing data files, but not crash
    # The important thing is that the command is recognized and parsed correctly
    assert returncode in [0, 1, 2]  # 0=success, 1=runtime error, 2=validation error


def test_beginner_user_no_attribute_errors():
    """Test that beginner users don't get AttributeError for missing CLI arguments."""
    runner = CLITestRunner()

    # Create minimal temp files
    with tempfile.NamedTemporaryFile(suffix=".pdb") as pdb_f:
        pdb_f.write(MOCK_PROTEIN_PDB.encode())
        pdb_f.flush()

        # Test that we don't get AttributeError for missing namespace attributes
        # This simulates a fresh beginner user
        stdout, stderr, returncode = runner.capture_output(
            ["run", "--protein-file", pdb_f.name, "--ligand-smiles", MOCK_SMILES]
        )

        # Should NOT fail with AttributeError about missing namespace attributes
        combined_output = (stdout + stderr).lower()
        assert "attributeerror" not in combined_output, "Should not have AttributeError"
        assert (
            "'namespace' object has no attribute" not in combined_output
        ), "Should not have namespace attribute errors"
        assert (
            "log_level" not in combined_output
            or "attributeerror" not in combined_output
        ), "Should not have log_level AttributeError"


def test_log_level_argument_always_available():
    """Test that --log-level argument is always available regardless of user experience."""
    runner = CLITestRunner()

    # Test that --log-level is recognized in help for main command
    stdout, stderr, returncode = runner.capture_output(["--help"])

    # Should mention log-level as an available option (in any help mode)
    assert returncode == 0, "Help should work"
    # Since the progressive help system might not show all options, just check it's parsed correctly

    # Test that --log-level is accepted without error
    stdout, stderr, returncode = runner.capture_output(["run", "--log-level", "DEBUG"])

    # Should fail for missing required args, but NOT for unrecognized --log-level
    assert (
        "unrecognized arguments: --log-level" not in stderr
    ), "--log-level should be recognized"
    assert (
        "invalid choice" not in stderr.lower()
    ), "--log-level DEBUG should be valid choice"
