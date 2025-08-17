# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""CLI execution helpers for testing."""

import os
import subprocess
import tempfile
import time
from typing import List, Tuple


class CLITestRunner:
    """Helper class for running CLI commands in tests."""

    def __init__(self):
        # Use the installed templ command if available, otherwise use module
        self.templ_cmd = ["templ"]

    def run_command(
        self, args: List[str], timeout: int = 30
    ) -> subprocess.CompletedProcess:
        """Run a CLI command with timeout."""
        cmd = self.templ_cmd + args
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def capture_output(self, args: List[str]) -> Tuple[str, str, int]:
        """Capture stdout, stderr, and return code."""
        result = self.run_command(args)
        return result.stdout, result.stderr, result.returncode

    def measure_execution_time(self, args: List[str]) -> float:
        """Measure command execution time."""
        start_time = time.time()
        self.run_command(args)
        return time.time() - start_time

    def check_exit_code(self, args: List[str], expected_code: int = 0) -> bool:
        """Check if command returns expected exit code."""
        result = self.run_command(args)
        return result.returncode == expected_code

    def create_temp_file(self, content: str, suffix: str = ".tmp") -> str:
        """Create temporary file with content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            f.write(content)
            return f.name

    def cleanup_temp_file(self, filepath: str):
        """Clean up temporary file."""
        try:
            os.unlink(filepath)
        except FileNotFoundError:
            pass
