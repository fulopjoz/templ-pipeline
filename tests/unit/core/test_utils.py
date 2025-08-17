# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Test utilities for TEMPL pipeline tests.

This module provides common utilities and fixtures for testing,
including directory management, cleanup, and test data handling.
"""

import tempfile
import shutil
import os
from pathlib import Path
from typing import Optional, List
import contextlib
import logging

logger = logging.getLogger(__name__)


class TempDirectoryManager:
    """Context manager for temporary directory creation and cleanup."""

    def __init__(self, prefix: str = "templ_test_", cleanup_on_error: bool = True):
        """
        Initialize temporary directory manager.

        Args:
            prefix: Prefix for the temporary directory name
            cleanup_on_error: Whether to cleanup if an exception occurs
        """
        self.prefix = prefix
        self.cleanup_on_error = cleanup_on_error
        self.temp_dir: Optional[Path] = None

    def __enter__(self) -> Path:
        """Create and return temporary directory."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix=self.prefix))
        logger.debug(f"Created temporary directory: {self.temp_dir}")
        return self.temp_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary directory."""
        if self.temp_dir and self.temp_dir.exists():
            if exc_type is None or self.cleanup_on_error:
                try:
                    shutil.rmtree(self.temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {self.temp_dir}: {e}")
            else:
                logger.warning(f"Preserving directory for debugging: {self.temp_dir}")


@contextlib.contextmanager
def temp_directory(prefix: str = "templ_test_", cleanup_on_error: bool = True):
    """
    Context manager for creating temporary directories.

    Args:
        prefix: Prefix for the temporary directory name
        cleanup_on_error: Whether to cleanup if an exception occurs

    Yields:
        Path: Path to the temporary directory
    """
    with TempDirectoryManager(prefix, cleanup_on_error) as temp_dir:
        yield temp_dir


def cleanup_test_directories(base_path: Path = None, patterns: List[str] = None):
    """
    Clean up test directories matching specified patterns.

    Args:
        base_path: Base directory to search in (default: current directory)
        patterns: List of glob patterns to match (default: common test patterns)
    """
    if base_path is None:
        base_path = Path(".")

    if patterns is None:
        patterns = [
            "templ_test_*",
            "test_temp_*",
            "qa_test_*",
            "temp_*",
            "output_test_*",
        ]

    removed_count = 0
    for pattern in patterns:
        for dir_path in base_path.glob(pattern):
            if dir_path.is_dir():
                try:
                    shutil.rmtree(dir_path)
                    logger.debug(f"Removed test directory: {dir_path}")
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {dir_path}: {e}")

    if removed_count > 0:
        logger.info(f"Cleaned up {removed_count} test directories")

    return removed_count


def safe_cleanup_tempdir(temp_dir: str):
    """
    Safely clean up a temporary directory with error handling.

    Args:
        temp_dir: Path to the temporary directory to clean up
    """
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {e}")


class TestDirectoryMixin:
    """
    Mixin class for test cases that need temporary directory management.

    Usage:
        class MyTest(unittest.TestCase, TestDirectoryMixin):
            def setUp(self):
                super().setUp()
                self.setup_temp_directory()

            def tearDown(self):
                self.cleanup_temp_directory()
                super().tearDown()
    """

    def setup_temp_directory(self, prefix: str = "templ_test_"):
        """Set up temporary directory for test."""
        self.temp_dir = tempfile.mkdtemp(prefix=prefix)
        self.temp_path = Path(self.temp_dir)

    def cleanup_temp_directory(self):
        """Clean up temporary directory."""
        if hasattr(self, "temp_dir"):
            safe_cleanup_tempdir(self.temp_dir)

    def get_temp_file(self, filename: str) -> Path:
        """Get path to a temporary file."""
        if not hasattr(self, "temp_path"):
            raise RuntimeError(
                "Temporary directory not set up. Call setup_temp_directory() first."
            )
        return self.temp_path / filename

    def create_temp_file(self, filename: str, content: str = "") -> Path:
        """Create a temporary file with content."""
        file_path = self.get_temp_file(filename)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return file_path


def ensure_test_cleanup_on_exit():
    """
    Register cleanup function to run on test process exit.
    This helps clean up any directories that might be left behind.
    """
    import atexit

    def final_cleanup():
        """Final cleanup function."""
        try:
            cleanup_test_directories()
        except Exception as e:
            logger.warning(f"Error in final test cleanup: {e}")

    atexit.register(final_cleanup)


# Register cleanup on module import
ensure_test_cleanup_on_exit()
