# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""Performance measurement utilities for CLI tests."""

import time
import psutil
import sys
from typing import Dict, Any
from contextlib import contextmanager


class PerformanceMonitor:
    """Monitor performance metrics during test execution."""

    def __init__(self):
        self.process = psutil.Process()

    @contextmanager
    def measure_execution(self):
        """Context manager to measure execution time and memory."""
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        yield

        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        self.last_execution_time = end_time - start_time
        self.last_memory_usage = end_memory
        self.memory_delta = end_memory - start_memory

    def get_import_time(self, module_name: str) -> float:
        """Measure time to import a module."""
        start_time = time.time()
        try:
            __import__(module_name)
            return time.time() - start_time
        except ImportError:
            return float("inf")

    def check_module_loaded(self, module_name: str) -> bool:
        """Check if a module is already loaded."""
        return module_name in sys.modules

    def get_loaded_modules(self) -> set:
        """Get set of currently loaded modules."""
        return set(sys.modules.keys())


def assert_performance_criteria(
    execution_time: float,
    memory_mb: float,
    max_time: float = 0.5,
    max_memory: float = 300,
):
    """Assert performance criteria are met."""
    assert (
        execution_time < max_time
    ), f"Execution time {execution_time:.3f}s exceeds {max_time}s"
    assert (
        memory_mb < max_memory
    ), f"Memory usage {memory_mb:.1f}MB exceeds {max_memory}MB"
