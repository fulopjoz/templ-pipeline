# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Test fixtures for TEMPL pipeline tests.

This package provides centralized test data management and fixtures
for consistent testing across all modules.
"""

from .data_factory import TestDataFactory

# Import available fixtures (not all functions exist yet)
try:
    from .benchmark_fixtures import create_benchmark_test_data
except ImportError:
    create_benchmark_test_data = None

__all__ = ["TestDataFactory"]
