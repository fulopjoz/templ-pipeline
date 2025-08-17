# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Time-split benchmark module for TEMPL pipeline.

This module provides time-split benchmarking functionality with proper data hygiene,
ensuring that test sets only use templates from earlier time periods to prevent
data leakage.

Main components:
- SimpleTimeSplitRunner: Streamlined benchmark execution with 2A/5A success rate generation
- benchmark.py: CLI entry point and workspace organization
- Integration with unified benchmark infrastructure and detailed summary generation
"""

from .benchmark import run_timesplit_benchmark
from .simple_runner import SimpleTimeSplitRunner

__all__ = [
    "run_timesplit_benchmark", 
    "SimpleTimeSplitRunner",
]