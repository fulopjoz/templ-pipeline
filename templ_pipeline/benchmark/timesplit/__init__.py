"""
Time-split benchmark module for TEMPL pipeline.

This module provides time-split benchmarking functionality with proper data hygiene,
ensuring that test sets only use templates from earlier time periods to prevent
data leakage.

Main components:
- TimeSplitBenchmarkRunner: Core benchmark execution
- benchmark.py: CLI entry point and workspace organization
- Integration with unified benchmark infrastructure
"""

from .benchmark import run_timesplit_benchmark
from .timesplit_runner import TimeSplitBenchmarkRunner

__all__ = [
    "run_timesplit_benchmark",
    "TimeSplitBenchmarkRunner",
]