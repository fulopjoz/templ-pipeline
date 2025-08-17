# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
TEMPL Pipeline Benchmark Module

Internal benchmark functionality for TEMPL pipeline.
"""

from .runner import (
    run_templ_pipeline_for_benchmark,
    BenchmarkRunner,
    BenchmarkParams,
    BenchmarkResult,
)


# Lazy import for timesplit to avoid dependency contamination
def run_timesplit_benchmark(*args, **kwargs):
    """Lazy import wrapper for timesplit benchmark to avoid dependency issues."""
    from .timesplit import run_timesplit_benchmark as _run_timesplit

    return _run_timesplit(*args, **kwargs)


__all__ = [
    "run_templ_pipeline_for_benchmark",
    "BenchmarkRunner",
    "BenchmarkParams",
    "BenchmarkResult",
    "run_timesplit_benchmark",
]
