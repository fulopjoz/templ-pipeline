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
from .timesplit import run_timesplit_benchmark
from .timesplit_stream import run_timesplit_streaming

__all__ = [
    "run_templ_pipeline_for_benchmark",
    "BenchmarkRunner",
    "BenchmarkParams",
    "BenchmarkResult",
    "run_timesplit_benchmark",
    "run_timesplit_streaming",
]
