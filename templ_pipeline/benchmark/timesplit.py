from typing import Dict, List, Optional, Set

"""templ_pipeline.benchmark.timesplit

Thin compatibility wrapper.

This module keeps the historic public API (`run_timesplit_benchmark`) that many
call-sites import, while delegating the actual work to the modern streaming
implementation located in :pymod:`templ_pipeline.benchmark.timesplit_stream`.

It also provides a placeholder ``SHARED_MOLECULE_CACHE`` so that legacy utility
functions importing this symbol continue to operate (they will simply see an
empty cache).
"""

# Public re-exports -----------------------------------------------------------
from .timesplit_stream import (
    run_timesplit_benchmark,  # noqa: F401 re-export
    run_timesplit_streaming,  # noqa: F401 re-export
)

# ---------------------------------------------------------------------------
# Legacy globals expected by *templ_pipeline.core.utils* and other helpers.
# We keep them as no-op placeholders to avoid breaking imports without
# incurring the heavyweight global caches of the old implementation.
# ---------------------------------------------------------------------------

SHARED_MOLECULE_CACHE: Dict[str, Dict] = {}

__all__: List[str] = [
    "run_timesplit_benchmark",
    "run_timesplit_streaming",
    "SHARED_MOLECULE_CACHE",
]
