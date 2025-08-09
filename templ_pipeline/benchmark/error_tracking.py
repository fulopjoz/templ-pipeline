

import json
import logging
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkErrorSummary:
    """A summary of all errors encountered during a benchmark."""
    total_errors: int
    error_breakdown: Dict[str, int]

    def to_dict(self):
        return asdict(self)

@dataclass
class MissingPDBRecord:
    """A record of a missing PDB file."""
    pdb_id: str
    reason: str
    details: str
    file_type: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        return asdict(self)

@dataclass
class ErrorRecord:
    """A record of a single error encountered during a benchmark."""
    pdb_id: str
    error_message: str
    context: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self):
        return asdict(self)

class BenchmarkErrorTracker:
    """A class to track and summarize errors during benchmark execution."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.errors: List[ErrorRecord] = []
        self.error_summary = defaultdict(int)
        self.error_file = self.workspace_dir / "benchmark_errors.jsonl"

    def record_target_failure(self, target_pdb: str, error_message: str, context: Optional[Dict[str, Any]] = None):
        """Records a failure for a specific target."""
        record = ErrorRecord(
            pdb_id=target_pdb,
            error_message=error_message,
            context=context
        )
        self.errors.append(record)
        self.error_summary[error_message] += 1
        self._write_record(record)

    def record_target_success(self, target_pdb: str):
        """Records a successful target processing (no-op for error log)."""
        # We keep success tracking out of the error jsonl to minimize file size.
        # Could be extended to maintain a separate success counter if needed.
        return

    def record_missing_pdb(self, pdb_id: str, reason: str, details: str, file_type: str):
        """Records a missing PDB file."""
        context = {
            "reason": reason,
            "details": details,
            "file_type": file_type
        }
        self.record_target_failure(pdb_id, "Missing PDB file", context)

    def _write_record(self, record: ErrorRecord):
        """Appends a single error record to the error file."""
        with open(self.error_file, 'a') as f:
            f.write(json.dumps(record.to_dict()) + '\n')

    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of all recorded errors."""
        return {
            "total_errors": len(self.errors),
            "error_breakdown": dict(self.error_summary)
        }

