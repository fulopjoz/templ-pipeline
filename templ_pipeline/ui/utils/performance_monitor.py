# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Performance Monitor for TEMPL Pipeline

Tracks and reports performance metrics for UI operations.
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Performance monitoring for UI operations"""

    def __init__(self):
        """Initialize performance monitor"""
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.active_timers: Dict[str, float] = {}
        self.operation_counts: Dict[str, int] = defaultdict(int)

    def start_render(self, component_name: str) -> None:
        """Start timing a render operation

        Args:
            component_name: Name of the component being rendered
        """
        self.active_timers[f"render_{component_name}"] = time.time()

    def end_render(self, component_name: str) -> float:
        """End timing a render operation

        Args:
            component_name: Name of the component being rendered

        Returns:
            Time taken in seconds
        """
        key = f"render_{component_name}"
        if key not in self.active_timers:
            logger.warning(f"No active timer for {key}")
            return 0.0

        start_time = self.active_timers.pop(key)
        duration = time.time() - start_time

        self.timings[key].append(duration)
        self.operation_counts[key] += 1

        return duration

    def start_operation(self, operation_name: str) -> None:
        """Start timing a general operation

        Args:
            operation_name: Name of the operation
        """
        self.active_timers[operation_name] = time.time()

    def end_operation(self, operation_name: str) -> float:
        """End timing a general operation

        Args:
            operation_name: Name of the operation

        Returns:
            Time taken in seconds
        """
        if operation_name not in self.active_timers:
            logger.warning(f"No active timer for {operation_name}")
            return 0.0

        start_time = self.active_timers.pop(operation_name)
        duration = time.time() - start_time

        self.timings[operation_name].append(duration)
        self.operation_counts[operation_name] += 1

        return duration

    def get_average_time(self, operation_name: str) -> float:
        """Get average time for an operation

        Args:
            operation_name: Name of the operation

        Returns:
            Average time in seconds
        """
        if operation_name not in self.timings:
            return 0.0

        timings = self.timings[operation_name]
        return sum(timings) / len(timings) if timings else 0.0

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive performance statistics

        Returns:
            Dictionary of statistics per operation
        """
        stats = {}

        for operation, timings in self.timings.items():
            if not timings:
                continue

            stats[operation] = {
                "count": self.operation_counts[operation],
                "total": sum(timings),
                "average": sum(timings) / len(timings),
                "min": min(timings),
                "max": max(timings),
                "last": timings[-1],
            }

        return stats

    def get_report(self) -> str:
        """Generate human-readable performance report

        Returns:
            Performance report string
        """
        stats = self.get_statistics()

        if not stats:
            return "No performance data collected yet"

        report_lines = ["=== Performance Report ==="]

        # Sort by total time
        sorted_ops = sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True)

        for operation, op_stats in sorted_ops:
            report_lines.append(
                f"\n{operation}:"
                f"\n  Count: {op_stats['count']}"
                f"\n  Total: {op_stats['total']:.3f}s"
                f"\n  Average: {op_stats['average']:.3f}s"
                f"\n  Min/Max: {op_stats['min']:.3f}s / {op_stats['max']:.3f}s"
            )

        return "\n".join(report_lines)

    def reset(self) -> None:
        """Reset all performance data"""
        self.timings.clear()
        self.active_timers.clear()
        self.operation_counts.clear()
        logger.info("Performance monitor reset")

    def log_slow_operations(self, threshold: float = 1.0) -> None:
        """Log operations that exceeded a time threshold

        Args:
            threshold: Time threshold in seconds
        """
        for operation, timings in self.timings.items():
            slow_ops = [t for t in timings if t > threshold]
            if slow_ops:
                logger.warning(
                    f"Slow operations detected for {operation}: "
                    f"{len(slow_ops)} exceeded {threshold}s threshold"
                )
