# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
TEMPL CLI Progress Indicators

This module provides user-friendly progress visualization for TEMPL pipeline operations,
with context-aware output that adapts to user experience levels and verbosity settings.

Features:
- Adaptive progress visualization (progress bars vs. simple status updates)
- Context-aware messaging based on operation complexity
- Performance timing and estimation
- Hardware utilization indicators
- Beginner-friendly explanations for long-running operations
"""

import time
import threading
import sys
from typing import Optional, Dict, Any, Callable, List
from contextlib import contextmanager
from enum import Enum
import logging

try:
    from tqdm import tqdm

    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .ux_config import VerbosityLevel, ExperienceLevel, get_ux_config

logger = logging.getLogger(__name__)


def simple_progress_wrapper(description: str, func: Callable, *args, **kwargs):
    """Simple progress wrapper that shows a basic spinner or progress message."""
    ux_config = get_ux_config()
    verbosity = ux_config.get_verbosity_level()

    # Only show progress for normal verbosity or higher
    if verbosity == VerbosityLevel.MINIMAL:
        return func(*args, **kwargs)

    print(f"Starting: {description}")
    start_time = time.time()

    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"Completed: {description} ({elapsed:.1f}s)")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"Failed: {description} ({elapsed:.1f}s)")
        raise


class OperationType(Enum):
    """Types of operations for context-aware progress indication."""

    EMBEDDING_GENERATION = "embedding_generation"
    TEMPLATE_SEARCH = "template_search"
    POSE_GENERATION = "pose_generation"
    MCS_CALCULATION = "mcs_calculation"
    SCORING = "scoring"
    FILE_IO = "file_io"
    VALIDATION = "validation"
    BENCHMARK = "benchmark"


class ProgressStyle(Enum):
    """Progress indication styles."""

    PROGRESS_BAR = "progress_bar"  # Visual progress bar with percentage
    SPINNER = "spinner"  # Simple spinning indicator
    STATUS_UPDATES = "status_updates"  # Periodic status messages
    MINIMAL = "minimal"  # Minimal output
    SILENT = "silent"  # No progress indication


@contextmanager
def progress_context(
    description: str,
    operation_type: OperationType,
    total_items: Optional[int] = None,
    show_eta: bool = True,
    show_rate: bool = True,
):
    """Context manager for progress indication during operations.

    Args:
        description: Human-readable description of the operation
        operation_type: Type of operation for context-aware messaging
        total_items: Total number of items to process (for progress bar)
        show_eta: Whether to show estimated time of arrival
        show_rate: Whether to show processing rate
    """
    ux_config = get_ux_config()
    verbosity = ux_config.get_verbosity_level()
    experience_level = ux_config.get_effective_experience_level()
    show_progress_bars = ux_config.should_show_progress_bars()

    # Determine progress style based on user preferences and context
    style = _determine_progress_style(
        verbosity, experience_level, show_progress_bars, total_items
    )

    progress_tracker = ProgressTracker(
        description=description,
        operation_type=operation_type,
        style=style,
        total_items=total_items,
        show_eta=show_eta,
        show_rate=show_rate,
        experience_level=experience_level,
        verbosity=verbosity,
    )

    try:
        progress_tracker.start()
        yield progress_tracker
    finally:
        progress_tracker.finish()


def _determine_progress_style(
    verbosity: VerbosityLevel,
    experience_level: ExperienceLevel,
    show_progress_bars: bool,
    total_items: Optional[int],
) -> ProgressStyle:
    """Determine appropriate progress style based on context."""

    if verbosity == VerbosityLevel.MINIMAL:
        return ProgressStyle.MINIMAL
    elif verbosity == VerbosityLevel.DEBUG:
        return ProgressStyle.STATUS_UPDATES

    # For normal/detailed verbosity, use preference and context
    if not show_progress_bars or not TQDM_AVAILABLE:
        if experience_level == ExperienceLevel.BEGINNER:
            return ProgressStyle.STATUS_UPDATES
        else:
            return ProgressStyle.SPINNER

    # Use progress bar if we have total items
    if total_items is not None and total_items > 1:
        return ProgressStyle.PROGRESS_BAR
    else:
        return ProgressStyle.SPINNER


class ProgressTracker:
    """Main progress tracking class."""

    def __init__(
        self,
        description: str,
        operation_type: OperationType,
        style: ProgressStyle,
        total_items: Optional[int] = None,
        show_eta: bool = True,
        show_rate: bool = True,
        experience_level: ExperienceLevel = ExperienceLevel.INTERMEDIATE,
        verbosity: VerbosityLevel = VerbosityLevel.NORMAL,
    ):
        self.description = description
        self.operation_type = operation_type
        self.style = style
        self.total_items = total_items
        self.show_eta = show_eta
        self.show_rate = show_rate
        self.experience_level = experience_level
        self.verbosity = verbosity

        self.start_time = None
        self.current_item = 0
        self.is_running = False
        self.progress_bar = None
        self.spinner_thread = None
        self.last_update_time = 0

        # Operation-specific context and tips
        self.operation_context = self._get_operation_context()

    def _get_operation_context(self) -> Dict[str, Any]:
        """Get context-specific information for the operation."""
        contexts = {
            OperationType.EMBEDDING_GENERATION: {
                "beginner_tip": "Generating protein embeddings (this may take a few minutes for large proteins)",
                "time_estimate": "1-5 minutes",
                "what_it_does": "Converting protein structure to numerical representation",
            },
            OperationType.TEMPLATE_SEARCH: {
                "beginner_tip": "Searching for similar protein templates in database",
                "time_estimate": "10-60 seconds",
                "what_it_does": "Finding proteins with similar binding sites",
            },
            OperationType.POSE_GENERATION: {
                "beginner_tip": "Generating molecular poses (this is the most compute-intensive step)",
                "time_estimate": "2-20 minutes",
                "what_it_does": "Creating 3D conformations of your molecule",
            },
            OperationType.MCS_CALCULATION: {
                "beginner_tip": "Finding common molecular patterns between molecules",
                "time_estimate": "10-30 seconds",
                "what_it_does": "Identifying shared chemical features",
            },
            OperationType.SCORING: {
                "beginner_tip": "Evaluating and ranking generated poses",
                "time_estimate": "30-120 seconds",
                "what_it_does": "Assessing quality of molecular conformations",
            },
        }

        return contexts.get(
            self.operation_type,
            {
                "beginner_tip": f"Processing {self.description.lower()}",
                "time_estimate": "Variable",
                "what_it_does": "Performing computational analysis",
            },
        )

    def start(self):
        """Start progress tracking."""
        self.start_time = time.time()
        self.is_running = True

        if self.style == ProgressStyle.MINIMAL:
            return
        elif self.style == ProgressStyle.SILENT:
            return

        # Show initial message with context for beginners
        if self.experience_level == ExperienceLevel.BEGINNER:
            tip = self.operation_context.get("beginner_tip", "")
            estimate = self.operation_context.get("time_estimate", "")
            if tip:
                print(f"TIP: {tip}")
                if estimate and estimate != "Variable":
                    print(f"ESTIMATED TIME: {estimate}")
                print()

        if self.style == ProgressStyle.PROGRESS_BAR and TQDM_AVAILABLE:
            self.progress_bar = tqdm(
                total=self.total_items,
                desc=self.description,
                unit="items" if self.total_items else "it",
                disable=False,
                leave=True,
                ncols=80,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        elif self.style == ProgressStyle.SPINNER:
            self._start_spinner()
        elif self.style == ProgressStyle.STATUS_UPDATES:
            print(f"STARTING: {self.description}")

    def _start_spinner(self):
        """Start a spinner in a separate thread."""

        def spin():
            spinner_chars = "|/-\\"
            i = 0
            while self.is_running:
                if time.time() - self.last_update_time > 0.1:  # Update every 100ms
                    sys.stdout.write(
                        f"\r{spinner_chars[i % len(spinner_chars)]} {self.description}"
                    )
                    sys.stdout.flush()
                    i += 1
                time.sleep(0.1)
            sys.stdout.write(
                "\r" + " " * (len(self.description) + 5) + "\r"
            )  # Clear line
            sys.stdout.flush()

        self.spinner_thread = threading.Thread(target=spin, daemon=True)
        self.spinner_thread.start()

    def update(self, increment: int = 1, message: Optional[str] = None):
        """Update progress."""
        if not self.is_running:
            return

        self.current_item += increment
        current_time = time.time()

        if self.style == ProgressStyle.PROGRESS_BAR and self.progress_bar:
            if message:
                self.progress_bar.set_description(message)
            self.progress_bar.update(increment)

        elif self.style == ProgressStyle.STATUS_UPDATES:
            # Update every 5 seconds or on significant progress
            if current_time - self.last_update_time > 5.0 or (
                self.total_items
                and self.current_item % max(1, self.total_items // 10) == 0
            ):

                elapsed = current_time - self.start_time
                if self.total_items:
                    progress_pct = (self.current_item / self.total_items) * 100
                    eta = (
                        (elapsed / self.current_item)
                        * (self.total_items - self.current_item)
                        if self.current_item > 0
                        else 0
                    )
                    eta_str = f", ETA: {eta:.0f}s" if self.show_eta and eta > 0 else ""
                    print(
                        f"PROGRESS: {self.current_item}/{self.total_items} ({progress_pct:.1f}%) - {elapsed:.1f}s elapsed{eta_str}"
                    )
                else:
                    rate = self.current_item / elapsed if elapsed > 0 else 0
                    rate_str = f" ({rate:.1f} items/s)" if self.show_rate else ""
                    print(
                        f"PROCESSED: {self.current_item} items{rate_str} - {elapsed:.1f}s elapsed"
                    )

                if message:
                    print(f"   Current: {message}")

        self.last_update_time = current_time

    def set_description(self, description: str):
        """Update the operation description."""
        self.description = description
        if self.style == ProgressStyle.PROGRESS_BAR and self.progress_bar:
            self.progress_bar.set_description(description)

    def finish(self, success_message: Optional[str] = None):
        """Finish progress tracking."""
        if not self.is_running:
            return

        self.is_running = False
        elapsed = time.time() - self.start_time if self.start_time else 0

        if self.style == ProgressStyle.PROGRESS_BAR and self.progress_bar:
            self.progress_bar.close()

        if self.style == ProgressStyle.SPINNER and self.spinner_thread:
            self.spinner_thread.join(timeout=0.5)

        # Show completion message
        if self.style not in [ProgressStyle.MINIMAL, ProgressStyle.SILENT]:
            if success_message:
                print(f"SUCCESS: {success_message}")
            else:
                rate = (
                    self.current_item / elapsed
                    if elapsed > 0 and self.current_item > 0
                    else 0
                )
                rate_str = (
                    f" ({rate:.1f} items/s)" if self.show_rate and rate > 0 else ""
                )
                print(f"COMPLETED: {self.description} - {elapsed:.1f}s{rate_str}")

            # Show additional context for beginners
            if self.experience_level == ExperienceLevel.BEGINNER and elapsed > 30:
                what_happened = self.operation_context.get("what_it_does", "")
                if what_happened:
                    print(f"   SUMMARY: {what_happened}")


class SimpleProgressBar:
    """Fallback progress bar when tqdm is not available."""

    def __init__(self, total: int, description: str = "", width: int = 50):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()

    def update(self, n: int = 1):
        """Update progress."""
        self.current = min(self.current + n, self.total)
        self._display()

    def _display(self):
        """Display current progress."""
        if self.total == 0:
            return

        progress = self.current / self.total
        filled = int(self.width * progress)
        bar = "█" * filled + "░" * (self.width - filled)

        elapsed = time.time() - self.start_time
        eta = (
            (elapsed / self.current) * (self.total - self.current)
            if self.current > 0
            else 0
        )

        print(
            f"\r{self.description} |{bar}| {self.current}/{self.total} ({progress*100:.1f}%) "
            f"[{elapsed:.1f}s<{eta:.1f}s]",
            end="",
            flush=True,
        )

    def close(self):
        """Finish progress bar."""
        print()  # New line


def show_hardware_status():
    """Show current hardware utilization status."""
    ux_config = get_ux_config()

    if not ux_config.should_show_performance_hints():
        return

    try:
        from templ_pipeline.core.hardware_utils import get_hardware_status

        status = get_hardware_status()

        print("HARDWARE STATUS:")
        print(f"   CPU Usage: {status.get('cpu_percent', 'Unknown')}%")
        print(f"   Memory Usage: {status.get('memory_percent', 'Unknown')}%")

        if "gpu_info" in status:
            for i, gpu in enumerate(status["gpu_info"]):
                print(
                    f"   GPU {i}: {gpu.get('utilization', 'Unknown')}% util, "
                    f"{gpu.get('memory_percent', 'Unknown')}% memory"
                )

    except ImportError:
        logger.debug("Hardware monitoring not available")


def estimate_operation_time(
    operation_type: OperationType, context: Dict[str, Any]
) -> Optional[str]:
    """Estimate operation time based on context."""
    estimates = {
        OperationType.EMBEDDING_GENERATION: lambda ctx: "1-5 minutes",
        OperationType.TEMPLATE_SEARCH: lambda ctx: "10-60 seconds",
        OperationType.POSE_GENERATION: lambda ctx: f"2-{max(20, ctx.get('num_conformers', 100) // 10)} minutes",
        OperationType.MCS_CALCULATION: lambda ctx: "10-30 seconds",
        OperationType.SCORING: lambda ctx: "30-120 seconds",
    }

    estimator = estimates.get(operation_type)
    if estimator:
        try:
            return estimator(context)
        except Exception:
            return None

    return None


# Export main classes and functions
__all__ = [
    "progress_context",
    "ProgressTracker",
    "OperationType",
    "ProgressStyle",
    "show_hardware_status",
    "estimate_operation_time",
    "simple_progress_wrapper",
]
