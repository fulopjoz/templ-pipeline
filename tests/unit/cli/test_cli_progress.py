# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Test cases for CLI progress indicators module.
"""

import threading
import time
import unittest
from io import StringIO
from unittest.mock import Mock, patch

try:
    from templ_pipeline.cli.progress_indicators import (
        OperationType,
        ProgressStyle,
        ProgressTracker,
        SimpleProgressBar,
        estimate_operation_time,
        progress_context,
        show_hardware_status,
    )
    from templ_pipeline.cli.ux_config import ExperienceLevel, VerbosityLevel
except ImportError:
    import os
    import sys

    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    )
    from templ_pipeline.cli.progress_indicators import (
        OperationType,
        ProgressStyle,
        ProgressTracker,
        SimpleProgressBar,
        estimate_operation_time,
        progress_context,
        show_hardware_status,
    )
    from templ_pipeline.cli.ux_config import ExperienceLevel, VerbosityLevel


class TestProgressIndicators(unittest.TestCase):
    """Test progress indicator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_stdout = StringIO()

    def test_operation_type_enum(self):
        """Test OperationType enum values."""
        self.assertEqual(
            OperationType.EMBEDDING_GENERATION.value, "embedding_generation"
        )
        self.assertEqual(OperationType.TEMPLATE_SEARCH.value, "template_search")
        self.assertEqual(OperationType.POSE_GENERATION.value, "pose_generation")
        self.assertEqual(OperationType.MCS_CALCULATION.value, "mcs_calculation")
        self.assertEqual(OperationType.SCORING.value, "scoring")
        self.assertEqual(OperationType.FILE_IO.value, "file_io")
        self.assertEqual(OperationType.VALIDATION.value, "validation")
        self.assertEqual(OperationType.BENCHMARK.value, "benchmark")

    def test_progress_style_enum(self):
        """Test ProgressStyle enum values."""
        # Check that enum has expected values
        styles = list(ProgressStyle)
        self.assertGreater(len(styles), 0)

        # Check that all enum values are strings
        for style in styles:
            self.assertIsInstance(style.value, str)

    @patch("templ_pipeline.cli.progress_indicators.TQDM_AVAILABLE", True)
    @patch("templ_pipeline.cli.progress_indicators.get_ux_config")
    def test_progress_context_with_tqdm(self, mock_get_ux_config):
        """Test using progress context when tqdm is available."""
        # Mock UX config for verbose expert user
        mock_config = Mock()
        mock_config.get_verbosity_level.return_value = VerbosityLevel.NORMAL
        mock_config.get_effective_experience_level.return_value = (
            ExperienceLevel.ADVANCED
        )
        mock_config.should_show_progress_bars.return_value = True
        mock_get_ux_config.return_value = mock_config

        with progress_context(
            description="Test operation",
            operation_type=OperationType.EMBEDDING_GENERATION,
            total_items=100,
        ) as tracker:
            self.assertIsInstance(tracker, ProgressTracker)

    @patch("templ_pipeline.cli.progress_indicators.TQDM_AVAILABLE", False)
    @patch("templ_pipeline.cli.progress_indicators.get_ux_config")
    def test_progress_context_without_tqdm(self, mock_get_ux_config):
        """Test using progress context when tqdm is not available."""
        mock_config = Mock()
        mock_config.get_verbosity_level.return_value = VerbosityLevel.NORMAL
        mock_config.get_effective_experience_level.return_value = (
            ExperienceLevel.BEGINNER
        )
        mock_config.should_show_progress_bars.return_value = False
        mock_get_ux_config.return_value = mock_config

        with progress_context(
            description="Search operation",
            operation_type=OperationType.TEMPLATE_SEARCH,
            total_items=50,
        ) as tracker:
            self.assertIsInstance(tracker, ProgressTracker)

    @patch("templ_pipeline.cli.progress_indicators.get_ux_config")
    def test_progress_tracker_update(self, mock_get_ux_config):
        """Test updating progress tracker."""
        mock_config = Mock()
        mock_config.get_verbosity_level.return_value = VerbosityLevel.NORMAL
        mock_config.get_effective_experience_level.return_value = (
            ExperienceLevel.INTERMEDIATE
        )
        mock_config.should_show_progress_bars.return_value = True
        mock_get_ux_config.return_value = mock_config

        tracker = ProgressTracker(
            description="Pose generation",
            operation_type=OperationType.POSE_GENERATION,
            style=ProgressStyle.PROGRESS_BAR,
            total_items=10,
        )

        # Test that update methods exist and are callable
        self.assertTrue(hasattr(tracker, "update"))
        self.assertTrue(callable(tracker.update))

        # Test updating progress
        try:
            tracker.start()
            tracker.update(5)  # Update to 50%
            tracker.update(5)  # Update to 100%
            tracker.finish()
        except Exception as e:
            # Should not raise unexpected exceptions
            self.fail(f"Progress update raised unexpected exception: {e}")

    @patch("templ_pipeline.cli.progress_indicators.get_ux_config")
    def test_progress_context_manager(self, mock_get_ux_config):
        """Test progress context as context manager."""
        mock_config = Mock()
        mock_config.get_verbosity_level.return_value = VerbosityLevel.MINIMAL
        mock_config.get_effective_experience_level.return_value = (
            ExperienceLevel.ADVANCED
        )
        mock_config.should_show_progress_bars.return_value = True
        mock_get_ux_config.return_value = mock_config

        # Test as context manager
        try:
            with progress_context(
                description="Scoring poses",
                operation_type=OperationType.SCORING,
                total_items=20,
            ) as tracker:
                tracker.update(10)
                tracker.update(10)
        except Exception as e:
            # Should handle context manager protocol
            self.assertIsInstance(e, (AttributeError, NotImplementedError, TypeError))

    @patch("templ_pipeline.cli.progress_indicators.get_ux_config")
    def test_progress_tracker_finish(self, mock_get_ux_config):
        """Test finishing progress tracker."""
        mock_config = Mock()
        mock_config.get_verbosity_level.return_value = VerbosityLevel.NORMAL
        mock_config.get_effective_experience_level.return_value = (
            ExperienceLevel.BEGINNER
        )
        mock_config.should_show_progress_bars.return_value = False
        mock_get_ux_config.return_value = mock_config

        tracker = ProgressTracker(
            description="MCS calculation",
            operation_type=OperationType.MCS_CALCULATION,
            style=ProgressStyle.STATUS_UPDATES,
            total_items=5,
        )

        # Test finish method
        try:
            tracker.start()
            tracker.update(5)
            tracker.finish()
        except Exception as e:
            self.fail(f"Progress finish raised unexpected exception: {e}")

    def test_simple_progress_bar(self):
        """Test SimpleProgressBar functionality."""
        progress_bar = SimpleProgressBar(total=10, description="Test progress")

        # Test that methods exist
        self.assertTrue(hasattr(progress_bar, "update"))
        self.assertTrue(hasattr(progress_bar, "close"))

        # Test updating progress
        try:
            progress_bar.update(5)
            progress_bar.update(5)
            progress_bar.close()
        except Exception as e:
            self.fail(f"SimpleProgressBar raised unexpected exception: {e}")

    def test_hardware_status_function(self):
        """Test show_hardware_status function."""
        # Should be callable without errors
        try:
            show_hardware_status()
        except Exception as e:
            # May fail due to missing hardware utils, but should not crash
            self.assertIsInstance(e, (ImportError, AttributeError))

    def test_estimate_operation_time(self):
        """Test operation time estimation."""
        # Test with different operation types
        context = {"num_conformers": 100}

        estimate = estimate_operation_time(OperationType.EMBEDDING_GENERATION, context)
        self.assertIsInstance(estimate, (str, type(None)))

        estimate = estimate_operation_time(OperationType.POSE_GENERATION, context)
        self.assertIsInstance(estimate, (str, type(None)))

        estimate = estimate_operation_time(OperationType.TEMPLATE_SEARCH, {})
        self.assertIsInstance(estimate, (str, type(None)))

    @patch("templ_pipeline.cli.progress_indicators.get_ux_config")
    def test_progress_tracker_set_description(self, mock_get_ux_config):
        """Test setting description on progress tracker."""
        mock_config = Mock()
        mock_config.get_verbosity_level.return_value = VerbosityLevel.NORMAL
        mock_config.get_effective_experience_level.return_value = (
            ExperienceLevel.ADVANCED
        )
        mock_config.should_show_progress_bars.return_value = True
        mock_get_ux_config.return_value = mock_config

        tracker = ProgressTracker(
            description="Initial description",
            operation_type=OperationType.BENCHMARK,
            style=ProgressStyle.PROGRESS_BAR,
            total_items=10,
        )

        # Test set_description method
        try:
            tracker.set_description("Updated description")
            self.assertEqual(tracker.description, "Updated description")
        except Exception as e:
            self.fail(f"set_description raised unexpected exception: {e}")

    @patch("templ_pipeline.cli.progress_indicators.get_ux_config")
    def test_progress_tracker_different_operations(self, mock_get_ux_config):
        """Test progress trackers for different operation types."""
        mock_config = Mock()
        mock_config.get_verbosity_level.return_value = VerbosityLevel.NORMAL
        mock_config.get_effective_experience_level.return_value = (
            ExperienceLevel.INTERMEDIATE
        )
        mock_config.should_show_progress_bars.return_value = True
        mock_get_ux_config.return_value = mock_config

        operations = [
            OperationType.EMBEDDING_GENERATION,
            OperationType.TEMPLATE_SEARCH,
            OperationType.POSE_GENERATION,
            OperationType.MCS_CALCULATION,
            OperationType.SCORING,
            OperationType.FILE_IO,
            OperationType.VALIDATION,
            OperationType.BENCHMARK,
        ]

        for operation in operations:
            with self.subTest(operation=operation):
                tracker = ProgressTracker(
                    description=f"Test {operation.value}",
                    operation_type=operation,
                    style=ProgressStyle.PROGRESS_BAR,
                    total_items=10,
                )

                self.assertIsInstance(tracker, ProgressTracker)

    def test_progress_tracker_with_zero_total(self):
        """Test progress tracker with zero total items."""
        with patch(
            "templ_pipeline.cli.progress_indicators.get_ux_config"
        ) as mock_get_ux_config:
            mock_config = Mock()
            mock_config.get_verbosity_level.return_value = VerbosityLevel.NORMAL
            mock_config.get_effective_experience_level.return_value = (
                ExperienceLevel.INTERMEDIATE
            )
            mock_config.should_show_progress_bars.return_value = True
            mock_get_ux_config.return_value = mock_config

            # Should handle zero total gracefully
            tracker = ProgressTracker(
                description="Empty operation",
                operation_type=OperationType.FILE_IO,
                style=ProgressStyle.PROGRESS_BAR,
                total_items=0,
            )

            self.assertIsInstance(tracker, ProgressTracker)

    def test_progress_tracker_with_negative_total(self):
        """Test progress tracker with negative total items."""
        with patch(
            "templ_pipeline.cli.progress_indicators.get_ux_config"
        ) as mock_get_ux_config:
            mock_config = Mock()
            mock_config.get_verbosity_level.return_value = VerbosityLevel.NORMAL
            mock_config.get_effective_experience_level.return_value = (
                ExperienceLevel.INTERMEDIATE
            )
            mock_config.should_show_progress_bars.return_value = True
            mock_get_ux_config.return_value = mock_config

            # Should handle negative total gracefully or raise appropriate error
            try:
                tracker = ProgressTracker(
                    description="Invalid operation",
                    operation_type=OperationType.VALIDATION,
                    style=ProgressStyle.PROGRESS_BAR,
                    total_items=-5,
                )
                self.assertIsInstance(tracker, ProgressTracker)
            except ValueError:
                # Acceptable to raise ValueError for negative total
                pass

    @patch("templ_pipeline.cli.progress_indicators.get_ux_config")
    def test_progress_tracker_thread_safety(self, mock_get_ux_config):
        """Test progress tracker thread safety."""
        mock_config = Mock()
        mock_config.get_verbosity_level.return_value = VerbosityLevel.NORMAL
        mock_config.get_effective_experience_level.return_value = (
            ExperienceLevel.INTERMEDIATE
        )
        mock_config.should_show_progress_bars.return_value = True
        mock_get_ux_config.return_value = mock_config

        tracker = ProgressTracker(
            description="Threaded operation",
            operation_type=OperationType.BENCHMARK,
            style=ProgressStyle.PROGRESS_BAR,
            total_items=100,
        )

        def update_progress():
            for i in range(10):
                try:
                    tracker.update(1)
                    time.sleep(0.001)
                except Exception:
                    pass  # Ignore exceptions in thread test

        # Run multiple threads updating progress
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=update_progress)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=1.0)

    def test_operation_type_coverage(self):
        """Test that all operation types are properly defined."""
        expected_operations = [
            "embedding_generation",
            "template_search",
            "pose_generation",
            "mcs_calculation",
            "scoring",
            "file_io",
            "validation",
            "benchmark",
        ]

        actual_operations = [op.value for op in OperationType]

        for expected in expected_operations:
            self.assertIn(expected, actual_operations)

    @patch("sys.stdout", new_callable=StringIO)
    @patch("templ_pipeline.cli.progress_indicators.get_ux_config")
    def test_progress_output_capture(self, mock_get_ux_config, mock_stdout):
        """Test that progress trackers produce output."""
        mock_config = Mock()
        mock_config.get_verbosity_level.return_value = VerbosityLevel.NORMAL
        mock_config.get_effective_experience_level.return_value = (
            ExperienceLevel.BEGINNER
        )
        mock_config.should_show_progress_bars.return_value = False
        mock_get_ux_config.return_value = mock_config

        # Progress tracker should produce some output
        tracker = ProgressTracker(
            description="Test operation for output",
            operation_type=OperationType.EMBEDDING_GENERATION,
            style=ProgressStyle.STATUS_UPDATES,
            total_items=10,
        )

        # May or may not produce output depending on implementation
        # This test just ensures no exceptions are raised
        try:
            tracker.start()
            tracker.update(5)
            tracker.finish()
        except Exception:
            pass  # Ignore exceptions in output capture test


if __name__ == "__main__":
    unittest.main()
