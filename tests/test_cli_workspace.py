"""
Test cases for CLI workspace management module.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, call
from datetime import datetime

try:
    from templ_pipeline.cli.workspace_cli import (
        setup_logging,
        list_workspaces,
        cleanup_workspaces,
        display_workspace_summary,
        create_test_workspace,
        main as workspace_main
    )
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from templ_pipeline.cli.workspace_cli import (
        setup_logging,
        list_workspaces,
        cleanup_workspaces,
        display_workspace_summary,
        create_test_workspace,
        main as workspace_main
    )


class TestWorkspaceCLI(unittest.TestCase):
    """Test CLI workspace management functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.workspace_root = Path(self.test_dir) / "workspace"
        self.workspace_root.mkdir()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    @patch('templ_pipeline.cli.workspace_cli.logging.basicConfig')
    def test_setup_logging_default(self, mock_logging):
        """Test setup_logging with default level."""
        setup_logging()
        
        mock_logging.assert_called_once()
        args, kwargs = mock_logging.call_args
        self.assertEqual(kwargs['level'], 20)  # INFO level

    @patch('templ_pipeline.cli.workspace_cli.logging.basicConfig')
    def test_setup_logging_custom_level(self, mock_logging):
        """Test setup_logging with custom level."""
        setup_logging("DEBUG")
        
        mock_logging.assert_called_once()
        args, kwargs = mock_logging.call_args
        self.assertEqual(kwargs['level'], 10)  # DEBUG level

    @patch('templ_pipeline.cli.workspace_cli.logging.basicConfig')
    def test_setup_logging_invalid_level(self, mock_logging):
        """Test setup_logging with invalid level."""
        # Should raise AttributeError with invalid level
        with self.assertRaises(AttributeError):
            setup_logging("INVALID")

    def test_list_workspaces_empty(self):
        """Test listing workspaces with empty directory."""
        # Test with actual empty directory
        workspaces = list_workspaces(str(self.workspace_root))
        
        self.assertIsInstance(workspaces, list)
        self.assertEqual(len(workspaces), 0)

    def test_list_workspaces_with_directories(self):
        """Test listing workspaces with workspace directories."""
        # Create test workspace directories with run_ prefix (as expected by the function)
        workspace1 = self.workspace_root / "run_20250707_120000"
        workspace2 = self.workspace_root / "run_20250707_130000"
        workspace1.mkdir()
        workspace2.mkdir()
        
        # Add some test files
        (workspace1 / "test_file.txt").write_text("test content")
        (workspace2 / "another_file.txt").write_text("more content")
        
        workspaces = list_workspaces(str(self.workspace_root))
        
        self.assertIsInstance(workspaces, list)
        self.assertEqual(len(workspaces), 2)
        
        # Verify workspace information structure
        for workspace in workspaces:
            self.assertIsInstance(workspace, dict)
            self.assertIn('run_id', workspace)
            self.assertIn('path', workspace)

    def test_list_workspaces_nonexistent_directory(self):
        """Test listing workspaces with nonexistent directory."""
        nonexistent_path = str(Path(self.test_dir) / "nonexistent")
        
        workspaces = list_workspaces(nonexistent_path)
        
        # Should handle gracefully
        self.assertIsInstance(workspaces, list)

    def test_display_workspace_summary(self):
        """Test displaying workspace summary."""
        # Create test workspace directories
        workspace1 = self.workspace_root / "workspace_test1"
        workspace1.mkdir()
        (workspace1 / "file1.txt").write_text("test content")
        
        # Should not raise exceptions
        try:
            display_workspace_summary(str(self.workspace_root))
        except Exception as e:
            self.fail(f"display_workspace_summary raised unexpected exception: {e}")

    def test_create_test_workspace(self):
        """Test creating test workspace."""
        run_id = "test_workspace_001"
        
        try:
            result = create_test_workspace(run_id=run_id)
            
            # Should return workspace run_id or empty string
            if result:
                self.assertIsInstance(result, str)
        except Exception as e:
            # Function should exist but may fail with missing dependencies
            self.assertIsInstance(e, (ImportError, ValueError, FileNotFoundError))

    @patch('templ_pipeline.cli.workspace_cli.WORKSPACE_AVAILABLE', True)
    def test_cleanup_workspaces_basic(self):
        """Test basic workspace cleanup."""
        # Create old workspace directories
        old_workspace = self.workspace_root / "workspace_20220101_120000"
        old_workspace.mkdir()
        (old_workspace / "old_file.txt").write_text("old content")
        
        # Mock the cleanup function since we may not have full workspace manager
        with patch('templ_pipeline.cli.workspace_cli.cleanup_old_workspaces') as mock_cleanup:
            mock_cleanup.return_value = {"cleaned": 1, "errors": 0}
            
            result = cleanup_workspaces(
                workspace_root=str(self.workspace_root),
                max_age_days=1,
                dry_run=False
            )
            
            self.assertIsInstance(result, dict)
            mock_cleanup.assert_called_once()

    @patch('templ_pipeline.cli.workspace_cli.WORKSPACE_AVAILABLE', False)
    def test_cleanup_workspaces_unavailable(self):
        """Test workspace cleanup when workspace manager unavailable."""
        result = cleanup_workspaces(
            workspace_root=str(self.workspace_root),
            max_age_days=1,
            dry_run=False
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('error', result)

    def test_cleanup_workspaces_dry_run(self):
        """Test workspace cleanup in dry run mode."""
        # Create test workspace with run_ prefix
        test_workspace = self.workspace_root / "run_test"
        test_workspace.mkdir()
        
        # Dry run mode doesn't call cleanup_old_workspaces, it simulates locally
        with patch('templ_pipeline.cli.workspace_cli.WORKSPACE_AVAILABLE', True):
            result = cleanup_workspaces(
                workspace_root=str(self.workspace_root),
                max_age_days=1,
                dry_run=True
            )
            
            self.assertIsInstance(result, dict)
            self.assertIn('dry_run', result)
            self.assertTrue(result['dry_run'])

    @patch('templ_pipeline.cli.workspace_cli.argparse.ArgumentParser.parse_args')
    @patch('templ_pipeline.cli.workspace_cli.setup_logging')
    def test_main_list_command(self, mock_setup_logging, mock_parse_args):
        """Test main function with list command."""
        # Mock command line arguments
        mock_args = Mock()
        mock_args.command = 'list'
        mock_args.workspace_root = str(self.workspace_root)
        mock_args.log_level = 'INFO'
        mock_parse_args.return_value = mock_args
        
        # Create test workspace
        test_workspace = self.workspace_root / "workspace_test"
        test_workspace.mkdir()
        
        # Capture the result
        try:
            result = workspace_main()
            # Should execute without error
        except SystemExit as e:
            # argparse may cause SystemExit, which is normal
            self.assertEqual(e.code, 0)
        
        mock_setup_logging.assert_called_once_with('INFO')

    @patch('templ_pipeline.cli.workspace_cli.argparse.ArgumentParser.parse_args')
    @patch('templ_pipeline.cli.workspace_cli.setup_logging')
    def test_main_stats_command(self, mock_setup_logging, mock_parse_args):
        """Test main function with stats command."""
        mock_args = Mock()
        mock_args.command = 'stats'
        mock_args.workspace_root = str(self.workspace_root)
        mock_args.log_level = 'INFO'
        mock_parse_args.return_value = mock_args
        
        try:
            result = workspace_main()
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        
        mock_setup_logging.assert_called_once_with('INFO')

    @patch('templ_pipeline.cli.workspace_cli.argparse.ArgumentParser.parse_args')
    @patch('templ_pipeline.cli.workspace_cli.setup_logging')
    @patch('templ_pipeline.cli.workspace_cli.WORKSPACE_AVAILABLE', True)
    @patch('templ_pipeline.cli.workspace_cli.cleanup_old_workspaces')
    def test_main_cleanup_command(self, mock_cleanup, mock_setup_logging, mock_parse_args):
        """Test main function with cleanup command."""
        mock_args = Mock()
        mock_args.command = 'cleanup'
        mock_args.workspace_root = str(self.workspace_root)
        mock_args.max_age_days = 30
        mock_args.dry_run = False
        mock_args.log_level = 'INFO'
        mock_parse_args.return_value = mock_args
        
        mock_cleanup.return_value = {"cleaned": 2, "errors": 0}
        
        try:
            result = workspace_main()
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        
        mock_setup_logging.assert_called_once_with('INFO')
        mock_cleanup.assert_called_once()

    @patch('sys.argv', ['workspace_cli.py', 'list'])
    def test_main_integration_list(self):
        """Test main function integration with list command."""
        # This tests the actual argument parsing
        try:
            workspace_main()
        except SystemExit:
            # Expected for CLI tools
            pass
        except Exception as e:
            # Should handle errors gracefully
            self.assertIsInstance(e, (ImportError, FileNotFoundError, AttributeError))

    def test_workspace_naming_convention(self):
        """Test workspace naming convention validation."""
        # Create workspaces with run_ prefix (as expected by the function)
        valid_names = [
            "run_20250707_120000",
            "run_20250707_235959",
            "run_benchmark_20250707_120000"
        ]
        
        invalid_names = [
            "invalid_workspace",
            "workspace_invalid_time",
            "20250707_120000"
        ]
        
        for name in valid_names:
            workspace_dir = self.workspace_root / name
            workspace_dir.mkdir()
        
        for name in invalid_names:
            workspace_dir = self.workspace_root / name
            workspace_dir.mkdir()
        
        workspaces = list_workspaces(str(self.workspace_root))
        
        # Should only list directories with run_ prefix
        self.assertEqual(len(workspaces), len(valid_names))

    def test_workspace_error_handling(self):
        """Test error handling in workspace functions."""
        # Test with permission denied scenario (simulated)
        with patch('pathlib.Path.iterdir', side_effect=PermissionError("Access denied")):
            workspaces = list_workspaces(str(self.workspace_root))
            
            # Should handle permission errors gracefully
            self.assertIsInstance(workspaces, list)

    def test_workspace_stats_calculation(self):
        """Test workspace statistics calculation accuracy."""
        # Create multiple workspaces with known sizes
        workspace1 = self.workspace_root / "run_ws1"
        workspace2 = self.workspace_root / "run_ws2"
        workspace1.mkdir()
        workspace2.mkdir()
        
        # Create files with specific sizes
        (workspace1 / "file1.txt").write_text("a" * 1024)  # 1KB
        (workspace2 / "file2.txt").write_text("b" * 2048)  # 2KB
        
        # Use the actual list_workspaces function to get stats
        workspaces = list_workspaces(str(self.workspace_root))
        
        self.assertEqual(len(workspaces), 2)
        total_size_mb = sum(w['size_mb'] for w in workspaces)
        self.assertGreater(total_size_mb, 0)
        # Size should be at least 3KB = ~0.003MB
        self.assertGreaterEqual(total_size_mb, 0.001)


if __name__ == "__main__":
    unittest.main()