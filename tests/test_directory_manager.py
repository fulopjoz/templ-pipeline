"""
Tests for directory manager module.
"""

import unittest
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

try:
    from templ_pipeline.core.directory_manager import (
        DirectoryManager,
        TempDirectoryManager,
        register_directory_cleanup,
        cleanup_test_artifacts,
        emergency_cleanup
    )
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from templ_pipeline.core.directory_manager import (
        DirectoryManager,
        TempDirectoryManager,
        register_directory_cleanup,
        cleanup_test_artifacts,
        emergency_cleanup
    )


class TestDirectoryManager(unittest.TestCase):
    """Test DirectoryManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_base_dir = Path(tempfile.mkdtemp(prefix="test_dir_mgr_"))
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_base_dir.exists():
            shutil.rmtree(self.test_base_dir)
    
    def test_lazy_creation_enabled(self):
        """Test that lazy creation doesn't create directory immediately."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            manager = DirectoryManager(
                base_name="test_output",
                run_id="test123",
                lazy_creation=True
            )
            
            # Directory should not be created yet
            mock_mkdir.assert_not_called()
            
            # Accessing the property should create it
            _ = manager.directory
            mock_mkdir.assert_called_once()
    
    def test_lazy_creation_disabled(self):
        """Test that disabling lazy creation creates directory immediately."""
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            manager = DirectoryManager(
                base_name="test_output",
                run_id="test123", 
                lazy_creation=False
            )
            
            # Directory should be created immediately
            mock_mkdir.assert_called_once()
    
    def test_directory_naming_with_run_id(self):
        """Test directory naming with run_id."""
        manager = DirectoryManager(
            base_name="test_output",
            run_id="custom123",
            lazy_creation=False
        )
        
        directory_name = manager.directory.name
        self.assertEqual(directory_name, "test_output_custom123")
    
    def test_directory_naming_with_timestamp(self):
        """Test directory naming with timestamp."""
        manager = DirectoryManager(
            base_name="test_output",
            run_id=None,
            lazy_creation=False
        )
        
        directory_name = manager.directory.name
        self.assertTrue(directory_name.startswith("test_output_"))
        self.assertTrue(len(directory_name) > len("test_output_"))
    
    def test_cleanup_removes_directory(self):
        """Test that cleanup removes the directory."""
        manager = DirectoryManager(
            base_name="test_cleanup",
            run_id="test",
            lazy_creation=False
        )
        
        directory = manager.directory
        self.assertTrue(directory.exists())
        
        success = manager.cleanup()
        self.assertTrue(success)
        self.assertFalse(directory.exists())
    
    def test_context_manager_with_auto_cleanup(self):
        """Test context manager with auto cleanup."""
        directory_path = None
        
        with DirectoryManager(
            base_name="test_context",
            run_id="test",
            auto_cleanup=True,
            lazy_creation=False
        ) as manager:
            directory_path = manager.directory
            self.assertTrue(directory_path.exists())
        
        # Directory should be cleaned up after context exit
        self.assertFalse(directory_path.exists())
    
    def test_exists_method(self):
        """Test exists method."""
        manager = DirectoryManager(
            base_name="test_exists",
            run_id="test",
            lazy_creation=True
        )
        
        # Should return False before creation
        self.assertFalse(manager.exists())
        
        # Should return True after accessing directory
        _ = manager.directory
        self.assertTrue(manager.exists())
        
        # Should return False after cleanup
        manager.cleanup()
        self.assertFalse(manager.exists())


class TestTempDirectoryManager(unittest.TestCase):
    """Test TempDirectoryManager class."""
    
    def test_temp_directory_creation(self):
        """Test temporary directory creation."""
        manager = TempDirectoryManager(prefix="test_temp_")
        
        directory = manager.directory
        self.assertTrue(directory.exists())
        self.assertTrue(str(directory).find("test_temp_") >= 0)
        
        # Clean up
        manager.cleanup()
        self.assertFalse(directory.exists())
    
    def test_temp_directory_context_manager(self):
        """Test temporary directory as context manager."""
        directory_path = None
        
        with TempDirectoryManager(prefix="test_context_temp_") as manager:
            directory_path = manager.directory
            self.assertTrue(directory_path.exists())
        
        # Should be cleaned up automatically (if auto_cleanup was enabled)
        # Note: TempDirectoryManager doesn't enable auto_cleanup by default
    
    def test_temp_directory_auto_cleanup(self):
        """Test temporary directory with auto cleanup."""
        directory_path = None
        
        with TempDirectoryManager(
            prefix="test_auto_temp_",
            auto_cleanup=True
        ) as manager:
            directory_path = manager.directory
            self.assertTrue(directory_path.exists())
        
        # Directory should be cleaned up after context exit
        self.assertFalse(directory_path.exists())


class TestDirectoryManagerGlobalFunctions(unittest.TestCase):
    """Test global directory management functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_base_dir = Path(tempfile.mkdtemp(prefix="test_global_"))
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_base_dir.exists():
            shutil.rmtree(self.test_base_dir)
    
    def test_register_directory_cleanup(self):
        """Test directory cleanup registration."""
        test_dir = self.test_base_dir / "test_register"
        test_dir.mkdir()
        
        # Register for cleanup
        register_directory_cleanup(test_dir)
        
        # Verify it's in the registered set
        self.assertIn(test_dir, DirectoryManager._registered_directories)
    
    def test_cleanup_old_directories(self):
        """Test cleanup of old directories."""
        # Create test directories
        old_dirs = [
            self.test_base_dir / "templ_test_old1",
            self.test_base_dir / "test_temp_old2",
            self.test_base_dir / "qa_test_old3"
        ]
        
        new_dirs = [
            self.test_base_dir / "templ_test_new1",
            self.test_base_dir / "normal_directory"
        ]
        
        all_dirs = old_dirs + new_dirs
        for dir_path in all_dirs:
            dir_path.mkdir()
        
        # Make some directories appear old by modifying their timestamps
        import os
        old_time = time.time() - (25 * 3600)  # 25 hours ago
        for dir_path in old_dirs:
            os.utime(dir_path, (old_time, old_time))
        
        # Run cleanup for directories older than 24 hours
        removed_count = DirectoryManager.cleanup_old_directories(
            patterns=['templ_test_*', 'test_temp_*', 'qa_test_*'],
            max_age_hours=24,
            base_path=self.test_base_dir
        )
        
        # Should have removed the old directories
        self.assertEqual(removed_count, 3)
        
        # Check which directories remain
        for dir_path in old_dirs:
            self.assertFalse(dir_path.exists())
        
        for dir_path in new_dirs:
            self.assertTrue(dir_path.exists())
    
    def test_cleanup_test_artifacts(self):
        """Test cleanup of test artifacts."""
        # Create test artifact directories
        test_dirs = [
            self.test_base_dir / "templ_test_artifact1",
            self.test_base_dir / "test_temp_artifact2",
            self.test_base_dir / "qa_test_artifact3"
        ]
        
        for dir_path in test_dirs:
            dir_path.mkdir()
        
        # Make them appear old
        import os
        old_time = time.time() - (2 * 3600)  # 2 hours ago
        for dir_path in test_dirs:
            os.utime(dir_path, (old_time, old_time))
        
        # Run cleanup
        removed_count = cleanup_test_artifacts(base_path=self.test_base_dir)
        
        # Should have removed all test artifacts
        self.assertEqual(removed_count, 3)
        
        for dir_path in test_dirs:
            self.assertFalse(dir_path.exists())
    
    @patch('templ_pipeline.core.directory_manager.logger')
    def test_emergency_cleanup(self, mock_logger):
        """Test emergency cleanup function."""
        # Create some test directories
        test_dirs = [
            self.test_base_dir / "emergency_test1",
            self.test_base_dir / "emergency_test2"
        ]
        
        for dir_path in test_dirs:
            dir_path.mkdir()
            register_directory_cleanup(dir_path)
        
        # Run emergency cleanup
        emergency_cleanup()
        
        # Should have logged the cleanup
        mock_logger.info.assert_called()
        
        # Directories should be cleaned up
        for dir_path in test_dirs:
            self.assertFalse(dir_path.exists())


class TestDirectoryManagerIntegration(unittest.TestCase):
    """Integration tests for directory manager."""
    
    def test_multiple_managers_independent(self):
        """Test that multiple managers work independently."""
        manager1 = DirectoryManager(
            base_name="test_multi1",
            run_id="run1",
            lazy_creation=False
        )
        
        manager2 = DirectoryManager(
            base_name="test_multi2", 
            run_id="run2",
            lazy_creation=False
        )
        
        dir1 = manager1.directory
        dir2 = manager2.directory
        
        # Both should exist and be different
        self.assertTrue(dir1.exists())
        self.assertTrue(dir2.exists())
        self.assertNotEqual(dir1, dir2)
        
        # Cleanup one shouldn't affect the other
        manager1.cleanup()
        self.assertFalse(dir1.exists())
        self.assertTrue(dir2.exists())
        
        # Clean up the second
        manager2.cleanup()
        self.assertFalse(dir2.exists())
    
    def test_directory_reuse_after_cleanup(self):
        """Test that directory can be recreated after cleanup."""
        manager = DirectoryManager(
            base_name="test_reuse",
            run_id="reuse",
            lazy_creation=True
        )
        
        # Create directory
        dir1 = manager.directory
        self.assertTrue(dir1.exists())
        
        # Clean up
        manager.cleanup()
        self.assertFalse(dir1.exists())
        
        # Access again should create new directory
        dir2 = manager.directory
        self.assertTrue(dir2.exists())
        
        # Clean up again
        manager.cleanup()


if __name__ == "__main__":
    unittest.main()