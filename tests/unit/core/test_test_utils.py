"""
Tests for test utilities module.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from .test_utils import (
    TempDirectoryManager,
    temp_directory,
    cleanup_test_directories,
    safe_cleanup_tempdir,
    TestDirectoryMixin
)


class TestTempDirectoryManager(unittest.TestCase):
    """Test the TempDirectoryManager class."""
    
    def test_context_manager_creates_and_cleans_directory(self):
        """Test that context manager creates and cleans up directory."""
        temp_path = None
        
        with TempDirectoryManager(prefix="test_mgr_") as temp_dir:
            temp_path = temp_dir
            self.assertTrue(temp_path.exists())
            self.assertTrue(temp_path.is_dir())
            self.assertTrue(str(temp_path).find("test_mgr_") >= 0)
        
        # Directory should be cleaned up after context exit
        self.assertFalse(temp_path.exists())
    
    def test_context_manager_cleanup_on_error(self):
        """Test cleanup behavior when an exception occurs."""
        temp_path = None
        
        try:
            with TempDirectoryManager(prefix="test_error_", cleanup_on_error=True) as temp_dir:
                temp_path = temp_dir
                self.assertTrue(temp_path.exists())
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Directory should still be cleaned up
        self.assertFalse(temp_path.exists())
    
    def test_context_manager_preserve_on_error(self):
        """Test preserving directory when cleanup_on_error=False."""
        temp_path = None
        
        try:
            with TempDirectoryManager(prefix="test_preserve_", cleanup_on_error=False) as temp_dir:
                temp_path = temp_dir
                self.assertTrue(temp_path.exists())
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Directory should still exist for debugging
        self.assertTrue(temp_path.exists())
        
        # Clean up manually
        shutil.rmtree(temp_path)


class TestTempDirectoryFunction(unittest.TestCase):
    """Test the temp_directory context manager function."""
    
    def test_temp_directory_function(self):
        """Test temp_directory function works correctly."""
        temp_path = None
        
        with temp_directory(prefix="test_func_") as temp_dir:
            temp_path = temp_dir
            self.assertTrue(temp_path.exists())
            self.assertTrue(temp_path.is_dir())
        
        # Directory should be cleaned up
        self.assertFalse(temp_path.exists())


class TestCleanupUtilities(unittest.TestCase):
    """Test cleanup utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_dir = Path(tempfile.mkdtemp(prefix="cleanup_test_"))
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
    
    def test_cleanup_test_directories(self):
        """Test cleanup_test_directories function."""
        # Create test directories
        test_dirs = [
            self.base_dir / "templ_test_001",
            self.base_dir / "test_temp_002", 
            self.base_dir / "qa_test_003",
            self.base_dir / "should_not_match"
        ]
        
        for dir_path in test_dirs:
            dir_path.mkdir()
        
        # All directories should exist
        for dir_path in test_dirs:
            self.assertTrue(dir_path.exists())
        
        # Run cleanup
        removed_count = cleanup_test_directories(
            base_path=self.base_dir,
            patterns=['templ_test_*', 'test_temp_*', 'qa_test_*']
        )
        
        # Should have removed 3 directories
        self.assertEqual(removed_count, 3)
        
        # Check which directories remain
        self.assertFalse((self.base_dir / "templ_test_001").exists())
        self.assertFalse((self.base_dir / "test_temp_002").exists())
        self.assertFalse((self.base_dir / "qa_test_003").exists())
        self.assertTrue((self.base_dir / "should_not_match").exists())
    
    def test_safe_cleanup_tempdir(self):
        """Test safe_cleanup_tempdir function."""
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="safe_cleanup_test_")
        self.assertTrue(Path(temp_dir).exists())
        
        # Clean it up safely
        safe_cleanup_tempdir(temp_dir)
        self.assertFalse(Path(temp_dir).exists())
        
        # Should not error on non-existent directory
        safe_cleanup_tempdir(temp_dir)
        safe_cleanup_tempdir(None)
        safe_cleanup_tempdir("")


class TestDirectoryMixinTest(unittest.TestCase, TestDirectoryMixin):
    """Test the TestDirectoryMixin."""
    
    def setUp(self):
        """Set up test fixtures."""
        super().setUp()
        self.setup_temp_directory(prefix="mixin_test_")
    
    def tearDown(self):
        """Clean up test fixtures."""
        self.cleanup_temp_directory()
        super().tearDown()
    
    def test_mixin_provides_temp_directory(self):
        """Test that mixin provides working temporary directory."""
        self.assertTrue(hasattr(self, 'temp_dir'))
        self.assertTrue(hasattr(self, 'temp_path'))
        self.assertTrue(self.temp_path.exists())
        self.assertTrue(self.temp_path.is_dir())
    
    def test_mixin_temp_file_operations(self):
        """Test temporary file operations in mixin."""
        # Test getting temp file path
        file_path = self.get_temp_file("test.txt")
        self.assertTrue(str(file_path).endswith("test.txt"))
        
        # Test creating temp file
        created_file = self.create_temp_file("created.txt", "test content")
        self.assertTrue(created_file.exists())
        self.assertEqual(created_file.read_text(), "test content")
        
        # Test creating file in subdirectory
        subdir_file = self.create_temp_file("subdir/nested.txt", "nested content")
        self.assertTrue(subdir_file.exists())
        self.assertEqual(subdir_file.read_text(), "nested content")


if __name__ == "__main__":
    unittest.main()