"""
Directory lifecycle management for TEMPL pipeline.

This module provides utilities for managing temporary and output directories
with proper lifecycle management, cleanup, and error handling.
"""

import os
import shutil
import tempfile
import atexit
from pathlib import Path
from typing import Optional, Set, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DirectoryManager:
    """
    Manager for pipeline directories with lifecycle management.
    
    Features:
    - Lazy directory creation
    - Automatic cleanup registration
    - Error handling and recovery
    - Timeout-based cleanup for abandoned directories
    """
    
    _registered_directories: Set[Path] = set()
    _cleanup_registered = False
    
    def __init__(
        self, 
        base_name: str = "output",
        run_id: Optional[str] = None,
        auto_cleanup: bool = False,
        lazy_creation: bool = True,
        centralized_output: bool = True,
        output_root: Optional[str] = None
    ):
        """
        Initialize directory manager.
        
        Args:
            base_name: Base name for the directory
            run_id: Custom run identifier (default: timestamp)
            auto_cleanup: Whether to automatically clean up on exit
            lazy_creation: Whether to defer directory creation until needed
            centralized_output: Whether to use centralized output directory structure
            output_root: Root directory for centralized output (default: "output")
        """
        self.base_name = base_name
        self.run_id = run_id
        self.auto_cleanup = auto_cleanup
        self.lazy_creation = lazy_creation
        self.centralized_output = centralized_output
        self.output_root = Path(output_root or "output")
        self._directory: Optional[Path] = None
        self._created = False
        
        if not lazy_creation:
            self._create_directory()
    
    @property
    def directory(self) -> Path:
        """Get the managed directory, creating it if necessary."""
        if self._directory is None or not self._created:
            self._create_directory()
        return self._directory
    
    def _create_directory(self) -> None:
        """Create the timestamped directory."""
        if self._created and self._directory and self._directory.exists():
            return
            
        if self.run_id:
            dir_name = f"{self.base_name}_{self.run_id}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = f"{self.base_name}_{timestamp}"
        
        if self.centralized_output:
            # Create directory within centralized output structure
            self.output_root.mkdir(exist_ok=True)
            self._directory = self.output_root / dir_name
        else:
            # Use original behavior for backward compatibility
            self._directory = Path(dir_name)
            
        self._directory.mkdir(exist_ok=True)
        self._created = True
        
        # Register for cleanup
        if self.auto_cleanup:
            self._register_for_cleanup()
        
        logger.debug(f"Created directory: {self._directory}")
    
    def _register_for_cleanup(self) -> None:
        """Register directory for cleanup on exit."""
        DirectoryManager._registered_directories.add(self._directory)
        
        if not DirectoryManager._cleanup_registered:
            atexit.register(DirectoryManager._cleanup_all_registered)
            DirectoryManager._cleanup_registered = True
    
    def cleanup(self) -> bool:
        """
        Clean up the managed directory.
        
        Returns:
            bool: True if cleanup was successful
        """
        if not self._directory or not self._directory.exists():
            return True
            
        try:
            shutil.rmtree(self._directory)
            logger.debug(f"Cleaned up directory: {self._directory}")
            
            # Remove from registered directories
            DirectoryManager._registered_directories.discard(self._directory)
            
            self._created = False
            self._directory = None
            return True
            
        except Exception as e:
            logger.warning(f"Failed to cleanup directory {self._directory}: {e}")
            return False
    
    def exists(self) -> bool:
        """Check if the directory exists."""
        return self._directory is not None and self._directory.exists()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.auto_cleanup:
            self.cleanup()
    
    @classmethod
    def _cleanup_all_registered(cls) -> None:
        """Clean up all registered directories."""
        logger.debug(f"Cleaning up {len(cls._registered_directories)} registered directories")
        
        for directory in list(cls._registered_directories):
            try:
                if directory.exists():
                    shutil.rmtree(directory)
                    logger.debug(f"Cleaned up registered directory: {directory}")
            except Exception as e:
                logger.warning(f"Failed to cleanup registered directory {directory}: {e}")
        
        cls._registered_directories.clear()
    
    @classmethod
    def cleanup_old_directories(
        cls, 
        patterns: Optional[list] = None, 
        max_age_hours: int = 24,
        base_path: Path = None
    ) -> int:
        """
        Clean up old directories matching patterns.
        
        Args:
            patterns: List of glob patterns to match
            max_age_hours: Maximum age in hours before cleanup
            base_path: Base directory to search in
            
        Returns:
            int: Number of directories cleaned up
        """
        if base_path is None:
            base_path = Path('.')
            
        if patterns is None:
            patterns = ['output_*', 'qa_test_*', 'temp_*', 'debug_*']
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        removed_count = 0
        
        for pattern in patterns:
            for dir_path in base_path.glob(pattern):
                if not dir_path.is_dir():
                    continue
                    
                try:
                    # Check modification time
                    mtime = datetime.fromtimestamp(dir_path.stat().st_mtime)
                    
                    if mtime < cutoff_time:
                        # Check if directory is empty or has only empty files
                        files = list(dir_path.rglob('*'))
                        non_empty_files = [
                            f for f in files 
                            if f.is_file() and f.stat().st_size > 0
                        ]
                        
                        if not non_empty_files:
                            shutil.rmtree(dir_path)
                            logger.debug(f"Cleaned up old directory: {dir_path}")
                            removed_count += 1
                        else:
                            logger.debug(
                                f"Keeping directory with content: {dir_path} "
                                f"({len(non_empty_files)} non-empty files)"
                            )
                            
                except Exception as e:
                    logger.warning(f"Error processing directory {dir_path}: {e}")
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old directories")
        
        return removed_count


class TempDirectoryManager(DirectoryManager):
    """Specialized directory manager for temporary directories."""
    
    def __init__(self, prefix: str = "templ_temp_", **kwargs):
        """
        Initialize temporary directory manager.
        
        Args:
            prefix: Prefix for temporary directory name
            **kwargs: Additional arguments for DirectoryManager
        """
        # Use tempfile to generate unique directory
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        self._temp_dir_path = Path(temp_dir)
        
        # Initialize parent with the temp directory path as base
        super().__init__(
            base_name=self._temp_dir_path.name,
            lazy_creation=False,  # Already created by tempfile
            **kwargs
        )
        
        # Override the directory to use the tempfile path
        self._directory = self._temp_dir_path
        self._created = True
    
    def cleanup(self) -> bool:
        """Clean up temporary directory."""
        success = super().cleanup()
        if success:
            self._temp_dir_path = None
        return success


# Global cleanup utilities

def register_directory_cleanup(directory: Path, priority: int = 0) -> None:
    """
    Register a directory for cleanup on exit.
    
    Args:
        directory: Directory to register for cleanup
        priority: Cleanup priority (higher = cleaned up first)
    """
    DirectoryManager._registered_directories.add(directory)
    
    if not DirectoryManager._cleanup_registered:
        atexit.register(DirectoryManager._cleanup_all_registered)
        DirectoryManager._cleanup_registered = True


def cleanup_test_artifacts(base_path: Path = None) -> int:
    """
    Clean up test artifacts and temporary directories.
    
    Args:
        base_path: Base directory to search in
        
    Returns:
        int: Number of directories cleaned up
    """
    return DirectoryManager.cleanup_old_directories(
        patterns=[
            'templ_test_*',
            'test_temp_*', 
            'qa_test_*',
            'output_test_*',
            'debug_*',
            'temp_*'
        ],
        max_age_hours=1,  # Clean up test artifacts after 1 hour
        base_path=base_path
    )


def emergency_cleanup() -> None:
    """Emergency cleanup function for critical situations."""
    try:
        logger.info("Running emergency cleanup...")
        
        # Clean up all registered directories
        DirectoryManager._cleanup_all_registered()
        
        # Clean up old test artifacts
        cleanup_test_artifacts()
        
        # Clean up very old output directories
        DirectoryManager.cleanup_old_directories(max_age_hours=168)  # 1 week
        
        logger.info("Emergency cleanup completed")
        
    except Exception as e:
        logger.error(f"Emergency cleanup failed: {e}")


# Register emergency cleanup on module import
atexit.register(emergency_cleanup)