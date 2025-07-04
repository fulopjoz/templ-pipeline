"""
Resource Manager for TEMPL Pipeline

Smart resource management for memory and file cleanup with monitoring.
"""

import gc
import os
import psutil
import tempfile
import logging
from typing import Dict, Any, Optional, Set, List
from pathlib import Path
import time

import streamlit as st

logger = logging.getLogger(__name__)


class ResourceManager:
    """Smart resource management for memory and file cleanup"""

    def __init__(self, memory_limit_mb: int = 1024):
        """Initialize resource manager

        Args:
            memory_limit_mb: Memory limit in megabytes
        """
        self.memory_limit_mb = memory_limit_mb
        self.temp_files: Set[str] = set()
        self.large_objects: Dict[str, Any] = {}
        self.cleanup_history: List[Dict[str, Any]] = []

    def monitor_memory_usage(self) -> Dict[str, float]:
        """Monitor current memory usage

        Returns:
            Dictionary with memory statistics
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
            "total_mb": psutil.virtual_memory().total / 1024 / 1024,
            "used_mb": psutil.virtual_memory().used / 1024 / 1024,
        }

    def should_trigger_cleanup(self) -> bool:
        """Check if memory cleanup should be triggered

        Returns:
            True if cleanup is needed
        """
        memory_stats = self.monitor_memory_usage()

        # Trigger if RSS memory exceeds limit
        if memory_stats["rss_mb"] > self.memory_limit_mb:
            logger.warning(
                f"Memory limit exceeded: {memory_stats['rss_mb']:.1f}MB > {self.memory_limit_mb}MB"
            )
            return True

        # Trigger if system memory is low
        if memory_stats["available_mb"] < 500:  # Less than 500MB available
            logger.warning(
                f"Low system memory: {memory_stats['available_mb']:.1f}MB available"
            )
            return True

        # Trigger if memory usage is above 80% of limit
        if memory_stats["rss_mb"] > self.memory_limit_mb * 0.8:
            logger.info(
                f"Memory usage high: {memory_stats['rss_mb']:.1f}MB "
                f"({memory_stats['percent']:.1f}% of process limit)"
            )
            return True

        return False

    def cleanup_memory(self) -> Dict[str, Any]:
        """Perform comprehensive memory cleanup

        Returns:
            Dictionary with cleanup statistics
        """
        start_time = time.time()
        initial_memory = self.monitor_memory_usage()["rss_mb"]

        # Clear large objects
        objects_cleared = len(self.large_objects)
        self.large_objects.clear()

        # Clear Streamlit caches
        cache_cleared = False
        if initial_memory > self.memory_limit_mb * 0.9:
            st.cache_data.clear()
            st.cache_resource.clear()
            cache_cleared = True
            logger.info("Cleared Streamlit caches due to high memory usage")

        # Force garbage collection
        gc.collect(2)  # Full collection

        # Clean up temporary files
        files_cleaned = self.cleanup_temp_files()

        # Get final memory stats
        final_memory = self.monitor_memory_usage()["rss_mb"]
        memory_saved = max(0, initial_memory - final_memory)

        cleanup_stats = {
            "timestamp": time.time(),
            "duration_seconds": time.time() - start_time,
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_saved_mb": memory_saved,
            "objects_cleared": objects_cleared,
            "files_cleaned": files_cleaned,
            "cache_cleared": cache_cleared,
        }

        self.cleanup_history.append(cleanup_stats)

        logger.info(
            f"Memory cleanup completed: saved {memory_saved:.1f}MB in "
            f"{cleanup_stats['duration_seconds']:.2f}s"
        )

        return cleanup_stats

    def register_temp_file(self, filepath: str) -> str:
        """Register a temporary file for cleanup

        Args:
            filepath: Path to temporary file

        Returns:
            The filepath for chaining
        """
        self.temp_files.add(filepath)
        logger.debug(f"Registered temp file: {filepath}")
        return filepath

    def cleanup_temp_files(self, max_age_hours: Optional[float] = None) -> int:
        """Clean up registered temporary files

        Args:
            max_age_hours: Only clean files older than this (None = all files)

        Returns:
            Number of files cleaned
        """
        cleaned_count = 0
        current_time = time.time()

        for filepath in list(self.temp_files):
            try:
                if os.path.exists(filepath):
                    # Check age if specified
                    if max_age_hours is not None:
                        file_age = current_time - os.path.getmtime(filepath)
                        if file_age < max_age_hours * 3600:
                            continue

                    os.unlink(filepath)
                    cleaned_count += 1
                    logger.debug(f"Cleaned up temp file: {filepath}")

                self.temp_files.remove(filepath)

            except Exception as e:
                logger.error(f"Failed to clean temp file {filepath}: {e}")

        # Also clean system temp directory
        if max_age_hours is not None:
            cleaned_count += self._clean_old_temp_files(max_age_hours)

        return cleaned_count

    def _clean_old_temp_files(self, max_age_hours: float) -> int:
        """Clean old files from system temp directory

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of files cleaned
        """
        cleaned_count = 0
        temp_dir = Path(tempfile.gettempdir())
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600

        # Pattern for TEMPL-related temp files
        patterns = ["templ_*", "tmp*", "pose_*"]

        for pattern in patterns:
            for filepath in temp_dir.glob(pattern):
                try:
                    if filepath.is_file():
                        file_age = current_time - filepath.stat().st_mtime
                        if file_age > max_age_seconds:
                            filepath.unlink()
                            cleaned_count += 1
                except Exception:
                    # Skip files we can't access
                    pass

        if cleaned_count > 0:
            logger.info(f"Cleaned {cleaned_count} old temp files")

        return cleaned_count

    def store_large_object(
        self, key: str, obj: Any, metadata: Optional[Dict] = None
    ) -> None:
        """Store large object with memory monitoring

        Args:
            key: Object identifier
            obj: Object to store
            metadata: Optional metadata
        """
        # Check if cleanup is needed before storing
        if self.should_trigger_cleanup():
            self.cleanup_memory()

        self.large_objects[key] = {
            "object": obj,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "size_estimate": self._estimate_object_size(obj),
        }

        logger.debug(f"Stored large object: {key}")

    def get_large_object(self, key: str) -> Optional[Any]:
        """Retrieve large object

        Args:
            key: Object identifier

        Returns:
            Object or None
        """
        if key in self.large_objects:
            return self.large_objects[key]["object"]
        return None

    def _estimate_object_size(self, obj: Any) -> int:
        """Estimate object size in bytes

        Args:
            obj: Object to estimate

        Returns:
            Estimated size in bytes
        """
        try:
            import sys

            return sys.getsizeof(obj)
        except:
            # Fallback for complex objects
            return 1024  # 1KB default

    def get_resource_report(self) -> Dict[str, Any]:
        """Get comprehensive resource usage report

        Returns:
            Dictionary with resource statistics
        """
        memory_stats = self.monitor_memory_usage()

        # Calculate large objects size
        large_objects_size = (
            sum(item["size_estimate"] for item in self.large_objects.values())
            / 1024
            / 1024
        )  # Convert to MB

        return {
            "memory": {
                "current_usage_mb": memory_stats["rss_mb"],
                "limit_mb": self.memory_limit_mb,
                "utilization_percent": (memory_stats["rss_mb"] / self.memory_limit_mb)
                * 100,
                "system_available_mb": memory_stats["available_mb"],
                "large_objects_mb": large_objects_size,
            },
            "files": {
                "temp_files_tracked": len(self.temp_files),
                "temp_directory": tempfile.gettempdir(),
            },
            "cleanup": {
                "total_cleanups": len(self.cleanup_history),
                "last_cleanup": (
                    self.cleanup_history[-1] if self.cleanup_history else None
                ),
            },
        }

    def optimize_resources(self) -> Dict[str, Any]:
        """Perform comprehensive resource optimization

        Returns:
            Optimization results
        """
        logger.info("Starting resource optimization")

        results = {
            "memory_cleanup": self.cleanup_memory(),
            "file_cleanup": self.cleanup_temp_files(max_age_hours=24),
            "final_stats": self.get_resource_report(),
        }

        logger.info(
            f"Resource optimization complete: "
            f"{results['memory_cleanup']['memory_saved_mb']:.1f}MB saved, "
            f"{results['file_cleanup']} files cleaned"
        )

        return results


# Global resource manager instance
_resource_manager = None


def get_resource_manager() -> ResourceManager:
    """Get global resource manager instance

    Returns:
        ResourceManager instance
    """
    global _resource_manager
    if _resource_manager is None:
        _resource_manager = ResourceManager()
    return _resource_manager
