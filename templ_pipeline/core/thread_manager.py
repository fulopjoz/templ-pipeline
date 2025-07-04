"""
Thread resource management utilities for TEMPL pipeline.

Provides functions to detect and manage threading resources to prevent
"can't start new thread" errors in high-load scenarios.
"""

import logging
import threading
import os
from typing import Optional

# Optional psutil import for enhanced thread monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


class ThreadResourceManager:
    """Manages thread resources and provides adaptive thread pool sizing."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._max_threads = self._get_system_thread_limit()
            self._base_threads = threading.active_count()
            self._initialized = True
    
    def _get_system_thread_limit(self) -> int:
        """Get system thread limit with fallbacks."""
        try:
            # Try to get user process limit
            import resource
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NPROC)
            if soft_limit != resource.RLIM_INFINITY:
                # Reserve some threads for system operations
                return max(1, int(soft_limit * 0.8))
        except (ImportError, OSError):
            pass
        
        if HAS_PSUTIL:
            try:
                # Fallback to psutil if available
                return max(1, psutil.Process().num_threads() * 50)
            except (AttributeError, OSError):
                pass
        
        # Conservative fallback
        return max(1, os.cpu_count() * 4)
    
    def get_safe_worker_count(self, requested_workers: int, task_type: str = "general") -> int:
        """Get a safe number of workers based on current thread usage."""
        current_threads = threading.active_count()
        available_threads = max(1, self._max_threads - current_threads - 10)  # Reserve 10 threads
        
        # Different strategies for different task types
        if task_type == "scoring":
            # Scoring can be memory intensive, use fewer threads
            max_workers = min(requested_workers, available_threads // 2, 4)
        elif task_type == "conformer":
            # Conformer generation is CPU bound
            max_workers = min(requested_workers, available_threads, os.cpu_count())
        else:
            # Conservative default
            max_workers = min(requested_workers, available_threads // 3, 2)
        
        safe_workers = max(1, max_workers)
        
        if safe_workers < requested_workers:
            logger.warning(
                f"Reducing {task_type} workers from {requested_workers} to {safe_workers} "
                f"(active threads: {current_threads}, limit: {self._max_threads})"
            )
        
        return safe_workers
    
    def check_thread_health(self) -> dict:
        """Check current thread health and resource usage."""
        current_threads = threading.active_count()
        return {
            "active_threads": current_threads,
            "max_threads": self._max_threads,
            "available_threads": max(0, self._max_threads - current_threads),
            "usage_percent": (current_threads / self._max_threads) * 100,
            "thread_pressure": current_threads > (self._max_threads * 0.7)
        }
    
    def is_safe_to_create_threads(self, num_threads: int) -> bool:
        """Check if it's safe to create additional threads."""
        current_threads = threading.active_count()
        return (current_threads + num_threads) < (self._max_threads * 0.8)


def get_safe_worker_count(requested_workers: int, task_type: str = "general") -> int:
    """Convenience function to get safe worker count."""
    manager = ThreadResourceManager()
    return manager.get_safe_worker_count(requested_workers, task_type)


def check_thread_pressure() -> bool:
    """Check if system is under thread pressure."""
    manager = ThreadResourceManager()
    health = manager.check_thread_health()
    return health["thread_pressure"]


def log_thread_status(context: str = ""):
    """Log current thread status for debugging."""
    manager = ThreadResourceManager()
    health = manager.check_thread_health()
    logger.debug(
        f"Thread status {context}: {health['active_threads']}/{health['max_threads']} "
        f"({health['usage_percent']:.1f}%) - Pressure: {health['thread_pressure']}"
    )