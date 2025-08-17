# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Cache Manager for TEMPL Pipeline

Unified caching system with performance monitoring and memory-aware strategies.
"""

import hashlib
import json
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

import streamlit as st

logger = logging.getLogger(__name__)


class CacheManager:
    """Unified caching system for TEMPL Pipeline"""

    def __init__(self):
        """Initialize cache manager"""
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_time_saved": 0.0,
        }
        self.function_stats: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def resource_cache(ttl: Optional[int] = None, show_spinner: bool = False):
        """Decorator for caching expensive resources with performance monitoring

        Args:
            ttl: Time to live in seconds (None for no expiration)
            show_spinner: Whether to show spinner during computation

        Returns:
            Decorated function with caching
        """

        def decorator(func: Callable) -> Callable:
            # Create unique key for this function
            func_key = f"{func.__module__}.{func.__name__}"

            @st.cache_resource(ttl=ttl, show_spinner=show_spinner)
            @wraps(func)
            def cached_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # Log cache miss (first execution)
                    logger.info(
                        f"Cache MISS: {func.__name__} took {execution_time:.3f}s"
                    )

                    # Track statistics
                    manager = CacheManager._get_instance()
                    manager._record_miss(func_key, execution_time)

                    return result
                except Exception as e:
                    logger.error(f"Cache error in {func.__name__}: {e}")
                    raise

            # Wrap again to track hits
            @wraps(cached_wrapper)
            def wrapper(*args, **kwargs):
                manager = CacheManager._get_instance()
                start_time = time.time()

                # Check if this is a hit by seeing if execution is fast
                result = cached_wrapper(*args, **kwargs)
                execution_time = time.time() - start_time

                # If execution was very fast, it was likely a cache hit
                if execution_time < 0.01:  # Less than 10ms suggests cache hit
                    manager._record_hit(func_key)

                return result

            return wrapper

        return decorator

    @staticmethod
    def data_cache(
        ttl: int = 3600,
        max_entries: int = 100,
        persist: bool = False,
        show_spinner: bool = False,
    ):
        """Memory-aware data caching with size limits

        Args:
            ttl: Time to live in seconds
            max_entries: Maximum number of cache entries
            persist: Whether to persist across sessions
            show_spinner: Whether to show spinner

        Returns:
            Decorated function with data caching
        """

        def decorator(func: Callable) -> Callable:
            func_key = f"{func.__module__}.{func.__name__}"

            @st.cache_data(
                ttl=ttl,
                max_entries=max_entries,
                persist=persist,
                show_spinner=show_spinner,
            )
            @wraps(func)
            def cached_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # Log cache miss
                    logger.debug(
                        f"Data cache MISS: {func.__name__} took {execution_time:.3f}s"
                    )

                    # Track statistics
                    manager = CacheManager._get_instance()
                    manager._record_miss(func_key, execution_time)

                    return result
                except Exception as e:
                    logger.error(f"Data cache error in {func.__name__}: {e}")
                    raise

            # Wrap to track hits
            @wraps(cached_wrapper)
            def wrapper(*args, **kwargs):
                manager = CacheManager._get_instance()
                start_time = time.time()

                result = cached_wrapper(*args, **kwargs)
                execution_time = time.time() - start_time

                if execution_time < 0.01:  # Cache hit
                    manager._record_hit(func_key)

                return result

            return wrapper

        return decorator

    @staticmethod
    def compute_cache_key(*args, **kwargs) -> str:
        """Compute stable cache key from arguments

        Returns:
            Hash string for cache key
        """
        # Create a string representation of arguments
        key_parts = []

        # Handle positional arguments
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            elif isinstance(arg, (list, tuple)):
                key_parts.append(json.dumps(arg, sort_keys=True))
            elif isinstance(arg, dict):
                key_parts.append(json.dumps(arg, sort_keys=True))
            else:
                # For complex objects, use type and id
                key_parts.append(f"{type(arg).__name__}_{id(arg)}")

        # Handle keyword arguments
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        # Create hash
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    def clear_all_caches(self) -> Dict[str, Any]:
        """Clear all Streamlit caches and return statistics

        Returns:
            Dictionary with clearing statistics
        """
        start_time = time.time()

        # Clear caches
        st.cache_data.clear()
        st.cache_resource.clear()

        # Record evictions
        self.cache_stats["evictions"] += 1

        clear_time = time.time() - start_time

        logger.info(f"Cleared all caches in {clear_time:.3f}s")

        return {"clear_time": clear_time, "cache_stats": self.get_cache_statistics()}

    def selective_clear(self, pattern: Optional[str] = None) -> int:
        """Selectively clear cache entries matching pattern

        Args:
            pattern: Pattern to match function names

        Returns:
            Number of entries cleared
        """
        # Note: Streamlit doesn't provide direct access to cache keys
        # This is a placeholder for future enhancement
        logger.warning("Selective cache clearing not fully implemented in Streamlit")

        if pattern:
            # Clear specific function stats
            cleared = 0
            for func_key in list(self.function_stats.keys()):
                if pattern in func_key:
                    del self.function_stats[func_key]
                    cleared += 1
            return cleared

        return 0

    def _record_hit(self, func_key: str) -> None:
        """Record a cache hit

        Args:
            func_key: Function identifier
        """
        self.cache_stats["hits"] += 1

        if func_key not in self.function_stats:
            self.function_stats[func_key] = {
                "hits": 0,
                "misses": 0,
                "total_compute_time": 0.0,
                "avg_compute_time": 0.0,
            }

        self.function_stats[func_key]["hits"] += 1

    def _record_miss(self, func_key: str, compute_time: float) -> None:
        """Record a cache miss

        Args:
            func_key: Function identifier
            compute_time: Time taken to compute result
        """
        self.cache_stats["misses"] += 1

        if func_key not in self.function_stats:
            self.function_stats[func_key] = {
                "hits": 0,
                "misses": 0,
                "total_compute_time": 0.0,
                "avg_compute_time": 0.0,
            }

        stats = self.function_stats[func_key]
        stats["misses"] += 1
        stats["total_compute_time"] += compute_time
        stats["avg_compute_time"] = stats["total_compute_time"] / stats["misses"]

        # Estimate time saved by cache hits
        self.cache_stats["total_time_saved"] += (
            stats["hits"] * stats["avg_compute_time"]
        )

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (
            self.cache_stats["hits"] / total_requests * 100 if total_requests > 0 else 0
        )

        return {
            "overall": {
                "hits": self.cache_stats["hits"],
                "misses": self.cache_stats["misses"],
                "hit_rate": hit_rate,
                "evictions": self.cache_stats["evictions"],
                "time_saved_seconds": self.cache_stats["total_time_saved"],
            },
            "by_function": self.function_stats,
        }

    def get_performance_report(self) -> str:
        """Generate human-readable performance report

        Returns:
            Performance report string
        """
        stats = self.get_cache_statistics()
        overall = stats["overall"]

        report_lines = [
            "=== Cache Performance Report ===",
            f"Total Hits: {overall['hits']}",
            f"Total Misses: {overall['misses']}",
            f"Hit Rate: {overall['hit_rate']:.1f}%",
            f"Time Saved: {overall['time_saved_seconds']:.2f} seconds",
            f"Cache Evictions: {overall['evictions']}",
            "\n=== Top Cached Functions ===",
        ]

        # Sort functions by time saved
        function_savings = []
        for func_key, func_stats in stats["by_function"].items():
            time_saved = func_stats["hits"] * func_stats["avg_compute_time"]
            function_savings.append((func_key, time_saved, func_stats))

        function_savings.sort(key=lambda x: x[1], reverse=True)

        for func_key, time_saved, func_stats in function_savings[:10]:
            func_name = func_key.split(".")[-1]
            hit_rate = (
                func_stats["hits"] / (func_stats["hits"] + func_stats["misses"]) * 100
                if (func_stats["hits"] + func_stats["misses"]) > 0
                else 0
            )
            report_lines.append(
                f"{func_name}: {time_saved:.2f}s saved, "
                f"{hit_rate:.1f}% hit rate, "
                f"{func_stats['avg_compute_time']:.3f}s avg compute"
            )

        return "\n".join(report_lines)

    @classmethod
    def _get_instance(cls) -> "CacheManager":
        """Get or create singleton instance

        Returns:
            CacheManager instance
        """
        if not hasattr(cls, "_instance"):
            cls._instance = cls()
        return cls._instance


# Create global instance
_cache_manager = CacheManager()


# Export convenience functions
def get_cache_manager() -> CacheManager:
    """Get global cache manager instance"""
    return _cache_manager


def resource_cache(ttl: Optional[int] = None, show_spinner: bool = False):
    """Convenience wrapper for resource caching"""
    return CacheManager.resource_cache(ttl, show_spinner)


def data_cache(ttl: int = 3600, max_entries: int = 100):
    """Convenience wrapper for data caching"""
    return CacheManager.data_cache(ttl, max_entries)
