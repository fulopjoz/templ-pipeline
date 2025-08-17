# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Cache Optimization Utilities for TEMPL Pipeline

Provides smart cache management and optimization strategies.
"""

import streamlit as st
import logging
import time
from typing import Dict, Any, Optional, List
import psutil
import os

logger = logging.getLogger(__name__)


class CacheOptimizer:
    """Smart cache optimization for scientific computing workflows"""

    def __init__(self):
        """Initialize cache optimizer"""
        self.optimization_stats = {
            "cache_clears": 0,
            "memory_optimizations": 0,
            "selective_clears": 0,
            "last_optimization": None,
        }

    def analyze_cache_health(self) -> Dict[str, Any]:
        """Analyze current cache health and performance

        Returns:
            Dictionary with cache health metrics
        """
        try:
            # Get system memory info
            memory = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info()

            # Estimate Streamlit cache usage (rough approximation)
            total_memory_mb = memory.total / (1024 * 1024)
            used_memory_mb = memory.used / (1024 * 1024)
            process_memory_mb = process_memory.rss / (1024 * 1024)

            # Check if memory pressure exists
            memory_pressure = memory.percent > 80.0
            process_pressure = process_memory_mb > (
                total_memory_mb * 0.15
            )  # More than 15% of system

            # Analyze session state size
            session_size_estimate = self._estimate_session_state_size()

            health_score = self._calculate_health_score(
                memory.percent, process_memory_mb, session_size_estimate
            )

            return {
                "health_score": health_score,
                "system_memory_percent": memory.percent,
                "process_memory_mb": process_memory_mb,
                "session_state_size_mb": session_size_estimate,
                "memory_pressure": memory_pressure,
                "process_pressure": process_pressure,
                "needs_optimization": health_score < 70,
                "recommendation": self._get_optimization_recommendation(health_score),
            }

        except Exception as e:
            logger.error(f"Error analyzing cache health: {e}")
            return {
                "health_score": 50,
                "error": str(e),
                "needs_optimization": True,
                "recommendation": "Clear caches due to analysis error",
            }

    def optimize_caches_smart(self, force: bool = False) -> Dict[str, Any]:
        """Smart cache optimization based on current conditions

        Args:
            force: Force optimization regardless of conditions

        Returns:
            Optimization results
        """
        start_time = time.time()

        try:
            # Analyze current state
            health = self.analyze_cache_health()

            if not force and not health.get("needs_optimization", False):
                return {
                    "action": "no_action",
                    "reason": "Cache health is good",
                    "health_score": health.get("health_score", 0),
                    "time_taken": time.time() - start_time,
                }

            # Determine optimization strategy
            if (
                health.get("process_pressure", False)
                or health.get("health_score", 0) < 50
            ):
                # Aggressive optimization
                return self._aggressive_optimization(health)
            elif (
                health.get("memory_pressure", False)
                or health.get("health_score", 0) < 70
            ):
                # Selective optimization
                return self._selective_optimization(health)
            else:
                # Light optimization
                return self._light_optimization(health)

        except Exception as e:
            logger.error(f"Error during cache optimization: {e}")
            return {
                "action": "error",
                "error": str(e),
                "time_taken": time.time() - start_time,
            }

    def clear_stale_pipeline_data(self) -> int:
        """Clear stale pipeline data from session state

        Returns:
            Number of items cleared
        """
        stale_keys = [
            "poses_timestamp",
            "best_poses_refs",
            "pipeline_poses",
            "template_info",
            "mcs_details",
            "all_ranked_poses",
        ]

        cleared_count = 0
        for key in stale_keys:
            if key in st.session_state:
                try:
                    del st.session_state[key]
                    cleared_count += 1
                    logger.debug(f"Cleared stale key: {key}")
                except Exception as e:
                    logger.warning(f"Failed to clear {key}: {e}")

        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} stale pipeline data items")

        return cleared_count

    def reset_for_new_calculation(self) -> Dict[str, Any]:
        """Reset caches and state for new calculation

        Returns:
            Reset operation results
        """
        start_time = time.time()

        try:
            # Clear stale pipeline data
            stale_cleared = self.clear_stale_pipeline_data()

            # Clear relevant caches
            st.cache_data.clear()

            # Keep resource cache (hardware detection, etc.) but clear data cache
            # This preserves expensive operations while clearing calculation data

            self.optimization_stats["cache_clears"] += 1
            self.optimization_stats["last_optimization"] = time.time()

            return {
                "action": "reset_for_calculation",
                "stale_items_cleared": stale_cleared,
                "time_taken": time.time() - start_time,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error resetting for new calculation: {e}")
            return {
                "action": "reset_error",
                "error": str(e),
                "time_taken": time.time() - start_time,
                "success": False,
            }

    def _estimate_session_state_size(self) -> float:
        """Estimate session state size in MB"""
        try:
            import sys

            total_size = 0
            large_items = []

            for key in st.session_state:
                try:
                    item_size = sys.getsizeof(st.session_state[key])
                    total_size += item_size

                    # Track large items
                    if item_size > 1024 * 1024:  # > 1MB
                        large_items.append((key, item_size / (1024 * 1024)))

                except Exception:
                    pass  # Skip items that can't be sized

            size_mb = total_size / (1024 * 1024)

            if large_items:
                logger.debug(f"Large session items: {large_items}")

            return size_mb

        except Exception as e:
            logger.warning(f"Could not estimate session state size: {e}")
            return 10.0  # Default estimate

    def _calculate_health_score(
        self, memory_percent: float, process_mb: float, session_mb: float
    ) -> int:
        """Calculate cache health score (0-100)"""
        score = 100

        # Penalize high memory usage
        if memory_percent > 90:
            score -= 40
        elif memory_percent > 80:
            score -= 20
        elif memory_percent > 70:
            score -= 10

        # Penalize large process memory
        if process_mb > 2000:  # > 2GB
            score -= 30
        elif process_mb > 1000:  # > 1GB
            score -= 15

        # Penalize large session state
        if session_mb > 500:  # > 500MB
            score -= 25
        elif session_mb > 100:  # > 100MB
            score -= 10

        return max(0, score)

    def _get_optimization_recommendation(self, health_score: int) -> str:
        """Get optimization recommendation based on health score"""
        if health_score < 30:
            return "Immediate cache clearing and memory optimization needed"
        elif health_score < 50:
            return "Aggressive cache optimization recommended"
        elif health_score < 70:
            return "Selective cache cleaning would help performance"
        else:
            return "Cache health is good, no action needed"

    def _aggressive_optimization(self, health: Dict[str, Any]) -> Dict[str, Any]:
        """Perform aggressive cache optimization"""
        start_time = time.time()

        try:
            # Clear all caches
            st.cache_data.clear()
            st.cache_resource.clear()

            # Clear stale data
            stale_cleared = self.clear_stale_pipeline_data()

            # Clear large session state items
            large_items_cleared = self._clear_large_session_items()

            self.optimization_stats["memory_optimizations"] += 1

            return {
                "action": "aggressive_optimization",
                "stale_cleared": stale_cleared,
                "large_items_cleared": large_items_cleared,
                "time_taken": time.time() - start_time,
                "health_before": health.get("health_score", 0),
            }

        except Exception as e:
            logger.error(f"Aggressive optimization failed: {e}")
            return {"action": "aggressive_failed", "error": str(e)}

    def _selective_optimization(self, health: Dict[str, Any]) -> Dict[str, Any]:
        """Perform selective cache optimization"""
        start_time = time.time()

        try:
            # Clear only data cache, keep resource cache
            st.cache_data.clear()

            # Clear stale data
            stale_cleared = self.clear_stale_pipeline_data()

            self.optimization_stats["selective_clears"] += 1

            return {
                "action": "selective_optimization",
                "stale_cleared": stale_cleared,
                "time_taken": time.time() - start_time,
                "health_before": health.get("health_score", 0),
            }

        except Exception as e:
            logger.error(f"Selective optimization failed: {e}")
            return {"action": "selective_failed", "error": str(e)}

    def _light_optimization(self, health: Dict[str, Any]) -> Dict[str, Any]:
        """Perform light cache optimization"""
        start_time = time.time()

        try:
            # Only clear stale pipeline data
            stale_cleared = self.clear_stale_pipeline_data()

            return {
                "action": "light_optimization",
                "stale_cleared": stale_cleared,
                "time_taken": time.time() - start_time,
                "health_before": health.get("health_score", 0),
            }

        except Exception as e:
            logger.error(f"Light optimization failed: {e}")
            return {"action": "light_failed", "error": str(e)}

    def _clear_large_session_items(self) -> int:
        """Clear large items from session state"""
        large_item_keys = [
            "poses",
            "templates",
            "query_mol",
            "custom_templates",
            "all_ranked_poses",
            "pipeline_poses",
        ]

        cleared = 0
        for key in large_item_keys:
            if key in st.session_state:
                try:
                    del st.session_state[key]
                    cleared += 1
                except Exception as e:
                    logger.warning(f"Failed to clear large item {key}: {e}")

        return cleared

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            **self.optimization_stats,
            "current_health": self.analyze_cache_health(),
        }


# Global instance
_cache_optimizer = CacheOptimizer()


def get_cache_optimizer() -> CacheOptimizer:
    """Get global cache optimizer instance"""
    return _cache_optimizer


def smart_cache_reset():
    """Smart cache reset for new calculations"""
    optimizer = get_cache_optimizer()
    return optimizer.reset_for_new_calculation()


def auto_optimize_if_needed():
    """Automatically optimize caches if needed"""
    optimizer = get_cache_optimizer()
    health = optimizer.analyze_cache_health()

    if health.get("needs_optimization", False):
        logger.info("Auto-optimizing caches due to poor health")
        result = optimizer.optimize_caches_smart()
        logger.info(f"Auto-optimization result: {result.get('action', 'unknown')}")
        return result

    return {"action": "no_action", "reason": "Cache health is good"}
