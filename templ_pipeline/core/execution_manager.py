"""
Comprehensive execution management for TEMPL pipeline.

This module consolidates thread resource management and skip tracking to provide
unified execution monitoring and control. It handles:

- Thread resource management and adaptive sizing
- Skip tracking and reporting for failed molecules
- Execution health monitoring
- Resource constraint detection
- Safe worker count calculation
- Graceful error handling with skip exceptions

Features:
- Adaptive thread pool sizing based on system resources
- Comprehensive skip tracking with detailed reporting
- Thread health monitoring and pressure detection
- Integration with pipeline execution flow
- Backward compatibility with existing components
"""

import os
import json
import logging
import threading
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

# Optional psutil import for enhanced monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Optional pebble import for robust process management like true_mcs.py
try:
    import pebble
    from pebble import ProcessPool
    HAS_PEBBLE = True
except ImportError:
    HAS_PEBBLE = False

logger = logging.getLogger(__name__)


class SkipReason(Enum):
    """Enumeration of reasons why a molecule might be skipped."""
    LARGE_PEPTIDE = "large_peptide"
    ORGANOMETALLIC_COMPLEX = "organometallic_complex"
    RHENIUM_COMPLEX = "rhenium_complex"
    INVALID_MOLECULE = "invalid_molecule"
    EMBEDDING_FAILED = "embedding_failed"
    TEMPLATE_UNAVAILABLE = "template_unavailable"
    CONFORMER_GENERATION_FAILED = "conformer_generation_failed"
    SCORING_FAILED = "scoring_failed"
    THREAD_RESOURCE_EXHAUSTED = "thread_resource_exhausted"
    MEMORY_PRESSURE = "memory_pressure"
    TIMEOUT = "timeout"
    OTHER = "other"


@dataclass
class SkipRecord:
    """Record of a skipped molecule with detailed information."""
    molecule_id: str
    reason: SkipReason
    message: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "molecule_id": self.molecule_id,
            "reason": self.reason.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details or {}
        }


@dataclass
class ThreadHealthInfo:
    """Information about thread health and resource usage."""
    active_threads: int
    max_threads: int
    available_threads: int
    usage_percent: float
    thread_pressure: bool
    safe_worker_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "active_threads": self.active_threads,
            "max_threads": self.max_threads,
            "available_threads": self.available_threads,
            "usage_percent": self.usage_percent,
            "thread_pressure": self.thread_pressure,
            "safe_worker_count": self.safe_worker_count
        }


class MoleculeSkipException(Exception):
    """Exception that signals a molecule should be skipped gracefully."""
    
    def __init__(self, reason: SkipReason, message: str, details: Optional[Dict[str, Any]] = None):
        self.reason = reason
        self.message = message
        self.details = details or {}
        super().__init__(message)


class ExecutionManager:
    """
    Comprehensive execution manager for TEMPL pipeline.
    
    Combines thread resource management with skip tracking to provide
    unified execution monitoring and control.
    """
    
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
            # Thread management
            self._max_threads = self._get_system_thread_limit()
            self._base_threads = threading.active_count()
            
            # Skip management
            self.skipped_records: List[SkipRecord] = []
            self._skip_counts: Dict[SkipReason, int] = {}
            
            # Execution monitoring
            self._execution_start_time = None
            self._total_processed = 0
            self._total_failed = 0
            
            self._initialized = True
            logger.info(f"Execution manager initialized with max_threads={self._max_threads}")
    
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
    
    def start_execution(self) -> None:
        """Mark the start of execution for monitoring."""
        self._execution_start_time = time.time()
        self._total_processed = 0
        self._total_failed = 0
        logger.info("Execution monitoring started")
    
    def get_thread_health(self) -> ThreadHealthInfo:
        """Get current thread health and resource usage."""
        current_threads = threading.active_count()
        available_threads = max(0, self._max_threads - current_threads)
        usage_percent = (current_threads / self._max_threads) * 100
        thread_pressure = current_threads > (self._max_threads * 0.7)
        
        # Calculate safe worker count
        safe_workers = max(1, available_threads // 3)
        
        return ThreadHealthInfo(
            active_threads=current_threads,
            max_threads=self._max_threads,
            available_threads=available_threads,
            usage_percent=usage_percent,
            thread_pressure=thread_pressure,
            safe_worker_count=safe_workers
        )
    
    def get_safe_worker_count(self, requested_workers: int, task_type: str = "general") -> int:
        """Get a safe number of workers based on system capabilities."""
        system_cpu_count = os.cpu_count() or 1
        
        if requested_workers <= 0:
            recommended_workers = system_cpu_count
        else:
            recommended_workers = requested_workers
        
        # Use most available cores for better performance
        if task_type == "scoring":
            max_workers = min(recommended_workers, system_cpu_count)
        elif task_type == "conformer":
            max_workers = min(recommended_workers, system_cpu_count)
        elif task_type == "embedding":
            max_workers = min(recommended_workers, system_cpu_count // 2)
        elif task_type == "benchmark":
            # Benchmark tasks should use all available cores
            max_workers = system_cpu_count
        else:
            max_workers = min(recommended_workers, system_cpu_count)
        
        safe_workers = max(1, max_workers)
        
        if safe_workers < requested_workers:
            current_threads = threading.active_count()
            logger.warning(
                f"Reducing {task_type} workers from {requested_workers} to {safe_workers} "
                f"(active threads: {current_threads}, limit: {self._max_threads})"
            )
        
        return safe_workers
    
    def is_safe_to_create_threads(self, num_threads: int) -> bool:
        """Check if it's safe to create additional threads."""
        current_threads = threading.active_count()
        return (current_threads + num_threads) < (self._max_threads * 0.8)
    
    def check_thread_pressure(self) -> bool:
        """Check if system is under thread pressure."""
        health = self.get_thread_health()
        return health.thread_pressure
    
    def skip_molecule(
        self, 
        molecule_id: str, 
        reason: SkipReason, 
        message: str, 
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a skipped molecule."""
        record = SkipRecord(
            molecule_id=molecule_id,
            reason=reason,
            message=message,
            timestamp=datetime.now().isoformat(),
            details=details or {}
        )
        
        self.skipped_records.append(record)
        self._skip_counts[reason] = self._skip_counts.get(reason, 0) + 1
        self._total_failed += 1
        
        # Log the skip with appropriate level
        if reason in [SkipReason.LARGE_PEPTIDE, SkipReason.ORGANOMETALLIC_COMPLEX]:
            logger.info(f"SKIP {molecule_id}: {message}")
        else:
            logger.warning(f"SKIP {molecule_id}: {message}")
    
    def record_successful_processing(self, molecule_id: str) -> None:
        """Record successful processing of a molecule."""
        self._total_processed += 1
        logger.debug(f"Successfully processed {molecule_id}")
    
    def get_skip_summary(self) -> Dict[str, Any]:
        """Get summary of all skipped molecules."""
        total_skipped = len(self.skipped_records)
        
        summary = {
            "total_skipped": total_skipped,
            "total_processed": self._total_processed,
            "total_failed": self._total_failed,
            "skip_counts_by_reason": {reason.value: count for reason, count in self._skip_counts.items()},
            "skip_rate_by_reason": {},
            "recent_skips": [record.to_dict() for record in self.skipped_records[-5:]]
        }
        
        # Calculate percentages if we have skips
        if total_skipped > 0:
            for reason, count in self._skip_counts.items():
                summary["skip_rate_by_reason"][reason.value] = (count / total_skipped) * 100
        
        return summary
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary."""
        thread_health = self.get_thread_health()
        skip_summary = self.get_skip_summary()
        
        execution_time = None
        if self._execution_start_time:
            execution_time = time.time() - self._execution_start_time
        
        return {
            "execution_time_seconds": execution_time,
            "thread_health": thread_health.to_dict(),
            "skip_summary": skip_summary,
            "total_processed": self._total_processed,
            "total_failed": self._total_failed,
            "success_rate": self._total_processed / max(1, self._total_processed + self._total_failed) * 100
        }
    
    def get_skipped_by_reason(self, reason: SkipReason) -> List[SkipRecord]:
        """Get all molecules skipped for a specific reason."""
        return [record for record in self.skipped_records if record.reason == reason]
    
    def clear_skips(self) -> None:
        """Clear all skip records."""
        self.skipped_records.clear()
        self._skip_counts.clear()
        self._total_failed = 0
        logger.info("Skip records cleared")
    
    def reset_execution_stats(self) -> None:
        """Reset execution statistics."""
        self._execution_start_time = None
        self._total_processed = 0
        self._total_failed = 0
        self.clear_skips()
        logger.info("Execution statistics reset")
    
    def save_execution_report(self, output_path: str) -> bool:
        """Save comprehensive execution report to file."""
        try:
            report_data = {
                "execution_summary": self.get_execution_summary(),
                "detailed_skip_records": [record.to_dict() for record in self.skipped_records],
                "generation_time": datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Execution report saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save execution report: {e}")
            return False
    
    def save_skip_report(self, output_path: str) -> bool:
        """Save skip report to file (backward compatibility)."""
        try:
            summary = self.get_skip_summary()
            
            report_data = {
                "summary": summary,
                "detailed_records": [record.to_dict() for record in self.skipped_records]
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Skip report saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save skip report: {e}")
            return False
    
    def log_thread_status(self, context: str = "") -> None:
        """Log current thread status for debugging."""
        health = self.get_thread_health()
        logger.debug(
            f"Thread status {context}: {health.active_threads}/{health.max_threads} "
            f"({health.usage_percent:.1f}%) - Pressure: {health.thread_pressure}"
        )
    
    def log_skip_summary(self) -> None:
        """Log a summary of all skips."""
        summary = self.get_skip_summary()
        total = summary["total_skipped"]
        
        if total == 0:
            logger.info("No molecules were skipped")
            return
        
        logger.info(f"SKIP SUMMARY: {total} molecules skipped")
        for reason, count in summary["skip_counts_by_reason"].items():
            percentage = summary["skip_rate_by_reason"][reason]
            logger.info(f"  {reason}: {count} molecules ({percentage:.1f}%)")
    
    def create_validation_skip_wrapper(self, validation_func):
        """Decorator to convert validation failures into skip exceptions."""
        def wrapper(mol, mol_name="unknown", **kwargs):
            is_valid, message = validation_func(mol, mol_name, **kwargs)
            if not is_valid:
                # Determine skip reason from message content
                if "large peptide" in message.lower():
                    reason = SkipReason.LARGE_PEPTIDE
                elif "rhenium" in message.lower():
                    reason = SkipReason.RHENIUM_COMPLEX
                elif "organometallic" in message.lower():
                    reason = SkipReason.ORGANOMETALLIC_COMPLEX
                else:
                    reason = SkipReason.INVALID_MOLECULE
                
                raise MoleculeSkipException(reason, message, {"molecule_name": mol_name})
            
            return is_valid, message
        
        return wrapper
    
    def with_error_handling(self, func, molecule_id: str = "unknown"):
        """Decorator to handle errors and convert them to skips."""
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                self.record_successful_processing(molecule_id)
                return result
            except MoleculeSkipException as e:
                self.skip_molecule(molecule_id, e.reason, e.message, e.details)
                raise
            except Exception as e:
                self.skip_molecule(
                    molecule_id, 
                    SkipReason.OTHER, 
                    f"Unexpected error: {str(e)}", 
                    {"error_type": type(e).__name__}
                )
                raise MoleculeSkipException(SkipReason.OTHER, f"Unexpected error: {str(e)}")
        
        return wrapper


# Global execution manager instance
_global_execution_manager = ExecutionManager()


# Convenience functions for backward compatibility and easy access

def get_execution_manager() -> ExecutionManager:
    """Get the global execution manager instance."""
    return _global_execution_manager


def get_safe_worker_count(requested_workers: int, task_type: str = "general") -> int:
    """Convenience function to get safe worker count."""
    return _global_execution_manager.get_safe_worker_count(requested_workers, task_type)


def check_thread_pressure() -> bool:
    """Check if system is under thread pressure."""
    return _global_execution_manager.check_thread_pressure()


def log_thread_status(context: str = "") -> None:
    """Log current thread status for debugging."""
    _global_execution_manager.log_thread_status(context)


def skip_molecule(
    molecule_id: str, 
    reason: SkipReason, 
    message: str, 
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Convenience function to skip a molecule using the global manager."""
    _global_execution_manager.skip_molecule(molecule_id, reason, message, details)


def raise_skip_exception(
    reason: SkipReason, 
    message: str, 
    details: Optional[Dict[str, Any]] = None
) -> None:
    """Raise a skip exception that can be caught and handled gracefully."""
    raise MoleculeSkipException(reason, message, details)


def record_successful_processing(molecule_id: str) -> None:
    """Record successful processing of a molecule."""
    _global_execution_manager.record_successful_processing(molecule_id)


def get_skip_manager():
    """Get the global skip manager instance (backward compatibility)."""
    return _global_execution_manager


class RobustProcessPoolWrapper:
    """Wrapper to make pebble ProcessPool compatible with concurrent.futures interface."""
    
    def __init__(self, max_workers: int):
        if HAS_PEBBLE:
            self.pool = pebble.ProcessPool(max_workers=max_workers)
            self.use_pebble = True
            logger.debug(f"Created pebble ProcessPool with {max_workers} workers")
        else:
            from concurrent.futures import ProcessPoolExecutor
            self.pool = ProcessPoolExecutor(max_workers=max_workers)
            self.use_pebble = False
            logger.debug(f"Created ProcessPoolExecutor with {max_workers} workers")
    
    def map(self, func, iterable):
        """Map function compatible with both pebble and concurrent.futures."""
        if self.use_pebble:
            # Use pebble's schedule/result pattern like true_mcs.py
            futures = []
            for args in iterable:
                future = self.pool.schedule(func, args=[args])
                futures.append(future)
            
            # Collect results
            results = []
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Pebble task failed: {e}")
                    results.append(None)
            return results
        else:
            # Use standard map
            return list(self.pool.map(func, iterable))
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.close()
        if self.use_pebble:
            self.pool.join()
        else:
            self.pool.shutdown(wait=True)


def create_robust_process_pool(max_workers: int, task_type: str = "general") -> RobustProcessPoolWrapper:
    """Create a robust process pool using pebble if available, fallback to ProcessPoolExecutor.
    
    Args:
        max_workers: Maximum number of worker processes
        task_type: Type of task for worker optimization
        
    Returns:
        Wrapped process pool executor compatible with concurrent.futures interface
    """
    # Get safe worker count
    safe_workers = get_safe_worker_count(max_workers, task_type)
    
    logger.info(f"Creating robust process pool with {safe_workers} workers for {task_type}")
    return RobustProcessPoolWrapper(safe_workers)


def log_skip_summary() -> None:
    """Log a summary of all skips."""
    _global_execution_manager.log_skip_summary()


def create_validation_skip_wrapper(validation_func):
    """Decorator to convert validation failures into skip exceptions."""
    return _global_execution_manager.create_validation_skip_wrapper(validation_func)


def start_execution_monitoring() -> None:
    """Start execution monitoring."""
    _global_execution_manager.start_execution()


def get_execution_summary() -> Dict[str, Any]:
    """Get comprehensive execution summary."""
    return _global_execution_manager.get_execution_summary()


def save_execution_report(output_path: str) -> bool:
    """Save comprehensive execution report to file."""
    return _global_execution_manager.save_execution_report(output_path)


def reset_execution_stats() -> None:
    """Reset execution statistics."""
    _global_execution_manager.reset_execution_stats()


# Backward compatibility classes

class ThreadResourceManager:
    """Backward compatibility wrapper for thread management."""
    
    def __init__(self):
        self.manager = _global_execution_manager
    
    def get_safe_worker_count(self, requested_workers: int, task_type: str = "general") -> int:
        return self.manager.get_safe_worker_count(requested_workers, task_type)
    
    def check_thread_health(self) -> dict:
        health = self.manager.get_thread_health()
        return health.to_dict()
    
    def is_safe_to_create_threads(self, num_threads: int) -> bool:
        return self.manager.is_safe_to_create_threads(num_threads)


class SkipManager:
    """Backward compatibility wrapper for skip management."""
    
    def __init__(self):
        self.manager = _global_execution_manager
    
    def skip_molecule(self, molecule_id: str, reason: SkipReason, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.manager.skip_molecule(molecule_id, reason, message, details)
    
    def get_skip_summary(self) -> Dict[str, Any]:
        return self.manager.get_skip_summary()
    
    def get_skipped_by_reason(self, reason: SkipReason) -> List[SkipRecord]:
        return self.manager.get_skipped_by_reason(reason)
    
    def clear_skips(self) -> None:
        self.manager.clear_skips()
    
    def save_skip_report(self, output_path: str) -> bool:
        return self.manager.save_skip_report(output_path)
    
    @property
    def skipped_records(self) -> List[SkipRecord]:
        return self.manager.skipped_records