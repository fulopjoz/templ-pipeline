# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Hardware detection and optimization utilities for TEMPL pipeline.

This module provides comprehensive hardware detection, performance benchmarking,
and optimal configuration recommendations for different deployment scenarios.
"""

import logging
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Optional dependencies
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Hardware configuration information."""

    cpu_count: int
    cpu_model: str
    total_ram_gb: float
    gpu_available: bool
    gpu_count: int
    gpu_memory_gb: float
    gpu_models: List[str]
    recommended_config: str


def get_basic_hardware_info() -> HardwareInfo:
    """Get basic hardware information using standard library and psutil."""
    try:
        # CPU info
        cpu_count = os.cpu_count() or 4
        cpu_model = platform.processor() or "Unknown CPU"

        # Memory info
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            total_ram_gb = mem.total / (1024**3)
        else:
            # Fallback memory detection
            total_ram_gb = 8.0  # Conservative fallback
            try:
                # Try to get memory info from /proc/meminfo on Linux
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total_ram_gb = int(line.split()[1]) / (1024**2)
                            break
            except (FileNotFoundError, ValueError, IOError):
                pass

        # GPU detection
        gpu_available = False
        gpu_count = 0
        gpu_memory_gb = 0.0
        gpu_models = []

        # Try to detect NVIDIA GPUs
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpu_available = True
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = line.split(", ")
                        if len(parts) >= 2:
                            gpu_models.append(parts[0])
                            gpu_memory_gb += float(parts[1]) / 1024  # Convert MB to GB
                gpu_count = len(gpu_models)
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ):
            pass

        # Determine recommended config
        if gpu_available and gpu_memory_gb > 4.0:
            recommended_config = "gpu"
        elif total_ram_gb > 16.0 and cpu_count >= 8:
            recommended_config = "high_performance"
        elif total_ram_gb > 8.0 and cpu_count >= 4:
            recommended_config = "balanced"
        else:
            recommended_config = "conservative"

        return HardwareInfo(
            cpu_count=cpu_count,
            cpu_model=cpu_model,
            total_ram_gb=total_ram_gb,
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            gpu_models=gpu_models,
            recommended_config=recommended_config,
        )

    except Exception as e:
        logger.warning(f"Hardware detection failed: {e}")
        return HardwareInfo(
            cpu_count=4,
            cpu_model="Unknown",
            total_ram_gb=8.0,
            gpu_available=False,
            gpu_count=0,
            gpu_memory_gb=0.0,
            gpu_models=[],
            recommended_config="conservative",
        )


def get_memory_info() -> Dict[str, float]:
    """Get detailed system memory information."""
    try:
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            return {
                "total_gb": mem.total / (1024**3),
                "available_gb": mem.available / (1024**3),
                "percent_used": mem.percent,
                "free_gb": mem.free / (1024**3),
                "cached_gb": getattr(mem, "cached", 0) / (1024**3),
            }
        else:
            # Fallback for systems without psutil
            return {
                "total_gb": 8.0,
                "available_gb": 4.0,
                "percent_used": 50.0,
                "free_gb": 4.0,
                "cached_gb": 0.0,
            }
    except Exception as e:
        logger.warning(f"Memory info collection failed: {e}")
        return {
            "total_gb": 8.0,
            "available_gb": 4.0,
            "percent_used": 50.0,
            "free_gb": 4.0,
            "cached_gb": 0.0,
        }


def get_optimized_worker_config(
    workload_type: str = "balanced",
    dataset_size: int = 0,
    hardware_info: Optional[HardwareInfo] = None,
) -> Dict[str, Any]:
    """
    Return optimized worker configuration based on workload type and dataset size.

    Args:
        workload_type: "balanced", "cpu_intensive", "io_intensive", "memory_intensive"
        dataset_size: Number of items to process (for scaling decisions)
        hardware_info: Optional hardware info to use for optimization

    Returns:
        Dictionary containing optimized configuration
    """
    if hardware_info is None:
        hardware_info = get_basic_hardware_info()

    total_cpus = hardware_info.cpu_count  # Use all available CPUs
    memory_info = get_memory_info()

    # Base configuration based on workload type - CONSERVATIVE FOR SYSTEM STABILITY
    if workload_type == "cpu_intensive":
        # CRITICAL FIX: Use conservative worker count to prevent resource exhaustion
        # Cap at 20 workers maximum to prevent "cannot allocate memory for thread-local data" errors
        n_workers = max(
            2, min(20, int(total_cpus * 0.75))
        )  # Use 75% of CPUs, max 20 workers
        internal_pipeline_workers = (
            min(2, total_cpus // n_workers) if n_workers > 1 else 1
        )

    elif workload_type == "io_intensive":
        # Conservative parallel targets for I/O-bound tasks with system stability
        n_workers = min(total_cpus, 20)  # Cap at 20 workers for system stability
        internal_pipeline_workers = 1

    elif workload_type == "memory_intensive":
        # Conservative allocation for molecular tasks - use 2GB per worker for stability
        # This prevents system memory exhaustion seen with 4GB per worker
        max_workers_by_memory = max(
            1, int(memory_info["available_gb"] * 0.7 // 2)
        )  # 70% memory, 2GB per worker
        n_workers = min(total_cpus, max_workers_by_memory, 20)  # Cap at 20 workers
        internal_pipeline_workers = 1

    else:  # balanced
        # Default: conservative parallelization for system stability
        n_workers = min(total_cpus, 20)  # Cap at 20 workers
        internal_pipeline_workers = 1

    # Adjust for large datasets - conservative approach
    if dataset_size > 1000:
        # For very large datasets, maintain stability rather than maximizing throughput
        n_workers = min(n_workers, 20)  # Keep 20 worker cap even for large datasets
        internal_pipeline_workers = 1

    # Apply hardware-specific optimizations - SYSTEM STABILITY FIRST
    if hardware_info.gpu_available and hardware_info.gpu_memory_gb > 4.0:
        # GPU available - modest boost but maintain system stability
        n_workers = min(n_workers + 2, 20)  # Small boost, keep 20 worker cap
        internal_pipeline_workers = min(
            internal_pipeline_workers + 1, 2
        )  # Modest internal worker boost
    elif hardware_info.total_ram_gb < 8.0:
        # Limited RAM - be more conservative
        n_workers = min(
            n_workers, max(4, total_cpus // 2)
        )  # Use at least half CPUs, minimum 4 workers

    config = {
        "n_workers": n_workers,
        "internal_pipeline_workers": internal_pipeline_workers,
        "workload_type": workload_type,
        "total_cpus": total_cpus,
        "memory_gb": memory_info["available_gb"],
        "hardware_profile": hardware_info.recommended_config,
        "gpu_available": hardware_info.gpu_available,
    }

    logger.debug(
        f"Optimized config ({workload_type}) → n_workers={n_workers}, "
        f"internal_workers={internal_pipeline_workers}, CPUs={total_cpus}, "
        f"RAM={memory_info['available_gb']:.1f}GB"
    )

    return config


def get_suggested_worker_config() -> Dict[str, Any]:
    """
    Get suggested worker configuration for backward compatibility.

    Returns:
        Dictionary containing balanced worker configuration
    """
    return get_optimized_worker_config("balanced")


def get_hardware_info() -> Dict[str, Any]:
    """
    Get comprehensive hardware information for backward compatibility.

    Returns:
        Dictionary containing hardware information
    """
    hardware_info = get_basic_hardware_info()
    config = get_suggested_worker_config()
    memory_info = get_memory_info()

    return {
        "cpu_count": hardware_info.cpu_count,
        "cpu_model": hardware_info.cpu_model,
        "total_ram_gb": hardware_info.total_ram_gb,
        "available_ram_gb": memory_info["available_gb"],
        "gpu_available": hardware_info.gpu_available,
        "gpu_count": hardware_info.gpu_count,
        "gpu_memory_gb": hardware_info.gpu_memory_gb,
        "gpu_models": hardware_info.gpu_models,
        "recommended_config": hardware_info.recommended_config,
        "suggested_workers": config["n_workers"],
        "suggested_internal_workers": config["internal_pipeline_workers"],
    }


def benchmark_cpu_performance(duration: float = 1.0) -> float:
    """
    Simple CPU benchmark to estimate relative performance.

    Args:
        duration: Benchmark duration in seconds

    Returns:
        Performance score (higher is better)
    """
    import math

    start_time = time.time()
    operations = 0

    while time.time() - start_time < duration:
        # Simple mathematical operations
        for _ in range(10000):
            math.sqrt(operations * 2.5)
            operations += 1

    elapsed = time.time() - start_time
    score = operations / elapsed

    logger.debug(
        f"CPU benchmark: {operations} operations in {elapsed:.2f}s (score: {score:.0f})"
    )
    return score


def detect_optimal_configuration(
    target_workload: str = "pipeline", dataset_size: int = 0
) -> Dict[str, Any]:
    """
    Detect optimal configuration for a specific workload.

    Args:
        target_workload: Type of workload ("pipeline", "embedding", "benchmark")
        dataset_size: Expected dataset size

    Returns:
        Optimal configuration dictionary
    """
    hardware_info = get_basic_hardware_info()

    # Map workload types to configuration types
    workload_mapping = {
        "pipeline": "balanced",
        "embedding": (
            "gpu_intensive" if hardware_info.gpu_available else "cpu_intensive"
        ),
        "benchmark": "memory_intensive",
        "conformer_generation": "cpu_intensive",
        "template_search": "io_intensive",
    }

    workload_type = workload_mapping.get(target_workload, "balanced")

    config = get_optimized_worker_config(
        workload_type=workload_type,
        dataset_size=dataset_size,
        hardware_info=hardware_info,
    )

    # Add workload-specific optimizations
    if target_workload == "embedding" and hardware_info.gpu_available:
        config["use_gpu"] = True
        config["gpu_batch_size"] = min(32, int(hardware_info.gpu_memory_gb * 2))
    elif target_workload == "conformer_generation":
        config["max_conformers"] = min(100, int(hardware_info.total_ram_gb * 10))
    elif target_workload == "benchmark":
        config["max_concurrent_pdbs"] = min(config["n_workers"], 5)

    config["target_workload"] = target_workload
    config["hardware_summary"] = {
        "cpu_count": hardware_info.cpu_count,
        "total_ram_gb": hardware_info.total_ram_gb,
        "gpu_available": hardware_info.gpu_available,
        "recommended_profile": hardware_info.recommended_config,
    }

    return config


def get_environment_constraints() -> Dict[str, Any]:
    """
    Get environment constraints that might affect performance.

    Returns:
        Dictionary of environment constraints
    """
    constraints = {
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "architecture": platform.machine(),
        "in_docker": os.path.exists("/.dockerenv"),
        "has_nvidia_docker": False,
        "memory_limited": False,
        "cpu_limited": False,
    }

    # Check for NVIDIA Docker
    try:
        with open("/proc/1/cgroup", "r") as f:
            if "docker" in f.read():
                constraints["in_docker"] = True
                # Check for nvidia-docker
                if os.path.exists("/usr/local/nvidia"):
                    constraints["has_nvidia_docker"] = True
    except (FileNotFoundError, PermissionError):
        pass

    # Check for memory constraints
    memory_info = get_memory_info()
    if memory_info["total_gb"] < 4.0:
        constraints["memory_limited"] = True

    # Check for CPU constraints
    cpu_count = os.cpu_count()
    if cpu_count is not None and cpu_count < 4:
        constraints["cpu_limited"] = True

    return constraints


def recommend_installation_type() -> str:
    """
    Recommend installation type based on hardware capabilities.

    Returns:
        Recommended installation type
    """
    hardware_info = get_basic_hardware_info()
    constraints = get_environment_constraints()

    if constraints["memory_limited"] or constraints["cpu_limited"]:
        return "minimal"
    elif hardware_info.gpu_available and hardware_info.gpu_memory_gb > 4.0:
        return "gpu"
    elif hardware_info.total_ram_gb > 16.0 and hardware_info.cpu_count >= 8:
        return "full"
    else:
        return "standard"


def log_hardware_summary() -> None:
    """Log a summary of detected hardware for debugging."""
    hardware_info = get_basic_hardware_info()
    config = get_suggested_worker_config()
    constraints = get_environment_constraints()

    logger.info("Hardware Summary:")
    logger.info(f"  CPU: {hardware_info.cpu_model} ({hardware_info.cpu_count} cores)")
    logger.info(f"  RAM: {hardware_info.total_ram_gb:.1f} GB")
    logger.info(
        f"  GPU: {'Available' if hardware_info.gpu_available else 'Not available'}"
    )
    if hardware_info.gpu_available:
        logger.info(f"    Models: {', '.join(hardware_info.gpu_models)}")
        logger.info(f"    Total Memory: {hardware_info.gpu_memory_gb:.1f} GB")

    logger.info(f"  Recommended Profile: {hardware_info.recommended_config}")
    logger.info(f"  Suggested Workers: {config['n_workers']}")
    logger.info(
        f"  Platform: {constraints['platform']} ({constraints['architecture']})"
    )
    logger.info(f"  Installation Type: {recommend_installation_type()}")
