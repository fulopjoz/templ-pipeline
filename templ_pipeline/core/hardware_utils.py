import os
import logging

log = logging.getLogger(__name__)


def get_memory_info() -> dict:
    """Get system memory information."""
    try:
        import psutil

        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "percent_used": mem.percent,
        }
    except ImportError:
        return {"total_gb": 0, "available_gb": 0, "percent_used": 0}


def get_optimized_worker_config(
    workload_type: str = "balanced", dataset_size: int = 0
) -> dict:
    """Return optimized worker configuration based on workload type and dataset size.

    Args:
        workload_type: "balanced", "cpu_intensive", "io_intensive", "memory_intensive"
        dataset_size: Number of items to process (for scaling decisions)
    """
    total_cpus = (os.cpu_count() - 2) or 1
    memory_info = get_memory_info()

    # Base configuration
    if workload_type == "cpu_intensive":
        # Fewer parallel targets, more cores per target for heavy computation
        n_workers = max(1, total_cpus // 2)
        internal_pipeline_workers = min(4, total_cpus // n_workers)

    elif workload_type == "io_intensive":
        # More parallel targets, single core per target, cap to prevent I/O thrashing
        n_workers = min(total_cpus, 16)
        internal_pipeline_workers = 1

    elif workload_type == "memory_intensive":
        # Very conservative allocation for molecular tasks
        # Use 8GB per worker for molecular processing
        max_workers_by_memory = max(1, int(memory_info["available_gb"] * 0.8 // 8))
        n_workers = min(total_cpus, max_workers_by_memory, 3)  # Cap at 3
        internal_pipeline_workers = 1

    else:  # balanced
        # Default: maximize parallelization for most benchmarks
        n_workers = total_cpus
        internal_pipeline_workers = 1

    # Adjust for large datasets
    if dataset_size > 1000:
        # For very large datasets, prioritize throughput
        n_workers = min(n_workers * 2, total_cpus)
        internal_pipeline_workers = 1

    config = {
        "n_workers": n_workers,
        "internal_pipeline_workers": internal_pipeline_workers,
        "workload_type": workload_type,
        "total_cpus": total_cpus,
        "memory_gb": memory_info["available_gb"],
    }

    log.info(
        f"Optimized config ({workload_type}) → n_workers={n_workers}, "
        f"internal_workers={internal_pipeline_workers}, CPUs={total_cpus}, "
        f"RAM={memory_info['available_gb']:.1f}GB"
    )

    return config


def get_suggested_worker_config() -> dict:
    """Backward compatibility: Return balanced worker configuration."""
    return get_optimized_worker_config("balanced")


def get_hardware_info() -> dict:
    """Return basic hardware info using the simplified worker config."""
    total_cpus = os.cpu_count() or 1
    config = get_suggested_worker_config()

    try:
        import psutil  # Optional dependency

        mem = psutil.virtual_memory()
        mem_total_gb = mem.total / (1024**3)
        mem_free_gb = mem.available / (1024**3)
    except Exception:
        mem_total_gb = "Unknown"
        mem_free_gb = "Unknown"

    return {
        "total_cpus": total_cpus,
        "suggested_workers": config["n_workers"],
        "total_memory_gb": (
            f"{mem_total_gb:.1f}" if isinstance(mem_total_gb, float) else mem_total_gb
        ),
        "available_memory_gb": (
            f"{mem_free_gb:.1f}" if isinstance(mem_free_gb, float) else mem_free_gb
        ),
        "utilization": f"{config['n_workers']}/{total_cpus}",
    }


def get_molecular_worker_config(dataset_size: int = 0) -> dict:
    """Return optimized worker configuration for memory-intensive molecular tasks.

    Uses conservative 8GB per worker to prevent OOM with molecular datasets.
    """
    memory_info = get_memory_info()
    available_gb = memory_info["available_gb"]

    # Conservative: 8GB per worker for molecular processing
    max_workers_by_memory = max(1, int(available_gb * 0.8 // 8))
    n_workers = min(max_workers_by_memory, 3)  # Cap at 3 for stability

    config = {
        "n_workers": n_workers,
        "internal_pipeline_workers": 1,  # Force single pipeline worker
        "workload_type": "molecular",
        "estimated_memory_gb": n_workers * 8,
        "available_memory_gb": available_gb,
        "safe": n_workers * 8 <= available_gb * 0.8,
    }

    log.info(
        f"Molecular config → n_workers={n_workers}, "
        f"estimated_memory={n_workers * 8}GB, available={available_gb:.1f}GB"
    )

    return config
