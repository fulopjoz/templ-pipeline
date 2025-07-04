"""
Hardware Manager for TEMPL Pipeline

Optimized hardware detection with caching to prevent redundant calls.
"""

import streamlit as st
import logging
import multiprocessing
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Hardware information container"""

    cpu_count: int
    total_ram_gb: float
    gpu_available: bool
    gpu_models: list
    gpu_memory_gb: float
    recommended_config: str
    max_workers: int
    device_type: str  # 'cpu' or 'cuda'


# Standalone cached functions to avoid hashing issues with instance methods
@st.cache_resource
def _cached_hardware_detection() -> HardwareInfo:
    """Cached hardware detection function

    Returns:
        HardwareInfo object with detected capabilities
    """
    logger.info("Starting hardware detection...")

    # CPU detection
    cpu_count = multiprocessing.cpu_count()

    # RAM detection
    try:
        import psutil

        total_ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        total_ram_gb = 8.0  # Default assumption
        logger.warning("psutil not available, assuming 8GB RAM")

    # GPU detection
    gpu_available = False
    gpu_models = []
    gpu_memory_gb = 0.0
    device_type = "cpu"

    try:
        # Use safer PyTorch import to avoid Streamlit file watcher conflicts
        import sys

        if "torch" not in sys.modules:
            import torch
        else:
            torch = sys.modules["torch"]

        if torch.cuda.is_available():
            gpu_available = True
            device_type = "cuda"
            device_count = torch.cuda.device_count()

            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_models.append(gpu_name)

                # Get GPU memory
                props = torch.cuda.get_device_properties(i)
                gpu_memory_gb += props.total_memory / (1024**3)

            logger.info(f"Detected {device_count} GPU(s): {', '.join(gpu_models)}")
        else:
            logger.info("No CUDA-capable GPU detected")
    except ImportError:
        logger.info("PyTorch not available for GPU detection")
    except Exception as e:
        logger.warning(f"GPU detection error: {e}")

    # Determine recommended configuration
    if gpu_available:
        if gpu_memory_gb >= 16:
            recommended_config = "gpu-large"
        elif gpu_memory_gb >= 8:
            recommended_config = "gpu-medium"
        else:
            recommended_config = "gpu-small"
    else:
        if cpu_count >= 8 and total_ram_gb >= 16:
            recommended_config = "cpu-optimized"
        else:
            recommended_config = "cpu-minimal"

    # Calculate optimal workers
    if gpu_available:
        # GPU can handle more concurrent operations
        base_workers = min(cpu_count, 16)  # Increased from 8 to 16
    else:
        # CPU-only: Use more cores but leave some for system
        if cpu_count >= 16:
            base_workers = min(
                cpu_count - 2, 16
            )  # Use up to 16 workers, reserve 2 cores
        elif cpu_count >= 8:
            base_workers = min(
                cpu_count - 1, 12
            )  # Use most cores for 8-16 core systems
        else:
            base_workers = min(
                cpu_count - 1, 4
            )  # Original conservative approach for <8 cores

    # Adjust based on RAM
    if total_ram_gb < 8:
        max_workers = min(base_workers, 2)
    elif total_ram_gb < 16:
        max_workers = min(base_workers, 6)  # Increased from 4
    elif total_ram_gb < 32:
        max_workers = min(base_workers, 12)  # New tier for 16-32GB
    else:
        max_workers = base_workers  # Use full base_workers for 32GB+

    max_workers = max(1, max_workers)  # At least 1 worker

    # Log GPU detection issue if GPUs available but not detected
    if not gpu_available:
        try:
            import subprocess

            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.warning(
                    "NVIDIA GPUs detected by nvidia-smi but not by PyTorch. "
                    "Install PyTorch with CUDA support: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
                )
        except:
            pass

    # Create hardware info
    hardware_info = HardwareInfo(
        cpu_count=cpu_count,
        total_ram_gb=total_ram_gb,
        gpu_available=gpu_available,
        gpu_models=gpu_models,
        gpu_memory_gb=gpu_memory_gb,
        recommended_config=recommended_config,
        max_workers=max_workers,
        device_type=device_type,
    )

    logger.info(f"Hardware detection complete: {recommended_config}")

    return hardware_info


@st.cache_resource
def _cached_embedding_capabilities_check() -> Dict[str, bool]:
    """Cached check for embedding capabilities

    Returns:
        Dictionary with capability flags
    """
    capabilities = {
        "torch_available": False,
        "transformers_available": False,
        "embedding_available": False,
    }

    # Check PyTorch
    try:
        import torch

        capabilities["torch_available"] = True
        logger.info("PyTorch available")
    except ImportError:
        logger.warning("PyTorch not available")

    # Check Transformers
    try:
        import transformers

        capabilities["transformers_available"] = True
        logger.info("Transformers available")
    except ImportError:
        logger.warning("Transformers not available")

    # Overall embedding capability
    capabilities["embedding_available"] = (
        capabilities["torch_available"] and capabilities["transformers_available"]
    )

    if capabilities["embedding_available"]:
        logger.info("Protein embedding similarity features available")
    else:
        logger.warning("Embedding features disabled - missing dependencies")

    return capabilities


class HardwareManager:
    """Manages hardware detection and configuration with caching"""

    def __init__(self):
        """Initialize hardware manager"""
        self._hardware_info: Optional[HardwareInfo] = None
        self._embedding_capabilities: Optional[Dict[str, bool]] = None

    def detect_hardware(self) -> HardwareInfo:
        """Detect hardware capabilities with caching

        Returns:
            HardwareInfo object with detected capabilities
        """
        # Use the cached standalone function
        if self._hardware_info is None:
            self._hardware_info = _cached_hardware_detection()
        return self._hardware_info

    def check_embedding_capabilities(self) -> Dict[str, bool]:
        """Check embedding-related capabilities with caching

        Returns:
            Dictionary with capability flags
        """
        # Use the cached standalone function
        if self._embedding_capabilities is None:
            self._embedding_capabilities = _cached_embedding_capabilities_check()
        return self._embedding_capabilities

    def get_hardware_info(self) -> HardwareInfo:
        """Get cached hardware information

        Returns:
            HardwareInfo object
        """
        return self.detect_hardware()

    def get_device(self) -> str:
        """Get recommended compute device

        Returns:
            'cuda' or 'cpu'
        """
        info = self.get_hardware_info()
        return info.device_type

    def get_worker_config(self) -> Dict[str, Any]:
        """Get worker configuration for pipeline

        Returns:
            Worker configuration dictionary
        """
        info = self.get_hardware_info()

        return {
            "n_workers": info.max_workers,
            "device": info.device_type,
            "batch_size": self._get_batch_size(info),
            "enable_gpu": info.gpu_available,
            "memory_limit_mb": int(info.total_ram_gb * 1024 * 0.7),  # Use 70% of RAM
        }

    def _get_batch_size(self, info: HardwareInfo) -> int:
        """Determine optimal batch size based on hardware

        Returns:
            Batch size
        """
        if info.gpu_available:
            if info.gpu_memory_gb >= 16:
                return 64
            elif info.gpu_memory_gb >= 8:
                return 32
            else:
                return 16
        else:
            if info.total_ram_gb >= 16:
                return 16
            else:
                return 8

    def get_status_summary(self) -> Dict[str, Any]:
        """Get hardware status summary

        Returns:
            Status dictionary
        """
        info = self.get_hardware_info()
        capabilities = self.check_embedding_capabilities()

        return {
            "hardware": {
                "cpu_cores": info.cpu_count,
                "ram_gb": f"{info.total_ram_gb:.1f}",
                "gpu": "Available" if info.gpu_available else "Not available",
                "gpu_memory_gb": (
                    f"{info.gpu_memory_gb:.1f}" if info.gpu_available else "N/A"
                ),
                "config": info.recommended_config,
            },
            "capabilities": capabilities,
            "performance": {
                "max_workers": info.max_workers,
                "device": info.device_type,
                "batch_size": self._get_batch_size(info),
            },
        }


# Global hardware manager instance
_hardware_manager = None


def get_hardware_manager() -> HardwareManager:
    """Get global hardware manager instance

    Returns:
        HardwareManager instance
    """
    global _hardware_manager
    if _hardware_manager is None:
        _hardware_manager = HardwareManager()
    return _hardware_manager
