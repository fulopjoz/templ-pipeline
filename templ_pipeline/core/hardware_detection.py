"""
Hardware Detection and Performance Benchmarking for TEMPL Pipeline

This module provides intelligent hardware detection, GPU/CPU benchmarking,
and optimal dependency recommendations for different deployment scenarios.
"""

import logging
import platform
import subprocess
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Use basic libraries that are always available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class HardwareInfo:
    """Hardware configuration information"""
    cpu_count: int
    cpu_model: str
    total_ram_gb: float
    gpu_available: bool
    gpu_count: int
    gpu_memory_gb: float
    gpu_models: List[str]
    recommended_config: str

def get_basic_hardware_info() -> HardwareInfo:
    """Get basic hardware information using only standard library"""
    try:
        # CPU info
        cpu_count = os.cpu_count() or 4
        cpu_model = platform.processor() or "Unknown CPU"
        
        # Memory info - fallback if psutil not available
        if PSUTIL_AVAILABLE:
            import psutil
            total_ram_gb = psutil.virtual_memory().total / (1024**3)
        else:
            # Rough estimate from /proc/meminfo on Linux
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            total_ram_gb = int(line.split()[1]) / (1024**2)
                            break
                    else:
                        total_ram_gb = 8.0  # Default fallback
            except:
                total_ram_gb = 8.0  # Default fallback
        
        # GPU detection
        gpu_available = False
        gpu_count = 0
        gpu_memory_gb = 0.0
        gpu_models = []
        
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_available = True
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 2:
                            gpu_models.append(parts[0])
                            gpu_memory_gb += float(parts[1]) / 1024  # Convert MB to GB
                gpu_count = len(gpu_models)
        except:
            pass
        
        # Recommend configuration based on hardware
        if gpu_available and total_ram_gb >= 16 and gpu_memory_gb >= 8:
            recommended_config = "gpu-large"
        elif gpu_available and total_ram_gb >= 12 and gpu_memory_gb >= 4:
            recommended_config = "gpu-medium"
        elif gpu_available and total_ram_gb >= 8:
            recommended_config = "gpu-small"
        elif total_ram_gb >= 16 and cpu_count >= 8:
            recommended_config = "cpu-optimized"
        else:
            recommended_config = "cpu-minimal"
        
        return HardwareInfo(
            cpu_count=cpu_count,
            cpu_model=cpu_model,
            total_ram_gb=total_ram_gb,
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            gpu_memory_gb=gpu_memory_gb,
            gpu_models=gpu_models,
            recommended_config=recommended_config
        )
        
    except Exception as e:
        logger.error(f"Hardware detection failed: {e}")
        # Return safe defaults
        return HardwareInfo(
            cpu_count=4,
            cpu_model="Unknown CPU",
            total_ram_gb=8.0,
            gpu_available=False,
            gpu_count=0,
            gpu_memory_gb=0.0,
            gpu_models=[],
            recommended_config="cpu-minimal"
        )

class HardwareDetector:
    """Hardware detection and recommendation system"""
    
    def __init__(self):
        self.hardware_info = None
    
    def detect_hardware(self) -> HardwareInfo:
        """Detect and cache hardware information"""
        if self.hardware_info is None:
            self.hardware_info = get_basic_hardware_info()
        return self.hardware_info

def get_hardware_recommendation() -> Dict[str, Any]:
    """Get hardware recommendation for installation"""
    detector = HardwareDetector()
    hardware_info = detector.detect_hardware()
    
    # Map config to installation command
    config_commands = {
        "cpu-minimal": "uv pip install -e .",
        "cpu-optimized": "uv pip install -e .[ai-cpu,web]",
        "gpu-small": "uv pip install -e .[ai-gpu,web]",
        "gpu-medium": "uv pip install -e .[ai-gpu,web]",
        "gpu-large": "uv pip install -e .[ai-gpu,web]"
    }
    
    config_descriptions = {
        "cpu-minimal": "Minimal CPU-only installation",
        "cpu-optimized": "CPU-optimized with embedding features",
        "gpu-small": "GPU-enabled installation (small models)",
        "gpu-medium": "GPU-enabled installation (medium models)",
        "gpu-large": "GPU-enabled installation (large models)"
    }
    
    config = hardware_info.recommended_config
    
    return {
        "hardware_info": {
            "cpu_count": hardware_info.cpu_count,
            "cpu_model": hardware_info.cpu_model,
            "total_ram_gb": hardware_info.total_ram_gb,
            "gpu_available": hardware_info.gpu_available,
            "gpu_count": hardware_info.gpu_count,
            "gpu_memory_gb": hardware_info.gpu_memory_gb,
            "gpu_models": hardware_info.gpu_models,
            "recommended_config": config
        },
        "recommended_installation": {
            "command": config_commands.get(config, "uv pip install -e ."),
            "description": config_descriptions.get(config, "Basic installation")
        }
    }

# Simplified benchmark class for compatibility
class ProteinEmbeddingBenchmark:
    """Simple benchmark placeholder"""
    
    def benchmark_cpu_vs_gpu(self, model_sizes: List[str] = None) -> Dict[str, List]:
        """Placeholder benchmark function"""
        logger.info("Benchmarking functionality requires embedding dependencies")
        return {"cpu": [], "gpu": []} 