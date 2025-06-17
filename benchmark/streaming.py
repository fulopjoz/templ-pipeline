import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Iterator
from dataclasses import dataclass
import threading
import gc
import psutil


@dataclass
class StreamingConfig:
    """Configuration for streaming benchmark results."""
    output_dir: Path
    batch_size: int = 10
    memory_threshold_gb: float = 8.0
    auto_cleanup: bool = True


class StreamingResultWriter:
    """Writes benchmark results incrementally to disk to prevent memory accumulation."""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.config.output_dir / "results.jsonl"
        self.summary_file = self.config.output_dir / "summary.json"
        self.lock = threading.Lock()
        self.processed_count = 0
        self.successful_count = 0
        self.failed_count = 0
        self.start_time = time.time()
        
    def write_result(self, pdb_id: str, result: Dict[str, Any]) -> None:
        """Write single result immediately to disk."""
        result_entry = {
            "pdb_id": pdb_id,
            "timestamp": time.time(),
            "result": result
        }
        
        with self.lock:
            with open(self.results_file, 'a') as f:
                f.write(json.dumps(result_entry) + '\n')
            
            self.processed_count += 1
            if result.get("success", False):
                self.successful_count += 1
            else:
                self.failed_count += 1
                
            if self.config.auto_cleanup and self.processed_count % self.config.batch_size == 0:
                self._cleanup_memory()
                
    def _cleanup_memory(self) -> None:
        """Force garbage collection and memory cleanup."""
        gc.collect()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process()
        return process.memory_info().rss / (1024**3)
        
    def check_memory_pressure(self) -> bool:
        """Check if memory usage exceeds threshold."""
        return self.get_memory_usage() > self.config.memory_threshold_gb
        
    def write_summary(self) -> None:
        """Write final summary statistics."""
        runtime = time.time() - self.start_time
        summary = {
            "total_processed": self.processed_count,
            "successful": self.successful_count,
            "failed": self.failed_count,
            "success_rate": (self.successful_count / self.processed_count * 100) if self.processed_count > 0 else 0.0,
            "runtime_seconds": runtime,
            "timestamp": time.time()
        }
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
    def load_results(self) -> Iterator[Dict[str, Any]]:
        """Generator to load results from disk incrementally."""
        if not self.results_file.exists():
            return
            
        with open(self.results_file, 'r') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
                    
    def load_summary(self) -> Optional[Dict[str, Any]]:
        """Load summary statistics."""
        if self.summary_file.exists():
            with open(self.summary_file, 'r') as f:
                return json.load(f)
        return None


class MemoryManagedProcessor:
    """Base class for memory-managed processing with automatic cleanup."""
    
    def __init__(self, memory_limit_gb: float = 4.0):
        self.memory_limit_gb = memory_limit_gb
        self._cleanup_counter = 0
        
    def process_with_memory_management(self, process_func, *args, **kwargs):
        """Execute function with automatic memory management."""
        try:
            result = process_func(*args, **kwargs)
            self._cleanup_counter += 1
            
            if self._cleanup_counter % 5 == 0:
                self._force_cleanup()
                
            if self._check_memory_pressure():
                self._aggressive_cleanup()
                
            return result
        finally:
            self._cleanup_locals(locals())
            
    def _force_cleanup(self):
        """Force garbage collection."""
        gc.collect()
        
    def _aggressive_cleanup(self):
        """Aggressive memory cleanup under pressure."""
        for _ in range(3):
            gc.collect()
            
    def _check_memory_pressure(self) -> bool:
        """Check if memory usage exceeds limit."""
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024**3)
        return memory_gb > self.memory_limit_gb
        
    def _cleanup_locals(self, local_vars: Dict):
        """Clean up local variables to prevent reference cycles."""
        for name, obj in local_vars.items():
            if name not in ['self', 'result'] and hasattr(obj, '__del__'):
                try:
                    del obj
                except:
                    pass


def create_streaming_config(output_dir: str, memory_threshold_gb: float = 8.0) -> StreamingConfig:
    """Create streaming configuration with defaults."""
    return StreamingConfig(
        output_dir=Path(output_dir),
        memory_threshold_gb=memory_threshold_gb,
        auto_cleanup=True
    ) 