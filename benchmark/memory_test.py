#!/usr/bin/env python3

import time
import psutil
import tempfile
from pathlib import Path
from typing import Dict, List
import argparse

from templ_pipeline.benchmark.timesplit import run_timesplit_benchmark
from templ_pipeline.benchmark.streaming import create_streaming_config


def monitor_memory_usage(process_name: str = "python") -> Dict:
    """Monitor system memory usage for processes."""
    memory_info = psutil.virtual_memory()
    python_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            if process_name in proc.info['name'].lower():
                python_processes.append({
                    'pid': proc.info['pid'],
                    'memory_mb': proc.info['memory_info'].rss / (1024 * 1024)
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    total_python_memory = sum(p['memory_mb'] for p in python_processes)
    
    return {
        'system_memory_percent': memory_info.percent,
        'system_available_gb': memory_info.available / (1024**3),
        'python_processes': len(python_processes),
        'total_python_memory_mb': total_python_memory,
        'python_processes_details': python_processes
    }


def test_memory_streaming():
    """Test memory usage with streaming enabled vs disabled."""
    print("Testing TEMPL Pipeline Memory Usage")
    print("=" * 50)
    
    # Test parameters
    test_params = {
        'n_workers': 4,
        'n_conformers': 50,  # Reduced for testing
        'template_knn': 10,  # Reduced for testing
        'max_pdbs': 5,       # Very small test set
        'splits_to_run': ['val'],  # Test only val split
        'quiet': True
    }
    
    # Test 1: Traditional in-memory approach
    print("\nTest 1: Traditional in-memory processing")
    print("-" * 40)
    
    initial_memory = monitor_memory_usage()
    print(f"Initial memory: {initial_memory['total_python_memory_mb']:.1f}MB")
    
    start_time = time.time()
    results_memory = run_timesplit_benchmark(**test_params)
    end_time = time.time()
    
    final_memory = monitor_memory_usage()
    print(f"Final memory: {final_memory['total_python_memory_mb']:.1f}MB")
    print(f"Memory increase: {final_memory['total_python_memory_mb'] - initial_memory['total_python_memory_mb']:.1f}MB")
    print(f"Runtime: {end_time - start_time:.1f}s")
    
    # Test 2: Streaming approach
    print("\nTest 2: Streaming processing")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        streaming_params = test_params.copy()
        streaming_params['streaming_output_dir'] = temp_dir
        
        initial_memory = monitor_memory_usage()
        print(f"Initial memory: {initial_memory['total_python_memory_mb']:.1f}MB")
        
        start_time = time.time()
        results_streaming = run_timesplit_benchmark(**streaming_params)
        end_time = time.time()
        
        final_memory = monitor_memory_usage()
        print(f"Final memory: {final_memory['total_python_memory_mb']:.1f}MB")
        print(f"Memory increase: {final_memory['total_python_memory_mb'] - initial_memory['total_python_memory_mb']:.1f}MB")
        print(f"Runtime: {end_time - start_time:.1f}s")
        
        # Check streaming files
        streaming_files = list(Path(temp_dir).rglob("*.jsonl"))
        print(f"Streaming files created: {len(streaming_files)}")
        
        for file in streaming_files:
            file_size = file.stat().st_size
            print(f"  {file.name}: {file_size} bytes")
    
    # Compare results
    print("\nResults Comparison")
    print("-" * 20)
    
    def extract_metrics(results):
        metrics = {}
        for split_name, split_data in results.items():
            if split_name != 'params':
                split_metrics = split_data.get('metrics', {})
                metrics[split_name] = {
                    'total': split_metrics.get('total', 0),
                    'successful': split_metrics.get('successful', 0),
                    'success_rate': split_metrics.get('success_rate', 0.0)
                }
        return metrics
    
    memory_metrics = extract_metrics(results_memory)
    streaming_metrics = extract_metrics(results_streaming)
    
    print("Memory approach:")
    for split, metrics in memory_metrics.items():
        print(f"  {split}: {metrics['successful']}/{metrics['total']} ({metrics['success_rate']:.1f}%)")
    
    print("Streaming approach:")
    for split, metrics in streaming_metrics.items():
        print(f"  {split}: {metrics['successful']}/{metrics['total']} ({metrics['success_rate']:.1f}%)")
    
    # Verify results match
    results_match = memory_metrics == streaming_metrics
    print(f"\nResults match: {results_match}")
    
    return results_match


def continuous_memory_monitor(duration_minutes: int = 5):
    """Monitor memory usage continuously during benchmark execution."""
    print(f"Monitoring memory usage for {duration_minutes} minutes")
    print("Time\tSystem%\tPython_MB\tProcesses")
    print("-" * 40)
    
    start_time = time.time()
    max_memory = 0
    
    while time.time() - start_time < duration_minutes * 60:
        memory_info = monitor_memory_usage()
        current_time = time.strftime("%H:%M:%S")
        
        python_memory = memory_info['total_python_memory_mb']
        max_memory = max(max_memory, python_memory)
        
        print(f"{current_time}\t{memory_info['system_memory_percent']:.1f}%\t{python_memory:.1f}\t{memory_info['python_processes']}")
        
        time.sleep(10)  # Check every 10 seconds
    
    print(f"\nMax Python memory observed: {max_memory:.1f}MB")
    return max_memory


def main():
    parser = argparse.ArgumentParser(description="Test TEMPL pipeline memory usage")
    parser.add_argument("--test-streaming", action="store_true", help="Test streaming vs memory approaches")
    parser.add_argument("--monitor", type=int, help="Monitor memory for N minutes")
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark with memory monitoring")
    
    args = parser.parse_args()
    
    if args.test_streaming:
        success = test_memory_streaming()
        if success:
            print("\nMemory streaming test PASSED")
        else:
            print("\nMemory streaming test FAILED")
            return 1
    
    elif args.monitor:
        continuous_memory_monitor(args.monitor)
    
    elif args.benchmark:
        print("Running benchmark with memory monitoring...")
        with tempfile.TemporaryDirectory() as temp_dir:
            # Start memory monitoring in background
            import threading
            monitor_thread = threading.Thread(
                target=continuous_memory_monitor,
                args=(10,)  # Monitor for 10 minutes
            )
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # Run benchmark with streaming
            results = run_timesplit_benchmark(
                n_workers=4,
                n_conformers=100,
                template_knn=50,
                max_pdbs=10,
                streaming_output_dir=temp_dir,
                quiet=False
            )
            
            print("Benchmark completed successfully")
    
    else:
        print("Please specify --test-streaming, --monitor N, or --benchmark")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 