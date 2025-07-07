"""
Test profiling utilities for identifying performance bottlenecks.
Provides detailed profiling capabilities for test optimization.
"""

import cProfile
import io
import pstats
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytest
import psutil
import functools
import json


class TestProfiler:
    """Advanced test profiling with detailed performance analysis."""
    
    def __init__(self, output_dir: str = "test-profiles"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.profiles = {}
        self.memory_snapshots = {}
        
    def profile_test(self, test_name: str = None):
        """Decorator to profile a test function."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                name = test_name or func.__name__
                
                # Profile the test
                profiler = cProfile.Profile()
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss
                
                profiler.enable()
                try:
                    result = func(*args, **kwargs)
                finally:
                    profiler.disable()
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss
                
                # Save profile data
                self._save_profile(name, profiler, start_time, end_time, 
                                 start_memory, end_memory)
                
                return result
            return wrapper
        return decorator
    
    def _save_profile(self, test_name: str, profiler: cProfile.Profile,
                     start_time: float, end_time: float, 
                     start_memory: int, end_memory: int):
        """Save profiling data for a test."""
        # Save binary profile
        profile_file = self.output_dir / f"{test_name}.prof"
        profiler.dump_stats(str(profile_file))
        
        # Generate text report
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(50)  # Top 50 functions
        
        text_report = s.getvalue()
        
        # Save detailed analysis
        analysis = {
            'test_name': test_name,
            'total_time': end_time - start_time,
            'memory_delta': (end_memory - start_memory) / 1024 / 1024,  # MB
            'profile_file': str(profile_file),
            'top_functions': self._extract_top_functions(profiler),
            'text_report': text_report,
            'timestamp': time.time()
        }
        
        # Save JSON report
        json_file = self.output_dir / f"{test_name}.json"
        with open(json_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        self.profiles[test_name] = analysis
    
    def _extract_top_functions(self, profiler: cProfile.Profile, n: int = 10) -> List[Dict]:
        """Extract top N functions by cumulative time."""
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        top_functions = []
        for func_key, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:n]:
            filename, line_num, func_name = func_key
            top_functions.append({
                'function': func_name,
                'file': filename,
                'line': line_num,
                'calls': cc,
                'total_time': tt,
                'cumulative_time': ct,
                'per_call': ct / cc if cc > 0 else 0
            })
        
        return top_functions
    
    @contextmanager
    def profile_context(self, context_name: str):
        """Context manager for profiling code blocks."""
        profiler = cProfile.Profile()
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        profiler.enable()
        try:
            yield profiler
        finally:
            profiler.disable()
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            self._save_profile(context_name, profiler, start_time, end_time,
                             start_memory, end_memory)
    
    def compare_profiles(self, test_name1: str, test_name2: str) -> Dict:
        """Compare two test profiles."""
        if test_name1 not in self.profiles or test_name2 not in self.profiles:
            return {"error": "One or both profiles not found"}
        
        profile1 = self.profiles[test_name1]
        profile2 = self.profiles[test_name2]
        
        comparison = {
            'test1': test_name1,
            'test2': test_name2,
            'time_diff': profile2['total_time'] - profile1['total_time'],
            'memory_diff': profile2['memory_delta'] - profile1['memory_delta'],
            'time_ratio': profile2['total_time'] / profile1['total_time'] if profile1['total_time'] > 0 else 0,
            'memory_ratio': profile2['memory_delta'] / profile1['memory_delta'] if profile1['memory_delta'] > 0 else 0,
            'function_comparison': self._compare_functions(profile1, profile2)
        }
        
        return comparison
    
    def _compare_functions(self, profile1: Dict, profile2: Dict) -> List[Dict]:
        """Compare function performance between two profiles."""
        functions1 = {f['function']: f for f in profile1['top_functions']}
        functions2 = {f['function']: f for f in profile2['top_functions']}
        
        comparisons = []
        for func_name in set(functions1.keys()) | set(functions2.keys()):
            f1 = functions1.get(func_name, {})
            f2 = functions2.get(func_name, {})
            
            if f1 and f2:
                comparisons.append({
                    'function': func_name,
                    'time_diff': f2['cumulative_time'] - f1['cumulative_time'],
                    'calls_diff': f2['calls'] - f1['calls'],
                    'efficiency_change': (f2['per_call'] - f1['per_call']) / f1['per_call'] if f1['per_call'] > 0 else 0
                })
        
        return sorted(comparisons, key=lambda x: abs(x['time_diff']), reverse=True)
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report."""
        if not self.profiles:
            return {"error": "No profile data available"}
        
        total_time = sum(p['total_time'] for p in self.profiles.values())
        total_memory = sum(p['memory_delta'] for p in self.profiles.values())
        
        # Find slowest tests
        slowest_tests = sorted(
            self.profiles.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )[:10]
        
        # Find memory intensive tests
        memory_intensive = sorted(
            self.profiles.items(),
            key=lambda x: x[1]['memory_delta'],
            reverse=True
        )[:10]
        
        # Analyze common bottlenecks
        function_stats = {}
        for profile in self.profiles.values():
            for func in profile['top_functions']:
                func_key = f"{func['function']}:{func['file']}"
                if func_key not in function_stats:
                    function_stats[func_key] = {
                        'total_time': 0,
                        'total_calls': 0,
                        'occurrences': 0
                    }
                function_stats[func_key]['total_time'] += func['cumulative_time']
                function_stats[func_key]['total_calls'] += func['calls']
                function_stats[func_key]['occurrences'] += 1
        
        common_bottlenecks = sorted(
            function_stats.items(),
            key=lambda x: x[1]['total_time'],
            reverse=True
        )[:20]
        
        report = {
            'summary': {
                'total_tests_profiled': len(self.profiles),
                'total_execution_time': total_time,
                'total_memory_usage': total_memory,
                'average_test_time': total_time / len(self.profiles),
                'average_memory_per_test': total_memory / len(self.profiles)
            },
            'slowest_tests': [
                {
                    'name': name,
                    'time': data['total_time'],
                    'memory': data['memory_delta']
                }
                for name, data in slowest_tests
            ],
            'memory_intensive_tests': [
                {
                    'name': name,
                    'memory': data['memory_delta'],
                    'time': data['total_time']
                }
                for name, data in memory_intensive
            ],
            'common_bottlenecks': [
                {
                    'function': func_key,
                    'total_time': stats['total_time'],
                    'total_calls': stats['total_calls'],
                    'occurrences': stats['occurrences'],
                    'average_time_per_occurrence': stats['total_time'] / stats['occurrences']
                }
                for func_key, stats in common_bottlenecks
            ],
            'recommendations': self._generate_optimization_recommendations(
                slowest_tests, memory_intensive, common_bottlenecks
            )
        }
        
        return report
    
    def _generate_optimization_recommendations(self, slowest_tests: List, 
                                            memory_intensive: List,
                                            bottlenecks: List) -> List[str]:
        """Generate optimization recommendations based on profiling data."""
        recommendations = []
        
        if slowest_tests:
            slowest_name = slowest_tests[0][0]
            slowest_time = slowest_tests[0][1]['total_time']
            recommendations.append(
                f"Optimize '{slowest_name}' - takes {slowest_time:.2f}s"
            )
        
        if memory_intensive:
            memory_hog = memory_intensive[0][0]
            memory_usage = memory_intensive[0][1]['memory_delta']
            recommendations.append(
                f"Reduce memory usage in '{memory_hog}' - uses {memory_usage:.2f}MB"
            )
        
        if bottlenecks:
            top_bottleneck = bottlenecks[0][0]
            recommendations.append(
                f"Common bottleneck: {top_bottleneck}"
            )
        
        return recommendations


# Global profiler instance
profiler = TestProfiler()


@pytest.fixture
def test_profiler():
    """Fixture to provide test profiling capabilities."""
    return profiler


def test_profiler_functionality():
    """Test the profiler functionality itself."""
    test_profiler = TestProfiler()
    
    @test_profiler.profile_test("test_sample")
    def sample_test():
        time.sleep(0.1)
        return sum(range(1000))
    
    result = sample_test()
    assert result == 499500
    
    # Check if profile was recorded
    assert "test_sample" in test_profiler.profiles
    assert test_profiler.profiles["test_sample"]["total_time"] > 0.1


@pytest.mark.performance_critical
def test_profiler_context_manager():
    """Test the profiler context manager."""
    test_profiler = TestProfiler()
    
    with test_profiler.profile_context("context_test"):
        time.sleep(0.05)
        sum(range(500))
    
    assert "context_test" in test_profiler.profiles


if __name__ == "__main__":
    # Generate performance report for existing profiles
    test_profiler = TestProfiler()
    report = test_profiler.generate_performance_report()
    
    if "error" not in report:
        print("Performance Analysis Report:")
        print(json.dumps(report, indent=2))
    else:
        print("No profiling data available. Run profiled tests to generate data.")