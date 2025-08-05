"""
Test timing analysis and performance optimization utilities.
Provides tools for identifying slow tests and optimizing test execution.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pytest
import psutil
import statistics
from collections import defaultdict


class TimingAnalyzer:
    """Analyzes test execution times and identifies performance bottlenecks."""
    
    def __init__(self, results_file: str = "test-timing-results.json"):
        self.results_file = Path(results_file)
        self.timing_data = {}
        self.memory_data = {}
        self.slow_test_threshold = 10.0  # seconds
        self.medium_test_threshold = 3.0  # seconds
        
    def record_test_timing(self, test_name: str, duration: float, memory_usage: float):
        """Record timing and memory usage for a test."""
        self.timing_data[test_name] = {
            'duration': duration,
            'memory_usage': memory_usage,
            'timestamp': time.time()
        }
    
    def analyze_slow_tests(self) -> Dict[str, List[str]]:
        """Identify slow tests and categorize them."""
        slow_tests = []
        medium_tests = []
        fast_tests = []
        
        for test_name, data in self.timing_data.items():
            duration = data['duration']
            if duration > self.slow_test_threshold:
                slow_tests.append((test_name, duration))
            elif duration > self.medium_test_threshold:
                medium_tests.append((test_name, duration))
            else:
                fast_tests.append((test_name, duration))
        
        return {
            'slow': sorted(slow_tests, key=lambda x: x[1], reverse=True),
            'medium': sorted(medium_tests, key=lambda x: x[1], reverse=True),
            'fast': sorted(fast_tests, key=lambda x: x[1], reverse=True)
        }
    
    def generate_timing_report(self) -> Dict:
        """Generate comprehensive timing analysis report."""
        if not self.timing_data:
            return {"error": "No timing data available"}
        
        durations = [data['duration'] for data in self.timing_data.values()]
        memory_usage = [data['memory_usage'] for data in self.timing_data.values()]
        
        categorized = self.analyze_slow_tests()
        
        report = {
            'summary': {
                'total_tests': len(self.timing_data),
                'total_duration': sum(durations),
                'average_duration': statistics.mean(durations),
                'median_duration': statistics.median(durations),
                'max_duration': max(durations),
                'min_duration': min(durations),
                'std_dev': statistics.stdev(durations) if len(durations) > 1 else 0,
                'total_memory': sum(memory_usage),
                'average_memory': statistics.mean(memory_usage),
                'peak_memory': max(memory_usage)
            },
            'categorized_tests': categorized,
            'recommendations': self._generate_recommendations(categorized),
            'detailed_data': self.timing_data
        }
        
        return report
    
    def _generate_recommendations(self, categorized: Dict) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        slow_count = len(categorized['slow'])
        if slow_count > 0:
            recommendations.append(
                f"Consider optimizing {slow_count} slow tests (>{self.slow_test_threshold}s)"
            )
            recommendations.append(
                "Mark slow tests with @pytest.mark.slow and run separately"
            )
        
        medium_count = len(categorized['medium'])
        if medium_count > 5:
            recommendations.append(
                f"Consider optimizing {medium_count} medium tests (>{self.medium_test_threshold}s)"
            )
        
        total_duration = sum(data['duration'] for data in self.timing_data.values())
        if total_duration > 300:  # 5 minutes
            recommendations.append(
                "Consider parallel execution with pytest-xdist"
            )
        
        return recommendations
    
    def save_results(self):
        """Save timing analysis results to file."""
        report = self.generate_timing_report()
        with open(self.results_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def load_historical_data(self) -> Optional[Dict]:
        """Load historical timing data if available."""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return None
        return None


class PerformanceMonitor:
    """Monitor test performance during execution."""
    
    def __init__(self):
        self.analyzer = TimingAnalyzer()
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def monitor_test(self, test_function):
        """Decorator to monitor test performance."""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            
            try:
                result = test_function(*args, **kwargs)
                end_time = time.time()
                end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
                
                duration = end_time - start_time
                memory_delta = end_memory - start_memory
                
                self.analyzer.record_test_timing(
                    test_function.__name__,
                    duration,
                    memory_delta
                )
                
                return result
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                self.analyzer.record_test_timing(
                    test_function.__name__,
                    duration,
                    0
                )
                raise e
        
        return wrapper


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def pytest_configure(config):
    """Configure pytest with performance monitoring."""
    # Add custom markers for performance analysis
    config.addinivalue_line(
        "markers",
        "performance_critical: Mark test as performance critical"
    )
    config.addinivalue_line(
        "markers", 
        "memory_intensive: Mark test as memory intensive"
    )


def pytest_sessionstart(session):
    """Initialize performance monitoring session."""
    global performance_monitor
    performance_monitor = PerformanceMonitor()


def pytest_sessionfinish(session, exitstatus):
    """Generate performance report at session end."""
    global performance_monitor
    report = performance_monitor.analyzer.save_results()
    
    # Print summary to console
    if 'summary' in report:
        summary = report['summary']
        print("\n" + "="*80)
        print("TEST PERFORMANCE SUMMARY")
        print("="*80)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Total Duration: {summary['total_duration']:.2f}s")
        print(f"Average Duration: {summary['average_duration']:.2f}s")
        print(f"Median Duration: {summary['median_duration']:.2f}s")
        print(f"Slowest Test: {summary['max_duration']:.2f}s")
        print(f"Peak Memory Delta: {summary['peak_memory']:.2f}MB")
        
        if 'recommendations' in report:
            print("\nRECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        
        print("="*80)


@pytest.fixture
def performance_monitor_fixture():
    """Fixture to provide performance monitoring capabilities."""
    return performance_monitor


def test_timing_analysis_functionality():
    """Test the timing analysis functionality itself."""
    analyzer = TestTimingAnalyzer()
    
    # Add sample data
    analyzer.record_test_timing("test_fast", 0.5, 1.0)
    analyzer.record_test_timing("test_medium", 5.0, 2.0)
    analyzer.record_test_timing("test_slow", 15.0, 5.0)
    
    report = analyzer.generate_timing_report()
    
    assert 'summary' in report
    assert 'categorized_tests' in report
    assert 'recommendations' in report
    
    # Check categorization
    assert len(report['categorized_tests']['fast']) == 1
    assert len(report['categorized_tests']['medium']) == 1
    assert len(report['categorized_tests']['slow']) == 1
    
    # Check recommendations
    assert len(report['recommendations']) > 0


@pytest.mark.performance_critical
def test_performance_monitor_decorator():
    """Test the performance monitoring decorator."""
    @performance_monitor.monitor_test
    def dummy_test():
        time.sleep(0.1)  # Simulate work
        return "success"
    
    result = dummy_test()
    assert result == "success"
    
    # Check if timing was recorded
    assert "dummy_test" in performance_monitor.analyzer.timing_data


if __name__ == "__main__":
    # Run timing analysis on existing test results
    analyzer = TestTimingAnalyzer()
    historical_data = analyzer.load_historical_data()
    
    if historical_data:
        print("Historical Test Performance Data:")
        print(json.dumps(historical_data, indent=2))
    else:
        print("No historical timing data found.")
        print("Run tests with performance monitoring to generate data.")