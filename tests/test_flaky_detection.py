"""
Flaky test detection and retry mechanisms.
Provides automated identification and handling of unreliable tests.
"""

import json
import time
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pytest
import functools
from collections import defaultdict, deque
import traceback
import threading
import statistics


class FlakyTestDetector:
    """Detects and tracks flaky tests across multiple runs."""
    
    def __init__(self, history_file: str = "flaky-test-history.json"):
        self.history_file = Path(history_file)
        self.test_history = self._load_history()
        self.current_session = {}
        self.flaky_threshold = 0.1  # 10% failure rate
        self.min_runs_for_detection = 5
        self.lock = threading.Lock()
        
    def _load_history(self) -> Dict:
        """Load historical test results."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def _save_history(self):
        """Save test history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.test_history, f, indent=2)
    
    def record_test_result(self, test_name: str, passed: bool, 
                          duration: float, error_msg: str = None):
        """Record a test result for flaky detection."""
        with self.lock:
            if test_name not in self.test_history:
                self.test_history[test_name] = {
                    'results': deque(maxlen=100),  # Keep last 100 results
                    'first_seen': time.time(),
                    'last_run': time.time(),
                    'total_runs': 0,
                    'total_failures': 0,
                    'error_patterns': defaultdict(int)
                }
            
            test_data = self.test_history[test_name]
            test_data['results'].append({
                'passed': passed,
                'timestamp': time.time(),
                'duration': duration,
                'error_msg': error_msg
            })
            
            test_data['last_run'] = time.time()
            test_data['total_runs'] += 1
            if not passed:
                test_data['total_failures'] += 1
                if error_msg:
                    # Extract error pattern (first line of error)
                    error_pattern = error_msg.split('\n')[0][:100]
                    test_data['error_patterns'][error_pattern] += 1
            
            self.current_session[test_name] = passed
    
    def is_flaky(self, test_name: str) -> bool:
        """Check if a test is considered flaky."""
        if test_name not in self.test_history:
            return False
        
        test_data = self.test_history[test_name]
        
        # Need minimum runs to determine flakiness
        if test_data['total_runs'] < self.min_runs_for_detection:
            return False
        
        # Check recent failure rate
        recent_results = list(test_data['results'])[-20:]  # Last 20 runs
        if len(recent_results) < 5:
            return False
        
        failure_rate = sum(1 for r in recent_results if not r['passed']) / len(recent_results)
        return 0 < failure_rate < (1 - self.flaky_threshold)
    
    def get_flaky_tests(self) -> List[Dict]:
        """Get all currently identified flaky tests."""
        flaky_tests = []
        
        for test_name, test_data in self.test_history.items():
            if self.is_flaky(test_name):
                recent_results = list(test_data['results'])[-20:]
                failure_rate = sum(1 for r in recent_results if not r['passed']) / len(recent_results)
                
                flaky_tests.append({
                    'name': test_name,
                    'failure_rate': failure_rate,
                    'total_runs': test_data['total_runs'],
                    'total_failures': test_data['total_failures'],
                    'last_run': test_data['last_run'],
                    'common_errors': dict(test_data['error_patterns']),
                    'recent_pattern': [r['passed'] for r in recent_results[-10:]]
                })
        
        return sorted(flaky_tests, key=lambda x: x['failure_rate'], reverse=True)
    
    def analyze_flaky_patterns(self, test_name: str) -> Dict:
        """Analyze patterns in flaky test behavior."""
        if test_name not in self.test_history:
            return {"error": "Test not found in history"}
        
        test_data = self.test_history[test_name]
        results = list(test_data['results'])
        
        if len(results) < 5:
            return {"error": "Insufficient data for analysis"}
        
        # Analyze timing patterns
        passed_times = [r['duration'] for r in results if r['passed']]
        failed_times = [r['duration'] for r in results if not r['passed']]
        
        # Look for consecutive failures/passes
        consecutive_failures = []
        consecutive_passes = []
        current_streak = 1
        current_state = results[0]['passed']
        
        for i in range(1, len(results)):
            if results[i]['passed'] == current_state:
                current_streak += 1
            else:
                if current_state:
                    consecutive_passes.append(current_streak)
                else:
                    consecutive_failures.append(current_streak)
                current_streak = 1
                current_state = results[i]['passed']
        
        analysis = {
            'test_name': test_name,
            'total_runs': len(results),
            'failure_rate': sum(1 for r in results if not r['passed']) / len(results),
            'timing_analysis': {
                'passed_avg': statistics.mean(passed_times) if passed_times else 0,
                'failed_avg': statistics.mean(failed_times) if failed_times else 0,
                'timing_correlation': abs(statistics.mean(passed_times) - statistics.mean(failed_times)) > 1.0 if passed_times and failed_times else False
            },
            'pattern_analysis': {
                'max_consecutive_failures': max(consecutive_failures) if consecutive_failures else 0,
                'max_consecutive_passes': max(consecutive_passes) if consecutive_passes else 0,
                'avg_failure_streak': statistics.mean(consecutive_failures) if consecutive_failures else 0,
                'avg_pass_streak': statistics.mean(consecutive_passes) if consecutive_passes else 0
            },
            'error_patterns': dict(test_data['error_patterns']),
            'recent_trend': results[-10:] if len(results) >= 10 else results
        }
        
        return analysis
    
    def generate_flaky_report(self) -> Dict:
        """Generate comprehensive flaky test report."""
        flaky_tests = self.get_flaky_tests()
        
        if not flaky_tests:
            return {
                "summary": "No flaky tests detected",
                "total_tests_analyzed": len(self.test_history),
                "flaky_tests": []
            }
        
        detailed_analyses = []
        for test in flaky_tests:
            analysis = self.analyze_flaky_patterns(test['name'])
            detailed_analyses.append(analysis)
        
        report = {
            "summary": {
                "total_tests_analyzed": len(self.test_history),
                "flaky_tests_count": len(flaky_tests),
                "flaky_percentage": len(flaky_tests) / len(self.test_history) * 100,
                "most_flaky": flaky_tests[0]['name'] if flaky_tests else None,
                "highest_failure_rate": flaky_tests[0]['failure_rate'] if flaky_tests else 0
            },
            "flaky_tests": flaky_tests,
            "detailed_analyses": detailed_analyses,
            "recommendations": self._generate_flaky_recommendations(flaky_tests)
        }
        
        return report
    
    def _generate_flaky_recommendations(self, flaky_tests: List[Dict]) -> List[str]:
        """Generate recommendations for handling flaky tests."""
        recommendations = []
        
        if not flaky_tests:
            return ["No flaky tests detected. Good job!"]
        
        high_flaky = [t for t in flaky_tests if t['failure_rate'] > 0.3]
        if high_flaky:
            recommendations.append(
                f"Investigate {len(high_flaky)} highly flaky tests (>30% failure rate)"
            )
        
        recommendations.extend([
            "Mark identified flaky tests with @pytest.mark.flaky",
            "Consider using pytest-rerunfailures for automatic retries",
            "Investigate timing-related issues in flaky tests",
            "Add better error handling and logging to flaky tests",
            "Consider isolating flaky tests to prevent cascade failures"
        ])
        
        return recommendations
    
    def save_report(self):
        """Save flaky test analysis and update history."""
        report = self.generate_flaky_report()
        
        # Save detailed report
        report_file = self.history_file.parent / "flaky-test-report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Update history
        self._save_history()
        
        return report


class FlakyTestRetryMechanism:
    """Implements retry mechanisms for flaky tests."""
    
    def __init__(self, detector: FlakyTestDetector = None):
        self.detector = detector or FlakyTestDetector()
        self.default_retries = 3
        self.retry_delay = 1.0  # seconds
        
    def retry_on_failure(self, retries: int = None, delay: float = None, 
                        exceptions: Tuple = None):
        """Decorator to retry failed tests."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                max_retries = retries or self.default_retries
                retry_delay = delay or self.retry_delay
                allowed_exceptions = exceptions or (Exception,)
                
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        end_time = time.time()
                        
                        # Record successful result
                        self.detector.record_test_result(
                            func.__name__, True, end_time - start_time
                        )
                        
                        if attempt > 0:
                            print(f"Test {func.__name__} passed on attempt {attempt + 1}")
                        
                        return result
                        
                    except allowed_exceptions as e:
                        last_exception = e
                        error_msg = str(e)
                        
                        if attempt < max_retries:
                            print(f"Test {func.__name__} failed on attempt {attempt + 1}, retrying...")
                            time.sleep(retry_delay)
                        else:
                            # Record final failure
                            self.detector.record_test_result(
                                func.__name__, False, 0, error_msg
                            )
                
                # All retries exhausted
                raise last_exception
            
            return wrapper
        return decorator
    
    def smart_retry(self, func):
        """Smart retry based on test's flaky history."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            test_name = func.__name__
            
            # Determine retry strategy based on history
            if self.detector.is_flaky(test_name):
                # More aggressive retry for known flaky tests
                max_retries = 5
                delay = 2.0
            else:
                # Conservative retry for unknown tests
                max_retries = 2
                delay = 1.0
            
            return self.retry_on_failure(max_retries, delay)(func)(*args, **kwargs)
        
        return wrapper


# Global detector instance
flaky_detector = FlakyTestDetector()
retry_mechanism = FlakyTestRetryMechanism(flaky_detector)


# Pytest hooks for automatic flaky detection
def pytest_runtest_makereport(item, call):
    """Hook to capture test results for flaky detection."""
    if call.when == "call":
        test_name = item.nodeid
        passed = call.excinfo is None
        duration = call.stop - call.start
        error_msg = str(call.excinfo.value) if call.excinfo else None
        
        flaky_detector.record_test_result(test_name, passed, duration, error_msg)


def pytest_sessionfinish(session, exitstatus):
    """Generate flaky test report at session end."""
    report = flaky_detector.save_report()
    
    if report["summary"]["flaky_tests_count"] > 0:
        print("\n" + "="*80)
        print("FLAKY TEST DETECTION REPORT")
        print("="*80)
        print(f"Total Tests Analyzed: {report['summary']['total_tests_analyzed']}")
        print(f"Flaky Tests Found: {report['summary']['flaky_tests_count']}")
        print(f"Flaky Test Percentage: {report['summary']['flaky_percentage']:.1f}%")
        
        if report['summary']['most_flaky']:
            print(f"Most Flaky Test: {report['summary']['most_flaky']}")
        
        print("\nFlaky Tests:")
        for test in report["flaky_tests"][:5]:  # Show top 5
            print(f"  • {test['name']} - {test['failure_rate']:.1%} failure rate")
        
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")
        
        print("="*80)


@pytest.fixture
def flaky_detector_fixture():
    """Fixture to provide flaky detection capabilities."""
    return flaky_detector


@pytest.fixture
def retry_mechanism_fixture():
    """Fixture to provide retry mechanisms."""
    return retry_mechanism


def test_flaky_detection_functionality():
    """Test the flaky detection functionality."""
    detector = FlakyTestDetector()
    
    # Simulate a flaky test
    for i in range(10):
        passed = random.random() > 0.3  # 70% success rate
        detector.record_test_result("test_flaky_example", passed, 1.0, 
                                  "Random failure" if not passed else None)
    
    # Check if test is detected as flaky
    assert detector.is_flaky("test_flaky_example")
    
    flaky_tests = detector.get_flaky_tests()
    assert len(flaky_tests) > 0
    assert flaky_tests[0]['name'] == "test_flaky_example"


def test_retry_mechanism():
    """Test the retry mechanism functionality."""
    retry_mech = FlakyTestRetryMechanism()
    
    call_count = 0
    
    @retry_mech.retry_on_failure(retries=3, delay=0.1)
    def flaky_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Simulated failure")
        return "success"
    
    result = flaky_function()
    assert result == "success"
    assert call_count == 3


@pytest.mark.flaky
def test_intentionally_flaky():
    """An intentionally flaky test for demonstration."""
    # This test fails ~30% of the time
    if random.random() < 0.3:
        raise AssertionError("Intentional flaky failure")
    assert True


if __name__ == "__main__":
    # Generate flaky test report
    detector = FlakyTestDetector()
    report = detector.generate_flaky_report()
    
    print("Flaky Test Analysis Report:")
    print(json.dumps(report, indent=2, default=str))