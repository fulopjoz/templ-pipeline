# TEMPL Pipeline Testing Guide

A comprehensive guide for developers working with the TEMPL pipeline testing infrastructure.

## Table of Contents
- [Overview](#overview)
- [Test Environment Setup](#test-environment-setup)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Test Categories and Markers](#test-categories-and-markers)
- [Performance Testing](#performance-testing)
- [Advanced Testing Features](#advanced-testing-features)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

## Overview

The TEMPL pipeline testing infrastructure provides comprehensive testing capabilities including:

- **Performance-optimized test execution** with parallel processing
- **Smart fixture caching** for expensive operations
- **Flaky test detection** and automatic retry mechanisms
- **Advanced coverage analysis** with branch coverage
- **Test timing analysis** and performance monitoring
- **Multiple test environments** (development, CI/CD, performance)

## Test Environment Setup

### Prerequisites

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Or using the setup script
source setup_templ_env.sh --dev
```

### Required Dependencies

The testing infrastructure includes these key dependencies:
- `pytest>=7.0.0` - Core testing framework
- `pytest-xdist>=3.0.0` - Parallel test execution
- `pytest-cov>=3.0.0` - Coverage reporting
- `pytest-timeout>=2.0.0` - Test timeout handling
- `pytest-repeat>=0.9.0` - Test repetition for flaky detection
- `pytest-benchmark>=4.0.0` - Performance benchmarking
- `pytest-html>=3.0.0` - HTML test reports
- `pytest-json-report>=1.5.0` - JSON test reports

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m fast          # Fast tests only (< 10s)
pytest -m medium        # Medium tests (< 30s)  
pytest -m slow          # Slow tests (> 30s)
pytest -m integration   # Integration tests
pytest -m ui            # UI/Streamlit tests
pytest -m performance   # Performance tests
```

### Smart Test Runner

Use the intelligent test runner for optimized execution:

```bash
# Auto-detect optimal configuration
python run_tests.py

# Use specific configuration
python run_tests.py --config ci        # CI/CD optimized
python run_tests.py --config parallel  # Parallel execution
python run_tests.py --config performance # Performance testing

# Use test presets
python run_tests.py --preset quick     # Fast tests only
python run_tests.py --preset critical  # Critical path tests
python run_tests.py --preset flaky     # Flaky tests with retries

# Enable parallel execution
python run_tests.py --parallel

# List available options
python run_tests.py --list-configs
```

### Environment-Specific Configurations

```bash
# Development environment
pytest -c pytest.ini

# CI/CD environment  
pytest -c pytest-ci.ini

# Parallel execution
pytest -c pytest-parallel.ini

# Performance testing
pytest -c pytest-performance.ini
```

### Advanced Execution Options

```bash
# Run with coverage analysis
PYTEST_COVERAGE_ANALYSIS=1 pytest

# Run with flaky test detection
pytest --repeat=3 -m flaky

# Run with timing analysis
pytest --durations=20

# Run with detailed reporting
pytest --html=test-report.html --json-report --json-report-file=results.json
```

## Writing Tests

### Test Structure

Follow this structure for test files:

```python
"""
Test module for [component name].
Brief description of what this module tests.
"""

import pytest
from unittest.mock import Mock, patch
from templ_pipeline.core import ComponentUnderTest

class TestComponentUnderTest:
    """Test suite for ComponentUnderTest."""
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        component = ComponentUnderTest()
        
        # Act
        result = component.process()
        
        # Assert
        assert result is not None
    
    @pytest.mark.slow
    def test_expensive_operation(self):
        """Test expensive operation."""
        # Mark slow tests appropriately
        pass
    
    @pytest.mark.flaky
    def test_potentially_flaky(self):
        """Test that might be flaky."""
        # Mark potentially flaky tests
        pass
```

### Using Fixtures

```python
def test_with_standard_fixtures(cached_protein_data, cached_ligand_data):
    """Test using cached fixtures for performance."""
    # Use session-scoped cached fixtures for expensive data
    assert len(cached_protein_data) > 0
    assert len(cached_ligand_data['simple_molecules']) > 0

def test_with_custom_fixture(custom_fixture):
    """Test with custom fixture."""
    # Custom fixtures defined in conftest.py
    pass

@pytest.fixture
def local_fixture():
    """Local fixture for this test module."""
    return "test_data"

def test_with_local_fixture(local_fixture):
    """Test using local fixture."""
    assert local_fixture == "test_data"
```

### Performance Testing

```python
import pytest
from tests.test_profiler import profiler

class TestPerformance:
    """Performance tests."""
    
    @pytest.mark.performance
    def test_benchmark_operation(self, benchmark):
        """Benchmark an operation."""
        def operation_to_benchmark():
            # Your expensive operation here
            return expensive_computation()
        
        result = benchmark(operation_to_benchmark)
        assert result is not None
    
    @profiler.profile_test("custom_test_name")
    def test_with_profiling(self):
        """Test with detailed profiling."""
        # This test will be profiled automatically
        expensive_operation()
    
    def test_with_timing_analysis(self, performance_monitor_fixture):
        """Test with timing analysis."""
        # Uses automatic timing analysis
        time_consuming_operation()
```

### Async Testing

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_operation():
    """Test async operation."""
    result = await async_function()
    assert result is not None

def test_async_with_helper(async_test_helper):
    """Test async with helper utilities."""
    future = async_test_helper['create_resolved_future']("result")
    assert future.result() == "result"
```

## Test Categories and Markers

### Standard Markers

- `@pytest.mark.fast` - Tests that run in < 10 seconds
- `@pytest.mark.medium` - Tests that run in < 30 seconds  
- `@pytest.mark.slow` - Tests that run in > 30 seconds
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.ui` - UI/Streamlit tests
- `@pytest.mark.performance` - Performance and benchmark tests

### Advanced Markers

- `@pytest.mark.flaky` - Tests that may be flaky and need retries
- `@pytest.mark.critical` - Critical path tests requiring high coverage
- `@pytest.mark.isolation` - Tests requiring isolation verification
- `@pytest.mark.cached` - Tests using cached fixtures
- `@pytest.mark.coverage_critical` - Tests covering critical code paths

### Custom Markers Usage

```python
@pytest.mark.slow
@pytest.mark.integration
def test_full_pipeline():
    """Integration test for full pipeline."""
    pass

@pytest.mark.flaky
@pytest.mark.parametrize("attempt", range(3))
def test_potentially_unstable(attempt):
    """Test that might fail intermittently."""
    pass

@pytest.mark.performance
@pytest.mark.critical
def test_core_algorithm_performance():
    """Performance test for core algorithm."""
    pass
```

## Performance Testing

### Benchmark Tests

```python
def test_mcs_performance(benchmark):
    """Benchmark MCS computation."""
    def mcs_computation():
        return compute_mcs(mol1, mol2)
    
    result = benchmark(mcs_computation)
    assert result is not None

def test_scaling_behavior():
    """Test scaling behavior."""
    for size in [10, 50, 100, 500]:
        start_time = time.time()
        process_batch(size)
        duration = time.time() - start_time
        
        # Assert reasonable scaling
        assert duration < size * 0.1  # Linear scaling assumption
```

### Memory Testing

```python
import psutil

def test_memory_usage():
    """Test memory usage patterns."""
    process = psutil.Process()
    start_memory = process.memory_info().rss
    
    # Perform memory-intensive operation
    large_data_operation()
    
    end_memory = process.memory_info().rss
    memory_increase = (end_memory - start_memory) / 1024 / 1024  # MB
    
    # Assert reasonable memory usage
    assert memory_increase < 100  # Less than 100MB increase
```

## Advanced Testing Features

### Fixture Caching

The testing infrastructure includes smart fixture caching for expensive operations:

```python
from tests.test_fixture_caching import cached_fixture

@cached_fixture(scope="session", ttl=3600)  # Cache for 1 hour
def expensive_data_fixture():
    """Expensive fixture that gets cached."""
    # This will only run once per session
    return load_expensive_data()

def test_using_cached_fixture(expensive_data_fixture):
    """Test using cached fixture."""
    # Data is loaded from cache if available
    assert len(expensive_data_fixture) > 0
```

### Flaky Test Handling

```python
from tests.test_flaky_detection import retry_mechanism

@retry_mechanism.smart_retry
def test_with_smart_retry():
    """Test with intelligent retry logic."""
    # Automatically retries based on test history
    potentially_flaky_operation()

@retry_mechanism.retry_on_failure(retries=5, delay=1.0)
def test_with_custom_retry():
    """Test with custom retry configuration."""
    flaky_external_service_call()
```

### Coverage Analysis

```python
# Run with detailed coverage analysis
PYTEST_COVERAGE_ANALYSIS=1 pytest

# Generate coverage reports
python -m tests.test_coverage_analysis
```

### Test Profiling

```python
from tests.test_profiler import profiler

# Profile specific tests
with profiler.profile_context("test_context"):
    expensive_operation()

# View profiling results
python -m tests.test_profiler
```

## CI/CD Integration

### GitHub Actions Configuration

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install uv
          uv pip install -e ".[dev]"
      
      - name: Run tests
        run: |
          python run_tests.py --config ci --preset critical
        env:
          PYTEST_COVERAGE_ANALYSIS: 1
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage-analysis/coverage.xml
```

### Environment Variables

- `CI=true` - Enables CI-specific optimizations
- `PYTEST_PARALLEL=true` - Forces parallel execution
- `PYTEST_PERFORMANCE=true` - Enables performance testing mode
- `PYTEST_COVERAGE_ANALYSIS=1` - Enables detailed coverage analysis

## Troubleshooting

### Common Issues

#### Test Discovery Problems
```bash
# Clear pytest cache
pytest --cache-clear

# Verify test discovery
pytest --collect-only
```

#### Slow Test Execution
```bash
# Identify slow tests
pytest --durations=10

# Use parallel execution
python run_tests.py --parallel

# Use faster test categories
pytest -m fast
```

#### Memory Issues
```bash
# Monitor memory usage
python -c "from tests.test_profiler import profiler; profiler.generate_performance_report()"

# Use fixture caching
pytest -m cached
```

#### Flaky Test Issues
```bash
# Identify flaky tests
python -c "from tests.test_flaky_detection import flaky_detector; print(flaky_detector.generate_flaky_report())"

# Run flaky tests with retries
python run_tests.py --preset flaky
```

### Debug Mode

```bash
# Run with verbose output
pytest -v --tb=long

# Run single test with debugging
pytest -s tests/test_specific.py::test_function

# Use Python debugger
pytest --pdb tests/test_specific.py::test_function
```

### Performance Debugging

```bash
# Generate performance report
python -m tests.test_timing_analysis

# Profile specific tests
python -m tests.test_profiler

# Analyze coverage gaps
PYTEST_COVERAGE_ANALYSIS=1 pytest -m critical
```

## Best Practices

### Test Naming
- Use descriptive names: `test_mcs_computation_with_invalid_molecules`
- Follow pattern: `test_[functionality]_[condition]_[expected_result]`
- Group related tests in classes

### Test Organization
- One test file per module under test
- Use appropriate markers for categorization
- Keep tests focused and independent
- Use fixtures for shared setup

### Performance Considerations
- Mark expensive tests appropriately (`@pytest.mark.slow`)
- Use cached fixtures for expensive setup
- Profile tests regularly to identify bottlenecks
- Use parallel execution for large test suites

### Coverage Goals
- Aim for 90%+ line coverage on critical paths
- Achieve 80%+ branch coverage
- Focus on edge cases and error conditions
- Use coverage analysis to identify gaps

## Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [TEMPL Pipeline Architecture](./ARCHITECTURE.md)
- [Contributing Guidelines](./CONTRIBUTING.md)
- [QA Process Documentation](./QA_PROCESS.md)