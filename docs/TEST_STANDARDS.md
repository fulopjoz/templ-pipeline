# Test Writing Standards and Contribution Guidelines

Standards and guidelines for writing high-quality tests in the TEMPL pipeline project.

## Table of Contents
- [Overview](#overview)
- [Test Writing Standards](#test-writing-standards)
- [Testing Best Practices](#testing-best-practices)
- [Code Organization](#code-organization)
- [Test Data Management](#test-data-management)
- [Contribution Guidelines](#contribution-guidelines)
- [Review Process](#review-process)
- [Quality Metrics](#quality-metrics)

## Overview

This document establishes standards for writing, organizing, and maintaining tests in the TEMPL pipeline project. Following these guidelines ensures consistency, maintainability, and effectiveness of our test suite.

### Goals
- Maintain high code coverage (90%+ for critical paths)
- Ensure test reliability and consistency
- Facilitate easy test maintenance and updates
- Enable efficient test execution and debugging
- Support continuous integration and deployment

## Test Writing Standards

### Test Structure and Organization

#### File Naming Convention
```
tests/
├── test_[module_name].py          # Unit tests for specific modules
├── test_[component]_integration.py # Integration tests
├── test_[feature]_performance.py  # Performance tests
├── test_[system]_e2e.py          # End-to-end tests
└── fixtures/                      # Shared test data
```

#### Test Function Naming
```python
def test_[functionality]_[condition]_[expected_result]():
    """Test [functionality] [condition] [expected result]."""
    pass

# Examples:
def test_mcs_computation_with_valid_molecules_returns_structure():
    """Test MCS computation with valid molecules returns structure."""
    pass

def test_pipeline_execution_with_invalid_input_raises_error():
    """Test pipeline execution with invalid input raises error."""
    pass

def test_embedding_generation_with_large_protein_completes_successfully():
    """Test embedding generation with large protein completes successfully."""
    pass
```

#### Test Class Organization
```python
class TestComponentName:
    """Test suite for ComponentName."""
    
    def setup_method(self):
        """Set up test environment before each test method."""
        # Initialize test state
        pass
    
    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up test state
        pass
    
    def test_basic_functionality(self):
        """Test basic functionality."""
        pass
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        pass
    
    def test_error_conditions(self):
        """Test error handling and exception cases."""
        pass
```

### Test Implementation Standards

#### Test Structure (Arrange-Act-Assert)
```python
def test_example_functionality():
    """Test example functionality with proper structure."""
    # Arrange - Set up test data and conditions
    input_data = create_test_data()
    expected_result = "expected_value"
    
    # Act - Execute the functionality being tested
    actual_result = function_under_test(input_data)
    
    # Assert - Verify the results
    assert actual_result == expected_result
    assert isinstance(actual_result, str)
    assert len(actual_result) > 0
```

#### Assertion Standards
```python
# Use descriptive assertions
assert actual == expected, f"Expected {expected}, got {actual}"

# Use appropriate assertion methods
assert isinstance(result, list)
assert result is not None
assert len(result) == 3
assert "key" in result_dict
assert result > 0

# Use pytest helpers for better error messages
from pytest import approx
assert actual_float == approx(expected_float, rel=1e-3)

# Test exceptions explicitly
with pytest.raises(ValueError, match="Invalid input"):
    function_that_should_fail(invalid_input)
```

#### Docstring Standards
```python
def test_mcs_computation_performance():
    """
    Test MCS computation performance meets requirements.
    
    This test verifies that MCS computation completes within
    acceptable time limits for molecules of various sizes.
    
    Expected behavior:
    - Small molecules (< 50 atoms): < 1 second
    - Medium molecules (50-200 atoms): < 5 seconds
    - Large molecules (> 200 atoms): < 30 seconds
    
    Test data:
    - Uses standardized test molecules from fixtures
    - Includes edge cases with complex structures
    
    Performance criteria:
    - Execution time within specified limits
    - Memory usage < 100MB per computation
    - Successful completion for all test cases
    """
    pass
```

### Test Categories and Markers

#### Required Markers
```python
# Speed-based markers (required for all tests)
@pytest.mark.fast        # < 10 seconds
@pytest.mark.medium      # < 30 seconds
@pytest.mark.slow        # > 30 seconds

# Type-based markers
@pytest.mark.integration  # Integration tests
@pytest.mark.ui          # UI/Streamlit tests
@pytest.mark.performance # Performance tests

# Quality markers
@pytest.mark.flaky       # Potentially flaky tests
@pytest.mark.critical    # Critical path tests
@pytest.mark.isolation   # Requires test isolation
```

#### Marker Usage Examples
```python
@pytest.mark.fast
@pytest.mark.critical
def test_core_pipeline_initialization():
    """Test core pipeline initialization (fast, critical)."""
    pass

@pytest.mark.slow
@pytest.mark.integration
def test_full_pipeline_workflow():
    """Test complete pipeline workflow (slow, integration)."""
    pass

@pytest.mark.performance
@pytest.mark.parametrize("molecule_size", [10, 50, 100, 500])
def test_scaling_behavior(molecule_size):
    """Test performance scaling with molecule size."""
    pass
```

### Parametrized Testing
```python
@pytest.mark.parametrize("input_data,expected", [
    ("CCO", "ethanol"),
    ("c1ccccc1", "benzene"),
    ("CC(=O)O", "acetic_acid"),
])
def test_molecule_identification(input_data, expected):
    """Test molecule identification with various inputs."""
    result = identify_molecule(input_data)
    assert result == expected

@pytest.mark.parametrize("protein_size", [100, 500, 1000])
@pytest.mark.parametrize("embedding_type", ["esm2", "protbert"])
def test_embedding_generation(protein_size, embedding_type):
    """Test embedding generation with different configurations."""
    protein = create_protein_sequence(protein_size)
    embedding = generate_embedding(protein, embedding_type)
    assert embedding is not None
    assert len(embedding) == expected_embedding_size(embedding_type)
```

## Testing Best Practices

### Test Independence
```python
# Good: Each test is independent
def test_function_a():
    """Test function A independently."""
    data = create_fresh_test_data()
    result = function_a(data)
    assert result is not None

def test_function_b():
    """Test function B independently."""
    data = create_fresh_test_data()
    result = function_b(data)
    assert result is not None

# Bad: Tests depend on each other
class TestBadExample:
    def test_setup_data(self):
        self.shared_data = create_data()
    
    def test_use_data(self):
        # This test depends on test_setup_data running first
        result = process(self.shared_data)
        assert result is not None
```

### Fixture Usage
```python
# Use fixtures for shared setup
@pytest.fixture
def sample_protein():
    """Provide sample protein for testing."""
    return create_test_protein()

@pytest.fixture
def sample_molecules():
    """Provide sample molecules for testing."""
    return [
        create_test_molecule("CCO"),
        create_test_molecule("c1ccccc1"),
        create_test_molecule("CC(=O)O"),
    ]

def test_with_fixtures(sample_protein, sample_molecules):
    """Test using fixtures for setup."""
    result = process_protein_ligands(sample_protein, sample_molecules)
    assert len(result) == len(sample_molecules)

# Use cached fixtures for expensive operations
def test_with_cached_fixture(cached_protein_data):
    """Test using cached fixture for performance."""
    # Uses session-scoped cached data
    assert len(cached_protein_data) > 0
```

### Mocking and Test Doubles
```python
from unittest.mock import Mock, patch, MagicMock

def test_with_mock():
    """Test using mock objects."""
    # Mock external dependencies
    mock_service = Mock()
    mock_service.process.return_value = "mocked_result"
    
    # Use mock in test
    result = function_under_test(mock_service)
    
    # Verify mock was called correctly
    mock_service.process.assert_called_once()
    assert result == "mocked_result"

@patch('templ_pipeline.external_service')
def test_with_patch(mock_service):
    """Test using patch decorator."""
    mock_service.return_value = "patched_result"
    
    result = function_that_uses_external_service()
    assert result == "patched_result"

def test_with_context_manager():
    """Test using mock context manager."""
    with patch('builtins.open', mock_open(read_data="test_data")) as mock_file:
        result = read_file_function("test.txt")
        mock_file.assert_called_once_with("test.txt", 'r')
        assert result == "test_data"
```

### Error Testing
```python
def test_error_conditions():
    """Test various error conditions."""
    # Test specific exception types
    with pytest.raises(ValueError):
        function_with_invalid_input(None)
    
    # Test exception messages
    with pytest.raises(ValueError, match="Invalid molecule format"):
        parse_molecule("invalid_smiles")
    
    # Test exception attributes
    with pytest.raises(CustomException) as exc_info:
        function_that_raises_custom_exception()
    
    assert exc_info.value.error_code == "E001"
    assert "details" in str(exc_info.value)

def test_error_recovery():
    """Test error recovery mechanisms."""
    # Test graceful degradation
    result = function_with_fallback(invalid_input)
    assert result is not None
    assert result.status == "fallback_used"
    
    # Test error logging
    with pytest.raises(Exception) as exc_info:
        function_that_logs_errors()
    
    # Verify error was logged (if using logging)
    # assert "Error occurred" in caplog.text
```

## Code Organization

### Test File Structure
```python
"""
Test module for [component_name].

This module contains comprehensive tests for [component_name],
including unit tests, integration tests, and performance tests.

Test categories:
- Unit tests: Test individual functions and methods
- Integration tests: Test component interactions
- Performance tests: Verify performance requirements
- Error tests: Test error handling and edge cases
"""

import pytest
from unittest.mock import Mock, patch
import time
from pathlib import Path

# Import modules under test
from templ_pipeline.core.component import ComponentUnderTest
from templ_pipeline.utils.helpers import helper_function

# Import test utilities
from tests.utils import TestDataFactory, create_test_molecule
from tests.fixtures.data_factory import standard_test_data


class TestComponentUnderTest:
    """Comprehensive test suite for ComponentUnderTest."""
    
    def test_initialization(self):
        """Test component initialization."""
        pass
    
    def test_basic_operations(self):
        """Test basic component operations."""
        pass
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        pass
    
    def test_error_handling(self):
        """Test error handling mechanisms."""
        pass


class TestComponentIntegration:
    """Integration tests for ComponentUnderTest."""
    
    @pytest.mark.integration
    def test_integration_scenario_1(self):
        """Test integration scenario 1."""
        pass


class TestComponentPerformance:
    """Performance tests for ComponentUnderTest."""
    
    @pytest.mark.performance
    def test_performance_requirements(self):
        """Test performance meets requirements."""
        pass
```

### Import Standards
```python
# Standard library imports first
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports second
import pytest
import numpy as np
from unittest.mock import Mock, patch

# Local imports last, with absolute imports
from templ_pipeline.core.pipeline import TEMPLPipeline
from templ_pipeline.core.mcs import MCSComputer
from templ_pipeline.utils.validation import validate_molecule

# Test-specific imports
from tests.utils import TestDataFactory
from tests.fixtures.molecules import standard_molecules
```

## Test Data Management

### Test Data Organization
```
tests/
├── fixtures/
│   ├── __init__.py
│   ├── molecules.py          # Molecular test data
│   ├── proteins.py           # Protein test data
│   ├── data_factory.py       # Test data factory
│   └── benchmark_data.py     # Benchmark test data
├── data/
│   ├── sample_proteins/      # Sample protein files
│   ├── sample_ligands/       # Sample ligand files
│   └── expected_outputs/     # Expected output files
└── utils/
    ├── __init__.py
    ├── test_helpers.py       # Test utility functions
    └── assertions.py         # Custom assertion helpers
```

### Test Data Creation
```python
# Use factories for test data creation
class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_test_molecule(smiles: str, name: str = None):
        """Create a test molecule from SMILES."""
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        
        if name:
            mol.SetProp("_Name", name)
        
        return mol
    
    @staticmethod
    def create_test_protein(sequence: str, pdb_id: str = "test"):
        """Create a test protein from sequence."""
        return {
            'sequence': sequence,
            'pdb_id': pdb_id,
            'length': len(sequence),
            'chains': ['A']
        }

# Use constants for common test values
STANDARD_TEST_SMILES = [
    "CCO",              # Ethanol
    "CC(=O)O",          # Acetic acid
    "c1ccccc1",         # Benzene
    "CCN(CC)CC",        # Triethylamine
]

STANDARD_TEST_PROTEINS = {
    'small': 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR',
    'medium': 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUQTPQWNSPSYQPQLYITAQQTQRAADLGDGWKWSDLFLGPGMSEQHLAQQGKQGKGQ'
}
```

### Fixture Best Practices
```python
# Session-scoped fixtures for expensive setup
@pytest.fixture(scope="session")
def expensive_test_data():
    """Create expensive test data once per session."""
    # This runs once per test session
    return load_large_dataset()

# Function-scoped fixtures for test isolation
@pytest.fixture
def clean_test_environment():
    """Provide clean test environment for each test."""
    # Setup
    setup_clean_environment()
    
    yield  # Test runs here
    
    # Teardown
    cleanup_environment()

# Parameterized fixtures for test variations
@pytest.fixture(params=["small", "medium", "large"])
def protein_size(request):
    """Provide proteins of different sizes."""
    return STANDARD_TEST_PROTEINS[request.param]
```

## Contribution Guidelines

### Pre-Contribution Checklist
- [ ] Understand the component being tested
- [ ] Review existing tests for the component
- [ ] Identify test gaps in coverage
- [ ] Plan test structure and organization
- [ ] Consider performance implications

### Test Development Workflow
1. **Analysis Phase**
   - Analyze the code to be tested
   - Identify test scenarios and edge cases
   - Plan test data requirements
   - Estimate test complexity and timing

2. **Implementation Phase**
   - Write tests following standards
   - Use appropriate fixtures and utilities
   - Add proper markers and documentation
   - Implement error cases and edge cases

3. **Validation Phase**
   - Run tests locally with coverage
   - Verify performance within limits
   - Check for flaky behavior
   - Validate against coding standards

4. **Integration Phase**
   - Run full test suite
   - Verify CI/CD compatibility
   - Update documentation if needed
   - Submit for code review

### Code Review Requirements

#### Test Code Review Checklist
- [ ] Tests follow naming conventions
- [ ] Appropriate markers are applied
- [ ] Test independence is maintained
- [ ] Coverage targets are met
- [ ] Performance is acceptable
- [ ] Documentation is complete
- [ ] Error cases are covered

#### Review Focus Areas
1. **Test Quality**
   - Clear test intent and documentation
   - Proper use of assertions
   - Appropriate test data usage
   - Error condition coverage

2. **Performance**
   - Test execution time
   - Memory usage patterns
   - Fixture efficiency
   - Scaling behavior

3. **Maintainability**
   - Code clarity and readability
   - Proper use of fixtures
   - Minimal duplication
   - Future-proof design

## Quality Metrics

### Coverage Requirements
- **Unit Tests**: 90%+ line coverage for critical components
- **Integration Tests**: 80%+ coverage of integration scenarios
- **Branch Coverage**: 85%+ for complex logic paths
- **Critical Path Coverage**: 95%+ for core pipeline functionality

### Performance Standards
- **Fast Tests**: Complete in < 10 seconds
- **Medium Tests**: Complete in < 30 seconds
- **Slow Tests**: Complete in < 5 minutes
- **Total Suite**: Complete in < 30 minutes

### Quality Standards
- **Flaky Test Rate**: < 2% of total tests
- **Test Reliability**: > 99% pass rate in CI/CD
- **Maintenance Burden**: < 10% of development time
- **Documentation Coverage**: 100% of public APIs

### Monitoring and Reporting
```bash
# Generate coverage report
pytest --cov=templ_pipeline --cov-report=html

# Run performance analysis
python -m tests.test_timing_analysis

# Check for flaky tests
python -m tests.test_flaky_detection

# Generate quality metrics
python -m tests.test_coverage_analysis
```

## Additional Resources

### Tools and Utilities
- **pytest**: Primary testing framework
- **pytest-cov**: Coverage analysis
- **pytest-xdist**: Parallel execution
- **pytest-benchmark**: Performance testing
- **Custom test utilities**: Project-specific helpers

### Documentation
- [Testing Guide](./TESTING_GUIDE.md)
- [QA Process](./QA_PROCESS.md)
- [Contributing Guidelines](./CONTRIBUTING.md)
- [Architecture Documentation](./ARCHITECTURE.md)

### Examples and Templates
- [Test Template](../tests/test_template.py)
- [Integration Test Example](../tests/test_integration_example.py)
- [Performance Test Example](../tests/test_performance_example.py)