# CLAUDE.md - TEMPL Pipeline Project Guide

## Project Overview

**TEMPL Pipeline** is a template-based protein-ligand pose prediction system that uses ligand similarity and template superposition instead of exhaustive docking or deep neural networks. It's designed for fast, reasonable pose predictions within known chemical space.

**Key Features:**
- Template-based pose prediction using ligand similarity
- Alignment driven by maximal common substructure (MCS)
- Constrained conformer generation (ETKDG v3)
- Shape/pharmacophore scoring for pose selection
- Built-in benchmarks (Polaris, time-split PDBbind)
- CPU-only by default; GPU optional for protein embeddings

## Python Development Excellence Standards

### Code Quality Framework

**Must Follow PEP 8 Standards:**
- Use 4 spaces for indentation (never tabs)
- Limit lines to 79 characters for code, 72 for docstrings
- Use snake_case for functions/variables, PascalCase for classes
- Organize imports: standard library → third-party → local imports "explicit is better than implicit" principle

**Type Hints and Documentation:**
- Use type hints for ALL function parameters and return values
- Write comprehensive docstrings following PEP 257
- Include examples in docstrings for complex functions
- Use meaningful, descriptive variable and function names
- Avoid abbreviations unless universally understood

**Error Handling Best Practices:**
- Use specific exception types, never bare `except:`
- Implement proper exception chaining with `from` clause
- Keep try blocks focused and minimal
- Log errors with appropriate context and severity
- Provide user-friendly error messages in Streamlit components[22][25][28]

### Performance Optimization Guidelines

**Memory Management:**
- Use generators for large datasets instead of lists
- Implement `__slots__` for data classes with many instances
- Use `@st.cache_data` for expensive Streamlit computations
- Profile memory usage with `memory_profiler` for critical paths
- Explicit cleanup with `del` for large objects when done

**Code Optimization Strategies:**
- Use list comprehensions over explicit loops where appropriate
- Leverage built-in functions and libraries (NumPy, pandas)
- Avoid global variables; use function parameters
- Use `collections` module for efficient data structures
- Profile code with `cProfile` before optimizing

**I/O and Async Optimization:**
- Use `asyncio` for I/O-bound operations when beneficial
- Implement proper async/await patterns for concurrent operations
- Use context managers for resource management
- Optimize file I/O with appropriate buffer sizes

### Testing and Quality Assurance

**Testing Framework:**
- Use pytest as the primary testing framework
- Write focused tests that test one behavior at a time
- Use fixtures for test setup and teardown
- Mock external dependencies appropriately
- Aim for 80%+ test coverage

**Testing Best Practices:**
- Keep tests simple with single assertions where possible
- Use descriptive test names that explain the behavior
- Organize tests to mirror application structure
- Use parametrized tests for multiple input scenarios
- Separate unit, integration, and performance tests

## Project Structure

```
templ_pipeline/
├── core/          # Core pipeline functionality
├── cli/           # Command-line interface
├── ui/            # Streamlit web interface
├── benchmark/     # Benchmarking utilities
├── fair/          # FAIR evaluation components
├── __init__.py    # Package initialization
└── data/          # Data files and embeddings
```

### Configuration Management

**Centralized Configuration Pattern:**
- Use environment variables for deployment-specific settings
- Implement configuration classes with validation
- Separate sensitive data (API keys) from general config
- Use `.env` files for local development
- Document all configuration options

**Logging Configuration:**
- Configure logging at module level using `__name__`
- Use appropriate log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Include timestamps and context in log messages
- Centralize logging configuration in `utils/logging_config.py`
- Use structured logging for production environments[61][64][67]

### Performance Monitoring and Profiling

**Profiling Strategy:**
- Use `cProfile` for general performance profiling
- Apply `line_profiler` for line-by-line analysis
- Monitor memory usage with `memory_profiler`
- Use `py-spy` for production profiling with minimal overhead
- Profile before optimizing - measure, don't guess

**Performance Guidelines:**
- Optimize data structures before algorithms
- Use appropriate data types (sets for membership tests)
- Cache expensive computations appropriately
- Avoid premature optimization
- Focus on bottlenecks identified through profiling

## Streamlit-Specific Best Practices

**UI Architecture:**
- Keep main script focused on UI flow
- Separate business logic from presentation
- Use consistent layouts across all pages
- Implement proper error boundaries with user-friendly messages
- Add loading indicators for long-running operations

**Performance Optimization:**
- Use `@st.cache_data` for expensive computations
- Minimize use of `st.session_state`
- Optimize for different screen sizes
- Handle large datasets with pagination or streaming
- Implement proper timeout handling

**User Experience:**
- Add confirmation dialogs for destructive actions
- Provide clear progress indicators
- Use appropriate Streamlit components for data types
- Implement graceful error handling
- Ensure responsive design

## Development Workflow

### Code Quality Checks

Format code

black templ_pipeline/
isort templ_pipeline/
Type checking

mypy templ_pipeline/
Linting

flake8 templ_pipeline/
Testing

pytest -v tests/
pytest --cov=templ_pipeline tests/

text

### Performance Testing

Profile specific functions

python -m cProfile -s tottime script.py
Memory profiling

python -m memory_profiler script.py
Line-by-line profiling

kernprof -l -v script.py

text

### Environment Management

Setup development environment

source setup_templ_env.sh --dev
Activate environment

source .templ/bin/activate
Install dependencies

pip install -r requirements-dev.txt

text

## Production Readiness Checklist

**Code Quality:**
- [ ] All functions have type hints and docstrings
- [ ] Error handling covers all edge cases
- [ ] Logging is properly configured
- [ ] Code follows PEP 8 standards
- [ ] No hardcoded values or credentials

**Performance:**
- [ ] Profile analysis completed for critical paths
- [ ] Memory usage optimized for production loads
- [ ] Caching implemented for expensive operations
- [ ] I/O operations properly handled
- [ ] Resource cleanup implemented

**Testing:**
- [ ] Unit tests cover core functionality
- [ ] Integration tests validate component interaction
- [ ] Performance tests validate acceptable response times
- [ ] Error scenarios properly tested
- [ ] Test coverage above 80%

**Documentation:**
- [ ] README updated with setup instructions
- [ ] API documentation complete
- [ ] Configuration options documented
- [ ] Deployment guide available
- [ ] Troubleshooting guide provided

## Advanced Patterns and Techniques

**Memory-Efficient Patterns:**
- Use generators for data processing pipelines
- Implement lazy loading for large datasets
- Use `__slots__` for frequently instantiated classes
- Apply proper garbage collection strategies
- Monitor and optimize object lifespans

**Async Programming:**
- Use async/await for I/O-bound operations
- Implement proper error handling in async contexts
- Use asyncio.gather() for concurrent operations
- Avoid mixing sync and async code improperly
- Profile async performance separately

**Design Patterns:**
- Implement singleton pattern for configuration management
- Use factory patterns for object creation
- Apply observer pattern for event handling
- Use dependency injection for testability
- Implement proper separation of concerns

## Debugging and Troubleshooting

**Common Issues:**
- Memory leaks in long-running processes
- Performance bottlenecks in data processing
- Streamlit state management problems
- Type hint mismatches
- Configuration management errors

**Debugging Tools:**
- Use `pdb` for interactive debugging
- Apply logging for runtime behavior analysis
- Use profilers to identify performance issues
- Implement health checks for production monitoring
- Use memory profilers for memory leak detection

**Best Practices:**
- Write self-documenting code with clear variable names
- Implement comprehensive error messages
- Use assertions for debugging (remove in production)
- Create reproducible test cases for bugs
- Document known issues and workarounds

---

## Key Reminders for AI Assistant

**Always Prioritize:**
1. Code correctness and reliability
2. Performance optimization based on profiling
3. Type safety and clear documentation
4. Proper error handling and logging
5. Maintainable and readable code structure

**When Suggesting Improvements:**
- Provide specific, actionable recommendations
- Include performance implications
- Reference relevant sections of this guide
- Consider the scientific computing context
- Focus on production-ready solutions

**For Complex Changes:**
- Break down into smaller, testable components
- Provide migration strategies for existing code
- Include performance benchmarks where relevant
- Suggest appropriate testing strategies
- Consider backwards compatibility