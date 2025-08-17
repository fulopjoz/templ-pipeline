# Contributing to TEMPL Pipeline

Thank you for your interest in contributing to TEMPL Pipeline! This document provides guidelines for contributing to the project.

## Code Formatting

We use automated tools to maintain consistent code formatting:

### Setup Pre-commit Hooks
```bash
# Install pre-commit hooks for automatic formatting
pip install pre-commit
pre-commit install
```

### Manual Formatting
```bash
# Format code with Black
black templ_pipeline/ tests/ scripts/

# Sort imports with isort
isort templ_pipeline/ tests/ scripts/

# Check code quality with flake8
flake8 templ_pipeline/ tests/ scripts/
```

### Configuration
- **Black**: Line length 88, Python 3.9+ compatibility
- **isort**: Black profile, line length 88
- **flake8**: Line length 88, ignores E203, W503

## Development Setup

1. Clone the repository
2. Install dependencies: `pip install -e ".[dev]"`
3. Set up pre-commit hooks
4. Run tests: `pytest`

## Testing

- Run all tests: `pytest`
- Run specific test: `pytest tests/unit/core/test_embedding.py`
- Run with coverage: `pytest --cov=templ_pipeline`

## CI/CD

The project uses GitHub Actions for continuous integration:
- **CI**: Tests, linting, security checks
- **Citation**: Citation file management
- **Live Tests**: Manual test execution

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions
- Keep functions focused and small
- Use meaningful variable names

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and formatting tools
5. Submit a pull request with clear description

## Questions?

Feel free to open an issue or discussion for any questions about contributing.
