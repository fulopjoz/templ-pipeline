# Contributing to TEMPL Pipeline

Thank you for your interest in contributing to TEMPL Pipeline! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct.

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker
- Provide a clear description of the problem
- Include steps to reproduce the issue
- Specify your environment (OS, Python version, etc.)

### Suggesting Enhancements

- Use the GitHub issue tracker with the "enhancement" label
- Describe the proposed feature clearly
- Explain the benefits and use cases

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** following the coding standards below
4. **Add tests** for new functionality
5. **Run tests** to ensure everything works
6. **Commit your changes** with clear commit messages
7. **Push to your fork** and submit a pull request

## Coding Standards

### Python Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Keep functions focused and well-documented
- Add docstrings for all public functions and classes

### File Headers

All Python files must include SPDX headers:

```python
# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
```

### Commit Messages

Use clear, descriptive commit messages:
- Start with a verb in present tense
- Keep the first line under 50 characters
- Provide additional details in the body if needed

Example:
```
Add SPDX headers to all Python files

- Add copyright and license headers to 128 Python files
- Ensure FAIR compliance for academic citation
- Follow REUSE software guidelines
```

## Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting
- Run the test suite: `python -m pytest tests/`

## Documentation

- Update README.md if adding new features
- Add docstrings for new functions and classes
- Update examples if API changes

## License

By contributing to TEMPL Pipeline, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions about contributing, please open an issue or contact the maintainers.
