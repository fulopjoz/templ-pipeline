# Gemini Customization

This file provides instructions for Gemini on how to interact with the `templ_pipeline` project.

## General Principles

*   **Project Structure:** The core application logic is within the `templ_pipeline` directory. Tests are in the `tests` directory.
*   **Dependencies:** The project uses `pip` for dependency management. The main dependencies are in `requirements.txt`, and development dependencies are in `requirements-dev.txt`.
*   **Testing:** The project uses `pytest` for testing. To run the tests, use the `run_tests.py` script.
*   **Code Style:** The project follows the PEP 8 style guide. Use a linter to check for style issues.

## Common Tasks

*   **Running the pipeline:** The `run_pipeline.py` script is the main entry point for running the pipeline.
*   **Running the Streamlit app:** The `run_streamlit_app.py` script starts the Streamlit application.
*   **Adding new dependencies:** Add new dependencies to the appropriate `requirements.txt` file and then run `pip install -r requirements.txt` or `pip install -r requirements-dev.txt`.




## 🧱 Project Structure

```
my_project/
├── pyproject.toml
├── README.md
├── .gitignore
├── src/
│   └── my_package/
│       ├── __init__.py
│       ├── core/
│       ├── utils/
│       └── cli.py
├── tests/
│   ├── conftest.py
│   └── unit/
├── scripts/
└── docs/
```

- Use **`src/` layout** for test isolation and cleaner installs.
- Separate **unit** and **integration** tests under `tests/`.



## 🎨 Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/), enforced via `black`, `isort`, and `flake8`.
- Use **4 spaces**, limit lines to **88 characters**.
- Organize imports:
  1. Stdlib
  2. Third-party
  3. Local

Example:
```python
import os
import numpy as np
from src.my_package.utils import helper_function
```



## ✍️ Naming Conventions

```python
# snake_case for variables/functions
def get_user_profile(): ...

# PascalCase for classes
class DataProcessor: ...

# UPPER_CASE for constants
MAX_RETRIES = 5
```



## 🧠 Type Hints (≥ Python 3.10)

```python
from typing import Dict, Any

def process(data: list[Dict[str, Any]]) -> str | None:
    ...
```

Use `Literal`, `TypeAlias`, `Protocol` for clarity in APIs and class contracts.



## 🧾 Docstrings

Follow [PEP 257](https://peps.python.org/pep-0257/) + NumPy or Google style:

```python
def calculate_area(length: float, width: float) -> float:
    """Calculate area of a rectangle.

    Args:
        length: Rectangle length.
        width: Rectangle width.

    Returns:
        Computed area.
    """
```


## ❗ Error Handling

- Handle **specific exceptions**.
- Use `raise ... from e` for context.
- Define custom exceptions for domain-specific errors.

```python
class ValidationError(Exception): ...

try:
    ...
except FileNotFoundError as e:
    raise ConfigLoadError(...) from e
```



## ✅ Testing with Pytest

- Use `pytest`, fixtures in `conftest.py`, and `@pytest.mark.parametrize`.

Example:
```python
@pytest.mark.parametrize("input,expected", [(1, 2), (2, 3)])
def test_increment(input, expected):
    assert increment(input) == expected
```

Run tests:
```bash
pytest -v --cov=src
```



## 🪵 Logging

Use `logging.config.dictConfig` with both console and file handlers.

```python
import logging

logger = logging.getLogger(__name__)
logger.info("Processing started")
```



## 📦 Dependencies

- Use `pyproject.toml` + [uv](https://github.com/astral-sh/uv) for dependency management.

Minimal example:
```toml
[project]
name = "gemini"
dependencies = ["pydantic", "click"]
requires-python = ">=3.10"

[tool.black]
line-length = 88
```

## 🚀 Performance

```python
from functools import lru_cache

@lru_cache
def fib(n: int) -> int: ...
```

Use:
- Generators for large data
- List comprehensions
- Context managers


## 🔐 Security Basics

```python
import secrets

def generate_token() -> str:
    return secrets.token_urlsafe(32)
```

- Always validate file paths, sanitize user input.
- Never hardcode secrets or credentials.


## 🧪 Dev Commands

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt
black src/ tests/
pytest --cov=src
```


## 🧠 Summary

- ✅ Modern structure (`src`, `pyproject.toml`, `tests`)
- ✅ Auto-format, lint, type-check
- ✅ Full type hints + docstrings
- ✅ Domain-driven exceptions + logging
- ✅ Pytest-based test suite
- ✅ Secure, readable, maintainable code

---

> _“Readability counts. In 2025, so does maintainability.” – The Zen of Python, revised._
