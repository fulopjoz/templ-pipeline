# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""Output validation utilities for CLI tests."""

import re
from typing import List


def validate_help_output(output: str, expected_keywords: List[str]) -> bool:
    """Validate that help output contains expected keywords."""
    for keyword in expected_keywords:
        if keyword not in output:
            return False
    return True


def validate_ascii_banner(output: str, pattern: str) -> bool:
    """Validate ASCII banner appears in output."""
    return bool(re.search(pattern, output, re.DOTALL | re.IGNORECASE))


def validate_command_syntax(command: str) -> bool:
    """Basic validation of command syntax."""
    # Check for basic command structure
    if not command.startswith("templ "):
        return False

    # Check for balanced quotes
    single_quotes = command.count("'")
    double_quotes = command.count('"')

    return single_quotes % 2 == 0 and double_quotes % 2 == 0


def validate_no_error_patterns(output: str) -> bool:
    """Check output doesn't contain common error patterns."""
    error_patterns = [
        "Traceback",
        "ImportError",
        "ModuleNotFoundError",
        "AttributeError",
        "TypeError",
    ]

    for pattern in error_patterns:
        if pattern in output:
            return False
    return True


def extract_execution_time_from_output(output: str) -> float:
    """Extract execution time if present in output."""
    time_pattern = r"Execution time: ([\d.]+)s"
    match = re.search(time_pattern, output)
    return float(match.group(1)) if match else 0.0
