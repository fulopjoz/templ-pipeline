# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""
Script to add SPDX headers to all Python files in the TEMPL pipeline.

This script systematically adds SPDX copyright and license headers to all
Python source files to ensure FAIR compliance and proper licensing attribution.
"""

from pathlib import Path

SPDX_HEADER = """# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT"""


def has_spdx_header(content):
    """Check if file already has SPDX header."""
    lines = content.split("\n")
    if len(lines) < 2:
        return False

    # Check for SPDX header in first few lines
    for i in range(min(3, len(lines))):
        if (
            "SPDX-FileCopyrightText" in lines[i]
            or "SPDX-License-Identifier" in lines[i]
        ):
            return True
    return False


def add_spdx_header(content):
    """Add SPDX header to file content."""
    lines = content.split("\n")

    # Find the first non-empty, non-comment line
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            start_idx = i
            break

    # Insert SPDX header before the first content line
    new_lines = lines[:start_idx]
    new_lines.append(SPDX_HEADER)
    new_lines.extend(lines[start_idx:])

    return "\n".join(new_lines)


def process_file(file_path):
    """Process a single Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if has_spdx_header(content):
            print(f"  ✓ Already has SPDX header: {file_path}")
            return False

        new_content = add_spdx_header(content)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        print(f"  + Added SPDX header: {file_path}")
        return True

    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")
        return False


def main():
    """Main function to process all Python files."""
    project_root = Path(__file__).parent.parent
    python_files = list(project_root.rglob("*.py"))

    # Filter out files we don't want to modify
    exclude_patterns = [
        "__pycache__",
        ".git",
        ".templ",
        "venv",
        "env",
        "build",
        "dist",
        ".pytest_cache",
        "templ_pipeline.egg-info",
    ]

    filtered_files = []
    for file_path in python_files:
        if not any(pattern in str(file_path) for pattern in exclude_patterns):
            filtered_files.append(file_path)

    print(f"Found {len(filtered_files)} Python files to process")
    print("=" * 50)

    modified_count = 0
    for file_path in filtered_files:
        if process_file(file_path):
            modified_count += 1

    print("=" * 50)
    print(f"Modified {modified_count} files with SPDX headers")


if __name__ == "__main__":
    main()
