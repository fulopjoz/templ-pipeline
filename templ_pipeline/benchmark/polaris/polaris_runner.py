# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
import sys
import os

from .benchmark import main as _benchmark_main


def main(argv=None):
    """Entry point for the Polaris benchmark (TEMPL version)."""
    argv = argv or sys.argv[1:]
    return _benchmark_main(argv)


if __name__ == "__main__":
    sys.exit(main())
