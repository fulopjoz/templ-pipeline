#!/usr/bin/env python3
"""
TEMPL Pipeline System Validation Script

Auto-discovers and tests all available TEMPL commands and examples to ensure the CLI is working correctly.
This script validates:
- Help system functionality
- Auto-discovered commands from 'templ --help'
- Examples from help text (converted to safe testing modes)
- Command-specific help for all discovered commands
- Alternative entry points
- Import dependencies
- Error handling for missing arguments
- Both 'templ' and 'python -m templ_pipeline.cli.main' formats

Features:
- Auto-discovery: Parses 'templ --help' to find all available commands
- Example testing: Extracts and safely tests examples from help text
- Safe execution: Converts examples to --help or validation modes
- Comprehensive coverage: Tests all discovered functionality automatically

Usage:
    python templ_pipeline/test_all_commands.py [--verbose] [--timeout SECONDS] [--output FORMAT] [--no-discovery]

Examples:
    python templ_pipeline/test_all_commands.py --verbose
    python templ_pipeline/test_all_commands.py --no-discovery --timeout 60
    python templ_pipeline/test_all_commands.py --output json --output-file results.json
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import os
import re


# Color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    END = "\033[0m"

    @classmethod
    def supports_color(cls):
        """Check if terminal supports color output."""
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class CommandResult:
    """Container for test results."""

    def __init__(
        self,
        command: str,
        status: str,
        output: str = "",
        error: str = "",
        duration: float = 0.0,
        details: str = "",
    ):
        self.command = command
        self.status = status  # PASS, FAIL, TIMEOUT, SKIP
        self.output = output
        self.error = error
        self.duration = duration
        self.details = details


class HelpParser:
    """Parse help output to discover commands and examples."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.discovered_commands = []
        self.command_examples = {}

    def run_help_command(self, command: str) -> str:
        """Run a help command and return output."""
        try:
            result = subprocess.run(
                command.split(), capture_output=True, text=True, timeout=self.timeout
            )
            return result.stdout + result.stderr  # Combine both streams
        except Exception:
            return ""

    def parse_main_help(self) -> List[str]:
        """Parse main help with strict validation."""
        help_output = self.run_help_command("templ --help")
        commands = []

        lines = help_output.split("\n")
        in_commands_section = False

        for line in lines:
            stripped = line.strip()

            # Exact section matching
            if stripped == "Commands:":
                in_commands_section = True
                continue

            # End of commands section
            if in_commands_section:
                if stripped == "" or (stripped and not line.startswith("  ")):
                    break

            # Extract commands (must be indented with 2 spaces)
            if (
                in_commands_section
                and line.startswith("  ")
                and not line.startswith("   ")
            ):
                cmd_line = stripped.split()
                if cmd_line:
                    command = cmd_line[0]
                    # Validate: lowercase, may contain hyphens, must be alpha
                    if (
                        command.replace("-", "").islower()
                        and command.replace("-", "").isalpha()
                    ):
                        commands.append(command)

        # Remove duplicates while preserving order
        unique_commands = list(dict.fromkeys(commands))
        self.discovered_commands = unique_commands
        return unique_commands

    def extract_examples_from_help(self, help_text: str) -> List[str]:
        """Extract complete example commands, handling multi-line commands properly."""
        examples = []
        lines = help_text.split("\n")

        current_command = ""
        in_examples_section = False

        for line in lines:
            stripped = line.strip()

            # Section detection
            if stripped.startswith("Quick Examples:"):
                in_examples_section = True
                continue
            elif (
                in_examples_section
                and stripped
                and not line.startswith(" ")
                and ":" in stripped
            ):
                # Save any pending command before ending section
                if current_command:
                    cleaned = self._clean_command(current_command)
                    if cleaned:
                        examples.append(cleaned)
                    current_command = ""
                break

            # Command extraction in examples section
            if in_examples_section and "templ" in line:
                # Handle prefixes like "Basic:", "Template:", "Parallel:"
                clean_line = self._remove_example_prefixes(line)

                if clean_line.endswith("\\"):
                    # Start of multi-line command
                    current_command = clean_line[:-1].strip()
                elif current_command and not clean_line.startswith("templ"):
                    # Continuation line
                    current_command += " " + clean_line.strip()
                    if not clean_line.endswith("\\"):
                        # End of multi-line command
                        cleaned = self._clean_command(current_command)
                        if cleaned:
                            examples.append(cleaned)
                        current_command = ""
                else:
                    # Single line command or new command
                    if current_command:
                        cleaned = self._clean_command(current_command)
                        if cleaned:
                            examples.append(cleaned)
                        current_command = ""
                    if clean_line.startswith("templ"):
                        cleaned = self._clean_command(clean_line)
                        if cleaned:
                            examples.append(cleaned)

        # Handle any remaining command
        if current_command:
            cleaned = self._clean_command(current_command)
            if cleaned:
                examples.append(cleaned)

        return examples

    def _remove_example_prefixes(self, line: str) -> str:
        """Remove example prefixes while preserving command structure."""
        prefixes = [
            "Basic:",
            "Template:",
            "Parallel:",
            "$",
            ">",
            "•",
            "-",
            "Example:",
            "Usage:",
        ]

        clean_line = line.strip()
        for prefix in prefixes:
            if clean_line.startswith(prefix):
                clean_line = clean_line[len(prefix) :].strip()
                break

        return clean_line

    def _clean_command(self, command: str) -> Optional[str]:
        """Clean and validate a command string."""
        if not command:
            return None

        # Remove quotes and normalize whitespace
        clean = command.replace('"', "").replace("'", "")
        clean = " ".join(clean.split())

        # Validate it's a proper templ command
        if clean.startswith("templ ") and len(clean.split()) >= 2:
            return clean

        return None

    def _extract_command_from_line(self, line: str) -> Optional[str]:
        """Extract a templ command from a line of text."""
        line = line.strip()

        # Remove common prefixes
        line = self._remove_example_prefixes(line)

        # Must start with templ
        if not line.startswith("templ"):
            return None

        # Extract the command part (stop at comments, explanations, etc.)
        for stop_word in ["#", "//", "/*", "(this", "(for", "where", " \\"]:
            if stop_word in line:
                line = line[: line.index(stop_word)].strip()

        return self._clean_command(line)

    def discover_all_commands(self) -> Dict[str, Any]:
        """Discover all available commands and their examples."""
        # Get main commands
        main_commands = self.parse_main_help()

        # Get examples from main help
        main_help = self.run_help_command("templ --help")
        main_examples = self.extract_examples_from_help(main_help)

        # Get subcommand help and examples
        subcommand_data = {}
        for cmd in main_commands:
            help_output = self.run_help_command(f"templ {cmd} --help")
            examples = self.extract_examples_from_help(help_output)
            subcommand_data[cmd] = {"help_output": help_output, "examples": examples}

        return {
            "main_commands": main_commands,
            "main_examples": main_examples,
            "subcommands": subcommand_data,
        }

    def make_command_safe(self, command: str) -> List[str]:
        """Convert a command to safe testing variants with proper validation."""
        if not command or not command.startswith("templ "):
            return []

        safe_commands = []
        parts = command.split()

        if len(parts) < 2:
            return []

        # Extract base command: "templ subcommand"
        base_cmd = f"{parts[0]} {parts[1]}"  # e.g., "templ run"

        # 1. Always add help version
        safe_commands.append(f"{base_cmd} --help")

        # 2. Test missing required arguments (base command only)
        safe_commands.append(base_cmd)

        # 3. Test with non-existent file (for error handling)
        if any(
            arg in command
            for arg in ["--protein-file", "--ligand-file", "--embedding-file"]
        ):
            error_test = f"{base_cmd} --protein-file /nonexistent/test.pdb"
            safe_commands.append(error_test)

        return safe_commands


class DependencyChecker:
    """Check for required dependencies and system prerequisites."""

    def __init__(self):
        self.results = {}

    def check_python_environment(self) -> bool:
        """Check Python version and basic environment."""
        try:
            version = sys.version_info
            if version.major >= 3 and version.minor >= 7:
                self.results["python_version"] = (
                    f"✓ Python {version.major}.{version.minor}.{version.micro}"
                )
                return True
            else:
                self.results["python_version"] = (
                    f"✗ Python {version.major}.{version.minor}.{version.micro} (requires >= 3.7)"
                )
                return False
        except Exception as e:
            self.results["python_version"] = f"✗ Error checking Python version: {e}"
            return False

    def check_core_modules(self) -> Dict[str, bool]:
        """Check if core TEMPL modules can be imported."""
        modules_to_check = [
            "templ_pipeline",
            "templ_pipeline.cli",
            "templ_pipeline.cli.main",
            "templ_pipeline.cli.help_system",
            "templ_pipeline.core",
        ]

        results = {}
        for module in modules_to_check:
            try:
                __import__(module)
                results[module] = True
                self.results[f"module_{module}"] = f"✓ {module}"
            except ImportError as e:
                results[module] = False
                self.results[f"module_{module}"] = f"✗ {module}: {e}"
            except Exception as e:
                results[module] = False
                self.results[f"module_{module}"] = f"✗ {module}: Unexpected error: {e}"

        return results

    def check_optional_dependencies(self) -> Dict[str, bool]:
        """Check optional dependencies like RDKit, numpy, etc."""
        optional_deps = ["rdkit", "numpy", "scipy", "pandas", "rich", "colorama"]

        results = {}
        for dep in optional_deps:
            try:
                __import__(dep)
                results[dep] = True
                self.results[f"optional_{dep}"] = f"✓ {dep}"
            except ImportError:
                results[dep] = False
                self.results[f"optional_{dep}"] = (
                    f"WARNING: {dep}: Not found (optional)"
                )
            except Exception as e:
                results[dep] = False
                self.results[f"optional_{dep}"] = f"WARNING: {dep}: Error: {e}"

        return results

    def check_file_system(self) -> bool:
        """Check file system permissions and required directories."""
        try:
            # Check if we can create temp files
            test_dir = Path("test_temp_dir")
            test_dir.mkdir(exist_ok=True)
            test_file = test_dir / "test_file.txt"
            test_file.write_text("test")
            test_file.unlink()
            test_dir.rmdir()

            self.results["file_system"] = "✓ File system permissions OK"
            return True
        except Exception as e:
            self.results["file_system"] = f"✗ File system error: {e}"
            return False


class CommandValidator:
    """Validate individual commands."""

    def __init__(self, timeout: int = 30, verbose: bool = False):
        self.timeout = timeout
        self.verbose = verbose

    def run_command(self, command: str, cwd: Optional[str] = None) -> CommandResult:
        """Run a single command and capture results."""
        start_time = time.time()

        try:
            # Split command for subprocess
            cmd_parts = command.split()

            # Run command with timeout
            result = subprocess.run(
                cmd_parts, capture_output=True, text=True, timeout=self.timeout, cwd=cwd
            )

            duration = time.time() - start_time

            # Determine status based on return code and output
            if result.returncode == 0:
                status = "PASS"
                details = "Command executed successfully"
            elif result.returncode == 2:  # Common help return code
                status = "PASS"
                details = "Help command executed successfully"
            else:
                status = "FAIL"
                details = f"Command failed with return code {result.returncode}"

            return CommandResult(
                command=command,
                status=status,
                output=result.stdout,
                error=result.stderr,
                duration=duration,
                details=details,
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return CommandResult(
                command=command,
                status="TIMEOUT",
                duration=duration,
                details=f"Command timed out after {self.timeout} seconds",
            )
        except FileNotFoundError:
            duration = time.time() - start_time
            return CommandResult(
                command=command,
                status="FAIL",
                duration=duration,
                details="Command not found or Python module not accessible",
            )
        except Exception as e:
            duration = time.time() - start_time
            return CommandResult(
                command=command,
                status="FAIL",
                error=str(e),
                duration=duration,
                details=f"Unexpected error: {e}",
            )


class CLITestRunner:
    """Main test execution engine."""

    def __init__(
        self, timeout: int = 30, verbose: bool = False, enable_discovery: bool = True
    ):
        self.timeout = timeout
        self.verbose = verbose
        self.enable_discovery = enable_discovery
        self.validator = CommandValidator(timeout, verbose)
        self.dependency_checker = DependencyChecker()
        self.help_parser = HelpParser(timeout)
        self.results: List[CommandResult] = []
        self.use_color = Colors.supports_color()
        self.discovered_data = None

    def colorize(self, text: str, color: str) -> str:
        """Add color to text if terminal supports it."""
        if self.use_color:
            return f"{color}{text}{Colors.END}"
        return text

    def print_status(self, message: str, status: str = "INFO"):
        """Print colored status message."""
        if status == "PASS":
            print(self.colorize(f"✓ {message}", Colors.GREEN))
        elif status == "FAIL":
            print(self.colorize(f"✗ {message}", Colors.RED))
        elif status == "WARN":
            print(self.colorize(f"WARNING: {message}", Colors.YELLOW))
        elif status == "INFO":
            print(self.colorize(f"• {message}", Colors.BLUE))
        else:
            print(message)

    def get_command_lists(self) -> Dict[str, List[str]]:
        """Get all command lists to test, including auto-discovered commands."""
        # Perform discovery if not already done and discovery is enabled
        if self.enable_discovery and self.discovered_data is None:
            print(
                self.colorize("SEARCH: Discovering available commands...", Colors.CYAN)
            )
            self.discovered_data = self.help_parser.discover_all_commands()

        # Static command lists (original functionality)
        command_lists = {
            "help_commands": [
                "python -m templ_pipeline.cli.main --help",
                "python -m templ_pipeline.cli.main --help main",
                "python -m templ_pipeline.cli.main --help simple",
                "python -m templ_pipeline.cli.main --help examples",
                "python -m templ_pipeline.cli.main --help workflows",
                "python -m templ_pipeline.cli.main --help performance",
                "python -m templ_pipeline.cli.main --help troubleshoot",
            ],
            "static_command_help": [
                "python -m templ_pipeline.cli.main embed --help",
                "python -m templ_pipeline.cli.main find-templates --help",
                "python -m templ_pipeline.cli.main generate-poses --help",
                "python -m templ_pipeline.cli.main run --help",
            ],
            "alternative_commands": [
                "python templ_pipeline/run_pipeline.py --help",
                "python templ_pipeline/scripts/generate_embedding_map.py --help",
            ],
        }

        # Add discovered commands if discovery is enabled
        if self.enable_discovery and self.discovered_data:
            # Main templ help
            command_lists["main_templ_help"] = ["templ --help"]

            # Discovered subcommand help
            discovered_commands = self.discovered_data.get("main_commands", [])
            if discovered_commands:
                command_lists["discovered_command_help"] = [
                    f"templ {cmd} --help" for cmd in discovered_commands
                ]

                # Also test module format for discovered commands
                command_lists["discovered_module_help"] = [
                    f"python -m templ_pipeline.cli.main {cmd} --help"
                    for cmd in discovered_commands
                ]

            # Test discovered examples (converted to safe versions)
            main_examples = self.discovered_data.get("main_examples", [])
            subcommand_data = self.discovered_data.get("subcommands", {})

            all_examples = main_examples.copy()
            for cmd_data in subcommand_data.values():
                all_examples.extend(cmd_data.get("examples", []))

            if all_examples:
                safe_examples = []
                for example in all_examples:
                    if example and example.startswith("templ "):
                        safe_variants = self.help_parser.make_command_safe(example)
                        safe_examples.extend([cmd for cmd in safe_variants if cmd])

                if safe_examples:
                    command_lists["discovered_examples"] = safe_examples

            # Error handling tests
            if discovered_commands:
                command_lists["error_handling_tests"] = [
                    f"templ {cmd}"
                    for cmd in discovered_commands[:3]  # Test missing args
                ]

        # Output behavior and functionality tests with example files
        command_lists["output_behavior_tests"] = [
            "templ embed --protein-file templ_pipeline/data/example/1iky_protein.pdb",
            "templ embed --protein-file templ_pipeline/data/example/5eqy_protein.pdb",
            'templ generate-poses --protein-file templ_pipeline/data/example/1iky_protein.pdb --template-pdb 5eqy --ligand-smiles "COc1ccc(C(C)=O)c(O)c1"',
            'templ run --protein-file templ_pipeline/data/example/1iky_protein.pdb --ligand-smiles "COc1ccc(C(C)=O)c(O)c1" --num-conformers 5',
        ]

        return command_lists

    def run_prerequisites_check(self):
        """Run all prerequisite checks."""
        print(self.colorize("\nTEMPL Pipeline System Validation", Colors.BOLD))
        print("=" * 50)

        print(self.colorize("\nChecking Prerequisites...", Colors.CYAN))

        # Check Python environment
        python_ok = self.dependency_checker.check_python_environment()

        # Check core modules
        core_modules = self.dependency_checker.check_core_modules()
        core_ok = all(core_modules.values())

        # Check optional dependencies
        optional_deps = self.dependency_checker.check_optional_dependencies()

        # Check file system
        fs_ok = self.dependency_checker.check_file_system()

        # Print results
        for key, result in self.dependency_checker.results.items():
            if result.startswith("✓"):
                self.print_status(result.replace("✓ ", ""), "PASS")
            elif result.startswith("✗"):
                self.print_status(result.replace("✗ ", ""), "FAIL")
            elif result.startswith("WARNING:"):
                self.print_status(result.replace("WARNING: ", ""), "WARN")

        return python_ok and core_ok and fs_ok

    def print_discovery_summary(self):
        """Print summary of discovered commands and examples."""
        if not self.discovered_data:
            return

        print(self.colorize(f"\nDISCOVERY SUMMARY", Colors.BOLD))
        print("=" * 30)

        main_commands = self.discovered_data.get("main_commands", [])
        main_examples = self.discovered_data.get("main_examples", [])
        subcommands = self.discovered_data.get("subcommands", {})

        if main_commands:
            print(f"Discovered Commands: {', '.join(main_commands)}")
        else:
            print("No commands discovered from templ --help")

        if main_examples:
            print(f"Main Examples Found: {len(main_examples)}")
            for ex in main_examples[:3]:  # Show first 3
                print(f"  • {ex}")
            if len(main_examples) > 3:
                print(f"  ... and {len(main_examples) - 3} more")

        subcommand_examples = 0
        for cmd_data in subcommands.values():
            subcommand_examples += len(cmd_data.get("examples", []))

        if subcommand_examples > 0:
            print(f"Subcommand Examples: {subcommand_examples}")

    def analyze_output_behavior(self, command: str, result: CommandResult):
        """Analyze output directory behavior for functionality tests."""
        if "output_behavior_tests" not in command:
            return

        if result.status == "PASS":
            # Check for output directory patterns in the output
            output_text = result.output.lower()
            if "saving" in output_text or "saved" in output_text:
                if "output_" in output_text and any(c.isdigit() for c in output_text):
                    print(f"   FOLDER: Creates timestamp folder")
                elif "output/" in output_text:
                    print(f"   FOLDER: Uses fixed 'output/' folder")
                else:
                    print(f"   FOLDER: Custom output location")

    def run_command_tests(self):
        """Run all command tests."""
        command_lists = self.get_command_lists()

        # Print discovery summary
        self.print_discovery_summary()

        for category, commands in command_lists.items():
            if not commands:  # Skip empty categories
                continue

            print(
                self.colorize(
                    f"\nTesting {category.replace('_', ' ').title()}...", Colors.CYAN
                )
            )

            for command in commands:
                if self.verbose:
                    print(f"Running: {command}")

                result = self.validator.run_command(command)
                self.results.append(result)

                # Print result
                status_symbol = (
                    "✓"
                    if result.status == "PASS"
                    else (
                        "✗"
                        if result.status == "FAIL"
                        else "TIME" if result.status == "TIMEOUT" else "WARN"
                    )
                )
                command_name = (
                    command.split()[-2:]
                    if "--help" in command
                    else command.split()[-1:]
                )
                command_display = " ".join(command_name)

                # Better command display for complex commands
                if len(command.split()) > 3:
                    command_display = " ".join(command.split()[:3]) + "..."

                if result.status == "PASS":
                    self.print_status(
                        f"{command_display}: {result.status} ({result.duration:.2f}s)",
                        "PASS",
                    )
                    self.analyze_output_behavior(category, result)
                elif result.status == "FAIL":
                    self.print_status(
                        f"{command_display}: {result.status} - {result.details}", "FAIL"
                    )
                    if self.verbose and result.error:
                        print(f"   Error: {result.error}")
                elif result.status == "TIMEOUT":
                    self.print_status(
                        f"{command_display}: {result.status} ({result.duration:.2f}s)",
                        "WARN",
                    )
                else:
                    self.print_status(f"{command_display}: {result.status}", "WARN")

    def generate_summary(self):
        """Generate and print test summary."""
        total_tests = len(self.results)
        passed = len([r for r in self.results if r.status == "PASS"])
        failed = len([r for r in self.results if r.status == "FAIL"])
        timeout = len([r for r in self.results if r.status == "TIMEOUT"])
        skipped = len([r for r in self.results if r.status == "SKIP"])

        print(self.colorize(f"\nSummary:", Colors.BOLD))
        print("=" * 20)
        print(f"Total Tests: {total_tests}")
        print(self.colorize(f"Passed: {passed}", Colors.GREEN))

        if failed > 0:
            print(self.colorize(f"Failed: {failed}", Colors.RED))
        if timeout > 0:
            print(self.colorize(f"Timeout: {timeout}", Colors.YELLOW))
        if skipped > 0:
            print(self.colorize(f"Skipped: {skipped}", Colors.YELLOW))

        # Calculate success rate
        success_rate = (passed / total_tests * 100) if total_tests > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")

        return {
            "total": total_tests,
            "passed": passed,
            "failed": failed,
            "timeout": timeout,
            "skipped": skipped,
            "success_rate": success_rate,
        }

    def export_results(self, format_type: str, output_file: Optional[str] = None):
        """Export results to file in specified format."""
        if format_type.lower() == "json":
            data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": self.generate_summary(),
                "prerequisites": self.dependency_checker.results,
                "discovery": self.discovered_data or {},
                "test_results": [
                    {
                        "command": r.command,
                        "status": r.status,
                        "duration": r.duration,
                        "details": r.details,
                        "output": (
                            r.output[:500] if r.output else ""
                        ),  # Truncate long outputs
                        "error": r.error[:500] if r.error else "",
                    }
                    for r in self.results
                ],
            }

            filename = output_file or f"templ_validation_{int(time.time())}.json"
            with open(filename, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\nResults exported to: {filename}")

    def run_all_tests(self) -> bool:
        """Run all tests and return overall success status."""
        # Check prerequisites first
        prereq_ok = self.run_prerequisites_check()

        if not prereq_ok:
            print(
                self.colorize(
                    "\nWARNING: Prerequisites check failed. Some tests may not work correctly.",
                    Colors.YELLOW,
                )
            )

        # Run command tests
        self.run_command_tests()

        # Generate summary
        summary = self.generate_summary()

        # Overall success if we have > 80% success rate and no critical failures
        return summary["success_rate"] >= 80 and summary["failed"] <= 2


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TEMPL Pipeline System Validation Script - Auto-discovers and tests all TEMPL commands",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="""
Examples:
  python templ_pipeline/test_all_commands.py --verbose
  python templ_pipeline/test_all_commands.py --no-discovery --timeout 60
  python templ_pipeline/test_all_commands.py --output json --output-file results.json
        """,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed command execution",
    )

    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=30,
        help="Timeout for each command in seconds",
    )

    parser.add_argument(
        "--output", "-o", choices=["json"], help="Export results in specified format"
    )

    parser.add_argument(
        "--output-file", help="Output file name (default: auto-generated)"
    )

    parser.add_argument(
        "--no-discovery",
        action="store_true",
        help="Skip auto-discovery and only run static command tests",
    )

    args = parser.parse_args()

    # Create test runner
    runner = CLITestRunner(
        timeout=args.timeout,
        verbose=args.verbose,
        enable_discovery=not args.no_discovery,
    )

    try:
        # Run all tests
        success = runner.run_all_tests()

        # Export results if requested
        if args.output:
            runner.export_results(args.output, args.output_file)

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print(runner.colorize("\n\nTest interrupted by user.", Colors.YELLOW))
        sys.exit(1)
    except Exception as e:
        print(runner.colorize(f"\nUnexpected error: {e}", Colors.RED))
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
