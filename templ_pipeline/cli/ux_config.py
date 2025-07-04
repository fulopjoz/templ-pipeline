"""
TEMPL CLI User Experience Configuration

This module implements the Smart Progressive Interface design for the TEMPL CLI,
providing adaptive complexity management, intelligent defaults, and user preference learning.

Key Features:
- Progressive complexity revelation based on user experience level
- Smart verbosity controls with context-aware output
- User preference learning and adaptation
- Backward compatibility with existing CLI patterns

Design Philosophy:
- Beginners get minimal, essential options with clear guidance
- Intermediate users see commonly used options grouped logically
- Advanced users have access to all options with expert shortcuts
- System learns from usage patterns to adapt interface complexity
"""

import os
import json
import logging
from typing import Dict, List, Optional, Set, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class VerbosityLevel(Enum):
    """Output verbosity levels for the CLI."""

    MINIMAL = "minimal"  # Only essential output, errors, and final results
    NORMAL = "normal"  # Standard user-friendly output with progress
    DETAILED = "detailed"  # Comprehensive output with technical details
    DEBUG = "debug"  # Full debugging information


class ExperienceLevel(Enum):
    """User experience levels for progressive interface adaptation."""

    BEGINNER = "beginner"  # New to TEMPL, needs guidance
    INTERMEDIATE = "intermediate"  # Familiar with basic operations
    ADVANCED = "advanced"  # Expert user, wants full control
    AUTO = "auto"  # System determines based on usage patterns


@dataclass
class UserPreferences:
    """User preferences for CLI behavior and output."""

    experience_level: ExperienceLevel = ExperienceLevel.AUTO
    default_verbosity: VerbosityLevel = VerbosityLevel.NORMAL
    show_progress_bars: bool = True
    show_performance_hints: bool = True
    preferred_output_format: str = "user-friendly"  # or "machine-readable"
    auto_detect_hardware: bool = True
    remember_recent_args: bool = True
    max_recent_args: int = 10

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "experience_level": self.experience_level.value,
            "default_verbosity": self.default_verbosity.value,
            "show_progress_bars": self.show_progress_bars,
            "show_performance_hints": self.show_performance_hints,
            "preferred_output_format": self.preferred_output_format,
            "auto_detect_hardware": self.auto_detect_hardware,
            "remember_recent_args": self.remember_recent_args,
            "max_recent_args": self.max_recent_args,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "UserPreferences":
        """Create from dictionary loaded from JSON."""
        return cls(
            experience_level=ExperienceLevel(data.get("experience_level", "auto")),
            default_verbosity=VerbosityLevel(data.get("default_verbosity", "normal")),
            show_progress_bars=data.get("show_progress_bars", True),
            show_performance_hints=data.get("show_performance_hints", True),
            preferred_output_format=data.get(
                "preferred_output_format", "user-friendly"
            ),
            auto_detect_hardware=data.get("auto_detect_hardware", True),
            remember_recent_args=data.get("remember_recent_args", True),
            max_recent_args=data.get("max_recent_args", 10),
        )


@dataclass
class UsagePattern:
    """Track user usage patterns for adaptive interface."""

    command_usage_count: Dict[str, int]
    argument_usage_frequency: Dict[str, int]
    error_patterns: List[str]
    successful_workflows: List[str]
    total_commands: int

    def calculate_experience_level(self) -> ExperienceLevel:
        """Determine user experience level based on usage patterns."""
        if self.total_commands < 5:
            return ExperienceLevel.BEGINNER
        elif self.total_commands < 20:
            return ExperienceLevel.INTERMEDIATE
        else:
            # Check for advanced usage patterns
            advanced_commands = {"benchmark", "embed", "find-templates"}
            advanced_usage = sum(
                self.command_usage_count.get(cmd, 0) for cmd in advanced_commands
            )

            if advanced_usage > 5 or len(self.argument_usage_frequency) > 15:
                return ExperienceLevel.ADVANCED
            else:
                return ExperienceLevel.INTERMEDIATE


class ArgumentComplexity(Enum):
    """Complexity levels for CLI arguments."""

    ESSENTIAL = "essential"  # Core arguments all users need
    COMMON = "common"  # Frequently used by intermediate users
    ADVANCED = "advanced"  # Expert-level options
    DEBUG = "debug"  # Debugging and development options


@dataclass
class ArgumentGroup:
    """Group CLI arguments by complexity and usage context."""

    name: str
    complexity: ArgumentComplexity
    arguments: List[str]
    description: str
    show_in_help: bool = True


class TEMPLUXConfig:
    """Main UX configuration manager for TEMPL CLI."""

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize UX configuration manager.

        Args:
            config_dir: Directory to store user preferences (default: ~/.templ)
        """
        self.config_dir = Path(config_dir or Path.home() / ".templ")
        self.config_dir.mkdir(exist_ok=True)

        self.preferences_file = self.config_dir / "preferences.json"
        self.usage_file = self.config_dir / "usage_patterns.json"

        # Load user preferences and usage patterns
        self.preferences = self._load_preferences()
        self.usage_patterns = self._load_usage_patterns()

        # Define argument groups for progressive revelation
        self._setup_argument_groups()

    def _load_preferences(self) -> UserPreferences:
        """Load user preferences from file."""
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file) as f:
                    data = json.load(f)
                return UserPreferences.from_dict(data)
            except Exception as e:
                logger.debug(f"Failed to load preferences: {e}")

        return UserPreferences()

    def _load_usage_patterns(self) -> UsagePattern:
        """Load usage patterns from file."""
        if self.usage_file.exists():
            try:
                with open(self.usage_file) as f:
                    data = json.load(f)
                return UsagePattern(
                    command_usage_count=data.get("command_usage_count", {}),
                    argument_usage_frequency=data.get("argument_usage_frequency", {}),
                    error_patterns=data.get("error_patterns", []),
                    successful_workflows=data.get("successful_workflows", []),
                    total_commands=data.get("total_commands", 0),
                )
            except Exception as e:
                logger.debug(f"Failed to load usage patterns: {e}")

        return UsagePattern(
            command_usage_count={},
            argument_usage_frequency={},
            error_patterns=[],
            successful_workflows=[],
            total_commands=0,
        )

    def _setup_argument_groups(self):
        """Define argument groups for progressive complexity revelation."""
        self.argument_groups = {
            # Essential arguments - shown to all users
            "essential": [
                ArgumentGroup(
                    "core_input",
                    ArgumentComplexity.ESSENTIAL,
                    ["protein-file", "protein-pdb-id", "ligand-smiles", "ligand-file"],
                    "Core input options (always shown)",
                ),
                ArgumentGroup(
                    "basic_control",
                    ArgumentComplexity.ESSENTIAL,
                    ["help", "version", "output-dir"],
                    "Basic control options",
                ),
            ],
            # Common arguments - shown to intermediate+ users
            "common": [
                ArgumentGroup(
                    "template_control",
                    ArgumentComplexity.COMMON,
                    ["num-templates", "num-conformers", "template-pdb"],
                    "Template and conformer settings",
                ),
                ArgumentGroup(
                    "performance",
                    ArgumentComplexity.COMMON,
                    ["workers", "embedding-file"],
                    "Performance optimization",
                ),
            ],
            # Advanced arguments - shown to expert users
            "advanced": [
                ArgumentGroup(
                    "fine_tuning",
                    ArgumentComplexity.ADVANCED,
                    ["similarity-threshold", "exclude-uniprot-file", "chain", "run-id"],
                    "Fine-tuning and filtering options",
                ),
                ArgumentGroup(
                    "output_control",
                    ArgumentComplexity.ADVANCED,
                    ["no-realign", "template-ligand-file"],
                    "Advanced output control",
                ),
            ],
            # Debug arguments - shown only when explicitly requested
            "debug": [
                ArgumentGroup(
                    "debugging",
                    ArgumentComplexity.DEBUG,
                    ["log-level", "verbose"],
                    "Debugging and development options",
                )
            ],
        }

    def get_effective_experience_level(self) -> ExperienceLevel:
        """Get the effective user experience level."""
        if self.preferences.experience_level == ExperienceLevel.AUTO:
            return self.usage_patterns.calculate_experience_level()
        return self.preferences.experience_level

    def get_arguments_for_user_level(
        self, user_level: Optional[ExperienceLevel] = None
    ) -> Set[str]:
        """Get the set of arguments appropriate for a user level."""
        if user_level is None:
            user_level = self.get_effective_experience_level()

        arguments = set()

        # Always include essential arguments
        for group in self.argument_groups["essential"]:
            arguments.update(group.arguments)

        # Add arguments based on experience level
        if user_level in [ExperienceLevel.INTERMEDIATE, ExperienceLevel.ADVANCED]:
            for group in self.argument_groups["common"]:
                arguments.update(group.arguments)

        if user_level == ExperienceLevel.ADVANCED:
            for group in self.argument_groups["advanced"]:
                arguments.update(group.arguments)

        return arguments

    def get_verbosity_level(self, override: Optional[str] = None) -> VerbosityLevel:
        """Get the effective verbosity level."""
        if override:
            try:
                return VerbosityLevel(override.lower())
            except ValueError:
                pass

        return self.preferences.default_verbosity

    def should_show_progress_bars(self) -> bool:
        """Check if progress bars should be shown."""
        return self.preferences.show_progress_bars

    def should_show_performance_hints(self) -> bool:
        """Check if performance optimization hints should be shown."""
        return (
            self.preferences.show_performance_hints
            and self.get_effective_experience_level() != ExperienceLevel.BEGINNER
        )

    def get_smart_defaults(
        self, command: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get smart default values based on user patterns and context."""
        defaults = {}

        # Hardware-aware defaults
        if self.preferences.auto_detect_hardware:
            try:
                from templ_pipeline.core.hardware_utils import (
                    get_suggested_worker_config,
                )

                hw_config = get_suggested_worker_config()
                defaults["workers"] = hw_config.get("n_workers", 4)
            except ImportError:
                defaults["workers"] = 4

        # Command-specific intelligent defaults
        if command == "run":
            # For full pipeline, use conservative defaults for beginners
            if self.get_effective_experience_level() == ExperienceLevel.BEGINNER:
                defaults["num-conformers"] = 100
                defaults["num-templates"] = 50
            else:
                # Use learned preferences from usage patterns
                defaults["num-conformers"] = 200
                defaults["num-templates"] = 100

        elif command == "benchmark":
            # Benchmark defaults based on experience
            if self.get_effective_experience_level() == ExperienceLevel.BEGINNER:
                defaults["quick"] = True

        return defaults

    def track_command_usage(self, command: str, args: Dict[str, Any], success: bool):
        """Track command usage for adaptive learning."""
        # Update command usage count
        self.usage_patterns.command_usage_count[command] = (
            self.usage_patterns.command_usage_count.get(command, 0) + 1
        )

        # Track argument usage frequency
        for arg_name, arg_value in args.items():
            if arg_value is not None:  # Only track used arguments
                self.usage_patterns.argument_usage_frequency[arg_name] = (
                    self.usage_patterns.argument_usage_frequency.get(arg_name, 0) + 1
                )

        # Track successful workflows
        if success:
            workflow = f"{command}:{':'.join(sorted(args.keys()))}"
            if workflow not in self.usage_patterns.successful_workflows:
                self.usage_patterns.successful_workflows.append(workflow)
                # Keep only recent successful workflows
                if len(self.usage_patterns.successful_workflows) > 20:
                    self.usage_patterns.successful_workflows = (
                        self.usage_patterns.successful_workflows[-20:]
                    )

        self.usage_patterns.total_commands += 1

        # Save patterns periodically
        if self.usage_patterns.total_commands % 5 == 0:
            self._save_usage_patterns()

    def track_error(self, error_context: str):
        """Track error patterns for better user guidance."""
        self.usage_patterns.error_patterns.append(error_context)
        # Keep only recent errors
        if len(self.usage_patterns.error_patterns) > 50:
            self.usage_patterns.error_patterns = self.usage_patterns.error_patterns[
                -50:
            ]

    def get_contextual_help_hints(
        self, command: str, partial_args: Dict[str, Any]
    ) -> List[str]:
        """Get contextual help hints based on current context and user patterns."""
        hints = []

        # Beginner-specific hints
        if self.get_effective_experience_level() == ExperienceLevel.BEGINNER:
            if (
                command == "run"
                and not partial_args.get("ligand-smiles")
                and not partial_args.get("ligand-file")
            ):
                hints.append(
                    "TIP: You need either --ligand-smiles or --ligand-file to specify your query molecule"
                )

            if (
                command == "run"
                and not partial_args.get("protein-file")
                and not partial_args.get("protein-pdb-id")
            ):
                hints.append(
                    "TIP: You need either --protein-file or --protein-pdb-id to specify your target protein"
                )

        # Performance hints
        if self.should_show_performance_hints():
            if command in ["run", "generate-poses"] and not partial_args.get("workers"):
                hints.append(
                    f"PERFORMANCE: Consider using --workers {self.get_smart_defaults(command, {})['workers']} for faster processing"
                )

        return hints

    def _save_preferences(self):
        """Save user preferences to file."""
        try:
            with open(self.preferences_file, "w") as f:
                json.dump(self.preferences.to_dict(), f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save preferences: {e}")

    def _save_usage_patterns(self):
        """Save usage patterns to file."""
        try:
            data = {
                "command_usage_count": self.usage_patterns.command_usage_count,
                "argument_usage_frequency": self.usage_patterns.argument_usage_frequency,
                "error_patterns": self.usage_patterns.error_patterns,
                "successful_workflows": self.usage_patterns.successful_workflows,
                "total_commands": self.usage_patterns.total_commands,
            }
            with open(self.usage_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save usage patterns: {e}")

    def update_preferences(self, **kwargs):
        """Update user preferences."""
        for key, value in kwargs.items():
            if hasattr(self.preferences, key):
                setattr(self.preferences, key, value)
        self._save_preferences()

    def reset_learning(self):
        """Reset usage patterns and learning data."""
        self.usage_patterns = UsagePattern(
            command_usage_count={},
            argument_usage_frequency={},
            error_patterns=[],
            successful_workflows=[],
            total_commands=0,
        )
        self._save_usage_patterns()


def get_ux_config() -> TEMPLUXConfig:
    """Get the global UX configuration instance."""
    global _ux_config
    if "_ux_config" not in globals():
        _ux_config = TEMPLUXConfig()
    return _ux_config


def configure_logging_for_verbosity(
    verbosity: VerbosityLevel, logger_name: str = "templ-cli"
):
    """Configure logging based on verbosity level."""
    root_logger = logging.getLogger()
    cli_logger = logging.getLogger(logger_name)

    # Configure formatters
    minimal_formatter = logging.Formatter("%(message)s")
    normal_formatter = logging.Formatter("%(levelname)s: %(message)s")
    detailed_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create new handler
    handler = logging.StreamHandler()

    if verbosity == VerbosityLevel.MINIMAL:
        root_logger.setLevel(logging.WARNING)
        cli_logger.setLevel(logging.WARNING)
        handler.setFormatter(minimal_formatter)
    elif verbosity == VerbosityLevel.NORMAL:
        root_logger.setLevel(logging.INFO)
        cli_logger.setLevel(logging.INFO)
        handler.setFormatter(normal_formatter)
    elif verbosity == VerbosityLevel.DETAILED:
        root_logger.setLevel(logging.INFO)
        cli_logger.setLevel(logging.DEBUG)
        handler.setFormatter(detailed_formatter)
    elif verbosity == VerbosityLevel.DEBUG:
        root_logger.setLevel(logging.DEBUG)
        cli_logger.setLevel(logging.DEBUG)
        handler.setFormatter(detailed_formatter)

    root_logger.addHandler(handler)


# Export main classes and functions
__all__ = [
    "TEMPLUXConfig",
    "VerbosityLevel",
    "ExperienceLevel",
    "UserPreferences",
    "ArgumentComplexity",
    "get_ux_config",
    "configure_logging_for_verbosity",
]
