#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Help system for TEMPL CLI commands.

This module provides comprehensive help content for all TEMPL commands,
organized by topic and complexity level.
"""

import argparse
import enum
from typing import List, Optional


class HelpTopic(enum.Enum):
    """Available help topics."""

    GETTING_STARTED = "getting-started"
    EXAMPLES = "examples"
    TROUBLESHOOTING = "troubleshooting"
    ADVANCED = "advanced"
    CONFIGURATION = "configuration"


class HelpContent:
    """Container for help content at different complexity levels."""

    def __init__(
        self,
        basic: str = "",
        intermediate: str = "",
        expert: str = "",
        examples: Optional[List[str]] = None,
    ):
        """Initialize help content.

        Args:
            basic: Basic level help content
            intermediate: Intermediate level help content
            expert: Expert level help content
            examples: List of example commands
        """
        self.basic = basic.strip()
        self.intermediate = intermediate.strip()
        self.expert = expert.strip()
        self.examples = examples or []


class HelpSystem:
    """Central help system for TEMPL CLI."""

    def __init__(self):
        """Initialize the help system with all help content."""
        # Main help content
        self.main_help = HelpContent(
            basic="""
TEMPL Pipeline - Template-based Protein-Ligand Pose Prediction

TEMPL predicts how small molecules bind to proteins using template-based methods.

Quick Start:
  templ run --protein-file protein.pdb --ligand-smiles "SMILES" [OPTIONS]

Common Commands:
  templ embed --protein-file protein.pdb [OPTIONS]
  templ find-templates --protein-file protein.pdb [OPTIONS]
  templ generate-poses --protein-file protein.pdb --ligand-smiles "SMILES" [OPTIONS]

Get Help:
  templ --help examples          # Usage examples
  templ --help getting-started   # Setup guide
  templ --help troubleshooting   # Common issues
""",
            intermediate="""
TEMPL Pipeline - Template-based Protein-Ligand Pose Prediction

TEMPL predicts how small molecules bind to proteins using template-based methods.

Quick Start:
  templ run --protein-file protein.pdb --ligand-smiles "SMILES" [OPTIONS]

Common Commands:
  templ embed --protein-file protein.pdb [OPTIONS]
  templ find-templates --protein-file protein.pdb [OPTIONS]
  templ generate-poses --protein-file protein.pdb --ligand-smiles "SMILES" [OPTIONS]

Advanced Options:
  --output-dir DIR              # Custom output directory (default: output)
  --log-level LEVEL             # Set logging level (DEBUG, INFO, WARNING, ERROR)
  --workers N                   # Number of worker processes
  --num-conformers N            # Number of ligand conformers to generate
  --enable-optimization         # Enable force field optimization

Get Help:
  templ --help examples          # Usage examples
  templ --help getting-started   # Setup guide
  templ --help troubleshooting   # Common issues
  templ --help expert            # Advanced usage
""",
            expert="""
TEMPL Pipeline - Template-based Protein-Ligand Pose Prediction

TEMPL predicts how small molecules bind to proteins using template-based methods.

Quick Start:
  templ run --protein-file protein.pdb --ligand-smiles "SMILES" [OPTIONS]

Common Commands:
  templ embed --protein-file protein.pdb [OPTIONS]
  templ find-templates --protein-file protein.pdb [OPTIONS]
  templ generate-poses --protein-file protein.pdb --ligand-smiles "SMILES" [OPTIONS]

Complete Command Reference:

Global Options:
  --output-dir DIR              # Custom output directory (default: output)
  --log-level LEVEL             # Set logging level (DEBUG, INFO, WARNING, ERROR)
  --verbosity LEVEL             # Output verbosity (minimal, normal, detailed, debug)
  --seed N                      # Random seed for reproducible results
  --version                     # Show version and exit

Performance Options:
  --workers N                   # Number of worker processes (default: auto-detect)
  --num-conformers N            # Number of ligand conformers (default: 100)
  --num-templates N             # Number of templates to use (default: 50)
  --similarity-threshold F      # Template similarity threshold (0.0-1.0)

Quality Options:
  --enable-optimization         # Enable force field optimization
  --no-realign                  # Skip ligand realignment
  --template-pdb PDBID          # Use specific template PDB ID

Advanced Options:
  --run-id ID                   # Custom run identifier
  --chain CHAIN                 # Specific protein chain to use
  --output-file FILE            # Custom output file path
  --embedding-file FILE         # Pre-computed embeddings file

Get Help:
  templ --help examples          # Usage examples
  templ --help getting-started   # Setup guide
  templ --help troubleshooting   # Common issues
  templ --help expert            # This help
""",
            examples=[
                "templ run --protein-file data/example/1iky_protein.pdb --ligand-smiles 'CCO'",
                "templ generate-poses --protein-file data/example/1iky_protein.pdb --ligand-smiles 'CCO' --template-pdb 5eqy",
                "templ embed --protein-file data/example/1iky_protein.pdb",
                "templ find-templates --protein-file data/example/1iky_protein.pdb",
            ],
        )

        # Examples help content
        self.examples_help = HelpContent(
            basic="""
BASIC EXAMPLES

Simple pose prediction:
  templ run --protein-file data/example/1iky_protein.pdb --ligand-smiles "CCO"

Using PDB ID instead of file:
  templ run --protein-pdb-id 1iky --ligand-smiles "CCO"

Using SDF file for ligand:
  templ run --protein-file data/example/1iky_protein.pdb --ligand-file data/example/1iky_ligand.sdf

With optimization enabled:
  templ run --protein-file data/example/1iky_protein.pdb --ligand-smiles "CCO" --enable-optimization
""",
            intermediate="",
            expert="",
            examples=[],
        )

        # Set up topic help dictionary
        self.topic_help = {
            HelpTopic.EXAMPLES: self.examples_help,
        }

        # Topic-specific help content
        self.topic_help = {
            HelpTopic.GETTING_STARTED: HelpContent(
                basic="""
GETTING STARTED WITH TEMPL

TEMPL predicts how small molecules bind to proteins using template-based methods.

Environment Setup:
  # FIRST TIME - Complete setup with activation:
  source setup_templ_env.sh

  # SUBSEQUENT USE - Activate existing environment:
  source .templ/bin/activate

Prerequisites:
  - Protein structure file (.pdb) OR PDB ID
  - Ligand as SMILES string OR SDF file

Simplest Usage:
  templ run --protein-file data/example/1iky_protein.pdb --ligand-smiles "CCO"

This command will:
  1. EMBED: Generate protein embedding
  2. SEARCH: Find similar protein templates
  3. GENERATE: Generate ligand poses
  4. SCORE: Score and rank results

Output: timestamped folder with top poses and all poses SDF files
""",
                intermediate="""
GETTING STARTED WITH TEMPL

Environment Setup:
  # FIRST TIME SETUP (creates environment + installs + activates):
  source setup_templ_env.sh

  # SUBSEQUENT USAGE (activate existing environment):
  source .templ/bin/activate

Installation Check:
  templ --help        # Should show this help
  templ benchmark polaris --quick  # Test installation

Basic Workflow:
  1. Prepare your files:
     - Protein: .pdb file (e.g., "data/example/1iky_protein.pdb")
     - Ligand: SMILES string or .sdf file

  2. Run prediction:
     templ run --protein-file data/example/1iky_protein.pdb --ligand-smiles "CCO"

  3. Results in output/ directory:
     - timestamped folders with top3 and all poses SDF files
     - metadata files: Scores and analysis

Advanced Options:
  --workers 8              # Use 8 CPU cores
  --num-conformers 200     # Generate 200 conformers
  --output-dir results/    # Custom output directory

Troubleshooting Setup:
  templ --help troubleshooting  # Common issues
  templ benchmark polaris --quick --verbose  # Diagnostic run
""",
                examples=[
                    'templ run --protein-file data/example/1iky_protein.pdb --ligand-smiles "COc1ccc(C(C)=O)c(O)c1[C@H]2C[C@H]2NC(=S)Nc3ccc(cn3)C#N"',
                    "templ run --protein-file data/example/5eqy_protein.pdb --ligand-file data/example/1iky_ligand.sdf --workers 4",
                ],
            ),
            HelpTopic.EXAMPLES: HelpContent(
                basic="""
BASIC EXAMPLES

Simple pose prediction:
  templ run --protein-file data/example/1iky_protein.pdb --ligand-smiles "CCO"

Using PDB ID instead of file:
  templ run --protein-pdb-id 1iky --ligand-smiles "CCO"

Using SDF file for ligand:
  templ run --protein-file data/example/1iky_protein.pdb --ligand-file data/example/1iky_ligand.sdf

With optimization enabled:
  templ run --protein-file data/example/1iky_protein.pdb --ligand-smiles "CCO" --enable-optimization
""",
                intermediate="""
COMPREHENSIVE EXAMPLES

Basic Examples:
  # Simple ethanol binding prediction
  templ run --protein-file data/example/1iky_protein.pdb --ligand-smiles "CCO"

  # Using actual example data
  templ run --protein-file data/example/5eqy_protein.pdb --ligand-smiles "CN1CCCN(CC1)Cc2ccc(cc2)c3ccc(CN4CCCN(C)CC4)cc3C#N"

  # Complex ligand from file
  templ run --protein-file data/example/1iky_protein.pdb --ligand-file data/example/1iky_ligand.sdf

Performance Examples:
  # Use 8 CPU cores for faster processing
  templ run --protein-file data/example/1iky_protein.pdb --ligand-smiles "CCO" --workers 8

  # Generate more conformers for better coverage
  templ run --protein-file data/example/1iky_protein.pdb --ligand-smiles "CCO" --num-conformers 200

  # Enable force field optimization for better quality
  templ run --protein-file data/example/1iky_protein.pdb --ligand-smiles "CCO" --enable-optimization

  # Custom output directory
  templ run --protein-file data/example/1iky_protein.pdb --ligand-smiles "CCO" --output-dir results/ethanol_binding/

Step-by-step Examples:
  # 1. Generate embedding first
  templ embed --protein-file data/example/1iky_protein.pdb --output-file protein_embedding.npz

  # 2. Find templates
  templ find-templates --query protein_embedding.npz --embedding-file data/embeddings/templ_protein_embeddings_v1.0.0.npz

  # 3. Generate poses with specific template
  templ generate-poses --protein-file data/example/1iky_protein.pdb --ligand-smiles "CCO" --template-pdb 5eqy
""",
                expert="""
EXPERT EXAMPLES

Production Workflows:
  # High-throughput screening setup
  templ run --protein-file data/example/1iky_protein.pdb --ligand-smiles "CCO" --workers 16 --num-conformers 500 --num-templates 200 --output-dir batch_results/compound_001/

  # High-quality poses with optimization
  templ run --protein-file data/example/1iky_protein.pdb --ligand-smiles "CCO" --num-conformers 1000 --enable-optimization --output-dir high_quality_poses/

  # Custom template database
  templ find-templates --query data/example/1iky_protein.pdb --embedding-file data/embeddings/templ_protein_embeddings_v1.0.0.npz --similarity-threshold 0.8

  # Fine-tuned pose generation
  templ generate-poses --protein-file data/example/1iky_protein.pdb --ligand-file data/example/1iky_ligand.sdf --template-pdb 5eqy --num-conformers 1000 --no-realign

Benchmarking Examples:
  # Quick validation
  templ benchmark polaris --quick --workers 8

  # Full time-split validation
  templ benchmark time-split --n-workers 32 --n-conformers 500 --max-ram 64.0 --per-worker-ram 2.0

  # Development testing
  templ benchmark time-split --val-only --max-pdbs 10 --verbose

Integration Examples:
  # Batch processing with custom run ID
  templ run --protein-file data/example/1iky_protein.pdb --ligand-file data/example/1iky_ligand.sdf --run-id "batch_20240101_120000" --log-level DEBUG

  # Production-quality processing
  templ run --protein-file data/example/1iky_protein.pdb \
    --ligand-file data/example/1iky_ligand.sdf --run-id "production_run" \
    --enable-optimization --num-conformers 1000 --log-level INFO

  # Multiple SMILES processing (run separately)
  templ run --protein-file data/example/5eqy_protein.pdb --ligand-smiles "CCO" --output-dir results/ethanol/
  templ run --protein-file data/example/5eqy_protein.pdb --ligand-smiles "CC(C)O" --output-dir results/isopropanol/

Advanced Configuration:
""",
                examples=[
                    "templ run --protein-file data/example/1iky_protein.pdb --ligand-smiles 'CCO' --workers 8",
                    "templ benchmark polaris --quick --verbose",
                ],
            ),
        }

    def get_help(self, topic: Optional[HelpTopic] = None, level: str = "basic") -> str:
        """Get help content for a specific topic and level.

        Args:
            topic: Help topic to retrieve
            level: Complexity level (basic, intermediate, expert)

        Returns:
            Formatted help content
        """
        if topic is None:
            content = self.main_help
        else:
            content = self.topic_help.get(topic, self.main_help)

        if level == "basic":
            return content.basic
        elif level == "intermediate":
            return content.intermediate
        elif level == "expert":
            return content.expert
        else:
            return content.basic

    def get_examples(self, topic: Optional[HelpTopic] = None) -> List[str]:
        """Get example commands for a topic.

        Args:
            topic: Help topic to retrieve examples for

        Returns:
            List of example commands
        """
        if topic is None:
            return self.main_help.examples
        else:
            content = self.topic_help.get(topic, self.main_help)
            return content.examples

    def list_topics(self) -> List[str]:
        """List all available help topics.

        Returns:
            List of topic names
        """
        return [topic.value for topic in HelpTopic]

    def format_help(
        self, topic: Optional[HelpTopic] = None, level: str = "basic"
    ) -> str:
        """Format complete help content with examples.

        Args:
            topic: Help topic to format
            level: Complexity level

        Returns:
            Complete formatted help string
        """
        help_text = self.get_help(topic, level)
        examples = self.get_examples(topic)

        if examples:
            help_text += "\n\nEXAMPLES:\n"
            for i, example in enumerate(examples, 1):
                help_text += f"  {i}. {example}\n"

        return help_text.strip()


def create_enhanced_parser():
    """Create argument parser with enhanced help system."""
    help_system = HelpSystem()

    parser = argparse.ArgumentParser(
        description="TEMPL Pipeline: Template-based Protein-Ligand Pose Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,  # We handle help ourselves
    )

    # Custom help arguments
    parser.add_argument(
        "-h",
        "--help",
        nargs="?",
        const="main",
        help="Show help. Options: basic, intermediate, expert, examples, getting-started, troubleshooting, <command>",
    )

    return parser, help_system


def handle_help_request(help_arg: Optional[str], help_system: HelpSystem):
    """Handle help requests with smart routing."""
    if not help_arg or help_arg == "main":
        print(help_system.get_help())
    elif help_arg in ["basic", "intermediate", "expert"]:
        print(help_system.get_help(level=help_arg))
    elif help_arg in [
        "examples",
        "getting-started",
        "troubleshooting",
        "advanced",
        "configuration",
    ]:
        try:
            topic = HelpTopic(help_arg)
            print(help_system.get_help(topic))
        except ValueError:
            print(f"Unknown help topic: {help_arg}")
            print(f"Available topics: {', '.join(help_system.list_topics())}")
    else:
        # Assume it's a command name
        print(f"Help not available for command: {help_arg}")


# Export main classes and functions
__all__ = [
    "HelpSystem",
    "HelpTopic",
    "create_enhanced_parser",
    "handle_help_request",
]
