#!/usr/bin/env python3
"""
TEMPL CLI Enhanced Help System

This module implements the Topic-Centered + Smart Contextual help architecture,
providing progressive disclosure and workflow-oriented assistance for TEMPL CLI users.

Features:
- Three-tier help system (basic, intermediate, expert)
- Topic-centered organization around user workflows
- Context-aware help suggestions
- Example-driven learning with copy-paste commands
- Smart help that adapts to user experience level
"""

import argparse
import sys
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from .ux_config import ExperienceLevel, get_ux_config


class HelpLevel(Enum):
    """Help detail levels for progressive disclosure."""

    BASIC = "basic"  # Essential info only, beginner-friendly
    INTERMEDIATE = "intermediate"  # Common options and workflows
    EXPERT = "expert"  # Complete reference with all options


class HelpTopic(Enum):
    """Help topics organized around user workflows."""

    GETTING_STARTED = "getting-started"
    BASIC_WORKFLOW = "basic-workflow"
    BATCH_PROCESSING = "batch-processing"
    PERFORMANCE_TUNING = "performance-tuning"
    TROUBLESHOOTING = "troubleshooting"
    EXAMPLES = "examples"
    REFERENCE = "reference"


@dataclass
class HelpContent:
    """Container for help content at different levels."""

    basic: str
    intermediate: str
    expert: str
    examples: List[str]

    def get_for_level(self, level: HelpLevel) -> str:
        """Get content appropriate for the specified help level."""
        if level == HelpLevel.BASIC:
            return self.basic
        elif level == HelpLevel.INTERMEDIATE:
            return self.intermediate
        else:
            return self.expert


class TEMPLHelpSystem:
    """Enhanced help system for TEMPL CLI."""

    def __init__(self):
        self.ux_config = get_ux_config()
        self._setup_help_content()

    def _setup_help_content(self):
        """Initialize help content organized by topics and complexity levels."""

        # Main help content
        self.main_help = HelpContent(
            basic="""
TEMPL Pipeline - Template-based Protein-Ligand Pose Prediction

Quick Start:
  templ run --protein-file protein.pdb --ligand-smiles "CCO"

Common Commands:
  run              FULL: Complete pipeline (recommended for beginners)
  embed            EMBED: Generate protein embeddings
  find-templates   SEARCH: Find similar protein templates
  generate-poses   GENERATE: Generate ligand poses

Get Help:
  templ --help examples     Show example commands
  templ --help basic        This help (beginner-friendly)
  templ --help intermediate More options and workflows
""",
            intermediate="""
TEMPL Pipeline - Template-based Protein-Ligand Pose Prediction

Commands by Workflow:

FULL PIPELINE (Most Common):
  templ run --protein-file protein.pdb --ligand-smiles "SMILES_STRING"
  templ run --protein-pdb-id 2hyy --ligand-file ligand.sdf

STEP-BY-STEP WORKFLOW:
  1. Generate embedding:     templ embed --protein-file protein.pdb
  2. Find templates:         templ find-templates --query protein.pdb --embedding-file embeddings.npz
  3. Generate poses:         templ generate-poses --protein-file protein.pdb --ligand-smiles "SMILES" --template-pdb 5eqy

PERFORMANCE OPTIONS:
  --workers N              Number of CPU cores to use
  --num-conformers N       Number of conformers to generate (default: 100)

BENCHMARKING:
  templ benchmark polaris  Run Polaris benchmark suite
  templ benchmark time-split --quick  Quick benchmark test

Get More Help:
  templ --help expert       Complete reference with all options
  templ --help examples     Comprehensive examples
  templ --help troubleshooting  Common issues and solutions
""",
            expert="""
TEMPL Pipeline - Template-based Protein-Ligand Pose Prediction

COMPLETE COMMAND REFERENCE:

GLOBAL OPTIONS:
  --log-level {DEBUG,INFO,WARNING,ERROR}  Set logging verbosity
  --output-dir PATH                       Output directory (default: output)

COMMANDS:

run - Full pipeline execution
  Input (required, choose one):
    --protein-file PATH | --protein-pdb-id ID
    --ligand-smiles STR | --ligand-file PATH
  
  Template control:
    --embedding-file PATH     Pre-computed embeddings (default: auto-download)
    --num-templates INT       Templates to consider (default: 100)
    --similarity-threshold FLOAT  Min similarity (overrides num-templates)
  
  Generation control:
    --num-conformers INT      Conformers to generate (default: 100)
    --workers INT             Parallel workers (default: auto-detect)
    --run-id STR             Custom identifier (default: timestamp)
    --no-realign             Use raw conformers (no shape alignment)
    --enable-optimization    Enable force field optimization (disabled by default)

embed - Generate protein embeddings
  --protein-file PATH       Protein PDB file (required)
  --chain STR               Specific chain (default: first chain)
  --output-file PATH        Output embedding file

find-templates - Search for similar templates
  --query PATH              Protein (.pdb) or embedding (.npz) (required)
  --embedding-file PATH     Database embeddings (required)
  --num-templates INT       Number to return (default: 10)
  --similarity-threshold FLOAT  Minimum similarity
  --exclude-uniprot-file PATH   UniProt IDs to exclude

generate-poses - Generate ligand poses
  --protein-file PATH       Query protein (required)
  --ligand-smiles STR | --ligand-file PATH  Query ligand (required)
  --template-pdb STR        Template PDB ID (required)
  --template-ligand-file PATH  Template ligand SDF (optional)
  --num-conformers INT      Conformers to generate (default: 100)
  --workers INT             Parallel workers (default: auto)
  --no-realign             Use raw conformers (no shape alignment)

benchmark - Run benchmark suites
  polaris                   Polaris benchmark suite
  time-split                Time-split validation
  
  Options:
    --n-workers INT         CPU workers (default: auto)
    --n-conformers INT      Conformers per molecule (default: 200)
    --quick                 Reduced subset for testing
    --verbose               Debug output
    
  Time-split specific:
    --template-knn INT      Template neighbors (default: 100)
    --max-pdbs INT          Limit PDBs for testing
    --val-only, --test-only, --train-only  Specific dataset
    --pipeline-timeout INT  Per-PDB timeout (default: 180s)
    --max-ram FLOAT         RAM limit in GiB
    --per-worker-ram FLOAT  Per-worker RAM cap (default: 4.0 GiB)

ENVIRONMENT VARIABLES:
  TEMPL_DATA_DIR           Data directory (default: ./data)
  TEMPL_CACHE_DIR          Cache directory (default: ~/.templ/cache)
  TEMPL_LOG_LEVEL          Global log level
""",
            examples=[],
        )

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
  templ run --protein-file your_protein.pdb --ligand-smiles "CCO"

This command will:
  1. EMBED: Generate protein embedding
  2. SEARCH: Find similar protein templates
  3. GENERATE: Generate ligand poses
  4. SCORE: Score and rank results

Output: poses_final.sdf with ranked conformations
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
     - Protein: .pdb file or PDB ID (e.g., "2hyy")
     - Ligand: SMILES string or .sdf file

  2. Run prediction:
     templ run --protein-file protein.pdb --ligand-smiles "YOUR_SMILES"

  3. Results in output/ directory:
     - poses_final.sdf: Ranked conformations
     - metadata files: Scores and analysis

Advanced Options:
  --workers 8              # Use 8 CPU cores
  --num-conformers 200     # Generate 200 conformers
  --output-dir results/    # Custom output directory

Understanding Output:
  - Lower RMSD = better pose
  - Higher scores = more confident predictions
  - Multiple conformations ranked by quality
""",
                expert="""
GETTING STARTED WITH TEMPL

Complete Setup and Configuration:

Environment Setup:
  # Complete setup (creates + installs + activates):
  source setup_templ_env.sh
  
  # Manual activation for subsequent use:
  source .templ/bin/activate
  
  # Verify installation:
  templ --help
  python -c "import templ_pipeline; print('TEMPL installed successfully')"

Data Requirements:
  - Protein: PDB format, clean structure preferred
  - Ligand: SMILES (ChEMBL format) or SDF with 3D coordinates
  - Templates: Automatic download or custom database

Performance Tuning:
  export OMP_NUM_THREADS=1  # Recommended for multi-worker setup
  export TEMPL_CACHE_DIR=/fast/disk/cache  # Custom cache location

Advanced Configuration:
  ~/.templ/preferences.json  # User preferences
  ~/.templ/usage_patterns.json  # Learned patterns

Integration with Other Tools:
  - Input: ChEMBL compounds, PDB structures
  - Output: Compatible with PyMOL, VMD, Schrödinger
  - Scripting: Python API available

Troubleshooting Setup:
  templ --help troubleshooting  # Common issues
  templ benchmark polaris --quick --verbose  # Diagnostic run
""",
                examples=[
                    'templ run --protein-file examples/2hyy_protein.pdb --ligand-smiles "COc1ccc(C(C)=O)c(O)c1[C@H]1C[C@H]1NC(=S)Nc1ccc(C#N)cn1"',
                    "templ run --protein-pdb-id 2hyy --ligand-file examples/ligand.sdf --workers 4",
                ],
            ),
            HelpTopic.EXAMPLES: HelpContent(
                basic="""
BASIC EXAMPLES

Simple pose prediction:
  templ run --protein-file protein.pdb --ligand-smiles "CCO"

Using PDB ID instead of file:
  templ run --protein-pdb-id 2hyy --ligand-smiles "CCO"

Using SDF file for ligand:
  templ run --protein-file protein.pdb --ligand-file ligand.sdf

With optimization enabled:
  templ run --protein-file protein.pdb --ligand-smiles "CCO" --enable-optimization
""",
                intermediate="""
COMPREHENSIVE EXAMPLES

Basic Examples:
  # Simple ethanol binding prediction
  templ run --protein-file protein.pdb --ligand-smiles "CCO"
  
  # Using PDB ID (automatically downloads structure)
  templ run --protein-pdb-id 2hyy --ligand-smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
  
  # Complex ligand from file
  templ run --protein-file protein.pdb --ligand-file complex_ligand.sdf

Performance Examples:
  # Use 8 CPU cores for faster processing
  templ run --protein-file protein.pdb --ligand-smiles "CCO" --workers 8
  
  # Generate more conformers for better coverage
  templ run --protein-file protein.pdb --ligand-smiles "CCO" --num-conformers 200
  
  # Enable force field optimization for better quality
  templ run --protein-file protein.pdb --ligand-smiles "CCO" --enable-optimization
  
  # Custom output directory
  templ run --protein-file protein.pdb --ligand-smiles "CCO" --output-dir results/ethanol_binding/

Step-by-step Examples:
  # 1. Generate embedding first
  templ embed --protein-file protein.pdb --output-file protein_embedding.npz
  
  # 2. Find templates
  templ find-templates --query protein_embedding.npz --embedding-file database_embeddings.npz
  
  # 3. Generate poses with specific template
  templ generate-poses --protein-file protein.pdb --ligand-smiles "CCO" --template-pdb 5eqy
""",
                expert="""
EXPERT EXAMPLES

Production Workflows:
  # High-throughput screening setup
  templ run --protein-file target.pdb --ligand-smiles "SMILES" \\
            --workers 16 --num-conformers 500 --num-templates 200 \\
            --output-dir batch_results/compound_001/

  # High-quality poses with optimization
  templ run --protein-file target.pdb --ligand-smiles "SMILES" \\
            --num-conformers 1000 --enable-optimization \\
            --output-dir high_quality_poses/

  # Custom template database
  templ find-templates --query protein.pdb \\
                      --embedding-file custom_templates.npz \\
                      --similarity-threshold 0.8 \\
                      --exclude-uniprot-file exclude_list.txt

  # Fine-tuned pose generation
  templ generate-poses --protein-file protein.pdb \\
                       --ligand-file ligand.sdf \\
                       --template-pdb 5eqy \\
                       --template-ligand-file custom_template.sdf \\
                       --num-conformers 1000 \\
                       --no-realign

Benchmarking Examples:
  # Quick validation
  templ benchmark polaris --quick --workers 8
  
  # Full time-split validation
  templ benchmark time-split --n-workers 32 --n-conformers 500 \\
                            --max-ram 64.0 --per-worker-ram 2.0

  # Development testing
  templ benchmark time-split --val-only --max-pdbs 10 --verbose

Integration Examples:
  # ChEMBL compound processing
  for smiles in $(cat chembl_compounds.txt); do
    templ run --protein-pdb-id 2hyy --ligand-smiles "$smiles" \\
              --output-dir "results/$(echo $smiles | md5sum | cut -d' ' -f1)/"
  done

  # High-quality batch processing with optimization
  for smiles in $(cat chembl_compounds.txt); do
    templ run --protein-pdb-id 2hyy --ligand-smiles "$smiles" \\
              --enable-optimization --num-conformers 500 \\
              --output-dir "results/$(echo $smiles | md5sum | cut -d' ' -f1)/"
  done

  # Batch processing with metadata
  templ run --protein-file protein.pdb --ligand-file compounds.sdf \\
            --run-id "batch_$(date +%Y%m%d_%H%M%S)" \\
            --log-level DEBUG > batch_log.txt 2>&1

  # Production-quality batch processing
  templ run --protein-file protein.pdb --ligand-file compounds.sdf \\
            --run-id "production_$(date +%Y%m%d_%H%M%S)" \\
            --enable-optimization --num-conformers 1000 \\
            --log-level INFO > production_log.txt 2>&1

Advanced Configuration:
  # Custom environment setup
  export TEMPL_DATA_DIR="/data/templ_database"
  export TEMPL_CACHE_DIR="/tmp/templ_cache" 
  export OMP_NUM_THREADS=1
  
  templ run --protein-file protein.pdb --ligand-smiles "CCO" \\
            --workers $(nproc) --num-conformers 1000 --enable-optimization
""",
                examples=[],
            ),
            HelpTopic.TROUBLESHOOTING: HelpContent(
                basic="""
COMMON ISSUES

Problem: "Command not found" or "templ: command not found"
Solution: 
  1. Make sure you're in the TEMPL environment:
     source .templ/bin/activate
  2. If no environment exists, run setup:
     source setup_templ_env.sh

Problem: "Invalid SMILES string"
Solution: Check your SMILES syntax at https://pubchem.ncbi.nlm.nih.gov/

Problem: "Out of memory"
Solution: Reduce --workers or --num-conformers

Problem: Slow performance
Solution: Increase --workers to match your CPU cores

Problem: Environment not activating
Solution: Use 'source' not './': source setup_templ_env.sh

Get more help: templ --help troubleshooting
""",
                intermediate="""
TROUBLESHOOTING GUIDE

Installation Issues:
  # First-time setup (creates + installs + activates):
  source setup_templ_env.sh
  
  # Activate existing environment:
  source .templ/bin/activate
  
  # Check installation
  templ --help
  python -c "import templ_pipeline; print('OK')"
  
  # Test with benchmark
  templ benchmark polaris --quick

Performance Issues:
  # Check hardware utilization
  htop  # Monitor CPU usage during run
  
  # Optimize worker count
  templ run ... --workers $(nproc)  # Use all cores
  templ run ... --workers 4         # Conservative setting
  
  # Reduce memory usage
  templ run ... --num-conformers 50 --workers 2

Common Errors:
  "RDKit not available"
    → pip install rdkit
  
  "SMILES validation failed"
    → Test SMILES: python -c "from rdkit import Chem; print(Chem.MolFromSmiles('YOUR_SMILES'))"
  
  "Protein file not found"
    → Check file path and permissions
  
  "Embedding file not found"
    → Will auto-download on first run, check internet connection

File Format Issues:
  # PDB files: Must be clean, single chain preferred
  # SDF files: Must contain 3D coordinates
  # SMILES: Standard format, test with RDKit

Getting Help:
  - Include full error message
  - Specify TEMPL version: templ --version
  - Test with provided examples first
""",
                expert="""
ADVANCED TROUBLESHOOTING

System-Level Diagnostics:
  # Check dependencies
  python -c "import rdkit, numpy, sklearn, tqdm; print('Dependencies OK')"
  
  # Hardware detection
  python -c "from templ_pipeline.core.hardware_utils import get_hardware_status; print(get_hardware_status())"
  
  # Memory monitoring
  /usr/bin/time -v templ run ... 2>&1 | grep -E "(Maximum resident|User time)"

Performance Profiling:
  # Enable detailed logging
  export TEMPL_LOG_LEVEL=DEBUG
  templ run ... --log-level DEBUG 2>&1 | tee debug.log
  
  # Profile with Python
  python -m cProfile -s cumtime -m templ_pipeline.cli.main run ... > profile.txt

Memory Issues:
  # Large conformer sets
  split -l 100 compounds.sdf compound_batch_
  for batch in compound_batch_*; do
    templ run --ligand-file $batch ...
  done
  
  # Reduce memory footprint
  export OMP_NUM_THREADS=1
  templ run ... --workers 1 --num-conformers 50

Debugging Failed Runs:
  # Check intermediate files
  ls -la output/
  
  # Validate inputs
  python -c "
  from rdkit import Chem
  mol = Chem.MolFromSmiles('YOUR_SMILES')
  print(f'Atoms: {mol.GetNumAtoms()}, MW: {Chem.Descriptors.MolWt(mol)}')
  "
  
  # Check template database
  python -c "
  import numpy as np
  data = np.load('embeddings.npz')
  print(f'Templates: {len(data[\"pdb_ids\"])}, Dimensions: {data[\"embeddings\"].shape}')
  "

Custom Database Issues:
  # Validate embedding format
  python -c "
  import numpy as np
  data = np.load('custom_embeddings.npz')
  assert 'embeddings' in data and 'pdb_ids' in data
  assert data['embeddings'].shape[1] == 1280
  print('Format OK')
  "

Environment Variables:
  TEMPL_DATA_DIR           # Database location
  TEMPL_CACHE_DIR          # Cache location  
  TEMPL_LOG_LEVEL          # Global logging
  OMP_NUM_THREADS=1        # Recommended for multi-worker
  CUDA_VISIBLE_DEVICES     # GPU selection (if applicable)

Log Analysis:
  # Find bottlenecks
  grep -E "(took|seconds|minutes)" debug.log
  
  # Memory usage patterns
  grep -E "(memory|RAM|MB|GB)" debug.log
  
  # Error patterns
  grep -E "(ERROR|CRITICAL|Failed)" debug.log
""",
                examples=[],
            ),
        }

    def get_help_level(self, requested_level: Optional[str] = None) -> HelpLevel:
        """Determine appropriate help level based on user and request."""
        if requested_level:
            try:
                return HelpLevel(requested_level.lower())
            except ValueError:
                pass

        # Auto-determine based on user experience
        experience = self.ux_config.get_effective_experience_level()
        if experience == ExperienceLevel.BEGINNER:
            return HelpLevel.BASIC
        elif experience == ExperienceLevel.INTERMEDIATE:
            return HelpLevel.INTERMEDIATE
        else:
            return HelpLevel.EXPERT

    def show_main_help(self, level: Optional[str] = None):
        """Show main help content."""
        help_level = self.get_help_level(level)
        content = self.main_help.get_for_level(help_level)
        print(content)

        # Add contextual hints
        if help_level == HelpLevel.BASIC:
            print("\nTIP: New to TEMPL? Try: templ --help getting-started")
            print("TIP: See examples: templ --help examples")

    def show_topic_help(self, topic: str, level: Optional[str] = None):
        """Show help for a specific topic."""
        try:
            topic_enum = HelpTopic(topic.lower().replace("_", "-"))
        except ValueError:
            available_topics = [t.value for t in HelpTopic]
            print(f"ERROR: Unknown help topic: {topic}")
            print(f"Available topics: {', '.join(available_topics)}")
            return

        help_level = self.get_help_level(level)

        if topic_enum in self.topic_help:
            content = self.topic_help[topic_enum].get_for_level(help_level)
            print(content)

            # Show examples if available
            examples = self.topic_help[topic_enum].examples
            if examples and help_level in [HelpLevel.INTERMEDIATE, HelpLevel.EXPERT]:
                print("\nCOPY-PASTE EXAMPLES:")
                for example in examples:
                    print(f"  {example}")
        else:
            print(f"ERROR: Help not available for topic: {topic}")

    def show_contextual_help(self, command: str, partial_args: Dict[str, any]):
        """Show contextual help based on current command and arguments."""
        hints = self.ux_config.get_contextual_help_hints(command, partial_args)

        if hints:
            print("\nCONTEXTUAL HELP:")
            for hint in hints:
                print(f"   {hint}")

    def get_command_help(self, command: str, level: Optional[str] = None) -> str:
        """Get help content for a specific command."""
        help_level = self.get_help_level(level)

        command_help = {
            "run": {
                HelpLevel.BASIC: 'templ run --protein-file protein.pdb --ligand-smiles "SMILES"',
                HelpLevel.INTERMEDIATE: """
run - Execute complete TEMPL pipeline

Required (choose one for each):
  --protein-file PATH  OR  --protein-pdb-id ID
  --ligand-smiles STR  OR  --ligand-file PATH

Common options:
  --workers N           Number of CPU cores (default: auto)
  --num-conformers N    Conformers to generate (default: 100)
  --output-dir PATH     Output directory (default: output)
  --enable-optimization Enable force field optimization (disabled by default)
  --no-realign         Use raw conformers (no shape alignment)

Example:
  templ run --protein-file protein.pdb --ligand-smiles "CCO" --workers 4
""",
                HelpLevel.EXPERT: "Use: templ --help expert  # For complete reference",
            },
            "embed": {
                HelpLevel.BASIC: "templ embed --protein-file protein.pdb",
                HelpLevel.INTERMEDIATE: """
embed - Generate protein embeddings

Required:
  --protein-file PATH   Protein PDB file

Optional:
  --chain STR          Specific chain (default: first chain)  
  --output-file PATH   Output file (default: auto-generated)

Example:
  templ embed --protein-file protein.pdb --chain A
""",
                HelpLevel.EXPERT: "Use: templ --help expert  # For complete reference",
            },
        }

        if command in command_help and help_level in command_help[command]:
            return command_help[command][help_level]

        return f"Help not available for command: {command}"


def create_enhanced_parser():
    """Create argument parser with enhanced help system."""
    help_system = TEMPLHelpSystem()

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


def handle_help_request(help_arg: Optional[str], help_system: TEMPLHelpSystem):
    """Handle help requests with smart routing."""
    if not help_arg or help_arg == "main":
        help_system.show_main_help()
    elif help_arg in ["basic", "intermediate", "expert"]:
        help_system.show_main_help(help_arg)
    elif help_arg in [
        "examples",
        "getting-started",
        "troubleshooting",
        "basic-workflow",
        "batch-processing",
        "performance-tuning",
        "reference",
    ]:
        help_system.show_topic_help(help_arg)
    else:
        # Assume it's a command name
        print(help_system.get_command_help(help_arg))


# Export main classes and functions
__all__ = [
    "TEMPLHelpSystem",
    "HelpLevel",
    "HelpTopic",
    "create_enhanced_parser",
    "handle_help_request",
]
