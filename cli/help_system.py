#!/usr/bin/env python3
"""
TEMPL Pipeline Enhanced Help System

Provides visually rich and comprehensive help output inspired by professional
scientific software like ROCS.
"""

import sys
from typing import Optional
from rich.console import Console
import colorama

# Initialize colorama for cross-platform color support
colorama.init()

class TEMPLHelpSystem:
    """Enhanced help system for TEMPL Pipeline CLI."""
    
    def __init__(self):
        self.console = Console()
        self.version = "0.1.0"
        self.build_date = "20250106"
        
    def get_ascii_banner(self) -> str:
        """Generate TEMPL ASCII banner."""
        # Include literal text for easier machine detection while keeping the ASCII art intact
        return """TEMPL Pipeline
 _______ ______ __  __ _____  _      
|__   __|  ____|  \/  |  __ \| |     
   | |  | |__  | \  / | |__) | |     
   | |  |  __| | |\/| |  ___/| |     
   | |  | |____| |  | | |    | |____ 
   |_|  |______|_|  |_|_|    |______|

Template-based Protein-Ligand Pose Prediction Pipeline
License: MIT

GitHub: https://github.com/username/templ_pipeline
        """
    
    def show_main_help(self):
        """Display the main help screen."""
        self.console.print(self.get_ascii_banner(), style="bold cyan")
        self.console.print()
        
        # Commands
        self.console.print("[bold]Commands:[/bold]")
        commands = [
            ("embed", "Generate protein embeddings for template search"),
            ("find-templates", "Find similar protein templates from embedding database"),
            ("generate-poses", "Generate ligand poses using specific templates"),
            ("run", "Execute complete pipeline with template discovery"),
            ("benchmark", "Run built-in benchmark suites (polaris, time-split)")
        ]
        
        for cmd, desc in commands:
            self.console.print(f"  [cyan]{cmd:<15}[/cyan] {desc}")
        self.console.print()
        
        # Benchmark Details
        self.console.print("[bold]Benchmark Suites:[/bold]")
        benchmarks = [
            ("polaris", "Polaris benchmark dataset evaluation"),
            ("time-split", "Time-based split evaluation (train/val/test) using BenchmarkRunner")
        ]
        for bench, desc in benchmarks:
            self.console.print(f"  [yellow]{bench:<15}[/yellow] {desc}")
        self.console.print()
        
        # Split Options for time-split
        self.console.print("[bold]Time-split Options:[/bold]")
        split_options = [
            ("--train-only", "Evaluate training split only"),
            ("--val-only", "Evaluate validation split only"),
            ("--test-only", "Evaluate test split only"),
            ("--max-pdbs N", "Limit to N PDBs for testing"),
            ("--template-knn N", "Number of templates for KNN search (default: 100)"),
            ("--verbose", "Enable detailed progress output")
        ]
        for opt, desc in split_options:
            self.console.print(f"  [magenta]{opt:<15}[/magenta] {desc}")
        self.console.print()
        
        # Quick examples using real data
        self.console.print("[bold]Quick Examples:[/bold]")
        self.console.print("  [green]Basic:[/green]     templ run --protein-file templ_pipeline/data/example/1iky_protein.pdb \\")
        self.console.print("                    --ligand-smiles \"COc1ccc(C(C)=O)c(O)c1[C@H]2C[C@H]2NC(=S)Nc3ccc(cn3)C#N\"")
        self.console.print()
        self.console.print("  [green]Template:[/green]  templ generate-poses --protein-file templ_pipeline/data/example/1iky_protein.pdb \\")
        self.console.print("                    --template-pdb 5eqy --ligand-file templ_pipeline/data/example/1iky_ligand.sdf")
        self.console.print()
        self.console.print("  [green]Benchmark:[/green] templ benchmark polaris --n-workers 8")
        self.console.print("  [green]Time-split:[/green] templ benchmark time-split --n-workers 8 --n-conformers 200")
        self.console.print("  [green]Val only:[/green]  templ benchmark time-split --val-only --max-pdbs 5 --verbose")
        self.console.print("  [green]Quick test:[/green] templ benchmark time-split --val-only --max-pdbs 2")
        self.console.print()
        
        # Default Settings
        self.console.print("[bold]Default Settings:[/bold]")
        self.console.print("  Output directory: output")
        self.console.print("  Log level: INFO")
        self.console.print("  Conformers: 200 | Templates: 100 | Workers: auto-detected")
        self.console.print()
        
        # Additional help functions
        self.console.print("[bold]Additional help functions:[/bold]")
        self.console.print("  templ --help simple      Quick parameter reference")
        self.console.print("  templ --help examples    Detailed usage examples with real data")
        self.console.print("  templ --help workflows   Common workflow patterns")
        self.console.print("  templ --help performance Performance optimization guide")
        self.console.print("  templ --help troubleshoot Common issues and solutions")
    
    def show_simple_help(self):
        """Display simple parameter list organized by category."""
        self.console.print("[bold cyan]TEMPL Pipeline - Quick Reference[/bold cyan]\n")
        
        # Pipeline Options
        self.console.print("[bold]Pipeline Options:[/bold]")
        pipeline_params = [
            ("--protein-file", "Input protein PDB file"),
            ("--ligand-smiles", "Ligand SMILES string"),
            ("--ligand-file", "Ligand SDF file"),
            ("--embedding-file", "Protein embedding database"),
            ("--template-pdb", "Specific template PDB ID")
        ]
        for param, desc in pipeline_params:
            self.console.print(f"  [cyan]{param:<20}[/cyan] {desc}")
        self.console.print()
        
        # Template Options
        self.console.print("[bold]Template Options:[/bold]")
        template_params = [
            ("--num-templates", "Number of templates to find (default: 100)"),
            ("--similarity-cutoff", "Minimum similarity threshold (default: 0.7)"),
            ("--ca-rmsd-cutoff", "Maximum CA RMSD threshold (default: 10.0)")
        ]
        for param, desc in template_params:
            self.console.print(f"  [cyan]{param:<20}[/cyan] {desc}")
        self.console.print()
        
        # Generation Options
        self.console.print("[bold]Generation Options:[/bold]")
        gen_params = [
            ("--num-conformers", "Number of conformers per ligand (default: 200)"),
            ("--workers", "Parallel workers (auto-detected based on hardware)"),
            ("--scoring-method", "Scoring method (shape, color, combo)")
        ]
        for param, desc in gen_params:
            self.console.print(f"  [cyan]{param:<20}[/cyan] {desc}")
        self.console.print()
        
        # Output Options
        self.console.print("[bold]Output Options:[/bold]")
        output_params = [
            ("--output-dir", "Output directory (default: output)"),
            ("--run-id", "Custom run identifier (default: timestamp)"),
            ("--enhanced-output", "Include CA RMSD statistics"),
            ("--log-level", "Logging level (default: INFO)")
        ]
        for param, desc in output_params:
            self.console.print(f"  [cyan]{param:<20}[/cyan] {desc}")
        self.console.print()
        
        # Default Values Summary
        self.console.print("[bold]Default Values:[/bold]")
        self.console.print("  Output directory: output")
        self.console.print("  Log level: INFO")
        self.console.print("  Number of conformers: 200")
        self.console.print("  Number of templates: 100")
        self.console.print("  Workers: auto-detected")
    
    def show_examples_help(self):
        """Display detailed examples using real data files."""
        self.console.print("[bold cyan]TEMPL Pipeline - Usage Examples[/bold cyan]\n")
        
        examples = [
            {
                "title": "Basic Protein-Ligand Pose Generation",
                "description": "Using 1iky kinase with its native ligand",
                "command": "templ run --protein-file templ_pipeline/data/example/1iky_protein.pdb \\\n          --ligand-smiles \"COc1ccc(C(C)=O)c(O)c1[C@H]2C[C@H]2NC(=S)Nc3ccc(cn3)C#N\""
            },
            {
                "title": "Cross-template Pose Generation",
                "description": "Using 1iky ligand with 5eqy protein template",
                "command": "templ generate-poses --protein-file templ_pipeline/data/example/1iky_protein.pdb \\\n                      --ligand-file templ_pipeline/data/example/1iky_ligand.sdf \\\n                      --template-pdb 5eqy"
            },
            {
                "title": "High-throughput Processing",
                "description": "Process with multiple conformers (workers auto-detected)",
                "command": "templ run --protein-file templ_pipeline/data/example/5eqy_protein.pdb \\\n          --ligand-smiles \"CN1CCCN(CC1)Cc2ccc(cc2)c3ccc(CN4CCCN(C)CC4)cc3C#N\" \\\n          --num-conformers 200 --run-id experiment_v1"
            },
            {
                "title": "Timesplit Benchmark - Full Dataset",
                "description": "Run complete timesplit benchmark (train/val/test) using BenchmarkRunner",
                "command": "templ benchmark time-split --n-workers 8 --n-conformers 200 --template-knn 100"
            },
            {
                "title": "Timesplit Benchmark - Validation Only",
                "description": "Run only validation split for quick testing with verbose output",
                "command": "templ benchmark time-split --val-only --max-pdbs 10 --n-conformers 100 --verbose"
            },
            {
                "title": "Timesplit Benchmark - Test Split",
                "description": "Run only test split for final evaluation",
                "command": "templ benchmark time-split --test-only --n-conformers 200 --template-knn 100"
            },
            {
                "title": "Step-by-step Workflow",
                "description": "Execute pipeline components separately",
                "command": "# Generate embeddings\ntempl embed --protein-file templ_pipeline/data/example/1iky_protein.pdb\n\n# Find templates\ntempl find-templates --protein-file templ_pipeline/data/example/1iky_protein.pdb \\\n                     --embedding-file templ_pipeline/data/embeddings/protein_embeddings_base.npz\n\n# Generate poses\ntempl generate-poses --protein-file templ_pipeline/data/example/1iky_protein.pdb \\\n                     --ligand-smiles \"COc1ccc(C(C)=O)c(O)c1[C@H]2C[C@H]2NC(=S)Nc3ccc(cn3)C#N\" \\\n                     --template-pdb 5eqy"
            }
        ]
        
        for i, example in enumerate(examples, 1):
            self.console.print(f"[bold green]{i}. {example['title']}[/bold green]")
            self.console.print(f"   [dim]{example['description']}[/dim]")
            self.console.print(f"   [yellow]{example['command']}[/yellow]")
            self.console.print()
    
    def show_workflows_help(self):
        """Display common workflow patterns."""
        self.console.print("[bold cyan]TEMPL Pipeline - Common Workflows[/bold cyan]\n")
        
        workflows = [
            {
                "title": "Cross-Virus Template Discovery",
                "description": "For discovering templates across different viral proteins",
                "command": "templ run --protein-file your_protein.pdb \\\n          --ligand-smiles \"your_smiles\" \\\n          --embedding-file data/embeddings/protein_embeddings_base.npz \\\n          --num-templates 10"
            },
            {
                "title": "High-Quality Pose Generation",
                "description": "For publication-quality results",
                "command": "templ run --protein-file templ_pipeline/data/example/1iky_protein.pdb \\\n          --ligand-smiles \"COc1ccc(C(C)=O)c(O)c1[C@H]2C[C@H]2NC(=S)Nc3ccc(cn3)C#N\" \\\n          --num-conformers 200"
            },
            {
                "title": "Timesplit Benchmark Workflow",
                "description": "Sequential evaluation of splits using BenchmarkRunner with provided split files",
                "command": "# Run training split evaluation\ntempl benchmark time-split --train-only --n-conformers 200 --template-knn 100\n\n# Run validation split evaluation\ntempl benchmark time-split --val-only --n-conformers 200 --template-knn 100\n\n# Run test split evaluation\ntempl benchmark time-split --test-only --n-conformers 200 --template-knn 100"
            },
            {
                "title": "Quick Development Testing",
                "description": "Fast testing with limited PDBs from timesplit data using BenchmarkRunner",
                "command": "templ benchmark time-split --val-only --max-pdbs 5 --n-conformers 50 --verbose"
            }
        ]
        
        for workflow in workflows:
            self.console.print(f"[bold green]{workflow['title']}:[/bold green]")
            self.console.print(f"  {workflow['description']}")
            self.console.print(f"  [yellow]{workflow['command']}[/yellow]")
            self.console.print()
    
    def show_performance_help(self):
        """Display performance tuning guide."""
        self.console.print("[bold cyan]TEMPL Pipeline - Performance Guide[/bold cyan]\n")
        
        self.console.print("[bold]Hardware Optimization:[/bold]")
        self.console.print("  • Workers auto-detected based on CPU cores and memory")
        self.console.print("  • Override with --workers N if needed")
        self.console.print("  • GPU acceleration automatically detected when available")
        self.console.print("  • Increase --num-conformers for better sampling (trade-off: time vs quality)")
        self.console.print("  • Use SSD storage for faster I/O operations")
        self.console.print()
        
        self.console.print("[bold]Memory Management:[/bold]")
        self.console.print("  • Large protein complexes may require more RAM")
        self.console.print("  • Reduce --num-conformers if memory issues occur")
        self.console.print("  • Use --output-dir on fast storage for temporary files")
        self.console.print("  • Monitor system resources during batch processing")
        self.console.print()
        
        self.console.print("[bold]Quality vs Speed:[/bold]")
        self.console.print("  • Default settings balance quality and performance")
        self.console.print("  • Increase conformers (200+) for publication-quality results")
        self.console.print("  • Use similarity thresholds to filter templates")
        self.console.print("  • Enable verbose logging for debugging: --log-level DEBUG")

    def show_troubleshoot_help(self):
        """Display troubleshooting guide for common issues."""
        self.console.print("[bold cyan]TEMPL Pipeline - Troubleshooting Guide[/bold cyan]\n")
        
        self.console.print("[bold red]Common Issues and Solutions:[/bold red]\n")
        
        issues = [
            {
                "problem": "File not found: data/example/protein.pdb",
                "solution": "Use correct path: templ_pipeline/data/example/protein.pdb\nRun commands from the main project directory (/home/ubuntu/mcs/)",
                "example": "templ run --protein-file templ_pipeline/data/example/1iky_protein.pdb ..."
            },
            {
                "problem": "ModuleNotFoundError: No module named 'templ_pipeline'",
                "solution": "Don't run commands from inside the templ_pipeline/ directory\nRun from the parent directory where the package was installed",
                "example": "cd /home/ubuntu/mcs  # Correct directory\ntempl --help"
            },
            {
                "problem": "unrecognized arguments: --output-dir",
                "solution": "Put global arguments before the subcommand",
                "example": "templ --output-dir mydir run --protein-file ...\n# NOT: templ run --protein-file ... --output-dir mydir"
            },
            {
                "problem": "Split file not found: timesplit_train",
                "solution": "Ensure timesplit split files exist in templ_pipeline/data/splits/\nRequired files: timesplit_train, timesplit_val, timesplit_test",
                "example": "ls templ_pipeline/data/splits/  # Check split files exist"
            },
            {
                "problem": "Time-split benchmark fails with import error",
                "solution": "Benchmark uses BenchmarkRunner with provided split files, no PDB data setup required\nEnsure split files contain valid PDB IDs (one per line) in templ_pipeline/data/splits/",
                "example": "templ benchmark time-split --val-only --max-pdbs 2 --verbose  # Quick test with verbose output"
            },
            {
                "problem": "RDKit compatibility errors (maxAttempts, coordMap)",
                "solution": "These are automatically handled with compatibility checks\nUpdate RDKit if issues persist: conda update rdkit",
                "example": "conda update rdkit  # or pip install --upgrade rdkit"
            }
        ]
        
        for i, issue in enumerate(issues, 1):
            self.console.print(f"[bold yellow]{i}. {issue['problem']}[/bold yellow]")
            self.console.print(f"   [green]Solution:[/green] {issue['solution']}")
            if 'example' in issue:
                self.console.print(f"   [blue]Example:[/blue] {issue['example']}")
            self.console.print()
        
        self.console.print("[bold]Quick Diagnostic Commands:[/bold]")
        self.console.print("  Check installation: templ --help")
        self.console.print("  Test timesplit benchmark: templ benchmark time-split --val-only --max-pdbs 2 --verbose")
        self.console.print("  Test polaris benchmark: templ benchmark polaris --quick")
        self.console.print("  Enable debug logging: templ --log-level DEBUG benchmark ...")
        self.console.print("  Check working directory: pwd  # Should be /home/ubuntu/mcs")
        self.console.print("  Check split files: ls templ_pipeline/data/splits/")


def show_enhanced_help(help_type: Optional[str] = None):
    """Main entry point for enhanced help system."""
    help_system = TEMPLHelpSystem()
    
    if help_type is None or help_type == "main":
        help_system.show_main_help()
    elif help_type == "simple":
        help_system.show_simple_help()
    elif help_type == "examples":
        help_system.show_examples_help()
    elif help_type == "workflows":
        help_system.show_workflows_help()
    elif help_type == "performance":
        help_system.show_performance_help()
    elif help_type == "troubleshoot":
        help_system.show_troubleshoot_help()
    else:
        help_system.show_main_help() 