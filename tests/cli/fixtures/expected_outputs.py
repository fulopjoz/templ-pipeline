"""Expected help outputs for validation."""

# Expected help output patterns
HELP_MAIN_KEYWORDS = [
    "TEMPL",
    "Template-based Protein-Ligand Pose Prediction Pipeline",
    "Commands:",
    "embed",
    "find-templates", 
    "generate-poses",
    "run",
    "Quick Examples:",
    "Additional help functions:"
]

HELP_SIMPLE_KEYWORDS = [
    "TEMPL Pipeline - Quick Reference",
    "Pipeline Options:",
    "Template Options:",
    "Generation Options:",
    "Output Options:",
    "--protein-file",
    "--ligand-smiles",
    "--num-templates",
    "--num-conformers"
]

HELP_EXAMPLES_KEYWORDS = [
    "TEMPL Pipeline - Usage Examples",
    "Basic Protein-Ligand Pose Generation",
    "Cross-template Pose Generation", 
    "High-throughput Processing",
    "Step-by-step Workflow"
]

HELP_PERFORMANCE_KEYWORDS = [
    "TEMPL Pipeline - Performance Guide",
    "Hardware Optimization:",
    "--workers",
    "GPU acceleration",
    "--num-conformers"
]

# Example commands that should be syntactically correct
EXAMPLE_COMMANDS = [
    "templ run --protein-file data/example/1iky_protein.pdb",
    "templ generate-poses --protein-file data/example/1iky_protein.pdb",
    "templ embed --protein-file data/example/1iky_protein.pdb",
    "templ find-templates --protein-file data/example/1iky_protein.pdb"
]

# ASCII banner pattern
ASCII_BANNER_PATTERN = r"TEMPL.*Template-based.*Pipeline" 