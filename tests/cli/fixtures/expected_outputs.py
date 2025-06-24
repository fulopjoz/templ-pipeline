"""Expected help outputs for validation."""

# Expected help output patterns
HELP_MAIN_KEYWORDS = [
    "TEMPL",
    "Template-based Protein-Ligand Pose Prediction",
    "Quick Start:",
    "embed",
    "find-templates", 
    "generate-poses",
    "run",
    "Common Commands:",
    "Get Help:"
]

HELP_SIMPLE_KEYWORDS = [
    "Help not available for command: simple",
]

HELP_EXAMPLES_KEYWORDS = [
    "BASIC EXAMPLES",
    "Simple pose prediction:",
    "Using PDB ID instead of file:",
    "Using SDF file for ligand:",
]

HELP_PERFORMANCE_KEYWORDS = [
    "Help not available for command: performance",
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