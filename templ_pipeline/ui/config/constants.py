"""
TEMPL Pipeline Constants

Application-wide constants for the TEMPL Pipeline UI.
"""


# Import centralized version
try:
    from templ_pipeline import __version__ as VERSION
except ImportError:
    VERSION = "1.0.0"  # Fallback

API_VERSION = "v1"

# File type constants
ALLOWED_MOLECULE_EXTENSIONS = {'.sdf', '.mol', '.smi'}
ALLOWED_PROTEIN_EXTENSIONS = {'.pdb'}
ALLOWED_TEMPLATE_EXTENSIONS = {'.sdf'}

# MIME types
MIME_TYPES = {
    '.sdf': ['chemical/x-mdl-sdfile', 'text/plain', 'application/octet-stream'],
    '.mol': ['chemical/x-mdl-molfile', 'text/plain', 'application/octet-stream'],
    '.pdb': ['chemical/x-pdb', 'text/plain', 'application/octet-stream'],
    '.smi': ['text/plain', 'application/octet-stream'],
}

# Molecular constraints
MIN_ATOMS = 3
MAX_ATOMS = 200
DEFAULT_NUM_CONFORMERS = 200

# Template settings
DEFAULT_MAX_TEMPLATES = 100
MIN_TEMPLATES = 10
MAX_TEMPLATES = 500

# Scoring thresholds - TEMPL Normalized TanimotoCombo Implementation
# 
# Scientific Methodology (Based on PMC9059856):
# Standard TanimotoCombo = ShapeTanimoto + ColorTanimoto (range 0-2)
# TEMPL Implementation: combo_score = 0.5 * (ShapeTanimoto + ColorTanimoto) = TanimotoCombo / 2 (range 0-1)
# 
# Scientific References:
# 1. "Sequential ligand- and structure-based virtual screening approach" (PMC9059856)
#    - Standard TanimotoCombo methodology: ShapeTanimoto + ColorTanimoto
#    - Literature cutoff: TanimotoCombo > 1.2 on 0-2 scale
# 2. ChemBioChem Study: Large-scale analysis of 269.7 billion conformer pairs
#    - Shape Tanimoto: average 0.54 ± 0.10
#    - Color Tanimoto: average 0.07 ± 0.05  
#    - ComboTanimoto: average 0.62 ± 0.13
# 3. Research Journal Applied Sciences: "Retrieval Performance using Different Type of Similarity Coefficient"
#    - Validates Tanimoto coefficient performance for virtual screening
#
# TEMPL Normalization Rationale:
# - PMC Article cutoff (1.2/2 = 0.6) converted to TEMPL's 0-1 scale
# - TEMPL uses more conservative thresholds (0.35/0.25/0.15) for higher quality pose discrimination
# - Normalization provides easier interpretation while maintaining scientific rigor
SCORE_EXCELLENT = 0.35  # Excellent: More stringent than PMC equivalent (0.6), ensures very high quality
SCORE_GOOD = 0.25      # Good: Conservative threshold for reliable pose prediction  
SCORE_FAIR = 0.15      # Fair: Acceptable quality threshold for further evaluation
SCORE_POOR = 0.0       # Poor: Below acceptance threshold

# UI Messages
MESSAGES = {
    "NO_INPUT": "Please provide both a molecule and protein target to begin",
    "PROCESSING": "Processing your request...",
    "SUCCESS": "Pose prediction completed successfully!",
    "ERROR": "An error occurred during processing",
    "INVALID_SMILES": "Invalid SMILES string format",
    "INVALID_PDB": "Invalid PDB ID format (should be 4 characters)",
    "FILE_TOO_LARGE": "File size exceeds the maximum limit",
    "NO_POSES": "No valid poses could be generated",
}

# Error categories
ERROR_CATEGORIES = {
    'FILE_UPLOAD': 'File Upload Error',
    'MOLECULAR_PROCESSING': 'Molecular Processing Error',
    'PIPELINE_ERROR': 'Pipeline Execution Error',
    'VALIDATION_ERROR': 'Input Validation Error',
    'MEMORY_ERROR': 'Memory Management Error',
    'NETWORK_ERROR': 'Network/Database Error',
    'CONFIGURATION_ERROR': 'Configuration Error',
    'CRITICAL': 'Critical System Error'
}

# Quality assessment labels
QUALITY_LABELS = {
    "excellent": "🟢 Excellent - High confidence pose",
    "good": "🔵 Good - Reliable pose prediction",
    "fair": "🟡 Fair - Moderate confidence",
    "poor": "🔴 Poor - Low confidence, consider alternatives"
}

# Progress messages
PROGRESS_MESSAGES = {
    10: "Initializing pipeline...",
    20: "Loading protein structure...",
    30: "Searching for templates...",
    40: "Generating molecular conformers...",
    50: "Performing MCS alignment...",
    60: "Scoring poses...",
    80: "Finalizing results...",
    90: "Preparing visualization...",
    100: "Complete!"
}

# CSS Color scheme
COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "success": "#28a745",
    "info": "#17a2b8",
    "warning": "#ffc107",
    "error": "#dc3545",
    "background": "rgba(30, 32, 48, 0.75)",
    "text": "#e0e0e0"
}

# Session state keys
SESSION_KEYS = {
    "APP_INITIALIZED": "app_initialized",
    "QUERY_MOL": "query_mol",
    "INPUT_SMILES": "input_smiles",
    "PROTEIN_PDB_ID": "protein_pdb_id",
    "PROTEIN_FILE_PATH": "protein_file_path",
    "CUSTOM_TEMPLATES": "custom_templates",
    "POSES": "poses",
    "TEMPLATE_USED": "template_used",
    "TEMPLATE_INFO": "template_info",
    "MCS_INFO": "mcs_info",
    "ALL_RANKED_POSES": "all_ranked_poses",
    "HARDWARE_INFO": "hardware_info",
    "FAIR_METADATA": "fair_metadata",
    "SHOW_FAIR_PANEL": "show_fair_panel",
    "PREDICTION_RUNNING": "prediction_running",  # Track prediction state for UI management
    # Protein similarity keys
    "PROTEIN_SIMILARITY_RESULTS": "protein_similarity_results",
    "PROTEIN_SIMILARITY_COUNT": "protein_similarity_count",
    "PROTEIN_SIMILARITY_STATUS": "protein_similarity_status",
    # User settings for GPU and pipeline configuration
    "USER_DEVICE_PREFERENCE": "user_device_preference",      # "auto", "gpu", "cpu"
    "USER_KNN_THRESHOLD": "user_knn_threshold",              # 10-500
    "USER_CHAIN_SELECTION": "user_chain_selection",          # "auto", "A", "B", etc.
    "USER_SIMILARITY_THRESHOLD": "user_similarity_threshold"  # 0.0-1.0
} 