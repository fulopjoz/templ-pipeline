"""
TEMPL Pipeline Constants

Application-wide constants for the TEMPL Pipeline UI.
"""

# Version information
VERSION = "2.0.0"
API_VERSION = "v2"

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

# Note: These thresholds are calibrated for shape/color/combo scores (0.0-1.0 range)
# Combo score = (shape + color) / 2, where both shape and color range 0.0-1.0
# Scoring thresholds
SCORE_EXCELLENT = 0.8   # Excellent match for shape/color scores (80%+ similarity)
SCORE_GOOD = 0.6       # Good match for shape/color scores (60%+ similarity)
SCORE_FAIR = 0.4       # Fair match for shape/color scores (40%+ similarity)
SCORE_POOR = 0.0

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
    "PROTEIN_SIMILARITY_STATUS": "protein_similarity_status"
} 