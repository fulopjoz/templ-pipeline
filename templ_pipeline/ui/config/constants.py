# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
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
ALLOWED_MOLECULE_EXTENSIONS = {".sdf", ".mol", ".smi"}
ALLOWED_PROTEIN_EXTENSIONS = {".pdb"}
ALLOWED_TEMPLATE_EXTENSIONS = {".sdf"}

# MIME types
MIME_TYPES = {
    ".sdf": ["chemical/x-mdl-sdfile", "text/plain", "application/octet-stream"],
    ".mol": ["chemical/x-mdl-molfile", "text/plain", "application/octet-stream"],
    ".pdb": ["chemical/x-pdb", "text/plain", "application/octet-stream"],
    ".smi": ["text/plain", "application/octet-stream"],
}

# Molecular constraints
MIN_ATOMS = 3
MAX_ATOMS = 200
DEFAULT_NUM_CONFORMERS = 200

# Template settings
DEFAULT_MAX_TEMPLATES = 100
MIN_TEMPLATES = 10
MAX_TEMPLATES = 500

# Scoring thresholds - TEMPL Pose Prediction Implementation
#
# Scientific Methodology for Pose Prediction:
# TEMPL uses normalized TanimotoCombo scores to evaluate pose quality against template structures
# Evaluation based on established pose prediction success criteria (RMSD ≤ 2.0 Å standard)
#
# Scientific References for Pose Prediction:
# 1. "CB-Dock2: improved protein–ligand blind docking" (Nucleic Acids Research, 2022)
#    - Achieved ~85% success rate for binding pose prediction (RMSD < 2.0 Å)
#    - Template-based docking approach similar to TEMPL methodology
#    - https://academic.oup.com/nar/article/50/W1/W159/6591526
# 2. "Uni-Mol Docking V2: Towards Realistic and Accurate Binding Pose Prediction" (2024)
#    - 77% accuracy for poses with RMSD < 2.0 Å, 75% passing quality checks
#    - Modern benchmark for pose prediction performance
#    - https://arxiv.org/abs/2405.11769
# 3. "POSIT: Flexible Shape-Guided Docking For Pose Prediction" (J. Chem. Inf. Model., 2015)
#    - Largest prospective validation study (71 crystal structures)
#    - Shape-guided approach with emphasis on pose prediction accuracy
#    - https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00142
# 4. "DeepBSP - Machine Learning Method for Accurate Prediction of Protein-Ligand Docking Structures" (2021)
#    - Direct RMSD prediction for pose evaluation
#    - Validates RMSD-based assessment for pose quality
#    - https://pubs.acs.org/doi/10.1021/acs.jcim.1c00334
#
# TEMPL Threshold Rationale for Pose Prediction:
# - Based on established RMSD success criteria where ≤2.0 Å represents successful pose prediction
# - Thresholds designed to reflect realistic pose prediction performance expectations
# - Higher thresholds appropriate for template-based pose prediction vs. ab-initio docking
# - Conservative approach ensuring meaningful discrimination between pose qualities

SCORE_EXCELLENT = 0.80  # Excellent: Top-tier poses, equivalent to RMSD ≤ 1.0 Å performance
SCORE_GOOD = 0.65       # Good: High-quality poses, RMSD ≤ 2.0 Å standard for success  
SCORE_FAIR = 0.45       # Fair: Moderate quality, requires validation (RMSD 2.0-3.0 Å range)
SCORE_POOR = 0.0        # Poor: Below acceptable threshold for pose prediction

# Individual component significance thresholds (based on pose prediction context)
SHAPE_TANIMOTO_SIGNIFICANT = 0.80   # High shape similarity for reliable pose prediction
COLOR_TANIMOTO_SIGNIFICANT = 0.50   # Meaningful pharmacophore alignment threshold

# Quality assessment labels for pose prediction
QUALITY_LABELS = {
    "excellent": "Excellent - High confidence pose (≤1.0 Å expected)",
    "good": "Good - Reliable pose prediction (≤2.0 Å expected)", 
    "fair": "Fair - Moderate confidence (2.0-3.0 Å expected)",
    "poor": "Poor - Low confidence, consider alternatives (>3.0 Å expected)",
}

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
    "FILE_UPLOAD": "File Upload Error",
    "MOLECULAR_PROCESSING": "Molecular Processing Error",
    "PIPELINE_ERROR": "Pipeline Execution Error",
    "VALIDATION_ERROR": "Input Validation Error",
    "MEMORY_ERROR": "Memory Management Error",
    "NETWORK_ERROR": "Network/Database Error",
    "CONFIGURATION_ERROR": "Configuration Error",
    "CRITICAL": "Critical System Error",
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
    100: "Complete!",
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
    "text": "#e0e0e0",
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
    "USER_DEVICE_PREFERENCE": "user_device_preference",  # "auto", "gpu", "cpu"
    "USER_KNN_THRESHOLD": "user_knn_threshold",  # 10-500
    "USER_CHAIN_SELECTION": "user_chain_selection",  # "auto", "A", "B", etc.
    "USER_SIMILARITY_THRESHOLD": "user_similarity_threshold",  # 0.0-1.0
}
