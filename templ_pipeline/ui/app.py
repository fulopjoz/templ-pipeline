"""
TEMPL Pipeline Web Application - Production Version 2.0

Professional molecular pose prediction interface with modern UI/UX design.
Enhanced with advanced error handling and configuration management.
"""

import streamlit as st
import time
import logging
import sys
import multiprocessing
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union

# PHASE 1: IMMEDIATE PAGE CONFIG
st.set_page_config(
    page_title="TEMPL Pipeline",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# IMMEDIATE CSS TEXT HIDING - Applied before anything else renders
st.markdown("""
<style>
/* EMERGENCY: Hide any accidentally displayed CSS text immediately */
body *:not(style):not(script) {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
}

/* Hide CSS-like text patterns and accidental displays */
.css-text-visible,
.css-text-display,
.accidentally-visible-css,
.css-visible-text {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}

/* Prevent CSS text from showing during load */
.stApp > div:first-child {
    display: block !important;
}

/* Override any accidentally visible text with CSS-like patterns */
*[style*="display: block"]:empty,
*[style*="visibility: visible"]:empty {
    display: none !important;
}
</style>
<script>
// Immediate JavaScript to hide CSS-like text content
(function() {
    const hidePattern = /\/\*[\s\S]*?\*\/|\.css-|@media|display:\s*block|visibility:\s*visible/;
    
    function hideCSSLikeText() {
        document.querySelectorAll('*').forEach(el => {
            if (el.textContent && hidePattern.test(el.textContent) && 
                !el.matches('style, script, .stApp, [class*="st"]')) {
                el.style.display = 'none';
                el.style.visibility = 'hidden';
            }
        });
    }
    
    // Run immediately
    hideCSSLikeText();
    
    // Run on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', hideCSSLikeText);
    }
})();
</script>
""", unsafe_allow_html=True)

# PHASE 2: IMMEDIATE LAYOUT FIXES  
# First, add emergency CSS/JS to hide any accidentally displayed CSS text
st.markdown("""
<style>
/* Emergency CSS text hiding */
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
/* Hide CSS-like text that might appear */
.css-text-visible {
    display: none !important;
    visibility: hidden !important;
}
</style>
<script>
// Hide any text content that looks like CSS
(function() {
    function hideCSSText() {
        const allElements = document.querySelectorAll('*');
        allElements.forEach(element => {
            if (element.textContent && 
                (element.textContent.includes('/* Streamlit app container') ||
                 element.textContent.includes('/* Main section containers') ||
                 element.textContent.includes('/* Block containers'))) {
                element.style.display = 'none';
                element.style.visibility = 'hidden';
            }
        });
    }
    
    // Run immediately and on DOM changes
    hideCSSText();
    
    // Also run after a short delay in case content loads later
    setTimeout(hideCSSText, 100);
    setTimeout(hideCSSText, 500);
    
    // Set up observer for dynamic content
    if (typeof MutationObserver !== 'undefined') {
        const observer = new MutationObserver(hideCSSText);
        observer.observe(document.body, { childList: true, subtree: true });
    }
})();
</script>
""", unsafe_allow_html=True)

from templ_pipeline.ui.styles.early_layout import apply_layout_fixes
apply_layout_fixes()

# PHASE 3: Essential imports after layout fixes
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PHASE 4: QA-Enhanced imports with advanced error handling and configuration
try:
    from templ_pipeline.ui.config.settings import get_config
    from templ_pipeline.ui.core.error_handling import get_error_manager
    app_config = get_config()
    error_manager = get_error_manager()
    logger.info("Advanced configuration and error handling loaded successfully")
except ImportError as e:
    logger.warning(f"Advanced QA modules not available: {e}")
    # Fallback configuration
    class FallbackConfig:
        def __init__(self):
            self.resource_limits = {"max_file_size_mb": 5, "max_templates": 100}
            self.ui_settings = {"default_pose_alignment": "aligned"}
            self.scientific = {"quality_thresholds": {"excellent": 0.8, "good": 0.6, "moderate": 0.4}}
        def get_setting(self, category: str, key: str, default: Any = None) -> Any:
            return getattr(self, category, {}).get(key, default)
    
    app_config = FallbackConfig()
    error_manager = None

# Import pipeline functionality
from templ_pipeline.ui.services.pipeline_service import run_pipeline
from templ_pipeline.ui.utils.molecular_utils import (
    validate_smiles_input, 
    validate_sdf_input,
    load_templates_from_uploaded_sdf,
    save_uploaded_file,
    create_best_poses_sdf,
    create_all_conformers_sdf,
    display_molecule,
    get_rdkit_modules
)

# Enhanced CSS for modern UI
st.markdown("""
<style>
/* Hide any accidentally displayed CSS text */
.css-visible-text {
    display: none !important;
    visibility: hidden !important;
}

/* Modern UI Enhancements */
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(168, 85, 247, 0.1));
    border-radius: 1rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(99, 102, 241, 0.2);
}

.feature-card {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 0.75rem;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
}

.input-section {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 1rem;
    padding: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 2rem;
}

.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 2rem;
    font-size: 0.9rem;
    font-weight: 500;
}

.status-success {
    background: rgba(34, 197, 94, 0.1);
    color: rgb(34, 197, 94);
    border: 1px solid rgba(34, 197, 94, 0.2);
}

.status-error {
    background: rgba(239, 68, 68, 0.1);
    color: rgb(239, 68, 68);
    border: 1px solid rgba(239, 68, 68, 0.2);
}

.status-warning {
    background: rgba(245, 158, 11, 0.1);
    color: rgb(245, 158, 11);
    border: 1px solid rgba(245, 158, 11, 0.2);
}

.progress-container {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 0.75rem;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.metric-container {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 0.5rem;
    padding: 1rem;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

/* Responsive improvements */
@media (max-width: 768px) {
    .main-header {
        padding: 1rem 0;
    }
    
    .input-section {
        padding: 1rem;
    }
    
    .feature-card {
        padding: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialize enhanced session state with QA improvements
def initialize_session_state() -> None:
    """Initialize session state with enhanced tracking and error context"""
    try:
        # Register session context for error tracking
        if error_manager:
            session_context = {
                'session_id': id(st.session_state),
                'app_version': getattr(app_config, 'app_version', '2.0.0'),
                'initialization_time': time.time()
            }
            error_manager.register_context('session_init', session_context)
        
        # Configuration-driven defaults
        max_file_size = app_config.get_setting('resource_limits', 'max_file_size_mb', 5)
        default_templates = app_config.get_setting('resource_limits', 'default_templates', 100)
        show_advanced = app_config.get_setting('ui_settings', 'show_advanced_options', False)
        
        default_state = {
            'app_initialized': True,
            'query_mol': None,
            'input_smiles': None,
            'protein_pdb_id': None,
            'protein_file_path': None,
            'custom_templates': None,
            'poses': {},
            'processing_stage': None,
            'show_advanced': show_advanced,
            'last_validation': None,
            'processing_start_time': None,
            'max_file_size_mb': max_file_size,
            'max_templates': default_templates,
            'error_count': 0,
            'session_context': {}
        }
        
        for key, default_value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        logger.debug("Session state initialized successfully")
        
    except Exception as e:
        logger.error(f"Session state initialization failed: {e}")
        if error_manager:
            error_manager.handle_error(
                'CONFIGURATION_ERROR',
                e,
                'session state initialization',
                show_recovery=False
            )

initialize_session_state()

# Enhanced helper functions with error handling
def show_status_indicator(status_type: str, message: str, error_id: Optional[str] = None) -> None:
    """Display modern status indicator with accessibility features and error handling
    
    Args:
        status_type: Type of status (success, error, warning, info)
        message: Message to display
        error_id: Optional error ID for tracking
    """
    try:
        # Input validation with detailed logging
        if not isinstance(status_type, str):
            logger.error(f"Invalid status_type: {type(status_type)} - {status_type}")
            status_type = "info"
        
        if not isinstance(message, str):
            logger.error(f"Invalid message: {type(message)} - {message}")
            message = "Status message display error"
        
        # Sanitize inputs
        status_type = status_type.lower().strip()
        valid_types = {"success", "error", "warning", "info"}
        if status_type not in valid_types:
            logger.warning(f"Unknown status type: {status_type}, defaulting to 'info'")
            status_type = "info"
        
        status_class = f"status-{status_type}"
        icon_map = {
            "success": "✅", 
            "error": "❌", 
            "warning": "⚠️", 
            "info": "ℹ️"
        }
        icon = icon_map[status_type]
        
        # Enhanced accessibility attributes
        aria_label = f"{status_type.title()} message: {message}"
        role = "status" if status_type == "info" else "alert"
        
        # Include error ID if provided
        display_message = message
        if error_id and status_type == "error":
            display_message = f"{message} (Ref: {error_id})"
        
        st.markdown(f"""
        <div class="status-indicator {status_class}" 
             role="{role}" 
             aria-label="{aria_label}"
             tabindex="0">
            <span aria-hidden="true">{icon}</span>
            <span>{display_message}</span>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Status indicator display failed: {e}")
        # Fallback to simple Streamlit error
        if status_type == "error":
            st.error(message)
        elif status_type == "warning":
            st.warning(message)
        elif status_type == "success":
            st.success(message)
        else:
            st.info(message)

def show_processing_progress(stage: str, progress: float, estimated_time: Optional[str] = None) -> None:
    """Display enhanced progress indicator with error handling
    
    Args:
        stage: Current processing stage description
        progress: Progress value between 0.0 and 1.0
        estimated_time: Optional estimated time remaining
    """
    try:
        # Input validation
        if not isinstance(stage, str):
            logger.error(f"Invalid stage type: {type(stage)}")
            stage = "Processing..."
        
        if not isinstance(progress, (int, float)) or not 0 <= progress <= 1:
            logger.error(f"Invalid progress value: {progress}")
            progress = 0.0
        
        # Register processing context for error tracking
        if error_manager:
            processing_context = {
                'stage': stage,
                'progress': progress,
                'estimated_time': estimated_time,
                'start_time': st.session_state.get('processing_start_time'),
                'timestamp': time.time()
            }
            error_manager.register_context('processing_progress', processing_context)
        
        st.markdown('<div class="progress-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{stage}**")
            progress_bar = st.progress(progress)
            
            if estimated_time:
                st.caption(f"Estimated time remaining: {estimated_time}")
        
        with col2:
            if st.button("Cancel", type="secondary", key="cancel_processing"):
                # Register cancellation event
                if error_manager:
                    cancel_context = {
                        'cancelled_stage': stage,
                        'cancelled_progress': progress,
                        'cancellation_time': time.time()
                    }
                    error_manager.register_context('processing_cancelled', cancel_context)
                
                st.session_state.processing_stage = None
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Progress display failed: {e}")
        if error_manager:
            error_manager.handle_error(
                'CRITICAL',
                e,
                'progress display',
                show_recovery=False
            )
        # Fallback progress display
        st.progress(progress if isinstance(progress, (int, float)) and 0 <= progress <= 1 else 0.0)

def validate_inputs() -> Tuple[bool, str]:
    """Enhanced input validation with helpful messages and error tracking
    
    Returns:
        Tuple of (is_valid, message)
    """
    try:
        # Get configuration-driven limits
        min_atoms = app_config.get_setting('resource_limits', 'min_molecule_atoms', 3)
        max_atoms = app_config.get_setting('resource_limits', 'max_molecule_atoms', 200)
        
        # Collect validation context
        validation_context = {
            'timestamp': time.time(),
            'session_id': id(st.session_state),
            'validation_attempt': st.session_state.get('error_count', 0) + 1
        }
        
        molecule_input = st.session_state.get('input_smiles')
        protein_input = st.session_state.get('protein_pdb_id') or st.session_state.get('protein_file_path')
        custom_templates = st.session_state.get('custom_templates')
        
        # Add inputs to validation context
        validation_context.update({
            'has_molecule': bool(molecule_input),
            'has_protein': bool(protein_input),
            'has_custom_templates': bool(custom_templates),
            'molecule_length': len(molecule_input) if molecule_input else 0
        })
        
        # Register validation context
        if error_manager:
            error_manager.register_context('input_validation', validation_context)
        
        # Validate molecule input
        if not molecule_input:
            error_msg = "Please provide a molecule (SMILES string or upload SDF/MOL file)"
            if error_manager:
                error_manager.handle_error(
                    'VALIDATION_ERROR',
                    ValueError("Missing molecule input"),
                    'input validation',
                    error_subtype='empty_input',
                    show_recovery=True
                )
            return False, error_msg
        
        # Validate molecule complexity (basic check)
        if len(molecule_input) < 2:
            error_msg = f"Molecule too simple. Minimum {min_atoms} atoms required."
            if error_manager:
                error_manager.handle_error(
                    'VALIDATION_ERROR',
                    ValueError("Molecule too small"),
                    'input validation',
                    user_message=error_msg,
                    error_subtype='molecule_too_small',
                    show_recovery=True
                )
            return False, error_msg
        
        if len(molecule_input) > 500:  # SMILES length check as proxy
            error_msg = f"Molecule too complex. Maximum ~{max_atoms} atoms supported."
            if error_manager:
                error_manager.handle_error(
                    'VALIDATION_ERROR',
                    ValueError("Molecule too large"),
                    'input validation',
                    user_message=error_msg,
                    error_subtype='molecule_too_large',
                    show_recovery=True
                )
            return False, error_msg
        
        # Validate protein/template input
        if not (protein_input or custom_templates):
            error_msg = "Please provide a protein target (PDB ID, upload PDB file, or custom templates)"
            if error_manager:
                error_manager.handle_error(
                    'VALIDATION_ERROR',
                    ValueError("Missing protein target"),
                    'input validation',
                    user_message=error_msg,
                    error_subtype='empty_input',
                    show_recovery=True
                )
            return False, error_msg
        
        # Validate PDB ID format if provided
        if protein_input and isinstance(protein_input, str) and len(protein_input) == 4:
            if not protein_input.isalnum():
                error_msg = "PDB ID must contain only letters and numbers (e.g., '1abc')"
                if error_manager:
                    error_manager.handle_error(
                        'VALIDATION_ERROR',
                        ValueError("Invalid PDB ID format"),
                        'input validation',
                        user_message=error_msg,
                        error_subtype='invalid_pdb_id',
                        show_recovery=True
                    )
                return False, error_msg
        
        logger.info("Input validation successful")
        return True, "All inputs valid"
        
    except Exception as e:
        logger.error(f"Validation function failed: {e}")
        if error_manager:
            error_manager.handle_error(
                'CRITICAL',
                e,
                'input validation system',
                show_recovery=False
            )
        # Fallback validation
        return False, "Validation system error - please try again"

# MAIN APPLICATION LAYOUT

# Professional Header
st.markdown("""
<div class="main-header">
    <h1>TEMPL Pipeline</h1>
    <h3>Template-based Protein-Ligand Pose Prediction</h3>
    <p>Predict 3D binding poses for small molecules using advanced template-guided methods</p>
</div>
""", unsafe_allow_html=True)

# Feature Overview (Compact)
with st.expander("About TEMPL Pipeline ❓", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Core Features**
        - 3D pose prediction
        - Template-guided generation
        - Comprehensive scoring
        - Multiple conformers
        """)
    
    with col2:
        st.markdown("""
        **How It Works**
        1. Analyze your molecule
        2. Find similar templates
        3. Generate conformers
        4. Score and rank poses
        """)
    
    with col3:
        st.markdown("""
        **Output**
        - Ranked pose predictions
        - Quality assessments
        - Downloadable SDF files
        - Detailed scoring metrics
        """)

# Enhanced Input Section
st.markdown('<div class="input-section">', unsafe_allow_html=True)
st.markdown("## Input Configuration")

# Input validation status
is_valid, validation_message = validate_inputs()
if st.session_state.get('last_validation') != validation_message:
    st.session_state.last_validation = validation_message

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### Query Molecule")
    
    input_method = st.radio(
        "Choose input method:",
        ["SMILES", "Upload File"],
        horizontal=True,
        key="mol_input_method",
        help="SMILES: Text representation of molecular structure. File: Upload SDF or MOL format"
    )
    
    if input_method == "SMILES":
        smiles = st.text_input(
            "SMILES String",
            placeholder="e.g., COc1ccc(C(C)=O)c(O)c1 (aspirin-like compound)",
            key="smiles_input",
            help="Enter a valid SMILES representation. Need help? Try pubchem.ncbi.nlm.nih.gov"
        )
        
        if smiles:
            valid, msg, mol_data = validate_smiles_input(smiles)
            if valid:
                if mol_data is not None:
                    Chem, AllChem, Draw = get_rdkit_modules()
                    mol = Chem.Mol(mol_data)
                else:
                    mol = None
                
                st.session_state.query_mol = mol
                st.session_state.input_smiles = smiles
                
                show_status_indicator("success", msg)
                
                # Enhanced molecule preview
                if mol:
                    with st.container():
                        st.markdown("**Molecule Preview:**")
                        display_molecule(mol, width=280, height=200)
            else:
                show_status_indicator("error", msg)
                with st.expander("💡 Need help with SMILES?"):
                    st.markdown("""
                    **Common SMILES examples:**
                    - Water: `O`
                    - Ethanol: `CCO`
                    - Benzene: `c1ccccc1`
                    - Aspirin: `CC(=O)OC1=CC=CC=C1C(=O)O`
                    
                    **Resources:**
                    - [PubChem](https://pubchem.ncbi.nlm.nih.gov) - Search compounds
                    - [SMILES Tutorial](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html)
                    """)
    else:
        uploaded_file = st.file_uploader(
            "Upload Molecule File",
            type=["sdf", "mol"],
            key="mol_file_upload",
            help="Supported formats: SDF, MOL (max 5MB)"
        )
        
        if uploaded_file is not None:
            try:
                # Use configuration-driven file size limit
                max_size_mb = st.session_state.get('max_file_size_mb', 5)
                max_size_bytes = max_size_mb * 1024 * 1024
                
                # Register file upload context
                if error_manager:
                    upload_context = {
                        'filename': uploaded_file.name,
                        'file_size': uploaded_file.size,
                        'file_type': uploaded_file.type,
                        'max_allowed_size': max_size_bytes,
                        'upload_timestamp': time.time()
                    }
                    error_manager.register_context('file_upload', upload_context)
                
                if uploaded_file.size > max_size_bytes:
                    error_msg = f"File too large (max {max_size_mb}MB). File size: {uploaded_file.size / (1024*1024):.1f}MB"
                    if error_manager:
                        error_id = error_manager.handle_error(
                            'FILE_UPLOAD',
                            ValueError(f"File size {uploaded_file.size} exceeds limit {max_size_bytes}"),
                            'file upload validation',
                            user_message=error_msg,
                            error_subtype='size_exceeded'
                        )
                        show_status_indicator("error", error_msg, error_id)
                    else:
                        show_status_indicator("error", error_msg)
                else:
                    # Attempt file validation with enhanced error handling
                    try:
                        valid, msg, mol = validate_sdf_input(uploaded_file)
                        if valid:
                            st.session_state.query_mol = mol
                            Chem, AllChem, Draw = get_rdkit_modules()
                            st.session_state.input_smiles = Chem.MolToSmiles(mol)
                            
                            show_status_indicator("success", msg)
                            
                            # Show molecule preview with error handling
                            if mol:
                                try:
                                    with st.container():
                                        st.markdown("**Uploaded Molecule:**")
                                        display_molecule(mol, width=280, height=200)
                                except Exception as display_error:
                                    logger.warning(f"Molecule display failed: {display_error}")
                                    st.info("Molecule uploaded successfully (preview unavailable)")
                        else:
                            # Enhanced error handling for validation failures
                            if error_manager:
                                error_id = error_manager.handle_error(
                                    'MOLECULAR_PROCESSING',
                                    ValueError(f"Molecule validation failed: {msg}"),
                                    'molecule file validation',
                                    user_message=msg,
                                    error_subtype='processing_failed',
                                    additional_context={'filename': uploaded_file.name}
                                )
                                show_status_indicator("error", msg, error_id)
                            else:
                                show_status_indicator("error", msg)
                    
                    except Exception as validation_error:
                        logger.error(f"File validation failed: {validation_error}")
                        if error_manager:
                            error_id = error_manager.handle_error(
                                'FILE_UPLOAD',
                                validation_error,
                                'file validation process',
                                error_subtype='processing_failed'
                            )
                            show_status_indicator("error", "File processing failed", error_id)
                        else:
                            show_status_indicator("error", "File processing failed")
                            
            except Exception as upload_error:
                logger.error(f"File upload handling failed: {upload_error}")
                if error_manager:
                    error_id = error_manager.handle_error(
                        'CRITICAL',
                        upload_error,
                        'file upload system'
                    )
                    show_status_indicator("error", "File upload system error", error_id)
                else:
                    show_status_indicator("error", "File upload system error")

with col2:
    st.markdown("### Target Protein")
    
    protein_method = st.radio(
        "Choose input method:",
        ["PDB ID", "Upload File", "Custom Templates"],
        horizontal=True,
        key="prot_input_method",
        help="PDB ID: 4-character code from Protein Data Bank. File: Upload PDB structure. Templates: Use custom template molecules"
    )
    
    if protein_method == "PDB ID":
        pdb_id = st.text_input(
            "PDB ID",
            placeholder="e.g., 1iky, 2xyz, 3abc",
            key="pdb_id_input",
            help="Enter a 4-character PDB identifier. Browse structures at rcsb.org"
        )
        
        if pdb_id:
            if len(pdb_id) == 4 and pdb_id.isalnum():
                st.session_state.protein_pdb_id = pdb_id.lower()
                st.session_state.protein_file_path = None
                st.session_state.custom_templates = None
                
                show_status_indicator("success", f"PDB ID: {pdb_id.upper()}")
                
                # Show PDB info
                with st.container():
                    st.markdown(f"**Target Structure:** [View at RCSB](https://www.rcsb.org/structure/{pdb_id.upper()})")
            else:
                show_status_indicator("error", "PDB ID must be exactly 4 alphanumeric characters")
                with st.expander("💡 Need help finding PDB IDs?"):
                    st.markdown("""
                    **Popular examples:**
                    - `1iky` - HIV protease
                    - `2xyz` - Various enzymes
                    - `3abc` - Different proteins
                    
                    **Find structures at:**
                    - [RCSB Protein Data Bank](https://www.rcsb.org)
                    - Search by protein name or function
                    """)
                    
    elif protein_method == "Upload File":
        pdb_file = st.file_uploader(
            "Upload PDB File",
            type=["pdb"],
            key="pdb_file_upload",
            help="Upload a protein structure file in PDB format (max 5MB)"
        )
        
        if pdb_file is not None:
            if pdb_file.size > 5 * 1024 * 1024:
                show_status_indicator("error", "File too large (max 5MB)")
            else:
                file_path = save_uploaded_file(pdb_file)
                st.session_state.protein_file_path = file_path
                st.session_state.protein_pdb_id = None
                st.session_state.custom_templates = None
                
                show_status_indicator("success", f"PDB file uploaded: {pdb_file.name}")
            
    else:  # Custom Templates
        st.markdown("**Advanced: Custom Template Molecules**")
        st.caption("Upload SDF containing template molecules for MCS-based pose generation")
        
        template_file = st.file_uploader(
            "Upload Template SDF",
            type=["sdf"],
            key="template_file_upload",
            help="SDF file with multiple template molecules (max 10MB)"
        )
        
        if template_file is not None:
            if template_file.size > 10 * 1024 * 1024:
                show_status_indicator("error", "File too large (max 10MB)")
            else:
                templates = load_templates_from_uploaded_sdf(template_file)
                if templates:
                    st.session_state.custom_templates = templates
                    st.session_state.protein_pdb_id = None
                    st.session_state.protein_file_path = None
                    
                    show_status_indicator("success", f"Loaded {len(templates)} template molecules")
                else:
                    show_status_indicator("error", "No valid molecules found in SDF")

st.markdown('</div>', unsafe_allow_html=True)

# Advanced Settings (Enhanced with Configuration)
with st.expander("Advanced Settings", expanded=st.session_state.get('show_advanced', False)):
    col1, col2 = st.columns(2)
    
    with col1:
        # Use configuration-driven default for pose alignment
        default_alignment = app_config.get_setting('ui_settings', 'default_pose_alignment', 'aligned')
        alignment_index = 0 if default_alignment == 'aligned' else 1
        
        use_aligned_poses = st.radio(
            "Pose Alignment Mode:",
            ["Aligned Poses", "Original Geometry"],
            index=alignment_index,
            help="Aligned: Poses positioned relative to templates (recommended). Original: Preserves conformer shape.",
            key="pose_alignment_mode"
        ) == "Aligned Poses"
        
        num_conformers = st.slider(
            "Number of Conformers:",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            help="More conformers = better coverage but longer processing time"
        )
    
    with col2:
        # Use configuration-driven default for max templates
        default_max_templates = st.session_state.get('max_templates', 100)
        config_max_templates = app_config.get_setting('resource_limits', 'max_templates', 500)
        
        max_templates = st.slider(
            "Maximum Templates:",
            min_value=10,
            max_value=min(200, config_max_templates),
            value=default_max_templates,
            step=10,
            help=f"Number of template structures to consider (system limit: {config_max_templates})"
        )
        
        # QA: Show system status if error manager is available
        if error_manager and st.checkbox("Show System Status", help="Display error tracking and system health information"):
            with st.expander("System Status", expanded=True):
                error_stats = error_manager.get_error_stats()
                
                status_col1, status_col2 = st.columns(2)
                with status_col1:
                    st.metric("Session Errors", st.session_state.get('error_count', 0))
                    st.metric("Total Errors", error_stats.get('total_errors', 0))
                
                with status_col2:
                    if error_stats.get('most_common_category'):
                        st.metric("Most Common", error_stats['most_common_category'])
                    recent_count = error_stats.get('recent_errors_count', 0)
                    st.metric("Recent Errors (1h)", recent_count)
                
                # Show recent errors if any
                if error_stats.get('total_errors', 0) > 0:
                    recent_errors = error_manager.get_recent_errors(3)
                    if recent_errors:
                        st.markdown("**Recent Issues:**")
                        for error in recent_errors[-3:]:
                            st.caption(f"• {error['category']}: {error['user_message'][:50]}...")
        
        # Configuration validation status
        if hasattr(app_config, 'validate_configuration'):
            config_status = app_config.validate_configuration()
            if config_status.get('warnings'):
                with st.expander("Configuration Warnings", expanded=False):
                    for warning in config_status['warnings']:
                        st.warning(warning)

# Processing Section
if is_valid:
    st.markdown("## Run Prediction")
    
    # Show input summary
    with st.container():
        st.markdown("### Input Summary")
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.markdown("**Molecule:**")
            if st.session_state.get('input_smiles'):
                st.code(st.session_state.input_smiles[:50] + "..." if len(st.session_state.input_smiles) > 50 else st.session_state.input_smiles)
        
        with summary_col2:
            st.markdown("**Protein:**")
            if st.session_state.get('protein_pdb_id'):
                st.code(f"PDB ID: {st.session_state.protein_pdb_id.upper()}")
            elif st.session_state.get('protein_file_path'):
                st.code("Uploaded PDB file")
            elif st.session_state.get('custom_templates'):
                st.code(f"{len(st.session_state.custom_templates)} custom templates")
    
    st.divider()
    
    # Enhanced prediction button
    prediction_col1, prediction_col2, prediction_col3 = st.columns([1, 2, 1])
    
    with prediction_col2:
        if st.button(
            "PREDICT POSES",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.get('processing_stage') is not None
        ):
            st.session_state.processing_stage = "initializing"
            st.session_state.processing_start_time = time.time()
            st.rerun()

# Processing Status Display
if st.session_state.get('processing_stage'):
    st.markdown("## Processing Your Request")
    
    processing_stages = {
        "initializing": ("Initializing pipeline...", 0.1),
        "loading": ("Loading protein structure...", 0.2),
        "templates": ("Finding template molecules...", 0.4),
        "conformers": ("Generating molecular conformers...", 0.6),
        "scoring": ("Scoring and ranking poses...", 0.8),
        "finalizing": ("Finalizing results...", 0.9),
        "complete": ("Processing complete!", 1.0)
    }
    
    stage = st.session_state.processing_stage
    if stage in processing_stages:
        stage_text, progress = processing_stages[stage]
        
        # Calculate estimated time
        if st.session_state.processing_start_time:
            elapsed = time.time() - st.session_state.processing_start_time
            if progress > 0.1:
                estimated_total = elapsed / progress
                remaining = max(0, estimated_total - elapsed)
                est_time = f"{remaining:.0f} seconds" if remaining > 0 else "Almost done"
            else:
                est_time = "Calculating..."
        else:
            est_time = None
        
        show_processing_progress(stage_text, progress, est_time)
        
        # Simulate processing (in real app, this would be actual pipeline execution)
        if stage != "complete":
            time.sleep(1)
            next_stages = list(processing_stages.keys())
            current_idx = next_stages.index(stage)
            if current_idx < len(next_stages) - 1:
                st.session_state.processing_stage = next_stages[current_idx + 1]
            st.rerun()
        else:
            # Execute actual pipeline with enhanced error handling
            try:
                logger.info("Running TEMPL pipeline...")
                
                # Register pipeline execution context
                if error_manager:
                    pipeline_context = {
                        'smiles': st.session_state.input_smiles,
                        'protein_source': 'pdb_id' if st.session_state.get('protein_pdb_id') else 'uploaded_file' if st.session_state.get('protein_file_path') else 'custom_templates',
                        'protein_identifier': st.session_state.get('protein_pdb_id') or 'uploaded_file',
                        'use_aligned_poses': use_aligned_poses,
                        'max_templates': max_templates,
                        'execution_start': time.time(),
                        'session_id': id(st.session_state)
                    }
                    error_manager.register_context('pipeline_execution', pipeline_context)
                
                poses = run_pipeline(
                    st.session_state.input_smiles,
                    st.session_state.get('protein_pdb_id') or st.session_state.get('protein_file_path'),
                    st.session_state.get('custom_templates'),
                    use_aligned_poses=use_aligned_poses,
                    max_templates=max_templates,
                    similarity_threshold=None
                )
                
                if poses:
                    st.session_state.poses = poses
                    st.session_state.processing_stage = None
                    
                    # Log successful completion
                    execution_time = time.time() - st.session_state.get('processing_start_time', time.time())
                    logger.info(f"Pipeline execution successful in {execution_time:.2f} seconds")
                    
                    # Register success context
                    if error_manager:
                        success_context = {
                            'execution_time': execution_time,
                            'poses_generated': len(poses),
                            'completion_timestamp': time.time()
                        }
                        error_manager.register_context('pipeline_success', success_context)
                    
                    st.success("Pose prediction completed successfully!")
                    st.rerun()
                else:
                    st.session_state.processing_stage = None
                    
                    # Handle case where pipeline returns empty results
                    error_msg = "No poses could be generated. This might indicate incompatible molecule-protein combination or insufficient templates."
                    
                    if error_manager:
                        error_id = error_manager.handle_error(
                            'PIPELINE_ERROR',
                            ValueError("Pipeline returned no poses"),
                            'pose generation',
                            user_message=error_msg,
                            error_subtype='no_poses_generated',
                            additional_context={
                                'molecule': st.session_state.input_smiles[:50] + "..." if len(st.session_state.input_smiles) > 50 else st.session_state.input_smiles,
                                'protein': st.session_state.get('protein_pdb_id', 'unknown')
                            }
                        )
                        show_status_indicator("error", error_msg, error_id)
                    else:
                        st.error(f"{error_msg}")
                    
            except ImportError as import_error:
                st.session_state.processing_stage = None
                logger.error(f"Pipeline import error: {import_error}")
                
                if error_manager:
                    error_id = error_manager.handle_error(
                        'CONFIGURATION_ERROR',
                        import_error,
                        'pipeline module import',
                        error_subtype='missing_dependencies'
                    )
                    show_status_indicator("error", "Pipeline dependencies not available", error_id)
                else:
                    st.error("Pipeline dependencies not available. Please check installation.")
                    
            except MemoryError as memory_error:
                st.session_state.processing_stage = None
                logger.error(f"Pipeline memory error: {memory_error}")
                
                if error_manager:
                    error_id = error_manager.handle_error(
                        'MEMORY_ERROR',
                        memory_error,
                        'pipeline execution',
                        user_message="Not enough memory to complete processing. Try a smaller molecule or restart the application."
                    )
                    show_status_indicator("error", "Memory limit exceeded", error_id)
                else:
                    st.error("Memory limit exceeded. Try a smaller molecule.")
                    
            except Exception as e:
                st.session_state.processing_stage = None
                logger.error(f"Pipeline execution failed: {e}")
                
                # Increment error count for tracking
                st.session_state.error_count = st.session_state.get('error_count', 0) + 1
                
                if error_manager:
                    error_id = error_manager.handle_error(
                        'PIPELINE_ERROR',
                        e,
                        'pipeline execution',
                        additional_context={
                            'error_count': st.session_state.error_count,
                            'molecule': st.session_state.input_smiles[:100] if st.session_state.input_smiles else 'none',
                            'protein': st.session_state.get('protein_pdb_id', 'unknown')
                        }
                    )
                    show_status_indicator("error", "Pipeline execution failed", error_id)
                else:
                    st.error(f"Pipeline error: {str(e)}")

else:
    if not is_valid:
        st.info(f"{validation_message}")

# Enhanced Results Section
if st.session_state.get('poses'):
    st.markdown("## Prediction Results")
    
    poses = st.session_state.poses
    
    # Find best pose by combo score
    best_method, (best_mol, best_scores) = max(poses.items(), 
                                             key=lambda x: x[1][1].get('combo_score', x[1][1].get('combo', 0)))
    
    shape_score = best_scores.get('shape_score', best_scores.get('shape', 0))
    color_score = best_scores.get('color_score', best_scores.get('color', 0))
    combo_score = best_scores.get('combo_score', best_scores.get('combo', 0))
    
    # Results header with quality indicator
    result_col1, result_col2 = st.columns([2, 1])
    
    with result_col1:
        st.markdown("### Best Predicted Pose")
        st.caption(f"Method: {best_method}")
    
    with result_col2:
        # Enhanced quality assessment
        if combo_score >= 0.8:
            quality_text = "High Confidence"
            quality_description = "Excellent pose prediction"
        elif combo_score >= 0.6:
            quality_text = "Good Prediction"
            quality_description = "Reliable pose prediction"
        elif combo_score >= 0.4:
            quality_text = "Moderate Confidence"
            quality_description = "Fair pose prediction"
        else:
            quality_text = "Low Confidence"
            quality_description = "Consider alternative approaches"
        
        st.markdown(f"**{quality_text}**")
        st.caption(quality_description)
    
    # Enhanced score metrics
    st.markdown("### Scoring Metrics")
    
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            "Shape Similarity",
            f"{shape_score:.3f}",
            help="Geometric similarity to template (0-1 scale)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            "Pharmacophore",
            f"{color_score:.3f}",
            help="Chemical feature similarity (0-1 scale)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            "Overall Score",
            f"{combo_score:.3f}",
            help="Combined shape and pharmacophore score"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced download section
    st.markdown("### Download Results")
    
    download_col1, download_col2 = st.columns(2)
    
    with download_col1:
        # Best poses download
        sdf_data, file_name = create_best_poses_sdf(poses)
        st.download_button(
            f"Best Poses ({len(poses)})",
            data=sdf_data,
            file_name=file_name,
            mime="chemical/x-mdl-sdfile",
            help="Download top scoring poses for each method",
            use_container_width=True
        )
    
    with download_col2:
        # All conformers download (if available)
        if hasattr(st.session_state, 'all_ranked_poses') and st.session_state.all_ranked_poses:
            all_sdf_data, all_file_name = create_all_conformers_sdf()
            st.download_button(
                f"All Conformers ({len(st.session_state.all_ranked_poses)})",
                data=all_sdf_data,
                file_name=all_file_name,
                mime="chemical/x-mdl-sdfile",
                help="Download all generated conformers ranked by score",
                use_container_width=True
            )
        else:
            st.button(
                "All Conformers (N/A)",
                disabled=True,
                help="All ranked poses not available for this prediction",
                use_container_width=True
            )
    
    # Results interpretation
    with st.expander("How to Interpret Results ❓", expanded=False):
        st.markdown("""
        **Score Interpretation:**
        - **Shape Similarity (0-1):** How well the predicted pose matches the geometric shape of template molecules
        - **Pharmacophore (0-1):** Similarity of chemical features and functional groups
        - **Overall Score:** Combined metric indicating prediction confidence
        
        **Quality Guidelines:**
        - **High (≥0.8):** Excellent prediction, high confidence in binding mode
        - **Good (≥0.6):** Reliable prediction, suitable for further analysis
        - **Moderate (≥0.4):** Fair prediction, consider experimental validation
        - **Low (<0.4):** Poor prediction, may need different approach or templates
        
        **Next Steps:**
        - Visualize poses in molecular graphics software (PyMOL, ChimeraX)
        - Perform molecular dynamics simulations for validation
        - Consider experimental testing of predicted binding modes
        """)

# Footer with QA Information
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>TEMPL Pipeline v1.0 | Template-based Protein-Ligand Pose Prediction</p>
    <p>For questions or support, please refer to the documentation</p>
</div>
""", unsafe_allow_html=True)
