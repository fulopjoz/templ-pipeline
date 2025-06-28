"""
TEMPL Pipeline - Simple One-Click Web Application

PERFORMANCE OPTIMIZATIONS APPLIED:
- Fixed relative imports to absolute imports (prevents import failures)
- Moved hardware detection to cached function (prevents redundant execution)
- Moved embedding capability checks to cached function (prevents repeated checks)  
- Fixed session state initialization logic (prevents double initialization)
- Added caching to expensive molecular operations (faster validation)
- Moved page config to main() with exception handling (prevents conflicts)
- Added performance monitoring and cache statistics
"""

import os
import io
import sys
import tempfile
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
# Remove gzip import - now handled by standardized template loading
import logging
import multiprocessing
import time
import functools
import re
import json
from datetime import datetime

# Configure logging for better pipeline visibility
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Import streamlit early for cache decorators
import streamlit as st

# FAIR metadata availability flag (will be checked via cached function)
FAIR_AVAILABLE = False

@st.cache_resource
def check_fair_availability():
    """Check FAIR metadata availability with caching"""
    global FAIR_AVAILABLE
    try:
        from templ_pipeline.fair.core.metadata_engine import MetadataEngine, generate_quick_metadata
        from templ_pipeline.fair.biology.molecular_descriptors import calculate_comprehensive_descriptors
        FAIR_AVAILABLE = True
        logger.info("FAIR metadata engine available")
        return True, {
            'MetadataEngine': MetadataEngine,
            'generate_quick_metadata': generate_quick_metadata,
            'calculate_molecular_descriptors': calculate_comprehensive_descriptors
        }
    except ImportError:
        FAIR_AVAILABLE = False
        logger.warning("FAIR metadata engine not available")
        return False, {}

# Performance timing decorator for diagnostics
def time_function(func):
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Performance: {func.__name__} took {end_time - start_time:.3f} seconds")
        return result
    return wrapper

def is_streamlit_rerun():
    """Detect if this is a Streamlit rerun vs initial load"""
    return hasattr(st.session_state, 'app_initialized') and st.session_state.app_initialized

def log_once(message, level="info"):
    """Log a message only once during the session"""
    if not hasattr(st.session_state, 'logged_messages'):
        st.session_state.logged_messages = set()
    
    if message not in st.session_state.logged_messages:
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        st.session_state.logged_messages.add(message)

# Hardware-aware imports and dependency checking

# Hardware detection variables (will be initialized via cached function)
HARDWARE_DETECTOR = None
HARDWARE_INFO = None

@st.cache_resource
def get_hardware_info():
    """Get hardware information with caching to prevent redundant detection"""
    global HARDWARE_DETECTOR, HARDWARE_INFO
    if HARDWARE_INFO is None:
        try:
            from templ_pipeline.core.hardware_detection import HardwareDetector, get_hardware_recommendation
            HARDWARE_DETECTOR = HardwareDetector()
            HARDWARE_INFO = HARDWARE_DETECTOR.detect_hardware()
            logger.info(f"Hardware detected: {HARDWARE_INFO.recommended_config}")
        except ImportError:
            HARDWARE_DETECTOR = None
            HARDWARE_INFO = None
            logger.warning("Hardware detection not available")
    return HARDWARE_INFO

# Embedding capability variables (will be initialized via cached function)
EMBEDDING_FEATURES_AVAILABLE = False
TORCH_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

@st.cache_resource
def check_embedding_capabilities():
    """Check embedding capabilities with caching to prevent redundant checks"""
    global EMBEDDING_FEATURES_AVAILABLE, TORCH_AVAILABLE, TRANSFORMERS_AVAILABLE
    
    try:
        import torch
        TORCH_AVAILABLE = True
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        else:
            logger.info("PyTorch CPU-only mode")
    except ImportError:
        logger.warning("PyTorch not available - embedding similarity disabled")
        TORCH_AVAILABLE = False

    try:
        import transformers
        TRANSFORMERS_AVAILABLE = True
        EMBEDDING_FEATURES_AVAILABLE = TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE
        logger.info("Protein embedding similarity available")
    except ImportError:
        logger.warning("Transformers not available - embedding similarity disabled")
        TRANSFORMERS_AVAILABLE = False
        EMBEDDING_FEATURES_AVAILABLE = False
    
    return {
        'embedding_available': EMBEDDING_FEATURES_AVAILABLE,
        'torch_available': TORCH_AVAILABLE,
        'transformers_available': TRANSFORMERS_AVAILABLE
    }

# Fix torch.classes compatibility issue with Streamlit
try:
    from templ_pipeline.core.torch_fix import fix_torch_streamlit_compatibility
except ImportError:
    pass

# Apply scoring fixes on module load
try:
    from templ_pipeline.core.scoring import FixedMolecularProcessor, ScoringFixer
    log_once("Enhanced scoring modules loaded successfully")
except ImportError:
    log_once("Enhanced scoring modules not available", "warning")

# Import new optimization modules
try:
    from templ_pipeline.ui.secure_upload import SecureFileUploadHandler, validate_file_secure
    from templ_pipeline.ui.error_handling import get_error_manager, ContextualErrorManager
    from templ_pipeline.ui.memory_manager import get_memory_manager, MolecularSessionManager
    from templ_pipeline.ui.molecular_processor import get_molecular_processor, CachedMolecularProcessor
    OPTIMIZATION_MODULES_AVAILABLE = True
    log_once("Security and memory optimization modules loaded successfully")
except ImportError as e:
    OPTIMIZATION_MODULES_AVAILABLE = False
    log_once(f"Optimization modules not available: {e}", "warning")

# Add project root to path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))

# Lazy imports to improve startup performance
_rdkit_modules = None
_py3dmol = None
_core_modules = None

@time_function
def get_rdkit_modules():
    """Lazy load RDKit modules to improve startup performance"""
    global _rdkit_modules
    if _rdkit_modules is None:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, Draw
        RDLogger.DisableLog('rdApp.*')
        _rdkit_modules = (Chem, AllChem, Draw)
    return _rdkit_modules

@time_function
def get_py3dmol():
    """Lazy load py3Dmol to improve startup performance"""
    global _py3dmol
    if _py3dmol is None:
        import py3Dmol
        _py3dmol = py3Dmol
    return _py3dmol

@time_function
def get_core_modules():
    """Lazy load core pipeline modules to improve startup performance"""
    global _core_modules
    if _core_modules is None:
        from templ_pipeline.core.embedding import EmbeddingManager, get_protein_embedding, _get_gpu_info
        from templ_pipeline.core.mcs import generate_conformers, prepare_mol
        from templ_pipeline.core.scoring import select_best
        _core_modules = {
            'EmbeddingManager': EmbeddingManager,
            'get_protein_embedding': get_protein_embedding,
            '_get_gpu_info': _get_gpu_info,
            'generate_conformers': generate_conformers,
            'prepare_mol': prepare_mol,
            'select_best': select_best
        }
    return _core_modules

# Page configuration will be done in main() to prevent redundant execution

# Enhanced CSS
st.markdown("""
<style>
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

.stButton > button {
    font-size: 18px;
    font-weight: bold;
    padding: 0.75rem 2rem;
    border-radius: 0.5rem;
    width: 100%;
}

.predict-button {
    background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
    color: white;
    border: none;
    font-size: 20px;
    font-weight: bold;
    padding: 1rem 2rem;
    border-radius: 0.8rem;
    margin: 1rem 0;
}

.input-section {
    background: rgba(128, 128, 128, 0.1);
    padding: 1.5rem;
    border-radius: 0.8rem;
    border: 1px solid rgba(128, 128, 128, 0.2);
    margin-bottom: 1rem;
}

.score-excellent { color: #28a745; font-weight: bold; }
.score-good { color: #17a2b8; font-weight: bold; }
.score-fair { color: #ffc107; font-weight: bold; }
.score-poor { color: #dc3545; font-weight: bold; }

.molecule-preview {
    text-align: center;
    padding: 1rem;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 0.5rem;
    border: 1px solid rgba(128, 128, 128, 0.2);
    margin-top: 0.5rem;
}

.status-success { 
    color: #28a745; 
    font-weight: 500;
}
.status-error { 
    color: #dc3545; 
    font-weight: 500;
}
.status-warning { 
    color: #ffc107; 
    font-weight: 500;
}

/* Theme-adaptive card backgrounds */
.info-card {
    padding: 1.5rem;
    border-radius: 0.8rem;
    border: 1px solid rgba(128, 128, 128, 0.2);
    margin-bottom: 1rem;
    background: rgba(128, 128, 128, 0.05);
}

.info-card h4 {
    margin-top: 0;
    opacity: 0.9;
}

.info-card ul, .info-card ol {
    margin-bottom: 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state with lazy loading to improve startup performance
@time_function
def initialize_session_state():
    """Initialize session state variables with performance timing"""
    defaults = {
        "protein_embedding": None,
        "query_mol": None,
        "poses": {},
        "template_used": None,
        "mcs_info": None,
        "protein_pdb_id": None,
        "protein_file_path": None,
        "input_smiles": None,
        "custom_templates": None,
        "all_ranked_poses": None,
        "file_cache": {},
        "hardware_info": None,
        "show_fair_panel": False,
        "fair_metadata": None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


@st.cache_resource
@time_function
def get_embedding_manager():
    """Get embedding manager singleton"""
    # Get EmbeddingManager from lazy loader
    core_modules = get_core_modules()
    EmbeddingManager = core_modules['EmbeddingManager']
    
    # Get path relative to templ_pipeline directory (standalone)
    current_dir = Path(__file__).resolve().parent  # ui/
    templ_pipeline_root = current_dir.parent       # templ_pipeline/
    embedding_path = templ_pipeline_root / "data" / "embeddings" / "protein_embeddings_base.npz"
    
    if embedding_path.exists():
        return EmbeddingManager(str(embedding_path))
    return None

def save_uploaded_file(uploaded_file, suffix=".pdb"):
    """Save uploaded file to temp location with security enhancements"""
    if OPTIMIZATION_MODULES_AVAILABLE:
        try:
            # Use secure file upload handler
            handler = SecureFileUploadHandler()
            file_type = suffix.lstrip('.')
            success, message, secure_path = handler.validate_and_save(uploaded_file, file_type)
            
            if success:
                logger.info(f"Secure file upload: {message}")
                return secure_path
            else:
                # Handle upload error with error manager
                error_manager = get_error_manager()
                error_manager.handle_error(
                    'FILE_UPLOAD', 
                    Exception(message), 
                    'file_upload',
                    user_message=message
                )
                return None
        except Exception as e:
            # Fallback to original method on error
            logger.warning(f"Secure upload failed, using fallback: {e}")
    
    # Original fallback method
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

def validate_smiles_input_impl(smiles):
    """Core SMILES validation logic with optimization enhancements"""
    if OPTIMIZATION_MODULES_AVAILABLE:
        try:
            # Use cached molecular processor for validation
            processor = get_molecular_processor()
            valid, message, canonical = processor.validate_smiles(smiles)
            
            if valid and canonical:
                # Get molecule for binary conversion
                Chem, AllChem, Draw = get_rdkit_modules()
                mol = Chem.MolFromSmiles(canonical)
                if mol:
                    mol_pickle = mol.ToBinary()
                    return True, message, mol_pickle
            
            return valid, message, None
            
        except Exception as e:
            # Fallback to original implementation
            logger.warning(f"Cached validation failed, using fallback: {e}")
    
    # Original implementation as fallback
    try:
        if not smiles or not smiles.strip():
            return False, "Please enter a SMILES string", None
        
        Chem, AllChem, Draw = get_rdkit_modules()
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return False, "Invalid SMILES string format", None
        
        num_atoms = mol.GetNumAtoms()
        if num_atoms < 3:
            return False, "Molecule too small (minimum 3 atoms)", None
        if num_atoms > 200:
            return False, "Molecule too large (maximum 200 atoms)", None
        
        # Convert mol to pickle-able format for caching
        mol_pickle = mol.ToBinary()
        return True, f"Valid molecule ({num_atoms} atoms)", mol_pickle
        
    except Exception as e:
        if OPTIMIZATION_MODULES_AVAILABLE:
            # Use error manager for enhanced error handling
            error_manager = get_error_manager()
            error_manager.handle_error(
                'MOLECULAR_PROCESSING',
                e,
                'smiles_validation',
                error_subtype='invalid_smiles'
            )
        return False, f"Error parsing SMILES: {str(e)}", None

@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour, hide spinner for better UX
@time_function
def validate_smiles_input(smiles):
    """Validate SMILES input with detailed feedback - cached wrapper"""
    return validate_smiles_input_impl(smiles)

@time_function
def validate_sdf_input(sdf_file):
    """Validate SDF file input with security enhancements and caching"""
    if OPTIMIZATION_MODULES_AVAILABLE:
        try:
            # Use secure file upload handler for validation
            handler = SecureFileUploadHandler()
            success, message, secure_path = handler.validate_and_save(sdf_file, 'sdf')
            
            if not success:
                return False, message, None
            
            # Use memory manager for efficient caching
            memory_manager = get_memory_manager()
            
            # Create cache key from file content
            sdf_data = sdf_file.read()
            sdf_file.seek(0)  # Reset file pointer
            file_hash = hash(sdf_data)
            cache_key = f"sdf_{file_hash}_{sdf_file.name}"
            
            # Check memory manager cache first
            cached_mol = memory_manager.get_molecule(cache_key)
            if cached_mol:
                num_atoms = cached_mol.GetNumAtoms()
                return True, f"Valid molecule ({num_atoms} atoms) [cached]", cached_mol
            
            # Process the validated file
            Chem, AllChem, Draw = get_rdkit_modules()
            supplier = Chem.SDMolSupplier()
            supplier.SetData(sdf_data)
            
            mol = None
            for m in supplier:
                if m is not None:
                    mol = m
                    break
            
            if mol is None:
                result = (False, "No valid molecules found in file", None)
            else:
                num_atoms = mol.GetNumAtoms()
                if num_atoms < 3:
                    result = (False, "Molecule too small (minimum 3 atoms)", None)
                elif num_atoms > 200:
                    result = (False, "Molecule too large (maximum 200 atoms)", None)
                else:
                    # Store in memory manager
                    memory_manager.store_molecule(cache_key, mol, {'source': 'sdf_upload'})
                    result = (True, f"Valid molecule ({num_atoms} atoms)", mol)
            
            return result
            
        except Exception as e:
            if OPTIMIZATION_MODULES_AVAILABLE:
                error_manager = get_error_manager()
                error_manager.handle_error(
                    'FILE_UPLOAD',
                    e,
                    'sdf_validation',
                    error_subtype='processing_failed'
                )
            logger.warning(f"Secure SDF validation failed, using fallback: {e}")
    
    # Original implementation as fallback
    try:
        # Create cache key from file content hash
        sdf_data = sdf_file.read()
        file_hash = hash(sdf_data)
        cache_key = f"sdf_{file_hash}_{sdf_file.name}"
        
        # Check cache first
        if cache_key in st.session_state.file_cache:
            cached_result = st.session_state.file_cache[cache_key]
            return cached_result['valid'], cached_result['message'], cached_result['mol']
        
        Chem, AllChem, Draw = get_rdkit_modules()
        supplier = Chem.SDMolSupplier()
        supplier.SetData(sdf_data)
        
        mol = None
        for m in supplier:
            if m is not None:
                mol = m
                break
        
        if mol is None:
            result = (False, "No valid molecules found in file", None)
        else:
            num_atoms = mol.GetNumAtoms()
            if num_atoms < 3:
                result = (False, "Molecule too small (minimum 3 atoms)", None)
            elif num_atoms > 200:
                result = (False, "Molecule too large (maximum 200 atoms)", None)
            else:
                result = (True, f"Valid molecule ({num_atoms} atoms)", mol)
        
        # Cache the result
        st.session_state.file_cache[cache_key] = {
            'valid': result[0],
            'message': result[1], 
            'mol': result[2]
        }
        
        return result
        
    except Exception as e:
        return False, f"Error reading file: {str(e)}", None

def extract_pdb_id_from_file(file_path):
    """Extract PDB ID from uploaded PDB file header"""
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('HEADER'):
                    # PDB ID is typically at positions 62-66 in HEADER line
                    if len(line) >= 66:
                        pdb_id = line[62:66].strip().lower()
                        if len(pdb_id) == 4 and pdb_id.isalnum():
                            return pdb_id
                elif line.startswith('TITLE') or line.startswith('ATOM'):
                    # Stop searching after HEADER section
                    break
        return None
    except Exception:
        return None

def load_templates_from_sdf(template_pdbs, max_templates=100, exclude_target_smiles=None):
    """Load template molecules from SDF file using standardized approach"""
    # Import the standardized template loading function
    from templ_pipeline.core.templates import load_template_molecules_standardized
    
    try:
        # Use the standardized template loading function
        templates, loading_stats = load_template_molecules_standardized(
            template_pdb_ids=template_pdbs[:max_templates],
            max_templates=max_templates,
            exclude_target_smiles=exclude_target_smiles
        )
        
        if "error" in loading_stats:
            st.error(f"Template loading error: {loading_stats['error']}")
            return []
        
        if loading_stats.get("missing_pdbs"):
            missing_count = len(loading_stats["missing_pdbs"])
            if missing_count > 5:
                st.warning(f"Could not find {missing_count} templates in the database")
            else:
                st.warning(f"Could not find templates for: {', '.join(loading_stats['missing_pdbs'][:5])}")
        
        return templates
        
    except Exception as e:
        st.error(f"Error loading templates: {e}")
        return []

def load_templates_from_uploaded_sdf(uploaded_file):
    """Load template molecules from uploaded SDF file"""
    try:
        Chem, AllChem, Draw = get_rdkit_modules()
        sdf_data = uploaded_file.read()
        supplier = Chem.SDMolSupplier()
        supplier.SetData(sdf_data)
        
        templates = []
        pid_pattern = re.compile(r"[0-9][A-Za-z0-9]{3}")  # 4-char PDB-like ID
        
        for idx, mol in enumerate(supplier):
            if mol is None or mol.GetNumAtoms() < 3:
                continue
            
            # Try to derive an identifier for better UI display
            try:
                name_field = mol.GetProp('_Name') if mol.HasProp('_Name') else ""
            except Exception:
                name_field = ""
            
            pid_match = pid_pattern.search(name_field) if name_field else None
            pid = pid_match.group(0).upper() if pid_match else f"TPL{idx:03d}"
            
            # Store as properties recognised by extract_pdb_id_from_template
            mol.SetProp('template_pid', pid)
            mol.SetProp('template_name', name_field or pid)
            
            templates.append(mol)
        
        return templates
    except Exception as e:
        st.error(f"Error reading template SDF file: {e}")
        return []

# @st.cache_data(ttl=3600, show_spinner=False)  # Cache molecule images, hide spinner
# when we used cache there was a distruption in the image generation because of 3D coordinates
def generate_molecule_image(mol_binary, width=400, height=300, highlight_atoms=None):
    """Generate molecule image from binary representation"""
    try:
        Chem, AllChem, Draw = get_rdkit_modules()
        mol = Chem.Mol(mol_binary)
        
        # SMART FIX: Use original molecular structure for visualization if available
        if mol.HasProp("original_smiles"):
            try:
                original_smiles = mol.GetProp("original_smiles")
                clean_mol = Chem.MolFromSmiles(original_smiles)
                if clean_mol:
                    mol = clean_mol
                    logger.debug("Using original molecular structure for image generation")
            except Exception as e:
                logger.debug(f"Could not use original SMILES for image generation: {e}")
        
        # Proper sanitization and hydrogen removal
        mol_copy = Chem.RemoveHs(mol)
        
        # Ensure proper sanitization for visualization
        try:
            Chem.SanitizeMol(mol_copy)
        except Exception as e:
            logger.debug(f"Sanitization failed during image generation: {e}")
            # Continue with unsanitized molecule
        
        # Ensure we have valid 2D coordinates
        if mol_copy.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol_copy)
        
        mol_copy = Chem.MolToSmiles(mol_copy)
        mol_copy = Chem.MolFromSmiles(mol_copy)

        if highlight_atoms:
            img = Draw.MolToImage(mol_copy, size=(width, height), highlightAtoms=highlight_atoms)
        else:
            img = Draw.MolToImage(mol_copy, size=(width, height))
        return img
    except Exception as e:
        logger.error(f"Error generating molecule image: {e}")
        return None



@st.cache_data(ttl=1800, show_spinner=False)  # 30 minute cache for molecular displays
def cached_display_molecule_data(mol_binary, width=400, height=300, highlight_atoms=None):
    """Cached wrapper for molecular display data processing"""
    try:
        Chem, AllChem, Draw = get_rdkit_modules()
        mol = Chem.Mol(mol_binary)
        
        # Create safe copy for processing
        mol_copy = Chem.RemoveHs(mol)
        
        # Generate coordinates if needed
        if mol_copy.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol_copy)
        
        return mol_copy.ToBinary()
    except Exception as e:
        logger.error(f"Error in cached molecular display: {e}")
        return None

def display_molecule(mol, width=400, height=300, title="", highlight_atoms=None):
    """Display molecule as 2D image with optional atom highlighting"""
    if mol is None:
        return
    
    try:
        # Convert mol to binary for caching
        # sanitize the molecule carefully
        import rdkit.Chem as Chem
        
        # SMART FIX: Use original molecular structure for visualization if available
        mol_work = mol
        if mol.HasProp("original_smiles"):
            try:
                original_smiles = mol.GetProp("original_smiles")
                clean_mol = Chem.MolFromSmiles(original_smiles)
                if clean_mol:
                    mol_work = clean_mol
                    logger.debug("Using original molecular structure for visualization")
            except Exception as e:
                logger.debug(f"Could not use original SMILES for visualization: {e}")
                # Fall back to coordinate-manipulated molecule
                mol_work = Chem.Mol(mol)
        else:
            # Create a working copy to avoid modifying the original
            mol_work = Chem.Mol(mol)
        
        try:
            Chem.SanitizeMol(mol_work)
        except:
            # If sanitization fails, try without it
            mol_work = mol
        
        mol_binary = mol_work.ToBinary()
        
        # Ensure highlight_atoms is hashable for caching (convert list to tuple)
        if highlight_atoms is not None:
            highlight_atoms = tuple(highlight_atoms)
            
        img = generate_molecule_image(mol_binary, width, height, highlight_atoms)
        
        if img:
            if title:
                st.write(f"**{title}**")
            st.image(img)
    except Exception as e:
        st.error(f"Error displaying molecule: {e}")

def get_mcs_mol(mol1, mol2):
    """Get MCS as a molecule object"""
    try:
        Chem, AllChem, Draw = get_rdkit_modules()
        mcs = Chem.rdFMCS.FindMCS([mol1, mol2])
        if mcs.numAtoms > 0:
            mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
            if mcs_mol:
                AllChem.Compute2DCoords(mcs_mol)
                return mcs_mol, mcs.smartsString
    except:
        pass
    return None, None

def get_score_color_class(score):
    """Get CSS class for score color coding"""
    if score >= 0.8:
        return "score-excellent"
    elif score >= 0.6:
        return "score-good"
    elif score >= 0.4:
        return "score-fair"
    else:
        return "score-poor"

def get_score_interpretation(score):
    """Get human-readable score interpretation with guidance"""
    if score >= 0.8:
        return "Excellent - High confidence pose"
    elif score >= 0.6:
        return "Good - Reliable pose prediction"
    elif score >= 0.4:
        return "Fair - Moderate confidence"
    else:
        return "Poor - Low confidence, consider alternative approaches"

def safe_get_mcs_mol(mcs_info):
    """Safe access to MCS molecule from mcs_info with debugging"""
    try:        
        if isinstance(mcs_info, (list, tuple)) and len(mcs_info) > 0:
            return mcs_info[0]
        elif isinstance(mcs_info, dict):
            # Handle new dict format from pipeline
            if 'mcs_mol' in mcs_info:
                return mcs_info['mcs_mol']
            elif 'smarts' in mcs_info:
                # Create mol from SMARTS if available
                Chem, AllChem, Draw = get_rdkit_modules()
                smarts = mcs_info['smarts']
                mol = Chem.MolFromSmarts(smarts)
                if mol:
                    try:
                        Chem.SanitizeMol(mol)
                        AllChem.Compute2DCoords(mol)
                        return mol
                    except:
                        return None
        return None
    except Exception as e:
        logger.warning(f"MCS access failed: {e}")
        return None


# not implemented now
# def render_3d_mcs_view(predicted_pose, template_mol, mcs_smarts):
#     """Render 3D view with predicted pose, template, and MCS highlighting"""
#     try:
#         Chem, AllChem, Draw = get_rdkit_modules()
#         py3Dmol = get_py3dmol()
        
#         # Make clean copies
#         pose_copy = Chem.Mol(predicted_pose)
#         template_copy = Chem.Mol(template_mol)
        
#         # Ensure 3D coordinates
#         if pose_copy.GetNumConformers() == 0:
#             AllChem.EmbedMolecule(pose_copy)
#         if template_copy.GetNumConformers() == 0:
#             AllChem.EmbedMolecule(template_copy)
        
#         # Get MCS matches
#         patt = Chem.MolFromSmarts(mcs_smarts)
#         pose_mcs_atoms = list(pose_copy.GetSubstructMatch(patt))
#         template_mcs_atoms = list(template_copy.GetSubstructMatch(patt))
        
#         # Create view
#         view = py3Dmol.view(width=700, height=400)
        
#         # Add predicted pose (blue)
#         pose_block = Chem.MolToMolBlock(pose_copy)
#         view.addModel(pose_block, 'sdf')
#         view.setStyle({'model': 0}, {'stick': {'color': 'blue', 'radius': 0.15}})
        
#         # Add template (gray)  
#         template_block = Chem.MolToMolBlock(template_copy)
#         view.addModel(template_block, 'sdf')
#         view.setStyle({'model': 1}, {'stick': {'color': 'gray', 'radius': 0.15}})
        
#         # Highlight MCS atoms (red)
#         if pose_mcs_atoms:
#             view.addStyle({'model': 0, 'atom': pose_mcs_atoms}, {'stick': {'color': 'red', 'radius': 0.25}})
#         if template_mcs_atoms:
#             view.addStyle({'model': 1, 'atom': template_mcs_atoms}, {'stick': {'color': 'red', 'radius': 0.25}})
        
#         view.zoomTo()
#         return view
#     except Exception as e:
#         st.error(f"Error creating 3D visualization: {e}")
#         return None

def create_best_poses_sdf(poses):
    """Create SDF for the best poses using the same helper used by the CLI.

    This now delegates to `TEMPLPipeline.save_results`, guaranteeing coordinate
    handling identical to the command-line workflow (heavy-atom coords are
    remapped back to the original conformer before writing).
    """

    from templ_pipeline.core.pipeline import TEMPLPipeline
    import tempfile, os
    from pathlib import Path

    # Resolve template identifier for metadata
    template_pid = "unknown"
    if hasattr(st.session_state, "template_info") and st.session_state.template_info:
        template_pid = st.session_state.template_info.get("name", "unknown")

    # Create a temporary output folder
    tmp_dir = Path(tempfile.mkdtemp())

    # Lightweight pipeline instance just for writing
    pipeline = TEMPLPipeline(output_dir=str(tmp_dir))

    sdf_path = pipeline.save_results(poses, template_pid)

    # Read the generated SDF
    with open(sdf_path, "r") as fh:
        sdf_content = fh.read()

    # Return content and a friendly filename
    return sdf_content, "templ_best_poses.sdf"

def create_all_conformers_sdf():
    """Create SDF with all ranked conformers including scores"""
    if not hasattr(st.session_state, 'all_ranked_poses') or not st.session_state.all_ranked_poses:
        return "No ranked poses available", "error.sdf"
    
    Chem, AllChem, Draw = get_rdkit_modules()
    sdf_buffer = io.StringIO()
    writer = Chem.SDWriter(sdf_buffer)
    
    for rank, (pose, scores, original_cid) in enumerate(st.session_state.all_ranked_poses, 1):
        # Create a safe copy of the molecule
        try:
            if isinstance(pose, int):
                # Skip invalid poses
                continue
            mol_copy = Chem.Mol(pose)
        except Exception as e:
            logger.warning(f"Skipping pose {rank} due to copy error: {e}")
            continue
        mol_copy.SetProp("Rank", str(rank))
        mol_copy.SetProp("Shape_Score", f"{scores['shape']:.3f}")
        mol_copy.SetProp("Color_Score", f"{scores['color']:.3f}")
        mol_copy.SetProp("Combo_Score", f"{scores['combo']:.3f}")
        mol_copy.SetProp("Original_Conformer_ID", str(original_cid))
        
        # Add template info if available - use resolved PDB ID
        if hasattr(st.session_state, 'template_info') and st.session_state.template_info:
            template_name = st.session_state.template_info.get('name', 'unknown')
            mol_copy.SetProp("Template_Used", template_name)
            if 'mcs_smarts' in st.session_state.template_info:
                mol_copy.SetProp("MCS_SMARTS", st.session_state.template_info['mcs_smarts'])
            
            # Add additional template metadata  
            mol_copy.SetProp("Template_Index", str(st.session_state.template_info.get('index', 0)))
            mol_copy.SetProp("Total_Templates", str(st.session_state.template_info.get('total_templates', 1)))
            mol_copy.SetProp("Atoms_Matched", str(st.session_state.template_info.get('atoms_matched', 0)))
        
        writer.write(mol_copy)
    
    writer.close()
    return sdf_buffer.getvalue(), "templ_all_conformers_ranked.sdf"

def extract_best_poses_from_ranked(all_ranked_poses):
    """Extract best poses for each scoring method from ranked results"""
    if not all_ranked_poses:
        return {}
    
    best = {"shape": (None, {"shape": -1.0, "color": -1.0, "combo": -1.0}),
            "color": (None, {"shape": -1.0, "color": -1.0, "combo": -1.0}),
            "combo": (None, {"shape": -1.0, "color": -1.0, "combo": -1.0})}
    
    for pose, scores, _ in all_ranked_poses:
        for metric in ["shape", "color", "combo"]:
            if scores[metric] > best[metric][1][metric]:
                best[metric] = (pose, scores)
    
    return best

def extract_pdb_id_from_template(template_mol, index=0):
    """Extract PDB ID from template molecule properties with synthetic template ID support"""
    import re
    
    # Get available properties
    props = list(template_mol.GetPropNames())
    prop_values = {}
    for prop in props:
        try:
            value = template_mol.GetProp(prop)
            prop_values[prop] = value
        except Exception:
            continue
    
    # Comprehensive list of property names that might contain PDB ID
    pdb_properties = [
        # Primary sources (most likely to contain PDB ID)
        'template_pid',      # From true_mcs_pipeline
        'Template_PDB',      # From scoring module  
        'template_pdb',      # Common variant
        'pdb_id', 'PDB_ID',  # Standard names
        'pdb', 'PDB',        # Short forms
        'OriginalPDB',       # From similarity pipeline
        'TemplatePDB',       # Variant form
        # Secondary sources
        '_Name',             # RDKit default
        'template_name',     # Template-specific
        'name', 'Name',      # Generic names
        'title', 'Title',    # Title fields
        'source', 'Source',  # Source fields
        'id', 'ID',          # Generic ID fields
    ]
    
    # Enhanced PDB ID validation patterns including synthetic templates
    pdb_id_patterns = [
        r'\b([0-9][A-Za-z0-9]{3})\b',          # Standard: digit + 3 alphanumeric
        r'^([0-9][A-Za-z0-9]{3})$',            # Exact match
        r'^(TPL\d{3})$',                       # Synthetic template IDs: TPL000, TPL001, etc.
        r'\b(TPL\d{3})\b',                     # Synthetic template IDs in text
        r'^([0-9][A-Za-z0-9]{3})_',            # With underscore suffix
        r'_([0-9][A-Za-z0-9]{3})_',            # Between underscores
        r'([0-9][A-Za-z0-9]{3})\.pdb',         # With .pdb extension
        r'([0-9][A-Za-z0-9]{3})\.',            # With any extension
        r'pdb[_\-:]([0-9][A-Za-z0-9]{3})',     # pdb:1abc format
        r'([0-9][A-Za-z0-9]{3})[_\-]',         # With dash/underscore
    ]
    
    def validate_pdb_id(pdb_candidate):
        """Validate that a candidate string is a proper PDB ID or synthetic template ID"""
        if not pdb_candidate:
            return False
        
        # Check for synthetic template IDs (TPL000, TPL001, etc.)
        if re.match(r'^TPL\d{3}$', pdb_candidate):
            return True
            
        # Check for standard PDB IDs (4 chars, starts with digit)
        if len(pdb_candidate) == 4:
            return pdb_candidate[0].isdigit() and pdb_candidate[1:].isalnum()
            
        return False
    
    # Try to extract PDB ID from molecule properties
    for prop in pdb_properties:
        if template_mol.HasProp(prop):
            value = template_mol.GetProp(prop).strip()
            
            if not value:
                continue
                
            # Direct PDB ID check (standard or synthetic)
            if validate_pdb_id(value):
                return value.upper()
            
            # Try regex patterns to extract PDB ID from longer strings
            for pattern in pdb_id_patterns:
                match = re.search(pattern, value, re.IGNORECASE)
                if match:
                    pdb_candidate = match.group(1).upper()
                    if validate_pdb_id(pdb_candidate):
                        return pdb_candidate
    
    # Enhanced fallback: try to find ANY valid PDB ID pattern
    for prop, value in prop_values.items():
        if isinstance(value, str) and len(value) >= 4:
            # Look for standard PDB IDs
            matches = re.findall(r'[0-9][A-Za-z0-9]{3}', value)
            for match in matches:
                if validate_pdb_id(match):
                    return match.upper()
            
            # Look for synthetic template IDs
            matches = re.findall(r'TPL\d{3}', value, re.IGNORECASE)
            for match in matches:
                return match.upper()
    
    # Final fallback - return template_name if available, otherwise generate synthetic ID
    if template_mol.HasProp('template_name'):
        fallback_name = template_mol.GetProp('template_name').strip()
        if fallback_name:
            return fallback_name.upper()
    
    return f"TPL{index:03d}"

def _format_pipeline_results_for_ui(results, template_mol=None, query_mol=None):
    """Format TEMPLPipeline results for web UI display with enhanced template PDB ID resolution"""
    
    if 'error' in results:
        st.error(f"Pipeline Error: {results['error']}")
        return None
    
    poses = results.get('poses', {})
    if not poses:
        st.error("No poses generated")
        return None

    # Store alignment mode information for UI display
    alignment_used = results.get('alignment_used', True)  # Default to True for backward compatibility
    st.session_state.alignment_used = alignment_used
    logger.info(f"Storing alignment mode in session state: {alignment_used}")

    def resolve_template_pdb_id(template_mol, index, mcs_info, results):
        """Resolve template PDB ID using multiple data sources"""
        
        # Priority 1: Use selected_template_pdb from MCS info if available
        if isinstance(mcs_info, dict) and mcs_info.get('selected_template_pdb'):
            pdb_id = mcs_info['selected_template_pdb']
            return pdb_id.upper()
        
        # Priority 2: Use templates list from results if available
        if 'templates' in results and isinstance(results['templates'], list):
            templates_list = results['templates']
            if index < len(templates_list):
                template_entry = templates_list[index]
                
                # Handle different template list formats
                if isinstance(template_entry, (list, tuple)) and len(template_entry) >= 1:
                    template_pdb = template_entry[0]
                    if isinstance(template_pdb, str) and len(template_pdb) >= 4:
                        return template_pdb.upper()
                elif isinstance(template_entry, str) and len(template_entry) >= 4:
                    return template_entry.upper()
        
        # Priority 3: Extract from template molecule properties
        pdb_id = extract_pdb_id_from_template(template_mol, index)
        return pdb_id
        
        # Priority 4: Check if there's a template_pid property in results
        if 'template_info' in results:
            template_info = results['template_info']
            if isinstance(template_info, dict) and template_info.get('template_pdb'):
                pdb_id = template_info['template_pdb']
                return pdb_id.upper()
        
        # Priority 5: Try to extract from any other results metadata
        for key, value in results.items():
            if 'template' in key.lower() and isinstance(value, (str, list)):
                if isinstance(value, str) and len(value) >= 4:
                    return value.upper()
                elif isinstance(value, list) and value:
                    first_item = value[0]
                    if isinstance(first_item, str) and len(first_item) >= 4:
                        return first_item.upper()
        
        # Final fallback
        return pdb_id

    # Store results in session state for UI components with memory optimization
    if OPTIMIZATION_MODULES_AVAILABLE:
        try:
            memory_manager = get_memory_manager()
            
            # Store template molecules efficiently
            if 'template_molecules' in results and results['template_molecules']:
                template_molecules = results['template_molecules']
                for i, template in enumerate(template_molecules):
                    memory_manager.store_molecule(f"template_{i}", template, {'source': 'pipeline'})
            
            # Store query molecule efficiently
            if query_mol:
                memory_manager.store_molecule("query_molecule", query_mol, {'source': 'user_input'})
            
            # Store poses efficiently using memory manager
            if poses:
                memory_manager.store_pose_results(poses)
                logger.info(f"Stored pose results using memory optimization")
            
        except Exception as e:
            logger.warning(f"Memory optimization failed, using fallback storage: {e}")
    
    # Store results in session state for UI components (original/fallback)
    if 'template_molecules' in results and results['template_molecules']:
        # Get the actually selected template from MCS info
        mcs_info = results.get('mcs_info', {})
        selected_template_index = 0  # Default fallback
        
        # Extract selected template index from MCS results
        if isinstance(mcs_info, dict) and 'selected_template_index' in mcs_info:
            selected_template_index = mcs_info['selected_template_index']
        elif 'templates' in results and mcs_info.get('selected_template_pdb'):
            # Find the template index by PDB ID
            selected_pdb = mcs_info['selected_template_pdb']
            template_pdbs = [t[0] for t in results['templates']]
            if selected_pdb in template_pdbs:
                selected_template_index = template_pdbs.index(selected_pdb)
        
        # Use the actually selected template
        template_molecules = results['template_molecules']
        if selected_template_index < len(template_molecules):
            template_mol = template_molecules[selected_template_index]
        else:
            template_mol = template_molecules[0]  # Fallback
            
        st.session_state.template_used = template_mol
        
        # Resolve the actual PDB ID using enhanced logic
        resolved_pdb_id = resolve_template_pdb_id(template_mol, selected_template_index, mcs_info, results)
        
        st.session_state.template_info = {
            "name": resolved_pdb_id,
            "index": selected_template_index,
            "total_templates": len(template_molecules),
            "mcs_smarts": mcs_info.get('smarts', ""),
            "atoms_matched": mcs_info.get('atom_count', 0) if isinstance(mcs_info, dict) else 0,
            "similarity_score": mcs_info.get('similarity_score', 0.0) if isinstance(mcs_info, dict) else 0.0
        }
        
        # Additional debugging for template resolution
        logger.info(f"Resolved template info: name={resolved_pdb_id}, index={selected_template_index}, total={len(template_molecules)}")
        
    elif template_mol:
        st.session_state.template_used = template_mol
        resolved_pdb_id = resolve_template_pdb_id(template_mol, 0, results.get('mcs_info', {}), results)
        
        st.session_state.template_info = {
            "name": resolved_pdb_id,
            "index": 0,
            "total_templates": 1,
            "mcs_smarts": results.get('mcs_info', {}).get('smarts', ""),
            "atoms_matched": 0,
            "similarity_score": 0.0
        }
    
    # Store MCS info if available - handle dict format
    if 'mcs_info' in results:
        st.session_state.mcs_info = results['mcs_info']
    
    # Store all ranked poses if available
    if 'all_ranked_poses' in results:
        st.session_state.all_ranked_poses = results['all_ranked_poses']
        logger.info(f"Stored {len(results['all_ranked_poses'])} ranked poses in session state")
    
    if query_mol:
        st.session_state.query_mol = query_mol
    
    return poses

class StreamlitProgressHandler(logging.Handler):
    """Custom handler to redirect logs to Streamlit progress display"""
    
    def __init__(self, progress_placeholder):
        super().__init__()
        self.progress_placeholder = progress_placeholder
        self.steps = []
        self.start_time = time.time()
        self.current_phase = None
        self.phase_progress = {}
    
    def emit(self, record):
        try:
            msg = self.format(record)
            
            # Parse log message for progress information
            if any(keyword in msg for keyword in ["Step", "Progress", "Starting", "Loading", "Processing", "Generating"]):
                # Extract phase information
                if "Starting" in msg or "Loading" in msg:
                    self.current_phase = msg.split(":")[-1].strip() if ":" in msg else msg
                    self.phase_progress[self.current_phase] = 0
                
                # Update progress
                elapsed = time.time() - self.start_time
                self.steps.append({
                    'time': elapsed,
                    'message': msg,
                    'level': record.levelname
                })
                
                # Display progress in a structured way
                with self.progress_placeholder.container():
                    # Show phase progress
                    if self.current_phase:
                        st.markdown(f"**Current Phase:** {self.current_phase}")
                    
                    # Progress bar if we can extract percentage
                    if "%" in msg:
                        try:
                            pct_match = re.search(r'(\d+)%', msg)
                            if pct_match:
                                progress = int(pct_match.group(1))
                                st.progress(progress / 100, text=f"Progress: {progress}%")
                        except:
                            pass
                    
                    # Show time elapsed
                    st.caption(f"Time elapsed: {elapsed:.1f}s")
                    
                    # Show recent steps with appropriate styling
                    st.markdown("**Recent Steps:**")
                    for step in self.steps[-5:]:  # Show last 5 steps
                        if step['level'] == 'ERROR':
                            st.error(step['message'])
                        elif step['level'] == 'WARNING':
                            st.warning(step['message'])
                        elif "Success" in step['message'] or "Complete" in step['message']:
                            st.success(step['message'])
                        else:
                            st.info(step['message'])
        except Exception as e:
            # Don't break on display errors
            logger.debug(f"Progress display error: {e}")





def run_pipeline(smiles, protein_input=None, custom_templates=None, use_aligned_poses=True, max_templates=None, similarity_threshold=None):
    """Run the complete TEMPL pipeline using TEMPLPipeline class"""
    
    progress_placeholder = st.empty()
    
    # Detect available hardware using TEMPL hardware detection
    try:
        from templ_pipeline.core.hardware_utils import get_suggested_worker_config, get_hardware_info
        hardware_config = get_suggested_worker_config()
        hardware_info = get_hardware_info()
        max_workers = hardware_config["n_workers"]
        
        # Store hardware info in session state
        st.session_state.hardware_info = hardware_info
    except ImportError:
        # Fallback to basic detection
        total_cores = multiprocessing.cpu_count()
        max_workers = min(total_cores, 4)
        st.session_state.hardware_info = {
            'total_cores': total_cores,
            'used_cores': max_workers,
            'utilization': f"{max_workers}/{total_cores}"
        }
    
    # Set up logging handler for progress
    handler = StreamlitProgressHandler(progress_placeholder)
    handler.setLevel(logging.INFO)
    pipeline_logger = logging.getLogger('templ_pipeline')
    pipeline_logger.addHandler(handler)
    
    try:
        from templ_pipeline.core.pipeline import TEMPLPipeline
        
        with progress_placeholder.container():
            st.info("Starting TEMPL Pipeline")
            hardware_display = st.session_state.hardware_info.get('utilization', f"{max_workers}/unknown")
            st.markdown(f"Hardware: Using {hardware_display} CPU cores")
            alignment_status = "aligned" if use_aligned_poses else "original geometry"
            st.markdown(f"Mode: Returning {alignment_status} poses")
        
        # Initialize pipeline
        pipeline = TEMPLPipeline(
            embedding_path=None,  # Auto-detect
            output_dir=st.session_state.get('temp_dir', 'temp'),
            run_id=None
        )
        
        # Check if using custom templates (MCS-only workflow)
        if custom_templates:
            with progress_placeholder.container():
                st.info("Using custom template molecules...")
                st.success(f"Loaded {len(custom_templates)} custom templates")
                st.info("Generating molecular conformers...")
            
            # Generate poses using custom templates
            query_mol = pipeline.prepare_query_molecule(ligand_smiles=smiles)
            
            pose_results = pipeline.generate_poses(
                query_mol=query_mol,
                template_mols=custom_templates,
                num_conformers=200,
                n_workers=max_workers,
                use_aligned_poses=use_aligned_poses  # Use the alignment control parameter
            )
            
            # Handle both dict and poses format for backward compatibility
            if isinstance(pose_results, dict) and 'poses' in pose_results:
                results = pose_results
            else:
                results = {'poses': pose_results}
            
            with progress_placeholder.container():
                st.success("Pipeline completed successfully!")
            
            return _format_pipeline_results_for_ui(results, custom_templates[0], query_mol)
            
        else:
            # Full pipeline workflow
            with progress_placeholder.container():
                st.info("Processing protein structure...")
                if len(protein_input) == 4 and protein_input.isalnum():
                    st.markdown(f"Database lookup: Checking for PDB {protein_input.upper()}")
            
            # Determine protein input type
            protein_file = None
            protein_pdb_id = None
            
            if len(protein_input) == 4 and protein_input.isalnum():
                protein_pdb_id = protein_input.lower()
            else:
                protein_file = protein_input
            
            # Determine filtering parameters based on user selection
            if max_templates is not None:
                # KNN mode: use max_templates, no similarity threshold
                num_templates = max_templates
                sim_threshold = None
            elif similarity_threshold is not None:
                # Similarity threshold mode: no limit on count, use threshold
                num_templates = None  # No limit
                sim_threshold = similarity_threshold
            else:
                # Fallback to default KNN mode
                num_templates = 100
                sim_threshold = None
            
            # Run full pipeline with real-time progress updates
            results = pipeline.run_full_pipeline(
                protein_file=protein_file,
                protein_pdb_id=protein_pdb_id,
                ligand_smiles=smiles,
                num_templates=num_templates,
                num_conformers=200,
                n_workers=max_workers,
                similarity_threshold=sim_threshold,
                use_aligned_poses=use_aligned_poses  # Use the alignment control parameter
            )
            
            # Show completion
            with progress_placeholder.container():
                st.success("Pipeline completed successfully!")
                if 'templates' in results:
                    st.markdown(f"Found {len(results['templates'])} templates")
                if 'poses' in results:
                    st.markdown(f"Generated {len(results['poses'])} poses")
                alignment_used = results.get('alignment_used', 'unknown')
                st.markdown(f"Poses: Using {'aligned' if alignment_used else 'original geometry'} coordinates")
            
            return _format_pipeline_results_for_ui(results)
    except Exception as e:
        import traceback
        error_details = str(e)
        
        with progress_placeholder.container():
            st.error(f"Pipeline Error: {error_details}")
            
            # Provide specific guidance based on error type
            if "not found in database" in error_details:
                st.info(f"Suggestion: PDB ID '{protein_input}' not available. Try uploading the PDB file directly.")
            elif "Invalid SMILES" in error_details:
                st.info("Suggestion: Check your molecule format. Try a different SMILES string or upload an SDF file.")
            elif "No conformers generated" in error_details:
                st.info("Suggestion: Molecule might be too complex or templates incompatible. Try simpler molecules.")
            elif "hydrogen" in error_details.lower() or "alignment" in error_details.lower():
                st.info("Suggestion: Try using 'Original Geometry' mode in Advanced Settings to avoid alignment issues.")
            

        
        logger.error(f"Pipeline execution failed: {error_details}")
    finally:
        # Clean up logging handler
        pipeline_logger = logging.getLogger('templ_pipeline')
        if handler in pipeline_logger.handlers:
            pipeline_logger.removeHandler(handler)
    
    return None

async def run_pipeline_async(smiles, protein_input=None, custom_templates=None, use_aligned_poses=True, max_templates=None, similarity_threshold=None):
    """
    Async wrapper for run_pipeline to prevent UI blocking.
    
    This function provides non-blocking execution of the TEMPL pipeline within 
    the Streamlit web interface, enabling progress feedback and responsive UI
    during long-running molecular processing operations.
    
    Args:
        smiles (str): SMILES string for target molecule
        protein_input (str, optional): PDB ID or file path for protein target
        custom_templates (list, optional): Custom template molecules for MCS-based pose generation
        use_aligned_poses (bool): Whether to return aligned poses (default: True)
        max_templates (int, optional): Maximum number of templates for KNN filtering
        similarity_threshold (float, optional): Similarity threshold for template filtering
    
    Returns:
        dict or None: Same format as run_pipeline - dict with pose results or None on failure
        
    Raises:
        Same exceptions as run_pipeline, properly propagated from the async wrapper
        
    Note:
        This is a simple ThreadPoolExecutor-based async wrapper that maintains
        full compatibility with the synchronous run_pipeline function while
        enabling async/await usage patterns in UI code.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    # Get the current event loop
    loop = asyncio.get_event_loop()
    
    # Execute the synchronous pipeline in a thread pool to avoid blocking
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = await loop.run_in_executor(
            executor, 
            run_pipeline,
            smiles, protein_input, custom_templates, 
            use_aligned_poses, max_templates, similarity_threshold
        )
    
    return result

def validate_smiles(smiles: str) -> bool:
    """Validate SMILES string using RDKit (real validation)."""
    if not smiles or not isinstance(smiles, str):
        return False
    
    try:
        Chem, AllChem, Draw = get_rdkit_modules()
        mol = Chem.MolFromSmiles(smiles.strip())
        return mol is not None and mol.GetNumAtoms() > 0
    except Exception:
        return False

def process_molecule(smiles: str, engine=None) -> Dict[str, Any]:
    """Process molecule through real TEMPL pipeline."""
    if not validate_smiles(smiles):
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    try:
        if engine is None:
            from templ_pipeline.core.template_engine import TemplateEngine
            engine = TemplateEngine()
        
        # Run actual TEMPL pipeline
        result = engine.run(smiles)
        
        # Add UI-specific metadata
        result['ui_processed'] = True
        result['input_smiles'] = smiles
        
        return result
        
    except Exception as e:
        raise Exception(f"Molecule processing failed: {str(e)}")

def get_pipeline_status() -> Dict[str, Any]:
    """Get current pipeline configuration and status."""
    try:
        from templ_pipeline.core.embedding import EmbeddingManager
        from templ_pipeline.core.template_engine import TemplateEngine
        
        return {
            'embedding_manager_available': True,
            'template_engine_available': True,
            'pipeline_ready': True
        }
    except ImportError as e:
        return {
            'embedding_manager_available': False,
            'template_engine_available': False,
            'pipeline_ready': False,
            'error': str(e)
        }

# Molecular validation functions


def show_hardware_status():
    """Display hardware status and AI capability information"""
    
    HARDWARE_INFO = get_hardware_info()
    if not HARDWARE_INFO:
        return
    
    with st.expander("System Information", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Hardware Configuration:**")
            st.markdown(f"CPU: {HARDWARE_INFO.cpu_count} cores")
            st.markdown(f"RAM: {HARDWARE_INFO.total_ram_gb:.1f} GB")
            if HARDWARE_INFO.gpu_available:
                gpu_info = ", ".join(HARDWARE_INFO.gpu_models[:2])  # Show first 2 GPUs
                st.markdown(f"GPU: {gpu_info}")
                st.markdown(f"VRAM: {HARDWARE_INFO.gpu_memory_gb:.1f} GB")
            else:
                st.markdown("GPU: Not detected")
        
        with col2:
            st.markdown("**Embedding Features:**")
            
            if EMBEDDING_FEATURES_AVAILABLE:
                st.markdown("Protein Embedding Similarity Available")
                if HARDWARE_INFO.gpu_available and TORCH_AVAILABLE:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            st.markdown(f"GPU Acceleration: {torch.cuda.device_count()} device(s)")
                        else:
                            st.markdown("CPU Mode: GPU libraries available but no GPU detected")
                    except:
                        st.markdown("CPU Mode: PyTorch CPU-only")
                else:
                    st.markdown("CPU Mode: Optimized for embedding computation")
            else:
                st.markdown("Limited Features Available")
                missing = []
                if not TORCH_AVAILABLE:
                    missing.append("PyTorch")
                if not TRANSFORMERS_AVAILABLE:
                    missing.append("Transformers")
                
                if missing:
                    st.markdown(f"Missing: {', '.join(missing)}")
                    st.markdown("Solution: Install embedding dependencies:")
                    
                    if HARDWARE_INFO.gpu_available:
                        st.code("pip install torch transformers", language="bash")
                    else:
                        st.code("pip install torch transformers --index-url https://download.pytorch.org/whl/cpu", language="bash")
            
            # Performance recommendation
            config_labels = {
                "cpu-minimal": "BASIC",
                "cpu-optimized": "STANDARD", 
                "gpu-small": "ACCELERATED",
                "gpu-medium": "HIGH-PERFORMANCE",
                "gpu-large": "MAXIMUM"
            }
            
            label = config_labels.get(HARDWARE_INFO.recommended_config, "HARDWARE")
            st.markdown(f"{label}: Recommended: {HARDWARE_INFO.recommended_config}")
        
        # Add optimization status section
        if OPTIMIZATION_MODULES_AVAILABLE:
            st.markdown("---")
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("**Security & Performance:**")
                st.markdown("✅ Secure file upload enabled")
                st.markdown("✅ Enhanced error handling active")
                st.markdown("✅ Memory optimization active")
                st.markdown("✅ Molecular processing cache enabled")
            
            with col4:
                st.markdown("**Optimization Stats:**")
                try:
                    memory_manager = get_memory_manager()
                    mol_processor = get_molecular_processor()
                    
                    memory_stats = memory_manager.get_memory_stats()
                    cache_stats = mol_processor.get_cache_stats()
                    
                    st.markdown(f"Memory cache: {memory_stats['cache_size_mb']:.1f}MB used")
                    st.markdown(f"Cache hit rate: {cache_stats.get('validate_smiles', {}).get('hit_rate', 0):.1f}%")
                    st.markdown(f"Molecules cached: {memory_stats['smiles_cache_entries']}")
                    
                    # Memory optimization button
                    if st.button("Optimize Memory", help="Clean up cache and free memory"):
                        optimization_result = memory_manager.optimize_memory()
                        st.success(f"Freed {optimization_result['memory_saved_mb']:.1f}MB")
                except Exception as e:
                    st.markdown("Stats: Available after first use")
                    
            # Cache performance section
            with st.expander("Cache Performance", expanded=False):
                try:
                    perf_stats = get_performance_stats()
                    st.markdown("**Streamlit Cache Statistics:**")
                    for cache_name, info in perf_stats.items():
                        if info and hasattr(info, 'hits'):
                            hit_rate = (info.hits / max(1, info.hits + info.misses)) * 100
                            st.markdown(f"• {cache_name}: {hit_rate:.1f}% hit rate ({info.hits} hits, {info.misses} misses)")
                        else:
                            st.markdown(f"• {cache_name}: No cache info available")
                except Exception as e:
                    st.markdown("Cache stats: Available after usage")
        else:
            st.markdown("---")
            st.markdown("**Optimization Status:**")
            st.warning("⚠️ Security and performance modules not loaded - using basic functionality")

def check_embedding_requirements_for_feature(feature_name: str) -> bool:
    """Check if embedding requirements are met for a specific feature"""
    if not EMBEDDING_FEATURES_AVAILABLE:
        st.error(f"{feature_name} requires embedding dependencies (PyTorch + Transformers)")
        st.info("Quick Fix: Install missing dependencies and restart the app")
        
        with st.expander("Installation Instructions", expanded=False):
            HARDWARE_INFO = get_hardware_info()
            if HARDWARE_INFO and HARDWARE_INFO.gpu_available:
                st.code("""
# For GPU acceleration
pip install torch transformers

# Or install full embedding bundle
./setup_env_smart.sh --gpu-force
                """, language="bash")
            else:
                st.code("""
# For CPU-only (smaller download)
pip install torch transformers --index-url https://download.pytorch.org/whl/cpu

# Or install full bundle
./setup_env_smart.sh --cpu-only
                """, language="bash")
        
        return False
    
    return True

def validate_molecular_connectivity(mol, step_name="unknown"):
    """Comprehensive molecular connectivity validation"""
    if not mol:
        return False, f"{step_name}: Molecule is None"
        
    try:
        Chem, AllChem, Draw = get_rdkit_modules()
        
        # Check basic validity
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            return False, f"{step_name}: Sanitization failed - {str(e)}"
        
        # Get molecular properties
        atoms = mol.GetNumAtoms()
        bonds = mol.GetNumBonds()
        fragments = Chem.GetMolFrags(mol)
        
        # Basic connectivity checks
        if atoms == 0:
            return False, f"{step_name}: No atoms"
        if bonds == 0 and atoms > 1:
            return False, f"{step_name}: No bonds but multiple atoms"
        if len(fragments) > 1:
            return False, f"{step_name}: Molecule is disconnected ({len(fragments)} fragments)"
            
        # Check for reasonable bonding
        expected_min_bonds = max(0, atoms - len(fragments))
        if bonds < expected_min_bonds:
            return False, f"{step_name}: Too few bonds ({bonds} < {expected_min_bonds})"
            
        # Try to generate SMILES as connectivity test
        try:
            smiles = Chem.MolToSmiles(mol)
            if not smiles or smiles == "":
                return False, f"{step_name}: Cannot generate SMILES"
        except Exception as e:
            return False, f"{step_name}: SMILES generation failed - {str(e)}"
            
        return True, f"{step_name}: Connectivity valid"
        
    except Exception as e:
        return False, f"{step_name}: Validation error - {str(e)}"

def create_safe_molecular_copy(mol, step_name="copy"):
    """Create a safe copy of a molecule with validation"""
    if mol is None:
        return None
    
    try:
        Chem, AllChem, Draw = get_rdkit_modules()
        
        # Create a copy using SMILES roundtrip
        smiles = Chem.MolToSmiles(mol)
        new_mol = Chem.MolFromSmiles(smiles)
        
        if new_mol:
            # Copy coordinates if available
            if mol.GetNumConformers() > 0:
                conf = mol.GetConformer()
                new_conf = Chem.Conformer(new_mol.GetNumAtoms())
                for i in range(new_mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    new_conf.SetAtomPosition(i, pos)
                new_mol.AddConformer(new_conf)
            
            return new_mol
        else:
            logger.error(f"Failed to create safe copy in {step_name}")
            return mol  # Return original if copy fails
            
    except Exception as e:
        logger.error(f"Error in create_safe_molecular_copy ({step_name}): {e}")
        return mol  # Return original if copy fails

# FAIR Metadata Functions
def generate_fair_metadata_for_results(poses, input_data):
    """Generate comprehensive FAIR metadata for pose prediction results"""
    if not FAIR_AVAILABLE:
        return None
    
    try:
        # Extract input information
        target_id = input_data.get('protein_pdb_id') or 'custom'
        input_type = 'pdb_id' if input_data.get('protein_pdb_id') else 'custom'
        input_smiles = input_data.get('input_smiles', '')
        
        # Calculate basic metrics
        poses_count = len(poses) if poses else 0
        best_scores = {}
        if poses:
            best_method, (best_mol, scores) = max(poses.items(), 
                                                key=lambda x: x[1][1].get('combo_score', x[1][1].get('combo', 0)))
            best_scores = {
                'shape_score': scores.get('shape_score', scores.get('shape', 0)),
                'color_score': scores.get('color_score', scores.get('color', 0)),
                'combo_score': scores.get('combo_score', scores.get('combo', 0))
            }
        
        # Get FAIR functions
        fair_available, fair_funcs = check_fair_availability()
        if not fair_available:
            return None
        
        if 'generate_quick_metadata' not in fair_funcs:
            return None
            
        generate_quick_metadata = fair_funcs['generate_quick_metadata']
        calculate_molecular_descriptors = fair_funcs['calculate_molecular_descriptors']
        
        # Generate quick metadata
        metadata = generate_quick_metadata(
            target_id=target_id,
            input_type=input_type,
            input_value=input_smiles,
            output_file="web_interface_results.sdf",
            poses_count=poses_count,
            scores=best_scores
        )
        
        # Add web interface specific metadata
        metadata['interface'] = {
            'type': 'web_interface',
            'platform': 'streamlit',
            'session_id': id(st.session_state),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add molecular descriptors if available
        if input_smiles and hasattr(st.session_state, 'query_mol') and st.session_state.query_mol:
            try:
                if input_smiles and 'calculate_molecular_descriptors' in locals():
                    descriptors = calculate_molecular_descriptors(input_smiles)
                    metadata['molecular_descriptors'] = descriptors
            except Exception as e:
                logger.warning(f"Could not calculate molecular descriptors: {e}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error generating FAIR metadata: {e}")
        return None

def render_fair_panel_trigger():
    """Render the subtle trigger button for the FAIR panel"""
    if st.session_state.get('poses') and FAIR_AVAILABLE:
        # Create a subtle trigger in the top-right area
        col1, col2, col3 = st.columns([20, 1, 1])
        with col3:
            if st.button("📊", 
                        help="View Scientific Data & Metadata", 
                        key="fair_trigger",
                        use_container_width=True):
                st.session_state.show_fair_panel = True
                # Generate metadata if not already done
                if not st.session_state.get('fair_metadata'):
                    input_data = {
                        'protein_pdb_id': st.session_state.get('protein_pdb_id'),
                        'input_smiles': st.session_state.get('input_smiles')
                    }
                    st.session_state.fair_metadata = generate_fair_metadata_for_results(
                        st.session_state.poses, input_data
                    )
                st.rerun()

def render_fair_sliding_panel():
    """Render the FAIR data in a sliding sidebar panel"""
    if st.session_state.get('show_fair_panel', False) and FAIR_AVAILABLE:
        with st.sidebar:
            # Panel Header
            st.markdown("### 🔬 Scientific Data & Metadata")
            col1, col2 = st.columns([4, 1])
            with col2:
                if st.button("✕", help="Close Panel", key="close_fair"):
                    st.session_state.show_fair_panel = False
                    st.rerun()
            
            # FAIR Compliance Badge
            if st.session_state.get('fair_metadata'):
                st.success("✅ FAIR Compliant Dataset")
            
            # Content Tabs
            panel_tab = st.selectbox(
                "Select Data Type:",
                ["📋 Metadata", "🧬 Properties", "🔄 Provenance", "📥 Export"],
                key="fair_panel_tab"
            )
            
            # Render selected content
            if panel_tab == "📋 Metadata":
                render_fair_metadata_tab()
            elif panel_tab == "🧬 Properties":
                render_molecular_properties_tab()
            elif panel_tab == "🔄 Provenance":
                render_provenance_tab()
            elif panel_tab == "📥 Export":
                render_enhanced_exports_tab()

def render_fair_metadata_tab():
    """Render the metadata tab content"""
    metadata = st.session_state.get('fair_metadata')
    if not metadata:
        st.warning("No metadata available")
        return
    
    st.markdown("#### Quick Summary")
    
    # Key metrics
    if 'scores' in metadata and metadata['scores']:
        scores = metadata['scores']
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best Score", f"{scores.get('combo_score', 0):.3f}")
        with col2:
            st.metric("Poses Generated", metadata.get('poses_count', 0))
    
    # Expandable metadata sections
    with st.expander("📋 Core Metadata"):
        if 'input' in metadata:
            st.json(metadata['input'])
    
    with st.expander("⚙️ Computational Environment"):
        if 'computational' in metadata:
            st.json(metadata['computational'])
    
    with st.expander("🔬 Scientific Context"):
        if 'scientific' in metadata:
            st.json(metadata['scientific'])

def render_molecular_properties_tab():
    """Render the molecular properties tab content"""
    metadata = st.session_state.get('fair_metadata')
    if not metadata or 'molecular_descriptors' not in metadata:
        st.warning("No molecular descriptors available")
        return
    
    descriptors = metadata['molecular_descriptors']
    
    st.markdown("#### Molecular Properties")
    
    # Basic properties
    if 'basic_properties' in descriptors:
        basic = descriptors['basic_properties']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Molecular Weight", f"{basic.get('molecular_weight', 0):.2f}")
        with col2:
            st.metric("LogP", f"{basic.get('logp', 0):.2f}")
        with col3:
            st.metric("TPSA", f"{basic.get('tpsa', 0):.2f}")
    
    # Drug-likeness
    if 'drug_likeness' in descriptors:
        drug_like = descriptors['drug_likeness']
        st.markdown("#### Drug-Likeness Assessment")
        
        lipinski_violations = drug_like.get('lipinski_violations', 0)
        if lipinski_violations == 0:
            st.success("✅ Passes Lipinski Rule of Five")
        else:
            st.warning(f"⚠️ {lipinski_violations} Lipinski violations")
    
    # Detailed properties
    with st.expander("📊 All Molecular Descriptors"):
        st.json(descriptors)

def render_provenance_tab():
    """Render the provenance tab content"""
    metadata = st.session_state.get('fair_metadata')
    if not metadata:
        st.warning("No provenance data available")
        return
    
    st.markdown("#### Workflow Provenance")
    
    # Timeline
    if 'computational' in metadata:
        comp = metadata['computational']
        st.markdown("**Execution Timeline:**")
        st.write(f"• Started: {comp.get('start_time', 'Unknown')}")
        if comp.get('end_time'):
            st.write(f"• Completed: {comp.get('end_time')}")
        if comp.get('execution_time'):
            st.write(f"• Duration: {comp.get('execution_time'):.2f} seconds")
    
    # Parameters
    if 'input' in metadata and 'parameters' in metadata['input']:
        with st.expander("⚙️ Input Parameters"):
            st.json(metadata['input']['parameters'])
    
    # System info
    if 'computational' in metadata:
        with st.expander("💻 System Information"):
            comp = metadata['computational']
            sys_info = {
                'platform': comp.get('platform'),
                'python_version': comp.get('python_version'),
                'pipeline_version': comp.get('pipeline_version'),
                'gpu_available': comp.get('gpu_available')
            }
            st.json(sys_info)

def render_enhanced_exports_tab():
    """Render the enhanced exports tab content"""
    st.markdown("#### 📥 Scientific Export Options")
    
    # Standard downloads with metadata
    st.markdown("**Results with Metadata:**")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📄 Download SDF + Metadata", use_container_width=True):
            create_fair_sdf_download()
    
    with col2:
        if st.button("📊 Download JSON Metadata", use_container_width=True):
            create_metadata_download()
    
    # Citation information
    st.markdown("**Citation & Attribution:**")
    if st.button("🔗 Copy Citation Information", use_container_width=True):
        create_citation_export()

def create_fair_sdf_download():
    """Create SDF download with embedded FAIR metadata"""
    if not st.session_state.get('poses'):
        st.error("No poses available for download")
        return
    
    # Create enhanced SDF with metadata
    sdf_data, file_name = create_best_poses_sdf(st.session_state.poses)
    
    # Add metadata to download
    metadata = st.session_state.get('fair_metadata')
    if metadata:
        # Create a bundle with both SDF and metadata
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add SDF file
            zip_file.writestr(file_name, sdf_data)
            # Add metadata file
            metadata_json = json.dumps(metadata, indent=2)
            zip_file.writestr(f"{file_name.replace('.sdf', '_metadata.json')}", metadata_json)
        
        zip_buffer.seek(0)
        st.download_button(
            "📦 Download Complete Package",
            data=zip_buffer.getvalue(),
            file_name=f"{file_name.replace('.sdf', '_with_metadata.zip')}",
            mime="application/zip"
        )
    else:
        st.download_button(
            "📄 Download SDF",
            data=sdf_data,
            file_name=file_name,
            mime="chemical/x-mdl-sdfile"
        )

def create_metadata_download():
    """Create standalone metadata download"""
    metadata = st.session_state.get('fair_metadata')
    if not metadata:
        st.error("No metadata available")
        return
    
    metadata_json = json.dumps(metadata, indent=2)
    st.download_button(
        "📊 Download Metadata",
        data=metadata_json,
        file_name="templ_prediction_metadata.json",
        mime="application/json"
    )

def create_citation_export():
    """Create citation information"""
    citation_text = """
TEMPL Pipeline Pose Prediction

Generated using TEMPL (Template-based Protein-Ligand Pose Prediction)
Web Interface: Streamlit Application
Timestamp: {timestamp}

Please cite:
[Add appropriate citation information here]

Data generated following FAIR principles:
- Findable: Unique identifiers and comprehensive metadata
- Accessible: Standard formats and open access
- Interoperable: JSON/SDF formats with standard descriptors  
- Reusable: Complete provenance and methodology documentation
    """.format(timestamp=datetime.now().isoformat())
    
    st.text_area("Citation Information", citation_text, height=300)
    st.info("Citation information copied to display. Copy text above for your records.")

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_performance_stats():
    """Get performance statistics for monitoring"""
    return {
        'embedding_cached': check_embedding_capabilities.cache_info() if hasattr(check_embedding_capabilities, 'cache_info') else None,
        'hardware_cached': get_hardware_info.cache_info() if hasattr(get_hardware_info, 'cache_info') else None,
        'molecule_validation_cached': validate_smiles_input.cache_info() if hasattr(validate_smiles_input, 'cache_info') else None,
        'image_generation_cached': generate_molecule_image.cache_info() if hasattr(generate_molecule_image, 'cache_info') else None,
    }

def health_check():
    """Health check endpoint for DigitalOcean deployment monitoring"""
    st.write("OK")
    st.stop()

def main():
    """Main application"""
    
    # Configure page only once
    try:
        st.set_page_config(
            page_title="TEMPL Pipeline",
            page_icon="🧪",
            layout="wide"
        )
    except st.StreamlitAPIException:
        # Already configured
        pass
    
    # Handle health check endpoint
    if st.query_params.get("health") == "check":
        health_check()
    
    # Handle DigitalOcean health check endpoint
    if st.query_params.get("healthz") is not None:
        health_check()
    
    # Initialize session state only once
    if 'app_initialized' not in st.session_state:
        init_start_time = time.time()
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.info("🔄 Initializing TEMPL Pipeline...")
            st.write("Loading molecular processing modules and hardware detection...")
        
        # Initialize session state
        initialize_session_state()
        
        # Initialize hardware detection (cached)
        get_hardware_info()
        
        # Initialize embedding capabilities (cached)
        check_embedding_capabilities()
        
        # Initialize FAIR metadata availability (cached)
        check_fair_availability()
        
        # Mark app as initialized
        st.session_state.app_initialized = True
        init_time = time.time() - init_start_time
        logger.info(f"Performance: App initialization completed in {init_time:.3f} seconds")
        loading_placeholder.empty()
    # No else clause - already initialized
    
    # Display optimization status badge
    if OPTIMIZATION_MODULES_AVAILABLE:
        st.success("🚀 Enhanced Performance Mode: Security & Memory Optimization Active")
    else:
        st.info("⚡ Standard Mode: Basic functionality available")
    
    # Performance optimization notice
    st.info("✅ Performance Optimizations: Redundant execution fixes applied, caching enabled for faster loading")
    
    # Header with branding and hardware status
    header_color = "#667eea" if EMBEDDING_FEATURES_AVAILABLE else "#FFA500"  # Blue if embeddings available, orange if not
    
    st.markdown(f"""
    <style>
    .header-glass {{
        background: rgba(30, 32, 48, 0.75);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.18);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
        margin-bottom: 1.5rem;
        animation: fadeIn 1.2s ease;
    }}
    .header-title {{
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        font-size: 2.8rem;
        letter-spacing: 0.04em;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #fff;
        text-shadow: 0 2px 12px #764ba2aa, 0 1px 0 #0003;
    }}
    .header-subtitle {{
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        font-size: 1.18rem;
        font-weight: 400;
        color: #e0e0e0;
        opacity: 0.92;
        margin: 0;
        letter-spacing: 0.02em;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(-16px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    <div class="header-glass">
        <div class="header-title">TEMPL Pipeline</div>
        <div class="header-subtitle">TEMplate-based Protein-Ligand Pose Prediction</div>
    </div>
    """, unsafe_allow_html=True)

    
    # Hardware and dependency status
    show_hardware_status()
    
    # Quick overview
    col_desc1, col_desc2 = st.columns(2, gap="large")
    
    with col_desc1:
        st.markdown("""
        <div class="info-card">
            <h4>What it does</h4>
            <ul>
                <li>Predicts 3D binding poses for small molecules</li> 
                <li>Uses template-guided conformer generation with MCS</li>
                <li>Provides shape, pharmacophore, and combined scoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_desc2:
        st.markdown("""
        <div class="info-card">
            <h4>How it works</h4>
            <ol>
                <li>Enter your molecule (SMILES or file)</li>
                <li>Provide protein target or SDF templates</li>
                <li>Get ranked poses with confidence scores</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    st.divider()
    
    # Enhanced Input section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.header("Input Configuration")
    st.markdown("Provide your molecule and protein target for pose prediction")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### Query Molecule")
        
        molecule_method = st.radio("Input method:", ["SMILES", "Upload File"], horizontal=True, key="mol_input")
        
        if molecule_method == "SMILES":
            # Add performance timing for SMILES input
            smiles_start = time.time()
            smiles = st.text_input(
                "SMILES String", 
                placeholder="Cn1c(=O)c2c(ncn2C)n(C)c1=O",
                help="Enter a valid SMILES string e.g. Cn1c(=O)c2c(ncn2C)n(C)c1=O"
            )
            # Only log performance on initial load, not on every rerun
            if not is_streamlit_rerun():
                logger.info(f"Performance: SMILES text_input creation took {time.time() - smiles_start:.3f} seconds")
            
            if smiles:
                valid, msg, mol_data = validate_smiles_input(smiles)
                if valid:
                    # Convert from cached binary format if needed
                    if mol_data is not None:
                        Chem, AllChem, Draw = get_rdkit_modules()
                        mol = Chem.Mol(mol_data)
                    else:
                        mol = None
                    st.session_state.query_mol = mol
                    st.session_state.input_smiles = smiles
                    st.markdown(f'<p class="status-success">{msg}</p>', unsafe_allow_html=True)
                    st.markdown('<div class="molecule-preview">', unsafe_allow_html=True)
                    display_molecule(mol, width=280, height=200)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="status-error">{msg}</p>', unsafe_allow_html=True)
        else:
            # Add performance timing for file uploader
            uploader_start = time.time()
            sdf_file = st.file_uploader("Upload SDF/MOL File", type=["sdf", "mol"], help="Upload a molecule file (≤5MB)")
            # Only log performance on initial load, not on every rerun
            if not is_streamlit_rerun():
                logger.info(f"Performance: SDF file_uploader creation took {time.time() - uploader_start:.3f} seconds")
            
            if sdf_file is not None:  # More explicit check
                if sdf_file.size > 5 * 1024 * 1024:  # 5MB limit
                    st.error("File too large (max 5MB)")
                else:
                    valid, msg, mol = validate_sdf_input(sdf_file)
                    if valid:
                        st.session_state.query_mol = mol
                        Chem, AllChem, Draw = get_rdkit_modules()
                        st.session_state.input_smiles = Chem.MolToSmiles(mol)
                        st.markdown(f'<p class="status-success">{msg}</p>', unsafe_allow_html=True)
                        st.markdown('<div class="molecule-preview">', unsafe_allow_html=True)
                        display_molecule(mol, width=280, height=200)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="status-error">{msg}</p>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Target Protein")
        
        protein_method = st.radio("Input method:", ["PDB ID", "Upload File", "Custom Templates"], horizontal=True, key="prot_input")
        
        if protein_method == "PDB ID":
            pdb_id = st.text_input("PDB ID", placeholder="1iky", help="4-character PDB identifier e.g. 1iky")
            if pdb_id:
                if len(pdb_id) == 4 and pdb_id.isalnum():
                    st.session_state.protein_pdb_id = pdb_id.lower()
                    st.session_state.protein_file_path = None
                    st.session_state.custom_templates = None
                    st.markdown(f'<p class="status-success">PDB ID: {pdb_id.upper()}</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="status-error">PDB ID must be 4 alphanumeric characters</p>', unsafe_allow_html=True)
        elif protein_method == "Upload File":
            # Add performance timing for PDB file uploader
            pdb_uploader_start = time.time()
            pdb_file = st.file_uploader("Upload PDB File", type=["pdb"], help="Upload protein structure file (≤5MB)")
            # Only log performance on initial load, not on every rerun
            if not is_streamlit_rerun():
                logger.info(f"Performance: PDB file_uploader creation took {time.time() - pdb_uploader_start:.3f} seconds")
            
            if pdb_file is not None:  # More explicit check
                if pdb_file.size > 5 * 1024 * 1024:  # 5MB limit
                    st.error("File too large (max 5MB)")
                else:
                    file_path = save_uploaded_file(pdb_file)
                    st.session_state.protein_file_path = file_path
                    st.session_state.protein_pdb_id = None
                    st.session_state.custom_templates = None
                    st.markdown('<p class="status-success">PDB file uploaded</p>', unsafe_allow_html=True)
        else:  # Custom Templates
            st.markdown("Upload SDF with template molecules for MCS-based pose generation")
            # Add performance timing for template file uploader
            template_uploader_start = time.time()
            template_file = st.file_uploader("Upload Template SDF", type=["sdf"], help="SDF file with template molecules (≤10MB)")
            # Only log performance on initial load, not on every rerun
            if not is_streamlit_rerun():
                logger.info(f"Performance: Template file_uploader creation took {time.time() - template_uploader_start:.3f} seconds")
            
            if template_file is not None:  # More explicit check
                if template_file.size > 10 * 1024 * 1024:  # 10MB limit
                    st.error("File too large (max 10MB)")
                else:
                    templates = load_templates_from_uploaded_sdf(template_file)
                    if templates:
                        st.session_state.custom_templates = templates
                        st.session_state.protein_pdb_id = None
                        st.session_state.protein_file_path = None
                        st.markdown(f'<p class="status-success">Loaded {len(templates)} template molecules</p>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p class="status-error">No valid molecules found in SDF</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Streamlined Execution
    protein_input = st.session_state.get('protein_pdb_id') or st.session_state.get('protein_file_path')
    custom_templates = st.session_state.get('custom_templates')
    molecule_input = st.session_state.get('input_smiles')
    ready = molecule_input and (protein_input or custom_templates)
    
    # Advanced Settings (Collapsible)
    with st.expander("Advanced Settings", expanded=False):
        st.markdown("### Pose Generation Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pose_mode = st.radio(
                "Pose Alignment Mode:",
                ["Aligned Poses", "Original Geometry"],
                index=0,  # Default to aligned poses for compatibility
                help="""
                • **Aligned Poses**: Molecules are aligned to templates
                • **Original Geometry**: Preserves conformer geometry, scores only
                """,
                key="pose_alignment_mode"
            )
            use_aligned_poses = (pose_mode == "Aligned Poses")
        
        with col2:
            if use_aligned_poses:
                st.info("Using alignment - poses will be positioned relative to templates")
            else:
                st.success("Preserving geometry - conformers maintain their original shape")
        
        # Template Filtering Options
        st.markdown("### Template Filtering Options")
        
        # Template filtering method selection - only show similarity if embeddings available
        filter_options = ["KNN (Top-K)"]
        filter_help = "Choose how to filter templates: by count (KNN)"
        
        if EMBEDDING_FEATURES_AVAILABLE:
            filter_options.append("Similarity Threshold")
            filter_help = "Choose how to filter templates: by count (KNN) or by embedding similarity"
        
        filter_method = st.radio(
            "Filtering Method:",
            filter_options,
            index=0,
            help=filter_help,
            key="template_filter_method",
            horizontal=True
        )
        
        # Show embedding requirement notice if similarity desired but not available
        if not EMBEDDING_FEATURES_AVAILABLE and len(filter_options) == 1:
            st.info("Embedding Similarity: Install embedding dependencies to enable protein embedding-based filtering")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            if filter_method == "KNN (Top-K)":
                max_templates = st.slider(
                    "Max Templates (KNN)",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                    help="Select top K most similar templates based on protein embedding similarity",
                    key="max_templates_slider"
                )
                similarity_threshold = None
                
            else:  # Similarity Threshold
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.70,
                    max_value=0.99,
                    value=0.80,
                    step=0.01,
                    format="%.2f",
                    help="Minimum cosine similarity for template selection (higher = fewer, better templates)",
                    key="similarity_threshold_slider"
                )
                max_templates = None
        
        with col2:
            if filter_method == "KNN (Top-K)":
                st.info("KNN Mode  \nSelects top K most similar templates")
            else:
                st.info("Threshold Mode  \nFilters by embedding similarity")
        
        # Widget values are automatically stored in session state via their keys
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if ready:
            st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
            if st.button("PREDICT POSES", type="primary", use_container_width=True):
                # Create placeholders for non-blocking progress
                progress_placeholder = st.empty()
                result_placeholder = st.empty()
                
                with progress_placeholder.container():
                    st.info("Initializing TEMPL Pipeline...")
                
                # Run pipeline directly
                poses = run_pipeline(
                    molecule_input, 
                    protein_input, 
                    custom_templates,
                    use_aligned_poses=use_aligned_poses,
                    max_templates=max_templates,
                    similarity_threshold=similarity_threshold
                )
                
                if poses:
                    st.session_state.poses = poses
                    progress_placeholder.empty()
                    result_placeholder.success("Pose prediction completed!")
                    st.rerun()
                else:
                    progress_placeholder.empty()
                    result_placeholder.error("Pipeline failed. Check error messages above.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True)
            st.info("Ready to start?  \nProvide molecule input and either protein target or custom templates above")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Enhanced Results Section
    if st.session_state.poses:
        st.markdown("---")
        
        # Add FAIR panel trigger (subtle, top-right)
        render_fair_panel_trigger()
        
        st.header("Prediction Results")
        
        # Detect and fix corruption
        fallback_applied = detect_and_fix_corruption()
        
        poses = st.session_state.poses
        
        # Find best pose by combo score
        best_method, (best_mol, best_scores) = max(poses.items(), 
                                                 key=lambda x: x[1][1].get('combo_score', x[1][1].get('combo', 0)))
        
        shape_score = best_scores.get('shape_score', best_scores.get('shape', 0))
        color_score = best_scores.get('color_score', best_scores.get('color', 0))
        combo_score = best_scores.get('combo_score', best_scores.get('combo', 0))
        
        # Best Pose Highlight
        st.markdown("### Best Predicted Pose")
        
        # Show alignment mode used
        if hasattr(st.session_state, 'poses') and isinstance(st.session_state.poses, dict):
            # Check if we have alignment info from the latest run
            alignment_used = getattr(st.session_state, 'alignment_used', None)
            if alignment_used is not None:
                if alignment_used:
                    st.info("Alignment Mode: Poses aligned to template structure")
                else:
                    st.success("Geometry Mode: Original conformer geometry preserved")
        
        # Score metrics
        col1a, col1b, col1c = st.columns(3)
        
        with col1a:
            st.metric("Shape Similarity", f"{shape_score:.3f}")
        with col1b:
            st.metric("Pharmacophore", f"{color_score:.3f}")
        with col1c:
            st.metric("Overall Score", f"{combo_score:.3f}")
        
        # Template Information (Collapsible)
        with st.expander("Template Details", expanded=False):
            if st.session_state.template_used and st.session_state.query_mol:
                col1, col2, col3 = st.columns(3)
                with col1:
                    display_molecule(st.session_state.query_mol, width=220, height=180, title="Query")
                with col2:
                    display_molecule(st.session_state.template_used, width=220, height=180, title="Template")
                with col3:
                    mcs_mol = safe_get_mcs_mol(st.session_state.mcs_info) if st.session_state.mcs_info else None
                    if mcs_mol:
                        display_molecule(mcs_mol, width=220, height=180, title="MCS")
                    else:
                        st.info("No significant common substructure found")
                
                if hasattr(st.session_state, 'template_info'):
                    info = st.session_state.template_info
                    template_name = info.get('name', 'unknown')
                    template_rank = info.get('index', 0) + 1
                    total_templates = info.get('total_templates', 1)
                    atoms_matched = info.get('atoms_matched', 0)
                    
                    # Build display text with PDB ID
                    display_parts = [f"Selected template: {template_name}"]
                    display_parts.append(f"ranked #{template_rank} of {total_templates}")
                    
                    if atoms_matched > 0:
                        display_parts.append(f"{atoms_matched} atoms matched")
                    
                    display_text = f"{display_parts[0]} ({', '.join(display_parts[1:])})"
                    st.info(display_text)
                    

        
        
        # Download section
        st.markdown("### Download Results")
        
        # Calculate counts for display
        best_poses_count = len(poses) if poses else 0
        all_poses_count = len(st.session_state.all_ranked_poses) if hasattr(st.session_state, 'all_ranked_poses') and st.session_state.all_ranked_poses else 0
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sdf_data, file_name = create_best_poses_sdf(poses)
            st.download_button(
                f"Best Poses ({best_poses_count})",
                data=sdf_data,
                file_name=file_name,
                mime="chemical/x-mdl-sdfile",
                help="Top scoring poses for each method (shape, color, combo)",
                use_container_width=True
            )
        
        with col2:
            if hasattr(st.session_state, 'all_ranked_poses') and st.session_state.all_ranked_poses:
                all_sdf_data, all_file_name = create_all_conformers_sdf()
                st.download_button(
                    f"All Conformers ({all_poses_count})",
                    data=all_sdf_data,
                    file_name=all_file_name,
                    mime="chemical/x-mdl-sdfile",
                    help="All generated conformers ranked by combo score",
                    use_container_width=True
                )
            else:
                st.button(
                    "All Conformers (N/A)",
                    disabled=True,
                    help="All ranked poses not available - try regenerating poses",
                    use_container_width=True
                )
        
    # Render FAIR sliding panel if activated
    render_fair_sliding_panel()

def simple_alignment_fallback(poses_dict):
    """Simple fallback processing when molecular corruption is detected"""
    try:
        Chem, AllChem, Draw = get_rdkit_modules()
        
        logger.info("=== APPLYING SIMPLE ALIGNMENT FALLBACK ===")
        fallback_poses = {}
        
        for method, (mol, scores) in poses_dict.items():
            if mol:
                # Validate the original molecule
                is_valid, msg = validate_molecular_connectivity(mol, f"fallback_{method}")
                
                if is_valid:
                    # Use the molecule as-is if it's valid
                    fallback_poses[method] = (mol, scores)
                    logger.info(f"Fallback: {method} molecule is valid, using directly")
                else:
                    # Try to reconstruct from SMILES
                    try:
                        smiles = Chem.MolToSmiles(mol)
                        reconstructed_mol = Chem.MolFromSmiles(smiles)
                        
                        if reconstructed_mol:
                            # Generate 3D coordinates
                            AllChem.EmbedMolecule(reconstructed_mol)
                            AllChem.UFFOptimizeMolecule(reconstructed_mol)
                            
                            fallback_poses[method] = (reconstructed_mol, scores)
                            logger.info(f"Fallback: {method} molecule reconstructed from SMILES")
                        else:
                            logger.error(f"Fallback: Could not reconstruct {method} molecule")
                    except Exception as e:
                        logger.error(f"Fallback: Failed to process {method} - {e}")
        
        return fallback_poses
        
    except Exception as e:
        logger.error(f"Simple alignment fallback failed: {e}")
        return poses_dict  # Return original if fallback fails

def detect_and_fix_corruption():
    """Detect molecular corruption and apply fixes"""
    if not hasattr(st.session_state, 'poses') or not st.session_state.poses:
        return False
        
    corruption_detected = False
    
    # Check for corruption indicators
    pipeline_issues = []
    sdf_issues = []
    
    if hasattr(st.session_state, 'pose_integrity_pipeline'):
        for method, info in st.session_state.pose_integrity_pipeline.items():
            if info and info.get('issues'):
                pipeline_issues.extend(info['issues'])
                corruption_detected = True
    
    if hasattr(st.session_state, 'sdf_integrity'):
        for method, integrity_info in st.session_state.sdf_integrity.items():
            before = integrity_info.get('before', {})
            after = integrity_info.get('after', {})
            if before and after:
                if before.get('smiles') != after.get('smiles'):
                    sdf_issues.append(f"{method}: SMILES corruption")
                    corruption_detected = True
    
    if corruption_detected:
        logger.warning("Molecular corruption detected, applying fallback processing")
        
        # Apply fallback
        original_poses = st.session_state.poses
        fallback_poses = simple_alignment_fallback(original_poses)
        
        # Update poses with fallback
        st.session_state.poses = fallback_poses
        st.session_state.fallback_applied = True
        
        return True
    
    return False



# Call main function only when run directly
if __name__ == "__main__":
    main()
