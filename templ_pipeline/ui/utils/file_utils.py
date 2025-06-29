"""
File utility functions for TEMPL Pipeline UI

Contains file upload, processing, and template loading functions.
"""

import tempfile
import logging
import re
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def save_uploaded_file(uploaded_file, suffix=".pdb"):
    """Save uploaded file to temp location with security enhancements"""
    try:
        # Try to use secure upload if available
        from ..core.secure_upload import SecureFileUploadHandler
        
        handler = SecureFileUploadHandler()
        file_type = suffix.lstrip('.')
        success, message, secure_path = handler.validate_and_save(uploaded_file, file_type)
        
        if success:
            logger.info(f"Secure file upload: {message}")
            return secure_path
        else:
            logger.warning(f"Secure upload failed: {message}")
            
    except ImportError:
        logger.info("Secure upload handler not available, using fallback")
    except Exception as e:
        logger.warning(f"Secure upload failed, using fallback: {e}")
    
    # Original fallback method
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


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


def load_templates_from_uploaded_sdf(uploaded_file):
    """Load template molecules from uploaded SDF file"""
    try:
        from .molecular_utils import get_rdkit_modules
        
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
        logger.error(f"Error reading template SDF file: {e}")
        return []


def load_templates_from_sdf(template_pdbs, max_templates=100, exclude_target_smiles=None):
    """Load template molecules from SDF file using standardized approach"""
    # Import the standardized template loading function
    try:
        from templ_pipeline.core.templates import load_template_molecules_standardized
        
        # Use the standardized template loading function
        templates, loading_stats = load_template_molecules_standardized(
            template_pdb_ids=template_pdbs[:max_templates],
            max_templates=max_templates,
            exclude_target_smiles=exclude_target_smiles
        )
        
        if "error" in loading_stats:
            logger.error(f"Template loading error: {loading_stats['error']}")
            return []
        
        if loading_stats.get("missing_pdbs"):
            missing_count = len(loading_stats["missing_pdbs"])
            logger.warning(f"Could not find {missing_count} templates in the database")
        
        return templates
        
    except Exception as e:
        logger.error(f"Error loading templates: {e}")
        return []
