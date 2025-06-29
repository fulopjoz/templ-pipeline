"""
Export utility functions for TEMPL Pipeline UI

Contains SDF export and molecular data extraction functions.
"""

import io
import re
import logging
from typing import Dict, Tuple, Any, Optional

# Import streamlit for session state access
import streamlit as st

logger = logging.getLogger(__name__)


def create_best_poses_sdf(poses):
    """Create SDF for the best poses using the same helper used by the CLI.

    This now delegates to `TEMPLPipeline.save_results`, guaranteeing coordinate
    handling identical to the command-line workflow (heavy-atom coords are
    remapped back to the original conformer before writing).
    """
    try:
        from templ_pipeline.core.pipeline import TEMPLPipeline
        import tempfile
        import os
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
        
    except Exception as e:
        logger.error(f"Error creating best poses SDF: {e}")
        return "", "error.sdf"


def create_all_conformers_sdf():
    """Create SDF with all ranked conformers including scores"""
    if not hasattr(st.session_state, 'all_ranked_poses') or not st.session_state.all_ranked_poses:
        return "No ranked poses available", "error.sdf"
    
    try:
        from ..utils.molecular_utils import get_rdkit_modules
        
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
        
    except Exception as e:
        logger.error(f"Error creating all conformers SDF: {e}")
        return "", "error.sdf"


def extract_pdb_id_from_template(template_mol, index=0):
    """Extract PDB ID from template molecule properties with synthetic template ID support"""
    
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
