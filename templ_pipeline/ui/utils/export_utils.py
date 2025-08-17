# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
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
        if (
            hasattr(st.session_state, "template_info")
            and st.session_state.template_info
        ):
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
    import streamlit as st
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Try to get all_ranked_poses from session using the session manager
    try:
        from ..core.session_manager import get_session_manager
        session = get_session_manager()
        # Prefer fresh in-session data
        all_ranked_poses = st.session_state.get("all_ranked_poses")
        if not all_ranked_poses:
            all_ranked_poses = session.get("all_ranked_poses")
        logger.info(f"Retrieved all_ranked_poses from session manager: {type(all_ranked_poses)}")
    except Exception as e:
        logger.warning(f"Failed to get session manager, using direct access: {e}")
        # Fallback to direct session state access
        all_ranked_poses = st.session_state.get("all_ranked_poses")
        logger.info(f"Retrieved all_ranked_poses from session state: {type(all_ranked_poses)}")
    
    if not all_ranked_poses:
        logger.warning("No all_ranked_poses data available")
        return "No ranked poses available", "error.sdf"
    
    logger.info(f"Processing {len(all_ranked_poses)} ranked poses")
    
    # DEBUGGING: Log detailed information about the data structure
    logger.info(f"DEBUG: all_ranked_poses type: {type(all_ranked_poses)}")
    logger.info(f"DEBUG: all_ranked_poses length: {len(all_ranked_poses)}")
    if len(all_ranked_poses) > 0:
        logger.info(f"DEBUG: First pose structure: {type(all_ranked_poses[0])}")
        if hasattr(all_ranked_poses[0], '__len__'):
            logger.info(f"DEBUG: First pose length: {len(all_ranked_poses[0])}")
        if len(all_ranked_poses) > 10:
            logger.info(f"DEBUG: 11th pose structure: {type(all_ranked_poses[10])}")
    
    # DEBUGGING: Also check if there are more poses in the session under different keys
    all_session_keys = list(st.session_state.keys()) if hasattr(st, 'session_state') else []
    pose_related_keys = [key for key in all_session_keys if 'pose' in key.lower()]
    logger.info(f"DEBUG: All pose-related session keys: {pose_related_keys}")
    
    for key in pose_related_keys:
        value = st.session_state.get(key)
        if value is not None:
            logger.info(f"DEBUG: Session key '{key}': type={type(value)}, length={len(value) if hasattr(value, '__len__') else 'N/A'}")

    try:
        from ..utils.molecular_utils import get_rdkit_modules

        Chem, AllChem, Draw = get_rdkit_modules()
        sdf_buffer = io.StringIO()
        writer = Chem.SDWriter(sdf_buffer)
        
        valid_poses_count = 0

        # Handle the correct data structure from pipeline: (conf_id, scores, mol)
        for rank, pose_data in enumerate(all_ranked_poses, 1):
            try:
                logger.debug(f"Processing pose {rank}: {type(pose_data)}, length: {len(pose_data) if hasattr(pose_data, '__len__') else 'N/A'}")
                
                # Unpack the tuple correctly: (conf_id, scores, mol)
                if len(pose_data) == 3:
                    conf_id, scores, mol = pose_data
                    logger.debug(f"Pose {rank}: conf_id={conf_id}, scores={type(scores)}, mol={type(mol)}")
                else:
                    # Handle legacy format: (pose, scores, original_cid)
                    mol, scores, conf_id = pose_data
                    logger.debug(f"Pose {rank} (legacy): mol={type(mol)}, scores={type(scores)}, conf_id={conf_id}")
                
                # Skip invalid poses
                if mol is None or not hasattr(mol, 'ToBinary'):
                    logger.warning(f"Skipping invalid pose at rank {rank}: mol={type(mol)}")
                    continue
                    
                # Create a safe copy of the molecule
                mol_copy = Chem.Mol(mol)
                # Ensure the copy carries current query identifier for clarity
                try:
                    input_smiles = st.session_state.get("input_smiles")
                    if input_smiles:
                        mol_copy.SetProp("Query_SMILES", input_smiles)
                except Exception:
                    pass
                valid_poses_count += 1
                
            except Exception as e:
                logger.warning(f"Skipping pose {rank} due to unpacking error: {e}")
                continue

            # Set properties
            mol_copy.SetProp("Rank", str(rank))
            mol_copy.SetProp("Shape_Score", f"{scores.get('shape', 0.0):.3f}")
            mol_copy.SetProp("Color_Score", f"{scores.get('color', 0.0):.3f}")
            mol_copy.SetProp("Combo_Score", f"{scores.get('combo', 0.0):.3f}")
            mol_copy.SetProp("Original_Conformer_ID", str(conf_id))

            # Add template info if available
            try:
                template_info = session.get("template_info") if 'session' in locals() else st.session_state.get("template_info")
                if template_info:
                    template_name = template_info.get("name", "unknown")
                    mol_copy.SetProp("Template_Used", template_name)
                    if "mcs_smarts" in template_info:
                        mol_copy.SetProp("MCS_SMARTS", template_info["mcs_smarts"])

                    # Add additional template metadata
                    mol_copy.SetProp("Template_Index", str(template_info.get("index", 0)))
                    mol_copy.SetProp("Total_Templates", str(template_info.get("total_templates", 1)))
                    mol_copy.SetProp("Atoms_Matched", str(template_info.get("atoms_matched", 0)))
            except Exception as e:
                logger.warning(f"Could not add template info to pose {rank}: {e}")

            writer.write(mol_copy)

        writer.close()
        sdf_content = sdf_buffer.getvalue()
        
        logger.info(f"Generated SDF with {valid_poses_count} valid poses")
        logger.info(f"DEBUG: SDF content length: {len(sdf_content)} characters")
        
        if not sdf_content.strip():
            logger.error("No valid poses to export - SDF content is empty")
            return "No valid poses to export", "error.sdf"
            
        return sdf_content, "templ_all_conformers_ranked.sdf"

    except Exception as e:
        logger.error(f"Error creating all conformers SDF: {e}", exc_info=True)
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
        "template_pid",  # From true_mcs_pipeline
        "Template_PDB",  # From scoring module
        "template_pdb",  # Common variant
        "pdb_id",
        "PDB_ID",  # Standard names
        "pdb",
        "PDB",  # Short forms
        "OriginalPDB",  # From similarity pipeline
        "TemplatePDB",  # Variant form
        # Secondary sources
        "_Name",  # RDKit default
        "template_name",  # Template-specific
        "name",
        "Name",  # Generic names
        "title",
        "Title",  # Title fields
        "source",
        "Source",  # Source fields
        "id",
        "ID",  # Generic ID fields
    ]

    # Enhanced PDB ID validation patterns including synthetic templates
    pdb_id_patterns = [
        r"\b([0-9][A-Za-z0-9]{3})\b",  # Standard: digit + 3 alphanumeric
        r"^([0-9][A-Za-z0-9]{3})$",  # Exact match
        r"^(TPL\d{3})$",  # Synthetic template IDs: TPL000, TPL001, etc.
        r"\b(TPL\d{3})\b",  # Synthetic template IDs in text
        r"^([0-9][A-Za-z0-9]{3})_",  # With underscore suffix
        r"_([0-9][A-Za-z0-9]{3})_",  # Between underscores
        r"([0-9][A-Za-z0-9]{3})\.pdb",  # With .pdb extension
        r"([0-9][A-Za-z0-9]{3})\.",  # With any extension
        r"pdb[_\-:]([0-9][A-Za-z0-9]{3})",  # pdb:1abc format
        r"([0-9][A-Za-z0-9]{3})[_\-]",  # With dash/underscore
    ]

    def validate_pdb_id(pdb_candidate):
        """Validate that a candidate string is a proper PDB ID or synthetic template ID"""
        if not pdb_candidate:
            return False

        # Check for synthetic template IDs (TPL000, TPL001, etc.)
        if re.match(r"^TPL\d{3}$", pdb_candidate):
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
            matches = re.findall(r"[0-9][A-Za-z0-9]{3}", value)
            for match in matches:
                if validate_pdb_id(match):
                    return match.upper()

            # Look for synthetic template IDs
            matches = re.findall(r"TPL\d{3}", value, re.IGNORECASE)
            for match in matches:
                return match.upper()

    # Final fallback - return template_name if available, otherwise generate synthetic ID
    if template_mol.HasProp("template_name"):
        fallback_name = template_mol.GetProp("template_name").strip()
        if fallback_name:
            return fallback_name.upper()

    return f"TPL{index:03d}"


def extract_best_poses_from_ranked(all_ranked_poses):
    """Extract best poses for each scoring method from ranked results"""
    if not all_ranked_poses:
        return {}

    best = {
        "shape": (None, {"shape": -1.0, "color": -1.0, "combo": -1.0}),
        "color": (None, {"shape": -1.0, "color": -1.0, "combo": -1.0}),
        "combo": (None, {"shape": -1.0, "color": -1.0, "combo": -1.0}),
    }

    for pose, scores, _ in all_ranked_poses:
        for metric in ["shape", "color", "combo"]:
            if scores[metric] > best[metric][1][metric]:
                best[metric] = (pose, scores)

    return best
