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
    """Save uploaded file to temp location with security enhancements and workspace integration"""
    try:
        # Try to use secure upload with workspace integration if available
        from ..core.secure_upload import SecureFileUploadHandler
        
        # Try to get workspace manager from active pipeline service
        workspace_manager = None
        try:
            import streamlit as st
            # Check if we have workspace integration in session state
            if hasattr(st, 'session_state') and hasattr(st.session_state, '_workspace_manager'):
                workspace_manager = st.session_state._workspace_manager
        except:
            pass
        
        handler = SecureFileUploadHandler(workspace_manager=workspace_manager)
        file_type = suffix.lstrip(".")
        success, message, secure_path = handler.validate_and_save(
            uploaded_file, file_type
        )

        if success:
            logger.info(f"Secure file upload: {message}")
            return secure_path
        else:
            logger.warning(f"Secure upload failed: {message}")

    except ImportError:
        logger.info("Secure upload handler not available, using fallback")
    except Exception as e:
        logger.warning(f"Secure upload failed, using fallback: {e}")

    # Memory-efficient fallback method
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        # Write file in chunks to avoid memory issues
        uploaded_file.seek(0)  # Reset file pointer
        chunk_size = 8192  # 8KB chunks
        while True:
            chunk = uploaded_file.read(chunk_size)
            if not chunk:
                break
            tmp.write(chunk)
        return tmp.name


def extract_pdb_id_from_file_robust(file_path):
    """Extract PDB ID from uploaded PDB file using robust core pipeline logic
    
    Uses the same multi-strategy approach as the core pipeline for maximum compatibility.
    """
    if not file_path:
        return None
        
    try:
        with open(file_path, "r") as f:
            for line in f:
                if line.startswith("HEADER"):
                    # Strategy 1: Standard PDB format - PDB ID at positions 62-66
                    if len(line) >= 66:
                        pdb_id = line[62:66].strip().lower()
                        if len(pdb_id) == 4 and pdb_id.isalnum():
                            logger.info(f"Extracted PDB ID '{pdb_id}' from standard header format")
                            return pdb_id
                    
                    # Strategy 2: Simple header format - "HEADER    PDB_ID" or "HEADER    PDB_ID_PROTEIN"
                    header_parts = line.strip().split()
                    if len(header_parts) >= 2:
                        potential_id = header_parts[1]
                        
                        # Remove common suffixes like "_PROTEIN"
                        if potential_id.endswith("_PROTEIN"):
                            potential_id = potential_id[:-8]
                        elif potential_id.endswith("_COMPLEX"):
                            potential_id = potential_id[:-8]
                        
                        # Validate as 4-character PDB ID
                        if len(potential_id) == 4 and potential_id.isalnum():
                            pdb_id = potential_id.lower()
                            logger.info(f"Extracted PDB ID '{pdb_id}' from simple header format")
                            return pdb_id
                            
                elif line.startswith("TITLE") or line.startswith("ATOM"):
                    # Stop searching after HEADER section
                    break
                    
        # Strategy 3: Filename fallback - extract from filename as last resort
        import os
        filename = os.path.basename(file_path)
        if filename:
            # Remove extension
            name_part = os.path.splitext(filename)[0]
            
            # Remove common suffixes
            if name_part.endswith("_protein"):
                name_part = name_part[:-8]
            elif name_part.endswith("_complex"):
                name_part = name_part[:-8]
            
            # Check if it looks like a PDB ID
            if len(name_part) == 4 and name_part.isalnum():
                pdb_id = name_part.lower()
                logger.info(f"Extracted PDB ID '{pdb_id}' from filename as fallback")
                return pdb_id
                
        logger.warning(f"No valid PDB ID found in file header or filename: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error extracting PDB ID from file {file_path}: {e}")
        return None


def extract_pdb_id_from_file(file_path):
    """Legacy function - redirects to robust implementation"""
    return extract_pdb_id_from_file_robust(file_path)


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
                name_field = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
            except Exception:
                name_field = ""

            pid_match = pid_pattern.search(name_field) if name_field else None
            pid = pid_match.group(0).upper() if pid_match else f"TPL{idx:03d}"

            # Store as properties recognised by extract_pdb_id_from_template
            mol.SetProp("template_pid", pid)
            mol.SetProp("template_name", name_field or pid)

            templates.append(mol)

        return templates
    except Exception as e:
        logger.error(f"Error reading template SDF file: {e}")
        return []


def validate_pdb_file_content(file_path):
    """Validate PDB file structure and content
    
    Returns:
        Tuple of (is_valid, message, extracted_info)
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        if not lines:
            return False, "File is empty", {}
            
        # Check for basic PDB structure
        has_header = any(line.startswith("HEADER") for line in lines[:10])
        has_atoms = any(line.startswith("ATOM") or line.startswith("HETATM") for line in lines)
        
        if not has_atoms:
            return False, "No ATOM or HETATM records found", {}
            
        # Count atoms
        atom_count = sum(1 for line in lines if line.startswith(("ATOM", "HETATM")))
        
        # Extract basic info
        extracted_info = {
            "has_header": has_header,
            "atom_count": atom_count,
            "total_lines": len(lines)
        }
        
        if atom_count < 10:
            return False, f"Too few atoms ({atom_count}) for valid protein structure", extracted_info
            
        return True, f"Valid PDB file with {atom_count} atoms", extracted_info
        
    except Exception as e:
        return False, f"Error reading PDB file: {str(e)}", {}


def integrate_uploaded_pdb_with_pipeline(file_path, session_manager):
    """Integrate uploaded PDB file with pipeline system
    
    Args:
        file_path: Path to uploaded PDB file
        session_manager: Session manager instance
        
    Returns:
        Tuple of (success, message, pdb_id, embedding_info)
    """
    try:
        # Validate file content
        is_valid, validation_msg, file_info = validate_pdb_file_content(file_path)
        if not is_valid:
            return False, validation_msg, None, None
            
        # Extract PDB ID using robust method
        extracted_pdb_id = extract_pdb_id_from_file_robust(file_path)
        
        # Check if we can create embeddings
        embedding_info = None
        if extracted_pdb_id:
            try:
                # Try to create or get embedding for this PDB
                embedding_info = create_embedding_for_uploaded_file(
                    file_path, extracted_pdb_id, session_manager
                )
                
                if embedding_info["success"]:
                    success_msg = f"Successfully processed PDB file. ID: {extracted_pdb_id.upper()}, {validation_msg}"
                    return True, success_msg, extracted_pdb_id, embedding_info
                else:
                    warning_msg = f"PDB ID extracted ({extracted_pdb_id.upper()}) but embedding generation failed. {validation_msg}"
                    return True, warning_msg, extracted_pdb_id, embedding_info
                    
            except Exception as e:
                logger.warning(f"Embedding creation failed for {extracted_pdb_id}: {e}")
                warning_msg = f"PDB ID extracted ({extracted_pdb_id.upper()}) but embedding creation failed. {validation_msg}"
                return True, warning_msg, extracted_pdb_id, None
        else:
            # No PDB ID but file is valid
            warning_msg = f"Valid PDB file but no PDB ID could be extracted. {validation_msg}. Pipeline will work but may not find database templates."
            return True, warning_msg, None, None
            
    except Exception as e:
        logger.error(f"Error integrating PDB file: {e}")
        return False, f"Error processing PDB file: {str(e)}", None, None


def create_embedding_for_uploaded_file(file_path, pdb_id, session_manager):
    """Create or retrieve embedding for uploaded PDB file
    
    Args:
        file_path: Path to PDB file
        pdb_id: Extracted PDB ID
        session_manager: Session manager instance
        
    Returns:
        Dictionary with embedding creation results
    """
    try:
        # Import pipeline components
        from templ_pipeline.core.embedding import EmbeddingManager, get_protein_embedding
        from templ_pipeline.core.pipeline import DEFAULT_DATA_DIR
        
        # Get or create embedding manager
        embedding_path = f"{DEFAULT_DATA_DIR}/embeddings/templ_protein_embeddings_v1.0.0.npz"
        embedding_manager = EmbeddingManager(embedding_path)
        
        # Check if embedding already exists
        if embedding_manager.has_embedding(pdb_id):
            logger.info(f"Found existing embedding for PDB ID '{pdb_id}'")
            embedding, chains_str = embedding_manager.get_embedding(pdb_id)
            return {
                "success": True,
                "source": "database",
                "embedding_shape": embedding.shape if embedding is not None else None,
                "chains": chains_str.split(',') if chains_str else [],
                "message": f"Retrieved existing embedding for {pdb_id.upper()}"
            }
        
        # Generate new embedding from uploaded file
        logger.info(f"Generating new embedding for uploaded file: {pdb_id}")
        embedding, chains_str = embedding_manager.get_embedding(pdb_id, pdb_file=file_path)
        
        if embedding is not None:
            return {
                "success": True,
                "source": "generated",
                "embedding_shape": embedding.shape,
                "chains": chains_str.split(',') if chains_str else [],
                "message": f"Generated new embedding for {pdb_id.upper()}"
            }
        else:
            return {
                "success": False,
                "source": None,
                "embedding_shape": None,
                "chains": [],
                "message": f"Failed to generate embedding for {pdb_id.upper()}"
            }
            
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        return {
            "success": False,
            "source": None,
            "embedding_shape": None,
            "chains": [],
            "message": f"Error during embedding creation: {str(e)}"
        }


def load_templates_from_sdf(
    template_pdbs, max_templates=100, exclude_target_smiles=None
):
    """Load template molecules from SDF file using standardized approach"""
    # Import the standardized template loading function
    try:
        from templ_pipeline.core.templates import load_template_molecules_standardized

        # Use the standardized template loading function
        templates, loading_stats = load_template_molecules_standardized(
            template_pdb_ids=template_pdbs[:max_templates],
            max_templates=max_templates,
            exclude_target_smiles=exclude_target_smiles,
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
