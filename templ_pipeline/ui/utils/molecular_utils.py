"""
Molecular Utilities for TEMPL Pipeline

Essential molecular processing functions extracted from app.py for app_v2.py compatibility.
"""

import streamlit as st
import logging
import tempfile
import io
import re
from pathlib import Path
from typing import Tuple, Optional, Any

logger = logging.getLogger(__name__)

# Lazy imports to improve startup performance
_rdkit_modules = None

def get_rdkit_modules():
    """Lazy load RDKit modules to improve startup performance"""
    global _rdkit_modules
    if _rdkit_modules is None:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, Draw
        RDLogger.DisableLog('rdApp.*')
        _rdkit_modules = (Chem, AllChem, Draw)
    return _rdkit_modules

def validate_smiles_input(smiles: str) -> Tuple[bool, str, Optional[bytes]]:
    """Validate SMILES input with detailed feedback"""
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
        return False, f"Error parsing SMILES: {str(e)}", None

def validate_sdf_input(sdf_file) -> Tuple[bool, str, Optional[Any]]:
    """Validate SDF file input"""
    try:
        # Create cache key from file content hash
        sdf_data = sdf_file.read()
        file_hash = hash(sdf_data)
        cache_key = f"sdf_{file_hash}_{sdf_file.name}"
        
        # Check cache first
        if hasattr(st.session_state, 'file_cache') and cache_key in st.session_state.file_cache:
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
        if not hasattr(st.session_state, 'file_cache'):
            st.session_state.file_cache = {}
        st.session_state.file_cache[cache_key] = {
            'valid': result[0],
            'message': result[1], 
            'mol': result[2]
        }
        
        return result
        
    except Exception as e:
        return False, f"Error reading file: {str(e)}", None

def save_uploaded_file(uploaded_file, suffix=".pdb") -> Optional[str]:
    """Save uploaded file to temp location"""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        return None

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

def display_molecule(mol, width=400, height=300, title="", highlight_atoms=None):
    """Display molecule as 2D image with optional atom highlighting"""
    if mol is None:
        return
    
    try:
        # Convert mol to binary for processing
        Chem, AllChem, Draw = get_rdkit_modules()
        
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
        
        # Create safe copy for processing
        mol_copy = Chem.RemoveHs(mol_work)
        
        # Generate coordinates if needed
        if mol_copy.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol_copy)
        
        # Ensure highlight_atoms is hashable for caching (convert list to tuple)
        if highlight_atoms is not None:
            highlight_atoms = tuple(highlight_atoms)
        
        # Generate image
        if highlight_atoms:
            img = Draw.MolToImage(mol_copy, size=(width, height), highlightAtoms=highlight_atoms)
        else:
            img = Draw.MolToImage(mol_copy, size=(width, height))
        
        if img:
            if title:
                st.write(f"**{title}**")
            st.image(img)
            
    except Exception as e:
        st.error(f"Error displaying molecule: {e}")

def create_best_poses_sdf(poses):
    """Create SDF for the best poses using the same helper used by the CLI"""
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