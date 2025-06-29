"""
Input Section Component for TEMPL Pipeline

Handles molecule and protein input functionality.
"""

import streamlit as st
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import tempfile

from ...config.settings import AppConfig
from ...config.constants import SESSION_KEYS, MESSAGES
from ...core.session_manager import SessionManager

logger = logging.getLogger(__name__)


class InputSection:
    """Component for handling molecular and protein inputs"""
    
    def __init__(self, config: AppConfig, session: SessionManager):
        """Initialize input section
        
        Args:
            config: Application configuration
            session: Session manager
        """
        self.config = config
        self.session = session
    
    def render(self):
        """Render the input section"""
        st.markdown("### Input Configuration")
        st.markdown("Provide your molecule and protein target for pose prediction")
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            self._render_molecule_input()
        
        with col2:
            self._render_protein_input()
    
    def _render_molecule_input(self):
        """Render molecule input section"""
        st.markdown("#### Query Molecule")
        
        input_method = st.radio(
            "Input method:",
            ["SMILES", "Upload File"],
            horizontal=True,
            key="mol_input_method"
        )
        
        if input_method == "SMILES":
            smiles = st.text_input(
                "SMILES String",
                placeholder="Enter SMILES (e.g., C1CC(=O)N(C1)CC(=O)N )",
                key="smiles_input"
            )
            
            if smiles:
                # Import validation functions
                try:
                    from ...app import validate_smiles_input, get_rdkit_modules
                    valid, msg, mol_data = validate_smiles_input(smiles)
                    if valid:
                        # Convert from cached binary format if needed
                        if mol_data is not None:
                            Chem, AllChem, Draw = get_rdkit_modules()
                            mol = Chem.Mol(mol_data)
                        else:
                            mol = None
                        self.session.set(SESSION_KEYS["QUERY_MOL"], mol)
                        self.session.set(SESSION_KEYS["INPUT_SMILES"], smiles)
                        st.success(f"SMILES entered: {smiles}")
                    else:
                        st.error(msg)
                except Exception as e:
                    logger.error(f"Error validating SMILES: {e}")
                    # Fallback - just store the SMILES
                    self.session.set(SESSION_KEYS["INPUT_SMILES"], smiles)
                    st.success(f"SMILES entered: {smiles}")
        else:
            uploaded_file = st.file_uploader(
                "Upload SDF/MOL File",
                type=["sdf", "mol"],
                key="mol_file_upload"
            )
            
            if uploaded_file:
                try:
                    from ...app import validate_sdf_input, get_rdkit_modules
                    valid, msg, mol = validate_sdf_input(uploaded_file)
                    if valid:
                        self.session.set(SESSION_KEYS["QUERY_MOL"], mol)
                        Chem, AllChem, Draw = get_rdkit_modules()
                        self.session.set(SESSION_KEYS["INPUT_SMILES"], Chem.MolToSmiles(mol))
                        st.success(msg)
                    else:
                        st.error(msg)
                except Exception as e:
                    logger.error(f"Error processing SDF file: {e}")
                    st.error(f"Error processing file: {str(e)}")
    
    def _render_protein_input(self):
        """Render protein input section"""
        st.markdown("#### Target Protein")
        
        input_method = st.radio(
            "Input method:",
            ["PDB ID", "Upload File", "Custom Templates"],
            horizontal=True,
            key="prot_input_method"
        )
        
        if input_method == "PDB ID":
            pdb_id = st.text_input(
                "PDB ID",
                placeholder="Enter 4-character PDB ID (e.g., 1iky)",
                key="pdb_id_input"
            )
            
            if pdb_id:
                if len(pdb_id) == 4 and pdb_id.isalnum():
                    self.session.set(SESSION_KEYS["PROTEIN_PDB_ID"], pdb_id.lower())
                    # Clear file path when using PDB ID
                    self.session.set(SESSION_KEYS["PROTEIN_FILE_PATH"], None)
                    self.session.set(SESSION_KEYS["CUSTOM_TEMPLATES"], None)
                    st.success(f"PDB ID: {pdb_id.upper()}")
                else:
                    st.error("PDB ID must be 4 alphanumeric characters")
                    
        elif input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload PDB File",
                type=["pdb"],
                key="pdb_file_upload"
            )
            
            if uploaded_file:
                try:
                    # Save uploaded file
                    from ...app import save_uploaded_file
                    file_path = save_uploaded_file(uploaded_file)
                    
                    # Store file path in session
                    self.session.set(SESSION_KEYS["PROTEIN_FILE_PATH"], file_path)
                    # Clear PDB ID when using file upload
                    self.session.set(SESSION_KEYS["PROTEIN_PDB_ID"], None)
                    self.session.set(SESSION_KEYS["CUSTOM_TEMPLATES"], None)
                    
                    st.success(f"PDB file uploaded: {uploaded_file.name}")
                    
                    # Optional: Try to extract PDB ID from file for display purposes only
                    try:
                        from ...app import extract_pdb_id_from_file
                        extracted_pdb_id = extract_pdb_id_from_file(file_path)
                        if extracted_pdb_id:
                            st.info(f"Detected PDB ID from file: {extracted_pdb_id.upper()}")
                    except:
                        pass
                        
                except Exception as e:
                    logger.error(f"Error processing PDB file: {e}")
                    st.error(f"Error processing file: {str(e)}")
                
        else:  # Custom Templates
            st.markdown("Upload SDF with template molecules for MCS-based pose generation")
            template_file = st.file_uploader(
                "Upload Template SDF",
                type=["sdf"],
                key="template_file_upload"
            )
            
            if template_file:
                try:
                    from ...app import load_templates_from_uploaded_sdf
                    templates = load_templates_from_uploaded_sdf(template_file)
                    if templates:
                        self.session.set(SESSION_KEYS["CUSTOM_TEMPLATES"], templates)
                        # Clear protein inputs when using custom templates
                        self.session.set(SESSION_KEYS["PROTEIN_PDB_ID"], None)
                        self.session.set(SESSION_KEYS["PROTEIN_FILE_PATH"], None)
                        st.success(f"Loaded {len(templates)} template molecules")
                    else:
                        st.error("No valid molecules found in SDF")
                except Exception as e:
                    logger.error(f"Error processing template file: {e}")
                    st.error(f"Error processing file: {str(e)}")
