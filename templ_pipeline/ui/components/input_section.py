# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Input Section Component for TEMPL Pipeline

Handles molecule and protein input functionality.
"""

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from ..config.constants import MESSAGES, SESSION_KEYS
from ..config.settings import AppConfig
from ..core.session_manager import SessionManager
from ..utils.file_utils import (
    extract_pdb_id_from_file_robust,
    integrate_uploaded_pdb_with_pipeline,
    load_templates_from_uploaded_sdf,
    save_uploaded_file,
    validate_pdb_file_content,
)
from ..utils.molecular_utils import (
    get_rdkit_modules,
    validate_sdf_input,
    validate_smiles_input,
)

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
            key="mol_input_method",
        )

        if input_method == "SMILES":
            # Define callback function for setting example SMILES
            def set_example_smiles(smiles_value):
                st.session_state.smiles_input = smiles_value

            # Create columns for input, example button, and validation tick
            col_inp, col_btn, col_tick = st.columns([4, 1, 0.5])

            with col_inp:
                smiles = st.text_input(
                    "SMILES String",
                    placeholder="Enter SMILES string",
                    key="smiles_input",
                )

            with col_btn:
                # Add some spacing to align with the input field
                st.markdown("<br>", unsafe_allow_html=True)
                example_smiles = "c1ncccc1NC(=O)C1CCNCC1"
                st.button(
                    "Use Example",
                    key="use_example_smiles",
                    help="Fill with example SMILES",
                    on_click=set_example_smiles,
                    args=[example_smiles],
                )

            # Inline, subtle validation indicator (✅/❌) instead of large success box
            if smiles:
                try:
                    valid, msg, mol_data = validate_smiles_input(smiles)
                    if valid:
                        if mol_data is not None:
                            Chem, AllChem, Draw = get_rdkit_modules()
                            mol = Chem.Mol(mol_data)
                            if mol is not None:
                                mol.SetProp("original_smiles", smiles)
                                mol.SetProp("input_method", "smiles")
                        else:
                            mol = None
                        self.session.set(SESSION_KEYS["QUERY_MOL"], mol)
                        self.session.set(SESSION_KEYS["INPUT_SMILES"], smiles)
                        with col_tick:
                            st.markdown(
                                "<div style='font-size:20px;color:#22c55e;text-align:center;'>✅</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        self.session.set(SESSION_KEYS["INPUT_SMILES"], None)
                        with col_tick:
                            st.markdown(
                                "<div style='font-size:20px;color:#ef4444;text-align:center;'>❌</div>",
                                unsafe_allow_html=True,
                            )
                except Exception as e:
                    logger.error(f"Error validating SMILES: {e}")
                    self.session.set(SESSION_KEYS["INPUT_SMILES"], smiles)
                    with col_tick:
                        st.markdown(
                            "<div style='font-size:20px;color:#f59e0b;text-align:center;'>⚠️</div>",
                            unsafe_allow_html=True,
                        )
            else:
                # Clear SMILES when input is empty
                self.session.set(SESSION_KEYS["INPUT_SMILES"], None)
                self.session.set(SESSION_KEYS["QUERY_MOL"], None)
        else:
            uploaded_file = st.file_uploader(
                "Upload SDF/MOL File", type=["sdf", "mol"], key="mol_file_upload"
            )

            if uploaded_file:
                try:
                    valid, msg, mol = validate_sdf_input(uploaded_file)
                    if valid:
                        # Preserve original SMILES and input method for visualization
                        if mol is not None:
                            Chem, AllChem, Draw = get_rdkit_modules()
                            smiles = Chem.MolToSmiles(mol)
                            mol.SetProp("original_smiles", smiles)
                            mol.SetProp("input_method", "file_upload")
                            self.session.set(SESSION_KEYS["INPUT_SMILES"], smiles)
                        self.session.set(SESSION_KEYS["QUERY_MOL"], mol)
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
            key="prot_input_method",
        )

        if input_method == "PDB ID":
            # Clear any stale custom templates when switching to PDB ID mode
            try:
                self.session.set(SESSION_KEYS["CUSTOM_TEMPLATES"], None)
            except Exception:
                pass

            # Define callback function for setting example PDB ID
            def set_example_pdb(pdb_value):
                st.session_state.pdb_id_input = pdb_value

            # Create columns for input, example button, and validation tick
            col_inp, col_btn, col_tick = st.columns([4, 1, 0.5])

            with col_inp:
                pdb_id = st.text_input(
                    "PDB ID",
                    placeholder="Enter 4-character PDB ID",
                    key="pdb_id_input",
                )

            with col_btn:
                # Add some spacing to align with the input field
                st.markdown("<br>", unsafe_allow_html=True)
                example_pdb = "2etr"
                st.button(
                    "Use Example",
                    key="use_example_pdb",
                    help="Fill with example PDB ID",
                    on_click=set_example_pdb,
                    args=[example_pdb],
                )

            if pdb_id:
                if len(pdb_id) == 4 and pdb_id.isalnum():
                    self.session.set(SESSION_KEYS["PROTEIN_PDB_ID"], pdb_id.lower())
                    # Clear file path when using PDB ID
                    self.session.set(SESSION_KEYS["PROTEIN_FILE_PATH"], None)
                    self.session.set(SESSION_KEYS["CUSTOM_TEMPLATES"], None)
                    with col_tick:
                        st.markdown(
                            "<div style='font-size:20px;color:#22c55e;text-align:center;'>✅</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    self.session.set(SESSION_KEYS["PROTEIN_PDB_ID"], None)
                    with col_tick:
                        st.markdown(
                            "<div style='font-size:20px;color:#ef4444;text-align:center;'>❌</div>",
                            unsafe_allow_html=True,
                        )
            else:
                # Clear PDB ID when input is empty
                self.session.set(SESSION_KEYS["PROTEIN_PDB_ID"], None)

        elif input_method == "Upload File":
            # Clear any stale custom templates when switching to file upload mode
            try:
                self.session.set(SESSION_KEYS["CUSTOM_TEMPLATES"], None)
            except Exception:
                pass
            uploaded_file = st.file_uploader(
                "Upload PDB File", type=["pdb"], key="pdb_file_upload"
            )

            if uploaded_file:
                try:
                    # Save uploaded file
                    file_path = save_uploaded_file(uploaded_file)

                    # Integrate with pipeline system for robust processing
                    success, message, extracted_pdb_id, embedding_info = (
                        integrate_uploaded_pdb_with_pipeline(file_path, self.session)
                    )

                    if success:
                        # Store file path and PDB ID in session
                        self.session.set(SESSION_KEYS["PROTEIN_FILE_PATH"], file_path)

                        if extracted_pdb_id:
                            # Store extracted PDB ID for database integration
                            self.session.set(
                                SESSION_KEYS["PROTEIN_PDB_ID"], extracted_pdb_id
                            )
                            # Store embedding info for pipeline use
                            if embedding_info:
                                self.session.set("pdb_embedding_info", embedding_info)
                        else:
                            # Clear PDB ID if not found
                            self.session.set(SESSION_KEYS["PROTEIN_PDB_ID"], None)

                        # Clear custom templates when using file upload
                        self.session.set(SESSION_KEYS["CUSTOM_TEMPLATES"], None)

                        # Show appropriate success/warning message
                        if "Successfully processed" in message:
                            st.success(message)
                            if embedding_info and embedding_info.get("success"):
                                st.info(
                                    f"✓ Vector embedding ready: {embedding_info['message']}"
                                )
                        else:
                            st.warning(message)
                            if extracted_pdb_id:
                                st.info(
                                    "Note: The pipeline will attempt to generate embeddings during processing."
                                )
                    else:
                        st.error(f"Failed to process PDB file: {message}")
                        # Clear session data on failure
                        self.session.set(SESSION_KEYS["PROTEIN_FILE_PATH"], None)
                        self.session.set(SESSION_KEYS["PROTEIN_PDB_ID"], None)

                except Exception as e:
                    logger.error(f"Error processing PDB file: {e}")
                    st.error(f"Error processing file: {str(e)}")
                    # Clear session data on error
                    self.session.set(SESSION_KEYS["PROTEIN_FILE_PATH"], None)
                    self.session.set(SESSION_KEYS["PROTEIN_PDB_ID"], None)

        else:  # Custom Templates
            st.markdown(
                "Upload SDF with template molecules for MCS-based pose generation"
            )
            template_file = st.file_uploader(
                "Upload Template SDF", type=["sdf"], key="template_file_upload"
            )

            if template_file:
                try:
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
