"""
Results Section Component for TEMPL Pipeline

Displays pose prediction results.
"""

import streamlit as st
import logging
from typing import Optional, Dict, Any

from ..config.settings import AppConfig
from ..config.constants import (
    SESSION_KEYS,
    MESSAGES,
    SCORE_EXCELLENT,
    SCORE_GOOD,
    SCORE_FAIR,
)
from ..core.session_manager import SessionManager
from ..utils.export_utils import (
    create_best_poses_sdf,
    create_all_conformers_sdf,
    extract_pdb_id_from_template,
)
from ..utils.visualization_utils import display_molecule, safe_get_mcs_mol

logger = logging.getLogger(__name__)


class ResultsSection:
    """Component for displaying prediction results"""

    def __init__(self, config: AppConfig, session: SessionManager):
        """Initialize results section

        Args:
            config: Application configuration
            session: Session manager
        """
        self.config = config
        self.session = session

    def render(self):
        """Render the results section"""
        poses = self.session.get(SESSION_KEYS["POSES"])

        if not poses:
            st.info("No results to display. Run a prediction first.")
            return

        st.header("Prediction Results")

        # Template Information Badge (NEW)
        self._render_template_badge()

        # Find best pose
        best_method, best_data = self._find_best_pose(poses)

        if best_method and best_data:
            mol, scores = best_data
            self._render_best_pose(best_method, scores)
            self._render_score_details(scores)

            # Template Details Section (NEW)
            with st.expander("Template Details", expanded=False):
                self._render_template_comparison()

            # Downloads with actual functionality
            self._render_download_section()
        else:
            st.error("No valid poses found in results")

        # FAIR Metadata Trigger (NEW)
        if (
            self.config.features.get("fair_metadata")
            or self.config.ui_settings.get("enable_fair_metadata")
        ) and poses:
            col1, col2 = st.columns([20, 1])
            with col2:
                if st.button(
                    "Metadata",
                    help="View Scientific Data & Metadata",
                    key="fair_trigger",
                ):
                    st.session_state.show_fair_panel = True
                    st.rerun()

    def _render_template_badge(self):
        """Display template information prominently"""
        template_info = self.session.get("template_info")
        if template_info:
            template_pdb = template_info.get("name", "Unknown")
            template_rank = template_info.get("index", 0) + 1
            total_templates = template_info.get("total_templates", 1)
            atoms_matched = template_info.get("atoms_matched", 0)

            # Create informative badge
            badge_text = f"**Poses generated using template: {template_pdb}**"
            if total_templates > 1:
                badge_text += f" (ranked #{template_rank} of {total_templates})"
            if atoms_matched > 0:
                badge_text += f" | {atoms_matched} atoms matched"

            st.info(badge_text)

    def _find_best_pose(self, poses: Dict) -> tuple:
        """Find the best pose by combo score

        Args:
            poses: Dictionary of poses

        Returns:
            Tuple of (method_name, (mol, scores))
        """
        if not poses:
            return None, None

        best_method = None
        best_score = -1
        best_data = None

        for method, (mol, scores) in poses.items():
            combo_score = scores.get("combo_score", scores.get("combo", 0))
            if combo_score > best_score:
                best_score = combo_score
                best_method = method
                best_data = (mol, scores)

        return best_method, best_data

    def _render_best_pose(self, method: str, scores: Dict):
        """Render best pose information

        Args:
            method: Method name
            scores: Score dictionary
        """
        st.markdown("### Best Predicted Pose")
        st.info(f"Best method: {method}")

        # Score metrics
        col1, col2, col3 = st.columns(3)

        shape_score = scores.get("shape_score", scores.get("shape", 0))
        color_score = scores.get("color_score", scores.get("color", 0))
        combo_score = scores.get("combo_score", scores.get("combo", 0))

        with col1:
            st.metric("ShapeTanimoto", f"{shape_score:.3f}")
        with col2:
            st.metric("ColorTanimoto", f"{color_score:.3f}")
        with col3:
            st.metric("TanimotoCombo (Normalized)", f"{combo_score:.3f}")

    def _render_score_details(self, scores: Dict):
        """Render detailed score interpretation with scientific explanations

        Args:
            scores: Score dictionary
        """
        combo_score = scores.get("combo_score", scores.get("combo", 0))

        # Determine quality level using updated thresholds
        if combo_score >= SCORE_EXCELLENT:
            quality = "Excellent - High confidence pose"
            color = "green"
            explanation = (
                "Top 10% of meaningful results. Highly reliable pose prediction."
            )
        elif combo_score >= SCORE_GOOD:
            quality = "Good - Reliable pose prediction"
            color = "blue"
            explanation = "Top 25% of results. Reliable pose with good shape and pharmacophore alignment."
        elif combo_score >= SCORE_FAIR:
            quality = "Fair - Moderate confidence"
            color = "orange"
            explanation = "Acceptable quality pose. Consider for further evaluation or optimization."
        else:
            quality = "Poor - Low confidence, consider alternatives"
            color = "red"
            explanation = "Below acceptance threshold. May require different templates or approaches."

        # Display quality assessment with help
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Quality Assessment:** :{color}[{quality}]")
        with col2:
            if st.button(
                "Help", key="score_help", help="Learn about scoring methodology"
            ):
                st.info(
                    f"""
                    **TEMPL Normalized TanimotoCombo Scoring:**
                    
                    {explanation}
                    
                    **Methodology (PMC9059856 Implementation):**
                    • **ShapeTanimoto**: 3D molecular shape overlap using Gaussian volume comparison
                    • **ColorTanimoto**: Chemical feature alignment (H-bond donors/acceptors, hydrophobic regions)
                    • **TEMPL Combo Score**: Normalized TanimotoCombo = (ShapeTanimoto + ColorTanimoto) / 2
                    
                    **Scale & Threshold Comparison:**
                    • **PMC Article**: TanimotoCombo range 0-2, cutoff >1.2 (equivalent to >0.6 normalized)
                    • **TEMPL Normalized**: Range 0-1, more conservative thresholds for higher quality
                    
                    **TEMPL Conservative Thresholds:**
                    • ≥0.35: Excellent (PMC equivalent: ≥0.7, more stringent than literature)
                    • ≥0.25: Good (PMC equivalent: ≥0.5, reliable pose prediction)
                    • ≥0.15: Fair (PMC equivalent: ≥0.3, acceptable for optimization)
                    
                    **Scientific References:**
                    1. "Sequential ligand- and structure-based virtual screening" (PMC9059856)
                    2. ChemBioChem: Large-scale TanimotoCombo analysis (269.7B pairs)
                    3. ROCS methodology validation (OpenEye Scientific)
                    
                    **Current Score: {combo_score:.3f}** (normalized 0-1 scale)
                    """
                )

    def _render_template_comparison(self):
        """Render template molecule comparison in details section"""
        template_mol = self.session.get("template_used")
        query_mol = self.session.get("query_mol")
        mcs_info = self.session.get("mcs_info")

        # Debug: Check what type of objects we have
        logger.info(f"template_mol type: {type(template_mol)}")
        logger.info(f"query_mol type: {type(query_mol)}")

        # Handle different types of stored molecule data
        def safe_get_mol(mol_data):
            """Safely extract RDKit mol from various data formats with enhanced error handling"""
            if mol_data is None:
                logger.debug("safe_get_mol: mol_data is None")
                return None

            logger.debug(f"safe_get_mol: Processing mol_data of type {type(mol_data)}")

            # Try to import RDKit (with error handling)
            try:
                from rdkit import Chem
            except ImportError as e:
                logger.error(f"safe_get_mol: RDKit import failed: {e}")
                return None

            # If it's already an RDKit molecule, validate and return it
            try:
                if hasattr(mol_data, "HasProp") and hasattr(mol_data, "GetNumAtoms"):
                    logger.debug("safe_get_mol: Found RDKit molecule object")
                    # Validate the molecule is not None
                    if mol_data is not None and mol_data.GetNumAtoms() > 0:
                        return mol_data
                    else:
                        logger.warning(
                            "safe_get_mol: RDKit molecule object is invalid or empty"
                        )
            except Exception as e:
                logger.warning(f"safe_get_mol: Error validating RDKit molecule: {e}")

            # If it's a dictionary, try to extract the molecule
            if isinstance(mol_data, dict):
                logger.debug("safe_get_mol: Processing dictionary data")
                # Try common keys where molecule might be stored
                for key in ["mol", "molecule", "rdkit_mol", "data"]:
                    if key in mol_data:
                        mol = mol_data[key]
                        logger.debug(
                            f"safe_get_mol: Found key '{key}' with type {type(mol)}"
                        )
                        if hasattr(mol, "HasProp") and hasattr(mol, "GetNumAtoms"):
                            try:
                                if mol.GetNumAtoms() > 0:
                                    return mol
                            except:
                                pass

                # Try to recreate from SMILES if available
                if "smiles" in mol_data:
                    try:
                        smiles = mol_data["smiles"]
                        logger.debug(
                            f"safe_get_mol: Attempting to create molecule from SMILES: {smiles}"
                        )
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            logger.debug(
                                "safe_get_mol: Successfully created molecule from SMILES"
                            )
                            return mol
                        else:
                            logger.warning(
                                f"safe_get_mol: Failed to create molecule from SMILES: {smiles}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"safe_get_mol: Error creating molecule from SMILES: {e}"
                        )

            # Try to use session SMILES as fallback
            try:
                input_smiles = self.session.get(SESSION_KEYS["INPUT_SMILES"])
                if input_smiles:
                    logger.debug(
                        f"safe_get_mol: Trying fallback with session SMILES: {input_smiles}"
                    )
                    mol = Chem.MolFromSmiles(input_smiles)
                    if mol is not None:
                        logger.debug(
                            "safe_get_mol: Successfully created molecule from session SMILES"
                        )
                        return mol
                    else:
                        logger.warning(
                            f"safe_get_mol: Failed to create molecule from session SMILES: {input_smiles}"
                        )
            except Exception as e:
                logger.warning(
                    f"safe_get_mol: Error using session SMILES fallback: {e}"
                )

            logger.error(
                f"safe_get_mol: Unable to extract valid molecule from data: {type(mol_data)}"
            )
            return None

        if template_mol or query_mol:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Query Molecule**")
                safe_query_mol = safe_get_mol(query_mol)
                if safe_query_mol:
                    display_molecule(safe_query_mol, width=220, height=180)
                else:
                    # Try to show SMILES as fallback
                    input_smiles = self.session.get(SESSION_KEYS["INPUT_SMILES"])
                    if input_smiles:
                        st.error(
                            "Error displaying molecule: Unable to parse query molecule data"
                        )
                        st.info(f"Query SMILES: `{input_smiles}`")
                        logger.error(
                            f"Query molecule visualization failed for SMILES: {input_smiles}"
                        )
                    else:
                        st.error(
                            "Error displaying molecule: Unable to parse query molecule data"
                        )
                        logger.error(
                            "Query molecule visualization failed - no SMILES available"
                        )

            with col2:
                st.markdown("**Template**")
                safe_template_mol = safe_get_mol(template_mol)
                if safe_template_mol:
                    display_molecule(safe_template_mol, width=220, height=180)
                else:
                    st.error(
                        "Error displaying molecule: Unable to parse template molecule data"
                    )

            with col3:
                st.markdown("**Common Substructure**")
                if mcs_info:
                    mcs_mol = safe_get_mcs_mol(mcs_info)
                    if mcs_mol:
                        display_molecule(mcs_mol, width=220, height=180)
                    else:
                        st.info("No significant MCS found")
                else:
                    st.info("MCS information not available")

            # Additional template information
            template_info = self.session.get("template_info")
            if template_info and template_info.get("mcs_smarts"):
                st.markdown(f"**MCS SMARTS:** `{template_info['mcs_smarts']}`")
        else:
            st.info("Template comparison not available")

    def _render_download_section(self):
        """Render download options with actual functionality"""
        st.markdown("### Download Results")

        poses = self.session.get("poses")
        all_ranked = self.session.get("all_ranked_poses")

        col1, col2 = st.columns(2)

        with col1:
            if poses:
                try:
                    # Use the function from utils.export_utils
                    sdf_data, filename = create_best_poses_sdf(poses)
                    st.download_button(
                        f"Best Poses ({len(poses)})",
                        data=sdf_data,
                        file_name=filename,
                        mime="chemical/x-mdl-sdfile",
                        help="Download top scoring poses for each method (shape, color, combo)",
                        use_container_width=True,
                        key="download_best_functional",
                    )
                except Exception as e:
                    logger.error(f"Error creating best poses SDF: {e}")
                    st.error("Failed to generate SDF file")
            else:
                st.button(
                    "Best Poses (N/A)",
                    disabled=True,
                    use_container_width=True,
                    key="download_best_disabled",
                )

        with col2:
            if all_ranked:
                try:
                    # Use the function from utils.export_utils
                    sdf_data, filename = create_all_conformers_sdf()
                    st.download_button(
                        f"All Conformers ({len(all_ranked)})",
                        data=sdf_data,
                        file_name=filename,
                        mime="chemical/x-mdl-sdfile",
                        help="Download all generated conformers ranked by combo score",
                        use_container_width=True,
                        key="download_all_functional",
                    )
                except Exception as e:
                    logger.error(f"Error creating all conformers SDF: {e}")
                    st.error("Failed to generate SDF file")
            else:
                st.button(
                    "All Conformers (N/A)",
                    disabled=True,
                    help="All ranked poses not available - try regenerating poses",
                    use_container_width=True,
                    key="download_all_disabled",
                )
