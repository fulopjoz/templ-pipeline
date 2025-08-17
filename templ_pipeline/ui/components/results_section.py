# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Results Section Component for TEMPL Pipeline

Displays pose prediction results.
"""

import logging
from typing import Any, Dict, Optional

import streamlit as st

from ..config.constants import (
    MESSAGES,
    SCORE_EXCELLENT,
    SCORE_FAIR,
    SCORE_GOOD,
    SESSION_KEYS,
)
from ..config.settings import AppConfig
from ..core.session_manager import SessionManager
from ..utils.export_utils import (
    create_all_conformers_sdf,
    create_best_poses_sdf,
)
from ..utils.visualization_utils import (
    create_mcs_molecule_from_info,
    display_molecule,
    get_molecule_from_session,
    safe_get_mcs_mol,
)

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

        # Find best pose
        best_method, best_data = self._find_best_pose(poses)

        if best_method and best_data:
            mol, scores = best_data

            # Create unified single-line layout: Template info (left) | Tanimoto scores (right)
            self._render_unified_results_layout(scores)
            self._render_score_details(scores)

            # Template Details Section - More compact and focused
            with st.expander("Template Analysis", expanded=True):
                self._render_template_comparison()

            # Downloads with actual functionality
            self._render_download_section()
        else:
            st.error("No valid poses found in results")

    def _render_consolidated_template_header(self):
        """Display template information using native Streamlit components"""
        template_info = self.session.get(SESSION_KEYS["TEMPLATE_INFO"])

        if template_info and isinstance(template_info, dict):
            # Extract template data
            template_name = template_info.get("name", "Unknown")
            ca_rmsd_value = template_info.get("ca_rmsd")
            mcs_smarts = template_info.get("mcs_smarts")

            # Process RMSD value and quality
            rmsd_display = "N/A"
            quality_indicator = "⚪ Unknown"
            if ca_rmsd_value:
                try:
                    rmsd_val = float(ca_rmsd_value)
                    rmsd_display = f"{rmsd_val:.2f} Å"
                    if rmsd_val <= 2.0:
                        quality_indicator = "🟢 High Quality"
                    elif rmsd_val <= 5.0:
                        quality_indicator = "🟡 Moderate"
                    else:
                        quality_indicator = "🔴 Low Quality"
                except (ValueError, TypeError):
                    rmsd_display = f"{ca_rmsd_value} Å"

            # Process MCS status
            mcs_status = (
                "✅ Found" if mcs_smarts and len(mcs_smarts.strip()) > 0 else "❌ None"
            )

            # Create simple, clean display using native Streamlit components
            st.markdown("#### Template Context")

            # Use columns for horizontal layout
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"**Template:** `{template_name}`")

            with col2:
                st.markdown(f"**Cα RMSD:** `{rmsd_display}`")

            with col3:
                st.markdown(f"**MCS:** {mcs_status}")

            with col4:
                st.markdown(f"**Quality:** {quality_indicator}")

            st.markdown("---")  # Clean separator
        else:
            # Fallback for cases without template info
            st.info("Template information not available")

    def _render_unified_results_layout(self, scores: Dict):
        """Create unified single-line layout with Tanimoto scores (left) and template info (right)"""

        # Create two-column layout for the unified display - scores first, template second
        left_col, right_col = st.columns([5, 4], gap="medium")

        with left_col:
            # Tanimoto scores section with TanimotoCombo highlighted and first
            # Extract score values
            shape_score = scores.get("shape_score", scores.get("shape", 0))
            color_score = scores.get("color_score", scores.get("color", 0))
            combo_score = scores.get("combo_score", scores.get("combo", 0))

            # Create three columns for scores with TanimotoCombo first and highlighted
            score_cols = st.columns(3, gap="small")

            # TanimotoCombo first with highlighting
            with score_cols[0]:
                # Determine highlighting based on score quality
                combo_delta = None
                combo_color = "normal"
                if combo_score >= SCORE_EXCELLENT:
                    combo_delta = "Excellent pose"
                    combo_color = "normal"
                elif combo_score >= SCORE_GOOD:
                    combo_delta = "Good pose"
                    combo_color = "normal"
                elif combo_score >= SCORE_FAIR:
                    combo_delta = "Fair pose"
                    combo_color = "normal"
                else:
                    combo_delta = "Poor pose"
                    combo_color = "inverse"

                st.metric(
                    "TanimotoCombo",
                    f"{combo_score:.3f}",
                    delta=combo_delta,
                    delta_color=combo_color,
                    help="Primary quality metric - combined shape and pharmacophore similarity (normalized)",
                )

            # Shape and Color scores
            with score_cols[1]:
                st.metric(
                    "ShapeTanimoto",
                    f"{shape_score:.3f}",
                    help="3D molecular shape similarity to template",
                )
            with score_cols[2]:
                st.metric(
                    "ColorTanimoto",
                    f"{color_score:.3f}",
                    help="Pharmacophore feature similarity to template",
                )

        with right_col:
            # Template information section using consistent st.metric components
            template_info = self.session.get(SESSION_KEYS["TEMPLATE_INFO"])

            if template_info and isinstance(template_info, dict):
                # Extract and process template data
                template_name = template_info.get("name", "Unknown")
                ca_rmsd_value = template_info.get("ca_rmsd")
                mcs_smarts = template_info.get("mcs_smarts")

                # Process RMSD and quality
                rmsd_display = "N/A"
                rmsd_delta = None
                rmsd_color = "normal"
                quality_label = "Unknown"
                quality_delta = None
                quality_color = "normal"

                if ca_rmsd_value:
                    try:
                        rmsd_val = float(ca_rmsd_value)
                        rmsd_display = f"{rmsd_val:.2f} Å"
                        if rmsd_val <= 2.0:
                            quality_label = "High"
                            quality_delta = "Excellent template"
                            quality_color = "normal"
                            rmsd_delta = "Good alignment"
                        elif rmsd_val <= 5.0:
                            quality_label = "Moderate"
                            quality_delta = "Acceptable"
                            quality_color = "normal"
                        else:
                            quality_label = "Low"
                            quality_delta = "Poor alignment"
                            quality_color = "inverse"
                            rmsd_delta = "High deviation"
                            rmsd_color = "inverse"
                    except (ValueError, TypeError):
                        rmsd_display = str(ca_rmsd_value)

                # Process MCS status - extract atom count instead of Found/None
                mcs_atom_count = "N/A"
                mcs_delta = "No common structure"
                mcs_color = "inverse"

                if mcs_smarts and len(mcs_smarts.strip()) > 0:
                    try:
                        # Try to get atom count from MCS info
                        mcs_info = self.session.get(SESSION_KEYS["MCS_INFO"])
                        if mcs_info and hasattr(mcs_info, "GetNumAtoms"):
                            mcs_atom_count = str(mcs_info.GetNumAtoms())
                            mcs_delta = f"{mcs_atom_count} atoms matched"
                            mcs_color = "normal"
                        else:
                            # Fallback: create molecule from SMARTS to get atom count
                            from rdkit import Chem

                            mcs_mol = Chem.MolFromSmarts(mcs_smarts)
                            if mcs_mol:
                                atom_count = mcs_mol.GetNumAtoms()
                                mcs_atom_count = str(atom_count)
                                mcs_delta = f"{atom_count} atoms matched"
                                mcs_color = "normal"
                            else:
                                mcs_atom_count = "Found"
                                mcs_delta = "Common structure identified"
                                mcs_color = "normal"
                    except Exception as e:
                        # If anything fails, fallback to simple "Found"
                        mcs_atom_count = "Found"
                        mcs_delta = "Common structure identified"
                        mcs_color = "normal"

                # Create unified template metrics layout
                template_cols = st.columns(4, gap="small")

                with template_cols[0]:
                    st.metric(
                        "Template",
                        template_name,
                        help="Template structure used for pose prediction",
                    )
                with template_cols[1]:
                    st.metric(
                        "Cα RMSD",
                        rmsd_display,
                        delta=rmsd_delta,
                        delta_color=rmsd_color,
                        help="Root Mean Square Deviation of template alignment",
                    )
                with template_cols[2]:
                    st.metric(
                        "MCS",
                        mcs_atom_count,
                        delta=mcs_delta,
                        delta_color=mcs_color,
                        help="Maximum Common Substructure atom count between query and template",
                    )
                with template_cols[3]:
                    st.metric(
                        "Quality",
                        quality_label,
                        delta=quality_delta,
                        delta_color=quality_color,
                        help="Overall template quality assessment",
                    )
            else:
                st.info("Template information not available")

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
        """Render best pose information using native Streamlit components

        Args:
            method: Method name
            scores: Score dictionary
        """
        st.markdown("### Best Predicted Pose")

        # Extract score values
        shape_score = scores.get("shape_score", scores.get("shape", 0))
        color_score = scores.get("color_score", scores.get("color", 0))
        combo_score = scores.get("combo_score", scores.get("combo", 0))

        # Use native Streamlit metric components for clean, consistent display
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("ShapeTanimoto", f"{shape_score:.3f}")
        with col2:
            st.metric("ColorTanimoto", f"{color_score:.3f}")
        with col3:
            st.metric("TanimotoCombo", f"{combo_score:.3f}")

    def _render_score_details(self, scores: Dict):
        """Render scientific methodology information (quality assessment now in unified layout)

        Args:
            scores: Score dictionary
        """
        combo_score = scores.get("combo_score", scores.get("combo", 0))

        # Determine explanation for methodology section
        if combo_score >= SCORE_EXCELLENT:
            explanation = "High confidence pose"
        elif combo_score >= SCORE_GOOD:
            explanation = "Reliable pose prediction"
        elif combo_score >= SCORE_FAIR:
            explanation = "Moderate confidence"
        else:
            explanation = "Low confidence"

        # Scientific methodology section (quality assessment now integrated in unified layout)
        with st.expander("Scientific Assessment & Methodology", expanded=False):
            self._render_enhanced_help_content(combo_score, explanation)

    def _render_enhanced_help_content(self, combo_score: float, explanation: str):
        """Render enhanced help content with modern UI/UX patterns

        Args:
            combo_score: Current combo score
            explanation: Context-specific explanation
        """

        # Create simplified 2-section interface for better user experience
        tab1, tab2 = st.tabs(["Methodology & Assessment", "Scientific Basis"])

        with tab1:
            self._render_methodology_and_assessment(combo_score, explanation)

        with tab2:
            self._render_scientific_basis()

    def _render_methodology_and_assessment(self, combo_score: float, explanation: str):
        """Render consolidated methodology and assessment section using native Streamlit components"""

        # Current Assessment - Clean, minimal presentation
        st.markdown("#### Current Assessment")

        # Create containers for clean layout
        assessment_container = st.container()

        with assessment_container:
            # TanimotoCombo Score
            st.markdown(f"**TanimotoCombo Score:** `{combo_score:.3f}` - {explanation}")

            # Cα RMSD information now displayed in consolidated header above

            # Quality indicator
            if combo_score >= SCORE_EXCELLENT:
                quality_desc = "Excellent - High confidence"
            elif combo_score >= SCORE_GOOD:
                quality_desc = "Good - Reliable prediction"
            elif combo_score >= SCORE_FAIR:
                quality_desc = "Fair - Moderate confidence"
            else:
                quality_desc = "Poor - Low confidence"

            st.markdown(f"**Quality Assessment:** {quality_desc}")

        st.divider()

        # Methodology
        st.markdown("### TEMPL Scoring Methodology")
        st.markdown(
            "**Template-based pose prediction** using 3D molecular similarity and constrained conformer generation."
        )

        st.markdown("")  # Add some spacing

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**TanimotoCombo Score**")
            st.markdown("- Combined shape and pharmacophore similarity")
            st.markdown("- Normalized scale: 0.0 → 1.0")
            st.markdown("- Formula: (ShapeTanimoto + ColorTanimoto) / 2")

        with col2:
            st.markdown("**Cα RMSD Context**")
            st.markdown("- Measures template alignment quality")
            st.markdown("- Lower values indicate better structural match")
            st.markdown("- >5 Å may compromise pose accuracy")

        st.divider()

        # Quality Thresholds
        st.markdown("### Quality Thresholds & Expected Performance")

        # Current assessment summary at the top
        current_quality = self._get_quality_label(combo_score)
        expected_rmsd = self._get_expected_rmsd(combo_score)
        st.markdown(f"**Your prediction:** {current_quality} - {expected_rmsd}")

        st.markdown("")  # Add spacing

        # Use 2x2 grid layout for thresholds
        row1_col1, row1_col2 = st.columns(2)
        row2_col1, row2_col2 = st.columns(2)

        with row1_col1:
            st.markdown("**Excellent (≥ 0.80)**")
            st.text("RMSD ≤ 1.0 Å expected")

        with row1_col2:
            st.markdown("**Good (≥ 0.65)**")
            st.text("RMSD ≤ 2.0 Å expected")

        with row2_col1:
            st.markdown("**Fair (≥ 0.45)**")
            st.text("RMSD 2.0-3.0 Å expected")

        with row2_col2:
            st.markdown("**Poor (< 0.45)**")
            st.text("RMSD > 3.0 Å expected")

    def _render_scientific_basis(self):
        """Render simplified scientific basis section focusing on POSIT"""

        st.markdown("### Scientific Foundation")

        st.markdown(
            "TEMPL's scoring methodology is inspired by **Figure 4** from the POSIT paper."
        )

        st.markdown("")

        # Primary reference
        st.markdown("**POSIT: Flexible Shape-Guided Docking For Pose Prediction**")
        st.markdown(
            "- [Read Paper](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00142) | *J. Chem. Inf. Model., 2015*"
        )

        st.markdown("")

        st.markdown("### TEMPL Implementation")
        st.markdown(
            "- **3D similarity scoring** based on shape and pharmacophore alignment"
        )
        st.markdown(
            "- **Template-guided approach** leveraging known protein-ligand complexes"
        )
        st.markdown(
            "- **Conservative thresholds** ensuring meaningful quality discrimination"
        )
        st.markdown(
            "- **Cα RMSD integration** for template alignment quality assessment"
        )

    def _get_quality_label(self, score: float) -> str:
        """Get quality label for a given score"""
        if score >= SCORE_EXCELLENT:
            return "Excellent"
        elif score >= SCORE_GOOD:
            return "Good"
        elif score >= SCORE_FAIR:
            return "Fair"
        else:
            return "Poor"

    def _get_expected_rmsd(self, score):
        """Helper function to estimate expected RMSD based on combo score"""
        if score >= SCORE_EXCELLENT:
            return "RMSD ≤ 1.0 Å (high precision)"
        elif score >= SCORE_GOOD:
            return "RMSD ≤ 2.0 Å (acceptable quality)"
        elif score >= SCORE_FAIR:
            return "RMSD 2.0-3.0 Å (moderate quality)"
        else:
            return "RMSD > 3.0 Å (poor quality)"

    def _render_template_comparison(self):
        """Render compact template molecule comparison"""
        try:
            template_mol = self.session.get(SESSION_KEYS["TEMPLATE_USED"])
            query_mol = self.session.get(SESSION_KEYS["QUERY_MOL"])
            mcs_info = self.session.get(SESSION_KEYS["MCS_INFO"])
            template_info = self.session.get(SESSION_KEYS["TEMPLATE_INFO"])

            # Template information now consolidated in header - proceed directly to molecular visualization

            # Molecular visualization section
            if template_mol or query_mol:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Query Molecule**")
                    # Use utility function with fallback to current INPUT_SMILES
                    input_smiles = self.session.get(SESSION_KEYS["INPUT_SMILES"])
                    query_molecule = get_molecule_from_session(
                        self.session,
                        SESSION_KEYS["QUERY_MOL"],
                        fallback_smiles=input_smiles,
                    )

                    if query_molecule:
                        display_molecule(query_molecule, width=300, height=300)
                    else:
                        # Always attempt to render from current SMILES directly as ultimate fallback
                        if input_smiles:
                            try:
                                from rdkit import Chem

                                fallback_mol = Chem.MolFromSmiles(input_smiles)
                                if fallback_mol:
                                    fallback_mol.SetProp(
                                        "original_smiles", input_smiles
                                    )
                                    display_molecule(
                                        fallback_mol, width=300, height=300
                                    )
                                else:
                                    st.info("Query structure not available")
                            except Exception as e:
                                logger.error(
                                    f"SMILES fallback visualization failed: {e}"
                                )
                                st.info("Query structure not available")
                        else:
                            st.info("Query structure not available")

                with col2:
                    st.markdown("**Template**")
                    # Use utility function for template molecule
                    template_molecule = get_molecule_from_session(
                        self.session, SESSION_KEYS["TEMPLATE_USED"]
                    )

                    if template_molecule:
                        display_molecule(template_molecule, width=300, height=300)
                    else:
                        # Enhanced fallback with template info
                        if template_info and isinstance(template_info, dict):
                            template_name = template_info.get("name", "Unknown")
                            template_smiles = template_info.get("template_smiles")

                            if template_smiles:
                                # Try to create and display template molecule from SMILES
                                try:
                                    from rdkit import Chem

                                    fallback_template_mol = Chem.MolFromSmiles(
                                        template_smiles
                                    )
                                    if fallback_template_mol:
                                        display_molecule(
                                            fallback_template_mol, width=300, height=300
                                        )
                                    else:
                                        st.info(f"Template: {template_name}")
                                except Exception as e:
                                    logger.error(
                                        f"Template SMILES fallback visualization failed: {e}"
                                    )
                                    st.info(f"Template: {template_name}")
                            else:
                                st.info(f"Template: {template_name}")
                        else:
                            st.info("Template not available")

                with col3:
                    st.markdown("**Common Substructure**")

                    # Use utility function for MCS
                    mcs_molecule = create_mcs_molecule_from_info(mcs_info)

                    if mcs_molecule:
                        display_molecule(mcs_molecule, width=300, height=300)
                    else:
                        # Try to get MCS from template_info as fallback
                        mcs_found = False

                        if template_info and isinstance(template_info, dict):
                            mcs_smarts = template_info.get("mcs_smarts")
                            if mcs_smarts and len(mcs_smarts.strip()) > 0:
                                mcs_mol_fallback = create_mcs_molecule_from_info(
                                    mcs_smarts
                                )
                                if mcs_mol_fallback:
                                    display_molecule(
                                        mcs_mol_fallback, width=300, height=300
                                    )
                                    mcs_found = True

                        if not mcs_found:
                            st.info("No significant MCS found")

                # Compact additional information
                if template_info and isinstance(template_info, dict):
                    mcs_smarts = template_info.get("mcs_smarts")
                    if mcs_smarts and len(mcs_smarts.strip()) > 0:
                        with st.expander("MCS Details", expanded=False):
                            st.code(f"SMARTS: {mcs_smarts}")

            else:
                st.info("Template comparison not available")
                logger.debug(
                    "No template or query molecule data available for comparison"
                )
        except Exception as e:
            logger.error(f"Error rendering template comparison: {e}")
            st.error("Error displaying template comparison")
            if st.session_state.get("debug_mode", False):
                st.exception(e)

    def _render_download_section(self):
        """Render download options with actual functionality"""
        st.markdown("### Download Results")

        poses = self.session.get(SESSION_KEYS["POSES"])
        all_ranked = self.session.get(SESSION_KEYS["ALL_RANKED_POSES"])

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

                    # Check if the export was successful
                    if filename == "error.sdf" or not sdf_data:
                        st.button(
                            "All Conformers (Error)",
                            disabled=True,
                            help="Error generating SDF file - check logs for details",
                            use_container_width=True,
                            key="download_all_error",
                        )
                    else:
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
                    st.button(
                        "All Conformers (Error)",
                        disabled=True,
                        help=f"Error generating SDF file: {str(e)}",
                        use_container_width=True,
                        key="download_all_exception",
                    )
            else:
                st.button(
                    "All Conformers (N/A)",
                    disabled=True,
                    help="All ranked poses not available - try regenerating poses",
                    use_container_width=True,
                    key="download_all_disabled",
                )
