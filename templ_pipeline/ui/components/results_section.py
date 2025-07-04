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
        """Render detailed score interpretation for pose prediction with scientific explanations

        Args:
            scores: Score dictionary
        """
        combo_score = scores.get("combo_score", scores.get("combo", 0))

        # Determine quality level using pose prediction standards
        if combo_score >= SCORE_EXCELLENT:
            quality = "Excellent - High confidence pose"
            color = "green"
            explanation = (
                "Top 10-15% performance tier. Expected RMSD ≤ 1.0 Å from native structure. "
                "Suitable for lead optimization and structure-activity relationship studies."
            )
        elif combo_score >= SCORE_GOOD:
            quality = "Good - Reliable pose prediction"
            color = "blue"
            explanation = (
                "Meets established success criterion (RMSD ≤ 2.0 Å). Reliable for drug design applications. "
                "Comparable to high-performing docking tools like CB-Dock2 and Uni-Mol Docking V2."
            )
        elif combo_score >= SCORE_FAIR:
            quality = "Fair - Moderate confidence"
            color = "orange"
            explanation = (
                "Moderate quality pose (expected RMSD 2.0-3.0 Å). Consider validation with additional methods "
                "or use for initial screening with caution."
            )
        else:
            quality = "Poor - Low confidence, consider alternatives"
            color = "red"
            explanation = (
                "Below acceptable threshold for pose prediction (expected RMSD > 3.0 Å). "
                "Consider alternative templates or docking approaches."
            )

        # Display quality assessment with enhanced help
        st.markdown(f"**Quality Assessment:** :{color}[{quality}]")
        
        # Enhanced help section with modern UI/UX
        self._render_enhanced_help_section(combo_score, explanation)

    def _render_enhanced_help_section(self, combo_score: float, explanation: str):
        """Render enhanced help section with modern UI/UX patterns
        
        Args:
            combo_score: Current combo score
            explanation: Context-specific explanation
        """
        # Create help trigger with better UX - progressive disclosure
        with st.expander("Scoring Guide & Scientific References", expanded=False):
            # Add custom CSS for better styling
            st.markdown("""
            <style>
            .help-tab-content {
                padding: 10px 0;
                line-height: 1.6;
            }
            .help-metric {
                background-color: #f0f2f6;
                padding: 8px 12px;
                border-radius: 6px;
                margin: 5px 0;
                border-left: 4px solid #1f77b4;
            }
            .help-link {
                color: #1f77b4;
                text-decoration: none;
                font-weight: 500;
            }
            .help-link:hover {
                text-decoration: underline;
                color: #0d5aa7;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create tabbed interface for organized information
            tab1, tab2, tab3, tab4 = st.tabs([
                "Quick Guide",
                "Methodology", 
                "References",
                "Thresholds"
            ])
            
            with tab1:
                self._render_quick_guide(combo_score, explanation)
            
            with tab2:
                self._render_methodology_section()
                
            with tab3:
                self._render_references_section()
                
            with tab4:
                self._render_thresholds_section(combo_score)

    def _render_quick_guide(self, combo_score: float, explanation: str):
        """Render quick guide tab with essential information"""
        st.markdown('<div class="help-tab-content">', unsafe_allow_html=True)
        
        st.markdown("### Your Result")
        st.info(f"**Current Score: {combo_score:.3f}** - {explanation}")
        
        st.markdown("### Quick Interpretation")
        if combo_score >= SCORE_EXCELLENT:
            st.success("**Excellent**: Top-tier pose quality - proceed with confidence")
        elif combo_score >= SCORE_GOOD:
            st.info("**Good**: Reliable pose prediction - suitable for drug design")
        elif combo_score >= SCORE_FAIR:
            st.warning("**Fair**: Moderate confidence - consider additional validation")
        else:
            st.error("**Poor**: Low confidence - try alternative approaches")
        
        st.markdown("### Key Points")
        st.markdown("""
        - **Higher scores** indicate better pose accuracy
        - **RMSD** measures deviation from experimental structure
        - **Validation** recommended for critical applications
        - **Literature** thresholds guide interpretation
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

    def _render_methodology_section(self):
        """Render methodology tab with scientific explanation"""
        st.markdown('<div class="help-tab-content">', unsafe_allow_html=True)
        
        st.markdown("### TEMPL Scoring Methodology")
        
        st.markdown("""
        **Template-Based Pose Prediction** leverages 3D molecular similarity 
        to predict binding conformations from known ligand-protein complexes.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="help-metric">', unsafe_allow_html=True)
            st.markdown("""
            **Shape Tanimoto (ST)**
            - Volumetric overlap coefficient
            - Measures 3D shape complementarity
            - Score: 0.0 (no overlap) → 1.0 (identical)
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="help-metric">', unsafe_allow_html=True)
            st.markdown("""
            **Color Tanimoto (CT)**
            - Pharmacophoric feature similarity
            - Score: 0.0 (dissimilar) → 1.0 (identical)
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        
        st.markdown("### TanimotoCombo Score")
        st.markdown("""
        **Combined 3D Similarity Metric** integrating both shape and pharmacophoric features:
        """)

        # Use LaTeX for better formula rendering
        st.latex(r'''
        \text{TanimotoCombo} = \frac{\text{Shape}_T + \text{Color}_T}{2}
        ''')

        st.info("""
        **Interpretation:** Scores > 0.7 indicate high 3D similarity and reliable pose prediction
        """)

        
        st.markdown('</div>', unsafe_allow_html=True)

    def _render_references_section(self):
        """Render references tab with hyperlinked scientific papers"""
        st.markdown('<div class="help-tab-content">', unsafe_allow_html=True)
        
        st.markdown("### Key Scientific References")
        
        st.markdown("""
        Studies used to determine final TEMPL score thresholds:
        """)
        
        # Reference 1: CB-Dock2
        st.markdown("""
        **1. CB-Dock2: Improved Protein-Ligand Blind Docking**
        - 85% success rate at RMSD < 2.0 Å
        - Protein-ligand blind docking
        - [Read Paper](https://academic.oup.com/nar/article/50/W1/W159/6591526)
        - *Nucleic Acids Research, 2022*
        """)
        
        # Reference 2: Uni-Mol Docking V2
        st.markdown("""
        **2. Uni-Mol Docking V2: Realistic Binding Pose Prediction**
        - 77% accuracy for poses with RMSD < 2.0 Å
        - Modern benchmark for pose prediction
        - [Read Paper](https://arxiv.org/abs/2405.11769)
        - *arXiv preprint, 2024*
        """)
        
        # Reference 3: POSIT
        st.markdown("""
        **3. POSIT: Flexible Shape-Guided Docking**
        - Largest prospective validation (71 structures)
        - Shape-guided pose prediction emphasis
        - [Read Paper](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00142)
        - *J. Chem. Inf. Model., 2015*
        """)
        
        # Reference 4: DeepBSP
        st.markdown("""
        **4. DeepBSP: Machine Learning Pose Quality Assessment**
        - Direct RMSD prediction methodology
        - Validates RMSD-based quality assessment
        - [Read Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.1c00334)
        - *J. Chem. Inf. Model., 2021*
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

    def _render_thresholds_section(self, combo_score: float):
        """Render thresholds tab with quality assessment details"""
        st.markdown('<div class="help-tab-content">', unsafe_allow_html=True)
        
        st.markdown("### TEMPL Quality Thresholds")
        
        # Visual indicators for each threshold
        st.success("**Excellent (≥ 0.80)**: Expected RMSD ≤ 1.0 Å")
        st.markdown("Top 10-15% performance tier. Suitable for lead optimization.")
        
        st.info("**Good (≥ 0.65)**: Expected RMSD ≤ 2.0 Å") 
        st.markdown("Meets established success criterion. Reliable for drug design.")
        
        st.warning("**Fair (≥ 0.45)**: Expected RMSD 2.0-3.0 Å")
        st.markdown("Moderate confidence. Consider additional validation.")
        
        st.error("**Poor (< 0.45)**: Expected RMSD > 3.0 Å")
        st.markdown("Below acceptable threshold. Try alternative approaches.")
        
        st.markdown("### Current Assessment")
        current_quality = self._get_quality_label(combo_score)
        expected_rmsd = self._get_expected_rmsd(combo_score)
        
        st.markdown(f"""
        **Your Score: {combo_score:.3f}**
        - Quality: {current_quality}
        - {expected_rmsd}
        """)
        
        st.markdown("### Performance Standards")
        st.markdown("""
        Based on literature benchmarks:
        - **Success Criterion**: RMSD ≤ 2.0 Å from experimental structure
        - **High Performance**: 75-85% success rate (modern tools)
        - **Typical Performance**: 60-75% success rate (traditional methods)
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)

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
