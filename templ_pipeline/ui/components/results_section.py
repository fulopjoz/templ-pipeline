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
from ..utils.visualization_utils import (
    display_molecule, 
    safe_get_mcs_mol,
    get_molecule_from_session,
    create_mcs_molecule_from_info
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

        # Template Information Badge (NEW)
        self._render_template_badge()

        # Find best pose
        best_method, best_data = self._find_best_pose(poses)

        if best_method and best_data:
            mol, scores = best_data
            self._render_best_pose(best_method, scores)
            self._render_score_details(scores)

            # Template Details Section (NEW) - Show first and expanded
            with st.expander("Template Details", expanded=True):
                self._render_template_comparison()

            # Downloads with actual functionality
            self._render_download_section()
        else:
            st.error("No valid poses found in results")



    def _render_template_badge(self):
        """Display template information prominently"""
        template_info = self.session.get("template_info")
        if template_info:
            template_pdb = template_info.get("name", "Unknown")
            template_rank = template_info.get("index", 0) + 1

            # Check if custom templates are being used
            if self.session.get(SESSION_KEYS["CUSTOM_TEMPLATES"]):
                total_templates = template_info.get("total_templates", 1)
            else:
                # Get user-defined k-NN value from session for standard pipeline runs
                total_templates = self.session.get(
                    SESSION_KEYS["USER_KNN_THRESHOLD"], 100
                )

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
            explanation = "High confidence pose - proceed with confidence"
        elif combo_score >= SCORE_GOOD:
            quality = "Good - Reliable pose prediction"
            color = "blue"
            explanation = "Reliable pose prediction - suitable for drug design"
        elif combo_score >= SCORE_FAIR:
            quality = "Fair - Moderate confidence"
            color = "orange"
            explanation = "Moderate confidence - consider additional validation"
        else:
            quality = "Poor - Low confidence, consider alternatives"
            color = "red"
            explanation = "Low confidence - try alternative approaches"

        # Display quality assessment with enhanced help
        st.markdown(f"**Quality Assessment:** :{color}[{quality}]")
        
        # Enhanced help section with modern UI/UX - moved after template details
        with st.expander("Scoring Guide & Scientific References", expanded=False):
            self._render_enhanced_help_content(combo_score, explanation)

    def _render_enhanced_help_content(self, combo_score: float, explanation: str):
        """Render enhanced help content with modern UI/UX patterns
        
        Args:
            combo_score: Current combo score
            explanation: Context-specific explanation
        """
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
        st.success("**Excellent (≥ 0.80)**: High confidence pose")
        st.markdown("Proceed with confidence")
        
        st.info("**Good (≥ 0.65)**: Reliable pose prediction") 
        st.markdown("Suitable for drug design")
        
        st.warning("**Fair (≥ 0.45)**: Moderate confidence")
        st.markdown("Consider additional validation")
        
        st.error("**Poor (< 0.45)**: Low confidence")
        st.markdown("Try alternative approaches")
        
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
        try:
            template_mol = self.session.get(SESSION_KEYS["TEMPLATE_USED"])
            query_mol = self.session.get(SESSION_KEYS["QUERY_MOL"])
            mcs_info = self.session.get(SESSION_KEYS["MCS_INFO"])

            # Debug: Check what type of objects we have
            logger.info(f"template_mol type: {type(template_mol)}")
            logger.info(f"query_mol type: {type(query_mol)}")
            logger.info(f"mcs_info type: {type(mcs_info)}")

            if template_mol or query_mol:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Query Molecule**")
                    # Use utility function with fallback SMILES
                    input_smiles = self.session.get(SESSION_KEYS["INPUT_SMILES"])
                    query_molecule = get_molecule_from_session(
                        self.session, SESSION_KEYS["QUERY_MOL"], fallback_smiles=input_smiles
                    )
                    
                    if query_molecule:
                        display_molecule(query_molecule, width=220, height=300)
                    else:
                        # Show fallback information
                        if input_smiles:
                            st.warning("Error displaying query molecule")
                            st.info(f"Query SMILES: `{input_smiles}`")
                            
                            # Try to create and display from SMILES directly
                            try:
                                from rdkit import Chem
                                fallback_mol = Chem.MolFromSmiles(input_smiles)
                                if fallback_mol:
                                    display_molecule(fallback_mol, width=220, height=300)
                                    st.success("Displayed from SMILES fallback")
                            except Exception as e:
                                logger.error(f"SMILES fallback visualization failed: {e}")
                        else:
                            st.error("Query molecule data not available")

                with col2:
                    st.markdown("**Template**")
                    # Use utility function for template molecule
                    template_molecule = get_molecule_from_session(
                        self.session, SESSION_KEYS["TEMPLATE_USED"]
                    )
                    
                    if template_molecule:
                        display_molecule(template_molecule, width=220, height=300)
                    else:
                        # Enhanced fallback with template info
                        template_info = self.session.get(SESSION_KEYS["TEMPLATE_INFO"])
                        if template_info and isinstance(template_info, dict):
                            template_name = template_info.get("name", "Unknown")
                            template_smiles = template_info.get("template_smiles")
                            
                            if template_smiles:
                                # Try to create and display template molecule from SMILES
                                try:
                                    from rdkit import Chem
                                    fallback_template_mol = Chem.MolFromSmiles(template_smiles)
                                    if fallback_template_mol:
                                        display_molecule(fallback_template_mol, width=220, height=300)
                                        st.success(f"Template: {template_name}")
                                    else:
                                        st.warning(f"Template: {template_name}")
                                        st.code(f"SMILES: {template_smiles}")
                                except Exception as e:
                                    logger.error(f"Template SMILES fallback visualization failed: {e}")
                                    st.warning(f"Template: {template_name}")
                                    st.code(f"SMILES: {template_smiles}")
                            else:
                                st.warning(f"Template: {template_name}")
                                st.info("No molecular structure available")
                        else:
                            st.error("Template information not available")

                with col3:
                    st.markdown("**Common Substructure**")
                    
                    # Use utility function for MCS
                    mcs_molecule = create_mcs_molecule_from_info(mcs_info)
                    
                    if mcs_molecule:
                        display_molecule(mcs_molecule, width=220, height=300)
                        # Show atom count if available
                        try:
                            atom_count = mcs_molecule.GetNumAtoms()
                            st.success(f"MCS found ({atom_count} atoms)")
                        except:
                            st.success("MCS structure displayed")
                    else:
                        # Try to get MCS from template_info as fallback
                        template_info = self.session.get(SESSION_KEYS["TEMPLATE_INFO"])
                        mcs_found = False
                        
                        if template_info and isinstance(template_info, dict):
                            mcs_smarts = template_info.get("mcs_smarts")
                            if mcs_smarts and len(mcs_smarts.strip()) > 0:
                                mcs_mol_fallback = create_mcs_molecule_from_info(mcs_smarts)
                                if mcs_mol_fallback:
                                    display_molecule(mcs_mol_fallback, width=220, height=300)
                                    try:
                                        atom_count = mcs_mol_fallback.GetNumAtoms()
                                        st.success(f"MCS found ({atom_count} atoms)")
                                    except:
                                        st.success("MCS structure displayed")
                                    mcs_found = True
                        
                        if not mcs_found:
                            # Show informative message about MCS status
                            if mcs_info is None:
                                st.info("No MCS analysis performed")
                            elif isinstance(mcs_info, dict) and not any(mcs_info.get(key) for key in ["smarts", "mcs_smarts"]):
                                st.info("No significant common substructure found")
                            else:
                                st.warning("Could not display MCS structure")

                # Additional template information
                template_info = self.session.get(SESSION_KEYS["TEMPLATE_INFO"])
                if template_info:
                    st.markdown("---")
                    
                    # Display MCS SMARTS if available
                    if isinstance(template_info, dict):
                        mcs_smarts = template_info.get("mcs_smarts")
                        if mcs_smarts and len(mcs_smarts.strip()) > 0:
                            st.markdown(f"**MCS SMARTS:** `{mcs_smarts}`")
                        
                        # Display additional template info in columns
                        info_col1, info_col2 = st.columns(2)
                        
                        with info_col1:
                            if template_info.get("name"):
                                st.markdown(f"**Template:** {template_info['name']}")
                            if template_info.get("atoms_matched"):
                                st.markdown(f"**Atoms Matched:** {template_info['atoms_matched']}")
                        
                        with info_col2:
                            if template_info.get("index") is not None:
                                total = template_info.get("total_templates", 1)
                                rank = template_info.get("index", 0) + 1
                                st.markdown(f"**Template Rank:** {rank}/{total}")
                            if template_info.get("ca_rmsd"):
                                try:
                                    ca_rmsd_value = float(template_info['ca_rmsd'])
                                    st.markdown(f"**CA RMSD:** {ca_rmsd_value:.2f} Å")
                                except (ValueError, TypeError):
                                    st.markdown(f"**CA RMSD:** {template_info['ca_rmsd']} Å")
                            
                # Debug info display for troubleshooting (only in debug mode)
                if st.session_state.get("debug_mode", False):
                    with st.expander("🔍 Debug Info - Molecule Data"):
                        st.markdown("**Session Data Types:**")
                        st.write(f"- Template: {type(template_mol)}")
                        st.write(f"- Query: {type(query_mol)}")
                        st.write(f"- MCS Info: {type(mcs_info)}")
                        st.write(f"- Template Info: {type(template_info)}")
                        
                        if template_info and isinstance(template_info, dict):
                            st.markdown("**Template Info Contents:**")
                            for key, value in template_info.items():
                                st.write(f"  - {key}: {value}")
                        
                        if mcs_info:
                            st.markdown("**MCS Info Contents:**")
                            if isinstance(mcs_info, dict):
                                for key, value in mcs_info.items():
                                    if key == "smarts" or key == "mcs_smarts":
                                        st.write(f"  - {key}: `{value}`")
                                    else:
                                        st.write(f"  - {key}: {value} ({type(value)})")
                            else:
                                st.write(f"  - Type: {type(mcs_info)}")
                                st.write(f"  - Value: {str(mcs_info)[:200]}...")
                        
                        # Test molecule retrieval functions
                        st.markdown("**Retrieval Function Tests:**")
                        try:
                            # Test template retrieval (functions already imported at top)
                            test_template = get_molecule_from_session(self.session, SESSION_KEYS["TEMPLATE_USED"])
                            st.write(f"  - Template retrieval test: {type(test_template)} ({'success' if test_template else 'failed'})")
                            
                            # Test MCS creation
                            test_mcs = create_mcs_molecule_from_info(mcs_info)
                            st.write(f"  - MCS creation test: {type(test_mcs)} ({'success' if test_mcs else 'failed'})")
                            
                            if test_mcs:
                                try:
                                    st.write(f"  - MCS atoms: {test_mcs.GetNumAtoms()}")
                                except:
                                    st.write("  - Could not get MCS atom count")
                                    
                        except Exception as debug_error:
                            st.write(f"  - Debug test error: {debug_error}")
            else:
                st.info("Template comparison not available")
                logger.debug("No template or query molecule data available for comparison")
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
