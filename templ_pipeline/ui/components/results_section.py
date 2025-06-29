"""
Results Section Component for TEMPL Pipeline

Displays pose prediction results.
"""

import streamlit as st
import logging
from typing import Optional, Dict, Any

from ..config.settings import AppConfig
from ..config.constants import SESSION_KEYS, MESSAGES, SCORE_EXCELLENT, SCORE_GOOD, SCORE_FAIR
from ..core.session_manager import SessionManager
from ..utils.export_utils import create_best_poses_sdf, create_all_conformers_sdf, extract_pdb_id_from_template
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
        if (self.config.features.get('fair_metadata') or self.config.ui_settings.get('enable_fair_metadata')) and poses:
            col1, col2 = st.columns([20, 1])
            with col2:
                if st.button("Metadata 📊", help="View Scientific Data & Metadata", key="fair_trigger"):
                    st.session_state.show_fair_panel = True
                    st.rerun()
    
    def _render_template_badge(self):
        """Display template information prominently"""
        template_info = self.session.get("template_info")
        if template_info:
            template_pdb = template_info.get('name', 'Unknown')
            template_rank = template_info.get('index', 0) + 1
            total_templates = template_info.get('total_templates', 1)
            atoms_matched = template_info.get('atoms_matched', 0)
            
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
            combo_score = scores.get('combo_score', scores.get('combo', 0))
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
        
        shape_score = scores.get('shape_score', scores.get('shape', 0))
        color_score = scores.get('color_score', scores.get('color', 0))
        combo_score = scores.get('combo_score', scores.get('combo', 0))
        
        with col1:
            st.metric("Shape Similarity", f"{shape_score:.3f}")
        with col2:
            st.metric("Pharmacophore", f"{color_score:.3f}")
        with col3:
            st.metric("Overall Score", f"{combo_score:.3f}")
    
    def _render_score_details(self, scores: Dict):
        """Render detailed score interpretation
        
        Args:
            scores: Score dictionary
        """
        combo_score = scores.get('combo_score', scores.get('combo', 0))
        
        # Determine quality level
        if combo_score >= SCORE_EXCELLENT:
            quality = "Excellent - High confidence pose"
            color = "green"
        elif combo_score >= SCORE_GOOD:
            quality = "Good - Reliable pose prediction"
            color = "blue"
        elif combo_score >= SCORE_FAIR:
            quality = "Fair - Moderate confidence"
            color = "orange"
        else:
            quality = "Poor - Low confidence, consider alternatives"
            color = "red"
        
        st.markdown(f"**Quality Assessment:** :{color}[{quality}]")
    
    def _render_template_comparison(self):
        """Render template molecule comparison in details section"""
        template_mol = self.session.get("template_used")
        query_mol = self.session.get("query_mol")
        mcs_info = self.session.get("mcs_info")
        
        if template_mol and query_mol:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Query Molecule**")
                display_molecule(query_mol, width=220, height=180)
            
            with col2:
                st.markdown("**Template**")
                display_molecule(template_mol, width=220, height=180)
            
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
            if template_info and template_info.get('mcs_smarts'):
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
                        key="download_best_functional"
                    )
                except Exception as e:
                    logger.error(f"Error creating best poses SDF: {e}")
                    st.error("Failed to generate SDF file")
            else:
                st.button(
                    "Best Poses (N/A)",
                    disabled=True,
                    use_container_width=True,
                    key="download_best_disabled"
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
                        key="download_all_functional"
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
                    key="download_all_disabled"
                )
