"""
TEMPL Pipeline Web Application - Version 2.0

Clean, modular entry point for the refactored TEMPL Pipeline UI.
This replaces the monolithic app.py with a clean architecture.
"""

import streamlit as st
import logging
import sys
from pathlib import Path
import traceback

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import refactored modules
from templ_pipeline.ui.config.settings import get_config
from templ_pipeline.ui.core.session_manager import get_session_manager
from templ_pipeline.ui.core.hardware_manager import get_hardware_manager
from templ_pipeline.ui.ui.layouts.main_layout import MainLayout

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def handle_health_check():
    """Handle health check endpoints for deployment monitoring"""
    query_params = st.query_params
    
    # Check for health check endpoints
    if query_params.get("health") == "check" or query_params.get("healthz") is not None:
        st.write("OK")
        st.stop()


def initialize_app():
    """Initialize application configuration and core services"""
    # Get configuration
    config = get_config()
    
    # Configure Streamlit page
    try:
        st.set_page_config(**config.page_config)
    except st.errors.StreamlitAPIException:
        # Page already configured (on rerun)
        pass
    
    # Get session manager
    session = get_session_manager(config)
    
    # Initialize session state
    session.initialize()
    
    # Initialize hardware detection (cached)
    hardware_manager = get_hardware_manager()
    hardware_info = hardware_manager.detect_hardware()
    session.set("hardware_info", hardware_info)
    
    # Log initialization
    if not session.get("initialization_logged", False):
        logger.info(f"TEMPL Pipeline v{config.app_version} initialized")
        logger.info(f"Hardware: {hardware_info.recommended_config}")
        logger.info(f"Features: {config.features}")
        session.set("initialization_logged", True)
    
    return config, session


def main():
    """Main application entry point"""
    try:
        # Add debug marker
        st.session_state._debug_marker = "app_v2_main_started"
        logger.info("Main function started")
        
        # Handle health checks first
        handle_health_check()
        
        # Initialize application
        logger.info("Initializing application...")
        config, session = initialize_app()
        logger.info("Application initialized successfully")
        
        # Debug: Log session state
        logger.info(f"Session has_results: {session.has_results()}")
        logger.info(f"Session has_valid_input: {session.has_valid_input()}")
        
        # Create layout components
        logger.info("Creating main layout...")
        layout = MainLayout(config, session)
        
        # Import header component
        from templ_pipeline.ui.ui.components.header import render_header
        
        # Render header
        render_header(config, session)
        
        # Add pipeline explanation
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div style="background: rgba(128, 128, 128, 0.1); padding: 1.5rem; border-radius: 0.8rem; border: 1px solid rgba(128, 128, 128, 0.2);">
                <h4 style="margin-top: 0;">What it does</h4>
                <ul>
                    <li>Predicts 3D binding poses for small molecules</li> 
                    <li>Uses template-guided conformer generation with MCS</li>
                    <li>Provides shape, pharmacophore, and combined scoring</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(128, 128, 128, 0.1); padding: 1.5rem; border-radius: 0.8rem; border: 1px solid rgba(128, 128, 128, 0.2);">
                <h4 style="margin-top: 0;">How it works</h4>
                <ol>
                    <li>Enter your molecule (SMILES or file)</li>
                    <li>Provide protein target or SDF templates</li>
                    <li>Get ranked poses with confidence scores</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Render main content without system status
        logger.info("Rendering main content...")
        
        # Check if we have results to show
        if session.has_results():
            # Create tabs for input and results
            tab1, tab2 = st.tabs(["New Prediction", "Results"])
            
            with tab1:
                layout.input_section.render()
                if session.has_valid_input():
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        if st.button("PREDICT POSES", type="primary", use_container_width=True):
                            layout._handle_prediction()
            
            with tab2:
                layout.results_section.render()
        else:
            # Show input area
            if not session.has_valid_input():
                st.info("Provide your molecule and protein target for pose prediction")
            
            layout.input_section.render()
            
            # Show action button if inputs are valid
            if session.has_valid_input():
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("PREDICT POSES", type="primary", use_container_width=True):
                        layout._handle_prediction()
        
        # Render status bar
        from templ_pipeline.ui.ui.components.status_bar import render_status_bar
        render_status_bar(session)
        
        logger.info("Main function completed successfully")
        
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        
        # Store error in session for debugging
        if 'session' in locals():
            session.set("last_error", {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "type": type(e).__name__
            })
        
        # Show error page with more details
        st.error("An unexpected error occurred")
        
        # Show the actual error message
        st.error(f"Error: {str(e)}")
        
        with st.expander("Full Error Details", expanded=True):
            st.code(traceback.format_exc())
            
            # Show session state for debugging
            st.subheader("Session State Debug Info")
            try:
                if 'st.session_state' in globals():
                    debug_state = {}
                    for key in st.session_state:
                        try:
                            value = st.session_state[key]
                            # Skip large objects
                            if key in ['query_mol', 'poses', 'custom_templates', 'all_ranked_poses']:
                                debug_state[key] = f"<{type(value).__name__} object>"
                            else:
                                debug_state[key] = str(value)[:100]  # Truncate long values
                        except:
                            debug_state[key] = "<error accessing>"
                    st.json(debug_state)
            except Exception as debug_error:
                st.write(f"Could not display session state: {debug_error}")
            
            # Recovery options
            st.subheader("Recovery Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Restart Application"):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            
            with col2:
                if st.button("Clear Session Only"):
                    for key in list(st.session_state.keys()):
                        if key != '_debug_marker':
                            del st.session_state[key]
                    st.rerun()
            
            with col3:
                if st.button("📋 Copy Error Info"):
                    error_info = f"Error: {str(e)}\n\n{traceback.format_exc()}"
                    st.code(error_info)
                    st.info("Copy the error information above")


if __name__ == "__main__":
    # Wrap in another try-catch for absolute safety
    try:
        main()
    except Exception as critical_error:
        # Last resort error display
        st.error(f"CRITICAL ERROR: {critical_error}")
        st.code(traceback.format_exc())
        if st.button("Emergency Restart"):
            st.rerun()
