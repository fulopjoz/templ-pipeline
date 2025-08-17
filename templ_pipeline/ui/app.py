# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
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
from typing import Dict, Any, Optional, Callable

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import refactored modules
from templ_pipeline.ui.config.settings import get_config
from templ_pipeline.ui.core.session_manager import get_session_manager
from templ_pipeline.ui.core.hardware_manager import get_hardware_manager
from templ_pipeline.ui.layouts.main_layout import MainLayout

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def handle_health_check():
    """Handle health check endpoints for deployment monitoring"""
    query_params = st.query_params

    # Check for health check endpoints
    if query_params.get("health") == "check" or query_params.get("healthz") is not None:
        logger.info("Health check requested")
        
        # Perform basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": str(Path(__file__).stat().st_mtime),
            "python_version": sys.version,
            "working_directory": str(Path.cwd()),
        }
        
        # Check critical imports
        try:
            from templ_pipeline.ui.config.settings import get_config
            health_status["config_import"] = "  OK"
        except ImportError as e:
            health_status["config_import"] = f"  FAIL: {e}"
            health_status["status"] = "unhealthy"
        
        try:
            from templ_pipeline.ui.core.session_manager import get_session_manager
            health_status["session_manager_import"] = "  OK"
        except ImportError as e:
            health_status["session_manager_import"] = f"  FAIL: {e}"
            health_status["status"] = "unhealthy"
        
        if health_status["status"] == "healthy":
            st.success("  TEMPL Pipeline Health Check: OK")
        else:
            st.error("  TEMPL Pipeline Health Check: FAILED")
            
        st.json(health_status)
        st.stop()


def initialize_app():
    """Initialize application configuration and core services with comprehensive error handling"""
    try:
        logger.info("Starting application initialization...")
        
        # Get configuration
        logger.info("Loading configuration...")
        config = get_config()
        logger.info(f"Configuration loaded successfully: {config.app_version}")

        # Page config is set once in main(); avoid duplicate calls per Streamlit guidance

        # Get session manager
        logger.info("Initializing session manager...")
        session = get_session_manager(config)
        logger.info("Session manager created successfully")

        # Initialize session state
        logger.info("Initializing session state...")
        session.initialize()
        logger.info("Session state initialized successfully")

        # Initialize hardware detection (cached)
        logger.info("Detecting hardware configuration...")
        hardware_manager = get_hardware_manager()
        hardware_info = hardware_manager.detect_hardware()
        session.set("hardware_info", hardware_info)
        logger.info(f"Hardware detection completed: {hardware_info.recommended_config}")

        # Log initialization
        if not session.get("initialization_logged", False):
            logger.info(f"TEMPL Pipeline v{config.app_version} initialized")
            logger.info(f"Hardware: {hardware_info.recommended_config}")
            logger.info(f"Features: {config.features}")
            session.set("initialization_logged", True)

        logger.info("Application initialization completed successfully")
        return config, session
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
        
        # Display detailed error information
        st.error("Application Initialization Failed")
        st.error(f"Error: {str(e)}")
        
        with st.expander("Detailed Error Information", expanded=True):
            st.code(traceback.format_exc())
            
            # Show import status
            st.subheader("Import Status Check")
            imports_to_check = [
                "templ_pipeline.ui.config.settings",
                "templ_pipeline.ui.core.session_manager", 
                "templ_pipeline.ui.core.hardware_manager",
                "templ_pipeline.ui.layouts.main_layout"
            ]
            
            for import_name in imports_to_check:
                try:
                    __import__(import_name)
                    st.success(f"{import_name}")
                except ImportError as ie:
                    st.error(f"{import_name}: {ie}")
                    
        # Allow user to continue with basic functionality
        st.info("You can try refreshing the page or check the server logs for more details.")
        
        raise


def main():
    try:
        st.set_page_config(
            # page_title="TEMPL Pipeline",
            page_icon="♟",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        with st.spinner("Initializing TEMPL Pipeline..."):
            # Add debug marker
            st.session_state._debug_marker = "app_v2_main_started"
            logger.info("Main function started")

            # Handle health checks first
            logger.info("Handling health checks...")
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
            logger.info("Main layout created successfully")

            # Render the layout
            logger.info("Rendering main layout...")
            layout.render()

            logger.info("Main function completed successfully")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)

        # Store error in session for debugging
        if "session" in locals():
            session.set(
                "last_error",
                {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "type": type(e).__name__,
                },
            )

        # Show error page with more details
        st.error("TEMPL Pipeline Application Error")
        st.error(f"**Error Type**: {type(e).__name__}")
        st.error(f"**Error Message**: {str(e)}")

        with st.expander("📋 Full Error Details", expanded=True):
            st.code(traceback.format_exc())

            # Show environment information
            st.subheader("Environment Information")
            st.write(f"**Python Version**: {sys.version}")
            st.write(f"**Working Directory**: {Path.cwd()}")
            st.write(f"**Python Path**: {sys.path[:3]}...")

            # Show session state for debugging
            st.subheader("Session State Debug Info")
            try:
                if "st.session_state" in globals():
                    debug_state = {}
                    for key in st.session_state:
                        try:
                            value = st.session_state[key]
                            # Skip large objects
                            if key in [
                                "query_mol",
                                "poses",
                                "custom_templates",
                                "all_ranked_poses",
                            ]:
                                debug_state[key] = f"<{type(value).__name__} object>"
                            else:
                                debug_state[key] = str(value)[
                                    :100
                                ]  # Truncate long values
                        except:
                            debug_state[key] = "<error accessing>"
                    st.json(debug_state)
            except Exception as debug_error:
                st.write(f"Could not display session state: {debug_error}")

            # Recovery options
            st.subheader("Recovery Options")
            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("Restart Application"):
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.rerun()

            with col2:
                if st.button("Clear Session Only"):
                    for key in list(st.session_state.keys()):
                        if key != "_debug_marker":
                            del st.session_state[key]
                    st.rerun()

            with col3:
                if st.button("Copy Error Info"):
                    error_info = f"Error: {str(e)}\n\n{traceback.format_exc()}"
                    st.code(error_info)
                    st.info("Copy the error information above")

        # Show helpful suggestions
        st.subheader("Troubleshooting Suggestions")
        st.info("1. Try refreshing the page (Ctrl+F5)")
        st.info("2. Check the server logs for more details")
        st.info("3. Ensure all required dependencies are installed")
        st.info("4. Verify the TEMPL data files are accessible")


# Compatibility layer for backward compatibility and testing
# These functions provide easy access to core functionality from app.py


def run_pipeline(
    smiles: str,
    protein_input: str,
    custom_templates: Optional[list] = None,
    use_aligned_poses: bool = True,
    max_templates: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Compatibility function for running the TEMPL pipeline synchronously

    This function provides backward compatibility for tests and external code
    that expects to import run_pipeline from app.py.

    Args:
        smiles: SMILES string for the query molecule
        protein_input: PDB ID or file path for protein input
        custom_templates: Optional custom template molecules
        use_aligned_poses: Whether to use aligned poses
        max_templates: Maximum number of templates to use
        similarity_threshold: Similarity threshold for template search

    Returns:
        Results dictionary or None on failure
    """
    from .services.pipeline_service import PipelineService
    from .config.settings import get_config
    from .core.session_manager import SessionManager

    # Create temporary configuration and session
    config = get_config()
    session = SessionManager(config)
    session.initialize()

    # Create pipeline service and prepare data
    service = PipelineService(config, session)

    # Prepare molecule and protein data dictionaries
    molecule_data = {
        "input_smiles": smiles,
        "custom_templates": custom_templates,
    }

    # Determine if protein_input is a file path or PDB ID
    if isinstance(protein_input, str):
        if protein_input.lower().endswith((".pdb", ".ent")):
            # It's a file path
            protein_data = {"file_path": protein_input}
        else:
            # It's a PDB ID
            protein_data = {"pdb_id": protein_input}
    else:
        protein_data = {"pdb_id": str(protein_input)}

    return service.run_pipeline(molecule_data, protein_data)


async def run_pipeline_async(
    smiles: str,
    protein_input: str,
    custom_templates: Optional[list] = None,
    use_aligned_poses: bool = True,
    max_templates: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    progress_callback: Optional[Callable] = None,
) -> Optional[Dict[str, Any]]:
    """Compatibility function for running the TEMPL pipeline asynchronously

    This function provides backward compatibility for tests and external code
    that expects to import run_pipeline_async from app.py.

    Args:
        smiles: SMILES string for the query molecule
        protein_input: PDB ID or file path for protein input
        custom_templates: Optional custom template molecules
        use_aligned_poses: Whether to use aligned poses
        max_templates: Maximum number of templates to use
        similarity_threshold: Similarity threshold for template search
        progress_callback: Optional callback for progress updates

    Returns:
        Results dictionary or None on failure
    """
    from .services.pipeline_service import (
        run_pipeline_async as service_run_pipeline_async,
    )

    return await service_run_pipeline_async(
        smiles=smiles,
        protein_input=protein_input,
        custom_templates=custom_templates,
        use_aligned_poses=use_aligned_poses,
        max_templates=max_templates,
        similarity_threshold=similarity_threshold,
        progress_callback=progress_callback,
    )


def validate_smiles_input(smiles: str):
    """Compatibility function for SMILES validation

    This function provides backward compatibility for tests and external code
    that expects to import validate_smiles_input from app.py.

    Args:
        smiles: SMILES string to validate

    Returns:
        Tuple of (is_valid, message, molecule_data)
    """
    from .utils.molecular_utils import (
        validate_smiles_input as utils_validate_smiles_input,
    )

    return utils_validate_smiles_input(smiles)


def generate_molecule_image(mol_binary, width=400, height=300, highlight_atoms=None):
    """Compatibility function for molecule image generation

    This function provides backward compatibility for tests and external code
    that expects to import generate_molecule_image from app.py.

    Args:
        mol_binary: Binary molecule data
        width: Image width in pixels
        height: Image height in pixels
        highlight_atoms: Optional atoms to highlight

    Returns:
        Generated molecule image
    """
    from .utils.visualization_utils import (
        generate_molecule_image as utils_generate_molecule_image,
    )

    return utils_generate_molecule_image(mol_binary, width, height, highlight_atoms)


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
