"""
Main Layout for TEMPL Pipeline

Orchestrates the overall application layout and component rendering.
"""

import streamlit as st
import logging
from typing import Optional, Dict, Any

from ..config.settings import AppConfig
from ..config.constants import MESSAGES, VERSION, SESSION_KEYS
from ..core.session_manager import SessionManager
from ..core.hardware_manager import get_hardware_manager
from ..core.cache_manager import get_cache_manager
from ..components.header import render_header
from ..components.input_section import InputSection
from ..components.results_section import ResultsSection
from ..components.status_bar import render_status_bar
from ..utils.performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class MainLayout:
    """Main application layout orchestrator"""
    
    def __init__(self, config: AppConfig, session: SessionManager):
        """Initialize main layout
        
        Args:
            config: Application configuration
            session: Session manager instance
        """
        self.config = config
        self.session = session
        self.hardware_manager = get_hardware_manager()
        self.cache_manager = get_cache_manager()
        self.performance_monitor = PerformanceMonitor()
        
        # Component instances
        self.input_section = InputSection(config, session)
        self.results_section = ResultsSection(config, session)
        
        # Progress tracking
        self.progress_bar = None
        self.progress_text = None
    
    def render(self):
        """Render the complete application layout"""
        try:
            # Start performance monitoring
            self.performance_monitor.start_render("main_layout")
            
            # Render header
            render_header(self.config, self.session)
            
            # Show system status if configured
            if self.config.ui_settings.get("show_system_status", True):
                self._render_system_status()
            
            # Main content area
            self._render_main_content()
            
            # Status bar at bottom
            if self.config.ui_settings.get("show_status_bar", True):
                render_status_bar(self.session)
            
            # End performance monitoring
            render_time = self.performance_monitor.end_render("main_layout")
            logger.debug(f"Main layout rendered in {render_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error rendering main layout: {e}", exc_info=True)
            self._render_error_state(e)
    
    def _render_system_status(self):
        """Render system status section"""
        with st.expander("System Information", expanded=False):
            hardware_status = self.hardware_manager.get_status_summary()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Hardware**")
                hw = hardware_status['hardware']
                st.markdown(f"CPU: {hw['cpu_cores']} cores")
                st.markdown(f"RAM: {hw['ram_gb']} GB")
                st.markdown(f"GPU: {hw['gpu']}")
                if hw['gpu'] != 'Not available':
                    st.markdown(f"VRAM: {hw['gpu_memory_gb']} GB")
            
            with col2:
                st.markdown("**Features**")
                caps = hardware_status['capabilities']
                
                # Show feature availability with icons
                features = [
                    ("PyTorch", caps['torch_available']),
                    ("Transformers", caps['transformers_available']),
                    ("Embeddings", caps['embedding_available']),
                    ("FAIR Metadata", self.config.features['fair_metadata'])
                ]
                
                for name, available in features:
                    status = "Available" if available else "Not Available"
                    st.markdown(f"{name}: {status}")
            
            with col3:
                st.markdown("**Performance**")
                perf = hardware_status['performance']
                st.markdown(f"Workers: {perf['max_workers']}")
                st.markdown(f"Device: {perf['device'].upper()}")
                st.markdown(f"Batch Size: {perf['batch_size']}")
                
                # Cache statistics button
                if st.button("View Cache Stats", key="cache_stats_btn"):
                    self._show_cache_statistics()
    
    def _render_main_content(self):
        """Render the main content area"""
        # Check if we have results to show
        if self.session.has_results():
            # Handle automatic tab switching after prediction completion
            if st.session_state.get('prediction_just_completed', False):
                # Set the active tab to Results and clear the flag
                st.session_state.active_tab = "Results"
                st.session_state.prediction_just_completed = False
                logger.info("Automatically switched to Results tab after prediction completion")
            
            # Initialize active tab if not set
            if 'active_tab' not in st.session_state:
                st.session_state.active_tab = "New Prediction"
            
            # Create tabs for input and results with controlled selection
            # Add custom CSS for tab-like radio buttons
            st.markdown("""
            <style>
            div[data-testid="stRadio"] > div {
                display: flex;
                justify-content: center;
                gap: 0;
                margin-bottom: 1.5rem;
            }
            
            div[data-testid="stRadio"] > div > label {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 0.75rem;
                padding: 0.75rem 1.5rem;
                margin: 0 0.25rem;
                cursor: pointer;
                transition: all 0.3s ease;
                font-size: 1.1rem;
                font-weight: 500;
                color: rgba(255, 255, 255, 0.8);
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
                min-width: 140px;
                text-align: center;
            }
            
            div[data-testid="stRadio"] > div > label:hover {
                background: rgba(255, 255, 255, 0.1);
                border-color: rgba(255, 255, 255, 0.2);
                color: rgba(255, 255, 255, 0.95);
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            }
            
            div[data-testid="stRadio"] > div > label > div {
                display: none !important;
            }
            
            div[data-testid="stRadio"] > div > label[data-checked="true"] {
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.8), rgba(168, 85, 247, 0.8));
                border-color: rgba(99, 102, 241, 0.4);
                color: white;
                box-shadow: 0 4px 20px rgba(99, 102, 241, 0.3);
                transform: translateY(-2px);
            }
            
            div[data-testid="stRadio"] > div > label[data-checked="true"]:hover {
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.9), rgba(168, 85, 247, 0.9));
                transform: translateY(-2px);
            }
            </style>
            """, unsafe_allow_html=True)
            
            tab_names = ["New Prediction", "Results"]
            selected_tab = st.radio(
                "Select View:",
                tab_names,
                index=tab_names.index(st.session_state.active_tab),
                horizontal=True,
                key="tab_selector",
                label_visibility="collapsed"
            )
            
            # Update session state when user manually changes tabs
            if selected_tab != st.session_state.active_tab:
                st.session_state.active_tab = selected_tab
                logger.debug(f"User manually switched to tab: {selected_tab}")
            
            # Render content based on selected tab
            if st.session_state.active_tab == "New Prediction":
                self._render_input_area()
            else:  # Results tab
                self.results_section.render()
        else:
            # Just show input area
            self._render_input_area()
    

    def _render_input_area(self):
        """Render the input section"""
        # Show welcome message if no input yet
        if not self.session.has_valid_input():
            st.info(MESSAGES['NO_INPUT'])
        
        # Render input section
        self.input_section.render()
        
        # Advanced settings panel
        self._render_advanced_settings()
        
        # Show action button if inputs are valid
        if self.session.has_valid_input():
            self._render_action_button()
    
    def _render_advanced_settings(self):
        """Render advanced pipeline settings panel"""
        with st.expander("Advanced Settings", expanded=False):
            st.markdown("Configure advanced pipeline parameters for optimal performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Hardware & Performance**")
                
                # Device Selection
                hardware_info = self.session.get(SESSION_KEYS["HARDWARE_INFO"])
                device_options = ["Auto"]
                device_help = "Auto: Use GPU if available, fallback to CPU"
                
                if hardware_info and hardware_info.gpu_available:
                    device_options.extend(["Force GPU", "Force CPU"])
                    device_help = "Auto: Use GPU if available | Force GPU: Always use GPU | Force CPU: Always use CPU"
                else:
                    device_options.append("Force CPU")
                    device_help = "Auto/Force CPU: Use CPU (no GPU detected)"
                
                current_device = self.session.get(SESSION_KEYS["USER_DEVICE_PREFERENCE"], "Auto")
                if current_device not in device_options:
                    current_device = "Auto"
                
                device_pref = st.selectbox(
                    "Compute Device:",
                    options=device_options,
                    index=device_options.index(current_device),
                    help=device_help,
                    key="device_preference_selectbox"
                )
                
                # Store device preference in session
                device_mapping = {
                    "Auto": "auto",
                    "Force GPU": "gpu", 
                    "Force CPU": "cpu"
                }
                self.session.set(SESSION_KEYS["USER_DEVICE_PREFERENCE"], device_mapping[device_pref])
                
                # KNN Threshold
                current_knn = self.session.get(SESSION_KEYS["USER_KNN_THRESHOLD"], 100)
                knn_threshold = st.slider(
                    "Template Search Count:", 
                    min_value=10, 
                    max_value=500, 
                    value=current_knn,
                    step=10,
                    help="Number of similar proteins to find (more = slower but potentially better results)",
                    key="knn_threshold_slider"
                )
                self.session.set(SESSION_KEYS["USER_KNN_THRESHOLD"], knn_threshold)
            
            with col2:
                st.markdown("**Protein Configuration**")
                
                # Chain Selection for PDB uploads
                chain_options = ["Auto-detect"] + [chr(65+i) for i in range(26)]  # A-Z
                current_chain = self.session.get(SESSION_KEYS["USER_CHAIN_SELECTION"], "Auto-detect")
                if current_chain not in chain_options:
                    current_chain = "Auto-detect"
                
                chain_selection = st.selectbox(
                    "PDB Chain Selection:", 
                    options=chain_options, 
                    index=chain_options.index(current_chain),
                    help="Select specific protein chain from PDB file (only applies to uploaded PDB files)",
                    key="chain_selection_selectbox"
                )
                
                # Store chain selection in session
                chain_mapping = {"Auto-detect": "auto"}
                chain_mapping.update({chr(65+i): chr(65+i) for i in range(26)})
                self.session.set(SESSION_KEYS["USER_CHAIN_SELECTION"], chain_mapping[chain_selection])
                
                # Similarity Threshold
                current_similarity = self.session.get(SESSION_KEYS["USER_SIMILARITY_THRESHOLD"], 0.5)
                similarity_threshold = st.slider(
                    "Similarity Threshold:", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=current_similarity,
                    step=0.05,
                    help="Minimum similarity score for template selection (higher = more stringent)",
                    key="similarity_threshold_slider"
                )
                self.session.set(SESSION_KEYS["USER_SIMILARITY_THRESHOLD"], similarity_threshold)
            
            # Show current settings status
            st.markdown("---")
            status_col1, status_col2, status_col3 = st.columns(3)
            
            with status_col1:
                device_status = self.session.get(SESSION_KEYS["USER_DEVICE_PREFERENCE"], "auto")
                if device_status == "auto":
                    device_text = "Auto"
                elif device_status == "gpu":
                    device_text = "GPU"
                else:
                    device_text = "CPU"
                st.markdown(f"**Device:** {device_text}")
            
            with status_col2:
                knn_count = self.session.get(SESSION_KEYS["USER_KNN_THRESHOLD"], 100)
                st.markdown(f"**Templates:** {knn_count}")
            
            with status_col3:
                chain_choice = self.session.get(SESSION_KEYS["USER_CHAIN_SELECTION"], "auto")
                chain_display = "Auto" if chain_choice == "auto" else f"Chain {chain_choice}"
                st.markdown(f"**Chain:** {chain_display}")
    
    def _render_action_button(self):
        """Render the main action button"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button(
                "PREDICT POSES",
                type="primary",
                use_container_width=True,
                key="predict_button"
            ):
                self._handle_prediction()
    
    def _handle_prediction(self):
        """Handle the prediction button click"""
        logger.info("Prediction button clicked")
        
        try:
            # Create progress containers
            progress_container = st.container()
            
            with progress_container:
                progress_text = st.empty()
                progress_bar = st.empty()
                
                progress_text.info(MESSAGES['PROCESSING'])
                
                # Get input data
                molecule_data = self.session.get_molecule_data()
                protein_data = self.session.get_protein_data()
                
                logger.info(f"Input data - Molecule: {molecule_data}, Protein: {protein_data}")
                
                # Import pipeline service
                try:
                    from ..services.pipeline_service import PipelineService
                    pipeline_service = PipelineService(self.config, self.session)
                except Exception as e:
                    logger.error(f"Failed to import PipelineService: {e}", exc_info=True)
                    st.error(f"Failed to load pipeline service: {str(e)}")
                    return
                
                # Create a callback that updates the existing elements
                def update_progress(message: str, progress: int):
                    logger.debug(f"Progress update: {progress}% - {message}")
                    progress_text.info(message)
                    progress_bar.progress(progress / 100)
                
                # Run pipeline
                logger.info("Running pipeline...")
                results = pipeline_service.run_pipeline(
                    molecule_data=molecule_data,
                    protein_data=protein_data,
                    progress_callback=update_progress
                )
                
                logger.info(f"Pipeline returned: {results is not None}")
                
                if results:
                    logger.info(f"Results structure: {list(results.keys())}")
                    
                    # Store results in session
                    try:
                        poses = results.get("poses", {})
                        self.session.set("poses", poses)
                        logger.info(f"Stored {len(poses)} poses")
                        
                        template_info = results.get("template_info")
                        self.session.set("template_info", template_info)
                        logger.info(f"Stored template info: {template_info}")
                        
                        mcs_info = results.get("mcs_info")
                        self.session.set("mcs_info", mcs_info)
                        logger.info(f"Stored MCS info: {mcs_info}")
                        
                        all_ranked_poses = results.get("all_ranked_poses")
                        self.session.set("all_ranked_poses", all_ranked_poses)
                        
                        # Store template and query molecules for visualization
                        template_mol = results.get("template_mol")
                        if template_mol:
                            self.session.set("template_used", template_mol)
                            logger.info("Stored template molecule")
                        
                        query_mol = results.get("query_mol")
                        if query_mol:
                            self.session.set("query_mol", query_mol)
                            logger.info("Stored query molecule")
                        
                        # Increment pipeline runs
                        self.session.increment_pipeline_runs()
                        
                    except Exception as e:
                        logger.error(f"Failed to store results: {e}", exc_info=True)
                        st.error(f"Failed to store results: {str(e)}")
                        return
                    
                    # Clear progress and show success
                    progress_text.success(MESSAGES['SUCCESS'])
                    progress_bar.empty()
                    
                    # Set flag for automatic tab switching
                    st.session_state.prediction_just_completed = True
                    logger.info("Set prediction_just_completed flag for automatic tab switching")
                    
                    # Debugging: Check if results are actually stored
                    logger.info(f"Session has results: {self.session.has_results()}")
                    logger.info(f"Poses in session: {self.session.get('poses')}")
                    
                    # Force a rerun to show results
                    logger.info("Triggering rerun to show results...")
                    st.rerun()
                else:
                    logger.warning("Pipeline returned None")
                    progress_text.error(MESSAGES['ERROR'])
                    progress_bar.empty()
                    
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            st.error(f"Prediction failed: {str(e)}")
            
            # Show more details in an expander
            with st.expander("Error Details"):
                st.code(str(e))
                import traceback
                st.code(traceback.format_exc())
    
    def _update_progress(self, message: str, progress: int):
        """Update progress display
        
        Args:
            message: Progress message
            progress: Progress percentage (0-100)
        """
        # This method is no longer used - replaced by inline callback
        pass
    
    def _show_cache_statistics(self):
        """Display cache statistics in a modal"""
        with st.expander("Cache Performance", expanded=True):
            stats = self.cache_manager.get_cache_statistics()
            report = self.cache_manager.get_performance_report()
            
            # Show summary metrics
            overall = stats['overall']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Hit Rate", f"{overall['hit_rate']:.1f}%")
            with col2:
                st.metric("Time Saved", f"{overall['time_saved_seconds']:.1f}s")
            with col3:
                st.metric("Total Requests", overall['hits'] + overall['misses'])
            
            # Show detailed report
            st.text(report)
            
            # Cache management buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear All Caches"):
                    result = self.cache_manager.clear_all_caches()
                    st.success(f"Caches cleared in {result['clear_time']:.3f}s")
            
            with col2:
                if st.button("Export Stats"):
                    st.download_button(
                        "Download Cache Stats",
                        data=str(stats),
                        file_name="cache_stats.json",
                        mime="application/json"
                    )
    
    def _render_error_state(self, error: Exception):
        """Render error state UI
        
        Args:
            error: The exception that occurred
        """
        st.error("An error occurred while rendering the application")
        
        with st.expander("Error Details", expanded=False):
            st.code(str(error))
            
            # Offer recovery options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Refresh Page"):
                    st.rerun()
            
            with col2:
                if st.button("Clear Session"):
                    self.session.clear()
                    st.rerun()
