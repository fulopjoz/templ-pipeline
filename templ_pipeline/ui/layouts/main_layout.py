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

    def _validate_chain_input(self, chain_input: str):
        """Validate and parse chain input string

        Args:
            chain_input: User input for chain selection

        Returns:
            Validated chain selection ("auto" or list of chain IDs)
        """
        if not chain_input or chain_input.strip() == "":
            return "auto"

        # Clean input
        chain_input = chain_input.strip()

        # Special case: if user types "auto", treat as auto-detect
        if chain_input.lower() == "auto":
            return "auto"

        # Convert to uppercase for chain IDs
        chain_input = chain_input.upper()

        # Handle various formats
        chains = []

        # Split by common separators
        if "+" in chain_input:
            chains = [c.strip() for c in chain_input.split("+")]
        elif "," in chain_input:
            chains = [c.strip() for c in chain_input.split(",")]
        elif " " in chain_input:
            chains = [c.strip() for c in chain_input.split()]
        else:
            # Single chain or concatenated (e.g., "AB" -> ["A", "B"])
            if len(chain_input) == 1:
                chains = [chain_input]
            else:
                # Split each character as separate chain
                chains = list(chain_input)

        # Validate each chain ID
        valid_chains = []
        for chain in chains:
            if chain and len(chain) == 1 and chain.isalpha():
                valid_chains.append(chain)

        if not valid_chains:
            return "auto"
        elif len(valid_chains) == 1:
            return valid_chains[0]
        else:
            return valid_chains

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
                hw = hardware_status["hardware"]
                st.markdown(f"CPU: {hw['cpu_cores']} cores")
                st.markdown(f"RAM: {hw['ram_gb']} GB")
                st.markdown(f"GPU: {hw['gpu']}")
                if hw["gpu"] != "Not available":
                    st.markdown(f"VRAM: {hw['gpu_memory_gb']} GB")

            with col2:
                st.markdown("**Features**")
                caps = hardware_status["capabilities"]

                # Show feature availability with icons
                features = [
                    ("PyTorch", caps["torch_available"]),
                    ("Transformers", caps["transformers_available"]),
                    ("Embeddings", caps["embedding_available"]),
                    ("FAIR Metadata", self.config.features["fair_metadata"]),
                ]

                for name, available in features:
                    status = "Available" if available else "Not Available"
                    st.markdown(f"{name}: {status}")

            with col3:
                st.markdown("**Performance**")
                perf = hardware_status["performance"]
                st.markdown(f"Workers: {perf['max_workers']}")
                st.markdown(f"Device: {perf['device'].upper()}")
                st.markdown(f"Batch Size: {perf['batch_size']}")

                # Simple, functional cleanup button
                if st.button("Clean Up Application", key="cleanup_btn", help="Clear caches and free up memory"):
                    self._cleanup_application()

    def _cleanup_application(self):
        """Perform comprehensive application cleanup with user feedback"""
        cleanup_results = []
        
        try:
            # Clear Streamlit caches
            cache_result = self.cache_manager.clear_all_caches()
            cleanup_results.append(f"Cleared Streamlit caches ({cache_result['clear_time']:.2f}s)")
        except Exception as e:
            cleanup_results.append(f"Cache clearing failed: {str(e)}")
            logger.error(f"Cache clearing failed: {e}")

        try:
            # Memory optimization
            memory_result = self.session.memory_manager.optimize_memory()
            if memory_result.get('memory_saved_mb', 0) > 0:
                cleanup_results.append(f"Freed {memory_result['memory_saved_mb']:.1f}MB memory")
            else:
                cleanup_results.append("Memory already optimized")
        except Exception as e:
            cleanup_results.append(f"Memory optimization failed: {str(e)}")
            logger.error(f"Memory optimization failed: {e}")

        try:
            # Clear molecular cache
            self.session.memory_manager.clear_cache()
            cleanup_results.append("Cleared molecular data cache")
        except Exception as e:
            cleanup_results.append(f"Molecular cache clearing failed: {str(e)}")
            logger.error(f"Molecular cache clearing failed: {e}")

        try:
            # Force garbage collection
            import gc
            collected = gc.collect()
            if collected > 0:
                cleanup_results.append(f"Garbage collected {collected} objects")
            else:
                cleanup_results.append("No garbage to collect")
        except Exception as e:
            cleanup_results.append(f"Garbage collection failed: {str(e)}")
            logger.error(f"Garbage collection failed: {e}")

        # Show results to user
        if cleanup_results:
            st.success("Application Cleanup Complete!")
            for result in cleanup_results:
                st.write(result)
            
            # Show memory stats after cleanup
            memory_stats = self.session.memory_manager.get_memory_stats()
            st.info(f"Current memory usage: {memory_stats['cache_size_mb']:.1f}MB")
        else:
            st.warning("No cleanup actions were performed")

    def _render_main_content(self):
        """Render the main content area using native Streamlit tabs"""
        # Check if we have results to show
        if self.session.has_results():
            # Handle automatic tab switching after prediction completion
            active_tab_index = (
                1 if st.session_state.get("prediction_just_completed", False) else 0
            )
            if st.session_state.get("prediction_just_completed", False):
                st.session_state.prediction_just_completed = False
                logger.info(
                    "Automatically switched to Results tab after prediction completion"
                )

            # Use native Streamlit tabs - eliminates button visibility issues
            tab1, tab2 = st.tabs(["New Prediction", "Results"])

            with tab1:
                self._render_input_area()

            with tab2:
                self.results_section.render()
        else:
            # Just show input area
            self._render_input_area()

    def _render_input_area(self):
        """Render the input section"""
        # Render input section first
        self.input_section.render()

        # Show welcome message if no input yet (after inputs are rendered so validation works)
        if not self.session.has_valid_input():
            st.info(MESSAGES["NO_INPUT"])

        # Show action button if inputs are valid - placed closer to inputs
        if self.session.has_valid_input():
            # Add some spacing before the button
            st.markdown("---")
            self._render_action_button()

        # Advanced settings panel (moved after action button)
        self._render_advanced_settings()

    def _render_advanced_settings(self):
        """Render advanced pipeline settings panel"""
        # Simplified CSS for elegant, minimalistic styling
        st.markdown(
            """
        <style>
        /* Simplified theme-aware styling */
        
        /* Button styling with subtle shadows */
        .stButton > button {
            border-radius: 6px !important;
            font-weight: 500 !important;
            transition: all 0.15s ease !important;
        }
        
        /* Primary button with reduced shine */
        .stButton > button[kind="primary"] {
            font-weight: 600 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
        }
        
        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Slider styling - elegant and neutral */
        .stSlider {
            padding: 1rem 0 !important;
        }
        
        /* Radio buttons - simplified */
        div[data-testid="stRadio"] > div {
            background-color: var(--bg-light) !important;
            border: 2px solid var(--border-light) !important;
            border-radius: 8px !important;
            padding: 8px !important;
        }
        
        div[data-testid="stRadio"] > div > label {
            background-color: var(--bg-white) !important;
            border: 2px solid var(--border-light) !important;
            border-radius: 6px !important;
            padding: 8px 16px !important;
            margin: 4px !important;
            color: var(--text-color) !important;
            font-weight: 500 !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
        }
        
        div[data-testid="stRadio"] > div > label:hover {
            background-color: var(--button-hover-bg) !important;
            border-color: var(--border-hover) !important;
        }
        
        div[data-testid="stRadio"] > div > label[data-checked="true"] {
            background-color: var(--primary-color) !important;
            border-color: var(--primary-color) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
        }
        
        /* Hide radio circles */
        div[data-testid="stRadio"] > div > label > div[data-testid="stMarkdownContainer"] > div > input {
            display: none !important;
        }
        
        /* Selectbox styling - Theme-aware */
        .stSelectbox > div > div {
            background-color: var(--bg-white) !important;
            border: 2px solid var(--border-light) !important;
            border-radius: 8px !important;
            color: var(--text-color) !important;
            font-weight: 500 !important;
            box-shadow: 0 2px 4px var(--button-shadow) !important;
        }
        
        .stSelectbox > div > div:hover {
            border-color: var(--border-hover) !important;
        }
        
        /* Text input styling - Theme-aware */
        .stTextInput > div > div > input {
            color: var(--text-color) !important;
            background-color: var(--bg-white) !important;
            border: 2px solid var(--border-light) !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            padding: 12px !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: var(--primary-color) !important;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
        }
        
        /* File uploader styling - Theme-aware */
        .stFileUploader > div {
            border: 2px dashed var(--border-light) !important;
            border-radius: 8px !important;
            background-color: var(--bg-light) !important;
            padding: 20px !important;
        }
        
        .stFileUploader > div > div > button {
            color: var(--primary-color) !important;
            background-color: var(--bg-white) !important;
            border: 2px solid var(--primary-color) !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
        }
        
        /* Slider styling - Theme-aware */
        
        /* Expander styling - minimal */
        .streamlit-expanderHeader {
            border-radius: 6px !important;
            font-weight: 500 !important;
        }
        
        /* Clean button text */
        .stButton > button {
            -webkit-font-smoothing: antialiased !important;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        with st.expander("Advanced Settings", expanded=False):
            st.markdown(
                "Configure advanced pipeline parameters for optimal performance"
            )

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

                    # Show GPU installation hint if GPUs are available
                    try:
                        import subprocess

                        result = subprocess.run(
                            ["nvidia-smi"], capture_output=True, text=True
                        )
                        if result.returncode == 0:
                            st.info(
                                "**GPU Available**: Install PyTorch with CUDA to enable GPU acceleration"
                            )
                    except:
                        pass

                current_device = self.session.get(
                    SESSION_KEYS["USER_DEVICE_PREFERENCE"], "auto"
                )
                # Map session values to display values for device selection
                device_display_mapping = {
                    "auto": "Auto",
                    "gpu": "Force GPU",
                    "cpu": "Force CPU",
                }
                display_device = device_display_mapping.get(current_device, "Auto")

                device_pref = st.selectbox(
                    "Compute Device:",
                    options=device_options,
                    index=(
                        device_options.index(display_device)
                        if display_device in device_options
                        else 0
                    ),
                    help=device_help,
                    key="device_preference_selectbox",
                )

                # Store device preference in session only if changed
                device_mapping = {
                    "Auto": "auto",
                    "Force GPU": "gpu",
                    "Force CPU": "cpu",
                }
                new_device_value = device_mapping[device_pref]
                if (
                    self.session.get(SESSION_KEYS["USER_DEVICE_PREFERENCE"])
                    != new_device_value
                ):
                    self.session.set(
                        SESSION_KEYS["USER_DEVICE_PREFERENCE"], new_device_value
                    )

                # KNN Threshold - simplified session state management
                st.markdown("**Template Search Count**")
                knn_threshold = st.slider(
                    "Number of templates to search:",
                    min_value=10,
                    max_value=500,
                    value=self.session.get(SESSION_KEYS["USER_KNN_THRESHOLD"], 100),
                    step=10,
                    help="Recommended: 100-200 for balanced speed/quality. More templates = slower but potentially better results.",
                    key="knn_threshold_slider",
                    on_change=lambda: self.session.set(
                        SESSION_KEYS["USER_KNN_THRESHOLD"],
                        st.session_state.knn_threshold_slider,
                    ),
                )
                # Update session when value changes
                self.session.set(SESSION_KEYS["USER_KNN_THRESHOLD"], knn_threshold)

            with col2:
                st.markdown("**Protein Configuration**")

                # Chain Selection for PDB uploads - flexible input
                current_chains = self.session.get(
                    SESSION_KEYS["USER_CHAIN_SELECTION"], "auto"
                )

                # Display current chain selection in user-friendly format
                if current_chains == "auto":
                    display_value = ""
                elif isinstance(current_chains, list):
                    display_value = "+".join(current_chains)
                else:
                    display_value = str(current_chains)

                def update_chain_selection():
                    """Callback to validate and update chain selection"""
                    chain_input = st.session_state.chain_input_field.strip()
                    validated_chains = self._validate_chain_input(chain_input)
                    self.session.set(
                        SESSION_KEYS["USER_CHAIN_SELECTION"], validated_chains
                    )

                chain_input = st.text_input(
                    "PDB Chain Selection:",
                    value=display_value,
                    help="Enter chain ID(s): A, B, A+B, A,B, AB, or leave empty for auto-detect. Supports multiple chains for concatenated embeddings.",
                    key="chain_input_field",
                    on_change=update_chain_selection,
                    placeholder="e.g., A or A+B or A,B (empty = auto-detect)",
                )

                # Update session state immediately
                validated_chains = self._validate_chain_input(chain_input)
                if (
                    self.session.get(SESSION_KEYS["USER_CHAIN_SELECTION"])
                    != validated_chains
                ):
                    self.session.set(
                        SESSION_KEYS["USER_CHAIN_SELECTION"], validated_chains
                    )

                # Similarity Threshold - simplified session state management
                st.markdown("**Similarity Threshold**")
                similarity_threshold = st.slider(
                    "Minimum similarity for templates:",
                    min_value=0.0,
                    max_value=1.0,
                    value=self.session.get(
                        SESSION_KEYS["USER_SIMILARITY_THRESHOLD"], 0.5
                    ),
                    step=0.05,
                    help="Recommended: 0.3-0.7 for good results. Higher = more stringent template selection.",
                    key="similarity_threshold_slider",
                    on_change=lambda: self.session.set(
                        SESSION_KEYS["USER_SIMILARITY_THRESHOLD"],
                        st.session_state.similarity_threshold_slider,
                    ),
                )
                # Update session when value changes
                self.session.set(
                    SESSION_KEYS["USER_SIMILARITY_THRESHOLD"], similarity_threshold
                )

            # Show current settings status with better formatting
            st.markdown("---")
            st.markdown("**Current Configuration**")

            status_col1, status_col2, status_col3 = st.columns(3)

            with status_col1:
                device_status = self.session.get(
                    SESSION_KEYS["USER_DEVICE_PREFERENCE"], "auto"
                )
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
                chain_choice = self.session.get(
                    SESSION_KEYS["USER_CHAIN_SELECTION"], "auto"
                )
                if chain_choice == "auto":
                    chain_display = "Auto"
                elif isinstance(chain_choice, list) and len(chain_choice) > 1:
                    chain_display = f"Chains {'+'.join(chain_choice)}"
                else:
                    chain_display = f"Chain {chain_choice if isinstance(chain_choice, str) else chain_choice[0]}"
                st.markdown(f"**Chain:** {chain_display}")

    def _render_action_button(self):
        """Render the main action button with loading states"""
        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            # Check if prediction is currently running
            prediction_running = self.session.get(
                SESSION_KEYS.get("PREDICTION_RUNNING", "prediction_running"), False
            )

            if prediction_running:
                # Show loading button when prediction is running
                st.button(
                    "Predicting... Please Wait",
                    type="primary",
                    use_container_width=True,
                    disabled=True,
                    key="predict_poses_button_loading",
                )
                # Show spinner
                with st.spinner("Running prediction pipeline..."):
                    st.empty()  # Placeholder to keep spinner visible
            else:
                # Normal button when not running
                if st.button(
                    "Predict Poses",
                    type="primary",
                    use_container_width=True,
                    key="predict_poses_button",
                ):
                    self._handle_prediction()

    def _handle_prediction(self):
        """Handle the prediction button click"""
        logger.info("Prediction button clicked")

        try:
            # Set prediction running state
            self.session.set(
                SESSION_KEYS.get("PREDICTION_RUNNING", "prediction_running"), True
            )

            # Validate inputs first
            if not self.session.has_valid_input():
                st.error(
                    "Please provide both a molecule (SMILES or file) and protein before running prediction"
                )
                self.session.set(
                    SESSION_KEYS.get("PREDICTION_RUNNING", "prediction_running"), False
                )
                return

            # Create progress containers
            progress_container = st.container()

            with progress_container:
                progress_text = st.empty()
                progress_bar = st.empty()

                progress_text.info("Starting prediction pipeline...")
                progress_bar.progress(0.1)

                # Get input data
                molecule_data = self.session.get_molecule_data()
                protein_data = self.session.get_protein_data()

                logger.info(
                    f"Input data - Molecule: {molecule_data}, Protein: {protein_data}"
                )

                # Check if we have the required data
                if not molecule_data:
                    st.error("No valid molecule data found")
                    return
                if not protein_data:
                    st.error("No valid protein data found")
                    return

                # Import pipeline service
                try:
                    from ..services.pipeline_service import PipelineService

                    pipeline_service = PipelineService(self.config, self.session)
                    st.success("Pipeline service loaded successfully")
                except Exception as e:
                    logger.error(
                        f"Failed to import PipelineService: {e}", exc_info=True
                    )
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
                    progress_callback=update_progress,
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
                    progress_text.success(MESSAGES["SUCCESS"])
                    progress_bar.empty()

                    # Set flag for automatic tab switching
                    st.session_state.prediction_just_completed = True
                    logger.info(
                        "Set prediction_just_completed flag for automatic tab switching"
                    )

                    # Clear prediction running state
                    self.session.set(
                        SESSION_KEYS.get("PREDICTION_RUNNING", "prediction_running"),
                        False,
                    )

                    # Debugging: Check if results are actually stored
                    logger.info(f"Session has results: {self.session.has_results()}")
                    logger.info(f"Poses in session: {self.session.get('poses')}")

                    # Force a rerun to show results
                    logger.info("Triggering rerun to show results...")
                    st.rerun()
                else:
                    logger.warning("Pipeline returned None")
                    progress_text.error(MESSAGES["ERROR"])
                    progress_bar.empty()
                    # Clear prediction running state
                    self.session.set(
                        SESSION_KEYS.get("PREDICTION_RUNNING", "prediction_running"),
                        False,
                    )

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            # Clear prediction running state on error
            self.session.set(
                SESSION_KEYS.get("PREDICTION_RUNNING", "prediction_running"), False
            )
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
                if st.button("Clean Up & Restart"):
                    self._cleanup_application()
                    st.rerun()
