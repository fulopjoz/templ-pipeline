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
from ..services.pipeline_service import PipelineService
from ..utils.workspace_integration import get_workspace_integration

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

        # Initialize pipeline service (workspace panel removed for a cleaner UI)
        self.pipeline_service = PipelineService(config, session)

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

            # Show about section if configured
            if self.config.ui_settings.get("show_system_status", True):
                self._render_about_section()

            # Main content area
            self._render_main_content()

            # Workspace panel removed per UX simplification

            # Status bar at bottom
            if self.config.ui_settings.get("show_status_bar", True):
                render_status_bar(self.session)

            # End performance monitoring
            render_time = self.performance_monitor.end_render("main_layout")
            logger.debug(f"Main layout rendered in {render_time:.3f}s")

        except Exception as e:
            logger.error(f"Error rendering main layout: {e}", exc_info=True)
            self._render_error_state(e)

    def _render_about_section(self):
        """Render About TEMPL section"""
        with st.expander("About TEMPL", expanded=False):
            # Main description
            st.markdown("""
            **TEMPL** is a template-based method for rapid protein-ligand pose prediction that leverages 
            ligand similarity and template superposition for pose generation within known chemical space.
            """)
            
            st.markdown("---")
            
            # Two-column layout for main content
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Key Features**")
                st.markdown("""
                * **MCS-driven alignment** - Uses maximal common substructure matching
                * **Constrained embedding** - ETKDG v3 conformer generation  
                * **Shape scoring** - Pharmacophore-based pose selection
                * **Built-in benchmarks** - Polaris and time-split PDBbind
                * **CPU-optimized** - Optional GPU acceleration available
                """)
                
            
            with col2:
                st.markdown("**How It Works**")
                st.markdown("""
                1. **Template Matching** - Find similar protein-ligand complexes
                2. **MCS Detection** - Identify common substructures with reference ligands
                3. **Conformer Generation** - Create poses using constrained embedding
                4. **Pose Ranking** - Score using shape and pharmacophore alignment
                5. **Limitations**: Novel scaffolds, allosteric sites, insufficient templates
                """)
            
            st.markdown("---")
            
            # Compact system status at bottom
            st.markdown("**System Status**")
            hardware_status = self.hardware_manager.get_status_summary()
            hw = hardware_status["hardware"]
            perf = hardware_status["performance"]
            
            status_col1, status_col2, status_col3 = st.columns(3)
            
            with status_col1:
                st.markdown(f"**Hardware:** {hw['cpu_cores']} cores, {hw['ram_gb']} GB RAM")
                if hw["gpu"] != "Not available":
                    st.markdown(f"**GPU:** {hw['gpu_memory_gb']} GB VRAM")
            
            with status_col2:
                caps = hardware_status["capabilities"]
                embedding_status = "✓" if caps["embedding_available"] else "✗"
                st.markdown(f"**Embeddings:** {embedding_status}")
                st.markdown(f"**Device:** {perf['device'].upper()}")
            
            with status_col3:
                st.markdown(f"**Workers:** {perf['max_workers']}")
                # Subtle cleanup button
                if st.button("Clean Up", key="cleanup_btn", help="Clear caches and free up memory"):
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

    # Workspace status UI fully removed

    def _render_main_content(self):
        """Render the main content area using native Streamlit tabs"""
        # Check if we have results to show
        has_results = self.session.has_results()
        logger.info(f"Main content rendering - has_results: {has_results}")
        
        if has_results:
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
            logger.info("No results found - only showing input area")
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
            st.markdown("##### Pipeline Configuration")
            
            # GPU/Device selection
            device_pref = st.selectbox(
                "Device Preference", 
                ["auto", "gpu", "cpu"], 
                index=0,
                help="Choose compute device. Auto will use GPU if available."
            )
            
            # KNN threshold
            knn_threshold = st.slider(
                "Template Selection (k-NN)", 
                10, 500, 100,
                help="Number of similar templates to consider"
            )
            
            # Similarity threshold
            similarity_threshold = st.slider(
                "Similarity Threshold", 
                0.0, 1.0, 0.9, 0.05,
                help="Minimum similarity score for template selection"
            )
            
            # Chain selection
            chain_selection = st.text_input(
                "Chain Selection",
                value="auto",
                help="Specify protein chain(s) to use (e.g., 'A', 'B', 'AB') or 'auto' for automatic selection"
            )
            

            
            # Removed Debug Mode and FAIR metadata toggles for a cleaner UI

            # Store settings in session
            self.session.set(SESSION_KEYS["USER_DEVICE_PREFERENCE"], device_pref)
            self.session.set(SESSION_KEYS["USER_KNN_THRESHOLD"], knn_threshold)
            self.session.set(SESSION_KEYS["USER_SIMILARITY_THRESHOLD"], similarity_threshold)
            self.session.set(SESSION_KEYS["USER_CHAIN_SELECTION"], chain_selection)
            # Ensure flags are off
            self.session.set(SESSION_KEYS["SHOW_FAIR_PANEL"], False)
            st.session_state["debug_mode"] = False
            
            # Validation callback for chain selection
            def update_chain_selection():
                """Update chain selection with validation"""
                if chain_selection != "auto":
                    validated_chains = self._validate_chain_input(chain_selection)
                    if validated_chains != chain_selection:
                        st.warning(f"Chain selection normalized to: {validated_chains}")
                        self.session.set(SESSION_KEYS["USER_CHAIN_SELECTION"], validated_chains)
                else:
                    self.session.set(SESSION_KEYS["USER_CHAIN_SELECTION"], "auto")
            
            # Validate chain selection
            update_chain_selection()
            

            
            # Performance monitoring
            if hasattr(self, 'performance_monitor'):
                stats = self.performance_monitor.get_statistics()
                if stats:
                    st.markdown("##### Performance Statistics")
                    for component, component_stats in stats.items():
                        if component_stats.get('count', 0) > 0:
                            avg_time = component_stats.get('average', 0)
                            st.write(f"**{component}:** {avg_time:.2f}s avg")
            


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
                        logger.info(f"Storing {len(poses)} poses in session")
                        logger.debug(f"Poses keys: {list(poses.keys()) if poses else 'None'}")
                        
                        # Validate poses before storing
                        if poses and isinstance(poses, dict):
                            # Check if poses contain valid data
                            valid_poses = {}
                            for method, pose_data in poses.items():
                                if isinstance(pose_data, tuple) and len(pose_data) == 2:
                                    mol, scores = pose_data
                                    if hasattr(mol, 'ToBinary') and isinstance(scores, dict):
                                        valid_poses[method] = pose_data
                                    else:
                                        logger.warning(f"Invalid pose data for method {method}: mol={type(mol)}, scores={type(scores)}")
                                else:
                                    logger.warning(f"Invalid pose structure for method {method}: {type(pose_data)}")
                            
                            if valid_poses:
                                self.session.set(SESSION_KEYS["POSES"], valid_poses)
                                logger.info(f"Stored {len(valid_poses)} valid poses")
                            else:
                                logger.error("No valid poses found to store")
                        else:
                            logger.error(f"Invalid poses data structure: {type(poses)}")
                            self.session.set(SESSION_KEYS["POSES"], poses)  # Store anyway for debugging

                        template_info = results.get("template_info")
                        self.session.set(SESSION_KEYS["TEMPLATE_INFO"], template_info)
                        logger.info(f"Stored template info: {template_info}")
                        logger.info(f"Template info type: {type(template_info)}")

                        mcs_info = results.get("mcs_info")
                        # Fallback: build MCS info from template_info if missing
                        if not mcs_info:
                            try:
                                tinfo = template_info or {}
                                mcs_smarts_candidate = None
                                if isinstance(tinfo, dict):
                                    mcs_smarts_candidate = tinfo.get("mcs_smarts")
                                # Also check raw results in case it exists there
                                if not mcs_smarts_candidate:
                                    mcs_smarts_candidate = results.get("mcs_smarts")
                                # Final fallback to structured details
                                if not mcs_smarts_candidate and isinstance(results.get("mcs_details"), dict):
                                    mcs_smarts_candidate = results["mcs_details"].get("smarts")
                                if mcs_smarts_candidate and isinstance(mcs_smarts_candidate, str) and mcs_smarts_candidate.strip():
                                    # Use pipeline service processor to normalize
                                    processed = self.pipeline_service._process_mcs_info(mcs_smarts_candidate, template_info)
                                    if processed:
                                        mcs_info = processed
                                        logger.info("Reconstructed MCS info from template info fallback")
                            except Exception as mcs_fallback_err:
                                logger.warning(f"Failed to reconstruct MCS info: {mcs_fallback_err}")

                        self.session.set(SESSION_KEYS["MCS_INFO"], mcs_info)
                        logger.info(f"Stored MCS info: {mcs_info}")
                        logger.info(f"MCS info type: {type(mcs_info)}")

                        all_ranked_poses = results.get("all_ranked_poses")
                        logger.info(f"DEBUG: Storing all_ranked_poses: type={type(all_ranked_poses)}, length={len(all_ranked_poses) if hasattr(all_ranked_poses, '__len__') else 'N/A'}")
                        self.session.set(SESSION_KEYS["ALL_RANKED_POSES"], all_ranked_poses)
                        
                        # Verify storage immediately
                        stored_all_ranked = self.session.get(SESSION_KEYS["ALL_RANKED_POSES"])
                        logger.info(f"DEBUG: Verified stored all_ranked_poses: type={type(stored_all_ranked)}, length={len(stored_all_ranked) if hasattr(stored_all_ranked, '__len__') else 'N/A'}")

                        # Store template and query molecules for visualization
                        template_mol = results.get("template_mol")
                        if template_mol:
                            self.session.set(SESSION_KEYS["TEMPLATE_USED"], template_mol)
                            logger.info("Stored template molecule")
                            logger.info(f"Template molecule type: {type(template_mol)}")
                        else:
                            logger.warning("No template molecule in results")

                        query_mol = results.get("query_mol")
                        if query_mol:
                            self.session.set(SESSION_KEYS["QUERY_MOL"], query_mol)
                            logger.info("Stored query molecule")
                        else:
                            logger.warning("No query molecule in results")

                        # Increment pipeline runs
                        self.session.increment_pipeline_runs()
                        
                        # Verify results are stored properly
                        stored_poses = self.session.get(SESSION_KEYS["POSES"])
                        has_results = self.session.has_results()
                        logger.info(f"Verification - Has results: {has_results}")
                        logger.info(f"Verification - Stored poses: {type(stored_poses)} with {len(stored_poses) if isinstance(stored_poses, dict) else 'N/A'} entries")
                        
                        # Additional debugging
                        logger.info(f"Verification - Session state poses: {type(st.session_state.get(SESSION_KEYS['POSES']))}")
                        logger.info(f"Verification - Session state poses length: {len(st.session_state.get(SESSION_KEYS['POSES'], {}))}")
                        logger.info(f"Verification - Best poses refs in session: {'best_poses_refs' in st.session_state}")
                        if "best_poses_refs" in st.session_state:
                            refs = st.session_state["best_poses_refs"]
                            logger.info(f"Verification - Best poses refs: {type(refs)}, length: {len(refs) if isinstance(refs, dict) else 'N/A'}")
                        
                        if not has_results:
                            logger.error("CRITICAL: Session shows no results despite successful storage attempt")
                            # Force store in session state directly as fallback
                            st.session_state[SESSION_KEYS["POSES"]] = poses
                            logger.info("Forced direct storage in session state as fallback")
                            
                            # Try to verify again
                            has_results_after_force = self.session.has_results()
                            logger.info(f"Has results after forced storage: {has_results_after_force}")

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
        # Deprecated: replaced by inline progress callback. Remove if not reintroduced.
        return None

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
