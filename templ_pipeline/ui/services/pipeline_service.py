"""
Enhanced Pipeline Service for TEMPL Pipeline

Handles pipeline execution with proper error reporting and embedding generation.
"""

import logging
import time
import asyncio
import concurrent.futures
from typing import Dict, Any, Optional, Callable, Tuple
from pathlib import Path
import sys
import os

from ..config.settings import AppConfig
from ..core.session_manager import SessionManager
from ..config.constants import SESSION_KEYS

logger = logging.getLogger(__name__)


class PipelineService:
    """Service for managing pipeline execution with enhanced error handling and unified workspace management"""

    def __init__(self, config: AppConfig, session: SessionManager):
        """Initialize pipeline service

        Args:
            config: Application configuration
            session: Session manager
        """
        self.config = config
        self.session = session
        self.pipeline = None
        self.workspace_manager = None
        
        # Initialize workspace manager
        self._initialize_workspace_manager()

    def _initialize_workspace_manager(self):
        """Initialize unified workspace manager for the session"""
        try:
            # Import workspace manager
            from templ_pipeline.core.workspace_manager import (
                WorkspaceManager, 
                WorkspaceConfig
            )
            
            # Create workspace configuration
            workspace_config = WorkspaceConfig(
                base_dir=self.config.paths.get("workspace_dir", "workspace"),
                auto_cleanup=True,
                temp_retention_hours=24
            )
            
            # Generate unique run ID for this session
            session_id = self.session.get("session_id", None)
            if not session_id:
                import uuid
                session_id = str(uuid.uuid4())[:8]
                self.session.set("session_id", session_id)
            
            # Create workspace manager
            self.workspace_manager = WorkspaceManager(
                run_id=f"ui_{session_id}",
                config=workspace_config
            )
            
            # Store workspace info in session
            self.session.set("workspace_run_id", self.workspace_manager.run_id)
            self.session.set("workspace_dir", str(self.workspace_manager.run_dir))
            
            logger.info(f"Initialized workspace manager: {self.workspace_manager.run_dir}")
            
        except ImportError:
            logger.warning("WorkspaceManager not available, using legacy file handling")
            self.workspace_manager = None
        except Exception as e:
            logger.warning(f"Failed to initialize workspace manager: {e}")
            self.workspace_manager = None

    def get_workspace_manager(self):
        """Get the workspace manager instance"""
        return self.workspace_manager

    def _configure_device_preference(self, device_pref: str):
        """Configure device preference for embedding generation

        Args:
            device_pref: User device preference ("auto", "gpu", "cpu")
        """
        # Set environment variable that embedding functions can check
        if device_pref == "gpu":
            os.environ["TEMPL_FORCE_DEVICE"] = "cuda"
            logger.info("User forced GPU usage - setting TEMPL_FORCE_DEVICE=cuda")
        elif device_pref == "cpu":
            os.environ["TEMPL_FORCE_DEVICE"] = "cpu"
            logger.info("User forced CPU usage - setting TEMPL_FORCE_DEVICE=cpu")
        else:
            # Auto mode - let embedding functions decide
            if "TEMPL_FORCE_DEVICE" in os.environ:
                del os.environ["TEMPL_FORCE_DEVICE"]
            logger.info(
                "Auto device mode - letting embedding functions choose optimal device"
            )

    def _resolve_chain_selection(self, user_chain_selection: str):
        """Resolve chain selection for PDB processing

        Args:
            user_chain_selection: User chain preference ("auto" or specific chain like "A")

        Returns:
            Chain ID to use or None for auto-detection
        """
        if user_chain_selection == "auto":
            return None
        else:
            return user_chain_selection

    def run_pipeline(
        self,
        molecule_data: Dict[str, Any],
        protein_data: Dict[str, Any],
        progress_callback: Optional[Callable] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run the TEMPL pipeline with enhanced error handling

        Args:
            molecule_data: Molecule input data
            protein_data: Protein input data
            progress_callback: Optional callback for progress updates

        Returns:
            Results dictionary or None on failure
        """
        try:
            # Log inputs
            logger.info("Starting pipeline execution")
            logger.info(f"Molecule data: {molecule_data.get('input_smiles', 'N/A')}")
            logger.info(f"Protein data: {protein_data}")

            # Get user settings from session
            user_device_pref = self.session.get(
                SESSION_KEYS["USER_DEVICE_PREFERENCE"], "auto"
            )
            user_knn_threshold = self.session.get(
                SESSION_KEYS["USER_KNN_THRESHOLD"], 100
            )
            user_chain_selection = self.session.get(
                SESSION_KEYS["USER_CHAIN_SELECTION"], "auto"
            )
            user_similarity_threshold = self.session.get(
                SESSION_KEYS["USER_SIMILARITY_THRESHOLD"], 0.5
            )

            logger.info(
                f"User settings - Device: {user_device_pref}, KNN: {user_knn_threshold}, Chain: {user_chain_selection}, Similarity: {user_similarity_threshold}"
            )

            # Configure device preference for embedding generation
            self._configure_device_preference(user_device_pref)

            # Progress updates
            if progress_callback:
                progress_callback("Initializing pipeline...", 10)

            # Import the actual pipeline
            logger.info("Attempting to load real TEMPL pipeline...")

            # Add parent directory to path for imports
            parent_dir = Path(__file__).parent.parent.parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))

            from templ_pipeline.core.pipeline import TEMPLPipeline

            if progress_callback:
                progress_callback("Loading TEMPL pipeline...", 20)

            # Initialize pipeline if not already done
            if self.pipeline is None:
                # Always use simplified initialization to avoid Streamlit errors
                # Previous workspace manager integration was causing parameter issues
                self.pipeline = TEMPLPipeline(
                    embedding_path=None,  # Auto-detect
                    output_dir=self.config.paths.get("output_dir", "temp"),
                )
                logger.info("Pipeline initialized with simplified configuration")

            # Prepare inputs
            smiles = molecule_data.get("input_smiles")
            pdb_id = protein_data.get("pdb_id")
            pdb_file = protein_data.get("file_path")
            custom_templates = molecule_data.get("custom_templates")

            # Check if PDB ID exists in database first
            if pdb_id and not pdb_file:
                embedding_manager = self.pipeline._get_embedding_manager()
                if not embedding_manager.has_embedding(pdb_id):
                    # PDB ID not found - provide clear error message
                    error_msg = f"PDB ID '{pdb_id.upper()}' not found in database"

                    # Safe streamlit call
                    try:
                        import streamlit as st

                        if hasattr(st, "error"):
                            st.error(f" {error_msg}")
                            st.info(
                                f"**Suggestion**: Upload the PDB file directly using the 'Upload File' option"
                            )

                            # Show help for getting PDB files
                            with st.expander("How to get PDB files", expanded=True):
                                st.markdown(
                                    f"""
                                You can download PDB files from:
                                
                                1. **RCSB PDB**: [https://www.rcsb.org/structure/{pdb_id.upper()}](https://www.rcsb.org/structure/{pdb_id.upper()})
                                2. **PDBe**: [https://www.ebi.ac.uk/pdbe/entry/pdb/{pdb_id.lower()}](https://www.ebi.ac.uk/pdbe/entry/pdb/{pdb_id.lower()})
                                
                                Click "Download Files" → "PDB Format" on either site.
                                """
                                )
                    except:
                        logger.error(error_msg)

                    return None

            if progress_callback:
                if pdb_file:
                    progress_callback("Generating protein embedding from file...", 30)
                else:
                    progress_callback("Loading protein embedding from database...", 30)

            # Handle different input scenarios with user settings
            if custom_templates:
                # MCS-only workflow with custom templates
                results = self._run_custom_template_pipeline(
                    smiles,
                    custom_templates,
                    progress_callback,
                    user_settings={
                        "device_pref": user_device_pref,
                        "knn_threshold": user_knn_threshold,
                        "chain_selection": user_chain_selection,
                        "similarity_threshold": user_similarity_threshold,
                    },
                )
            elif pdb_file:
                # Full pipeline with uploaded PDB file - generate embedding and search
                results = self._run_uploaded_pdb_pipeline(
                    smiles,
                    pdb_file,
                    progress_callback,
                    user_settings={
                        "device_pref": user_device_pref,
                        "knn_threshold": user_knn_threshold,
                        "chain_selection": user_chain_selection,
                        "similarity_threshold": user_similarity_threshold,
                    },
                )
            elif pdb_id:
                # Full pipeline with PDB ID from database
                results = self._run_pdb_id_pipeline(
                    smiles,
                    pdb_id,
                    progress_callback,
                    user_settings={
                        "device_pref": user_device_pref,
                        "knn_threshold": user_knn_threshold,
                        "chain_selection": user_chain_selection,
                        "similarity_threshold": user_similarity_threshold,
                    },
                )
            else:
                # Safe streamlit call
                try:
                    import streamlit as st

                    if hasattr(st, "error"):
                        st.error("No valid protein input provided")
                except:
                    logger.error("No valid protein input provided")
                return None

            if progress_callback:
                progress_callback("Pipeline complete!", 100)

            logger.info("Pipeline completed successfully")
            return results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)

            # Provide user-friendly error messages
            error_str = str(e)

            # Safe streamlit calls
            try:
                import streamlit as st

                if hasattr(st, "error"):
                    if "not found in database" in error_str:
                        # Already handled above
                        pass
                    elif "No templates found" in error_str:
                        st.error("No similar templates found in database")
                        st.info(
                            "Try uploading custom template molecules or using a different protein"
                        )
                    elif "embedding" in error_str.lower():
                        st.error("Failed to generate protein embedding")
                        st.info(
                            "Check that the PDB file is valid and contains protein chains"
                        )
                    else:
                        st.error(f"Pipeline error: {error_str}")
            except:
                logger.error(f"Pipeline error: {error_str}")

            return None

    def _run_custom_template_pipeline(
        self,
        smiles: str,
        custom_templates: list,
        progress_callback: Optional[Callable],
        user_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run pipeline with custom template molecules"""
        if progress_callback:
            progress_callback("Processing custom templates...", 40)

        # For custom templates, we need to use the pipeline's individual methods
        # since run_full_pipeline doesn't support custom templates directly
        logger.info("Custom template pipeline not fully implemented yet")
        
        # For now, return an error indicating this feature needs implementation
        return {
            "success": False,
            "error": "Custom template pipeline not yet implemented",
            "poses": {},
            "template_info": {}
        }

    def _run_uploaded_pdb_pipeline(
        self,
        smiles: str,
        pdb_file: str,
        progress_callback: Optional[Callable],
        user_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run pipeline with uploaded PDB file - generate embedding and search"""
        logger.info(f"Running pipeline with uploaded PDB file: {pdb_file}")

        if progress_callback:
            progress_callback("Generating protein embedding...", 35)

        # Use run_full_pipeline with the uploaded PDB file
        try:
            results = self.pipeline.run_full_pipeline(
                protein_file=pdb_file,
                ligand_smiles=smiles,
                num_templates=user_settings["knn_threshold"],
                num_conformers=200,
                n_workers=4,
                similarity_threshold=user_settings["similarity_threshold"],
            )

            return self._format_pipeline_results(results, user_settings)

        except Exception as e:
            logger.error(f"Failed to process uploaded PDB file: {e}")

            # Safe streamlit call
            try:
                import streamlit as st

                if hasattr(st, "error"):
                    st.error(f"❌ Failed to process PDB file: {str(e)}")
                    st.info(
                        "💡 Make sure the file is a valid PDB format with protein chains"
                    )
            except:
                pass

            raise

    def _run_pdb_id_pipeline(
        self,
        smiles: str,
        pdb_id: str,
        progress_callback: Optional[Callable],
        user_settings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run pipeline with PDB ID from database"""
        logger.info(f"Running pipeline with PDB ID: {pdb_id}")

        # This will use the existing embedding from database
        results = self.pipeline.run_full_pipeline(
            protein_pdb_id=pdb_id,
            ligand_smiles=smiles,
            num_templates=user_settings["knn_threshold"],
            num_conformers=200,
            n_workers=4,
        )

        return self._format_pipeline_results(results, user_settings)

    def _run_full_pipeline_with_templates(
        self,
        smiles: str,
        templates: list,
        embedding: Any,
        chain_id: str,
        progress_callback: Optional[Callable],
        user_settings: Dict[str, Any],
        target_protein_file: Optional[str] = None,
        target_protein_pdb_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Continue full pipeline after finding templates - NOT IMPLEMENTED"""
        logger.warning("_run_full_pipeline_with_templates method not implemented")
        return {
            "success": False,
            "error": "Template-based pipeline not yet implemented",
            "poses": {},
            "template_info": {}
        }

    def _format_pipeline_results(
        self,
        results: Dict[str, Any],
        query_mol: Any = None,
        user_settings: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Format pipeline results for UI consumption"""
        if "error" in results:
            logger.error(f"Pipeline Error: {results['error']}")
            return None

        poses = results.get("poses", {})
        if not poses:
            logger.error("No poses generated")
            return None

        # Format the results
        # Ensure query molecule has original SMILES for visualization
        final_query_mol = query_mol or results.get("query_molecule")
        if final_query_mol is not None and hasattr(final_query_mol, "HasProp"):
            # If original SMILES not set, try to get from session
            if not final_query_mol.HasProp("original_smiles"):
                try:
                    input_smiles = self.session.get(SESSION_KEYS["INPUT_SMILES"])
                    if input_smiles:
                        final_query_mol.SetProp("original_smiles", input_smiles)
                        logger.debug(
                            f"Added original_smiles property to query molecule: {input_smiles}"
                        )
                except Exception as e:
                    logger.warning(f"Could not set original_smiles property: {e}")

        formatted = {
            "poses": poses,
            "template_info": None,
            "mcs_info": results.get("mcs_info"),
            "all_ranked_poses": results.get("all_ranked_poses"),
            "template_mol": None,
            "query_mol": final_query_mol,
        }

        # Extract template information
        if "template_molecules" in results and results["template_molecules"]:
            template_mol = results["template_molecules"][0]
            template_index = 0

            if "mcs_info" in results:
                mcs_info = results["mcs_info"]
                if isinstance(mcs_info, dict) and "selected_template_index" in mcs_info:
                    template_index = mcs_info["selected_template_index"]
                    if template_index < len(results["template_molecules"]):
                        template_mol = results["template_molecules"][template_index]

            # Store the actual template molecule
            formatted["template_mol"] = template_mol
            formatted["template_info"] = {
                "name": self._extract_pdb_id(template_mol, results),
                "index": template_index,
                "total_templates": len(results["template_molecules"]),
                "mcs_smarts": (
                    results.get("mcs_info", {}).get("smarts", "")
                    if isinstance(results.get("mcs_info"), dict)
                    else ""
                ),
                "atoms_matched": (
                    results.get("mcs_info", {}).get("atom_count", 0)
                    if isinstance(results.get("mcs_info"), dict)
                    else 0
                ),
            }

        return formatted

    def _extract_pdb_id(self, template_mol: Any, results: Dict[str, Any]) -> str:
        """Extract PDB ID from template molecule"""
        # Try multiple sources for PDB ID
        if hasattr(template_mol, "GetProp"):
            for prop in [
                "template_pid",
                "Template_PDB",
                "template_pdb",
                "pdb_id",
                "_Name",
            ]:
                try:
                    if template_mol.HasProp(prop):
                        return template_mol.GetProp(prop).upper()
                except:
                    pass

        # Check results for template info
        if "templates" in results and results["templates"]:
            if isinstance(results["templates"][0], (list, tuple)):
                return results["templates"][0][0].upper()
            elif isinstance(results["templates"][0], str):
                return results["templates"][0].upper()

        return "UNKN"

    async def run_pipeline_async(
        self,
        smiles: str,
        protein_input: str,
        custom_templates: Optional[list] = None,
        use_aligned_poses: bool = True,
        max_templates: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run the TEMPL pipeline asynchronously for non-blocking UI execution

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
        logger.info(f"Starting async pipeline execution for SMILES: {smiles}")

        try:
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

            # Update session with user preferences if provided
            if max_templates is not None:
                self.session.set(SESSION_KEYS["USER_KNN_THRESHOLD"], max_templates)
            if similarity_threshold is not None:
                self.session.set(
                    SESSION_KEYS["USER_SIMILARITY_THRESHOLD"], similarity_threshold
                )

            # Use ThreadPoolExecutor to run the synchronous pipeline in a separate thread
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Submit the synchronous run_pipeline method to the thread pool
                future = loop.run_in_executor(
                    executor,
                    self.run_pipeline,
                    molecule_data,
                    protein_data,
                    progress_callback,
                )

                # Await the result
                result = await future

            logger.info("Async pipeline execution completed")
            return result

        except Exception as e:
            logger.error(f"Async pipeline execution failed: {e}", exc_info=True)
            # Re-raise the exception to maintain proper async error handling
            raise


# Convenience function for backward compatibility and easy testing
async def run_pipeline_async(
    smiles: str,
    protein_input: str,
    custom_templates: Optional[list] = None,
    use_aligned_poses: bool = True,
    max_templates: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
    progress_callback: Optional[Callable] = None,
) -> Optional[Dict[str, Any]]:
    """Standalone async function for running the TEMPL pipeline

    This is a convenience function that creates a temporary pipeline service
    and runs the pipeline asynchronously. For production use, prefer using
    the PipelineService class directly.

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
    # Import here to avoid circular imports
    from ..config.settings import get_config
    from ..core.session_manager import SessionManager

    # Create temporary configuration and session for standalone use
    config = get_config()
    session = SessionManager(config)
    session.initialize()

    # Create pipeline service and run async
    service = PipelineService(config, session)
    return await service.run_pipeline_async(
        smiles=smiles,
        protein_input=protein_input,
        custom_templates=custom_templates,
        use_aligned_poses=use_aligned_poses,
        max_templates=max_templates,
        similarity_threshold=similarity_threshold,
        progress_callback=progress_callback,
    )
