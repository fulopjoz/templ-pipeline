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

    def _get_optimal_workers(self) -> int:
        """Get optimal number of workers based on hardware configuration"""
        try:
            hardware_info = self.session.get(SESSION_KEYS["HARDWARE_INFO"])
            if hardware_info and hasattr(hardware_info, 'max_workers'):
                return hardware_info.max_workers
            else:
                # Fallback: use hardware manager directly
                from ..core.hardware_manager import get_hardware_manager
                hw_manager = get_hardware_manager()
                hw_info = hw_manager.detect_hardware()
                return hw_info.max_workers
        except Exception as e:
            logger.warning(f"Failed to get optimal workers: {e}")
            return 4  # Safe default

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
                        "num_conformers": 200,  # Standard conformer count
                        "n_workers": self._get_optimal_workers(),
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
        """Run pipeline with custom template molecules using MCS-based pose generation"""
        logger.info(f"Running custom template pipeline with {len(custom_templates)} templates")
        
        if progress_callback:
            progress_callback("Processing custom templates...", 40)

        try:
            # Import RDKit for molecule handling
            from rdkit import Chem
            
            # Create query molecule from SMILES
            query_mol = Chem.MolFromSmiles(smiles)
            if not query_mol:
                logger.error(f"Invalid SMILES: {smiles}")
                return {
                    "success": False,
                    "error": f"Invalid SMILES: {smiles}",
                    "poses": {},
                    "template_info": {}
                }
            
            # Validate custom templates
            valid_templates = []
            for i, template_mol in enumerate(custom_templates):
                if template_mol and hasattr(template_mol, 'GetNumAtoms'):
                    # Add template index for tracking
                    template_mol.SetProp("template_index", str(i))
                    template_mol.SetProp("template_pid", f"custom_{i}")
                    valid_templates.append(template_mol)
                else:
                    logger.warning(f"Invalid template molecule at index {i}")
            
            if not valid_templates:
                logger.error("No valid template molecules found")
                return {
                    "success": False,
                    "error": "No valid template molecules found in SDF file",
                    "poses": {},
                    "template_info": {}
                }
            
            logger.info(f"Found {len(valid_templates)} valid template molecules")
            
            if progress_callback:
                progress_callback("Finding maximum common substructure...", 50)
            
            # Use TEMPL's find_mcs function to find best template and MCS
            from templ_pipeline.core.mcs import find_mcs
            
            try:
                best_template_idx, mcs_smarts, mcs_details = find_mcs(
                    query_mol, valid_templates, return_details=True
                )
                best_template = valid_templates[best_template_idx]
                logger.info(f"Best template found at index {best_template_idx} with MCS: {mcs_smarts}")
                
            except Exception as e:
                logger.error(f"MCS finding failed: {e}")
                return {
                    "success": False,
                    "error": f"Failed to find common substructure: {str(e)}",
                    "poses": {},
                    "template_info": {}
                }
            
            if progress_callback:
                progress_callback("Generating conformers with MCS constraints...", 60)
            
            # Set target molecule in pipeline for conformer generation
            self.pipeline.target_mol = query_mol
            
            # Generate conformers using MCS constraints
            num_conformers = user_settings.get("num_conformers", 200)
            conformers_mol = self.pipeline.generate_conformers(
                best_template, mcs_smarts, num_conformers
            )
            
            if not conformers_mol or conformers_mol.GetNumConformers() == 0:
                logger.error("Failed to generate conformers")
                return {
                    "success": False,
                    "error": "Failed to generate conformers with MCS constraints",
                    "poses": {},
                    "template_info": {}
                }
            
            logger.info(f"Generated {conformers_mol.GetNumConformers()} conformers")
            
            if progress_callback:
                progress_callback("Scoring poses...", 80)
            
            # Score conformers against template
            scored_poses = self.pipeline.score_conformers(conformers_mol, best_template)
            
            if not scored_poses:
                logger.error("Failed to score conformers")
                return {
                    "success": False,
                    "error": "Failed to score generated poses",
                    "poses": {},
                    "template_info": {}
                }
            
            logger.info(f"Scored poses: {list(scored_poses.keys())}")
            
            if progress_callback:
                progress_callback("Finalizing results...", 90)
            
            # Ensure template molecule has proper properties for visualization
            if best_template and hasattr(best_template, "SetProp"):
                try:
                    # Add template SMILES for fallback
                    template_smiles = Chem.MolToSmiles(best_template)
                    best_template.SetProp("template_smiles", template_smiles)
                    best_template.SetProp("template_pdb_id", f"custom_{best_template_idx}")
                    logger.debug(f"Added properties to custom template molecule: custom_{best_template_idx}")
                except Exception as e:
                    logger.warning(f"Could not set template properties: {e}")
            
            # Create template info
            template_info = {
                "name": f"custom_{best_template_idx}",
                "index": best_template_idx,
                "total_templates": len(valid_templates),
                "template_pdb": f"custom_{best_template_idx}",
                "mcs_smarts": mcs_smarts,
                "atoms_matched": mcs_details.get("atom_count", 0),
            }
            
            # Add template SMILES to template_info for fallback
            if best_template and hasattr(best_template, "GetProp"):
                try:
                    if best_template.HasProp("template_smiles"):
                        template_info["template_smiles"] = best_template.GetProp("template_smiles")
                except Exception as e:
                    logger.debug(f"Could not get template SMILES: {e}")
            
            # Process MCS info for UI - ensure proper format
            mcs_info_for_processing = {
                "smarts": mcs_smarts,
                "mcs_smarts": mcs_smarts,
                "atom_count": mcs_details.get("atom_count", 0),
                "selected_template_index": best_template_idx
            }
            processed_mcs_info = self._process_mcs_info(mcs_info_for_processing, template_info)
            
            # Generate all ranked poses for download functionality
            from templ_pipeline.core.scoring import select_best
            
            try:
                all_ranked_poses = select_best(
                    conformers_mol, best_template,
                    no_realign=False,
                    n_workers=user_settings.get("n_workers", 4),
                    return_all_ranked=True
                )
                logger.info(f"Generated {len(all_ranked_poses)} ranked poses")
            except Exception as e:
                logger.warning(f"Failed to generate all ranked poses: {e}")
                all_ranked_poses = []
            
            # Format results for UI
            results = {
                "poses": scored_poses,
                "template_info": template_info,
                "mcs_info": processed_mcs_info,
                "all_ranked_poses": all_ranked_poses,
                "template_mol": best_template,
                "query_mol": query_mol,
                "template_molecules": valid_templates,  # Include all templates
                "success": True
            }
            
            logger.info("Custom template pipeline completed successfully")
            logger.info(f"Results summary: {len(scored_poses)} poses, template: {template_info['name']}")
            logger.info(f"Pose methods: {list(scored_poses.keys())}")
            logger.info(f"MCS SMARTS: {mcs_smarts}")
            logger.info(f"Template molecule type: {type(best_template)}")
            logger.info(f"Template info: {template_info}")
            logger.info(f"Processed MCS info: {processed_mcs_info}")
            return results
            
        except Exception as e:
            logger.error(f"Custom template pipeline failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": f"Custom template pipeline failed: {str(e)}",
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

        # Get hardware-optimized worker count instead of hardcoded value
        hardware_info = self.session.get(SESSION_KEYS["HARDWARE_INFO"])
        if hardware_info and hasattr(hardware_info, 'max_workers'):
            n_workers = hardware_info.max_workers
        else:
            # Fallback: use hardware manager directly
            n_workers = self.session.get(SESSION_KEYS.get("USER_DEVICE_PREFERENCE", "auto"))
            if isinstance(n_workers, str):
                from ..core.hardware_manager import get_hardware_manager
                hw_manager = get_hardware_manager()
                hw_info = hw_manager.detect_hardware()
                n_workers = hw_info.max_workers

        # Use run_full_pipeline with the uploaded PDB file
        try:
            # Try to extract PDB ID from the uploaded file for better identification
            # Use file header extraction instead of filename extraction for uploaded files
            from ..utils.file_utils import extract_pdb_id_from_file
            extracted_pdb_id = extract_pdb_id_from_file(pdb_file)
            if extracted_pdb_id:
                logger.info(f"Extracted PDB ID '{extracted_pdb_id}' from uploaded file header")
                
                # Check if this PDB ID already has an embedding in the database
                embedding_manager = self.pipeline._get_embedding_manager()
                if embedding_manager.has_embedding(extracted_pdb_id):
                    logger.info(f"Found existing embedding for PDB ID '{extracted_pdb_id}' - will use pre-computed embedding instead of generating new one")
                    # Use the PDB ID from database but exclude it from templates to avoid confusion
                    exclude_pdb_ids = {extracted_pdb_id.lower()}
                    logger.info(f"Excluding target PDB ID '{extracted_pdb_id}' from template search to avoid self-templating")
                else:
                    logger.info(f"No existing embedding found for PDB ID '{extracted_pdb_id}' - will generate on-demand embedding from uploaded file")
                    exclude_pdb_ids = set()
            else:
                logger.info("Could not extract PDB ID from filename - will generate on-demand embedding from uploaded file")
                exclude_pdb_ids = set()
            
            # Reduce worker count for scoring to avoid parallel processing issues
            # Use sequential processing for scoring to ensure all conformers are processed
            scoring_workers = 1  # Force sequential scoring to avoid Pebble failures
            logger.info(f"Using sequential scoring (n_workers=1) to ensure all conformers are processed reliably")
            
            results = self.pipeline.run_full_pipeline(
                protein_file=pdb_file,
                protein_pdb_id=extracted_pdb_id,  # Provide extracted PDB ID if available
                ligand_smiles=smiles,
                num_templates=user_settings["knn_threshold"],
                num_conformers=200,
                n_workers=scoring_workers,  # Use sequential scoring
                similarity_threshold=user_settings["similarity_threshold"],
                exclude_pdb_ids=exclude_pdb_ids,  # Exclude target from templates
                enable_optimization=False,  # Disable UFF/MMFF minimization by default (matches CLI behavior)
            )

            return self._format_pipeline_results(results, query_mol=None, user_settings=user_settings)

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

        # Get hardware-optimized worker count
        hardware_info = self.session.get(SESSION_KEYS["HARDWARE_INFO"])
        if hardware_info and hasattr(hardware_info, 'max_workers'):
            n_workers = hardware_info.max_workers
        else:
            # Fallback: use hardware manager directly
            from ..core.hardware_manager import get_hardware_manager
            hw_manager = get_hardware_manager()
            hw_info = hw_manager.detect_hardware()
            n_workers = hw_info.max_workers

        # Exclude the target PDB ID from templates to avoid self-templating
        exclude_pdb_ids = {pdb_id.lower()}
        logger.info(f"Excluding target PDB ID '{pdb_id}' from template search to avoid self-templating")

        # Use sequential scoring to ensure all conformers are processed
        scoring_workers = 1  # Force sequential scoring to avoid Pebble failures
        logger.info(f"Using sequential scoring (n_workers=1) to ensure all conformers are processed reliably")

        # This will use the existing embedding from database
        results = self.pipeline.run_full_pipeline(
            protein_pdb_id=pdb_id,
            ligand_smiles=smiles,
            num_templates=user_settings["knn_threshold"],
            num_conformers=200,
            n_workers=scoring_workers,  # Use sequential scoring
            exclude_pdb_ids=exclude_pdb_ids,  # Exclude target from templates
            enable_optimization=False,  # Disable UFF/MMFF minimization by default (matches CLI behavior)
        )

        return self._format_pipeline_results(results, query_mol=None, user_settings=user_settings)

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
        """Format pipeline results for UI consumption with enhanced molecule handling"""
        if "error" in results:
            logger.error(f"Pipeline Error: {results['error']}")
            return None

        poses = results.get("poses", {})
        if not poses:
            logger.error("No poses generated")
            return None

        # Import RDKit for molecule handling
        try:
            from rdkit import Chem
        except ImportError:
            logger.error("RDKit not available for molecule processing")
            return None

        # Ensure query molecule has original SMILES for visualization
        final_query_mol = query_mol or results.get("query_molecule")
        
        # Validate and fix query molecule if it's not a proper RDKit object
        if final_query_mol is not None:
            # If it's a dictionary, try to reconstruct the molecule from SMILES
            if isinstance(final_query_mol, dict):
                logger.warning("Query molecule is a dictionary, attempting to reconstruct from SMILES")
                try:
                    input_smiles = self.session.get(SESSION_KEYS["INPUT_SMILES"])
                    if input_smiles:
                        final_query_mol = Chem.MolFromSmiles(input_smiles)
                        if final_query_mol:
                            final_query_mol.SetProp("original_smiles", input_smiles)
                            final_query_mol.SetProp("input_method", "reconstructed")
                            logger.info("Successfully reconstructed query molecule from SMILES")
                        else:
                            logger.error("Failed to reconstruct molecule from SMILES")
                            final_query_mol = None
                    else:
                        logger.error("No input SMILES available for reconstruction")
                        final_query_mol = None
                except Exception as e:
                    logger.error(f"Error reconstructing query molecule: {e}")
                    final_query_mol = None
            
            # Validate RDKit molecule object
            elif not (hasattr(final_query_mol, 'ToBinary') and hasattr(final_query_mol, 'GetNumAtoms')):
                logger.warning(f"Invalid query molecule object: {type(final_query_mol)}")
                final_query_mol = None
        
        # Set original SMILES property if valid molecule
        if final_query_mol is not None and hasattr(final_query_mol, "HasProp"):
            # If original SMILES not set, try to get from session
            if not final_query_mol.HasProp("original_smiles"):
                try:
                    input_smiles = self.session.get(SESSION_KEYS["INPUT_SMILES"])
                    if input_smiles:
                        final_query_mol.SetProp("original_smiles", input_smiles)
                        logger.debug(f"Added original_smiles property to query molecule: {input_smiles}")
                except Exception as e:
                    logger.warning(f"Could not set original_smiles property: {e}")

        # Process template molecule information
        template_mol = None
        template_info = None
        template_index = 0

        # Extract template information from actual TEMPL pipeline results structure
        pipeline_template_info = results.get("template_info", {})
        templates_list = results.get("templates", [])
        
        # Extract the ACTUAL template molecule that was selected by MCS process
        template_mol = None
        best_template_pdb = None
        
        if isinstance(pipeline_template_info, dict):
            best_template_pdb = pipeline_template_info.get("best_template_pdb")
            logger.info(f"Pipeline selected template PDB: {best_template_pdb}")
            
            # Find the corresponding template molecule by matching PDB ID
            if best_template_pdb and templates_list:
                for template in templates_list:
                    if hasattr(template, "HasProp"):
                        # Check various property names for PDB ID
                        prop_names = ["template_pid", "template_pdb", "pdb_id", "_Name", "ID"]
                        for prop_name in prop_names:
                            if template.HasProp(prop_name):
                                template_pdb_id = template.GetProp(prop_name)
                                if template_pdb_id.upper() == best_template_pdb.upper():
                                    template_mol = template
                                    logger.info(f"Found matching template molecule for {best_template_pdb}: {prop_name}={template_pdb_id}")
                                    break
                        if template_mol:
                            break
                
                if not template_mol:
                    logger.warning(f"Could not find template molecule for PDB {best_template_pdb}, using first template as fallback")
                    template_mol = templates_list[0] if templates_list else None
            else:
                logger.warning("No best_template_pdb found, using first template as fallback")
                template_mol = templates_list[0] if templates_list else None
                
        if template_mol:
            logger.info(f"Selected template molecule: {type(template_mol)}")
        else:
            logger.warning("No template molecule found")
            
        # Extract template information from pipeline results  
        if isinstance(pipeline_template_info, dict) and best_template_pdb:
            template_info = {
                "name": best_template_pdb,
                "index": 0,
                "total_templates": len(templates_list) if templates_list else 1,
                "template_pdb": best_template_pdb,
                "mcs_smarts": pipeline_template_info.get("mcs_smarts", ""),
                "atoms_matched": 0,  # Will be updated from MCS info
                "ca_rmsd": pipeline_template_info.get("ca_rmsd", "unknown"),
                "similarity_score": pipeline_template_info.get("similarity_score", "unknown"),
            }
            logger.info(f"Extracted template info from pipeline: {template_info}")
        else:
            template_info = None
            
        # Ensure template molecule has proper properties for visualization
        if template_mol and hasattr(template_mol, "SetProp"):
            try:
                # Use the best_template_pdb from pipeline selection
                if best_template_pdb:
                    template_mol.SetProp("template_pdb_id", best_template_pdb)
                else:
                    # Fallback to extraction from molecule
                    extracted_pdb_id = self._extract_pdb_id(template_mol, results)
                    if extracted_pdb_id:
                        template_mol.SetProp("template_pdb_id", extracted_pdb_id)
                        if template_info:
                            template_info["name"] = extracted_pdb_id
                            template_info["template_pdb"] = extracted_pdb_id
                
                # Add template SMILES for fallback
                template_smiles = Chem.MolToSmiles(template_mol)
                template_mol.SetProp("template_smiles", template_smiles)
                if template_info:
                    template_info["template_smiles"] = template_smiles
                    
                logger.debug(f"Added properties to template molecule: {best_template_pdb}")
                
            except Exception as e:
                logger.warning(f"Could not set template properties: {e}")
        else:
            logger.warning(f"Template molecule invalid or missing SetProp: {type(template_mol)}")

        # Process MCS information from the actual pipeline template info
        raw_mcs_info = None
        
        # First try to get MCS from the pipeline template info (primary source)
        if pipeline_template_info and isinstance(pipeline_template_info, dict):
            mcs_smarts_from_pipeline = pipeline_template_info.get("mcs_smarts")
            if mcs_smarts_from_pipeline and mcs_smarts_from_pipeline.strip():
                raw_mcs_info = mcs_smarts_from_pipeline
                logger.info(f"Found MCS SMARTS from pipeline template_info: {mcs_smarts_from_pipeline}")
        
        # Fallback to other possible locations
        if not raw_mcs_info:
            raw_mcs_info = results.get("mcs_info") or results.get("mcs") or results.get("mcs_smarts")
            if raw_mcs_info:
                logger.info(f"Found MCS info from fallback location: {type(raw_mcs_info)}")
        
        # Update template_info with MCS data if we have it
        if raw_mcs_info and template_info:
            if isinstance(raw_mcs_info, str):
                template_info["mcs_smarts"] = raw_mcs_info
                # Try to count atoms in the SMARTS
                try:
                    mcs_mol = Chem.MolFromSmarts(raw_mcs_info)
                    if mcs_mol:
                        template_info["atoms_matched"] = mcs_mol.GetNumAtoms()
                except:
                    pass
        
        logger.debug(f"Raw MCS info from pipeline: {type(raw_mcs_info)} - {raw_mcs_info}")
        processed_mcs_info = self._process_mcs_info(raw_mcs_info, template_info)
        logger.debug(f"Processed MCS info: {processed_mcs_info}")

        # Update template_info with processed MCS data
        if template_info and processed_mcs_info:
            # Only update if we don't already have the information
            if not template_info.get("mcs_smarts"):
                template_info["mcs_smarts"] = processed_mcs_info.get("smarts", "")
            if not template_info.get("atoms_matched"):
                template_info["atoms_matched"] = processed_mcs_info.get("atom_count", 0)

        # DEBUGGING: Log all_ranked_poses information
        all_ranked_poses = results.get("all_ranked_poses")
        logger.info(f"DEBUG: Raw all_ranked_poses from pipeline: type={type(all_ranked_poses)}, length={len(all_ranked_poses) if hasattr(all_ranked_poses, '__len__') else 'N/A'}")
        if all_ranked_poses and len(all_ranked_poses) > 0:
            logger.info(f"DEBUG: First all_ranked_pose: {type(all_ranked_poses[0])}")
            if hasattr(all_ranked_poses[0], '__len__'):
                logger.info(f"DEBUG: First all_ranked_pose length: {len(all_ranked_poses[0])}")

        # Final fallback: if we still don't have template info but have template molecule
        if not template_info and template_mol:
            # Try to extract template PDB ID from the molecule itself
            extracted_pdb_id = self._extract_pdb_id(template_mol, results)
            if extracted_pdb_id and extracted_pdb_id != "UNKN":
                template_info = {
                    "name": extracted_pdb_id,
                    "index": 0,
                    "total_templates": 1,
                    "template_pdb": extracted_pdb_id,
                    "mcs_smarts": "",
                    "atoms_matched": 0,
                }
                
                # Add template SMILES
                try:
                    template_smiles = Chem.MolToSmiles(template_mol)
                    template_mol.SetProp("template_smiles", template_smiles)
                    template_info["template_smiles"] = template_smiles
                except Exception as e:
                    logger.warning(f"Could not add template SMILES: {e}")
                
                logger.info(f"Created fallback template info: {template_info}")

        formatted = {
            "poses": poses,
            "template_info": template_info,
            "mcs_info": processed_mcs_info,
            "all_ranked_poses": all_ranked_poses,
            "template_mol": template_mol,
            "query_mol": final_query_mol,
        }

        logger.info(f"Formatted results with template: {template_info.get('name') if template_info else 'None'}")
        logger.info(f"MCS info available: {processed_mcs_info is not None}")
        logger.info(f"Template molecule available: {template_mol is not None}")
        logger.info(f"DEBUG: Formatted all_ranked_poses: type={type(formatted['all_ranked_poses'])}, length={len(formatted['all_ranked_poses']) if hasattr(formatted['all_ranked_poses'], '__len__') else 'N/A'}")

        return formatted

    def _process_mcs_info(self, mcs_info: Any, template_info: Dict = None) -> Optional[Dict[str, Any]]:
        """Process MCS information into a standardized format for visualization
        
        Args:
            mcs_info: Raw MCS information from pipeline
            template_info: Template information for context
            
        Returns:
            Processed MCS information dictionary
        """
        if not mcs_info:
            return None
        
        try:
            from rdkit import Chem
        except ImportError:
            logger.error("RDKit not available for MCS processing")
            return None

        processed = {}
        
        # Handle different MCS info formats
        if isinstance(mcs_info, dict):
            # Copy relevant fields
            for key in ["smarts", "mcs_smarts", "atom_count", "query_match", "template_match", "selected_template_index"]:
                if key in mcs_info:
                    processed[key] = mcs_info[key]
            
            # Ensure we have a valid SMARTS pattern
            smarts = processed.get("smarts") or processed.get("mcs_smarts")
            if smarts and isinstance(smarts, str) and len(smarts.strip()) > 0:
                processed["smarts"] = smarts.strip()
                
                # Try to validate the SMARTS and get atom count
                try:
                    mcs_mol = Chem.MolFromSmarts(smarts)
                    if mcs_mol:
                        processed["atom_count"] = mcs_mol.GetNumAtoms()
                        processed["valid_smarts"] = True
                        logger.debug(f"Validated MCS SMARTS: {smarts} ({processed['atom_count']} atoms)")
                    else:
                        logger.warning(f"Invalid MCS SMARTS pattern: {smarts}")
                        processed["valid_smarts"] = False
                except Exception as e:
                    logger.warning(f"Error validating MCS SMARTS: {e}")
                    processed["valid_smarts"] = False
            
            # Add template context if available
            if template_info:
                processed["template_name"] = template_info.get("name", "")
                processed["template_index"] = template_info.get("index", 0)
                
        elif isinstance(mcs_info, str):
            # Assume it's a SMARTS string
            smarts = mcs_info.strip()
            if len(smarts) > 0:
                processed["smarts"] = smarts
                try:
                    mcs_mol = Chem.MolFromSmarts(smarts)
                    if mcs_mol:
                        processed["atom_count"] = mcs_mol.GetNumAtoms()
                        processed["valid_smarts"] = True
                    else:
                        processed["valid_smarts"] = False
                except Exception as e:
                    logger.warning(f"Error processing MCS string: {e}")
                    processed["valid_smarts"] = False
        
        return processed if processed else None

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
