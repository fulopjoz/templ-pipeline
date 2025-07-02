"""
Enhanced Pipeline Service for TEMPL Pipeline

Handles pipeline execution with proper error reporting and embedding generation.
"""

import logging
import time
from typing import Dict, Any, Optional, Callable, Tuple
from pathlib import Path
import sys
import os

from ..config.settings import AppConfig
from ..core.session_manager import SessionManager
from ..config.constants import SESSION_KEYS

logger = logging.getLogger(__name__)


class PipelineService:
    """Service for managing pipeline execution with enhanced error handling"""
    
    def __init__(self, config: AppConfig, session: SessionManager):
        """Initialize pipeline service
        
        Args:
            config: Application configuration
            session: Session manager
        """
        self.config = config
        self.session = session
        self.pipeline = None
    
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
            logger.info("Auto device mode - letting embedding functions choose optimal device")
    
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
    
    def run_pipeline(self, 
                    molecule_data: Dict[str, Any],
                    protein_data: Dict[str, Any],
                    progress_callback: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
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
            user_device_pref = self.session.get(SESSION_KEYS["USER_DEVICE_PREFERENCE"], "auto")
            user_knn_threshold = self.session.get(SESSION_KEYS["USER_KNN_THRESHOLD"], 100)
            user_chain_selection = self.session.get(SESSION_KEYS["USER_CHAIN_SELECTION"], "auto")
            user_similarity_threshold = self.session.get(SESSION_KEYS["USER_SIMILARITY_THRESHOLD"], 0.5)
            
            logger.info(f"User settings - Device: {user_device_pref}, KNN: {user_knn_threshold}, Chain: {user_chain_selection}, Similarity: {user_similarity_threshold}")
            
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
                self.pipeline = TEMPLPipeline(
                    embedding_path=None,  # Auto-detect
                    output_dir=self.config.paths.get('output_dir', 'temp'),
                    run_id=None
                )
            
            # Prepare inputs
            smiles = molecule_data.get('input_smiles')
            pdb_id = protein_data.get('pdb_id')
            pdb_file = protein_data.get('file_path')
            custom_templates = molecule_data.get('custom_templates')
            
            # Check if PDB ID exists in database first
            if pdb_id and not pdb_file:
                embedding_manager = self.pipeline._get_embedding_manager()
                if not embedding_manager.has_embedding(pdb_id):
                    # PDB ID not found - provide clear error message
                    error_msg = f"PDB ID '{pdb_id.upper()}' not found in database"
                    
                    # Safe streamlit call
                    try:
                        import streamlit as st
                        if hasattr(st, 'error'):
                            st.error(f"❌ {error_msg}")
                            st.info(f"💡 **Suggestion**: Upload the PDB file directly using the 'Upload File' option")
                            
                            # Show help for getting PDB files
                            with st.expander("How to get PDB files", expanded=True):
                                st.markdown(f"""
                                You can download PDB files from:
                                
                                1. **RCSB PDB**: [https://www.rcsb.org/structure/{pdb_id.upper()}](https://www.rcsb.org/structure/{pdb_id.upper()})
                                2. **PDBe**: [https://www.ebi.ac.uk/pdbe/entry/pdb/{pdb_id.lower()}](https://www.ebi.ac.uk/pdbe/entry/pdb/{pdb_id.lower()})
                                
                                Click "Download Files" → "PDB Format" on either site.
                                """)
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
                    smiles, custom_templates, progress_callback, user_settings={
                        'device_pref': user_device_pref,
                        'knn_threshold': user_knn_threshold,
                        'chain_selection': user_chain_selection,
                        'similarity_threshold': user_similarity_threshold
                    }
                )
            elif pdb_file:
                # Full pipeline with uploaded PDB file - generate embedding and search
                results = self._run_uploaded_pdb_pipeline(
                    smiles, pdb_file, progress_callback, user_settings={
                        'device_pref': user_device_pref,
                        'knn_threshold': user_knn_threshold,
                        'chain_selection': user_chain_selection,
                        'similarity_threshold': user_similarity_threshold
                    }
                )
            elif pdb_id:
                # Full pipeline with PDB ID from database
                results = self._run_pdb_id_pipeline(
                    smiles, pdb_id, progress_callback, user_settings={
                        'device_pref': user_device_pref,
                        'knn_threshold': user_knn_threshold,
                        'chain_selection': user_chain_selection,
                        'similarity_threshold': user_similarity_threshold
                    }
                )
            else:
                # Safe streamlit call
                try:
                    import streamlit as st
                    if hasattr(st, 'error'):
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
                if hasattr(st, 'error'):
                    if "not found in database" in error_str:
                        # Already handled above
                        pass
                    elif "No templates found" in error_str:
                        st.error("❌ No similar templates found in database")
                        st.info("💡 Try uploading custom template molecules or using a different protein")
                    elif "embedding" in error_str.lower():
                        st.error("❌ Failed to generate protein embedding")
                        st.info("💡 Check that the PDB file is valid and contains protein chains")
                    else:
                        st.error(f"❌ Pipeline error: {error_str}")
            except:
                logger.error(f"Pipeline error: {error_str}")
            
            return None
    
    def _run_custom_template_pipeline(self, smiles: str, custom_templates: list, 
                                    progress_callback: Optional[Callable], user_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Run pipeline with custom template molecules"""
        if progress_callback:
            progress_callback("Processing custom templates...", 40)
        
        query_mol = self.pipeline.prepare_query_molecule(ligand_smiles=smiles)
        
        if progress_callback:
            progress_callback("Generating conformers...", 60)
        
        pose_results = self.pipeline.generate_poses(
            query_mol=query_mol,
            template_mols=custom_templates,
            num_conformers=200,
            n_workers=4,
            use_aligned_poses=True
        )
        
        if progress_callback:
            progress_callback("Scoring poses...", 80)
        
        # Format results
        if isinstance(pose_results, dict) and 'poses' in pose_results:
            results = pose_results
        else:
            results = {'poses': pose_results}
        
        return self._format_pipeline_results(results, query_mol, user_settings)
    
    def _run_uploaded_pdb_pipeline(self, smiles: str, pdb_file: str, 
                                  progress_callback: Optional[Callable], user_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Run pipeline with uploaded PDB file - generate embedding and search"""
        logger.info(f"Running pipeline with uploaded PDB file: {pdb_file}")
        
        if progress_callback:
            progress_callback("Generating protein embedding...", 35)
        
        # Generate embedding for the uploaded protein
        try:
            # Resolve chain selection from user settings
            target_chain = self._resolve_chain_selection(user_settings['chain_selection'])
            embedding, chain_id = self.pipeline.generate_embedding(pdb_file, chain=target_chain)
            logger.info(f"Generated embedding for chain {chain_id}")
            
            if progress_callback:
                progress_callback("Searching for similar proteins...", 45)
            
            # Find similar templates using KNN
            templates = self.pipeline.find_templates(
                protein_embedding=embedding,
                num_templates=user_settings['knn_threshold'],
                similarity_threshold=user_settings['similarity_threshold'],
                allow_self_as_template=False
            )
            
            if not templates:
                # Safe streamlit call
                try:
                    import streamlit as st
                    if hasattr(st, 'warning'):
                        st.warning("⚠️ No similar proteins found in database")
                        st.info("💡 Consider uploading custom template molecules instead")
                except:
                    logger.warning("No similar proteins found in database")
                return None
            
            logger.info(f"Found {len(templates)} similar templates")
            
            # Safe streamlit calls for showing results
            try:
                import streamlit as st
                if hasattr(st, 'success'):
                    st.success(f"✅ Found {len(templates)} similar protein structures")
                    
                    # Show top templates
                    with st.expander("Top similar proteins", expanded=False):
                        for i, (pdb_id, similarity) in enumerate(templates[:5]):
                            st.write(f"{i+1}. **{pdb_id.upper()}** (similarity: {similarity:.3f})")
            except:
                # Just log if streamlit is not available
                logger.info(f"Top 5 similar proteins: {[(p, f'{s:.3f}') for p, s in templates[:5]]}")
            
            if progress_callback:
                progress_callback("Loading template molecules...", 55)
            
            # Continue with full pipeline
            return self._run_full_pipeline_with_templates(
                smiles, templates, embedding, chain_id, progress_callback, user_settings
            )
            
        except Exception as e:
            logger.error(f"Failed to process uploaded PDB file: {e}")
            
            # Safe streamlit call
            try:
                import streamlit as st
                if hasattr(st, 'error'):
                    st.error(f"❌ Failed to process PDB file: {str(e)}")
                    st.info("💡 Make sure the file is a valid PDB format with protein chains")
            except:
                pass
            
            raise
    
    def _run_pdb_id_pipeline(self, smiles: str, pdb_id: str, 
                            progress_callback: Optional[Callable], user_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Run pipeline with PDB ID from database"""
        logger.info(f"Running pipeline with PDB ID: {pdb_id}")
        
        # This will use the existing embedding from database
        results = self.pipeline.run_full_pipeline(
            protein_pdb_id=pdb_id,
            ligand_smiles=smiles,
            num_templates=user_settings['knn_threshold'],
            num_conformers=200,
            n_workers=4,
            use_aligned_poses=True
        )
        
        return self._format_pipeline_results(results, user_settings)
    
    def _run_full_pipeline_with_templates(self, smiles: str, templates: list, 
                                         embedding: Any, chain_id: str,
                                         progress_callback: Optional[Callable], user_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Continue full pipeline after finding templates"""
        
        # Load template molecules
        template_pdbs = [t[0] for t in templates]
        template_mols = self.pipeline.load_template_molecules(template_pdbs)
        
        if not template_mols:
            raise RuntimeError("Failed to load template molecules")
        
        if progress_callback:
            progress_callback("Preparing query molecule...", 65)
        
        # Prepare query molecule
        query_mol = self.pipeline.prepare_query_molecule(ligand_smiles=smiles)
        
        if progress_callback:
            progress_callback("Generating poses...", 75)
        
        # Generate poses
        pose_results = self.pipeline.generate_poses(
            query_mol,
            template_mols,
            num_conformers=200,
            n_workers=4,
            use_aligned_poses=True
        )
        
        if progress_callback:
            progress_callback("Finalizing results...", 90)
        
        # Format results
        if isinstance(pose_results, dict):
            results = {
                'poses': pose_results['poses'],
                'mcs_info': pose_results.get('mcs_info'),
                'all_ranked_poses': pose_results.get('all_ranked_poses'),
                'alignment_used': pose_results.get('alignment_used', True),
                'templates': templates,
                'template_molecules': template_mols,
                'embedding': embedding,
                'chain_id': chain_id
            }
        else:
            results = {'poses': pose_results}
        
        return self._format_pipeline_results(results, query_mol, user_settings)
    
    def _format_pipeline_results(self, results: Dict[str, Any], query_mol: Any = None, user_settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format pipeline results for UI consumption"""
        if 'error' in results:
            logger.error(f"Pipeline Error: {results['error']}")
            return None
        
        poses = results.get('poses', {})
        if not poses:
            logger.error("No poses generated")
            return None
        
        # Format the results
        formatted = {
            "poses": poses,
            "template_info": None,
            "mcs_info": results.get('mcs_info'),
            "all_ranked_poses": results.get('all_ranked_poses'),
            "template_mol": None,
            "query_mol": query_mol or results.get('query_molecule')
        }
        
        # Extract template information
        if 'template_molecules' in results and results['template_molecules']:
            template_mol = results['template_molecules'][0]
            template_index = 0
            
            if 'mcs_info' in results:
                mcs_info = results['mcs_info']
                if isinstance(mcs_info, dict) and 'selected_template_index' in mcs_info:
                    template_index = mcs_info['selected_template_index']
                    if template_index < len(results['template_molecules']):
                        template_mol = results['template_molecules'][template_index]
            
            # Store the actual template molecule
            formatted['template_mol'] = template_mol
            formatted['template_info'] = {
                "name": self._extract_pdb_id(template_mol, results),
                "index": template_index,
                "total_templates": len(results['template_molecules']),
                "mcs_smarts": results.get('mcs_info', {}).get('smarts', "") if isinstance(results.get('mcs_info'), dict) else "",
                "atoms_matched": results.get('mcs_info', {}).get('atom_count', 0) if isinstance(results.get('mcs_info'), dict) else 0
            }
        
        return formatted
    
    def _extract_pdb_id(self, template_mol: Any, results: Dict[str, Any]) -> str:
        """Extract PDB ID from template molecule"""
        # Try multiple sources for PDB ID
        if hasattr(template_mol, 'GetProp'):
            for prop in ['template_pid', 'Template_PDB', 'template_pdb', 'pdb_id', '_Name']:
                try:
                    if template_mol.HasProp(prop):
                        return template_mol.GetProp(prop).upper()
                except:
                    pass
        
        # Check results for template info
        if 'templates' in results and results['templates']:
            if isinstance(results['templates'][0], (list, tuple)):
                return results['templates'][0][0].upper()
            elif isinstance(results['templates'][0], str):
                return results['templates'][0].upper()
        
        return "UNKN"
