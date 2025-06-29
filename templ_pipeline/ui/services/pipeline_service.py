"""
Pipeline Service for TEMPL Pipeline

Core pipeline execution functionality extracted from app.py for app_v2.py compatibility.
"""

import streamlit as st
import logging
import multiprocessing
from pathlib import Path

logger = logging.getLogger(__name__)

class StreamlitProgressHandler(logging.Handler):
    """Custom logging handler for Streamlit progress display"""
    
    def __init__(self, progress_placeholder):
        super().__init__()
        self.progress_placeholder = progress_placeholder
        self.setLevel(logging.INFO)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            with self.progress_placeholder.container():
                if record.levelno >= logging.ERROR:
                    st.error(msg)
                elif record.levelno >= logging.WARNING:
                    st.warning(msg)
                else:
                    st.info(msg)
        except Exception:
            pass

def _format_pipeline_results_for_ui(results, template_mol=None, query_mol=None):
    """Format pipeline results for UI display"""
    if not results or 'poses' not in results:
        return None
    
    poses = results['poses']
    
    # Store template information for metadata
    if template_mol:
        st.session_state.template_info = {
            'name': getattr(template_mol, 'template_pid', 'custom'),
            'index': 0,
            'total_templates': 1
        }
    
    return poses

def run_pipeline(smiles, protein_input=None, custom_templates=None, use_aligned_poses=True, max_templates=None, similarity_threshold=None):
    """Run the complete TEMPL pipeline using TEMPLPipeline class"""
    
    progress_placeholder = st.empty()
    
    # Detect available hardware using TEMPL hardware detection
    try:
        from templ_pipeline.core.hardware_utils import get_suggested_worker_config, get_hardware_info
        hardware_config = get_suggested_worker_config()
        hardware_info = get_hardware_info()
        max_workers = hardware_config["n_workers"]
        
        # Store hardware info in session state
        st.session_state.hardware_info = hardware_info
    except ImportError:
        # Fallback to basic detection
        total_cores = multiprocessing.cpu_count()
        max_workers = min(total_cores, 4)
        st.session_state.hardware_info = {
            'total_cores': total_cores,
            'used_cores': max_workers,
            'utilization': f"{max_workers}/{total_cores}"
        }
    
    # Set up logging handler for progress
    handler = StreamlitProgressHandler(progress_placeholder)
    handler.setLevel(logging.INFO)
    pipeline_logger = logging.getLogger('templ_pipeline')
    pipeline_logger.addHandler(handler)
    
    try:
        from templ_pipeline.core.pipeline import TEMPLPipeline
        
        with progress_placeholder.container():
            st.info("Starting TEMPL Pipeline")
            hardware_display = st.session_state.hardware_info.get('utilization', f"{max_workers}/unknown")
            st.markdown(f"Hardware: Using {hardware_display} CPU cores")
            alignment_status = "aligned" if use_aligned_poses else "original geometry"
            st.markdown(f"Mode: Returning {alignment_status} poses")
        
        # Initialize pipeline
        pipeline = TEMPLPipeline(
            embedding_path=None,  # Auto-detect
            output_dir=st.session_state.get('temp_dir', 'temp'),
            run_id=None
        )
        
        # Check if using custom templates (MCS-only workflow)
        if custom_templates:
            with progress_placeholder.container():
                st.info("Using custom template molecules...")
                st.success(f"Loaded {len(custom_templates)} custom templates")
                st.info("Generating molecular conformers...")
            
            # Generate poses using custom templates
            query_mol = pipeline.prepare_query_molecule(ligand_smiles=smiles)
            
            pose_results = pipeline.generate_poses(
                query_mol=query_mol,
                template_mols=custom_templates,
                num_conformers=200,
                n_workers=max_workers,
                use_aligned_poses=use_aligned_poses  # Use the alignment control parameter
            )
            
            # Handle both dict and poses format for backward compatibility
            if isinstance(pose_results, dict) and 'poses' in pose_results:
                results = pose_results
            else:
                results = {'poses': pose_results}
            
            with progress_placeholder.container():
                st.success("Pipeline completed successfully!")
            
            return _format_pipeline_results_for_ui(results, custom_templates[0], query_mol)
            
        else:
            # Full pipeline workflow
            with progress_placeholder.container():
                st.info("Processing protein structure...")
                if len(protein_input) == 4 and protein_input.isalnum():
                    st.markdown(f"Database lookup: Checking for PDB {protein_input.upper()}")
            
            # Determine protein input type
            protein_file = None
            protein_pdb_id = None
            
            if len(protein_input) == 4 and protein_input.isalnum():
                protein_pdb_id = protein_input.lower()
            else:
                protein_file = protein_input
            
            # Determine filtering parameters based on user selection
            if max_templates is not None:
                # KNN mode: use max_templates, no similarity threshold
                num_templates = max_templates
                sim_threshold = None
            elif similarity_threshold is not None:
                # Similarity threshold mode: no limit on count, use threshold
                num_templates = None  # No limit
                sim_threshold = similarity_threshold
            else:
                # Fallback to default KNN mode
                num_templates = 100
                sim_threshold = None
            
            # Run full pipeline with real-time progress updates
            results = pipeline.run_full_pipeline(
                protein_file=protein_file,
                protein_pdb_id=protein_pdb_id,
                ligand_smiles=smiles,
                num_templates=num_templates,
                num_conformers=200,
                n_workers=max_workers,
                similarity_threshold=sim_threshold,
                use_aligned_poses=use_aligned_poses  # Use the alignment control parameter
            )
            
            # Show completion
            with progress_placeholder.container():
                st.success("Pipeline completed successfully!")
                if 'templates' in results:
                    st.markdown(f"Found {len(results['templates'])} templates")
                if 'poses' in results:
                    st.markdown(f"Generated {len(results['poses'])} poses")
                alignment_used = results.get('alignment_used', 'unknown')
                st.markdown(f"Poses: Using {'aligned' if alignment_used else 'original geometry'} coordinates")
            
            return _format_pipeline_results_for_ui(results)
    except Exception as e:
        import traceback
        error_details = str(e)
        
        with progress_placeholder.container():
            st.error(f"Pipeline Error: {error_details}")
            
            # Provide specific guidance based on error type
            if "not found in database" in error_details:
                st.info(f"Suggestion: PDB ID '{protein_input}' not available. Try uploading the PDB file directly.")
            elif "Invalid SMILES" in error_details:
                st.info("Suggestion: Check your molecule format. Try a different SMILES string or upload an SDF file.")
            elif "No conformers generated" in error_details:
                st.info("Suggestion: Molecule might be too complex or templates incompatible. Try simpler molecules.")
            elif "hydrogen" in error_details.lower() or "alignment" in error_details.lower():
                st.info("Suggestion: Try using 'Original Geometry' mode in Advanced Settings to avoid alignment issues.")
        
        logger.error(f"Pipeline execution failed: {error_details}")
    finally:
        # Clean up logging handler
        pipeline_logger = logging.getLogger('templ_pipeline')
        if handler in pipeline_logger.handlers:
            pipeline_logger.removeHandler(handler)
    
    return None
