"""
TEMPL Pipeline Streamlit UI

A simple, clean interface for template-based protein ligand pose prediction.
"""

import os
import io
import sys
import tempfile
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import uuid
import gzip
import logging
import json
import traceback

import streamlit as st
from streamlit import session_state as ss
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdShapeHelpers
import py3Dmol
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("streamlit_app")

# Import stmol for 3D visualization
try:
    from stmol import showmol
    STMOL_AVAILABLE = True
except ImportError:
    STMOL_AVAILABLE = False

# Initialize torch availability flag
TORCH_AVAILABLE = False

def get_torch():
    """Lazy import of torch to avoid Streamlit conflicts during startup"""
    try:
        import torch
        return torch
    except ImportError:
        return None

@st.cache_data
def get_system_capabilities():
    """Get system capabilities including GPU status"""
    capabilities = {
        "torch_available": False,
        "cuda_available": False,
        "gpu_count": 0,
        "cuda_version": None
    }
    
    torch = get_torch()
    if torch is not None:
        capabilities["torch_available"] = True
        try:
            capabilities["cuda_available"] = torch.cuda.is_available()
            if capabilities["cuda_available"]:
                capabilities["gpu_count"] = torch.cuda.device_count()
                capabilities["cuda_version"] = torch.version.cuda
        except Exception:
            capabilities["cuda_available"] = False
    
    return capabilities

# Import psutil for memory detection
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Add parent directory to path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import pipeline modules
try:
    from templ_pipeline.core.embedding import (
        EmbeddingManager, 
        get_protein_embedding,
        get_embedding,
        get_sample_pdb_ids,
        is_pdb_id_in_database,
        _resolve_embedding_path,
        select_templates,
        analyze_embedding_database,
        filter_templates_by_ca_rmsd,
        get_templates_with_progressive_fallback,
        CA_RMSD_THRESHOLD,
        CA_RMSD_FALLBACK_THRESHOLDS
    )
    from templ_pipeline.core.utils import find_pocket_chains, find_pdbbind_paths
    from templ_pipeline.core.mcs import (
        find_mcs, 
        generate_conformers, 
        transform_ligand,
        prepare_mol,
        detect_and_substitute_organometallic,
        get_central_atom,
        central_atom_embed,
        find_best_ca_rmsd_template
    )
    from templ_pipeline.core.scoring import select_best as scoring_select_best
except ImportError:
    try:
        parent_dir = str(Path(__file__).parent.parent)
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        
        from templ_pipeline.core.embedding import (
            EmbeddingManager, 
            get_protein_embedding,
            get_embedding,
            get_sample_pdb_ids,
            is_pdb_id_in_database,
            _resolve_embedding_path,
            select_templates,
            analyze_embedding_database,
            filter_templates_by_ca_rmsd,
            get_templates_with_progressive_fallback,
            CA_RMSD_THRESHOLD,
            CA_RMSD_FALLBACK_THRESHOLDS
        )
        from templ_pipeline.core.utils import find_pocket_chains, find_pdbbind_paths
        from templ_pipeline.core.mcs import (
            find_mcs, 
            generate_conformers, 
            transform_ligand,
            prepare_mol,
            detect_and_substitute_organometallic,
            get_central_atom,
            central_atom_embed,
            find_best_ca_rmsd_template
        )
        from templ_pipeline.core.scoring import select_best as scoring_select_best
    except ImportError as e:
        st.error(f"Failed to import TEMPL pipeline modules: {str(e)}")
        st.info("Please run this application from the project root directory using: python -m templ_pipeline.ui.streamlit_app")
        st.stop()

# Import embedding map component
try:
    from templ_pipeline.ui.components.embedding_map import (
        render_embedding_landscape, 
        get_method_toggle,
        load_embedding_map
    )
    EMBEDDING_MAP_AVAILABLE = True
except ImportError:
    EMBEDDING_MAP_AVAILABLE = False
    logger.warning("Embedding map component not available")

@st.cache_data
def load_embedding_map_cached():
    """Cached version of embedding map loading"""
    try:
        from templ_pipeline.ui.components.embedding_map import load_embedding_map
        return load_embedding_map()
    except Exception as e:
        st.error(f"Failed to load embedding map: {e}")
        return None

# Configure page
st.set_page_config(
    page_title="TEMPL Pipeline",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal CSS for clean appearance
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "protein_embedding" not in ss:
    ss.protein_embedding = None
if "template_results" not in ss:
    ss.template_results = None
if "query_mol" not in ss:
    ss.query_mol = None
if "template_mol" not in ss:
    ss.template_mol = None
if "conformers" not in ss:
    ss.conformers = None
if "poses" not in ss:
    ss.poses = {}
if "protein_file" not in ss:
    ss.protein_file = None
if "protein_pdb_id" not in ss:
    ss.protein_pdb_id = None
if "pocket_chains" not in ss:
    ss.pocket_chains = []
if "mcs_indices" not in ss:
    ss.mcs_indices = {}
if "crystal_mol" not in ss:
    ss.crystal_mol = None
if "pdb_header_info" not in ss:
    ss.pdb_header_info = None

# Helper functions
def parse_pdb_header(file_content: bytes) -> Dict[str, str]:
    """Parse PDB file header and extract metadata"""
    try:
        content = file_content.decode('utf-8')
        lines = content.split('\n')
        
        for line in lines:
            if line.startswith('HEADER'):
                return {
                    'pdb_id': line[62:66].strip().lower() if len(line) >= 66 else None,
                    'classification': line[10:50].strip(),
                    'deposition_date': line[50:59].strip(),
                    'header_line': line
                }
    except Exception as e:
        logger.error(f"Error parsing PDB header: {e}")
    
    return {}

def handle_pdb_upload(uploaded_file):
    """Handle PDB file upload with automatic PDB ID extraction"""
    # Save file temporarily
    pdb_path = read_pdb_file(uploaded_file)
    
    # Parse header
    header_info = parse_pdb_header(uploaded_file.getvalue())
    
    if header_info.get('pdb_id'):
        ss.protein_pdb_id = header_info['pdb_id']
        ss.protein_file = pdb_path
        ss.pdb_header_info = header_info
        
        return {
            'success': True,
            'pdb_id': header_info['pdb_id'],
            'classification': header_info.get('classification'),
            'file_path': pdb_path
        }
    else:
        return {
            'success': False,
            'error': 'Could not extract PDB ID from HEADER line'
        }

def generate_embedding_smart():
    """Generate embedding using best available method"""
    pdb_id = ss.get('protein_pdb_id')
    pdb_file = ss.get('protein_file')
    
    if not pdb_id:
        return False, "No PDB ID available"
    
    # Initialize embedding manager if needed
    if not ss.get('embedding_manager'):
        try:
            ss.embedding_manager = get_embedding_manager_singleton(
                st.session_state.embedding_path
            )
        except Exception as e:
            return False, f"Failed to initialize embedding database: {e}"
    
    # Try database first
    embedding, chain_id = ss.embedding_manager.get_embedding(pdb_id)
    
    if embedding is not None:
        ss.protein_embedding = embedding
        return True, f"Found existing embedding for {pdb_id.upper()}"
    
    # If no embedding in database but we have a file
    if pdb_file and os.path.exists(pdb_file):
        try:
            # Generate new embedding from uploaded file
            from templ_pipeline.core.embedding import get_protein_embedding
            embedding = get_protein_embedding(pdb_file)
            if embedding is not None:
                ss.protein_embedding = embedding
                return True, f"Generated new embedding for {pdb_id.upper()}"
        except Exception as e:
            return False, f"Failed to generate embedding: {e}"
    
    return False, f"PDB {pdb_id.upper()} not found in database and no file available"

def ensure_rdkit_mol(obj) -> Optional[Chem.Mol]:
    """Ensure that an object is a valid RDKit molecule"""
    if obj is None:
        return None
        
    try:
        if isinstance(obj, Chem.Mol):
            return obj
        if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], Chem.Mol):
            return obj[0]
        if isinstance(obj, str):
            mol = Chem.MolFromSmiles(obj)
            if mol:
                return mol
        return None
    except Exception as e:
        logger.error(f"Error converting to RDKit Mol: {str(e)}")
        return None

def display_molecule(mol, width=400, height=300):
    """Display an RDKit molecule using 2D representation"""
    if mol is None:
        return
    
    try:    
        mol = ensure_rdkit_mol(mol)
        if mol is None:
            st.error("Invalid molecule")
            return
        
        mol_copy = Chem.RemoveHs(Chem.Mol(mol))
        if mol_copy.GetNumConformers() > 0 and mol_copy.GetConformer().Is3D():
            mol_copy.Compute2DCoords()
        
        img = Draw.MolToImage(mol_copy, size=(width, height), kekulize=True, wedgeBonds=True, fitImage=True)
        st.image(img)
    except Exception as e:
        st.error(f"Error displaying molecule: {str(e)}")

def read_pdb_file(file):
    """Read a PDB file from streamlit uploaded file"""
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as fp:
        fp.write(file.getvalue())
        pdb_path = fp.name
    return pdb_path

def display_molecule_stmol(mol, height=400, width=600, style="stick"):
    """Display a molecule using stmol for 3D visualization"""
    if not STMOL_AVAILABLE:
        st.error("stmol not available")
        return None
    
    mol = ensure_rdkit_mol(mol)
    if mol is None:
        st.warning("No valid molecule provided")
        return None
    
    try:
        view = py3Dmol.view(width=width, height=height)
        mol_block = Chem.MolToMolBlock(mol)
        view.addModel(mol_block, 'mol')
        
        if style == "stick":
            style_options = {"stick": {}}
        elif style == "ball+stick":
            style_options = {"stick": {}, "sphere": {"scale": 0.3}}
        elif style == "sphere":
            style_options = {"sphere": {}}
        else:
            style_options = {"stick": {}}
            
        view.setStyle({}, style_options)
        view.setBackgroundColor('white')
        view.zoomTo()
        
        return showmol(view, height=height, width=width)
    except Exception as e:
        st.error(f"Error displaying molecule: {str(e)}")
        return None

def load_templates_from_processed_sdf(template_pdbs, gz_path="templ_pipeline/data/ligands/processed_ligands_new.sdf.gz"):
    """Load template molecules from the pre-processed SDF.gz file"""
    template_pdbs_lower = [pdb_id.lower() for pdb_id in template_pdbs]
    
    possible_paths = [
        gz_path,
        os.path.join(os.getcwd(), gz_path),
        "/home/ubuntu/mcs/templ_pipeline/data/ligands/processed_ligands_new.sdf.gz",
        "/home/ubuntu/mcs/mcs_bench/data/processed_ligands_new.sdf.gz"
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if not file_path:
        st.error("Processed ligands file not found")
        return []
    
    templates = []
    loaded_pdb_ids = set()
    
    try:
        with gzip.open(file_path, 'rb') as fh:
            for mol in Chem.ForwardSDMolSupplier(fh, removeHs=False, sanitize=False):
                if not mol or not mol.GetNumConformers():
                    continue
                
                if mol.HasProp("_Name"):
                    mol_name = mol.GetProp("_Name")
                    pdb_id = mol_name[:4].lower()
                    
                    if pdb_id in template_pdbs_lower and pdb_id not in loaded_pdb_ids:
                        if mol.GetNumAtoms() >= 3 and mol.GetNumBonds() >= 1:
                            mol.SetProp("_Name", pdb_id)
                            templates.append(mol)
                            loaded_pdb_ids.add(pdb_id)
        
        missing_pdbs = set(template_pdbs_lower) - loaded_pdb_ids
        if missing_pdbs:
            st.warning(f"Could not find templates: {', '.join(missing_pdbs)}")
        
        return templates
    except Exception as e:
        st.error(f"Error loading templates: {str(e)}")
        return []

@st.cache_resource
def get_embedding_manager_singleton(embedding_path):
    """Get or create a singleton EmbeddingManager instance"""
    try:
        return EmbeddingManager(embedding_path)
    except Exception as e:
        st.error(f"Failed to initialize EmbeddingManager: {e}")
        return None

# Main application
def main():
    """Main Streamlit application"""
    
    # Simple header
    st.title("TEMPL Pipeline")
    st.markdown("Template-based protein ligand pose prediction")
    
    # Simple progress indicator
    steps = ["Protein Embedding", "Template Finding", "Pose Generation", "Results"]
    completed = [
        ss.get("protein_embedding") is not None,
        bool(ss.get("template_results")),
        bool(ss.get("poses")),
        bool(ss.get("poses"))
    ]
    
    progress_text = " → ".join([
        f"**{step}**" if completed[i] else step 
        for i, step in enumerate(steps)
    ])
    st.markdown(progress_text)
    st.divider()
    
    # Initialize paths
    DEFAULT_EMBEDDING_PATH = os.getenv("TEMPL_EMBEDDING_PATH", "data/embeddings/protein_embeddings_base.npz")
    ABSOLUTE_DEFAULT_PATH = "/home/ubuntu/mcs/templ_pipeline/data/embeddings/protein_embeddings_base.npz"
    
    if "embedding_path" not in st.session_state:
        if os.path.exists(ABSOLUTE_DEFAULT_PATH):
            st.session_state.embedding_path = ABSOLUTE_DEFAULT_PATH
        else:
            st.session_state.embedding_path = DEFAULT_EMBEDDING_PATH
    
    # Embedding manager will be initialized on demand
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Protein Embedding", 
        "Template Finding",
        "Pose Generation", 
        "Results"
    ])
    
    # Tab 1: Protein Embedding
    with tab1:
        st.header("Protein Embedding")
        
        # Input method
        input_method = st.radio("Input method:", ["PDB ID", "Upload PDB File"], horizontal=True)
        
        if input_method == "PDB ID":
            pdb_id = st.text_input("PDB ID", placeholder="e.g., 1a1e").strip().lower()
            
            if pdb_id and len(pdb_id) == 4 and pdb_id.isalnum():
                ss.protein_pdb_id = pdb_id
                
                # Initialize embedding manager for database check
                if not ss.get("embedding_manager"):
                    try:
                        ss.embedding_manager = get_embedding_manager_singleton(st.session_state.embedding_path)
                    except Exception:
                        pass
                
                # Check if exists in database
                if ss.get("embedding_manager"):
                    if ss.embedding_manager.get_embedding(pdb_id)[0] is not None:
                        st.success(f"PDB {pdb_id.upper()} found in database")
                    else:
                        st.warning(f"PDB {pdb_id.upper()} not found in database")
            elif pdb_id:
                st.error("PDB ID must be exactly 4 alphanumeric characters")
        else:
            uploaded_file = st.file_uploader("Upload PDB File", type=["pdb"])
            if uploaded_file:
                result = handle_pdb_upload(uploaded_file)
                if result['success']:
                    st.success(f"PDB file uploaded - ID: {result['pdb_id'].upper()}")
                    st.info(f"Classification: {result['classification']}")
                else:
                    st.error(result['error'])
        
        # Generate embedding
        if st.button("Generate Embedding", type="primary"):
            with st.spinner("Generating embedding..."):
                success, message = generate_embedding_smart()
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Status
        st.subheader("Status")
        if ss.get("protein_pdb_id"):
            st.write(f"PDB ID: `{ss.protein_pdb_id.upper()}`")
            if ss.get("pdb_header_info"):
                st.write(f"Classification: {ss.pdb_header_info.get('classification', 'Unknown')}")
                st.write(f"Source: {'Database' if input_method == 'PDB ID' else 'Uploaded File'}")
        elif ss.get("protein_file"):
            st.write(f"File: `{os.path.basename(ss.protein_file)}`")
        
        if ss.get("protein_embedding") is not None:
            st.success("Embedding ready")
            st.write(f"Dimensions: {ss.protein_embedding.shape}")
        else:
            st.info("No embedding generated")
        
        # Embedding landscape (simplified)
        if EMBEDDING_MAP_AVAILABLE and ss.get("protein_embedding") is not None:
            st.subheader("Embedding Landscape")
            
            method = get_method_toggle() if EMBEDDING_MAP_AVAILABLE else "t-SNE"
            
            if st.button("View Embedding Landscape"):
                try:
                    method_key = method.lower().replace('-', '')
                    
                    with st.spinner(f"Loading {method} visualization..."):
                        result = render_embedding_landscape(
                            query_embedding=ss.protein_embedding,
                            query_pdb_id=ss.protein_pdb_id,
                            method=method_key
                        )
                    
                    if result and isinstance(result, dict):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Proteins", result.get('total_proteins', 'Unknown'))
                        with col2:
                            if 'similarities' in result:
                                avg_sim = result['similarities'].mean()
                                st.metric("Avg Similarity", f"{avg_sim:.3f}")
                        with col3:
                            if 'similarities' in result:
                                max_sim = result['similarities'].max()
                                st.metric("Best Match", f"{max_sim:.3f}")
                        
                        # Quick template selection
                        if 'similarities' in result and 'pdb_ids' in result:
                            top_indices = np.argsort(result['similarities'])[-5:][::-1]
                            if st.button("Use Top 5 as Templates"):
                                top_5_templates = [(result['pdb_ids'][idx], result['similarities'][idx]) for idx in top_indices]
                                ss.template_results = top_5_templates
                                st.success("Top 5 templates selected!")
                                st.rerun()
                        
                except Exception as e:
                    st.error(f"Error loading embedding map: {e}")
    
    # Tab 2: Template Finding
    with tab2:
        st.header("Template Finding")
        
        # Check prerequisites
        if ss.get("protein_embedding") is None:
            st.warning("Please generate a protein embedding first")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Search Parameters")
                
                k = st.slider("Number of Templates", 1, 50, 10)
                
                use_threshold = st.checkbox("Use Similarity Threshold", value=False)
                if use_threshold:
                    similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7, 0.05)
                else:
                    similarity_threshold = None
                
                if st.button("Find Templates", type="primary"):
                    with st.spinner("Finding templates..."):
                        try:
                            threshold_param = similarity_threshold if use_threshold else None
                            
                            templates = ss.embedding_manager.find_neighbors(
                                query_pdb_id=ss.protein_pdb_id,
                                query_embedding=ss.protein_embedding,
                                k=k if not use_threshold else None,
                                similarity_threshold=threshold_param,
                                return_similarities=True,
                                allow_self_as_template=True  # Allow self when using PDB ID
                            )
                            
                            ss.template_results = templates
                            
                            if templates:
                                st.success(f"Found {len(templates)} templates")
                            else:
                                st.warning("No templates found")
                                
                        except Exception as e:
                            st.error(f"Error finding templates: {str(e)}")
            
            with col2:
                st.subheader("Results")
                
                if ss.get("template_results"):
                    st.metric("Templates Found", len(ss.template_results))
                    
                    # Template table
                    template_data = []
                    for i, (pdb_id, sim) in enumerate(ss.template_results[:10]):
                        template_data.append({
                            "Rank": i+1,
                            "PDB ID": pdb_id,
                            "Similarity": f"{sim:.3f}"
                        })
                    st.table(template_data)
                else:
                    st.info("No templates found yet")
            
            # Query ligand section
            st.subheader("Query Ligand")
            
            ligand_input_method = st.radio("Input method:", ["Upload File", "SMILES"], horizontal=True)
            
            if ligand_input_method == "Upload File":
                ligand_file = st.file_uploader("Upload Ligand", type=["sdf", "mol"])
                if ligand_file and st.button("Load Ligand"):
                    try:
                        ligand_bytes = ligand_file.read()
                        supplier = Chem.SDMolSupplier()
                        supplier.SetData(ligand_bytes)
                        
                        for mol in supplier:
                            if mol:
                                ss.query_mol = Chem.RemoveHs(mol)
                                st.success("Ligand loaded")
                                break
                    except Exception as e:
                        st.error(f"Error loading ligand: {str(e)}")
            else:
                smiles_input = st.text_input("SMILES", placeholder="e.g., CCO")
                if smiles_input and st.button("Create from SMILES"):
                    try:
                        mol = Chem.MolFromSmiles(smiles_input)
                        if mol:
                            AllChem.Compute2DCoords(mol)
                            mol = Chem.AddHs(mol)
                            AllChem.EmbedMolecule(mol)
                            AllChem.MMFFOptimizeMolecule(mol)
                            ss.query_mol = Chem.RemoveHs(mol)
                            st.success("Ligand created")
                        else:
                            st.error("Invalid SMILES")
                    except Exception as e:
                        st.error(f"Error creating ligand: {str(e)}")
            
            # Display ligand if available
            if ss.get("query_mol"):
                st.success("Query ligand loaded")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Atoms", ss.query_mol.GetNumAtoms())
                with col2:
                    st.metric("Bonds", ss.query_mol.GetNumBonds())
                
                display_molecule(ss.query_mol, width=300, height=250)
            
            # Ready check
            if ss.get("template_results") and ss.get("query_mol"):
                st.success("Ready for pose generation!")
    
    # Tab 3: Pose Generation  
    with tab3:
        st.header("Pose Generation")
        
        # Check prerequisites
        if not ss.get("template_results"):
            st.warning("No templates available. Please find templates first.")
            return
        
        if not ss.get("query_mol"):
            st.warning("No query molecule available. Please upload a ligand first.")
            return
        
        # Load template molecules
        template_pdbs = [result[0] for result in ss.template_results[:5]]
        
        with st.spinner("Loading template molecules..."):
            template_mols = load_templates_from_processed_sdf(template_pdbs)
        
        if template_mols:
            st.subheader(f"Templates ({len(template_mols)})")
            
            # Template overview
            template_data = []
            for i, mol in enumerate(template_mols):
                template_data.append({
                    "#": i+1,
                    "PDB ID": mol.GetProp('_Name'),
                    "Atoms": mol.GetNumAtoms()
                })
            st.table(template_data)
            
            # Template selection
            template_names = [mol.GetProp("_Name") for mol in template_mols]
            selected_template_name = st.selectbox("Select template", template_names)
            
            # Find selected template
            template_mol = None
            for mol in template_mols:
                if mol.GetProp("_Name") == selected_template_name:
                    template_mol = mol
                    break
            
            if template_mol:
                ss.template_mol = template_mol
                
                st.subheader("Template Molecule")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Atoms", template_mol.GetNumAtoms())
                with col2:
                    st.metric("Bonds", template_mol.GetNumBonds())
                
                display_molecule_stmol(template_mol)
                
                st.subheader("Query Molecule")
                display_molecule_stmol(ss.query_mol)
                
                # Generate conformers
                st.subheader("Step 1: Generate Conformers")
                
                n_confs = st.slider("Number of Conformers", 10, 500, 100, 10)
                n_workers_local = st.slider("Number of Workers", 1, 8, 4)
                
                if st.button("Generate Conformers"):
                    with st.spinner("Generating conformers..."):
                        try:
                            conformers, mcs_idx = generate_conformers(
                                ss.query_mol, 
                                [template_mol], 
                                n_confs, 
                                n_workers_local
                            )
                            
                            if conformers is not None:
                                ss.conformers = conformers
                                ss.mcs_indices = {selected_template_name: mcs_idx}
                                
                                n_generated = conformers.GetNumConformers()
                                st.success(f"Generated {n_generated}/{n_confs} conformers")
                                
                                if isinstance(mcs_idx, dict):
                                    if mcs_idx.get("central_atom_fallback", False):
                                        st.warning("Used central atom positioning fallback")
                                    else:
                                        st.info(f"MCS found with {mcs_idx.get('atom_count', 0)} atoms")
                            else:
                                st.error("Failed to generate conformers")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                # Pose selection
                if "conformers" in ss and ss.conformers:
                    st.subheader("Step 2: Pose Selection")
                    
                    if st.button("Score and Select Poses"):
                        with st.spinner("Scoring poses..."):
                            try:
                                results = scoring_select_best(ss.conformers, template_mol, n_workers=n_workers_local)
                                
                                if results:
                                    ss.poses = results
                                    st.success("Poses scored and selected!")
                                    
                                    # Display results
                                    score_data = []
                                    for method, (mol, scores) in results.items():
                                        score_data.append({
                                            "Method": method,
                                            "Shape Score": f"{scores.get('shape_score', 0):.3f}",
                                            "Color Score": f"{scores.get('color_score', 0):.3f}"
                                        })
                                    st.table(score_data)
                                else:
                                    st.error("Failed to score poses")
                            except Exception as e:
                                st.error(f"Error scoring poses: {str(e)}")
                    
                    # Crystal structure comparison
                    st.subheader("Step 3: Crystal Structure (Optional)")
                    crystal_file = st.file_uploader("Upload Crystal Ligand", type=["sdf", "mol"])
                    
                    if crystal_file:
                        try:
                            crystal_bytes = crystal_file.read()
                            crystal_supplier = Chem.SDMolSupplier()
                            crystal_supplier.SetData(crystal_bytes)
                            
                            for mol in crystal_supplier:
                                if mol:
                                    ss.crystal_mol = mol
                                    st.success("Crystal structure loaded")
                                    break
                        except Exception as e:
                            st.error(f"Error loading crystal: {str(e)}")
        
        # Navigation
        if "poses" in ss and ss.poses:
            st.info("Proceed to the Results tab to view poses")
    
    # Tab 4: Results
    with tab4:
        st.header("Results")
        
        if not ss.poses:
            st.info("No results to display. Generate poses first.")
            return
        
        # Simple results display
        st.subheader("Generated Poses")
        
        for method, (mol, scores) in ss.poses.items():
            st.write(f"**{method.capitalize()} Pose**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("2D Structure:")
                display_molecule(mol, width=300, height=200)
            
            with col2:
                st.write("Scores:")
                for score_name, value in scores.items():
                    st.write(f"{score_name}: {value:.3f}")
                
                # RMSD if available
                if mol.HasProp("rmsd_to_crystal"):
                    st.write(f"**RMSD to Crystal: {mol.GetProp('rmsd_to_crystal')} Å**")
            
            st.divider()
        
        # Download options
        st.subheader("Download Results")
        
        combined_sdf = io.StringIO()
        writer = Chem.SDWriter(combined_sdf)
        
        try:
            for method, (mol, scores) in ss.poses.items():
                mol_copy = ensure_rdkit_mol(mol)
                if mol_copy:
                    writer.write(mol_copy)
        except Exception as e:
            st.error(f"Error creating download: {str(e)}")
        finally:
            writer.close()
        
        st.download_button(
            label="Download All Results (SDF)",
            data=combined_sdf.getvalue(),
            file_name="templ_results.sdf",
            mime="chemical/x-mdl-sdfile"
        )
    
    # Simplified sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Database settings
        st.subheader("Database")
        new_embedding_path = st.text_input(
            "Embedding Database Path",
            value=st.session_state.embedding_path
        )
        
        if new_embedding_path != st.session_state.embedding_path:
            st.session_state.embedding_path = new_embedding_path
            st.session_state.embedding_path_changed = True
        
        if os.path.exists(st.session_state.embedding_path):
            st.success("Database found")
        else:
            st.error("Database not found")
        
        # System status
        st.subheader("System Status")
        
        if ss.get("embedding_manager"):
            st.success("EmbeddingManager ready")
        else:
            st.error("EmbeddingManager not initialized")
        
        capabilities = get_system_capabilities()
        if capabilities["cuda_available"]:
            st.success(f"GPU Available ({capabilities['gpu_count']})")
        else:
            st.info("CPU Only")

if __name__ == "__main__":
    main()
