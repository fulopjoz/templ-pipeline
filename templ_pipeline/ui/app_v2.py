"""
TEMPL Pipeline Web Application - Version 2.0

Hybrid solution: Working layout (direct execution) + Working pipeline (real functionality)
"""

import streamlit as st

# PHASE 1: IMMEDIATE PAGE CONFIG - Set page config first (exactly like working version)
st.set_page_config(
    page_title="TEMPL Pipeline",
    page_icon="🧪",
    layout="wide"
)

# PHASE 2: IMMEDIATE LAYOUT FIXES - Apply immediately (exactly like working version)
from templ_pipeline.ui.ui.styles.early_layout import apply_layout_fixes
apply_layout_fixes()

# PHASE 3: Essential imports after layout fixes
import logging
import sys
import multiprocessing
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PHASE 4: Import pipeline functionality (after layout fixes)
from templ_pipeline.ui.app import (
    run_pipeline, 
    validate_smiles_input, 
    validate_sdf_input,
    load_templates_from_uploaded_sdf,
    save_uploaded_file,
    create_best_poses_sdf,
    create_all_conformers_sdf,
    display_molecule,
    get_rdkit_modules
)

# Initialize session state if needed (simple, direct)
if 'app_initialized' not in st.session_state:
    st.session_state.app_initialized = True
    st.session_state.query_mol = None
    st.session_state.input_smiles = None
    st.session_state.protein_pdb_id = None
    st.session_state.protein_file_path = None
    st.session_state.custom_templates = None
    st.session_state.poses = {}

# DIRECT EXECUTION - NO FUNCTION WRAPPING (like working test_layout_fix.py)

# Header
st.title("TEMPL Pipeline")
st.markdown("**Template-based Protein-Ligand Pose Prediction**")

# Layout test
st.info("🧪 Layout Test: This should be full-width from first load")

# Test columns (exactly like working version)
col1, col2, col3 = st.columns(3)
with col1:
    st.success("Left column - should be full width immediately")
with col2:
    st.info("Middle column - no narrow->wide transition")  
with col3:
    st.warning("Right column - consistent width on first load")

# Add divider
st.divider()

# What it does section
st.markdown("### What it does")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **TEMPL Pipeline Features:**
    - Predicts 3D binding poses for small molecules
    - Uses template-guided conformer generation
    - Provides comprehensive scoring
    """)

with col2:
    st.markdown("""
    **How it works:**
    1. Enter your molecule (SMILES or file)
    2. Provide protein target
    3. Get ranked poses with scores
    """)

st.divider()

# Enhanced Input section with REAL functionality
st.header("Input Configuration")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### Query Molecule")
    
    input_method = st.radio(
        "Input method:",
        ["SMILES", "Upload File"],
        horizontal=True,
        key="mol_input_method"
    )
    
    if input_method == "SMILES":
        smiles = st.text_input(
            "SMILES String", 
            placeholder="Enter SMILES (e.g., C1CC(=O)N(C1)CC(=O)N )",
            key="smiles_input"
        )
        
        if smiles:
            # Use REAL validation function
            valid, msg, mol_data = validate_smiles_input(smiles)
            if valid:
                # Convert from cached binary format if needed
                if mol_data is not None:
                    Chem, AllChem, Draw = get_rdkit_modules()
                    mol = Chem.Mol(mol_data)
                else:
                    mol = None
                st.session_state.query_mol = mol
                st.session_state.input_smiles = smiles
                st.success(f"✅ {msg}")
                # Show molecule preview
                if mol:
                    display_molecule(mol, width=280, height=200)
            else:
                st.error(f"❌ {msg}")
    else:
        uploaded_file = st.file_uploader(
            "Upload SDF/MOL File",
            type=["sdf", "mol"],
            key="mol_file_upload"
        )
        
        if uploaded_file is not None:
            if uploaded_file.size > 5 * 1024 * 1024:  # 5MB limit
                st.error("File too large (max 5MB)")
            else:
                # Use REAL validation function
                valid, msg, mol = validate_sdf_input(uploaded_file)
                if valid:
                    st.session_state.query_mol = mol
                    Chem, AllChem, Draw = get_rdkit_modules()
                    st.session_state.input_smiles = Chem.MolToSmiles(mol)
                    st.success(f"✅ {msg}")
                    # Show molecule preview
                    display_molecule(mol, width=280, height=200)
                else:
                    st.error(f"❌ {msg}")

with col2:
    st.markdown("#### Target Protein")
    
    protein_method = st.radio(
        "Input method:",
        ["PDB ID", "Upload File", "Custom Templates"],
        horizontal=True,
        key="prot_input_method"
    )
    
    if protein_method == "PDB ID":
        pdb_id = st.text_input(
            "PDB ID",
            placeholder="Enter 4-character PDB ID (e.g., 1iky)",
            key="pdb_id_input"
        )
        
        if pdb_id:
            if len(pdb_id) == 4 and pdb_id.isalnum():
                st.session_state.protein_pdb_id = pdb_id.lower()
                st.session_state.protein_file_path = None
                st.session_state.custom_templates = None
                st.success(f"✅ PDB ID: {pdb_id.upper()}")
            else:
                st.error("❌ PDB ID must be 4 alphanumeric characters")
                
    elif protein_method == "Upload File":
        pdb_file = st.file_uploader(
            "Upload PDB File",
            type=["pdb"],
            key="pdb_file_upload"
        )
        
        if pdb_file is not None:
            if pdb_file.size > 5 * 1024 * 1024:  # 5MB limit
                st.error("File too large (max 5MB)")
            else:
                # Use REAL file handling function
                file_path = save_uploaded_file(pdb_file)
                st.session_state.protein_file_path = file_path
                st.session_state.protein_pdb_id = None
                st.session_state.custom_templates = None
                st.success(f"✅ PDB file uploaded: {pdb_file.name}")
            
    else:  # Custom Templates
        st.markdown("Upload SDF with template molecules for MCS-based pose generation")
        template_file = st.file_uploader(
            "Upload Template SDF",
            type=["sdf"],
            key="template_file_upload"
        )
        
        if template_file is not None:
            if template_file.size > 10 * 1024 * 1024:  # 10MB limit
                st.error("File too large (max 10MB)")
            else:
                # Use REAL template loading function
                templates = load_templates_from_uploaded_sdf(template_file)
                if templates:
                    st.session_state.custom_templates = templates
                    st.session_state.protein_pdb_id = None
                    st.session_state.protein_file_path = None
                    st.success(f"✅ Loaded {len(templates)} template molecules")
                else:
                    st.error("❌ No valid molecules found in SDF")

# Check if we have valid input
protein_input = st.session_state.get('protein_pdb_id') or st.session_state.get('protein_file_path')
custom_templates = st.session_state.get('custom_templates')
molecule_input = st.session_state.get('input_smiles')
ready = molecule_input and (protein_input or custom_templates)

# Advanced Settings (simple version)
with st.expander("Advanced Settings", expanded=False):
    use_aligned_poses = st.radio(
        "Pose Alignment Mode:",
        ["Aligned Poses", "Original Geometry"],
        index=0,
        help="Aligned poses are positioned relative to templates. Original geometry preserves conformer shape.",
        key="pose_alignment_mode"
    ) == "Aligned Poses"

# Action button with REAL pipeline execution
if ready:
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 PREDICT POSES", type="primary", use_container_width=True):
            # REAL PIPELINE EXECUTION - Using the actual run_pipeline function
            logger.info("Running REAL TEMPL pipeline...")
            
            poses = run_pipeline(
                molecule_input, 
                protein_input, 
                custom_templates,
                use_aligned_poses=use_aligned_poses,
                max_templates=100,  # Default
                similarity_threshold=None
            )
            
            if poses:
                st.session_state.poses = poses
                st.success("✅ Pose prediction completed!")
                logger.info("Pipeline execution successful")
            else:
                st.error("❌ Pipeline failed. Check error messages above.")

# Results Section with REAL functionality
if st.session_state.get('poses'):
    st.divider()
    st.header("🎯 Prediction Results")
    
    poses = st.session_state.poses
    
    # Find best pose by combo score
    best_method, (best_mol, best_scores) = max(poses.items(), 
                                             key=lambda x: x[1][1].get('combo_score', x[1][1].get('combo', 0)))
    
    shape_score = best_scores.get('shape_score', best_scores.get('shape', 0))
    color_score = best_scores.get('color_score', best_scores.get('color', 0))
    combo_score = best_scores.get('combo_score', best_scores.get('combo', 0))
    
    # Show results
    st.markdown("### Best Predicted Pose")
    st.info(f"Best method: {best_method}")
    
    # Score metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Shape Similarity", f"{shape_score:.3f}")
    with col2:
        st.metric("Pharmacophore", f"{color_score:.3f}")
    with col3:
        st.metric("Overall Score", f"{combo_score:.3f}")
    
    # Quality assessment
    if combo_score >= 0.8:
        quality = "Excellent - High confidence pose"
        color = "green"
    elif combo_score >= 0.6:
        quality = "Good - Reliable pose prediction"
        color = "blue"
    elif combo_score >= 0.4:
        quality = "Fair - Moderate confidence"
        color = "orange"
    else:
        quality = "Poor - Low confidence, consider alternatives"
        color = "red"
    
    st.markdown(f"**Quality Assessment:** :{color}[{quality}]")
    
    # Download section with REAL functionality
    st.markdown("### 📥 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Use REAL SDF creation function
        sdf_data, file_name = create_best_poses_sdf(poses)
        st.download_button(
            f"📄 Best Poses ({len(poses)})",
            data=sdf_data,
            file_name=file_name,
            mime="chemical/x-mdl-sdfile",
            help="Top scoring poses for each method",
            use_container_width=True
        )
    
    with col2:
        if hasattr(st.session_state, 'all_ranked_poses') and st.session_state.all_ranked_poses:
            # Use REAL SDF creation function
            all_sdf_data, all_file_name = create_all_conformers_sdf()
            st.download_button(
                f"📊 All Conformers ({len(st.session_state.all_ranked_poses)})",
                data=all_sdf_data,
                file_name=all_file_name,
                mime="chemical/x-mdl-sdfile",
                help="All generated conformers ranked by score",
                use_container_width=True
            )
        else:
            st.button(
                "📊 All Conformers (N/A)",
                disabled=True,
                help="All ranked poses not available",
                use_container_width=True
            )

# Wide test element (exactly like working version)
st.divider()
st.markdown("### Width Test Elements")

st.code("""
This code block should span the full browser width from the moment the page loads.
If you see a narrow layout that then expands to full width, the fix needs adjustment.
The layout should be consistent on first load and after refresh.
""")

st.success("✅ If this layout looks the same on first load and after refresh, the fix is working!")
