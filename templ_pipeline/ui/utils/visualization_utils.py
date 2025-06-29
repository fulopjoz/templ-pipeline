"""
Visualization utility functions for TEMPL Pipeline UI

Contains molecule display and image generation functions.
"""

import streamlit as st
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


def generate_molecule_image(mol_binary, width=400, height=300, highlight_atoms=None):
    """Generate molecule image from binary representation"""
    try:
        from .molecular_utils import get_rdkit_modules
        
        Chem, AllChem, Draw = get_rdkit_modules()
        mol = Chem.Mol(mol_binary)
        
        # SMART FIX: Use original molecular structure for visualization if available
        if mol.HasProp("original_smiles"):
            try:
                original_smiles = mol.GetProp("original_smiles")
                clean_mol = Chem.MolFromSmiles(original_smiles)
                if clean_mol:
                    mol = clean_mol
                    logger.debug("Using original molecular structure for image generation")
            except Exception as e:
                logger.debug(f"Could not use original SMILES for image generation: {e}")
        
        # Proper sanitization and hydrogen removal
        mol_copy = Chem.RemoveHs(mol)
        
        # Ensure proper sanitization for visualization
        try:
            Chem.SanitizeMol(mol_copy)
        except Exception as e:
            logger.debug(f"Sanitization failed during image generation: {e}")
            # Continue with unsanitized molecule
        
        # Ensure we have valid 2D coordinates
        if mol_copy.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol_copy)
        
        mol_copy = Chem.MolToSmiles(mol_copy)
        mol_copy = Chem.MolFromSmiles(mol_copy)

        if highlight_atoms:
            img = Draw.MolToImage(mol_copy, size=(width, height), highlightAtoms=highlight_atoms)
        else:
            img = Draw.MolToImage(mol_copy, size=(width, height))
        return img
    except Exception as e:
        logger.error(f"Error generating molecule image: {e}")
        return None


@st.cache_data(ttl=1800, show_spinner=False)  # 30 minute cache for molecular displays
def cached_display_molecule_data(mol_binary, width=400, height=300, highlight_atoms=None):
    """Cached wrapper for molecular display data processing"""
    try:
        from .molecular_utils import get_rdkit_modules
        
        Chem, AllChem, Draw = get_rdkit_modules()
        mol = Chem.Mol(mol_binary)
        
        # Create safe copy for processing
        mol_copy = Chem.RemoveHs(mol)
        
        # Generate coordinates if needed
        if mol_copy.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol_copy)
        
        return mol_copy.ToBinary()
    except Exception as e:
        logger.error(f"Error in cached molecular display: {e}")
        return None


def display_molecule(mol, width=400, height=300, title="", highlight_atoms=None):
    """Display molecule as 2D image with optional atom highlighting"""
    if mol is None:
        return
    
    try:
        from .molecular_utils import get_rdkit_modules
        
        # Convert mol to binary for caching
        # sanitize the molecule carefully
        Chem, AllChem, Draw = get_rdkit_modules()
        
        # SMART FIX: Use original molecular structure for visualization if available
        mol_work = mol
        if mol.HasProp("original_smiles"):
            try:
                original_smiles = mol.GetProp("original_smiles")
                clean_mol = Chem.MolFromSmiles(original_smiles)
                if clean_mol:
                    mol_work = clean_mol
                    logger.debug("Using original molecular structure for visualization")
            except Exception as e:
                logger.debug(f"Could not use original SMILES for visualization: {e}")
                # Fall back to coordinate-manipulated molecule
                mol_work = Chem.Mol(mol)
        else:
            # Create a working copy to avoid modifying the original
            mol_work = Chem.Mol(mol)
        
        try:
            Chem.SanitizeMol(mol_work)
        except:
            # If sanitization fails, try without it
            mol_work = mol
        
        mol_binary = mol_work.ToBinary()
        
        # Ensure highlight_atoms is hashable for caching (convert list to tuple)
        if highlight_atoms is not None:
            highlight_atoms = tuple(highlight_atoms)
            
        img = generate_molecule_image(mol_binary, width, height, highlight_atoms)
        
        if img:
            if title:
                st.write(f"**{title}**")
            st.image(img)
    except Exception as e:
        st.error(f"Error displaying molecule: {e}")


def get_mcs_mol(mol1, mol2):
    """Get MCS as a molecule object"""
    try:
        from .molecular_utils import get_rdkit_modules
        
        Chem, AllChem, Draw = get_rdkit_modules()
        mcs = Chem.rdFMCS.FindMCS([mol1, mol2])
        if mcs.numAtoms > 0:
            mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
            if mcs_mol:
                AllChem.Compute2DCoords(mcs_mol)
                return mcs_mol, mcs.smartsString
    except:
        pass
    return None, None


def safe_get_mcs_mol(mcs_info):
    """Safe access to MCS molecule from mcs_info with debugging"""
    try:        
        if isinstance(mcs_info, (list, tuple)) and len(mcs_info) > 0:
            return mcs_info[0]
        elif isinstance(mcs_info, dict):
            # Handle new dict format from pipeline
            if 'mcs_mol' in mcs_info:
                return mcs_info['mcs_mol']
            elif 'smarts' in mcs_info:
                # Create mol from SMARTS if available
                from .molecular_utils import get_rdkit_modules
                
                Chem, AllChem, Draw = get_rdkit_modules()
                smarts = mcs_info['smarts']
                mol = Chem.MolFromSmarts(smarts)
                if mol:
                    try:
                        Chem.SanitizeMol(mol)
                        AllChem.Compute2DCoords(mol)
                        return mol
                    except:
                        return None
        return None
    except Exception as e:
        logger.warning(f"MCS access failed: {e}")
        return None
