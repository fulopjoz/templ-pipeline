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
                    logger.debug(
                        "Using original molecular structure for image generation"
                    )
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
            img = Draw.MolToImage(
                mol_copy, size=(width, height), highlightAtoms=highlight_atoms
            )
        else:
            img = Draw.MolToImage(mol_copy, size=(width, height))
        return img
    except Exception as e:
        logger.error(f"Error generating molecule image: {e}")
        return None


@st.cache_data(ttl=1800, show_spinner=False)  # 30 minute cache for molecular displays
def cached_display_molecule_data(
    mol_binary, width=400, height=300, highlight_atoms=None
):
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

        try:
            mol_binary = mol_work.ToBinary()
        except Exception as e:
            logger.warning(f"Failed to convert molecule to binary: {e}")
            # If ToBinary fails, try to recreate from SMILES
            if mol.HasProp("original_smiles"):
                try:
                    original_smiles = mol.GetProp("original_smiles")
                    logger.debug(
                        f"Attempting to recreate molecule from original SMILES: {original_smiles}"
                    )
                    fallback_mol = Chem.MolFromSmiles(original_smiles)
                    if fallback_mol:
                        mol_binary = fallback_mol.ToBinary()
                        logger.debug(
                            "Successfully recreated molecule from original SMILES"
                        )
                    else:
                        raise Exception(
                            "Could not recreate molecule from original SMILES"
                        )
                except Exception as e2:
                    logger.error(f"Failed to recreate molecule from SMILES: {e2}")
                    st.error(f"Error converting molecule for display: {e}")
                    return
            else:
                st.error(f"Error converting molecule for display: {e}")
                return

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


def safe_get_mcs_mol(mcs_data):
    """Safely extract MCS molecule from various data formats with enhanced error handling"""
    if mcs_data is None:
        logger.debug("safe_get_mcs_mol: mcs_data is None")
        return None

    logger.debug(f"safe_get_mcs_mol: Processing mcs_data of type {type(mcs_data)}")

    try:
        from .molecular_utils import get_rdkit_modules
        Chem, AllChem, Draw = get_rdkit_modules()
    except ImportError as e:
        logger.error(f"safe_get_mcs_mol: RDKit import failed: {e}")
        return None

    try:
        # Handle different MCS data formats
        if isinstance(mcs_data, dict):
            logger.debug("safe_get_mcs_mol: Processing dictionary data")
            
            # Try to get MCS molecule directly
            if "mcs_mol" in mcs_data:
                mol = mcs_data["mcs_mol"]
                if hasattr(mol, "HasProp") and hasattr(mol, "GetNumAtoms"):
                    try:
                        if mol.GetNumAtoms() > 0:
                            return mol
                    except:
                        pass
            
            # Try to get from SMARTS
            if "smarts" in mcs_data:
                smarts = mcs_data["smarts"]
                logger.debug(f"safe_get_mcs_mol: Creating molecule from SMARTS: {smarts}")
                mol = Chem.MolFromSmarts(smarts)
                if mol is not None:
                    try:
                        # Try to sanitize and add 2D coordinates
                        Chem.SanitizeMol(mol)
                        AllChem.Compute2DCoords(mol)
                        return mol
                    except Exception as e:
                        logger.warning(f"safe_get_mcs_mol: SMARTS sanitization failed: {e}")
                        # Return unsanitized molecule
                        return mol
            
            # Try to get from mcs_smarts key
            if "mcs_smarts" in mcs_data:
                smarts = mcs_data["mcs_smarts"]
                logger.debug(f"safe_get_mcs_mol: Creating molecule from mcs_smarts: {smarts}")
                mol = Chem.MolFromSmarts(smarts)
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol)
                        AllChem.Compute2DCoords(mol)
                        return mol
                    except Exception as e:
                        logger.warning(f"safe_get_mcs_mol: mcs_smarts sanitization failed: {e}")
                        return mol

        # Handle list/tuple format (legacy)
        elif isinstance(mcs_data, (list, tuple)) and len(mcs_data) > 0:
            logger.debug("safe_get_mcs_mol: Processing list/tuple data")
            mol = mcs_data[0]
            if hasattr(mol, "HasProp") and hasattr(mol, "GetNumAtoms"):
                try:
                    if mol.GetNumAtoms() > 0:
                        return mol
                except:
                    pass

        # Handle string format (SMARTS)
        elif isinstance(mcs_data, str):
            logger.debug(f"safe_get_mcs_mol: Processing string data: {mcs_data}")
            mol = Chem.MolFromSmarts(mcs_data)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                    AllChem.Compute2DCoords(mol)
                    return mol
                except Exception as e:
                    logger.warning(f"safe_get_mcs_mol: String SMARTS sanitization failed: {e}")
                    return mol

        # Handle direct molecule object
        elif hasattr(mcs_data, "HasProp") and hasattr(mcs_data, "GetNumAtoms"):
            try:
                if mcs_data.GetNumAtoms() > 0:
                    return mcs_data
            except:
                pass

    except Exception as e:
        logger.warning(f"safe_get_mcs_mol: Error processing MCS data: {e}")

    logger.debug("safe_get_mcs_mol: Unable to extract valid MCS molecule")
    return None
