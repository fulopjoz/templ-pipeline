# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Visualization utility functions for TEMPL Pipeline UI

Contains molecule display and image generation functions.
"""

import logging
from typing import Any, Optional

import streamlit as st

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


@st.cache_data(ttl=300, show_spinner=False)
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
        except (
            Chem.rdchem.KekulizeException,
            Chem.rdchem.AtomValenceException,
            ValueError,
        ):
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
    except (ValueError, TypeError, RuntimeError):
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
                    except (AttributeError, ValueError, RuntimeError):
                        pass

            # Try to get from SMARTS
            if "smarts" in mcs_data:
                smarts = mcs_data["smarts"]
                logger.debug(
                    f"safe_get_mcs_mol: Creating molecule from SMARTS: {smarts}"
                )
                mol = Chem.MolFromSmarts(smarts)
                if mol is not None:
                    try:
                        # Try to sanitize and add 2D coordinates
                        Chem.SanitizeMol(mol)
                        AllChem.Compute2DCoords(mol)
                        return mol
                    except Exception as e:
                        logger.warning(
                            f"safe_get_mcs_mol: SMARTS sanitization failed: {e}"
                        )
                        # Return unsanitized molecule
                        return mol

            # Try to get from mcs_smarts key
            if "mcs_smarts" in mcs_data:
                smarts = mcs_data["mcs_smarts"]
                logger.debug(
                    f"safe_get_mcs_mol: Creating molecule from mcs_smarts: {smarts}"
                )
                mol = Chem.MolFromSmarts(smarts)
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol)
                        AllChem.Compute2DCoords(mol)
                        return mol
                    except Exception as e:
                        logger.warning(
                            f"safe_get_mcs_mol: mcs_smarts sanitization failed: {e}"
                        )
                        return mol

        # Handle list/tuple format (legacy)
        elif isinstance(mcs_data, (list, tuple)) and len(mcs_data) > 0:
            logger.debug("safe_get_mcs_mol: Processing list/tuple data")
            mol = mcs_data[0]
            if hasattr(mol, "HasProp") and hasattr(mol, "GetNumAtoms"):
                try:
                    if mol.GetNumAtoms() > 0:
                        return mol
                except (AttributeError, ValueError, RuntimeError):
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
                    logger.warning(
                        f"safe_get_mcs_mol: String SMARTS sanitization failed: {e}"
                    )
                    return mol

        # Handle direct molecule object
        elif hasattr(mcs_data, "HasProp") and hasattr(mcs_data, "GetNumAtoms"):
            try:
                if mcs_data.GetNumAtoms() > 0:
                    return mcs_data
            except (AttributeError, ValueError, RuntimeError):
                pass

    except Exception as e:
        logger.warning(f"safe_get_mcs_mol: Error processing MCS data: {e}")

    logger.debug("safe_get_mcs_mol: Unable to extract valid MCS molecule")
    return None


def get_molecule_from_session(
    session_manager, key: str, fallback_smiles: str = None
) -> Optional[Any]:
    """Robustly retrieve molecule from session with memory manager integration

    Args:
        session_manager: SessionManager instance
        key: Session key for the molecule
        fallback_smiles: Optional SMILES string as fallback

    Returns:
        RDKit molecule object or None
    """
    try:
        from rdkit import Chem
    except ImportError:
        logger.error("RDKit not available for molecule retrieval")
        return None

    # First try to reconstruct from the latest INPUT_SMILES when key is QUERY_MOL
    try:
        from ..config.constants import SESSION_KEYS as _SK

        if key == _SK["QUERY_MOL"]:
            current_smiles = session_manager.get(_SK["INPUT_SMILES"])
            if current_smiles:
                mol = Chem.MolFromSmiles(current_smiles)
                if mol:
                    try:
                        mol.SetProp("original_smiles", current_smiles)
                        mol.SetProp("input_method", "smiles")
                    except Exception:
                        pass
                    return mol
    except Exception:
        pass

    # Then try to get from session by key
    mol_data = session_manager.get(key)

    if mol_data is not None:
        # If it's already an RDKit molecule, return it
        if hasattr(mol_data, "ToBinary") and hasattr(mol_data, "GetNumAtoms"):
            if mol_data.GetNumAtoms() > 0:
                return mol_data

        # If it's binary data, try to deserialize
        if isinstance(mol_data, bytes):
            try:
                mol = Chem.Mol(mol_data)
                if mol and mol.GetNumAtoms() > 0:
                    return mol
            except Exception as e:
                logger.warning(f"Failed to deserialize molecule from binary: {e}")

        # If it's a string, try as SMILES
        if isinstance(mol_data, str):
            try:
                mol = Chem.MolFromSmiles(mol_data)
                if mol:
                    return mol
            except Exception as e:
                logger.warning(f"Failed to create molecule from SMILES: {e}")

    # Try memory manager fallback for specific keys
    try:
        from ..config.constants import SESSION_KEYS
        from ..core.memory_manager import get_memory_manager

        memory_manager = get_memory_manager()

        # Map session keys to memory manager keys
        memory_key_map = {
            SESSION_KEYS["QUERY_MOL"]: "query",
            SESSION_KEYS["TEMPLATE_USED"]: "template",
        }

        memory_key = memory_key_map.get(key)
        if memory_key:
            mol = memory_manager.get_molecule(memory_key)
            if mol:
                logger.debug(f"Retrieved molecule {key} from memory manager")
                return mol

    except Exception as e:
        logger.warning(f"Memory manager fallback failed for {key}: {e}")

    # Enhanced fallback for template molecules using template_info
    if key == SESSION_KEYS.get("TEMPLATE_USED"):
        try:
            template_info = session_manager.get(SESSION_KEYS.get("TEMPLATE_INFO"))
            if template_info and isinstance(template_info, dict):
                template_smiles = template_info.get("template_smiles")
                if template_smiles:
                    try:
                        mol = Chem.MolFromSmiles(template_smiles)
                        if mol:
                            logger.debug(
                                f"Created template molecule from template_info SMILES: {template_smiles}"
                            )
                            return mol
                    except Exception as e:
                        logger.warning(
                            f"Failed to create template from template_info SMILES: {e}"
                        )
        except Exception as e:
            logger.warning(f"Template info fallback failed: {e}")

    # Final fallback to SMILES
    if fallback_smiles:
        try:
            mol = Chem.MolFromSmiles(fallback_smiles)
            if mol:
                logger.debug(
                    f"Created molecule from fallback SMILES: {fallback_smiles}"
                )
                return mol
        except Exception as e:
            logger.warning(f"Fallback SMILES failed: {e}")

    logger.debug(f"Unable to retrieve molecule for key: {key}")
    return None


def create_mcs_molecule_from_info(mcs_info: Any) -> Optional[Any]:
    """Create MCS molecule from various MCS info formats

    Args:
        mcs_info: MCS information in various formats

    Returns:
        RDKit molecule object for MCS or None
    """
    if not mcs_info:
        return None

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        logger.error("RDKit not available for MCS processing")
        return None

    # Collect all possible SMARTS patterns
    smarts_candidates = []

    if isinstance(mcs_info, dict):
        # Try different SMARTS keys in order of preference
        for smarts_key in ["smarts", "mcs_smarts", "mcs_pattern"]:
            if smarts_key in mcs_info and mcs_info[smarts_key]:
                smarts_value = mcs_info[smarts_key]
                if isinstance(smarts_value, str) and len(smarts_value.strip()) > 0:
                    smarts_candidates.append(smarts_value.strip())

        # Also check if there's a pre-created molecule
        if "mcs_mol" in mcs_info:
            mcs_mol = mcs_info["mcs_mol"]
            if hasattr(mcs_mol, "GetNumAtoms"):
                try:
                    if mcs_mol.GetNumAtoms() > 0:
                        return mcs_mol
                except Exception:
                    pass

    elif isinstance(mcs_info, str):
        smarts_value = mcs_info.strip()
        if len(smarts_value) > 0:
            smarts_candidates.append(smarts_value)
    elif hasattr(mcs_info, "GetNumAtoms"):
        # It's already a molecule object
        try:
            if mcs_info.GetNumAtoms() > 0:
                return mcs_info
        except Exception:
            pass

    # Try each SMARTS pattern
    for smarts in smarts_candidates:
        if smarts and isinstance(smarts, str) and len(smarts.strip()) > 0:
            try:
                mol = Chem.MolFromSmarts(smarts.strip())
                if mol and mol.GetNumAtoms() > 0:
                    try:
                        # Try to sanitize and add 2D coordinates
                        Chem.SanitizeMol(mol)
                        AllChem.Compute2DCoords(mol)
                        logger.debug(
                            f"Successfully created MCS molecule from SMARTS: {smarts}"
                        )
                        return mol
                    except Exception as e:
                        logger.debug(
                            f"Could not sanitize MCS molecule, returning unsanitized: {e}"
                        )
                        # Return unsanitized molecule if sanitization fails
                        return mol
                else:
                    logger.debug(f"Invalid SMARTS pattern (no atoms): {smarts}")
            except Exception as e:
                logger.debug(f"Failed to create molecule from SMARTS '{smarts}': {e}")
                continue

    logger.debug("No valid SMARTS pattern found in MCS info")
    return None
