# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Molecular utility functions for TEMPL Pipeline UI

Contains molecular validation, processing, and RDKit integration functions.
"""

import streamlit as st
import logging
from typing import Tuple, Optional, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

# Global variables for lazy loading
_rdkit_modules = None


@st.cache_resource
def get_rdkit_modules():
    """Lazy load RDKit modules to improve startup performance"""
    global _rdkit_modules
    if _rdkit_modules is None:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem, Draw

        RDLogger.DisableLog("rdApp.*")
        _rdkit_modules = (Chem, AllChem, Draw)
    return _rdkit_modules


def validate_smiles_input_impl(smiles):
    """Core SMILES validation logic with optimization enhancements"""
    try:
        if not smiles or not smiles.strip():
            return False, "Please enter a SMILES string", None

        Chem, AllChem, Draw = get_rdkit_modules()
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return False, "Invalid SMILES string format", None

        num_atoms = mol.GetNumAtoms()
        if num_atoms < 3:
            return False, "Molecule too small (minimum 3 atoms)", None
        if num_atoms > 200:
            return False, "Molecule too large (maximum 200 atoms)", None

        # Convert mol to pickle-able format for caching
        mol_pickle = mol.ToBinary()
        return True, f"Valid molecule ({num_atoms} atoms)", mol_pickle

    except Exception as e:
        return False, f"Error parsing SMILES: {str(e)}", None


@st.cache_data(
    ttl=3600, show_spinner=False
)  # Cache for 1 hour, hide spinner for better UX
def validate_smiles_input(smiles):
    """Validate SMILES input with detailed feedback - cached wrapper"""
    return validate_smiles_input_impl(smiles)


def validate_sdf_input(sdf_file):
    """Validate SDF file input with caching"""
    try:
        # Create cache key from file content hash
        sdf_data = sdf_file.read()
        file_hash = hash(sdf_data)
        cache_key = f"sdf_{file_hash}_{sdf_file.name}"

        # Check cache first
        if cache_key in st.session_state.get("file_cache", {}):
            cached_result = st.session_state.file_cache[cache_key]
            return (
                cached_result["valid"],
                cached_result["message"],
                cached_result["mol"],
            )

        Chem, AllChem, Draw = get_rdkit_modules()
        supplier = Chem.SDMolSupplier()
        supplier.SetData(sdf_data)

        mol = None
        for m in supplier:
            if m is not None:
                mol = m
                break

        if mol is None:
            result = (False, "No valid molecules found in file", None)
        else:
            num_atoms = mol.GetNumAtoms()
            if num_atoms < 3:
                result = (False, "Molecule too small (minimum 3 atoms)", None)
            elif num_atoms > 200:
                result = (False, "Molecule too large (maximum 200 atoms)", None)
            else:
                result = (True, f"Valid molecule ({num_atoms} atoms)", mol)

        # Cache the result
        if "file_cache" not in st.session_state:
            st.session_state.file_cache = {}
        st.session_state.file_cache[cache_key] = {
            "valid": result[0],
            "message": result[1],
            "mol": result[2],
        }

        return result

    except Exception as e:
        return False, f"Error reading file: {str(e)}", None


def validate_molecular_connectivity(mol, step_name="unknown"):
    """Comprehensive molecular connectivity validation"""
    if not mol:
        return False, f"{step_name}: Molecule is None"

    try:
        Chem, AllChem, Draw = get_rdkit_modules()

        # Check basic validity
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            return False, f"{step_name}: Sanitization failed - {str(e)}"

        # Get molecular properties
        atoms = mol.GetNumAtoms()
        bonds = mol.GetNumBonds()
        fragments = Chem.GetMolFrags(mol)

        # Basic connectivity checks
        if atoms == 0:
            return False, f"{step_name}: No atoms"
        if bonds == 0 and atoms > 1:
            return False, f"{step_name}: No bonds but multiple atoms"
        if len(fragments) > 1:
            return (
                False,
                f"{step_name}: Molecule is disconnected ({len(fragments)} fragments)",
            )

        # Check for reasonable bonding
        expected_min_bonds = max(0, atoms - len(fragments))
        if bonds < expected_min_bonds:
            return False, f"{step_name}: Too few bonds ({bonds} < {expected_min_bonds})"

        # Try to generate SMILES as connectivity test
        try:
            smiles = Chem.MolToSmiles(mol)
            if not smiles or smiles == "":
                return False, f"{step_name}: Cannot generate SMILES"
        except Exception as e:
            return False, f"{step_name}: SMILES generation failed - {str(e)}"

        return True, f"{step_name}: Connectivity valid"

    except Exception as e:
        return False, f"{step_name}: Validation error - {str(e)}"


def create_safe_molecular_copy(mol, step_name="copy"):
    """Create a safe copy of a molecule with validation"""
    if mol is None:
        return None

    try:
        Chem, AllChem, Draw = get_rdkit_modules()

        # Create a copy using SMILES roundtrip
        smiles = Chem.MolToSmiles(mol)
        new_mol = Chem.MolFromSmiles(smiles)

        if new_mol:
            # Copy coordinates if available
            if mol.GetNumConformers() > 0:
                conf = mol.GetConformer()
                new_conf = Chem.Conformer(new_mol.GetNumAtoms())
                for i in range(new_mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    new_conf.SetAtomPosition(i, pos)
                new_mol.AddConformer(new_conf)

            return new_mol
        else:
            logger.error(f"Failed to create safe copy in {step_name}")
            return mol  # Return original if copy fails

    except Exception as e:
        logger.error(f"Error in create_safe_molecular_copy ({step_name}): {e}")
        return mol  # Return original if copy fails
