# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Chemical processing utilities for the TEMPL pipeline.

This module provides:
- Molecule validation and filtering
- Organometallic detection and handling
- Large biomolecule detection (peptides and polysaccharides)
"""

import logging
from typing import List, Tuple

from rdkit import Chem, RDLogger  # type: ignore

# Disable RDKit logging noise
try:
    RDLogger.DisableLog("rdApp.*")  # type: ignore
except (AttributeError, ImportError):
    pass

logger = logging.getLogger(__name__)

# Constants
ORGANOMETALLIC_ELEMENTS = {
    75: "Re",  # Rhenium
    26: "Fe",  # Iron
    29: "Cu",  # Copper
    30: "Zn",  # Zinc
    25: "Mn",  # Manganese
    24: "Cr",  # Chromium
    23: "V",  # Vanadium
    22: "Ti",  # Titanium
    27: "Co",  # Cobalt
    28: "Ni",  # Nickel
    42: "Mo",  # Molybdenum
    74: "W",  # Tungsten
    44: "Ru",  # Ruthenium
    45: "Rh",  # Rhodium
    46: "Pd",  # Palladium
    47: "Ag",  # Silver
    48: "Cd",  # Cadmium
    77: "Ir",  # Iridium
    78: "Pt",  # Platinum
    79: "Au",  # Gold
    80: "Hg",  # Mercury
    76: "Os",  # Osmium
    21: "Sc",  # Scandium
    39: "Y",  # Yttrium
    40: "Zr",  # Zirconium
    41: "Nb",  # Niobium
    43: "Tc",  # Technetium
}


# Organometallic detection and handling
def detect_and_substitute_organometallic(
    mol: Chem.Mol, molecule_name: str = "unknown"
) -> Tuple[Chem.Mol, bool, List[str]]:
    """
    Detect and substitute organometallic atoms with carbon for conformer generation.

    This enables processing of molecules containing metal atoms that would otherwise fail
    in RDKit sanitization and downstream operations.

    Args:
        mol: RDKit molecule object
        molecule_name: Name/identifier for the molecule (for logging)

    Returns:
        Tuple[Mol, bool, List[str]]: (modified_mol, was_modified, substitution_log)
    """
    if mol is None:
        return None, False, ["Input molecule is None"]

    substitution_log = []
    modified = False

    try:
        # Create a copy to modify
        mol_copy = Chem.Mol(mol)

        # Find organometallic atoms
        organometallic_atoms = []
        for atom in mol_copy.GetAtoms():
            if atom.GetAtomicNum() in ORGANOMETALLIC_ELEMENTS:
                organometallic_atoms.append(atom.GetIdx())
                element_symbol = ORGANOMETALLIC_ELEMENTS[atom.GetAtomicNum()]
                substitution_log.append(
                    f"Found {element_symbol} at atom index {atom.GetIdx()}"
                )

        if not organometallic_atoms:
            return mol_copy, False, ["No organometallic atoms detected"]

        # Substitute with carbon (atomic number 6)
        for atom_idx in organometallic_atoms:
            atom = mol_copy.GetAtomWithIdx(atom_idx)
            old_element = ORGANOMETALLIC_ELEMENTS[atom.GetAtomicNum()]
            atom.SetAtomicNum(6)  # Carbon
            atom.SetFormalCharge(0)  # Reset charge
            substitution_log.append(
                f"Substituted {old_element} with C at index {atom_idx}"
            )
            modified = True

        if modified:
            # Try to sanitize the modified molecule
            try:
                Chem.SanitizeMol(mol_copy)
                substitution_log.append("Successfully sanitized modified molecule")
            except Exception as e:
                substitution_log.append(f"Sanitization failed: {e}")
                # Continue without sanitization

        return mol_copy, modified, substitution_log

    except Exception as e:
        error_msg = f"Organometallic substitution failed for {molecule_name}: {e}"
        logger.warning(error_msg)
        return mol, False, [error_msg]


# Molecule validation functions
def has_rhenium_complex(mol: Chem.Mol, pdb_id: str = "") -> Tuple[bool, str]:
    """
    Check if molecule contains rhenium complexes that cannot be processed.
    Special handling for 3rj7 (incorrect oxidation state in PDBbind).

    Args:
        mol: RDKit molecule object
        pdb_id: PDB ID for special case handling

    Returns:
        Tuple[bool, str]: (has_rhenium, warning_message)
    """
    if mol is None:
        return False, ""

    # Special case for 3rj7 - allow processing with substitution
    if pdb_id and pdb_id.lower() == "3rj7":
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 75:  # Rhenium
                logger.info(
                    "3rj7: Allowing rhenium processing with substitution "
                    "(PDBbind oxidation state issue)"
                )
                return False, ""

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 75:  # Rhenium
            warning_msg = (
                "ERROR: Target contains rhenium complex - cannot be processed. "
                "Rhenium complexes require specialized force fields not available "
                "in this pipeline. Consider using quantum mechanical methods "
                "for conformer generation."
            )
            return True, warning_msg

    return False, ""


def is_large_peptide_or_polysaccharide(
    mol: Chem.Mol, residue_threshold: int = 8, sugar_ring_threshold: int = 3
) -> Tuple[bool, str]:
    """
    Unified filter: catches large peptides OR large polysaccharides.

    This function detects both large peptides and complex polysaccharides that
    cannot be processed by the pipeline due to their size and complexity.
    Includes fallback mechanisms for robust detection.

    Args:
        mol: RDKit molecule object
        residue_threshold: Number of amino acid residues above which to consider "large"
        sugar_ring_threshold: Number of sugar rings above which to consider "large"

    Returns:
        Tuple[bool, str]: (True if large peptide/polysaccharide, warning message)
    """
    if mol is None:
        return False, ""

    # Try peptide detection first
    peptide_backbone_pattern = "[NX3][CX3](=O)[CX4]"
    try:
        pattern = Chem.MolFromSmarts(peptide_backbone_pattern)
        if pattern is not None:
            matches = mol.GetSubstructMatches(pattern)
            if len(matches) > residue_threshold:
                warning_msg = (
                    f" Target is a large peptide ({len(matches)} residues > "
                    f"{residue_threshold} threshold) - cannot be processed. "
                    "This pipeline is designed for drug-like small molecules only. "
                    "Large peptides require specialized conformational sampling methods."
                )
                return True, warning_msg
        else:
            # Pattern compilation failed, fall back to conservative estimation
            raise Exception("SMARTS pattern compilation failed")

    except Exception as e:
        # Fallback to atom counting if SMARTS fails for peptides
        logger.debug(
            f"Peptide pattern matching failed, falling back to atom count "
            f"estimation: {e}"
        )
        non_h_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
        # Conservative conversion: ~6-8 atoms per residue average
        estimated_residues = non_h_atoms // 6
        if estimated_residues > residue_threshold:
            warning_msg = (
                f" Target appears to be a large peptide (estimated "
                f"{estimated_residues} residues > {residue_threshold} threshold) - "
                f"cannot be processed. This pipeline is designed for drug-like "
                f"small molecules only. Large peptides require specialized "
                f"conformational sampling methods."
            )
            return True, warning_msg

    # Check for polysaccharide detection
    # SMARTS pattern for 6-membered sugar rings:
    # [C;R1]1[C;R1][C;R1][C;R1][C;R1][O;R1]1
    sugar_ring_pattern = "[C;R1]1[C;R1][C;R1][C;R1][C;R1][O;R1]1"
    try:
        sugar_pattern = Chem.MolFromSmarts(sugar_ring_pattern)
        if sugar_pattern is not None:
            sugar_matches = mol.GetSubstructMatches(sugar_pattern)
            if len(sugar_matches) > sugar_ring_threshold:
                warning_msg = (
                    f" Target is a complex polysaccharide ({len(sugar_matches)} "
                    f"sugar-like rings > {sugar_ring_threshold} threshold) - "
                    f"cannot be processed. This pipeline is designed for drug-like "
                    f"small molecules only. Complex polysaccharides require "
                    f"specialized conformational sampling methods."
                )
                return True, warning_msg
    except Exception as e:
        logger.debug(f"Sugar ring pattern matching failed: {e}")

    return False, ""


def validate_target_molecule(
    mol: Chem.Mol,
    mol_name: str = "unknown",
    pdb_id: str = "",
    peptide_threshold: int = 8,
    sugar_ring_threshold: int = 3,
) -> Tuple[bool, str]:
    """
    Validate if a target molecule can be processed by the pipeline.

    Args:
        mol: RDKit molecule object
        mol_name: Name/identifier for the molecule (for logging)
        pdb_id: PDB ID for special case handling
        peptide_threshold: Maximum number of peptide residues allowed
        sugar_ring_threshold: Maximum number of sugar rings allowed

    Returns:
        Tuple[bool, str]: (True if valid, warning message if invalid)
    """
    if mol is None:
        return False, f"ERROR: Invalid molecule object for {mol_name}"

    # Check for rhenium complexes (with 3rj7 exception)
    has_re, re_msg = has_rhenium_complex(mol, pdb_id)
    if has_re:
        return False, f"{mol_name}: {re_msg}"

    # Check for large peptides or polysaccharides
    is_large_bio, bio_msg = is_large_peptide_or_polysaccharide(
        mol, peptide_threshold, sugar_ring_threshold
    )
    if is_large_bio:
        return False, f"{mol_name}: {bio_msg}"

    return True, ""
