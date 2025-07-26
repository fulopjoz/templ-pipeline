"""
Chemical processing utilities for TEMPL pipeline.

This module provides comprehensive chemical processing functionality including:
- Molecule validation and filtering
- Organometallic detection and handling  
- Molecule standardization
- Force field compatibility checking
- Large biomolecule detection (peptides and polysaccharides)
- Problematic compound filtering
"""

import logging
from typing import Tuple, List, Optional, Set
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

# Disable RDKit logging noise
RDLogger.DisableLog("rdApp.*")

logger = logging.getLogger(__name__)

# Constants
ORGANOMETALLIC_ELEMENTS = {
    75: "Re",  # Rhenium
    26: "Fe",  # Iron  
    29: "Cu",  # Copper
    30: "Zn",  # Zinc
    25: "Mn",  # Manganese
    24: "Cr",  # Chromium
    23: "V",   # Vanadium
    22: "Ti",  # Titanium
    27: "Co",  # Cobalt
    28: "Ni",  # Nickel
    42: "Mo",  # Molybdenum
    74: "W",   # Tungsten
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
    39: "Y",   # Yttrium
    40: "Zr",  # Zirconium
    41: "Nb",  # Niobium
    43: "Tc",  # Technetium
}

ORGANOMETALLIC_SYMBOLS = {
    'Fe', 'Mn', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Pd', 'Ag', 'Cd', 'Pt', 'Au', 'Hg',
    'Mo', 'W', 'Cr', 'V', 'Ti', 'Sc', 'Y', 'Zr', 'Nb', 'Tc', 'Re', 'Os', 'Ir'
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


def has_problematic_organometallics(mol: Chem.Mol) -> Tuple[bool, str]:
    """
    Check if molecule contains problematic organometallic atoms.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Tuple[bool, str]: (has_problematic_metals, warning_message)
    """
    if mol is None:
        return False, ""
    
    problematic_atoms = []
    
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in ORGANOMETALLIC_SYMBOLS:
            problematic_atoms.append(f"{symbol}(idx:{atom.GetIdx()})")
    
    if problematic_atoms:
        warning_msg = (
            f"Target contains organometallic atoms: {', '.join(problematic_atoms)}. "
            "These may cause issues with force field calculations. "
            "Consider using organometallic substitution or UFF fallback."
        )
        return True, warning_msg
    
    return False, ""


def needs_uff_fallback(mol: Chem.Mol) -> bool:
    """
    Determine if a molecule needs UFF fallback for force field calculations.

    Returns True if MMFF is likely to fail and UFF should be used instead.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        bool: True if UFF should be used instead of MMFF
    """
    if mol is None:
        return True

    # Check for elements that MMFF doesn't handle well
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in ORGANOMETALLIC_ELEMENTS:
            return True

    # Check for unusual bonding patterns that might cause MMFF issues
    try:
        # Try to get MMFF properties - if this fails, we need UFF
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        if mmff_props is None:
            return True
    except:
        return True

    return False


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
    if pdb_id.lower() == "3rj7":
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 75:  # Rhenium
                logger.info(
                    f"3rj7: Allowing rhenium processing with substitution (PDBbind oxidation state issue)"
                )
                return False, ""

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 75:  # Rhenium
            warning_msg = (
                "ERROR: Target contains rhenium complex - cannot be processed. "
                "Rhenium complexes require specialized force fields not available in this pipeline. "
                "Consider using quantum mechanical methods for conformer generation."
            )
            return True, warning_msg

    return False, ""


def is_large_peptide_or_polysaccharide(
    mol: Chem.Mol, 
    residue_threshold: int = 8, 
    sugar_ring_threshold: int = 3
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
                    f" Target is a large peptide ({len(matches)} residues > {residue_threshold} threshold) - cannot be processed. "
                    "This pipeline is designed for drug-like small molecules only. "
                    "Large peptides require specialized conformational sampling methods."
                )
                return True, warning_msg
        else:
            # Pattern compilation failed, fall back to conservative estimation
            raise Exception("SMARTS pattern compilation failed")
            
    except Exception as e:
        # Fallback to atom counting if SMARTS fails for peptides
        logger.debug(f"Peptide pattern matching failed, falling back to atom count estimation: {e}")
        non_h_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
        # Conservative conversion: ~6-8 atoms per residue average
        estimated_residues = non_h_atoms // 6
        if estimated_residues > residue_threshold:
            warning_msg = (
                f" Target appears to be a large peptide (estimated {estimated_residues} residues > {residue_threshold} threshold) - cannot be processed. "
                "This pipeline is designed for drug-like small molecules only. "
                "Large peptides require specialized conformational sampling methods."
            )
            return True, warning_msg

    # Check for polysaccharide detection
    # SMARTS pattern for 6-membered sugar rings: [C;R1]1[C;R1][C;R1][C;R1][C;R1][O;R1]1
    sugar_ring_pattern = "[C;R1]1[C;R1][C;R1][C;R1][C;R1][O;R1]1"
    try:
        sugar_pattern = Chem.MolFromSmarts(sugar_ring_pattern)
        if sugar_pattern is not None:
            sugar_matches = mol.GetSubstructMatches(sugar_pattern)
            if len(sugar_matches) > sugar_ring_threshold:
                warning_msg = (
                    f" Target is a complex polysaccharide ({len(sugar_matches)} sugar-like rings > {sugar_ring_threshold} threshold) - cannot be processed. "
                    "This pipeline is designed for drug-like small molecules only. "
                    "Complex polysaccharides require specialized conformational sampling methods."
                )
                return True, warning_msg
    except Exception as e:
        logger.debug(f"Sugar ring pattern matching failed: {e}")

    return False, ""


def is_large_peptide(mol: Chem.Mol, residue_threshold: int = 12) -> Tuple[bool, str]:
    """
    Backward compatibility wrapper for is_large_peptide_or_polysaccharide.
    
    This function is deprecated. Use is_large_peptide_or_polysaccharide instead.
    
    Args:
        mol: RDKit molecule object
        residue_threshold: Number of amino acid residues above which to consider "large"
        
    Returns:
        Tuple[bool, str]: (True if large peptide, warning message)
    """
    # Only check for peptides, not polysaccharides, to maintain exact backward compatibility
    result, msg = is_large_peptide_or_polysaccharide(mol, residue_threshold, float('inf'))
    return result, msg


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


# Molecule standardization functions
def standardize_molecule_smiles(mol: Chem.Mol, method: str = "canonical") -> Chem.Mol:
    """
    Standardize molecule SMILES representation to ensure consistent comparisons.

    Args:
        mol: RDKit molecule object
        method: Standardization method (canonical, isomeric, kekule)

    Returns:
        Molecule with standardized SMILES representation
    """
    try:
        if method == "canonical":
            smiles = Chem.MolToSmiles(mol, canonical=True)
        elif method == "isomeric":
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        elif method == "kekule":
            Chem.Kekulize(mol)
            smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
        else:
            smiles = Chem.MolToSmiles(mol)

        # Recreate molecule from standardized SMILES
        standardized_mol = Chem.MolFromSmiles(smiles)

        if standardized_mol is None:
            logger.warning(
                f"Failed to recreate molecule from standardized SMILES: {smiles}"
            )
            return mol

        # Copy coordinates if available and atoms match
        if (
            mol.GetNumConformers() > 0
            and standardized_mol.GetNumAtoms() == mol.GetNumAtoms()
        ):
            try:
                conf = mol.GetConformer(0)
                new_conf = Chem.Conformer(standardized_mol.GetNumAtoms())
                for i in range(standardized_mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    new_conf.SetAtomPosition(i, pos)
                standardized_mol.AddConformer(new_conf)
            except Exception as e:
                logger.warning(
                    f"Failed to copy coordinates during standardization: {e}"
                )
                # Continue without coordinates

        return standardized_mol

    except Exception as e:
        logger.warning(f"SMILES standardization failed: {e}")
        return mol


def standardize_molecule_list(
    molecules: List[Chem.Mol], method: str = "canonical"
) -> List[Chem.Mol]:
    """
    Standardize a list of molecules.

    Args:
        molecules: List of RDKit molecule objects
        method: Standardization method

    Returns:
        List of standardized molecules
    """
    standardized = []

    for i, mol in enumerate(molecules):
        if mol is None:
            logger.warning(f"Skipping None molecule at index {i}")
            continue

        try:
            std_mol = standardize_molecule_smiles(mol, method)
            standardized.append(std_mol)
        except Exception as e:
            logger.error(f"Failed to standardize molecule at index {i}: {e}")
            # Include original molecule as fallback
            standardized.append(mol)

    return standardized


def remove_problematic_molecules(
    molecules: List[Chem.Mol], 
    strict: bool = False
) -> Tuple[List[Chem.Mol], List[str]]:
    """
    Remove molecules that cannot be processed by the pipeline.
    
    Args:
        molecules: List of RDKit molecule objects
        strict: If True, remove molecules with any organometallic atoms
        
    Returns:
        Tuple containing:
            - List of filtered molecules that can be processed
            - List of removal reasons for rejected molecules
    """
    filtered = []
    removal_reasons = []
    
    for i, mol in enumerate(molecules):
        if mol is None:
            removal_reasons.append(f"Molecule {i}: None object")
            continue
            
        # Check for validation issues
        is_valid, error_msg = validate_target_molecule(mol, f"molecule_{i}")
        if not is_valid:
            removal_reasons.append(f"Molecule {i}: {error_msg}")
            continue
            
        # Check for problematic organometallics if strict mode
        if strict:
            has_problems, problem_msg = has_problematic_organometallics(mol)
            if has_problems:
                removal_reasons.append(f"Molecule {i}: {problem_msg}")
                continue
        
        filtered.append(mol)
    
    return filtered, removal_reasons


def sanitize_molecule_safe(mol: Chem.Mol) -> Tuple[Chem.Mol, bool, str]:
    """
    Safely sanitize a molecule with fallback handling.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Tuple containing:
            - Sanitized molecule (or original if sanitization failed)
            - Success flag
            - Status message
    """
    if mol is None:
        return None, False, "Input molecule is None"
    
    try:
        # Try normal sanitization first
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)
        return mol_copy, True, "Successfully sanitized"
        
    except Exception as e:
        # Try with organometallic substitution
        try:
            substituted_mol, was_modified, sub_log = detect_and_substitute_organometallic(mol)
            if was_modified:
                return substituted_mol, True, f"Sanitized after organometallic substitution: {'; '.join(sub_log)}"
            else:
                return mol, False, f"Sanitization failed: {e}"
                
        except Exception as e2:
            return mol, False, f"Sanitization failed even with organometallic substitution: {e2}"


def get_molecule_properties(mol: Chem.Mol) -> dict:
    """
    Get comprehensive properties of a molecule for validation and debugging.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary containing molecule properties and validation flags:
            - valid: Whether the molecule is valid
            - num_atoms: Total number of atoms
            - num_heavy_atoms: Number of non-hydrogen atoms  
            - num_conformers: Number of conformers
            - molecular_weight: Molecular weight
            - smiles: SMILES representation
            - has_rhenium: Contains rhenium atoms
            - is_large_biomolecule: Is a large peptide or polysaccharide
            - has_organometallics: Contains organometallic atoms
            - needs_uff_fallback: Requires UFF instead of MMFF
    """
    if mol is None:
        return {"valid": False, "error": "None molecule"}
    
    try:
        props = {
            "valid": True,
            "num_atoms": mol.GetNumAtoms(),
            "num_heavy_atoms": mol.GetNumHeavyAtoms(),
            "num_conformers": mol.GetNumConformers(),
            "molecular_weight": Descriptors.MolWt(mol),
            "smiles": Chem.MolToSmiles(mol),
        }
        
        # Check for special cases
        has_re, _ = has_rhenium_complex(mol)
        is_large_bio, _ = is_large_peptide_or_polysaccharide(mol)
        has_metals, _ = has_problematic_organometallics(mol)
        needs_uff = needs_uff_fallback(mol)
        
        props.update({
            "has_rhenium": has_re,
            "is_large_biomolecule": is_large_bio,
            "has_organometallics": has_metals,
            "needs_uff_fallback": needs_uff,
        })
        
        return props
        
    except Exception as e:
        return {"valid": False, "error": str(e)}