"""
Chemical processing utilities for complex molecules.
"""
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
from typing import Tuple, List, Optional


def detect_and_substitute_organometallic(mol: Chem.Mol, molecule_name: str = "unknown") -> Tuple[Chem.Mol, bool, List[str]]:
    """
    Detect and substitute organometallic atoms with carbon for conformer generation.
    
    Returns:
        Tuple[Mol, bool, List[str]]: (modified_mol, was_modified, substitution_log)
    """
    if mol is None:
        return None, False, ["Input molecule is None"]
    
    # Common organometallic elements that cause issues
    organometallic_elements = {
        75: 'Re',  # Rhenium
        26: 'Fe',  # Iron
        29: 'Cu',  # Copper
        30: 'Zn',  # Zinc
        25: 'Mn',  # Manganese
        24: 'Cr',  # Chromium
        23: 'V',   # Vanadium
        22: 'Ti',  # Titanium
        27: 'Co',  # Cobalt
        28: 'Ni',  # Nickel
        42: 'Mo',  # Molybdenum
        74: 'W',   # Tungsten
        44: 'Ru',  # Ruthenium
        45: 'Rh',  # Rhodium
        46: 'Pd',  # Palladium
        47: 'Ag',  # Silver
        48: 'Cd',  # Cadmium
        77: 'Ir',  # Iridium
        78: 'Pt',  # Platinum
        79: 'Au',  # Gold
        80: 'Hg',  # Mercury
    }
    
    substitution_log = []
    modified = False
    
    try:
        # Create a copy to modify
        mol_copy = Chem.Mol(mol)
        
        # Find organometallic atoms
        organometallic_atoms = []
        for atom in mol_copy.GetAtoms():
            if atom.GetAtomicNum() in organometallic_elements:
                organometallic_atoms.append(atom.GetIdx())
                element_symbol = organometallic_elements[atom.GetAtomicNum()]
                substitution_log.append(f"Found {element_symbol} at atom index {atom.GetIdx()}")
        
        if not organometallic_atoms:
            return mol_copy, False, ["No organometallic atoms detected"]
        
        # Substitute with carbon (atomic number 6)
        for atom_idx in organometallic_atoms:
            atom = mol_copy.GetAtomWithIdx(atom_idx)
            old_element = organometallic_elements[atom.GetAtomicNum()]
            atom.SetAtomicNum(6)  # Carbon
            atom.SetFormalCharge(0)  # Reset charge
            substitution_log.append(f"Substituted {old_element} with C at index {atom_idx}")
            modified = True
        
        if modified:
            # Try to sanitize the modified molecule
            try:
                Chem.SanitizeMol(mol_copy)
                substitution_log.append("Successfully sanitized modified molecule")
            except Exception as e:
                substitution_log.append(f"Sanitization failed: {e}")
                # Try without sanitization
                pass
        
        return mol_copy, modified, substitution_log
    
    except Exception as e:
        error_msg = f"Organometallic substitution failed for {molecule_name}: {e}"
        logging.warning(error_msg)
        return mol, False, [error_msg]


def needs_uff_fallback(mol: Chem.Mol) -> bool:
    """
    Determine if a molecule needs UFF fallback for force field calculations.
    
    Returns True if MMFF is likely to fail and UFF should be used instead.
    """
    if mol is None:
        return True
    
    # Check for elements that MMFF doesn't handle well
    problematic_elements = {
        75,  # Rhenium
        26,  # Iron
        29,  # Copper
        30,  # Zinc
        25,  # Manganese
        24,  # Chromium
        23,  # Vanadium
        22,  # Titanium
        27,  # Cobalt
        28,  # Nickel
        42,  # Molybdenum
        74,  # Tungsten
        44,  # Ruthenium
        45,  # Rhodium
        46,  # Palladium
        47,  # Silver
        48,  # Cadmium
        77,  # Iridium
        78,  # Platinum
        79,  # Gold
        80,  # Mercury
    }
    
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in problematic_elements:
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


def has_rhenium_complex(mol: Chem.Mol, pdb_id: str = "") -> Tuple[bool, str]:
    """
    Check if molecule contains rhenium complexes that cannot be processed.
    Special handling for 3rj7 (incorrect oxidation state in PDBbind).
    
    Returns:
        Tuple[bool, str]: (has_rhenium, warning_message)
    """
    if mol is None:
        return False, ""
    
    # Special case for 3rj7 - allow processing with substitution
    if pdb_id.lower() == "3rj7":
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 75:  # Rhenium
                logging.info(f"3rj7: Allowing rhenium processing with substitution (PDBbind oxidation state issue)")
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


def is_large_peptide(mol: Chem.Mol, residue_threshold: int = 8) -> Tuple[bool, str]:
    """
    Check if molecule is a large peptide that should not be processed.
    
    Returns:
        Tuple[bool, str]: (is_large_peptide, warning_message)
    """
    if mol is None:
        return False, ""
    
    try:
        # SMARTS pattern for peptide bonds
        peptide_pattern = Chem.MolFromSmarts('[NX3][CX3](=[OX1])[CX4]')
        
        if peptide_pattern is not None:
            matches = mol.GetSubstructMatches(peptide_pattern)
            num_peptide_bonds = len(matches)
            
            if num_peptide_bonds > residue_threshold:
                warning_msg = (
                    f"ERROR: Target appears to be a large peptide ({num_peptide_bonds} peptide bonds > {residue_threshold} threshold) - cannot be processed. "
                    "This pipeline is designed for drug-like small molecules only. "
                    "Large peptides require specialized conformational sampling methods."
                )
                return True, warning_msg
        else:
            # Pattern compilation failed, fall back to conservative estimation
            raise Exception("SMARTS pattern compilation failed")
            
    except Exception as e:
        # Fallback to atom counting if SMARTS fails
        logging.debug(f"SMARTS pattern matching failed, falling back to atom count estimation: {e}")
        non_h_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
        # Conservative conversion: ~6-8 atoms per residue average
        estimated_residues = non_h_atoms // 6
        if estimated_residues > residue_threshold:
            warning_msg = (
                f"ERROR: Target appears to be a large peptide (estimated {estimated_residues} residues > {residue_threshold} threshold) - cannot be processed. "
                "This pipeline is designed for drug-like small molecules only. "
                "Large peptides require specialized conformational sampling methods."
            )
            return True, warning_msg
    
    return False, ""


def validate_target_molecule(mol: Chem.Mol, mol_name: str = "unknown", pdb_id: str = "", peptide_threshold: int = 8) -> Tuple[bool, str]:
    """
    Validate if a target molecule can be processed by the pipeline.
    
    Args:
        mol: RDKit molecule object
        mol_name: Name/identifier for the molecule (for logging)
        pdb_id: PDB ID for special case handling
        peptide_threshold: Maximum number of peptide residues allowed
        
    Returns:
        Tuple[bool, str]: (True if valid, warning message if invalid)
    """
    if mol is None:
        return False, f"ERROR: Invalid molecule object for {mol_name}"
    
    # Check for rhenium complexes (with 3rj7 exception)
    has_re, re_msg = has_rhenium_complex(mol, pdb_id)
    if has_re:
        return False, f"{mol_name}: {re_msg}"
    
    # Check for large peptides
    is_peptide, peptide_msg = is_large_peptide(mol, peptide_threshold)
    if is_peptide:
        return False, f"{mol_name}: {peptide_msg}"
    
    return True, "" 