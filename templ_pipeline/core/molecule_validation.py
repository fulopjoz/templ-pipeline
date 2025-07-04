"""
Molecular validation module for TEMPL pipeline.

Provides validation functions to filter problematic molecules that the pipeline
cannot handle effectively, including large peptides and complex organometallics.
"""

import logging
from typing import Tuple, Optional
from rdkit import Chem

logger = logging.getLogger(__name__)

# Common organometallic atoms that cause issues
ORGANOMETALLIC_ATOMS = {
    'Fe', 'Mn', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Pd', 'Ag', 'Cd', 'Pt', 'Au', 'Hg',
    'Mo', 'W', 'Cr', 'V', 'Ti', 'Sc', 'Y', 'Zr', 'Nb', 'Tc', 'Re', 'Os', 'Ir'
}


def has_rhenium_complex(mol: Chem.Mol) -> Tuple[bool, str]:
    """
    Check for rhenium atoms which are problematic for embedding.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Tuple[bool, str]: (True if rhenium found, warning message)
    """
    if mol is None:
        return False, ""
    
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 75:  # Rhenium
            warning_msg = (
                "❌ Target contains Re+9 organometallic complex - cannot be processed. "
                "This pipeline is designed for small molecules only. "
                "Note: PDBBind often has incorrect SMILES for organometallics. "
                "For correct structure see: https://www.rcsb.org/ligand/RCS "
                "Technical reference: https://pubs.acs.org/doi/10.1021/om00086a032"
            )
            return True, warning_msg
    
    return False, ""


def is_large_peptide(mol: Chem.Mol, residue_threshold: int = 8) -> Tuple[bool, str]:
    """
    Check if molecule is a large peptide (>8 amino acid residues).
    
    Args:
        mol: RDKit molecule object
        residue_threshold: Number of amino acid residues above which to consider "large"
        
    Returns:
        Tuple[bool, str]: (True if large peptide, warning message)
    """
    if mol is None:
        return False, ""
    
    # SMARTS pattern for amino acid backbone (amide bond pattern)
    # [NX3][CX3](=O)[CX4] matches N-C(=O)-C typical of peptide bonds
    peptide_backbone_pattern = "[NX3][CX3](=O)[CX4]"
    
    try:
        pattern = Chem.MolFromSmarts(peptide_backbone_pattern)
        if pattern is not None:
            matches = mol.GetSubstructMatches(pattern)
            residue_count = len(matches)
            
            if residue_count > residue_threshold:
                warning_msg = (
                    f"❌ Target is a large peptide ({residue_count} residues > {residue_threshold} threshold) - cannot be processed. "
                    "This pipeline is designed for drug-like small molecules only. "
                    "Large peptides require specialized conformational sampling methods."
                )
                return True, warning_msg
        else:
            # Pattern compilation failed, fall back to conservative estimation
            raise Exception("SMARTS pattern compilation failed")
            
    except Exception as e:
        # Fallback to atom counting if SMARTS fails
        logger.debug(f"SMARTS pattern matching failed, falling back to atom count estimation: {e}")
        non_h_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
        # Conservative conversion: ~6-8 atoms per residue average
        estimated_residues = non_h_atoms // 6
        if estimated_residues > residue_threshold:
            warning_msg = (
                f"❌ Target appears to be a large peptide (estimated {estimated_residues} residues > {residue_threshold} threshold) - cannot be processed. "
                "This pipeline is designed for drug-like small molecules only. "
                "Large peptides require specialized conformational sampling methods."
            )
            return True, warning_msg
    
    return False, ""


def has_problematic_organometallics(mol: Chem.Mol) -> Tuple[bool, str]:
    """
    Check for organometallic atoms that may cause processing issues.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Tuple[bool, str]: (True if problematic metals found, warning message)
    """
    if mol is None:
        return False, ""
    
    found_metals = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in ORGANOMETALLIC_ATOMS:
            found_metals.append(symbol)
    
    if found_metals:
        warning_msg = (
            f"⚠️ Target contains organometallic atoms: {', '.join(set(found_metals))}. "
            "Processing may be less reliable. Consider using UFF fallback if errors occur."
        )
        return True, warning_msg
    
    return False, ""


def validate_target_molecule(mol: Chem.Mol, mol_name: str = "unknown", 
                           peptide_threshold: int = 8) -> Tuple[bool, str]:
    """
    Validate if a target molecule can be processed by the pipeline.
    
    Args:
        mol: RDKit molecule object
        mol_name: Name/identifier for the molecule (for logging)
        peptide_threshold: Maximum number of peptide residues allowed
        
    Returns:
        Tuple[bool, str]: (True if valid, warning message if invalid)
    """
    if mol is None:
        return False, f"❌ Invalid molecule object for {mol_name}"
    
    # Check for rhenium complexes (like 3rj7)
    has_re, re_msg = has_rhenium_complex(mol)
    if has_re:
        return False, f"{mol_name}: {re_msg}"
    
    # Check for large peptides
    is_large, large_msg = is_large_peptide(mol, peptide_threshold)
    if is_large:
        return False, f"{mol_name}: {large_msg}"
    
    # Check for organometallics (warning only, not blocking)
    has_metals, metal_msg = has_problematic_organometallics(mol)
    if has_metals:
        logger.warning(f"{mol_name}: {metal_msg}")
    
    return True, ""


def get_molecule_complexity_info(mol: Chem.Mol) -> dict:
    """
    Get complexity information about a molecule for debugging.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary with complexity metrics
    """
    if mol is None:
        return {"error": "Invalid molecule"}
    
    info = {
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "num_rings": len(Chem.GetSymmSSSR(mol)),
        "molecular_weight": Chem.rdMolDescriptors.CalcExactMolWt(mol),
        "has_organometallics": has_problematic_organometallics(mol)[0],
        "estimated_peptide_residues": 0
    }
    
    # Estimate peptide residues
    try:
        peptide_pattern = Chem.MolFromSmarts("[NX3][CX3](=O)[CX4]")
        if peptide_pattern:
            matches = mol.GetSubstructMatches(peptide_pattern)
            info["estimated_peptide_residues"] = len(matches)
    except:
        pass
    
    return info