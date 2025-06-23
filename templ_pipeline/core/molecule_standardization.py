"""
Molecule Standardization Module

This module provides standardized molecular processing to ensure consistent
SMILES representations and prevent benchmarking issues caused by different
aromatic ring notations.
"""

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from typing import List, Optional, Tuple
import logging

RDLogger.DisableLog('rdApp.*')
logger = logging.getLogger(__name__)

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
            logger.warning(f"Failed to recreate molecule from standardized SMILES: {smiles}")
            return mol
        
        # Copy coordinates if available and atoms match
        if mol.GetNumConformers() > 0 and standardized_mol.GetNumAtoms() == mol.GetNumAtoms():
            try:
                conf = mol.GetConformer(0)
                new_conf = Chem.Conformer(standardized_mol.GetNumAtoms())
                for i in range(standardized_mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    new_conf.SetAtomPosition(i, pos)
                standardized_mol.AddConformer(new_conf)
            except Exception as e:
                logger.warning(f"Failed to copy coordinates during standardization: {e}")
                # Continue without coordinates
        
        return standardized_mol
        
    except Exception as e:
        logger.warning(f"SMILES standardization failed: {e}")
        return mol

def standardize_molecule_list(molecules: List[Chem.Mol], method: str = "canonical") -> List[Chem.Mol]:
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

def compare_smiles_representations(mol1: Chem.Mol, mol2: Chem.Mol) -> dict:
    """
    Compare different SMILES representations of two molecules.
    
    Args:
        mol1: First molecule
        mol2: Second molecule
        
    Returns:
        Dictionary with comparison results
    """
    results = {
        "molecules_valid": mol1 is not None and mol2 is not None,
        "raw_smiles": {},
        "standardized_smiles": {},
        "identical_after_standardization": False
    }
    
    if not results["molecules_valid"]:
        return results
    
    try:
        # Raw SMILES
        raw1 = Chem.MolToSmiles(mol1)
        raw2 = Chem.MolToSmiles(mol2) 
        results["raw_smiles"] = {
            "mol1": raw1,
            "mol2": raw2,
            "identical": raw1 == raw2
        }
        
        # Standardized SMILES using our standardization function
        methods = ["canonical", "isomeric", "kekule"]
        for method in methods:
            try:
                # Use our standardization function instead of direct RDKit
                std_mol1 = standardize_molecule_smiles(mol1, method)
                std_mol2 = standardize_molecule_smiles(mol2, method)
                
                std1 = Chem.MolToSmiles(std_mol1)
                std2 = Chem.MolToSmiles(std_mol2)
                
                results["standardized_smiles"][method] = {
                    "mol1": std1,
                    "mol2": std2,
                    "identical": std1 == std2
                }
                
                # Check if any method produces identical results
                if std1 == std2:
                    results["identical_after_standardization"] = True
                    
            except Exception as e:
                results["standardized_smiles"][method] = {"error": str(e)}
        
    except Exception as e:
        results["error"] = str(e)
    
    return results

def ensure_molecule_conformers(mol: Chem.Mol, force_generation: bool = False) -> Chem.Mol:
    """
    Ensure molecule has conformers, generating them if necessary.
    
    Args:
        mol: RDKit molecule
        force_generation: Force generation even if conformers exist
        
    Returns:
        Molecule with conformers
    """
    if mol is None:
        return None
    
    # Check if conformers already exist
    if mol.GetNumConformers() > 0 and not force_generation:
        return mol
    
    try:
        # Create a copy to avoid modifying the original
        mol_copy = Chem.Mol(mol)
        
        # Add hydrogens for better embedding using enhanced method
        try:
            from .scoring import FixedMolecularProcessor
            mol_with_h = FixedMolecularProcessor.safe_add_hydrogens(mol_copy, preserve_coords=True)
        except ImportError:
            # Fallback to standard method if scoring module not available
            mol_with_h = Chem.AddHs(mol_copy)
        
        # Generate 3D coordinates
        embed_result = AllChem.EmbedMolecule(mol_with_h)
        
        if embed_result == 0:  # Success
            # Optimize geometry
            try:
                AllChem.MMFFOptimizeMolecule(mol_with_h)
            except Exception as e:
                logger.warning(f"MMFF optimization failed: {e}")
            
            # Remove hydrogens
            mol_final = Chem.RemoveHs(mol_with_h)
            
            # Ensure we have coordinates
            if mol_final.GetNumConformers() > 0:
                return mol_final
            else:
                logger.warning("Conformer generation succeeded but no conformers found")
        else:
            logger.debug("Failed to embed molecule - using original")
            return mol
            
    except Exception as e:
        logger.debug(f"Failed to embed molecule - using original: {e}")
        return mol

def standardize_and_prepare_molecule(mol: Chem.Mol, 
                                   ensure_conformers: bool = True,
                                   standardization_method: str = "canonical") -> Chem.Mol:
    """
    Complete standardization and preparation of a molecule.
    
    Args:
        mol: Input molecule
        ensure_conformers: Whether to ensure 3D conformers exist
        standardization_method: SMILES standardization method
        
    Returns:
        Fully prepared molecule
    """
    if mol is None:
        return None
    
    try:
        # Step 1: Standardize SMILES representation
        standardized_mol = standardize_molecule_smiles(mol, standardization_method)
        
        # Step 2: Ensure conformers if requested
        if ensure_conformers:
            final_mol = ensure_molecule_conformers(standardized_mol)
        else:
            final_mol = standardized_mol
        
        # Step 3: Final sanitization
        try:
            Chem.SanitizeMol(final_mol)
            Chem.SetAromaticity(final_mol)
        except Exception as e:
            logger.warning(f"Final sanitization failed: {e}")
        
        return final_mol
        
    except Exception as e:
        logger.error(f"Complete molecule standardization failed: {e}")
        return mol

# Default standardization parameters
DEFAULT_STANDARDIZATION_METHOD = "canonical"
ENABLE_CONFORMER_GENERATION = True

def get_standardization_config() -> dict:
    """Get current standardization configuration."""
    return {
        "method": DEFAULT_STANDARDIZATION_METHOD,
        "ensure_conformers": ENABLE_CONFORMER_GENERATION,
        "available_methods": ["canonical", "isomeric", "kekule"]
    } 