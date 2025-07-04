"""
Organometallic molecule handling for TEMPL pipeline.

Provides functions to detect and handle organometallic molecules that can cause
issues with standard RDKit operations and force field calculations.
"""

import logging
from typing import Tuple, List
from rdkit import Chem

logger = logging.getLogger(__name__)

# Common organometallic atoms found in drug discovery
ORGANOMETALLIC_ATOMS = {
    'Fe', 'Mn', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Pd', 'Ag', 'Cd', 'Pt', 'Au', 'Hg',
    'Mo', 'W', 'Cr', 'V', 'Ti', 'Sc', 'Y', 'Zr', 'Nb', 'Tc', 'Re', 'Os', 'Ir'
}


def detect_and_substitute_organometallic(mol: Chem.Mol, molecule_name: str = "unknown") -> Tuple[Chem.Mol, bool, List[str]]:
    """Detect organometallic atoms and substitute with carbon to enable processing.
    
    This enables processing of molecules containing metal atoms that would otherwise fail
    in RDKit sanitization and downstream operations.
    
    Args:
        mol: Input molecule
        molecule_name: Name for logging purposes
        
    Returns:
        Tuple of (processed_molecule, had_metals, substitutions_made)
    """
    if mol is None:
        return None, False, []
    
    # Create a copy to avoid modifying the original
    mol_copy = Chem.Mol(mol)
    
    # Check for organometallic atoms
    organometallic_found = []
    substitutions = []
    
    for atom in mol_copy.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in ORGANOMETALLIC_ATOMS:
            organometallic_found.append(f"{symbol}(idx:{atom.GetIdx()})")
            
            # Substitute with carbon - preserve connectivity
            atom.SetAtomicNum(6)  # Carbon
            atom.SetFormalCharge(0)
            substitutions.append(f"{symbol}→C(idx:{atom.GetIdx()})")
    
    had_metals = len(organometallic_found) > 0
    
    if had_metals:
        logger.info(f"Organometallic handling for {molecule_name}: Found {organometallic_found}, substituted with carbon: {substitutions}")
        
        try:
            # Attempt to sanitize after substitution
            Chem.SanitizeMol(mol_copy)
        except Exception as e:
            logger.warning(f"Sanitization still failed for {molecule_name} after organometallic substitution: {e}")
            # Return the modified molecule even if sanitization fails - some operations might still work
    
    return mol_copy, had_metals, substitutions


def needs_uff_fallback(mol: Chem.Mol) -> bool:
    """
    Check if a molecule needs UFF instead of MMFF due to problematic atoms.
    
    Args:
        mol: RDKit molecule object
    
    Returns:
        bool: True if UFF should be used instead of MMFF
        
    Common problematic atoms that work better with UFF:
    - Transition metals (particularly organometallics)
    - Some metalloids in unusual coordination environments
    """
    if mol is None:
        return False
    
    problematic_atomic_nums = {
        # Transition metals that commonly cause MMFF issues
        25,  # Mn
        26,  # Fe  
        27,  # Co
        28,  # Ni
        29,  # Cu
        30,  # Zn
        42,  # Mo
        74,  # W
        75,  # Re
        76,  # Os
        77,  # Ir
        78,  # Pt
        79,  # Au
        80,  # Hg
    }
    
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in problematic_atomic_nums:
            return True
    
    return False


def has_organometallic_atoms(mol: Chem.Mol) -> Tuple[bool, List[str]]:
    """
    Check if molecule contains organometallic atoms.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Tuple of (has_metals, list_of_metal_symbols)
    """
    if mol is None:
        return False, []
    
    found_metals = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in ORGANOMETALLIC_ATOMS:
            found_metals.append(symbol)
    
    return len(found_metals) > 0, found_metals


def minimize_with_uff(mol: Chem.Mol, conf_ids: List[int], fixed_atoms: List[int] = None, max_its: int = 200):
    """UFF minimization for organometallic molecules.
    
    Args:
        mol: Molecule with conformers
        conf_ids: List of conformer IDs to minimize
        fixed_atoms: List of atom indices to keep fixed (optional)
        max_its: Maximum iterations for minimization
    """
    from rdkit.Chem import rdForceFieldHelpers
    
    if not conf_ids:
        return
        
    fixed_atoms = fixed_atoms or []
    successful_minimizations = 0
    
    for cid in conf_ids:
        try:
            ff = rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=cid)
            if ff:
                # Add fixed points if specified
                for fixed_idx in fixed_atoms:
                    if fixed_idx < mol.GetNumAtoms():
                        ff.AddFixedPoint(fixed_idx)
                
                # Perform minimization
                result = ff.Minimize(maxIts=max_its)
                if result == 0:  # Success
                    successful_minimizations += 1
                else:
                    logger.debug(f"UFF minimization returned code {result} for conformer {cid}")
            else:
                logger.debug(f"Could not create UFF force field for conformer {cid}")
        except Exception as e:
            logger.debug(f"UFF minimization failed for conformer {cid}: {e}")
    
    logger.debug(f"UFF minimization succeeded for {successful_minimizations}/{len(conf_ids)} conformers")


def embed_with_uff_fallback(mol: Chem.Mol, n_conformers: int, ps, coordMap: dict = None) -> List[int]:
    """Embedding with UFF compatibility for organometallic molecules.
    
    Args:
        mol: Molecule to embed (should have hydrogens added)
        n_conformers: Number of conformers to generate
        ps: Original embedding parameters
        coordMap: Coordinate constraints (optional)
        
    Returns:
        List of conformer IDs
    """
    from rdkit.Chem import rdDistGeom
    
    # Try standard embedding first (might work even with metals)
    try:
        # Use individual parameters instead of EmbedParameters when coordMap is needed
        conf_ids = rdDistGeom.EmbedMultipleConfs(
            mol, n_conformers,
            maxAttempts=ps.maxIterations if hasattr(ps, 'maxIterations') else 1000,
            randomSeed=-1,
            useRandomCoords=ps.useRandomCoords if hasattr(ps, 'useRandomCoords') else True,
            enforceChirality=ps.enforceChirality if hasattr(ps, 'enforceChirality') else False,
            numThreads=ps.numThreads if hasattr(ps, 'numThreads') else 0,
            coordMap=coordMap or {}
        )
        if conf_ids:
            logger.debug(f"Standard embedding succeeded for organometallic molecule")
            return conf_ids
    except Exception as e:
        logger.debug(f"Standard embedding failed for organometallic molecule: {e}")
    
    # Fallback: simplified embedding parameters for difficult molecules
    logger.info(f"Using UFF-compatible embedding parameters for organometallic molecule")
    ps_uff = rdDistGeom.ETKDGv3()
    ps_uff.numThreads = ps.numThreads if hasattr(ps, 'numThreads') else 0
    ps_uff.maxIterations = (ps.maxIterations if hasattr(ps, 'maxIterations') else 1000) * 2  # More attempts
    ps_uff.useRandomCoords = True
    ps_uff.randomSeed = 42  # Reproducible results
    ps_uff.enforceChirality = False  # Critical: match original working code
    
    # Remove coordinate constraints that might cause issues with metals
    if hasattr(ps, 'coordMap') and ps.coordMap:
        logger.debug("Removing coordinate constraints for UFF embedding")
        ps_uff.coordMap = {}
    
    try:
        conf_ids = rdDistGeom.EmbedMultipleConfs(mol, n_conformers, ps_uff)
        if conf_ids:
            logger.debug(f"UFF-compatible embedding succeeded, generated {len(conf_ids)} conformers")
            return conf_ids
    except Exception as e:
        logger.warning(f"UFF-compatible embedding also failed: {e}")
    
    # Final fallback: try with even simpler parameters
    logger.warning("Attempting final fallback embedding with minimal constraints")
    ps_minimal = rdDistGeom.ETKDGv3()
    ps_minimal.numThreads = 1  # Single thread for stability
    ps_minimal.maxIterations = 500  # Fewer attempts but simpler
    ps_minimal.useRandomCoords = True
    ps_minimal.randomSeed = 42
    ps_minimal.enforceChirality = False  # Critical: match original working code
    
    try:
        return rdDistGeom.EmbedMultipleConfs(mol, min(n_conformers, 50), ps_minimal)  # Limit conformers
    except Exception as e:
        logger.error(f"All embedding attempts failed for organometallic molecule: {e}")
        return []