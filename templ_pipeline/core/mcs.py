#!/usr/bin/env python3
"""Maximum Common Substructure (MCS) functionality for template-based pose prediction."""

import logging
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    rdDistGeom,
    rdForceFieldHelpers,
    rdRascalMCES,
    rdMolAlign,
)
from rdkit.Geometry import Point3D

# Import unified error framework
try:
    from templ_pipeline.core.error_framework import UnifiedErrorTracker, ErrorCategory, ErrorSeverity
except ImportError:
    # Fallback if unified framework not available
    class ErrorCategory:
        SYSTEM = "system"
        MCS_FINDING = "mcs_finding"
        CONFORMER_GENERATION = "conformer_generation"
        MOLECULAR_ALIGNMENT = "molecular_alignment"
        COORDINATE_TRANSFORMATION = "coordinate_transformation"
        FORCE_FIELD = "force_field"
        TEMPLATE_PROCESSING = "template_processing"
        VALIDATION = "validation"
        LIGAND_EMBEDDING = "ligand_embedding"
    
    class ErrorSeverity:
        ERROR = "error"
        WARNING = "warning"
        INFO = "info"
    
    class UnifiedErrorTracker:
        def __init__(self, storage_mode="dict"):
            self._errors = {}
            self.storage_mode = storage_mode
        
        def track_error(self, pdb_id, category, message, severity="error", component="unknown", **kwargs):
            self._errors[pdb_id] = {
                "category": category,
                "message": message,
                "severity": severity,
                "component": component,
                "timestamp": time.time()
            }
        
        def has_errors(self, pdb_id=None):
            if pdb_id:
                return pdb_id in self._errors
            return len(self._errors) > 0
        
        def get_errors(self, pdb_id=None):
            if pdb_id:
                return self._errors.get(pdb_id)
            return self._errors
        
        def clear_errors(self, pdb_id=None):
            if pdb_id:
                self._errors.pop(pdb_id, None)
            else:
                self._errors.clear()

log = logging.getLogger(__name__)

# Constants from original code
N_CONFS = 50
MIN_PROTEIN_LENGTH = 20
DEFAULT_MMFF_ITERATIONS = 1000
CA_RMSD_THRESHOLD = 2.0

#  Molecular Geometry Validation 

def validate_molecular_geometry(mol: Chem.Mol, step_name: str = "unknown", log_level: str = "warning") -> bool:
    """Validate molecular geometry by checking bond lengths and connectivity.
    
    Args:
        mol: RDKit molecule to validate
        step_name: Name of the processing step for logging
        log_level: Logging level (warning, info, debug)
        
    Returns:
        True if geometry is valid, False if distorted
    """
    if mol is None or mol.GetNumConformers() == 0:
        return True
        
    try:
        conf = mol.GetConformer(0)
        bond_lengths = []
        suspicious_bonds = []
        
        for bond in mol.GetBonds():
            atom1_idx = bond.GetBeginAtomIdx()
            atom2_idx = bond.GetEndAtomIdx()
            
            pos1 = conf.GetAtomPosition(atom1_idx)
            pos2 = conf.GetAtomPosition(atom2_idx)
            
            distance = ((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)**0.5
            bond_lengths.append(distance)
            
            # Flag suspicious bond lengths
            if distance < 0.5 or distance > 3.0:
                suspicious_bonds.append((atom1_idx, atom2_idx, distance))
        
        if bond_lengths:
            min_length = min(bond_lengths)
            max_length = max(bond_lengths)
            avg_length = sum(bond_lengths) / len(bond_lengths)
            
            log_msg = f"Geometry validation at {step_name}: bonds={len(bond_lengths)}, min={min_length:.3f}Å, max={max_length:.3f}Å, avg={avg_length:.3f}Å"
            
            if suspicious_bonds:
                log_msg += f", suspicious_bonds={len(suspicious_bonds)}"
                log.warning(f"{log_msg}")
                for atom1, atom2, dist in suspicious_bonds:
                    log.warning(f"  Suspicious bond {atom1}-{atom2}: {dist:.3f}Å")
                return False
            else:
                if log_level == "info":
                    log.info(log_msg)
                elif log_level == "debug":
                    log.debug(log_msg)
                return True
                
    except Exception as e:
        log.error(f"Geometry validation failed at {step_name}: {e}")
        return False
    
    return True


def log_coordinate_map(coord_map: dict, step_name: str = "unknown"):
    """Log coordinate map details for debugging.
    
    Args:
        coord_map: Dictionary mapping atom indices to coordinates
        step_name: Name of the processing step
    """
    if not coord_map:
        log.warning(f"Empty coordinate map at {step_name}")
        return
        
    log.info(f"Coordinate map at {step_name}: {len(coord_map)} constraints")
    
    # Check for unreasonable constraint distances
    constraint_distances = []
    coord_list = list(coord_map.values())
    
    for i in range(len(coord_list)):
        for j in range(i + 1, len(coord_list)):
            pos1 = coord_list[i]
            pos2 = coord_list[j]
            dist = ((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)**0.5
            constraint_distances.append(dist)
    
    if constraint_distances:
        min_dist = min(constraint_distances)
        max_dist = max(constraint_distances)
        avg_dist = sum(constraint_distances) / len(constraint_distances)
        
        log.info(f"Constraint distances: min={min_dist:.3f}Å, max={max_dist:.3f}Å, avg={avg_dist:.3f}Å")
        
        # Flag problematic constraints
        if min_dist < 1.0:
            log.warning(f"Very close constraints detected: min_distance={min_dist:.3f}Å")
        if max_dist > 20.0:
            log.warning(f"Very distant constraints detected: max_distance={max_dist:.3f}Å")


def relax_close_constraints(coord_map: dict, min_distance: float = 1.0) -> dict:
    """Relax constraints that are too close together.
    
    Args:
        coord_map: Dictionary mapping atom indices to coordinates
        min_distance: Minimum allowed distance between constraints
        
    Returns:
        Dictionary with relaxed constraints
    """
    if not coord_map or len(coord_map) < 2:
        return coord_map
    
    from rdkit.Geometry import Point3D
    
    # Convert to list for easier manipulation
    atom_indices = list(coord_map.keys())
    coordinates = list(coord_map.values())
    
    # Find pairs that are too close
    close_pairs = []
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            pos1 = coordinates[i]
            pos2 = coordinates[j]
            dist = ((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)**0.5
            if dist < min_distance:
                close_pairs.append((i, j, dist))
    
    if not close_pairs:
        return coord_map
    
    log.info(f"Found {len(close_pairs)} constraint pairs closer than {min_distance:.3f}Å")
    
    # Remove constraints that are too close, keeping the one with lower atom index
    indices_to_remove = set()
    for i, j, dist in close_pairs:
        # Keep the constraint with lower atom index (more central)
        if atom_indices[i] < atom_indices[j]:
            indices_to_remove.add(j)
            log.debug(f"Removing constraint {atom_indices[j]} (too close to {atom_indices[i]}: {dist:.3f}Å)")
        else:
            indices_to_remove.add(i)
            log.debug(f"Removing constraint {atom_indices[i]} (too close to {atom_indices[j]}: {dist:.3f}Å)")
    
    # Build relaxed coordinate map
    relaxed_coord_map = {}
    for i, atom_idx in enumerate(atom_indices):
        if i not in indices_to_remove:
            relaxed_coord_map[atom_idx] = coordinates[i]
    
    log.info(f"Relaxed constraints: {len(coord_map)} -> {len(relaxed_coord_map)} constraints")
    return relaxed_coord_map


#  Core MCS Functions 

def get_central_atom(mol: Chem.Mol) -> int:
    """Get the idx of the central atom by picking the atom with the smallest
    sum of shortest path to all the other atoms. In case of tie returns the first one.
    
    Args:
        mol: Input molecule
        
    Returns:
        Index of the central atom
    """
    tdm = Chem.GetDistanceMatrix(mol)
    pathsum = [(idx, sum(row)) for idx, row in enumerate(tdm)]
    return sorted(pathsum, key=lambda x: x[1])[0][0]


def find_best_ca_rmsd_template(refs: List[Chem.Mol]) -> int:
    """Find template with best (lowest) CA RMSD from the reference list.
    
    Args:
        refs: List of template molecules with CA RMSD properties
        
    Returns:
        Index of template with best CA RMSD, defaults to 0 if none found
    """
    best_idx = 0
    best_rmsd = float('inf')
    
    for i, tpl in enumerate(refs):
        if tpl.HasProp("ca_rmsd"):
            try:
                ca_rmsd = float(tpl.GetProp("ca_rmsd"))
                if ca_rmsd < best_rmsd:
                    best_rmsd = ca_rmsd
                    best_idx = i
            except (ValueError, TypeError):
                continue
    
    log.info(f"Selected template {best_idx} with CA RMSD {best_rmsd:.3f}Å for central atom fallback")
    return best_idx


def find_mcs(tgt: Chem.Mol, refs: List[Chem.Mol], return_details: bool = False) -> Union[Tuple[int, str], Tuple[int, str, Dict]]:
    """RascalMCES-only MCS finding with progressive fallback strategy.
    
    This function now uses RascalMCES exclusively for all molecule sizes to avoid
    timeout issues in benchmarking. The rdFMCS path has been disabled but kept
    for potential future implementation.
    
    Args:
        tgt: Target molecule
        refs: Reference molecules
        return_details: If True, return detailed MCS information
        
    Returns:
        If return_details=False: (best_template_index, smarts)
        If return_details=True: (best_template_index, smarts, mcs_details_dict)
    """
    # TODO: Future implementation - rdFMCS for very large molecules (>200 atoms)
    # Currently disabled due to timeout issues in benchmarking.
    # The rdFMCS path can be re-enabled by uncommenting the following lines:
    # 
    # from rdkit.Chem import rdFMCS
    # target_atoms = tgt.GetNumAtoms()
    # max_template_atoms = max(mol.GetNumAtoms() for mol in refs)
    # if target_atoms > 200 or max_template_atoms > 200:
    #     log.info(f"Very large molecules detected (target: {target_atoms}, max template: {max_template_atoms}), using rdFMCS with timeout")
    #     return _find_mcs_with_fmcs(tgt, refs, return_details)
    
    mcs_details = {}  # Only populated if return_details=True
    min_acceptable_size = 5
    
    # Log molecule sizes for monitoring
    target_atoms = tgt.GetNumAtoms()
    max_template_atoms = max(mol.GetNumAtoms() for mol in refs)
    log.debug(f"Using RascalMCES for molecules (target: {target_atoms}, max template: {max_template_atoms})")
    
    # Use continuous threshold reduction like true_mcs.py
    opts = rdRascalMCES.RascalOptions()
    opts.singleLargestFrag = True
    opts.similarityThreshold = 0.9  # Start at high threshold
    
    # Continuous threshold reduction with 0.1 steps
    while opts.similarityThreshold >= 0.0:
        hits = []
        
        for i, r in enumerate(refs):
            try:
                mcr = rdRascalMCES.FindMCES(tgt, Chem.RemoveHs(r), opts)
                if mcr:
                    mcs_mol = mcr[0]
                    atom_matches = mcs_mol.atomMatches()
                    smarts = mcs_mol.smartsString
                    
                    # Store details if requested
                    if return_details:
                        bond_matches = mcs_mol.bondMatches()
                        mcs_info = {
                            "atom_count": len(atom_matches),
                            "bond_count": len(bond_matches),
                            "similarity_score": opts.similarityThreshold,
                            "query_atoms": [match[0] for match in atom_matches],
                            "template_atoms": [match[1] for match in atom_matches],
                            "smarts": smarts
                        }
                        hits.append((len(atom_matches), i, smarts, mcs_info))
                    else:
                        hits.append((len(atom_matches), i, smarts))
            except (MemoryError, RuntimeError) as e:
                log.warning(f"RascalMCES failed for template {i} at threshold {opts.similarityThreshold:.2f}: {e}")
                continue
        
        if hits:
            # Get best match by size
            if return_details:
                best_size, idx, smarts, details = max(hits)
            else:
                best_size, idx, smarts = max(hits)
            
            # Quality control: accept small matches only at low thresholds
            if best_size >= min_acceptable_size or opts.similarityThreshold <= 0.2:
                log.info(f"MCS found: size={best_size}, threshold={opts.similarityThreshold:.2f}")
                if return_details:
                    return idx, smarts, details
                return idx, smarts
            else:
                log.warning(f"Rejecting small MCS (size={best_size}) at threshold {opts.similarityThreshold:.2f}")
        else:
            log.info(f"No MCS found at threshold {opts.similarityThreshold:.2f}, trying next threshold")
        
        # Reduce threshold by 0.1 for next iteration
        opts.similarityThreshold = round(opts.similarityThreshold - 0.1, 1)
    
    # Central atom fallback: use template with best CA RMSD
    log.warning("RascalMCS search failed at all thresholds - using central atom fallback with best CA RMSD template")
    best_template_idx = find_best_ca_rmsd_template(refs)
    
    if return_details:
        central_details = {
            "atom_count": 1,
            "bond_count": 0,
            "similarity_score": 0.0,
            "query_atoms": [get_central_atom(tgt)],
            "template_atoms": [get_central_atom(refs[best_template_idx])],
            "smarts": "*",  # Single atom SMARTS
            "central_atom_fallback": True
        }
        return best_template_idx, "*", central_details
    return best_template_idx, "*"


def _find_mcs_with_fmcs(tgt: Chem.Mol, refs: List[Chem.Mol], return_details: bool = False) -> Union[Tuple[int, str], Tuple[int, str, Dict]]:
    """Memory-efficient MCS finding using rdFMCS with timeout.
    
    Args:
        tgt: Target molecule
        refs: Reference molecules
        return_details: If True, return detailed MCS information
        
    Returns:
        If return_details=False: (best_template_index, smarts)
        If return_details=True: (best_template_index, smarts, mcs_details_dict)
    """
    from rdkit.Chem import rdFMCS
    
    min_acceptable_size = 5
    threshold = 0.9
    timeout = 10  # seconds
    
    while threshold > 0.0:
        hits = []
        for i, r in enumerate(refs):
            try:
                # Use rdFMCS with timeout for memory safety
                result = rdFMCS.FindMCS(
                    [tgt, r],
                    threshold=threshold,
                    timeout=timeout,
                    atomCompare=rdFMCS.AtomCompare.CompareElements,
                    bondCompare=rdFMCS.BondCompare.CompareOrder,
                    completeRingsOnly=True
                )
                
                if result.smartsString and not result.canceled:
                    atom_count = result.numAtoms
                    bond_count = result.numBonds
                    smarts = result.smartsString
                    
                    # Store details if requested
                    if return_details:
                        # Get atom matches for detailed info
                        pattern = Chem.MolFromSmarts(smarts)
                        if pattern:
                            tgt_matches = tgt.GetSubstructMatch(pattern)
                            ref_matches = r.GetSubstructMatch(pattern)
                            
                            mcs_info = {
                                "atom_count": atom_count,
                                "bond_count": bond_count,
                                "similarity_score": threshold,
                                "query_atoms": list(tgt_matches),
                                "template_atoms": list(ref_matches),
                                "smarts": smarts
                            }
                            hits.append((atom_count, i, smarts, mcs_info))
                        else:
                            hits.append((atom_count, i, smarts))
                    else:
                        hits.append((atom_count, i, smarts))
                        
            except (MemoryError, RuntimeError) as e:
                log.warning(f"rdFMCS failed for template {i}: {e}")
                continue
        
        if hits:
            # Get best match by size
            if return_details:
                best_size, idx, smarts, details = max(hits)
            else:
                best_size, idx, smarts = max(hits)
            
            # Accept matches based on size and threshold
            if best_size >= min_acceptable_size or threshold <= 0.3:
                log.info(f"rdFMCS found: size={best_size}, threshold={threshold:.2f}")
                if return_details:
                    return idx, smarts, details
                return idx, smarts
            else:
                log.warning(f"Rejecting small rdFMCS (size={best_size}) at threshold {threshold:.2f}")
        
        # Reduce threshold and continue
        log.warning(f"No rdFMCS at threshold {threshold:.2f}, reducing…")
        threshold -= 0.1
    
    # Final fallback to central atom
    log.warning("rdFMCS search failed - using central atom fallback with best CA RMSD template")
    best_template_idx = find_best_ca_rmsd_template(refs)
    
    if return_details:
        central_details = {
            "atom_count": 1,
            "bond_count": 0,
            "similarity_score": 0.0,
            "query_atoms": [get_central_atom(tgt)],
            "template_atoms": [get_central_atom(refs[best_template_idx])],
            "smarts": "*",
            "central_atom_fallback": True
        }
        return best_template_idx, "*", central_details
    return best_template_idx, "*"


#  Conformer Generation Functions 

def needs_uff_fallback(mol: Chem.Mol) -> bool:
    """Check if molecule needs UFF fallback due to organometallic atoms.
    
    Args:
        mol: Input molecule
        
    Returns:
        True if UFF fallback is needed, False otherwise
    """
    # Check for organometallic atoms (transition metals and metalloids)
    organometallic_atoms = {
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,  # Sc-Zn
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48,  # Y-Cd
        57, 72, 73, 74, 75, 76, 77, 78, 79, 80,  # La, Hf-Hg
        89, 104, 105, 106, 107, 108, 109, 110, 111, 112  # Ac, Rf-Cn
    }
    
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in organometallic_atoms:
            return True
    return False


def minimize_with_uff(mol: Chem.Mol, conf_ids: List[int], fixed_atoms: List[int] = None, max_its: int = DEFAULT_MMFF_ITERATIONS) -> bool:
    """UFF minimization for organometallic molecules.
    
    Args:
        mol: Molecule to minimize
        conf_ids: List of conformer IDs to minimize
        fixed_atoms: List of atom indices to keep fixed
        max_its: Maximum iterations for minimization
        
    Returns:
        True if minimization succeeded, False otherwise
    """
    if fixed_atoms is None:
        fixed_atoms = []
        
    try:
        for conf_id in conf_ids:
            ff = rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=conf_id)
            if ff is None:
                log.debug(f"Could not get UFF force field for conformer {conf_id}")
                continue
                
            for idx in fixed_atoms:
                if idx < mol.GetNumAtoms():
                    ff.AddFixedPoint(idx)
                    
            ff.Minimize(maxIts=max_its)
            
        return True
        
    except Exception as e:
        log.warning(f"UFF minimization failed: {e}")
        return False


def embed_organometallic_with_constraints(mol: Chem.Mol, n_conformers: int, coordMap: dict) -> List[int]:
    """Organometallic-compatible embedding with constraints."""
    try:
        # First try with standard ETKDGv3 + constraints
        conf_ids = rdDistGeom.EmbedMultipleConfs(
            mol, 
            n_conformers, 
            randomSeed=42,
            numThreads=1,
            useRandomCoords=False,
            enforceChirality=False,
            maxAttempts=1000,
            coordMap=coordMap
        )
        if conf_ids:
            log.debug(f"Standard constrained embedding succeeded: {len(conf_ids)} conformers")
            return conf_ids
    except Exception as e:
        log.debug(f"Standard constrained embedding failed: {e}")
    
    # Try with UFF-compatible settings
    try:
        conf_ids = rdDistGeom.EmbedMultipleConfs(
            mol, 
            n_conformers, 
            randomSeed=42,
            numThreads=1,
            useRandomCoords=True,
            enforceChirality=False,
            maxAttempts=2000,
            coordMap=coordMap
        )
        if conf_ids:
            log.debug(f"UFF-compatible constrained embedding succeeded: {len(conf_ids)} conformers")
            return conf_ids
    except Exception as e:
        log.debug(f"UFF-compatible constrained embedding failed: {e}")
    
    # Final fallback: unconstrained embedding
    try:
        conf_ids = rdDistGeom.EmbedMultipleConfs(
            mol, 
            min(n_conformers, 50), 
            randomSeed=42,
            numThreads=1,
            useRandomCoords=True,
            enforceChirality=False,
            maxAttempts=500
        )
        if conf_ids:
            log.debug(f"Unconstrained fallback embedding succeeded: {len(conf_ids)} conformers")
            return conf_ids
    except Exception as e:
        log.error(f"All organometallic embedding attempts failed: {e}")
    
    return []


def embed_with_uff_fallback(mol: Chem.Mol, n_conformers: int, coordMap: dict = None, numThreads: int = 0) -> List[int]:
    """Embedding function with UFF force field fallback for organometallic molecules.
    
    Args:
        mol: Molecule to embed
        n_conformers: Number of conformers to generate
        coordMap: Coordinate map for constrained embedding
        numThreads: Number of threads to use
        
    Returns:
        List of conformer IDs generated, empty list if failed
    """
    
    # Simple thread limiting to prevent resource exhaustion
    safe_threads = min(numThreads if numThreads > 0 else mp.cpu_count(), 3)
    
    # Use fixed number of conformers regardless of molecule size
    adaptive_conformers = 200
    
    try:
                    # Skip memory check - always use 200 conformers
            
        # Try standard embedding with coordinate map
        if coordMap:
            cids = rdDistGeom.EmbedMultipleConfs(
                mol, 
                numConfs=adaptive_conformers,
                randomSeed=42,
                numThreads=safe_threads,
                coordMap=coordMap,
                useRandomCoords=False,
                enforceChirality=False,
                maxAttempts=1000
            )
        else:
            cids = rdDistGeom.EmbedMultipleConfs(
                mol, 
                numConfs=adaptive_conformers,
                randomSeed=42,
                numThreads=safe_threads,
                useRandomCoords=False,
                enforceChirality=False,
                maxAttempts=1000
            )
        
        if cids:
            log.debug(f"Standard embedding succeeded: {len(cids)} conformers")
            return list(cids)
        
        # Fallback 1: Try with UFF for organometallic molecules
        log.debug("Standard embedding failed, trying UFF fallback")
        if needs_uff_fallback(mol):
            # Use UFF-compatible embedding with random coordinates
            if coordMap:
                cids = rdDistGeom.EmbedMultipleConfs(
                    mol, 
                    numConfs=adaptive_conformers,
                    randomSeed=42,
                    numThreads=safe_threads,
                    coordMap=coordMap,
                    useRandomCoords=True,
                    enforceChirality=False,
                    maxAttempts=1000
                )
            else:
                cids = rdDistGeom.EmbedMultipleConfs(
                    mol, 
                    numConfs=adaptive_conformers,
                    randomSeed=42,
                    numThreads=safe_threads,
                    useRandomCoords=True,
                    enforceChirality=False,
                    maxAttempts=1000
                )
            if cids:
                minimize_with_uff(mol, list(cids))
                log.debug(f"UFF fallback succeeded: {len(cids)} conformers")
                return list(cids)
        
        # Fallback 2: Try with relaxed parameters and no coordinate map
        log.debug("UFF fallback failed, trying relaxed parameters")
        cids = rdDistGeom.EmbedMultipleConfs(
            mol, 
            numConfs=min(adaptive_conformers, 50),  # Further reduce conformer count
            randomSeed=42,
            numThreads=safe_threads,
            useRandomCoords=True,
            enforceChirality=False,
            maxAttempts=2000  # Increase attempts
        )
        
        if cids:
            log.debug(f"Relaxed embedding succeeded: {len(cids)} conformers")
            return list(cids)
        
        # Fallback 3: Minimal parameters for difficult molecules
        log.debug("Relaxed embedding failed, trying minimal parameters")
        cids = rdDistGeom.EmbedMultipleConfs(
            mol, 
            numConfs=min(adaptive_conformers, 20),  # Minimal conformer count
            randomSeed=42,
            numThreads=1,  # Single thread
            useRandomCoords=True,
            enforceChirality=False,
            maxAttempts=5000  # Many attempts
        )
        
        if cids:
            log.debug(f"Minimal embedding succeeded: {len(cids)} conformers")
            return list(cids)
        
        log.warning("All embedding methods failed")
        return []
        
    except Exception as e:
        log.error(f"Embedding with UFF fallback failed: {e}")
        log.error(f"Molecule info: atoms={mol.GetNumAtoms()}, heavy_atoms={mol.GetNumHeavyAtoms()}")
        if coordMap:
            log.error(f"Coordinate map size: {len(coordMap)} constraints")
        log.error(f"Target conformers: {n_conformers}, numThreads: {numThreads}")
        return []


def constrained_embed(tgt: Chem.Mol, ref: Chem.Mol, smarts: str, n_conformers: int = N_CONFS, n_workers_pipeline: int = 0, enable_optimization: bool = True) -> Optional[Chem.Mol]:
    """Generate N_CONFS conformations of tgt, locking MCS atoms to ref coords."""
    
    # Handle central atom fallback case
    if smarts == "*":
        log.info("Using central atom positioning for pose generation")
        return central_atom_embed(tgt, ref, n_conformers, n_workers_pipeline, enable_optimization)
    
    # Check if UFF fallback is needed for organometallic molecules
    use_uff = needs_uff_fallback(tgt)
    if use_uff:
        log.info("Detected organometallic atoms in target molecule, using UFF-compatible embedding")
    
    # Early detection for extremely large molecules
    num_atoms = tgt.GetNumAtoms()
    if num_atoms > 150:
        log.warning(f"Extremely large molecule ({num_atoms} atoms) - skipping constrained embedding and using central atom fallback")
        return central_atom_embed(tgt, ref, n_conformers, n_workers_pipeline)
    
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        log.warning(f"Invalid SMARTS pattern: {smarts}, falling back to central atom")
        return central_atom_embed(tgt, ref, n_conformers, n_workers_pipeline)
    
    tgt_idxs = tgt.GetSubstructMatch(patt)
    ref_idxs = ref.GetSubstructMatch(patt)
    
    # Check for valid MCS match
    if not tgt_idxs or not ref_idxs or len(tgt_idxs) != len(ref_idxs) or len(tgt_idxs) < 3:
        log.warning(f"Invalid MCS match for constrained embedding. Target idx: {tgt_idxs}, Ref idx: {ref_idxs}. Proceeding with central atom embedding.")
        return central_atom_embed(tgt, ref, n_conformers, n_workers_pipeline)
    
    # Use robust hydrogen addition for the target
    try:
        if tgt.GetNumConformers() > 0:
            tgt_no_h = Chem.RemoveHs(tgt)
            if tgt_no_h.GetNumConformers() == 0:
                orig_conf = tgt.GetConformer(0)
                new_conf = Chem.Conformer(tgt_no_h.GetNumAtoms())
                heavy_atom_idx = 0
                for i in range(tgt.GetNumAtoms()):
                    if tgt.GetAtomWithIdx(i).GetAtomicNum() != 1:  # Not hydrogen
                        if heavy_atom_idx < tgt_no_h.GetNumAtoms():
                            pos = orig_conf.GetAtomPosition(i)
                            new_conf.SetAtomPosition(heavy_atom_idx, pos)
                            heavy_atom_idx += 1
                tgt_no_h.AddConformer(new_conf, assignId=True)
            target_h = Chem.AddHs(tgt_no_h, addCoords=True)
        else:
            target_h = Chem.AddHs(tgt)
    except Exception as e:
        log.warning(f"Hydrogen addition failed: {e}, using original molecule")
        target_h = Chem.AddHs(tgt)
    
    # Recalculate matches for hydrogen-added molecule
    tgt_idxs_h = target_h.GetSubstructMatch(patt)
    if not tgt_idxs_h:
        log.warning("MCS match failed after hydrogen addition, falling back to central atom")
        return central_atom_embed(tgt, ref, n_conformers, n_workers_pipeline)
    
    # Validate index lengths before zip operation to prevent "zip() argument 2 is longer than argument 1" error
    if len(tgt_idxs_h) != len(ref_idxs):
        log.warning(f"Index length mismatch after hydrogen addition: target={len(tgt_idxs_h)} vs ref={len(ref_idxs)}")
        log.warning(f"Target indices: {tgt_idxs_h}")
        log.warning(f"Reference indices: {ref_idxs}")
        log.warning("This indicates MCS matching inconsistency, falling back to central atom")
        return central_atom_embed(tgt, ref, n_conformers, n_workers_pipeline)
    
    # Build coordinate map for constrained embedding
    coordMap = {}
    ref_conf = ref.GetConformer()
    for i, (tgt_idx, ref_idx) in enumerate(zip(tgt_idxs_h, ref_idxs)):
        try:
            pos = ref_conf.GetAtomPosition(ref_idx)
            coordMap[tgt_idx] = pos
        except Exception as e:
            log.warning(f"Error mapping atom {tgt_idx} -> {ref_idx}: {e}")
            continue
    
    if len(coordMap) < 3:
        log.warning("Insufficient coordinate constraints for embedding, falling back to central atom")
        return central_atom_embed(tgt, ref, n_conformers, n_workers_pipeline)

    # Log coordinate map details for debugging
    log_coordinate_map(coordMap, "initial_coordinate_map")
    
    # Relax constraints that are too close together
    coordMap = relax_close_constraints(coordMap, min_distance=1.0)
    
    if len(coordMap) < 3:
        log.warning("Insufficient coordinate constraints after relaxation, falling back to central atom")
        return central_atom_embed(tgt, ref, n_conformers, n_workers_pipeline)
    
    log.info(f"Using {len(coordMap)} constraints after relaxation")
    
    # Use proven working pattern from reference code
    try:
        ps = rdDistGeom.ETKDGv3()
        ps.randomSeed = 42
        ps.enforceChirality = False
        # Reduce thread pressure to prevent memory allocation errors
        # Use single thread for conformer generation to reduce thread-local storage pressure
        ps.numThreads = 1
        
        # Progressive coordinate mapping - reduce constraints until embedding succeeds
        r = -1
        lrm = 0
        
        # Try with relaxed constraints first
        log.info("Attempting embedding with relaxed constraints")
        log_coordinate_map(coordMap, "relaxed_coordinate_map")
        ps.SetCoordMap(coordMap)
        
                # Enhanced error handling for conformer generation
        try:
            # Use the parameter-specified number of conformers
            r = rdDistGeom.EmbedMultipleConfs(target_h, n_conformers, ps)
        except Exception as e:
            log.error(f"RDKit EmbedMultipleConfs failed: {e}")
            log.error(f"Target molecule info: atoms={target_h.GetNumAtoms()}, heavy_atoms={target_h.GetNumHeavyAtoms()}")
            log.error(f"Coordinate map size: {len(coordMap)} constraints")
            log.error("Falling back to central atom embedding")
            return central_atom_embed(tgt, ref, n_conformers, n_workers_pipeline)
        
        if r != -1:
            log.info(f"Embedding succeeded with relaxed constraints, generated {len(r)} conformers")
        else:
            log.warning("Embedding failed with relaxed constraints, trying progressive reduction")
            
            # Progressive coordinate mapping - reduce constraints until embedding succeeds
            while r == -1:
                cmap = {}
                for i, t in enumerate(tgt_idxs_h[lrm:]):
                    if i < len(ref_idxs):
                        cmap[t] = ref_conf.GetAtomPosition(ref_idxs[i])
                
                log.info(f"Progressive embedding attempt {lrm + 1}: using {len(cmap)} constraints")
                log_coordinate_map(cmap, f"progressive_attempt_{lrm + 1}")
                
                ps.SetCoordMap(cmap)
                r = rdDistGeom.EmbedMultipleConfs(target_h, n_conformers, ps)
                
                if r == -1:
                    log.warning(f"Embedding attempt {lrm + 1} failed with {len(cmap)} constraints")
                else:
                    log.info(f"Embedding succeeded at attempt {lrm + 1} with {len(cmap)} constraints, generated {len(r)} conformers")
                
                lrm += 1
                if lrm >= len(tgt_idxs_h):
                    break
        
        if r == -1:
            log.warning("Progressive embedding failed, falling back to central atom")
            return central_atom_embed(tgt, ref, n_conformers, n_workers_pipeline)
        
        conf_ids = r
        
        # Check if we actually generated conformers
        if len(conf_ids) == 0:
            log.warning("Embedding succeeded but generated 0 conformers, falling back to central atom")
            return central_atom_embed(tgt, ref, n_conformers, n_workers_pipeline)
        
        log.info(f"Constrained embedding succeeded: {len(conf_ids)} conformers")
        
        # Validate geometry after embedding
        is_valid_after_embedding = validate_molecular_geometry(target_h, "after_embedding", "info")
        if not is_valid_after_embedding:
            log.warning("Molecular distortion detected after embedding")
        
        # Post-embedding alignment to correct any distortion
        # Validate index lengths again before alignment (safety check)
        if len(tgt_idxs_h) != len(ref_idxs):
            log.error(f"Index length mismatch during alignment: target={len(tgt_idxs_h)} vs ref={len(ref_idxs)}")
            log.error("This should not happen after earlier validation - indicates code logic error")
            return None
        
        try:
            for i in range(len(conf_ids)):
                rdMolAlign.AlignMol(target_h, ref, atomMap=list(zip(tgt_idxs_h, ref_idxs)), prbCid=i)
        except Exception as e:
            log.error(f"RDKit AlignMol failed: {e}")
            log.error(f"Target conformer {i}, indices: target={tgt_idxs_h}, ref={ref_idxs}")
            # Continue without alignment rather than failing completely
            log.warning("Continuing without post-embedding alignment")
        
        # Validate geometry after alignment
        is_valid_after_alignment = validate_molecular_geometry(target_h, "after_alignment", "info")
        if not is_valid_after_alignment:
            log.warning("Molecular distortion detected after alignment")
        
        # Apply force field optimization if enabled
        if enable_optimization:
            log.info("Applying unconstrained force field optimization to whole molecule after MCS positioning")
            if use_uff:
                minimize_with_uff(target_h, list(conf_ids), [])  # Empty list = unconstrained optimization
            else:
                mmff_minimise_fixed_parallel(target_h, list(conf_ids), [], n_workers=n_workers_pipeline)  # Empty list = unconstrained optimization
            
            # Validate geometry after minimization
            is_valid_after_minimization = validate_molecular_geometry(target_h, "after_minimization", "info")
            if not is_valid_after_minimization:
                log.warning("Molecular distortion detected after minimization")
        else:
            log.info("Skipping force field minimization (use --enable-optimization to enable)")
            log.info("ETKDGv3 embedding + alignment provides sufficient geometry optimization")
        
        return target_h
        
    except Exception as e:
        log.error(f"Constrained embedding failed: {e}")
        return central_atom_embed(tgt, ref, n_conformers, n_workers_pipeline)


def central_atom_embed(tgt: Chem.Mol, ref: Chem.Mol, n_conformers: int, n_workers_pipeline: int = 0, enable_optimization: bool = True) -> Optional[Chem.Mol]:
    """Fallback embedding method using central atom positioning.
    
    Args:
        tgt: Target molecule
        ref: Reference molecule
        n_conformers: Number of conformers to generate
        n_workers_pipeline: Number of workers for parallel processing
        
    Returns:
        Molecule with embedded conformers or None if failed
    """
    try:
        tgt_copy = Chem.Mol(tgt)
        tgt_copy = Chem.AddHs(tgt_copy)
        
        # Use even fewer conformers for central atom fallback (more conservative)
        num_atoms = tgt_copy.GetNumAtoms()
        if num_atoms > 100:
            # Large molecules: very few conformers for fallback
            fallback_conformers = min(n_conformers, 20)
            log.info(f"Central atom fallback for large molecule ({num_atoms} atoms): using {fallback_conformers} conformers")
        else:
            # Smaller molecules: use standard adaptive count
            fallback_conformers = n_conformers
        
        # Generate unconstrained conformers
        conf_ids = embed_with_uff_fallback(tgt_copy, fallback_conformers, numThreads=n_workers_pipeline)
        
        if not conf_ids:
            log.error("Central atom embedding failed")
            return None
        
        # Position at central atom of reference
        ref_center = ref.GetConformer().GetAtomPosition(get_central_atom(ref))
        tgt_center_idx = get_central_atom(tgt_copy)
        
        # Translate all conformers to position central atom at reference center
        for conf_id in range(tgt_copy.GetNumConformers()):
            conf = tgt_copy.GetConformer(conf_id)
            current_center = conf.GetAtomPosition(tgt_center_idx)
            translation = ref_center - current_center
            
            for atom_idx in range(conf.GetNumAtoms()):
                pos = conf.GetAtomPosition(atom_idx)
                conf.SetAtomPosition(atom_idx, pos + translation)
        
        # Apply force field optimization if enabled
        if enable_optimization:
            log.info("Applying force field optimization to central atom fallback conformers")
            # Use UFF for organometallic molecules, MMFF for others
            if needs_uff_fallback(tgt_copy):
                minimize_with_uff(tgt_copy, list(range(tgt_copy.GetNumConformers())))
            else:
                mmff_minimise_fixed_parallel(tgt_copy, list(range(tgt_copy.GetNumConformers())), [], n_workers=n_workers_pipeline)
        else:
            log.info("Skipping force field minimization for central atom fallback (use --enable-optimization to enable)")
            log.info("Unconstrained embedding provides sufficient geometry optimization")
        
        return tgt_copy
        
    except Exception as e:
        log.error(f"Central atom embedding failed: {e}")
        return None


#  MMFF Minimization Functions 

def mmff_minimise_fixed_sequential(mol: Chem.Mol, conf_ids, fixed_idx, its: int = DEFAULT_MMFF_ITERATIONS):
    """Sequential MMFF minimization with fixed atom constraints.
    
    Args:
        mol: Molecule to minimize
        conf_ids: List of conformer IDs to minimize
        fixed_idx: List of atom indices to keep fixed
        its: Maximum iterations for minimization
    """
    try:
        for conf_id in conf_ids:
            mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
            if mp is None:
                log.debug(f"Could not get MMFF params for conformer {conf_id}")
                continue
                
            ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, mp, confId=conf_id)
            if ff is None:
                log.debug(f"Could not get MMFF force field for conformer {conf_id}")
                continue
                
            for idx in fixed_idx:
                if idx < mol.GetNumAtoms():
                    ff.AddFixedPoint(idx)
                    
            ff.Minimize(maxIts=its)
            
    except Exception as e:
        log.warning(f"Sequential MMFF minimization failed: {e}")


def mmff_minimise_fixed_parallel(mol_original: Chem.Mol, conf_ids: List[int], fixed_idx: List[int], its: int = DEFAULT_MMFF_ITERATIONS, n_workers: int = 0):
    """Parallel MMFF minimization with fixed atom constraints.
    
    Args:
        mol_original: Original molecule
        conf_ids: List of conformer IDs to minimize
        fixed_idx: List of atom indices to keep fixed
        its: Maximum iterations for minimization
        n_workers: Number of workers for parallel processing
    """
    # Handle n_workers = 0 (use all CPUs) or n_workers = 1 (sequential)
    if n_workers == 0:
        n_workers = mp.cpu_count() or 4  # Use all CPUs, fallback to 4
    elif n_workers == 1:
        return mmff_minimise_fixed_sequential(mol_original, conf_ids, fixed_idx, its)
    
    try:
        # Prepare data for parallel processing
        mol_smiles = Chem.MolToSmiles(mol_original)
        tasks = []
        
        for conf_id in conf_ids:
            if conf_id < mol_original.GetNumConformers():
                conf = mol_original.GetConformer(conf_id)
                coords = conf.GetPositions().tolist()
                tasks.append((mol_smiles, conf_id, coords, fixed_idx, 'MMFF94s', its))
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Check if UFF fallback is needed
            if needs_uff_fallback(mol_original):
                futures = [executor.submit(_mmff_minimize_single_conformer_task_uff_aware, task) for task in tasks]
            else:
                futures = [executor.submit(_mmff_minimize_single_conformer_task, task) for task in tasks]
            
            # Collect results
            for future in as_completed(futures):
                try:
                    conf_id, minimized_coords = future.result()
                    if minimized_coords is not None and conf_id < mol_original.GetNumConformers():
                        conf = mol_original.GetConformer(conf_id)
                        for atom_idx, coord in enumerate(minimized_coords):
                            if atom_idx < conf.GetNumAtoms():
                                conf.SetAtomPosition(atom_idx, Point3D(*coord))
                except Exception as e:
                    log.warning(f"Parallel minimization task failed: {e}")
                    
    except Exception as e:
        log.warning(f"Parallel MMFF minimization failed, falling back to sequential: {e}")
        mmff_minimise_fixed_sequential(mol_original, conf_ids, fixed_idx, its)


#  Worker Functions for Parallel Processing 

def _mmff_minimize_single_conformer_task(args_tuple):
    """Helper for parallel MMFF minimization. Modifies a conformer of a molecule (passed as SMILES).
    
    Args:
        args_tuple: Tuple containing (mol_smiles, conformer_id, initial_conformer_coords, 
                   fixed_atom_indices, mmff_variant, max_its)
    
    Returns:
        Tuple (conformer_id, list_of_minimized_coords) or None if failed
    """
    mol_smiles, conformer_id, initial_conformer_coords, fixed_atom_indices, mmff_variant, max_its = args_tuple
    
    try:
        mol = Chem.MolFromSmiles(mol_smiles)
        if not mol:
            return conformer_id, None
        
        # Add hydrogens and set up conformer
        mol = Chem.AddHs(mol)
        conf = Chem.Conformer(mol.GetNumAtoms())
        
        # Set coordinates
        for i, coord in enumerate(initial_conformer_coords):
            if i < mol.GetNumAtoms():
                conf.SetAtomPosition(i, Point3D(*coord))
        
        mol.AddConformer(conf, assignId=True)
        
        # Set up MMFF
        mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant=mmff_variant)
        if mp is None:
            return conformer_id, None
            
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, mp, confId=0)
        if ff is None:
            return conformer_id, None
            
        # Add fixed points
        for idx in fixed_atom_indices:
            if idx < mol.GetNumAtoms():
                ff.AddFixedPoint(idx)
        
        # Minimize
        ff.Minimize(maxIts=max_its)
        minimized_coords = mol.GetConformer(0).GetPositions().tolist()
        
        return conformer_id, minimized_coords
        
    except Exception as e:
        log.error(f"MMFF minimization task failed for conformer {conformer_id}: {e}")
        return conformer_id, None


def _mmff_minimize_single_conformer_task_uff_aware(args_tuple):
    """UFF-aware helper for parallel MMFF minimization with automatic fallback.
    
    Args:
        args_tuple: Tuple containing (mol_smiles, conformer_id, initial_conformer_coords, 
                   fixed_atom_indices, mmff_variant, max_its)
    
    Returns:
        Tuple (conformer_id, list_of_minimized_coords) or None if failed
    """
    mol_smiles, conformer_id, initial_conformer_coords, fixed_atom_indices, mmff_variant, max_its = args_tuple
    
    try:
        mol = Chem.MolFromSmiles(mol_smiles)
        if not mol:
            return conformer_id, None
        
        # Add hydrogens and set up conformer
        mol = Chem.AddHs(mol)
        conf = Chem.Conformer(mol.GetNumAtoms())
        
        # Set coordinates
        for i, coord in enumerate(initial_conformer_coords):
            if i < mol.GetNumAtoms():
                conf.SetAtomPosition(i, Point3D(*coord))
        
        mol.AddConformer(conf, assignId=True)
        
        # Check if UFF fallback is needed
        use_uff = needs_uff_fallback(mol)
        
        if use_uff:
            # Use UFF for organometallic molecules
            ff = rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=0)
            if ff is None:
                return conformer_id, None
            
            for idx in fixed_atom_indices:
                if idx < mol.GetNumAtoms():
                    ff.AddFixedPoint(idx)
                    
            ff.Minimize(maxIts=max_its)
            
        else:
            # Use MMFF for regular molecules
            mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant=mmff_variant)
            if mp is None:
                return conformer_id, None
                
            ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, mp, confId=0)
            if ff is None:
                return conformer_id, None
                
            for idx in fixed_atom_indices:
                if idx < mol.GetNumAtoms():
                    ff.AddFixedPoint(idx)
                    
            ff.Minimize(maxIts=max_its)
        
        minimized_coords = mol.GetConformer(0).GetPositions().tolist()
        return conformer_id, minimized_coords
        
    except Exception as e:
        log.error(f"UFF-aware minimization task failed for conformer {conformer_id}: {e}")
        return conformer_id, None


#  Utility Functions 

def safe_name(mol_or_name: Union[Chem.Mol, str], fallback: str = "unnamed") -> str:
    """Convert a molecule name or string to a filesystem-safe version.
    
    Args:
        mol_or_name: RDKit molecule object or string name
        fallback: Fallback name if extraction fails
        
    Returns:
        Filesystem-safe version of the name
    """
    import re
    
    # Extract name from molecule or use provided string
    if isinstance(mol_or_name, Chem.Mol):
        if mol_or_name.HasProp('_Name'):
            name = mol_or_name.GetProp('_Name')
        elif mol_or_name.HasProp('name'):
            name = mol_or_name.GetProp('name')
        else:
            name = fallback
    else:
        name = str(mol_or_name) if mol_or_name else fallback
    
    # Replace problematic characters with underscores
    safe = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Remove multiple underscores
    safe = re.sub(r'_+', '_', safe)
    # Remove leading/trailing underscores
    safe = safe.strip('_')
    return safe if safe else fallback