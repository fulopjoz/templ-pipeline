# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""Maximum Common Substructure (MCS) functionality for template-based pose prediction."""

import logging
import multiprocessing as mp
import time
from typing import Dict, List, Optional, Tuple, Union

from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    rdDistGeom,
    rdForceFieldHelpers,
    rdMolAlign,
    rdRascalMCES,
)
from rdkit.Geometry import Point3D

log = logging.getLogger(__name__)

N_CONFS = 200

# Molecular Geometry Validation


def validate_molecular_connectivity(mol: Chem.Mol, step_name: str = "unknown") -> bool:
    """Validate that molecular connectivity is preserved after processing.

    Args:
        mol: RDKit molecule to validate
        step_name: Name of the processing step for logging

    Returns:
        True if connectivity is preserved, False if broken
    """
    if mol is None or mol.GetNumConformers() == 0:
        return True

    try:
        # Check for disconnected fragments
        fragments = Chem.GetMolFrags(mol, asMols=False)
        if len(fragments) > 1:
            log.warning(
                f"Molecular fragmentation detected at {step_name}: {len(fragments)} fragments"
            )
            return False

        # Check for unreasonable atom positions (NaN or infinity)
        conf = mol.GetConformer(0)
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            if not (abs(pos.x) < 1000 and abs(pos.y) < 1000 and abs(pos.z) < 1000):
                log.warning(
                    f"Unreasonable atom position detected at {step_name}: atom {i} at ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})"
                )
                return False

        return True

    except Exception as e:
        log.warning(f"Connectivity validation failed at {step_name}: {e}")
        return False


def validate_molecular_geometry(
    mol: Chem.Mol, step_name: str = "unknown", log_level: str = "warning"
) -> bool:
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

            distance = (
                (pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2 + (pos1.z - pos2.z) ** 2
            ) ** 0.5
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
            dist = (
                (pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2 + (pos1.z - pos2.z) ** 2
            ) ** 0.5
            constraint_distances.append(dist)

    if constraint_distances:
        min_dist = min(constraint_distances)
        max_dist = max(constraint_distances)
        avg_dist = sum(constraint_distances) / len(constraint_distances)

        log.info(
            f"Constraint distances: min={min_dist:.3f}Å, max={max_dist:.3f}Å, avg={avg_dist:.3f}Å"
        )

        # Flag problematic constraints
        if min_dist < 1.0:
            log.warning(
                f"Very close constraints detected: min_distance={min_dist:.3f}Å"
            )
        if max_dist > 20.0:
            log.warning(
                f"Very distant constraints detected: max_distance={max_dist:.3f}Å"
            )


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

    # Convert to list for easier manipulation
    atom_indices = list(coord_map.keys())
    coordinates = list(coord_map.values())

    # Find pairs that are too close
    close_pairs = []
    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            pos1 = coordinates[i]
            pos2 = coordinates[j]
            dist = (
                (pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2 + (pos1.z - pos2.z) ** 2
            ) ** 0.5
            if dist < min_distance:
                close_pairs.append((i, j, dist))

    if not close_pairs:
        return coord_map

    log.info(
        f"Found {len(close_pairs)} constraint pairs closer than {min_distance:.3f}Å"
    )

    # Remove constraints that are too close, keeping the one with lower atom index
    indices_to_remove = set()
    for i, j, dist in close_pairs:
        # Keep the constraint with lower atom index (more central)
        if atom_indices[i] < atom_indices[j]:
            indices_to_remove.add(j)
            log.debug(
                f"Removing constraint {atom_indices[j]} (too close to {atom_indices[i]}: {dist:.3f}Å)"
            )
        else:
            indices_to_remove.add(i)
            log.debug(
                f"Removing constraint {atom_indices[i]} (too close to {atom_indices[j]}: {dist:.3f}Å)"
            )

    # Build relaxed coordinate map
    relaxed_coord_map = {}
    for i, atom_idx in enumerate(atom_indices):
        if i not in indices_to_remove:
            relaxed_coord_map[atom_idx] = coordinates[i]

    log.info(
        f"Relaxed constraints: {len(coord_map)} -> {len(relaxed_coord_map)} constraints"
    )
    return relaxed_coord_map


def simple_minimize_molecule(mol: Chem.Mol) -> bool:
    """Simple unconstrained minimization using MMFF or UFF fallback.

    This replaces the complex constrained minimization system with a simple
    approach that won't shift the molecule from its template-aligned coordinates.

    Args:
        mol: RDKit molecule with conformers to minimize

    Returns:
        True if minimization succeeded, False otherwise
    """
    if mol is None or mol.GetNumConformers() == 0:
        log.warning("Cannot minimize molecule: no conformers present")
        return False

    try:
        # Check if UFF fallback is needed for organometallic molecules
        if needs_uff_fallback(mol):
            log.debug("Using UFF minimization for organometallic molecule")
            all_ok = True
            for conf_id in range(mol.GetNumConformers()):
                try:
                    rc = AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)  # type: ignore
                except TypeError:
                    # Older RDKit versions may not support confId; optimize default conf only
                    rc = AllChem.UFFOptimizeMolecule(mol, maxIters=1000)  # type: ignore
                if rc != 0:
                    all_ok = False
            return all_ok
        else:
            log.debug("Using MMFF minimization for standard molecule")
            # Use MMFF for standard molecules - optimize all conformers
            props = rdForceFieldHelpers.MMFFGetMoleculeProperties(
                mol, mmffVariant="MMFF94s"
            )
            if props is None:
                log.debug("MMFF properties unavailable; cannot minimize with MMFF")
                return False
            all_ok = True
            for conf_id in range(mol.GetNumConformers()):
                try:
                    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
                        mol, props, confId=conf_id
                    )
                    rc = ff.Minimize(maxIts=1000)
                except Exception:
                    log.exception("MMFF minimization error on conformer %d", conf_id)
                    rc = 1
                if rc != 0:
                    all_ok = False
            return all_ok

    except Exception as e:
        log.exception(f"Simple minimization failed: {e}")
        return False


# Core MCS Functions


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
    best_rmsd = float("inf")

    for i, tpl in enumerate(refs):
        if tpl.HasProp("ca_rmsd"):
            try:
                ca_rmsd = float(tpl.GetProp("ca_rmsd"))
                if ca_rmsd < best_rmsd:
                    best_rmsd = ca_rmsd
                    best_idx = i
            except (ValueError, TypeError):
                continue

    log.info(
        f"Selected template {best_idx} with CA RMSD {best_rmsd:.3f}Å for central atom fallback"
    )
    return best_idx


def find_mcs(
    tgt: Chem.Mol, refs: List[Chem.Mol], return_details: bool = False
) -> Union[Tuple[int, str], Tuple[int, str, Dict]]:
    """RascalMCES-only MCS finding with progressive fallback strategy.

    This function uses RascalMCES exclusively for all molecule sizes to ensure
    stable performance and memory efficiency in benchmarking environments.

    Args:
        tgt: Target molecule
        refs: Reference molecules
        return_details: If True, return detailed MCS information

    Returns:
        If return_details=False: (best_template_index, smarts)
        If return_details=True: (best_template_index, smarts, mcs_details_dict)
    """
    mcs_details = {}  # Only populated if return_details=True
    min_acceptable_size = 5

    # Log molecule sizes for monitoring
    target_atoms = tgt.GetNumAtoms()
    max_template_atoms = max(mol.GetNumAtoms() for mol in refs)
    log.debug(
        f"Using RascalMCES for molecules (target: {target_atoms}, max template: {max_template_atoms})"
    )

    # Use continuous threshold reduction for optimal MCS finding
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
                            "smarts": smarts,
                        }
                        hits.append((len(atom_matches), i, smarts, mcs_info))
                    else:
                        hits.append((len(atom_matches), i, smarts))
            except (MemoryError, RuntimeError) as e:
                log.warning(
                    f"RascalMCES failed for template {i} at threshold {opts.similarityThreshold:.2f}: {e}"
                )
                continue

        if hits:
            # Get best match by size
            if return_details:
                best_size, idx, smarts, details = max(hits)
            else:
                best_size, idx, smarts = max(hits)

            # Quality control: accept small matches only at low thresholds
            if best_size >= min_acceptable_size or opts.similarityThreshold <= 0.2:
                log.info(
                    f"MCS found: size={best_size}, threshold={opts.similarityThreshold:.2f}"
                )
                if return_details:
                    return idx, smarts, details
                return idx, smarts
            else:
                log.warning(
                    f"Rejecting small MCS (size={best_size}) at threshold {opts.similarityThreshold:.2f}"
                )
        else:
            log.info(
                f"No MCS found at threshold {opts.similarityThreshold:.2f}, trying next threshold"
            )

        # Reduce threshold by 0.1 for next iteration
        opts.similarityThreshold = round(opts.similarityThreshold - 0.1, 1)

    # Central atom fallback: use template with best CA RMSD
    log.warning(
        "RascalMCS search failed at all thresholds - using central atom fallback with best CA RMSD template"
    )
    best_template_idx = find_best_ca_rmsd_template(refs)

    if return_details:
        central_details = {
            "atom_count": 1,
            "bond_count": 0,
            "similarity_score": 0.0,
            "query_atoms": [get_central_atom(tgt)],
            "template_atoms": [get_central_atom(refs[best_template_idx])],
            "smarts": "*",  # Single atom SMARTS
            "central_atom_fallback": True,
        }
        return best_template_idx, "*", central_details
    return best_template_idx, "*"


# Conformer Generation Functions


def validate_mmff_parameters(mol: Chem.Mol) -> bool:
    """Check if MMFF parameters are available for all atoms in the molecule.

    Args:
        mol: Input molecule

    Returns:
        True if MMFF parameters are available for all atoms, False otherwise
    """
    if mol is None:
        return False

    try:
        # Check if MMFF parameters are available for all atoms
        mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if mp is None:
            log.debug("MMFF parameters not available for molecule")
            return False

        # Additional check using MMFFHasAllMoleculeParams if available
        try:
            has_params = rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol)
            log.debug(f"MMFF parameter availability: {has_params}")
            return has_params
        except AttributeError:
            # Fallback if MMFFHasAllMoleculeParams is not available
            log.debug("Using fallback MMFF parameter validation")
            return mp is not None

    except Exception as e:
        log.warning(f"MMFF parameter validation failed: {e}")
        return False


def needs_uff_fallback(mol: Chem.Mol) -> bool:
    """Check if molecule needs UFF fallback due to organometallic atoms or MMFF parameter issues.

    Args:
        mol: Input molecule

    Returns:
        True if UFF fallback is needed, False otherwise
    """
    if mol is None:
        return True

    # Check for organometallic atoms (transition metals and metalloids)
    organometallic_atoms = {
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,  # Sc-Zn
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,  # Y-Cd
        57,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,  # La, Hf-Hg
        89,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,  # Ac, Rf-Cn
    }

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in organometallic_atoms:
            return True

    # Check if MMFF parameters are available
    if not validate_mmff_parameters(mol):
        log.info("MMFF parameters not available, using UFF fallback")
        return True

    return False


def embed_with_uff_fallback(
    mol: Chem.Mol,
    n_conformers: int,
    coordMap: Optional[dict] = None,
    numThreads: int = 0,
) -> List[int]:
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

    # Validate and ensure minimum conformer count of 1
    requested_conformers = max(1, int(n_conformers))

    try:

        # Try standard embedding with coordinate map
        if coordMap:
            cids = rdDistGeom.EmbedMultipleConfs(
                mol,
                numConfs=requested_conformers,
                randomSeed=42,
                numThreads=safe_threads,
                coordMap=coordMap,
                useRandomCoords=False,
                enforceChirality=False,
                maxAttempts=1000,
            )
        else:
            cids = rdDistGeom.EmbedMultipleConfs(
                mol,
                numConfs=requested_conformers,
                randomSeed=42,
                numThreads=safe_threads,
                useRandomCoords=False,
                enforceChirality=False,
                maxAttempts=1000,
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
                    numConfs=requested_conformers,
                    randomSeed=42,
                    numThreads=safe_threads,
                    coordMap=coordMap,
                    useRandomCoords=True,
                    enforceChirality=False,
                    maxAttempts=1000,
                )
            else:
                cids = rdDistGeom.EmbedMultipleConfs(
                    mol,
                    numConfs=requested_conformers,
                    randomSeed=42,
                    numThreads=safe_threads,
                    useRandomCoords=True,
                    enforceChirality=False,
                    maxAttempts=1000,
                )
            if cids:
                # Use simple UFF optimization for organogmetallic molecules
                AllChem.UFFOptimizeMolecule(mol, maxIters=1000)  # type: ignore
                log.debug(f"UFF fallback succeeded: {len(cids)} conformers")
                return list(cids)

        # Fallback 2: Try with relaxed parameters and no coordinate map
        log.debug("UFF fallback failed, trying relaxed parameters")
        cids = rdDistGeom.EmbedMultipleConfs(
            mol,
            numConfs=requested_conformers,  # Use requested conformer count for fair comparison
            randomSeed=42,
            numThreads=safe_threads,
            useRandomCoords=True,
            enforceChirality=False,
            maxAttempts=2000,  # Increase attempts
        )

        if cids:
            log.debug(f"Relaxed embedding succeeded: {len(cids)} conformers")
            return list(cids)

        # Fallback 3: Minimal parameters for difficult molecules
        log.debug("Relaxed embedding failed, trying minimal parameters")
        cids = rdDistGeom.EmbedMultipleConfs(
            mol,
            numConfs=requested_conformers,  # Use requested conformer count for fair comparison
            randomSeed=42,
            numThreads=1,  # Single thread
            useRandomCoords=True,
            enforceChirality=False,
            maxAttempts=5000,  # Many attempts
        )

        if cids:
            log.debug(f"Minimal embedding succeeded: {len(cids)} conformers")
            return list(cids)

        log.warning("All embedding methods failed")
        return []

    except Exception as e:
        log.error(f"Embedding with UFF fallback failed: {e}")
        log.error(
            f"Molecule info: atoms={mol.GetNumAtoms()}, heavy_atoms={mol.GetNumHeavyAtoms()}"
        )
        if coordMap:
            log.error(f"Coordinate map size: {len(coordMap)} constraints")
        log.error(f"Target conformers: {n_conformers}, numThreads: {numThreads}")
        return []


def constrained_embed(
    tgt: Chem.Mol,
    ref: Chem.Mol,
    smarts: str,
    n_conformers: int = N_CONFS,
    n_workers_pipeline: int = 0,
    enable_optimization: bool = False,
) -> Optional[Chem.Mol]:
    """Generate N_CONFS conformations of tgt, locking MCS atoms to ref coords."""

    # Handle central atom fallback case
    if smarts == "*":
        log.info("Using central atom positioning for pose generation")
        return central_atom_embed(
            tgt, ref, n_conformers, n_workers_pipeline, enable_optimization
        )

    # Check if UFF fallback is needed for organometallic molecules
    use_uff = needs_uff_fallback(tgt)
    if use_uff:
        log.info(
            "Detected organometallic atoms in target molecule, using UFF-compatible embedding"
        )

    # Early detection for extremely large molecules
    num_atoms = tgt.GetNumAtoms()
    if num_atoms > 150:
        log.warning(
            f"Extremely large molecule ({num_atoms} atoms) - skipping constrained embedding and using central atom fallback"
        )
        return central_atom_embed(
            tgt, ref, n_conformers, n_workers_pipeline, enable_optimization
        )

    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        log.warning(f"Invalid SMARTS pattern: {smarts}, falling back to central atom")
        return central_atom_embed(
            tgt, ref, n_conformers, n_workers_pipeline, enable_optimization
        )

    tgt_idxs = tgt.GetSubstructMatch(patt)
    ref_idxs = ref.GetSubstructMatch(patt)

    # Check for valid MCS match
    if (
        not tgt_idxs
        or not ref_idxs
        or len(tgt_idxs) != len(ref_idxs)
        or len(tgt_idxs) < 3
    ):
        log.warning(
            f"Invalid MCS match for constrained embedding. Target idx: {tgt_idxs}, Ref idx: {ref_idxs}. Proceeding with central atom embedding."
        )
        return central_atom_embed(
            tgt, ref, n_conformers, n_workers_pipeline, enable_optimization
        )

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
        log.warning(
            "MCS match failed after hydrogen addition, falling back to central atom"
        )
        return central_atom_embed(
            tgt, ref, n_conformers, n_workers_pipeline, enable_optimization
        )

    # Validate index lengths before zip operation to prevent "zip() argument 2 is longer than argument 1" error
    if len(tgt_idxs_h) != len(ref_idxs):
        log.warning(
            f"Index length mismatch after hydrogen addition: target={len(tgt_idxs_h)} vs ref={len(ref_idxs)}"
        )
        log.warning(f"Target indices: {tgt_idxs_h}")
        log.warning(f"Reference indices: {ref_idxs}")
        log.warning(
            "This indicates MCS matching inconsistency, falling back to central atom"
        )
        return central_atom_embed(
            tgt, ref, n_conformers, n_workers_pipeline, enable_optimization
        )

    # Build coordinate map for constrained embedding
    coordMap = {}
    if ref.GetNumConformers() == 0:
        # Attempt to create a minimal conformer for the reference if missing
        try:
            rdDistGeom.EmbedMolecule(ref, rdDistGeom.ETKDGv3())
        except Exception:
            log.warning(
                "Reference molecule has no conformers and embedding failed; falling back to central atom"
            )
            return central_atom_embed(
                tgt, ref, n_conformers, n_workers_pipeline, enable_optimization
            )
    ref_conf = ref.GetConformer()
    for i, (tgt_idx, ref_idx) in enumerate(zip(tgt_idxs_h, ref_idxs)):
        try:
            pos = ref_conf.GetAtomPosition(ref_idx)
            coordMap[tgt_idx] = pos
        except Exception as e:
            log.warning(f"Error mapping atom {tgt_idx} -> {ref_idx}: {e}")
            continue

    if len(coordMap) < 3:
        log.warning(
            "Insufficient coordinate constraints for embedding, falling back to central atom"
        )
        return central_atom_embed(
            tgt, ref, n_conformers, n_workers_pipeline, enable_optimization
        )

    # Log coordinate map details for debugging
    log_coordinate_map(coordMap, "initial_coordinate_map")

    # Relax constraints that are too close together
    coordMap = relax_close_constraints(coordMap, min_distance=1.0)

    if len(coordMap) < 3:
        log.warning(
            "Insufficient coordinate constraints after relaxation, falling back to central atom"
        )
        return central_atom_embed(
            tgt, ref, n_conformers, n_workers_pipeline, enable_optimization
        )

    log.info(f"Using {len(coordMap)} constraints after relaxation")

    try:
        ps = rdDistGeom.ETKDGv3()
        ps.randomSeed = 42
        ps.enforceChirality = False

        ps.numThreads = 1

        # Progressive coordinate mapping - reduce constraints until embedding succeeds
        r = []
        lrm = 0

        # Try with current constraints first (possibly relaxed if any were removed)
        log.info("Attempting embedding with current constraints")
        log_coordinate_map(coordMap, "relaxed_coordinate_map")
        ps.coordMap = coordMap

        # Enhanced error handling for conformer generation
        try:
            # Use the parameter-specified number of conformers
            r = rdDistGeom.EmbedMultipleConfs(
                target_h, numConfs=n_conformers, params=ps
            )
        except Exception as e:
            log.error(f"RDKit EmbedMultipleConfs failed: {e}")
            log.error(
                f"Target molecule info: atoms={target_h.GetNumAtoms()}, heavy_atoms={target_h.GetNumHeavyAtoms()}"
            )
            log.error(f"Coordinate map size: {len(coordMap)} constraints")
            log.error("Falling back to central atom embedding")
            return central_atom_embed(
                tgt, ref, n_conformers, n_workers_pipeline, enable_optimization
            )

        if r:
            log.info(
                f"Embedding succeeded with relaxed constraints, generated {len(r)} conformers"
            )
        else:
            log.warning(
                "Embedding failed with relaxed constraints, trying progressive reduction"
            )

            # Progressive coordinate mapping - reduce constraints until embedding succeeds
            while not r:
                cmap = {}
                for t_idx, r_idx in zip(tgt_idxs_h[lrm:], ref_idxs[lrm:]):
                    cmap[t_idx] = ref_conf.GetAtomPosition(r_idx)

                log.info(
                    f"Progressive embedding attempt {lrm + 1}: using {len(cmap)} constraints"
                )
                log_coordinate_map(cmap, f"progressive_attempt_{lrm + 1}")

                ps.coordMap = cmap
                r = rdDistGeom.EmbedMultipleConfs(
                    target_h, numConfs=n_conformers, params=ps
                )

                if not r:
                    log.warning(
                        f"Embedding attempt {lrm + 1} failed with {len(cmap)} constraints"
                    )
                else:
                    log.info(
                        f"Embedding succeeded at attempt {lrm + 1} with {len(cmap)} constraints, generated {len(r)} conformers"
                    )

                lrm += 1
                if lrm >= len(tgt_idxs_h):
                    break

        if not r:
            log.warning("Progressive embedding failed, falling back to central atom")
            return central_atom_embed(
                tgt, ref, n_conformers, n_workers_pipeline, enable_optimization
            )

        conf_ids = r

        # Check if we actually generated conformers
        if len(conf_ids) == 0:
            log.warning(
                "Embedding succeeded but generated 0 conformers, falling back to central atom"
            )
            return central_atom_embed(
                tgt, ref, n_conformers, n_workers_pipeline, enable_optimization
            )

        log.info(f"Constrained embedding succeeded: {len(conf_ids)} conformers")

        # Validate geometry after embedding
        is_valid_after_embedding = validate_molecular_geometry(
            target_h, "after_embedding", "info"
        )
        if not is_valid_after_embedding:
            log.warning("Molecular distortion detected after embedding")

        # Post-embedding alignment to correct any distortion
        # Validate index lengths again before alignment (safety check)
        if len(tgt_idxs_h) != len(ref_idxs):
            log.error(
                f"Index length mismatch during alignment: target={len(tgt_idxs_h)} vs ref={len(ref_idxs)}"
            )
            log.error(
                "This should not happen after earlier validation - indicates code logic error"
            )
            return None

        try:
            for i in range(len(conf_ids)):
                rdMolAlign.AlignMol(
                    target_h, ref, atomMap=list(zip(tgt_idxs_h, ref_idxs)), prbCid=i
                )
        except Exception as e:
            log.error(f"RDKit AlignMol failed: {e}")
            log.error(
                f"Target conformer {i}, indices: target={tgt_idxs_h}, ref={ref_idxs}"
            )
            # Continue without alignment rather than failing completely
            log.warning("Continuing without post-embedding alignment")

        # Validate geometry after alignment
        is_valid_after_alignment = validate_molecular_geometry(
            target_h, "after_alignment", "info"
        )
        if not is_valid_after_alignment:
            log.warning("Molecular distortion detected after alignment")

        # Apply force field optimization if enabled
        log.info(
            f"Force field optimization parameter: enable_optimization={enable_optimization}"
        )
        if enable_optimization:
            log.info("Applying simple unconstrained force field minimization")

            # Use simple minimization approach
            minimization_success = simple_minimize_molecule(target_h)

            if minimization_success:
                log.info("Simple minimization completed successfully")
                # Validate connectivity and geometry after minimization
                is_connected = validate_molecular_connectivity(
                    target_h, "after_minimization"
                )
                if not is_connected:
                    log.warning(
                        "Molecular connectivity issues detected after minimization"
                    )

                is_valid_after_minimization = validate_molecular_geometry(
                    target_h, "after_minimization", "info"
                )
                if not is_valid_after_minimization:
                    log.warning("Molecular distortion detected after minimization")
            else:
                log.warning("Simple minimization failed, using unoptimized conformers")
        else:
            log.info("Skipping force field minimization (enable_optimization=False)")
            log.info(
                "ETKDGv3 embedding + alignment provides sufficient geometry optimization"
            )

        return target_h

    except Exception as e:
        log.error(f"Constrained embedding failed: {e}")
        return central_atom_embed(
            tgt, ref, n_conformers, n_workers_pipeline, enable_optimization
        )


def central_atom_embed(
    tgt: Chem.Mol,
    ref: Chem.Mol,
    n_conformers: int,
    n_workers_pipeline: int = 0,
    enable_optimization: bool = False,
) -> Optional[Chem.Mol]:
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

        # Use requested conformer count for fair ablation study comparison
        # When users specify large conformer counts, it's typically for difficult cases (large molecules)
        # so artificially limiting conformers for large molecules is counterproductive
        num_atoms = tgt_copy.GetNumAtoms()
        fallback_conformers = n_conformers
        log.info(
            f"Central atom embedding for molecule ({num_atoms} atoms): using {fallback_conformers} conformers"
        )

        # Generate unconstrained conformers
        conf_ids = embed_with_uff_fallback(
            tgt_copy, fallback_conformers, numThreads=n_workers_pipeline
        )

        if not conf_ids:
            log.error("Central atom embedding failed")
            return None

        # Position at central atom of reference
        if ref.GetNumConformers() == 0:
            try:
                rdDistGeom.EmbedMolecule(ref, rdDistGeom.ETKDGv3())
            except Exception:
                log.warning(
                    "Reference molecule has no conformers and embedding failed in central_atom_embed"
                )
                return None
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
            log.info(
                "Applying simple force field minimization to central atom fallback conformers"
            )

            # Use simple minimization approach
            minimization_success = simple_minimize_molecule(tgt_copy)

            if minimization_success:
                log.info(
                    "Simple minimization completed successfully for central atom fallback"
                )
                # Validate connectivity after minimization
                is_connected = validate_molecular_connectivity(
                    tgt_copy, "central_atom_fallback_after_minimization"
                )
                if not is_connected:
                    log.warning(
                        "Molecular connectivity issues detected after central atom fallback minimization"
                    )
            else:
                log.warning("Simple minimization failed in central atom fallback")
        else:
            log.info(
                "Skipping force field minimization for central atom fallback (use --enable-optimization to enable)"
            )
            log.info(
                "Unconstrained embedding provides sufficient geometry optimization"
            )

        return tgt_copy

    except Exception as e:
        log.error(f"Central atom embedding failed: {e}")
        return None


# Utility Functions


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
        if mol_or_name.HasProp("_Name"):
            name = mol_or_name.GetProp("_Name")
        elif mol_or_name.HasProp("name"):
            name = mol_or_name.GetProp("name")
        else:
            name = fallback
    else:
        name = str(mol_or_name) if mol_or_name else fallback

    # Replace problematic characters with underscores
    safe = re.sub(r'[<>:"/\\|?*]', "_", name)
    # Remove multiple underscores
    safe = re.sub(r"_+", "_", safe)
    # Remove leading/trailing underscores
    safe = safe.strip("_")
    return safe if safe else fallback
