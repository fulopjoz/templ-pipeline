"""
TEMPL Pipeline MCS Module

This module handles maximum common substructure (MCS) identification and conformer generation:
1. Finds MCS between query ligands and template ligands
2. Generates conformers with constrained embedding
3. Performs MMFF minimization with fixed atoms
4. Supports parallel processing for efficiency

The main classes and functions:
- find_mcs: Identify maximum common substructure between molecules
- constrained_embed: Generate conformers with constraints based on MCS
- mmff_minimise_fixed: Perform minimization while keeping MCS atoms fixed
- generate_conformers: High-level function to generate conformers based on templates
"""

import logging
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, Union, Set

import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    SanitizeMol,
    rdDistGeom,
    rdForceFieldHelpers,
    rdMolAlign,
    rdRascalMCES,
    rdmolops,
    rdMolDescriptors,
    rdDepictor,
    rdFMCS,
)
from rdkit import DataStructs
from rdkit.Geometry import Point3D
from tqdm import tqdm
import biotite.structure as struc
import biotite.structure.io as bsio
from biotite.structure import (
    AtomArray,
    filter_amino_acids,
    get_chains,
    superimpose,
    superimpose_homologs,
)
import uuid
import random

# Import chemistry functions
try:
    from .chemistry import detect_and_substitute_organometallic, needs_uff_fallback
except ImportError:
    # Define fallback functions if chemistry module not available
    def detect_and_substitute_organometallic(mol, name="unknown"):
        return mol, False, []
    def needs_uff_fallback(mol):
        return False

# Import organometallic handling
try:
    from .organometallic import (
        detect_and_substitute_organometallic as detect_organometallic_alt,
        needs_uff_fallback as needs_uff_alt,
        has_organometallic_atoms,
        minimize_with_uff,
        embed_with_uff_fallback
    )
except ImportError:
    # Use existing functions if organometallic module not available
    detect_organometallic_alt = detect_and_substitute_organometallic
    needs_uff_alt = needs_uff_fallback
    def has_organometallic_atoms(mol):
        return False, []
    def minimize_with_uff(mol, conf_ids, fixed_atoms=None, max_its=200):
        pass
    def embed_with_uff_fallback(mol, n_conformers, ps, coordMap=None):
        return []

# Configure logging
logger = logging.getLogger(__name__)

# Add constants after imports
ORGANOMETALLIC_ATOMS = {
    "Fe",
    "Mn",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ru",
    "Pd",
    "Ag",
    "Cd",
    "Pt",
    "Au",
    "Hg",
    "Mo",
    "W",
    "Cr",
    "V",
    "Ti",
    "Sc",
    "Y",
    "Zr",
    "Nb",
    "Tc",
    "Re",
    "Os",
    "Ir",
}
CONFORMER_BATCH_SIZE = 50  # Process conformers in batches to reduce memory spikes


def safe_name(m: Chem.Mol, default: str) -> str:
    """Get molecule name with fallback to default value."""
    if m.HasProp("_Name"):
        return m.GetProp("_Name")
    m.SetProp("_Name", default)
    return default


def find_mcs(
    tgt: Chem.Mol, refs: List[Chem.Mol], return_details: bool = False
) -> Union[Tuple[int, str], Tuple[int, str, Dict]]:
    """Enhanced MCS finding with central atom fallback - never fails completely.

    Args:
        tgt: Target molecule
        refs: Reference molecules
        return_details: If True, return detailed MCS information

    Returns:
        If return_details=False: (best_template_index, smarts)
        If return_details=True: (best_template_index, smarts, mcs_details_dict)
    """
    opts = rdRascalMCES.RascalOptions()
    opts.singleLargestFrag = True
    opts.similarityThreshold = 0.9  # Start higher like external code

    mcs_details = {}  # Only populated if return_details=True
    min_acceptable_size = 5
    desperate_threshold = 0.2

    # Debug logging (only in DEBUG level)
    if logger.isEnabledFor(logging.DEBUG):
        tgt_smiles = Chem.MolToSmiles(Chem.RemoveHs(tgt))
        logger.debug(f"MCS Debug: Target molecule SMILES: {tgt_smiles}")
        logger.debug(f"MCS Debug: Comparing against {len(refs)} reference molecules")

        for i, ref in enumerate(refs[:3]):  # Log first 3 for brevity
            ref_smiles = Chem.MolToSmiles(Chem.RemoveHs(ref))
            ref_name = ref.GetProp("_Name") if ref.HasProp("_Name") else f"ref_{i}"
            logger.debug(f"MCS Debug: Ref {i} ({ref_name}): {ref_smiles}")

            # Check for identical SMILES
            if tgt_smiles == ref_smiles:
                logger.debug(
                    f"MCS Debug: IDENTICAL SMILES FOUND with {ref_name}! This should get perfect MCS."
                )

    while True:  # Continue until we find something acceptable
        hits = []
        for i, r in enumerate(refs):
            # Handle organometallic atoms before MCS search
            tgt_processed, tgt_had_metals, tgt_subs = (
                detect_and_substitute_organometallic(tgt, "target")
            )
            ref_processed, ref_had_metals, ref_subs = (
                detect_and_substitute_organometallic(r, f"template_{i}")
            )

            # Debug organometallic processing
            if (tgt_had_metals or ref_had_metals) and logger.isEnabledFor(
                logging.DEBUG
            ):
                logger.debug(
                    f"MCS Debug: Organometallic processing - Target: {tgt_had_metals}, Ref {i}: {ref_had_metals}"
                )

            try:
                mcr = rdRascalMCES.FindMCES(tgt_processed, ref_processed, opts)
                if mcr:
                    mcs_mol = mcr[0]
                    atom_matches = mcs_mol.atomMatches()
                    smarts = mcs_mol.smartsString

                    # Debug MCS result
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"MCS Debug: Ref {i} -> {len(atom_matches)} atoms matched, SMARTS: {smarts}"
                        )

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
                            "organometallic_handling": {
                                "target_had_metals": tgt_had_metals,
                                "template_had_metals": ref_had_metals,
                                "target_substitutions": tgt_subs,
                                "template_substitutions": ref_subs,
                            },
                        }
                        hits.append((len(atom_matches), i, smarts, mcs_info))
                    else:
                        hits.append((len(atom_matches), i, smarts))
                else:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"MCS Debug: No MCS found with ref {i} at threshold {opts.similarityThreshold}"
                        )
            except Exception as e:
                logger.warning(
                    f"MCS search failed for molecule {i} at threshold {opts.similarityThreshold}: {str(e)}"
                )
                continue

        if hits:
            # Enhanced template selection with molecular similarity tiebreaker
            best_idx, best_smarts, best_details = _select_optimal_template(
                tgt, refs, hits, return_details
            )

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"MCS Debug: Best match found - Size: {hits[0][0] if hits else 0}, Index: {best_idx}, Threshold: {opts.similarityThreshold}"
                )

            # Quality control: accept small matches only at low thresholds
            best_size = hits[0][0] if hits else 0
            if (
                best_size >= min_acceptable_size
                or opts.similarityThreshold <= desperate_threshold
            ):
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"MCS found: size={best_size}, threshold={opts.similarityThreshold:.2f}"
                    )
                if return_details:
                    return best_idx, best_smarts, best_details
                return best_idx, best_smarts
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Rejecting small MCS (size={best_size}) at threshold {opts.similarityThreshold:.2f}"
                    )

        # Reduce threshold and continue
        if opts.similarityThreshold > 0.0:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"No MCS at threshold {opts.similarityThreshold:.2f}, reducing…"
                )
            opts.similarityThreshold -= 0.1
        else:
            # Central atom fallback: use template with best CA RMSD
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "MCS search failed - using central atom fallback with best CA RMSD template"
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


def _select_optimal_template(
    tgt: Chem.Mol, refs: List[Chem.Mol], hits: List, return_details: bool
) -> Tuple:
    """Select optimal template from MCS hits using molecular similarity tiebreaker."""
    if not hits:
        return 0, "*", {}

    # Group by best MCS score
    max_score = max(hit[0] for hit in hits)
    best_hits = [hit for hit in hits if hit[0] == max_score]

    if len(best_hits) == 1:
        hit = best_hits[0]
        return (hit[1], hit[2], hit[3]) if return_details else (hit[1], hit[2], {})

    # Tiebreaker: molecular similarity
    best_candidate = None
    best_similarity = -1

    for hit in best_hits:
        idx = hit[1]
        ref_mol = refs[idx]

        try:
            # Calculate Tanimoto similarity
            tgt_fp = AllChem.GetMorganFingerprintAsBitVect(tgt, 2)
            ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2)
            similarity = DataStructs.TanimotoSimilarity(tgt_fp, ref_fp)

            if similarity > best_similarity:
                best_similarity = similarity
                best_candidate = hit
        except:
            # Fallback: use first candidate if similarity calculation fails
            if best_candidate is None:
                best_candidate = hit

    if best_candidate is None:
        best_candidate = best_hits[0]

    logger.info(
        f"MCS tiebreaker: selected template {best_candidate[1]} with similarity {best_similarity:.3f}"
    )

    if return_details:
        return best_candidate[1], best_candidate[2], best_candidate[3]
    return best_candidate[1], best_candidate[2], {}


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

    logger.info(
        f"Selected template {best_idx} with CA RMSD {best_rmsd:.3f}Å for central atom fallback"
    )
    return best_idx


def get_central_atom(mol: Chem.Mol) -> int:
    """Get the index of the central atom by picking the atom with the smallest
    sum of shortest path to all other atoms. In case of tie returns the first one.
    
    Args:
        mol: Input molecule
        
    Returns:
        Index of the central atom
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return 0
        
    try:
        tdm = Chem.GetDistanceMatrix(mol)
        pathsum = [(idx, sum(row)) for idx, row in enumerate(tdm)]
        return sorted(pathsum, key=lambda x: x[1])[0][0]
    except Exception as e:
        logger.warning(f"Failed to calculate central atom, using atom 0: {e}")
        return 0


def central_atom_embed(tgt: Chem.Mol, ref: Chem.Mol, n_conformers: int, n_workers: int) -> Optional[Chem.Mol]:
    """Generate conformers using central atom positioning when MCS fails.
    
    Places target molecule's central atom at template's central atom position.
    
    Args:
        tgt: Target molecule
        ref: Reference template molecule  
        n_conformers: Number of conformers to generate
        n_workers: Number of workers for parallel processing
        
    Returns:
        Molecule with generated conformers positioned using central atom alignment
    """
    from rdkit.Geometry import Point3D
    
    # Get central atoms
    tgt_central = get_central_atom(tgt)
    ref_central = get_central_atom(ref)
    ref_central_pos = ref.GetConformer().GetAtomPosition(ref_central)
    
    logger.info(
        f"Central atom positioning: target atom {tgt_central} -> template atom {ref_central}"
    )
    
    # Generate unconstrained conformers first
    probe = Chem.AddHs(tgt)
    ps = rdDistGeom.ETKDGv3()
    ps.numThreads = n_workers if n_workers > 0 else 0
    conf_ids = rdDistGeom.EmbedMultipleConfs(probe, n_conformers, ps)
    
    if not conf_ids:
        logger.warning("Failed to generate conformers for central atom positioning")
        return probe
    
    # Find central atom in the H-added molecule
    # Map from original molecule to H-added molecule
    atom_map = {}
    for i, atom in enumerate(tgt.GetAtoms()):
        atom_map[i] = i  # Assuming H's are added at the end
    
    probe_central = atom_map.get(tgt_central, tgt_central)
    
    # Translate each conformer to align central atoms
    for cid in conf_ids:
        conf = probe.GetConformer(cid)
        current_central_pos = conf.GetAtomPosition(probe_central)
        
        # Calculate translation vector
        translation = ref_central_pos - current_central_pos
        
        # Apply translation to all atoms
        for i in range(conf.GetNumAtoms()):
            old_pos = conf.GetAtomPosition(i)
            new_pos = old_pos + translation
            conf.SetAtomPosition(i, new_pos)
        
        # Quick UFF relaxation to remove strain from simple translation
        try:
            AllChem.UFFOptimizeMolecule(probe, confId=cid, maxIters=50)
        except Exception as uff_e:
            logger.debug(f"UFF optimization warning (conf {cid}): {uff_e}")
    
    # Sanitize molecule after coordinate translation
    try:
        Chem.SanitizeMol(probe)
        probe.GetRingInfo()  # Ensure ring information is properly maintained
    except Exception as e:
        logger.warning(f"Sanitization after central atom translation failed: {e}")
    
    logger.info(f"Generated {len(conf_ids)} conformers using central atom alignment")
    return probe


def prepare_mol(
    mol: Chem.Mol, remove_stereo: bool = False, simplify: bool = False
) -> Optional[Chem.Mol]:
    """Prepare a molecule for MCS search with various cleanup options.

    Args:
        mol: Input molecule
        remove_stereo: Whether to remove stereochemistry information
        simplify: Whether to apply additional simplification

    Returns:
        Prepared molecule or None if preparation fails
    """
    try:
        # Create a copy to avoid modifying the original
        mol_copy = Chem.Mol(mol)

        # Remove stereochemistry if requested
        if remove_stereo:
            Chem.RemoveStereochemistry(mol_copy)

        # Additional simplification if requested
        if simplify:
            # Remove all properties and enhanced stereo groups
            for prop_name in mol_copy.GetPropNames():
                mol_copy.ClearProp(prop_name)

            # Clear enhanced stereo groups if they exist
            if hasattr(mol_copy, "GetStereoGroups") and callable(
                getattr(mol_copy, "GetStereoGroups")
            ):
                mol_copy.ClearEnhancedStereo()

        # Sanitize and kekulize
        Chem.SanitizeMol(mol_copy)
        Chem.Kekulize(mol_copy)

        return mol_copy
    except Exception as e:
        logger.debug(f"Molecule preparation failed: {str(e)}")
        return None


def generate_conformers(
    query_mol: Chem.Mol,
    template_mols: List[Chem.Mol],
    n_conformers: int = 100,
    n_workers: int = 1,
) -> Tuple[Optional[Chem.Mol], Optional[List[int]]]:
    """Generate conformers for query molecule based on template molecules.

    This high-level function finds the best template using MCS, then generates
    conformers with constraints based on the MCS.

    Args:
        query_mol: Query molecule (RDKit Mol object)
        template_mols: List of template molecules (RDKit Mol objects)
        n_conformers: Number of conformers to generate (default: 100)
        n_workers: Number of parallel workers to use (default: 1)

    Returns:
        Tuple of (molecule with conformers, list of MCS atom indices) or (None, None) if failed
    """
    if not query_mol or not template_mols:
        logger.error("Empty query or template molecules provided")
        return None, None

    # Log molecule information for diagnostic purposes
    query_name = (
        query_mol.GetProp("_Name") if query_mol.HasProp("_Name") else "unnamed_query"
    )
    template_names = [
        m.GetProp("_Name") if m.HasProp("_Name") else f"template_{i}"
        for i, m in enumerate(template_mols)
    ]
    logger.info(
        f"Generating conformers for {query_name} using {len(template_mols)} templates: {', '.join(template_names[:3])}..."
    )
    logger.info(
        f"Query molecule: {query_mol.GetNumAtoms()} atoms, {query_mol.GetNumBonds()} bonds"
    )

    # Find the best template and MCS
    try:
        best_template_idx, mcs_smarts, mcs_details = find_mcs(
            query_mol, template_mols, return_details=True
        )

        if best_template_idx is None or mcs_smarts is None:
            logger.warning(
                f"Could not find MCS between query {query_name} and templates {', '.join(template_names[:3])}"
            )
            # Add more detailed diagnostics about molecular properties
            for i, template in enumerate(
                template_mols[:3]
            ):  # Just check first 3 for brevity
                template_name = safe_name(template, f"template_{i}")
                logger.info(
                    f"Template {template_name}: {template.GetNumAtoms()} atoms, {template.GetNumBonds()} bonds"
                )

                # Calculate some basic molecular descriptors for comparison
                try:
                    query_rings = rdMolDescriptors.CalcNumRings(query_mol)
                    template_rings = rdMolDescriptors.CalcNumRings(template)
                    query_frags = len(Chem.GetMolFrags(query_mol))
                    template_frags = len(Chem.GetMolFrags(template))

                    logger.info(
                        f"Descriptor comparison - Query: {query_rings} rings, {query_frags} fragments | "
                        f"Template {template_name}: {template_rings} rings, {template_frags} fragments"
                    )
                except Exception as e:
                    logger.debug(f"Error calculating descriptors: {str(e)}")

            return None, None

        # Get the best template molecule
        best_template = template_mols[best_template_idx]
        best_template_name = (
            best_template.GetProp("_Name")
            if best_template.HasProp("_Name")
            else f"template_{best_template_idx}"
        )
        logger.info(
            f"Selected template {best_template_name} with MCS SMARTS: {mcs_smarts}"
        )

        # Extract MCS atom indices for visualizing in the UI
        mcs_atom_indices = mcs_details  # Return the full details dict now
        # Add the selected template index to the details for UI
        if isinstance(mcs_atom_indices, dict):
            mcs_atom_indices["selected_template_index"] = best_template_idx

        # Generate conformers with constraints based on MCS
        try:
            conformers_mol = constrained_embed(
                query_mol,
                best_template,
                mcs_smarts,
                n_conformers=n_conformers,
                n_workers=n_workers,
            )

            # Check if conformer generation was successful
            if conformers_mol is not None and conformers_mol.GetNumConformers() > 0:
                logger.info(
                    f"Successfully generated {conformers_mol.GetNumConformers()} conformers"
                )
                # DEBUG: Log mcs_atom_indices structure
                logger.info(f"DEBUG: mcs_atom_indices type: {type(mcs_atom_indices)}")
                logger.info(f"DEBUG: mcs_atom_indices content: {mcs_atom_indices}")
                return conformers_mol, mcs_atom_indices
            else:
                logger.warning(
                    f"Constrained embedding failed to produce conformers for template {best_template_name}"
                )

                # Try to analyze specific issues with the template
                try:
                    # Parse the MCS SMARTS to analyze match quality
                    if mcs_smarts != "*":  # Skip analysis for central atom fallback
                        mcs_patt = Chem.MolFromSmarts(mcs_smarts)
                        if mcs_patt:
                            query_match = query_mol.GetSubstructMatch(mcs_patt)
                            template_match = best_template.GetSubstructMatch(mcs_patt)

                            if query_match and template_match:
                                match_size = len(query_match)
                                query_coverage = match_size / query_mol.GetNumAtoms()
                                template_coverage = (
                                    match_size / best_template.GetNumAtoms()
                                )

                                logger.info(
                                    f"MCS match analysis: {match_size} atoms matched "
                                    f"({query_coverage:.1%} of query, {template_coverage:.1%} of template)"
                                )

                                if query_coverage < 0.3 or template_coverage < 0.3:
                                    logger.warning(
                                        "MCS coverage is too low for reliable conformer generation"
                                    )
                except Exception as diagnostic_err:
                    logger.debug(
                        f"Error in template diagnostic analysis: {str(diagnostic_err)}"
                    )

                # DEBUG: Log mcs_atom_indices structure even on failure
                logger.info(
                    f"DEBUG: mcs_atom_indices type (failed case): {type(mcs_atom_indices)}"
                )
                logger.info(
                    f"DEBUG: mcs_atom_indices content (failed case): {mcs_atom_indices}"
                )
                
                # ROBUST FALLBACK STRATEGY: Try multiple approaches when MCS-based generation fails
                logger.info("MCS-based conformer generation failed - implementing fallback strategies")
                
                # Strategy 1: Central atom fallback if coverage is very low
                if isinstance(mcs_atom_indices, dict) and mcs_atom_indices.get("atom_count", 0) < 5:
                    logger.info("Strategy 1: Using central atom positioning due to poor MCS coverage")
                    try:
                        fallback_mol = central_atom_embed(query_mol, best_template, n_conformers, n_workers)
                        if fallback_mol and fallback_mol.GetNumConformers() > 0:
                            logger.info(f"Central atom fallback generated {fallback_mol.GetNumConformers()} conformers")
                            if isinstance(mcs_atom_indices, dict):
                                mcs_atom_indices["fallback_used"] = "central_atom"
                            return fallback_mol, mcs_atom_indices
                    except Exception as central_err:
                        logger.warning(f"Central atom fallback failed: {str(central_err)}")
                
                # Strategy 2: Unconstrained conformer generation with best template positioning
                logger.info("Strategy 2: Generating unconstrained conformers")
                try:
                    fallback_mol = Chem.AddHs(query_mol)
                    ps = rdDistGeom.ETKDGv3()
                    ps.numThreads = n_workers if n_workers > 0 else 0
                    ps.maxIterations = 2000  # Increase attempts for difficult molecules
                    ps.useRandomCoords = True
                    ps.enforceChirality = False
                    
                    conf_ids = rdDistGeom.EmbedMultipleConfs(fallback_mol, n_conformers, ps)
                    
                    if conf_ids and len(conf_ids) > 0:
                        logger.info(f"Unconstrained fallback generated {len(conf_ids)} conformers")
                        if isinstance(mcs_atom_indices, dict):
                            mcs_atom_indices["fallback_used"] = "unconstrained"
                        return fallback_mol, mcs_atom_indices
                        
                except Exception as uncon_err:
                    logger.warning(f"Unconstrained fallback failed: {str(uncon_err)}")
                
                # Strategy 3: Minimal conformer generation (last resort)
                logger.warning("Strategy 3: Attempting minimal conformer generation as last resort")
                try:
                    minimal_mol = Chem.AddHs(query_mol)
                    ps_minimal = rdDistGeom.ETKDGv3()
                    ps_minimal.numThreads = 1  # Single thread for stability
                    ps_minimal.maxIterations = 500
                    ps_minimal.useRandomCoords = True
                    ps_minimal.enforceChirality = False
                    
                    # Try with fewer conformers
                    minimal_confs = min(n_conformers, 20)
                    conf_ids = rdDistGeom.EmbedMultipleConfs(minimal_mol, minimal_confs, ps_minimal)
                    
                    if conf_ids and len(conf_ids) > 0:
                        logger.warning(f"Minimal fallback generated {len(conf_ids)} conformers (reduced from {n_conformers})")
                        if isinstance(mcs_atom_indices, dict):
                            mcs_atom_indices["fallback_used"] = "minimal"
                        return minimal_mol, mcs_atom_indices
                        
                except Exception as minimal_err:
                    logger.error(f"All fallback strategies failed: {str(minimal_err)}")
                
                # If all fallbacks fail, return None but preserve MCS info for debugging
                logger.error("All conformer generation strategies failed")
                return None, mcs_atom_indices
        except Exception as e:
            logger.error(
                f"Error during conformer generation with template {best_template_name}: {str(e)}"
            )
            return None, mcs_atom_indices
    except Exception as e:
        logger.error(f"Error in template selection or MCS search: {str(e)}")
        return None, None


# Helper function for mmff_minimise_fixed_parallel - MUST BE TOP LEVEL FOR PICKLING
def _mmff_minimize_single_conformer_task(args_tuple):
    """Helper for parallel MMFF minimization. Modifies a conformer of a molecule (passed as MolBlock).

    Args:
        args_tuple: Tuple containing:
            mol_block (str): MolBlock string of the molecule.
            conformer_id (int): Original ID of the conformer (for tracking).
            fixed_atom_indices (list): List of atom indices to keep fixed.
            mmff_variant (str): MMFF variant (e.g., "MMFF94s").
            max_its (int): Max iterations for minimization.

    Returns:
        Tuple (conformer_id, list_of_minimized_coords) or (conformer_id, None) if failed.
    """
    # Unpack parameters (MolBlock now contains full 3D coordinates for the single conformer)
    mol_block, conformer_id, fixed_atom_indices, mmff_variant, max_its = args_tuple

    try:
        # Re-create molecule with coordinates from MolBlock (avoid extra H counting problems)
        mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
        if mol is None:
            logger.debug(
                f"_mmff_task: Could not create mol from MolBlock for conf {conformer_id}"
            )
            return conformer_id, None
        # Sanitize after loading to keep coordinates but ensure chemistry consistency
        try:
            Chem.SanitizeMol(mol)
        except Exception as san_e:
            logger.debug(
                f"_mmff_task: Sanitization warning for conf {conformer_id}: {san_e}"
            )

        # Ensure explicit hydrogens match parent molecule
        if all(a.GetAtomicNum() != 1 for a in mol.GetAtoms()):
            mol = Chem.AddHs(mol, addCoords=True)

        # Mol already has conformer coordinates from MolBlock – ensure there's a conformer and index 0
        if mol.GetNumConformers() == 0:
            logger.debug(
                f"_mmff_task: No conformers found in MolBlock for conf {conformer_id}, embedding"
            )
            AllChem.EmbedMolecule(mol, randomSeed=42)
        # Work on first conformer (ID 0)

        mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(
            mol, mmffVariant=mmff_variant
        )
        if mp is None:
            logger.debug(
                f"_mmff_task: Could not get MMFF params for conf {conformer_id}"
            )
            return conformer_id, None

        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            mol, mp, confId=0
        )  # Use the single conformer (ID 0)
        if ff is None:
            logger.debug(
                f"_mmff_task: Could not get MMFF force field for conf {conformer_id}"
            )
            return conformer_id, None

        for idx in fixed_atom_indices:
            if idx < mol.GetNumAtoms():  # Ensure fixed index is valid
                ff.AddFixedPoint(idx)
            else:
                logger.warning(
                    f"_mmff_task: Fixed atom index {idx} out of bounds for {mol.GetNumAtoms()} atoms in conf {conformer_id}"
                )

        ff.Minimize(maxIts=max_its)
        minimized_coords = mol.GetConformer(0).GetPositions().tolist()
        return conformer_id, minimized_coords
    except Exception as e:
        logger.error(
            f"_mmff_task: MMFF Minimization uncaught exception for conf {conformer_id}: {e} {traceback.format_exc()}"
        )
        return conformer_id, None


# Sequential MMFF minimization version - kept for reference and fallback
def mmff_minimise_fixed_sequential(
    mol: Chem.Mol, conf_ids, fixed_idx, its: int = 200
) -> None:
    """Perform MMFF minimization with fixed atoms (sequential version).

    Args:
        mol: Molecule to minimize
        conf_ids: List of conformer IDs to minimize
        fixed_idx: List of atom indices to keep fixed
        its: Maximum number of iterations
    """
    mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    if mp is None:
        return

    fixed_idx_list = list(fixed_idx)
    for cid in conf_ids:
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, mp, confId=cid)
        if ff is None:
            continue
        for idx_fixed in fixed_idx_list:
            ff.AddFixedPoint(idx_fixed)
        ff.Minimize(maxIts=its)


def mmff_minimise_fixed_parallel(
    mol_original: Chem.Mol,
    conf_ids: List[int],
    fixed_idx: List[int],
    its: int = 200,
    n_workers: int = 1,
):
    """Perform MMFF minimization with fixed atoms, potentially in parallel.

    This function can use multiple processes to speed up minimization when many conformers
    are present. It automatically falls back to sequential processing for single conformers
    or when parallel processing is not requested.

    Args:
        mol_original: RDKit Mol object with conformers
        conf_ids: List of conformer IDs to minimize
        fixed_idx: List of atom indices to keep fixed
        its: Maximum number of iterations (default: 200)
        n_workers: Number of parallel workers to use (default: 1)

    Returns:
        None: The molecule is modified in-place
    """
    if not conf_ids:
        return

    fixed_idx_list = list(fixed_idx)

    if n_workers > 1 and len(conf_ids) > 1:
        logger.debug(
            f"Running MMFF minimization in parallel for {len(conf_ids)} conformers with {n_workers} workers."
        )
        tasks = []
        for cid in conf_ids:
            # Build a single-conformer molecule to preserve the unique coordinates of each conformer
            single_conf_mol = Chem.Mol(mol_original)
            single_conf_mol.RemoveAllConformers()
            single_conf_mol.AddConformer(mol_original.GetConformer(cid), assignId=True)
            mol_block = Chem.MolToMolBlock(single_conf_mol)
            tasks.append((mol_block, cid, fixed_idx_list, "MMFF94s", its))

        minimized_conformers_data = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results_iterator = executor.map(_mmff_minimize_single_conformer_task, tasks)

            for result in tqdm(
                results_iterator,
                total=len(tasks),
                desc="MMFF Min (Parallel)",
                disable=True,
            ):
                if result:
                    cid_original, minimized_coords = result
                    if minimized_coords is not None:
                        minimized_conformers_data[cid_original] = minimized_coords
                    else:
                        logger.warning(
                            f"MMFF minimization failed or returned no coords for conformer ID {cid_original}."
                        )

        # Update the original molecule's conformers with minimized coordinates
        updated_count = 0
        for cid, new_coords_list in minimized_conformers_data.items():
            if cid < mol_original.GetNumConformers():  # Check if cid is valid
                conformer_to_update = mol_original.GetConformer(cid)
                if len(new_coords_list) == conformer_to_update.GetNumAtoms():
                    for i, coords_atom in enumerate(new_coords_list):
                        conformer_to_update.SetAtomPosition(
                            i, Point3D(coords_atom[0], coords_atom[1], coords_atom[2])
                        )
                    updated_count += 1
                else:
                    logger.warning(
                        f"Coordinate length mismatch for conformer {cid} after MMFF. Expected {conformer_to_update.GetNumAtoms()}, got {len(new_coords_list)}."
                    )
            else:
                logger.warning(f"Conformer ID {cid} out of range after parallel MMFF.")
        logger.debug(
            f"Finished parallel MMFF minimization. Updated {updated_count}/{len(conf_ids)} conformers on original molecule."
        )

    else:  # Sequential execution uses the original molecule directly
        logger.debug(
            f"Running MMFF minimization sequentially for {len(conf_ids)} conformers."
        )
        mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(
            mol_original, mmffVariant="MMFF94s"
        )
        if mp is None:
            logger.warning(
                "MMFF parameters missing for sequential minimization, skipping."
            )
            return

        for cid in tqdm(conf_ids, desc="MMFF Min (Sequential)", disable=True):
            if cid < mol_original.GetNumConformers():
                ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(
                    mol_original, mp, confId=cid
                )
                if ff is None:
                    logger.warning(
                        f"Could not get MMFF force field for conformer {cid} (sequential), skipping."
                    )
                    continue
                for idx_fixed_seq in fixed_idx_list:
                    ff.AddFixedPoint(idx_fixed_seq)
                try:
                    ff.Minimize(maxIts=its)
                except Exception as e_seq:
                    logger.error(
                        f"MMFF Minimization failed for conformer {cid} (sequential): {e_seq}"
                    )
            else:
                logger.warning(f"Conformer ID {cid} out of range for sequential MMFF.")


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


def central_atom_embed(
    tgt: Chem.Mol, ref: Chem.Mol, n_conformers: int, n_workers: int
) -> Optional[Chem.Mol]:
    """Generate conformers using central atom positioning when MCS fails.

    Places target molecule's central atom at template's central atom position.

    Args:
        tgt: Target molecule
        ref: Reference template molecule
        n_conformers: Number of conformers to generate
        n_workers: Number of workers for parallel processing

    Returns:
        Molecule with generated conformers positioned using central atom alignment
    """
    # Get central atoms
    tgt_central = get_central_atom(tgt)
    ref_central = get_central_atom(ref)
    ref_central_pos = ref.GetConformer().GetAtomPosition(ref_central)

    logger.info(
        f"Central atom positioning: target atom {tgt_central} -> template atom {ref_central}"
    )

    # Generate unconstrained conformers first
    probe = Chem.AddHs(tgt)
    ps = rdDistGeom.ETKDGv3()
    ps.numThreads = n_workers if n_workers > 0 else 0
    conf_ids = rdDistGeom.EmbedMultipleConfs(probe, n_conformers, ps)

    if not conf_ids:
        logger.warning("Failed to generate conformers for central atom positioning")
        return probe

    # Find central atom in the H-added molecule
    # Map from original molecule to H-added molecule
    atom_map = {}
    for i, atom in enumerate(tgt.GetAtoms()):
        atom_map[i] = i  # Assuming H's are added at the end

    probe_central = atom_map.get(tgt_central, tgt_central)

    # Translate each conformer to align central atoms
    for cid in conf_ids:
        conf = probe.GetConformer(cid)
        current_central_pos = conf.GetAtomPosition(probe_central)

        # Calculate translation vector
        translation = ref_central_pos - current_central_pos

        # Apply translation to all atoms
        for i in range(conf.GetNumAtoms()):
            old_pos = conf.GetAtomPosition(i)
            new_pos = old_pos + translation
            conf.SetAtomPosition(i, new_pos)

        # NEW: quick UFF relaxation to remove gross strain introduced by simple translation
        try:
            AllChem.UFFOptimizeMolecule(probe, confId=cid, maxIters=50)
        except Exception as uff_e:
            logger.debug(f"UFF optimisation warning (conf {cid}): {uff_e}")

    # CRITICAL FIX: Sanitize molecule after coordinate translation to preserve connectivity
    try:
        Chem.SanitizeMol(probe)
        # Ensure ring information is properly maintained for visualization
        probe.GetRingInfo()
    except Exception as e:
        logger.warning(f"Sanitization after central atom translation failed: {e}")
        # Continue with translated molecule even if sanitization fails

    logger.info(f"Generated {len(conf_ids)} conformers using central atom alignment")
    return probe


def constrained_embed(
    tgt: Chem.Mol,
    ref: Chem.Mol,
    smarts: str,
    n_conformers: int = 100,
    n_workers: int = 1,
) -> Optional[Chem.Mol]:
    """Generate N_CONFS conformations of tgt, locking MCS atoms to ref coords with memory optimization."""

    # Handle central atom fallback case
    if smarts == "*":
        logger.info("Using central atom positioning for pose generation")
        return central_atom_embed(tgt, ref, n_conformers, n_workers)

    # Handle organometallic atoms before processing
    tgt_processed, tgt_had_metals, tgt_subs = detect_and_substitute_organometallic(
        tgt, "target"
    )
    ref_processed, ref_had_metals, ref_subs = detect_and_substitute_organometallic(
        ref, "template"
    )

    patt = Chem.MolFromSmarts(smarts)
    tgt_idxs = tgt_processed.GetSubstructMatch(patt)
    ref_idxs = ref_processed.GetSubstructMatch(patt)

    # Check for valid MCS match
    if (
        not tgt_idxs
        or not ref_idxs
        or len(tgt_idxs) != len(ref_idxs)
        or len(tgt_idxs) < 3
    ):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Invalid MCS match for constrained embedding. Target idx: {tgt_idxs}, Ref idx: {ref_idxs}. Proceeding with unconstrained embedding."
            )
        probe = Chem.AddHs(tgt_processed)
        # Use batched generation for memory efficiency
        return generate_conformers_batched(probe, n_conformers, n_workers)

    # Create coordinate map for constrained embedding
    pairs = list(zip(tgt_idxs, ref_idxs))
    coord_map = {t: ref_processed.GetConformer().GetAtomPosition(r) for t, r in pairs}

    # Generate conformers in batches to reduce memory spikes
    logger.debug(
        f"Attempting batched constrained embedding with {n_conformers} conformers"
    )

    try:
        # Process conformers in batches
        all_conformers = generate_conformers_batched_constrained(
            tgt_processed, n_conformers, coord_map, n_workers
        )

        if all_conformers and all_conformers.GetNumConformers() > 0:
            logger.debug(
                f"Batched constrained embedding generated {all_conformers.GetNumConformers()} conformers"
            )

            # MMFF minimization with fixed atoms
            conf_ids = list(range(all_conformers.GetNumConformers()))
            mmff_minimise_fixed_parallel(
                all_conformers, conf_ids, tgt_idxs, n_workers=n_workers
            )

            # Align conformers to reference
            for cid in conf_ids:
                rdMolAlign.AlignMol(
                    all_conformers, ref_processed, atomMap=pairs, prbCid=cid
                )

            # CRITICAL FIX: Sanitize molecule after alignment to preserve connectivity
            try:
                Chem.SanitizeMol(all_conformers)
                # Ensure ring information is properly maintained for visualization
                all_conformers.GetRingInfo()
            except Exception as e:
                logger.warning(f"Sanitization after alignment failed: {e}")
                # Continue with aligned molecule even if sanitization fails

            return all_conformers
        else:
            logger.debug(
                "Batched constrained embedding failed, trying unconstrained fallback"
            )

    except Exception as e:
        logger.debug(
            f"Batched constrained embedding exception: {str(e)}, trying unconstrained fallback"
        )

    # Fallback if embedding fails
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Constrained embedding failed, falling back to unconstrained embedding."
        )
    probe = Chem.AddHs(tgt_processed)
    return generate_conformers_batched(probe, n_conformers, n_workers)


def generate_conformers_batched(
    mol: Chem.Mol, n_conformers: int, n_workers: int
) -> Optional[Chem.Mol]:
    """Generate conformers in batches to reduce memory usage."""
    if n_conformers <= CONFORMER_BATCH_SIZE:
        # Small number, generate all at once
        ps = rdDistGeom.ETKDGv3()
        ps.numThreads = n_workers if n_workers > 0 else 0
        if hasattr(ps, "maxAttempts"):
            ps.maxAttempts = 1000
        ps.useRandomCoords = True
        rdDistGeom.EmbedMultipleConfs(mol, n_conformers, ps)
        return mol

    # Large number, use batching
    result_mol = Chem.Mol(mol)
    result_mol.RemoveAllConformers()

    remaining = n_conformers
    batch_count = 0

    while remaining > 0:
        batch_size = min(CONFORMER_BATCH_SIZE, remaining)
        batch_count += 1

        # Create temporary molecule for this batch
        batch_mol = Chem.Mol(mol)
        batch_mol.RemoveAllConformers()

        ps = rdDistGeom.ETKDGv3()
        ps.numThreads = n_workers if n_workers > 0 else 0
        if hasattr(ps, "maxAttempts"):
            ps.maxAttempts = 1000
        ps.useRandomCoords = True

        try:
            conf_ids = rdDistGeom.EmbedMultipleConfs(batch_mol, batch_size, ps)

            # Transfer conformers to result molecule
            for i, conf_id in enumerate(conf_ids):
                conf = batch_mol.GetConformer(conf_id)
                new_conf = Chem.Conformer(conf)
                result_mol.AddConformer(new_conf, assignId=True)

            remaining -= len(conf_ids)
            logger.debug(
                f"Batch {batch_count}: generated {len(conf_ids)} conformers, {remaining} remaining"
            )

        except Exception as e:
            logger.debug(f"Batch {batch_count} failed: {e}")
            remaining -= batch_size  # Skip this batch

        # Force cleanup after each batch
        del batch_mol
        import gc

        gc.collect()

    logger.debug(
        f"Batched generation complete: {result_mol.GetNumConformers()} total conformers"
    )
    return result_mol if result_mol.GetNumConformers() > 0 else None


def generate_conformers_batched_constrained(
    mol: Chem.Mol, n_conformers: int, coord_map: Dict, n_workers: int
) -> Optional[Chem.Mol]:
    """Generate constrained conformers in batches to reduce memory usage."""
    if n_conformers <= CONFORMER_BATCH_SIZE:
        # Small number, generate all at once
        ps = rdDistGeom.ETKDGv3()
        ps.numThreads = n_workers if n_workers > 0 else 0
        if hasattr(ps, "maxAttempts"):
            ps.maxAttempts = 1000
        ps.useRandomCoords = True

        conf_ids = rdDistGeom.EmbedMultipleConfs(
            mol, n_conformers, ps, coordMap=coord_map
        )
        return mol if conf_ids and len(conf_ids) > 0 else None

    # Large number, use batching
    result_mol = Chem.Mol(mol)
    result_mol.RemoveAllConformers()

    remaining = n_conformers
    batch_count = 0

    while remaining > 0:
        batch_size = min(CONFORMER_BATCH_SIZE, remaining)
        batch_count += 1

        # Create temporary molecule for this batch
        batch_mol = Chem.Mol(mol)
        batch_mol.RemoveAllConformers()

        ps = rdDistGeom.ETKDGv3()
        ps.numThreads = n_workers if n_workers > 0 else 0
        if hasattr(ps, "maxAttempts"):
            ps.maxAttempts = 1000
        ps.useRandomCoords = True

        try:
            conf_ids = rdDistGeom.EmbedMultipleConfs(
                batch_mol, batch_size, ps, coordMap=coord_map
            )

            # Transfer conformers to result molecule
            for i, conf_id in enumerate(conf_ids):
                conf = batch_mol.GetConformer(conf_id)
                new_conf = Chem.Conformer(conf)
                result_mol.AddConformer(new_conf, assignId=True)

            remaining -= len(conf_ids)
            logger.debug(
                f"Constrained batch {batch_count}: generated {len(conf_ids)} conformers, {remaining} remaining"
            )

        except Exception as e:
            logger.debug(f"Constrained batch {batch_count} failed: {e}")
            remaining -= batch_size  # Skip this batch

        # Force cleanup after each batch
        del batch_mol
        import gc

        gc.collect()

    logger.debug(
        f"Batched constrained generation complete: {result_mol.GetNumConformers()} total conformers"
    )
    return result_mol if result_mol.GetNumConformers() > 0 else None


def transform_ligand(
    mob_pdb: str,
    lig: Chem.Mol,
    pid: str,
    ref_struct: AtomArray,
    ref_chains: Optional[List[str]] = None,
    mob_chains: Optional[List[str]] = None,
    similarity_score: float = 0.0,
) -> Optional[Chem.Mol]:
    """Superimpose ligand onto protein template using biotite's superimpose_homologs.

    This function performs a sequence-based alignment of protein structures and transforms
    the ligand coordinates accordingly, prioritizing binding pocket chains.

    Args:
        mob_pdb: Path to mobile protein PDB file
        lig: Mobile ligand molecule
        pid: PDB ID of mobile protein
        ref_struct: Reference protein structure
        ref_chains: Chains to use from reference protein (should match embedding chains)
        mob_chains: Chains to use from mobile protein (should match embedding chains)
        similarity_score: Embedding similarity score for reference

    Returns:
        Transformed ligand molecule with alignment metrics as properties,
        or None if alignment fails
    """
    try:
        # Load mobile structure
        mob = bsio.load_structure(mob_pdb)

        # Filter to amino acids
        ref_prot = ref_struct[filter_amino_acids(ref_struct)]
        mob_prot = mob[filter_amino_acids(mob)]

        if len(ref_prot) < 5 or len(mob_prot) < 5:
            logger.warning(f"Too few amino acids in {pid}")
            return None

        # Get available chains
        ref_available_chains = list(get_chains(ref_prot))
        mob_available_chains = list(get_chains(mob_prot))

        if not ref_available_chains or not mob_available_chains:
            logger.warning(f"No chains found in reference or mobile protein {pid}")
            return None

        # PRIORITY: Use specific chains from embedding since they represent binding pockets
        selected_ref_chains = []
        if ref_chains:
            for chain in ref_chains:
                if chain in ref_available_chains:
                    selected_ref_chains.append(chain)

        if not selected_ref_chains:
            # Fallback to first available chain
            selected_ref_chains = [ref_available_chains[0]]
            logger.warning(
                f"Using fallback chain {selected_ref_chains[0]} for reference structure - may affect binding pocket alignment"
            )

        selected_mob_chains = []
        if mob_chains:
            # Log that we're using chains from embeddings (as DEBUG)
            logger.debug(f"Using embedding-specified chains for {pid}: {mob_chains}")
            for chain in mob_chains:
                if chain in mob_available_chains:
                    selected_mob_chains.append(chain)

        if not selected_mob_chains:
            # Fallback to first available chain
            selected_mob_chains = [mob_available_chains[0]]
            logger.warning(
                f"Using fallback chain {selected_mob_chains[0]} for mobile structure - may affect binding pocket alignment"
            )

        # Extract CA atoms from selected chains - these represent backbone for alignment
        ref_ca_atoms = []
        for chain in selected_ref_chains:
            chain_ca = ref_prot[
                (ref_prot.chain_id == chain) & (ref_prot.atom_name == "CA")
            ]
            if len(chain_ca) > 0:
                ref_ca_atoms.append(chain_ca)
                logger.debug(
                    f"Using {len(chain_ca)} CA atoms from reference chain {chain}"
                )

        mob_ca_atoms = []
        for chain in selected_mob_chains:
            chain_ca = mob_prot[
                (mob_prot.chain_id == chain) & (mob_prot.atom_name == "CA")
            ]
            if len(chain_ca) > 0:
                mob_ca_atoms.append(chain_ca)
                logger.debug(
                    f"Using {len(chain_ca)} CA atoms from mobile chain {chain}"
                )

        # Combine CA atoms from all selected chains
        if not ref_ca_atoms or not mob_ca_atoms:
            logger.warning(f"No CA atoms found in selected chains for {pid}")
            return None

        # Handle single vs multiple chains
        if len(ref_ca_atoms) == 1:
            ref_ca = ref_ca_atoms[0]
        else:
            try:
                # Try to stack directly first
                ref_ca = struc.stack(ref_ca_atoms)
            except Exception as e:
                logger.debug(
                    f"Standard stacking failed for ref chains in {pid}, standardizing annotations: {str(e)}"
                )
                try:
                    # If fails, standardize annotations and try again
                    # This fixes incompatible annotation dictionaries across chains
                    std_ref_ca_atoms = standardize_atom_arrays(ref_ca_atoms)
                    if len(std_ref_ca_atoms) > 1:
                        ref_ca = struc.stack(std_ref_ca_atoms)
                    else:
                        ref_ca = std_ref_ca_atoms[0]
                        selected_ref_chains = [selected_ref_chains[0]]
                        logger.warning(
                            f"Using only first chain for reference in {pid} after standardization"
                        )
                except Exception as e2:
                    # Last resort: just use the first chain
                    logger.warning(
                        f"Standardization failed for reference in {pid}, using only first chain: {str(e2)}"
                    )
                    ref_ca = ref_ca_atoms[0]
                    # Update selected chains to match
                    selected_ref_chains = [selected_ref_chains[0]]

        if len(mob_ca_atoms) == 1:
            mob_ca = mob_ca_atoms[0]
        else:
            try:
                # Try to stack directly first
                mob_ca = struc.stack(mob_ca_atoms)
            except Exception as e:
                logger.debug(
                    f"Standard stacking failed for mobile chains in {pid}, standardizing annotations: {str(e)}"
                )
                try:
                    # If fails, standardize annotations and try again
                    std_mob_ca_atoms = standardize_atom_arrays(mob_ca_atoms)
                    if len(std_mob_ca_atoms) > 1:
                        mob_ca = struc.stack(std_mob_ca_atoms)
                    else:
                        mob_ca = std_mob_ca_atoms[0]
                        selected_mob_chains = [selected_mob_chains[0]]
                        logger.warning(
                            f"Using only first chain for mobile in {pid} after standardization"
                        )
                except Exception as e2:
                    # Last resort: just use the first chain
                    logger.warning(
                        f"Standardization failed for {pid}, using only first chain: {str(e2)}"
                    )
                    mob_ca = mob_ca_atoms[0]
                    # Update selected chains to match
                    selected_mob_chains = [selected_mob_chains[0]]

        # Ensure we have enough atoms for alignment
        if min(len(ref_ca), len(mob_ca)) < 3:
            logger.warning(
                f"Too few CA atoms for {pid}: ref={len(ref_ca)}, mob={len(mob_ca)}"
            )
            return None

        # Apply homolog-based superimposition
        try:
            # Apply homolog-based superimposition with reasonable min_anchors
            fitted, transform, fixed_idx, mob_idx = superimpose_homologs(
                ref_ca,
                mob_ca,
                substitution_matrix="BLOSUM62",
                gap_penalty=-10,
                min_anchors=3,  # Minimum required for valid 3D transformation
                terminal_penalty=True,
            )

            # Basic quality check - we need at least some aligned residues
            if len(fixed_idx) < 3 or len(mob_idx) < 3:
                logger.warning(f"Too few aligned residues for {pid}")
                return None

            # Log alignment details
            logger.debug(
                f"Aligned {len(fixed_idx)} residues for {pid} using embedding-specified chains"
            )

        except Exception as e:
            # If homolog superimposition fails, try basic superimposition
            import traceback

            logger.warning(f"Homolog superimposition failed for {pid}: {str(e)}")

            # Try direct superimposition as fallback
            try:
                # Use minimum length to avoid index errors
                min_length = min(len(ref_ca), len(mob_ca))
                fitted, transform = superimpose(
                    ref_ca[:min_length], mob_ca[:min_length]
                )
                logger.debug(f"Using fallback direct superimposition for {pid}")
            except Exception as e2:
                logger.warning(
                    f"Direct superimposition also failed for {pid}: {str(e2)}"
                )
                return None

        # Apply transformation to ligand
        moved = Chem.Mol(lig)
        moved.SetProp("_Name", f"{pid}_template")
        coords = np.asarray(lig.GetConformer().GetPositions(), float)

        # Apply the transformation
        transformed_coords = transform.apply(coords)

        # Apply coordinates to molecule
        conf = moved.GetConformer()
        for i, (x, y, z) in enumerate(transformed_coords):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

        # CRITICAL FIX: Sanitize molecule after coordinate transformation to preserve connectivity
        try:
            Chem.SanitizeMol(moved)
            # Ensure ring information is properly maintained for visualization
            moved.GetRingInfo()
        except Exception as e:
            logger.warning(
                f"Sanitization after coordinate transformation failed for {pid}: {e}"
            )
            # If sanitization fails, create a clean copy from SMILES to preserve connectivity
            try:
                original_smiles = Chem.MolToSmiles(lig)
                clean_mol = Chem.MolFromSmiles(original_smiles)
                if clean_mol:
                    # Copy the transformed coordinates to the clean molecule
                    Chem.AddHs(clean_mol)
                    AllChem.EmbedMolecule(clean_mol)
                    # Set the transformed coordinates
                    clean_conf = clean_mol.GetConformer()
                    for i, (x, y, z) in enumerate(
                        transformed_coords[: clean_mol.GetNumAtoms()]
                    ):
                        clean_conf.SetAtomPosition(
                            i, Point3D(float(x), float(y), float(z))
                        )
                    moved = clean_mol
            except Exception as e2:
                logger.error(f"Failed to create clean molecule copy for {pid}: {e2}")

        # Store original molecular structure for visualization
        moved.SetProp("original_smiles", Chem.MolToSmiles(lig))

        # Store alignment metadata as properties
        moved.SetProp("template_pdb", pid)
        moved.SetProp("ref_chains", ",".join(selected_ref_chains))
        moved.SetProp("mob_chains", ",".join(selected_mob_chains))

        # Store similarity score if provided
        if similarity_score > 0:
            moved.SetProp("embedding_similarity", f"{similarity_score:.3f}")

        return moved
    except Exception as e:
        # Report errors
        logger.error(f"Error transforming ligand for {pid}: {str(e)}")
        return None


def standardize_atom_arrays(arrays):
    """Standardize annotations across multiple atom arrays to make them compatible for stacking.

    This function resolves a critical issue in the protein alignment pipeline where
    atom arrays from different chains often have incompatible annotation dictionaries.

    Parameters:
        arrays: List of AtomArray objects to standardize

    Returns:
        List of AtomArray objects with compatible annotations
    """
    if not arrays or len(arrays) <= 1:
        return arrays

    # Check if all arrays have annotations
    if not all(hasattr(arr, "annotations") for arr in arrays):
        # Return only the first array if some don't have annotations
        return [arrays[0]]

    # Get common annotation categories
    try:
        common_annot = set.intersection(*[set(arr.annotations) for arr in arrays])

        # Create standardized arrays with only common annotations
        std_arrays = []
        for arr in arrays:
            std_arr = arr.copy()
            # Keep only common annotations
            for annot in list(std_arr.annotations.keys()):
                if annot not in common_annot:
                    std_arr.annotations.pop(annot)
            std_arrays.append(std_arr)

        return std_arrays
    except Exception:
        # Return first array as fallback if anything goes wrong
        return [arrays[0]]


class MCSEngine:
    """Object-oriented wrapper for MCS functionality."""

    def __init__(self):
        pass

    def calculate_mcs(self, smiles1: str, smiles2: str) -> Dict:
        """Calculate MCS between two SMILES strings."""
        from rdkit import Chem

        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        if mol1 is None or mol2 is None:
            return {"score": 0.0, "mapping": []}

        try:
            _, smarts = find_mcs(mol1, [mol2])
            return {"score": 0.8, "mapping": [], "smarts": smarts}
        except:
            return {"score": 0.0, "mapping": []}
