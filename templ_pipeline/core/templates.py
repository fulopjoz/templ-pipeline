#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""Template processing functionality for the TEMPL pipeline."""

import logging
import os
from typing import Dict, List, Optional, Tuple, Set, Any

import numpy as np
from rdkit import Chem
import biotite.structure as struc

try:
    from biotite.structure import (
        AtomArray,
        filter_amino_acids,
        get_chains,
        superimpose,
        superimpose_homologs,
        to_sequence,
    )
    import biotite.structure.io as bsio
    import biotite.sequence as seq
    import biotite.sequence.align as align

    try:
        import biotite.structure.alphabet as strucalph

        STRUCTURAL_ALPHABET_AVAILABLE = True
    except ImportError:
        STRUCTURAL_ALPHABET_AVAILABLE = False
    BIOTITE_AVAILABLE = True
except ImportError:
    BIOTITE_AVAILABLE = False
    STRUCTURAL_ALPHABET_AVAILABLE = False

log = logging.getLogger(__name__)

# Constants
MIN_ANCHOR_RESIDUES = 15
MIN_PROTEIN_LENGTH = 20
MIN_CA_ATOMS_FOR_ALIGNMENT = 3
BLOSUM_GAP_PENALTY = -10
CA_RMSD_THRESHOLD = 10.0
CA_RMSD_FALLBACK_THRESHOLDS = [10.0, 15.0, 20.0]


def load_reference_protein(target_pdb: str) -> Optional[AtomArray]:
    """Load reference protein structure from PDB file.

    Args:
        target_pdb: Path to reference protein PDB file

    Returns:
        Reference protein structure as AtomArray or None if failed
    """
    if not BIOTITE_AVAILABLE:
        log.error("Biotite not available for protein loading")
        return None

    try:
        if not os.path.exists(target_pdb):
            log.error(f"Reference PDB file not found: {target_pdb}")
            return None

        ref_struct = bsio.load_structure(target_pdb)
        log.info(f"Loaded reference protein structure from {target_pdb}")
        return ref_struct

    except Exception as e:
        log.error(f"Failed to load reference protein from {target_pdb}: {e}")
        return None


def load_target_data(
    target_pdb: str, target_smiles: Optional[str] = None
) -> Tuple[Optional[str], Optional[Chem.Mol]]:
    """Load target protein and ligand data.

    Args:
        target_pdb: Path to target protein PDB file
        target_smiles: Target SMILES string (optional)

    Returns:
        Tuple of (target_smiles, target_molecule) or (None, None) if failed
    """
    try:
        if not target_smiles:
            log.error("Target SMILES not provided")
            return None, None

        # Create target molecule from SMILES
        target_mol = Chem.MolFromSmiles(target_smiles)
        if not target_mol:
            log.error(f"Failed to create molecule from SMILES: {target_smiles}")
            return None, None

        # Add hydrogens
        target_mol = Chem.AddHs(target_mol)

        log.info(f"Loaded target molecule with {target_mol.GetNumAtoms()} atoms")
        return target_smiles, target_mol

    except Exception as e:
        log.error(f"Failed to load target data: {e}")
        return None, None


def transform_ligand(
    mob_pdb: str,
    lig: Chem.Mol,
    pid: str,
    ref_struct: AtomArray,
    ref_chains: Optional[List[str]] = None,
    mob_chains: Optional[List[str]] = None,
    similarity_score: float = 0.0,
) -> Optional[Chem.Mol]:
    """Superimpose ligand onto protein template using biotite's superimpose.

    This function performs a structure-based alignment of protein C-alpha atoms and transforms
    the ligand coordinates accordingly.
    """
    if not BIOTITE_AVAILABLE:
        log.error("Biotite not available for protein alignment")
        return None

    try:
        mob = bsio.load_structure(mob_pdb)
        ref_prot = ref_struct[filter_amino_acids(ref_struct)]
        mob_prot = mob[filter_amino_acids(mob)]

        if len(ref_prot) == 0 or len(mob_prot) == 0:
            log.warning(f"No amino acids found in {pid}")
            return None

        # Validate and optimize chain selection for binding site alignment
        ref_chain_ids, mob_chain_ids = _validate_and_select_chains(
            ref_prot, mob_prot, ref_chains, mob_chains, pid
        )

        if not ref_chain_ids or not mob_chain_ids:
            log.warning(f"No valid chains found for alignment in {pid}")
            return None

        ref_ca = ref_prot[
            np.isin(ref_prot.chain_id, ref_chain_ids) & (ref_prot.atom_name == "CA")
        ]
        mob_ca = mob_prot[
            np.isin(mob_prot.chain_id, mob_chain_ids) & (mob_prot.atom_name == "CA")
        ]

        if (
            len(ref_ca) < MIN_CA_ATOMS_FOR_ALIGNMENT
            or len(mob_ca) < MIN_CA_ATOMS_FOR_ALIGNMENT
        ):
            log.warning(
                f"Not enough C-alpha atoms for alignment in {pid}: ref={len(ref_ca)}, mob={len(mob_ca)}"
            )
            return None

        # Perform structural alignment with multi-level fallback strategy
        ca_rmsd = None
        transformation = None
        alignment_method = None
        anchor_count = 0

        # Level 1: Try superimpose_homologs for similar-length sequences
        try:
            if abs(len(ref_ca) - len(mob_ca)) / max(len(ref_ca), len(mob_ca)) < 0.3:
                # Only try homolog alignment if sequences are reasonably similar in length
                fitted, transformation, fixed_idx, mob_idx = superimpose_homologs(
                    ref_ca,
                    mob_ca,
                    substitution_matrix="BLOSUM62",
                    gap_penalty=-10,
                    min_anchors=3,
                    terminal_penalty=True,
                )

                if len(fixed_idx) >= 3 and len(mob_idx) >= 3:
                    ref_subset = ref_ca[fixed_idx]
                    mob_subset = mob_ca[mob_idx]
                    fitted_mob_ca, _ = superimpose(ref_subset, mob_subset)
                    ca_rmsd = struc.rmsd(ref_subset, fitted_mob_ca)
                    anchor_count = len(fixed_idx)
                    alignment_method = "homologs"
                    log.info(
                        f"Homologous alignment for {pid}: CA RMSD = {ca_rmsd:.3f}Å using {anchor_count} anchors"
                    )
                else:
                    raise ValueError("Insufficient matched residues from homologs")
            else:
                raise ValueError(
                    "Sequence length difference too large for homolog alignment"
                )

        except Exception as e:
            log.debug(f"Homologous alignment failed for {pid}: {e}")

            # Level 2: Biotite optimal sequence alignment with anchor extraction
            try:
                transformation, ca_rmsd, anchor_count = _align_with_biotite_sequence(
                    ref_ca, mob_ca, pid
                )
                alignment_method = "sequence"

            except Exception as e2:
                log.debug(f"Biotite sequence alignment failed for {pid}: {e2}")

                # Level 3: 3Di structural alphabet alignment for remote homologs
                try:
                    transformation, ca_rmsd, anchor_count = _align_with_3di_structural(
                        ref_ca, mob_ca, pid
                    )
                    alignment_method = "3di"

                except Exception as e3:
                    log.debug(f"3Di structural alignment failed for {pid}: {e3}")

                    # Level 4: Simple centroid-based alignment as final fallback
                    try:
                        transformation, ca_rmsd, anchor_count = (
                            _align_with_centroid_fallback(ref_ca, mob_ca, pid)
                        )
                        alignment_method = "centroid"

                    except Exception as e4:
                        log.error(f"All alignment methods failed for {pid}: {e4}")
                        return None

        # Log final alignment results
        if transformation is not None:
            log.info(
                f"Structural alignment for {pid} using {alignment_method}: "
                f"CA RMSD = {ca_rmsd:.3f}Å, anchors = {anchor_count}"
            )
        else:
            log.error(f"No valid alignment found for {pid}")
            return None

        if lig.GetNumConformers() == 0:
            log.warning(f"No conformers in ligand for {pid}")
            return None

        transformed_lig = Chem.Mol(lig)
        conf = transformed_lig.GetConformer()

        # Extract all coordinates at once for efficient transformation
        coords = conf.GetPositions()

        # Apply transformation matrix to all coordinates simultaneously
        transformed_coords = transformation.apply(coords)

        # Set all transformed coordinates back to conformer
        from rdkit.Geometry import Point3D

        for i in range(conf.GetNumAtoms()):
            conf.SetAtomPosition(
                i,
                Point3D(
                    float(transformed_coords[i][0]),
                    float(transformed_coords[i][1]),
                    float(transformed_coords[i][2]),
                ),
            )

        # Validate bond lengths to ensure reasonable molecular geometry
        try:
            from rdkit.Chem import Descriptors

            bond_lengths = []
            for bond in transformed_lig.GetBonds():
                atom1_idx = bond.GetBeginAtomIdx()
                atom2_idx = bond.GetEndAtomIdx()
                pos1 = conf.GetAtomPosition(atom1_idx)
                pos2 = conf.GetAtomPosition(atom2_idx)
                distance = (
                    (pos1.x - pos2.x) ** 2
                    + (pos1.y - pos2.y) ** 2
                    + (pos1.z - pos2.z) ** 2
                ) ** 0.5
                bond_lengths.append(distance)

            # Check for unreasonable bond lengths (outside 0.5-3.0 Å range)
            if bond_lengths and (min(bond_lengths) < 0.5 or max(bond_lengths) > 3.0):
                log.warning(
                    f"Suspicious bond lengths after transformation for {pid}: min={min(bond_lengths):.2f}Å, max={max(bond_lengths):.2f}Å"
                )
        except Exception as e:
            log.debug(f"Bond length validation failed for {pid}: {e}")

        transformed_lig.SetProp("ca_rmsd", f"{ca_rmsd:.3f}")
        transformed_lig.SetProp("template_pid", pid)
        transformed_lig.SetProp("similarity_score", f"{similarity_score:.3f}")
        transformed_lig.SetProp("ref_chains", ",".join(ref_chain_ids))
        transformed_lig.SetProp("mob_chains", ",".join(mob_chain_ids))
        transformed_lig.SetProp("alignment_method", alignment_method)
        transformed_lig.SetProp("anchor_count", str(anchor_count))

        return transformed_lig

    except Exception as e:
        log.error(f"Template transformation failed for {pid}: {e}", exc_info=True)
        return None


def filter_templates_by_ca_rmsd(
    all_templates: List[Chem.Mol], ca_rmsd_threshold: float
) -> List[Chem.Mol]:
    """Filter templates by CA RMSD threshold.

    Args:
        all_templates: List of template molecules with CA RMSD properties
        ca_rmsd_threshold: Maximum CA RMSD allowed (Angstroms)

    Returns:
        List of templates that pass the CA RMSD threshold
    """
    if ca_rmsd_threshold == float("inf"):
        return all_templates

    filtered_templates = []
    for tpl in all_templates:
        if tpl.HasProp("ca_rmsd"):
            try:
                ca_rmsd = float(tpl.GetProp("ca_rmsd"))
                if ca_rmsd <= ca_rmsd_threshold:
                    filtered_templates.append(tpl)
            except (ValueError, TypeError):
                # If CA RMSD property exists but can't be parsed, skip this template
                continue
        else:
            # If no CA RMSD property, include in filtered list (shouldn't happen in normal flow)
            filtered_templates.append(tpl)

    return filtered_templates


def get_templates_with_progressive_fallback(
    all_templates: List[Chem.Mol], fallback_thresholds: List[float] = None
) -> Tuple[List[Chem.Mol], float, bool]:
    """Apply progressive CA RMSD fallback with central atom final fallback.

    Args:
        all_templates: List of all available template molecules
        fallback_thresholds: List of CA RMSD thresholds to try (default: global constant)

    Returns:
        Tuple of (valid_templates, threshold_used, use_central_atom_fallback)
    """
    if fallback_thresholds is None:
        fallback_thresholds = CA_RMSD_FALLBACK_THRESHOLDS

    for threshold in fallback_thresholds:
        filtered_templates = filter_templates_by_ca_rmsd(all_templates, threshold)
        if filtered_templates:
            if threshold > CA_RMSD_THRESHOLD:
                log.warning(
                    f"Using relaxed CA RMSD threshold ({threshold}Å) - found {len(filtered_templates)} templates (poses may be less accurate)"
                )
            else:
                log.info(
                    f"Found {len(filtered_templates)} templates with CA RMSD ≤ {threshold}Å"
                )
            return filtered_templates, threshold, False

    # Fallback: find template with smallest CA RMSD and use central atom positioning
    best_template = None
    best_rmsd = float("inf")

    for tpl in all_templates:
        if tpl.HasProp("ca_rmsd"):
            try:
                ca_rmsd = float(tpl.GetProp("ca_rmsd"))
                if ca_rmsd < best_rmsd:
                    best_rmsd = ca_rmsd
                    best_template = tpl
            except (ValueError, TypeError):
                continue

    # If no template has CA RMSD, use first available
    if best_template is None and all_templates:
        best_template = all_templates[0]
        best_rmsd = "N/A"

    if best_template:
        log.warning(
            f"Using central atom fallback with best available template (CA RMSD: {best_rmsd}Å)"
        )
        return [best_template], float("inf"), True

    # This should never happen since we have templates
    log.error("No templates available for central atom fallback")
    return [], float("inf"), False


def pdb_path(pid: str, data_dir: str = "data") -> Optional[str]:
    """Find protein PDB file in either refined-set or other-PL directories.

    Args:
        pid: PDB ID to find
        data_dir: Base data directory

    Returns:
        Path to PDB file or None if not found
    """
    # Convert PDB ID to lowercase for file system compatibility
    pid_lower = pid.lower()

    # Common PDB file locations
    possible_paths = [
        f"{data_dir}/PDBBind/PDBbind_v2020_refined/{pid_lower}/{pid_lower}_protein.pdb",
        f"{data_dir}/PDBBind/PDBbind_v2020_other_PL/{pid_lower}/{pid_lower}_protein.pdb",
        f"{data_dir}/PDBBind/PDBbind_v2020_refined/refined-set/{pid_lower}/{pid_lower}_protein.pdb",
        f"{data_dir}/PDBBind/PDBbind_v2020_other_PL/v2020-other-PL/{pid_lower}/{pid_lower}_protein.pdb",
        f"{data_dir}/pdbs/{pid_lower}.pdb",
        f"{data_dir}/proteins/{pid_lower}_protein.pdb",
        f"{pid_lower}.pdb",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    log.warning(f"PDB file for {pid} not found in any standard location")
    return None


def ligand_path(pid: str, data_dir: str = "data") -> Optional[str]:
    """Find ligand file for a given PDB ID.

    Args:
        pid: PDB ID to find
        data_dir: Base data directory

    Returns:
        Path to ligand file or None if not found
    """
    # Convert PDB ID to lowercase for file system compatibility
    pid_lower = pid.lower()

    # Common ligand file locations
    possible_paths = [
        f"{data_dir}/PDBBind/PDBbind_v2020_refined/refined-set/{pid_lower}/{pid_lower}_ligand.sdf",
        f"{data_dir}/PDBBind/PDBbind_v2020_other_PL/v2020-other-PL/{pid_lower}/{pid_lower}_ligand.sdf",
        f"{data_dir}/ligands/{pid_lower}.sdf",
        f"{data_dir}/ligands/{pid_lower}_ligand.sdf",
        f"{pid_lower}_ligand.sdf",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    log.warning(f"Ligand file for {pid} not found in any standard location")
    return None


def _rmsd_from_alignment(
    fixed_ca: AtomArray, mobile_ca: AtomArray, alignment, pid: str
) -> Tuple[AtomArray, float, int]:
    """
    Extract corresponding residues from alignment and perform superimposition.
    Based on biotite documentation pattern.
    """
    if not BIOTITE_AVAILABLE:
        raise ImportError("Biotite not available")

    # Get alignment codes and find anchors
    alignment_codes = align.get_codes(alignment)

    # Create substitution matrix for anchor quality assessment
    matrix = align.SubstitutionMatrix.std_protein_matrix()

    # Anchors must be structurally similar and without gaps
    anchor_mask = (
        # Anchors must be structurally similar (positive score)
        (matrix.score_matrix()[alignment_codes[0], alignment_codes[1]] > 0)
        # Gaps are not anchors
        & (alignment_codes != -1).all(axis=0)
    )

    superimposition_anchors = alignment.trace[anchor_mask]

    if len(superimposition_anchors) < MIN_CA_ATOMS_FOR_ALIGNMENT:
        raise ValueError(f"Insufficient anchors found: {len(superimposition_anchors)}")

    # Extract anchor atoms
    fixed_anchors = fixed_ca[superimposition_anchors[:, 0]]
    mobile_anchors = mobile_ca[superimposition_anchors[:, 1]]

    # Perform superimposition
    mobile_anchors_fitted, transformation = struc.superimpose(
        fixed_anchors, mobile_anchors
    )
    rmsd = struc.rmsd(fixed_anchors, mobile_anchors_fitted)

    return transformation, rmsd, len(superimposition_anchors)


def _align_with_biotite_sequence(
    ref_ca: AtomArray, mob_ca: AtomArray, pid: str
) -> Tuple[object, float, int]:
    """
    Level 2: Biotite optimal sequence alignment with anchor extraction.
    """
    if not BIOTITE_AVAILABLE:
        raise ImportError("Biotite not available")

    # Convert structures to sequences
    ref_sequences, _ = to_sequence(ref_ca)
    mob_sequences, _ = to_sequence(mob_ca)

    if not ref_sequences or not mob_sequences:
        raise ValueError("Could not extract sequences from structures")

    ref_seq = ref_sequences[0]  # Take first chain
    mob_seq = mob_sequences[0]  # Take first chain

    # Perform optimal alignment
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    alignments = align.align_optimal(
        ref_seq, mob_seq, matrix, gap_penalty=(-10, -1), terminal_penalty=False
    )

    if not alignments:
        raise ValueError("No alignment found")

    best_alignment = alignments[0]

    # Extract anchors and perform superimposition
    transformation, rmsd, anchor_count = _rmsd_from_alignment(
        ref_ca, mob_ca, best_alignment, pid
    )

    log.info(
        f"Biotite sequence alignment for {pid}: "
        f"CA RMSD = {rmsd:.3f}Å using {anchor_count} anchors"
    )

    return transformation, rmsd, anchor_count


def _align_with_3di_structural(
    ref_ca: AtomArray, mob_ca: AtomArray, pid: str
) -> Tuple[object, float, int]:
    """
    Level 3: 3Di structural alphabet alignment for remote homologs.
    """
    if not BIOTITE_AVAILABLE or not STRUCTURAL_ALPHABET_AVAILABLE:
        raise ImportError("Biotite structural alphabet not available")

    try:
        # Convert structures to 3Di sequences
        ref_3di_sequences, _ = strucalph.to_3di(ref_ca)
        mob_3di_sequences, _ = strucalph.to_3di(mob_ca)

        if not ref_3di_sequences or not mob_3di_sequences:
            raise ValueError("Could not extract 3Di sequences")

        ref_3di = ref_3di_sequences[0]  # Take first chain
        mob_3di = mob_3di_sequences[0]  # Take first chain

        # Perform 3Di alignment
        matrix = align.SubstitutionMatrix.std_3di_matrix()
        alignments = align.align_optimal(
            ref_3di, mob_3di, matrix, gap_penalty=(-10, -1), terminal_penalty=False
        )

        if not alignments:
            raise ValueError("No 3Di alignment found")

        best_alignment = alignments[0]

        # Extract anchors and perform superimposition
        transformation, rmsd, anchor_count = _rmsd_from_alignment(
            ref_ca, mob_ca, best_alignment, pid
        )

        log.info(
            f"3Di structural alignment for {pid}: "
            f"CA RMSD = {rmsd:.3f}Å using {anchor_count} anchors"
        )

        return transformation, rmsd, anchor_count

    except Exception as e:
        raise ValueError(f"3Di alignment failed: {e}")


def _align_with_centroid_fallback(
    ref_ca: AtomArray, mob_ca: AtomArray, pid: str
) -> Tuple[object, float, int]:
    """
    Level 4: Simple centroid-based alignment as final fallback.
    """
    if not BIOTITE_AVAILABLE:
        raise ImportError("Biotite not available")

    # Use the first N atoms from both structures where N is the minimum length
    min_length = min(len(ref_ca), len(mob_ca))

    if min_length < MIN_CA_ATOMS_FOR_ALIGNMENT:
        raise ValueError(f"Insufficient atoms for centroid alignment: {min_length}")

    # Use first N atoms for alignment
    ref_subset = ref_ca[:min_length]
    mob_subset = mob_ca[:min_length]

    # Perform simple superimposition
    fitted_mob_ca, transformation = struc.superimpose(ref_subset, mob_subset)
    rmsd = struc.rmsd(ref_subset, fitted_mob_ca)

    log.warning(
        f"Centroid fallback alignment for {pid}: "
        f"CA RMSD = {rmsd:.3f}Å using {min_length} atoms (may be less accurate)"
    )

    return transformation, rmsd, min_length


def _validate_and_select_chains(
    ref_prot: AtomArray,
    mob_prot: AtomArray,
    ref_chains: Optional[List[str]],
    mob_chains: Optional[List[str]],
    pid: str,
) -> Tuple[List[str], List[str]]:
    """
    Validate and select optimal chains for binding site alignment.

    Args:
        ref_prot: Reference protein structure
        mob_prot: Mobile protein structure
        ref_chains: Specified reference chains (from embedding data)
        mob_chains: Specified mobile chains (from embedding data)
        pid: PDB ID for logging

    Returns:
        Tuple of (validated_ref_chains, validated_mob_chains)
    """
    if not BIOTITE_AVAILABLE:
        return [], []

    # Get all available chains (convert to list for easier handling)
    available_ref_chains = list(get_chains(ref_prot))
    available_mob_chains = list(get_chains(mob_prot))

    # If chains are specified (from embedding data), validate them
    if ref_chains and mob_chains:
        # Validate specified chains exist
        valid_ref_chains = [c for c in ref_chains if c in available_ref_chains]
        valid_mob_chains = [c for c in mob_chains if c in available_mob_chains]

        if valid_ref_chains and valid_mob_chains:
            log.info(
                f"Using specified binding site chains for {pid}: "
                f"ref={valid_ref_chains}, mob={valid_mob_chains}"
            )
            return valid_ref_chains, valid_mob_chains
        else:
            log.warning(
                f"Specified chains not found in {pid}, falling back to all chains"
            )

    # Fallback to all available chains
    if available_ref_chains and available_mob_chains:
        log.debug(
            f"Using all available chains for {pid}: "
            f"ref={available_ref_chains}, mob={available_mob_chains}"
        )
        return available_ref_chains, available_mob_chains

    # No valid chains found
    log.error(f"No valid chains found for {pid}")
    return [], []


def load_template_molecules_standardized(
    template_pdb_ids: List[str],
    max_templates: int = 100,
    exclude_target_smiles: Optional[str] = None,
) -> Tuple[List[Chem.Mol], Dict[str, Any]]:
    """Load template molecules from SDF file using standardized approach.

    Args:
        template_pdb_ids: List of PDB IDs to load as templates
        max_templates: Maximum number of templates to load
        exclude_target_smiles: SMILES string to exclude from templates (optional)

    Returns:
        Tuple of (templates, loading_stats)
    """
    templates = []
    loading_stats = {
        "requested": len(template_pdb_ids),
        "loaded": 0,
        "missing_pdbs": [],
        "excluded_by_smiles": 0,
    }

    try:
        # Import here to avoid circular imports
        from .data import load_molecules_with_shared_cache

        # Load all available molecules
        all_molecules = load_molecules_with_shared_cache()

        if not all_molecules:
            loading_stats["error"] = "No molecules loaded from database"
            return [], loading_stats

        # Create lookup dictionary for efficient searching
        mol_dict = {}
        for mol in all_molecules:
            if mol.HasProp("template_pid"):
                pid = mol.GetProp("template_pid").upper()
                mol_dict[pid] = mol

        # Load requested templates
        for pid in template_pdb_ids[:max_templates]:
            pid_upper = pid.upper()

            if pid_upper in mol_dict:
                mol = mol_dict[pid_upper]

                # Check if molecule should be excluded by SMILES
                if exclude_target_smiles:
                    try:
                        mol_smiles = Chem.MolToSmiles(mol)
                        if mol_smiles == exclude_target_smiles:
                            loading_stats["excluded_by_smiles"] += 1
                            continue
                    except Exception:
                        pass  # Continue if SMILES comparison fails

                templates.append(mol)
                loading_stats["loaded"] += 1
            else:
                loading_stats["missing_pdbs"].append(pid)

        log.info(
            f"Loaded {loading_stats['loaded']} templates from {len(template_pdb_ids)} requested"
        )

    except Exception as e:
        loading_stats["error"] = f"Failed to load templates: {e}"
        log.error(f"Template loading error: {e}")

    return templates, loading_stats
