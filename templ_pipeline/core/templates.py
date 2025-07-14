#!/usr/bin/env python3
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
    )
    import biotite.structure.io as bsio
    BIOTITE_AVAILABLE = True
except ImportError:
    BIOTITE_AVAILABLE = False

log = logging.getLogger(__name__)

# Constants
MIN_ANCHOR_RESIDUES = 15
MIN_PROTEIN_LENGTH = 20
MIN_CA_ATOMS_FOR_ALIGNMENT = 3
BLOSUM_GAP_PENALTY = -10
CA_RMSD_THRESHOLD = 10.0
CA_RMSD_FALLBACK_THRESHOLDS = [10.0, 15.0, 20.0]

#  Template Loading Functions 

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


def load_target_data(target_pdb: str, target_smiles: Optional[str] = None) -> Tuple[Optional[str], Optional[Chem.Mol]]:
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


#  Template Transformation Functions 

def transform_ligand(mob_pdb: str, lig: Chem.Mol, pid: str, ref_struct: AtomArray, 
                    ref_chains: Optional[List[str]] = None,
                    mob_chains: Optional[List[str]] = None,
                    similarity_score: float = 0.0) -> Optional[Chem.Mol]:
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

        # Use specified chains or fallback to all available chains
        ref_chain_ids = ref_chains if ref_chains else get_chains(ref_prot)
        mob_chain_ids = mob_chains if mob_chains else get_chains(mob_prot)

        ref_ca = ref_prot[np.isin(ref_prot.chain_id, ref_chain_ids) & (ref_prot.atom_name == "CA")]
        mob_ca = mob_prot[np.isin(mob_prot.chain_id, mob_chain_ids) & (mob_prot.atom_name == "CA")]

        if len(ref_ca) < MIN_CA_ATOMS_FOR_ALIGNMENT or len(mob_ca) < MIN_CA_ATOMS_FOR_ALIGNMENT:
            log.warning(f"Not enough C-alpha atoms for alignment in {pid}: ref={len(ref_ca)}, mob={len(mob_ca)}")
            return None

        # Perform structural alignment with sequence matching
        try:
            # First try sequence-aware alignment (homologs) which is more robust
            fitted, transformation, fixed_idx, mob_idx = superimpose_homologs(
                ref_ca, mob_ca,
                substitution_matrix="BLOSUM62",
                gap_penalty=-10,
                min_anchors=3,  # Minimum required for valid alignment
                terminal_penalty=True
            )
            
            # Calculate RMSD using matched CA pairs
            if len(fixed_idx) >= 3 and len(mob_idx) >= 3:
                ref_subset = ref_ca[fixed_idx]
                mob_subset = mob_ca[mob_idx]
                fitted_mob_ca, _ = superimpose(ref_subset, mob_subset)
                ca_rmsd = struc.rmsd(ref_subset, fitted_mob_ca)
                log.info(f"Homologous structural alignment for {pid}: CA RMSD = {ca_rmsd:.3f}Å using {len(fixed_idx)} matched residues")
            else:
                log.warning(f"Too few matched residues for homolog alignment in {pid}: {len(fixed_idx)}")
                raise ValueError("Insufficient matched residues")
                
        except Exception as e:
            log.warning(f"Homologous structural alignment failed for {pid}: {e}")
            # Fallback to simple superimposition
            try:
                min_length = min(len(ref_ca), len(mob_ca))
                if min_length < MIN_CA_ATOMS_FOR_ALIGNMENT:
                    log.error(f"Insufficient CA atoms for alignment in {pid}: {min_length}")
                    return None
                    
                fitted_mob_ca, transformation = superimpose(ref_ca[:min_length], mob_ca[:min_length])
                ca_rmsd = struc.rmsd(ref_ca[:min_length], fitted_mob_ca)
                log.info(f"Direct structural alignment for {pid}: CA RMSD = {ca_rmsd:.3f}Å using {min_length} residues")
            except Exception as e2:
                log.error(f"All alignment methods failed for {pid}: {e2}")
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
            conf.SetAtomPosition(i, Point3D(float(transformed_coords[i][0]), 
                                           float(transformed_coords[i][1]), 
                                           float(transformed_coords[i][2])))
        
        # Validate bond lengths to ensure reasonable molecular geometry
        try:
            from rdkit.Chem import Descriptors
            bond_lengths = []
            for bond in transformed_lig.GetBonds():
                atom1_idx = bond.GetBeginAtomIdx()
                atom2_idx = bond.GetEndAtomIdx()
                pos1 = conf.GetAtomPosition(atom1_idx)
                pos2 = conf.GetAtomPosition(atom2_idx)
                distance = ((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)**0.5
                bond_lengths.append(distance)
            
            # Check for unreasonable bond lengths (outside 0.5-3.0 Å range)
            if bond_lengths and (min(bond_lengths) < 0.5 or max(bond_lengths) > 3.0):
                log.warning(f"Suspicious bond lengths after transformation for {pid}: min={min(bond_lengths):.2f}Å, max={max(bond_lengths):.2f}Å")
        except Exception as e:
            log.debug(f"Bond length validation failed for {pid}: {e}")
        
        transformed_lig.SetProp("ca_rmsd", f"{ca_rmsd:.3f}")
        transformed_lig.SetProp("template_pid", pid)
        transformed_lig.SetProp("similarity_score", f"{similarity_score:.3f}")
        transformed_lig.SetProp("ref_chains", ",".join(ref_chain_ids))
        transformed_lig.SetProp("mob_chains", ",".join(mob_chain_ids))
        
        return transformed_lig
        
    except Exception as e:
        log.error(f"Template transformation failed for {pid}: {e}", exc_info=True)
        return None


#  Template Filtering Functions 

def filter_templates_by_ca_rmsd(all_templates: List[Chem.Mol], ca_rmsd_threshold: float) -> List[Chem.Mol]:
    """Filter templates by CA RMSD threshold.
    
    Args:
        all_templates: List of template molecules with CA RMSD properties
        ca_rmsd_threshold: Maximum CA RMSD allowed (Angstroms)
        
    Returns:
        List of templates that pass the CA RMSD threshold
    """
    if ca_rmsd_threshold == float('inf'):
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
    all_templates: List[Chem.Mol], 
    fallback_thresholds: List[float] = None
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
                    f"Found {len(filtered_templates)} templates with CA RMSD d {threshold}Å"
                )
            return filtered_templates, threshold, False
    
    #  fallback: find template with smallest CA RMSD and use central atom positioning
    best_template = None
    best_rmsd = float('inf')
    
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
        return [best_template], float('inf'), True
    
    # This should never happen since we have templates
    log.error("No templates available for central atom fallback")
    return [], float('inf'), False


#  Template Validation Functions 

def validate_template_molecule(mol: Chem.Mol, mol_name: str = "unknown") -> Tuple[bool, str]:
    """Validate template molecule for processing.
    
    Args:
        mol: Template molecule to validate
        mol_name: Name for logging purposes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if mol is None:
        return False, f"{mol_name}: Template molecule is None"
        
    try:
        # Check if atoms are present
        if mol.GetNumAtoms() == 0:
            return False, f"{mol_name}: Template has no atoms"
            
        # Check for conformers
        if mol.GetNumConformers() == 0:
            return False, f"{mol_name}: Template has no conformers"
            
        # Check for required properties
        if not mol.HasProp("template_pid"):
            return False, f"{mol_name}: Template missing PDB ID property"
            
        # Try to sanitize
        try:
            test_mol = Chem.Mol(mol)
            Chem.SanitizeMol(test_mol)
        except Exception as e:
            return False, f"{mol_name}: Template fails sanitization: {e}"
            
        return True, f"{mol_name}: Template validation passed"
        
    except Exception as e:
        return False, f"{mol_name}: Template validation error: {e}"


def validate_template_compatibility(query_mol: Chem.Mol, template_mol: Chem.Mol) -> Tuple[bool, str]:
    """Validate compatibility between query and template molecules.
    
    Args:
        query_mol: Query molecule
        template_mol: Template molecule
        
    Returns:
        Tuple of (is_compatible, compatibility_message)
    """
    try:
        # Check both molecules are valid
        if query_mol is None or template_mol is None:
            return False, "One or both molecules are None"
            
        # Check both have atoms
        if query_mol.GetNumAtoms() == 0 or template_mol.GetNumAtoms() == 0:
            return False, "One or both molecules have no atoms"
            
        # Check both have conformers
        if query_mol.GetNumConformers() == 0 or template_mol.GetNumConformers() == 0:
            return False, "One or both molecules have no conformers"
            
        # Check for reasonable size compatibility
        query_atoms = query_mol.GetNumAtoms()
        template_atoms = template_mol.GetNumAtoms()
        
        # Allow significant size differences but flag extreme cases
        size_ratio = max(query_atoms, template_atoms) / min(query_atoms, template_atoms)
        if size_ratio > 10.0:
            return False, f"Extreme size difference: query={query_atoms}, template={template_atoms} atoms"
            
        return True, "Template compatibility check passed"
        
    except Exception as e:
        return False, f"Template compatibility check error: {e}"


#  Template Processing Utilities 

def extract_template_metadata(mol: Chem.Mol) -> Dict[str, Any]:
    """Extract metadata from template molecule properties.
    
    Args:
        mol: Template molecule with properties
        
    Returns:
        Dictionary of template metadata
    """
    metadata = {}
    
    if mol is None:
        return metadata
        
    try:
        # Extract standard properties
        prop_names = [
            "template_pid", "ca_rmsd", "similarity_score", 
            "ref_chains", "mob_chains", "template_source"
        ]
        
        for prop_name in prop_names:
            if mol.HasProp(prop_name):
                metadata[prop_name] = mol.GetProp(prop_name)
                
        # Extract numeric properties with validation
        numeric_props = ["ca_rmsd", "similarity_score"]
        for prop_name in numeric_props:
            if prop_name in metadata:
                try:
                    metadata[prop_name] = float(metadata[prop_name])
                except (ValueError, TypeError):
                    log.warning(f"Invalid numeric value for {prop_name}: {metadata[prop_name]}")
                    metadata[prop_name] = None
                    
        # Extract list properties
        list_props = ["ref_chains", "mob_chains"]
        for prop_name in list_props:
            if prop_name in metadata and metadata[prop_name]:
                try:
                    metadata[prop_name] = metadata[prop_name].split(",")
                except AttributeError:
                    metadata[prop_name] = [metadata[prop_name]]
                    
        return metadata
        
    except Exception as e:
        log.error(f"Failed to extract template metadata: {e}")
        return {}


def enhance_template_with_metadata(mol: Chem.Mol, metadata: Dict[str, Any]) -> Chem.Mol:
    """Enhance template molecule with additional metadata.
    
    Args:
        mol: Template molecule to enhance
        metadata: Additional metadata to add
        
    Returns:
        Enhanced template molecule
    """
    if mol is None:
        return None
        
    try:
        enhanced_mol = Chem.Mol(mol)
        
        for key, value in metadata.items():
            if value is not None:
                if isinstance(value, list):
                    enhanced_mol.SetProp(key, ",".join(map(str, value)))
                else:
                    enhanced_mol.SetProp(key, str(value))
                    
        return enhanced_mol
        
    except Exception as e:
        log.error(f"Failed to enhance template with metadata: {e}")
        return mol


#  Template Selection Utilities 

def rank_templates_by_quality(templates: List[Chem.Mol]) -> List[Tuple[Chem.Mol, float]]:
    """Rank templates by quality score (combination of CA RMSD and similarity).
    
    Args:
        templates: List of template molecules
        
    Returns:
        List of (template, quality_score) tuples sorted by quality (higher is better)
    """
    template_scores = []
    
    for template in templates:
        try:
            # Extract CA RMSD (lower is better)
            ca_rmsd = float('inf')
            if template.HasProp("ca_rmsd"):
                try:
                    ca_rmsd = float(template.GetProp("ca_rmsd"))
                except (ValueError, TypeError):
                    pass
                    
            # Extract similarity score (higher is better)
            similarity = 0.0
            if template.HasProp("similarity_score"):
                try:
                    similarity = float(template.GetProp("similarity_score"))
                except (ValueError, TypeError):
                    pass
                    
            # Calculate quality score (normalize and combine)
            # Lower CA RMSD is better, so invert it
            rmsd_score = 1.0 / (1.0 + ca_rmsd) if ca_rmsd != float('inf') else 0.0
            
            # Combine scores (weighted average)
            quality_score = 0.6 * similarity + 0.4 * rmsd_score
            
            template_scores.append((template, quality_score))
            
        except Exception as e:
            log.warning(f"Failed to calculate quality score for template: {e}")
            template_scores.append((template, 0.0))
            
    # Sort by quality score (descending)
    template_scores.sort(key=lambda x: x[1], reverse=True)
    
    return template_scores


def select_diverse_templates(templates: List[Chem.Mol], max_templates: int = 50) -> List[Chem.Mol]:
    """Select diverse templates to avoid redundancy.
    
    Args:
        templates: List of template molecules
        max_templates: Maximum number of templates to select
        
    Returns:
        List of selected diverse templates
    """
    if len(templates) <= max_templates:
        return templates
        
    # Rank templates by quality first
    ranked_templates = rank_templates_by_quality(templates)
    
    # Select top templates ensuring diversity
    selected = []
    used_pids = set()
    
    for template, quality_score in ranked_templates:
        if len(selected) >= max_templates:
            break
            
        # Check for PDB ID diversity
        pid = template.GetProp("template_pid") if template.HasProp("template_pid") else "unknown"
        
        if pid not in used_pids:
            selected.append(template)
            used_pids.add(pid)
            
    # If we still need more templates, add remaining ones
    if len(selected) < max_templates:
        for template, _ in ranked_templates:
            if len(selected) >= max_templates:
                break
            if template not in selected:
                selected.append(template)
                
    log.info(f"Selected {len(selected)} diverse templates from {len(templates)} total")
    return selected


#  Path Utilities 

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
        f"{data_dir}/PDBBind/PDBbind_v2020_refined/refined-set/{pid_lower}/{pid_lower}_protein.pdb",
        f"{data_dir}/PDBBind/PDBbind_v2020_other_PL/v2020-other-PL/{pid_lower}/{pid_lower}_protein.pdb",
        f"{data_dir}/pdbs/{pid_lower}.pdb",
        f"{data_dir}/proteins/{pid_lower}_protein.pdb",
        f"{pid_lower}.pdb"
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
        f"{pid_lower}_ligand.sdf"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
            
    log.warning(f"Ligand file for {pid} not found in any standard location")
    return None