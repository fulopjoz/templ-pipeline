"""
Template selection and filtering utilities.

This module provides functions for selecting and filtering template molecules
based on structural quality metrics, particularly CA RMSD (alpha-carbon RMSD)
values that indicate protein structural similarity.

This module also provides standardized template molecule loading functionality 
consistent with the true_mcs_pipeline.py approach to ensure identical template 
molecules are loaded across the entire pipeline.
"""

import gzip
import logging
import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Set, Dict
from rdkit import Chem

log = logging.getLogger(__name__)

# Default CA RMSD fallback thresholds
DEFAULT_CA_RMSD_FALLBACK_THRESHOLDS = [2.0, 3.0, 5.0, 10.0, 20.0]

# ═══════════════════════════════════════════════════════════════════════════════
# STANDARDIZED TEMPLATE LOADING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def safe_name(mol, default: str) -> str:
    """Get molecule name with fallback to default value.
    
    This function matches the safe_name implementation from true_mcs_pipeline.py
    to ensure consistent molecule naming across pipelines.
    """
    try:
        if mol.HasProp("_Name"):
            return mol.GetProp("_Name")
        mol.SetProp("_Name", default)
        return default
    except Exception:
        return default

def resolve_ligands_file_path() -> Optional[str]:
    """Resolve the path to processed_ligands_new.sdf.gz with fallback strategy.
    
    Tries multiple possible locations in order of preference:
    1. Environment variable TEMPL_LIGANDS_PATH if set
    2. Relative paths from current working directory
    3. Absolute paths for common deployment scenarios
    
    Returns:
        Path to the ligands file if found, None otherwise
    """
    # Check environment variable first
    env_path = os.environ.get('TEMPL_LIGANDS_PATH')
    if env_path and os.path.exists(env_path):
        log.debug(f"Found ligands file via TEMPL_LIGANDS_PATH: {env_path}")
        return env_path
    
    # Possible paths in order of preference - PRIORITIZE .gz files (fewer format errors)
    possible_paths = [
        # PRIORITY 1: .gz files (relative paths) - fewer SDF format errors
        "data/ligands/processed_ligands_new.sdf.gz",
        "templ_pipeline/data/ligands/processed_ligands_new.sdf.gz",
        
        # PRIORITY 2: .gz files (absolute paths) - fewer SDF format errors
        "/home/ubuntu/mcs/templ_pipeline/data/ligands/processed_ligands_new.sdf.gz",
        "/home/ubuntu/mcs/mcs_bench/data/processed_ligands_new.sdf.gz",
        "data-minimal/ligands/processed_ligands_new.sdf.gz",
        
        # FALLBACK: Unzipped versions (more format errors, use only if .gz unavailable)
        "data/ligands/processed_ligands_new_unzipped.sdf",
        "templ_pipeline/data/ligands/processed_ligands_new_unzipped.sdf",
        "/home/ubuntu/mcs/templ_pipeline/data/ligands/processed_ligands_new_unzipped.sdf",
        "/home/ubuntu/mcs/mcs_bench/data/processed_ligands_new_unzipped.sdf",
        "data-minimal/ligands/processed_ligands_new_unzipped.sdf"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            log.debug(f"Found ligands file at: {path}")
            return path
    
    log.error("Processed ligands file not found at any of the expected locations")
    log.error("Expected locations searched:")
    for path in possible_paths:
        log.error(f"  - {path}")
    
    return None

def ensure_molecule_ringinfo(mol: Chem.Mol) -> Chem.Mol:
    """Ensure molecule has properly initialized RingInfo to prevent MCS errors.
    
    Args:
        mol: Input molecule
        
    Returns:
        Molecule with properly initialized RingInfo
    """
    if mol is None:
        return None
    
    try:
        # Create a copy to avoid modifying original
        mol_copy = Chem.Mol(mol)
        
        # Force RingInfo initialization by accessing ring information
        mol_copy.GetRingInfo().NumRings()
        
        # Ensure proper sanitization which initializes RingInfo
        Chem.SanitizeMol(mol_copy)
        Chem.SetAromaticity(mol_copy)
        
        # Double check RingInfo is accessible
        ring_info = mol_copy.GetRingInfo()
        if ring_info is not None:
            # Access ring data to ensure it's fully initialized
            ring_info.NumRings()
            return mol_copy
        else:
            log.warning("RingInfo still not accessible after initialization attempt")
            return mol
            
    except Exception as e:
        log.warning(f"Failed to initialize RingInfo: {e}, using original molecule")
        return mol

def validate_template_molecule(mol: Chem.Mol, pdb_id: str) -> Tuple[bool, str]:
    """Enhanced validation including RingInfo check.
    
    Args:
        mol: Molecule to validate
        pdb_id: PDB ID for logging
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if mol is None:
        return False, "Molecule is None"
    
    try:
        # Basic checks
        if mol.GetNumAtoms() == 0:
            return False, "No atoms"
        
        if mol.GetNumAtoms() > 500:
            return False, f"Too many atoms: {mol.GetNumAtoms()}"
        
        # Check RingInfo accessibility - critical for MCS operations
        try:
            ring_info = mol.GetRingInfo()
            if ring_info is None:
                return False, "RingInfo not accessible"
            
            # Try to access ring data to ensure it's properly initialized
            num_rings = ring_info.NumRings()
            log.debug(f"Template {pdb_id}: {num_rings} rings detected")
            
        except Exception as e:
            return False, f"RingInfo error: {str(e)}"
        
        # Test MCS readiness by attempting basic ring operations
        try:
            from rdkit.Chem import rdMolDescriptors
            rdMolDescriptors.CalcNumRings(mol)
        except Exception as e:
            return False, f"Ring descriptor calculation failed: {str(e)}"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def load_template_molecules_standardized(
    template_pdb_ids: List[str],
    max_templates: Optional[int] = None,
    exclude_target_smiles: Optional[str] = None,
    allowed_pdb_ids: Optional[Set[str]] = None
) -> Tuple[List, Dict[str, str]]:
    """Load template molecules using the exact same approach as true_mcs_pipeline.py.
    
    This function ensures identical template loading behavior across all pipeline
    components by using the same RDKit parameters, validation, and filtering.
    Enhanced with error tolerance for SDF format issues.
    
    Args:
        template_pdb_ids: List of PDB IDs to load
        max_templates: Maximum number of templates to load (None for no limit)
        exclude_target_smiles: SMILES string to exclude (target molecule)
        allowed_pdb_ids: Set of allowed PDB IDs for filtering
        
    Returns:
        Tuple of (template_molecules_list, loading_stats_dict)
    """
    # CRITICAL: Suppress RDKit warnings for SDF format issues
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    # Resolve the ligands file path
    ligands_file_path = resolve_ligands_file_path()
    if not ligands_file_path:
        return [], {"error": "Ligands file not found"}
    
    log.info(f"Loading templates from: {ligands_file_path} (with error tolerance)")
    
    # Convert inputs to consistent format
    template_pdb_ids_lower = [pdb_id.lower() for pdb_id in template_pdb_ids]
    templates = []
    loaded_pdb_ids = set()
    skipped_reasons = {}
    
    # Prepare target canonical SMILES for exclusion if provided
    target_canonical = None
    if exclude_target_smiles:
        try:
            target_mol = Chem.MolFromSmiles(exclude_target_smiles)
            if target_mol:
                target_canonical = Chem.MolToSmiles(target_mol)
        except Exception as e:
            log.warning(f"Could not process target SMILES for exclusion: {e}")
    
    try:
        # Open file - handle both .gz and uncompressed
        if ligands_file_path.endswith('.gz'):
            file_handle = gzip.open(ligands_file_path, 'rb')
        else:
            file_handle = open(ligands_file_path, 'rb')
        
        # Use enhanced parameters with error tolerance for corrupted SDF entries
        supplier = Chem.ForwardSDMolSupplier(
            file_handle, 
            removeHs=False, 
            sanitize=False,
            strictParsing=False  # CRITICAL: Allow malformed entries to be skipped
        )
        
        molecules_processed = 0
        molecules_loaded = 0
        format_errors = 0
        
        for mol in supplier:
            molecules_processed += 1
            
            # Enhanced error handling for corrupted SDF entries
            if not mol:
                format_errors += 1
                continue
            
            # Critical: Check for 3D conformers (same as true_mcs_pipeline.py)
            if not mol.GetNumConformers():
                continue
            
            # Extract PDB ID using same safe_name approach
            mol_name = safe_name(mol, "lig_placeholder")
            pdb_id = mol_name[:4].lower()
            
            # Filter by requested PDB IDs
            if template_pdb_ids and pdb_id not in template_pdb_ids_lower:
                continue
            
            # Apply allowed_pdb_ids filtering if provided
            if allowed_pdb_ids is not None and pdb_id not in allowed_pdb_ids:
                continue
            
            # Avoid duplicate PDB IDs
            if pdb_id in loaded_pdb_ids:
                continue
            
            # Exclude target molecule if specified (same logic as true_mcs_pipeline.py)
            if target_canonical:
                try:
                    mol_smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))
                    if mol_smiles == target_canonical:
                        skipped_reasons[pdb_id] = "Target molecule self-exclusion"
                        continue
                except Exception:
                    # If SMILES generation fails, still include the molecule
                    pass
            
            # Validate template quality
            is_valid, reason = validate_template_molecule(mol, pdb_id)
            if not is_valid:
                skipped_reasons[pdb_id] = f"Quality validation failed: {reason}"
                continue
            
            # Set consistent naming and standardize aromaticity
            mol.SetProp("_Name", pdb_id)
            
            # CRITICAL: Ensure RingInfo is properly initialized before any downstream operations
            mol = ensure_molecule_ringinfo(mol)
            
            # Apply SMILES standardization to ensure consistent representation
            try:
                from .molecule_standardization import standardize_molecule_smiles
                standardized_mol = standardize_molecule_smiles(mol, method="canonical")
                
                # Ensure RingInfo is preserved after standardization
                standardized_mol = ensure_molecule_ringinfo(standardized_mol)
                templates.append(standardized_mol)
                
            except Exception as e:
                log.warning(f"SMILES standardization failed for {pdb_id}: {e}, using ring-validated original")
                templates.append(mol)
            
            loaded_pdb_ids.add(pdb_id)
            molecules_loaded += 1
            
            # Check max templates limit
            if max_templates and len(templates) >= max_templates:
                log.info(f"Reached maximum template limit: {max_templates}")
                break
        
        file_handle.close()
        
        # Compile enhanced loading statistics with error tolerance info
        missing_pdb_ids = set(template_pdb_ids_lower) - loaded_pdb_ids
        loading_stats = {
            "molecules_processed": molecules_processed,
            "molecules_loaded": molecules_loaded,
            "format_errors": format_errors,
            "error_rate": format_errors / molecules_processed if molecules_processed > 0 else 0,
            "requested_pdbs": len(template_pdb_ids_lower),
            "loaded_pdbs": len(loaded_pdb_ids),
            "missing_pdbs": list(missing_pdb_ids),
            "skipped_reasons": dict(skipped_reasons),
            "source_file": ligands_file_path,
            "error_tolerance_enabled": True
        }
        
        log.info(f"Template loading complete: {molecules_loaded} templates loaded from {molecules_processed} processed")
        
        # Report format errors with tolerance
        if format_errors > 0:
            error_rate = format_errors / molecules_processed * 100
            log.info(f"Format errors handled gracefully: {format_errors}/{molecules_processed} ({error_rate:.1f}%) - pipeline continues normally")
        
        if missing_pdb_ids:
            log.warning(f"Could not find templates for: {', '.join(sorted(missing_pdb_ids))}")
        
        if skipped_reasons:
            log.debug(f"Skipped templates: {dict(list(skipped_reasons.items())[:5])}...")  # Show first 5
        
        return templates, loading_stats
        
    except Exception as e:
        log.error(f"Error loading templates: {str(e)}")
        return [], {"error": f"Loading failed: {str(e)}"}

def load_all_template_molecules(
    max_templates: Optional[int] = None,
    exclude_target_smiles: Optional[str] = None
) -> Tuple[Dict[str, object], Dict[str, str]]:
    """Load all available template molecules from the processed ligands file.
    
    This function loads all templates similar to how true_mcs_pipeline.py loads
    the complete ligands dictionary for neighbor searching.
    Enhanced with error tolerance for SDF format issues.
    
    Args:
        max_templates: Maximum number of templates to load
        exclude_target_smiles: SMILES to exclude (target molecule)
        
    Returns:
        Tuple of (pdb_id_to_molecule_dict, loading_stats_dict)
    """
    # Suppress RDKit warnings for SDF format issues
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    ligands_file_path = resolve_ligands_file_path()
    if not ligands_file_path:
        return {}, {"error": "Ligands file not found"}
    
    log.info(f"Loading all templates from: {ligands_file_path} (with error tolerance)")
    
    templates_dict = {}
    
    # Prepare target canonical SMILES for exclusion
    target_canonical = None
    if exclude_target_smiles:
        try:
            target_mol = Chem.MolFromSmiles(exclude_target_smiles)
            if target_mol:
                target_canonical = Chem.MolToSmiles(target_mol)
        except Exception as e:
            log.warning(f"Could not process target SMILES for exclusion: {e}")
    
    try:
        if ligands_file_path.endswith('.gz'):
            file_handle = gzip.open(ligands_file_path, 'rb')
        else:
            file_handle = open(ligands_file_path, 'rb')
        
        supplier = Chem.ForwardSDMolSupplier(
            file_handle, 
            removeHs=False, 
            sanitize=False,
            strictParsing=False  # Allow malformed entries to be skipped
        )
        
        molecules_processed = 0
        molecules_loaded = 0
        format_errors = 0
        
        for mol in supplier:
            molecules_processed += 1
            
            if not mol:
                format_errors += 1
                continue
                
            if not mol.GetNumConformers():
                continue
            
            pdb_id = safe_name(mol, "lig_placeholder")[:4].lower()
            
            # Exclude target molecule
            if target_canonical:
                try:
                    mol_smiles = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))
                    if mol_smiles == target_canonical:
                        continue
                except Exception:
                    pass
            
            # Basic validation
            is_valid, _ = validate_template_molecule(mol, pdb_id)
            if not is_valid:
                continue
            
            mol.SetProp("_Name", pdb_id)
            templates_dict[pdb_id] = mol
            molecules_loaded += 1
            
            if max_templates and len(templates_dict) >= max_templates:
                break
        
        file_handle.close()
        
        loading_stats = {
            "molecules_processed": molecules_processed,
            "molecules_loaded": molecules_loaded,
            "format_errors": format_errors,
            "error_rate": format_errors / molecules_processed if molecules_processed > 0 else 0,
            "source_file": ligands_file_path,
            "error_tolerance_enabled": True
        }
        
        log.info(f"All templates loaded: {molecules_loaded} from {molecules_processed} processed")
        if format_errors > 0:
            error_rate = format_errors / molecules_processed * 100
            log.info(f"Format errors handled gracefully: {format_errors}/{molecules_processed} ({error_rate:.1f}%)")
        return templates_dict, loading_stats
        
    except Exception as e:
        log.error(f"Error loading all templates: {str(e)}")
        return {}, {"error": f"Loading failed: {str(e)}"}

# ═══════════════════════════════════════════════════════════════════════════════
# EXISTING TEMPLATE FILTERING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

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
    fallback_thresholds: Optional[List[float]] = None,
    primary_threshold: float = 2.0
) -> Tuple[List[Chem.Mol], float, bool]:
    """Apply progressive CA RMSD fallback with central atom final fallback.
    
    Args:
        all_templates: List of all available template molecules
        fallback_thresholds: List of CA RMSD thresholds to try (default: [2.0, 3.0, 5.0])
        primary_threshold: Primary threshold to determine if fallback was used
        
    Returns:
        Tuple of (valid_templates, threshold_used, use_central_atom_fallback)
    """
    if fallback_thresholds is None:
        fallback_thresholds = DEFAULT_CA_RMSD_FALLBACK_THRESHOLDS
    
    for threshold in fallback_thresholds:
        filtered_templates = filter_templates_by_ca_rmsd(all_templates, threshold)
        if filtered_templates:
            if threshold > primary_threshold:
                log.warning(f"Using relaxed CA RMSD threshold ({threshold}Å) - found {len(filtered_templates)} templates (poses may be less accurate)")
            else:
                log.info(f"Found {len(filtered_templates)} templates with CA RMSD ≤ {threshold}Å")
            return filtered_templates, threshold, False
    
    # Ultimate fallback: find template with smallest CA RMSD and use central atom positioning
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
        log.warning(f"Using central atom fallback with best available template (CA RMSD: {best_rmsd}Å)")
        return [best_template], float('inf'), True
    
    # This should never happen since we have templates
    log.error("No templates available for central atom fallback")
    return [], float('inf'), False


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


def load_uniprot_exclude(exclude_file: str) -> Set[str]:
    """Load UniProt IDs to exclude from a file.
    
    Args:
        exclude_file: Path to file containing UniProt IDs to exclude
        
    Returns:
        Set of UniProt IDs to exclude
    """
    exclude_uniprot = set()
    if os.path.exists(exclude_file):
        try:
            with open(exclude_file) as f:
                exclude_uniprot = {line.strip() for line in f if line.strip()}
            log.info(f"Loaded {len(exclude_uniprot)} UniProt IDs to exclude from {exclude_file}")
        except Exception as e:
            log.error(f"Error loading UniProt exclude file {exclude_file}: {e}")
    else:
        log.warning(f"UniProt exclude file not found: {exclude_file}")
    
    return exclude_uniprot


def get_uniprot_mapping(uniprot_map_file: str) -> Dict[str, str]:
    """Load PDB to UniProt ID mapping from a JSON file.
    
    Args:
        uniprot_map_file: Path to JSON file containing PDB to UniProt mapping
        
    Returns:
        Dict mapping PDB IDs to UniProt IDs
    """
    pdb_to_uniprot = {}
    if os.path.exists(uniprot_map_file):
        try:
            with open(uniprot_map_file) as f:
                data = json.load(f)
                for pdb_id, info in data.items():
                    if isinstance(info, dict) and "uniprot" in info:
                        pdb_to_uniprot[pdb_id] = info["uniprot"]
            log.info(f"Loaded PDB to UniProt mapping for {len(pdb_to_uniprot)} entries from {uniprot_map_file}")
        except Exception as e:
            log.error(f"Error loading UniProt mapping file {uniprot_map_file}: {e}")
    else:
        log.warning(f"UniProt mapping file not found: {uniprot_map_file}")
    
    return pdb_to_uniprot


def load_pdb_filter(filter_file: str) -> Set[str]:
    """Load PDB IDs from a filter file.
    
    Args:
        filter_file: Path to file containing PDB IDs (one per line)
        
    Returns:
        Set of PDB IDs
    """
    pdb_ids = set()
    
    if not os.path.exists(filter_file):
        log.warning(f"PDB filter file not found: {filter_file}")
        return pdb_ids
    
    try:
        with open(filter_file, 'r') as f:
            for line in f:
                pdb_id = line.strip().lower()
                if pdb_id and not pdb_id.startswith('#'):
                    pdb_ids.add(pdb_id)
        
        log.info(f"Loaded {len(pdb_ids)} PDB IDs from {filter_file}")
    
    except Exception as e:
        log.error(f"Error loading PDB filter file {filter_file}: {e}")
    
    return pdb_ids


def standardize_atom_arrays(arrays):
    """Standardize annotations across multiple atom arrays to make them compatible for stacking.
    
    This function resolves a critical issue in the protein alignment pipeline where
    atom arrays from different chains often have incompatible annotation dictionaries,
    causing errors when trying to stack them together for multi-chain analysis.
    
    Technical details:
    1. Biotite's AtomArray objects contain 'annotations' dictionaries with metadata
    2. When calling struc.stack(), all arrays must have identical annotation keys
    3. Different protein chains often have different annotations (chain-specific data)
    4. This function finds the intersection of annotations across all arrays
    5. It then creates new arrays with only the common annotations
    
    Used in transform_ligand() when:
    - Processing proteins with multiple chains from embedding data
    - Attempting to combine CA atoms from different chains into one array
    - Regular stacking fails due to annotation mismatches
    
    Without this function, multi-chain template alignment would fail for many proteins,
    forcing fallback to single-chain alignment and potentially reducing binding pocket
    alignment quality.
    
    Args:
        arrays: List of AtomArray objects to standardize
        
    Returns:
        List of AtomArray objects with compatible annotations, or just the first array
        if standardization is not possible
    """
    if not arrays or len(arrays) <= 1:
        return arrays
        
    # Check if all arrays have annotations
    if not all(hasattr(arr, 'annotations') for arr in arrays):
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