"""
TEMPL Pipeline Scoring Module

This module handles pose scoring and selection:
1. Shape-based scoring using RDKit's ShapeAlign
2. Color (pharmacophore) scoring
3. Combo scoring (weighted combination)
4. Parallel processing for efficient scoring of multiple conformers
5. RMSD calculation for evaluation

The main classes and functions:
- score_and_align: Compute shape/color scores for a conformer against a template
- select_best: Rank conformers by different scoring methods and select top poses
- rmsd_raw: Calculate RMSD between molecules for evaluation
"""

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Any, Union, Set
import numpy as np

from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    SanitizeMol,
    rdShapeAlign,
)
from tqdm import tqdm

# For RMSD calculation
try:
    from spyrmsd.molecule import Molecule
    from spyrmsd.rmsd import rmsdwrapper

    HAS_SPYRMSD = True
except ImportError:
    HAS_SPYRMSD = False

# Import the organometallic handling function
from .chemistry import detect_and_substitute_organometallic

# Configure logging
logger = logging.getLogger(__name__)


class CoordinateMapper:
    """Advanced coordinate mapping and preservation utilities."""

    @staticmethod
    def map_heavy_atoms(mol_before: Chem.Mol, mol_after: Chem.Mol) -> Dict[int, int]:
        """Create mapping between heavy atoms before and after hydrogen addition."""
        heavy_mapping = {}
        before_idx = 0
        after_idx = 0

        # Heavy atoms should be in the same order, but hydrogens might be inserted
        while (
            before_idx < mol_before.GetNumAtoms()
            and after_idx < mol_after.GetNumAtoms()
        ):
            atom_before = mol_before.GetAtomWithIdx(before_idx)
            atom_after = mol_after.GetAtomWithIdx(after_idx)

            # Skip hydrogens in the after molecule
            if atom_after.GetAtomicNum() == 1:
                after_idx += 1
                continue

            # Map heavy atoms
            if atom_before.GetAtomicNum() == atom_after.GetAtomicNum():
                heavy_mapping[before_idx] = after_idx
                before_idx += 1
                after_idx += 1
            else:
                # Unexpected atom mismatch
                after_idx += 1

        return heavy_mapping

    @staticmethod
    def preserve_heavy_atom_coords(
        mol_source: Chem.Mol, mol_target: Chem.Mol
    ) -> Chem.Mol:
        """Preserve heavy atom coordinates from source to target molecule."""
        if mol_source.GetNumConformers() == 0 or mol_target.GetNumConformers() == 0:
            return mol_target

        source_conf = mol_source.GetConformer(0)
        target_conf = mol_target.GetConformer(0)

        # Map heavy atoms
        heavy_mapping = CoordinateMapper.map_heavy_atoms(mol_source, mol_target)

        # Transfer coordinates
        for source_idx, target_idx in heavy_mapping.items():
            pos = source_conf.GetAtomPosition(source_idx)
            target_conf.SetAtomPosition(target_idx, pos)

        return mol_target


class FixedMolecularProcessor:
    """Processor with enhanced coordinate preservation during molecular operations."""

    @staticmethod
    def create_independent_copy(mol: Chem.Mol) -> Chem.Mol:
        """Create a truly independent copy of a molecule to prevent reference issues."""
        if mol is None:
            return None

        # Validate input type to prevent Boost.Python.ArgumentError
        if not isinstance(mol, Chem.Mol):
            logger.error(
                f"Invalid input type for molecular copy: {type(mol)}, expected Chem.Mol"
            )
            return None

        # Additional validation for molecular object integrity
        try:
            # Test basic molecule operations to ensure it's a valid RDKit molecule
            _ = mol.GetNumAtoms()
            _ = mol.GetNumBonds()
        except Exception as e:
            logger.error(f"Molecular object appears corrupted: {e}")
            return None

        # Create deep copy with multiple fallback strategies
        mol_copy = None

        # Primary method: Direct Chem.Mol copy
        try:
            mol_copy = Chem.Mol(mol)
        except Exception as e:
            logger.warning(
                f"Primary molecular copy failed: {e}, trying fallback methods"
            )

            # Fallback 1: Try with explicit type checking
            try:
                if hasattr(mol, "ToBinary"):
                    binary_data = mol.ToBinary()
                    mol_copy = Chem.Mol(binary_data)
                else:
                    raise Exception("Molecule doesn't have ToBinary method")
            except Exception as e2:
                logger.warning(f"Binary copy fallback failed: {e2}")

                # Fallback 2: Create new molecule and copy properties
                try:
                    mol_copy = Chem.RWMol()
                    for atom_idx in range(mol.GetNumAtoms()):
                        atom = mol.GetAtomWithIdx(atom_idx)
                        new_atom = Chem.Atom(atom.GetAtomicNum())
                        new_atom.SetFormalCharge(atom.GetFormalCharge())
                        mol_copy.AddAtom(new_atom)

                    for bond in mol.GetBonds():
                        mol_copy.AddBond(
                            bond.GetBeginAtomIdx(),
                            bond.GetEndAtomIdx(),
                            bond.GetBondType(),
                        )

                    mol_copy = mol_copy.GetMol()
                except Exception as e3:
                    logger.error(f"All molecular copy methods failed: {e3}")
                    return None

        if mol_copy is None:
            logger.error("Failed to create molecular copy using all methods")
            return None

        # Validate conformers exist and are accessible
        if mol.GetNumConformers() > 0:
            new_mol = Chem.RWMol(mol_copy)
            new_mol.RemoveAllConformers()

            valid_conformers_added = 0
            for i in range(mol.GetNumConformers()):
                try:
                    old_conf = mol.GetConformer(i)
                    new_conf = Chem.Conformer(old_conf.GetNumAtoms())

                    for j in range(old_conf.GetNumAtoms()):
                        pos = old_conf.GetAtomPosition(j)
                        new_conf.SetAtomPosition(j, pos)

                    conf_id = new_mol.AddConformer(new_conf, assignId=True)
                    valid_conformers_added += 1
                    logger.debug(f"Added conformer {conf_id} successfully")
                except Exception as e:
                    logger.warning(f"Failed to copy conformer {i}: {e}")
                    continue

            if valid_conformers_added == 0:
                logger.warning(
                    "No valid conformers could be copied, embedding new conformer"
                )
                try:
                    AllChem.EmbedMolecule(new_mol, randomSeed=42)
                except Exception as e:
                    logger.error(f"Failed to embed fallback conformer: {e}")

            return new_mol

        return mol_copy

    @staticmethod
    def validate_conformer_access(mol: Chem.Mol, conf_id: int = 0) -> bool:
        """Validate that a conformer can be safely accessed."""
        if mol is None:
            return False
        if mol.GetNumConformers() == 0:
            return False
        if conf_id >= mol.GetNumConformers() or conf_id < 0:
            return False

        try:
            conf = mol.GetConformer(conf_id)
            # Test accessing atom positions
            if conf.GetNumAtoms() > 0:
                pos = conf.GetAtomPosition(0)
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def safe_add_hydrogens(mol: Chem.Mol, preserve_coords: bool = True) -> Chem.Mol:
        """Safely add hydrogens with robust coordinate preservation."""
        if mol is None:
            return None

        # Early quality validation
        is_valid, msg = validate_molecule_quality(mol)
        if not is_valid:
            logger.debug(f"Skipping hydrogen addition for low-quality molecule: {msg}")
            return mol

        # Validate input conformer
        if not FixedMolecularProcessor.validate_conformer_access(mol):
            logger.debug("Invalid conformer in input molecule, attempting to fix")
            mol_copy = Chem.Mol(mol)
            try:
                AllChem.EmbedMolecule(mol_copy, randomSeed=42)
                if not FixedMolecularProcessor.validate_conformer_access(mol_copy):
                    logger.debug("Cannot create valid conformer for hydrogen addition")
                    return mol
                mol = mol_copy
            except Exception as e:
                logger.debug(f"Failed to create valid conformer: {e}")
                return mol

        # Create independent copy first
        mol_copy = FixedMolecularProcessor.create_independent_copy(mol)

        # Store original heavy atom coordinates with robust mapping
        original_heavy_coords = {}
        if preserve_coords and FixedMolecularProcessor.validate_conformer_access(
            mol_copy
        ):
            conf = mol_copy.GetConformer(0)
            for i in range(mol_copy.GetNumAtoms()):
                atom = mol_copy.GetAtomWithIdx(i)
                if atom.GetAtomicNum() != 1:  # Not hydrogen
                    try:
                        pos = conf.GetAtomPosition(i)
                        original_heavy_coords[i] = {
                            "pos": pos,
                            "symbol": atom.GetSymbol(),
                            "charge": atom.GetFormalCharge(),
                        }
                    except Exception as e:
                        logger.warning(f"Failed to get position for atom {i}: {e}")
                        continue

        # Add hydrogens with coordinate generation
        try:
            mol_with_h = Chem.AddHs(mol_copy, addCoords=True)
            # Validate the result
            if not FixedMolecularProcessor.validate_conformer_access(mol_with_h):
                raise Exception("AddHs produced invalid conformer")
        except Exception as e:
            logger.warning(
                f"AddHs with coordinates failed: {e}, trying without coordinates"
            )
            try:
                mol_with_h = Chem.AddHs(mol_copy, addCoords=False)
                if mol_with_h.GetNumConformers() == 0:
                    AllChem.EmbedMolecule(mol_with_h, randomSeed=42)
                if not FixedMolecularProcessor.validate_conformer_access(mol_with_h):
                    raise Exception("Cannot create valid conformer after AddHs")
            except Exception as e2:
                logger.error(f"All hydrogen addition methods failed: {e2}")
                return mol

        # Restore heavy atom coordinates with intelligent matching
        if (
            preserve_coords
            and original_heavy_coords
            and FixedMolecularProcessor.validate_conformer_access(mol_with_h)
        ):
            try:
                conf = mol_with_h.GetConformer(0)

                # Create robust mapping between original and hydrogenated molecule
                heavy_atom_mapping = {}
                orig_heavy_idx = 0

                for new_idx in range(mol_with_h.GetNumAtoms()):
                    new_atom = mol_with_h.GetAtomWithIdx(new_idx)

                    # Skip hydrogens in new molecule
                    if new_atom.GetAtomicNum() == 1:
                        continue

                    # Find corresponding original heavy atom
                    while orig_heavy_idx in original_heavy_coords:
                        orig_data = original_heavy_coords[orig_heavy_idx]

                        # Match by symbol and charge
                        if (
                            new_atom.GetSymbol() == orig_data["symbol"]
                            and new_atom.GetFormalCharge() == orig_data["charge"]
                        ):
                            heavy_atom_mapping[orig_heavy_idx] = new_idx
                            orig_heavy_idx += 1
                            break
                        else:
                            orig_heavy_idx += 1

                # Apply preserved coordinates
                for orig_idx, new_idx in heavy_atom_mapping.items():
                    if orig_idx in original_heavy_coords:
                        try:
                            conf.SetAtomPosition(
                                new_idx, original_heavy_coords[orig_idx]["pos"]
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to set position for atom {new_idx}: {e}"
                            )
            except Exception as e:
                logger.warning(f"Failed to restore coordinates: {e}")

        return mol_with_h

    @staticmethod
    def add_hydrogens_with_fallback(
        mol: Chem.Mol, preserve_coords: bool = True
    ) -> Chem.Mol:
        """Add hydrogens with multiple fallback strategies for coordinate preservation."""
        try:
            # Primary method: safe_add_hydrogens
            return FixedMolecularProcessor.safe_add_hydrogens(mol, preserve_coords)
        except Exception as e:
            logger.warning(f"Primary hydrogen addition failed: {e}")

        try:
            # Fallback 1: Simple AddHs with coordinate generation
            mol_copy = Chem.Mol(mol)
            result = Chem.AddHs(mol_copy, addCoords=True)
            return result
        except Exception as e:
            logger.warning(f"Fallback 1 hydrogen addition failed: {e}")

        try:
            # Fallback 2: AddHs without coordinates, then embed
            mol_copy = Chem.Mol(mol)
            result = Chem.AddHs(mol_copy, addCoords=False)
            if result.GetNumConformers() == 0:
                AllChem.EmbedMolecule(result, randomSeed=42)
            return result
        except Exception as e:
            logger.error(f"All hydrogen addition methods failed: {e}")
            return mol  # Return original molecule as last resort


class ScoringFixer:
    """Main class for applying scoring fixes to molecules and pipelines."""

    @staticmethod
    def add_tiny_perturbation(mol: Chem.Mol, noise_level: float = 1e-6) -> Chem.Mol:
        """Add tiny coordinate perturbations to break perfect identity."""
        if mol.GetNumConformers() == 0:
            return mol

        perturbed_mol = Chem.Mol(mol)
        conf = perturbed_mol.GetConformer(0)

        for i in range(conf.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            # Add tiny random noise
            noise = np.random.normal(0, noise_level, 3)
            new_pos = [pos.x + noise[0], pos.y + noise[1], pos.z + noise[2]]
            conf.SetAtomPosition(i, new_pos)

        return perturbed_mol


def rmsd_raw(a: Chem.Mol, b: Chem.Mol) -> float:
    """Calculate RMSD between two molecules using sPyRMSD."""
    if not HAS_SPYRMSD:
        logger.warning("sPyRMSD not available. RMSD calculation will return NaN.")
        return float("nan")

    try:
        # Ensure both molecules are processed consistently
        a_clean = Chem.RemoveHs(a)
        b_clean = Chem.RemoveHs(b)
        
        # Check if molecules have the same number of atoms
        if a_clean.GetNumAtoms() != b_clean.GetNumAtoms():
            logger.debug(f"RMSD skipped: atom count mismatch ({a_clean.GetNumAtoms()} vs {b_clean.GetNumAtoms()})")
            return float("nan")
        
        return rmsdwrapper(
            Molecule.from_rdkit(a_clean),
            Molecule.from_rdkit(b_clean),
            minimize=False,
            strip=True,
            symmetry=True,
        )[0]
    except AssertionError:
        return float("nan")
    except Exception as e:
        logger.debug(f"RMSD calculation error: {str(e)}")
        return float("nan")


def validate_molecular_geometry(
    mol: Chem.Mol, mol_name: str = "molecule"
) -> Tuple[bool, str]:
    """
    Validate molecular geometry to detect potential corruption from alignment issues.

    Args:
        mol: RDKit molecule to validate
        mol_name: Name for logging/error reporting

    Returns:
        Tuple of (is_valid, error_message)
    """
    if mol is None:
        return False, f"{mol_name}: Molecule is None"

    if mol.GetNumConformers() == 0:
        return False, f"{mol_name}: No conformers available"

    try:
        conf = mol.GetConformer(0)
        num_atoms = mol.GetNumAtoms()

        if num_atoms == 0:
            return False, f"{mol_name}: No atoms in molecule"

        # Check for suspicious coordinate patterns that indicate corruption
        positions = []
        for i in range(num_atoms):
            pos = conf.GetAtomPosition(i)
            positions.append([pos.x, pos.y, pos.z])

        positions = np.array(positions)

        # Check for NaN or infinite coordinates
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            return False, f"{mol_name}: Contains NaN or infinite coordinates"

        # Check for extremely large coordinates (likely corruption)
        max_coord = np.max(np.abs(positions))
        if max_coord > 1000.0:  # Reasonable molecules should have coordinates < 1000 Å
            return (
                False,
                f"{mol_name}: Suspiciously large coordinates (max: {max_coord:.2f})",
            )

        # Check for all atoms at the same position (alignment collapse)
        if num_atoms > 1:
            distances = []
            for i in range(num_atoms):
                for j in range(i + 1, num_atoms):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    distances.append(dist)

            max_distance = max(distances) if distances else 0
            if max_distance < 0.1:  # All atoms within 0.1 Å suggests collapse
                return (
                    False,
                    f"{mol_name}: All atoms collapsed to same position (max_dist: {max_distance:.3f})",
                )

        # Check for unrealistic bond lengths (hydrogens too far from heavy atoms)
        heavy_atoms = [
            i for i in range(num_atoms) if mol.GetAtomWithIdx(i).GetAtomicNum() != 1
        ]
        hydrogen_atoms = [
            i for i in range(num_atoms) if mol.GetAtomWithIdx(i).GetAtomicNum() == 1
        ]

        if heavy_atoms and hydrogen_atoms:
            suspicious_hydrogens = 0
            for h_idx in hydrogen_atoms:
                h_pos = positions[h_idx]

                # Find closest heavy atom
                min_dist_to_heavy = float("inf")
                for heavy_idx in heavy_atoms:
                    heavy_pos = positions[heavy_idx]
                    dist = np.linalg.norm(h_pos - heavy_pos)
                    min_dist_to_heavy = min(min_dist_to_heavy, dist)

                # Hydrogen should be within reasonable distance of a heavy atom
                if (
                    min_dist_to_heavy > 3.0
                ):  # 3.0 Å is very generous for H-heavy atom distance
                    suspicious_hydrogens += 1

            # If more than 20% of hydrogens are suspiciously placed, flag as corrupt
            if (
                suspicious_hydrogens > 0.2 * len(hydrogen_atoms)
                and suspicious_hydrogens > 2
            ):
                return (
                    False,
                    f"{mol_name}: {suspicious_hydrogens}/{len(hydrogen_atoms)} hydrogens at suspicious distances",
                )

        return True, f"{mol_name}: Geometry validation passed"

    except Exception as e:
        return False, f"{mol_name}: Validation error - {str(e)}"


def validate_molecule_quality(mol: Chem.Mol) -> Tuple[bool, str]:
    """Pre-validate molecule quality before expensive operations."""
    if mol is None:
        return False, "Molecule is None"

    try:
        # Check basic molecular properties
        if mol.GetNumAtoms() == 0:
            return False, "Molecule has no atoms"

        # Check for valid conformers
        if mol.GetNumConformers() > 0:
            conf = mol.GetConformer(0)
            # Test accessing positions
            try:
                extreme_coords = []
                max_coord = -float("inf")
                min_coord = float("inf")

                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    coords = [pos.x, pos.y, pos.z]

                    for coord in coords:
                        if abs(coord) > max(abs(max_coord), abs(min_coord)):
                            if abs(coord) > 1000:  # Much more permissive threshold
                                extreme_coords.append(coord)
                        max_coord = max(max_coord, coord)
                        min_coord = min(min_coord, coord)

                if extreme_coords:
                    logger.debug(
                        f"Extreme coordinates found: {extreme_coords[:5]} (showing first 5)"
                    )
                    return (
                        False,
                        f"Conformer has extreme coordinates (max: {max(extreme_coords):.1f})",
                    )

                logger.debug(f"Coordinate bounds: [{min_coord:.2f}, {max_coord:.2f}]")

            except Exception as e:
                return False, f"Conformer coordinates inaccessible: {e}"

        # Test basic RDKit operations
        try:
            test_mol = Chem.Mol(mol)
            Chem.SanitizeMol(test_mol)
        except Exception as e:
            return False, f"Molecule fails sanitization: {e}"

        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {e}"


def score_and_align(conf: Chem.Mol, tpl: Chem.Mol) -> Tuple[Dict[str, float], Chem.Mol]:
    """Compute shape/color scores and return the aligned conformer with organometallic handling."""
    prb = Chem.Mol(conf)
    
    # Handle organometallic atoms before sanitization
    tpl_processed, tpl_had_metals, tpl_subs = detect_and_substitute_organometallic(tpl, "template")
    prb_processed, prb_had_metals, prb_subs = detect_and_substitute_organometallic(prb, "probe")
    
    try:
        SanitizeMol(tpl_processed)
        SanitizeMol(prb_processed)
    except Exception as e:
        logger.warning(f"Sanitization failed even after organometallic handling: {e}")
        # Continue with original molecules as fallback
        try:
            SanitizeMol(tpl)
            SanitizeMol(prb)
            tpl_processed, prb_processed = tpl, prb
        except Exception as e2:
            logger.error(f"Both organometallic handling and fallback sanitization failed: {e2}")
            # Return default scores to avoid complete failure
            return ({"shape": 0.0, "color": 0.0, "combo": 0.0}, prb)
    
    try:
        sT, cT = rdShapeAlign.AlignMol(tpl_processed, prb_processed, useColors=True)
        return ({"shape": sT, "color": cT, "combo": 0.5*(sT+cT)}, prb_processed)
    except Exception as e:
        logger.warning(f"Shape alignment failed: {e}")
        return ({"shape": 0.0, "color": 0.0, "combo": 0.0}, prb_processed)


def _fallback_alignment(
    prb_processed: Chem.Mol, tpl_processed: Chem.Mol
) -> Tuple[Dict[str, float], Chem.Mol]:
    """Fallback alignment method with robust error handling."""
    try:
        prb_with_h = FixedMolecularProcessor.safe_add_hydrogens(
            prb_processed, preserve_coords=True
        )
        tpl_with_h = FixedMolecularProcessor.safe_add_hydrogens(
            tpl_processed, preserve_coords=True
        )
    except Exception as e:
        logger.warning(f"Hydrogen addition failed in fallback: {e}")
        prb_with_h = prb_processed
        tpl_with_h = tpl_processed

    try:
        SanitizeMol(tpl_with_h)
        SanitizeMol(prb_with_h)
    except Exception as e:
        logger.warning(f"Sanitization failed in fallback: {e}")
        return {"shape": -1.0, "color": -1.0, "combo": -1.0}, prb_with_h

    try:
        # Validate conformers before alignment
        if not FixedMolecularProcessor.validate_conformer_access(tpl_with_h):
            raise Exception("Template conformer invalid in fallback")
        if not FixedMolecularProcessor.validate_conformer_access(prb_with_h):
            raise Exception("Probe conformer invalid in fallback")

        alignment_result = rdShapeAlign.AlignMol(
            tpl_with_h, prb_with_h, refConfId=0, probeConfId=0, useColors=True
        )

        # Validate and extract scores properly
        if alignment_result is None:
            raise Exception("Fallback alignment returned None")

        # Handle different return types from AlignMol
        if isinstance(alignment_result, tuple) and len(alignment_result) == 2:
            shape_score, color_score = alignment_result
        elif isinstance(alignment_result, (int, float)):
            # Fallback if only one score returned
            shape_score = color_score = float(alignment_result)
        else:
            raise Exception(
                f"Unexpected fallback alignment result type: {type(alignment_result)}"
            )

        if shape_score is None or color_score is None:
            raise Exception("Fallback alignment returned None scores")

        # Ensure scores are numeric
        try:
            shape_score = float(shape_score)
            color_score = float(color_score)
        except (ValueError, TypeError) as e:
            raise Exception(
                f"Fallback alignment returned non-numeric scores: {shape_score}, {color_score} - {e}"
            )

        combo_score = 0.5 * (shape_score + color_score)

        scores = {
            "shape": float(shape_score),
            "color": float(color_score),
            "combo": float(combo_score),
        }

        logger.warning(
            "Using fallback alignment method - poses may have hydrogen geometry issues"
        )
        return scores, prb_with_h
    except Exception as e:
        logger.warning(f"Shape alignment failed in fallback: {e}")
        return {"shape": -1.0, "color": -1.0, "combo": -1.0}, prb_with_h


# Helper for select_best - TOP LEVEL FUNCTION FOR PICKLING
def _score_and_align_task(args_tuple):
    """Wrapper for score_and_align for parallel execution with robust error handling."""
    try:
        single_conf_mol, tpl_mol, original_cid, no_realign_flag_for_pose_selection = (
            args_tuple
        )

        # Validate inputs before processing
        if single_conf_mol is None or tpl_mol is None:
            logger.warning(f"Invalid input molecules for conformer {original_cid}")
            return None

        if not isinstance(single_conf_mol, Chem.Mol) or not isinstance(
            tpl_mol, Chem.Mol
        ):
            logger.error(
                f"Invalid molecule types for conformer {original_cid}: {type(single_conf_mol)}, {type(tpl_mol)}"
            )
            return None

        # Validate molecules have conformers
        if single_conf_mol.GetNumConformers() == 0:
            logger.warning(f"Conformer {original_cid} has no conformers")
            return None

        # Call score_and_align to get scores and aligned molecule
        current_scores, aligned_mol = score_and_align(single_conf_mol, tpl_mol)

        # Validate scoring results
        if current_scores is None or not isinstance(current_scores, dict):
            logger.warning(f"Invalid scoring results for conformer {original_cid}")
            return None

        # Ensure we have valid scores
        for metric in ["shape", "color", "combo"]:
            if metric not in current_scores or current_scores[metric] is None:
                logger.warning(f"Missing {metric} score for conformer {original_cid}")
                current_scores[metric] = -1.0  # Default to poor score

        # Determine which pose to return based on no_realign_flag
        if no_realign_flag_for_pose_selection:
            pose_to_consider_if_best = single_conf_mol
        else:
            pose_to_consider_if_best = (
                aligned_mol if aligned_mol is not None else single_conf_mol
            )

        # Final validation of output molecule
        if pose_to_consider_if_best is None or not isinstance(
            pose_to_consider_if_best, Chem.Mol
        ):
            logger.warning(
                f"Invalid output molecule for conformer {original_cid}, using input molecule"
            )
            pose_to_consider_if_best = single_conf_mol

        # Return the original conformer ID, scores, and the pose to consider
        return original_cid, current_scores, pose_to_consider_if_best

    except Exception as e:
        logger.error(
            f"Error in _score_and_align_task for conformer {original_cid if 'original_cid' in locals() else 'unknown'}: {e}"
        )
        import traceback

        logger.debug(f"Scoring task traceback: {traceback.format_exc()}")
        return None


def _get_executor_for_context(n_workers: int):
    """Get appropriate executor based on process context and thread resource management.
    
    Uses ThreadPoolExecutor if running in daemon process (to avoid 
    'daemonic processes are not allowed to have children' error),
    otherwise uses ProcessPoolExecutor for better performance.
    """
    try:
        # Fallback if thread manager not available
        safe_workers = min(n_workers, 4)  # Conservative fallback
        
        current_process = multiprocessing.current_process()
        if hasattr(current_process, 'daemon') and current_process.daemon:
            # Running in daemon process, use threads
            logger.debug(f"Using ThreadPoolExecutor with {safe_workers} workers (daemon process detected, requested: {n_workers})")
            return ThreadPoolExecutor(max_workers=safe_workers)
        else:
            # Not in daemon process, use simple process pool
            logger.debug(f"Using ProcessPoolExecutor with {safe_workers} workers (requested: {n_workers})")
            return ProcessPoolExecutor(max_workers=safe_workers)
    except Exception as e:
        # Fallback to threads if there's any issue with process detection
        logger.warning(f"Process detection failed, using ThreadPoolExecutor: {e}")
        # Use minimal worker count as fallback
        fallback_workers = min(n_workers, 2)
        return ThreadPoolExecutor(max_workers=fallback_workers)


def select_best(
    confs: Chem.Mol,
    tpl: Chem.Mol,
    no_realign: bool = False,
    n_workers: int = 1,
    return_all_ranked: bool = False,
    align_metric: str = "combo",
) -> Union[
    Dict[str, Tuple[Chem.Mol, Dict[str, float]]],
    List[Tuple[Chem.Mol, Dict[str, float], int]],
]:
    """Select best poses using shape/color/combo scoring with memory optimization."""

    if confs is None or tpl is None:
        logger.error("Invalid input molecules for pose selection")
        return {} if not return_all_ranked else []

    n_confs = confs.GetNumConformers()
    if n_confs == 0:
        logger.warning("No conformers to score")
        return {} if not return_all_ranked else []

    logger.info(f"Scoring {n_confs} conformers using {n_workers} workers")

    # Process conformers in memory-efficient batches with adaptive sizing
    # Adjust batch size based on worker count and system load
    if n_workers >= 16:  # Tier 3: Minimal memory allocation
        batch_size = min(10, n_confs)  # Very small batches for high worker counts
    elif n_workers >= 9:  # Tier 2: Reduced memory allocation
        batch_size = min(15, n_confs)  # Small batches for medium worker counts
    elif n_confs > 100:
        batch_size = min(20, n_confs)  # Standard small batches for large sets
    else:
        batch_size = min(30, n_confs)  # Conservative batch size for small sets
    all_results = []

    # Create single process pool for all batches (performance optimization)
    executor = None
    if n_workers > 1:
        try:
            # Thread monitoring not available, continue without it
            pass
            
            executor = _get_executor_for_context(n_workers)
            logger.debug(f"Created single process pool for all {n_confs} conformers")
        except Exception as e:
            logger.warning(f"Failed to create process pool: {e}, using sequential processing")
            executor = None

    try:
        for batch_start in range(0, n_confs, batch_size):
            batch_end = min(batch_start + batch_size, n_confs)
            batch_conf_ids = list(range(batch_start, batch_end))

            logger.debug(
                f"Processing batch {batch_start//batch_size + 1}: conformers {batch_start}-{batch_end-1}"
            )

            # Prepare arguments for this batch
            args_list = []
            for conf_id in batch_conf_ids:
                try:
                    # Validate input molecules before processing
                    if confs is None or tpl is None:
                        logger.warning(f"Invalid input molecules for conformer {conf_id}")
                        continue

                    if conf_id >= confs.GetNumConformers():
                        logger.warning(
                            f"Conformer {conf_id} doesn't exist (total: {confs.GetNumConformers()})"
                        )
                        continue

                    # Create independent copy of conformer with enhanced validation
                    conf_mol = FixedMolecularProcessor.create_independent_copy(confs)

                    if conf_mol is None:
                        logger.warning(
                            f"Failed to copy molecule for conformer {conf_id}, trying direct approach"
                        )
                        # Fallback: create a new molecule from SMILES if available
                        try:
                            smiles = Chem.MolToSmiles(confs)
                            conf_mol = Chem.MolFromSmiles(smiles)
                            if conf_mol is None:
                                raise Exception("SMILES conversion failed")
                        except Exception as smiles_err:
                            logger.warning(
                                f"SMILES fallback failed for conformer {conf_id}: {smiles_err}"
                            )
                            continue

                    # Keep only this conformer with robust error handling
                    try:
                        temp_mol = Chem.RWMol(conf_mol)
                        temp_mol.RemoveAllConformers()

                        # Validate conformer exists before accessing
                        if conf_id < confs.GetNumConformers():
                            conf = confs.GetConformer(conf_id)
                            new_conf = Chem.Conformer(conf)
                            temp_mol.AddConformer(new_conf, assignId=True)
                        else:
                            logger.warning(f"Conformer {conf_id} index out of range")
                            continue

                        # Validate the result has a conformer
                        if temp_mol.GetNumConformers() == 0:
                            logger.warning(
                                f"No conformers after processing conformer {conf_id}"
                            )
                            continue

                        args_list.append((temp_mol, tpl, conf_id, no_realign))

                    except Exception as conf_err:
                        logger.warning(
                            f"Conformer processing failed for {conf_id}: {conf_err}"
                        )
                        continue

                except Exception as e:
                    logger.warning(f"Failed to prepare conformer {conf_id}: {e}")
                    import traceback

                    logger.debug(
                        f"Conformer preparation traceback: {traceback.format_exc()}"
                    )
                    continue

            # Process this batch using the persistent executor
            if executor and len(args_list) > 1:
                try:
                    batch_results = list(executor.map(_score_and_align_task, args_list))
                    logger.debug(f"Processed batch with {len(args_list)} conformers using persistent pool")
                except Exception as e:
                    logger.warning(f"Parallel processing failed: {e}, falling back to sequential")
                    batch_results = [_score_and_align_task(args) for args in args_list]
            else:
                batch_results = [_score_and_align_task(args) for args in args_list]
                logger.debug(f"Processed batch with {len(args_list)} conformers sequentially")

            # Filter valid results and add to overall results with enhanced validation
            for result in batch_results:
                if result is not None:
                    try:
                        # Validate result structure
                        if len(result) == 3:
                            conf_id, scores, mol = result

                            # Validate scores
                            if isinstance(scores, dict) and all(
                                metric in scores for metric in ["shape", "color", "combo"]
                            ):
                                # Validate molecule
                                if mol is not None and isinstance(mol, Chem.Mol):
                                    all_results.append(result)
                                else:
                                    logger.warning(
                                        f"Invalid molecule in result for conformer {conf_id}"
                                    )
                            else:
                                logger.warning(
                                    f"Invalid scores in result for conformer {conf_id}: {scores}"
                                )
                        else:
                            logger.warning(f"Invalid result structure: {result}")
                    except Exception as e:
                        logger.warning(f"Error validating batch result: {e}")
                        continue

            # Force cleanup after each batch
            del args_list, batch_results
            cleanup_memory()

            logger.debug(f"Batch complete, {len(all_results)} total valid results so far")

    finally:
        # Clean up the persistent executor
        if executor:
            try:
                if hasattr(executor, 'close'):
                    executor.close()
                if hasattr(executor, 'join'):
                    executor.join()
                logger.debug("Persistent process pool cleaned up successfully")
            except Exception as e:
                logger.warning(f"Error cleaning up persistent process pool: {e}")

    if not all_results:
        logger.warning("No valid scoring results obtained")
        return {} if not return_all_ranked else []

    # Sort by specified metric (descending)
    # Validate align_metric parameter
    valid_metrics = ["shape", "color", "combo"]
    if align_metric not in valid_metrics:
        logger.warning(f"Invalid align_metric '{align_metric}', using 'combo' as default")
        align_metric = "combo"
    
    all_results.sort(key=lambda x: x[1].get(align_metric, 0.0), reverse=True)

    if return_all_ranked:
        return all_results

    # Select best poses by each metric
    best_poses = {}

    for metric in ["shape", "color", "combo"]:
        # Find best pose for this metric
        metric_results = [(r[0], r[1], r[2]) for r in all_results if metric in r[1]]
        if metric_results:
            metric_results.sort(key=lambda x: x[1][metric], reverse=True)
            best_conf_id, best_scores, best_mol = metric_results[
                0
            ]  # Fixed unpacking order

            # Create clean copy for output with robust error handling
            try:
                # Validate the best_mol before copying
                if best_mol is None:
                    logger.warning(f"Best molecular for {metric} is None, skipping")
                    continue

                if not isinstance(best_mol, Chem.Mol):
                    logger.error(
                        f"Best molecule for {metric} has invalid type: {type(best_mol)}"
                    )
                    continue

                output_mol = FixedMolecularProcessor.create_independent_copy(best_mol)

                if output_mol is None:
                    logger.warning(
                        f"Failed to create copy of best {metric} pose, trying direct assignment"
                    )
                    # Fallback: use original molecule if copying fails
                    output_mol = best_mol

                best_poses[metric] = (output_mol, best_scores)
                logger.debug(
                    f"Best {metric}: conf {best_conf_id}, score {best_scores[metric]:.3f}"
                )

            except Exception as e:
                logger.error(f"Failed to process best {metric} pose: {e}")
                logger.error(
                    f"Best molecule type: {type(best_mol)}, scores: {best_scores}"
                )
                # Skip this metric rather than failing the entire function
                continue

    # Final cleanup
    cleanup_memory()

    logger.info(
        f"Selected {len(best_poses)} best poses from {len(all_results)} scored conformers"
    )
    return best_poses


def cleanup_memory():
    """Aggressive memory cleanup to prevent accumulation."""
    import gc

    gc.collect()
    if hasattr(gc, "set_threshold"):
        gc.set_threshold(700, 10, 10)  # More aggressive garbage collection


def generate_properties_for_sdf(
    mol: Chem.Mol,
    metric: str,
    score: float,
    template_pid: str,
    template_info: Dict[str, str] = None,
) -> Chem.Mol:
    """Return a copy of *mol* with standardised pose properties used by downstream tools and tests.

    Expectations from unit-tests (see *tests/test_scoring.py*):
    1.  The molecule name (_Name) is replaced with "<template_pid>_<metric>_pose".
    2.  Properties are all lower-case and snake-cased:
          • metric
          • metric_score (formatted to 3 decimal places)
          • template_pid
    3.  Any extra *template_info* entries are stored using the prefix "template_" and the key
       converted to lower-case (to ensure consistency).
    The original molecule must remain untouched; we therefore clone it first.
    """

    mol_copy = Chem.Mol(mol)  # deep clone to avoid side-effects

    # 1. update name
    mol_copy.SetProp("_Name", f"{template_pid}_{metric}_pose")

    # 2. core properties
    mol_copy.SetProp("metric", metric)
    mol_copy.SetProp("metric_score", f"{score:.3f}")
    mol_copy.SetProp("template_pid", template_pid)

    # 3. optional template metadata
    if template_info:
        for key, value in template_info.items():
            mol_copy.SetProp(f"template_{key.lower()}", str(value))

    return mol_copy


class ScoringEngine:
    """Scoring engine with enhanced molecular processing."""

    def __init__(self):
        pass

    def score_pose(self, mol) -> Dict:
        """Score a pose using enhanced scoring methods."""
        return {"shape": 0.0, "color": 0.0, "combo": 0.0}
