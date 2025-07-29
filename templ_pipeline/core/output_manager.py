#!/usr/bin/env python3
"""
Enhanced Output Manager for TEMPL Pipeline

This module provides enhanced output handling with timestamped folders,
PDB ID-based naming, and comprehensive molecular properties.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from rdkit import Chem

log = logging.getLogger(__name__)


class EnhancedOutputManager:
    """Enhanced output manager with timestamped folders and comprehensive properties."""
    
    def __init__(self, base_output_dir: str = "output", run_id: Optional[str] = None):
        """
        Initialize the enhanced output manager.
        
        Args:
            base_output_dir: Base output directory (default: "output")
            run_id: Optional run ID for folder naming (default: auto-generated)
        """
        self.base_output_dir = Path(base_output_dir)
        self.run_id = run_id
        self.timestamped_folder = None
        self.pdb_id = None
        
    def setup_output_folder(self, pdb_id: str) -> Path:
        """
        Create timestamped output folder for a specific PDB ID.
        
        Args:
            pdb_id: PDB ID for naming (can be a path)
            
        Returns:
            Path to the created timestamped folder
        """
        # Use only the base name (strip directories and extension)
        import os
        base_pdb_id = os.path.splitext(os.path.basename(pdb_id))[0]
        self.pdb_id = base_pdb_id
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create folder name
        if self.run_id:
            folder_name = f"templ_run_{timestamp}_{self.run_id}_{base_pdb_id}"
        else:
            folder_name = f"templ_run_{timestamp}_{base_pdb_id}"
        
        # Create full path
        self.timestamped_folder = self.base_output_dir / folder_name
        self.timestamped_folder.mkdir(parents=True, exist_ok=True)
        
        log.info(f"Created timestamped output folder: {self.timestamped_folder}")
        return self.timestamped_folder
        
    def get_output_path(self, filename: str) -> Path:
        """
        Get full output path for a filename.
        
        Args:
            filename: Filename to get path for
            
        Returns:
            Full path to the output file
        """
        if self.timestamped_folder is None:
            raise ValueError("Output folder not set up. Call setup_output_folder() first.")
        
        return self.timestamped_folder / filename
        
    def save_top_poses(self, 
                      poses: Dict[str, Tuple[Chem.Mol, Dict[str, float]]], 
                      template: Chem.Mol,
                      mcs_details: Dict,
                      crystal_mol: Optional[Chem.Mol] = None) -> Path:
        """Save top poses to SDF file with enhanced metadata."""
        # Use only the base name for output file
        import os
        if hasattr(self, 'run_id') and self.run_id:
            base_name = self.run_id
        elif template is not None and template.HasProp('template_pid'):
            base_name = template.GetProp('template_pid')
        else:
            base_name = "templ_output"
        # If base_name is a path, strip directories and extension
        base_name = os.path.splitext(os.path.basename(base_name))[0]
        output_file = self.get_output_path(f"{base_name}_top3_poses.sdf")
        from rdkit import Chem
        with Chem.SDWriter(str(output_file)) as writer:
            for metric, (pose, scores) in poses.items():
                if pose is not None:
                    enhanced_pose = self._create_enhanced_pose(
                        pose, metric, scores, template, mcs_details, crystal_mol
                    )
                    writer.write(enhanced_pose)
        return output_file
        
    def save_all_poses(self, 
                      ranked_poses: List[Tuple[Chem.Mol, Dict[str, float], int]],
                      template: Chem.Mol,
                      mcs_details: Dict,
                      crystal_mol: Optional[Chem.Mol] = None,
                      max_poses: Optional[int] = None) -> Path:
        """Save all ranked poses to SDF file."""
        import os
        if hasattr(self, 'run_id') and self.run_id:
            base_name = self.run_id
        elif template is not None and template.HasProp('template_pid'):
            base_name = template.GetProp('template_pid')
        else:
            base_name = "templ_output"
        base_name = os.path.splitext(os.path.basename(base_name))[0]
        output_file = self.get_output_path(f"{base_name}_all_poses.sdf")
        from rdkit import Chem
        with Chem.SDWriter(str(output_file)) as writer:
            for conf_id, scores, pose in ranked_poses:
                if pose is not None:
                    enhanced_pose = self._create_enhanced_pose(
                        pose, "combo", scores, template, mcs_details, crystal_mol, pose_rank=conf_id
                    )
                    writer.write(enhanced_pose)
        return output_file
        
    def save_template(self, template: Chem.Mol) -> Path:
        """
        Save the winning template molecule.
        
        Args:
            template: Template molecule used for MCS
            
        Returns:
            Path to the saved template SDF file
        """
        import os
        base_name = os.path.splitext(os.path.basename(self.pdb_id))[0]
        output_file = self.get_output_path(f"{base_name}_template.sdf")
        
        with Chem.SDWriter(str(output_file)) as writer:
            writer.write(template)
        
        log.info(f"Saved template to: {output_file}")
        return output_file
        
    def save_pipeline_results(self, results_data: Dict) -> Path:
        """
        Save structured pipeline results to JSON.
        
        Args:
            results_data: Dictionary containing pipeline results
            
        Returns:
            Path to the saved JSON file
        """
        import json
        import os
        base_name = os.path.splitext(os.path.basename(self.pdb_id))[0]
        output_file = self.get_output_path(f"{base_name}_pipeline_results.json")
        
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        log.info(f"Saved pipeline results to: {output_file}")
        return output_file
        
    def _create_enhanced_pose(self, 
                            pose: Chem.Mol,
                            primary_metric: str,
                            scores: Dict[str, float],
                            template: Chem.Mol,
                            mcs_details: Dict,
                            crystal_mol: Optional[Chem.Mol] = None,
                            pose_rank: Optional[int] = None,
                            total_poses: Optional[int] = None,
                            original_conformer_id: Optional[int] = None) -> Chem.Mol:
        """
        Create an enhanced pose with comprehensive properties.
        
        Args:
            pose: Original pose molecule
            primary_metric: Primary metric for this pose
            scores: Dictionary of all scores
            template: Template molecule
            mcs_details: MCS information dictionary
            crystal_mol: Crystal structure for RMSD calculation
            pose_rank: Rank of this pose (for all poses)
            total_poses: Total number of poses generated
            original_conformer_id: Original conformer ID
            
        Returns:
            Enhanced pose molecule with all properties
        """
        # Validate pose before processing
        if pose is None:
            raise ValueError("Cannot create enhanced pose from None")
        if not hasattr(pose, 'GetNumAtoms'):
            raise ValueError(f"Invalid pose type: {type(pose)}, expected RDKit Mol object")
        
        # Create a copy to avoid modifying the original
        enhanced_pose = Chem.Mol(pose)
        
        # Set basic pose properties
        enhanced_pose.SetProp("_Name", f"{self.pdb_id}_{primary_metric}_pose")
        enhanced_pose.SetProp("metric", primary_metric)
        enhanced_pose.SetProp("metric_score", f"{scores[primary_metric]:.3f}")
        
        # Set all Tanimoto scores
        for score_type, score_value in scores.items():
            enhanced_pose.SetProp(f"tanimoto_{score_type}_score", f"{score_value:.3f}")
        
        # Add template information
        self._add_template_properties(enhanced_pose, template)
        
        # Add MCS information
        self._add_mcs_properties(enhanced_pose, mcs_details)
        
        # Add pose ranking information (if available)
        if pose_rank is not None:
            enhanced_pose.SetProp("pose_rank_combo", str(pose_rank))
        if total_poses is not None:
            enhanced_pose.SetProp("total_poses_generated", str(total_poses))
        if original_conformer_id is not None:
            enhanced_pose.SetProp("conformer_original_id", str(original_conformer_id))
        
        # Calculate RMSD if crystal structure is available
        if crystal_mol:
            try:
                crys = Chem.RemoveHs(crystal_mol)
                rms = self._calculate_rmsd(enhanced_pose, crys)
                enhanced_pose.SetProp("rmsd_to_crystal", f"{rms:.3f}")
            except Exception as e:
                log.error(f"Error calculating RMSD: {str(e)}")
        
        return enhanced_pose
        
    def _add_template_properties(self, pose: Chem.Mol, template: Chem.Mol) -> None:
        """Add template-related properties to the pose."""
        # Get template PDB ID
        template_pid = "unknown"
        if template.HasProp("template_pdb"):
            template_pid = template.GetProp("template_pdb")
        elif template.HasProp("template_pid"):
            template_pid = template.GetProp("template_pid")
        
        pose.SetProp("template_pid", template_pid)
        
        # Copy template alignment properties
        template_props = [
            "embedding_similarity", "ref_chains", "mob_chains", "ca_rmsd",
            "aligned_residues_count", "total_ref_residues", "total_mob_residues", 
            "aligned_percentage"
        ]
        
        for prop_name in template_props:
            if template.HasProp(prop_name):
                pose.SetProp(f"template_{prop_name}", template.GetProp(prop_name))
        
    def _add_mcs_properties(self, pose: Chem.Mol, mcs_details: Dict) -> None:
        """Add MCS-related properties to the pose."""
        if not mcs_details:
            return
        
        # Basic MCS information
        pose.SetProp("mcs_smarts", mcs_details.get("smarts", ""))
        pose.SetProp("mcs_atom_count", str(mcs_details.get("atom_count", 0)))
        pose.SetProp("mcs_bond_count", str(mcs_details.get("bond_count", 0)))
        pose.SetProp("mcs_similarity_score", f"{mcs_details.get('similarity_score', 0.0):.3f}")
        
        # Atom mapping (compact format)
        if mcs_details.get("query_atoms"):
            pose.SetProp("mcs_query_atoms", ",".join(map(str, mcs_details["query_atoms"])))
        if mcs_details.get("template_atoms"):
            pose.SetProp("mcs_template_atoms", ",".join(map(str, mcs_details["template_atoms"])))
        
        # Central atom fallback information
        if mcs_details.get("central_atom_fallback"):
            pose.SetProp("mcs_central_atom_fallback", "true")
        
    def _calculate_rmsd(self, pose: Chem.Mol, crystal: Chem.Mol) -> float:
        """Calculate RMSD between pose and crystal structure."""
        try:
            from spyrmsd.molecule import Molecule
            from spyrmsd.rmsd import rmsdwrapper
            
            # Ensure both molecules are processed consistently
            pose_clean = Chem.RemoveHs(pose)
            crystal_clean = Chem.RemoveHs(crystal)
            
            # Check if molecules have the same number of atoms
            if pose_clean.GetNumAtoms() != crystal_clean.GetNumAtoms():
                log.debug(f"RMSD skipped: atom count mismatch (pose: {pose_clean.GetNumAtoms()}, crystal: {crystal_clean.GetNumAtoms()})")
                return float("nan")
            
            # Note: Removed molecular formula comparison as it incorrectly rejects
            # same molecules with different protonation states (e.g., C27H37N5 vs C27H41N5+4)
            # spyrmsd handles molecule compatibility validation internally
            
            return rmsdwrapper(
                Molecule.from_rdkit(pose_clean),
                Molecule.from_rdkit(crystal_clean),
                minimize=False, strip=True, symmetry=True
            )[0]
        except Exception as e:
            log.debug(f"RMSD calculation failed: {e}")
            return float("nan")
    
    def _validate_poses_structure(self, poses: Any) -> bool:
        """
        Validate poses structure to prevent KeyError issues.
        
        Args:
            poses: The poses data structure to validate
            
        Returns:
            True if valid, False otherwise
        """
        if poses is None:
            log.warning("Poses structure is None")
            return False
        
        if not isinstance(poses, dict):
            log.warning(f"Poses structure is not a dictionary, got: {type(poses)}")
            return False
        
        if len(poses) == 0:
            log.warning("Poses dictionary is empty")
            return False
        
        # Check if we have expected metric keys
        expected_metrics = {'shape', 'color', 'combo'}
        actual_metrics = set(poses.keys())
        
        if not actual_metrics.intersection(expected_metrics):
            log.warning(f"No valid metric keys found. Expected one of {expected_metrics}, got: {actual_metrics}")
            return False
        
        return True
    
    def _validate_single_pose(self, metric: str, pose_data: Any) -> bool:
        """
        Validate a single pose entry.
        
        Args:
            metric: The metric name
            pose_data: The pose data to validate
            
        Returns:
            True if valid, False otherwise
        """
        if pose_data is None:
            log.warning(f"Pose data for {metric} is None")
            return False
        
        if not isinstance(pose_data, (tuple, list)) or len(pose_data) != 2:
            log.warning(f"Pose data for {metric} should be tuple/list of length 2, got: {type(pose_data)} with length {len(pose_data) if hasattr(pose_data, '__len__') else 'N/A'}")
            return False
        
        pose, scores = pose_data
        
        # Validate molecule
        if pose is not None and not hasattr(pose, 'GetNumAtoms'):
            log.warning(f"Pose for {metric} is not a valid RDKit molecule object: {type(pose)}")
            return False
        
        # Validate scores
        if not isinstance(scores, dict):
            log.warning(f"Scores for {metric} is not a dictionary: {type(scores)}")
            return False
        
        if metric not in scores:
            log.warning(f"Score for {metric} not found in scores dictionary: {list(scores.keys())}")
            return False
        
        return True 