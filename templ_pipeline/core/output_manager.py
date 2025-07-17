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
            pdb_id: PDB ID for naming
            
        Returns:
            Path to the created timestamped folder
        """
        self.pdb_id = pdb_id
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create folder name
        if self.run_id:
            folder_name = f"templ_run_{timestamp}_{self.run_id}_{pdb_id}"
        else:
            folder_name = f"templ_run_{timestamp}_{pdb_id}"
        
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
        """
        Save top 3 poses with enhanced properties.
        
        Args:
            poses: Dictionary of poses from select_best
            template: Template molecule used for MCS
            mcs_details: MCS information dictionary
            crystal_mol: Crystal structure for RMSD calculation (optional)
            
        Returns:
            Path to the saved SDF file
        """
        output_file = self.get_output_path(f"{self.pdb_id}_top3_poses.sdf")
        
        with Chem.SDWriter(str(output_file)) as writer:
            for metric, (pose, scores) in poses.items():
                if pose is None:
                    continue
                
                # Create enhanced pose with all properties
                enhanced_pose = self._create_enhanced_pose(
                    pose, metric, scores, template, mcs_details, crystal_mol
                )
                
                writer.write(enhanced_pose)
        
        log.info(f"Saved top 3 poses to: {output_file}")
        return output_file
        
    def save_all_poses(self, 
                      ranked_poses: List[Tuple[Chem.Mol, Dict[str, float], int]],
                      template: Chem.Mol,
                      mcs_details: Dict,
                      crystal_mol: Optional[Chem.Mol] = None,
                      max_poses: Optional[int] = None) -> Path:
        """
        Save all ranked poses with enhanced properties.
        
        Args:
            ranked_poses: List of (pose, scores, original_cid) tuples sorted by combo score
            template: Template molecule used for MCS
            mcs_details: MCS information dictionary
            crystal_mol: Crystal structure for RMSD calculation (optional)
            max_poses: Maximum number of poses to save (optional)
            
        Returns:
            Path to the saved SDF file
        """
        output_file = self.get_output_path(f"{self.pdb_id}_all_poses.sdf")
        
        # Limit poses if requested
        poses_to_save = ranked_poses[:max_poses] if max_poses else ranked_poses
        
        with Chem.SDWriter(str(output_file)) as writer:
            for rank, (original_cid, scores, pose) in enumerate(poses_to_save, 1):
                if pose is None:
                    continue
                
                # Create enhanced pose with all properties
                enhanced_pose = self._create_enhanced_pose(
                    pose, "combo", scores, template, mcs_details, crystal_mol,
                    pose_rank=rank, total_poses=len(ranked_poses), 
                    original_conformer_id=original_cid
                )
                
                # Set name for all poses
                enhanced_pose.SetProp("_Name", f"{self.pdb_id}_pose_rank_{rank}")
                
                writer.write(enhanced_pose)
        
        log.info(f"Saved {len(poses_to_save)} poses to: {output_file}")
        return output_file
        
    def save_template(self, template: Chem.Mol) -> Path:
        """
        Save the winning template molecule.
        
        Args:
            template: Template molecule used for MCS
            
        Returns:
            Path to the saved template SDF file
        """
        output_file = self.get_output_path(f"{self.pdb_id}_template.sdf")
        
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
        
        output_file = self.get_output_path(f"{self.pdb_id}_pipeline_results.json")
        
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
            
            return rmsdwrapper(
                Molecule.from_rdkit(pose),
                Molecule.from_rdkit(crystal),
                minimize=False, strip=True, symmetry=True
            )[0]
        except Exception as e:
            log.error(f"RMSD calculation failed: {e}")
            return float("nan") 