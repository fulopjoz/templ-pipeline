"""
TEMPL Pipeline Output Management Module

This module implements the adaptive file naming system and output management
following FAIR principles. It provides context-aware file naming that adapts
to different prediction scenarios (PDB-based, template-based, batch, custom).

Key Features:
- Adaptive file naming based on prediction context
- Timestamp-based unique identifiers
- FAIR-compliant output organization
- Backward compatibility with existing pipeline
"""

import os
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PredictionContext:
    """Container for prediction context information used in file naming."""
    pdb_id: Optional[str] = None
    template_source: Optional[str] = None  # 'sdf', 'database', 'custom'
    batch_id: Optional[str] = None
    input_file: Optional[str] = None
    smiles: Optional[str] = None
    custom_prefix: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return {
            'pdb_id': self.pdb_id,
            'template_source': self.template_source,
            'batch_id': self.batch_id,
            'input_file': self.input_file,
            'smiles': self.smiles,
            'custom_prefix': self.custom_prefix
        }

class AdaptiveFileNamingEngine:
    """
    Intelligent file naming engine that generates context-appropriate filenames.
    
    Implements the naming patterns designed in FAIR architecture:
    - PDB-based: {pdb_id}_{timestamp}_poses.sdf
    - Template-based: templ_{timestamp}_poses.sdf
    - Batch: batch_{batch_id}_{timestamp}_poses.sdf
    - Custom: custom_{hash}_{timestamp}_poses.sdf
    """
    
    def __init__(self, timestamp_format: str = "%Y%m%d_%H%M%S"):
        """
        Initialize the file naming engine.
        
        Args:
            timestamp_format: strftime format for timestamps (default: YYYYMMDD_HHMMSS)
        """
        self.timestamp_format = timestamp_format
        
    def generate_filename(self, 
                         context: PredictionContext, 
                         extension: str = "sdf",
                         custom_timestamp: Optional[str] = None) -> str:
        """
        Generate context-appropriate filename.
        
        Args:
            context: Prediction context information
            extension: File extension (default: 'sdf')
            custom_timestamp: Optional custom timestamp (for testing/reproducibility)
            
        Returns:
            Generated filename following FAIR naming conventions
        """
        # Generate timestamp
        timestamp = custom_timestamp or datetime.now().strftime(self.timestamp_format)
        
        # Determine naming pattern based on context
        if context.pdb_id:
            # PDB-based prediction
            pdb_id = context.pdb_id.lower()
            filename = f"{pdb_id}_{timestamp}_poses.{extension}"
            logger.debug(f"PDB-based naming: {filename}")
            
        elif context.template_source == 'sdf':
            # Template-based prediction from SDF file
            filename = f"templ_{timestamp}_poses.{extension}"
            logger.debug(f"Template-based naming: {filename}")
            
        elif context.batch_id:
            # Batch processing
            batch_id = context.batch_id
            filename = f"batch_{batch_id}_{timestamp}_poses.{extension}"
            logger.debug(f"Batch-based naming: {filename}")
            
        elif context.custom_prefix:
            # Custom prefix provided
            prefix = context.custom_prefix
            filename = f"{prefix}_{timestamp}_poses.{extension}"
            logger.debug(f"Custom prefix naming: {filename}")
            
        else:
            # Custom ligand or unknown source - generate hash
            input_hash = self._generate_input_hash(context)
            filename = f"custom_{input_hash}_{timestamp}_poses.{extension}"
            logger.debug(f"Custom hash naming: {filename}")
        
        return filename
    
    def _generate_input_hash(self, context: PredictionContext) -> str:
        """
        Generate short hash from input for identification.
        
        Args:
            context: Prediction context
            
        Returns:
            8-character hex hash for identification
        """
        # Create a string from available context information
        hash_input = ""
        
        if context.smiles:
            hash_input = context.smiles
        elif context.input_file:
            hash_input = str(context.input_file)
        else:
            # Fallback to timestamp-based hash
            hash_input = str(datetime.now().timestamp())
        
        # Generate MD5 hash and take first 8 characters
        return hashlib.md5(hash_input.encode()).hexdigest()[:8]
    
    def generate_metadata_filename(self, base_filename: str) -> str:
        """
        Generate corresponding metadata filename.
        
        Args:
            base_filename: Base output filename
            
        Returns:
            Metadata filename (JSON format)
        """
        base_path = Path(base_filename)
        metadata_filename = base_path.with_suffix('.json').name
        return metadata_filename
    
    def generate_archive_filename(self, base_filename: str) -> str:
        """
        Generate archive filename for complete results package.
        
        Args:
            base_filename: Base output filename
            
        Returns:
            Archive filename (ZIP format)
        """
        base_path = Path(base_filename)
        # Remove the final extension and add _complete.zip
        name_without_ext = base_path.stem
        if name_without_ext.endswith('_poses'):
            name_without_ext = name_without_ext[:-6]  # Remove '_poses'
        archive_filename = f"{name_without_ext}_complete.zip"
        return archive_filename

class OutputManager:
    """
    Main output management class that coordinates file naming, saving, and organization.
    
    Provides a unified interface for all TEMPL pipeline outputs while maintaining
    backward compatibility with existing code.
    """
    
    def __init__(self, output_dir: str = "output", run_id: Optional[str] = None):
        """
        Initialize the output manager.
        
        Args:
            output_dir: Base directory for outputs
            run_id: Optional run identifier for organization
        """
        self.naming_engine = AdaptiveFileNamingEngine()
        
        # Create output directory structure
        if run_id:
            self.output_dir = Path(f"{output_dir}_{run_id}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"{output_dir}_{timestamp}")
            
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Output manager initialized: {self.output_dir}")
    
    def generate_output_filename(self, 
                                context: PredictionContext,
                                file_type: str = "poses",
                                extension: str = "sdf") -> str:
        """
        Generate output filename with full path.
        
        Args:
            context: Prediction context for naming
            file_type: Type of output file (poses, metadata, etc.)
            extension: File extension
            
        Returns:
            Full path to output file
        """
        # Modify context for different file types
        if file_type == "metadata":
            extension = "json"
        elif file_type == "archive":
            extension = "zip"
        
        filename = self.naming_engine.generate_filename(context, extension)
        full_path = self.output_dir / filename
        
        logger.debug(f"Generated {file_type} filename: {full_path}")
        return str(full_path)
    
    def save_poses(self, 
                   poses: Dict[str, Tuple[Any, Dict[str, float]]], 
                   context: PredictionContext,
                   template_info: Optional[Dict[str, str]] = None) -> str:
        """
        Save poses to SDF file using adaptive naming.
        
        Args:
            poses: Dictionary of poses from pipeline
            context: Prediction context for file naming
            template_info: Optional template metadata
            
        Returns:
            Path to saved SDF file
        """
        from rdkit import Chem
        from .scoring import generate_properties_for_sdf
        
        output_file = self.generate_output_filename(context, "poses", "sdf")
        logger.info(f"Saving poses to {output_file}")
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with Chem.SDWriter(output_file) as writer:
            for method, (pose, scores) in poses.items():
                if pose is None:
                    logger.warning(f"No valid pose for {method}")
                    continue
                
                # Generate template PDB for metadata (backward compatibility)
                template_pdb = context.pdb_id or "template"
                
                # Add properties using existing function
                pose_with_props = generate_properties_for_sdf(
                    pose,
                    method,
                    scores.get(method, 0.0),
                    template_pdb,
                    {
                        "shape_score": f"{scores.get('shape', 0.0):.3f}",
                        "color_score": f"{scores.get('color', 0.0):.3f}",
                        "combo_score": f"{scores.get('combo', 0.0):.3f}",
                        "timestamp": datetime.now().isoformat(),
                        "prediction_context": str(context.template_source or "pdb_based")
                    }
                )
                
                writer.write(pose_with_props)
        
        logger.info(f"Successfully saved poses to {output_file}")
        return output_file
    
    def create_prediction_context(self, 
                                 pdb_id: Optional[str] = None,
                                 template_source: Optional[str] = None,
                                 ligand_smiles: Optional[str] = None,
                                 ligand_file: Optional[str] = None,
                                 batch_id: Optional[str] = None,
                                 custom_prefix: Optional[str] = None) -> PredictionContext:
        """
        Create prediction context from pipeline parameters.
        
        Args:
            pdb_id: Target PDB ID (if available)
            template_source: Source of templates ('sdf', 'database', 'custom')
            ligand_smiles: Input SMILES string
            ligand_file: Input ligand file path
            batch_id: Batch processing identifier
            custom_prefix: Custom filename prefix
            
        Returns:
            Prediction context for file naming
        """
        return PredictionContext(
            pdb_id=pdb_id,
            template_source=template_source,
            batch_id=batch_id,
            input_file=ligand_file,
            smiles=ligand_smiles,
            custom_prefix=custom_prefix
        )
    
    def get_output_summary(self) -> Dict[str, Any]:
        """
        Get summary of outputs in the output directory.
        
        Returns:
            Dictionary with output file information
        """
        if not self.output_dir.exists():
            return {"error": "Output directory does not exist"}
        
        files = list(self.output_dir.glob("*"))
        summary = {
            "output_directory": str(self.output_dir),
            "total_files": len(files),
            "files": []
        }
        
        for file_path in files:
            file_info = {
                "name": file_path.name,
                "size_bytes": file_path.stat().st_size,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                "type": file_path.suffix[1:] if file_path.suffix else "unknown"
            }
            summary["files"].append(file_info)
        
        return summary

# Convenience functions for backward compatibility
def create_output_manager(output_dir: str = "output", run_id: Optional[str] = None) -> OutputManager:
    """Create an output manager instance."""
    return OutputManager(output_dir, run_id)

def generate_adaptive_filename(pdb_id: Optional[str] = None, 
                              template_source: Optional[str] = None,
                              batch_id: Optional[str] = None,
                              smiles: Optional[str] = None,
                              extension: str = "sdf") -> str:
    """
    Quick function to generate adaptive filename without full context.
    
    Args:
        pdb_id: PDB ID if available
        template_source: Template source type
        batch_id: Batch identifier
        smiles: SMILES string
        extension: File extension
        
    Returns:
        Generated filename
    """
    engine = AdaptiveFileNamingEngine()
    context = PredictionContext(
        pdb_id=pdb_id,
        template_source=template_source,
        batch_id=batch_id,
        smiles=smiles
    )
    return engine.generate_filename(context, extension) 