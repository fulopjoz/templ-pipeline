"""
Comprehensive file management for TEMPL pipeline.

This module provides unified file management capabilities including:
- Adaptive file naming following FAIR principles
- File tracking and metadata management
- Output organization and archiving
- Integration with workspace management
- Backward compatibility with existing pipeline components

Key Features:
- Context-aware file naming based on prediction scenarios
- Comprehensive file tracking and metadata
- Support for multiple output formats
- Integration with workspace structure
- Automated cleanup and archiving
"""

import os
import json
import hashlib
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict

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
    run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility."""
        return asdict(self)


@dataclass
class FileMetadata:
    """Metadata about a managed file."""
    
    path: str
    filename: str
    size_bytes: int
    created_time: float
    modified_time: float
    file_type: str  # 'poses', 'metadata', 'log', 'archive', 'temp'
    category: str   # 'result', 'intermediate', 'analysis', 'backup'
    context: Optional[Dict[str, Any]] = None
    checksum: Optional[str] = None
    
    @property
    def age_hours(self) -> float:
        """Get file age in hours."""
        return (datetime.now().timestamp() - self.created_time) / 3600.0
    
    @property
    def size_mb(self) -> float:
        """Get file size in MB."""
        return self.size_bytes / (1024 * 1024)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


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

    def generate_filename(
        self,
        context: PredictionContext,
        file_type: str = "poses",
        extension: str = "sdf",
        custom_timestamp: Optional[str] = None,
    ) -> str:
        """
        Generate context-appropriate filename.

        Args:
            context: Prediction context information
            file_type: Type of file ('poses', 'metadata', 'log', 'archive')
            extension: File extension
            custom_timestamp: Optional custom timestamp (for testing/reproducibility)

        Returns:
            Generated filename following FAIR naming conventions
        """
        # Generate timestamp
        timestamp = custom_timestamp or datetime.now().strftime(self.timestamp_format)

        # Determine base name based on context
        if context.pdb_id:
            # PDB-based prediction
            pdb_id = context.pdb_id.lower()
            base_name = f"{pdb_id}_{timestamp}"
            logger.debug(f"PDB-based naming: {base_name}")

        elif context.template_source == "sdf":
            # Template-based prediction from SDF file
            base_name = f"templ_{timestamp}"
            logger.debug(f"Template-based naming: {base_name}")

        elif context.batch_id:
            # Batch processing
            batch_id = context.batch_id
            base_name = f"batch_{batch_id}_{timestamp}"
            logger.debug(f"Batch-based naming: {base_name}")

        elif context.custom_prefix:
            # Custom prefix provided
            prefix = context.custom_prefix
            base_name = f"{prefix}_{timestamp}"
            logger.debug(f"Custom prefix naming: {base_name}")

        else:
            # Custom ligand or unknown source - generate hash
            input_hash = self._generate_input_hash(context)
            base_name = f"custom_{input_hash}_{timestamp}"
            logger.debug(f"Custom hash naming: {base_name}")

        # Add file type suffix
        if file_type == "poses":
            filename = f"{base_name}_poses.{extension}"
        elif file_type == "metadata":
            filename = f"{base_name}_metadata.{extension}"
        elif file_type == "log":
            filename = f"{base_name}_log.{extension}"
        elif file_type == "archive":
            filename = f"{base_name}_complete.{extension}"
        elif file_type == "analysis":
            filename = f"{base_name}_analysis.{extension}"
        else:
            filename = f"{base_name}_{file_type}.{extension}"

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

    def generate_related_filename(
        self, 
        base_filename: str, 
        file_type: str,
        extension: Optional[str] = None
    ) -> str:
        """
        Generate related filename from base filename.

        Args:
            base_filename: Base filename (e.g., poses file)
            file_type: Type of related file ('metadata', 'log', 'archive')
            extension: Optional extension override

        Returns:
            Related filename
        """
        base_path = Path(base_filename)
        name_parts = base_path.stem.split('_')
        
        # Remove existing type suffix if present
        if name_parts[-1] in ['poses', 'metadata', 'log', 'archive', 'analysis']:
            name_parts = name_parts[:-1]
        
        # Add new type suffix
        new_name = '_'.join(name_parts + [file_type])
        
        # Determine extension
        if extension is None:
            if file_type == 'metadata':
                extension = 'json'
            elif file_type == 'log':
                extension = 'log'
            elif file_type == 'archive':
                extension = 'zip'
            else:
                extension = base_path.suffix[1:]  # Remove dot
        
        return f"{new_name}.{extension}"


class FileManager:
    """
    Comprehensive file management class for TEMPL pipeline.

    Provides unified interface for file operations including:
    - Adaptive file naming
    - File tracking and metadata
    - Output organization and archiving
    - Integration with workspace management
    """

    def __init__(
        self, 
        output_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        workspace_manager: Optional[Any] = None
    ):
        """
        Initialize the file manager.

        Args:
            output_dir: Base directory for outputs (optional if workspace_manager provided)
            run_id: Optional run identifier for organization
            workspace_manager: Optional workspace manager for integration
        """
        self.naming_engine = AdaptiveFileNamingEngine()
        self.workspace_manager = workspace_manager
        self.run_id = run_id
        
        # Setup output directory
        if workspace_manager:
            self.output_dir = workspace_manager.output_dir
        elif output_dir:
            if run_id:
                self.output_dir = Path(f"{output_dir}_{run_id}")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.output_dir = Path(f"{output_dir}_{timestamp}")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Default behavior
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"output_{timestamp}")
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # File tracking
        self.managed_files: Dict[str, FileMetadata] = {}
        self.file_history: List[Dict[str, Any]] = []

        logger.info(f"File manager initialized: {self.output_dir}")

    def generate_filename(
        self,
        context: PredictionContext,
        file_type: str = "poses",
        extension: str = "sdf",
        relative: bool = False
    ) -> str:
        """
        Generate filename with full path.

        Args:
            context: Prediction context for naming
            file_type: Type of output file ('poses', 'metadata', 'log', 'archive')
            extension: File extension
            relative: If True, return relative path; if False, return absolute path

        Returns:
            Generated filename path
        """
        # Override extension for certain file types
        if file_type == "metadata":
            extension = "json"
        elif file_type == "log":
            extension = "log"
        elif file_type == "archive":
            extension = "zip"

        filename = self.naming_engine.generate_filename(
            context, file_type, extension
        )
        
        if relative:
            return filename
        else:
            full_path = self.output_dir / filename
            return str(full_path)

    def save_file(
        self,
        content: Union[str, bytes, Any],
        context: PredictionContext,
        file_type: str = "poses",
        extension: str = "sdf",
        metadata: Optional[Dict[str, Any]] = None,
        calculate_checksum: bool = True
    ) -> str:
        """
        Save content to file with tracking.

        Args:
            content: Content to save
            context: Prediction context
            file_type: Type of file
            extension: File extension
            metadata: Optional metadata to associate with file
            calculate_checksum: Whether to calculate file checksum

        Returns:
            Path to saved file
        """
        output_file = self.generate_filename(context, file_type, extension)
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Save content based on type
        if isinstance(content, str):
            Path(output_file).write_text(content, encoding='utf-8')
        elif isinstance(content, bytes):
            Path(output_file).write_bytes(content)
        else:
            # Try to serialize as JSON
            Path(output_file).write_text(json.dumps(content, indent=2), encoding='utf-8')
        
        # Calculate checksum if requested
        checksum = None
        if calculate_checksum:
            checksum = self._calculate_file_checksum(output_file)
        
        # Track the file
        self._register_file(
            output_file, 
            file_type, 
            context,
            metadata,
            checksum
        )
        
        logger.info(f"Saved {file_type} file: {output_file}")
        return output_file

    def save_poses(
        self,
        poses: Dict[str, Tuple[Any, Dict[str, float]]],
        context: PredictionContext,
        template_info: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> Tuple[str, Optional[str]]:
        """
        Save poses to SDF file using adaptive naming.

        Args:
            poses: Dictionary of poses from pipeline
            context: Prediction context for file naming
            template_info: Optional template metadata
            include_metadata: Whether to save metadata file

        Returns:
            Tuple of (poses_file_path, metadata_file_path)
        """
        from rdkit import Chem
        
        # Import scoring function with fallback
        try:
            from .scoring import generate_properties_for_sdf
        except ImportError:
            # Fallback for basic property generation
            def generate_properties_for_sdf(mol, method, score, template_pdb, props):
                for prop_name, prop_value in props.items():
                    mol.SetProp(prop_name, str(prop_value))
                mol.SetProp("method", method)
                mol.SetProp("score", str(score))
                mol.SetProp("template_pdb", template_pdb)
                return mol

        output_file = self.generate_filename(context, "poses", "sdf")
        
        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # Save poses
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
                        "prediction_context": str(
                            context.template_source or "pdb_based"
                        ),
                        "run_id": context.run_id or "unknown",
                    },
                )

                writer.write(pose_with_props)

        # Track the poses file
        self._register_file(
            output_file,
            "poses",
            context,
            {
                "pose_count": len(poses),
                "methods": list(poses.keys()),
                "template_info": template_info,
            }
        )

        metadata_file = None
        if include_metadata:
            # Save metadata file
            metadata = {
                "prediction_context": context.to_dict(),
                "poses_file": Path(output_file).name,
                "pose_count": len(poses),
                "methods": list(poses.keys()),
                "template_info": template_info,
                "creation_time": datetime.now().isoformat(),
                "run_id": context.run_id or "unknown",
            }
            
            metadata_file = self.save_file(
                metadata,
                context,
                "metadata",
                "json",
                {"related_file": output_file}
            )

        logger.info(f"Successfully saved poses to {output_file}")
        return output_file, metadata_file

    def save_analysis(
        self,
        analysis_data: Dict[str, Any],
        context: PredictionContext,
        analysis_type: str = "general"
    ) -> str:
        """
        Save analysis results to file.

        Args:
            analysis_data: Analysis data to save
            context: Prediction context
            analysis_type: Type of analysis

        Returns:
            Path to saved analysis file
        """
        # Add analysis metadata
        analysis_with_metadata = {
            "analysis_type": analysis_type,
            "creation_time": datetime.now().isoformat(),
            "context": context.to_dict(),
            "data": analysis_data,
        }
        
        return self.save_file(
            analysis_with_metadata,
            context,
            "analysis",
            "json",
            {"analysis_type": analysis_type}
        )

    def create_archive(
        self,
        context: PredictionContext,
        include_temp: bool = False,
        file_types: Optional[List[str]] = None
    ) -> str:
        """
        Create archive of related files.

        Args:
            context: Prediction context
            include_temp: Whether to include temporary files
            file_types: List of file types to include (None for all)

        Returns:
            Path to created archive
        """
        archive_file = self.generate_filename(context, "archive", "zip")
        
        # Find related files
        related_files = self._find_related_files(context, file_types)
        
        with zipfile.ZipFile(archive_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in related_files:
                if Path(file_path).exists():
                    arcname = Path(file_path).name
                    zipf.write(file_path, arcname)
        
        # Track the archive
        self._register_file(
            archive_file,
            "archive",
            context,
            {
                "archived_files": len(related_files),
                "file_list": [Path(f).name for f in related_files],
            }
        )
        
        logger.info(f"Created archive: {archive_file}")
        return archive_file

    def _find_related_files(
        self, 
        context: PredictionContext, 
        file_types: Optional[List[str]] = None
    ) -> List[str]:
        """Find files related to a prediction context."""
        related_files = []
        
        # Generate expected filenames for this context
        if file_types is None:
            file_types = ["poses", "metadata", "log", "analysis"]
        
        for file_type in file_types:
            filename = self.generate_filename(context, file_type, "sdf")
            if Path(filename).exists():
                related_files.append(filename)
        
        return related_files

    def _register_file(
        self,
        file_path: str,
        file_type: str,
        context: PredictionContext,
        metadata: Optional[Dict[str, Any]] = None,
        checksum: Optional[str] = None
    ):
        """Register a file for tracking."""
        try:
            stat_info = Path(file_path).stat()
            
            file_metadata = FileMetadata(
                path=file_path,
                filename=Path(file_path).name,
                size_bytes=stat_info.st_size,
                created_time=stat_info.st_ctime,
                modified_time=stat_info.st_mtime,
                file_type=file_type,
                category=self._determine_category(file_type),
                context=context.to_dict(),
                checksum=checksum
            )
            
            self.managed_files[file_path] = file_metadata
            
            # Record in history
            self.file_history.append({
                "action": "created",
                "file_path": file_path,
                "file_type": file_type,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata,
            })
            
        except Exception as e:
            logger.warning(f"Could not register file {file_path}: {e}")

    def _determine_category(self, file_type: str) -> str:
        """Determine file category based on type."""
        if file_type in ["poses", "metadata"]:
            return "result"
        elif file_type in ["log"]:
            return "intermediate"
        elif file_type in ["analysis"]:
            return "analysis"
        elif file_type in ["archive"]:
            return "backup"
        else:
            return "unknown"

    def _calculate_file_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum of file."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate checksum for {file_path}: {e}")
            return None

    def get_file_summary(self) -> Dict[str, Any]:
        """Get summary of managed files."""
        summary = {
            "total_files": len(self.managed_files),
            "total_size_mb": sum(f.size_mb for f in self.managed_files.values()),
            "file_types": {},
            "categories": {},
            "recent_files": [],
        }
        
        # Count by type and category
        for file_metadata in self.managed_files.values():
            # By type
            if file_metadata.file_type not in summary["file_types"]:
                summary["file_types"][file_metadata.file_type] = 0
            summary["file_types"][file_metadata.file_type] += 1
            
            # By category
            if file_metadata.category not in summary["categories"]:
                summary["categories"][file_metadata.category] = 0
            summary["categories"][file_metadata.category] += 1
        
        # Recent files (last 5)
        recent_files = sorted(
            self.managed_files.values(),
            key=lambda f: f.created_time,
            reverse=True
        )[:5]
        
        summary["recent_files"] = [
            {
                "filename": f.filename,
                "file_type": f.file_type,
                "size_mb": f.size_mb,
                "age_hours": f.age_hours,
            }
            for f in recent_files
        ]
        
        return summary

    def create_prediction_context(
        self,
        pdb_id: Optional[str] = None,
        template_source: Optional[str] = None,
        ligand_smiles: Optional[str] = None,
        ligand_file: Optional[str] = None,
        batch_id: Optional[str] = None,
        custom_prefix: Optional[str] = None,
    ) -> PredictionContext:
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
            custom_prefix=custom_prefix,
            run_id=self.run_id,
        )


# Backward compatibility classes and functions

class OutputManager(FileManager):
    """Backward compatibility wrapper for OutputManager."""
    
    def __init__(self, output_dir: str = "output", run_id: Optional[str] = None):
        super().__init__(output_dir=output_dir, run_id=run_id)
    
    def generate_output_filename(
        self,
        context: PredictionContext,
        file_type: str = "poses",
        extension: str = "sdf",
    ) -> str:
        """Backward compatibility method."""
        return self.generate_filename(context, file_type, extension)
    
    def get_output_summary(self) -> Dict[str, Any]:
        """Backward compatibility method."""
        return self.get_file_summary()


# Convenience functions
def create_file_manager(
    output_dir: Optional[str] = None,
    run_id: Optional[str] = None,
    workspace_manager: Optional[Any] = None
) -> FileManager:
    """Create a file manager instance."""
    return FileManager(output_dir, run_id, workspace_manager)


def create_output_manager(
    output_dir: str = "output", 
    run_id: Optional[str] = None
) -> OutputManager:
    """Create an output manager instance for backward compatibility."""
    return OutputManager(output_dir, run_id)


def generate_adaptive_filename(
    pdb_id: Optional[str] = None,
    template_source: Optional[str] = None,
    batch_id: Optional[str] = None,
    smiles: Optional[str] = None,
    extension: str = "sdf",
) -> str:
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
    return engine.generate_filename(context, "poses", extension)