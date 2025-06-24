"""
TEMPL Pipeline FAIR Metadata Engine

This module implements comprehensive metadata generation following FAIR principles
for computational biology workflows. It provides structured metadata capture,
validation, and export functionality for scientific reproducibility.

Key Features:
- Domain-specific metadata for computational biology
- Comprehensive provenance tracking
- Publication-ready output formats
- Integration with existing TEMPL pipeline components
"""

import os
import json
import platform
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from uuid import uuid4

logger = logging.getLogger(__name__)

@dataclass
class ComputationalMetadata:
    """Computational environment and execution metadata."""
    python_version: str
    platform: str
    cpu_count: int
    memory_gb: float
    gpu_available: bool
    gpu_info: Optional[str]
    execution_time: Optional[float]
    start_time: str
    end_time: Optional[str]
    pipeline_version: str
    dependencies: Dict[str, str]

@dataclass
class InputMetadata:
    """Input data and parameters metadata."""
    target_identifier: Optional[str]  # PDB ID or custom identifier
    input_type: str  # 'pdb_id', 'smiles', 'sdf_file', 'custom'
    input_value: str  # The actual input (SMILES, file path, etc.)
    input_file_hash: Optional[str]
    template_source: str  # 'database', 'sdf', 'custom'
    template_count: int
    template_identifiers: List[str]
    parameters: Dict[str, Any]
    
@dataclass
class OutputMetadata:
    """Output files and results metadata."""
    primary_output: str  # Main output file path
    output_format: str  # 'sdf', 'pdb', etc.
    file_size_bytes: int
    file_hash: str
    poses_generated: int
    best_scores: Dict[str, float]
    additional_outputs: List[str]

@dataclass
class ScientificMetadata:
    """Scientific context and methodology metadata."""
    methodology: str  # 'template_based_pose_prediction'
    algorithm_description: str
    validation_metrics: List[str]
    quality_indicators: Dict[str, Any]
    biological_context: Dict[str, Any]
    citation_info: Dict[str, str]

@dataclass
class ProvenanceRecord:
    """Complete provenance record for a TEMPL prediction."""
    unique_id: str
    timestamp: str
    computational: ComputationalMetadata
    input: InputMetadata
    output: OutputMetadata
    scientific: ScientificMetadata
    fair_compliance: Dict[str, bool]

class MetadataEngine:
    """
    Core metadata engine for FAIR-compliant scientific data management.
    
    This engine captures, validates, and exports comprehensive metadata
    for TEMPL pipeline executions to ensure full reproducibility and
    scientific rigor.
    """
    
    def __init__(self, pipeline_version: str = "2.0.0"):
        """
        Initialize the metadata engine.
        
        Args:
            pipeline_version: Version of the TEMPL pipeline
        """
        self.pipeline_version = pipeline_version
        self.execution_start = datetime.now(timezone.utc)
        self.computational_env = self._capture_computational_environment()
        
    def _capture_computational_environment(self) -> ComputationalMetadata:
        """Capture computational environment information."""
        import sys
        
        # Get memory information
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            memory_gb = 0.0
        
        # Get GPU information
        gpu_available = False
        gpu_info = None
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_info = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        
        # Get dependencies
        dependencies = self._get_key_dependencies()
        
        return ComputationalMetadata(
            python_version=sys.version,
            platform=platform.platform(),
            cpu_count=os.cpu_count() or 1,
            memory_gb=memory_gb,
            gpu_available=gpu_available,
            gpu_info=gpu_info,
            execution_time=None,  # Will be set at completion
            start_time=self.execution_start.isoformat(),
            end_time=None,  # Will be set at completion
            pipeline_version=self.pipeline_version,
            dependencies=dependencies
        )
    
    def _get_key_dependencies(self) -> Dict[str, str]:
        """Get versions of key dependencies."""
        dependencies = {}
        
        key_packages = [
            'rdkit', 'numpy', 'scipy', 'sklearn', 'biotite', 
            'transformers', 'torch', 'spyrmsd'
        ]
        
        for package in key_packages:
            try:
                if package == 'rdkit':
                    from rdkit import rdBase
                    dependencies[package] = rdBase.rdkitVersion
                elif package == 'numpy':
                    import numpy as np
                    dependencies[package] = np.__version__
                elif package == 'scipy':
                    import scipy
                    dependencies[package] = scipy.__version__
                elif package == 'sklearn':
                    import sklearn
                    dependencies[package] = sklearn.__version__
                elif package == 'biotite':
                    import biotite
                    dependencies[package] = biotite.__version__
                elif package == 'transformers':
                    import transformers
                    dependencies[package] = transformers.__version__
                elif package == 'torch':
                    import torch
                    dependencies[package] = torch.__version__
                elif package == 'spyrmsd':
                    import spyrmsd
                    dependencies[package] = spyrmsd.__version__
            except ImportError:
                dependencies[package] = "not_installed"
            except AttributeError:
                dependencies[package] = "version_unknown"
        
        return dependencies
    
    def create_input_metadata(self,
                            target_identifier: Optional[str] = None,
                            input_type: str = "unknown",
                            input_value: str = "",
                            input_file: Optional[str] = None,
                            template_source: str = "database",
                            template_identifiers: List[str] = None,
                            parameters: Dict[str, Any] = None) -> InputMetadata:
        """
        Create comprehensive input metadata.
        
        Args:
            target_identifier: PDB ID or custom identifier
            input_type: Type of input ('pdb_id', 'smiles', 'sdf_file', 'custom')
            input_value: The actual input value
            input_file: Path to input file (if applicable)
            template_source: Source of templates
            template_identifiers: List of template IDs used
            parameters: Pipeline parameters
            
        Returns:
            InputMetadata object
        """
        # Calculate file hash if input file provided
        input_file_hash = None
        if input_file and os.path.exists(input_file):
            input_file_hash = self._calculate_file_hash(input_file)
        
        return InputMetadata(
            target_identifier=target_identifier,
            input_type=input_type,
            input_value=input_value,
            input_file_hash=input_file_hash,
            template_source=template_source,
            template_count=len(template_identifiers) if template_identifiers else 0,
            template_identifiers=template_identifiers or [],
            parameters=parameters or {}
        )
    
    def create_output_metadata(self,
                             primary_output: str,
                             output_format: str = "sdf",
                             poses_generated: int = 0,
                             best_scores: Dict[str, float] = None,
                             additional_outputs: List[str] = None) -> OutputMetadata:
        """
        Create comprehensive output metadata.
        
        Args:
            primary_output: Path to primary output file
            output_format: Format of primary output
            poses_generated: Number of poses generated
            best_scores: Best scores achieved
            additional_outputs: List of additional output files
            
        Returns:
            OutputMetadata object
        """
        # Calculate file size and hash
        file_size_bytes = 0
        file_hash = ""
        
        if os.path.exists(primary_output):
            file_size_bytes = os.path.getsize(primary_output)
            file_hash = self._calculate_file_hash(primary_output)
        
        return OutputMetadata(
            primary_output=primary_output,
            output_format=output_format,
            file_size_bytes=file_size_bytes,
            file_hash=file_hash,
            poses_generated=poses_generated,
            best_scores=best_scores or {},
            additional_outputs=additional_outputs or []
        )
    
    def create_scientific_metadata(self,
                                 methodology: str = "template_based_pose_prediction",
                                 validation_metrics: List[str] = None,
                                 quality_indicators: Dict[str, Any] = None,
                                 biological_context: Dict[str, Any] = None) -> ScientificMetadata:
        """
        Create scientific methodology and context metadata.
        
        Args:
            methodology: Description of methodology used
            validation_metrics: List of validation metrics applied
            quality_indicators: Quality assessment results
            biological_context: Biological context information
            
        Returns:
            ScientificMetadata object
        """
        algorithm_description = (
            "Template-based pose prediction using maximum common substructure (MCS) "
            "identification, constrained conformer generation, and shape-based scoring. "
            "Templates are selected based on protein sequence similarity using ESM2 embeddings."
        )
        
        citation_info = {
            "method": "TEMPL: Template-based Protein Ligand Pose Prediction",
            "authors": "Fülöp et al.",
            "description": "A template-based protein ligand pose prediction baseline",
            "version": self.pipeline_version
        }
        
        return ScientificMetadata(
            methodology=methodology,
            algorithm_description=algorithm_description,
            validation_metrics=validation_metrics or ["shape_tanimoto", "color_tanimoto", "combo_score"],
            quality_indicators=quality_indicators or {},
            biological_context=biological_context or {},
            citation_info=citation_info
        )
    
    def create_provenance_record(self,
                               input_metadata: InputMetadata,
                               output_metadata: OutputMetadata,
                               scientific_metadata: ScientificMetadata) -> ProvenanceRecord:
        """
        Create complete provenance record.
        
        Args:
            input_metadata: Input metadata
            output_metadata: Output metadata
            scientific_metadata: Scientific metadata
            
        Returns:
            Complete ProvenanceRecord
        """
        # Finalize computational metadata
        execution_end = datetime.now(timezone.utc)
        execution_time = (execution_end - self.execution_start).total_seconds()
        
        self.computational_env.execution_time = execution_time
        self.computational_env.end_time = execution_end.isoformat()
        
        # Assess FAIR compliance
        fair_compliance = self._assess_fair_compliance(
            input_metadata, output_metadata, scientific_metadata
        )
        
        return ProvenanceRecord(
            unique_id=str(uuid4()),
            timestamp=execution_end.isoformat(),
            computational=self.computational_env,
            input=input_metadata,
            output=output_metadata,
            scientific=scientific_metadata,
            fair_compliance=fair_compliance
        )
    
    def _assess_fair_compliance(self,
                              input_metadata: InputMetadata,
                              output_metadata: OutputMetadata,
                              scientific_metadata: ScientificMetadata) -> Dict[str, bool]:
        """Assess FAIR compliance of the workflow."""
        return {
            "findable": bool(input_metadata.target_identifier and output_metadata.file_hash),
            "accessible": bool(output_metadata.primary_output and os.path.exists(output_metadata.primary_output)),
            "interoperable": bool(output_metadata.output_format in ["sdf", "pdb"]),
            "reusable": bool(
                scientific_metadata.methodology and 
                input_metadata.parameters and 
                self.computational_env.dependencies
            )
        }
    
    def export_metadata(self,
                       provenance_record: ProvenanceRecord,
                       output_path: str,
                       format: str = "json") -> str:
        """
        Export metadata in specified format.
        
        Args:
            provenance_record: Complete provenance record
            output_path: Path for metadata export
            format: Export format ('json', 'yaml', 'xml')
            
        Returns:
            Path to exported metadata file
        """
        if format == "json":
            return self._export_json(provenance_record, output_path)
        elif format == "yaml":
            return self._export_yaml(provenance_record, output_path)
        elif format == "xml":
            return self._export_xml(provenance_record, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self, record: ProvenanceRecord, output_path: str) -> str:
        """Export metadata as JSON."""
        metadata_dict = asdict(record)
        
        json_path = Path(output_path).with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadata exported to JSON: {json_path}")
        return str(json_path)
    
    def _export_yaml(self, record: ProvenanceRecord, output_path: str) -> str:
        """Export metadata as YAML."""
        try:
            import yaml
            metadata_dict = asdict(record)
            
            yaml_path = Path(output_path).with_suffix('.yaml')
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(metadata_dict, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"Metadata exported to YAML: {yaml_path}")
            return str(yaml_path)
        except ImportError:
            logger.warning("PyYAML not available, falling back to JSON")
            return self._export_json(record, output_path)
    
    def _export_xml(self, record: ProvenanceRecord, output_path: str) -> str:
        """Export metadata as XML."""
        # Simplified XML export - could be enhanced with proper XML libraries
        metadata_dict = asdict(record)
        
        xml_content = self._dict_to_xml(metadata_dict, "templ_metadata")
        
        xml_path = Path(output_path).with_suffix('.xml')
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        logger.info(f"Metadata exported to XML: {xml_path}")
        return str(xml_path)
    
    def _dict_to_xml(self, data: Dict, root_name: str) -> str:
        """Convert dictionary to simple XML format."""
        def _to_xml(obj, name):
            if isinstance(obj, dict):
                xml = f"<{name}>\n"
                for key, value in obj.items():
                    xml += _to_xml(value, key)
                xml += f"</{name}>\n"
                return xml
            elif isinstance(obj, list):
                xml = f"<{name}>\n"
                for item in obj:
                    xml += _to_xml(item, "item")
                xml += f"</{name}>\n"
                return xml
            else:
                return f"<{name}>{str(obj)}</{name}>\n"
        
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{_to_xml(data, root_name)}'
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate hash for {file_path}: {e}")
            return ""

# Convenience functions for easy integration
def create_metadata_engine(pipeline_version: str = "2.0.0") -> MetadataEngine:
    """Create a metadata engine instance."""
    return MetadataEngine(pipeline_version)

def generate_quick_metadata(target_id: Optional[str],
                          input_type: str,
                          input_value: str,
                          output_file: str,
                          poses_count: int = 0,
                          scores: Dict[str, float] = None) -> Dict[str, Any]:
    """Quick metadata generation for simple workflows."""
    engine = MetadataEngine()
    
    input_meta = engine.create_input_metadata(
        target_identifier=target_id,
        input_type=input_type,
        input_value=input_value
    )
    
    output_meta = engine.create_output_metadata(
        primary_output=output_file,
        poses_generated=poses_count,
        best_scores=scores or {}
    )
    
    scientific_meta = engine.create_scientific_metadata()
    
    record = engine.create_provenance_record(input_meta, output_meta, scientific_meta)
    
    return asdict(record) 