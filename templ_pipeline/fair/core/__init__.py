"""
TEMPL Pipeline FAIR Core Module

Core FAIR infrastructure for metadata management, identifier generation,
and validation framework.
"""

from .metadata_engine import (
    MetadataEngine,
    create_metadata_engine,
    generate_quick_metadata,
    ComputationalMetadata,
    InputMetadata,
    OutputMetadata,
    ScientificMetadata,
    ProvenanceRecord,
)

__all__ = [
    "MetadataEngine",
    "create_metadata_engine",
    "generate_quick_metadata",
    "ComputationalMetadata",
    "InputMetadata",
    "OutputMetadata",
    "ScientificMetadata",
    "ProvenanceRecord",
]

# This will be imported once metadata_engine.py is created
