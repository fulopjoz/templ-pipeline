"""
TEMPL Pipeline FAIR Module

This module implements FAIR (Findable, Accessible, Interoperable, Reusable)
principles for scientific data management in the TEMPL pipeline.

Submodules:
- core: Core FAIR infrastructure (metadata engine, identifiers, validation)
- biology: Biology-specific metadata and descriptors
- outputs: Output formatting and validation
- integration: Integration with existing pipeline components
- utils: FAIR utilities and helper functions
"""

from .core.metadata_engine import MetadataEngine, create_metadata_engine

__all__ = ["MetadataEngine", "create_metadata_engine"]
