"""
Protein Similarity Service for TEMPL Pipeline

Handles protein similarity search functionality.
For custom templates (pure MCS), this service is not used.
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional, List

from ..config.settings import AppConfig
from ..core.session_manager import SessionManager

logger = logging.getLogger(__name__)


class ProteinSimilarityService:
    """Service for protein similarity search functionality"""

    def __init__(self, config: AppConfig, session: SessionManager):
        """Initialize protein similarity service

        Args:
            config: Application configuration
            session: Session manager instance
        """
        self.config = config
        self.session = session

    def search_similar_proteins(self, pdb_id: str) -> Dict[str, Any]:
        """Search for similar proteins by PDB ID

        Args:
            pdb_id: PDB ID to search for

        Returns:
            Dictionary with similarity results
        """
        logger.info(
            f"Protein similarity search not applicable for custom templates workflow"
        )
        return {
            "found": False,
            "count": 0,
            "results": [],
            "message": "Not applicable for custom templates",
        }

    def is_available(self) -> bool:
        """Check if protein similarity service is available

        Returns:
            False for custom templates workflow
        """
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get service status

        Returns:
            Dictionary with service status
        """
        return {
            "available": False,
            "reason": "Custom templates mode - protein similarity not needed",
        }
