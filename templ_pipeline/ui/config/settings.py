# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
TEMPL Pipeline Configuration Settings

Centralized configuration management for the TEMPL Pipeline UI.
All application settings, defaults, and configuration options are defined here.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

# Import centralized version
try:
    from templ_pipeline import __version__
except ImportError:
    __version__ = "1.0.0"  # Fallback
import logging

logger = logging.getLogger(__name__)


class AppConfig:
    """Centralized application configuration"""

    def __init__(self):
        """Initialize application configuration"""
        # Application metadata
        self.app_name = "TEMPL Pipeline"
        self.app_version = __version__
        self.app_description = "TEMplate-based Protein-Ligand Pose Prediction"

        # Page configuration for Streamlit
        self.page_config = {
            "page_title": "TEMPL Pipeline",
            "page_icon": "◈",
            "layout": "wide",
            "initial_sidebar_state": "auto",
        }

        # Resource limits
        self.resource_limits = {
            "max_file_size_mb": 10,
            "max_molecule_atoms": 200,
            "min_molecule_atoms": 3,
            "cache_size_mb": 100,
            "max_templates": 500,
            "default_templates": 100,
            "memory_limit_mb": 1024,
            "cleanup_threshold": 0.8,
        }

        # Performance settings
        self.performance = {
            "cache_ttl": 3600,  # 1 hour
            "max_cache_entries": 100,
            "enable_progress_tracking": True,
            "enable_async_execution": True,
            "max_workers": self._get_max_workers(),
            "enable_memory_optimization": True,
        }

        # UI/UX settings
        self.ui_settings = {
            "show_advanced_options": True,
            "enable_fair_metadata": True,
            "enable_tooltips": True,
            "default_pose_alignment": "aligned",  # "aligned" or "original"
            "progressive_disclosure": True,
            "show_technical_details": False,
        }

        # Scientific settings - TEMPL Normalized TanimotoCombo (PMC9059856 implementation)
        #
        # TEMPL implements the standard ROCS TanimotoCombo methodology but with normalization:
        # combo_score = 0.5 * (ShapeTanimoto + ColorTanimoto) = TanimotoCombo / 2
        #
        # Benefits of normalization:
        # 1. 0-1 scale for easier user interpretation
        # 2. More conservative quality thresholds than PMC article (better pose discrimination)
        # 3. Maintains scientific rigor while improving usability
        self.scientific = {
            "confidence_level": 0.95,
            "min_combo_score": 0.45,  # Pose prediction threshold aligned with SCORE_FAIR
            "quality_thresholds": {
                "excellent": 0.80,  # Top-tier pose prediction performance (RMSD ≤ 1.0 Å)
                "good": 0.65,  # High-quality poses meeting RMSD ≤ 2.0 Å success criterion
                "moderate": 0.45,  # Fair quality poses (RMSD 2.0-3.0 Å range)
                "poor": 0.0,
            },
            "normalization_info": {
                "method": "TanimotoCombo normalization",
                "scale": "0-1 (normalized from 0-2 standard ROCS scale)",
                "formula": "combo_score = 0.5 * (ShapeTanimoto + ColorTanimoto)",
                "literature_basis": "PMC9059856 methodology with conservative thresholds",
            },
            "enable_uncertainty_estimates": True,
            "enable_statistical_analysis": True,
        }

        # File paths
        self.paths = self._setup_paths()

        # Logging configuration
        self.logging = {
            "level": logging.INFO,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "enable_performance_logging": True,
        }

        # Feature flags
        self.features = {
            "embedding_similarity": self._check_embedding_features(),
            "fair_metadata": self._check_fair_availability(),
            "optimization_modules": self._check_optimization_modules(),
            "hardware_detection": True,
            "async_pipeline": True,
        }

    def _get_max_workers(self) -> int:
        """Determine optimal number of workers based on CPU count"""
        try:
            import multiprocessing

            cpu_count = multiprocessing.cpu_count()
            # Use 75% of available CPUs, minimum 2, maximum 8
            return max(2, min(8, int(cpu_count * 0.75)))
        except:
            return 4  # Safe default

    def _setup_paths(self) -> Dict[str, Path]:
        """Setup application paths"""
        # Get base paths
        ui_dir = Path(__file__).parent.parent
        project_root = ui_dir.parent.parent

        return {
            "ui_dir": ui_dir,
            "project_root": project_root,
            "data_dir": project_root / "data",
            "temp_dir": project_root / "temp",
            "embeddings_path": project_root
            / "data"
            / "embeddings"
            / "templ_protein_embeddings_v1.0.0.npz",
            "templates_path": project_root / "data" / "templates",
        }

    def _check_embedding_features(self) -> bool:
        """Check if embedding features are available"""
        try:
            import torch
            import transformers

            return True
        except ImportError:
            return False

    def _check_fair_availability(self) -> bool:
        """Check if FAIR metadata features are available"""
        try:
            from templ_pipeline.fair.core.metadata_engine import MetadataEngine

            return True
        except ImportError:
            return False

    def _check_optimization_modules(self) -> bool:
        """Check if optimization modules are available"""
        try:
            from templ_pipeline.ui.core.error_handling import ContextualErrorManager
            from templ_pipeline.ui.core.memory_manager import MolecularSessionManager
            from templ_pipeline.ui.core.molecular_processor import (
                CachedMolecularProcessor,
            )
            from templ_pipeline.ui.core.secure_upload import SecureFileUploadHandler

            return True
        except ImportError:
            return False

    def get_setting(self, category: str, key: str, default: Any = None) -> Any:
        """Get a specific setting value

        Args:
            category: Setting category (e.g., 'resource_limits', 'performance')
            key: Setting key
            default: Default value if not found

        Returns:
            Setting value or default
        """
        category_dict = getattr(self, category, {})
        if isinstance(category_dict, dict):
            return category_dict.get(key, default)
        return default

    def update_setting(self, category: str, key: str, value: Any) -> None:
        """Update a specific setting

        Args:
            category: Setting category
            key: Setting key
            value: New value
        """
        if hasattr(self, category):
            category_dict = getattr(self, category)
            if isinstance(category_dict, dict):
                category_dict[key] = value
                logger.info(f"Updated setting: {category}.{key} = {value}")

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings as a dictionary

        Returns:
            Dictionary of all settings
        """
        return {
            "app_metadata": {
                "name": self.app_name,
                "version": self.app_version,
                "description": self.app_description,
            },
            "page_config": self.page_config,
            "resource_limits": self.resource_limits,
            "performance": self.performance,
            "ui_settings": self.ui_settings,
            "scientific": self.scientific,
            "paths": {k: str(v) for k, v in self.paths.items()},
            "features": self.features,
        }

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration and return status

        Returns:
            Dictionary with validation results
        """
        issues = []
        warnings = []

        # Check paths exist
        for name, path in self.paths.items():
            if "embeddings" in name or "templates" in name:
                if not path.exists():
                    warnings.append(f"Path not found: {name} ({path})")

        # Check resource limits
        if self.resource_limits["max_file_size_mb"] > 50:
            warnings.append("Large file size limit may cause memory issues")

        # Check performance settings
        if self.performance["max_workers"] > 16:
            warnings.append("High worker count may not improve performance")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "features_available": self.features,
        }


# Create a singleton instance
_app_config = None


def get_config() -> AppConfig:
    """Get the global application configuration instance

    Returns:
        AppConfig instance
    """
    global _app_config
    if _app_config is None:
        _app_config = AppConfig()
    return _app_config
