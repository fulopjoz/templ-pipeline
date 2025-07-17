"""
Session Manager for TEMPL Pipeline

Centralized session state management with type safety, validation,
and automatic cleanup capabilities.
"""

import streamlit as st
import logging
from typing import Any, Dict, List, Optional, Set, Callable
from datetime import datetime
from pathlib import Path

from ..config.constants import SESSION_KEYS
from .memory_manager import get_memory_manager

logger = logging.getLogger(__name__)


class SessionManager:
    """Centralized session state manager with enhanced features"""

    def __init__(self, config: Optional[Any] = None):
        """Initialize session manager

        Args:
            config: Optional AppConfig instance
        """
        self.config = config
        self.memory_manager = get_memory_manager()

        # Define default values for all session keys
        self.defaults = {
            SESSION_KEYS["APP_INITIALIZED"]: False,
            SESSION_KEYS["QUERY_MOL"]: None,
            SESSION_KEYS["INPUT_SMILES"]: None,
            SESSION_KEYS["PROTEIN_PDB_ID"]: None,
            SESSION_KEYS["PROTEIN_FILE_PATH"]: None,
            SESSION_KEYS["CUSTOM_TEMPLATES"]: None,
            SESSION_KEYS["POSES"]: {},
            SESSION_KEYS["TEMPLATE_USED"]: None,
            SESSION_KEYS["TEMPLATE_INFO"]: None,
            SESSION_KEYS["MCS_INFO"]: None,
            SESSION_KEYS["ALL_RANKED_POSES"]: None,
            SESSION_KEYS["HARDWARE_INFO"]: None,
            SESSION_KEYS["FAIR_METADATA"]: None,
            SESSION_KEYS["SHOW_FAIR_PANEL"]: False,
            # Additional tracking
            "session_start_time": None,
            "pipeline_runs": 0,
            "last_error": None,
            "file_cache": {},
            "logged_messages": set(),
        }

        # Track large objects for memory management
        self.large_object_keys: Set[str] = {
            SESSION_KEYS["QUERY_MOL"],
            SESSION_KEYS["CUSTOM_TEMPLATES"],
            SESSION_KEYS["POSES"],
            SESSION_KEYS["ALL_RANKED_POSES"],
        }

        # Callbacks for state changes
        self.change_callbacks: Dict[str, List[Callable]] = {}

    def initialize(self) -> None:
        """Initialize session state with defaults"""
        if not self.is_initialized():
            logger.info("Initializing session state")

            # Set all defaults
            for key, default_value in self.defaults.items():
                if key not in st.session_state:
                    st.session_state[key] = default_value

            # Set initialization timestamp
            st.session_state["session_start_time"] = datetime.now()
            st.session_state[SESSION_KEYS["APP_INITIALIZED"]] = True

            logger.info("Session state initialized successfully")

    def is_initialized(self) -> bool:
        """Check if session is initialized

        Returns:
            True if initialized
        """
        return st.session_state.get(SESSION_KEYS["APP_INITIALIZED"], False)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from session state with memory-aware handling

        Args:
            key: Session key
            default: Default value if key not found

        Returns:
            Value from session state or default
        """
        # For molecule keys, try memory manager first
        if key in self.large_object_keys:
            try:
                # Try to get from memory manager
                memory_value = self.memory_manager.get_molecule(key)
                if memory_value is not None:
                    logger.debug(f"Retrieved {key} from memory manager")
                    return memory_value
                
                # Try to get from memory manager for poses
                if key == SESSION_KEYS["POSES"]:
                    pose_results = self.memory_manager.get_pose_results()
                    if pose_results:
                        logger.debug(f"Retrieved poses from memory manager")
                        return pose_results
                        
            except Exception as e:
                logger.warning(f"Memory manager retrieval failed for {key}: {e}")
        
        # Fallback to session state
        if key in st.session_state:
            return st.session_state[key]
        
        # Return default
        return self.defaults.get(key, default)

    def set(self, key: str, value: Any, track_large: bool = None) -> None:
        """Set value in session state with memory-aware handling

        Args:
            key: Session key
            value: Value to store
            track_large: Whether to track as large object (auto-detected if None)
        """
        # Add debugging for poses
        if key == SESSION_KEYS["POSES"]:
            logger.info(f"Setting poses: {type(value)}, length: {len(value) if isinstance(value, dict) else 'N/A'}")
            if isinstance(value, dict):
                logger.info(f"Pose methods: {list(value.keys())}")
        
        # Auto-detect large objects if not specified
        if track_large is None:
            track_large = key in self.large_object_keys or self._is_large_object(value)

        # Store in memory manager for large objects and molecules
        if track_large or key in self.large_object_keys:
            try:
                # Special handling for different data types
                if key == SESSION_KEYS["QUERY_MOL"] and value is not None:
                    success = self.memory_manager.store_molecule("query", value)
                    if success:
                        logger.debug(f"Stored query molecule in memory manager")
                    # Also store in session state as fallback
                    st.session_state[key] = value
                    
                elif key == SESSION_KEYS["TEMPLATE_USED"] and value is not None:
                    success = self.memory_manager.store_molecule("template", value)
                    if success:
                        logger.debug(f"Stored template molecule in memory manager")
                    # Also store in session state as fallback
                    st.session_state[key] = value
                    
                elif key == SESSION_KEYS["POSES"] and value is not None:
                    # Always store in session state first as fallback
                    st.session_state[key] = value
                    logger.info(f"Stored poses directly in session state as primary storage")
                    
                    # Also try to store in memory manager for optimization
                    try:
                        success = self.memory_manager.store_pose_results(value)
                        if success:
                            logger.info(f"Successfully stored poses in memory manager as secondary storage")
                        else:
                            logger.warning(f"Failed to store poses in memory manager, but session state storage succeeded")
                    except Exception as mem_error:
                        logger.warning(f"Memory manager storage failed for poses: {mem_error}, but session state storage succeeded")
                    
                else:
                    # General large object storage
                    success = self.memory_manager.store_large_object(key, value)
                    if success:
                        logger.debug(f"Stored {key} in memory manager")
                    # Always store in session state as well for backward compatibility
                    st.session_state[key] = value
                    
            except Exception as e:
                logger.warning(f"Memory manager storage failed for {key}: {e}")
                # Fallback to session state
                st.session_state[key] = value
        else:
            # Direct storage for small objects
            st.session_state[key] = value

        # Track change
        self._trigger_callbacks(key, value)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple session values at once

        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            self.set(key, value)

    def clear(
        self, keys: Optional[List[str]] = None, preserve_core: bool = True
    ) -> None:
        """Clear session state

        Args:
            keys: Specific keys to clear (None for all)
            preserve_core: Whether to preserve core keys like initialization
        """
        if keys:
            # Clear specific keys
            for key in keys:
                if key in st.session_state:
                    del st.session_state[key]
                    logger.debug(f"Cleared session key: {key}")
        else:
            # Clear all except core keys
            preserve_keys = (
                {SESSION_KEYS["APP_INITIALIZED"], "session_start_time"}
                if preserve_core
                else set()
            )

            keys_to_clear = [
                k for k in st.session_state.keys() if k not in preserve_keys
            ]
            for key in keys_to_clear:
                del st.session_state[key]

            logger.info(f"Cleared {len(keys_to_clear)} session keys")

        # Trigger memory cleanup
        self.memory_manager.cleanup_memory()
        
        # Cleanup temporary files
        self._cleanup_temp_files()

    def has_key(self, key: str) -> bool:
        """Check if key exists in session state

        Args:
            key: Session key

        Returns:
            True if key exists
        """
        return key in st.session_state

    def get_molecule_data(self) -> Dict[str, Any]:
        """Get all molecule-related data from session

        Returns:
            Dictionary with molecule data
        """
        return {
            "query_mol": self.get(SESSION_KEYS["QUERY_MOL"]),
            "input_smiles": self.get(SESSION_KEYS["INPUT_SMILES"]),
            "template_used": self.get(SESSION_KEYS["TEMPLATE_USED"]),
            "custom_templates": self.get(SESSION_KEYS["CUSTOM_TEMPLATES"]),
        }

    def get_protein_data(self) -> Dict[str, Any]:
        """Get all protein-related data from session

        Returns:
            Dictionary with protein data
        """
        return {
            "pdb_id": self.get(SESSION_KEYS["PROTEIN_PDB_ID"]),
            "file_path": self.get(SESSION_KEYS["PROTEIN_FILE_PATH"]),
        }

    def get_results_data(self) -> Dict[str, Any]:
        """Get all results-related data from session

        Returns:
            Dictionary with results data
        """
        return {
            "poses": self.get(SESSION_KEYS["POSES"], {}),
            "template_info": self.get(SESSION_KEYS["TEMPLATE_INFO"]),
            "mcs_info": self.get(SESSION_KEYS["MCS_INFO"]),
            "all_ranked_poses": self.get(SESSION_KEYS["ALL_RANKED_POSES"]),
        }

    def has_valid_input(self) -> bool:
        """Check if session has valid input for pipeline execution

        Returns:
            True if valid input is present
        """
        has_molecule = bool(self.get(SESSION_KEYS["INPUT_SMILES"]))
        has_protein = bool(
            self.get(SESSION_KEYS["PROTEIN_PDB_ID"])
            or self.get(SESSION_KEYS["PROTEIN_FILE_PATH"])
            or self.get(SESSION_KEYS["CUSTOM_TEMPLATES"])
        )
        return has_molecule and has_protein

    def has_results(self) -> bool:
        """Check if session has pipeline results

        Returns:
            True if results are present
        """
        # First check session state directly
        poses_in_session = st.session_state.get(SESSION_KEYS["POSES"], {})
        if poses_in_session and isinstance(poses_in_session, dict) and len(poses_in_session) > 0:
            logger.debug(f"Found poses in session state: {len(poses_in_session)} poses")
            return True
        
        # Check memory manager for pose results
        try:
            memory_poses = self.memory_manager.get_pose_results()
            if memory_poses and isinstance(memory_poses, dict) and len(memory_poses) > 0:
                logger.debug(f"Found poses in memory manager: {len(memory_poses)} poses")
                return True
        except Exception as e:
            logger.warning(f"Error checking memory manager for poses: {e}")
        
        # Check for pose references in session state (memory manager storage)
        if "best_poses_refs" in st.session_state:
            refs = st.session_state["best_poses_refs"]
            if refs and isinstance(refs, dict) and len(refs) > 0:
                logger.debug(f"Found pose references in session state: {len(refs)} references")
                return True
        
        logger.debug("No poses found in session state or memory manager")
        return False

    def increment_pipeline_runs(self) -> int:
        """Increment pipeline run counter

        Returns:
            New run count
        """
        count = self.get("pipeline_runs", 0) + 1
        self.set("pipeline_runs", count)
        return count

    def register_callback(self, key: str, callback: Callable) -> None:
        """Register callback for state changes

        Args:
            key: Session key to monitor
            callback: Function to call on change
        """
        if key not in self.change_callbacks:
            self.change_callbacks[key] = []
        self.change_callbacks[key].append(callback)

    def _trigger_callbacks(self, key: str, value: Any) -> None:
        """Trigger registered callbacks for a key

        Args:
            key: Changed key
            value: New value
        """
        if key in self.change_callbacks:
            for callback in self.change_callbacks[key]:
                try:
                    callback(key, value)
                except Exception as e:
                    logger.error(f"Callback error for {key}: {e}")

    def get_session_info(self) -> Dict[str, Any]:
        """Get session metadata and statistics

        Returns:
            Dictionary with session information
        """
        start_time = self.get("session_start_time")
        duration = (datetime.now() - start_time).total_seconds() if start_time else 0

        return {
            "session_id": id(st.session_state),
            "initialized": self.is_initialized(),
            "start_time": start_time.isoformat() if start_time else None,
            "duration_seconds": duration,
            "pipeline_runs": self.get("pipeline_runs", 0),
            "has_input": self.has_valid_input(),
            "has_results": self.has_results(),
            "memory_stats": self.memory_manager.get_memory_stats(),
            "state_size": len(st.session_state),
        }

    def export_state(self) -> Dict[str, Any]:
        """Export session state for debugging or persistence

        Returns:
            Serializable dictionary of session state
        """
        export_data = {}

        for key in st.session_state:
            value = st.session_state[key]

            # Skip non-serializable objects
            if key in self.large_object_keys:
                export_data[key] = f"<{type(value).__name__} object>"
            else:
                try:
                    # Attempt to include serializable data
                    if isinstance(value, (str, int, float, bool, list, dict)):
                        export_data[key] = value
                    else:
                        export_data[key] = str(value)
                except:
                    export_data[key] = "<non-serializable>"

        return export_data

    def _is_large_object(self, value: Any) -> bool:
        """Check if an object should be considered large

        Args:
            value: Object to check

        Returns:
            True if object is considered large
        """
        # Check for RDKit molecules
        if hasattr(value, "ToBinary") and hasattr(value, "GetNumAtoms"):
            return True
            
        # Check for pose dictionaries
        if isinstance(value, dict) and value:
            sample_key = next(iter(value))
            sample_value = value[sample_key]
            if isinstance(sample_value, tuple) and len(sample_value) == 2:
                if hasattr(sample_value[0], "ToBinary"):
                    return True
        
        # Check for large lists/tuples
        if isinstance(value, (list, tuple)) and len(value) > 10:
            return True
            
        # Check for large dictionaries
        if isinstance(value, dict) and len(value) > 50:
            return True
            
        return False

    def _cleanup_temp_files(self):
        """Clean up temporary files from uploads and processing"""
        try:
            import tempfile
            import os
            import glob
            
            # Get workspace directory from session
            workspace_dir = self.get(SESSION_KEYS.get("WORKSPACE_DIR"))
            if workspace_dir and os.path.exists(workspace_dir):
                # Clean up uploaded files older than 1 hour
                uploaded_dir = os.path.join(workspace_dir, "temp", "uploaded")
                if os.path.exists(uploaded_dir):
                    for file_path in glob.glob(os.path.join(uploaded_dir, "*")):
                        try:
                            if os.path.getctime(file_path) < (datetime.now().timestamp() - 3600):
                                os.remove(file_path)
                                logger.debug(f"Cleaned up old temp file: {file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to cleanup temp file {file_path}: {e}")
            
            # Clean up system temp files with templ prefix
            temp_dir = tempfile.gettempdir()
            for file_path in glob.glob(os.path.join(temp_dir, "templ_*")):
                try:
                    if os.path.getctime(file_path) < (datetime.now().timestamp() - 3600):
                        os.remove(file_path)
                        logger.debug(f"Cleaned up old system temp file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup system temp file {file_path}: {e}")
                    
        except Exception as e:
            logger.error(f"Error during temp file cleanup: {e}")


# Global session manager instance
_session_manager = None


def get_session_manager(config: Optional[Any] = None) -> SessionManager:
    """Get global session manager instance

    Args:
        config: Optional AppConfig instance

    Returns:
        SessionManager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(config)
    return _session_manager
