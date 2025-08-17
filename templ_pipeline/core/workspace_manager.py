# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Comprehensive workspace management for TEMPL pipeline.

This module consolidates directory lifecycle management, workspace organization,
and file tracking for both temporary and persistent files. It provides:

- Unified workspace structure with clear separation
- Intelligent cleanup policies and lifecycle management
- File tracking and metadata management
- Context manager support for automated cleanup
- Backward compatibility with existing manager classes

Architecture:
    workspace/
    ├── run_YYYYMMDD_HHMMSS/
    │   ├── temp/              # Temporary processing files
    │   │   ├── uploaded/      # Secure uploaded files (UI)
    │   │   ├── processing/    # Intermediate processing files
    │   │   └── cache/         # Cache files (can be deleted)
    │   ├── output/            # Final persistent results
    │   │   ├── poses_final.sdf
    │   │   ├── poses_final_metadata.json
    │   │   └── analysis/
    │   └── logs/              # Processing logs
"""

import os
import shutil
import tempfile
import atexit
import time
import json
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceConfig:
    """Configuration for workspace management."""

    base_dir: str = "workspace"
    auto_cleanup: bool = True
    temp_retention_hours: int = 24
    failed_run_retention_hours: int = 168  # 7 days
    max_workspace_size_gb: float = 10.0
    lazy_creation: bool = True
    centralized_output: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FileInfo:
    """Information about a managed file."""

    path: str
    size_bytes: int
    created_time: float
    file_type: str  # 'temp', 'output', 'log'
    category: str  # 'uploaded', 'processing', 'cache', 'result', 'metadata'

    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_time) / 3600.0

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


class WorkspaceManager:
    """
    Unified workspace manager for TEMPL Pipeline.

    Combines the functionality of DirectoryManager and UnifiedWorkspaceManager
    to provide comprehensive workspace management with intelligent cleanup
    policies and file tracking.
    """

    _registered_workspaces: Set[Path] = set()
    _cleanup_registered = False

    def __init__(
        self,
        run_id: Optional[str] = None,
        config: Optional[WorkspaceConfig] = None,
        base_name: str = "run",
        create_structure: Optional[bool] = None,
    ):
        """
        Initialize workspace manager.

        Args:
            run_id: Unique run identifier (generated if None)
            config: Workspace configuration
            base_name: Base name for run directory
            create_structure: Whether to create directory structure immediately
        """
        self.config = config or WorkspaceConfig()
        self.run_id = run_id or self._generate_run_id()
        self.base_name = base_name

        # Determine creation behavior
        if create_structure is None:
            create_structure = not self.config.lazy_creation
        self.create_structure = create_structure

        # Define workspace structure
        self.workspace_root = Path(self.config.base_dir)
        self.run_dir = self.workspace_root / f"{base_name}_{self.run_id}"
        self.temp_dir = self.run_dir / "temp"
        self.output_dir = self.run_dir / "output"
        self.logs_dir = self.run_dir / "logs"

        # Subdirectories for organization
        self.uploaded_dir = self.temp_dir / "uploaded"
        self.processing_dir = self.temp_dir / "processing"
        self.cache_dir = self.temp_dir / "cache"
        self.analysis_dir = self.output_dir / "analysis"

        # File tracking
        self.managed_files: Dict[str, FileInfo] = {}
        self.cleanup_history: List[Dict[str, Any]] = []
        self._created = False

        # Create structure if requested
        if create_structure:
            self._create_directory_structure()

        # Register for cleanup
        if self.config.auto_cleanup:
            self._register_for_cleanup()

        logger.info(f"Workspace manager initialized: {self.run_dir}")

    def _generate_run_id(self) -> str:
        """Generate unique run identifier."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _create_directory_structure(self):
        """Create the complete directory structure."""
        if self._created and self.run_dir.exists():
            return

        directories = [
            self.run_dir,
            self.temp_dir,
            self.output_dir,
            self.logs_dir,
            self.uploaded_dir,
            self.processing_dir,
            self.cache_dir,
            self.analysis_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        self._created = True
        logger.debug(f"Created workspace structure: {self.run_dir}")

    def _register_for_cleanup(self) -> None:
        """Register workspace for cleanup on exit."""
        WorkspaceManager._registered_workspaces.add(self.run_dir)

        if not WorkspaceManager._cleanup_registered:
            atexit.register(WorkspaceManager._cleanup_all_registered)
            WorkspaceManager._cleanup_registered = True

    @property
    def directory(self) -> Path:
        """Get the main workspace directory, creating it if necessary."""
        if not self._created:
            self._create_directory_structure()
        return self.run_dir

    def get_temp_file(
        self,
        prefix: str = "templ",
        suffix: str = ".tmp",
        category: str = "processing",
        delete: bool = False,
    ) -> str:
        """
        Create temporary file in managed temp directory.

        Args:
            prefix: File prefix
            suffix: File extension
            category: File category ('uploaded', 'processing', 'cache')
            delete: Whether file should be auto-deleted on close

        Returns:
            Path to temporary file
        """
        # Ensure directories exist
        if not self._created:
            self._create_directory_structure()

        # Choose appropriate subdirectory
        if category == "uploaded":
            temp_dir = self.uploaded_dir
        elif category == "cache":
            temp_dir = self.cache_dir
        else:
            temp_dir = self.processing_dir

        # Create temporary file
        fd, temp_path = tempfile.mkstemp(
            prefix=f"{prefix}_", suffix=suffix, dir=temp_dir
        )

        os.close(fd)

        # Track the file
        self._register_file(temp_path, "temp", category)

        logger.debug(f"Created temp file: {temp_path}")
        return temp_path

    def save_uploaded_file(
        self,
        uploaded_file_content: bytes,
        original_filename: str,
        secure_hash: Optional[str] = None,
    ) -> str:
        """
        Save uploaded file securely in managed directory.

        Args:
            uploaded_file_content: File content
            original_filename: Original filename
            secure_hash: Optional hash for secure naming

        Returns:
            Path to saved file
        """
        # Ensure directories exist
        if not self._created:
            self._create_directory_structure()

        # Generate secure filename
        if secure_hash:
            filename = (
                f"{secure_hash}_{int(time.time())}{Path(original_filename).suffix}"
            )
        else:
            import hashlib

            content_hash = hashlib.sha256(uploaded_file_content).hexdigest()[:16]
            filename = (
                f"{content_hash}_{int(time.time())}{Path(original_filename).suffix}"
            )

        file_path = self.uploaded_dir / filename

        # Save file
        file_path.write_bytes(uploaded_file_content)

        # Set restrictive permissions
        try:
            os.chmod(file_path, 0o600)
        except OSError:
            logger.warning(f"Could not set restrictive permissions on {file_path}")

        # Track the file
        self._register_file(str(file_path), "temp", "uploaded")

        logger.info(
            f"Saved uploaded file: {filename} ({len(uploaded_file_content)} bytes)"
        )
        return str(file_path)

    def save_output(
        self,
        filename: str,
        content: Any,
        category: str = "result",
        subdirectory: Optional[str] = None,
    ) -> str:
        """
        Save content to persistent output directory.

        Args:
            filename: Output filename
            content: Content to save (str, bytes, or object with write method)
            category: File category ('result', 'metadata', 'analysis')
            subdirectory: Optional subdirectory within output

        Returns:
            Path to saved file
        """
        # Ensure directories exist
        if not self._created:
            self._create_directory_structure()

        # Determine output location
        if subdirectory:
            output_location = self.output_dir / subdirectory
            output_location.mkdir(parents=True, exist_ok=True)
        elif category == "analysis":
            output_location = self.analysis_dir
        else:
            output_location = self.output_dir

        file_path = output_location / filename

        # Save content based on type
        if isinstance(content, str):
            file_path.write_text(content, encoding="utf-8")
        elif isinstance(content, bytes):
            file_path.write_bytes(content)
        elif hasattr(content, "write"):
            # File-like object
            with open(file_path, "wb") as f:
                if hasattr(content, "read"):
                    f.write(content.read())
                else:
                    content.write(f)
        else:
            # Try to serialize as JSON
            file_path.write_text(json.dumps(content, indent=2), encoding="utf-8")

        # Track the file
        self._register_file(str(file_path), "output", category)

        logger.info(f"Saved output file: {file_path}")
        return str(file_path)

    def save_metadata(
        self,
        base_filename: str,
        metadata: Dict[str, Any],
        include_workspace_info: bool = True,
    ) -> str:
        """
        Save metadata JSON file alongside results.

        Args:
            base_filename: Base filename (metadata will be {base}_metadata.json)
            metadata: Metadata dictionary
            include_workspace_info: Whether to include workspace structure info

        Returns:
            Path to metadata file
        """
        # Generate metadata filename
        base_path = Path(base_filename)
        if base_path.suffix:
            metadata_filename = base_path.stem + "_metadata.json"
        else:
            metadata_filename = base_filename + "_metadata.json"

        # Enhance metadata with workspace information
        if include_workspace_info:
            metadata = metadata.copy()
            metadata["workspace_info"] = {
                "run_id": self.run_id,
                "workspace_root": str(self.workspace_root),
                "run_directory": str(self.run_dir),
                "creation_time": datetime.now().isoformat(),
                "file_structure": self.get_workspace_summary(),
            }

        # Save metadata
        return self.save_output(metadata_filename, metadata, "metadata")

    def _register_file(self, file_path: str, file_type: str, category: str):
        """Register a file for tracking."""
        try:
            stat_info = Path(file_path).stat()
            file_info = FileInfo(
                path=file_path,
                size_bytes=stat_info.st_size,
                created_time=stat_info.st_ctime,
                file_type=file_type,
                category=category,
            )
            self.managed_files[file_path] = file_info
        except Exception as e:
            logger.warning(f"Could not register file {file_path}: {e}")

    def cleanup_temp_files(
        self,
        max_age_hours: Optional[int] = None,
        categories: Optional[List[str]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Clean up temporary files based on age and category.

        Args:
            max_age_hours: Maximum age in hours (uses config default if None)
            categories: File categories to clean (all temp if None)
            force: Force cleanup regardless of age

        Returns:
            Cleanup statistics
        """
        max_age = max_age_hours or self.config.temp_retention_hours
        categories = categories or ["uploaded", "processing", "cache"]

        cleanup_stats = {
            "files_removed": 0,
            "bytes_freed": 0,
            "categories_cleaned": categories,
            "max_age_hours": max_age,
            "timestamp": datetime.now().isoformat(),
        }

        current_time = time.time()
        cutoff_time = current_time - (max_age * 3600)

        files_to_remove = []

        # Find files to remove
        for file_path, file_info in self.managed_files.items():
            if (
                file_info.file_type == "temp"
                and file_info.category in categories
                and (force or file_info.created_time < cutoff_time)
            ):
                files_to_remove.append((file_path, file_info))

        # Remove files
        for file_path, file_info in files_to_remove:
            try:
                if Path(file_path).exists():
                    Path(file_path).unlink()
                    cleanup_stats["files_removed"] += 1
                    cleanup_stats["bytes_freed"] += file_info.size_bytes

                # Remove from tracking
                del self.managed_files[file_path]

                logger.debug(f"Cleaned up temp file: {file_path}")

            except Exception as e:
                logger.warning(f"Failed to remove temp file {file_path}: {e}")

        # Clean empty directories
        self._clean_empty_directories()

        # Record cleanup
        self.cleanup_history.append(cleanup_stats)

        logger.info(
            f"Cleanup completed: {cleanup_stats['files_removed']} files, "
            f"{cleanup_stats['bytes_freed'] / (1024*1024):.1f}MB freed"
        )

        return cleanup_stats

    def _clean_empty_directories(self):
        """Remove empty directories in temp structure."""
        temp_subdirs = [self.cache_dir, self.processing_dir, self.uploaded_dir]

        for directory in temp_subdirs:
            try:
                if directory.exists() and not any(directory.iterdir()):
                    directory.rmdir()
                    logger.debug(f"Removed empty directory: {directory}")
            except Exception as e:
                logger.debug(f"Could not remove directory {directory}: {e}")

    def archive_workspace(
        self, include_temp: bool = False, archive_name: Optional[str] = None
    ) -> str:
        """
        Create archive of workspace.

        Args:
            include_temp: Whether to include temporary files
            archive_name: Custom archive name

        Returns:
            Path to created archive
        """
        if not archive_name:
            archive_name = f"templ_run_{self.run_id}_complete.zip"

        archive_path = self.workspace_root / archive_name

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Always include output and logs
            for directory in [self.output_dir, self.logs_dir]:
                if directory.exists():
                    for file_path in directory.rglob("*"):
                        if file_path.is_file():
                            arcname = file_path.relative_to(self.run_dir)
                            zipf.write(file_path, arcname)

            # Optionally include temp files
            if include_temp and self.temp_dir.exists():
                for file_path in self.temp_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(self.run_dir)
                        zipf.write(file_path, arcname)

        logger.info(f"Created workspace archive: {archive_path}")
        return str(archive_path)

    def get_workspace_summary(self) -> Dict[str, Any]:
        """Get comprehensive workspace summary."""
        summary = {
            "run_id": self.run_id,
            "workspace_root": str(self.workspace_root),
            "run_directory": str(self.run_dir),
            "directories": {
                "temp": str(self.temp_dir),
                "output": str(self.output_dir),
                "logs": str(self.logs_dir),
            },
            "file_counts": {"temp": 0, "output": 0, "log": 0},
            "total_size_mb": 0,
            "temp_size_mb": 0,
            "output_size_mb": 0,
        }

        # Calculate file statistics
        for file_info in self.managed_files.values():
            summary["file_counts"][file_info.file_type] += 1
            summary["total_size_mb"] += file_info.size_mb

            if file_info.file_type == "temp":
                summary["temp_size_mb"] += file_info.size_mb
            elif file_info.file_type == "output":
                summary["output_size_mb"] += file_info.size_mb

        # Add cleanup history
        summary["cleanup_history"] = self.cleanup_history[-5:]  # Last 5 cleanups

        return summary

    def cleanup_workspace(self, keep_outputs: bool = True) -> Dict[str, Any]:
        """
        Clean up entire workspace.

        Args:
            keep_outputs: Whether to preserve output files

        Returns:
            Cleanup statistics
        """
        if keep_outputs:
            # Clean only temp files
            return self.cleanup_temp_files(force=True)
        else:
            # Remove entire workspace
            if self.run_dir.exists():
                shutil.rmtree(self.run_dir)
                logger.info(f"Removed entire workspace: {self.run_dir}")

                # Remove from registered workspaces
                WorkspaceManager._registered_workspaces.discard(self.run_dir)

                self._created = False
                return {
                    "workspace_removed": True,
                    "path": str(self.run_dir),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                return {"workspace_removed": False, "reason": "Directory not found"}

    def cleanup(self) -> bool:
        """
        Clean up the managed workspace.

        Returns:
            bool: True if cleanup was successful
        """
        return self.cleanup_workspace(keep_outputs=False)["workspace_removed"]

    def exists(self) -> bool:
        """Check if the workspace exists."""
        return self.run_dir.exists()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with optional cleanup."""
        if self.config.auto_cleanup:
            try:
                self.cleanup_temp_files()
            except Exception as e:
                logger.warning(f"Auto-cleanup failed: {e}")

    @classmethod
    def _cleanup_all_registered(cls) -> None:
        """Clean up all registered workspaces."""
        logger.debug(
            f"Cleaning up {len(cls._registered_workspaces)} registered workspaces"
        )

        for workspace in list(cls._registered_workspaces):
            try:
                if workspace.exists():
                    shutil.rmtree(workspace)
                    logger.debug(f"Cleaned up registered workspace: {workspace}")
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup registered workspace {workspace}: {e}"
                )

        cls._registered_workspaces.clear()

    @classmethod
    def cleanup_old_workspaces(
        cls,
        workspace_root: str = "workspace",
        max_age_hours: int = 24,
        patterns: Optional[List[str]] = None,
        keep_successful: bool = True,
    ) -> int:
        """
        Clean up old workspace directories.

        Args:
            workspace_root: Root workspace directory
            max_age_hours: Maximum age in hours before cleanup
            patterns: List of glob patterns to match
            keep_successful: Whether to keep workspaces with output files

        Returns:
            Number of workspaces cleaned up
        """
        workspace_path = Path(workspace_root)
        if not workspace_path.exists():
            return 0

        if patterns is None:
            patterns = ["run_*", "output_*", "temp_*"]

        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        removed_count = 0

        for pattern in patterns:
            for workspace_dir in workspace_path.glob(pattern):
                if not workspace_dir.is_dir():
                    continue

                try:
                    # Check modification time
                    mtime = datetime.fromtimestamp(workspace_dir.stat().st_mtime)

                    if mtime < cutoff_time:
                        # Check if we should keep successful runs
                        if keep_successful and (workspace_dir / "output").exists():
                            output_files = list((workspace_dir / "output").glob("*"))
                            if output_files:
                                continue  # Skip this workspace

                        # Check if directory is empty or has only empty files
                        files = list(workspace_dir.rglob("*"))
                        non_empty_files = [
                            f for f in files if f.is_file() and f.stat().st_size > 0
                        ]

                        if not non_empty_files or not keep_successful:
                            shutil.rmtree(workspace_dir)
                            logger.debug(f"Cleaned up old workspace: {workspace_dir}")
                            removed_count += 1

                except Exception as e:
                    logger.warning(f"Error processing workspace {workspace_dir}: {e}")

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old workspaces")

        return removed_count


# Backward compatibility classes and functions


class DirectoryManager(WorkspaceManager):
    """Backward compatibility wrapper for DirectoryManager."""

    def __init__(
        self,
        base_name: str = "output",
        run_id: Optional[str] = None,
        auto_cleanup: bool = False,
        lazy_creation: bool = True,
        centralized_output: bool = True,
        output_root: Optional[str] = None,
    ):
        config = WorkspaceConfig(
            base_dir=output_root or "output",
            auto_cleanup=auto_cleanup,
            lazy_creation=lazy_creation,
            centralized_output=centralized_output,
        )
        super().__init__(run_id=run_id, config=config, base_name=base_name)


class TempDirectoryManager(WorkspaceManager):
    """Backward compatibility wrapper for TempDirectoryManager."""

    def __init__(self, prefix: str = "templ_temp_", **kwargs):
        # Use tempfile to generate unique directory
        temp_dir = tempfile.mkdtemp(prefix=prefix)
        temp_path = Path(temp_dir)

        # Create config with temp directory
        config = WorkspaceConfig(
            base_dir=str(temp_path.parent), auto_cleanup=True, lazy_creation=False
        )

        super().__init__(run_id=temp_path.name, config=config, base_name="", **kwargs)


class UnifiedWorkspaceManager(WorkspaceManager):
    """Backward compatibility wrapper for UnifiedWorkspaceManager."""

    pass


# Convenience functions


def create_workspace_manager(
    run_id: Optional[str] = None, base_dir: str = "workspace", auto_cleanup: bool = True
) -> WorkspaceManager:
    """Create workspace manager with simple configuration."""
    config = WorkspaceConfig(base_dir=base_dir, auto_cleanup=auto_cleanup)
    return WorkspaceManager(run_id=run_id, config=config)


def register_directory_cleanup(directory: Path, priority: int = 0) -> None:
    """
    Register a directory for cleanup on exit.

    Args:
        directory: Directory to register for cleanup
        priority: Cleanup priority (higher = cleaned up first)
    """
    WorkspaceManager._registered_workspaces.add(directory)

    if not WorkspaceManager._cleanup_registered:
        atexit.register(WorkspaceManager._cleanup_all_registered)
        WorkspaceManager._cleanup_registered = True


def cleanup_test_artifacts(base_path: Optional[Path] = None) -> int:
    """
    Clean up test artifacts and temporary directories.

    Args:
        base_path: Base directory to search in

    Returns:
        Number of directories cleaned up
    """
    if base_path is None:
        base_path = Path(".")

    return WorkspaceManager.cleanup_old_workspaces(
        workspace_root=str(base_path),
        max_age_hours=1,
        patterns=[
            "templ_test_*",
            "test_temp_*",
            "qa_test_*",
            "output_test_*",
            "debug_*",
            "temp_*",
        ],
        keep_successful=False,
    )


def emergency_cleanup() -> None:
    """Emergency cleanup function for critical situations."""
    try:
        logger.info("Running emergency cleanup...")

        # Clean up all registered workspaces
        WorkspaceManager._cleanup_all_registered()

        # Clean up old test artifacts
        cleanup_test_artifacts()

        # Clean up very old workspaces
        WorkspaceManager.cleanup_old_workspaces(max_age_hours=168)  # 1 week

        logger.info("Emergency cleanup completed")

    except Exception as e:
        logger.error(f"Emergency cleanup failed: {e}")


# Register emergency cleanup on module import
atexit.register(emergency_cleanup)
