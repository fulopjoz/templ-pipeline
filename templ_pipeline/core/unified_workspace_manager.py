"""
TEMPL Pipeline - Unified Workspace Manager

Consolidates temporary and output file management for both UI and CLI components.
Provides clear separation between temporary processing files and persistent results.

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
import time
import shutil
import tempfile
import logging
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceConfig:
    """Configuration for workspace management."""
    
    base_dir: str = "workspace"
    auto_cleanup: bool = True
    temp_retention_hours: int = 24
    failed_run_retention_hours: int = 168  # 7 days
    max_workspace_size_gb: float = 10.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FileInfo:
    """Information about a managed file."""
    
    path: str
    size_bytes: int
    created_time: float
    file_type: str  # 'temp', 'output', 'log'
    category: str   # 'uploaded', 'processing', 'cache', 'result', 'metadata'
    
    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_time) / 3600.0
    
    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)


class UnifiedWorkspaceManager:
    """
    Unified workspace manager for TEMPL Pipeline.
    
    Manages both temporary and output files with clear separation and
    intelligent cleanup policies.
    """
    
    def __init__(
        self, 
        run_id: Optional[str] = None,
        config: Optional[WorkspaceConfig] = None,
        create_structure: bool = True
    ):
        """
        Initialize workspace manager.
        
        Args:
            run_id: Unique run identifier (generated if None)
            config: Workspace configuration
            create_structure: Whether to create directory structure immediately
        """
        self.config = config or WorkspaceConfig()
        self.run_id = run_id or self._generate_run_id()
        
        # Define workspace structure
        self.workspace_root = Path(self.config.base_dir)
        self.run_dir = self.workspace_root / f"run_{self.run_id}"
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
        
        if create_structure:
            self._create_directory_structure()
            
        logger.info(f"Workspace manager initialized: {self.run_dir}")
    
    def _generate_run_id(self) -> str:
        """Generate unique run identifier."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _create_directory_structure(self):
        """Create the complete directory structure."""
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
            
        logger.debug(f"Created workspace structure: {self.run_dir}")
    
    def get_temp_file(
        self, 
        prefix: str = "templ", 
        suffix: str = ".tmp", 
        category: str = "processing",
        delete: bool = False
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
        # Choose appropriate subdirectory
        if category == "uploaded":
            temp_dir = self.uploaded_dir
        elif category == "cache":
            temp_dir = self.cache_dir
        else:
            temp_dir = self.processing_dir
            
        # Create temporary file
        fd, temp_path = tempfile.mkstemp(
            prefix=f"{prefix}_",
            suffix=suffix,
            dir=temp_dir
        )
        
        if delete:
            os.close(fd)
        else:
            os.close(fd)
            
        # Track the file
        self._register_file(temp_path, "temp", category)
        
        logger.debug(f"Created temp file: {temp_path}")
        return temp_path
    
    def save_uploaded_file(
        self, 
        uploaded_file_content: bytes, 
        original_filename: str,
        secure_hash: Optional[str] = None
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
        # Generate secure filename
        if secure_hash:
            filename = f"{secure_hash}_{int(time.time())}{Path(original_filename).suffix}"
        else:
            import hashlib
            content_hash = hashlib.sha256(uploaded_file_content).hexdigest()[:16]
            filename = f"{content_hash}_{int(time.time())}{Path(original_filename).suffix}"
        
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
        
        logger.info(f"Saved uploaded file: {filename} ({len(uploaded_file_content)} bytes)")
        return str(file_path)
    
    def save_output(
        self, 
        filename: str, 
        content: Any, 
        category: str = "result",
        subdirectory: Optional[str] = None
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
        # Determine output location
        if subdirectory:
            output_location = self.output_dir / subdirectory
            output_location.mkdir(parents=True, exist_ok=True)  # Create nested directories
        elif category == "analysis":
            output_location = self.analysis_dir
        else:
            output_location = self.output_dir
            
        file_path = output_location / filename
        
        # Save content based on type
        if isinstance(content, str):
            file_path.write_text(content, encoding='utf-8')
        elif isinstance(content, bytes):
            file_path.write_bytes(content)
        elif hasattr(content, 'write'):
            # File-like object
            with open(file_path, 'wb') as f:
                if hasattr(content, 'read'):
                    f.write(content.read())
                else:
                    content.write(f)
        else:
            # Try to serialize as JSON
            file_path.write_text(json.dumps(content, indent=2), encoding='utf-8')
        
        # Track the file
        self._register_file(str(file_path), "output", category)
        
        logger.info(f"Saved output file: {file_path}")
        return str(file_path)
    
    def save_metadata(
        self, 
        base_filename: str, 
        metadata: Dict[str, Any],
        include_workspace_info: bool = True
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
                "file_structure": self.get_workspace_summary()
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
                category=category
            )
            self.managed_files[file_path] = file_info
        except Exception as e:
            logger.warning(f"Could not register file {file_path}: {e}")
    
    def cleanup_temp_files(
        self, 
        max_age_hours: Optional[int] = None,
        categories: Optional[List[str]] = None,
        force: bool = False
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
            "timestamp": datetime.now().isoformat()
        }
        
        current_time = time.time()
        cutoff_time = current_time - (max_age * 3600)
        
        files_to_remove = []
        
        # Find files to remove
        for file_path, file_info in self.managed_files.items():
            if (file_info.file_type == "temp" and 
                file_info.category in categories and
                (force or file_info.created_time < cutoff_time)):
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
        
        logger.info(f"Cleanup completed: {cleanup_stats['files_removed']} files, "
                   f"{cleanup_stats['bytes_freed'] / (1024*1024):.1f}MB freed")
        
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
        self, 
        include_temp: bool = False,
        archive_name: Optional[str] = None
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
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Always include output and logs
            for directory in [self.output_dir, self.logs_dir]:
                if directory.exists():
                    for file_path in directory.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(self.run_dir)
                            zipf.write(file_path, arcname)
            
            # Optionally include temp files
            if include_temp and self.temp_dir.exists():
                for file_path in self.temp_dir.rglob('*'):
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
                "logs": str(self.logs_dir)
            },
            "file_counts": {"temp": 0, "output": 0, "log": 0},
            "total_size_mb": 0,
            "temp_size_mb": 0,
            "output_size_mb": 0
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
                return {
                    "workspace_removed": True,
                    "path": str(self.run_dir),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"workspace_removed": False, "reason": "Directory not found"}
    
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


# Convenience functions for backward compatibility
def create_workspace_manager(
    run_id: Optional[str] = None,
    base_dir: str = "workspace",
    auto_cleanup: bool = True
) -> UnifiedWorkspaceManager:
    """Create workspace manager with simple configuration."""
    config = WorkspaceConfig(
        base_dir=base_dir,
        auto_cleanup=auto_cleanup
    )
    return UnifiedWorkspaceManager(run_id=run_id, config=config)


def cleanup_old_workspaces(
    workspace_root: str = "workspace",
    max_age_days: int = 30,
    keep_successful: bool = True
) -> Dict[str, Any]:
    """
    Clean up old workspace directories.
    
    Args:
        workspace_root: Root workspace directory
        max_age_days: Maximum age in days
        keep_successful: Whether to keep workspaces with output files
        
    Returns:
        Cleanup statistics
    """
    workspace_path = Path(workspace_root)
    if not workspace_path.exists():
        return {"error": "Workspace root does not exist"}
    
    cutoff_time = time.time() - (max_age_days * 24 * 3600)
    cleaned_count = 0
    bytes_freed = 0
    
    for run_dir in workspace_path.glob("run_*"):
        if not run_dir.is_dir():
            continue
            
        try:
            dir_mtime = run_dir.stat().st_mtime
            if dir_mtime < cutoff_time:
                # Check if we should keep successful runs
                if keep_successful and (run_dir / "output").exists():
                    output_files = list((run_dir / "output").glob("*"))
                    if output_files:
                        continue  # Skip this directory
                
                # Calculate size before removal
                dir_size = sum(
                    f.stat().st_size for f in run_dir.rglob('*') if f.is_file()
                )
                
                # Remove directory
                shutil.rmtree(run_dir)
                cleaned_count += 1
                bytes_freed += dir_size
                
                logger.info(f"Cleaned up old workspace: {run_dir}")
                
        except Exception as e:
            logger.warning(f"Failed to clean workspace {run_dir}: {e}")
    
    return {
        "workspaces_cleaned": cleaned_count,
        "bytes_freed": bytes_freed,
        "mb_freed": bytes_freed / (1024 * 1024),
        "cutoff_days": max_age_days,
        "timestamp": datetime.now().isoformat()
    } 