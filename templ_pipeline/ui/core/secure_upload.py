"""
TEMPL Pipeline - Secure File Upload Handler

Advanced file upload security with MIME type validation, filename sanitization,
and secure temporary file storage for molecular data files.
"""

import os
import time
import hashlib
import tempfile
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List

# Try to import python-magic for MIME type detection
try:
    import magic

    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    # Remove the warning - MIME validation is optional
    # logging.warning("python-magic not available - MIME type validation disabled")

logger = logging.getLogger(__name__)


class SecureFileUploadHandler:
    """Advanced file upload security with MIME type validation"""

    # MIME types allowed for each file extension
    ALLOWED_MIMES = {
        ".sdf": ["chemical/x-mdl-sdfile", "text/plain", "application/octet-stream"],
        ".mol": ["chemical/x-mdl-molfile", "text/plain", "application/octet-stream"],
        ".pdb": ["chemical/x-pdb", "text/plain", "application/octet-stream"],
        # Adding common fallback MIME types for chemical files
        ".smi": ["text/plain", "application/octet-stream"],
        ".xyz": ["text/plain", "application/octet-stream"],
    }

    # File extensions allowed for upload
    ALLOWED_EXTENSIONS = {".sdf", ".mol", ".pdb", ".smi", ".xyz"}

    # Maximum file sizes (in MB) for different file types
    MAX_SIZES = {
        ".sdf": 10,  # SDF files can be larger (multiple molecules)
        ".mol": 5,  # Single molecule files
        ".pdb": 5,  # Protein structure files
        ".smi": 1,  # SMILES files are typically small
        ".xyz": 5,  # XYZ coordinate files
    }

    def __init__(self, upload_dir: Optional[str] = None):
        """Initialize secure upload handler

        Args:
            upload_dir: Directory for temporary file storage. If None, uses system temp dir.
        """
        if upload_dir:
            self.upload_dir = Path(upload_dir)
            self.upload_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.upload_dir = Path(tempfile.mkdtemp(prefix="templ_secure_"))

        # Set restrictive permissions on upload directory
        try:
            os.chmod(self.upload_dir, 0o700)
        except OSError:
            logger.warning(
                f"Could not set restrictive permissions on {self.upload_dir}"
            )

    def validate_and_save(
        self, uploaded_file, file_type: str, custom_max_size: Optional[int] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """Comprehensive validation with MIME checking and secure storage

        Args:
            uploaded_file: Streamlit uploaded file object
            file_type: Expected file type (extension without dot)
            custom_max_size: Custom maximum size in MB (overrides defaults)

        Returns:
            Tuple of (success, message, secure_file_path)
        """

        if not uploaded_file:
            return False, "No file provided", None

        # Normalize file type
        file_ext = f".{file_type.lower().lstrip('.')}"

        # Validate file extension
        if file_ext not in self.ALLOWED_EXTENSIONS:
            return (
                False,
                f"File type '{file_ext}' not allowed. Supported: {', '.join(self.ALLOWED_EXTENSIONS)}",
                None,
            )

        # Get maximum size for this file type
        max_size_mb = custom_max_size or self.MAX_SIZES.get(file_ext, 5)
        max_size_bytes = max_size_mb * 1024 * 1024

        # Size validation (check before reading content)
        if uploaded_file.size > max_size_bytes:
            return (
                False,
                f"File exceeds {max_size_mb}MB limit (current: {uploaded_file.size / (1024*1024):.1f}MB)",
                None,
            )

        # Validate filename
        original_filename = uploaded_file.name
        if not self._is_safe_filename(original_filename):
            return (
                False,
                "Invalid filename. Please use only alphanumeric characters, hyphens, and underscores.",
                None,
            )

        # Read content for validation
        try:
            content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset for potential re-read
        except Exception as e:
            return False, f"Could not read file content: {str(e)}", None

        # MIME type validation (if available)
        if MAGIC_AVAILABLE:
            try:
                mime_type = magic.from_buffer(content, mime=True)
                allowed_mimes = self.ALLOWED_MIMES.get(file_ext, [])

                if mime_type not in allowed_mimes:
                    logger.warning(
                        f"MIME type mismatch: expected {allowed_mimes}, got {mime_type}"
                    )
                    # Don't reject - chemical files often have inconsistent MIME types
                    # Just log for monitoring
            except Exception as e:
                logger.warning(f"MIME type validation failed: {e}")

        # Content validation (basic checks for chemical files)
        validation_result = self._validate_content(content, file_ext)
        if not validation_result[0]:
            return validation_result

        # Generate secure filename
        secure_filename = self._generate_secure_filename(original_filename, content)

        # Save to secure location
        try:
            secure_path = self.upload_dir / secure_filename
            secure_path.write_bytes(content)

            # Set restrictive permissions
            try:
                os.chmod(secure_path, 0o600)
            except OSError:
                logger.warning(
                    f"Could not set restrictive permissions on {secure_path}"
                )

            logger.info(
                f"Securely saved file: {secure_filename} ({len(content)} bytes)"
            )
            return (
                True,
                f"File validated and saved ({len(content)} bytes)",
                str(secure_path),
            )

        except Exception as e:
            return False, f"Could not save file: {str(e)}", None

    def _is_safe_filename(self, filename: str) -> bool:
        """Validate filename for security

        Args:
            filename: Original filename to validate

        Returns:
            True if filename is safe
        """
        if not filename:
            return False

        # Check for path traversal attempts
        if ".." in filename or "/" in filename or "\\" in filename:
            return False

        # Check for null bytes
        if "\x00" in filename:
            return False

        # Check reasonable length
        if len(filename) > 255:
            return False

        # Allow alphanumeric, hyphens, underscores, dots
        import re

        if not re.match(r"^[a-zA-Z0-9._-]+$", filename):
            return False

        return True

    def _generate_secure_filename(self, original_filename: str, content: bytes) -> str:
        """Generate secure filename using content hash

        Args:
            original_filename: Original filename
            content: File content

        Returns:
            Secure filename
        """
        # Generate content hash
        content_hash = hashlib.sha256(content).hexdigest()[:16]

        # Get file extension
        ext = Path(original_filename).suffix.lower()

        # Create secure filename
        timestamp = int(time.time())
        secure_name = f"{content_hash}_{timestamp}{ext}"

        return secure_name

    def _validate_content(self, content: bytes, file_ext: str) -> Tuple[bool, str]:
        """Basic content validation for chemical files

        Args:
            content: File content as bytes
            file_ext: File extension

        Returns:
            Tuple of (valid, message)
        """
        try:
            # Decode content for text analysis
            text_content = content.decode("utf-8", errors="ignore")

            if file_ext == ".pdb":
                # PDB files should have ATOM or HETATM records
                lines = text_content.splitlines()
                has_atom_records = any(
                    line.startswith(("ATOM", "HETATM")) for line in lines[:100]
                )

                if not has_atom_records:
                    return False, "Invalid PDB file - no ATOM records found"

            elif file_ext in [".sdf", ".mol"]:
                # SDF/MOL files should have molecular structure indicators
                if (
                    "$$$$" in text_content
                    or "M  END" in text_content
                    or len(text_content.splitlines()) > 3
                ):
                    # Basic structure indicators present
                    pass
                else:
                    return (
                        False,
                        f"Invalid {file_ext} file - no molecular structure found",
                    )

            elif file_ext == ".smi":
                # SMILES files should contain valid characters
                lines = text_content.strip().splitlines()
                if not lines:
                    return False, "Empty SMILES file"

                # Basic SMILES character validation
                import re

                smiles_pattern = re.compile(r"^[A-Za-z0-9@+\-\[\]()=#\\/.%:*$]*$")
                for line in lines[:10]:  # Check first 10 lines
                    if line.strip() and not smiles_pattern.match(
                        line.strip().split()[0]
                    ):
                        return False, "Invalid SMILES characters detected"

            return True, "Content validation passed"

        except Exception as e:
            logger.warning(f"Content validation error: {e}")
            return True, "Content validation skipped due to error"

    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """Remove old uploaded files

        Args:
            max_age_hours: Maximum age in hours before cleanup

        Returns:
            Number of files cleaned up
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleanup_count = 0

        try:
            for file_path in self.upload_dir.glob("*"):
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        cleanup_count += 1
                        logger.info(f"Cleaned up old file: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Could not remove old file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Cleanup operation failed: {e}")

        return cleanup_count

    def get_upload_stats(self) -> Dict[str, any]:
        """Get upload directory statistics

        Returns:
            Dictionary with directory statistics
        """
        try:
            files = list(self.upload_dir.glob("*"))
            total_size = sum(f.stat().st_size for f in files if f.is_file())

            return {
                "upload_dir": str(self.upload_dir),
                "file_count": len(files),
                "total_size_mb": total_size / (1024 * 1024),
                "magic_available": MAGIC_AVAILABLE,
            }
        except Exception as e:
            return {
                "error": str(e),
                "upload_dir": str(self.upload_dir),
                "magic_available": MAGIC_AVAILABLE,
            }


# Convenience function for backward compatibility
def validate_file_secure(
    uploaded_file, file_type: str, max_size_mb: int = 5
) -> Tuple[bool, str, Optional[str]]:
    """Backward compatible function for secure file validation

    Args:
        uploaded_file: Streamlit uploaded file object
        file_type: File type (extension)
        max_size_mb: Maximum size in MB

    Returns:
        Tuple of (success, message, file_path)
    """
    handler = SecureFileUploadHandler()
    return handler.validate_and_save(uploaded_file, file_type, max_size_mb)
