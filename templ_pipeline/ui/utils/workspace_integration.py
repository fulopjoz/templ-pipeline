"""
TEMPL Pipeline - UI Workspace Integration

Utility classes for integrating the UnifiedWorkspaceManager with UI components,
providing seamless file management and workspace tracking for the web interface.
"""

import logging
import streamlit as st
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class UIWorkspaceIntegration:
    """Integration utility for UnifiedWorkspaceManager with Streamlit UI"""
    
    def __init__(self, session_manager, pipeline_service):
        """
        Initialize workspace integration.
        
        Args:
            session_manager: SessionManager instance
            pipeline_service: PipelineService instance
        """
        self.session = session_manager
        self.pipeline_service = pipeline_service
        self.workspace_manager = pipeline_service.get_workspace_manager()
    
    def get_secure_upload_handler(self):
        """
        Get a SecureFileUploadHandler configured with workspace manager.
        
        Returns:
            SecureFileUploadHandler instance
        """
        from ..core.secure_upload import SecureFileUploadHandler
        
        return SecureFileUploadHandler(
            workspace_manager=self.workspace_manager
        )
    
    def handle_file_upload(self, uploaded_file, file_type: str, 
                          custom_max_size: Optional[int] = None) -> Tuple[bool, str, Optional[str]]:
        """
        Handle file upload with workspace integration.
        
        Args:
            uploaded_file: Streamlit uploaded file object
            file_type: Expected file type
            custom_max_size: Custom maximum size in MB
            
        Returns:
            Tuple of (success, message, file_path)
        """
        if not uploaded_file:
            return False, "No file provided", None
        
        try:
            upload_handler = self.get_secure_upload_handler()
            success, message, file_path = upload_handler.validate_and_save(
                uploaded_file, file_type, custom_max_size
            )
            
            if success and file_path:
                # Track uploaded file in session
                uploaded_files = self.session.get("uploaded_files", [])
                uploaded_files.append({
                    "original_name": uploaded_file.name,
                    "secure_path": file_path,
                    "file_type": file_type,
                    "size_bytes": uploaded_file.size,
                    "upload_time": time.time()
                })
                self.session.set("uploaded_files", uploaded_files)
                
                logger.info(f"File uploaded successfully: {uploaded_file.name} -> {file_path}")
            
            return success, message, file_path
            
        except Exception as e:
            logger.error(f"File upload failed: {e}")
            return False, f"Upload failed: {str(e)}", None
    
    def get_workspace_summary(self) -> Dict[str, Any]:
        """
        Get workspace summary for display in UI.
        
        Returns:
            Dictionary with workspace information
        """
        if not self.workspace_manager:
            return {
                "enabled": False,
                "message": "Unified workspace not available"
            }
        
        try:
            summary = self.workspace_manager.get_workspace_summary()
            summary["enabled"] = True
            
            # Add UI-specific information
            summary["uploaded_files"] = self.session.get("uploaded_files", [])
            summary["session_id"] = self.session.get("session_id")
            summary["workspace_run_id"] = self.session.get("workspace_run_id")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get workspace summary: {e}")
            return {
                "enabled": False,
                "error": str(e)
            }
    
    def cleanup_session_files(self, force: bool = False) -> Dict[str, Any]:
        """
        Cleanup temporary files for the current session.
        
        Args:
            force: Force cleanup regardless of age
            
        Returns:
            Cleanup statistics
        """
        if not self.workspace_manager:
            return {"error": "Workspace manager not available"}
        
        try:
            # Cleanup temporary files
            stats = self.workspace_manager.cleanup_temp_files(force=force)
            
            # Clear uploaded files from session
            if force:
                self.session.set("uploaded_files", [])
            
            logger.info(f"Session cleanup completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
            return {"error": str(e)}
    
    def get_output_files(self) -> List[Dict[str, Any]]:
        """
        Get list of output files from the workspace.
        
        Returns:
            List of output file information
        """
        if not self.workspace_manager:
            return []
        
        try:
            output_files = []
            output_dir = self.workspace_manager.output_dir
            
            if output_dir.exists():
                for file_path in output_dir.rglob('*'):
                    if file_path.is_file():
                        try:
                            stat_info = file_path.stat()
                            output_files.append({
                                "name": file_path.name,
                                "path": str(file_path),
                                "relative_path": str(file_path.relative_to(output_dir)),
                                "size_bytes": stat_info.st_size,
                                "size_mb": stat_info.st_size / (1024 * 1024),
                                "modified_time": stat_info.st_mtime,
                                "extension": file_path.suffix.lower()
                            })
                        except Exception as e:
                            logger.warning(f"Could not get info for {file_path}: {e}")
            
            return sorted(output_files, key=lambda x: x["modified_time"], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to get output files: {e}")
            return []
    
    def download_file(self, file_path: str, display_name: Optional[str] = None) -> Tuple[bool, bytes, str]:
        """
        Prepare file for download.
        
        Args:
            file_path: Path to file to download
            display_name: Display name for download
            
        Returns:
            Tuple of (success, file_content, filename)
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return False, b"", f"File not found: {file_path}"
            
            content = path.read_bytes()
            filename = display_name or path.name
            
            return True, content, filename
            
        except Exception as e:
            logger.error(f"Failed to prepare file for download: {e}")
            return False, b"", str(e)
    
    def display_workspace_status(self):
        """Display workspace status in Streamlit sidebar"""
        try:
            with st.sidebar:
                st.markdown("### 🗂️ Workspace Panel")
                st.markdown("*Manage files and workspace settings*")
                st.markdown("---")
                
                # Quick workspace actions
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📋 View Files", help="Show workspace files", use_container_width=True):
                        st.session_state.show_workspace_files = not st.session_state.get('show_workspace_files', False)
                        st.rerun()
                        
                with col2:
                    if st.button("🧹 Cleanup", help="Clean temporary files", use_container_width=True):
                        cleanup_stats = self.cleanup_session_files()
                        if "error" not in cleanup_stats:
                            st.success("Cleanup completed!")
                        else:
                            st.error(f"Cleanup failed: {cleanup_stats['error']}")
                
                st.markdown("---")
                
                # Use expander for the detailed workspace status (collapsed by default)
                with st.expander("📊 Detailed Status", expanded=False):
                    summary = self.get_workspace_summary()
                    
                    if not summary.get("enabled", False):
                        st.warning("Unified workspace not available")
                        if "error" in summary:
                            st.error(f"Error: {summary['error']}")
                        return
                    
                    # Display workspace information
                    st.success("✅ Unified workspace active")
                    
                    # Show workspace details directly (no nested expander)
                    st.write(f"**Run ID:** `{summary.get('run_id', 'N/A')}`")
                    st.write(f"**Directory:** `{summary.get('run_directory', 'N/A')}`")
                    
                    # File counts
                    file_counts = summary.get("file_counts", {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Temp Files", file_counts.get("temp", 0))
                    with col2:
                        st.metric("Output Files", file_counts.get("output", 0))
                    with col3:
                        st.metric("Log Files", file_counts.get("log", 0))
                    
                    # Storage usage
                    total_mb = summary.get("total_size_mb", 0)
                    temp_mb = summary.get("temp_size_mb", 0)
                    output_mb = summary.get("output_size_mb", 0)
                    
                    st.write("**Storage Usage:**")
                    st.write(f"- Total: {total_mb:.1f} MB")
                    st.write(f"- Temporary: {temp_mb:.1f} MB")
                    st.write(f"- Output: {output_mb:.1f} MB")
                    
                
                # Show workspace files if requested
                if st.session_state.get('show_workspace_files', False):
                    st.markdown("---")
                    st.markdown("### 📁 Recent Files")
                    
                    output_files = self.get_output_files()
                    if output_files:
                        for file_info in output_files[:5]:  # Show max 5 recent files
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**{file_info['name']}** ({file_info['size_mb']:.1f} MB)")
                            with col2:
                                success, content, filename = self.download_file(file_info['path'])
                                if success:
                                    st.download_button(
                                        "⬇️",
                                        data=content,
                                        file_name=filename,
                                        mime="application/octet-stream",
                                        key=f"dl_{file_info['name']}",
                                        help="Download file"
                                    )
                        
                        if len(output_files) > 5:
                            st.info(f"+ {len(output_files) - 5} more files...")
                    else:
                        st.info("No output files found")
                
        except Exception as e:
            logger.error(f"Failed to display workspace status: {e}")
            with st.sidebar:
                st.error(f"Workspace status error: {str(e)}")
                # Add troubleshooting info
                with st.expander("Troubleshooting", expanded=False):
                    st.code(str(e))
                    st.info("Try refreshing the page or restarting the application.")
    
    def display_output_files(self):
        """Display output files section in UI"""
        try:
            output_files = self.get_output_files()
            
            if not output_files:
                st.info("No output files generated yet")
                return
            
            st.subheader("📁 Output Files")
            
            for file_info in output_files:
                with st.expander(f"📄 {file_info['name']} ({file_info['size_mb']:.1f} MB)", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Path:** `{file_info['relative_path']}`")
                        st.write(f"**Size:** {file_info['size_mb']:.1f} MB")
                        st.write(f"**Modified:** {time.ctime(file_info['modified_time'])}")
                    
                    with col2:
                        success, content, filename = self.download_file(file_info['path'])
                        if success:
                            st.download_button(
                                "⬇️ Download",
                                data=content,
                                file_name=filename,
                                mime="application/octet-stream",
                                key=f"download_{file_info['name']}"
                            )
                        else:
                            st.error("Download failed")
                
        except Exception as e:
            logger.error(f"Failed to display output files: {e}")
            st.error(f"Error displaying output files: {str(e)}")
    
    def archive_session_workspace(self, include_temp: bool = False) -> Tuple[bool, str]:
        """
        Create archive of the session workspace.
        
        Args:
            include_temp: Whether to include temporary files
            
        Returns:
            Tuple of (success, archive_path_or_error)
        """
        if not self.workspace_manager:
            return False, "Workspace manager not available"
        
        try:
            archive_path = self.workspace_manager.archive_workspace(
                include_temp=include_temp
            )
            return True, archive_path
            
        except Exception as e:
            logger.error(f"Failed to create workspace archive: {e}")
            return False, str(e)


# Convenience function for creating workspace integration
def get_workspace_integration(session_manager, pipeline_service) -> UIWorkspaceIntegration:
    """
    Create workspace integration instance.
    
    Args:
        session_manager: SessionManager instance
        pipeline_service: PipelineService instance
        
    Returns:
        UIWorkspaceIntegration instance
    """
    return UIWorkspaceIntegration(session_manager, pipeline_service) 