"""
TEMPL Pipeline - Contextual Error Management

Advanced error handling with context preservation, structured logging,
and user-friendly error messages without exposing stack traces.
"""

import json
import time
import logging
import traceback
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

# Try to import streamlit for UI integration
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ErrorContext:
    """Structure for error context information"""
    context_type: str
    data: Dict[str, Any]
    timestamp: str


@dataclass
class ErrorRecord:
    """Structure for complete error records"""
    error_id: str
    timestamp: str
    category: str
    operation: str
    exception_type: str
    exception_message: str
    user_message: str
    app_context: Dict[str, ErrorContext]
    additional_context: Dict[str, Any]
    session_id: Optional[str] = None
    stack_trace: Optional[str] = None


class ContextualErrorManager:
    """Advanced error handling with context preservation and user-friendly messages"""
    
    # Error categories for better organization
    ERROR_CATEGORIES = {
        'FILE_UPLOAD': 'File Upload Error',
        'MOLECULAR_PROCESSING': 'Molecular Processing Error',
        'PIPELINE_ERROR': 'Pipeline Execution Error',
        'VALIDATION_ERROR': 'Input Validation Error',
        'MEMORY_ERROR': 'Memory Management Error',
        'NETWORK_ERROR': 'Network/Database Error',
        'CONFIGURATION_ERROR': 'Configuration Error',
        'CRITICAL': 'Critical System Error'
    }
    
    # User-friendly messages for common error types
    USER_FRIENDLY_MESSAGES = {
        'FILE_UPLOAD': {
            'size_exceeded': "The uploaded file is too large. Please use a smaller file.",
            'invalid_format': "The file format is not supported. Please check the file type.",
            'corrupted_file': "The file appears to be corrupted. Please try uploading again.",
            'processing_failed': "Could not process the uploaded file. Please verify the file content."
        },
        'MOLECULAR_PROCESSING': {
            'invalid_smiles': "The molecular structure is invalid. Please check your SMILES string.",
            'sanitization_failed': "Could not process the molecular structure. Try a simpler molecule.",
            'conformer_generation_failed': "Could not generate 3D conformations for this molecule.",
            'alignment_failed': "Could not align the molecule to the template."
        },
        'PIPELINE_ERROR': {
            'template_not_found': "The specified protein template was not found in the database.",
            'embedding_failed': "Could not generate protein embeddings. Check your configuration.",
            'scoring_failed': "Pose scoring failed. The results may not be reliable.",
            'no_poses_generated': "No valid poses could be generated for this molecule."
        },
        'VALIDATION_ERROR': {
            'empty_input': "Please provide the required input.",
            'invalid_pdb_id': "The PDB ID format is invalid. Use 4-character codes like '1abc'.",
            'molecule_too_large': "The molecule is too large for processing. Try smaller molecules.",
            'molecule_too_small': "The molecule is too small. Minimum 3 atoms required."
        }
    }
    
    def __init__(self, app_name: str = "templ_pipeline", max_history: int = 100):
        """Initialize error manager
        
        Args:
            app_name: Application name for error tracking
            max_history: Maximum number of errors to keep in history
        """
        self.app_name = app_name
        self.max_history = max_history
        self.error_context: Dict[str, ErrorContext] = {}
        self.error_history: List[ErrorRecord] = []
        
        # Setup logging format
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup structured logging format"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Ensure logger has at least one handler
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    def register_context(self, context_type: str, context_data: Dict[str, Any]):
        """Register context that might be useful for error diagnosis
        
        Args:
            context_type: Type of context (e.g., 'user_input', 'system_state')
            context_data: Context data dictionary
        """
        self.error_context[context_type] = ErrorContext(
            context_type=context_type,
            data=context_data,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.debug(f"Registered context: {context_type}")
    
    def handle_error(self, 
                    error_category: str,
                    exception: Exception,
                    operation: str,
                    user_message: Optional[str] = None,
                    error_subtype: Optional[str] = None,
                    additional_context: Optional[Dict] = None,
                    show_recovery: bool = True) -> str:
        """Handle error with full context preservation and user-friendly display
        
        Args:
            error_category: Category of error (from ERROR_CATEGORIES)
            exception: The exception that occurred
            operation: Description of the operation that failed
            user_message: Custom user message (overrides automatic message)
            error_subtype: Specific error subtype for better messaging
            additional_context: Additional context for this specific error
            show_recovery: Whether to show recovery suggestions
        
        Returns:
            Error ID for reference
        """
        
        # Generate unique error ID
        error_id = f"{error_category}_{int(time.time() * 1000)}"
        
        # Get session ID if available
        session_id = None
        if STREAMLIT_AVAILABLE and hasattr(st, 'session_state'):
            session_id = id(st.session_state)
        
        # Generate user-friendly message
        if not user_message:
            user_message = self._generate_user_message(error_category, error_subtype, exception)
        
        # Create error record
        error_record = ErrorRecord(
            error_id=error_id,
            timestamp=datetime.utcnow().isoformat(),
            category=error_category,
            operation=operation,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            user_message=user_message,
            app_context={k: v for k, v in self.error_context.items()},
            additional_context=additional_context or {},
            session_id=session_id,
            stack_trace=traceback.format_exc()
        )
        
        # Log detailed error for developers
        error_log = {
            'error_id': error_id,
            'category': error_category,
            'operation': operation,
            'exception': f"{type(exception).__name__}: {str(exception)}",
            'session_id': session_id
        }
        logger.error(f"Error {error_id}: {json.dumps(error_log)}")
        
        # Store in history
        self._add_to_history(error_record)
        
        # Display user-friendly error
        self._display_user_error(error_record, show_recovery)
        
        return error_id
    
    def _generate_user_message(self, category: str, subtype: Optional[str], exception: Exception) -> str:
        """Generate user-friendly error message
        
        Args:
            category: Error category
            subtype: Error subtype
            exception: The exception
        
        Returns:
            User-friendly message
        """
        # Try to get specific message for subtype
        if subtype and category in self.USER_FRIENDLY_MESSAGES:
            category_messages = self.USER_FRIENDLY_MESSAGES[category]
            if subtype in category_messages:
                return category_messages[subtype]
        
        # Try to infer message from exception
        exception_str = str(exception).lower()
        
        if 'file' in exception_str and 'size' in exception_str:
            return "The file is too large. Please use a smaller file."
        elif 'invalid' in exception_str and 'smiles' in exception_str:
            return "The molecular structure format is invalid. Please check your input."
        elif 'memory' in exception_str or 'allocation' in exception_str:
            return "Not enough memory to complete this operation. Try a smaller molecule."
        elif 'network' in exception_str or 'connection' in exception_str:
            return "Network connection error. Please check your internet connection."
        
        # Default messages by category
        default_messages = {
            'FILE_UPLOAD': "There was a problem with the uploaded file.",
            'MOLECULAR_PROCESSING': "Could not process the molecular structure.",
            'PIPELINE_ERROR': "The pose prediction pipeline encountered an error.",
            'VALIDATION_ERROR': "The input validation failed.",
            'MEMORY_ERROR': "A memory-related error occurred.",
            'NETWORK_ERROR': "A network error occurred.",
            'CONFIGURATION_ERROR': "There is a configuration problem.",
            'CRITICAL': "A critical system error occurred."
        }
        
        return default_messages.get(category, "An unexpected error occurred.")
    
    def _display_user_error(self, error_record: ErrorRecord, show_recovery: bool):
        """Display user-friendly error message
        
        Args:
            error_record: Complete error record
            show_recovery: Whether to show recovery suggestions
        """
        if not STREAMLIT_AVAILABLE:
            print(f"Error: {error_record.user_message}")
            return
        
        # Main error message
        st.error(f"{error_record.user_message} (Reference: {error_record.error_id})")
        
        # Recovery suggestions for specific categories
        if show_recovery:
            recovery_suggestions = self._get_recovery_suggestions(error_record.category)
            if recovery_suggestions:
                st.info(f"💡 Suggestion: {recovery_suggestions}")
        
        # Support information for critical errors
        if error_record.category in ['CRITICAL', 'PIPELINE_ERROR']:
            with st.expander("Need help?", expanded=False):
                st.write("If this error persists, please contact support with:")
                st.code(f"Error Reference: {error_record.error_id}")
                st.write("This reference contains diagnostic information to help resolve the issue.")
        
        # Optional debug information
        if st.checkbox("Show technical details", key=f"debug_{error_record.error_id}"):
            st.code(f"""
Error ID: {error_record.error_id}
Category: {error_record.category}
Operation: {error_record.operation}
Exception: {error_record.exception_type}: {error_record.exception_message}
Timestamp: {error_record.timestamp}
            """)
    
    def _get_recovery_suggestions(self, category: str) -> Optional[str]:
        """Get recovery suggestions for error categories
        
        Args:
            category: Error category
        
        Returns:
            Recovery suggestion or None
        """
        suggestions = {
            'FILE_UPLOAD': "Try using a smaller file or check the file format.",
            'MOLECULAR_PROCESSING': "Verify your molecule structure or try a simpler molecule.",
            'PIPELINE_ERROR': "Check your inputs and try again, or contact support if the issue persists.",
            'VALIDATION_ERROR': "Review the input requirements and correct any formatting issues.",
            'MEMORY_ERROR': "Try processing smaller molecules or restart the application.",
            'NETWORK_ERROR': "Check your internet connection and try again.",
            'CONFIGURATION_ERROR': "Contact your administrator to check the system configuration."
        }
        
        return suggestions.get(category)
    
    def _add_to_history(self, error_record: ErrorRecord):
        """Add error record to history with size management
        
        Args:
            error_record: Error record to add
        """
        self.error_history.append(error_record)
        
        # Maintain history size limit
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
    
    def get_error_report(self, error_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve detailed error report for support
        
        Args:
            error_id: Error ID to look up
        
        Returns:
            Error report dictionary or None
        """
        for error_record in self.error_history:
            if error_record.error_id == error_id:
                return asdict(error_record)
        return None
    
    def get_recent_errors(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent error summaries
        
        Args:
            count: Number of recent errors to return
        
        Returns:
            List of error summaries
        """
        recent = self.error_history[-count:] if self.error_history else []
        return [
            {
                'error_id': error.error_id,
                'timestamp': error.timestamp,
                'category': error.category,
                'operation': error.operation,
                'user_message': error.user_message
            }
            for error in recent
        ]
    
    def clear_context(self, context_type: Optional[str] = None):
        """Clear error context
        
        Args:
            context_type: Specific context type to clear, or None to clear all
        """
        if context_type:
            self.error_context.pop(context_type, None)
        else:
            self.error_context.clear()
        
        logger.debug(f"Cleared context: {context_type or 'all'}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics
        
        Returns:
            Dictionary with error statistics
        """
        if not self.error_history:
            return {'total_errors': 0}
        
        # Count by category
        category_counts = {}
        for error in self.error_history:
            category_counts[error.category] = category_counts.get(error.category, 0) + 1
        
        # Recent errors (last hour)
        recent_threshold = datetime.utcnow().timestamp() - 3600
        recent_errors = [
            error for error in self.error_history
            if datetime.fromisoformat(error.timestamp.replace('Z', '+00:00')).timestamp() > recent_threshold
        ]
        
        return {
            'total_errors': len(self.error_history),
            'category_counts': category_counts,
            'recent_errors_count': len(recent_errors),
            'most_common_category': max(category_counts, key=category_counts.get) if category_counts else None
        }


# Convenience functions for backward compatibility
def handle_upload_error(exception: Exception, operation: str = "file upload") -> str:
    """Handle file upload errors with appropriate context"""
    manager = ContextualErrorManager()
    return manager.handle_error('FILE_UPLOAD', exception, operation)


def handle_molecular_error(exception: Exception, operation: str = "molecular processing") -> str:
    """Handle molecular processing errors with appropriate context"""
    manager = ContextualErrorManager()
    return manager.handle_error('MOLECULAR_PROCESSING', exception, operation)


def handle_pipeline_error(exception: Exception, operation: str = "pipeline execution") -> str:
    """Handle pipeline errors with appropriate context"""
    manager = ContextualErrorManager()
    return manager.handle_error('PIPELINE_ERROR', exception, operation)


# Global error manager instance
_global_error_manager = None

def get_error_manager() -> ContextualErrorManager:
    """Get global error manager instance"""
    global _global_error_manager
    if _global_error_manager is None:
        _global_error_manager = ContextualErrorManager()
    return _global_error_manager 