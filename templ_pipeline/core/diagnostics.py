"""
Diagnostic and error tracking utilities for the TEMPL pipeline.
"""
import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging


class PipelineErrorTracker:
    """Lightweight error tracking context manager for pipeline failures."""
    
    _current_pdb_id = None
    _current_stage = None
    _errors = {}
    _start_time = None
    
    def __init__(self, pdb_id: str, stage: str):
        self.pdb_id = pdb_id
        self.stage = stage
        self.start_time = time.time()
    
    def __enter__(self):
        PipelineErrorTracker._current_pdb_id = self.pdb_id
        PipelineErrorTracker._current_stage = self.stage
        PipelineErrorTracker._start_time = self.start_time
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_info = {
                'stage': self.stage,
                'error_type': exc_type.__name__,
                'error_message': str(exc_val),
                'timestamp': time.time(),
                'duration': time.time() - self.start_time
            }
            
            if self.pdb_id not in PipelineErrorTracker._errors:
                PipelineErrorTracker._errors[self.pdb_id] = []
            PipelineErrorTracker._errors[self.pdb_id].append(error_info)
        
        PipelineErrorTracker._current_pdb_id = None
        PipelineErrorTracker._current_stage = None
        PipelineErrorTracker._start_time = None
        return False  # Don't suppress exceptions
    
    @classmethod
    def get_error_summary(cls) -> Dict[str, Any]:
        """Get summary of all tracked errors."""
        total_errors = sum(len(errors) for errors in cls._errors.values())
        error_by_stage = {}
        error_by_type = {}
        
        for pdb_id, errors in cls._errors.items():
            for error in errors:
                stage = error['stage']
                error_type = error['error_type']
                
                error_by_stage[stage] = error_by_stage.get(stage, 0) + 1
                error_by_type[error_type] = error_by_type.get(error_type, 0) + 1
        
        return {
            'total_errors': total_errors,
            'failed_pdbs': len(cls._errors),
            'errors_by_stage': error_by_stage,
            'errors_by_type': error_by_type,
            'detailed_errors': cls._errors
        }
    
    @classmethod
    def save_error_report(cls, output_dir: str, target_pdb: str) -> Optional[str]:
        """Save error report to JSON file."""
        if not cls._errors:
            return None
        
        report_path = Path(output_dir) / f"{target_pdb}_error_report.json"
        try:
            with open(report_path, 'w') as f:
                json.dump(cls.get_error_summary(), f, indent=2)
            return str(report_path)
        except Exception as e:
            logging.error(f"Failed to save error report: {e}")
            return None
    
    @classmethod
    def clear_errors(cls):
        """Clear all tracked errors."""
        cls._errors.clear()


class ProteinAlignmentTracker:
    """Track protein alignment success/failure patterns for investigation."""
    
    _alignment_logs = []
    
    @classmethod
    def track_alignment_attempt(cls, pdb_id: str, stage: str, success: bool, details: Dict):
        """Track an alignment attempt with context."""
        log_entry = {
            'pdb_id': pdb_id,
            'stage': stage,
            'success': success,
            'timestamp': time.time(),
            'details': details
        }
        cls._alignment_logs.append(log_entry)
    
    @classmethod
    def get_alignment_summary(cls) -> Dict[str, Any]:
        """Get summary of alignment attempts."""
        total_attempts = len(cls._alignment_logs)
        successful = sum(1 for log in cls._alignment_logs if log['success'])
        failed = total_attempts - successful
        
        success_by_stage = {}
        failure_by_stage = {}
        
        for log in cls._alignment_logs:
            stage = log['stage']
            if log['success']:
                success_by_stage[stage] = success_by_stage.get(stage, 0) + 1
            else:
                failure_by_stage[stage] = failure_by_stage.get(stage, 0) + 1
        
        return {
            'total_attempts': total_attempts,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total_attempts if total_attempts > 0 else 0,
            'success_by_stage': success_by_stage,
            'failure_by_stage': failure_by_stage,
            'detailed_logs': cls._alignment_logs
        }
    
    @classmethod
    def save_alignment_report(cls, output_dir: str, target_pdb: str) -> Optional[str]:
        """Save alignment report to JSON file."""
        if not cls._alignment_logs:
            return None
        
        report_path = Path(output_dir) / f"{target_pdb}_alignment_report.json"
        try:
            with open(report_path, 'w') as f:
                json.dump(cls.get_alignment_summary(), f, indent=2)
            return str(report_path)
        except Exception as e:
            logging.error(f"Failed to save alignment report: {e}")
            return None
    
    @classmethod
    def clear_logs(cls):
        """Clear all alignment logs."""
        cls._alignment_logs.clear() 