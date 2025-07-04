"""
Skip management system for TEMPL pipeline.

Provides functionality to gracefully handle molecules that cannot be processed
by the pipeline, with proper tracking and reporting of skip reasons.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


class SkipReason(Enum):
    """Enumeration of reasons why a molecule might be skipped."""
    LARGE_PEPTIDE = "large_peptide"
    ORGANOMETALLIC_COMPLEX = "organometallic_complex"
    RHENIUM_COMPLEX = "rhenium_complex"
    INVALID_MOLECULE = "invalid_molecule"
    EMBEDDING_FAILED = "embedding_failed"
    TEMPLATE_UNAVAILABLE = "template_unavailable"
    CONFORMER_GENERATION_FAILED = "conformer_generation_failed"
    SCORING_FAILED = "scoring_failed"
    OTHER = "other"


@dataclass
class SkipRecord:
    """Record of a skipped molecule with detailed information."""
    molecule_id: str
    reason: SkipReason
    message: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None


class MoleculeSkipException(Exception):
    """Exception that signals a molecule should be skipped gracefully."""
    
    def __init__(self, reason: SkipReason, message: str, details: Optional[Dict[str, Any]] = None):
        self.reason = reason
        self.message = message
        self.details = details or {}
        super().__init__(message)


class SkipManager:
    """Manages skipped molecules and provides reporting functionality."""
    
    def __init__(self):
        self.skipped_records: List[SkipRecord] = []
        self._skip_counts: Dict[SkipReason, int] = {}
    
    def skip_molecule(self, molecule_id: str, reason: SkipReason, 
                     message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Record a skipped molecule."""
        record = SkipRecord(
            molecule_id=molecule_id,
            reason=reason,
            message=message,
            timestamp=datetime.now().isoformat(),
            details=details or {}
        )
        
        self.skipped_records.append(record)
        self._skip_counts[reason] = self._skip_counts.get(reason, 0) + 1
        
        # Log the skip with appropriate level
        if reason in [SkipReason.LARGE_PEPTIDE, SkipReason.ORGANOMETALLIC_COMPLEX]:
            logger.info(f"SKIP {molecule_id}: {message}")
        else:
            logger.warning(f"SKIP {molecule_id}: {message}")
    
    def get_skip_summary(self) -> Dict[str, Any]:
        """Get summary of all skipped molecules."""
        total_skipped = len(self.skipped_records)
        
        summary = {
            "total_skipped": total_skipped,
            "skip_counts_by_reason": dict(self._skip_counts),
            "skip_rate_by_reason": {},
            "recent_skips": self.skipped_records[-5:] if self.skipped_records else []
        }
        
        # Calculate percentages if we have skips
        if total_skipped > 0:
            for reason, count in self._skip_counts.items():
                summary["skip_rate_by_reason"][reason.value] = (count / total_skipped) * 100
        
        return summary
    
    def get_skipped_by_reason(self, reason: SkipReason) -> List[SkipRecord]:
        """Get all molecules skipped for a specific reason."""
        return [record for record in self.skipped_records if record.reason == reason]
    
    def clear_skips(self) -> None:
        """Clear all skip records."""
        self.skipped_records.clear()
        self._skip_counts.clear()
    
    def save_skip_report(self, output_path: str) -> bool:
        """Save skip report to file."""
        try:
            import json
            summary = self.get_skip_summary()
            
            # Convert SkipRecord objects to dicts for JSON serialization
            detailed_records = []
            for record in self.skipped_records:
                detailed_records.append({
                    "molecule_id": record.molecule_id,
                    "reason": record.reason.value,
                    "message": record.message,
                    "timestamp": record.timestamp,
                    "details": record.details
                })
            
            report_data = {
                "summary": summary,
                "detailed_records": detailed_records
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Skip report saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save skip report: {e}")
            return False


# Global skip manager instance
_global_skip_manager = SkipManager()


def skip_molecule(molecule_id: str, reason: SkipReason, message: str, 
                 details: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function to skip a molecule using the global manager."""
    _global_skip_manager.skip_molecule(molecule_id, reason, message, details)


def raise_skip_exception(reason: SkipReason, message: str, 
                        details: Optional[Dict[str, Any]] = None) -> None:
    """Raise a skip exception that can be caught and handled gracefully."""
    raise MoleculeSkipException(reason, message, details)


def get_skip_manager() -> SkipManager:
    """Get the global skip manager instance."""
    return _global_skip_manager


def log_skip_summary() -> None:
    """Log a summary of all skips."""
    summary = _global_skip_manager.get_skip_summary()
    total = summary["total_skipped"]
    
    if total == 0:
        logger.info("No molecules were skipped")
        return
    
    logger.info(f"SKIP SUMMARY: {total} molecules skipped")
    for reason, count in summary["skip_counts_by_reason"].items():
        percentage = summary["skip_rate_by_reason"][reason]
        logger.info(f"  {reason}: {count} molecules ({percentage:.1f}%)")


def create_validation_skip_wrapper(validation_func):
    """Decorator to convert validation failures into skip exceptions."""
    def wrapper(mol, mol_name="unknown", **kwargs):
        is_valid, message = validation_func(mol, mol_name, **kwargs)
        if not is_valid:
            # Determine skip reason from message content
            if "large peptide" in message.lower():
                reason = SkipReason.LARGE_PEPTIDE
            elif "rhenium" in message.lower():
                reason = SkipReason.RHENIUM_COMPLEX
            elif "organometallic" in message.lower():
                reason = SkipReason.ORGANOMETALLIC_COMPLEX
            else:
                reason = SkipReason.INVALID_MOLECULE
            
            raise_skip_exception(reason, message, {"molecule_name": mol_name})
        
        return is_valid, message
    
    return wrapper