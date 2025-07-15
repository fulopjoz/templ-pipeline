#!/usr/bin/env python3
"""
Error Tracking and Missing PDB Handling for TEMPL Benchmarks

This module provides comprehensive error tracking, missing PDB handling,
and graceful degradation for benchmark operations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from datetime import datetime
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class MissingPDBRecord:
    """Record for a missing PDB entry."""
    pdb_id: str
    error_type: str  # "file_not_found", "load_failed", "invalid_structure", etc.
    error_message: str
    component: str  # "protein", "ligand", "template", "embedding"
    timestamp: str
    context: Dict[str, Any] = None  # Additional context info
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkErrorSummary:
    """Summary of errors encountered during benchmark."""
    total_targets: int
    successful_targets: int
    failed_targets: int
    missing_pdbs: Dict[str, List[MissingPDBRecord]]
    error_categories: Dict[str, int]
    error_timeline: List[Tuple[str, str]]  # (timestamp, error_type)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_targets": self.total_targets,
            "successful_targets": self.successful_targets,
            "failed_targets": self.failed_targets,
            "missing_pdbs": {k: [r.to_dict() for r in v] for k, v in self.missing_pdbs.items()},
            "error_categories": self.error_categories,
            "error_timeline": self.error_timeline
        }


class BenchmarkErrorTracker:
    """Comprehensive error tracking and missing PDB handling for benchmarks."""
    
    def __init__(self, workspace_dir: Optional[Path] = None):
        self.workspace_dir = workspace_dir or Path.cwd()
        self.missing_pdbs: Dict[str, List[MissingPDBRecord]] = defaultdict(list)
        self.error_counts: Counter = Counter()
        self.error_timeline: List[Tuple[str, str]] = []
        self.successful_targets: Set[str] = set()
        self.failed_targets: Set[str] = set()
        
        # Create error tracking directory
        self.error_dir = self.workspace_dir / "error_tracking"
        self.error_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize error log file
        self.error_log_file = self.error_dir / "benchmark_errors.jsonl"
        
    def record_missing_pdb(
        self, 
        pdb_id: str, 
        error_type: str, 
        error_message: str, 
        component: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a missing PDB with comprehensive tracking."""
        timestamp = datetime.now().isoformat()
        
        record = MissingPDBRecord(
            pdb_id=pdb_id,
            error_type=error_type,
            error_message=error_message,
            component=component,
            timestamp=timestamp,
            context=context or {}
        )
        
        self.missing_pdbs[pdb_id].append(record)
        self.error_counts[error_type] += 1
        self.error_timeline.append((timestamp, error_type))
        
        # Log to file immediately for persistence
        self._log_error_to_file(record)
        
        logger.debug(f"Recorded missing PDB {pdb_id}: {error_type} in {component}")
        
    def record_target_success(self, pdb_id: str) -> None:
        """Record a successful target processing."""
        self.successful_targets.add(pdb_id)
        
    def record_target_failure(self, pdb_id: str, error_message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Record a failed target processing."""
        self.failed_targets.add(pdb_id)
        if "ligand_not_found" in error_message.lower():
            self.record_missing_pdb(
                pdb_id,
                "ligand_not_found",
                error_message,
                "ligand",
                context=context
            )
        else:
            self.record_missing_pdb(
                pdb_id, 
                "target_processing_failed", 
                error_message, 
                "pipeline",
                context=context
            )
        
    def get_missing_pdbs_by_component(self) -> Dict[str, Set[str]]:
        """Get missing PDBs grouped by component type."""
        by_component = defaultdict(set)
        for pdb_id, records in self.missing_pdbs.items():
            for record in records:
                by_component[record.component].add(pdb_id)
        return dict(by_component)
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        total_targets = len(self.successful_targets) + len(self.failed_targets)
        
        return {
            "total_targets_attempted": total_targets,
            "successful_targets": len(self.successful_targets),
            "failed_targets": len(self.failed_targets),
            "success_rate": len(self.successful_targets) / total_targets * 100 if total_targets > 0 else 0,
            "unique_missing_pdbs": len(self.missing_pdbs),
            "total_error_instances": sum(self.error_counts.values()),
            "error_categories": dict(self.error_counts),
            "missing_by_component": self.get_missing_pdbs_by_component(),
        }
        
    def generate_summary_report(self) -> BenchmarkErrorSummary:
        """Generate a comprehensive error summary report."""
        total_targets = len(self.successful_targets) + len(self.failed_targets)
        
        return BenchmarkErrorSummary(
            total_targets=total_targets,
            successful_targets=len(self.successful_targets),
            failed_targets=len(self.failed_targets),
            missing_pdbs=dict(self.missing_pdbs),
            error_categories=dict(self.error_counts),
            error_timeline=self.error_timeline.copy()
        )
        
    def save_error_report(self, filename: Optional[str] = None) -> Path:
        """Save comprehensive error report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_error_report_{timestamp}.json"
            
        report_path = self.error_dir / filename
        summary = self.generate_summary_report()
        
        with open(report_path, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
            
        logger.info(f"Error report saved to {report_path}")
        return report_path
        
    def _log_error_to_file(self, record: MissingPDBRecord) -> None:
        """Log error record to JSONL file for persistence."""
        try:
            with open(self.error_log_file, 'a') as f:
                json.dump(record.to_dict(), f)
                f.write('\n')
        except Exception as e:
            logger.warning(f"Failed to log error to file: {e}")
            
    def create_missing_pdb_recovery_plan(self) -> Dict[str, List[str]]:
        """Create a recovery plan for missing PDBs."""
        recovery_plan = {
            "file_not_found": [],
            "load_failed": [],
            "invalid_structure": [],
            "network_download": [],
            "alternative_sources": []
        }
        
        for pdb_id, records in self.missing_pdbs.items():
            primary_error = records[0].error_type
            
            if "file_not_found" in primary_error:
                recovery_plan["file_not_found"].append(pdb_id)
            elif "load_failed" in primary_error:
                recovery_plan["load_failed"].append(pdb_id)
            elif "invalid" in primary_error:
                recovery_plan["invalid_structure"].append(pdb_id)
            else:
                recovery_plan["alternative_sources"].append(pdb_id)
                
        # Remove empty categories
        recovery_plan = {k: v for k, v in recovery_plan.items() if v}
        
        return recovery_plan
        
    def print_error_summary(self, include_skips: bool = True) -> None:
        """Print a formatted error summary to console."""
        stats = self.get_error_statistics()
        
        # Get skip information if available
        skip_summary = {}
        if include_skips:
            try:
                from ..core.skip_manager import get_skip_manager
                skip_manager = get_skip_manager()
                skip_summary = skip_manager.get_skip_summary()
            except ImportError:
                skip_summary = {}
        
        print("\n" + "="*60)
        print("BENCHMARK ERROR SUMMARY")
        print("="*60)
        print(f"Total targets attempted: {stats['total_targets_attempted']}")
        print(f"Successful targets: {stats['successful_targets']}")
        print(f"Failed targets: {stats['failed_targets']}")
        
        # Include skip information if available
        if skip_summary.get('total_skipped', 0) > 0:
            skipped_count = skip_summary['total_skipped']
            print(f"Skipped targets: {skipped_count}")
            adjusted_total = stats['total_targets_attempted'] + skipped_count
            adjusted_success_rate = (stats['successful_targets'] / adjusted_total) * 100 if adjusted_total > 0 else 0
            print(f"Success rate (including skips): {adjusted_success_rate:.1f}%")
            print(f"Success rate (excluding skips): {stats['success_rate']:.1f}%")
        else:
            print(f"Success rate: {stats['success_rate']:.1f}%")
        
        print(f"Unique missing PDBs: {stats['unique_missing_pdbs']}")
        print(f"Total error instances: {stats['total_error_instances']}")
        
        # Show skip breakdown if available
        if skip_summary.get('total_skipped', 0) > 0:
            print("\nSkip Categories:")
            for reason, count in skip_summary.get('skip_counts_by_reason', {}).items():
                percentage = skip_summary.get('skip_rate_by_reason', {}).get(reason, 0)
                print(f"  {reason}: {count} targets ({percentage:.1f}% of skips)")
        
        if stats['error_categories']:
            print("\nError Categories:")
            for error_type, count in sorted(stats['error_categories'].items()):
                print(f"  {error_type}: {count}")
                
        if stats['missing_by_component']:
            print("\nMissing PDBs by Component:")
            for component, pdbs in stats['missing_by_component'].items():
                print(f"  {component}: {len(pdbs)} PDBs")
                if len(pdbs) <= 10:
                    print(f"    {', '.join(sorted(pdbs))}")
                else:
                    sample = sorted(list(pdbs))[:5]
                    print(f"    {', '.join(sample)} ... (+{len(pdbs)-5} more)")
                    
        print("="*60)


def create_graceful_pdb_loader(error_tracker: BenchmarkErrorTracker):
    """Create PDB loading functions with graceful error handling."""
    
    def load_protein_gracefully(pdb_id: str, data_dir: Path) -> Optional[str]:
        """Load protein file with graceful error handling."""
        try:
            # Import here to avoid circular imports
            from templ_pipeline.core.utils import get_protein_file_paths
            
            search_paths = get_protein_file_paths(pdb_id, data_dir)
            
            for protein_file in search_paths:
                if protein_file.exists():
                    return str(protein_file)
                    
            # No protein file found
            error_tracker.record_missing_pdb(
                pdb_id,
                "protein_file_not_found",
                f"Protein file not found in any search path",
                "protein",
                {"search_paths": [str(p) for p in search_paths]}
            )
            return None
            
        except Exception as e:
            error_tracker.record_missing_pdb(
                pdb_id,
                "protein_load_exception",
                str(e),
                "protein",
                {"exception_type": type(e).__name__}
            )
            return None
            
    def load_ligand_gracefully(pdb_id: str, molecules: List[Any]) -> Tuple[Optional[str], Optional[Any]]:
        """Load ligand with graceful error handling."""
        try:
            from templ_pipeline.core.utils import find_ligand_by_pdb_id
            
            smiles, mol = find_ligand_by_pdb_id(pdb_id, molecules)
            
            if smiles is None or mol is None:
                error_tracker.record_missing_pdb(
                    pdb_id,
                    "ligand_not_found",
                    f"Ligand not found in molecule database",
                    "ligand",
                    {"database_size": len(molecules)}
                )
                return None, None
                
            return smiles, mol
            
        except Exception as e:
            error_tracker.record_missing_pdb(
                pdb_id,
                "ligand_load_exception",
                str(e),
                "ligand",
                {"exception_type": type(e).__name__}
            )
            return None, None
            
    def load_embedding_gracefully(pdb_id: str, embedding_manager) -> Optional[Any]:
        """Load embedding with graceful error handling."""
        try:
            if not embedding_manager.has_embedding(pdb_id):
                error_tracker.record_missing_pdb(
                    pdb_id,
                    "embedding_not_found",
                    f"Embedding not found in database",
                    "embedding"
                )
                return None
                
            embedding, chain_id = embedding_manager.get_embedding(pdb_id)
            
            if embedding is None:
                error_tracker.record_missing_pdb(
                    pdb_id,
                    "embedding_load_failed",
                    f"Embedding found but failed to load",
                    "embedding",
                    {"chain_id": chain_id}
                )
                return None
                
            return embedding
            
        except Exception as e:
            error_tracker.record_missing_pdb(
                pdb_id,
                "embedding_load_exception",
                str(e),
                "embedding",
                {"exception_type": type(e).__name__}
            )
            return None
    
    return {
        "load_protein": load_protein_gracefully,
        "load_ligand": load_ligand_gracefully,
        "load_embedding": load_embedding_gracefully
    }


# Convenience function for integration
def setup_error_tracking(workspace_dir: Path) -> Tuple[BenchmarkErrorTracker, Dict[str, Any]]:
    """Setup error tracking and return tracker plus graceful loaders."""
    error_tracker = BenchmarkErrorTracker(workspace_dir)
    graceful_loaders = create_graceful_pdb_loader(error_tracker)
    
    return error_tracker, graceful_loaders