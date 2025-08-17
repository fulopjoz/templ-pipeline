#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Skip tracking system for TEMPL pipeline benchmarking.

This module provides comprehensive tracking of skipped molecules during benchmarking
with detailed reasons and statistics generation.
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

# Optional RDKit import for molecular information
try:
    from rdkit.Chem import rdMolDescriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class SkipRecord:
    """Record for a skipped molecule."""

    pdb_id: str
    reason: str
    details: str
    timestamp: float
    molecule_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class BenchmarkSkipTracker:
    """Track skipped molecules during benchmarking with detailed reporting."""

    # Standard skip reasons with user-friendly descriptions
    SKIP_REASONS = {
        "large_peptide": "Large peptide (>8 residues)",
        "organometallic": "Organometallic complex",
        "rhenium_complex": "Rhenium complex",
        "invalid_molecule": "Invalid molecule structure",
        "mcs_timeout": "MCS calculation timeout",
        "mcs_failure": "MCS calculation failed",
        "mcs_calculation_failed": "Maximum Common Substructure calculation failed",
        "conformer_generation_failed": "Conformer generation failed",
        "template_processing_failed": "Template processing failed",
        "embedding_failed": "Protein embedding failed",
        "validation_failed": "Molecule validation failed",
        "pose_generation_failed": "Pose generation failed",
        "rmsd_calculation_failed": "RMSD calculation failed",
        "alignment_failed": "RDKit molecular alignment failed",
        "index_mismatch_error": "MCS index length mismatch during alignment",
        "pipeline_general_error": "General pipeline error",
        "target_processing_failed": "Target processing failed",
        "file_not_found": "Required file not found",
        "parsing_error": "Structure parsing error",
        "sanitization_failed": "Molecule sanitization failed",
        "other": "Other reason",
    }

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize skip tracker.

        Args:
            output_dir: Directory to save skip tracking files
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.skip_records: List[SkipRecord] = []
        self.skip_counts: Dict[str, int] = defaultdict(int)
        self.skip_file = self.output_dir / "benchmark_skips.jsonl"

        # Initialize skip file
        self._init_skip_file()

    def _init_skip_file(self):
        """Initialize the skip tracking file."""
        try:
            # Create file with header if it doesn't exist
            if not self.skip_file.exists():
                with open(self.skip_file, "w") as f:
                    header = {
                        "header": "TEMPL Pipeline Benchmark Skip Tracking",
                        "timestamp": time.time(),
                        "format": "Each line contains a JSON object representing a skipped molecule",
                    }
                    f.write(json.dumps(header) + "\n")
                logger.info(f"Initialized skip tracking file: {self.skip_file}")
        except Exception as e:
            logger.error(f"Failed to initialize skip file: {e}")

    def track_skip(
        self,
        pdb_id: str,
        reason: str,
        details: str,
        molecule_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Record a skipped molecule with detailed reason.

        Args:
            pdb_id: PDB ID of the skipped molecule
            reason: Skip reason (should be from SKIP_REASONS keys)
            details: Detailed description of why molecule was skipped
            molecule_info: Additional molecular information (atom count, MW, etc.)
        """
        # Normalize reason
        normalized_reason = reason if reason in self.SKIP_REASONS else "other"

        # Create skip record
        skip_record = SkipRecord(
            pdb_id=pdb_id,
            reason=normalized_reason,
            details=details,
            timestamp=time.time(),
            molecule_info=molecule_info or {},
        )

        # Add to tracking
        self.skip_records.append(skip_record)
        self.skip_counts[normalized_reason] += 1

        # Log the skip
        reason_desc = self.SKIP_REASONS.get(normalized_reason, normalized_reason)
        logger.info(f"SKIPPED {pdb_id}: {reason_desc} - {details}")

        # Write to file immediately for real-time tracking
        self._write_skip_record(skip_record)

    def _write_skip_record(self, record: SkipRecord):
        """Write a single skip record to the tracking file."""
        try:
            with open(self.skip_file, "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write skip record for {record.pdb_id}: {e}")

    def get_skip_count(self, reason: Optional[str] = None) -> int:
        """
        Get count of skipped molecules.

        Args:
            reason: Specific skip reason to count, or None for total

        Returns:
            Number of skipped molecules
        """
        if reason is None:
            return len(self.skip_records)
        return self.skip_counts.get(reason, 0)

    def get_skipped_pdb_ids(self, reason: Optional[str] = None) -> Set[str]:
        """
        Get set of skipped PDB IDs.

        Args:
            reason: Specific skip reason to filter by, or None for all

        Returns:
            Set of PDB IDs that were skipped
        """
        if reason is None:
            return {record.pdb_id for record in self.skip_records}
        return {
            record.pdb_id for record in self.skip_records if record.reason == reason
        }

    def generate_skip_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive skip statistics.

        Returns:
            Dictionary containing detailed skip statistics
        """
        total_skipped = len(self.skip_records)

        # Count by reason
        reason_breakdown = {}
        for reason, count in self.skip_counts.items():
            reason_desc = self.SKIP_REASONS.get(reason, reason)
            reason_breakdown[reason] = {
                "count": count,
                "description": reason_desc,
                "percentage": (count / total_skipped * 100) if total_skipped > 0 else 0,
            }

        # Top skip reasons
        top_reasons = sorted(
            self.skip_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Recent skips (last 10)
        recent_skips = [
            {
                "pdb_id": record.pdb_id,
                "reason": self.SKIP_REASONS.get(record.reason, record.reason),
                "details": record.details,
            }
            for record in self.skip_records[-10:]
        ]

        summary = {
            "total_skipped": total_skipped,
            "skip_reasons": reason_breakdown,
            "top_skip_reasons": [
                {
                    "reason": reason,
                    "description": self.SKIP_REASONS.get(reason, reason),
                    "count": count,
                }
                for reason, count in top_reasons
            ],
            "recent_skips": recent_skips,
            "tracking_file": str(self.skip_file),
            "generated_at": time.time(),
        }

        return summary

    def save_summary(self, filename: Optional[str] = None) -> Path:
        """
        Save skip summary to JSON file.

        Args:
            filename: Output filename, defaults to benchmark_skip_summary.json

        Returns:
            Path to saved summary file
        """
        if filename is None:
            filename = "benchmark_skip_summary.json"

        summary_file = self.output_dir / filename
        summary = self.generate_skip_summary()

        try:
            with open(summary_file, "w") as f:
                f.write(json.dumps(summary, indent=2))
            logger.info(f"Skip summary saved to: {summary_file}")
            return summary_file
        except Exception as e:
            logger.error(f"Failed to save skip summary: {e}")
            raise

    def print_summary(self, detailed: bool = False):
        """
        Print skip summary to console.

        Args:
            detailed: Whether to print detailed breakdown
        """
        summary = self.generate_skip_summary()
        total = summary["total_skipped"]

        print(f"\n{'='*60}")
        print("BENCHMARK SKIP SUMMARY")
        print(f"{'='*60}")
        print(f"Total molecules skipped: {total}")

        if total == 0:
            print("No molecules were skipped!")
            return

        print("\nTop skip reasons:")
        for item in summary["top_skip_reasons"]:
            count = item["count"]
            desc = item["description"]
            pct = (count / total * 100) if total > 0 else 0
            print(f"  {desc}: {count} ({pct:.1f}%)")

        if detailed and summary["recent_skips"]:
            print("\nRecent skips:")
            for skip in summary["recent_skips"]:
                print(f"  {skip['pdb_id']}: {skip['reason']} - {skip['details']}")

        print(f"\nDetailed tracking file: {summary['tracking_file']}")
        print(f"{'='*60}\n")

    def get_formatted_skip_statistics(self) -> Dict[str, Any]:
        """
        Get formatted skip statistics for logging and progress reporting.

        Returns:
            Dictionary with formatted statistics suitable for logging
        """
        total_skipped = len(self.skip_records)
        if total_skipped == 0:
            return {
                "total_skipped": 0,
                "breakdown": {},
                "formatted_summary": "No molecules skipped",
            }

        # Get top 5 skip reasons
        top_reasons = sorted(
            self.skip_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        # Create formatted breakdown
        breakdown = {}
        formatted_lines = []

        for reason, count in top_reasons:
            description = self.SKIP_REASONS.get(reason, reason)
            percentage = (count / total_skipped) * 100
            breakdown[reason] = {
                "count": count,
                "description": description,
                "percentage": percentage,
            }
            formatted_lines.append(f"  {description}: {count} ({percentage:.1f}%)")

        formatted_summary = f"Total skipped: {total_skipped}\n" + "\n".join(
            formatted_lines
        )

        return {
            "total_skipped": total_skipped,
            "breakdown": breakdown,
            "formatted_summary": formatted_summary,
            "top_reasons": top_reasons,
        }

    def log_skip_statistics(self, logger_instance=None):
        """
        Log skip statistics to the provided logger.

        Args:
            logger_instance: Logger instance to use, defaults to module logger
        """
        log = logger_instance if logger_instance else logger
        stats = self.get_formatted_skip_statistics()

        if stats["total_skipped"] > 0:
            log.info("Skip Statistics Summary:")
            log.info(f"Total molecules skipped: {stats['total_skipped']}")
            for reason, data in stats["breakdown"].items():
                log.info(
                    f"  {data['description']}: {data['count']} ({data['percentage']:.1f}%)"
                )
        else:
            log.info("No molecules were skipped during processing")

    def load_existing_skips(self) -> int:
        """
        Load existing skip records from file.

        Returns:
            Number of skip records loaded
        """
        if not self.skip_file.exists():
            return 0

        loaded_count = 0
        try:
            with open(self.skip_file, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        # Skip header line
                        if "header" in data:
                            continue

                        # Create SkipRecord from data
                        record = SkipRecord(
                            pdb_id=data["pdb_id"],
                            reason=data["reason"],
                            details=data["details"],
                            timestamp=data["timestamp"],
                            molecule_info=data.get("molecule_info", {}),
                        )

                        self.skip_records.append(record)
                        self.skip_counts[record.reason] += 1
                        loaded_count += 1

                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Failed to parse skip record line: {e}")
                        continue

            logger.info(f"Loaded {loaded_count} existing skip records")
            return loaded_count

        except Exception as e:
            logger.error(f"Failed to load existing skip records: {e}")
            return 0


def create_molecule_info(mol, smiles: str = "") -> Dict[str, Any]:
    """
    Create molecular information dictionary for skip tracking.

    Args:
        mol: RDKit molecule object (can be None)
        smiles: SMILES string (optional)

    Returns:
        Dictionary with molecular information
    """
    info = {}

    if mol is not None and RDKIT_AVAILABLE:
        try:
            info["num_atoms"] = mol.GetNumAtoms()
            info["num_heavy_atoms"] = mol.GetNumHeavyAtoms()
            info["num_bonds"] = mol.GetNumBonds()
            info["molecular_weight"] = rdMolDescriptors.CalcExactMolWt(mol)

            # Ring information
            ring_info = mol.GetRingInfo()
            info["num_rings"] = ring_info.NumRings()
            info["num_aromatic_rings"] = len(
                [
                    ring
                    for ring in ring_info.AtomRings()
                    if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
                ]
            )
        except Exception as e:
            logger.debug(f"Failed to extract molecular information: {e}")
    elif mol is not None and not RDKIT_AVAILABLE:
        logger.debug("RDKit not available, skipping molecular information extraction")

    if smiles:
        info["smiles"] = smiles

    return info
