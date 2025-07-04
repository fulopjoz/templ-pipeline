"""
Processing Pipeline Guards

Protective measures for molecular processing pipeline.
"""

from typing import Optional, Any, Dict
from rdkit import Chem
import logging

logger = logging.getLogger(__name__)


class PipelineGuards:
    """Guards for protecting molecular processing pipeline."""

    @staticmethod
    def validate_input_molecule(mol: Chem.Mol) -> Dict[str, Any]:
        """Validate input molecule before processing."""
        validation = {"valid": True, "issues": [], "severity": "low"}

        if mol is None:
            validation["valid"] = False
            validation["issues"].append("Molecule is None")
            validation["severity"] = "critical"
            return validation

        # Check atom count
        if mol.GetNumAtoms() == 0:
            validation["valid"] = False
            validation["issues"].append("Molecule has no atoms")
            validation["severity"] = "critical"

        # Check for unreasonable atom counts
        if mol.GetNumAtoms() > 1000:
            validation["issues"].append("Molecule has unusually high atom count")
            validation["severity"] = "warning"

        # Test SMILES generation
        try:
            smiles = Chem.MolToSmiles(mol)
            if not smiles:
                validation["issues"].append("Cannot generate SMILES")
                validation["severity"] = "high"
        except Exception as e:
            validation["issues"].append(f"SMILES generation failed: {e}")
            validation["severity"] = "high"

        return validation

    @staticmethod
    def safe_sanitize(mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Safely sanitize molecule with fallback options."""
        if mol is None:
            return None

        # Try standard sanitization
        try:
            mol_copy = Chem.Mol(mol)
            Chem.SanitizeMol(mol_copy)
            return mol_copy
        except:
            pass

        # Try without kekulization
        try:
            mol_copy = Chem.Mol(mol)
            Chem.SanitizeMol(
                mol_copy,
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
            )
            return mol_copy
        except:
            pass

        # Last resort - return original
        logger.warning("Sanitization failed, returning original molecule")
        return mol
