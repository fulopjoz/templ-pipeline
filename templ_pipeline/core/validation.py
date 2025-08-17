# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Data validation utilities for TEMPL pipeline.

This module provides comprehensive validation functionality including:
- Dataset split validation
- Molecular data validation
- Database consistency checking
- Validation framework for testing
"""

import os
import logging
import json
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)


class SplitDataValidator:
    """Validates split data against available embeddings."""

    def __init__(self, embedding_path: str, splits_dir: str):
        """
        Initialize validator with embedding and splits paths.
        
        Args:
            embedding_path: Path to embedding NPZ file
            splits_dir: Directory containing split files
        """
        self.embedding_path = embedding_path
        self.splits_dir = splits_dir
        self._available_pdbs = None
        self._validated_splits = None

    def _load_available_pdbs(self) -> Set[str]:
        """Load available PDB IDs from embedding database."""
        if self._available_pdbs is None:
            try:
                data = np.load(self.embedding_path, allow_pickle=True)
                self._available_pdbs = set(data["pdb_ids"])
                logger.info(
                    f"Loaded {len(self._available_pdbs)} available PDB embeddings"
                )
            except Exception as e:
                logger.error(f"Failed to load embeddings: {e}")
                self._available_pdbs = set()
        return self._available_pdbs

    def _load_split_file(self, filename: str) -> Set[str]:
        """Load PDB IDs from split file."""
        filepath = os.path.join(self.splits_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"Split file not found: {filepath}")
            return set()

        with open(filepath) as f:
            return {line.strip().lower() for line in f if line.strip()}

    def get_validated_splits(self) -> Dict[str, Set[str]]:
        """Get validated splits with only available PDB IDs."""
        if self._validated_splits is None:
            available_pdbs = self._load_available_pdbs()
            
            splits = {}
            for split_name in ["train", "val", "test"]:
                filename = f"{split_name}_pdbs.txt"
                split_pdbs = self._load_split_file(filename)
                
                # Only include PDbs that have embeddings
                validated_pdbs = split_pdbs.intersection(available_pdbs)
                splits[split_name] = validated_pdbs
                
                logger.info(
                    f"{split_name}: {len(validated_pdbs)}/{len(split_pdbs)} PDbs have embeddings"
                )
            
            self._validated_splits = splits
        
        return self._validated_splits

    def validate_split_completeness(self) -> Dict[str, Any]:
        """Validate that splits are complete and non-overlapping."""
        splits = self.get_validated_splits()
        
        # Check for overlaps
        train_pdbs = splits.get("train", set())
        val_pdbs = splits.get("val", set())
        test_pdbs = splits.get("test", set())
        
        train_val_overlap = train_pdbs.intersection(val_pdbs)
        train_test_overlap = train_pdbs.intersection(test_pdbs)
        val_test_overlap = val_pdbs.intersection(test_pdbs)
        
        total_unique = len(train_pdbs.union(val_pdbs).union(test_pdbs))
        
        validation_result = {
            "total_unique_pdbs": total_unique,
            "train_count": len(train_pdbs),
            "val_count": len(val_pdbs),
            "test_count": len(test_pdbs),
            "has_overlaps": len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0,
            "overlaps": {
                "train_val": list(train_val_overlap),
                "train_test": list(train_test_overlap),
                "val_test": list(val_test_overlap),
            },
            "is_valid": True,
        }
        
        # Check for critical issues
        if validation_result["has_overlaps"]:
            validation_result["is_valid"] = False
            logger.error("Dataset splits have overlapping PDB IDs")
        
        if any(count == 0 for count in [len(train_pdbs), len(val_pdbs), len(test_pdbs)]):
            validation_result["is_valid"] = False
            logger.error("One or more dataset splits is empty")
        
        return validation_result


class DatabaseValidator:
    """Validates database consistency and integrity."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize database validator.
        
        Args:
            data_dir: Base data directory
        """
        self.data_dir = Path(data_dir)
        self.embedding_path = self.data_dir / "embeddings" / "templ_protein_embeddings_v1.0.0.npz"
        self.ligands_path = self.data_dir / "ligands" / "templ_processed_ligands_v1.0.0.sdf.gz"
    
    def validate_file_integrity(self) -> Dict[str, Any]:
        """Validate integrity of essential data files."""
        results = {
            "embeddings_file": self._validate_embeddings_file(),
            "ligands_file": self._validate_ligands_file(),
            "overall_valid": True,
        }
        
        # Check if any validation failed
        if not all(result["valid"] for result in results.values() if isinstance(result, dict)):
            results["overall_valid"] = False
        
        return results
    
    def _validate_embeddings_file(self) -> Dict[str, Any]:
        """Validate embeddings NPZ file."""
        if not self.embedding_path.exists():
            return {
                "valid": False,
                "error": "Embeddings file not found",
                "path": str(self.embedding_path),
            }
        
        try:
            data = np.load(self.embedding_path, allow_pickle=True)
            required_keys = ["pdb_ids", "embeddings", "chain_ids"]
            
            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                return {
                    "valid": False,
                    "error": f"Missing required keys: {missing_keys}",
                    "available_keys": list(data.keys()),
                }
            
            pdb_count = len(data["pdb_ids"])
            embedding_count = len(data["embeddings"])
            
            if pdb_count != embedding_count:
                return {
                    "valid": False,
                    "error": f"Mismatch: {pdb_count} PDB IDs vs {embedding_count} embeddings",
                }
            
            return {
                "valid": True,
                "pdb_count": pdb_count,
                "embedding_dimension": data["embeddings"][0].shape[0] if pdb_count > 0 else 0,
                "file_size_mb": self.embedding_path.stat().st_size / (1024 * 1024),
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Failed to load embeddings: {e}",
                "path": str(self.embedding_path),
            }
    
    def _validate_ligands_file(self) -> Dict[str, Any]:
        """Validate ligands SDF file."""
        if not self.ligands_path.exists():
            return {
                "valid": False,
                "error": "Ligands file not found",
                "path": str(self.ligands_path),
            }
        
        try:
            import gzip
            import tempfile
            
            # Create temporary uncompressed file for validation
            with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp_file:
                with gzip.open(self.ligands_path, 'rb') as gz_file:
                    tmp_file.write(gz_file.read())
                tmp_sdf_path = tmp_file.name
            
            try:
                supplier = Chem.SDMolSupplier(tmp_sdf_path, removeHs=False)
                
                molecule_count = 0
                valid_molecules = 0
                pdb_ids = set()
                
                for mol in supplier:
                    molecule_count += 1
                    if mol is not None:
                        valid_molecules += 1
                        if mol.HasProp("template_pid"):
                            pdb_ids.add(mol.GetProp("template_pid"))
                    
                    # Only check first 100 for performance
                    if molecule_count >= 100:
                        break
                
                return {
                    "valid": True,
                    "molecules_checked": molecule_count,
                    "valid_molecules": valid_molecules,
                    "unique_pdb_ids": len(pdb_ids),
                    "file_size_mb": self.ligands_path.stat().st_size / (1024 * 1024),
                }
                
            finally:
                os.unlink(tmp_sdf_path)
                
        except Exception as e:
            return {
                "valid": False,
                "error": f"Failed to validate ligands: {e}",
                "path": str(self.ligands_path),
            }
    
    def validate_data_consistency(self) -> Dict[str, Any]:
        """Validate consistency between embeddings and ligands."""
        try:
            # Load embeddings
            embedding_data = np.load(self.embedding_path, allow_pickle=True)
            embedding_pdb_ids = set(embedding_data["pdb_ids"])
            
            # Sample ligands
            import gzip
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp_file:
                with gzip.open(self.ligands_path, 'rb') as gz_file:
                    tmp_file.write(gz_file.read())
                tmp_sdf_path = tmp_file.name
            
            try:
                supplier = Chem.SDMolSupplier(tmp_sdf_path, removeHs=False)
                ligand_pdb_ids = set()
                
                for mol in supplier:
                    if mol is not None and mol.HasProp("template_pid"):
                        ligand_pdb_ids.add(mol.GetProp("template_pid"))
                
                # Calculate overlap
                common_ids = embedding_pdb_ids.intersection(ligand_pdb_ids)
                embedding_only = embedding_pdb_ids - ligand_pdb_ids
                ligand_only = ligand_pdb_ids - embedding_pdb_ids
                
                return {
                    "valid": True,
                    "embedding_pdbs": len(embedding_pdb_ids),
                    "ligand_pdbs": len(ligand_pdb_ids),
                    "common_pdbs": len(common_ids),
                    "embedding_only": len(embedding_only),
                    "ligand_only": len(ligand_only),
                    "consistency_ratio": len(common_ids) / max(len(embedding_pdb_ids), len(ligand_pdb_ids)),
                }
                
            finally:
                os.unlink(tmp_sdf_path)
                
        except Exception as e:
            return {
                "valid": False,
                "error": f"Consistency validation failed: {e}",
            }


class MolecularValidationFramework:
    """Framework for validating molecular processing consistency."""

    def __init__(self):
        """Initialize validation framework with test cases."""
        self.test_cases = [
            {
                "id": "ethanol",
                "smiles": "CCO",
                "expected_atoms": 3,
                "expected_heavy_atoms": 3,
                "type": "simple_molecule"
            },
            {
                "id": "benzene",
                "smiles": "c1ccccc1",
                "expected_atoms": 6,
                "expected_heavy_atoms": 6,
                "type": "aromatic"
            },
            {
                "id": "caffeine",
                "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
                "expected_atoms": 14,
                "expected_heavy_atoms": 14,
                "type": "complex_molecule"
            },
        ]

    def run_validation_suite(self, processor_func) -> Dict[str, Any]:
        """
        Run complete validation suite.
        
        Args:
            processor_func: Function to process molecules
            
        Returns:
            Validation results dictionary
        """
        results = {
            "total_tests": len(self.test_cases),
            "passed_tests": 0,
            "failed_tests": 0,
            "test_results": [],
            "success_rate": 0.0,
        }

        for test_case in self.test_cases:
            result = self._validate_molecule(processor_func, test_case)
            results["test_results"].append(result)

            if result["passed"]:
                results["passed_tests"] += 1
            else:
                results["failed_tests"] += 1

        results["success_rate"] = results["passed_tests"] / results["total_tests"]
        return results

    def _validate_molecule(self, processor_func, test_case: Dict) -> Dict:
        """Validate individual molecule processing."""
        result = {
            "test_case": test_case,
            "passed": False,
            "errors": [],
            "warnings": [],
            "processing_time": 0.0,
        }

        try:
            import time
            
            # Create molecule from SMILES
            mol = Chem.MolFromSmiles(test_case["smiles"])
            if mol is None:
                result["errors"].append(f"Failed to create molecule from SMILES: {test_case['smiles']}")
                return result
            
            # Add hydrogens for accurate atom count
            mol = Chem.AddHs(mol)
            
            # Process molecule
            start_time = time.time()
            processed_mol = processor_func(mol)
            result["processing_time"] = time.time() - start_time
            
            if processed_mol is None:
                result["errors"].append("Processor returned None")
                return result
            
            # Validate expected properties
            actual_atoms = processed_mol.GetNumAtoms()
            expected_atoms = test_case.get("expected_atoms")
            
            if expected_atoms and actual_atoms != expected_atoms:
                result["errors"].append(
                    f"Atom count mismatch: expected {expected_atoms}, got {actual_atoms}"
                )
            
            # Check for conformers if expected
            if processed_mol.GetNumConformers() == 0:
                result["warnings"].append("No conformers generated")
            
            # If we got here without errors, test passed
            if not result["errors"]:
                result["passed"] = True
                result["actual_atoms"] = actual_atoms
                result["conformers"] = processed_mol.GetNumConformers()
                
        except Exception as e:
            result["errors"].append(f"Processing failed: {e}")

        return result

    def validate_molecule_properties(self, mol: Chem.Mol) -> Dict[str, Any]:
        """Validate basic molecule properties."""
        if mol is None:
            return {"valid": False, "error": "Molecule is None"}
        
        try:
            properties = {
                "valid": True,
                "num_atoms": mol.GetNumAtoms(),
                "num_heavy_atoms": mol.GetNumHeavyAtoms(),
                "num_bonds": mol.GetNumBonds(),
                "num_conformers": mol.GetNumConformers(),
                "molecular_weight": Chem.Descriptors.MolWt(mol),
                "num_rings": Chem.rdMolDescriptors.CalcNumRings(mol),
                "num_aromatic_rings": Chem.rdMolDescriptors.CalcNumAromaticRings(mol),
            }
            
            # Check for potential issues
            if properties["num_atoms"] == 0:
                properties["valid"] = False
                properties["error"] = "Molecule has no atoms"
            elif properties["num_heavy_atoms"] == 0:
                properties["valid"] = False
                properties["error"] = "Molecule has no heavy atoms"
            elif properties["molecular_weight"] > 1000:
                properties["warning"] = "Large molecule (MW > 1000)"
            
            return properties
            
        except Exception as e:
            return {"valid": False, "error": f"Property calculation failed: {e}"}


def validate_pipeline_components(
    embedding_path: str,
    ligands_path: str,
    splits_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive validation of pipeline components.
    
    Args:
        embedding_path: Path to embeddings file
        ligands_path: Path to ligands file
        splits_dir: Optional path to splits directory
        
    Returns:
        Comprehensive validation results
    """
    results = {
        "timestamp": logger.handlers[0].formatter.formatTime(
            logger.handlers[0].formatter, logger.handlers[0].emit.__self__
        ) if logger.handlers else "unknown",
        "database_validation": {},
        "split_validation": {},
        "molecular_validation": {},
        "overall_valid": True,
    }
    
    # Validate database files
    try:
        db_validator = DatabaseValidator(os.path.dirname(embedding_path))
        results["database_validation"] = db_validator.validate_file_integrity()
        
        if not results["database_validation"]["overall_valid"]:
            results["overall_valid"] = False
    except Exception as e:
        results["database_validation"] = {"error": f"Database validation failed: {e}"}
        results["overall_valid"] = False
    
    # Validate splits if provided
    if splits_dir and os.path.exists(splits_dir):
        try:
            split_validator = SplitDataValidator(embedding_path, splits_dir)
            results["split_validation"] = split_validator.validate_split_completeness()
            
            if not results["split_validation"]["is_valid"]:
                results["overall_valid"] = False
        except Exception as e:
            results["split_validation"] = {"error": f"Split validation failed: {e}"}
            results["overall_valid"] = False
    
    # Validate molecular processing
    try:
        mol_validator = MolecularValidationFramework()
        # Simple identity processor for testing
        results["molecular_validation"] = mol_validator.run_validation_suite(lambda x: x)
        
        if results["molecular_validation"]["success_rate"] < 0.9:
            results["overall_valid"] = False
    except Exception as e:
        results["molecular_validation"] = {"error": f"Molecular validation failed: {e}"}
        results["overall_valid"] = False
    
    return results


def quick_validation_check(data_dir: str = "data") -> bool:
    """
    Quick validation check for essential files.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        True if basic validation passes
    """
    try:
        data_path = Path(data_dir)
        
        # Check for essential files
        essential_files = [
            data_path / "embeddings" / "templ_protein_embeddings_v1.0.0.npz",
            data_path / "ligands" / "templ_processed_ligands_v1.0.0.sdf.gz",
        ]
        
        for file_path in essential_files:
            if not file_path.exists():
                logger.error(f"Essential file missing: {file_path}")
                return False
        
        # Quick size check
        embedding_size = essential_files[0].stat().st_size
        ligand_size = essential_files[1].stat().st_size
        
        if embedding_size < 1024 * 1024:  # Less than 1MB
            logger.error("Embedding file too small")
            return False
        
        if ligand_size < 1024 * 1024:  # Less than 1MB
            logger.error("Ligand file too small")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Quick validation failed: {e}")
        return False