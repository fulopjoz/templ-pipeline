
"""
Database Validation Tools

Tools for monitoring SDF database integrity and consistency.
"""

import gzip
from rdkit import Chem
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class DatabaseValidator:
    """Validates SDF database integrity."""
    
    def __init__(self, sdf_paths: Dict[str, str]):
        self.sdf_paths = sdf_paths
    
    def validate_database_integrity(self) -> Dict:
        """Run comprehensive database validation."""
        results = {
            'total_files': len(self.sdf_paths),
            'file_results': {},
            'cross_file_consistency': {},
            'critical_issues': []
        }
        
        # Validate each SDF file
        for name, path in self.sdf_paths.items():
            results['file_results'][name] = self._validate_sdf_file(path)
        
        # Check consistency between files
        if len(self.sdf_paths) > 1:
            results['cross_file_consistency'] = self._check_cross_file_consistency()
        
        return results
    
    def _validate_sdf_file(self, sdf_path: str) -> Dict:
        """Validate individual SDF file."""
        validation = {
            'total_molecules': 0,
            'valid_molecules': 0,
            'parsing_errors': 0,
            'corruption_indicators': []
        }
        
        try:
            if sdf_path.endswith('.gz'):
                with gzip.open(sdf_path, 'rt') as f:
                    supplier = Chem.ForwardSDMolSupplier(f, removeHs=False, sanitize=False)
                    
                    for mol in supplier:
                        validation['total_molecules'] += 1
                        
                        if mol is None:
                            validation['parsing_errors'] += 1
                            continue
                        
                        # Check for corruption indicators
                        if self._check_molecule_corruption(mol):
                            validation['corruption_indicators'].append(validation['total_molecules'])
                        
                        validation['valid_molecules'] += 1
            else:
                supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
                
                for mol in supplier:
                    validation['total_molecules'] += 1
                    
                    if mol is None:
                        validation['parsing_errors'] += 1
                        continue
                    
                    # Check for corruption indicators
                    if self._check_molecule_corruption(mol):
                        validation['corruption_indicators'].append(validation['total_molecules'])
                    
                    validation['valid_molecules'] += 1
        
        except Exception as e:
            validation['file_error'] = str(e)
        
        return validation
    
    def _check_molecule_corruption(self, mol: Chem.Mol) -> bool:
        """Check individual molecule for corruption signs."""
        try:
            # Test SMILES generation
            Chem.MolToSmiles(mol)
            
            # Test sanitization
            mol_copy = Chem.Mol(mol)
            Chem.SanitizeMol(mol_copy)
            
            return False
        except:
            return True
    
    def _check_cross_file_consistency(self) -> Dict:
        """Check consistency between different SDF files."""
        # Implementation for cross-file consistency checking
        return {'implemented': False, 'note': 'Cross-file consistency checking'}
