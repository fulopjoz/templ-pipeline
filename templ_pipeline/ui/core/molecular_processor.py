"""
TEMPL Pipeline - Cached Molecular Processor

Molecular processing with operation caching for improved performance
and reduced redundant calculations.
"""

import logging
import hashlib
from functools import lru_cache
from typing import Dict, Any, Optional, Tuple, Union, List

# Try to import RDKit for molecular operations
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


class CachedMolecularProcessor:
    """Molecular processor with operation caching for performance optimization"""
    
    def __init__(self, cache_size: int = 256):
        """Initialize cached molecular processor
        
        Args:
            cache_size: Size of LRU caches for molecular operations
        """
        self.cache_size = cache_size
        self.processing_stats = {
            'validations': 0,
            'properties_calculated': 0,
            'coordinates_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Setup cached methods
        self._setup_caches()
        
        if not RDKIT_AVAILABLE:
            logger.warning("RDKit not available - molecular processing disabled")
        else:
            logger.info(f"Initialized CachedMolecularProcessor with cache size {cache_size}")
    
    def _setup_caches(self):
        """Setup LRU caches for common molecular operations"""
        
        # Cache for SMILES validation
        self.validate_smiles = lru_cache(maxsize=self.cache_size)(
            self._validate_smiles_impl
        )
        
        # Cache for molecular properties
        self.get_properties = lru_cache(maxsize=self.cache_size)(
            self._get_properties_impl
        )
        
        # Cache for 2D coordinate generation
        self.generate_2d_coords = lru_cache(maxsize=self.cache_size)(
            self._generate_2d_coords_impl
        )
    
    def _validate_smiles_impl(self, smiles: str) -> Tuple[bool, str, Optional[str]]:
        """Core SMILES validation implementation
        
        Args:
            smiles: SMILES string to validate
        
        Returns:
            Tuple of (valid, message, canonical_smiles)
        """
        if not RDKIT_AVAILABLE:
            return False, "RDKit not available", None
        
        try:
            # Basic checks
            if not smiles or not isinstance(smiles, str):
                return False, "Empty or invalid SMILES string", None
            
            smiles = smiles.strip()
            if not smiles:
                return False, "Empty SMILES string", None
            
            # Parse with RDKit
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, "Invalid SMILES format", None
            
            # Check atom count
            num_atoms = mol.GetNumAtoms()
            if num_atoms < 3:
                return False, f"Too few atoms ({num_atoms}, minimum 3)", None
            if num_atoms > 200:
                return False, f"Too many atoms ({num_atoms}, maximum 200)", None
            
            # Generate canonical SMILES for consistency
            try:
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                return True, f"Valid molecule ({num_atoms} atoms)", canonical_smiles
            except Exception as e:
                return False, f"Cannot generate canonical SMILES: {str(e)}", None
                
        except Exception as e:
            return False, f"Validation error: {str(e)}", None
    
    def _get_properties_impl(self, smiles: str) -> Dict[str, Any]:
        """Calculate basic molecular properties implementation
        
        Args:
            smiles: Canonical SMILES string
        
        Returns:
            Dictionary of molecular properties
        """
        if not RDKIT_AVAILABLE:
            return {'error': 'RDKit not available'}
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return {'error': 'Invalid SMILES'}
            
            # Calculate basic properties
            properties = {
                'num_atoms': mol.GetNumAtoms(),
                'num_bonds': mol.GetNumBonds(),
                'molecular_weight': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'num_rings': mol.GetRingInfo().NumRings(),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol)
            }
            
            return properties
            
        except Exception as e:
            return {'error': f'Property calculation failed: {str(e)}'}
    
    def _generate_2d_coords_impl(self, smiles: str) -> Optional[bytes]:
        """Generate 2D coordinates implementation
        
        Args:
            smiles: Canonical SMILES string
        
        Returns:
            Molecule binary data with 2D coordinates or None
        """
        if not RDKIT_AVAILABLE:
            return None
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return None
            
            # Generate 2D coordinates
            AllChem.Compute2DCoords(mol)
            
            # Return binary representation
            return mol.ToBinary()
            
        except Exception as e:
            logger.warning(f"2D coordinate generation failed: {e}")
            return None
    
    def process_molecule_efficient(self, mol_input: Union[str, Any]) -> Dict[str, Any]:
        """Process molecule with comprehensive caching
        
        Args:
            mol_input: SMILES string or RDKit molecule object
        
        Returns:
            Dictionary with processing results
        """
        if not RDKIT_AVAILABLE:
            return {'valid': False, 'error': 'RDKit not available'}
        
        # Update stats
        self.processing_stats['validations'] += 1
        
        try:
            # Convert to SMILES if needed
            if isinstance(mol_input, str):
                smiles = mol_input.strip()
            else:
                # Assume it's a molecule object
                smiles = Chem.MolToSmiles(mol_input)
            
            # Validate SMILES (cached)
            valid, message, canonical = self.validate_smiles(smiles)
            if not valid:
                return {
                    'valid': False,
                    'message': message,
                    'input_smiles': smiles
                }
            
            # Get properties (cached)
            self.processing_stats['properties_calculated'] += 1
            properties = self.get_properties(canonical)
            
            # Generate 2D coordinates if needed (cached)
            self.processing_stats['coordinates_generated'] += 1
            coords_binary = self.generate_2d_coords(canonical)
            
            result = {
                'valid': True,
                'smiles': canonical,
                'message': message,
                'properties': properties,
                'mol_binary': coords_binary
            }
            
            return result
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Processing failed: {str(e)}',
                'input': str(mol_input)[:100]  # Limit length for safety
            }
    
    def clear_caches(self):
        """Clear all caches and reset statistics"""
        self.validate_smiles.cache_clear()
        self.get_properties.cache_clear()
        self.generate_2d_coords.cache_clear()
        
        # Reset statistics
        self.processing_stats = {
            'validations': 0,
            'properties_calculated': 0,
            'coordinates_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("Cleared all molecular processing caches")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        cache_stats = {}
        
        # Get cache info for each cached method
        methods = ['validate_smiles', 'get_properties', 'generate_2d_coords']
        
        for method_name in methods:
            method = getattr(self, method_name)
            if hasattr(method, 'cache_info'):
                info = method.cache_info()
                cache_stats[method_name] = {
                    'hits': info.hits,
                    'misses': info.misses,
                    'currsize': info.currsize,
                    'maxsize': info.maxsize,
                    'hit_rate': info.hits / max(1, info.hits + info.misses) * 100
                }
        
        # Add processing statistics
        cache_stats['processing'] = self.processing_stats.copy()
        cache_stats['rdkit_available'] = RDKIT_AVAILABLE
        
        return cache_stats


# Global instance management
_global_processor = None

def get_molecular_processor() -> CachedMolecularProcessor:
    """Get global molecular processor instance
    
    Returns:
        Global CachedMolecularProcessor instance
    """
    global _global_processor
    if _global_processor is None:
        _global_processor = CachedMolecularProcessor()
    return _global_processor


# Convenience functions for backward compatibility
def validate_smiles_cached(smiles: str) -> Tuple[bool, str, Optional[str]]:
    """Validate SMILES using cached processor
    
    Args:
        smiles: SMILES string to validate
    
    Returns:
        Tuple of (valid, message, canonical_smiles)
    """
    processor = get_molecular_processor()
    return processor.validate_smiles(smiles)


def get_molecular_properties_cached(smiles: str) -> Dict[str, Any]:
    """Get molecular properties using cached processor
    
    Args:
        smiles: SMILES string
    
    Returns:
        Dictionary of molecular properties
    """
    processor = get_molecular_processor()
    return processor.get_properties(smiles) 