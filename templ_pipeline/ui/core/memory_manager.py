"""
TEMPL Pipeline - Molecular Session Manager

Memory-efficient session state management optimized for molecular data
with SMILES-based regeneration and automatic cleanup.
"""

import gc
import time
import pickle
import gzip
import logging
from typing import Any, Optional, Dict, Tuple, List
from collections import OrderedDict
from pathlib import Path

# Try to import streamlit and RDKit for molecular operations
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


class MolecularSessionManager:
    """Session manager optimized for molecular data with memory efficiency"""
    
    def __init__(self, cache_size_mb: int = 100, cleanup_threshold: float = 0.8):
        """Initialize molecular session manager
        
        Args:
            cache_size_mb: Maximum cache size in megabytes
            cleanup_threshold: Cleanup when cache reaches this fraction of max size
        """
        self.cache_size = cache_size_mb * 1024 * 1024  # Convert to bytes
        self.cleanup_threshold = cleanup_threshold
        
        # Cache storage
        self.molecule_cache = OrderedDict()  # Full molecule objects
        self.metadata_cache = {}             # Molecular metadata
        self.smiles_cache = {}              # Lightweight SMILES representations
        self.compressed_cache = {}          # Compressed large objects
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.regenerations = 0
        self.cleanup_count = 0
        
        logger.info(f"Initialized MolecularSessionManager with {cache_size_mb}MB cache")
    
    def store_molecule(self, key: str, mol: Any, metadata: Optional[Dict] = None) -> bool:
        """Store molecule with optimized caching strategy
        
        Args:
            key: Storage key for the molecule
            mol: RDKit molecule object
            metadata: Optional metadata dictionary
        
        Returns:
            True if stored successfully
        """
        if not RDKIT_AVAILABLE or mol is None:
            return False
        
        try:
            # Always store SMILES (lightweight fallback)
            smiles = Chem.MolToSmiles(mol)
            self.smiles_cache[key] = smiles
            
            # Store metadata separately
            if metadata:
                self.metadata_cache[key] = metadata.copy()
            
            # Estimate molecule size
            mol_binary = mol.ToBinary()
            mol_size = len(mol_binary)
            
            # Storage strategy based on size
            if mol_size < 50000:  # 50KB threshold for direct storage
                self._store_direct(key, mol)
                
            elif mol_size < 200000:  # 200KB threshold for compressed storage
                self._store_compressed(key, mol)
                
            else:
                # Very large molecules - store only SMILES
                if STREAMLIT_AVAILABLE:
                    st.session_state[f"{key}_large"] = True
                
                logger.info(f"Large molecule {key} stored as SMILES only ({mol_size} bytes)")
            
            # Trigger cleanup if needed
            if self._should_cleanup():
                self._cleanup_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store molecule {key}: {e}")
            return False
    
    def get_molecule(self, key: str) -> Optional[Any]:
        """Retrieve molecule with automatic regeneration from SMILES if needed
        
        Args:
            key: Storage key for the molecule
        
        Returns:
            RDKit molecule object or None
        """
        if not RDKIT_AVAILABLE:
            return None
        
        # Check direct cache
        if key in self.molecule_cache:
            self.cache_hits += 1
            self.molecule_cache.move_to_end(key)  # LRU update
            return self.molecule_cache[key]
        
        # Check session state
        if STREAMLIT_AVAILABLE and key in st.session_state:
            mol = st.session_state[key]
            if mol is not None:
                self.cache_hits += 1
                # Add to cache for faster future access
                self._store_direct(key, mol)
                return mol
        
        # Check compressed cache
        if key in self.compressed_cache:
            try:
                mol = self._decompress_molecule(key)
                if mol:
                    self.cache_hits += 1
                    # Move to direct cache for faster access
                    self._store_direct(key, mol)
                    return mol
            except Exception as e:
                logger.warning(f"Failed to decompress molecule {key}: {e}")
        
        # Regenerate from SMILES if available
        if key in self.smiles_cache:
            try:
                mol = Chem.MolFromSmiles(self.smiles_cache[key])
                if mol:
                    self.cache_misses += 1
                    self.regenerations += 1
                    
                    # Restore metadata if available
                    if key in self.metadata_cache:
                        for prop, value in self.metadata_cache[key].items():
                            mol.SetProp(prop, str(value))
                    
                    # Cache regenerated molecule
                    self._store_direct(key, mol)
                    logger.debug(f"Regenerated molecule {key} from SMILES")
                    return mol
                    
            except Exception as e:
                logger.error(f"Failed to regenerate molecule from SMILES: {e}")
        
        self.cache_misses += 1
        return None
    
    def store_pose_results(self, poses: Dict[str, Tuple[Any, Dict]]) -> bool:
        """Optimized storage for pose prediction results
        
        Args:
            poses: Dictionary of poses with scores
        
        Returns:
            True if stored successfully
        """
        try:
            # Store individual poses efficiently
            best_poses = {}
            all_scores = []
            
            for method, (mol, scores) in poses.items():
                # Store molecule with metadata
                pose_key = f"pose_{method}"
                if self.store_molecule(pose_key, mol, metadata=scores):
                    
                    # Track if this is a good pose
                    combo_score = scores.get('combo', scores.get('combo_score', 0))
                    if combo_score > 0.15:  # Threshold for keeping poses - Updated from 0.3 to 0.15 based on literature validation
                        best_poses[method] = {
                            'key': pose_key,
                            'scores': scores,
                            'combo_score': combo_score
                        }
                    
                    all_scores.append((method, scores))
            
            # Store lightweight references in session state
            if STREAMLIT_AVAILABLE:
                st.session_state['best_poses_refs'] = best_poses
                st.session_state['all_scores'] = all_scores
                st.session_state['poses_timestamp'] = time.time()
            
            logger.info(f"Stored {len(poses)} poses, {len(best_poses)} above threshold")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store pose results: {e}")
            return False
    
    def get_pose_results(self) -> Optional[Dict[str, Tuple[Any, Dict]]]:
        """Retrieve pose results with automatic regeneration
        
        Returns:
            Dictionary of poses with scores or None
        """
        if not STREAMLIT_AVAILABLE:
            return None
        
        # Check if we have pose references
        if 'best_poses_refs' not in st.session_state:
            return None
        
        try:
            poses = {}
            best_poses_refs = st.session_state['best_poses_refs']
            
            for method, pose_info in best_poses_refs.items():
                mol = self.get_molecule(pose_info['key'])
                if mol:
                    poses[method] = (mol, pose_info['scores'])
            
            return poses if poses else None
            
        except Exception as e:
            logger.error(f"Failed to retrieve pose results: {e}")
            return None
    
    def store_large_object(self, key: str, value: Any, metadata: Optional[Dict] = None) -> bool:
        """Store large object with intelligent routing to appropriate storage method
        
        This method provides the interface expected by SessionManager.set() and routes
        objects to the most appropriate storage method based on their type.
        
        Args:
            key: Storage key for the object
            value: Object to store
            metadata: Optional metadata dictionary
        
        Returns:
            True if stored successfully
        """
        if value is None:
            logger.debug(f"Skipping storage of None value for key: {key}")
            return True
        
        try:
            # Route to appropriate storage method based on object type
            
            # Check if it's an RDKit molecule object
            if hasattr(value, 'ToBinary') and hasattr(value, 'GetNumAtoms'):
                logger.debug(f"Storing {key} as molecule object")
                return self.store_molecule(key, value, metadata)
            
            # Check if it's a poses dictionary (dict with tuple values containing molecules)
            elif isinstance(value, dict) and value:
                # Check if it looks like poses (contains tuples with molecule-like objects)
                sample_key = next(iter(value))
                sample_value = value[sample_key]
                
                if (isinstance(sample_value, tuple) and len(sample_value) == 2 and
                    hasattr(sample_value[0], 'ToBinary')):
                    logger.debug(f"Storing {key} as pose results")
                    return self.store_pose_results(value)
            
            # For other large objects, store using general strategy
            logger.debug(f"Storing {key} as general large object")
            
            # Estimate object size for storage strategy
            try:
                if hasattr(value, '__sizeof__'):
                    obj_size = value.__sizeof__()
                else:
                    # Rough estimate using pickle
                    obj_size = len(pickle.dumps(value))
                    
                logger.debug(f"Object {key} estimated size: {obj_size} bytes")
                
                # Storage strategy based on size
                if obj_size < 100000:  # 100KB threshold for direct storage
                    self._store_general_direct(key, value, metadata)
                    
                elif obj_size < 500000:  # 500KB threshold for compressed storage
                    self._store_general_compressed(key, value, metadata)
                    
                else:
                    # Very large objects - store reference only
                    logger.warning(f"Very large object {key} ({obj_size} bytes) - storing reference only")
                    if STREAMLIT_AVAILABLE:
                        st.session_state[f"{key}_large_ref"] = True
                        # Store minimal reference info
                        if metadata:
                            self.metadata_cache[key] = metadata.copy()
                
                return True
                
            except Exception as size_error:
                logger.warning(f"Could not estimate size for {key}: {size_error}")
                # Fallback to direct storage
                self._store_general_direct(key, value, metadata)
                return True
            
        except Exception as e:
            logger.error(f"Failed to store large object {key}: {e}")
            # Last resort - try to store in session state directly
            try:
                if STREAMLIT_AVAILABLE:
                    st.session_state[key] = value
                    if metadata:
                        self.metadata_cache[key] = metadata.copy()
                    logger.debug(f"Fallback storage successful for {key}")
                    return True
            except Exception as fallback_error:
                logger.error(f"Fallback storage also failed for {key}: {fallback_error}")
                return False
            
            return False
    
    def _store_general_direct(self, key: str, obj: Any, metadata: Optional[Dict] = None):
        """Store general object directly
        
        Args:
            key: Storage key
            obj: Object to store
            metadata: Optional metadata
        """
        if STREAMLIT_AVAILABLE:
            st.session_state[key] = obj
        
        if metadata:
            self.metadata_cache[key] = metadata.copy()
            
        logger.debug(f"Stored general object {key} directly")
    
    def _store_general_compressed(self, key: str, obj: Any, metadata: Optional[Dict] = None):
        """Store general object in compressed format
        
        Args:
            key: Storage key
            obj: Object to store
            metadata: Optional metadata
        """
        try:
            # Serialize and compress
            serialized = pickle.dumps(obj)
            compressed = gzip.compress(serialized)
            
            # Store in compressed cache
            self.compressed_cache[f"{key}_general"] = compressed
            
            # Store metadata
            if metadata:
                self.metadata_cache[key] = metadata.copy()
            
            # Remove from session state if present to save memory
            if STREAMLIT_AVAILABLE and key in st.session_state:
                del st.session_state[key]
            
            compression_ratio = len(compressed) / len(serialized)
            logger.debug(f"Compressed general object {key}: {len(serialized)} -> {len(compressed)} bytes ({compression_ratio:.2%})")
            
        except Exception as e:
            logger.error(f"Failed to compress general object {key}: {e}")
            # Fallback to direct storage
            self._store_general_direct(key, obj, metadata)
    
    def _store_direct(self, key: str, mol: Any):
        """Store molecule directly in cache
        
        Args:
            key: Storage key
            mol: RDKit molecule object
        """
        self.molecule_cache[key] = mol
        if STREAMLIT_AVAILABLE:
            st.session_state[key] = mol
    
    def _store_compressed(self, key: str, mol: Any):
        """Store molecule in compressed format
        
        Args:
            key: Storage key
            mol: RDKit molecule object
        """
        try:
            mol_binary = mol.ToBinary()
            compressed = gzip.compress(mol_binary)
            
            self.compressed_cache[key] = compressed
            
            # Remove from direct cache if present
            self.molecule_cache.pop(key, None)
            
            compression_ratio = len(compressed) / len(mol_binary)
            logger.debug(f"Compressed {key}: {len(mol_binary)} -> {len(compressed)} bytes ({compression_ratio:.2%})")
            
        except Exception as e:
            logger.error(f"Failed to compress molecule {key}: {e}")
            # Fallback to direct storage
            self._store_direct(key, mol)
    
    def _decompress_molecule(self, key: str) -> Optional[Any]:
        """Decompress molecule from compressed cache
        
        Args:
            key: Storage key
        
        Returns:
            RDKit molecule object or None
        """
        if key not in self.compressed_cache:
            return None
        
        try:
            compressed_data = self.compressed_cache[key]
            mol_binary = gzip.decompress(compressed_data)
            mol = Chem.Mol(mol_binary)
            return mol
            
        except Exception as e:
            logger.error(f"Failed to decompress molecule {key}: {e}")
            return None
    
    def _should_cleanup(self) -> bool:
        """Check if cache cleanup is needed
        
        Returns:
            True if cleanup should be performed
        """
        current_size = self._calculate_cache_size()
        threshold_size = self.cache_size * self.cleanup_threshold
        
        return current_size > threshold_size
    
    def _calculate_cache_size(self) -> int:
        """Calculate current cache size in bytes
        
        Returns:
            Current cache size in bytes
        """
        size = 0
        
        # Direct cache
        for mol in self.molecule_cache.values():
            try:
                if hasattr(mol, 'ToBinary'):
                    size += len(mol.ToBinary())
            except:
                size += 1000  # Estimate for objects that can't be serialized
        
        # Compressed cache
        for compressed_data in self.compressed_cache.values():
            size += len(compressed_data)
        
        # Metadata cache
        for metadata in self.metadata_cache.values():
            try:
                size += len(pickle.dumps(metadata))
            except:
                size += 100  # Estimate
        
        return size
    
    def _cleanup_cache(self):
        """Clean up cache to free memory"""
        logger.info("Starting cache cleanup")
        
        cleanup_start = time.time()
        initial_size = self._calculate_cache_size()
        
        # Remove least recently used molecules from direct cache
        target_removal = max(1, len(self.molecule_cache) // 4)  # Remove 25%
        removed_keys = []
        
        for _ in range(target_removal):
            if self.molecule_cache:
                key, mol = self.molecule_cache.popitem(last=False)  # Remove LRU
                removed_keys.append(key)
        
        # Clean up orphaned metadata
        for key in removed_keys:
            self.metadata_cache.pop(key, None)
            
            # Remove from session state if present
            if STREAMLIT_AVAILABLE and key in st.session_state:
                del st.session_state[key]
        
        # Clean up very old compressed items
        if len(self.compressed_cache) > 20:
            # Keep only the most recent 15 compressed items
            items = list(self.compressed_cache.items())
            self.compressed_cache = dict(items[-15:])
        
        # Force garbage collection
        gc.collect()
        
        final_size = self._calculate_cache_size()
        cleanup_time = time.time() - cleanup_start
        self.cleanup_count += 1
        
        logger.info(f"Cache cleanup completed: {initial_size / 1024 / 1024:.1f}MB -> {final_size / 1024 / 1024:.1f}MB in {cleanup_time:.2f}s")
    
    def clear_cache(self, key_pattern: Optional[str] = None):
        """Clear cache entries
        
        Args:
            key_pattern: Optional pattern to match keys (simple substring match)
        """
        if key_pattern:
            # Remove matching keys
            keys_to_remove = [k for k in self.molecule_cache.keys() if key_pattern in k]
            for key in keys_to_remove:
                self.molecule_cache.pop(key, None)
                self.metadata_cache.pop(key, None)
                self.smiles_cache.pop(key, None)
                self.compressed_cache.pop(key, None)
                
                if STREAMLIT_AVAILABLE and key in st.session_state:
                    del st.session_state[key]
                    
            logger.info(f"Cleared {len(keys_to_remove)} cache entries matching '{key_pattern}'")
        else:
            # Clear all caches
            self.molecule_cache.clear()
            self.metadata_cache.clear()
            self.smiles_cache.clear()
            self.compressed_cache.clear()
            
            logger.info("Cleared all cache entries")
        
        gc.collect()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory usage statistics
        
        Returns:
            Dictionary with memory statistics
        """
        current_size = self._calculate_cache_size()
        
        return {
            'cache_size_mb': current_size / (1024 * 1024),
            'cache_limit_mb': self.cache_size / (1024 * 1024),
            'utilization_percent': (current_size / self.cache_size) * 100,
            'direct_cache_entries': len(self.molecule_cache),
            'compressed_cache_entries': len(self.compressed_cache),
            'smiles_cache_entries': len(self.smiles_cache),
            'metadata_cache_entries': len(self.metadata_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'regenerations': self.regenerations,
            'cleanup_count': self.cleanup_count,
            'hit_rate_percent': (self.cache_hits / max(1, self.cache_hits + self.cache_misses)) * 100,
            'rdkit_available': RDKIT_AVAILABLE,
            'streamlit_available': STREAMLIT_AVAILABLE
        }
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform comprehensive memory optimization
        
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        initial_size = self._calculate_cache_size()
        
        # Force cleanup
        self._cleanup_cache()
        
        # Compress molecules that should be compressed
        molecules_compressed = 0
        for key, mol in list(self.molecule_cache.items()):
            try:
                mol_size = len(mol.ToBinary())
                if mol_size > 50000:  # Compress large molecules
                    self._store_compressed(key, mol)
                    molecules_compressed += 1
            except:
                continue
        
        # Force garbage collection
        gc.collect()
        
        final_size = self._calculate_cache_size()
        optimization_time = time.time() - start_time
        
        result = {
            'initial_size_mb': initial_size / (1024 * 1024),
            'final_size_mb': final_size / (1024 * 1024),
            'memory_saved_mb': (initial_size - final_size) / (1024 * 1024),
            'molecules_compressed': molecules_compressed,
            'optimization_time_s': optimization_time
        }
        
        logger.info(f"Memory optimization: {result['memory_saved_mb']:.1f}MB saved in {optimization_time:.2f}s")
        
        return result


# Global instance management
_global_memory_manager = None

def get_memory_manager() -> MolecularSessionManager:
    """Get global memory manager instance
    
    Returns:
        Global MolecularSessionManager instance
    """
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MolecularSessionManager()
    return _global_memory_manager


# Convenience functions for backward compatibility
def store_molecule_efficient(key: str, mol: Any, metadata: Optional[Dict] = None) -> bool:
    """Store molecule using global memory manager
    
    Args:
        key: Storage key
        mol: RDKit molecule object  
        metadata: Optional metadata
    
    Returns:
        True if stored successfully
    """
    manager = get_memory_manager()
    return manager.store_molecule(key, mol, metadata)


def get_molecule_efficient(key: str) -> Optional[Any]:
    """Retrieve molecule using global memory manager
    
    Args:
        key: Storage key
    
    Returns:
        RDKit molecule object or None
    """
    manager = get_memory_manager()
    return manager.get_molecule(key) 