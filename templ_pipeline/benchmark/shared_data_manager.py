#!/usr/bin/env python3
"""
Enhanced shared data manager for TEMPL benchmark to prevent memory explosion.

This module implements improved shared memory solutions for:
1. Protein embeddings - shared across all processes using SharedMemory + .npy
2. Ligand index - memory-mapped for efficient lookup with binary format
3. Pre-computed KNN indices - to avoid repeated similarity calculations
4. Centralized memory lifecycle management with reference counting
"""

import logging
import os
import tempfile
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import pickle
import mmap
import json
import weakref
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager as StdSharedMemoryManager

import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)


class BenchmarkSharedMemoryManager:
    """
    Centralized memory lifecycle management for shared data.
    
    This class manages the creation, sharing, and cleanup of shared memory
    buffers with proper reference counting and coordinated cleanup.
    """
    
    def __init__(self):
        """Initialize the shared memory manager."""
        self._shared_buffers = {}  # name -> (buffer, ref_count, size)
        self._lock = threading.Lock()
        self._manager = None
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_all()
            
    def create_shared_buffer(self, name: str, size: int, data: Optional[bytes] = None) -> shared_memory.SharedMemory:
        """
        Create a shared memory buffer with reference counting.
        
        Args:
            name: Unique name for the shared buffer
            size: Size in bytes
            data: Optional initial data
            
        Returns:
            SharedMemory object
        """
        with self._lock:
            if name in self._shared_buffers:
                # Increment reference count
                buffer_info = self._shared_buffers[name]
                buffer_info[1] += 1  # Increment ref_count
                logger.debug(f"Incremented ref count for {name}: {buffer_info[1]}")
                return buffer_info[0]
            
            # Create new shared buffer
            try:
                shm = shared_memory.SharedMemory(create=True, size=size, name=name)
                
                if data:
                    shm.buf[:len(data)] = data
                    
                self._shared_buffers[name] = (shm, 1, size)
                logger.info(f"Created shared buffer {name} ({size} bytes)")
                return shm
                
            except Exception as e:
                logger.error(f"Failed to create shared buffer {name}: {e}")
                raise
                
    def attach_shared_buffer(self, name: str) -> Optional[shared_memory.SharedMemory]:
        """
        Attach to existing shared buffer with reference counting.
        
        Args:
            name: Name of existing shared buffer
            
        Returns:
            SharedMemory object or None if not found
        """
        with self._lock:
            if name in self._shared_buffers:
                buffer_info = self._shared_buffers[name]
                buffer_info[1] += 1  # Increment ref_count
                logger.debug(f"Attached to {name}, ref count: {buffer_info[1]}")
                return buffer_info[0]
            else:
                # Try to attach to existing buffer
                try:
                    shm = shared_memory.SharedMemory(name=name)
                    self._shared_buffers[name] = (shm, 1, 0)  # Size unknown
                    logger.info(f"Attached to existing shared buffer {name}")
                    return shm
                except Exception as e:
                    logger.warning(f"Failed to attach to shared buffer {name}: {e}")
                    return None
                    
    def release_shared_buffer(self, name: str) -> bool:
        """
        Release a shared buffer with reference counting.
        
        Args:
            name: Name of shared buffer to release
            
        Returns:
            True if buffer was actually freed
        """
        with self._lock:
            if name not in self._shared_buffers:
                return False
                
            buffer_info = self._shared_buffers[name]
            buffer_info[1] -= 1  # Decrement ref_count
            
            if buffer_info[1] <= 0:
                # No more references, cleanup
                shm = buffer_info[0]
                shm.close()
                shm.unlink()
                del self._shared_buffers[name]
                logger.info(f"Freed shared buffer {name}")
                return True
            else:
                logger.debug(f"Released {name}, ref count: {buffer_info[1]}")
                return False
                
    def cleanup_all(self):
        """Clean up all shared buffers."""
        with self._lock:
            logger.info(f"Cleaning up {len(self._shared_buffers)} shared buffers...")
            for name, (shm, ref_count, size) in list(self._shared_buffers.items()):
                try:
                    shm.close()
                    shm.unlink()
                    logger.info(f"Cleaned up shared buffer {name}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {name}: {e}")
            self._shared_buffers.clear()
            
            # Additional cleanup for any remaining shared memory objects
            try:
                import multiprocessing.resource_tracker as rt
                
                # Force cleanup of any remaining shared memory objects
                if hasattr(rt, '_CLEANUP_CALLBACKS'):
                    for callback in rt._CLEANUP_CALLBACKS:
                        try:
                            callback()
                        except Exception:
                            pass
                            
            except Exception as e:
                logger.debug(f"Additional shared memory cleanup failed: {e}")
                
            logger.info("✓ Shared memory manager cleanup completed")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about shared buffers."""
        with self._lock:
            total_size = sum(size for _, _, size in self._shared_buffers.values())
            return {
                'buffer_count': len(self._shared_buffers),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'buffers': {name: {'ref_count': ref_count, 'size': size} 
                           for name, (_, ref_count, size) in self._shared_buffers.items()}
            }


class EnhancedSharedDataManager:
    """
    Enhanced shared data manager with SharedMemory and binary formats.
    
    This class pre-loads embeddings and ligand data once in the main process
    and makes them available to all subprocesses via shared memory buffers
    and memory-mapped files to prevent memory explosion.
    """
    
    def __init__(self, data_dir: Path, cache_dir: Optional[Path] = None):
        """
        Initialize enhanced shared data manager.
        
        Args:
            data_dir: Directory containing benchmark data
            cache_dir: Directory for shared cache files (default: temp directory)
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "templ_shared_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Shared memory manager
        self.shm_manager = BenchmarkSharedMemoryManager()
        
        # Shared data references
        self.embedding_shm = None
        self.embedding_metadata_file = None
        self.ligand_index_file = None
        self.knn_cache_file = None
        
        # Pre-loaded data flags
        self.embeddings_loaded = False
        self.ligand_index_loaded = False
        
        # Reference counting for cleanup coordination
        self._ref_count = 0
        self._lock = threading.Lock()
        
        logger.info(f"EnhancedSharedDataManager initialized with cache dir: {self.cache_dir}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def _increment_ref_count(self):
        """Increment reference count for cleanup coordination."""
        with self._lock:
            self._ref_count += 1
            logger.debug(f"Incremented ref count: {self._ref_count}")
    
    def _decrement_ref_count(self):
        """Decrement reference count and cleanup if zero."""
        with self._lock:
            self._ref_count -= 1
            logger.debug(f"Decremented ref count: {self._ref_count}")
            if self._ref_count <= 0:
                self._cleanup_internal()
    
    def preload_embeddings(self, embedding_path: Optional[str] = None) -> Tuple[str, str]:
        """
        Pre-load embeddings and create shared memory buffer with .npy format.
        
        Args:
            embedding_path: Path to embedding file (auto-detected if None)
            
        Returns:
            Tuple of (shared_memory_name, metadata_file_path)
        """
        if self.embeddings_loaded:
            return self.embedding_shm.name, self.embedding_metadata_file
            
        # Auto-detect embedding path
        if embedding_path is None:
            embedding_path = self.data_dir / "embeddings" / "templ_protein_embeddings_v1.0.0.npz"
        
        if not Path(embedding_path).exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        logger.info(f"Pre-loading embeddings from: {embedding_path}")
        
        try:
            # Load embeddings from NPZ file
            data = np.load(embedding_path, allow_pickle=True)
            pdb_ids = data["pdb_ids"]
            embeddings = data["embeddings"]
            chain_ids = data.get("chain_ids", None)
            
            # Prepare embedding data for sharing
            embedding_db = {}
            embedding_chain_data = {}
            
            for i, pid in enumerate(pdb_ids):
                pid_upper = str(pid).upper()
                embedding_db[pid_upper] = embeddings[i]
                if chain_ids is not None:
                    embedding_chain_data[pid_upper] = chain_ids[i]
                else:
                    embedding_chain_data[pid_upper] = ""
            
            # Create binary .npy file for embeddings
            embedding_array = np.array([embedding_db[pid] for pid in sorted(embedding_db.keys())])
            pdb_id_list = sorted(embedding_db.keys())
            
            # Save as .npy file for memory mapping
            npy_file = self.cache_dir / "embeddings.npy"
            np.save(npy_file, embedding_array)
            
            # Create metadata file
            metadata = {
                'pdb_ids': pdb_id_list,
                'embedding_chain_data': embedding_chain_data,
                'total_embeddings': len(embedding_db),
                'embedding_shape': embedding_array.shape,
                'loaded_at': time.time()
            }
            
            metadata_file = self.cache_dir / "embedding_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create shared memory buffer for the .npy file
            with open(npy_file, 'rb') as f:
                npy_data = f.read()
            
            shm_name = f"templ_embeddings_{os.getpid()}_{int(time.time())}"
            self.embedding_shm = self.shm_manager.create_shared_buffer(
                shm_name, len(npy_data), npy_data
            )
            
            self.embedding_metadata_file = str(metadata_file)
            self.embeddings_loaded = True
            
            logger.info(f"Pre-loaded {len(embedding_db)} embeddings to shared memory: {shm_name}")
            
            return shm_name, self.embedding_metadata_file
            
        except Exception as e:
            logger.error(f"Failed to pre-load embeddings: {e}")
            raise
    
    def create_ligand_index(self, sdf_path: Optional[str] = None) -> str:
        """
        Create memory-mapped ligand index with binary format for efficient lookup.
        
        Instead of loading all molecules, create an index that maps PDB IDs
        to file positions for efficient random access.
        
        Args:
            sdf_path: Path to ligand SDF file (auto-detected if None)
            
        Returns:
            Path to ligand index file
        """
        if self.ligand_index_loaded:
            return self.ligand_index_file
            
        # Auto-detect SDF path
        if sdf_path is None:
            from templ_pipeline.core.utils import find_ligand_file_paths
            ligand_paths = find_ligand_file_paths(self.data_dir)
            if not ligand_paths:
                raise FileNotFoundError("No ligand SDF file found")
            sdf_path = ligand_paths[0]
        
        if not Path(sdf_path).exists():
            raise FileNotFoundError(f"Ligand SDF file not found: {sdf_path}")
        
        logger.info(f"Creating ligand index for: {sdf_path}")
        
        try:
            # Create index file path with binary format
            index_file = self.cache_dir / f"ligand_index_{Path(sdf_path).stem}.bin"
            
            # Build index by scanning SDF file
            ligand_index = {}
            
            if sdf_path.endswith('.gz'):
                import gzip
                with gzip.open(sdf_path, 'rt') as f:
                    self._build_sdf_index(f, ligand_index, sdf_path)
            else:
                with open(sdf_path, 'r') as f:
                    self._build_sdf_index(f, ligand_index, sdf_path)
            
            # Save index in binary format for faster access
            self._save_binary_index(index_file, ligand_index)
            
            self.ligand_index_file = str(index_file)
            self.ligand_index_loaded = True
            
            logger.info(f"Created binary ligand index with {len(ligand_index)} molecules: {self.ligand_index_file}")
            
            return self.ligand_index_file
            
        except Exception as e:
            logger.error(f"Failed to create ligand index: {e}")
            raise
    
    def _save_binary_index(self, index_file: Path, ligand_index: Dict):
        """Save ligand index in binary format for faster access."""
        # Convert to binary format: pdb_id -> (file_path_len, file_path, start_pos, molecule_index)
        with open(index_file, 'wb') as f:
            # Write number of entries
            f.write(len(ligand_index).to_bytes(8, 'little'))
            
            for pdb_id, entry in ligand_index.items():
                # Write PDB ID length and PDB ID
                pdb_id_bytes = pdb_id.encode('utf-8')
                f.write(len(pdb_id_bytes).to_bytes(4, 'little'))
                f.write(pdb_id_bytes)
                
                # Write file path length and file path
                file_path_bytes = entry['file_path'].encode('utf-8')
                f.write(len(file_path_bytes).to_bytes(4, 'little'))
                f.write(file_path_bytes)
                
                # Write start position and molecule index
                f.write(entry['start_pos'].to_bytes(8, 'little'))
                f.write(entry['molecule_index'].to_bytes(4, 'little'))
    
    def _load_binary_index(self, index_file: Path) -> Dict:
        """Load ligand index from binary format."""
        ligand_index = {}
        
        with open(index_file, 'rb') as f:
            # Read number of entries
            num_entries = int.from_bytes(f.read(8), 'little')
            
            for _ in range(num_entries):
                # Read PDB ID
                pdb_id_len = int.from_bytes(f.read(4), 'little')
                pdb_id = f.read(pdb_id_len).decode('utf-8')
                
                # Read file path
                file_path_len = int.from_bytes(f.read(4), 'little')
                file_path = f.read(file_path_len).decode('utf-8')
                
                # Read start position and molecule index
                start_pos = int.from_bytes(f.read(8), 'little')
                molecule_index = int.from_bytes(f.read(4), 'little')
                
                ligand_index[pdb_id] = {
                    'file_path': file_path,
                    'start_pos': start_pos,
                    'molecule_index': molecule_index
                }
        
        return ligand_index
    
    def _build_sdf_index(self, file_handle, index: Dict, sdf_path: str):
        """Build index mapping PDB IDs to file positions."""
        current_pos = 0
        molecule_count = 0
        
        for line in file_handle:
            if line.startswith('$$$$'):
                # End of molecule
                molecule_count += 1
            elif line.startswith('> <_Name>'):
                # Molecule name line
                name_line = line.strip()
                # Read the actual name on next line
                name = next(file_handle).strip()
                if name:
                    pdb_id = name.lower()
                    index[pdb_id] = {
                        'file_path': sdf_path,
                        'start_pos': current_pos,
                        'molecule_index': molecule_count
                    }
            
            current_pos = file_handle.tell()
    
    def preload_all_data(self, embedding_path: Optional[str] = None, 
                        sdf_path: Optional[str] = None) -> Dict[str, str]:
        """
        Pre-load all shared data (embeddings and ligand index).
        
        Args:
            embedding_path: Path to embedding file
            sdf_path: Path to ligand SDF file
            
        Returns:
            Dictionary with paths to shared cache files
        """
        logger.info("Pre-loading all shared data...")
        
        # Pre-load embeddings
        embedding_shm_name, embedding_metadata = self.preload_embeddings(embedding_path)
        
        # Create ligand index
        ligand_index = self.create_ligand_index(sdf_path)
        
        # Create KNN cache for common queries
        knn_cache = self._create_knn_cache()
        
        shared_data = {
            'embedding_shm_name': embedding_shm_name,
            'embedding_metadata': embedding_metadata,
            'ligand_index': ligand_index,
            'knn_cache': knn_cache
        }
        
        logger.info("All shared data pre-loaded successfully")
        return shared_data
    
    def _create_knn_cache(self) -> str:
        """Create cache for pre-computed KNN queries."""
        cache_file = self.cache_dir / "knn_cache.json"
        
        # For now, create empty cache - can be extended to pre-compute common KNN queries
        knn_cache = {
            'created_at': time.time(),
            'cache_type': 'knn_precomputed',
            'entries': {}
        }
        
        with open(cache_file, 'w') as f:
            json.dump(knn_cache, f, indent=2)
        
        return str(cache_file)
    
    def get_shared_cache_files(self) -> Dict[str, str]:
        """Get paths to all shared cache files."""
        return {
            'embedding_shm_name': self.embedding_shm.name if self.embedding_shm else None,
            'embedding_metadata': self.embedding_metadata_file,
            'ligand_index': self.ligand_index_file,
            'knn_cache': self.knn_cache_file
        }
    
    def _cleanup_internal(self):
        """Internal cleanup method called when ref count reaches zero."""
        try:
            if self.embedding_shm:
                self.shm_manager.release_shared_buffer(self.embedding_shm.name)
                self.embedding_shm = None
                logger.info("Cleaned up embedding shared memory")
            
            if self.embedding_metadata_file and Path(self.embedding_metadata_file).exists():
                Path(self.embedding_metadata_file).unlink()
                logger.info(f"Cleaned up embedding metadata: {self.embedding_metadata_file}")
            
            if self.ligand_index_file and Path(self.ligand_index_file).exists():
                Path(self.ligand_index_file).unlink()
                logger.info(f"Cleaned up ligand index: {self.ligand_index_file}")
                
        except Exception as e:
            logger.warning(f"Failed to cleanup shared cache files: {e}")
    
    def cleanup(self):
        """Clean up shared cache files with reference counting."""
        self._decrement_ref_count()
        
        # Additional cleanup for any remaining shared memory objects
        try:
            import multiprocessing.resource_tracker as rt
            
            # Force cleanup of any remaining shared memory objects
            if hasattr(rt, '_CLEANUP_CALLBACKS'):
                for callback in rt._CLEANUP_CALLBACKS:
                    try:
                        callback()
                    except Exception:
                        pass
                        
        except Exception as e:
            logger.debug(f"Additional cleanup failed: {e}")


class EnhancedMemoryMappedLigandLoader:
    """
    Enhanced memory-efficient ligand loader using binary index format.
    
    This class loads specific molecules on-demand using file position
    information from the binary ligand index, avoiding loading the entire SDF file.
    """
    
    def __init__(self, ligand_index_file: str):
        """
        Initialize enhanced memory-mapped ligand loader.
        
        Args:
            ligand_index_file: Path to binary ligand index file
        """
        self.ligand_index_file = ligand_index_file
        self.index = self._load_binary_index()
        self.log = logging.getLogger(__name__)
    
    def _load_binary_index(self) -> Dict:
        """Load ligand index from binary file."""
        index_file = Path(self.ligand_index_file)
        
        # Check if binary format exists, fallback to JSON
        if index_file.suffix == '.bin':
            return self._load_binary_index_format(index_file)
        else:
            # Fallback to JSON format
            with open(self.ligand_index_file, 'r') as f:
                return json.load(f)
    
    def _load_binary_index_format(self, index_file: Path) -> Dict:
        """Load ligand index from binary format."""
        ligand_index = {}
        
        with open(index_file, 'rb') as f:
            # Read number of entries
            num_entries = int.from_bytes(f.read(8), 'little')
            
            for _ in range(num_entries):
                # Read PDB ID
                pdb_id_len = int.from_bytes(f.read(4), 'little')
                pdb_id = f.read(pdb_id_len).decode('utf-8')
                
                # Read file path
                file_path_len = int.from_bytes(f.read(4), 'little')
                file_path = f.read(file_path_len).decode('utf-8')
                
                # Read start position and molecule index
                start_pos = int.from_bytes(f.read(8), 'little')
                molecule_index = int.from_bytes(f.read(4), 'little')
                
                ligand_index[pdb_id] = {
                    'file_path': file_path,
                    'start_pos': start_pos,
                    'molecule_index': molecule_index
                }
        
        return ligand_index
    
    def get_ligand_data(self, pdb_id: str) -> Tuple[Optional[str], Optional["Chem.Mol"]]:
        """
        Load specific ligand data using memory-mapped access.
        
        Args:
            pdb_id: PDB ID of the ligand to load
            
        Returns:
            Tuple of (SMILES, molecule) or (None, None) if not found
        """
        pdb_id_lower = pdb_id.lower()
        
        if pdb_id_lower not in self.index:
            self.log.warning(f"PDB ID {pdb_id} not found in ligand index")
            return None, None
        
        index_entry = self.index[pdb_id_lower]
        sdf_path = index_entry['file_path']
        
        try:
            # Load specific molecule using file position
            molecule, smiles = self._load_molecule_at_position(sdf_path, index_entry)
            return smiles, molecule
            
        except Exception as e:
            self.log.error(f"Failed to load ligand {pdb_id}: {e}")
            return None, None
    
    def _load_molecule_at_position(self, sdf_path: str, index_entry: Dict) -> Tuple[Optional["Chem.Mol"], Optional[str]]:
        """Load molecule at specific file position."""
        try:
            if sdf_path.endswith('.gz'):
                import gzip
                with gzip.open(sdf_path, 'rt') as f:
                    return self._extract_molecule_at_position(f, index_entry)
            else:
                with open(sdf_path, 'r') as f:
                    return self._extract_molecule_at_position(f, index_entry)
                    
        except Exception as e:
            self.log.error(f"Failed to load molecule at position: {e}")
            return None, None
    
    def _extract_molecule_at_position(self, file_handle, index_entry: Dict) -> Tuple[Optional["Chem.Mol"], Optional[str]]:
        """Extract molecule starting at specific file position."""
        start_pos = index_entry['start_pos']
        
        # Seek to position
        file_handle.seek(start_pos)
        
        # Read molecule block
        mol_lines = []
        in_molecule = False
        
        for line in file_handle:
            if line.startswith('$$$$'):
                mol_lines.append(line)
                break
            mol_lines.append(line)
        
        if not mol_lines:
            return None, None
        
        # Parse molecule
        mol_block = ''.join(mol_lines)
        
        try:
            # Use SDMolSupplier for parsing
            from io import StringIO
            supplier = Chem.SDMolSupplier(StringIO(mol_block), removeHs=False, sanitize=False)
            
            for mol in supplier:
                if mol is not None:
                    smiles = Chem.MolToSmiles(mol) if mol else None
                    return mol, smiles
            
            return None, None
            
        except Exception as e:
            self.log.error(f"Failed to parse molecule block: {e}")
            return None, None


def create_enhanced_shared_data_manager(data_dir: Path) -> EnhancedSharedDataManager:
    """Create and initialize enhanced shared data manager."""
    manager = EnhancedSharedDataManager(data_dir)
    return manager


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


# Backward compatibility aliases
SharedDataManager = EnhancedSharedDataManager
MemoryMappedLigandLoader = EnhancedMemoryMappedLigandLoader
create_shared_data_manager = create_enhanced_shared_data_manager 