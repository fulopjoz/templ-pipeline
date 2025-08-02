#!/usr/bin/env python3
"""Template-based MCS pose prediction pipeline"""

import argparse
import gzip
import json
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from datetime import datetime


# TODO: implement the PB - Wim will use it as test 
# remove 3D from the streamlit app and 3D for the aligned - remove features (hide them to the toggle plane
# 

import numpy as np
from tqdm import tqdm

try:
    import pebble
    from pebble import ProcessPool
    PEBBLE_AVAILABLE = True
except ImportError:
    PEBBLE_AVAILABLE = False

try:
    import Bio
    from Bio.PDB import PDBParser, PPBuilder
    from Bio.PDB.PDBExceptions import PDBConstructionWarning
    from Bio.SeqUtils import seq1
    from biotite.structure import (
        AtomArray,
        filter_amino_acids,
        get_chains,
        superimpose,
        superimpose_homologs,
    )
    import biotite.structure as struc
    import biotite.structure.io as bsio
    BIOTITE_AVAILABLE = True
except ImportError:
    BIOTITE_AVAILABLE = False

from rdkit import Chem, RDLogger
from rdkit.Chem import (
    AllChem,
    SanitizeMol,
    rdDistGeom,
    rdForceFieldHelpers,
    rdMolAlign,
    rdRascalMCES,
    rdShapeAlign,
)
from rdkit.Geometry import Point3D
from sklearn.metrics.pairwise import cosine_similarity
from spyrmsd.molecule import Molecule
from spyrmsd.rmsd import rmsdwrapper

try:
    import torch
    # Fix torch.classes compatibility issue  
    torch.classes.__path__ = []
    from transformers import EsmModel, EsmTokenizer
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False

# Import unified error framework
try:
    from templ_pipeline.core.error_framework import UnifiedErrorTracker, ErrorCategory, ErrorSeverity
except ImportError:
    # Fallback if unified framework not available
    class ErrorCategory:
        SYSTEM = "system"
        MCS_FINDING = "mcs_finding"
        CONFORMER_GENERATION = "conformer_generation"
        MOLECULAR_ALIGNMENT = "molecular_alignment"
        COORDINATE_TRANSFORMATION = "coordinate_transformation"
        FORCE_FIELD = "force_field"
        TEMPLATE_PROCESSING = "template_processing"
        VALIDATION = "validation"
        LIGAND_EMBEDDING = "ligand_embedding"
    
    class ErrorSeverity:
        ERROR = "error"
        WARNING = "warning"
        INFO = "info"
    
    class UnifiedErrorTracker:
        def __init__(self, storage_mode="dict"):
            self._errors = {}
            self.storage_mode = storage_mode
        
        def track_error(self, pdb_id, category, message, severity="error", component="unknown", **kwargs):
            self._errors[pdb_id] = {
                "category": category,
                "message": message,
                "severity": severity,
                "component": component,
                "timestamp": time.time()
            }
        
        def has_errors(self, pdb_id=None):
            if pdb_id:
                return pdb_id in self._errors
            return len(self._errors) > 0
        
        def get_errors(self, pdb_id=None):
            if pdb_id:
                return self._errors.get(pdb_id)
            return self._errors
        
        def clear_errors(self, pdb_id=None):
            if pdb_id:
                self._errors.pop(pdb_id, None)
            else:
                self._errors.clear()

# Error tracking system
class PipelineErrorTracker:
    """Enhanced error tracking using unified framework with backward compatibility."""
    
    _unified_tracker = UnifiedErrorTracker(storage_mode="dict")
    _current_pdb_id = None
    _current_stage = None
    _start_time = None
    
    def __init__(self, pdb_id: str, stage: str):
        self.pdb_id = pdb_id
        self.stage = stage
        self.start_time = datetime.now()
        
    def __enter__(self):
        PipelineErrorTracker._current_pdb_id = self.pdb_id
        PipelineErrorTracker._current_stage = self.stage
        PipelineErrorTracker._start_time = self.start_time
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            duration = (datetime.now() - self.start_time).total_seconds()
            
            category = self._map_stage_to_category(self.stage)
            
            self._unified_tracker.track_error(
                pdb_id=self.pdb_id,
                category=category,
                message=str(exc_val),
                severity=ErrorSeverity.ERROR,
                component=self.stage,
                traceback=traceback.format_exc(),
                metadata={
                    "duration_seconds": duration,
                    "error_type": exc_type.__name__
                }
            )
            
            # Log the error
            log = logging.getLogger(__name__)
            log.error(f"Pipeline failure in {self.stage} for {self.pdb_id}: {exc_val}")
            log.debug(f"Full traceback for {self.pdb_id}: {traceback.format_exc()}")
            
        # Reset current context
        PipelineErrorTracker._current_pdb_id = None
        PipelineErrorTracker._current_stage = None
        PipelineErrorTracker._start_time = None
        
        # Don't suppress the exception - let it propagate
        return False
    
    def _map_stage_to_category(self, stage: str) -> str:
        """Map pipeline stage to error category."""
        stage_mapping = {
            "embedding": ErrorCategory.LIGAND_EMBEDDING,
            "template_finding": ErrorCategory.TEMPLATE_PROCESSING,
            "mcs_finding": ErrorCategory.MCS_FINDING,
            "conformer_generation": ErrorCategory.CONFORMER_GENERATION,
            "alignment": ErrorCategory.MOLECULAR_ALIGNMENT,
            "coordinate_transform": ErrorCategory.COORDINATE_TRANSFORMATION,
            "force_field": ErrorCategory.FORCE_FIELD,
            "validation": ErrorCategory.VALIDATION
        }
        return stage_mapping.get(stage.lower(), ErrorCategory.SYSTEM)
    
    @classmethod
    def get_error_summary(cls) -> Dict[str, Any]:
        """Generate comprehensive error summary using unified framework."""
        if hasattr(cls._unified_tracker, 'get_error_summary'):
            summary = cls._unified_tracker.get_error_summary()
            
            # Map to backward-compatible format
            total_errors = summary.get("total_errors", 0)
            
            if total_errors == 0:
                return {
                    "summary": {"total_failures": 0, "success": True},
                    "failures": {}
                }
            
            # Convert unified format to legacy format
            stage_counts = {}
            error_type_counts = {}
            failures = {}
            
            errors = cls._unified_tracker.get_errors()
            for pdb_id, error in errors.items():
                if hasattr(error, 'component'):
                    stage = error.component
                    error_type = error.metadata.get("error_type", "UnknownError") if hasattr(error, 'metadata') and error.metadata else "UnknownError"
                    
                    stage_counts[stage] = stage_counts.get(stage, 0) + 1
                    error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
                    
                    failures[pdb_id] = {
                        "stage": stage,
                        "error_type": error_type,
                        "error_message": error.message if hasattr(error, 'message') else str(error),
                        "timestamp": error.timestamp if hasattr(error, 'timestamp') else time.time(),
                        "duration_seconds": error.metadata.get("duration_seconds", 0) if hasattr(error, 'metadata') and error.metadata else 0,
                        "traceback": error.traceback if hasattr(error, 'traceback') else ""
                    }
                else:
                    # Fallback for simple error format
                    failures[pdb_id] = error
            
            return {
                "summary": {
                    "total_failures": total_errors,
                    "success": False,
                    "failure_by_stage": stage_counts,
                    "failure_by_type": error_type_counts
                },
                "failures": failures
            }
        else:
            # Fallback to simple implementation
            errors = cls._unified_tracker.get_errors()
            return {
                "summary": {
                    "total_failures": len(errors),
                    "success": len(errors) == 0
                },
                "failures": errors
            }
    
    @classmethod
    def save_error_report(cls, output_dir: str, target_pdb: str) -> Optional[str]:
        """Save error report to JSON file using unified framework."""
        if not cls._unified_tracker.has_errors():
            return None
            
        error_summary = cls.get_error_summary()
        
        # Generate unique filename with collision detection
        report_file = get_unique_filename(output_dir, f"{target_pdb}_pipeline_errors", ".json")
        
        try:
            with open(report_file, 'w') as f:
                json.dump(error_summary, f, indent=2)
            return report_file
        except Exception as e:
            log = logging.getLogger(__name__)
            log.error(f"Failed to save error report: {e}")
            return None
    
    @classmethod
    def clear_errors(cls):
        """Clear all recorded errors."""
        cls._unified_tracker.clear_errors()

# ─── suppress RDKit warnings ──────────────────────────────────────────────
RDLogger.DisableLog('rdApp.*')
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

# ─── suppress only the Biotite "guessed from atom name" noise ────────────
warnings.filterwarnings(
    "ignore",
    message=r".*elements were guessed from atom name.*",
    category=UserWarning
)

# ─── collision detection function ───────────────────────────────────────

def get_unique_filename(base_dir: str, base_name: str, extension: str) -> str:
    """Generate unique filename with collision detection.
    
    Args:
        base_dir: Directory where file will be saved
        base_name: Base filename without extension
        extension: File extension (e.g., '.sdf')
        
    Returns:
        Unique filename (with path) that doesn't exist
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Try base name first
    filename = f"{base_name}{extension}"
    full_path = os.path.join(base_dir, filename)
    
    if not os.path.exists(full_path):
        return full_path
    
    # If collision detected, add incremental suffix
    counter = 2
    while True:
        filename = f"{base_name}_v{counter}{extension}"
        full_path = os.path.join(base_dir, filename)
        if not os.path.exists(full_path):
            return full_path
        counter += 1

# ─── ESM model for on-demand embedding ───────────────────────────────────
_esm_components = None

def get_protein_sequence(pdb_file: str, target_chain_id: Optional[str] = None) -> Tuple[Optional[str], List[str]]:
    """Extract protein sequence from PDB file using structure-based coordinates first, with SEQRES fallback.
    
    Args:
        pdb_file (str): Path to PDB file
        target_chain_id (Optional[str]): Specific chain ID to extract
        
    Returns:
        Tuple[Optional[str], List[str]]: (1-letter sequence, List of chain IDs used)
    """
    if not os.path.exists(pdb_file):
        log.warning(f"PDB file not found: {pdb_file}")
        return None, []

    def get_structure_sequence(structure) -> Tuple[Dict[str, str], List[str]]:
        """Extract sequence from structure coordinates using biopython."""
        ppb = PPBuilder()
        chain_sequences = {}
        for chain in structure.get_chains():
            try:
                peptides = ppb.build_peptides(chain)
                seq = "".join(str(pp.get_sequence()) for pp in peptides)
                if seq:
                    chain_sequences[chain.id] = seq
            except Exception as e:
                log.debug(f"Chain {chain.id} sequence error: {str(e)}")
        return chain_sequences

    def seqres_to_1letter(seqres: str) -> str:
        """Convert SEQRES 3-letter codes to 1-letter with validation."""
        conversion = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from Bio.Data.PDBData import protein_letters_3to1_extended as aa_map
            conversion.update(aa_map)
        
        return "".join(
            conversion.get(seqres[i:i+3].upper(), "X")
            for i in range(0, len(seqres), 3)
            if i+3 <= len(seqres)
        )

    try:
        # First try structure-based sequence extraction
        parser = PDBParser(QUIET=True, PERMISSIVE=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PDBConstructionWarning)
            structure = parser.get_structure("protein", pdb_file)
        
        struct_sequences = get_structure_sequence(structure[0])  # Use first model
        if not struct_sequences:
            log.warning("No structure-based sequences found")
            return None, []

        # Now check SEQRES for potential full-length sequence
        seqres_sequences = {}
        with open(pdb_file) as f:
            current_chain = None
            current_seq = []
            for line in f:
                if line.startswith("SEQRES"):
                    chain_id = line[11:12].strip()
                    if chain_id != current_chain:
                        if current_chain is not None:
                            seqres_sequences[current_chain] = "".join(current_seq)
                        current_chain = chain_id
                        current_seq = []
                    current_seq.extend(line[19:].strip().split())

            if current_chain is not None:
                seqres_sequences[current_chain] = "".join(current_seq)

        # Sequence selection logic
        final_seq = None
        used_chains = []
        
        # Convert SEQRES sequences to 1-letter code if available
        if seqres_sequences:
            seqres_1letter = {
                chain: seqres_to_1letter(seq)
                for chain, seq in seqres_sequences.items()
            }
            
            # Find matching chains between SEQRES and structure
            common_chains = set(seqres_1letter) & set(struct_sequences)
            if common_chains:
                target_chain = target_chain_id if target_chain_id in common_chains else sorted(common_chains)[0]
                seqres_seq = seqres_1letter[target_chain]
                struct_seq = struct_sequences[target_chain]
                
                # Validate length consistency
                if len(seqres_seq) == len(struct_seq):
                    final_seq = seqres_seq
                    used_chains = [target_chain]
                    log.debug(f"Using SEQRES sequence for chain {target_chain} (length: {len(final_seq)})")
                else:
                    log.warning(
                        f"SEQRES/structure length mismatch for chain {target_chain}: "
                        f"{len(seqres_seq)} vs {len(struct_seq)}. Using structure sequence."
                    )
                    final_seq = struct_seq
                    used_chains = [target_chain]
        
        # Fallback to structure sequence
        if final_seq is None:
            target_chain = target_chain_id if target_chain_id in struct_sequences else sorted(struct_sequences.keys())[0]
            final_seq = struct_sequences[target_chain]
            used_chains = [target_chain]
            log.debug(f"Using structure-based sequence for chain {target_chain} (length: {len(final_seq)})")

        # Final validation
        if len(final_seq) < MIN_PROTEIN_LENGTH:  # Minimum reasonable protein length
            log.error(f"Sequence too short (length={len(final_seq)})")
            return None, []

        if 'X' in final_seq:
            log.warning(f"Sequence contains {final_seq.count('X')} unknown residues")

        return final_seq, used_chains

    except Exception as e:
        log.error(f"Sequence extraction failed: {str(e)}")
        return None, []

def initialize_esm_model():
    """Initialize ESM model for embedding calculation, cached for reuse."""
    global _esm_components
    if _esm_components is None:
        try:
            from transformers import EsmModel, EsmTokenizer, AutoConfig
            
            # Use the larger model to match create_embeddings_base.py
            model_id = "facebook/esm2_t33_650M_UR50D"
            
            config = AutoConfig.from_pretrained(model_id)
            if hasattr(config, 'use_flash_attention_2'):
                config.use_flash_attention_2 = True
                
            tokenizer = EsmTokenizer.from_pretrained(model_id)
            model = EsmModel.from_pretrained(model_id, config=config, add_pooling_layer=False)
            model.eval()
            
            # Use mixed precision for memory efficiency
            import torch
            if torch.cuda.is_available():
                model = model.to(device="cuda", dtype=torch.float16)
            
            _esm_components = {"tokenizer": tokenizer, "model": model}
            log.info(f"Initialized ESM model: {model_id}")
        except Exception as e:
            log.error(f"Failed to initialize ESM model: {str(e)}")
            return None
    return _esm_components

def calculate_embedding_single(sequence: str, esm_components):
    """Calculate embedding for a single protein sequence."""
    import torch
    
    tokenizer, model = esm_components["tokenizer"], esm_components["model"]
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1022)
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        with torch.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
            outputs = model(**inputs)
        
    # Mean pool over sequence length dimension to get fixed-size vector
    # This matches the approach in create_embeddings_base.py
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

def get_protein_embedding(pdb_file, target_chain_id=None):
    """Get embedding for a protein, extracting sequence first."""
    seq, chains = get_protein_sequence(pdb_file, target_chain_id)
    if not seq:
        log.error(f"Failed to extract sequence from {pdb_file}")
        return None, chains
        
    esm = initialize_esm_model()
    if not esm:
        log.error("Failed to initialize ESM model")
        return None, chains
        
    try:
        # Calculate embedding and ensure it has the right format
        emb = calculate_embedding_single(seq, esm)
        
        # Make sure embedding has the correct shape and is in float32 format
        # to match what's in the NPZ file
        if emb is not None:
            emb = emb.astype(np.float32)
            
        return emb, chains
    except Exception as e:
        log.error(f"Failed to calculate embedding: {str(e)}")
        return None, chains

# ─── EmbeddingManager for unified embedding handling ───────────────────────────────
class EmbeddingManager:
    """
    Unified manager for protein embeddings, handling both pre-computed and on-demand embeddings.
    
    This class manages:
    1. Loading pre-computed embeddings from npz files
    2. Generating on-demand embeddings for proteins not in the database
    3. Finding nearest neighbors using all available embeddings
    4. Maintaining chain information for protein alignment
    5. Caching on-demand embeddings to disk to avoid regeneration
    6. Processing embedding requests in batches for efficiency
    """
    def __init__(self, embedding_path: str, 
                 use_cache: bool = True, 
                 cache_dir: Optional[str] = None,
                 enable_batching: bool = True,
                 max_batch_size: int = 8):
        """Initialize the embedding manager with a path to pre-computed embeddings."""
        self.embedding_path = embedding_path
        self.embedding_db = {}  # Pre-calculated embeddings from NPZ
        self.embedding_chain_data = {}  # Chain data from NPZ
        self.on_demand_embeddings = {}  # Dynamically generated embeddings
        self.on_demand_chain_data = {}  # Chain data for on-demand embeddings
        self.pdb_to_uniprot = {}  # For UniProt exclusion
        
        # Cache configuration
        self.use_cache = use_cache
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/templ/embeddings")
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            # Create model-specific subdirectory based on the model used in initialize_esm_model
            model_dir = os.path.join(self.cache_dir, "esm2_t33_650M_UR50D")
            os.makedirs(model_dir, exist_ok=True)
            self.cache_dir = model_dir
            
        # Batch processing
        self.enable_batching = enable_batching
        self.max_batch_size = max_batch_size
        self.pending_embedding_requests = []
        
        self._load_embeddings()
        
    def _load_embeddings(self):
        """Load pre-computed embeddings from NPZ file."""
        if not os.path.exists(self.embedding_path):
            log.warning(f"Embedding file not found: {self.embedding_path}")
            return False
            
        try:
            data = np.load(self.embedding_path, allow_pickle=True)
            pdb_ids = data['pdb_ids']
            embeddings = data['embeddings']
            chain_ids = data.get('chain_ids', None)
            
            # Populate embedding database
            for i, pid in enumerate(pdb_ids):
                self.embedding_db[pid] = embeddings[i]
                if chain_ids is not None:
                    self.embedding_chain_data[pid] = chain_ids[i]
            
            log.info(f"Loaded {len(self.embedding_db)} embeddings from {self.embedding_path}")
            return True
        except Exception as e:
            log.error(f"Error loading embeddings: {str(e)}")
            return False
    
    def set_uniprot_mapping(self, pdb_to_uniprot_map: Dict[str, str]):
        """Set UniProt mapping for template filtering."""
        self.pdb_to_uniprot = pdb_to_uniprot_map
    
    def _get_cache_path(self, pdb_id: str, chain_id: Optional[str] = None) -> str:
        """Generate a path for the cached embedding file."""
        if chain_id:
            filename = f"{pdb_id}_{chain_id}.npz"
        else:
            filename = f"{pdb_id}.npz"
        return os.path.join(self.cache_dir, filename)
    
    def _save_to_cache(self, pdb_id: str, embedding: np.ndarray, chains: List[str]) -> bool:
        """Save embedding to cache."""
        try:
            if not self.use_cache:
                return False
                
            chain_str = ",".join(chains) if chains else ""
            cache_path = self._get_cache_path(pdb_id, chain_str)
            
            # Save embedding with metadata
            np.savez_compressed(
                cache_path, 
                embedding=embedding,
                chain_ids=chain_str,
                timestamp=time.time(),
                model_id="esm2_t33_650M_UR50D"  # Match the model used in initialize_esm_model
            )
            log.debug(f"Saved embedding to cache: {cache_path}")
            return True
        except Exception as e:
            log.error(f"Failed to save embedding to cache: {str(e)}")
            return False
    
    def _load_from_cache(self, pdb_id: str, target_chain_id: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Load embedding from cache if it exists."""
        try:
            if not self.use_cache:
                return None, None
                
            # Try with specific chain first if provided
            if target_chain_id:
                cache_path = self._get_cache_path(pdb_id, target_chain_id)
                if os.path.exists(cache_path):
                    data = np.load(cache_path, allow_pickle=True)
                    log.debug(f"Loaded embedding from cache (with chain): {cache_path}")
                    return data['embedding'], data['chain_ids']
            
            # Try without specific chain
            cache_path = self._get_cache_path(pdb_id)
            if os.path.exists(cache_path):
                data = np.load(cache_path, allow_pickle=True)
                log.debug(f"Loaded embedding from cache: {cache_path}")
                return data['embedding'], data['chain_ids']
            
            return None, None
        except Exception as e:
            log.error(f"Failed to load embedding from cache: {str(e)}")
            return None, None
    
    def is_in_cache(self, pdb_id: str, target_chain_id: Optional[str] = None) -> bool:
        """Check if embedding exists in cache."""
        if not self.use_cache:
            return False
            
        # Check with specific chain first if provided
        if target_chain_id:
            if os.path.exists(self._get_cache_path(pdb_id, target_chain_id)):
                return True
        
        # Check without specific chain
        return os.path.exists(self._get_cache_path(pdb_id))
    
    def clear_cache(self) -> bool:
        """Clear the embedding cache."""
        try:
            if not self.use_cache:
                return False
                
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.npz'):
                    os.remove(os.path.join(self.cache_dir, filename))
            log.info(f"Cleared embedding cache: {self.cache_dir}")
            return True
        except Exception as e:
            log.error(f"Failed to clear cache: {str(e)}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        try:
            if not self.use_cache or not os.path.exists(self.cache_dir):
                return {"enabled": False, "count": 0, "size_mb": 0}
                
            files = [f for f in os.listdir(self.cache_dir) if f.endswith('.npz')]
            total_size = sum(os.path.getsize(os.path.join(self.cache_dir, f)) for f in files)
            
            return {
                "enabled": True,
                "count": len(files),
                "size_mb": total_size / (1024 * 1024),
                "path": self.cache_dir
            }
        except Exception as e:
            log.error(f"Failed to get cache stats: {str(e)}")
            return {"enabled": self.use_cache, "error": str(e)}
    
    def _generate_embedding(self, pdb_id: str, pdb_file: Optional[str], 
                           target_chain_id: Optional[str]) -> Tuple[Optional[np.ndarray], List[str]]:
        """Generate embedding for a protein."""
        embedding, chains = get_protein_embedding(pdb_file, target_chain_id)
        return embedding, chains
    
    def get_embedding(self, pdb_id: str, pdb_file: Optional[str] = None, 
                      target_chain_id: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Get embedding for a protein, either from cache, pre-computed dataset, or generated on-demand.
        
        Args:
            pdb_id: PDB ID for the protein
            pdb_file: Path to PDB file for on-demand embedding generation
            target_chain_id: Specific chain to use
            
        Returns:
            Tuple of (embedding array, chain_ids string)
        """
        # Check if embedding exists in pre-computed database
        if pdb_id in self.embedding_db:
            return self.embedding_db[pdb_id], self.embedding_chain_data.get(pdb_id, "")
            
        # Check if embedding was already generated on-demand
        if pdb_id in self.on_demand_embeddings:
            return self.on_demand_embeddings[pdb_id], self.on_demand_chain_data.get(pdb_id, "")
            
        # Check cache
        if self.use_cache:
            cached_emb, cached_chains = self._load_from_cache(pdb_id, target_chain_id)
            if cached_emb is not None:
                # Store in memory for future use
                self.on_demand_embeddings[pdb_id] = cached_emb
                self.on_demand_chain_data[pdb_id] = cached_chains
                return cached_emb, cached_chains
            
        # Generate embedding on-demand if PDB file path is provided
        if pdb_file and os.path.exists(pdb_file):
            log.info(f"Generating on-demand embedding for {pdb_id}")
            embedding, chains = self._generate_embedding(pdb_id, pdb_file, target_chain_id)
            
            if embedding is not None and len(chains) > 0:
                self.on_demand_embeddings[pdb_id] = embedding
                self.on_demand_chain_data[pdb_id] = ",".join(chains)
                
                # Save to cache if enabled
                if self.use_cache:
                    self._save_to_cache(pdb_id, embedding, chains)
                    
                log.info(f"Successfully generated on-demand embedding for {pdb_id}")
                return embedding, self.on_demand_chain_data[pdb_id]
            else:
                log.error(f"Failed to generate on-demand embedding for {pdb_id}")
        
        return None, None
    
    def add_to_batch(self, pdb_id: str, pdb_file: str, target_chain_id: Optional[str] = None) -> bool:
        """Add a protein to the batch for embedding generation."""
        if not self.enable_batching:
            return False
            
        # Skip if already available
        if pdb_id in self.embedding_db or pdb_id in self.on_demand_embeddings:
            return False
            
        # Skip if in cache
        if self.use_cache and self.is_in_cache(pdb_id, target_chain_id):
            return False
            
        # Add to pending requests
        if os.path.exists(pdb_file):
            self.pending_embedding_requests.append((pdb_id, pdb_file, target_chain_id))
            return True
        else:
            log.warning(f"PDB file not found for batch request: {pdb_file}")
            return False
    
    def process_batch(self) -> int:
        """Process all pending embedding requests in batch."""
        if not self.enable_batching or not self.pending_embedding_requests:
            return 0
            
        log.info(f"Processing batch of {len(self.pending_embedding_requests)} embeddings")
        
        # Extract sequences first
        batch_data = []
        for pdb_id, pdb_file, target_chain_id in self.pending_embedding_requests:
            seq, chains = get_protein_sequence(pdb_file, target_chain_id)
            if seq and chains:
                batch_data.append((pdb_id, pdb_file, seq, chains, target_chain_id))
            else:
                log.warning(f"Failed to extract sequence for {pdb_id} in batch processing")
        
        if not batch_data:
            self.pending_embedding_requests = []
            return 0
            
        # Sort by sequence length for optimal batching
        batch_data.sort(key=lambda x: len(x[2]))
        
        # Initialize ESM model
        esm = initialize_esm_model()
        if not esm:
            log.error("Failed to initialize ESM model for batch processing")
            self.pending_embedding_requests = []
            return 0
        
        # Process in smaller batches based on sequence length
        successful_count = 0
        current_idx = 0
        
        # Group sequences by length ranges for better efficiency
        length_groups = {
            "small": [],  # <500
            "medium": [], # 500-1000
            "large": []   # >1000
        }
        
        for item in batch_data:
            seq_len = len(item[2])
            if seq_len < 500:
                length_groups["small"].append(item)
            elif seq_len < 1000:
                length_groups["medium"].append(item)
            else:
                length_groups["large"].append(item)
        
        # Process each length group with appropriate batch size
        for group_name, group_items in length_groups.items():
            if not group_items:
                continue
                
            # Set batch size based on group
            if group_name == "small":
                group_batch_size = min(self.max_batch_size, 8)
            elif group_name == "medium":
                group_batch_size = min(self.max_batch_size, 4)
            else:  # large
                group_batch_size = min(self.max_batch_size, 2)
                
            log.debug(f"Processing {len(group_items)} {group_name} sequences in batches of {group_batch_size}")
            
            # Process this group in batches
            for i in range(0, len(group_items), group_batch_size):
                batch_subset = group_items[i:i+group_batch_size]
                
                try:
                    # Calculate embeddings
                    batch_embeddings = self._process_sequence_batch(
                        [item[2] for item in batch_subset], 
                        esm
                    )
                    
                    # Store results
                    for j, emb in enumerate(batch_embeddings):
                        if emb is not None:
                            pdb_id, pdb_file, _, chains, _ = batch_subset[j]
                            chain_str = ",".join(chains)
                            
                            # Store in memory
                            self.on_demand_embeddings[pdb_id] = emb
                            self.on_demand_chain_data[pdb_id] = chain_str
                            
                            # Save to cache
                            if self.use_cache:
                                self._save_to_cache(pdb_id, emb, chains)
                                
                            successful_count += 1
                except Exception as e:
                    log.error(f"Error processing batch: {str(e)}")
                    # If we hit an error, try with smaller batch size
                    if group_batch_size > 1:
                        # Process one by one
                        for item in batch_subset:
                            pdb_id, pdb_file, seq, chains, target_chain_id = item
                            try:
                                emb = calculate_embedding_single(seq, esm)
                                if emb is not None:
                                    chain_str = ",".join(chains)
                                    
                                    # Store in memory
                                    self.on_demand_embeddings[pdb_id] = emb
                                    self.on_demand_chain_data[pdb_id] = chain_str
                                    
                                    # Save to cache
                                    if self.use_cache:
                                        self._save_to_cache(pdb_id, emb, chains)
                                        
                                    successful_count += 1
                            except Exception as e_inner:
                                log.error(f"Error processing individual item in batch fallback: {str(e_inner)}")
        
        # Clear pending requests
        self.pending_embedding_requests = []
        return successful_count
    
    def _process_sequence_batch(self, sequences: List[str], esm_components) -> List[Optional[np.ndarray]]:
        """Process a batch of sequences to generate embeddings."""
        import torch
        
        tokenizer, model = esm_components["tokenizer"], esm_components["model"]
        results = []
        
        try:
            # Tokenize all sequences
            inputs_list = [tokenizer(seq, return_tensors="pt", truncation=True, max_length=1022) 
                          for seq in sequences]
            
            # Move to device and stack into a batch
            device = next(model.parameters()).device
            
            # Process each sequence individually but in a loop to maintain batch context
            for inputs in inputs_list:
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                        outputs = model(**inputs)
                    
                # Mean pool over sequence length dimension
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                
                # Ensure correct format (float32 to match NPZ)
                if embedding is not None:
                    embedding = embedding.astype(np.float32)
                
                results.append(embedding)
            
            return results
        except Exception as e:
            log.error(f"Error in batch processing: {str(e)}")
            # Try to free memory
            if 'inputs' in locals():
                del inputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return [None] * len(sequences)
    
    def prepare_batch_embeddings(self, pdb_ids: List[str]) -> int:
        """Prepare and process embeddings for a list of PDB IDs."""
        added_count = 0
        for pdb_id in pdb_ids:
            # Skip if already available
            if pdb_id in self.embedding_db or pdb_id in self.on_demand_embeddings:
                continue
                
            # Skip if in cache
            if self.use_cache and self.is_in_cache(pdb_id):
                cached_emb, cached_chains = self._load_from_cache(pdb_id)
                if cached_emb is not None:
                    self.on_demand_embeddings[pdb_id] = cached_emb
                    self.on_demand_chain_data[pdb_id] = cached_chains
                continue
                
            # Add to batch request
            pdb_file_path = pdb_path(pdb_id)
            if pdb_file_path:
                if self.add_to_batch(pdb_id, pdb_file_path):
                    added_count += 1
        
        if added_count > 0:
            # Process the batch
            processed = self.process_batch()
            log.info(f"Batch processed {processed}/{added_count} embeddings")
            return processed
        
        return 0
    
    def find_neighbors(self, query_pdb: str, query_embedding: Optional[np.ndarray] = None, 
                      exclude_uniprot_ids: Set[str] = None, exclude_pdb_ids: Set[str] = None,
                      allowed_pdb_ids: Optional[Set[str]] = None,
                      k: Optional[int] = None, similarity_threshold: Optional[float] = None,
                      return_similarities: bool = False) -> List:
        """
        Find nearest neighbors for a query protein.
        
        Args:
            query_pdb: PDB ID of query protein
            query_embedding: Optional pre-computed embedding for the query
            exclude_uniprot_ids: UniProt IDs to exclude
            exclude_pdb_ids: PDB IDs to exclude
            allowed_pdb_ids: Only consider these PDB IDs (if provided)
            k: Number of nearest neighbors to return (if provided)
            similarity_threshold: Minimum similarity threshold (overrides KNN if provided)
            return_similarities: Return similarity scores with neighbors
            
        Returns:
            List of nearest neighbors (PDB IDs or tuples of (PDB ID, similarity))
        """
        if exclude_pdb_ids is None:
            exclude_pdb_ids = set()
        
        if exclude_uniprot_ids is None:
            exclude_uniprot_ids = set()
            
        # Ensure query PDB is not returned as neighbor
        exclude_pdb_ids.add(query_pdb)
        
        # Get query embedding
        if query_embedding is None:
            query_embedding, _ = self.get_embedding(query_pdb)
            if query_embedding is None:
                log.error(f"No embedding found for query PDB {query_pdb}")
                return []
        
        # Prepare all available embeddings for search
        all_candidate_pdb_ids = []
        all_embeddings = []
        
        # Combine pre-computed and on-demand embeddings
        for emb_dict, chain_dict in [(self.embedding_db, self.embedding_chain_data), 
                                     (self.on_demand_embeddings, self.on_demand_chain_data)]:
            for pid, emb in emb_dict.items():
                # Skip if in exclusion list
                if pid in exclude_pdb_ids:
                    continue
                    
                # Skip if not in allowed list (when provided)
                if allowed_pdb_ids is not None and pid not in allowed_pdb_ids:
                    continue
                    
                # Skip if UniProt ID should be excluded
                if exclude_uniprot_ids and pid in self.pdb_to_uniprot:
                    if self.pdb_to_uniprot[pid] in exclude_uniprot_ids:
                        continue
                
                all_candidate_pdb_ids.append(pid)
                all_embeddings.append(emb)
        
        if not all_embeddings:
            log.warning("No valid template candidates found after filtering.")
            return []
            
        # Calculate similarities
        stacked_embeddings = np.vstack(all_embeddings)
        similarities = cosine_similarity(stacked_embeddings, query_embedding.reshape(1, -1)).flatten()
        
        # Create list of (PDB ID, similarity) tuples and sort by similarity
        neighbor_candidates = list(zip(all_candidate_pdb_ids, similarities))
        neighbor_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Apply similarity threshold or k nearest neighbors filter
        if similarity_threshold is not None:
            neighbor_candidates = [(pid, sim) for pid, sim in neighbor_candidates if sim >= similarity_threshold]
            log.debug(f"Found {len(neighbor_candidates)} neighbors with similarity >= {similarity_threshold:.3f}")
        elif k is not None:
            neighbor_candidates = neighbor_candidates[:k]
            log.debug(f"Selected top {len(neighbor_candidates)} nearest neighbors")
        
        # Return with or without similarity scores
        if return_similarities:
            return neighbor_candidates
        else:
            return [pid for pid, _ in neighbor_candidates]
    
    def get_chain_data(self, pdb_id: str) -> Optional[str]:
        """Get chain data for a PDB ID from either pre-computed or on-demand sources."""
        if pdb_id in self.embedding_chain_data:
            return self.embedding_chain_data[pdb_id]
        elif pdb_id in self.on_demand_chain_data:
            return self.on_demand_chain_data[pdb_id]
        return None

# ─── configuration ──────────────────────────────────────────────────────
TARGET_PDB      = "1zoe"
TARGET_SMILES   = "CN(C)c1nc2c(Br)c(Br)c(Br)c(Br)c2[nH]1"
DATA_DIR        = "data"  # Fixed: Use correct relative path from mcs/ directory
LIGANDS_SDF_GZ  = f"{DATA_DIR}/ligands/templ_processed_ligands_v1.0.0.sdf.gz"
EMB_NPZ         = f"{DATA_DIR}/embeddings/templ_protein_embeddings_v1.0.0.npz"
UNIPROT_MAP     = f"{DATA_DIR}/pdbbind_dates.json"  # JSON with PDB deposition dates and UniProt IDs
OUT_DIR         = "output"
# os.makedirs(OUT_DIR, exist_ok=True) # Moved to main, after arg parsing for OUT_DIR

# Pipeline parameters (default values, can be overridden by command line args)
SIM_THRESHOLD   = 0.90
N_CONFS         = 200
CA_RMSD_THRESHOLD = 10.0  # Maximum C-alpha RMSD in Angstroms for protein filtering
# Progressive fallback thresholds for CA RMSD filtering
CA_RMSD_FALLBACK_THRESHOLDS = [10.0, 15.0, 20.0]

# ─── algorithm constants ─────────────────────────────────────────────────
# MMFF optimization
DEFAULT_MMFF_ITERATIONS = 200
MMFF_VARIANT = "MMFF94s"


# Protein alignment parameters
MIN_ANCHOR_RESIDUES = 15
MIN_PROTEIN_LENGTH = 20
MIN_CA_ATOMS_FOR_ALIGNMENT = 3
BLOSUM_GAP_PENALTY = -10

# ESM model configuration
ESM_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
ESM_MAX_SEQUENCE_LENGTH = 1022

# Shape alignment scoring
COMBO_WEIGHT = 0.5  # For combining shape and color scores

# ─── parallelism tuning ──────────────────────────────────────────────────
# N_WORKERS_PIPELINE will be determined in main() based on args.multiprocessing
# EMBED_THREADS for RDKit's EmbedMultipleConfs will also be based on N_WORKERS_PIPELINE
N_WORKERS_PIPELINE = 1 # Default, will be updated in main()
EMBED_THREADS = 0      # Default for RDKit (use all available for that step), might be updated

# Dictionary to store loaded structures
LOADED_STRUCTURES = {}

# Reference structure loaded on-demand
REF_STRUCT = None

# ─── logging setup ──────────────────────────────────────────────────────
# Allow setting log level via environment variable
default_log_level = os.environ.get('MCS_LOG_LEVEL', 'INFO')
try:
    log_level = getattr(logging, default_log_level)
except AttributeError:
    log_level = logging.WARNING

# Initialize with a default level - will be overridden by command line
logging.basicConfig(
    level=log_level,  # Use environment variable or default to WARNING
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s"
)
log = logging.getLogger("true-MCS")

# Force all loggers to use our log level (including third-party loggers)
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(log_level)


def reset_logging(level):
    """Reset all logging handlers to use the specified level.
    
    This fixes issues where changing log.setLevel() doesn't affect existing handlers.
    """
    # Set level for our logger
    log.setLevel(level)
    
    # Set level for root logger and its handlers
    logging.root.setLevel(level)
    for handler in logging.root.handlers:
        handler.setLevel(level)
        
    # Set level for all other loggers too
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(level)


def safe_name(m: Chem.Mol, default: str) -> str:
    if m.HasProp("_Name"):
        return m.GetProp("_Name")
    m.SetProp("_Name", default)
    return default


def rmsd_raw(a: Chem.Mol, b: Chem.Mol) -> float:
    """Evaluate final full‐ligand RMSD via sPyRMSD (evaluation only)."""
    try:
        return rmsdwrapper(
            Molecule.from_rdkit(a),
            Molecule.from_rdkit(b),
            minimize=False, strip=True, symmetry=True
        )[0]
    except AssertionError:
        return float("nan")


# ─── organometallic handling ──────────────────────────────────────────────

def detect_and_substitute_organometallic(mol: Chem.Mol, molecule_name: str = "unknown") -> Tuple[Chem.Mol, bool, List[str]]:
    """Detect organometallic atoms and substitute with carbon to enable processing.
    
    This enables processing of molecules containing metal atoms that would otherwise fail
    in RDKit sanitization and downstream operations.
    
    Args:
        mol: Input molecule
        molecule_name: Name for logging purposes
        
    Returns:
        Tuple of (processed_molecule, had_metals, substitutions_made)
    """
    # Common organometallic atoms found in drug discovery
    ORGANOMETALLIC_ATOMS = {
        'Fe', 'Mn', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Pd', 'Ag', 'Cd', 'Pt', 'Au', 'Hg',
        'Mo', 'W', 'Cr', 'V', 'Ti', 'Sc', 'Y', 'Zr', 'Nb', 'Tc', 'Re', 'Os', 'Ir'
    } # TODO: find out3rj7 is failing for this 
    
    if mol is None:
        return None, False, []
    
    # Create a copy to avoid modifying the original
    mol_copy = Chem.Mol(mol)
    
    # Check for organometallic atoms
    organometallic_found = []
    substitutions = []
    
    for atom in mol_copy.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in ORGANOMETALLIC_ATOMS:
            organometallic_found.append(f"{symbol}(idx:{atom.GetIdx()})")
            
            # Substitute with carbon - preserve connectivity
            atom.SetAtomicNum(6)  # Carbon
            atom.SetFormalCharge(0)
            substitutions.append(f"{symbol}→C(idx:{atom.GetIdx()})")
    
    had_metals = len(organometallic_found) > 0
    
    if had_metals:
        log.info(f"Organometallic handling for {molecule_name}: Found {organometallic_found}, substituted with carbon: {substitutions}")
        
        try:
            # Attempt to sanitize after substitution
            Chem.SanitizeMol(mol_copy)
        except Exception as e:
            log.warning(f"Sanitization still failed for {molecule_name} after organometallic substitution: {e}")
            # Return the modified molecule even if sanitization fails - some operations might still work
            return mol_copy, had_metals, substitutions
    
    return mol_copy, had_metals, substitutions


def needs_uff_fallback(mol: Chem.Mol) -> bool:
    """
    Check if a molecule needs UFF instead of MMFF due to problematic atoms.
    
    Args:
        mol: RDKit molecule object
    
    Returns:
        bool: True if UFF should be used instead of MMFF
        
    Common problematic atoms that work better with UFF:
    - Transition metals (particularly organometallics)
    - Some metalloids in unusual coordination environments
    """
    if mol is None:
        return False
    
    problematic_atomic_nums = {
        # Transition metals that commonly cause MMFF issues
        25,  # Mn
        26,  # Fe  
        27,  # Co
        28,  # Ni
        29,  # Cu
        30,  # Zn
        42,  # Mo
        74,  # W
        75,  # Re
        76,  # Os
        77,  # Ir
        78,  # Pt
        79,  # Au
        80,  # Hg
    }
    
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() in problematic_atomic_nums:
            return True
    
    return False

def has_rhenium_complex(mol: Chem.Mol) -> Tuple[bool, str]:
    """
    Check for rhenium atoms which are problematic for embedding.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Tuple[bool, str]: (True if rhenium found, warning message)
    """
    if mol is None:
        return False, ""
    
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 75:  # Rhenium
            warning_msg = (
                "❌ Target contains Re+9 organometallic complex - cannot be processed. "
                "This pipeline is designed for small molecules only. "
                "Note: PDBBind often has incorrect SMILES for organometallics. "
                "For correct structure see: https://www.rcsb.org/ligand/RCS "
                "Technical reference: https://pubs.acs.org/doi/10.1021/om00086a032"
            )
            return True, warning_msg
    
    return False, ""

def is_large_peptide(mol: Chem.Mol, residue_threshold: int = 8) -> Tuple[bool, str]:
    """
    Check if molecule is a large peptide (>8 amino acid residues).
    
    Args:
        mol: RDKit molecule object
        residue_threshold: Number of amino acid residues above which to consider "large"
        
    Returns:
        Tuple[bool, str]: (True if large peptide, warning message)
    """
    if mol is None:
        return False, ""
    
    # SMARTS pattern for amino acid backbone (amide bond pattern)
    # [NX3][CX3](=O)[CX4] matches N-C(=O)-C typical of peptide bonds
    peptide_backbone_pattern = "[NX3][CX3](=O)[CX4]"
    
    try:
        pattern = Chem.MolFromSmarts(peptide_backbone_pattern)
        if pattern is not None:
            matches = mol.GetSubstructMatches(pattern)
            residue_count = len(matches)
            
            if residue_count > residue_threshold:
                warning_msg = (
                    f" Target is a large peptide ({residue_count} residues > {residue_threshold} threshold) - cannot be processed. "
                    "This pipeline is designed for drug-like small molecules only. "
                    "Large peptides require specialized conformational sampling methods."
                )
                return True, warning_msg
        else:
            # Pattern compilation failed, fall back to conservative estimation
            raise Exception("SMARTS pattern compilation failed")
            
    except Exception as e:
        # Fallback to atom counting if SMARTS fails
        log.debug(f"SMARTS pattern matching failed, falling back to atom count estimation: {e}")
        non_h_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1)
        # Conservative conversion: ~6-8 atoms per residue average
        estimated_residues = non_h_atoms // 6
        if estimated_residues > residue_threshold:
            warning_msg = (
                f" Target appears to be a large peptide (estimated {estimated_residues} residues > {residue_threshold} threshold) - cannot be processed. "
                "This pipeline is designed for drug-like small molecules only. "
                "Large peptides require specialized conformational sampling methods."
            )
            return True, warning_msg
    
    return False, ""

def validate_target_molecule(mol: Chem.Mol, mol_name: str = "unknown", peptide_threshold: int = 8) -> Tuple[bool, str]:
    """
    Validate if a target molecule can be processed by the pipeline.
    
    Args:
        mol: RDKit molecule object
        mol_name: Name/identifier for the molecule (for logging)
        peptide_threshold: Maximum number of peptide residues allowed
        
    Returns:
        Tuple[bool, str]: (True if valid, warning message if invalid)
    """
    if mol is None:
        return False, f"❌ Invalid molecule object for {mol_name}"
    
    # Check for rhenium complexes (like 3rj7)
    has_re, re_msg = has_rhenium_complex(mol)
    if has_re:
        return False, f"{mol_name}: {re_msg}"
    
    # Check for large peptides
    is_large, large_msg = is_large_peptide(mol, peptide_threshold)
    if is_large:
        return False, f"{mol_name}: {large_msg}"
    
    return True, ""

def embed_with_uff_fallback(mol: Chem.Mol, n_conformers: int, ps, coordMap: dict = None) -> List[int]:
    """Embedding with UFF compatibility for organometallic molecules.
    
    Args:
        mol: Molecule to embed (should have hydrogens added)
        n_conformers: Number of conformers to generate
        ps: Original embedding parameters
        
    Returns:
        List of conformer IDs
    """
    # Try standard embedding first (might work even with metals)
    try:
        # Use individual parameters instead of EmbedParameters when coordMap is needed
        conf_ids = rdDistGeom.EmbedMultipleConfs(
            mol, n_conformers,
            maxAttempts=ps.maxIterations if hasattr(ps, 'maxIterations') else 1000,
            randomSeed=-1,
            useRandomCoords=ps.useRandomCoords if hasattr(ps, 'useRandomCoords') else True,
            enforceChirality=ps.enforceChirality if hasattr(ps, 'enforceChirality') else False,
            numThreads=ps.numThreads if hasattr(ps, 'numThreads') else 0,
            coordMap=coordMap or {}
        )
        if conf_ids:
            log.debug(f"Standard embedding succeeded for organometallic molecule")
            return conf_ids
    except Exception as e:
        log.debug(f"Standard embedding failed for organometallic molecule: {e}")
    
            # Fallback: simplified embedding parameters for difficult molecules
        log.info(f"Using UFF-compatible embedding parameters for organometallic molecule")
        ps_uff = rdDistGeom.ETKDGv3()
        ps_uff.numThreads = ps.numThreads if hasattr(ps, 'numThreads') else 0
        ps_uff.maxIterations = (ps.maxIterations if hasattr(ps, 'maxIterations') else 1000) * 2  # More attempts
        ps_uff.useRandomCoords = True
        ps_uff.randomSeed = 42  # Reproducible results
        ps_uff.enforceChirality = False  # Critical: match original working code
    
    # Remove coordinate constraints that might cause issues with metals
    if hasattr(ps, 'coordMap') and ps.coordMap:
        log.debug("Removing coordinate constraints for UFF embedding")
        ps_uff.coordMap = {}
    
    try:
        conf_ids = rdDistGeom.EmbedMultipleConfs(mol, n_conformers, ps_uff)
        if conf_ids:
            log.debug(f"UFF-compatible embedding succeeded, generated {len(conf_ids)} conformers")
            return conf_ids
    except Exception as e:
        log.warning(f"UFF-compatible embedding also failed: {e}")
    
    # Final fallback: try with even simpler parameters
    log.warning("Attempting final fallback embedding with minimal constraints")
    ps_minimal = rdDistGeom.ETKDGv3()
    ps_minimal.numThreads = 1  # Single thread for stability
    ps_minimal.maxIterations = 500  # Fewer attempts but simpler
    ps_minimal.useRandomCoords = True
    ps_minimal.randomSeed = 42
    ps_minimal.enforceChirality = False  # Critical: match original working code
    
    try:
        return rdDistGeom.EmbedMultipleConfs(mol, min(n_conformers, 50), ps_minimal)  # Limit conformers
    except Exception as e:
        log.error(f"All embedding attempts failed for organometallic molecule: {e}")
        return []


def minimize_with_uff(mol: Chem.Mol, conf_ids: List[int], fixed_atoms: List[int] = None, max_its: int = DEFAULT_MMFF_ITERATIONS):
    """UFF minimization for organometallic molecules.
    
    Args:
        mol: Molecule with conformers
        conf_ids: List of conformer IDs to minimize
        fixed_atoms: List of atom indices to keep fixed (optional)
        max_its: Maximum iterations for minimization
    """
    if not conf_ids:
        return
        
    fixed_atoms = fixed_atoms or []
    successful_minimizations = 0
    
    for cid in conf_ids:
        try:
            ff = rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=cid)
            if ff:
                # Add fixed points if specified
                for fixed_idx in fixed_atoms:
                    if fixed_idx < mol.GetNumAtoms():
                        ff.AddFixedPoint(fixed_idx)
                
                # Perform minimization
                result = ff.Minimize(maxIts=max_its)
                if result == 0:  # Success
                    successful_minimizations += 1
                else:
                    log.debug(f"UFF minimization returned code {result} for conformer {cid}")
            else:
                log.debug(f"Could not create UFF force field for conformer {cid}")
        except Exception as e:
            log.debug(f"UFF minimization failed for conformer {cid}: {e}")
    
    log.debug(f"UFF minimization succeeded for {successful_minimizations}/{len(conf_ids)} conformers")


def score_and_align(conf: Chem.Mol, tpl: Chem.Mol) -> Tuple[Dict[str, float], Chem.Mol]:
    """Compute shape/color scores and return the aligned conformer with organometallic handling."""
    prb = Chem.Mol(conf)
    
    # Handle organometallic atoms before sanitization
    tpl_processed, tpl_had_metals, tpl_subs = detect_and_substitute_organometallic(tpl, "template")
    prb_processed, prb_had_metals, prb_subs = detect_and_substitute_organometallic(prb, "probe")
    
    try:
        SanitizeMol(tpl_processed)
        SanitizeMol(prb_processed)
    except Exception as e:
        log.warning(f"Sanitization failed even after organometallic handling: {e}")
        # Continue with original molecules as fallback
        try:
            SanitizeMol(tpl)
            SanitizeMol(prb)
            tpl_processed, prb_processed = tpl, prb
        except Exception as e2:
            log.error(f"Both organometallic handling and fallback sanitization failed: {e2}")
            # Return default scores to avoid complete failure
            return ({"shape": 0.0, "color": 0.0, "combo": 0.0}, prb)
    
    try:
        sT, cT = rdShapeAlign.AlignMol(tpl_processed, prb_processed, useColors=True)
        return ({"shape": sT, "color": cT, "combo": COMBO_WEIGHT*(sT+cT)}, prb_processed)
    except Exception as e:
        log.warning(f"Shape alignment failed: {e}")
        return ({"shape": 0.0, "color": 0.0, "combo": 0.0}, prb_processed)


def get_central_atom(mol: Chem.Mol) -> int:
    """Get the idx of the central atom by picking the atom with the smallest
    sum of shortest path to all the other atoms. In case of tie returns the first one.
    
    Args:
        mol: Input molecule
        
    Returns:
        Index of the central atom
    """
    tdm = Chem.GetDistanceMatrix(mol)
    pathsum = [(idx, sum(row)) for idx, row in enumerate(tdm)]
    return sorted(pathsum, key=lambda x: x[1])[0][0]


def find_mcs(tgt: Chem.Mol, refs: List[Chem.Mol], return_details: bool = False) -> Union[Tuple[int, str], Tuple[int, str, Dict]]:
    """Enhanced MCS finding with central atom fallback - never fails completely.
    
    Args:
        tgt: Target molecule
        refs: Reference molecules
        return_details: If True, return detailed MCS information
        
    Returns:
        If return_details=False: (best_template_index, smarts)
        If return_details=True: (best_template_index, smarts, mcs_details_dict)
    """
    opts = rdRascalMCES.RascalOptions()
    opts.singleLargestFrag = True
    opts.similarityThreshold = 0.9  # Start higher like original working code
    
    mcs_details = {}  # Only populated if return_details=True
    min_acceptable_size = 5
    desperate_threshold = 0.2
    
    while True:  # Continue until we find something acceptable
        hits = []
        for i, r in enumerate(refs):
            mcr = rdRascalMCES.FindMCES(tgt, r, opts)
            if mcr:
                mcs_mol = mcr[0]
                atom_matches = mcs_mol.atomMatches()
                smarts = mcs_mol.smartsString
                
                # Store details if requested
                if return_details:
                    bond_matches = mcs_mol.bondMatches()
                    mcs_info = {
                        "atom_count": len(atom_matches),
                        "bond_count": len(bond_matches),
                        "similarity_score": opts.similarityThreshold,
                        "query_atoms": [match[0] for match in atom_matches],
                        "template_atoms": [match[1] for match in atom_matches],
                        "smarts": smarts
                    }
                    hits.append((len(atom_matches), i, smarts, mcs_info))
                else:
                    hits.append((len(atom_matches), i, smarts))
        
        if hits:
            # Get best match by size
            if return_details:
                best_size, idx, smarts, details = max(hits)
            else:
                best_size, idx, smarts = max(hits)
            
            # Quality control: accept small matches only at low thresholds
            if best_size >= min_acceptable_size or opts.similarityThreshold <= desperate_threshold:
                log.info(f"MCS found: size={best_size}, threshold={opts.similarityThreshold:.2f}")
                if return_details:
                    return idx, smarts, details
                return idx, smarts
            else:
                log.warning(f"Rejecting small MCS (size={best_size}) at threshold {opts.similarityThreshold:.2f}")
        
        # Reduce threshold and continue
        if opts.similarityThreshold > 0.0:
            log.warning(f"No MCS at threshold {opts.similarityThreshold:.2f}, reducing…")
            opts.similarityThreshold -= 0.1
        else:
            # Central atom fallback: use template with best CA RMSD
            log.warning("MCS search failed - using central atom fallback with best CA RMSD template")
            best_template_idx = find_best_ca_rmsd_template(refs)
            
            if return_details:
                central_details = {
                    "atom_count": 1,
                    "bond_count": 0,
                    "similarity_score": 0.0,
                    "query_atoms": [get_central_atom(tgt)],
                    "template_atoms": [get_central_atom(refs[best_template_idx])],
                    "smarts": "*",  # Single atom SMARTS
                    "central_atom_fallback": True
                }
                return best_template_idx, "*", central_details
            return best_template_idx, "*"


def find_best_ca_rmsd_template(refs: List[Chem.Mol]) -> int:
    """Find template with best (lowest) CA RMSD from the reference list.
    
    Args:
        refs: List of template molecules with CA RMSD properties
        
    Returns:
        Index of template with best CA RMSD, defaults to 0 if none found
    """
    best_idx = 0
    best_rmsd = float('inf')
    
    for i, tpl in enumerate(refs):
        if tpl.HasProp("ca_rmsd"):
            try:
                ca_rmsd = float(tpl.GetProp("ca_rmsd"))
                if ca_rmsd < best_rmsd:
                    best_rmsd = ca_rmsd
                    best_idx = i
            except (ValueError, TypeError):
                continue
    
    log.info(f"Selected template {best_idx} with CA RMSD {best_rmsd:.3f}Å for central atom fallback")
    return best_idx


# Helper function for mmff_minimise_fixed_parallel - MUST BE TOP LEVEL FOR PICKLING
def _mmff_minimize_single_conformer_task(args_tuple):
    """Helper for parallel MMFF minimization. Modifies a conformer of a molecule (passed as SMILES).
    Args:
        mol_smiles (str): SMILES string of the molecule.
        conformer_id (int): Original ID of the conformer (for tracking).
        initial_conformer_coords (list): List of (x,y,z) tuples for the conformer.
        fixed_atom_indices (list): List of atom indices to keep fixed.
        mmff_variant (str): MMFF variant (e.g., "MMFF94s").
        max_its (int): Max iterations for minimization.
    Returns:
        Tuple (conformer_id, list_of_minimized_coords) or None if failed.
    """
    mol_smiles, conformer_id, initial_conformer_coords, fixed_atom_indices, mmff_variant, max_its = args_tuple
    
    try:
        mol = Chem.MolFromSmiles(mol_smiles)
        if not mol:
            log.debug(f"_mmff_task: Could not create mol from SMILES for conf {conformer_id}")
            return conformer_id, None # Return ID for tracking, None for coords
        
        # Use robust hydrogen addition with coordinate preservation
        mol_no_h = Chem.RemoveHs(mol) if mol else None
        if mol_no_h:
            # Create temporary conformer for coordinate preservation
            temp_conf = Chem.Conformer(mol_no_h.GetNumAtoms())
            heavy_atom_idx = 0
            for i in range(len(initial_conformer_coords)):
                if heavy_atom_idx < mol_no_h.GetNumAtoms():
                    temp_conf.SetAtomPosition(heavy_atom_idx, Point3D(
                        initial_conformer_coords[i][0], 
                        initial_conformer_coords[i][1], 
                        initial_conformer_coords[i][2]
                    ))
                    heavy_atom_idx += 1
            mol_no_h.AddConformer(temp_conf, assignId=True)
            
            # Add hydrogens with coordinate preservation  
            mol = Chem.AddHs(mol_no_h, addCoords=True)
        else:
            mol = Chem.AddHs(mol, addCoords=True)
        
        # Verify coordinates were properly assigned
        if mol.GetNumConformers() == 0:
            log.debug(f"_mmff_task: No conformer after AddHs for conf {conformer_id}, creating new one")
            conf = Chem.Conformer(mol.GetNumAtoms())
            # Use provided coordinates for heavy atoms, generate for hydrogens
            for i in range(min(len(initial_conformer_coords), mol.GetNumAtoms())):
                conf.SetAtomPosition(i, Point3D(initial_conformer_coords[i][0], initial_conformer_coords[i][1], initial_conformer_coords[i][2]))
            mol.AddConformer(conf, assignId=True)
        
        # Additional safety check for coordinate consistency
        conf = mol.GetConformer(0)
        if conf.GetNumAtoms() != mol.GetNumAtoms():
            log.warning(f"_mmff_task: Atom count mismatch in conformer {conformer_id}, regenerating conformer")
            mol.RemoveAllConformers()
            new_conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(min(len(initial_conformer_coords), mol.GetNumAtoms())):
                new_conf.SetAtomPosition(i, Point3D(initial_conformer_coords[i][0], initial_conformer_coords[i][1], initial_conformer_coords[i][2]))
            mol.AddConformer(new_conf, assignId=True)

        mol.AddConformer(conf, assignId=True) # conf ID will be 0 here

        mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant=mmff_variant)
        if mp is None:
            log.debug(f"_mmff_task: Could not get MMFF params for conf {conformer_id}")
            return conformer_id, None
            
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, mp, confId=0) # Use the single conformer (ID 0)
        if ff is None:
            log.debug(f"_mmff_task: Could not get MMFF force field for conf {conformer_id}")
            return conformer_id, None
            
        for idx in fixed_atom_indices:
            if idx < mol.GetNumAtoms(): # Ensure fixed index is valid
                 ff.AddFixedPoint(idx)
            else:
                log.warning(f"_mmff_task: Fixed atom index {idx} out of bounds for {mol.GetNumAtoms()} atoms in conf {conformer_id}")
        
        ff.Minimize(maxIts=max_its)
        minimized_coords = mol.GetConformer(0).GetPositions().tolist()
        return conformer_id, minimized_coords
    except Exception as e:
        log.error(f"_mmff_task: MMFF Minimization uncaught exception for conf {conformer_id}: {e} {traceback.format_exc()}")
        return conformer_id, None

# ─── UFF-aware MMFF worker function for parallel processing ─────────────────

def _mmff_minimize_single_conformer_task_uff_aware(args_tuple):
    """UFF-aware helper for parallel MMFF minimization with automatic fallback.
    
    Args:
        mol_smiles (str): SMILES string of the molecule.
        conformer_id (int): Original ID of the conformer (for tracking).
        initial_conformer_coords (list): List of (x,y,z) tuples for the conformer.
        fixed_atom_indices (list): List of atom indices to keep fixed.
        mmff_variant (str): MMFF variant (e.g., "MMFF94s").
        max_its (int): Max iterations for minimization.
        
    Returns:
        Tuple (conformer_id, list_of_minimized_coords) or None if failed.
    """
    mol_smiles, conformer_id, initial_conformer_coords, fixed_atom_indices, mmff_variant, max_its = args_tuple
    
    try:
        mol = Chem.MolFromSmiles(mol_smiles)
        if not mol:
            log.debug(f"_mmff_uff_task: Could not create mol from SMILES for conf {conformer_id}")
            return conformer_id, None
        
        # Use robust hydrogen addition with coordinate preservation  
        mol_no_h = Chem.RemoveHs(mol) if mol else None
        if mol_no_h:
            # Create temporary conformer for coordinate preservation
            temp_conf = Chem.Conformer(mol_no_h.GetNumAtoms()) 
            heavy_atom_idx = 0
            for i in range(len(initial_conformer_coords)):
                if heavy_atom_idx < mol_no_h.GetNumAtoms():
                    temp_conf.SetAtomPosition(heavy_atom_idx, Point3D(
                        initial_conformer_coords[i][0], 
                        initial_conformer_coords[i][1], 
                        initial_conformer_coords[i][2]
                    ))
                    heavy_atom_idx += 1
            mol_no_h.AddConformer(temp_conf, assignId=True)
            
            # Add hydrogens with coordinate preservation
            mol = Chem.AddHs(mol_no_h, addCoords=True)
        else:
            mol = Chem.AddHs(mol, addCoords=True)
        
        # Verify coordinates were properly assigned
        if mol.GetNumConformers() == 0:
            log.debug(f"_mmff_uff_task: No conformer after AddHs for conf {conformer_id}, creating new one")
            conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(min(len(initial_conformer_coords), mol.GetNumAtoms())):
                conf.SetAtomPosition(i, Point3D(initial_conformer_coords[i][0], initial_conformer_coords[i][1], initial_conformer_coords[i][2]))
            mol.AddConformer(conf, assignId=True)
        
        # Additional safety check for coordinate consistency
        conf = mol.GetConformer(0)
        if conf.GetNumAtoms() != mol.GetNumAtoms():
            log.warning(f"_mmff_uff_task: Atom count mismatch in conformer {conformer_id}, regenerating conformer")
            mol.RemoveAllConformers() 
            new_conf = Chem.Conformer(mol.GetNumAtoms())
            for i in range(min(len(initial_conformer_coords), mol.GetNumAtoms())):
                new_conf.SetAtomPosition(i, Point3D(initial_conformer_coords[i][0], initial_conformer_coords[i][1], initial_conformer_coords[i][2]))
            mol.AddConformer(new_conf, assignId=True)

        mol.AddConformer(conf, assignId=True)
        
        # Check if UFF fallback is needed
        use_uff = needs_uff_fallback(mol)
        
        if use_uff:
            # Use UFF for organometallic molecules
            ff = rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=0)
            if ff is None:
                log.debug(f"_mmff_uff_task: Could not get UFF force field for conf {conformer_id}")
                return conformer_id, None
            
            for idx in fixed_atom_indices:
                if idx < mol.GetNumAtoms():
                    ff.AddFixedPoint(idx)
                    
            ff.Minimize(maxIts=max_its)
            
        else:
            # Use MMFF for regular molecules
            mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant=mmff_variant)
            if mp is None:
                log.debug(f"_mmff_uff_task: Could not get MMFF params for conf {conformer_id}")
                return conformer_id, None
                
            ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, mp, confId=0)
            if ff is None:
                log.debug(f"_mmff_uff_task: Could not get MMFF force field for conf {conformer_id}")
                return conformer_id, None
                
            for idx in fixed_atom_indices:
                if idx < mol.GetNumAtoms():
                     ff.AddFixedPoint(idx)
            
            ff.Minimize(maxIts=max_its)
        
        minimized_coords = mol.GetConformer(0).GetPositions().tolist()
        return conformer_id, minimized_coords
        
    except Exception as e:
        log.error(f"_mmff_uff_task: Minimization uncaught exception for conf {conformer_id}: {e} {traceback.format_exc()}")
        return conformer_id, None

# Helper for select_best - MUST BE TOP LEVEL FOR PICKLING
def _score_and_align_task(args_tuple):
    """ Wrapper for score_and_align for parallel execution. """
    # We expect args_tuple to be (single_conf_mol_smiles, tpl_mol_smiles, original_cid, use_colors_flag)
    # OR (single_conf_mol_with_one_conformer, tpl_mol, original_cid)
    # Let's assume the latter as RDKit Mol objects are generally picklable.
    
    single_conf_mol, tpl_mol, original_cid, no_realign_flag_for_pose_selection = args_tuple
    # score_and_align internally creates copies (Chem.Mol(conf)), so original moles aren't modified.
    current_scores, aligned_mol = score_and_align(single_conf_mol, tpl_mol)
    
    # Determine which pose to return based on no_realign_flag_for_pose_selection
    # The actual pose (Mol object) to be stored if this is the best.
    # This should be the Chem.Mol object of the single conformer, or the aligned_mol.
    pose_to_consider_if_best = single_conf_mol if no_realign_flag_for_pose_selection else aligned_mol
    
    # We need to return enough info to reconstruct the best[metric_name] tuple later.
    # Return the scores, and the *relevant* pose (either raw single_conf_mol or aligned_mol)
    return original_cid, current_scores, pose_to_consider_if_best


# restrained MMFF minimisation (anchors frozen)
# This is the original sequential version, kept for reference or single-core use.
def mmff_minimise_fixed_sequential(mol: Chem.Mol, conf_ids, fixed_idx, its: int = DEFAULT_MMFF_ITERATIONS) -> None:
    mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant=MMFF_VARIANT)
    if mp is None:
        return
    fixed_idx_list = list(fixed_idx)
    for cid in conf_ids:
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, mp, confId=cid)
        if ff is None:
            continue
        for idx_fixed in fixed_idx_list:
            ff.AddFixedPoint(idx_fixed)
        ff.Minimize(maxIts=its)

# New parallel version for mmff_minimise_fixed
def mmff_minimise_fixed_parallel(mol_original: Chem.Mol, conf_ids: List[int], fixed_idx: List[int], its: int = DEFAULT_MMFF_ITERATIONS, n_workers: int = 1):
    """Perform MMFF minimization, potentially in parallel using ProcessPoolExecutor."""
    if not conf_ids:
        return

    # Get SMILES from the original molecule WITH HYDROGENS for consistent atom indexing if AddHs is used in worker.
    # Ensure mol_original has Hs before getting SMILES, if the conformer coords are for a H-added molecule.
    mol_smiles_with_hs = Chem.MolToSmiles(mol_original)

    fixed_idx_list = list(fixed_idx)

    # Check if UFF fallback might be needed
    use_uff_aware = needs_uff_fallback(mol_original)

    if n_workers > 1 and len(conf_ids) > 1:
        log.debug(f"MMFF minimization: {len(conf_ids)} conformers, {n_workers} workers")
        tasks = []
        for cid in conf_ids:
            conformer = mol_original.GetConformer(cid)
            # GetNumAtoms() on mol_original (which has Hs) should match coords length
            if conformer.GetNumAtoms() != mol_original.GetNumAtoms():
                 log.warning(f"Atom count mismatch for conformer {cid}. Conformer: {conformer.GetNumAtoms()}, Mol: {mol_original.GetNumAtoms()}")
            initial_coords = conformer.GetPositions().tolist()
            tasks.append((mol_smiles_with_hs, cid, initial_coords, fixed_idx_list, "MMFF94s", its))

        minimized_conformers_data = {}
        # Use pebble for better process management if available
        if PEBBLE_AVAILABLE:
            with ProcessPool(max_workers=n_workers) as pool:
                futures = []
                for task in tasks:
                    if use_uff_aware:
                        # Use UFF-aware worker for potential organometallic molecules
                        future = pool.schedule(_mmff_minimize_single_conformer_task_uff_aware, args=[task])
                    else:
                        # Use standard MMFF worker for regular molecules
                        future = pool.schedule(_mmff_minimize_single_conformer_task, args=[task])
                    futures.append(future)
                
                for future in tqdm(futures, desc="MMFF Min (Parallel/Pebble)"):
                    try:
                        result = future.result()
                        if result:
                            cid_original, minimized_coords = result
                            if minimized_coords is not None:
                                minimized_conformers_data[cid_original] = minimized_coords
                            else:
                                log.warning(f"MMFF minimization failed or returned no coords for conformer ID {cid_original}.")
                    except Exception as e:
                        log.warning(f"MMFF minimization task failed: {str(e)}")
        else:
            # Fallback to ProcessPoolExecutor
            mp_context = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
                # Use appropriate worker function based on organometallic detection
                if use_uff_aware:
                    worker_func = _mmff_minimize_single_conformer_task_uff_aware
                else:
                    worker_func = _mmff_minimize_single_conformer_task
                
                results_iterator = executor.map(worker_func, tasks)
                
                for result in tqdm(results_iterator, total=len(tasks), desc="MMFF Min (Parallel)"):
                    if result:
                        cid_original, minimized_coords = result
                        if minimized_coords is not None:
                            minimized_conformers_data[cid_original] = minimized_coords
                        else:
                            log.warning(f"MMFF minimization failed or returned no coords for conformer ID {cid_original}.")
                    # else: task might have failed even before returning a tuple
        
        updated_count = 0
        for cid, new_coords_list in minimized_conformers_data.items():
            if cid < mol_original.GetNumConformers(): # Check if cid is valid
                conformer_to_update = mol_original.GetConformer(cid)
                if len(new_coords_list) == conformer_to_update.GetNumAtoms():
                    for i, coords_atom in enumerate(new_coords_list):
                        conformer_to_update.SetAtomPosition(i, Point3D(coords_atom[0], coords_atom[1], coords_atom[2]))
                    updated_count += 1
                else:
                    log.warning(f"Coordinate length mismatch for conformer {cid} after MMFF. Expected {conformer_to_update.GetNumAtoms()}, got {len(new_coords_list)}.")
            else:
                log.warning(f"Conformer ID {cid} out of range after parallel MMFF.")
        log.debug(f"Finished parallel MMFF minimization. Updated {updated_count}/{len(conf_ids)} conformers on original molecule.")

    else: # Sequential execution uses the original molecule directly
        log.debug(f"MMFF minimization: {len(conf_ids)} conformers (sequential)")
        
        # Use UFF for organometallic molecules, MMFF for regular molecules
        if use_uff_aware:
            log.debug("Using UFF for sequential minimization of organometallic molecule")
            minimize_with_uff(mol_original, conf_ids, fixed_idx_list, its)
        else:
            # Standard MMFF minimization
            mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol_original, mmffVariant="MMFF94s")
            if mp is None:
                log.warning("MMFF parameters missing for sequential minimization, skipping.")
                return

            for cid in tqdm(conf_ids, desc="MMFF Min (Sequential)"):
                if cid < mol_original.GetNumConformers():
                    ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol_original, mp, confId=cid)
                    if ff is None:
                        log.warning(f"Could not get MMFF force field for conformer {cid} (sequential), skipping.")
                        continue
                    for idx_fixed_seq in fixed_idx_list:
                        ff.AddFixedPoint(idx_fixed_seq)
                    try:
                        ff.Minimize(maxIts=its)
                    except Exception as e_seq:
                        log.error(f"MMFF Minimization failed for conformer {cid} (sequential): {e_seq}")
                else:
                     log.warning(f"Conformer ID {cid} out of range for sequential MMFF.")


# constrained embedding - using original optimization approach
def constrained_embed(tgt: Chem.Mol, ref: Chem.Mol, smarts: str, n_conformers: int = N_CONFS, n_workers_pipeline: int = 1) -> Chem.Mol:
    """Generate N_CONFS conformations of tgt, locking MCS atoms to ref coords."""
    
    # Handle central atom fallback case
    if smarts == "*":
        log.info("Using central atom positioning for pose generation")
        return central_atom_embed(tgt, ref, n_conformers, n_workers_pipeline)
    
    # Check if UFF fallback is needed for organometallic molecules
    use_uff = needs_uff_fallback(tgt)
    if use_uff:
        log.info("Detected organometallic atoms in target molecule, using UFF-compatible embedding")
    
    patt = Chem.MolFromSmarts(smarts)
    tgt_idxs = tgt.GetSubstructMatch(patt)
    ref_idxs = ref.GetSubstructMatch(patt)
    
    # Check for valid MCS match
    if not tgt_idxs or not ref_idxs or len(tgt_idxs) != len(ref_idxs) or len(tgt_idxs) < 3:
        log.warning(f"Invalid MCS match for constrained embedding. Target idx: {tgt_idxs}, Ref idx: {ref_idxs}. Proceeding with unconstrained embedding.")
        # Use robust hydrogen addition for fallback
        try:
            if tgt.GetNumConformers() > 0:
                tgt_no_h = Chem.RemoveHs(tgt)
                if tgt_no_h.GetNumConformers() == 0:
                    orig_conf = tgt.GetConformer(0)
                    new_conf = Chem.Conformer(tgt_no_h.GetNumAtoms())
                    heavy_idx = 0
                    for i in range(tgt.GetNumAtoms()):
                        if tgt.GetAtomWithIdx(i).GetAtomicNum() != 1:
                            if heavy_idx < tgt_no_h.GetNumAtoms():
                                new_conf.SetAtomPosition(heavy_idx, orig_conf.GetAtomPosition(i))
                                heavy_idx += 1
                    tgt_no_h.AddConformer(new_conf, assignId=True)
                probe = Chem.AddHs(tgt_no_h, addCoords=True)
            else:
                probe = Chem.AddHs(tgt, addCoords=True)
        except Exception as e:
            log.warning(f"Enhanced hydrogen addition failed in fallback: {e}")
            probe = Chem.AddHs(tgt, addCoords=True)
        # Use UFF-compatible embedding if organometallic
        if use_uff:
            ps_fallback = rdDistGeom.ETKDGv3()
            ps_fallback.numThreads = n_workers_pipeline if n_workers_pipeline > 0 else 0
            ps_fallback.maxIterations = 1000
            ps_fallback.useRandomCoords = True
            ps_fallback.enforceChirality = False  # Critical: match original working code
            conf_ids = embed_with_uff_fallback(probe, n_conformers, ps_fallback)
        else:
            # Use standard MMFF-compatible embedding
            ps_fallback = rdDistGeom.ETKDGv3()
            ps_fallback.numThreads = n_workers_pipeline if n_workers_pipeline > 0 else 0
            ps_fallback.maxIterations = 1000
            ps_fallback.useRandomCoords = True
            ps_fallback.enforceChirality = False  # Critical: match original working code
            rdDistGeom.EmbedMultipleConfs(probe, n_conformers, ps_fallback)
        return probe
    
    # Create coordinate map for constrained embedding
    pairs = list(zip(tgt_idxs, ref_idxs))
    coord_map = {t: ref.GetConformer().GetAtomPosition(r) for t, r in pairs}
    ps = rdDistGeom.ETKDGv3()
    # Use n_workers_pipeline for ETKDG if > 0, else RDKit default (0 means all)
    ps.numThreads = n_workers_pipeline if n_workers_pipeline > 0 else 0
    ps.maxIterations = 1000  # Increase attempts for difficult molecules
    ps.useRandomCoords = True  # Enable random coordinate initialization
    ps.enforceChirality = False  # Critical: prevents getting stuck with geometric conflicts
    
    # Generate conformations with UFF fallback if needed - use robust hydrogen addition
    try:
        # Preserve original coordinates if target has conformers
        if tgt.GetNumConformers() > 0:
            tgt_no_h = Chem.RemoveHs(tgt)
            if tgt_no_h.GetNumConformers() == 0:
                # Transfer conformer from original
                orig_conf = tgt.GetConformer(0)
                new_conf = Chem.Conformer(tgt_no_h.GetNumAtoms())
                heavy_idx = 0
                for i in range(tgt.GetNumAtoms()):
                    if tgt.GetAtomWithIdx(i).GetAtomicNum() != 1:  # Heavy atom
                        if heavy_idx < tgt_no_h.GetNumAtoms():
                            new_conf.SetAtomPosition(heavy_idx, orig_conf.GetAtomPosition(i))
                            heavy_idx += 1
                tgt_no_h.AddConformer(new_conf, assignId=True)
            probe = Chem.AddHs(tgt_no_h, addCoords=True)
        else:
            probe = Chem.AddHs(tgt, addCoords=True)
    except Exception as e:
        log.warning(f"Enhanced hydrogen addition failed: {e}, using fallback")
        probe = Chem.AddHs(tgt, addCoords=True)
    
    # Progressive constraint reduction like original working code
    lrm = 0  # "last removed match" - start with full constraints
    conf_ids = []
    max_retries = min(len(pairs), 5)  # Limit retries to avoid infinite loop
    
    while not conf_ids and lrm < max_retries:
        # Reduce constraints by removing some atom pairs
        reduced_pairs = pairs[lrm:] if lrm < len(pairs) else []
        if not reduced_pairs:
            break
            
        # Update coordinate map with reduced constraints
        reduced_coord_map = {t: ref.GetConformer().GetAtomPosition(r) for t, r in reduced_pairs}
        
        log.debug(f"Trying constrained embedding with {len(reduced_pairs)} constraints (removed {lrm})")
        
        if use_uff:
            # Use UFF-compatible embedding for organometallic molecules
            conf_ids = embed_with_uff_fallback(probe, n_conformers, ps, reduced_coord_map)
        else:
            # Use standard embedding for regular molecules (use individual parameters instead of EmbedParameters)
            conf_ids = rdDistGeom.EmbedMultipleConfs(
                probe, n_conformers, 
                maxAttempts=ps.maxIterations if hasattr(ps, 'maxIterations') else 1000,
                randomSeed=-1,
                useRandomCoords=ps.useRandomCoords if hasattr(ps, 'useRandomCoords') else True,
                enforceChirality=ps.enforceChirality if hasattr(ps, 'enforceChirality') else False,
                numThreads=ps.numThreads if hasattr(ps, 'numThreads') else n_workers_pipeline,
                coordMap=reduced_coord_map
            )
        
        lrm += 1  # Remove one more constraint next time
    
    if conf_ids:
        # Use appropriate minimization based on molecule type
        if use_uff:
            # Use UFF minimization for organometallic molecules
            minimize_with_uff(probe, conf_ids, tgt_idxs, n_workers_pipeline)
        else:
            # Use MMFF minimization for regular molecules
            mmff_minimise_fixed_parallel(probe, conf_ids, tgt_idxs, n_workers=n_workers_pipeline)
        
        # Align each conformer using the final working constraint set
        final_pairs = pairs[lrm-1:] if lrm > 0 else pairs
        for cid in conf_ids:
            if final_pairs:  # Only align if we have valid pairs
                rdMolAlign.AlignMol(probe, ref, atomMap=final_pairs, prbCid=cid)
        return probe
    
    # Fallback if embedding fails
    log.warning("Constrained embedding failed, falling back to unconstrained embedding.")
    # Use robust hydrogen addition for final fallback
    try:
        if tgt.GetNumConformers() > 0:
            tgt_no_h = Chem.RemoveHs(tgt)
            if tgt_no_h.GetNumConformers() == 0:
                orig_conf = tgt.GetConformer(0)
                new_conf = Chem.Conformer(tgt_no_h.GetNumAtoms())
                heavy_idx = 0
                for i in range(tgt.GetNumAtoms()):
                    if tgt.GetAtomWithIdx(i).GetAtomicNum() != 1:
                        if heavy_idx < tgt_no_h.GetNumAtoms():
                            new_conf.SetAtomPosition(heavy_idx, orig_conf.GetAtomPosition(i))
                            heavy_idx += 1
                tgt_no_h.AddConformer(new_conf, assignId=True)
            probe = Chem.AddHs(tgt_no_h, addCoords=True)
        else:
            probe = Chem.AddHs(tgt, addCoords=True)
    except Exception as e:
        log.warning(f"Enhanced hydrogen addition failed in final fallback: {e}")
        probe = Chem.AddHs(tgt, addCoords=True)
    
    if use_uff:
        # UFF fallback for organometallic
        ps_fallback_final = rdDistGeom.ETKDGv3()
        ps_fallback_final.numThreads = n_workers_pipeline if n_workers_pipeline > 0 else 0
        ps_fallback_final.maxIterations = 1000
        ps_fallback_final.useRandomCoords = True
        ps_fallback_final.enforceChirality = False  # Critical: match original working code
        embed_with_uff_fallback(probe, n_conformers, ps_fallback_final)
    else:
        # Standard fallback for regular molecules
        ps_fallback_final = rdDistGeom.ETKDGv3()
        ps_fallback_final.numThreads = n_workers_pipeline if n_workers_pipeline > 0 else 0
        ps_fallback_final.maxIterations = 1000
        ps_fallback_final.useRandomCoords = True
        ps_fallback_final.enforceChirality = False  # Critical: match original working code
        rdDistGeom.EmbedMultipleConfs(probe, n_conformers, ps_fallback_final)
    
    return probe


def central_atom_embed(tgt: Chem.Mol, ref: Chem.Mol, n_conformers: int, n_workers_pipeline: int) -> Chem.Mol:
    """Generate conformers using central atom positioning when MCS fails.
    
    Places target molecule's central atom at template's central atom position.
    
    Args:
        tgt: Target molecule
        ref: Reference template molecule
        n_conformers: Number of conformers to generate
        n_workers_pipeline: Number of workers for parallel processing
        
    Returns:
        Molecule with generated conformers positioned using central atom alignment
    """
    # Check if UFF fallback is needed for organometallic molecules
    use_uff = needs_uff_fallback(tgt)
    if use_uff:
        log.info("Using UFF-compatible embedding for organometallic target in central atom positioning")
    
    # Get central atoms
    tgt_central = get_central_atom(tgt)
    ref_central = get_central_atom(ref)
    ref_central_pos = ref.GetConformer().GetAtomPosition(ref_central)
    
    log.info(f"Central atom positioning: target atom {tgt_central} -> template atom {ref_central}")
    
    # Generate unconstrained conformers first - use robust hydrogen addition
    try:
        if tgt.GetNumConformers() > 0:
            tgt_no_h = Chem.RemoveHs(tgt)
            if tgt_no_h.GetNumConformers() == 0:
                orig_conf = tgt.GetConformer(0)
                new_conf = Chem.Conformer(tgt_no_h.GetNumAtoms())
                heavy_idx = 0
                for i in range(tgt.GetNumAtoms()):
                    if tgt.GetAtomWithIdx(i).GetAtomicNum() != 1:
                        if heavy_idx < tgt_no_h.GetNumAtoms():
                            new_conf.SetAtomPosition(heavy_idx, orig_conf.GetAtomPosition(i))
                            heavy_idx += 1
                tgt_no_h.AddConformer(new_conf, assignId=True)
            probe = Chem.AddHs(tgt_no_h, addCoords=True)
        else:
            probe = Chem.AddHs(tgt, addCoords=True)
    except Exception as e:
        log.warning(f"Enhanced hydrogen addition failed in central_atom_embed: {e}")
        probe = Chem.AddHs(tgt, addCoords=True)
    ps = rdDistGeom.ETKDGv3()
    ps.numThreads = n_workers_pipeline if n_workers_pipeline > 0 else 0
    
    if use_uff:
        # Use UFF-compatible embedding for organometallic molecules
        conf_ids = embed_with_uff_fallback(probe, n_conformers, ps, coordMap=coord_map)
    else:
        # Use standard embedding for regular molecules
        conf_ids = rdDistGeom.EmbedMultipleConfs(probe, n_conformers, ps, coordMap=coord_map)
    
    if not conf_ids:
        log.warning("Failed to generate conformers for central atom positioning")
        return probe
    
    # Find central atom in the H-added molecule
    # Map from original molecule to H-added molecule
    atom_map = {}
    for i, atom in enumerate(tgt.GetAtoms()):
        atom_map[i] = i  # Assuming H's are added at the end
    
    probe_central = atom_map.get(tgt_central, tgt_central)
    
    # Translate each conformer to align central atoms
    for cid in conf_ids:
        conf = probe.GetConformer(cid)
        current_central_pos = conf.GetAtomPosition(probe_central)
        
        # Calculate translation vector
        translation = ref_central_pos - current_central_pos
        
        # Apply translation to all atoms
        for i in range(conf.GetNumAtoms()):
            old_pos = conf.GetAtomPosition(i)
            new_pos = old_pos + translation
            conf.SetAtomPosition(i, new_pos)
    
    log.info(f"Generated {len(conf_ids)} conformers using central atom alignment")
    return probe


def select_best(
    confs: Chem.Mol,
    tpl: Chem.Mol,
    no_realign: bool = False,
    n_workers: int = 1,
    return_all_ranked: bool = False
) -> Union[Dict[str, Tuple[Chem.Mol, Dict[str, float]]], List[Tuple[Chem.Mol, Dict[str, float], int]]]:
    """Enhanced select_best - backward compatible with optional all poses return.
    
    Args:
        confs: Molecule with multiple conformers
        tpl: Template molecule for alignment
        no_realign: Use raw conformer before shape‐align
        n_workers: Number of workers for parallel processing
        return_all_ranked: If True, return all poses ranked by combo score
        
    Returns:
        If return_all_ranked=False: Dict mapping metric to (best_pose, scores)
        If return_all_ranked=True: List of (pose, scores, original_conformer_id) tuples sorted by combo score
    """
    best = {m: (None, {"shape": -1.0, "color": -1.0, "combo": -1.0}) for m in ("shape", "color", "combo")}
    all_poses_with_scores = []  # Only populated if return_all_ranked=True
    
    total_num_confs = confs.GetNumConformers()

    if total_num_confs == 0:
        log.warning("No conformers provided to select_best function.")
        return [] if return_all_ranked else best

    all_pose_results = [] # List to store (original_cid, current_scores, pose_to_consider_if_best)

    if n_workers > 1 and total_num_confs > 1:
        log.debug(f"Pose scoring: {total_num_confs} conformers, {n_workers} workers")
        tasks = []
        for cid in range(total_num_confs):
            single_conf_mol = Chem.Mol(confs) 
            single_conf_mol.RemoveAllConformers()
            conformer = confs.GetConformer(cid)
            single_conf_mol.AddConformer(conformer, assignId=True) 
            # Pass no_realign flag to the task for determining pose_to_consider_if_best inside worker
            tasks.append((single_conf_mol, Chem.Mol(tpl), cid, no_realign))

        # Use pebble for better process management if available
        if PEBBLE_AVAILABLE:
            with ProcessPool(max_workers=n_workers) as pool:
                futures = []
                for task in tasks:
                    future = pool.schedule(_score_and_align_task, args=[task])
                    futures.append(future)
                
                for future in tqdm(futures, desc="Score/Align (Parallel/Pebble)"):
                    try:
                        result = future.result()
                        if result:
                            all_pose_results.append(result) # result is (original_cid, current_scores, pose_to_consider_if_best)
                    except Exception as e:
                        log.warning(f"Score/align task failed: {str(e)}")
        else:
            # Fallback to ProcessPoolExecutor
            mp_context = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
                # _score_and_align_task is the top-level helper function
                results_iterator = executor.map(_score_and_align_task, tasks)
                for result in tqdm(results_iterator, total=len(tasks), desc="Score/Align (Parallel)"):
                    if result:
                        all_pose_results.append(result) # result is (original_cid, current_scores, pose_to_consider_if_best)
                    # else: task might have failed
            
    else: # Sequential execution
        log.debug(f"Pose scoring: {total_num_confs} conformers (sequential)")
        for cid_seq in tqdm(range(total_num_confs), desc="Score/Align (Sequential)"):
            raw_conf_mol_seq = Chem.Mol(confs) 
            raw_conf_mol_seq.RemoveAllConformers()
            conformer_seq = confs.GetConformer(cid_seq)
            raw_conf_mol_seq.AddConformer(conformer_seq, assignId=True)
            
            current_scores_seq, aligned_mol_seq = score_and_align(raw_conf_mol_seq, tpl)
            pose_to_consider_seq = raw_conf_mol_seq if no_realign else aligned_mol_seq
            all_pose_results.append((cid_seq, current_scores_seq, pose_to_consider_seq))
            
    # Process all results to find the best for each metric and collect all poses if requested
    for original_cid, current_scores_item, pose_to_consider_item in all_pose_results:
        # Store for all poses if requested
        if return_all_ranked:
            all_poses_with_scores.append((pose_to_consider_item, current_scores_item, original_cid))
        
        # Find best for each metric (existing logic)
        for metric_name, current_metric_score_val in current_scores_item.items():
            if current_metric_score_val > best[metric_name][1][metric_name]:
                best[metric_name] = (pose_to_consider_item, current_scores_item)
    
    if return_all_ranked:
        # Sort by combo score (descending)
        all_poses_with_scores.sort(key=lambda x: x[1]["combo"], reverse=True)
        return all_poses_with_scores
    
    return best


def pdb_path(pid: str) -> Optional[str]:
    """Find protein PDB file in either refined-set or other-PL directories.
    Returns path to PDB file or None if not found."""
    r = f"{DATA_DIR}/PDBBind/PDBbind_v2020_refined/refined-set/{pid}/{pid}_protein.pdb"
    g = f"{DATA_DIR}/PDBBind/PDBbind_v2020_other_PL/v2020-other-PL/{pid}/{pid}_protein.pdb"
    
    if os.path.exists(r):
        return r
    elif os.path.exists(g):
        return g
    
    log.warning(f"PDB file for {pid} not found in either refined or other-PL directories")
    return None


def transform_ligand(mob_pdb: str, lig: Chem.Mol, pid: str, ref_struct: AtomArray, 
                ref_crystal_ligand: Optional[Chem.Mol] = None, 
                ref_chains: Optional[List[str]] = None,
                mob_chains: Optional[List[str]] = None,
                similarity_score: float = 0.0) -> Optional[Chem.Mol]:
    """Superimpose ligand onto protein template using biotite's superimpose_homologs.
    
    This function performs a sequence-based alignment of protein structures and transforms
    the ligand coordinates accordingly:
    
    1. Loads and filters protein structures to amino acids only
    2. Prioritizes chains from embeddings that represent binding pockets
    3. Extracts CA atoms from selected chains for alignment 
    4. Handles multi-chain proteins with annotation compatibility checking
    5. Performs sequence-based homology superimposition with fallbacks
    6. Applies the transformation matrix to ligand coordinates
    7. Stores alignment metadata as molecule properties
    
    Detailed Process:
    
    Chain Selection and CA Extraction:
    - Chains from embedding data are prioritized as they likely represent binding pockets
    - CA atoms are extracted individually from each selected chain
    - If embedding-specified chains aren't found, falls back to first available chain
    - CA atoms from all chains are collected and combined (via stacking)
    
    Multi-Chain Handling:
    - For proteins with multiple chains, attempts to stack all CA atoms
    - If stacking fails due to annotation incompatibility, standardizes annotations
    - If standardization fails, falls back to using only the first chain
    - Chains are handled individually but transformed as a unified structure
    
    Sequence-Based Superimposition:
    - Uses BIOTITE's superimpose_homologs with precise parameters:
      * BLOSUM62 substitution matrix for sequence similarity
      * Gap penalty of -10
      * Minimum of 15 anchor residues required (min_anchors=15)
      * Terminal gap penalties enabled
    - Returns matching residue pairs via fixed_idx and mob_idx arrays
    
    RMSD Calculation Specifics:
    - RMSD is calculated ONLY on the matched subset of CA atoms, not entire structure
    - This subset approach explains why CA RMSD values can be surprisingly low (0.3-0.5Å)
    - The number of residues used in RMSD calculation is logged (typically 15+ residues)
    - This approach allows accurate comparison even when structures differ globally
    - CA RMSD measures backbone alignment, NOT ligand pose accuracy
    
    Transformation Application:
    - The transformation matrix from alignment is applied directly to ligand coordinates
    - All alignment metadata is stored as properties on the returned molecule
    - CA RMSD values are stored and used for filtering with CA_RMSD_THRESHOLD
    
    The function implements multiple fallback mechanisms to handle problematic structures:
    - If specific chains aren't found, uses the first available chain
    - If multi-chain stacking fails, uses annotation standardization
    - If standardization fails, falls back to single chain
    - If homology superimposition fails, uses simple superimposition
    
    Args:
        mob_pdb: Path to mobile protein PDB file
        lig: Mobile ligand molecule
        pid: PDB ID of mobile protein
        ref_struct: Reference protein structure
        ref_crystal_ligand: Reference crystal ligand (not used in simplified approach)
        ref_chains: Chains to use from reference protein (should match embedding chains)
        mob_chains: Chains to use from mobile protein (should match embedding chains)
        similarity_score: Embedding similarity score for reference
        
    Returns:
        Transformed ligand molecule with alignment metrics as properties,
        or None if alignment fails
    
    Notes:
        There is often a disconnect between CA RMSD (protein backbone alignment)
        and final ligand RMSD (pose accuracy) because good backbone alignment
        doesn't guarantee correct ligand placement. CA RMSD is primarily used
        as a filtering step for reasonable template selection.
    """
    # Collect initial alignment context for tracking
    alignment_context = collect_protein_alignment_context(pid, mob_pdb)
    
    try:
        # Load mobile structure
        mob = bsio.load_structure(mob_pdb)
        
        # Track successful structure loading
        ProteinAlignmentTracker.track_alignment_attempt(
            pid, "structure_loading", True, 
            {"total_atoms": len(mob), "file_size": alignment_context["protein_file_status"]["file_size"]}
        )
        
        # Filter to amino acids
        ref_prot = ref_struct[filter_amino_acids(ref_struct)]
        mob_prot = mob[filter_amino_acids(mob)]
        
        if len(ref_prot) < 5 or len(mob_prot) < 5:
            ProteinAlignmentTracker.track_alignment_attempt(
                pid, "amino_acid_filtering", False,
                {"failure_reason": f"Too few amino acids: ref={len(ref_prot)}, mob={len(mob_prot)}"}
            )
            log.warning(f"Too few amino acids in {pid}")
            return None
        
        # Track successful amino acid filtering
        ProteinAlignmentTracker.track_alignment_attempt(
            pid, "amino_acid_filtering", True,
            {"ref_amino_acids": len(ref_prot), "mob_amino_acids": len(mob_prot)}
        )
        
        # Get available chains
        ref_available_chains = list(get_chains(ref_prot))
        mob_available_chains = list(get_chains(mob_prot))
        
        if not ref_available_chains or not mob_available_chains:
            ProteinAlignmentTracker.track_alignment_attempt(
                pid, "chain_detection", False,
                {"failure_reason": f"No chains found: ref={len(ref_available_chains)}, mob={len(mob_available_chains)}"}
            )
            log.warning(f"No chains found in reference or mobile protein {pid}")
            return None
        
        # Track successful chain detection
        ProteinAlignmentTracker.track_alignment_attempt(
            pid, "chain_detection", True,
            {"ref_chains": len(ref_available_chains), "mob_chains": len(mob_available_chains)}
        )
        
        # PRIORITY: Use specific chains from embedding since they represent binding pockets
        selected_ref_chains = []
        if ref_chains:
            for chain in ref_chains:
                if chain in ref_available_chains:
                    selected_ref_chains.append(chain)
            
        if not selected_ref_chains:
            # Fallback to first available chain
            selected_ref_chains = [ref_available_chains[0]]
            log.warning(f"Using fallback chain {selected_ref_chains[0]} for reference structure - may affect binding pocket alignment")
            
        selected_mob_chains = []
        if mob_chains:
            # Log that we're using chains from embeddings (as DEBUG to respect log level)
            log.debug(f"Using embedding-specified chains for {pid}: {mob_chains}")
            for chain in mob_chains:
                if chain in mob_available_chains:
                    selected_mob_chains.append(chain)
                    
        if not selected_mob_chains:
            # Fallback to first available chain
            selected_mob_chains = [mob_available_chains[0]]
            log.warning(f"Using fallback chain {selected_mob_chains[0]} for mobile structure - may affect binding pocket alignment")
        
        # Extract CA atoms from selected chains - these represent backbone for alignment
        ref_ca_atoms = []
        for chain in selected_ref_chains:
            chain_ca = ref_prot[(ref_prot.chain_id == chain) & (ref_prot.atom_name == "CA")]
            if len(chain_ca) > 0:
                ref_ca_atoms.append(chain_ca)
                log.debug(f"Using {len(chain_ca)} CA atoms from reference chain {chain}")
        
        mob_ca_atoms = []
        for chain in selected_mob_chains:
            chain_ca = mob_prot[(mob_prot.chain_id == chain) & (mob_prot.atom_name == "CA")]
            if len(chain_ca) > 0:
                mob_ca_atoms.append(chain_ca)
                log.debug(f"Using {len(chain_ca)} CA atoms from mobile chain {chain}")
        
        # Combine CA atoms from all selected chains
        if not ref_ca_atoms or not mob_ca_atoms:
            ProteinAlignmentTracker.track_alignment_attempt(
                pid, "ca_extraction", False,
                {"failure_reason": f"No CA atoms: ref_chains={len(ref_ca_atoms)}, mob_chains={len(mob_ca_atoms)}"}
            )
            log.warning(f"No CA atoms found in selected chains for {pid}")
            return None
            
        # Handle single vs multiple chains
        if len(ref_ca_atoms) == 1:
            ref_ca = ref_ca_atoms[0]
        else:
            try:
                # Try to stack directly first
                ref_ca = struc.stack(ref_ca_atoms)
            except Exception as e:
                log.debug(f"Standard stacking failed for ref chains in {pid}, standardizing annotations: {str(e)}")
                try:
                    # If fails, standardize annotations and try again
                    # This fixes incompatible annotation dictionaries across chains
                    # which is a common issue with multi-chain proteins
                    std_ref_ca_atoms = standardize_atom_arrays(ref_ca_atoms)
                    if len(std_ref_ca_atoms) > 1:
                        ref_ca = struc.stack(std_ref_ca_atoms)
                    else:
                        ref_ca = std_ref_ca_atoms[0]
                        selected_ref_chains = [selected_ref_chains[0]]
                        log.warning(f"Using only first chain for reference in {pid} after standardization")
                except Exception as e2:
                    # Last resort: just use the first chain
                    log.warning(f"Standardization failed for reference in {pid}, using only first chain: {str(e2)}")
                    ref_ca = ref_ca_atoms[0]
                    # Update selected chains to match
                    selected_ref_chains = [selected_ref_chains[0]]
        
        if len(mob_ca_atoms) == 1:
            mob_ca = mob_ca_atoms[0]
        else:
            try:
                # Try to stack directly first
                mob_ca = struc.stack(mob_ca_atoms)
            except Exception as e:
                log.debug(f"Standard stacking failed for mobile chains in {pid}, standardizing annotations: {str(e)}")
                try:
                    # If fails, standardize annotations and try again
                    # Critical for optimal binding pocket alignment when multiple chains are involved
                    # Ensures we maintain the chains specified in the embedding file
                    std_mob_ca_atoms = standardize_atom_arrays(mob_ca_atoms)
                    if len(std_mob_ca_atoms) > 1:
                        mob_ca = struc.stack(std_mob_ca_atoms)
                    else:
                        mob_ca = std_mob_ca_atoms[0]
                        selected_mob_chains = [selected_mob_chains[0]]
                        log.warning(f"Using only first chain for mobile in {pid} after standardization")
                except Exception as e2:
                    # Last resort: just use the first chain
                    log.warning(f"Standardization failed for {pid}, using only first chain: {str(e2)}")
                    mob_ca = mob_ca_atoms[0]
                    # Update selected chains to match
                    selected_mob_chains = [selected_mob_chains[0]]
        
        # Track successful CA extraction with detailed metrics
        ProteinAlignmentTracker.track_alignment_attempt(
            pid, "ca_extraction", True,
            {
                "total_ref_residues": len(ref_ca),
                "total_mob_residues": len(mob_ca),
                "ref_chains_used": selected_ref_chains,
                "mob_chains_used": selected_mob_chains
            }
        )
        
        # Ensure we have enough atoms for alignment
        if min(len(ref_ca), len(mob_ca)) < 3:
            ProteinAlignmentTracker.track_alignment_attempt(
                pid, "ca_validation", False,
                {"failure_reason": f"Too few CA atoms for alignment: ref={len(ref_ca)}, mob={len(mob_ca)}"}
            )
            log.warning(f"Too few CA atoms for {pid}: ref={len(ref_ca)}, mob={len(mob_ca)}")
            return None

        # Use superimpose_homologs as mentioned in the project brief - we call this first to get matching CA pairs
        ca_rmsd_value = None
        try:
            # Apply homolog-based superimposition with reasonable min_anchors to get matched CA pairs
            fitted, transform, fixed_idx, mob_idx = superimpose_homologs(
                ref_ca, mob_ca,
                substitution_matrix="BLOSUM62",
                gap_penalty=-10,
                min_anchors=15,  # Minimum required for valid 3D transformation
                terminal_penalty=True
            )
            
            # Basic quality check - we need at least some aligned residues
            if len(fixed_idx) < 3 or len(mob_idx) < 3:
                ProteinAlignmentTracker.track_alignment_attempt(
                    pid, "superimpose_homologs", False,
                    {"failure_reason": f"Too few aligned residues: {len(fixed_idx)}"}
                )
                log.warning(f"Too few aligned residues for {pid}")
                return None
                
            # Log alignment details
            log.debug(f"Aligned {len(fixed_idx)} residues for {pid} using embedding-specified chains")
            
            # Calculate RMSD using only the matched CA pairs from superimpose_homologs
            # This works even when the original CA arrays have different lengths
            try:
                # Extract matched CA atoms from both structures using fixed_idx and mob_idx
                ref_subset = ref_ca[fixed_idx]
                mob_subset = mob_ca[mob_idx]
                
                # Now superimpose and calculate RMSD using these matched subsets
                fitted_mob_ca, _ = struc.superimpose(ref_subset, mob_subset)
                calculated_ca_rmsd = struc.rmsd(ref_subset, fitted_mob_ca)
                log.info(f"Calculated C-alpha RMSD for {pid} with reference using {len(fixed_idx)} matched residues: {calculated_ca_rmsd:.2f} Å")
                log.debug(f"Alignment details for {pid}: Reference length={len(ref_ca)}, Mobile length={len(mob_ca)}, Matched residues={len(fixed_idx)} ({(len(fixed_idx)/len(ref_ca))*100:.1f}% of ref, {(len(fixed_idx)/len(mob_ca))*100:.1f}% of mob)")
                
                # Store RMSD value for later progressive filtering (don't filter here)
                ca_rmsd_value = calculated_ca_rmsd
                
                # Track successful superimposition with detailed metrics
                ProteinAlignmentTracker.track_alignment_attempt(
                    pid, "superimpose_homologs", True,
                    {
                        "ca_rmsd": ca_rmsd_value,
                        "aligned_residues": len(fixed_idx),
                        "total_ref_residues": len(ref_ca),
                        "total_mob_residues": len(mob_ca),
                        "alignment_coverage_ref": (len(fixed_idx)/len(ref_ca))*100,
                        "alignment_coverage_mob": (len(fixed_idx)/len(mob_ca))*100
                    }
                )
                
            except Exception as e:
                ProteinAlignmentTracker.track_alignment_attempt(
                    pid, "ca_rmsd_calculation", False,
                    {"failure_reason": f"CA RMSD calculation failed: {str(e)}"}
                )
                log.warning(f"C-alpha RMSD calculation failed for {pid} using matched residues: {e}. Proceeding without RMSD filter.")
                
        except Exception as e:
            # If homolog superimposition fails, try basic superimposition
            import traceback # Ensure traceback is imported here
            log.warning(f"Homolog superimposition failed for {pid}: {str(e)}")
            log.warning("Traceback (homolog superimposition):")
            try:
                # Python 3.10+ compatible traceback formatting
                log.warning(''.join(traceback.format_exception(e)))
            except TypeError:
                # Fallback for older Python versions
                log.warning(''.join(traceback.format_exception(type(e), e, e.__traceback__)))
            
            # Track homolog superimposition failure
            ProteinAlignmentTracker.track_alignment_attempt(
                pid, "superimpose_homologs", False,
                {"failure_reason": str(e), "fallback_used": True}
            )
            
            # Try direct superimposition as fallback
            try:
                # Use minimum length to avoid index errors
                min_length = min(len(ref_ca), len(mob_ca))
                fitted, transform = superimpose(ref_ca[:min_length], mob_ca[:min_length])
                log.debug(f"Using fallback direct superimposition for {pid}")
                
                # Track successful fallback superimposition
                ProteinAlignmentTracker.track_alignment_attempt(
                    pid, "superimpose_direct_fallback", True,
                    {"residues_used": min_length, "fallback_used": True}
                )
                
            except Exception as e2:
                import traceback # Ensure traceback is imported here as well for safety, though already in scope if first block ran
                log.warning(f"Direct superimposition also failed for {pid}: {str(e2)}")
                log.warning("Traceback (direct superimposition fallback):")
                try:
                    # Python 3.10+ compatible traceback formatting
                    log.warning(''.join(traceback.format_exception(e2)))
                except TypeError:
                    # Fallback for older Python versions
                    log.warning(''.join(traceback.format_exception(type(e2), e2, e2.__traceback__)))
                
                # Track complete superimposition failure
                ProteinAlignmentTracker.track_alignment_attempt(
                    pid, "superimpose_direct_fallback", False,
                    {"failure_reason": str(e2)}
                )
                return None

        # Apply transformation to ligand - using the AffineTransformation directly
        moved = Chem.Mol(lig)
        moved.SetProp("_Name", f"{pid}_template")
        coords = np.asarray(lig.GetConformer().GetPositions(), float)
        
        # Apply the transformation
        transformed_coords = transform.apply(coords)
        
        # Apply coordinates to molecule
        conf = moved.GetConformer()
        for i, (x, y, z) in enumerate(transformed_coords):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
        
        # Store alignment metadata as properties
        moved.SetProp("template_pdb", pid)
        moved.SetProp("ref_chains", ",".join(selected_ref_chains))
        moved.SetProp("mob_chains", ",".join(selected_mob_chains))
        
        # Store similarity score if provided
        if similarity_score > 0:
            moved.SetProp("embedding_similarity", f"{similarity_score:.3f}")
            
        # Store the CA RMSD value if we calculated it
        if ca_rmsd_value is not None:
            moved.SetProp("ca_rmsd", f"{ca_rmsd_value:.3f}")
            
        # Store additional alignment details if available
        if 'fixed_idx' in locals() and 'ref_ca' in locals() and 'mob_ca' in locals():
            moved.SetProp("aligned_residues_count", str(len(fixed_idx)))
            moved.SetProp("total_ref_residues", str(len(ref_ca)))
            moved.SetProp("total_mob_residues", str(len(mob_ca)))
            moved.SetProp("aligned_percentage", f"{(len(fixed_idx)/min(len(ref_ca), len(mob_ca)))*100:.1f}%")

        # Track successful transformation
        ProteinAlignmentTracker.track_alignment_attempt(
            pid, "transformation_application", True,
            {
                "ca_rmsd": ca_rmsd_value,
                "has_ca_rmsd": ca_rmsd_value is not None,
                "similarity_score": similarity_score
            }
        )

        return moved
    except Exception as e:
        # Track overall failure
        ProteinAlignmentTracker.track_alignment_attempt(
            pid, "overall_transform_ligand", False,
            {"failure_reason": str(e)}
        )
        # Report errors
        log.error(f"Error transforming ligand for {pid}: {str(e)}")
        return None


def load_reference_protein(target_pdb: str) -> AtomArray:
    """Load the reference protein structure for alignment."""
    # Check refined-set structure first
    ref_path = f"{DATA_DIR}/PDBBind/PDBbind_v2020_refined/refined-set/{target_pdb}/{target_pdb}_protein.pdb"
    if not os.path.exists(ref_path):
        # Try general set
        ref_path = f"{DATA_DIR}/PDBBind/PDBbind_v2020_other_PL/v2020-other-PL/{target_pdb}/{target_pdb}_protein.pdb"
    
    if not os.path.exists(ref_path):
        log.error(f"No reference protein structure found for {target_pdb}")
        log.error(f"Refined: {DATA_DIR}/PDBBind/PDBbind_v2020_refined/refined-set/{target_pdb}/{target_pdb}_protein.pdb")
        log.error(f"Other: {DATA_DIR}/PDBBind/PDBbind_v2020_other_PL/v2020-other-PL/{target_pdb}/{target_pdb}_protein.pdb")
        raise FileNotFoundError(f"Reference protein structure not found for {target_pdb}")
    
    log.debug(f"Loading reference protein: {ref_path}")
    return bsio.load_structure(ref_path)


def load_target_data(target_pdb: str, target_smiles: Optional[str] = None) -> Tuple[Optional[str], Optional[Chem.Mol]]:
    """Load target data from processed SDF file.
    
    Args:
        target_pdb: Target PDB ID
        target_smiles: Optional target SMILES string
        
    Returns:
        Tuple of (SMILES string, crystal mol if available)
    """
    # Initialize return values
    smiles = target_smiles
    crystal_mol = None
    
    # Load all ligands from the processed SDF file
    with gzip.open(LIGANDS_SDF_GZ, 'rb') as fh:
        for mol in Chem.ForwardSDMolSupplier(fh, removeHs=False, sanitize=False):
            if not mol or not mol.GetNumConformers():
                continue
            
            # Check if this is our target
            mol_name = safe_name(mol, "").lower()
            if mol_name.startswith(target_pdb.lower()):
                # If we found our target, extract SMILES if needed
                if not smiles:
                    smiles = Chem.MolToSmiles(Chem.RemoveAllHs(Chem.Mol(mol)))
                    log.debug(f"Extracted SMILES from pre-processed data: {smiles}")
                
                # Save as crystal mol for RMSD comparison
                crystal_mol = Chem.Mol(mol)
                break
    
    if not crystal_mol:
        log.warning(f"No crystal ligand found for {target_pdb} in processed_ligands_new.sdf.gz")
        
    return smiles, crystal_mol


def load_uniprot_exclude(exclude_file: str) -> Set[str]:
    """Load UniProt IDs to exclude from a file.
    
    Args:
        exclude_file: Path to file containing UniProt IDs to exclude
        
    Returns:
        Set of UniProt IDs to exclude
    """
    exclude_uniprot = set()
    if os.path.exists(exclude_file):
        with open(exclude_file) as f:
            exclude_uniprot = {line.strip() for line in f if line.strip()}
    return exclude_uniprot


def get_uniprot_mapping() -> Dict[str, str]:
    """Load PDB to UniProt ID mapping from pdbbind_dates.json file.
    
    Returns:
        Dict mapping PDB IDs to UniProt IDs
    """
    pdb_to_uniprot = {}
    if os.path.exists(UNIPROT_MAP):
        with open(UNIPROT_MAP) as f:
            data = json.load(f)
            for pdb_id, info in data.items():
                if isinstance(info, dict) and "uniprot" in info:
                    pdb_to_uniprot[pdb_id] = info["uniprot"]
    return pdb_to_uniprot


def filter_templates_by_ca_rmsd(all_templates: List[Chem.Mol], ca_rmsd_threshold: float) -> List[Chem.Mol]:
    """Filter templates by CA RMSD threshold.
    
    Args:
        all_templates: List of template molecules with CA RMSD properties
        ca_rmsd_threshold: Maximum CA RMSD allowed (Angstroms)
        
    Returns:
        List of templates that pass the CA RMSD threshold
    """
    if ca_rmsd_threshold == float('inf'):
        return all_templates
        
    filtered_templates = []
    for tpl in all_templates:
        if tpl.HasProp("ca_rmsd"):
            try:
                ca_rmsd = float(tpl.GetProp("ca_rmsd"))
                if ca_rmsd <= ca_rmsd_threshold:
                    filtered_templates.append(tpl)
            except (ValueError, TypeError):
                # If CA RMSD property exists but can't be parsed, skip this template
                continue
        else:
            # If no CA RMSD property, include in filtered list (shouldn't happen in normal flow)
            filtered_templates.append(tpl)
    
    return filtered_templates


def get_templates_with_progressive_fallback(all_templates: List[Chem.Mol], 
                                          fallback_thresholds: List[float] = None) -> Tuple[List[Chem.Mol], float, bool]:
    """Apply progressive CA RMSD fallback with central atom final fallback.
    
    Args:
        all_templates: List of all available template molecules
        fallback_thresholds: List of CA RMSD thresholds to try (default: global constant)
        
    Returns:
        Tuple of (valid_templates, threshold_used, use_central_atom_fallback)
    """
    if fallback_thresholds is None:
        fallback_thresholds = CA_RMSD_FALLBACK_THRESHOLDS
    
    for threshold in fallback_thresholds:
        filtered_templates = filter_templates_by_ca_rmsd(all_templates, threshold)
        if filtered_templates:
            if threshold > CA_RMSD_THRESHOLD:
                log.warning(f"Using relaxed CA RMSD threshold ({threshold}Å) - found {len(filtered_templates)} templates (poses may be less accurate)")
            else:
                log.info(f"Found {len(filtered_templates)} templates with CA RMSD ≤ {threshold}Å")
            return filtered_templates, threshold, False
    
    # Ultimate fallback: find template with smallest CA RMSD and use central atom positioning
    best_template = None
    best_rmsd = float('inf')
    
    for tpl in all_templates:
        if tpl.HasProp("ca_rmsd"):
            try:
                ca_rmsd = float(tpl.GetProp("ca_rmsd"))
                if ca_rmsd < best_rmsd:
                    best_rmsd = ca_rmsd
                    best_template = tpl
            except (ValueError, TypeError):
                continue
    
    # If no template has CA RMSD, use first available
    if best_template is None and all_templates:
        best_template = all_templates[0]
        best_rmsd = "N/A"
    
    if best_template:
        log.warning(f"Using central atom fallback with best available template (CA RMSD: {best_rmsd}Å)")
        return [best_template], float('inf'), True
    
    # This should never happen since we have templates
    log.error("No templates available for central atom fallback")
    return [], float('inf'), False


def standardize_atom_arrays(arrays):
    """Standardize annotations across multiple atom arrays to make them compatible for stacking.
    
    This function resolves a critical issue in the protein alignment pipeline where
    atom arrays from different chains often have incompatible annotation dictionaries,
    causing errors when trying to stack them together for multi-chain analysis.
    
    Technical details:
    1. Biotite's AtomArray objects contain 'annotations' dictionaries with metadata
    2. When calling struc.stack(), all arrays must have identical annotation keys
    3. Different protein chains often have different annotations (chain-specific data)
    4. This function finds the intersection of annotations across all arrays
    5. It then creates new arrays with only the common annotations
    
    Used in transform_ligand() when:
    - Processing proteins with multiple chains from embedding data
    - Attempting to combine CA atoms from different chains into one array
    - Regular stacking fails due to annotation mismatches
    
    Without this function, multi-chain template alignment would fail for many proteins,
    forcing fallback to single-chain alignment and potentially reducing binding pocket
    alignment quality.
    
    Parameters:
        arrays: List of AtomArray objects to standardize
        
    Returns:
        List of AtomArray objects with compatible annotations, or just the first array
        if standardization is not possible
    """
    if not arrays or len(arrays) <= 1:
        return arrays
        
    # Check if all arrays have annotations
    if not all(hasattr(arr, 'annotations') for arr in arrays):
        # Return only the first array if some don't have annotations
        return [arrays[0]]
    
    # Get common annotation categories
    try:
        common_annot = set.intersection(*[set(arr.annotations) for arr in arrays])
        
        # Create standardized arrays with only common annotations
        std_arrays = []
        for arr in arrays:
            std_arr = arr.copy()
            # Keep only common annotations
            for annot in list(std_arr.annotations.keys()):
                if annot not in common_annot:
                    std_arr.annotations.pop(annot)
            std_arrays.append(std_arr)
        
        return std_arrays
    except Exception:
        # Return first array as fallback if anything goes wrong
        return [arrays[0]]



def validate_data_paths():
    """Verify data directories exist and are accessible."""
    required_dirs = [
        f"{DATA_DIR}/PDBBind/PDBbind_v2020_refined/refined-set/",
        f"{DATA_DIR}/PDBBind/PDBbind_v2020_other_PL/v2020-other-PL/"
    ]
    
    for d in required_dirs:
        if not os.path.exists(d):
            log.warning(f"Directory {d} not found. Some complexes may be missing.")
    
    # Check if output directory exists, create if not
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)
        log.debug(f"Created output directory: {OUT_DIR}")
    
    # Check if processed ligands file exists
    if not os.path.exists(LIGANDS_SDF_GZ):
        log.error(f"Processed ligands file not found: {LIGANDS_SDF_GZ}")
        log.error("This file is required as we must use pre-processed ligands, not the original PDBbind ligands")
        return False
        
    return True


def main():
    global REF_STRUCT, TARGET_PDB, TARGET_SMILES, N_CONFS, SIM_THRESHOLD, OUT_DIR, CA_RMSD_THRESHOLD
    global N_WORKERS_PIPELINE, EMBED_THREADS # Expose to global for functions that might need them
    
    # Clear any previous errors at the start of each run
    PipelineErrorTracker.clear_errors()
    
    # Clear any previous alignment tracking at the start of each run  
    ProteinAlignmentTracker.clear_logs()
    
    # Parse command line arguments first, to set logging level early
    parser = argparse.ArgumentParser(
        description="Template-based MCS pose prediction pipeline"
    )

    # Core arguments
    parser.add_argument(
        "--target-pdb",
        type=str,
        help="Target PDB ID (for PDBbind datasets and pre-calculated embeddings)"
    )
    parser.add_argument(
        "--target-smiles",
        type=str,
        help="Target ligand SMILES (used if --target-ligand-sdf or --target-pdb not providing ligand)"
    )
    parser.add_argument(
        "--no-realign",
        action="store_true",
        help="Use raw conformer before shape‐align"
    )
    parser.add_argument(
        "--exclude-uniprot",
        type=str,
        default=None,
        help="File containing UniProt IDs to exclude (for time splits, requires --enable-uniprot-exclusion)"
    )
    parser.add_argument(
        "--enable-uniprot-exclusion",
        action="store_true",
        default=False,
        help="Enable UniProt ID exclusion (default: disabled)"
    )
    parser.add_argument(
        "--n-conformers",
        type=int,
        default=N_CONFS,
        help=f"Number of conformers to generate (default: {N_CONFS})"
    )
    parser.add_argument(
               "--template-knn",
        type=int,
        default=100,
        help="Number of nearest neighbors to use for template selection (default: 100)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Optional similarity threshold (overrides KNN if provided)"
    )
    parser.add_argument(
        "--ca-rmsd-threshold",
        type=float,
        default=CA_RMSD_THRESHOLD,
        help=f"Maximum C-alpha RMSD in Angstroms for protein filtering (default: {CA_RMSD_THRESHOLD})"
    )
    parser.add_argument(
        "--ref-chains",
        type=str,
        default=None,
        help="Comma-separated list of reference protein chains to use"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Set the logging level (default: WARNING)"
    )
    # --internal-pipeline-workers is now supported for explicit control
    parser.add_argument(
        "--internal-pipeline-workers",
        type=int,
        default=None,
        help="Override the number of internal workers for template transformation and conformer scoring. If set, overrides --multiprocessing. Use 1 when running in parallel from a benchmark to avoid oversubscription."
    )

    # New multiprocessing flag
    parser.add_argument(
        "--multiprocessing",
        action=argparse.BooleanOptionalAction, # Provides --multiprocessing and --no-multiprocessing
        default=True, # Default to True (use multiprocessing)
        help="Enable multiprocessing for internal pipeline tasks (default: enabled, uses all cores). If running as part of a parallel benchmark, set --internal-pipeline-workers=1 to avoid oversubscription."
    )

    # Embedding cache control
    parser.add_argument(
        "--use-embedding-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable embedding disk cache (default: enabled)."
    )
    
    parser.add_argument(
        "--embedding-cache-dir",
        type=str,
        default=None,
        help="Custom directory for embedding cache (default: ~/.cache/templ/embeddings)."
    )
    
    parser.add_argument(
        "--clear-embedding-cache",
        action="store_true",
        help="Clear the embedding cache before running."
    )
    
    # Batch embedding processing
    parser.add_argument(
        "--batch-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable batch embedding processing (default: enabled)."
    )
    
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Maximum batch size for embedding processing (default: 8)."
    )

    # Phase A arguments
    parser.add_argument(
        "--target-protein-file",
        type=str,
        default=None,
        help="Path to a user-provided PDB file for the target protein."
    )
    parser.add_argument(
        "--target-ligand-sdf",
        type=str,
        default=None,
        help="Path to a user-provided SDF file for the target ligand."
    )
    parser.add_argument(
        "--reference-ligand-sdf",
        type=str,
        default=None,
        help="Path to an SDF file for the reference crystal ligand (for RMSD calculation)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUT_DIR,
        help=f"Directory to save output files (default: {OUT_DIR})"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output filenames. If not provided, derived from target PDB or protein filename."
    )
    parser.add_argument(
        "--template-pdb-ids-file",
        type=str,
        default=None,
        help="Optional path to a file containing a newline-separated list of PDB IDs to use as the template pool (requires --enable-pdb-filtering)."
    )
    parser.add_argument(
        "--enable-pdb-filtering",
        action="store_true",
        default=False,
        help="Enable PDB ID filtering based on --template-pdb-ids-file (default: disabled)."
    )

    # New arguments for enhanced pose output
    parser.add_argument(
        "--save-all-poses",
        action="store_true",
        help="Save all generated poses ranked by combo Tanimoto score"
    )
    parser.add_argument(
        "--save-mcs-info", 
        action="store_true",
        help="Include detailed MCS information in pose properties"
    )
    parser.add_argument(
        "--max-poses",
        type=int,
        default=None,
        help="Maximum number of poses to save (default: all)"
    )
    parser.add_argument(
        "--skip-target-validation",
        action="store_true",
        help="Skip target molecule validation (allows processing of large peptides and complex molecules)"
    )
    parser.add_argument(
        "--peptide-threshold",
        type=int,
        default=8,
        help="Maximum number of peptide residues allowed (default: 8, use with --skip-target-validation for unlimited)"
    )

    args = parser.parse_args()

    # Set log level based on argument
    log_level = getattr(logging, args.log_level)
    reset_logging(log_level)
    
    # Determine N_WORKERS_PIPELINE based on multiprocessing flag or explicit override
    if args.internal_pipeline_workers is not None:
        N_WORKERS_PIPELINE = args.internal_pipeline_workers
        EMBED_THREADS = N_WORKERS_PIPELINE
        if args.multiprocessing and N_WORKERS_PIPELINE > 1:
            log.warning("Both --multiprocessing and --internal-pipeline-workers > 1. This may cause oversubscription. It is recommended to set --internal-pipeline-workers=1 to avoid oversubscription.")
    else:
        if args.multiprocessing:
            N_WORKERS_PIPELINE = os.cpu_count() or 1
            EMBED_THREADS = N_WORKERS_PIPELINE
            log.info(f"Multiprocessing ENABLED. Using {N_WORKERS_PIPELINE} worker(s) for internal pipeline tasks.")
        else:
            N_WORKERS_PIPELINE = 1
            EMBED_THREADS = 1
            log.info("Multiprocessing DISABLED. Using 1 worker for internal pipeline tasks.")
    log.debug(f"RDKit EMBED_THREADS (for ps.numThreads) set to: {EMBED_THREADS}")

    # Validate data paths
    if not validate_data_paths():
        log.error("Path validation failed. Exiting.")
        return
    
    # Update global variables based on arguments
    N_CONFS = args.n_conformers
    if args.similarity_threshold is not None:
        SIM_THRESHOLD = args.similarity_threshold
    
    # Update CA_RMSD_THRESHOLD if specified
    CA_RMSD_THRESHOLD = args.ca_rmsd_threshold
    
    OUT_DIR = args.output_dir # Update global OUT_DIR
    os.makedirs(OUT_DIR, exist_ok=True)

    # Determine Target Identifier (for logging, output naming if no prefix)
    target_identifier = TARGET_PDB # Default
    if args.target_protein_file:
        target_identifier = Path(args.target_protein_file).stem
    elif args.target_pdb:
        target_identifier = args.target_pdb
    
    if args.output_prefix:
        output_file_prefix = args.output_prefix
    elif target_identifier:
        output_file_prefix = target_identifier
    else: # Should not happen if one of target_pdb or target_protein_file is required
        output_file_prefix = "mcs_output"
        log.warning("No target PDB or protein file specified, using generic output prefix 'mcs_output'.")

    # Define output file path with collision detection
    pose_sdf = get_unique_filename(OUT_DIR, f"{output_file_prefix}_poses_multi", ".sdf")
    
    # Initialize target_smiles and crystal_mol
    target_smiles_val = args.target_smiles # May be None
    crystal_mol = None
    tgt_noH = None # This will hold the query ligand molecule

    # --- Target Protein Loading ---
    custom_protein_embedding = None # For embedding of user-provided protein file

    if args.target_protein_file:
        if not os.path.exists(args.target_protein_file):
            log.error(f"Target protein file not found: {args.target_protein_file}")
            return
        try:
            REF_STRUCT = bsio.load_structure(args.target_protein_file)
            TARGET_PDB = target_identifier # Use stem for PDB ID context
            log.info(f"Loaded user-provided target protein: {args.target_protein_file} (using ID: {TARGET_PDB})")

            # On-demand embedding handled by EmbeddingManager later
            
        except Exception as e:
            log.error(f"Failed to load or process target protein file {args.target_protein_file}: {e}")
            log.error(traceback.format_exc())
            return
    elif args.target_pdb:
        TARGET_PDB = args.target_pdb # Set global
        target_identifier = TARGET_PDB
        try:
            REF_STRUCT = load_reference_protein(TARGET_PDB)
        except Exception as e:
            log.error(f"Failed to load reference protein for {TARGET_PDB}: {str(e)}")
            return
    else:
        log.error("Either --target-pdb or --target-protein-file must be specified.")
        return

    # Get reference chains from argument or determine from structure
    ref_chains = None
    if args.ref_chains:
        ref_chains = args.ref_chains.split(',')
        log.debug(f"Using user-specified reference chains: {ref_chains}")
    else:
        # Get available chains
        ref_prot = REF_STRUCT[filter_amino_acids(REF_STRUCT)]
        ref_available_chains = list(get_chains(ref_prot))
        if ref_available_chains:
            ref_chains = [ref_available_chains[0]]
            log.debug(f"Using automatically detected reference chain: {ref_chains}")

    # Initialize allowed_template_pdb_ids. This will be populated if the file is provided.
    allowed_template_pdb_ids: Optional[Set[str]] = None
    if args.template_pdb_ids_file and args.enable_pdb_filtering:
        if os.path.exists(args.template_pdb_ids_file):
            with open(args.template_pdb_ids_file, 'r') as f_template_ids:
                allowed_template_pdb_ids = {line.strip().lower() for line in f_template_ids if line.strip()}
            log.info(f"Loaded {len(allowed_template_pdb_ids)} PDB IDs from --template-pdb-ids-file: {args.template_pdb_ids_file}")
        else:
            log.warning(f"--template-pdb-ids-file specified ({args.template_pdb_ids_file}) but not found. Proceeding without this specific template filtering.")
    elif args.template_pdb_ids_file and not args.enable_pdb_filtering:
        log.info(f"--template-pdb-ids-file provided but --enable-pdb-filtering not set. Ignoring template filtering.")

    # Initialize the EmbeddingManager for unified embedding handling
    embedding_manager = EmbeddingManager(
        EMB_NPZ,
        use_cache=args.use_embedding_cache,
        cache_dir=args.embedding_cache_dir,
        enable_batching=args.batch_embedding,
        max_batch_size=args.max_batch_size
    )
    
    # Clear cache if requested
    if args.clear_embedding_cache and args.use_embedding_cache:
        if embedding_manager.clear_cache():
            log.info("Embedding cache cleared successfully.")
        else:
            log.warning("Failed to clear embedding cache. Proceeding with existing cache.")
    
    # Print cache stats 
    if args.use_embedding_cache:
        cache_stats = embedding_manager.get_cache_stats()
        log.info(f"Embedding cache: {cache_stats['count']} entries ({cache_stats['size_mb']:.2f} MB)")
    
    # Load UniProt mapping and set in manager for exclusion filtering only if enabled
    pdb_to_uniprot = {}
    if args.enable_uniprot_exclusion:
        pdb_to_uniprot = get_uniprot_mapping()
        embedding_manager.set_uniprot_mapping(pdb_to_uniprot)
        log.info("UniProt exclusion enabled. Loaded UniProt mapping.")
    else:
        log.info("UniProt exclusion disabled. Skipping UniProt mapping load.")
    
    # Load exclude UniProt IDs if provided and enabled
    exclude_uniprot = set()
    if args.exclude_uniprot and args.enable_uniprot_exclusion:
        exclude_uniprot = load_uniprot_exclude(args.exclude_uniprot)
        log.info(f"Loaded {len(exclude_uniprot)} UniProt IDs to exclude from templates")
    elif args.exclude_uniprot and not args.enable_uniprot_exclusion:
        log.warning(f"--exclude-uniprot provided but --enable-uniprot-exclusion not set. Ignoring UniProt exclusion file.")

    # Now handle target protein embedding with the manager
    if args.target_protein_file:
        # Generate embedding for custom protein file
        target_embedding, target_chains_str = embedding_manager.get_embedding(
            TARGET_PDB, 
            args.target_protein_file, 
            args.ref_chains
        )
        
        if target_embedding is None:
            log.error(f"Failed to generate embedding for custom protein file {args.target_protein_file}")
            return
        
        log.info(f"Successfully generated embedding for custom protein {TARGET_PDB}")
        target_chains = target_chains_str.split(",") if target_chains_str else (ref_chains or ["A"])
    elif TARGET_PDB:
        # Get embedding for existing PDB ID (from database or on-demand)
        target_pdb_path = pdb_path(TARGET_PDB)
        target_embedding, target_chains_str = embedding_manager.get_embedding(
            TARGET_PDB,
            target_pdb_path,
            args.ref_chains
        )
        
        if target_embedding is None:
            log.error(f"Failed to get embedding for {TARGET_PDB}")
            return
            
        target_chains = target_chains_str.split(",") if target_chains_str else (ref_chains or ["A"])
    else:
        log.error("No target information for embedding calculation")
        return
        
    # ─── load input ligands from pre-processed file ───────────────────────────────────
    ligs: Dict[str, Chem.Mol] = {}
    target_can = Chem.MolToSmiles(Chem.MolFromSmiles(TARGET_SMILES))
    
    with gzip.open(LIGANDS_SDF_GZ, 'rb') as fh:
        for mol in Chem.ForwardSDMolSupplier(fh, removeHs=False, sanitize=False):
            if not mol or not mol.GetNumConformers():
                continue
            
            pid_from_mol = safe_name(mol, f"lig_placeholder")[:4].lower()

            # Filter by allowed_template_pdb_ids only if enabled
            if args.enable_pdb_filtering and allowed_template_pdb_ids is not None and pid_from_mol not in allowed_template_pdb_ids:
                continue

            smi = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)))
            if smi == target_can: # This is the self-exclusion for the *target* ligand, which is fine.
                continue
            
            # Template validation removed - templates should remain available unless they're problematic targets
            ligs[pid_from_mol] = mol

    log.debug(f"Loaded {len(ligs)} input ligands from pre-processed SDF (after potential filtering by --template-pdb-ids-file)")
    
    # Removed template skipping tracking since templates are no longer filtered

    # --- Target Ligand Loading & Crystal Mol for RMSD ---
    # target_smiles_val was initialized from args.target_smiles
    # crystal_mol was initialized to None

    # Strategic error tracking: Target data loading
    with PipelineErrorTracker(TARGET_PDB, "target_data_loading"):
        # --- Reference Ligand (crystal_mol for RMSD) Loading ---
        if args.reference_ligand_sdf:
            if not os.path.exists(args.reference_ligand_sdf):
                log.error(f"Reference ligand SDF file not found: {args.reference_ligand_sdf}")
            else:
                try:
                    suppl_ref = Chem.SDMolSupplier(args.reference_ligand_sdf, removeHs=False, sanitize=True)
                    ref_mol_from_sdf = next(suppl_ref, None)
                    if ref_mol_from_sdf:
                        crystal_mol = ref_mol_from_sdf
                        log.info(f"Loaded reference crystal ligand from explicitly provided SDF: {args.reference_ligand_sdf}")
                    else:
                        log.warning(f"No molecule found in reference ligand SDF: {args.reference_ligand_sdf}. RMSD may not be calculated if not found via PDB ID.")
                except Exception as e:
                    log.error(f"Error loading reference ligand SDF {args.reference_ligand_sdf}: {e}. RMSD may not be calculated if not found via PDB ID.")
        
        # --- Query Ligand (tgt_noH and TARGET_SMILES) Loading ---
        if args.target_ligand_sdf:
            if not os.path.exists(args.target_ligand_sdf):
                log.error(f"Target (query) ligand SDF file not found: {args.target_ligand_sdf}")
                return # Cannot proceed without a query ligand
            try:
                suppl_query = Chem.SDMolSupplier(args.target_ligand_sdf, removeHs=False, sanitize=True)
                mol_from_sdf = next(suppl_query, None)
                if mol_from_sdf:
                    tgt_noH = mol_from_sdf
                    target_smiles_val = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol_from_sdf))) # Ensure SMILES matches SDF
                    log.info(f"Loaded target (query) ligand from user-provided SDF: {args.target_ligand_sdf}")
                    log.debug(f"Derived SMILES for query from SDF: {target_smiles_val}")
                else:
                    log.error(f"No molecule found in target (query) ligand SDF: {args.target_ligand_sdf}")
                    return
            except Exception as e:
                log.error(f"Error loading target (query) ligand SDF {args.target_ligand_sdf}: {e}")
                return
        elif target_smiles_val: # User provided --target-smiles (and not --target-ligand-sdf)
            try:
                mol_from_smiles = Chem.MolFromSmiles(target_smiles_val)
                if mol_from_smiles:
                    tgt_noH = mol_from_smiles
                    log.info(f"Using target (query) ligand from user-provided SMILES: {target_smiles_val}")
                else:
                    log.error(f"Could not parse user-provided SMILES for query ligand: {target_smiles_val}")
                    return
            except Exception as e:
                log.error(f"Error processing user-provided SMILES for query ligand {target_smiles_val}: {e}")
                return
        elif args.target_pdb: # No explicit query SDF or SMILES, try to get from PDBbind via --target-pdb
            log.info(f"Attempting to load query ligand (SMILES and potentially crystal structure for RMSD) for PDB ID {TARGET_PDB} from PDBbind data.")
            loaded_smiles_pdb, loaded_crystal_mol_pdb = load_target_data(TARGET_PDB, target_smiles_val) 
            
            if loaded_smiles_pdb:
                target_smiles_val = loaded_smiles_pdb # This becomes the definitive SMILES for the query
                tgt_noH = Chem.MolFromSmiles(target_smiles_val)
                if not tgt_noH:
                    log.error(f"Could not parse SMILES '{target_smiles_val}' derived from PDBbind for {TARGET_PDB}.")
                    return
                log.info(f"Derived query ligand SMILES '{target_smiles_val}' for {TARGET_PDB} from PDBbind data.")
            else:
                log.error(f"Could not derive query ligand SMILES for {TARGET_PDB} from PDBbind data.")
                return # Cannot proceed without a query SMILES

            if crystal_mol is None and loaded_crystal_mol_pdb: # If --reference-ligand-sdf wasn't used or failed, use PDBbind's
                crystal_mol = loaded_crystal_mol_pdb
                log.info(f"Loaded crystal ligand for RMSD for {TARGET_PDB} from PDBbind data.")
            elif crystal_mol is not None and loaded_crystal_mol_pdb:
                log.info(f"Explicit --reference-ligand-sdf was provided; PDBbind crystal data for {TARGET_PDB} (if found) will not override it.")
            elif crystal_mol is None and not loaded_crystal_mol_pdb:
                log.warning(f"No crystal ligand found for RMSD for {TARGET_PDB} from PDBbind data (and no --reference-ligand-sdf). RMSD will be skipped.")
                
        else: # No way to determine query ligand
            log.error("No target (query) ligand specified via --target-ligand-sdf, --target-smiles, or derivable from --target-pdb.")
            return

    # Final check and setup for TARGET_SMILES global (used for MCES query)
    if not target_smiles_val:
        log.error("TARGET_SMILES could not be determined after all checks. This is required for MCES.")
        return
    TARGET_SMILES = target_smiles_val

    # Ensure tgt_noH (query Mol object) is consistent with TARGET_SMILES
    if tgt_noH is None: # Should only happen if derived from PDBbind and MolFromSmiles failed (already errored) or logic error
        tgt_noH = Chem.MolFromSmiles(TARGET_SMILES)
        if not tgt_noH:
            log.error(f"Failed to create final query molecule from TARGET_SMILES: {TARGET_SMILES}")
            return

    # Validate target molecule for problematic structures
    tgt_no_h_clean = Chem.RemoveHs(Chem.Mol(tgt_noH))
    
    if not args.skip_target_validation:
        is_target_valid, target_warning = validate_target_molecule(tgt_no_h_clean, f"target_{TARGET_PDB}", args.peptide_threshold)
        if not is_target_valid:
            log.error(f"Target molecule validation failed: {target_warning}")
            log.error("Cannot proceed with problematic target molecule. Please provide a different target or use --skip-target-validation.")
            return
    else:
        log.warning("Target molecule validation SKIPPED - processing may fail for complex molecules")

    # Strategic error tracking: Template finding
    with PipelineErrorTracker(TARGET_PDB, "template_finding"):
        # Find neighbors using the EmbeddingManager
        k_neighbors = args.template_knn if args.similarity_threshold is None else None
        
        # If batch embedding is enabled, pre-process all potential templates
        if args.batch_embedding and allowed_template_pdb_ids:
            log.info(f"Pre-processing embeddings for template pool ({len(allowed_template_pdb_ids)} PDBs)")
            embedding_manager.prepare_batch_embeddings(list(allowed_template_pdb_ids))
        
        # Ensure template pool is filtered by available embeddings
        if allowed_template_pdb_ids:
            available_in_db = set(embedding_manager.pdb_to_idx.keys())
            allowed_template_pdb_ids = allowed_template_pdb_ids.intersection(available_in_db)
            log.info(f"Filtered template pool to {len(allowed_template_pdb_ids)} PDBs with embeddings")

        neighbor_candidates = embedding_manager.find_neighbors(
            TARGET_PDB,
            target_embedding,
            exclude_uniprot_ids=exclude_uniprot if args.enable_uniprot_exclusion else set(),
            allowed_pdb_ids=allowed_template_pdb_ids if args.enable_pdb_filtering else None,
            k=k_neighbors,
            similarity_threshold=args.similarity_threshold,
            return_similarities=True
        )
        
        if not neighbor_candidates:
            log.error(f"No neighbors found for {TARGET_PDB}! Cannot proceed.")
            return
        
        # Extract PDB IDs and similarity scores
        nbrs = [pid for pid, _ in neighbor_candidates]
        similarity_scores = {pid: score for pid, score in neighbor_candidates}
        log.info(f"Found {len(nbrs)} neighbors for {TARGET_PDB}")

    # ─── transform ligands into templates ────────────────────────────────
    templates: List[Chem.Mol] = []
    # Check if parallel execution is beneficial
    if N_WORKERS_PIPELINE > 1 and len(nbrs) > 1 :
        # Use pebble for better process management if available
        if PEBBLE_AVAILABLE:
            with ProcessPool(max_workers=N_WORKERS_PIPELINE) as pool:
                futures = []
                for pid in nbrs:
                    if pid in ligs and pdb_path(pid):
                        future = pool.schedule(
                            transform_ligand,
                            args=[
                                pdb_path(pid), 
                                ligs[pid], 
                                pid, 
                                REF_STRUCT, 
                                crystal_mol,
                                ref_chains,
                                embedding_manager.get_chain_data(pid).split(',') if embedding_manager.get_chain_data(pid) else None,
                                similarity_scores.get(pid, 0.0)
                            ]
                        )
                        futures.append(future)
                
                for future in tqdm(futures, desc="templates (parallel/pebble)"):
                    try:
                        tpl = future.result()
                        if tpl:
                            templates.append(tpl)
                    except Exception as e:
                        log.warning(f"Template transformation task failed: {str(e)}")
        else:
            # Fallback to ProcessPoolExecutor
            mp_context = mp.get_context('spawn')
            with ProcessPoolExecutor(max_workers=N_WORKERS_PIPELINE, mp_context=mp_context) as pool:
                # futures are already defined before this block in the original code
                # This re-scoping is to use the ProcessPoolExecutor
                # The original code structure for submitting futures:
                futures_map = { # Renamed to avoid conflict with futures from outer scope if any
                    pool.submit(
                        transform_ligand, 
                        pdb_path(pid), 
                        ligs[pid] if pid in ligs else None, # Ensure pid is in ligs
                        pid, 
                        REF_STRUCT, 
                        crystal_mol,
                        ref_chains,
                        embedding_manager.get_chain_data(pid).split(',') if embedding_manager.get_chain_data(pid) else None,
                        similarity_scores.get(pid, 0.0)
                    ): pid
                    for pid in nbrs if pid in ligs and pdb_path(pid) # Filter here
                }
                actual_submitted_tasks = len(futures_map)
                if actual_submitted_tasks > 0:
                     for fut in tqdm(as_completed(futures_map), total=actual_submitted_tasks, desc="templates (parallel)"):
                        tpl = fut.result()
                        if tpl:
                            templates.append(tpl)
                else:
                    log.debug("No valid template tasks to submit for parallel processing.")
    else: # Sequential execution
        log.debug(f"Template processing: {len(nbrs)} templates (sequential)")
        for pid in tqdm(nbrs, desc="templates (sequential)"):
            if pid in ligs and pdb_path(pid):
                tpl = transform_ligand(
                    pdb_path(pid),
                    ligs[pid],
                    pid,
                    REF_STRUCT,
                    crystal_mol,
                    ref_chains,
                    embedding_manager.get_chain_data(pid).split(',') if embedding_manager.get_chain_data(pid) else None,
                    similarity_scores.get(pid, 0.0)
                )
                if tpl:
                    templates.append(tpl)


    # Apply progressive CA RMSD fallback strategy
    if not templates:
        log.error("No valid templates found after initial transformation")
        sys.exit(1)
    
    log.info(f"Generated {len(templates)} templates from protein alignment")
    
    # Apply progressive CA RMSD filtering with fallback strategy
    filtered_templates, threshold_used, use_central_atom_fallback = get_templates_with_progressive_fallback(templates)
    
    if not filtered_templates:
        log.error("No valid templates found even after central atom fallback")
        sys.exit(1)
    
    # Use filtered templates for MCS search
    templates = filtered_templates
    log.info(f"Using {len(templates)} templates after CA RMSD filtering (threshold: {threshold_used}Å)")

    # Strategic error tracking: MCS computation and pose generation
    with PipelineErrorTracker(TARGET_PDB, "mcs_and_pose_generation"):
        # ─── find MCS against query ligand ───────────────────────────────────
        tgt_noH = Chem.RemoveHs(Chem.MolFromSmiles(TARGET_SMILES))
        
        # Force central atom MCS when in fallback mode
        if use_central_atom_fallback:
            log.info("Using central atom positioning due to poor protein alignment quality")
            idx, sm = 0, "*"  # Force central atom SMARTS
            mcs_details = {
                "atom_count": 1,
                "bond_count": 0,
                "similarity_score": 0.0,
                "query_atoms": [get_central_atom(tgt_noH)],
                "template_atoms": [get_central_atom(templates[0])],
                "smarts": "*",
                "central_atom_fallback": True
            }
        else:
            # Normal MCS search
            if args.save_mcs_info or args.save_all_poses:
                idx, sm, mcs_details = find_mcs(tgt_noH, templates, return_details=True)
            else:
                idx, sm = find_mcs(tgt_noH, templates)
                mcs_details = {}
        
        if idx is None:
            log.error("MCS search failed - no common substructure found between query and any template")
            log.error(f"Attempted MCS search with {len(templates)} templates using CA RMSD threshold {threshold_used}Å")
            
            # Log template details for debugging
            template_info = []
            for i, tpl in enumerate(templates[:5]):  # Show first 5 templates
                tpl_name = safe_name(tpl, f"tpl_{i}")[:4]
                ca_rmsd = tpl.GetProp("ca_rmsd") if tpl.HasProp("ca_rmsd") else "N/A"
                similarity = tpl.GetProp("embedding_similarity") if tpl.HasProp("embedding_similarity") else "N/A"
                template_info.append(f"  {tpl_name}: CA_RMSD={ca_rmsd}Å, Similarity={similarity}")
            
            if template_info:
                log.error("Available templates (showing first 5):")
                for info in template_info:
                    log.error(info)
            
            sys.exit(1)
        tpl = templates[idx]
        tpl_pid = safe_name(tpl, "tpl")[:4].lower()
        
        # Log MCS information
        if mcs_details:
            log.info(f"MCES winner: PID={tpl_pid}, SMARTS={sm}, Atoms={mcs_details.get('atom_count', 0)}, Bonds={mcs_details.get('bond_count', 0)}")
        else:
            log.info(f"MCES winner: PID={tpl_pid}")

        # Save the winning transformed template for visualization
        winning_template_sdf_filename = os.path.join(OUT_DIR, f"target_{output_file_prefix}_mces_winner_template_{tpl_pid}.sdf")
        try:
            with Chem.SDWriter(winning_template_sdf_filename) as w:
                w.write(tpl)
            log.info(f"Saved MCES winning transformed template to: {winning_template_sdf_filename}")
        except Exception as e:
            log.error(f"Failed to save MCES winning template SDF: {e}")

        # Extract and log alignment quality if available
        if tpl.HasProp("embedding_similarity"):
            log.debug(f"Template similarity score: {tpl.GetProp('embedding_similarity')}")
        if tpl.HasProp("ref_chains"):
            log.debug(f"Template aligned using reference chains: {tpl.GetProp('ref_chains')}")
        if tpl.HasProp("mob_chains"):
            log.debug(f"Template chains used: {tpl.GetProp('mob_chains')}")
        if tpl.HasProp("ca_rmsd"):
            log.debug(f"Template CA RMSD: {tpl.GetProp('ca_rmsd')}")

        # ─── constrained embedding ──────────────────────────────────────────
        log.debug(f"Generating {N_CONFS} conformers using constrained embedding")
        confs = constrained_embed(tgt_noH, tpl, sm, N_CONFS, n_workers_pipeline=N_WORKERS_PIPELINE)
        log.debug(f"Generated {confs.GetNumConformers()} conformers")

        # ─── select best by shape/color/combo ───────────────────────────────
        log.debug("Selecting best poses using all scoring methods (shape, color, combo)")
        
        # Use enhanced selection when all poses are requested
        if args.save_all_poses:
            all_ranked_poses = select_best(confs, tpl, args.no_realign, N_WORKERS_PIPELINE, return_all_ranked=True)
            
            # Save all ranked poses with collision detection
            all_poses_sdf = get_unique_filename(OUT_DIR, f"{output_file_prefix}_all_poses_ranked", ".sdf")
            save_all_ranked_poses(all_ranked_poses, all_poses_sdf, mcs_details, tpl, crystal_mol, TARGET_PDB, args.max_poses)
            log.info(f"Saved {len(all_ranked_poses[:args.max_poses] if args.max_poses else all_ranked_poses)} ranked poses to {all_poses_sdf}")
            
            # Extract best poses for backward compatibility
            sel = extract_best_from_ranked(all_ranked_poses)
        else:
            # Existing behavior - no changes
            sel = select_best(confs, tpl, args.no_realign, n_workers=N_WORKERS_PIPELINE)

    # Strategic error tracking: Output writing
    with PipelineErrorTracker(TARGET_PDB, "output_writing"):
        # ─── write poses to SDF ─────────────────────────────────────────────
        with Chem.SDWriter(pose_sdf) as w:
            for metric, (pose, all_scores_for_this_pose) in sel.items():
                if pose is None:
                    continue
                
                # Create a copy to avoid modifying the original                
                pose_copy = Chem.Mol(pose)
                
                # Enhance properties with MCS information if requested
                if args.save_mcs_info and mcs_details:
                    template_info = extract_template_info(tpl)
                    enhance_pose_properties(pose_copy, mcs_details, template_info, 0, 1, 1)
                
                # Set primary metric identification
                pose_copy.SetProp("_Name",        f"{TARGET_PDB}_{metric}_pose")
                pose_copy.SetProp("metric",       metric) # Identifies why this pose was selected (e.g., best for "shape")
                pose_copy.SetProp("metric_score", f"{all_scores_for_this_pose[metric]:.3f}") # The score for the primary metric
                
                # Set all three calculated scores as distinct properties
                # Using a consistent naming convention, e.g., "tanimoto_shape_score"
                if "shape" in all_scores_for_this_pose:
                    pose_copy.SetProp("tanimoto_shape_score", f"{all_scores_for_this_pose['shape']:.3f}")
                if "color" in all_scores_for_this_pose:
                    pose_copy.SetProp("tanimoto_color_score", f"{all_scores_for_this_pose['color']:.3f}")
                if "combo" in all_scores_for_this_pose:
                    pose_copy.SetProp("tanimoto_combo_score", f"{all_scores_for_this_pose['combo']:.3f}")

                # Copy template alignment properties from the MCES-winning template (tpl)
                pose_copy.SetProp("template_pid", tpl_pid) # tpl_pid is defined earlier and holds the MCES winner PDB ID
                for prop_name in ["embedding_similarity", "ref_chains", "mob_chains", "ca_rmsd", 
                                  "aligned_residues_count", "total_ref_residues", "total_mob_residues", "aligned_percentage"]:
                    if tpl.HasProp(prop_name):
                        pose_copy.SetProp(f"template_{prop_name}", tpl.GetProp(prop_name))
                
                # If crystal structure exists, calculate RMSD immediately
                if crystal_mol:
                    try:
                        crys = Chem.RemoveHs(crystal_mol)
                        rms = rmsd_raw(pose_copy, crys)
                        pose_copy.SetProp("rmsd_to_crystal", f"{rms:.3f}")
                    except Exception as e:
                        log.error(f"Error calculating RMSD for {metric}: {str(e)}")
                
                # Write molecule to SDF
                w.write(pose_copy)

    # ─── optional evaluation: RMSD to crystal ──────────────────────────
    if crystal_mol:
        try:
            print("\nFinal RMSD to crystal (Å):")
            print("┌────────┬────────┬────────┬──────────┐") 
            print("│ metric │ score  │ RMSD   │ Template │") 
            print("├────────┼────────┼────────┼──────────┤") 
            
            # sel items are: metric_name, (pose_object, {all_scores_for_that_pose})
            for metric, (pose, scores_dict) in sel.items(): 
                if pose is None:
                    continue
                
                # The score to print here is the one for the primary metric   
                primary_score_for_metric = scores_dict[metric]
                # Calculate RMSD directly instead of reading from properties
                try:
                    crys = Chem.RemoveHs(crystal_mol)
                    rms = rmsd_raw(pose, crys)
                except Exception as e:
                    log.debug(f"Error calculating RMSD for {metric}: {str(e)}")
                    rms = float("nan")
                # tpl_pid is defined earlier and holds the MCES winner PDB ID
                print(f"│ {metric:<6} │ {primary_score_for_metric:6.3f} │ {rms:6.3f} │ {tpl_pid:<8} │")
            print("└────────┴────────┴────────┴──────────┘") 
            
            # Print alignment details if available
            if tpl.HasProp("ca_rmsd") and tpl.HasProp("aligned_residues_count"):
                aligned_count = tpl.GetProp("aligned_residues_count")
                ca_rmsd = tpl.GetProp("ca_rmsd")
                aligned_percentage = tpl.GetProp("aligned_percentage") if tpl.HasProp("aligned_percentage") else "N/A"
                print(f"\nTemplate Alignment Details:")
                print(f"- C-alpha RMSD: {ca_rmsd} Å")
                print(f"- Matched residues: {aligned_count}")
                print(f"- Alignment coverage: {aligned_percentage}")
                print(f"- Note: CA RMSD is calculated only on the matched residues, not the entire structure")
                
        except Exception as e:
            log.error(f"Error printing RMSD table: {str(e)}")
            log.error(traceback.format_exc())
    else:
        # Refined logging: Only log warning if RMSD calculation was expected but crystal_mol is missing.
        # If --reference-ligand-sdf was provided but failed to load, a warning was already issued.
        # If --target-pdb was used, load_target_data would issue a warning if not found.
        # This message is now more of a final status.
        if args.reference_ligand_sdf and not crystal_mol: # If user intended to provide one and it failed to load
            log.warning(f"RMSD evaluation skipped because the reference ligand from {args.reference_ligand_sdf} could not be loaded or was empty.")
        elif args.target_pdb and not crystal_mol: # If lookup from PDBbind data was expected and failed (and no explicit ref)
            log.warning(f"RMSD evaluation skipped: No crystal ligand was found for PDB ID {TARGET_PDB} via PDBbind, and no --reference-ligand-sdf was provided/successful.")
        elif not crystal_mol: # General case (e.g. using protein file + smiles, no reference specified, no PDB ID for lookup)
            log.info("RMSD evaluation skipped: No reference crystal ligand provided or found.")

    # ─── save structured results for reliable benchmark data exchange ────────────────
    try:
        # Collect pipeline results in structured format
        structured_results = collect_pipeline_results(sel, crystal_mol, tpl, TARGET_PDB, mcs_details)
        
        # Save structured results to JSON for benchmark consumption
        structured_file = save_structured_results(structured_results, OUT_DIR, TARGET_PDB)
        if structured_file:
            log.debug(f"Saved structured results to {structured_file}")
        
    except Exception as e:
        log.warning(f"Failed to save structured results: {e}")

    # Generate error report if any errors occurred
    error_report_file = PipelineErrorTracker.save_error_report(OUT_DIR, TARGET_PDB)
    if error_report_file:
        log.info(f"Pipeline error report saved to: {error_report_file}")
    
    # Generate alignment tracking report for investigation
    alignment_report_file = ProteinAlignmentTracker.save_alignment_report(OUT_DIR, TARGET_PDB)
    if alignment_report_file:
        log.info(f"Protein alignment tracking report saved to: {alignment_report_file}")
    
    # Validate pipeline output completeness
    alignment_summary = ProteinAlignmentTracker.get_alignment_summary()
    if alignment_summary["total_attempts"] > 0:
        success_rate = alignment_summary["success_rate"] * 100
        missing_ca_rmsd = alignment_summary["missing_ca_rmsd_count"]
        
        log.info(f"Protein alignment success rate: {success_rate:.1f}% ({alignment_summary['total_attempts']} attempts)")
        
        if missing_ca_rmsd > 0:
            log.warning(f"VALIDATION: {missing_ca_rmsd} successful alignments missing CA RMSD values - requires investigation")
        
        if success_rate < 95.0:
            log.warning(f"VALIDATION: Low alignment success rate ({success_rate:.1f}%) - check alignment tracking report")
    
    # If cache is enabled, print stats at the end
    if args.use_embedding_cache:
        cache_stats_end = embedding_manager.get_cache_stats()
        cache_delta = cache_stats_end['count'] - cache_stats['count'] if 'count' in cache_stats else cache_stats_end['count']
        if cache_delta > 0:
            log.info(f"Added {cache_delta} new entries to embedding cache.")


def enhance_pose_properties(pose: Chem.Mol, mcs_details: Dict, template_info: Dict, 
                          conformer_id: int, rank: int, total_poses: int) -> None:
    """Add enhanced properties to existing pose - modifies in place.
    
    Args:
        pose: Molecule to enhance with properties
        mcs_details: MCS information dictionary
        template_info: Template information dictionary
        conformer_id: Original conformer ID
        rank: Pose rank based on combo score
        total_poses: Total number of poses generated
    """
    
    # MCS information (only if available)
    if mcs_details:
        pose.SetProp("mcs_smarts", mcs_details.get("smarts", ""))
        pose.SetProp("mcs_atom_count", str(mcs_details.get("atom_count", 0)))
        pose.SetProp("mcs_bond_count", str(mcs_details.get("bond_count", 0)))
        pose.SetProp("mcs_similarity_score", f"{mcs_details.get('similarity_score', 0.0):.3f}")
        
        # Atom mapping (compact format)
        if mcs_details.get("query_atoms"):
            pose.SetProp("mcs_query_atoms", ",".join(map(str, mcs_details["query_atoms"])))
        if mcs_details.get("template_atoms"):
            pose.SetProp("mcs_template_atoms", ",".join(map(str, mcs_details["template_atoms"])))
    
    # Pose ranking information
    pose.SetProp("conformer_original_id", str(conformer_id))
    pose.SetProp("pose_rank_combo", str(rank))
    pose.SetProp("total_poses_generated", str(total_poses))
    
    # Template information (only if not already present)
    for key, value in template_info.items():
        prop_name = f"template_{key}" if not key.startswith("template_") else key
        if not pose.HasProp(prop_name) and value is not None:
            pose.SetProp(prop_name, str(value))


def save_all_ranked_poses(ranked_poses: List, output_file: str, mcs_details: Dict, 
                         template: Chem.Mol, crystal_mol: Optional[Chem.Mol], 
                         target_pdb: str, max_poses: Optional[int] = None) -> None:
    """Save all ranked poses with enhanced properties.
    
    Args:
        ranked_poses: List of (pose, scores, original_cid) tuples sorted by combo score
        output_file: Output SDF file path
        mcs_details: MCS information dictionary
        template: Template molecule
        crystal_mol: Crystal structure for RMSD calculation (optional)
        target_pdb: Target PDB ID for naming
        max_poses: Maximum number of poses to save (optional)
    """
    template_info = extract_template_info(template)
    
    # Limit poses if requested
    poses_to_save = ranked_poses[:max_poses] if max_poses else ranked_poses
    
    with Chem.SDWriter(output_file) as writer:
        for rank, (pose, scores, original_cid) in enumerate(poses_to_save, 1):
            # Create a copy to avoid modifying the original
            pose_copy = Chem.Mol(pose)
            
            # Enhance properties
            enhance_pose_properties(pose_copy, mcs_details, template_info, original_cid, rank, len(ranked_poses))
            
            # Set name and primary properties
            pose_copy.SetProp("_Name", f"{target_pdb}_pose_rank_{rank}")
            pose_copy.SetProp("metric", "combo")  # Primary ranking metric
            pose_copy.SetProp("metric_score", f"{scores['combo']:.3f}")
            
            # Set all Tanimoto scores (existing pattern)
            for score_type, score_value in scores.items():
                pose_copy.SetProp(f"tanimoto_{score_type}_score", f"{score_value:.3f}")
            
            # Calculate RMSD if crystal available (existing pattern)
            if crystal_mol:
                try:
                    crys = Chem.RemoveHs(crystal_mol)
                    rms = rmsd_raw(pose_copy, crys)
                    pose_copy.SetProp("rmsd_to_crystal", f"{rms:.3f}")
                except Exception as e:
                    log.error(f"Error calculating RMSD for rank {rank}: {str(e)}")
            
            writer.write(pose_copy)


def extract_best_from_ranked(ranked_poses: List) -> Dict:
    """Extract best poses per metric from ranked list for backward compatibility.
    
    Args:
        ranked_poses: List of (pose, scores, original_cid) tuples
        
    Returns:
        Dictionary mapping metric names to (best_pose, scores) tuples
    """
    best = {"shape": (None, {}), "color": (None, {}), "combo": (None, {})}
    
    # Find best for each metric
    for metric in best.keys():
        best_score = -1.0
        for pose, scores, _ in ranked_poses:
            if scores[metric] > best_score:
                best_score = scores[metric]
                best[metric] = (pose, scores)
    
    return best


def extract_template_info(template: Chem.Mol) -> Dict:
    """Extract template information from molecule properties.
    
    Args:
        template: Template molecule with properties
        
    Returns:
        Dictionary of template information
    """
    info = {}
    
    # Extract existing properties
    for prop in ["embedding_similarity", "ca_rmsd", "aligned_residues_count", 
                 "total_ref_residues", "total_mob_residues", "aligned_percentage"]:
        if template.HasProp(prop):
            info[prop] = template.GetProp(prop)
    
    return info


# ─── structured data output ──────────────────────────────────────────────

def save_structured_results(results_data: Dict, output_dir: str, target_pdb: str) -> Optional[str]:
    """Save structured pipeline results to JSON for reliable benchmark data exchange.
    
    This creates a structured data file that benchmarks can reliably parse instead of
    relying on stdout parsing which can be inconsistent due to logging variations.
    
    Args:
        results_data: Dictionary containing pipeline results
        output_dir: Output directory
        target_pdb: Target PDB ID for filename
        
    Returns:
        Path to saved JSON file or None if failed
    """
    try:
        results_file = get_unique_filename(output_dir, f"{target_pdb}_pipeline_results", ".json")
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        log.debug(f"Saved structured results to {results_file}")
        return results_file
    except Exception as e:
        log.error(f"Failed to save structured results: {e}")
        return None


def collect_pipeline_results(sel: Dict, crystal_mol: Optional[Chem.Mol], 
                           tpl: Chem.Mol, target_pdb: str,
                           mcs_details: Dict = None) -> Dict:
    """Collect all pipeline results into a structured format.
    
    Args:
        sel: Selection results from select_best
        crystal_mol: Crystal molecule for RMSD calculation
        tpl: Template molecule with alignment data
        target_pdb: Target PDB ID
        mcs_details: MCS information if available
        
    Returns:
        Structured results dictionary
    """
    results = {
        "target_pdb": target_pdb,
        "timestamp": datetime.now().isoformat(),
        "poses": {},
        "template_info": {},
        "mcs_info": mcs_details or {},
        "processing_notes": []
    }
    
    # Extract template information
    if tpl:
        results["template_info"] = {
            "template_pdb": tpl.GetProp("template_pdb") if tpl.HasProp("template_pdb") else None,
            "embedding_similarity": tpl.GetProp("embedding_similarity") if tpl.HasProp("embedding_similarity") else None,
            "ca_rmsd": tpl.GetProp("ca_rmsd") if tpl.HasProp("ca_rmsd") else None,
            "ref_chains": tpl.GetProp("ref_chains") if tpl.HasProp("ref_chains") else None,
            "mob_chains": tpl.GetProp("mob_chains") if tpl.HasProp("mob_chains") else None,
            "aligned_residues_count": tpl.GetProp("aligned_residues_count") if tpl.HasProp("aligned_residues_count") else None,
            "aligned_percentage": tpl.GetProp("aligned_percentage") if tpl.HasProp("aligned_percentage") else None
        }
    
    # Process pose results
    for metric, (pose, scores) in sel.items():
        if pose is None:
            continue
            
        pose_data = {
            "metric": metric,
            "scores": scores,
            "rmsd_to_crystal": None
        }
        
        # Calculate RMSD if crystal available
        if crystal_mol:
            try:
                crys = Chem.RemoveHs(crystal_mol)
                rms = rmsd_raw(pose, crys)
                pose_data["rmsd_to_crystal"] = rms
            except Exception as e:
                pose_data["rmsd_to_crystal"] = None
                results["processing_notes"].append(f"RMSD calculation failed for {metric}: {str(e)}")
        else:
            results["processing_notes"].append("No crystal structure available for RMSD calculation")
        
        results["poses"][metric] = pose_data
    
    return results


# ─── Protein alignment tracking system ──────────────────────────────────────────────
class ProteinAlignmentTracker:
    """Track protein alignment success/failure patterns for investigation."""
    
    _alignment_logs = []
    
    @classmethod
    def track_alignment_attempt(cls, pdb_id: str, stage: str, success: bool, details: Dict):
        """Track protein alignment attempt with detailed context."""
        alignment_log = {
            "pdb_id": pdb_id,
            "stage": stage,
            "success": success,
            "ca_rmsd": details.get("ca_rmsd"),
            "aligned_residues": details.get("aligned_residues"),
            "total_ref_residues": details.get("total_ref_residues"),
            "total_mob_residues": details.get("total_mob_residues"),
            "failure_reason": details.get("failure_reason") if not success else None,
            "fallback_used": details.get("fallback_used", False),
            "timestamp": datetime.now().isoformat()
        }
        
        cls._alignment_logs.append(alignment_log)
        
        # Log structured data for analysis
        ca_rmsd_str = f"{details.get('ca_rmsd', 'None')}"
        log.info(f"ALIGNMENT_TRACK|{pdb_id}|{stage}|{success}|{ca_rmsd_str}|{details.get('failure_reason', '')}")
    
    @classmethod
    def get_alignment_summary(cls) -> Dict[str, Any]:
        """Generate summary of alignment attempts."""
        if not cls._alignment_logs:
            return {"total_attempts": 0, "success_rate": 0.0, "stage_failures": {}}
        
        total = len(cls._alignment_logs)
        successful = sum(1 for log in cls._alignment_logs if log["success"])
        
        stage_failures = {}
        for log in cls._alignment_logs:
            if not log["success"]:
                stage = log["stage"]
                stage_failures[stage] = stage_failures.get(stage, 0) + 1
        
        return {
            "total_attempts": total,
            "success_rate": successful / total if total > 0 else 0.0,
            "stage_failures": stage_failures,
            "missing_ca_rmsd_count": sum(1 for log in cls._alignment_logs 
                                       if log["success"] and not log["ca_rmsd"])
        }
    
    @classmethod
    def save_alignment_report(cls, output_dir: str, target_pdb: str) -> Optional[str]:
        """Save detailed alignment tracking report."""
        if not cls._alignment_logs:
            return None
            
        summary = cls.get_alignment_summary()
        
        report_data = {
            "summary": summary,
            "detailed_logs": cls._alignment_logs
        }
        
        report_file = get_unique_filename(output_dir, f"{target_pdb}_alignment_tracking", ".json")
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)  # Use default=str to handle numpy types
            return report_file
        except Exception as e:
            log.error(f"Failed to save alignment tracking report: {e}")
            return None
    
    @classmethod
    def clear_logs(cls):
        """Clear alignment logs."""
        cls._alignment_logs.clear()

# ─── Enhanced protein alignment context collection ──────────────────────────────────────
def collect_protein_alignment_context(pdb_id: str, protein_file: str) -> Dict:
    """Collect specific context for protein alignment diagnostic purposes."""
    
    context = {
        "protein_file_status": {
            "path": protein_file,
            "exists": os.path.exists(protein_file) if protein_file else False,
            "file_size": 0,
            "biotite_loadable": False
        },
        "protein_structure": {
            "chain_count": 0,
            "amino_acid_count": 0,
            "ca_atom_count": 0
        },
        "alignment_requirements": {
            "min_ca_atoms_needed": MIN_CA_ATOMS_FOR_ALIGNMENT,
            "min_anchor_residues": MIN_ANCHOR_RESIDUES
        }
    }
    
    if protein_file and os.path.exists(protein_file):
        try:
            context["protein_file_status"]["file_size"] = os.path.getsize(protein_file)
            
            # Test biotite loading
            struct = bsio.load_structure(protein_file)
            context["protein_file_status"]["biotite_loadable"] = len(struct) > 0
            
            # Count chains and residues
            amino_struct = struct[filter_amino_acids(struct)]
            context["protein_structure"]["amino_acid_count"] = len(amino_struct)
            context["protein_structure"]["chain_count"] = len(set(amino_struct.chain_id))
            
            # Count CA atoms
            ca_atoms = amino_struct[amino_struct.atom_name == "CA"]
            context["protein_structure"]["ca_atom_count"] = len(ca_atoms)
            
        except Exception as e:
            context["protein_file_status"]["load_error"] = str(e)
    
    return context

# ─── Error tracking ──────────────────────────────────────────────────────────────────


def validate_sdf_file(sdf_path: str) -> bool:
    """Validate that an SDF file is complete and readable.
    
    Args:
        sdf_path: Path to SDF file to validate
        
    Returns:
        True if file is valid and readable, False otherwise
    """
    if not os.path.exists(sdf_path):
        return False
        
    try:
        # Quick validation - try to read first molecule
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        first_mol = next(suppl, None)
        return first_mol is not None
    except Exception as e:
        log.warning(f"SDF validation failed for {sdf_path}: {e}")
        return False


def write_sdf_atomically(molecules: List[Chem.Mol], output_path: str) -> bool:
    """Write molecules to SDF file atomically to prevent corruption.
    
    Args:
        molecules: List of molecules to write
        output_path: Final output path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Write to temporary file first
        temp_path = f"{output_path}.tmp"
        
        with Chem.SDWriter(temp_path) as writer:
            for mol in molecules:
                if mol is not None:
                    writer.write(mol)
        
        # Validate the temporary file
        if validate_sdf_file(temp_path):
            # Atomic move to final location
            import shutil
            shutil.move(temp_path, output_path)
            return True
        else:
            log.error(f"SDF validation failed for temporary file {temp_path}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
            
    except Exception as e:
        log.error(f"Atomic SDF write failed: {e}")
        # Clean up temp file if it exists
        temp_path = f"{output_path}.tmp"
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False


if __name__ == "__main__":
    import time
    start = time.time()
    main()
    log.debug(f"done in {time.time() - start:.1f}s")

