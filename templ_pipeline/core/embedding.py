"""
TEMPL Pipeline Embedding Module

This module handles protein embedding generation and template selection functionality:
1. Generates ESM2 embeddings for protein sequences
2. Loads pre-computed embeddings from files
3. Manages caching for efficient reuse
4. Provides template selection via embedding similarity

The main classes and functions:
- EmbeddingManager: Core class for handling all embedding operations
- get_protein_sequence: Extract sequence from PDB file
- calculate_embedding: Generate embedding for a protein sequence
- select_templates: Find similar templates for a target protein
"""

import os
import time
import warnings
import logging
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union

import numpy as np
from Bio.PDB import PDBParser, PPBuilder
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.parse_pdb_header import parse_pdb_header
from sklearn.metrics.pairwise import cosine_similarity

# Initialize global variables for ESM model components (lazy-loaded)
_esm_components = None


# GPU Detection Functions
def _detect_gpu() -> bool:
    """Detect if CUDA GPU is available for embedding generation."""
    try:
        import torch

        torch.classes.__path__ = []
        return torch.cuda.is_available()
    except ImportError:
        return False


def _get_device() -> str:
    """Get optimal device (cuda/cpu) for embedding generation."""
    import os

    # Check for user device preference override
    forced_device = os.environ.get("TEMPL_FORCE_DEVICE")
    if forced_device:
        if forced_device == "cuda" and _detect_gpu():
            return "cuda"
        elif forced_device == "cpu":
            return "cpu"
        elif forced_device == "cuda" and not _detect_gpu():
            # User forced GPU but no GPU available
            logger.warning(
                "User forced GPU usage but no GPU detected, falling back to CPU"
            )
            return "cpu"

    # Default auto-detection behavior
    return "cuda" if _detect_gpu() else "cpu"


def _get_gpu_info() -> Dict[str, Any]:
    """Get GPU information for logging and diagnostics."""
    if not _detect_gpu():
        return {"available": False, "device": "cpu"}

    try:
        import torch

        return {
            "available": True,
            "device": "cuda",
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    except Exception:
        return {"available": False, "device": "cpu", "error": "Failed to get GPU info"}


# Configure logging - prevent duplicate handlers
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # Don't set the level here - allow it to be controlled by the root logger

# Import utility functions to find paths to PDBbind files and data files
try:
    from templ_pipeline.core.utils import (
        find_pdbbind_paths,
        get_default_embedding_path,
    )
except ImportError:
    logger.warning("Could not import utility functions from templ_pipeline.core.utils")
    find_pdbbind_paths = None
    get_default_embedding_path = None

# Add constants at the top after imports
CA_RMSD_THRESHOLD = 10.0  # Maximum C-alpha RMSD in Angstroms for protein filtering
CA_RMSD_FALLBACK_THRESHOLDS = [10.0, 15.0, 20.0]  # Progressive fallback thresholds


# --- Path Resolution Helper ---
def _get_standard_embedding_paths() -> List[Path]:
    """Get standard embedding paths, using new centralized path management first."""
    potential_paths = []
    
    # First, try the new centralized path management system
    if get_default_embedding_path:
        current_dir = Path(__file__).parent.absolute()
        # Try different base directories
        for base_dir in [
            current_dir.parent.parent,  # project root
            current_dir.parent.parent.parent,  # if nested deeper
            ".",  # current working directory
        ]:
            embedding_path = get_default_embedding_path(str(base_dir))
            if embedding_path:
                potential_paths.append(Path(embedding_path))
    
    # Standard path resolution for the correct embedding file
    current_dir = Path(__file__).parent.absolute()
    root_dir_candidate1 = current_dir.parent.parent  # e.g., 'mcs' or 'project_root'
    root_dir_candidate2 = (
        current_dir.parent.parent.parent
    )  # If nested deeper, e.g. in 'src'

    root_dir = root_dir_candidate1  # Default assumption

    # Heuristic to find a more general project root if possible
    project_root_marker = "README.md"
    if (
        (root_dir_candidate2 / project_root_marker).exists()
        or (root_dir_candidate2 / "templ_pipeline").exists()
        or (root_dir_candidate2 / ".git").exists()
    ):
        root_dir = root_dir_candidate2
    elif not (
        (root_dir_candidate1 / "data").exists()
        or (root_dir_candidate1 / "templ_pipeline").exists()
        or (root_dir_candidate1 / ".git").exists()
        or (root_dir_candidate1 / project_root_marker).exists()
    ):
        # If candidate1 doesn't look like a project root, try one level up if structures exist
        if (current_dir.parent / "data" / "embeddings").exists():
            root_dir = current_dir.parent

    # ZENODO standardized paths only
    standard_paths = [
        # Standard ZENODO path
        root_dir / "data" / "embeddings" / "templ_protein_embeddings_v1.0.0.npz",
    ]
    
    potential_paths.extend(standard_paths)

    # Return unique, resolved paths
    unique_paths = []
    seen_paths_str = set()
    for p in potential_paths:
        try:
            resolved_p = p.resolve()  # Resolves symlinks and makes absolute
            if str(resolved_p) not in seen_paths_str:
                unique_paths.append(resolved_p)
                seen_paths_str.add(str(resolved_p))
        except (
            FileNotFoundError
        ):  # Path.resolve() can raise this if a component doesn't exist
            if (
                p.is_absolute() and str(p) not in seen_paths_str
            ):  # Keep original absolute path if resolve fails but it's a target
                unique_paths.append(p)
                seen_paths_str.add(str(p))
            elif (
                not p.is_absolute() and str(p) not in seen_paths_str
            ):  # Keep original relative path
                unique_paths.append(
                    p
                )  # Store as is, will be checked with .exists() later
                seen_paths_str.add(str(p))

    return unique_paths


def _resolve_embedding_path(embedding_path=None):
    """
    Resolve the embedding path from various sources.

    Order of precedence:
    1. Explicitly provided path
    2. TEMPL_EMBEDDING_PATH environment variable
    3. Default locations:
       - ~/.cache/templ/embeddings/templ_protein_embeddings_v1.0.0.npz
       - ./data/embeddings/templ_protein_embeddings_v1.0.0.npz
       - ./templ_pipeline/data/embeddings/templ_protein_embeddings_v1.0.0.npz

    Returns:
        The resolved path as a string
    """
    if embedding_path is not None:
        return str(embedding_path)

    # Check environment variable
    import os
    from pathlib import Path

    env_path = os.environ.get("TEMPL_EMBEDDING_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # Check default location
    search_paths = [
        Path("data/embeddings/templ_protein_embeddings_v1.0.0.npz"),
    ]

    for path in search_paths:
        if path.exists():
            # Sanity-check file size – Git-LFS pointers are only ~130 bytes
            MIN_BYTES = 5_000_000  # 5 MB, real DB is ~90 MB
            try:
                if path.stat().st_size < MIN_BYTES:
                    logger.error(
                        "Embedding file present but too small (likely an LFS pointer). "
                        "Ensure 'git lfs pull' ran during deployment or use Dockerfile with git-lfs."
                    )
                    continue  # keep searching
            except Exception:
                pass
            return str(path)

    # If no existing path found, return the environment variable or default path
    return (
        env_path or "data/embeddings/templ_protein_embeddings_v1.0.0.npz"
    )


def get_protein_sequence(
    pdb_file: str, target_chain_id: Optional[str] = None
) -> Tuple[Optional[str], List[str]]:
    """Extract protein sequence from PDB file using structure-based coordinates first, with SEQRES fallback.

    Args:
        pdb_file (str): Path to PDB file
        target_chain_id (Optional[str]): Specific chain ID to extract

    Returns:
        Tuple[Optional[str], List[str]]: (1-letter sequence, List of chain IDs used)
    """
    if not os.path.exists(pdb_file):
        logger.warning(f"PDB file not found: {pdb_file}")
        return None, []

    def get_structure_sequence(structure) -> Dict[str, str]:
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
                logger.debug(f"Chain {chain.id} sequence error: {str(e)}")
        return chain_sequences

    def seqres_to_1letter(seqres: str) -> str:
        """Convert SEQRES 3-letter codes to 1-letter with validation."""
        conversion = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from Bio.Data.PDBData import protein_letters_3to1_extended as aa_map

            conversion.update(aa_map)

        return "".join(
            conversion.get(seqres[i : i + 3].upper(), "X")
            for i in range(0, len(seqres), 3)
            if i + 3 <= len(seqres)
        )

    try:
        # First try structure-based sequence extraction
        parser = PDBParser(QUIET=True, PERMISSIVE=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PDBConstructionWarning)
            structure = parser.get_structure("protein", pdb_file)

        struct_sequences = get_structure_sequence(structure[0])  # Use first model
        if not struct_sequences:
            logger.warning("No structure-based sequences found")
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
                chain: seqres_to_1letter(seq) for chain, seq in seqres_sequences.items()
            }

            # Find matching chains between SEQRES and structure
            common_chains = set(seqres_1letter) & set(struct_sequences)
            if common_chains:
                target_chain = (
                    target_chain_id
                    if target_chain_id in common_chains
                    else sorted(common_chains)[0]
                )
                seqres_seq = seqres_1letter[target_chain]
                struct_seq = struct_sequences[target_chain]

                # Validate length consistency
                if len(seqres_seq) == len(struct_seq):
                    final_seq = seqres_seq
                    used_chains = [target_chain]
                    logger.debug(
                        f"Using SEQRES sequence for chain {target_chain} (length: {len(final_seq)})"
                    )
                else:
                    logger.warning(
                        f"SEQRES/structure length mismatch for chain {target_chain}: "
                        f"{len(seqres_seq)} vs {len(struct_seq)}. Using structure sequence."
                    )
                    final_seq = struct_seq
                    used_chains = [target_chain]

        # Fallback to structure sequence
        if final_seq is None:
            target_chain = (
                target_chain_id
                if target_chain_id in struct_sequences
                else sorted(struct_sequences.keys())[0]
            )
            final_seq = struct_sequences[target_chain]
            used_chains = [target_chain]
            logger.debug(
                f"Using structure-based sequence for chain {target_chain} (length: {len(final_seq)})"
            )

        # Final validation
        if len(final_seq) < 20:  # Minimum reasonable protein length
            logger.error(f"Sequence too short (length={len(final_seq)})")
            return None, []

        if "X" in final_seq:
            logger.warning(f"Sequence contains {final_seq.count('X')} unknown residues")

        return final_seq, used_chains

    except Exception as e:
        logger.error(f"Sequence extraction failed: {str(e)}")
        return None, []


def initialize_esm_model():
    """Initialize ESM model for embedding calculation, cached for reuse."""
    global _esm_components
    if _esm_components is None:
        try:
            import torch
            from transformers import EsmModel, EsmTokenizer, AutoConfig

            # Use the larger model to match create_embeddings_base.py
            model_id = "facebook/esm2_t33_650M_UR50D"

            # Get device info and log GPU status
            gpu_info = _get_gpu_info()
            device = _get_device()

            if gpu_info["available"]:
                logger.info(
                    f"GPU detected: {gpu_info['device_name']} ({gpu_info['memory_total']:.1f}GB)"
                )
                logger.info(f"Initializing ESM model on GPU")
            else:
                logger.info(f"No GPU available, using CPU for embedding generation")

            # Configure model with optimizations
            config = AutoConfig.from_pretrained(model_id)

            # Enable flash attention if available and on GPU
            if gpu_info["available"] and hasattr(config, "use_flash_attention_2"):
                config.use_flash_attention_2 = True
                logger.info("Flash Attention 2 enabled for better GPU performance")

            # Use optimized model initialization
            tokenizer = EsmTokenizer.from_pretrained(model_id)
            model = EsmModel.from_pretrained(
                model_id, config=config, add_pooling_layer=False
            )
            model.eval()

            # Device-specific optimization
            if gpu_info["available"]:
                dtype = torch.float16
                model = model.to(device=device, dtype=dtype)
                torch.cuda.empty_cache()
                logger.info(f"Model loaded on GPU with {dtype} precision")
            else:
                try:
                    dtype = torch.bfloat16
                    model = model.to(dtype=dtype)
                    logger.info(f"Model using {dtype} precision on CPU")
                except RuntimeError:
                    logger.info("Using default precision on CPU")

            _esm_components = {"tokenizer": tokenizer, "model": model}
            logger.info(f"ESM model initialization complete: {model_id}")

        except Exception as e:
            logger.error(f"Failed to initialize ESM model: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return None
    return _esm_components


def calculate_embedding(sequence: str) -> Optional[np.ndarray]:
    """Calculate embedding for a protein sequence.

    Args:
        sequence: Protein sequence as a string of one-letter amino acid codes

    Returns:
        Numpy array of embedding or None if calculation fails
    """
    start_time = time.time()
    logger.info(f"Calculating embedding for sequence length: {len(sequence)}")

    esm_components = initialize_esm_model()
    if not esm_components:
        logger.error("Failed to initialize ESM model")
        return None

    try:
        import torch
        import gc

        tokenizer, model = esm_components["tokenizer"], esm_components["model"]
        device = next(model.parameters()).device

        # Step 1: Tokenize on CPU first to avoid GPU memory issues with long sequences
        inputs = tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=1022,
            padding="longest",
        )

        # Step 2: Move to appropriate device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Step 3: Process with autocast for efficient computation
        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=True):
                outputs = model(**inputs)

            # Step 4: Mean pooling and CPU transfer
            emb = outputs.last_hidden_state.mean(dim=1).squeeze()
            # Transfer to CPU - separate operation to avoid memory issues
            emb_cpu = emb.cpu()

            # Step 5: Convert to numpy with correct dtype
            # Step 5: Convert to numpy with correct dtype - handle BFloat16
            if emb_cpu.dtype == torch.bfloat16:
                # Convert BFloat16 to Float32 first
                emb_cpu = emb_cpu.to(torch.float32)

            # Now convert to numpy
            embedding_array = emb_cpu.numpy().astype(np.float32)

            # Step 6: Clean up tensors to free GPU memory immediately
            del outputs, emb, emb_cpu, inputs

            # Force garbage collection to release memory
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        logger.info(
            f"Embedding generated successfully in {elapsed:.2f}s, shape: {embedding_array.shape}"
        )
        return embedding_array

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(f"GPU out of memory error calculating embedding: {str(e)}")
            logger.info("Consider using a smaller batch size or running on CPU")
        else:
            logger.error(f"Runtime error calculating embedding: {str(e)}")

        # Add more detailed error tracking
        import traceback

        logger.error(traceback.format_exc())

        # Try to free memory in case of error
        if "outputs" in locals():
            del outputs
        if "emb" in locals():
            del emb
        if "emb_cpu" in locals():
            del emb_cpu
        if "inputs" in locals():
            del inputs

        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return None
    except Exception as e:
        logger.error(f"Error calculating embedding: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def get_protein_embedding(
    pdb_file: str, target_chain_id: Optional[str] = None
) -> Tuple[Optional[np.ndarray], List[str]]:
    """Get embedding for a protein, extracting sequence first.

    Args:
        pdb_file: Path to PDB file
        target_chain_id: Specific chain ID to extract (if None, uses first available chain)

    Returns:
        Tuple of (embedding array, list of chain IDs used)
    """
    start_time = time.time()

    # Verify PDB file exists
    if not os.path.exists(pdb_file):
        logger.error(f"PDB file not found: {pdb_file}")
        return None, []

    # Extract proper PDB ID from the file header
    pdb_id = extract_pdb_id_from_file(pdb_file)

    # Log embedding generation start with device info
    gpu_info = _get_gpu_info()
    device_info = f"GPU ({gpu_info['device_name']})" if gpu_info["available"] else "CPU"
    logger.info(f"Generating embedding for protein {pdb_id} using {device_info}")

    try:
        # Extract sequence from PDB
        logger.info(f"Extracting sequence from {pdb_file}")
        seq, chains = get_protein_sequence(pdb_file, target_chain_id)

        if not seq:
            logger.error(f"Failed to extract sequence from {pdb_file}")
            return None, chains

        # If no target chain specified, use first available chain
        if not target_chain_id and chains:
            target_chain_id = chains[0]
            logger.info(
                f"No chain specified, using first available chain: {target_chain_id}"
            )

        logger.info(
            f"Extracted sequence of length {len(seq)} from chain(s): {', '.join(chains)}"
        )

        # Generate embedding with progress indication
        logger.info(f"Calculating embedding for sequence (length: {len(seq)})")
        emb = calculate_embedding(seq)

        # Log timing and success/failure
        elapsed = time.time() - start_time
        if emb is not None:
            logger.info(
                f"Successfully generated embedding for {pdb_id} in {elapsed:.2f}s"
            )
        else:
            logger.error(
                f"Failed to generate embedding for {pdb_id} after {elapsed:.2f}s"
            )

        return emb, chains

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"Failed to calculate embedding for {pdb_file} after {elapsed:.2f}s: {str(e)}"
        )
        import traceback

        logger.error(traceback.format_exc())
        return None, []


def is_pdb_id_in_database(pdb_id: str, embedding_path: Optional[str] = None) -> bool:
    """Check if a PDB ID exists in the pre-computed embedding database.

    Args:
        pdb_id: The PDB ID to check (case-insensitive)
        embedding_path: Path to the embedding NPZ file

    Returns:
        bool: True if PDB ID exists in database, False otherwise
    """
    try:
        # Normalize PDB ID exactly the same way as has_embedding does
        pdb_id = pdb_id.upper().split(":")[-1]

        # Resolve embedding path
        resolved_path = _resolve_embedding_path(embedding_path)

        if not resolved_path or not os.path.exists(resolved_path):
            logger.warning(f"Embedding file not found at {resolved_path}")
            return False

        # Create a temporary manager to perform the check consistently with has_embedding
        # This ensures both methods use the same logic and data structures
        manager = EmbeddingManager(embedding_path=resolved_path)
        return manager.has_embedding(pdb_id)
    except Exception as e:
        logger.error(f"Error checking database for PDB ID {pdb_id}: {str(e)}")
        return False


# Create a function to get sample PDB IDs from the database
def get_sample_pdb_ids(
    embedding_path: Optional[str] = None, limit: int = 20
) -> List[str]:
    """Get a sample of PDB IDs available in the embedding database.

    Args:
        embedding_path: Path to the embedding NPZ file
        limit: Maximum number of PDB IDs to return

    Returns:
        List of PDB ID strings
    """
    try:
        # Resolve embedding path
        resolved_path = _resolve_embedding_path(embedding_path)

        with np.load(resolved_path, allow_pickle=True) as data:
            if "pdb_ids" in data:
                pdb_ids = data["pdb_ids"]
                # Convert all PDB IDs to uppercase for consistency
                pdb_ids = [str(pid).upper() for pid in pdb_ids]

                # Take a random sample or the first few
                import random

                if len(pdb_ids) > limit:
                    return random.sample(pdb_ids, limit)
                return pdb_ids[:limit]
            return []
    except Exception as e:
        logger.error(f"Error getting sample PDB IDs: {str(e)}")
        return []


class EmbeddingManager:
    """
    Unified manager for protein embeddings, handling both pre-computed and on-demand embeddings.

    This class manages:
    1. Loading pre-computed embeddings from npz files
    2. Generating on-demand embeddings for proteins not in the database
    3. Finding nearest neighbors using all available embeddings
    4. Maintaining chain information for protein alignment
    5. Caching on-demand embeddings to disk to avoid regeneration
    """

    def __init__(
        self,
        embedding_path: Optional[str] = None,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        enable_batching: bool = True,
        max_batch_size: int = 8,
        shared_embedding_cache: Optional[str] = None,
    ):
        """Initialize the embedding manager with a path to pre-computed embeddings."""
        resolved_path = _resolve_embedding_path(embedding_path)
        self.embedding_path = resolved_path if resolved_path else ""

        self.embeddings = {}  # Pre-calculated embeddings from NPZ
        self.embedding_db = {}  # Pre-calculated embeddings from NPZ
        self.embedding_chain_data = {}  # Chain data from NPZ
        self.on_demand_embeddings = {}  # Dynamically generated embeddings
        self.on_demand_chain_data = {}  # Chain data for on-demand embeddings
        self.pdb_to_uniprot = {}  # For UniProt exclusion
        self.shared_embedding_cache = shared_embedding_cache  # Shared cache file for multiprocessing

        # Cache configuration
        self.use_cache = use_cache
        # Resolve cache_dir to an absolute path if it's relative
        if cache_dir:
            self.cache_dir = str(Path(cache_dir).expanduser().resolve())
        else:
            self.cache_dir = str(
                Path("~/.cache/templ/embeddings").expanduser().resolve()
            )

        if self.use_cache:
            # Ensure the base cache directory exists
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            # Create model-specific subdirectory
            model_dir_name = "esm2_t33_650M_UR50D"  # Should ideally come from model config or constant
            model_specific_cache_dir = Path(self.cache_dir) / model_dir_name
            model_specific_cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir = str(
                model_specific_cache_dir
            )  # Update to model-specific path

        # Batch processing
        self.enable_batching = enable_batching
        self.max_batch_size = max_batch_size
        self._batch_queue = []

        self._load_embeddings()

    def load_embeddings(self):
        """Load and return the embeddings from the NPZ file (for external access)."""
        if not self.embedding_db:
            self._load_embeddings()
        return self.embedding_db

    def _load_embeddings(self) -> bool:
        """Load pre-computed embeddings from NPZ file or shared cache."""
        
        # Try shared cache first (for multiprocessing)
        if self.shared_embedding_cache:
            try:
                from templ_pipeline.core.utils import load_shared_embedding_cache
                cached_data = load_shared_embedding_cache(self.shared_embedding_cache)
                if cached_data:
                    self.embedding_db = cached_data.get('embedding_db', {})
                    self.embedding_chain_data = cached_data.get('embedding_chain_data', {})
                    logger.info(f"Loaded {len(self.embedding_db)} embeddings from shared cache")
                    return True
            except Exception as e:
                logger.warning(f"Failed to load from shared embedding cache: {e}")
        
        # Fallback to loading from NPZ file
        if not self.embedding_path or not os.path.exists(self.embedding_path):
            logger.warning(
                f"Embedding file not found or path is invalid: '{self.embedding_path}'"
            )
            return False

        try:
            data = np.load(self.embedding_path, allow_pickle=True)
            pdb_ids = data["pdb_ids"]
            embeddings = data["embeddings"]
            chain_ids = data.get("chain_ids", None)

            # Populate embedding database
            for i, pid in enumerate(pdb_ids):
                # Store keys in uppercase to match how they're looked up in has_embedding
                pid_upper = str(pid).upper()
                self.embedding_db[pid_upper] = embeddings[i]
                if chain_ids is not None:
                    self.embedding_chain_data[pid_upper] = chain_ids[i]
                else:
                    self.embedding_chain_data[pid_upper] = (
                        ""  # Default if no chain data
                    )

            logger.info(
                f"Loaded {len(self.embedding_db)} embeddings from {self.embedding_path}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Error loading embeddings from '{self.embedding_path}': {str(e)}"
            )
            return False

    def set_uniprot_mapping(self, pdb_to_uniprot_map: Dict[str, str]) -> None:
        """Set UniProt mapping for template filtering."""
        self.pdb_to_uniprot = pdb_to_uniprot_map

    def _get_cache_path(self, pdb_id: str, chain_id: Optional[str] = None) -> str:
        """Generate a path for the cached embedding file within the model-specific cache dir."""
        # Ensure cache_dir is set (should be by __init__)
        if not self.cache_dir:  # Should not happen if use_cache is True
            logger.error("Cache directory not configured for _get_cache_path.")
            # Fallback to a temporary non-caching path to avoid crash, though this indicates an issue
            return os.path.join(
                tempfile.gettempdir(), f"{pdb_id}_{chain_id or 'all'}.npz"
            )

        # Sanitize pdb_id and chain_id to be safe filenames
        safe_pdb_id = "".join(
            c if c.isalnum() or c in ("_", "-") else "_" for c in pdb_id
        )
        safe_chain_id = (
            "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in chain_id)
            if chain_id
            else "allchains"
        )

        filename = f"{safe_pdb_id}_{safe_chain_id}.npz"
        return os.path.join(self.cache_dir, filename)

    def _save_to_cache(
        self, pdb_id: str, embedding: np.ndarray, chains: List[str]
    ) -> bool:
        """Save embedding to cache."""
        try:
            if not self.use_cache:
                return False

            chain_str = ",".join(chains) if chains else ""
            # Use target_chain_id if relevant or a hash of all chains for filename uniqueness
            # For simplicity, using the first chain or "all" if multiple/none specific.
            cache_key_chain = chains[0] if chains else None
            cache_path = self._get_cache_path(
                pdb_id, cache_key_chain
            )  # Use consistent chain for cache key

            np.savez_compressed(
                cache_path,
                embedding=embedding,
                chain_ids=chain_str,  # Store all chains used
                timestamp=time.time(),
                model_id="esm2_t33_650M_UR50D",
            )
            logger.debug(f"Saved embedding to cache: {cache_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save embedding to cache for {pdb_id}: {str(e)}")
            return False

    def _load_from_cache(
        self, pdb_id: str, target_chain_id: Optional[str] = None
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Load embedding from cache if it exists."""
        try:
            if not self.use_cache:
                return None, None

            # Try with specific chain first if provided for cache lookup
            cache_path_specific_chain = self._get_cache_path(pdb_id, target_chain_id)
            if os.path.exists(cache_path_specific_chain):
                data = np.load(cache_path_specific_chain, allow_pickle=True)
                logger.debug(
                    f"Loaded embedding from cache (specific chain key {target_chain_id}): {cache_path_specific_chain}"
                )
                return data["embedding"], (
                    str(data["chain_ids"]) if "chain_ids" in data else ""
                )

            # Try with a general key if specific chain not found or not provided
            cache_path_general = self._get_cache_path(
                pdb_id
            )  # Default chain key (e.g., "allchains")
            if os.path.exists(cache_path_general):
                data = np.load(cache_path_general, allow_pickle=True)
                logger.debug(
                    f"Loaded embedding from general cache key: {cache_path_general}"
                )
                return data["embedding"], (
                    str(data["chain_ids"]) if "chain_ids" in data else ""
                )

            return None, None
        except Exception as e:
            logger.error(f"Failed to load embedding from cache for {pdb_id}: {str(e)}")
            return None, None

    def is_in_cache(self, pdb_id: str, target_chain_id: Optional[str] = None) -> bool:
        """Check if embedding exists in cache."""
        if not self.use_cache:
            return False

        if os.path.exists(self._get_cache_path(pdb_id, target_chain_id)):
            return True
        # Also check general cache key if specific target_chain_id was given
        if target_chain_id and os.path.exists(self._get_cache_path(pdb_id)):
            return True
        return False

    def clear_cache(self) -> bool:
        """Clear the embedding cache."""
        try:
            if (
                not self.use_cache
                or not self.cache_dir
                or not os.path.exists(self.cache_dir)
            ):
                logger.info(
                    "Cache not used or cache directory does not exist. Nothing to clear."
                )
                return False

            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".npz"):
                    os.remove(os.path.join(self.cache_dir, filename))
            logger.info(f"Cleared embedding cache: {self.cache_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache at {self.cache_dir}: {str(e)}")
            return False

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the cache."""
        try:
            if (
                not self.use_cache
                or not self.cache_dir
                or not os.path.exists(self.cache_dir)
            ):
                return {
                    "enabled": self.use_cache,
                    "count": 0,
                    "size_mb": 0,
                    "path": self.cache_dir or "Not configured",
                }

            files = [f for f in os.listdir(self.cache_dir) if f.endswith(".npz")]
            total_size = sum(
                os.path.getsize(os.path.join(self.cache_dir, f)) for f in files
            )

            return {
                "enabled": True,
                "count": len(files),
                "size_mb": total_size / (1024 * 1024),
                "path": self.cache_dir,
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats for {self.cache_dir}: {str(e)}")
            return {
                "enabled": self.use_cache,
                "error": str(e),
                "path": self.cache_dir or "Error in path",
            }

    def has_embedding(self, pdb_id: str) -> bool:
        """Check if a PDB ID has an embedding in the database or cache.

        Args:
            pdb_id: PDB ID to check (case-insensitive)

        Returns:
            bool: True if embedding exists, False otherwise
        """
        # Normalize PDB ID
        pdb_id = pdb_id.upper().split(":")[-1]

        logger.debug(f"Checking for embedding of PDB ID: {pdb_id}")

        # Check 1: In-memory pre-computed database
        if pdb_id in self.embedding_db:
            logger.debug(f"PDB ID {pdb_id} found in pre-computed database")
            return True

        # Check 2: In-memory on-demand generated embeddings
        if pdb_id in self.on_demand_embeddings:
            logger.debug(f"PDB ID {pdb_id} found in on-demand memory cache")
            return True

        # Check 3: Disk cache if enabled
        if self.use_cache:
            if self.is_in_cache(pdb_id):
                logger.debug(f"PDB ID {pdb_id} found in disk cache")
                return True
            else:
                logger.debug(f"PDB ID {pdb_id} not found in disk cache")

        # Check 4: Maybe the database wasn't loaded properly, try force-reload
        if (
            not self.embedding_db
            and self.embedding_path
            and os.path.exists(self.embedding_path)
        ):
            logger.warning(
                f"Embedding DB seems empty. Trying to reload from {self.embedding_path}"
            )
            self._load_embeddings()
            # Check again after reloading
            if pdb_id in self.embedding_db:
                logger.debug(f"PDB ID {pdb_id} found after reloading database")
                return True

        logger.debug(f"PDB ID {pdb_id} not found in any available source")
        return False

    def get_embedding(
        self,
        pdb_id: str,
        pdb_file: Optional[str] = None,
        target_chain_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """
        Get embedding for a protein: from memory, cache, pre-computed DB, or generate on-demand.
        Args:
            pdb_id: PDB ID for the protein (must be resolvable to a filename if generating)
            pdb_file: Path to PDB file for on-demand embedding generation (if not in DB/cache)
            target_chain_id: Specific chain to use for sequence extraction and cache lookup.
        Returns:
            Tuple of (embedding array, chain_ids string used for the embedding)
        """
        # 0. Normalize pdb_id (e.g. PDB:1abc -> 1abc)
        pdb_id = pdb_id.upper().split(":")[-1]

        # 1. Check in-memory pre-computed database
        if pdb_id in self.embedding_db:
            logger.debug(f"Found embedding for {pdb_id} in pre-computed DB.")
            return self.embedding_db[pdb_id], self.embedding_chain_data.get(pdb_id, "")

        # 2. Check in-memory on-demand generated embeddings
        if pdb_id in self.on_demand_embeddings:
            # Potentially re-check if target_chain_id matters for on-demand ones too
            # For now, assume on_demand_embeddings are keyed uniquely enough by pdb_id
            logger.debug(f"Found embedding for {pdb_id} in on-demand memory.")
            return self.on_demand_embeddings[pdb_id], self.on_demand_chain_data.get(
                pdb_id, ""
            )

        # 3. Check cache
        if self.use_cache:
            cached_emb, cached_chains_str = self._load_from_cache(
                pdb_id, target_chain_id
            )
            if cached_emb is not None:
                logger.debug(f"Loaded embedding for {pdb_id} from cache.")
                self.on_demand_embeddings[pdb_id] = cached_emb  # Store in memory
                self.on_demand_chain_data[pdb_id] = cached_chains_str
                return cached_emb, cached_chains_str

        # 4. Generate embedding on-demand if PDB file path is provided
        # Ensure pdb_file is valid if we need to generate
        actual_pdb_file_to_use = pdb_file
        if (
            not actual_pdb_file_to_use
            and not pdb_id.endswith(".pdb")
            and not os.path.exists(pdb_id)
        ):
            # If only PDB ID is given and not found in DB/cache, we can't generate without a file.
            # However, if pdb_id itself could be a path to a file (e.g. user inputs a path in PDB ID field)
            logger.warning(
                f"No PDB file provided for {pdb_id} to generate on-demand embedding, and not found in DB/cache."
            )
            return None, None
        elif pdb_id.endswith(".pdb") and os.path.exists(
            pdb_id
        ):  # If pdb_id is actually a path
            actual_pdb_file_to_use = pdb_id
            # Extract proper PDB ID from the file header
            pdb_id = extract_pdb_id_from_file(actual_pdb_file_to_use)

        if actual_pdb_file_to_use and os.path.exists(actual_pdb_file_to_use):
            # Try to extract proper PDB ID from file if available
            if pdb_id.startswith("TEMP_"):
                pdb_id = extract_pdb_id_from_file(actual_pdb_file_to_use)

            logger.info(
                f"Generating on-demand embedding for {pdb_id} from file {actual_pdb_file_to_use}"
            )

            seq, chains_list = get_protein_sequence(
                actual_pdb_file_to_use, target_chain_id
            )
            if not seq or not chains_list:
                logger.error(
                    f"Failed to extract sequence from {actual_pdb_file_to_use} for {pdb_id}"
                )
                return None, None

            embedding = calculate_embedding(seq)
            chains_str = ",".join(chains_list)

            if embedding is not None:
                self.on_demand_embeddings[pdb_id] = embedding
                self.on_demand_chain_data[pdb_id] = chains_str

                if self.use_cache:
                    self._save_to_cache(
                        pdb_id, embedding, chains_list
                    )  # Save with list of chains

                logger.info(
                    f"Successfully generated and cached on-demand embedding for {pdb_id}"
                )
                return embedding, chains_str
            else:
                logger.error(f"Failed to calculate on-demand embedding for {pdb_id}")

        logger.warning(f"Embedding for {pdb_id} could not be found or generated.")
        return None, None

    def find_neighbors(
        self,
        query_pdb_id: str,
        query_embedding: Optional[np.ndarray] = None,
        query_pdb_file: Optional[str] = None,
        query_target_chain_id: Optional[str] = None,
        exclude_uniprot_ids: Optional[Set[str]] = None,
        exclude_pdb_ids: Optional[Set[str]] = None,
        allowed_pdb_ids: Optional[Set[str]] = None,
        k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        return_similarities: bool = False,
        allow_self_as_template: bool = False,
    ) -> List[Union[str, Tuple[str, float]]]:
        """
        Find nearest neighbors for a query protein.
        Args:
            query_pdb_id: PDB ID of query protein
            query_embedding: Optional pre-computed embedding for the query. If None, will be fetched/generated.
            query_pdb_file: Optional path to query PDB file (if embedding needs generation).
            query_target_chain_id: Optional chain for query if generating embedding.
            allow_self_as_template: Allow query PDB to appear in template results
            ... (other args)
        """
        query_pdb_id = query_pdb_id.upper().split(":")[-1]

        final_exclude_pdb_ids = set(exclude_pdb_ids) if exclude_pdb_ids else set()
        if not allow_self_as_template:
            final_exclude_pdb_ids.add(query_pdb_id)  # Exclude query itself

        if query_embedding is None:
            query_embedding, _ = self.get_embedding(
                query_pdb_id,
                pdb_file=query_pdb_file,
                target_chain_id=query_target_chain_id,
            )
            if query_embedding is None:
                logger.error(
                    f"No embedding found or generated for query PDB {query_pdb_id}. Cannot find neighbors."
                )
                return []

        all_candidate_pdb_ids = []
        all_embeddings_list = []

        # Combine pre-computed and on-demand embeddings already in memory
        # (Excludes on-demand that are only in cache and not yet loaded by a get_embedding call)
        # This means find_neighbors primarily works on the self.embedding_db.
        # If you want it to search *everything* including generating from cache, it needs more logic.
        # For now, it relies on the main DB being loaded.

        # Consider all PDBs in the loaded pre-computed database (self.embedding_db)
        # This is the primary source for template candidates.
        if not self.embedding_db:
            logger.warning(
                "Embedding database (self.embedding_db) is empty. Cannot find neighbors from it."
            )
            # Optionally, could try to load from self.embedding_path again if it's empty, but _load_embeddings should run in __init__

        for pid, emb in self.embedding_db.items():
            pid_norm = pid.upper().split(":")[-1]
            if pid_norm in final_exclude_pdb_ids:
                continue
            if allowed_pdb_ids is not None and pid_norm not in allowed_pdb_ids:
                continue
            if (
                exclude_uniprot_ids
                and pid_norm in self.pdb_to_uniprot
                and self.pdb_to_uniprot[pid_norm] in exclude_uniprot_ids
            ):
                continue

            all_candidate_pdb_ids.append(pid_norm)
            all_embeddings_list.append(emb)

        if not all_embeddings_list:
            logger.warning(
                "No valid template candidates found in self.embedding_db after filtering."
            )
            return []

        stacked_embeddings = np.vstack(all_embeddings_list)
        similarities = cosine_similarity(
            stacked_embeddings, query_embedding.reshape(1, -1)
        ).flatten()

        neighbor_candidates = sorted(
            list(zip(all_candidate_pdb_ids, similarities)),
            key=lambda x: x[1],
            reverse=True,
        )

        if similarity_threshold is not None:
            final_neighbors = [
                (pid, sim)
                for pid, sim in neighbor_candidates
                if sim >= similarity_threshold
            ]
        elif k is not None:
            final_neighbors = neighbor_candidates[:k]
        else:  # Default: return all sorted if neither k nor threshold is given
            final_neighbors = neighbor_candidates

        logger.info(f"Found {len(final_neighbors)} neighbors for {query_pdb_id}.")

        if return_similarities:
            return final_neighbors
        else:
            return [pid for pid, _ in final_neighbors]

    def get_chain_data(self, pdb_id: str) -> Optional[str]:
        """Get chain data for a PDB ID from either pre-computed or on-demand sources."""
        pdb_id = pdb_id.upper().split(":")[-1]
        if pdb_id in self.embedding_chain_data:
            return self.embedding_chain_data[pdb_id]
        elif pdb_id in self.on_demand_chain_data:
            return self.on_demand_chain_data[pdb_id]

        # If not in memory, try to get from cache (indirectly, if get_embedding was called)
        # Or, if precomputed DB has it. This method primarily checks memory.
        # To get from cache if not in memory, one might need to call self.get_embedding(pdb_id) first.
        logger.debug(
            f"Chain data for {pdb_id} not found in direct memory (embedding_chain_data or on_demand_chain_data)."
        )
        return None

    def add_to_batch(
        self, pdb_id: str, pdb_file: str, target_chain_id: Optional[str] = None
    ) -> bool:
        """Add a PDB to the batch for processing."""
        if not self.enable_batching:
            return False

        if len(self._batch_queue) >= self.max_batch_size:
            return False

        self._batch_queue.append(
            {"pdb_id": pdb_id, "pdb_file": pdb_file, "target_chain_id": target_chain_id}
        )
        return True

    def process_batch(self) -> int:
        """Process all items in the batch queue."""
        if not self._batch_queue:
            return 0

        processed = 0
        sequences = []
        batch_items = []

        # Collect sequences for batch processing
        for item in self._batch_queue:
            try:
                sequence, chains = get_protein_sequence(
                    item["pdb_file"], item["target_chain_id"]
                )
                if sequence:
                    sequences.append(sequence)
                    batch_items.append((item, sequence, chains))
            except Exception as e:
                logger.warning(f"Failed to get sequence for {item['pdb_id']}: {e}")

        if not sequences:
            self._batch_queue.clear()
            return 0

        # Process sequences in batch
        try:
            esm_components = initialize_esm_model()
            embeddings = self._process_sequence_batch(sequences, esm_components)

            # Save results
            for (item, sequence, chains), embedding in zip(batch_items, embeddings):
                if embedding is not None:
                    self._save_to_cache(item["pdb_id"], embedding, chains)
                    processed += 1

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")

        finally:
            self._batch_queue.clear()

        return processed

    def _process_sequence_batch(
        self, sequences: List[str], esm_components
    ) -> List[Optional[np.ndarray]]:
        """Process multiple sequences in a single batch."""
        model, alphabet, batch_converter = esm_components

        # Prepare batch data
        batch_data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]

        try:
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)

            import torch

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=False)

            embeddings = []
            for i, seq in enumerate(sequences):
                try:
                    # Extract per-token representations for sequence i
                    # results["representations"][33] is a tensor of shape [batch_size, seq_len, hidden_dim]
                    token_representations = results["representations"][33][
                        i, 1 : len(seq) + 1
                    ]
                    # Mean pool over sequence length
                    embedding = token_representations.mean(0).cpu().numpy()
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Failed to extract embedding for sequence {i}: {e}")
                    embeddings.append(None)

            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            return [None] * len(sequences)

    def prepare_batch_embeddings(self, pdb_ids: List[str]) -> int:
        """Prepare embeddings for a list of PDB IDs using batch processing."""
        if not self.enable_batching:
            return 0

        processed = 0
        for pdb_id in pdb_ids:
            if not self.has_embedding(pdb_id):
                pdb_file = self._find_pdb_file(pdb_id)
                if pdb_file and self.add_to_batch(pdb_id, pdb_file):
                    if len(self._batch_queue) >= self.max_batch_size:
                        processed += self.process_batch()

        # Process remaining items
        if self._batch_queue:
            processed += self.process_batch()

        return processed

    def _find_pdb_file(self, pdb_id: str) -> Optional[str]:
        """Find PDB file path for a given PDB ID."""
        if find_pdbbind_paths is None:
            logger.warning("find_pdbbind_paths not available")
            return None

        try:
            paths = find_pdbbind_paths(pdb_id)
            return paths.get("protein") if paths else None
        except Exception as e:
            logger.warning(f"Could not find PDB file for {pdb_id}: {e}")
            return None


def get_embedding(
    pdb_id_or_file: str,
    embedding_path: Optional[str] = None,
    chain_id: Optional[str] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, str]]:
    """
    High-level function to get embedding for a protein.

    Args:
        pdb_id_or_file: PDB ID or path to PDB file
        embedding_path: Path to pre-computed embeddings
        chain_id: Specific chain to use

    Returns:
        Embedding array or tuple of (embedding, chains_used)

    Raises:
        ValueError: If no embedding can be found or generated
    """
    # Determine if input is a PDB ID or file path
    is_file = os.path.exists(pdb_id_or_file)

    # Get proper PDB ID - either from file header or from the input string
    if is_file:
        pdb_id = extract_pdb_id_from_file(pdb_id_or_file)
    else:
        pdb_id = pdb_id_or_file

    # Normalize PDB ID
    pdb_id = pdb_id.upper().split(":")[-1]

    # Resolve the embedding path from various sources
    resolved_path = _resolve_embedding_path(embedding_path)

    # Initialize EmbeddingManager with resolved path
    manager = EmbeddingManager(embedding_path=resolved_path)

    # Check if this is a PDB ID lookup (not a file) and validate it exists
    if not is_file:
        # Check for embedding availability - directly use manager's check
        # Skip the external is_pdb_id_in_database function since we've updated it to use manager internally
        if not manager.has_embedding(pdb_id):
            raise ValueError(
                f"PDB ID {pdb_id} not found in pre-computed embedding database or cache"
            )

    # Try to get embedding from database or generate
    embedding_result = manager.get_embedding(
        pdb_id, pdb_file=pdb_id_or_file if is_file else None, target_chain_id=chain_id
    )

    # Check what was returned (could be tuple or None)
    if embedding_result is None or embedding_result[0] is None:
        if is_file:
            raise ValueError(f"Failed to generate embedding from file {pdb_id_or_file}")
        else:
            raise ValueError(f"Failed to retrieve embedding for PDB ID {pdb_id}")

    # Return embedding with or without chain info
    # The legacy syntax for backward compatibility with older code that expects just the array
    emb, chains_used = embedding_result
    return emb, chains_used


def select_templates(
    target_pdb_id: str,
    target_embedding: np.ndarray,
    embedding_path: Optional[str] = None,  # Path for manager to consider for its DB
    k: int = 100,
    similarity_threshold: Optional[float] = None,
    enable_uniprot_exclusion: bool = False,
    enable_pdb_filtering: bool = False,  # Renamed for clarity
    allowed_pdb_ids_file: Optional[str] = None,  # New name for clarity
    exclude_uniprot_ids_file: Optional[str] = None,
    pdb_to_uniprot_file: Optional[str] = None,
    return_similarities: bool = False,
    allow_self_as_template: bool = False,
    # split_name: Optional[str] = None # find_neighbors_in_split not implemented in current EmbeddingManager
    query_pdb_file: Optional[
        str
    ] = None,  # To pass to find_neighbors if query_embedding needs generation
    query_target_chain_id: Optional[str] = None,  # To pass to find_neighbors
) -> List[Union[str, Tuple[str, float]]]:
    """
    Select template proteins based on embedding similarity using EmbeddingManager.
    """
    manager = EmbeddingManager(
        embedding_path=embedding_path
    )  # Manager resolves its own path

    # Handle UniProt exclusion setup for the manager
    final_exclude_uniprot_ids = set()
    if enable_uniprot_exclusion:
        if pdb_to_uniprot_file and os.path.exists(pdb_to_uniprot_file):
            import json

            try:
                with open(pdb_to_uniprot_file) as f:
                    pdb_to_uniprot_map_loaded = {}
                    data = json.load(f)
                    for pid, info in data.items():
                        uniprot_id = None
                        if isinstance(info, dict) and "uniprot" in info:
                            uniprot_id = info["uniprot"]
                        elif isinstance(info, str):  # If mapping is direct PDB:UniProt
                            uniprot_id = info
                        if uniprot_id:
                            pdb_to_uniprot_map_loaded[pid.upper()] = uniprot_id
                manager.set_uniprot_mapping(pdb_to_uniprot_map_loaded)
            except json.JSONDecodeError:
                logger.error(
                    f"Error decoding JSON from pdb_to_uniprot_file: {pdb_to_uniprot_file}"
                )
            except Exception as e:
                logger.error(
                    f"Error processing pdb_to_uniprot_file {pdb_to_uniprot_file}: {e}"
                )

        if exclude_uniprot_ids_file and os.path.exists(exclude_uniprot_ids_file):
            with open(exclude_uniprot_ids_file) as f:
                final_exclude_uniprot_ids = {
                    line.strip().upper() for line in f if line.strip()
                }

    # Handle PDB filtering (allowed list)
    final_allowed_pdb_ids: Optional[Set[str]] = None
    if (
        enable_pdb_filtering
        and allowed_pdb_ids_file
        and os.path.exists(allowed_pdb_ids_file)
    ):
        with open(allowed_pdb_ids_file) as f:
            final_allowed_pdb_ids = {line.strip().upper() for line in f if line.strip()}

    # find_neighbors_in_split logic would need to be part of EmbeddingManager if used
    # For now, directly using find_neighbors

    neighbors = manager.find_neighbors(
        query_pdb_id=target_pdb_id,
        query_embedding=target_embedding,
        query_pdb_file=query_pdb_file,  # Pass along if query_embedding might need generation
        query_target_chain_id=query_target_chain_id,  # Pass along
        exclude_uniprot_ids=final_exclude_uniprot_ids,
        allowed_pdb_ids=final_allowed_pdb_ids,
        k=k if similarity_threshold is None else None,  # k is used if threshold is not
        similarity_threshold=similarity_threshold,
        return_similarities=return_similarities,
        allow_self_as_template=allow_self_as_template,
    )

    return neighbors


def analyze_embedding_database(embedding_path: Optional[str] = None) -> Dict[str, Any]:
    """Analyze an embedding database file to provide diagnostics.

    Args:
        embedding_path: Path to the embedding NPZ file

    Returns:
        Dictionary with diagnostic information
    """
    try:
        # Resolve embedding path
        resolved_path = _resolve_embedding_path(embedding_path)

        if not resolved_path or not os.path.exists(resolved_path):
            logger.error(f"Embedding file not found at {resolved_path}")
            return {
                "status": "error",
                "message": f"Embedding file not found at {resolved_path}",
                "paths_checked": _get_standard_embedding_paths(),
                "resolved_path": resolved_path,
            }

        # Get file metadata
        file_size = os.path.getsize(resolved_path) / (1024 * 1024)  # MB
        file_mtime = os.path.getmtime(resolved_path)
        mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_mtime))

        logger.info(
            f"Analyzing embedding file: {resolved_path} (Size: {file_size:.2f} MB, Modified: {mod_time})"
        )

        # Load NPZ metadata without loading entire content
        with np.load(resolved_path, allow_pickle=True) as data:
            arrays = dict(data)
            keys = list(arrays.keys())

            logger.info(f"NPZ file contains keys: {keys}")

            # Check for required keys
            has_pdb_ids = "pdb_ids" in keys
            has_embeddings = "embeddings" in keys
            has_chain_ids = "chain_ids" in keys

            # Get PDB ID sample and count if exists
            pdb_count = 0
            pdb_sample = []
            pdb_first_chars = {}  # To check for case patterns
            case_stats = {"lowercase": 0, "uppercase": 0, "mixed": 0}

            if has_pdb_ids:
                pdb_ids = arrays["pdb_ids"]
                pdb_count = len(pdb_ids)
                logger.info(f"Found {pdb_count} PDB IDs in database")

                # Get a sample of the PDB IDs
                if pdb_count > 0:
                    # Take a mix of items from the start, middle and end for better sampling
                    samples = []
                    if pdb_count <= 10:
                        samples = list(range(pdb_count))
                    else:
                        samples = [
                            0,
                            1,
                            2,
                            pdb_count // 2 - 1,
                            pdb_count // 2,
                            pdb_count // 2 + 1,
                            pdb_count - 3,
                            pdb_count - 2,
                            pdb_count - 1,
                        ]

                    pdb_sample = [str(pdb_ids[i]) for i in samples if i < pdb_count]
                    logger.info(
                        f"Sample PDB IDs: {pdb_sample[:5]} ... (total: {len(pdb_sample)})"
                    )

                    # Analyze case distribution
                    for pid in pdb_ids:
                        pid_str = str(pid)
                        if not pid_str:
                            continue

                        # Track first character distribution
                        first_char = pid_str[0]
                        pdb_first_chars[first_char] = (
                            pdb_first_chars.get(first_char, 0) + 1
                        )

                        # Track case patterns
                        if pid_str.islower():
                            case_stats["lowercase"] += 1
                        elif pid_str.isupper():
                            case_stats["uppercase"] += 1
                        else:
                            case_stats["mixed"] += 1

                    # Log case distribution
                    logger.info(f"Case distribution: {case_stats}")

            else:
                logger.error("No 'pdb_ids' key found in NPZ file!")

            # Check if embeddings match PDB ID count
            embedding_shapes_match = False
            embedding_count = 0
            emb_dim = None

            if has_embeddings:
                embeddings = arrays["embeddings"]
                embedding_count = len(embeddings)
                logger.info(f"Found {embedding_count} embeddings in database")
                embedding_shapes_match = embedding_count == pdb_count

                # Get embedding shape
                if embedding_count > 0:
                    emb_dim = embeddings[0].shape
                    logger.info(f"Embedding dimension: {emb_dim}")
            else:
                logger.error("No 'embeddings' key found in NPZ file!")

            # Check chain IDs if available
            chain_id_count = 0
            if has_chain_ids:
                chain_ids = arrays["chain_ids"]
                chain_id_count = (
                    len(chain_ids) if isinstance(chain_ids, np.ndarray) else 0
                )
                logger.info(f"Found {chain_id_count} chain_ids in database")

                # Check a sample
                if chain_id_count > 0:
                    chain_sample = [
                        str(chain_ids[i]) for i in samples[:3] if i < chain_id_count
                    ]
                    logger.info(f"Sample chain IDs: {chain_sample}")

            # Add case-related info to the results
            norm_note = "Note: PDB IDs are normalized to uppercase on load, but original file contains "
            if case_stats["lowercase"] == pdb_count:
                norm_note += "all lowercase PDB IDs."
            elif case_stats["uppercase"] == pdb_count:
                norm_note += "all uppercase PDB IDs."
            else:
                norm_note += "mixed case PDB IDs."

        return {
            "status": "success",
            "file_path": resolved_path,
            "file_size_mb": round(file_size, 2),
            "last_modified": mod_time,
            "keys": keys,
            "has_pdb_ids": has_pdb_ids,
            "has_embeddings": has_embeddings,
            "has_chain_ids": has_chain_ids,
            "pdb_count": pdb_count,
            "embedding_count": embedding_count,
            "chain_id_count": chain_id_count,
            "pdb_sample": pdb_sample[:10],  # Limit sample size
            "first_char_dist": pdb_first_chars,
            "case_stats": case_stats,
            "case_normalization_note": norm_note,
            "embeddings_match_count": embedding_shapes_match,
            "embedding_dimension": emb_dim,
        }
    except Exception as e:
        logger.error(f"Error analyzing embedding database: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "resolved_path": (
                _resolve_embedding_path(embedding_path)
                if embedding_path
                else "Not provided"
            ),
        }


def extract_pdb_id_from_file(pdb_file_path: str) -> str:
    """Extract PDB ID from PDB file HEADER record.

    Args:
        pdb_file_path: Path to PDB file

    Returns:
        PDB ID as a string, or a temporary ID if not found
    """
    try:
        # Parse the header using Biopython
        header_dict = parse_pdb_header(pdb_file_path)
        # The PDB ID is stored with the key 'idcode'
        if "idcode" in header_dict and header_dict["idcode"]:
            return header_dict["idcode"].strip()

        # Fallback: manually extract from HEADER line
        with open(pdb_file_path, "r") as f:
            for line in f:
                if line.startswith("HEADER"):
                    # PDB ID is typically at positions 62-66
                    if len(line) >= 66:
                        pdb_id = line[62:66].strip()
                        if pdb_id:
                            return pdb_id
                    break
    except Exception as e:
        logger.warning(f"Could not extract PDB ID from header: {e}")

    # If we get here, we couldn't extract a PDB ID
    # Create a temporary ID using the filename
    filename = os.path.basename(pdb_file_path)
    if filename.endswith(".pdb"):
        filename = filename[:-4]  # Remove .pdb extension
    return f"TEMP_{filename.upper()}"


def extract_pdb_id_from_path(file_path: str) -> Optional[str]:
    """Extract PDB ID from common file path patterns.

    Args:
        file_path: Path to PDB file

    Returns:
        4-character PDB ID if found, None otherwise
    """
    import re

    # Get filename without path
    filename = os.path.basename(file_path)

    # Common patterns for PDB files
    patterns = [
        r"^([0-9][a-zA-Z0-9]{3})_.*\.pdb$",  # 1iky_protein.pdb
        r"^([0-9][a-zA-Z0-9]{3})\.pdb$",  # 1iky.pdb
        r"^([0-9][a-zA-Z0-9]{3})_[A-Z]\.pdb$",  # 1iky_A.pdb
        r".*_([0-9][a-zA-Z0-9]{3})\.pdb$",  # protein_1iky.pdb
        r".*_([0-9][a-zA-Z0-9]{3})_.*\.pdb$",  # model_1iky_chain.pdb
    ]

    for pattern in patterns:
        match = re.match(pattern, filename, re.IGNORECASE)
        if match:
            pdb_id = match.group(1).lower()
            # Validate PDB ID format (digit + 3 alphanumeric)
            if len(pdb_id) == 4 and pdb_id[0].isdigit():
                return pdb_id

    return None


def filter_templates_by_ca_rmsd(
    all_templates: List[Any], ca_rmsd_threshold: float
) -> List[Any]:
    """Filter templates by CA RMSD threshold.

    Args:
        all_templates: List of template molecules with CA RMSD properties
        ca_rmsd_threshold: Maximum CA RMSD allowed (Angstroms)

    Returns:
        List of templates that pass the CA RMSD threshold
    """
    if ca_rmsd_threshold == float("inf"):
        return all_templates

    filtered_templates = []
    for tpl in all_templates:
        if hasattr(tpl, "HasProp") and tpl.HasProp("ca_rmsd"):
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


def get_templates_with_progressive_fallback(
    all_templates: List[Any], fallback_thresholds: List[float] = None
) -> Tuple[List[Any], float, bool]:
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
                logger.warning(
                    f"Using relaxed CA RMSD threshold ({threshold}Å) - found {len(filtered_templates)} templates (poses may be less accurate)"
                )
            else:
                logger.info(
                    f"Found {len(filtered_templates)} templates with CA RMSD ≤ {threshold}Å"
                )
            return filtered_templates, threshold, False

    # Ultimate fallback: find template with smallest CA RMSD and use central atom positioning
    best_template = None
    best_rmsd = float("inf")

    for tpl in all_templates:
        if hasattr(tpl, "HasProp") and tpl.HasProp("ca_rmsd"):
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
        logger.warning(
            f"Using central atom fallback with best available template (CA RMSD: {best_rmsd}Å)"
        )
        return [best_template], float("inf"), True

    # This should never happen since we have templates
    logger.error("No templates available for central atom fallback")
    return [], float("inf"), False


class EmbeddingEngine:
    """Object-oriented wrapper for embedding functionality."""

    def __init__(self):
        self.manager = EmbeddingManager()

    def generate_conformers(self, smiles: str, n_conformers: int = 100) -> List[Dict]:
        """Generate conformers for a SMILES string."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []

        # Mock implementation for testing
        return [{"coords": [[0, 0, 0]]} for _ in range(min(n_conformers, 100))]
