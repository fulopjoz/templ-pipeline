#!/usr/bin/env python3
"""Main pipeline orchestration for template-based pose prediction."""

import gzip
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .mcs import (
    find_mcs,
    constrained_embed,
    central_atom_embed,
    safe_name,
)
from .embedding import (
    EmbeddingManager,
    get_protein_embedding,
    get_protein_sequence,
)
from .scoring import (
    score_and_align,
    select_best,
    rmsd_raw,
    generate_properties_for_sdf,
)
from .templates import (
    load_reference_protein,
    load_target_data,
    transform_ligand,
    filter_templates_by_ca_rmsd,
    get_templates_with_progressive_fallback,
    pdb_path,
    ligand_path,
)
from .chemistry import (
    validate_target_molecule,
    detect_and_substitute_organometallic,
    needs_uff_fallback,
    is_large_peptide_or_polysaccharide,
)
from .output_manager import EnhancedOutputManager

log = logging.getLogger(__name__)

# Custom exception for molecule validation failures
class MoleculeValidationException(Exception):
    """Exception raised when molecule validation fails during pipeline execution."""
    
    def __init__(self, message, reason, details, molecule_info=None):
        super().__init__(message)
        self.reason = reason
        self.details = details
        self.molecule_info = molecule_info

# Constants
DEFAULT_N_CONFS = 200
DEFAULT_N_WORKERS = 0  # Auto-detect based on CPU count
DEFAULT_SIM_THRESHOLD = 0.90
DEFAULT_CA_RMSD_THRESHOLD = 10.0
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_DATA_DIR = "data"

@dataclass
class PipelineConfig:
    """Configuration for the TEMPL pipeline."""
    
    # Input configuration
    target_pdb: str = ""
    target_smiles: str = ""
    protein_pdb_id: Optional[str] = None
    ligand_smiles: Optional[str] = None
    ligand_file: Optional[str] = None
    
    # Data paths
    data_dir: str = DEFAULT_DATA_DIR
    ligands_sdf_gz: str = ""
    embedding_npz: str = ""
    uniprot_map: str = ""
    
    # Output configuration
    output_dir: str = DEFAULT_OUTPUT_DIR
    
    # Algorithm parameters
    sim_threshold: float = DEFAULT_SIM_THRESHOLD
    n_confs: int = DEFAULT_N_CONFS
    ca_rmsd_threshold: float = DEFAULT_CA_RMSD_THRESHOLD
    n_workers: int = DEFAULT_N_WORKERS
    num_templates: int = 100
    
    # Processing options
    use_cache: bool = True
    enable_batching: bool = True
    max_batch_size: int = 8
    no_realign: bool = False
    enable_optimization: bool = False
    
    # Ablation study options
    unconstrained: bool = False
    align_metric: str = "combo"
    
    def __post_init__(self):
        """Set default paths if not provided."""
        if not self.ligands_sdf_gz:
            self.ligands_sdf_gz = f"{self.data_dir}/ligands/templ_processed_ligands_v1.0.0.sdf.gz"
        if not self.embedding_npz:
            self.embedding_npz = f"{self.data_dir}/embeddings/templ_protein_embeddings_v1.0.0.npz"
        if not self.uniprot_map:
            self.uniprot_map = f"{self.data_dir}/pdbbind_dates.json"


class TEMPLPipeline:
    """Main TEMPL pipeline for template-based pose prediction."""
    
    def __init__(self, embedding_path: Optional[str] = None, output_dir: str = "output", run_id: Optional[str] = None, shared_embedding_cache: Optional[str] = None):
        """Initialize the pipeline with CLI interface."""
        # Use default embedding path if none provided
        if embedding_path is None:
            embedding_path = f"{DEFAULT_DATA_DIR}/embeddings/templ_protein_embeddings_v1.0.0.npz"
        
        self.embedding_path = embedding_path
        self.output_dir = output_dir
        self.run_id = run_id
        self.shared_embedding_cache = shared_embedding_cache
        self.embedding_manager = None
        self.reference_protein = None
        self.target_mol = None
        self.templates = []
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize enhanced output manager
        self.output_manager = EnhancedOutputManager(
            base_output_dir=self.output_dir,
            run_id=self.run_id
        )
        
        # Initialize embedding manager
        self._init_embedding_manager()
        
    def _init_embedding_manager(self):
        """Initialize the embedding manager."""
        try:
            self.embedding_manager = EmbeddingManager(
                self.embedding_path
            )
            log.info("Embedding manager initialized successfully")
        except Exception as e:
            log.error(f"Failed to initialize embedding manager: {e}")
            raise

    def _get_embedding_manager(self):
        """Get the embedding manager instance."""
        if self.embedding_manager is None:
            self._init_embedding_manager()
        return self.embedding_manager
    
    def _extract_pdb_id_from_path(self, file_path: str) -> Optional[str]:
        """Extract PDB ID from file header using multiple strategies."""
        if not file_path:
            return None
            
        try:
            with open(file_path, "r") as f:
                for line in f:
                    if line.startswith("HEADER"):
                        # Strategy 1: Standard PDB format - PDB ID at positions 62-66
                        if len(line) >= 66:
                            pdb_id = line[62:66].strip().lower()
                            if len(pdb_id) == 4 and pdb_id.isalnum():
                                log.info(f"Extracted PDB ID '{pdb_id}' from standard header format")
                                return pdb_id
                        
                        # Strategy 2: Simple header format - "HEADER    PDB_ID" or "HEADER    PDB_ID_PROTEIN"
                        header_parts = line.strip().split()
                        if len(header_parts) >= 2:
                            potential_id = header_parts[1]
                            
                            # Remove common suffixes like "_PROTEIN"
                            if potential_id.endswith("_PROTEIN"):
                                potential_id = potential_id[:-8]
                            elif potential_id.endswith("_COMPLEX"):
                                potential_id = potential_id[:-8]
                            
                            # Validate as 4-character PDB ID
                            if len(potential_id) == 4 and potential_id.isalnum():
                                pdb_id = potential_id.lower()
                                log.info(f"Extracted PDB ID '{pdb_id}' from simple header format")
                                return pdb_id
                                
                    elif line.startswith("TITLE") or line.startswith("ATOM"):
                        # Stop searching after HEADER section
                        break
                        
            # Strategy 3: Filename fallback - extract from filename as last resort
            import os
            filename = os.path.basename(file_path)
            if filename:
                # Remove extension
                name_part = os.path.splitext(filename)[0]
                
                # Remove common suffixes
                if name_part.endswith("_protein"):
                    name_part = name_part[:-8]
                elif name_part.endswith("_complex"):
                    name_part = name_part[:-8]
                
                # Check if it looks like a PDB ID
                if len(name_part) == 4 and name_part.isalnum():
                    pdb_id = name_part.lower()
                    log.info(f"Extracted PDB ID '{pdb_id}' from filename as fallback")
                    return pdb_id
                    
            log.warning(f"No valid PDB ID found in file header or filename: {file_path}")
            return None
        except Exception as e:
            log.error(f"Error extracting PDB ID from file {file_path}: {e}")
            return None

    def load_target_data(self) -> bool:
        """Load target protein and ligand data."""
        try:
            # Load reference protein structure
            target_pdb_file = getattr(self.config, 'target_pdb', None)
            target_pdb_id = getattr(self.config, 'protein_pdb_id', None)

            pdb_to_load = target_pdb_file
            if not pdb_to_load and target_pdb_id:
                # Find the path from the ID
                pdb_to_load = pdb_path(target_pdb_id, getattr(self.config, 'data_dir', DEFAULT_DATA_DIR))

            # If we have a file path but no PDB ID, try to extract it from the filename
            if pdb_to_load and not target_pdb_id:
                extracted_id = self._extract_pdb_id_from_path(pdb_to_load)
                if extracted_id:
                    target_pdb_id = extracted_id
                    log.info(f"Extracted PDB ID '{target_pdb_id}' from filename")

            if pdb_to_load and os.path.exists(pdb_to_load):
                self.reference_protein = load_reference_protein(pdb_to_load)
                if self.reference_protein is None:
                    log.error(f"Failed to load reference protein: {pdb_to_load}")
                    return False
                log.info(f"Loaded reference protein from {pdb_to_load}")
            
            # Load target molecule
            target_smiles = getattr(self.config, 'target_smiles', None) or getattr(self.config, 'ligand_smiles', None)
            ligand_file = getattr(self.config, 'ligand_file', None)

            if target_smiles:
                _, self.target_mol = load_target_data(pdb_to_load or "", target_smiles)
                if self.target_mol is None:
                    log.error(f"Failed to create target molecule from SMILES: {target_smiles}")
                    return False
                log.info(f"Created target molecule from SMILES: {target_smiles}")
                
                # Try to load crystal molecule for RMSD calculation
                if target_pdb_id is not None:
                    self.crystal_mol = self._load_crystal_molecule(target_pdb_id)
                else:
                    self.crystal_mol = None
                
                # Validate target molecule for peptides and other issues
                # Ensure pdb_id is never None by using multiple fallback strategies
                pdb_id = target_pdb_id
                if not pdb_id:
                    pdb_id = getattr(self.config, 'protein_pdb_id', None)
                if not pdb_id:
                    pdb_id = 'unknown'
                peptide_threshold = getattr(self.config, 'peptide_threshold', 8)
                
                is_valid, validation_msg = validate_target_molecule(
                    self.target_mol, 
                    mol_name=target_smiles[:50], 
                    pdb_id=pdb_id, 
                    peptide_threshold=peptide_threshold
                )
                
                if not is_valid:
                    # Import skip tracker if available for benchmarking
                    try:
                        from ..benchmark.skip_tracker import create_molecule_info
                        mol_info = create_molecule_info(self.target_mol, target_smiles)
                        
                        # Determine skip reason and add clear filtering prefix
                        if "peptide" in validation_msg.lower():
                            skip_reason = "large_peptide"
                            # Ensure message has filtering prefix for proper CLI classification
                            if not validation_msg.lower().startswith("filtered:"):
                                validation_msg = f"FILTERED: {validation_msg}"
                        elif "polysaccharide" in validation_msg.lower():
                            skip_reason = "large_polysaccharide"
                            if not validation_msg.lower().startswith("filtered:"):
                                validation_msg = f"FILTERED: {validation_msg}"
                        elif "rhenium" in validation_msg.lower():
                            skip_reason = "rhenium_complex"
                            if not validation_msg.lower().startswith("filtered:"):
                                validation_msg = f"FILTERED: {validation_msg}"
                        else:
                            skip_reason = "validation_failed"
                            if not validation_msg.lower().startswith("filtered:"):
                                validation_msg = f"FILTERED: {validation_msg}"
                        
                        log.warning(f"Molecule validation failed for {pdb_id}: {validation_msg}")
                        
                        # Raise the module-level MoleculeValidationException
                        raise MoleculeValidationException(
                            validation_msg, 
                            skip_reason, 
                            f"PDB: {pdb_id}, SMILES: {target_smiles[:100]}",
                            mol_info
                        )
                        
                    except ImportError:
                        # Skip tracker not available, just log and fail
                        # Add filtering prefix for proper CLI classification
                        if not validation_msg.lower().startswith("filtered:"):
                            validation_msg = f"FILTERED: {validation_msg}"
                        log.error(f"Target molecule validation failed: {validation_msg}")
                        return False
            elif ligand_file:
                # Load the first molecule from the SDF file
                from rdkit import Chem
                supplier = Chem.SDMolSupplier(ligand_file, removeHs=False)
                mols = [mol for mol in supplier if mol is not None]
                if mols:
                    self.target_mol = mols[0]
                    log.info(f"Loaded target molecule from SDF: {ligand_file} with {self.target_mol.GetNumAtoms()} atoms")
                else:
                    log.error(f"Failed to load any valid molecule from SDF: {ligand_file}")
                    return False
            else:
                log.error("No ligand SMILES or ligand file provided")
                return False

            return True
            
        except Exception as e:
            import traceback
            log.error(f"Failed to load target data: {e}")
            log.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _load_crystal_molecule(self, pdb_id: str) -> Optional[Chem.Mol]:
        """Load crystal molecule for RMSD calculation."""
        try:
            if not pdb_id:
                log.info(f"CRYSTAL_LOADING: No PDB ID provided for crystal structure loading")
                return None
                
            # Try to find crystal ligand in processed SDF
            config_data_dir = getattr(self.config, 'data_dir', None)
            actual_data_dir = config_data_dir if config_data_dir else DEFAULT_DATA_DIR
            ligands_sdf_gz = f"{actual_data_dir}/ligands/templ_processed_ligands_v1.0.0.sdf.gz"
            
            log.info(f"CRYSTAL_LOADING: Loading crystal structure for {pdb_id}")
            log.info(f"CRYSTAL_LOADING:   Config data_dir: {config_data_dir}")
            log.info(f"CRYSTAL_LOADING:   Actual data_dir: {actual_data_dir}")
            log.info(f"CRYSTAL_LOADING:   Looking for SDF: {ligands_sdf_gz}")
            log.info(f"CRYSTAL_LOADING:   SDF exists: {os.path.exists(ligands_sdf_gz)}")
            
            if not os.path.exists(ligands_sdf_gz):
                log.warning(f"CRYSTAL_LOADING: Crystal ligand SDF not found: {ligands_sdf_gz}")
                return None
                
            molecules_searched = 0
            molecules_matched = 0
            
            with gzip.open(ligands_sdf_gz, 'rb') as fh:
                for mol in Chem.ForwardSDMolSupplier(fh, removeHs=False, sanitize=False):
                    molecules_searched += 1
                    if not mol or not mol.GetNumConformers():
                        continue
                    
                    # Check if this is our target
                    mol_name = safe_name(mol, "")
                    if mol_name and mol_name.lower().startswith(pdb_id.lower()):
                        molecules_matched += 1
                        log.info(f"CRYSTAL_LOADING: Found crystal ligand for {pdb_id}")
                        log.info(f"CRYSTAL_LOADING:   Molecule name: {mol_name}")
                        log.info(f"CRYSTAL_LOADING:   Molecule atoms: {mol.GetNumAtoms()}")
                        log.info(f"CRYSTAL_LOADING:   Molecule conformers: {mol.GetNumConformers()}")
                        log.info(f"CRYSTAL_LOADING:   Searched {molecules_searched} molecules total")
                        return Chem.Mol(mol)
                        
            log.warning(f"CRYSTAL_LOADING: No crystal ligand found for {pdb_id}")
            log.info(f"CRYSTAL_LOADING:   Searched {molecules_searched} molecules total")
            log.info(f"CRYSTAL_LOADING:   Matched {molecules_matched} molecules")
            return None
            
        except Exception as e:
            log.error(f"CRYSTAL_LOADING: Failed to load crystal molecule for {pdb_id}: {e}")
            import traceback
            log.error(f"CRYSTAL_LOADING: Traceback: {traceback.format_exc()}")
            return None
    
    def load_templates(self) -> bool:
        """Load and filter template molecules from SDF file."""
        try:
            # Use default data directory structure
            ligands_sdf_gz = "data/ligands/templ_processed_ligands_v1.0.0.sdf.gz"
            
            if not os.path.exists(ligands_sdf_gz):
                log.error(f"Template SDF file not found: {ligands_sdf_gz}")
                return False
                
            log.info(f"Loading templates from {ligands_sdf_gz}")
            
            # Initialize filtering statistics
            original_count = 0
            filtered_peptides = 0
            filtered_polysaccharides = 0
            
            # Load templates from compressed SDF
            # Create temporary uncompressed file for RDKit to read
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp_file:
                with gzip.open(ligands_sdf_gz, 'rb') as gz_file:
                    tmp_file.write(gz_file.read())
                tmp_sdf_path = tmp_file.name
            
            try:
                supplier = Chem.SDMolSupplier(tmp_sdf_path, removeHs=False)
                
                for mol in supplier:
                    if mol is not None:
                        original_count += 1
                        
                        # Apply same filtering as targets - filter out peptides and polysaccharides
                        is_filtered, msg = is_large_peptide_or_polysaccharide(
                            mol, residue_threshold=8, sugar_ring_threshold=3
                        )
                        
                        if is_filtered:
                            if "peptide" in msg.lower():
                                filtered_peptides += 1
                            elif "polysaccharide" in msg.lower():
                                filtered_polysaccharides += 1
                            continue  # Skip this template
                            
                        self.templates.append(mol)
            finally:
                # Clean up temporary file
                os.unlink(tmp_sdf_path)
            
            # Store filtering statistics for CLI output
            self.template_filtering_stats = {
                "original_templates": original_count,
                "final_templates": len(self.templates),
                "peptides_filtered": filtered_peptides,
                "polysaccharides_filtered": filtered_polysaccharides
            }
            
            # Log filtering results
            total_filtered = filtered_peptides + filtered_polysaccharides
            log.info(f"Template filtering: {original_count} → {len(self.templates)} templates")
            log.info(f"Filtered out: {filtered_peptides} peptides, {filtered_polysaccharides} polysaccharides ({total_filtered} total)")
            
            return len(self.templates) > 0
            
        except Exception as e:
            log.error(f"Failed to load templates: {e}")
            return False
    
    def get_protein_embedding(self, pdb_id: str, pdb_file: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """Get protein embedding for PDB ID and the chains used.

        When a PDB file is provided, this function will first extract the PDB ID, check for
        an existing embedding in the database, and only generate a new one if not found.
        """
        try:
            if self.embedding_manager is None:
                log.error("Embedding manager not initialized")
                return None, None

            # Input validation: detect if a file path was passed instead of PDB ID
            if pdb_id and ('/' in pdb_id or '\\' in pdb_id or pdb_id.endswith('.pdb')):
                log.warning(f"File path detected as PDB ID: '{pdb_id}'. Attempting to extract PDB ID from path.")
                extracted_id = self._extract_pdb_id_from_path(pdb_id)
                if extracted_id:
                    original_input = pdb_id
                    pdb_id = extracted_id
                    log.info(f"Extracted PDB ID '{pdb_id}' from path '{original_input}'")
                else:
                    log.error(f"Could not extract valid PDB ID from path: '{pdb_id}'")
                    return None, None

            # Normalize the provided PDB ID for consistent lookup
            if pdb_id:
                pdb_id = pdb_id.upper().split(':')[-1]

            # If a PDB file is provided, extract its PDB ID and prioritize it
            if pdb_file and os.path.exists(pdb_file):
                extracted_pdb_id = self._extract_pdb_id_from_path(pdb_file)
                if extracted_pdb_id:
                    pdb_id = extracted_pdb_id.upper()
                    log.info(f"Using PDB ID '{pdb_id}' extracted from file for embedding lookup")

            # Always check for an existing embedding first
            if pdb_id and self.embedding_manager.has_embedding(pdb_id):
                log.info(f"Found existing embedding for PDB ID '{pdb_id}'. Using it.")
                embedding, chains_str = self.embedding_manager.get_embedding(pdb_id)
                if embedding is not None:
                    return embedding, (chains_str.split(',') if chains_str else [])

            # If no embedding is found, generate a new one ONLY if a file is provided
            if pdb_file and os.path.exists(pdb_file):
                log.info(f"No existing embedding for '{pdb_id}'. Generating new embedding from {pdb_file}.")
                embedding, chains_str = self.embedding_manager.get_embedding(pdb_id, pdb_file=pdb_file)
                if embedding is not None:
                    return embedding, (chains_str.split(',') if chains_str else [])

            # If no file is provided and the PDB ID is not in the database, fail gracefully
            if pdb_id and not pdb_file:
                log.error(f"PDB ID '{pdb_id}' not found in embedding database and no PDB file was provided.")
                log.error(f"Available options: 1) Provide PDB file path, 2) Check if PDB ID exists in database, 3) Verify PDB ID format (should be 4 characters)")

            return None, None

        except Exception as e:
            log.error(f"Failed to get protein embedding for {pdb_id}: {e}", exc_info=True)
            return None, None

    def find_similar_templates(self, query_pdb_id: str, query_embedding: np.ndarray, k: int = 100, exclude_pdb_ids: Optional[set] = None, allowed_pdb_ids: Optional[set] = None) -> Tuple[List[str], Dict[str, float]]:
        """Find similar templates using embedding similarity.
        
        Returns:
            Tuple of (template_pdb_ids, embedding_similarities_dict)
        """
        try:
            if self.embedding_manager is None:
                log.error("Embedding manager not initialized")
                return [], {}
                
            # Find nearest neighbors with similarities
            neighbors_with_similarities = self.embedding_manager.find_neighbors(
                query_pdb_id=query_pdb_id,
                query_embedding=query_embedding,
                k=k,
                exclude_pdb_ids=exclude_pdb_ids,
                allowed_pdb_ids=allowed_pdb_ids,
                return_similarities=True
            )
            
            # Extract PDB IDs and similarities
            template_pdb_ids = []
            embedding_similarities = {}
            
            for pdb_id, similarity in neighbors_with_similarities:
                template_pdb_ids.append(pdb_id)
                embedding_similarities[pdb_id.upper()] = float(similarity)
            
            log.info(f"Found {len(template_pdb_ids)} templates with embedding similarities ranging from {min(embedding_similarities.values()):.3f} to {max(embedding_similarities.values()):.3f}")
            
            # Store similarity search information for JSON output
            self.similarity_search_info = {
                "method": "cosine_similarity_knn",
                "knn_k_value": k,
                "raw_similar_proteins_found": len(template_pdb_ids),
                "similarity_threshold_used": None,  # Could be enhanced if threshold parameter is added
                "embedding_dimension": query_embedding.shape[0] if query_embedding is not None else None,
                "exclude_pdb_ids_count": len(exclude_pdb_ids) if exclude_pdb_ids else 0,
                "allowed_pdb_ids_count": len(allowed_pdb_ids) if allowed_pdb_ids else None
            }
            
            return template_pdb_ids, embedding_similarities
            
        except Exception as e:
            log.error(f"Failed to find similar templates: {e}")
            return [], {}
    
    def process_templates(self, template_pdb_ids: List[str], ref_chains: List[str], embedding_similarities: Optional[Dict[str, float]] = None) -> List[Chem.Mol]:
        """Process and transform template molecules."""
        processed_templates = []
        failed_templates = []
        
        # Initialize alignment method tracking
        alignment_methods_used = {"homologs": 0, "sequence": 0, "3di": 0, "centroid": 0}
        
        for i, pdb_id in enumerate(template_pdb_ids):
            try:
                # Get template molecule
                template_mol = None
                for mol in self.templates:
                    # Check various property names
                    prop_names = ["template_pid", "pdb_id", "PDB_ID", "_Name", "ID"]
                    for prop_name in prop_names:
                        if mol.HasProp(prop_name):
                            if mol.GetProp(prop_name).upper() == pdb_id.upper():
                                template_mol = mol
                                break
                    if template_mol is not None:
                        break
                
                if template_mol is None:
                    if i < 3:
                        sample_mol = self.templates[0]
                        props = sample_mol.GetPropNames()
                        log.warning(f"Template molecule not found for {pdb_id}. Sample properties: {props}")
                    else:
                        log.debug(f"Template molecule not found for {pdb_id}")
                    failed_templates.append(pdb_id)
                    continue
                
                pdb_file = pdb_path(pdb_id, "data")
                if not pdb_file:
                    log.debug(f"PDB file not found for {pdb_id}")
                    failed_templates.append(pdb_id)
                    continue

                # Get chain info for the mobile protein
                _, mob_chains = self.get_protein_embedding(pdb_id, pdb_file=pdb_file)
                
                if self.reference_protein is not None:
                    # Get actual embedding similarity for this template
                    embedding_similarity = 1.0  # Default fallback
                    if embedding_similarities and pdb_id.upper() in embedding_similarities:
                        embedding_similarity = embedding_similarities[pdb_id.upper()]
                    
                    transformed_mol = transform_ligand(
                        mob_pdb=pdb_file, 
                        lig=template_mol, 
                        pid=pdb_id, 
                        ref_struct=self.reference_protein,
                        ref_chains=ref_chains,
                        mob_chains=mob_chains,
                        similarity_score=embedding_similarity
                    )
                    
                    if transformed_mol is not None:
                        processed_templates.append(transformed_mol)
                        # Track alignment method used
                        if transformed_mol.HasProp("alignment_method"):
                            alignment_method = transformed_mol.GetProp("alignment_method")
                            if alignment_method in alignment_methods_used:
                                alignment_methods_used[alignment_method] += 1
                        log.debug(f"Processed template {i+1}/{len(template_pdb_ids)}: {pdb_id}")
                    else:
                        log.debug(f"Failed to transform template {pdb_id}")
                        failed_templates.append(pdb_id)
                else:
                    template_mol.SetProp("template_pid", pdb_id)
                    processed_templates.append(template_mol)
                    
            except Exception as e:
                log.debug(f"Error processing template {pdb_id}: {e}")
                failed_templates.append(pdb_id)
                continue
        
        log.info(f"Successfully processed {len(processed_templates)} templates from {len(template_pdb_ids)} candidates")
        if failed_templates:
            log.info(f"Failed to process {len(failed_templates)} templates: {failed_templates[:5]}{'...' if len(failed_templates) > 5 else ''}")
        
        # Store protein alignment information for JSON output
        self.protein_alignment_info = {
            "uses_superimposition": len(processed_templates) > 0,
            "total_templates_processed": len(processed_templates),
            "superimposed_poses_used": len(processed_templates),
            "alignment_methods_breakdown": alignment_methods_used.copy()
        }
        
        return processed_templates
    
    def filter_templates_by_rmsd(self, templates: List[Chem.Mol]) -> List[Chem.Mol]:
        """Filter templates by CA RMSD threshold with progressive fallback."""
        try:
            if not templates:
                return templates
                
            ca_rmsd_threshold = getattr(self.config, 'ca_rmsd_threshold', DEFAULT_CA_RMSD_THRESHOLD)
            
            # Apply progressive fallback filtering
            filtered_templates, threshold_used, use_central_atom = get_templates_with_progressive_fallback(
                templates, [ca_rmsd_threshold, 15.0, 20.0]
            )
            
            if use_central_atom:
                log.warning("Using central atom fallback due to poor template quality")
            
            if filtered_templates:
                log.info(f"Filtered to {len(filtered_templates)} templates using CA RMSD threshold {threshold_used}Å")
                return filtered_templates
            else:
                log.warning("No templates passed RMSD filtering, returning original templates")
                return templates
                
        except Exception as e:
            log.error(f"Failed to filter templates by RMSD: {e}")
            return templates
    
    def generate_conformers(self, template_mol: Chem.Mol, mcs_smarts: str, num_conformers: int) -> Optional[Chem.Mol]:
        """Generate conformers for the target molecule using a specific template and MCS."""
        try:
            if self.target_mol is None:
                log.error("Target molecule not loaded")
                return None

            n_workers = getattr(self.config, 'n_workers', DEFAULT_N_WORKERS)
            enable_optimization = getattr(self, 'enable_optimization', False)
            unconstrained = getattr(self, 'unconstrained', False)
            
            log.info(f"Pipeline generate_conformers: enable_optimization={enable_optimization}, unconstrained={unconstrained}")

            # Force unconstrained embedding if ablation flag is set
            if unconstrained or mcs_smarts == "*":
                log.info("Using central atom embedding for conformer generation")
                conformers = central_atom_embed(
                    self.target_mol,
                    template_mol,
                    num_conformers,
                    enable_optimization
                )
            else:
                log.info(f"Using MCS-constrained embedding: {mcs_smarts}")
                conformers = constrained_embed(
                    self.target_mol,
                    template_mol,
                    mcs_smarts,
                    num_conformers,
                    n_workers,
                    enable_optimization
                )
            
            return conformers
            
        except Exception as e:
            log.error(f"Failed to generate conformers: {e}", exc_info=True)
            return None
    
    def score_conformers(self, conformers: Chem.Mol, template_mol: Chem.Mol) -> Union[Dict[str, Tuple[Chem.Mol, Dict[str, float]]], List[Tuple[Chem.Mol, Dict[str, float], int]]]:
        """Score conformers against template."""
        try:
            if conformers is None or template_mol is None:
                log.error("Invalid conformers or template for scoring")
                return {}
                
            n_workers = getattr(self.config, 'n_workers', DEFAULT_N_WORKERS)
            no_realign = getattr(self, 'no_realign', False)
            align_metric = getattr(self, 'align_metric', 'combo')
            
            best_poses = select_best(
                conformers, 
                template_mol, 
                no_realign=no_realign,
                n_workers=n_workers,
                align_metric=align_metric
            )
            
            return best_poses
            
        except Exception as e:
            log.error(f"Failed to score conformers: {e}")
            return {}
    
    def save_results(self, results: Dict[str, Any], output_name: str) -> str:
        """Save results to output directory."""
        try:
            output_file = os.path.join(self.output_dir, f"{output_name}.sdf")
            
            with Chem.SDWriter(output_file) as writer:
                for metric, (pose, scores) in results.items():
                    if pose is not None:
                        enhanced_pose = generate_properties_for_sdf(
                            pose, 
                            metric, 
                            scores[metric], 
                            output_name
                        )
                        writer.write(enhanced_pose)
            
            writer.close()
            log.info(f"Results saved to {output_file}")
            return output_file
            
        except Exception as e:
            log.error(f"Failed to save results: {e}")
            return ""
    
    def save_consolidated_results(self, poses: List[Chem.Mol], output_name: str) -> str:
        """Save all poses to a single consolidated SDF file."""
        try:
            output_file = os.path.join(self.output_dir, f"{output_name}.sdf")
            
            with Chem.SDWriter(output_file) as writer:
                for pose in poses:
                    if pose is not None:
                        writer.write(pose)
            
            writer.close()
            log.info(f"Consolidated results saved to {output_file}")
            return output_file
            
        except Exception as e:
            log.error(f"Failed to save consolidated results: {e}")
            return ""
    
    def run_full_pipeline(self, protein_file: Optional[str] = None, protein_pdb_id: Optional[str] = None, 
                         ligand_smiles: Optional[str] = None, ligand_file: Optional[str] = None,
                         num_templates: int = 100, num_conformers: int = 200,
                         n_workers: int = 4, similarity_threshold: float = 0.9,
                         exclude_pdb_ids: Optional[set] = None, allowed_pdb_ids: Optional[set] = None,
                         output_dir: Optional[str] = None, no_realign: bool = False, 
                         enable_optimization: bool = False, unconstrained: bool = False, 
                         align_metric: str = "combo") -> dict:
        """Run the full pipeline with CLI interface."""
        # Use provided output_dir or fall back to instance output_dir
        effective_output_dir = output_dir or self.output_dir
        
        config = PipelineConfig(
            target_pdb=protein_file or "",
            target_smiles=ligand_smiles or "",
            protein_pdb_id=protein_pdb_id,
            ligand_smiles=ligand_smiles,
            ligand_file=ligand_file,
            output_dir=effective_output_dir,
            n_confs=num_conformers,
            n_workers=n_workers,
            sim_threshold=similarity_threshold,
            num_templates=num_templates,
            no_realign=no_realign,
            enable_optimization=enable_optimization,
            unconstrained=unconstrained,
            align_metric=align_metric
        )
        
        self.config = config
        self.exclude_pdb_ids = exclude_pdb_ids or set()
        self.allowed_pdb_ids = allowed_pdb_ids or None
        self.no_realign = no_realign
        self.enable_optimization = enable_optimization
        self.unconstrained = unconstrained
        self.align_metric = align_metric
        
        # Store poses during pipeline execution
        self.pipeline_poses = {}
        self.pipeline_template_info = {}
        self.pipeline_all_ranked_poses = []
        
        success = self.run()
        
        # Get the timestamped output folder path
        output_folder = str(self.output_manager.timestamped_folder) if self.output_manager.timestamped_folder else self.output_dir
        
        return {
            "success": success,
            "templates": getattr(self, "templates", []),
            "filtered_templates": getattr(self, "similar_template_ids", []),
            "template_similarities": getattr(self, "embedding_similarities", {}),
            "filtering_info": {
                "exclude_pdb_ids": list(getattr(self, "exclude_pdb_ids_used", set())),
                "allowed_pdb_ids": list(getattr(self, "allowed_pdb_ids_used", set())) if getattr(self, "allowed_pdb_ids_used", None) is not None else None,
                "requested_templates": getattr(self, "num_templates_requested", 0),
                "found_templates": len(getattr(self, "similar_template_ids", []))
            },
            "template_processing_pipeline": {
                "processing_stats": getattr(self, "template_processing_stats", {}),
                "filtering_stats": getattr(self, "template_filtering_stats", {}),
                "total_available_ligands": len(getattr(self, "templates", [])),
                "final_usable_templates": getattr(self, "final_template_count", 0),
                "similarity_search": getattr(self, "similarity_search_info", {}),
                "protein_alignment": getattr(self, "protein_alignment_info", {}),
                "mcs_analysis_input": getattr(self, "mcs_analysis_input", {})
            },
            "poses": getattr(self, "pipeline_poses", {}),
            "template_info": getattr(self, "pipeline_template_info", {}),
            "all_ranked_poses": getattr(self, "pipeline_all_ranked_poses", []),
            "output_file": output_folder,
            "output_folder": output_folder,  # New field for timestamped folder
            "pipeline_config": config
        }
    
    def run(self) -> bool:
        """
        Run the complete pipeline based on the user-defined methodology.
        """
        try:
            log.info("Starting TEMPL pipeline with new methodology")
            
            # Initialize MCS stage tracking
            self.made_it_to_mcs = False

            if not self.load_target_data():
                return False

            query_pdb_id = getattr(self.config, 'protein_pdb_id', None)
            target_pdb_file = getattr(self.config, 'target_pdb', None)
            
            # If no explicit PDB ID is provided, try to extract it from the file path
            if not query_pdb_id and target_pdb_file:
                extracted_id = self._extract_pdb_id_from_path(target_pdb_file)
                if extracted_id:
                    query_pdb_id = extracted_id
                    log.info(f"Using extracted PDB ID '{query_pdb_id}' for pipeline execution")
                else:
                    # Use the filename without extension as fallback
                    query_pdb_id = os.path.splitext(os.path.basename(target_pdb_file))[0]
                    log.info(f"Using filename '{query_pdb_id}' as fallback PDB ID")
            elif not query_pdb_id and not target_pdb_file:
                log.error("No protein PDB ID or file provided")
                return False

            # Setup enhanced output folder structure
            if query_pdb_id is None:
                log.error("No valid PDB ID available for output folder setup")
                return False
            self.output_manager.setup_output_folder(query_pdb_id)

            # Get target embedding and chains
            query_embedding, ref_chains = self.get_protein_embedding(query_pdb_id, pdb_file=target_pdb_file)

            if query_embedding is None:
                log.error(f"Failed to get/generate protein embedding for {query_pdb_id}")
                return False

            # If chains were not found during embedding, try to get them now
            if not ref_chains:
                log.warning(f"No chains returned with embedding for {query_pdb_id}. Attempting to extract from file.")
                if target_pdb_file and os.path.exists(target_pdb_file):
                    _, ref_chains = get_protein_sequence(target_pdb_file)
                    if ref_chains:
                        log.info(f"Successfully extracted chains: {ref_chains}")
                    else:
                        log.error(f"Could not extract chains from {target_pdb_file}")
                        return False
                else:
                    # Attempt to get chain data from the embedding manager as a last resort
                    if self.embedding_manager is not None:
                        chains_str = self.embedding_manager.get_chain_data(query_pdb_id)
                        if chains_str:
                            ref_chains = chains_str.split(',')
                            log.info(f"Successfully retrieved chains from embedding metadata: {ref_chains}")
                        else:
                            log.error(f"No PDB file available and no chain data in embedding for {query_pdb_id}")
                            return False
                    else:
                        log.error(f"Embedding manager is None and no PDB file available for {query_pdb_id}")
                        return False

            num_templates = getattr(self.config, 'num_templates', 100)
            exclude_pdb_ids = getattr(self, 'exclude_pdb_ids', set())
            allowed_pdb_ids = getattr(self, 'allowed_pdb_ids', None)
            similar_template_ids, embedding_similarities = self.find_similar_templates(query_pdb_id, query_embedding, k=num_templates, exclude_pdb_ids=exclude_pdb_ids, allowed_pdb_ids=allowed_pdb_ids)
            log.info(f"Found {len(similar_template_ids)} similar templates for {query_pdb_id}")
            
            # Store filtered template information for JSON output
            self.similar_template_ids = similar_template_ids
            self.embedding_similarities = embedding_similarities
            self.exclude_pdb_ids_used = exclude_pdb_ids.copy()
            self.allowed_pdb_ids_used = allowed_pdb_ids.copy() if allowed_pdb_ids is not None else None
            self.num_templates_requested = num_templates
            
            # Validate timesplit constraints are respected
            if allowed_pdb_ids is not None:
                # Check that target is not in template list (should be excluded by constraints)
                if query_pdb_id.upper() in [pid.upper() for pid in similar_template_ids]:
                    log.warning(f"Timesplit constraint validation: target {query_pdb_id} found in template list - this may indicate constraint bypass")
                else:
                    log.info(f"Timesplit constraint validation: target {query_pdb_id} properly excluded from template list")
                
                # Validate all templates are within allowed set
                invalid_templates = [pid for pid in similar_template_ids if pid.upper() not in allowed_pdb_ids]
                if invalid_templates:
                    log.error(f"Constraint violation: templates {invalid_templates} not in allowed set")
                else:
                    log.info(f"Constraint validation passed: all {len(similar_template_ids)} templates within allowed set")
            
            if not similar_template_ids:
                log.error("No similar templates found.")
                return False
            
            # Only include target PDB ID as template when no constraints are applied (not in timesplit scenarios) and not excluded
            if (allowed_pdb_ids is None and 
                query_pdb_id.upper() not in [pid.upper() for pid in similar_template_ids] and
                (exclude_pdb_ids is None or query_pdb_id.upper() not in exclude_pdb_ids)):
                # Check if target exists in embedding database
                target_embedding, _ = self.get_protein_embedding(query_pdb_id)
                if target_embedding is not None:
                    similar_template_ids.insert(0, query_pdb_id)
                    # Add perfect similarity score for the target itself
                    embedding_similarities[query_pdb_id.upper()] = 1.0
                    log.info(f"Added target {query_pdb_id} to template list as it exists in database")
            elif allowed_pdb_ids is not None:
                # In constrained scenarios (like timesplit), respect the constraints strictly
                log.info(f"Template constraints active: target {query_pdb_id} excluded to maintain benchmark integrity")
            
            log.info(f"Found {len(similar_template_ids)} similar templates.")

            if not self.load_templates():
                return False
            
            # Prepare list for all transformed ligands, including native if applicable
            all_transformed_ligands = []
            
            # Initialize native pose tracking
            native_poses_used = 0
            
            # Handle native ligand use when target is legitimately found as a template
            # Note: In timesplit scenarios, target won't be in similar_template_ids due to constraints
            target_is_template = False
            if query_pdb_id.upper() in [pid.upper() for pid in similar_template_ids]:
                target_is_template = True
                if allowed_pdb_ids is not None:
                    log.info(f"Target {query_pdb_id} found as legitimate template within constraints, using native pose.")
                else:
                    log.info(f"Target {query_pdb_id} is in templates, using its native pose.")
                native_ligand_path = ligand_path(query_pdb_id, "data")
                if native_ligand_path and os.path.exists(native_ligand_path):
                    try:
                        native_ligand = Chem.SDMolSupplier(native_ligand_path, removeHs=False)[0]
                        if native_ligand:
                            native_ligand.SetProp("template_pid", query_pdb_id)
                            native_ligand.SetProp("ca_rmsd", "0.0") # Native pose has 0 RMSD
                            native_ligand.SetProp("similarity_score", "1.0") # Native pose has 1.0 similarity
                            native_ligand.SetProp("alignment_method", "native") # Native template uses no alignment
                            native_ligand.SetProp("anchor_count", "N/A") # Native template has no alignment anchors
                            all_transformed_ligands.append(native_ligand)
                            native_poses_used += 1  # Track native pose usage
                            # Remove target from list to be processed by transform_ligand
                            similar_template_ids = [pid for pid in similar_template_ids if pid.upper() != query_pdb_id.upper()]
                        else:
                            log.warning(f"Could not load native ligand for {query_pdb_id} from {native_ligand_path}")
                    except Exception as e:
                        log.warning(f"Error loading native ligand: {e}")
                else:
                    log.warning(f"Native ligand file not found for {query_pdb_id} at {native_ligand_path}")

            # Process remaining templates
            if similar_template_ids:
                transformed_ligands_from_templates = self.process_templates(similar_template_ids, ref_chains, embedding_similarities)
                
                # Store processing statistics
                self.template_processing_stats = {
                    "embedding_search_found": len(similar_template_ids),
                    "processing_attempted": len(similar_template_ids),
                    "processing_successful": len(transformed_ligands_from_templates),
                    "processing_failed": len(similar_template_ids) - len(transformed_ligands_from_templates)
                }
                
                # Apply RMSD filtering with progressive fallback
                filtered_templates = self.filter_templates_by_rmsd(transformed_ligands_from_templates)
                
                # Store filtering statistics
                self.template_filtering_stats = {
                    "before_rmsd_filtering": len(transformed_ligands_from_templates),
                    "after_rmsd_filtering": len(filtered_templates),
                    "removed_by_rmsd_filter": len(transformed_ligands_from_templates) - len(filtered_templates)
                }
                
                all_transformed_ligands.extend(filtered_templates)

            if not all_transformed_ligands:
                log.error("Failed to process and transform any template ligands.")
                return False
            
            # Store final template count after all filtering (including native template if added)
            self.final_template_count = len(all_transformed_ligands)
            
            # Update protein alignment info to include native pose tracking
            if hasattr(self, 'protein_alignment_info'):
                self.protein_alignment_info["native_poses_used"] = native_poses_used
                self.protein_alignment_info["alignment_methods_breakdown"]["native"] = native_poses_used
            else:
                # In case no templates were processed via transform_ligand, initialize the info
                self.protein_alignment_info = {
                    "uses_superimposition": False,
                    "total_templates_processed": 0,
                    "superimposed_poses_used": 0,
                    "native_poses_used": native_poses_used,
                    "alignment_methods_breakdown": {"homologs": 0, "sequence": 0, "3di": 0, "centroid": 0, "native": native_poses_used}
                }

            # Ensure target molecule is properly prepared
            if self.target_mol is None:
                log.error("Target molecule is None, cannot proceed with MCS finding")
                return False

            # Store MCS analysis input information for JSON output
            pre_rmsd_filtering_count = len(transformed_ligands_from_templates) if 'transformed_ligands_from_templates' in locals() else 0
            post_rmsd_filtering_count = len(filtered_templates) if 'filtered_templates' in locals() else 0
            
            self.mcs_analysis_input = {
                "total_templates_for_mcs": len(all_transformed_ligands),
                "native_templates": native_poses_used,
                "superimposed_templates": len(all_transformed_ligands) - native_poses_used,
                "pre_rmsd_filtering": pre_rmsd_filtering_count,
                "post_rmsd_filtering": post_rmsd_filtering_count
            }

            # Mark that we successfully reached the MCS stage 
            # (passed target validation, template loading, similarity search, etc.)
            self.made_it_to_mcs = True
            log.info("Reached MCS processing stage")

            find_mcs_result = find_mcs(self.target_mol, all_transformed_ligands, return_details=True)
            if len(find_mcs_result) == 3:
                best_template_idx, mcs_smarts, mcs_details = find_mcs_result
            else:
                best_template_idx, mcs_smarts = find_mcs_result
                mcs_details = {}
            best_template = all_transformed_ligands[best_template_idx]
            log.info(f"Best template found: {best_template.GetProp('template_pid')} with MCS: {mcs_smarts}")

            num_conformers = getattr(self.config, 'n_confs', 200)
            target_with_conformers = self.generate_conformers(best_template, mcs_smarts, num_conformers)
            if not target_with_conformers or target_with_conformers.GetNumConformers() == 0:
                log.error("Failed to generate conformers.")
                return False
            log.info(f"Generated {target_with_conformers.GetNumConformers()} conformers.")

            # Score conformers efficiently (single scoring call)
            n_workers = getattr(self.config, 'n_workers', DEFAULT_N_WORKERS)
            no_realign = getattr(self, 'no_realign', False)
            align_metric = getattr(self, 'align_metric', 'combo')
            
            # Get all ranked poses first (comprehensive results)
            all_ranked_poses = select_best(
                target_with_conformers, best_template, 
                no_realign=no_realign,
                n_workers=n_workers,
                return_all_ranked=True,
                align_metric=align_metric
            )
            
            if not all_ranked_poses:
                log.error("Failed to rank and select poses.")
                return False
            
            # Store all ranked poses for CLI interface
            self.pipeline_all_ranked_poses = all_ranked_poses.copy()
            
            # Extract top poses by metric from all_ranked_poses (avoid redundant scoring)
            top_poses = {}
            metrics = ['shape', 'color', 'combo']
            
            for metric in metrics:
                # Find best pose for this metric from all ranked poses
                best_pose = None
                best_score = -1
                
                if isinstance(all_ranked_poses, list):
                    for conf_id, scores, mol in all_ranked_poses:
                        if metric in scores and scores[metric] > best_score:
                            best_score = scores[metric]
                            best_pose = (mol, scores)
                else:
                    # Handle case where all_ranked_poses is a dict
                    for metric_name, (mol, scores) in all_ranked_poses.items():
                        if metric_name == metric and metric in scores and scores[metric] > best_score:
                            best_score = scores[metric]
                            best_pose = (mol, scores)
                
                if best_pose:
                    top_poses[metric] = best_pose
            
            if not top_poses:
                log.error("Failed to extract top poses by metric.")
                return False

            # Store poses for CLI interface
            if hasattr(self, 'pipeline_poses'):
                self.pipeline_poses = top_poses.copy()
            
            # Store template information for CLI interface  
            if hasattr(self, 'pipeline_template_info'):
                # Store template info (without MCS details which go in separate section)
                embedding_similarity = best_template.GetProp('similarity_score') if best_template.HasProp('similarity_score') else 'unknown'
                alignment_method = best_template.GetProp('alignment_method') if best_template.HasProp('alignment_method') else 'unknown'
                anchor_count = best_template.GetProp('anchor_count') if best_template.HasProp('anchor_count') else 'unknown'
                
                self.pipeline_template_info = {
                    "best_template_pdb": best_template.GetProp('template_pid') if best_template.HasProp('template_pid') else 'unknown',
                    "num_conformers_generated": target_with_conformers.GetNumConformers(),
                    "ca_rmsd": best_template.GetProp('ca_rmsd') if best_template.HasProp('ca_rmsd') else 'unknown',
                    "embedding_similarity": embedding_similarity,
                    "alignment_method": alignment_method,
                    "anchor_count": anchor_count
                }

            # Get crystal structure for RMSD calculation if available
            crystal_mol = None
            if hasattr(self, 'crystal_mol'):
                crystal_mol = self.crystal_mol
            
            # Use enhanced output manager to save all files
            top_poses_file = self.output_manager.save_top_poses(
                top_poses, best_template, mcs_details, crystal_mol
            )
            
            # Convert dict to list format if needed for save_all_poses
            poses_for_saving: List[Tuple[Chem.Mol, Dict[str, float], int]] = []
            if isinstance(all_ranked_poses, dict):
                # Convert dict format to list format
                for metric_name, (mol, scores) in all_ranked_poses.items():
                    poses_for_saving.append((mol, scores, 0))  # Add dummy conf_id
            else:
                # Already in list format, but need to reorder: (conf_id, scores, mol) -> (mol, scores, conf_id)
                poses_for_saving = [(mol, scores, conf_id) for conf_id, scores, mol in all_ranked_poses]
            
            all_poses_file = self.output_manager.save_all_poses(
                poses_for_saving, best_template, mcs_details, crystal_mol
            )
            
            template_file = self.output_manager.save_template(best_template)
            
            # Save structured pipeline results
            # Reorganize MCS details with similarity and SMARTS at top, plus additional useful info
            mcs_details_organized = {}
            if mcs_details:
                # Put key information first
                mcs_details_organized["mcs_similarity"] = mcs_details.get("similarity_score", 0.0)
                mcs_details_organized["smarts"] = mcs_details.get("smarts", "")
                mcs_details_organized["atom_count"] = mcs_details.get("atom_count", 0)
                mcs_details_organized["bond_count"] = mcs_details.get("bond_count", 0)
                
                # Add derived metrics for better understanding
                total_query_atoms = len(self.target_mol.GetAtoms()) if self.target_mol else 0
                total_template_atoms = len(best_template.GetAtoms()) if best_template else 0
                mcs_details_organized["query_molecule_atoms"] = total_query_atoms
                mcs_details_organized["template_molecule_atoms"] = total_template_atoms
                mcs_details_organized["mcs_coverage_query"] = round(mcs_details_organized["atom_count"] / total_query_atoms * 100, 1) if total_query_atoms > 0 else 0.0
                mcs_details_organized["mcs_coverage_template"] = round(mcs_details_organized["atom_count"] / total_template_atoms * 100, 1) if total_template_atoms > 0 else 0.0
                
                # Add atom mappings at the end
                mcs_details_organized["query_atoms"] = mcs_details.get("query_atoms", [])
                mcs_details_organized["template_atoms"] = mcs_details.get("template_atoms", [])
            
            pipeline_results = {
                "target_pdb": query_pdb_id,
                "template_info": self.pipeline_template_info,
                "mcs_details": mcs_details_organized,
                "pipeline_analysis": {
                    "similarity_search": getattr(self, "similarity_search_info", {}),
                    "protein_alignment": getattr(self, "protein_alignment_info", {}),
                    "mcs_analysis_input": getattr(self, "mcs_analysis_input", {})
                },
                "num_conformers": target_with_conformers.GetNumConformers(),
                "top_poses_file": str(top_poses_file),
                "all_poses_file": str(all_poses_file),
                "template_file": str(template_file),
                "timestamp": str(datetime.now())
            }
            
            results_file = self.output_manager.save_pipeline_results(pipeline_results)
            
            log.info(f"Pipeline completed successfully.")
            log.info(f"Files saved to: {self.output_manager.timestamped_folder}")
            log.info(f"- Top 3 poses: {top_poses_file}")
            log.info(f"- All poses: {all_poses_file}")
            log.info(f"- Template: {template_file}")
            log.info(f"- Results: {results_file}")
            
            return True

        except Exception as e:
            log.error(f"Pipeline failed: {e}", exc_info=True)
            return False


def run_pipeline(config: PipelineConfig) -> bool:
    """Run the TEMPL pipeline with given configuration."""
    pipeline = TEMPLPipeline(
        embedding_path=config.embedding_npz,
        output_dir=config.output_dir
    )
    pipeline.config = config
    return pipeline.run()


def create_default_config() -> PipelineConfig:
    """Create default pipeline configuration."""
    return PipelineConfig()


# CLI Integration Functions

def run_pipeline_from_args(args) -> bool:
    """Run pipeline from command line arguments."""
    try:
        config = PipelineConfig(
            target_pdb=getattr(args, 'protein_file', ''),
            target_smiles=getattr(args, 'ligand_smiles', ''),
            protein_pdb_id=getattr(args, 'protein_pdb_id', None),
            output_dir=getattr(args, 'output_dir', DEFAULT_OUTPUT_DIR),
            n_confs=getattr(args, 'n_confs', DEFAULT_N_CONFS),
            n_workers=getattr(args, 'n_workers', DEFAULT_N_WORKERS),
            ca_rmsd_threshold=getattr(args, 'ca_rmsd_threshold', DEFAULT_CA_RMSD_THRESHOLD),
        )
        
        return run_pipeline(config)
        
    except Exception as e:
        log.error(f"Failed to run pipeline from arguments: {e}")
        return False


def run_from_pdb_and_smiles(pdb_id: str, smiles: str, output_dir: str = DEFAULT_OUTPUT_DIR) -> bool:
    """Convenience function to run pipeline from PDB ID and SMILES."""
    try:
        config = PipelineConfig(
            protein_pdb_id=pdb_id,
            target_smiles=smiles,
            output_dir=output_dir
        )
        
        return run_pipeline(config)
        
    except Exception as e:
        log.error(f"Failed to run pipeline from PDB ID and SMILES: {e}")
        return False