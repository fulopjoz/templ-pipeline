"""
TEMPL Pipeline Orchestrator

This module provides a unified interface for running the complete TEMPL pipeline,
integrating embedding generation, template finding, and pose generation.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import tempfile
from datetime import datetime
from rdkit import Chem, RDLogger
import biotite.structure.io as bsio
from .directory_manager import DirectoryManager

RDLogger.DisableLog("rdApp.*")
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

logger = logging.getLogger(__name__)

# Import enhanced scoring components
try:
    from .scoring import FixedMolecularProcessor, ScoringFixer

    logger.debug("Enhanced scoring components loaded")
except ImportError:
    logger.warning("Enhanced scoring components not available")


class TEMPLPipeline:
    """Main pipeline orchestrator for TEMPL."""

    def __init__(
        self,
        embedding_path: Optional[str] = None,
        output_dir: str = "output",
        run_id: Optional[str] = None,
        auto_cleanup: bool = False,
        shared_embedding_cache: Optional[str] = None,
    ):
        """
        Initialize the TEMPL pipeline.

        Args:
            embedding_path: Path to embedding database
            output_dir: Base directory for output files
            run_id: Custom run identifier (default: timestamp)
            auto_cleanup: Whether to automatically clean up output directory on exit
        """
        self.embedding_path = embedding_path
        self.auto_cleanup = auto_cleanup
        self._cleanup_registered = False
        self.shared_embedding_cache = shared_embedding_cache

        # Use directory manager for better lifecycle management
        self._directory_manager = DirectoryManager(
            base_name="run",
            run_id=run_id,
            auto_cleanup=auto_cleanup,
            lazy_creation=True,
            centralized_output=True,
            output_root=output_dir
        )

        # Initialize components lazily
        self._embedding_manager = None
        self._rdkit_modules = None
        
        # Initialize reproducible environment for deterministic results
        try:
            from .utils import ensure_reproducible_environment
            ensure_reproducible_environment()
        except ImportError:
            logger.warning("Could not initialize reproducible environment - results may not be deterministic")

    @property
    def output_dir(self) -> Path:
        """Get the output directory, creating it if necessary."""
        return self._directory_manager.directory

    def cleanup(self):
        """Manually clean up the output directory."""
        return self._directory_manager.cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.auto_cleanup:
            self._directory_manager.cleanup()

    def _get_embedding_manager(self):
        """Lazy initialization of embedding manager."""
        if self._embedding_manager is None:
            from .embedding import EmbeddingManager

            if self.embedding_path is None:
                # Try default paths
                default_paths = [
                    "data/embeddings/templ_protein_embeddings_v1.0.0.npz",
                    "templ_pipeline/data/embeddings/templ_protein_embeddings_v1.0.0.npz",
                ]

                for path in default_paths:
                    if os.path.exists(path):
                        self.embedding_path = path
                        break

                if self.embedding_path is None:
                    raise FileNotFoundError(
                        "No embedding database found. Please specify embedding_path."
                    )

            self._embedding_manager = EmbeddingManager(
                self.embedding_path, 
                shared_embedding_cache=self.shared_embedding_cache
            )

        return self._embedding_manager

    def _get_rdkit(self):
        """Lazy initialization of RDKit modules."""
        if self._rdkit_modules is None:
            from rdkit import Chem, RDLogger
            from rdkit.Chem import AllChem

            RDLogger.DisableLog("rdApp.*")
            self._rdkit_modules = (Chem, AllChem)
        return self._rdkit_modules

    def generate_embedding(
        self, protein_file: str, chain: Optional[str] = None
    ) -> Tuple[Any, str]:
        """
        Generate embedding for a protein, checking database first.

        Args:
            protein_file: Path to protein PDB file
            chain: Specific chain to use

        Returns:
            Tuple of (embedding, chain_id)
        """
        from .embedding import get_protein_embedding, extract_pdb_id_from_path

        logger.info(f"Generating embedding for {protein_file}")

        if not os.path.exists(protein_file):
            raise FileNotFoundError(f"Protein file not found: {protein_file}")

        # Try to extract PDB ID and check database first
        pdb_id = extract_pdb_id_from_path(protein_file)
        if pdb_id:
            embedding_manager = self._get_embedding_manager()
            if embedding_manager.has_embedding(pdb_id):
                logger.info(f"Using cached embedding for PDB: {pdb_id}")
                embedding, chain_id = embedding_manager.get_embedding(pdb_id)
                if embedding is not None:
                    logger.info(
                        f"Retrieved cached embedding with shape {embedding.shape} for chain {chain_id or 'A'}"
                    )
                    return embedding, chain_id or "A"

        # Fallback to generating new embedding
        if pdb_id:
            logger.info(f"PDB {pdb_id} not in cache, generating new embedding")
        else:
            logger.info(
                "Could not extract PDB ID from filename, generating new embedding"
            )

        embedding, chain_id = get_protein_embedding(protein_file, chain)

        if embedding is None:
            raise RuntimeError("Failed to generate protein embedding")

        logger.info(
            f"Generated embedding with shape {embedding.shape} for chain {chain_id}"
        )
        return embedding, chain_id

    def find_templates(
        self,
        protein_file: Optional[str] = None,
        protein_embedding: Optional[Any] = None,
        num_templates: int = 10,
        similarity_threshold: Optional[float] = None,
        allow_self_as_template: bool = False,
        exclude_pdb_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Find similar protein templates.

        Args:
            protein_file: Path to protein PDB file (if embedding not provided)
            protein_embedding: Pre-computed embedding
            num_templates: Number of templates to return
            similarity_threshold: Minimum similarity threshold
            allow_self_as_template: Allow query to appear in results
            exclude_pdb_ids: Set of PDB IDs to exclude from results

        Returns:
            List of (pdb_id, similarity) tuples
        """
        embedding_manager = self._get_embedding_manager()

        # Get embedding if not provided
        if protein_embedding is None:
            if protein_file is None:
                raise ValueError(
                    "Either protein_file or protein_embedding must be provided"
                )
            protein_embedding, _ = self.generate_embedding(protein_file)

        logger.info("Finding similar templates")

        # Find templates
        templates = embedding_manager.find_neighbors(
            query_pdb_id="query",  # Dummy PDB ID since we provide embedding directly
            query_embedding=protein_embedding,
            exclude_pdb_ids=exclude_pdb_ids,
            k=num_templates if similarity_threshold is None else None,
            similarity_threshold=similarity_threshold,
            return_similarities=True,
            allow_self_as_template=allow_self_as_template,
        )

        logger.info(f"Found {len(templates)} templates")
        return templates

    def load_template_molecules(self, template_pdbs: List[str]) -> List[Any]:
        """
        Load template molecules from processed SDF file using standardized approach.

        Args:
            template_pdbs: List of PDB IDs

        Returns:
            List of RDKit molecules
        """
        # Import standardized template loading function
        from templ_pipeline.core.templates import load_template_molecules_standardized

        try:
            # Use the standardized template loading function - this works consistently
            templates, loading_stats = load_template_molecules_standardized(
                template_pdb_ids=template_pdbs
            )

            if "error" in loading_stats:
                raise FileNotFoundError(
                    f"Template loading failed: {loading_stats['error']}"
                )

            if loading_stats.get("missing_pdbs"):
                missing_pdbs = loading_stats["missing_pdbs"]
                logger.warning(f"Could not find templates: {', '.join(missing_pdbs)}")

            logger.info(
                f"Loaded {len(templates)} template molecules using standardized loader"
            )
            return templates

        except Exception as e:
            logger.error(f"Failed to load template molecules: {e}")
            # Fallback to direct database access if standardized loading fails
            return self._load_templates_direct_fallback(template_pdbs)

    def _load_templates_direct_fallback(self, template_pdbs: List[str]) -> List[Any]:
        """
        Fallback method to load templates directly from database.

        Args:
            template_pdbs: List of PDB IDs

        Returns:
            List of RDKit molecules
        """
        logger.info("Using direct database fallback for template loading")
        Chem, _ = self._get_rdkit()

        templates = []
        sdf_path = "templ_pipeline/data/ligands/templ_processed_ligands_v1.0.0.sdf.gz"

        if not os.path.exists(sdf_path):
            logger.error(f"Database file not found: {sdf_path}")
            return []

        try:
            supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)

            for mol in supplier:
                if mol is None:
                    continue

                # Get molecule name
                mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""

                # Check if this matches any of our target PDBs
                for pdb_id in template_pdbs:
                    if pdb_id.lower() in mol_name.lower():
                        # Apply consistent processing
                        mol_processed = Chem.Mol(mol)
                        Chem.SanitizeMol(mol_processed)
                        Chem.SetAromaticity(mol_processed)
                        templates.append(mol_processed)
                        logger.info(f"Loaded template {pdb_id} using direct fallback")
                        break

            logger.info(f"Direct fallback loaded {len(templates)} templates")
            return templates

        except Exception as e:
            logger.error(f"Direct fallback failed: {e}")
            return []

    def prepare_query_molecule(
        self, ligand_smiles: Optional[str] = None, ligand_file: Optional[str] = None
    ) -> Any:
        """
        Prepare query molecule from SMILES or file with standardization.

        Args:
            ligand_smiles: SMILES string
            ligand_file: Path to ligand file (SDF/MOL)

        Returns:
            RDKit molecule with standardized representation
        """
        from .molecule_standardization import standardize_and_prepare_molecule

        Chem, AllChem = self._get_rdkit()

        if ligand_smiles:
            logger.info(f"Creating molecule from SMILES: {ligand_smiles}")
            mol = Chem.MolFromSmiles(ligand_smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {ligand_smiles}")

            # Use standardized preparation
            mol = standardize_and_prepare_molecule(mol, ensure_conformers=True)

        elif ligand_file:
            logger.info(f"Loading molecule from file: {ligand_file}")
            if not os.path.exists(ligand_file):
                raise FileNotFoundError(f"Ligand file not found: {ligand_file}")

            if ligand_file.endswith(".sdf"):
                supplier = Chem.SDMolSupplier(ligand_file, removeHs=False)
            elif ligand_file.endswith(".mol"):
                mol = Chem.MolFromMolFile(ligand_file, removeHs=False)
                supplier = [mol] if mol else []
            else:
                raise ValueError("Ligand file must be SDF or MOL format")

            mol = None
            for m in supplier:
                if m:
                    mol = Chem.RemoveHs(m)
                    break

            if mol is None:
                raise ValueError(f"Could not load molecule from {ligand_file}")

            # Use standardized preparation
            mol = standardize_and_prepare_molecule(mol, ensure_conformers=True)

        else:
            raise ValueError("Either ligand_smiles or ligand_file must be provided")

        logger.info(
            f"Query molecule prepared (standardized): {mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds"
        )
        return mol

    def generate_poses(
        self,
        query_mol: Any,
        template_mols: List[Any],
        num_conformers: int = 100,
        n_workers: int = 4,
        use_aligned_poses: bool = True,
        target_protein_file: Optional[str] = None,
        target_protein_pdb_id: Optional[str] = None,
        target_chain_id: Optional[str] = None,
        template_pdbs: Optional[List[str]] = None,
        template_similarities: Optional[List[float]] = None,
    ) -> Dict[str, Tuple[Any, Dict[str, float]]]:
        """
        Generate poses using template-based conformer generation with protein alignment.

        Args:
            query_mol: Query molecule
            template_mols: List of template molecules
            num_conformers: Number of conformers to generate
            n_workers: Number of worker processes
            use_aligned_poses: If True, return aligned poses. If False, return original conformers with scores.
            target_protein_file: Path to target protein PDB file
            target_protein_pdb_id: Target protein PDB ID
            target_chain_id: Target protein chain ID from embedding
            template_pdbs: List of template PDB IDs corresponding to template_mols
            template_similarities: List of similarity scores for templates

        Returns:
            Dictionary with poses and metadata, including MCS info
        """
        from .mcs import generate_conformers, find_mcs, transform_ligand
        from .scoring import select_best
        from .molecule_validation import validate_target_molecule, get_molecule_complexity_info
        import biotite.structure.io as bsio
        import os

        if not template_mols:
            raise ValueError("No template molecules provided")

        # Validate query molecule before processing with skip management
        from .skip_manager import MoleculeSkipException, create_validation_skip_wrapper
        
        query_mol_clean = Chem.RemoveHs(query_mol) if query_mol else None
        
        # Create validation wrapper that converts failures to skip exceptions
        validation_wrapper = create_validation_skip_wrapper(validate_target_molecule)
        
        try:
            is_valid, validation_msg = validation_wrapper(query_mol_clean, "query", peptide_threshold=8)
            
            # Log complexity information for debugging
            complexity_info = get_molecule_complexity_info(query_mol_clean)
            logger.debug(f"Query molecule complexity: {complexity_info}")
            
        except MoleculeSkipException as skip_ex:
            # Log the skip and re-raise to be handled by caller
            complexity_info = get_molecule_complexity_info(query_mol_clean)
            logger.info(f"Molecule complexity: {complexity_info}")
            raise skip_ex

        logger.info(
            f"Generating poses with {len(template_mols)} templates, {num_conformers} conformers, {n_workers} workers"
        )
        
        # CRITICAL FIX: Align template molecules to target protein binding site
        aligned_template_mols = template_mols
        if (target_protein_file or target_protein_pdb_id) and template_pdbs:
            logger.info("Step 5a: Aligning template molecules to target protein binding site")
            aligned_template_mols = self._align_template_molecules(
                template_mols=template_mols,
                template_pdbs=template_pdbs,
                target_protein_file=target_protein_file,
                target_protein_pdb_id=target_protein_pdb_id,
                target_chain_id=target_chain_id,
                template_similarities=template_similarities or [0.0] * len(template_mols)
            )
        else:
            logger.warning("No protein structure information provided - using original template coordinates (poses may be misaligned)")

        # Generate conformers with MCS-based constraints using aligned templates
        logger.info("Step 5b: Finding MCS and generating conformers with aligned templates")
        conformers, mcs_info = generate_conformers(
            query_mol, aligned_template_mols, n_conformers=num_conformers, n_workers=n_workers
        )

        # Store MCS info in results
        if mcs_info:
            logger.info(f"MCS info captured for template selection")

        if conformers is None:
            # Check if fallback info is available in mcs_info
            if mcs_info and isinstance(mcs_info, dict) and mcs_info.get("fallback_used"):
                fallback_type = mcs_info.get("fallback_used")
                logger.error(f"Conformer generation failed even with {fallback_type} fallback strategy")
                raise RuntimeError(f"All conformer generation strategies failed (including {fallback_type} fallback)")
            else:
                logger.error("Conformer generation failed without attempting fallback strategies")
                raise RuntimeError("Failed to generate conformers - no fallback strategies were attempted")

        n_generated = conformers.GetNumConformers()
        logger.info(f"Generated {n_generated}/{num_conformers} conformers")
        
        # Log if fallback was used
        if mcs_info and isinstance(mcs_info, dict) and mcs_info.get("fallback_used"):
            fallback_type = mcs_info.get("fallback_used") 
            logger.warning(f"Used {fallback_type} fallback strategy for conformer generation")

        if n_generated == 0:
            raise RuntimeError("No conformers generated")

        # PERFORMANCE FIX: Score conformers once and extract both best poses and ranked poses
        logger.info(
            f"Scoring and selecting best poses (aligned_poses={use_aligned_poses})"
        )

        # Score all conformers once (get all ranked poses first)
        all_ranked_poses = select_best(
            conformers,
            template_mols[0],
            n_workers=n_workers,
            return_all_ranked=True,  # Get all ranked poses
            no_realign=not use_aligned_poses,
        )

        # Extract best poses from the ranked results (no additional scoring)
        best_poses = {}
        if all_ranked_poses:
            # Group poses by scoring metric and select best for each
            for metric in ["shape", "color", "combo"]:
                # Find best pose for this metric from already-scored poses
                # Note: all_ranked_poses format is (conf_id, scores, mol)
                metric_poses = [
                    (cid, scores, mol)
                    for cid, scores, mol in all_ranked_poses
                    if metric in scores
                ]
                if metric_poses:
                    # Sort by this metric and get the best
                    metric_poses.sort(key=lambda x: x[1][metric], reverse=True)
                    _, best_scores, best_mol = metric_poses[0]  # Fixed unpacking order
                    best_poses[metric] = (best_mol, best_scores)
                    logger.debug(
                        f"Best {metric}: score {best_scores[metric]:.3f} (extracted from ranked results)"
                    )

        if not best_poses:
            raise RuntimeError("Failed to score poses")

        logger.info(
            f"Generated {len(best_poses)} best poses, {len(all_ranked_poses)} total ranked poses"
        )

        # Create results dictionary with MCS info and all poses
        results = {
            "poses": best_poses,
            "mcs_info": mcs_info,
            "all_ranked_poses": all_ranked_poses,
            "alignment_used": use_aligned_poses,
        }

        return results

    def _align_template_molecules(
        self,
        template_mols: List[Any],
        template_pdbs: List[str],
        target_protein_file: Optional[str] = None,
        target_protein_pdb_id: Optional[str] = None,
        target_chain_id: Optional[str] = None,
        template_similarities: Optional[List[float]] = None,
    ) -> List[Any]:
        """
        Align template molecules to target protein binding site using biotite protein alignment.
        
        This is the CRITICAL FIX for the misalignment issue - template ligands are transformed
        to match the target protein's binding site coordinates.
        
        Args:
            template_mols: List of template molecules in original coordinates
            template_pdbs: List of template PDB IDs
            target_protein_file: Path to target protein PDB file
            target_protein_pdb_id: Target protein PDB ID
            target_chain_id: Target protein chain ID from embedding
            template_similarities: List of similarity scores for logging
            
        Returns:
            List of aligned template molecules with transformed coordinates
        """
        from .mcs import transform_ligand
        import biotite.structure.io as bsio
        import numpy as np
        
        aligned_mols = []
        
        try:
            # Load target protein structure
            target_struct = None
            if target_protein_file and os.path.exists(target_protein_file):
                logger.info(f"Loading target protein structure from file: {target_protein_file}")
                target_struct = bsio.load_structure(target_protein_file)
            elif target_protein_pdb_id:
                # Try to find the PDB file in common locations including PDBBind dataset
                possible_paths = [
                    f"data/pdbs/{target_protein_pdb_id.lower()}.pdb",
                    f"templ_pipeline/data/pdbs/{target_protein_pdb_id.lower()}.pdb", 
                    f"/home/ubuntu/mcs/templ_pipeline/data/pdbs/{target_protein_pdb_id.lower()}.pdb",
                    f"data/{target_protein_pdb_id.lower()}.pdb",
                    # PDBBind dataset paths
                    f"data/PDBBind/PDBbind_v2020_refined/refined-set/{target_protein_pdb_id.lower()}/{target_protein_pdb_id.lower()}_protein.pdb",
                    f"data/PDBBind/PDBbind_v2020_other_PL/v2020-other-PL/{target_protein_pdb_id.lower()}/{target_protein_pdb_id.lower()}_protein.pdb",
                    f"templ_pipeline/data/PDBBind/PDBbind_v2020_refined/refined-set/{target_protein_pdb_id.lower()}/{target_protein_pdb_id.lower()}_protein.pdb",
                    f"templ_pipeline/data/PDBBind/PDBbind_v2020_other_PL/v2020-other-PL/{target_protein_pdb_id.lower()}/{target_protein_pdb_id.lower()}_protein.pdb"
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        logger.info(f"Loading target protein structure for {target_protein_pdb_id} from: {path}")
                        target_struct = bsio.load_structure(path)
                        break
                
                if target_struct is None:
                    logger.warning(f"Could not find PDB file for {target_protein_pdb_id} - alignment disabled")
                    return template_mols
            else:
                logger.warning("No target protein structure available - alignment disabled")
                return template_mols
                
            if target_struct is None:
                logger.warning("Failed to load target protein structure - using original template coordinates")
                return template_mols
                
            # Prepare target chains for alignment
            target_chains = [target_chain_id] if target_chain_id else None
            
            # Prioritize self-templates (same PDB as target) for better alignment
            template_items = list(zip(template_mols, template_pdbs, template_similarities or [0.0] * len(template_mols)))
            template_items.sort(key=lambda x: (x[1].upper() != target_protein_pdb_id.upper(), -x[2]))  # Self-templates first, then by similarity
            
            # Update the order to match prioritization (important for MCS selection consistency)
            template_mols = [item[0] for item in template_items]
            template_pdbs = [item[1] for item in template_items]
            template_similarities = [item[2] for item in template_items]
            
            # Align each template molecule
            for i, (template_mol, template_pdb, similarity) in enumerate(template_items):
                try:
                    # Find template PDB file
                    template_pdb_file = None
                    possible_template_paths = [
                        f"data/pdbs/{template_pdb.lower()}.pdb",
                        f"templ_pipeline/data/pdbs/{template_pdb.lower()}.pdb",
                        f"/home/ubuntu/mcs/templ_pipeline/data/pdbs/{template_pdb.lower()}.pdb",
                        f"data/{template_pdb.lower()}.pdb",
                        # PDBBind dataset paths for templates
                        f"data/PDBBind/PDBbind_v2020_refined/refined-set/{template_pdb.lower()}/{template_pdb.lower()}_protein.pdb",
                        f"data/PDBBind/PDBbind_v2020_other_PL/v2020-other-PL/{template_pdb.lower()}/{template_pdb.lower()}_protein.pdb",
                        f"templ_pipeline/data/PDBBind/PDBbind_v2020_refined/refined-set/{template_pdb.lower()}/{template_pdb.lower()}_protein.pdb",
                        f"templ_pipeline/data/PDBBind/PDBbind_v2020_other_PL/v2020-other-PL/{template_pdb.lower()}/{template_pdb.lower()}_protein.pdb"
                    ]
                    
                    for path in possible_template_paths:
                        if os.path.exists(path):
                            template_pdb_file = path
                            break
                    
                    if not template_pdb_file:
                        logger.warning(f"Could not find PDB file for template {template_pdb} - using original coordinates")
                        aligned_mols.append(template_mol)
                        continue
                    
                    # Similarity score already extracted from tuple above
                    
                    # Transform ligand using biotite protein alignment
                    logger.debug(f"Aligning template {template_pdb} (similarity: {similarity:.3f}) to target binding site")
                    
                    # Get template chain information from embedding database
                    template_chain_id = None
                    try:
                        embedding_manager = self._get_embedding_manager()
                        if embedding_manager.has_embedding(template_pdb):
                            _, template_chain_id = embedding_manager.get_embedding(template_pdb)
                            logger.debug(f"Retrieved chain {template_chain_id} for template {template_pdb}")
                        else:
                            logger.debug(f"Template {template_pdb} not found in embedding database")
                    except Exception as e:
                        logger.debug(f"Could not get chain info for template {template_pdb}: {e}")
                    
                    # Use embedding-specified chains for both target and template
                    mob_chains = [template_chain_id] if template_chain_id else None
                    logger.debug(f"Template {template_pdb}: using mob_chains={mob_chains}, ref_chains={target_chains}")
                    
                    aligned_mol = transform_ligand(
                        mob_pdb=template_pdb_file,
                        lig=template_mol,
                        pid=template_pdb,
                        ref_struct=target_struct,
                        ref_chains=target_chains,
                        mob_chains=mob_chains,  # Use embedding-specified chain
                        similarity_score=similarity
                    )
                    
                    if aligned_mol is not None:
                        # Check if alignment produced reasonable coordinates
                        conf = aligned_mol.GetConformer()
                        coords = np.array([conf.GetAtomPosition(i) for i in range(aligned_mol.GetNumAtoms())])
                        center = np.mean(coords, axis=0)
                        
                        # If template is same as target (self-template), expect similar coordinates
                        if template_pdb.upper() == target_protein_pdb_id.upper():
                            # For self-templates, coordinates should be very close to original
                            aligned_mols.append(aligned_mol)
                            logger.debug(f"Self-template {template_pdb} aligned successfully")
                        else:
                            # For different templates, check if coordinates are reasonable
                            # Use a more general approach: reject if coordinates are extremely far from reasonable range
                            coord_magnitude = np.linalg.norm(center)
                            
                            # Also check if they're clearly wrong by being too far from template's self-alignment
                            if template_pdb.upper() == target_protein_pdb_id.upper():
                                # This is handled above
                                pass
                            elif coord_magnitude > 300:  # Coordinates too far from origin are likely wrong alignment
                                logger.warning(f"Template {template_pdb} alignment produced extreme coordinates (center: {center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) - SKIPPING")
                                continue
                            else:
                                aligned_mols.append(aligned_mol)
                                logger.debug(f"Successfully aligned template {template_pdb} (center: {center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
                    else:
                        logger.warning(f"Failed to align template {template_pdb} - using original coordinates")
                        aligned_mols.append(template_mol)
                        
                except Exception as e:
                    logger.warning(f"Error aligning template {template_pdb}: {e} - using original coordinates")
                    aligned_mols.append(template_mol)
                    
            alignment_success_rate = sum(1 for mol in aligned_mols if mol.HasProp("template_pdb")) / len(aligned_mols) if aligned_mols else 0
            logger.info(f"Template alignment complete: {len(aligned_mols)} molecules processed, {alignment_success_rate:.1%} successfully aligned")
            logger.info(f"DEBUG_ALIGNMENT_FIX: Using fixed alignment code with embedding-specified chains")
            
            return aligned_mols if aligned_mols else template_mols
            
        except Exception as e:
            logger.error(f"Error in template alignment: {e} - using original coordinates")
            return template_mols

    def save_results(
        self,
        poses: Dict[str, Tuple[Any, Dict[str, float]]],
        template_pdb: str = "unknown",
        target_pdb: Optional[str] = None,
        ligand_smiles: Optional[str] = None,
        ligand_file: Optional[str] = None,
        template_source: Optional[str] = None,
        batch_id: Optional[str] = None,
        custom_prefix: Optional[str] = None,
        generate_fair_metadata: bool = True,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Save poses to SDF file using adaptive file naming system.

        Args:
            poses: Dictionary of poses from generate_poses
            template_pdb: Template PDB ID for metadata (kept for backwards compatibility)
            target_pdb: Target PDB ID or identifier
            ligand_smiles: SMILES string of the ligand
            ligand_file: Path to ligand file (if applicable)
            template_source: Source of templates ('database', 'sdf', 'custom')
            batch_id: Batch identifier for batch processing
            custom_prefix: Custom prefix for filename
            generate_fair_metadata: Whether to generate FAIR metadata (default: True)
            output_dir: Custom output directory (overrides default timestamped directory)

        Returns:
            Path to the saved SDF file
        """
        if not poses:
            logger.warning("No poses to save")
            return ""

        try:
            # Create output manager and prediction context
            from ..core.output_manager import OutputManager, PredictionContext

            # Use custom output directory if provided, otherwise use default timestamped directory
            if output_dir:
                output_manager = OutputManager(output_dir=output_dir, run_id="")
            else:
                output_manager = OutputManager()

            # Create prediction context for adaptive naming
            context = PredictionContext(
                pdb_id=target_pdb,
                smiles=ligand_smiles,
                input_file=ligand_file,
                template_source=template_source or "database",
                batch_id=batch_id,
                custom_prefix=custom_prefix,
            )

            # Generate filename using adaptive naming
            filename = output_manager.generate_output_filename(context, "poses", "sdf")

            # Create SDF writer
            Chem, _ = self._get_rdkit()
            writer = Chem.SDWriter(filename)

            best_scores = {}
            poses_written = 0

            # Write poses to file
            for pose_id, (mol, scores) in poses.items():
                try:
                    # Add pose metadata to molecule
                    mol.SetProp("_Name", pose_id)
                    mol.SetProp("Template_PDB", template_pdb)
                    if target_pdb:
                        mol.SetProp("Target_PDB", target_pdb)

                    # Add scores as properties
                    for score_name, score_value in scores.items():
                        mol.SetProp(score_name, str(score_value))

                        # Track best scores
                        if (
                            score_name not in best_scores
                            or score_value > best_scores[score_name]
                        ):
                            best_scores[score_name] = score_value

                    writer.write(mol)
                    poses_written += 1

                except Exception as e:
                    logger.warning(f"Failed to write pose {pose_id}: {e}")
                    continue

            writer.close()

            # Generate FAIR metadata if requested
            if generate_fair_metadata:
                try:
                    self._generate_fair_metadata(
                        filename=filename,
                        context=context,
                        poses_count=poses_written,
                        best_scores=best_scores,
                        template_pdb=template_pdb,
                        ligand_smiles=ligand_smiles,
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate FAIR metadata: {e}")

            logger.info(f"Saved {poses_written} poses to {filename}")
            return filename

        except Exception as e:
            logger.error(f"Error saving poses: {e}")
            # Fallback to simple naming if adaptive naming fails
            fallback_filename = "poses.sdf"
            try:
                Chem, _ = self._get_rdkit()
                writer = Chem.SDWriter(fallback_filename)
                for pose_id, (mol, scores) in poses.items():
                    mol.SetProp("_Name", pose_id)
                    writer.write(mol)
                writer.close()
                return fallback_filename
            except:
                return ""

    def _generate_fair_metadata(
        self,
        filename: str,
        context,
        poses_count: int,
        best_scores: Dict[str, float],
        template_pdb: str,
        ligand_smiles: Optional[str],
    ) -> None:
        """
        Generate comprehensive FAIR metadata for the prediction results.

        Args:
            filename: Output filename
            context: PredictionContext object
            poses_count: Number of poses generated
            best_scores: Best scoring values
            template_pdb: Template PDB identifier
            ligand_smiles: SMILES string of ligand
        """
        try:
            from ..fair.core.metadata_engine import MetadataEngine
            from ..fair.biology.molecular_descriptors import (
                calculate_comprehensive_descriptors,
            )

            # Create metadata engine
            metadata_engine = MetadataEngine(pipeline_version="2.0.0")

            # Determine input type and value
            input_type = "unknown"
            input_value = ""
            template_identifiers = []

            if context.pdb_id:
                input_type = "pdb_id"
                input_value = context.pdb_id
            elif ligand_smiles:
                input_type = "smiles"
                input_value = ligand_smiles
            elif context.input_file:
                input_type = "sdf_file"
                input_value = context.input_file

            if template_pdb and template_pdb != "unknown":
                template_identifiers = [template_pdb]

            # Create input metadata
            input_metadata = metadata_engine.create_input_metadata(
                target_identifier=context.pdb_id,
                input_type=input_type,
                input_value=input_value,
                input_file=context.input_file,
                template_source=context.template_source,
                template_identifiers=template_identifiers,
                parameters={
                    "batch_id": context.batch_id,
                    "custom_prefix": context.custom_prefix,
                    "template_pdb": template_pdb,
                },
            )

            # Create output metadata
            output_metadata = metadata_engine.create_output_metadata(
                primary_output=filename,
                output_format="sdf",
                poses_generated=poses_count,
                best_scores=best_scores,
            )

            # Generate molecular descriptors if SMILES available
            biological_context = {}
            if ligand_smiles:
                try:
                    descriptors = calculate_comprehensive_descriptors(ligand_smiles)
                    if descriptors.get("calculation_success"):
                        biological_context["molecular_descriptors"] = descriptors
                except Exception as e:
                    logger.warning(f"Failed to calculate molecular descriptors: {e}")

            # Create scientific metadata
            scientific_metadata = metadata_engine.create_scientific_metadata(
                biological_context=biological_context
            )

            # Create provenance record
            provenance_record = metadata_engine.create_provenance_record(
                input_metadata, output_metadata, scientific_metadata
            )

            # Export metadata to JSON
            metadata_path = filename.replace(".sdf", "_metadata.json")
            metadata_engine.export_metadata(
                provenance_record, metadata_path, format="json"
            )

            logger.info(f"FAIR metadata saved to {metadata_path}")

        except Exception as e:
            logger.error(f"Error generating FAIR metadata: {e}")
            # Don't raise exception - metadata generation failure shouldn't stop pipeline

    def run_full_pipeline(
        self,
        protein_file: Optional[str] = None,
        protein_pdb_id: Optional[str] = None,
        ligand_smiles: Optional[str] = None,
        ligand_file: Optional[str] = None,
        num_templates: int = 100,
        num_conformers: int = 200,
        n_workers: int = None,
        similarity_threshold: Optional[float] = None,
        use_aligned_poses: bool = True,
        exclude_pdb_ids: Optional[Set[str]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete TEMPL pipeline.

        Args:
            protein_file: Path to protein PDB file
            protein_pdb_id: PDB ID (alternative to protein_file)
            ligand_smiles: SMILES string for query ligand
            ligand_file: Path to ligand file
            num_templates: Number of templates to find
            num_conformers: Number of conformers to generate
            n_workers: Number of worker processes
            similarity_threshold: Minimum similarity threshold for templates
            use_aligned_poses: If True, return aligned poses. If False, return original conformers.
            exclude_pdb_ids: Set of PDB IDs to exclude from template search
            output_dir: Custom output directory for saving results

        Returns:
            Dictionary with pipeline results
        """
        # Auto-detect workers if not specified
        if n_workers is None:
            try:
                from .hardware_utils import get_suggested_worker_config

                n_workers = get_suggested_worker_config()["n_workers"]
                logger.info(f"Auto-detected {n_workers} workers based on hardware")
            except ImportError:
                n_workers = 4  # Conservative fallback
                logger.warning("Hardware detection unavailable, using 4 workers")

        results = {}
        embedding = None
        template_mols = None
        query_mol = None

        try:
            # Step 1: Generate or get protein embedding
            if protein_file:
                logger.info("Step 1: Generating protein embedding")
                embedding, chain_id = self.generate_embedding(protein_file)
                results["embedding"] = embedding
                results["chain_id"] = chain_id
            elif protein_pdb_id:
                logger.info("Step 1: Getting protein embedding from database")
                embedding_manager = self._get_embedding_manager()

                # Fast path check
                if embedding_manager.has_embedding(protein_pdb_id):
                    logger.info(
                        f"SUCCESS: Fast path: Using cached embedding for {protein_pdb_id}"
                    )
                    embedding, chain_id = embedding_manager.get_embedding(
                        protein_pdb_id
                    )
                    if embedding is not None:
                        logger.info(
                            f"Retrieved embedding with shape {embedding.shape} for chain {chain_id or 'A'}"
                        )
                    else:
                        raise ValueError(
                            f"Failed to retrieve embedding for PDB ID {protein_pdb_id}"
                        )
                else:
                    logger.warning(
                        f"PDB {protein_pdb_id} not in database - this should not happen for known PDBs"
                    )
                    raise ValueError(f"PDB ID {protein_pdb_id} not found in database")

                results["embedding"] = embedding
                results["chain_id"] = chain_id
            else:
                raise ValueError(
                    "Either protein_file or protein_pdb_id must be provided"
                )

            # Step 2: Find templates
            logger.info("Step 2: Finding similar templates")
            # Allow self as template when using PDB ID input
            allow_self = protein_pdb_id is not None
            templates = self.find_templates(
                protein_embedding=embedding,
                num_templates=num_templates,
                similarity_threshold=similarity_threshold,
                allow_self_as_template=allow_self,
                exclude_pdb_ids=exclude_pdb_ids,
            )
            results["templates"] = templates

            if not templates:
                raise RuntimeError("No templates found")

            # Step 3: Load template molecules
            logger.info("Step 3: Loading template molecules")
            template_pdbs = [t[0] for t in templates]  # Use all templates
            template_mols = self.load_template_molecules(template_pdbs)
            results["template_molecules"] = template_mols

            if not template_mols:
                raise RuntimeError("No template molecules loaded")

            # Step 4: Prepare query molecule
            logger.info("Step 4: Preparing query molecule")
            query_mol = self.prepare_query_molecule(ligand_smiles, ligand_file)
            results["query_molecule"] = query_mol

            # Step 5: Generate poses
            logger.info(
                f"Step 5: Generating poses (alignment={'enabled' if use_aligned_poses else 'disabled'})"
            )
            pose_results = self.generate_poses(
                query_mol,
                template_mols,
                num_conformers,
                n_workers,
                use_aligned_poses=use_aligned_poses,
                target_protein_file=protein_file,
                target_protein_pdb_id=protein_pdb_id,
                target_chain_id=results.get("chain_id"),
                template_pdbs=template_pdbs,
                template_similarities=[t[1] for t in templates] if templates else None,
            )

            # Handle new dict format from generate_poses
            if isinstance(pose_results, dict):
                results["poses"] = pose_results["poses"]
                results["mcs_info"] = pose_results.get("mcs_info")
                results["all_ranked_poses"] = pose_results.get("all_ranked_poses")
                results["alignment_used"] = pose_results.get(
                    "alignment_used", use_aligned_poses
                )
            else:
                results["poses"] = pose_results  # Backward compatibility
                results["alignment_used"] = use_aligned_poses

            # Step 6: Save results
            logger.info("Step 6: Saving results")
            output_file = self.save_results(
                results["poses"],
                template_pdbs[0],
                target_pdb=protein_pdb_id if protein_pdb_id else None,
                ligand_smiles=ligand_smiles,
                ligand_file=ligand_file,
                template_source="database" if protein_pdb_id else None,
                batch_id=None,
                custom_prefix=None,
                output_dir=output_dir,
            )
            results["output_file"] = output_file

            logger.info(
                f"Pipeline completed successfully (alignment_used={results.get('alignment_used', 'unknown')})"
            )
            return results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            results["error"] = str(e)
            raise
        finally:
            # Explicit cleanup of large objects
            self._cleanup_pipeline_objects(embedding, template_mols, query_mol)

    def _cleanup_pipeline_objects(
        self, embedding=None, template_mols=None, query_mol=None
    ):
        """Clean up large pipeline objects to prevent memory accumulation."""
        import gc

        try:
            if embedding is not None:
                del embedding
            if template_mols is not None:
                for mol in template_mols:
                    if mol is not None:
                        del mol
                del template_mols
            if query_mol is not None:
                del query_mol

            # Force garbage collection
            gc.collect()

        except Exception as e:
            logger.debug(f"Error during pipeline cleanup: {e}")
