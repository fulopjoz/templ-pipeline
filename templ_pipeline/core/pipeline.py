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

RDLogger.DisableLog("rdApp.*")
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

logger = logging.getLogger(__name__)

# Import enhanced scoring components
try:
    from .scoring import FixedMolecularProcessor, ScoringFixer

    logger.debug("Enhanced scoring components loaded")
except ImportError:
    logger.warning("Enhanced scoring components not available")

# Import unified workspace management
try:
    from .unified_workspace_manager import UnifiedWorkspaceManager, WorkspaceConfig
    
    logger.debug("Unified workspace manager loaded")
except ImportError:
    logger.warning("Unified workspace manager not available")


class TEMPLPipeline:
    """Main pipeline orchestrator for TEMPL."""

    def __init__(
        self,
        embedding_path: Optional[str] = None,
        output_dir: str = "workspace",
        run_id: Optional[str] = None,
        auto_cleanup: bool = False,
        use_unified_workspace: bool = True,
        workspace_config: Optional[WorkspaceConfig] = None,
    ):
        """
        Initialize the TEMPL pipeline.

        Args:
            embedding_path: Path to embedding database
            output_dir: Base directory for workspace (formerly output only)
            run_id: Custom run identifier (default: timestamp)
            auto_cleanup: Whether to automatically clean up temp files on exit
            use_unified_workspace: Whether to use the new unified workspace manager
            workspace_config: Configuration for workspace management
        """
        self.embedding_path = embedding_path
        self.auto_cleanup = auto_cleanup
        self._cleanup_registered = False
        self.use_unified_workspace = use_unified_workspace

        # Initialize workspace management
        if self.use_unified_workspace:
            try:
                # Create workspace configuration
                if workspace_config is None:
                    workspace_config = WorkspaceConfig(
                        base_dir=output_dir,
                        auto_cleanup=auto_cleanup
                    )
                
                # Initialize unified workspace manager
                self.workspace_manager = UnifiedWorkspaceManager(
                    run_id=run_id,
                    config=workspace_config
                )
                
                # For backward compatibility, expose output_dir
                self.output_dir = self.workspace_manager.output_dir
                self.run_id = self.workspace_manager.run_id
                
                logger.info(f"Initialized unified workspace: {self.workspace_manager.run_dir}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize unified workspace: {e}")
                logger.info("Falling back to legacy output directory management")
                self.use_unified_workspace = False
                self._init_legacy_output_dir(output_dir, run_id)
        else:
            self._init_legacy_output_dir(output_dir, run_id)

        # Register cleanup if requested
        if self.auto_cleanup and not self._cleanup_registered:
            import atexit
            atexit.register(self._cleanup_workspace)
            self._cleanup_registered = True
            logger.debug(f"Registered auto-cleanup for workspace")

        # Initialize components lazily
        self._embedding_manager = None
        self._rdkit_modules = None

    def _init_legacy_output_dir(self, output_dir: str, run_id: Optional[str]):
        """Initialize legacy output directory management."""
        # Create timestamped output directory (legacy behavior)
        if run_id:
            self.output_dir = Path(f"{output_dir}_{run_id}")
            self.run_id = run_id
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"{output_dir}_{timestamp}")
            self.run_id = timestamp

        self.output_dir.mkdir(exist_ok=True)
        self.workspace_manager = None
        logger.info(f"Initialized legacy output directory: {self.output_dir}")

    def _cleanup_workspace(self):
        """Clean up workspace using appropriate method."""
        if self.use_unified_workspace and self.workspace_manager:
            try:
                self.workspace_manager.cleanup_temp_files()
                logger.debug(f"Cleaned up workspace temp files: {self.workspace_manager.run_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up workspace: {e}")
        else:
            self._cleanup_output_dir()

    def _cleanup_output_dir(self):
        """Clean up the output directory (legacy method)."""
        if self.output_dir and self.output_dir.exists():
            try:
                import shutil

                shutil.rmtree(self.output_dir)
                logger.debug(f"Cleaned up output directory: {self.output_dir}")
            except Exception as e:
                logger.warning(
                    f"Failed to clean up output directory {self.output_dir}: {e}"
                )

    def cleanup(self):
        """Manually clean up workspace or output directory."""
        self._cleanup_workspace()

    def get_temp_file(self, prefix: str = "templ", suffix: str = ".tmp", category: str = "processing") -> str:
        """
        Get a temporary file path using workspace manager.
        
        Args:
            prefix: File prefix
            suffix: File extension  
            category: File category ('uploaded', 'processing', 'cache')
            
        Returns:
            Path to temporary file
        """
        if self.use_unified_workspace and self.workspace_manager:
            return self.workspace_manager.get_temp_file(prefix, suffix, category)
        else:
            # Fallback to tempfile for legacy mode
            import tempfile
            fd, temp_path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=suffix)
            os.close(fd)
            return temp_path

    def save_uploaded_file(self, content: bytes, filename: str, secure_hash: Optional[str] = None) -> str:
        """
        Save uploaded file using workspace manager.
        
        Args:
            content: File content
            filename: Original filename
            secure_hash: Optional security hash
            
        Returns:
            Path to saved file
        """
        if self.use_unified_workspace and self.workspace_manager:
            return self.workspace_manager.save_uploaded_file(content, filename, secure_hash)
        else:
            # Fallback to output directory for legacy mode
            import hashlib
            content_hash = hashlib.sha256(content).hexdigest()[:16]
            safe_filename = f"{content_hash}_{int(datetime.now().timestamp())}{Path(filename).suffix}"
            file_path = self.output_dir / safe_filename
            file_path.write_bytes(content)
            return str(file_path)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.auto_cleanup:
            self._cleanup_workspace()

    def _get_embedding_manager(self):
        """Lazy initialization of embedding manager."""
        if self._embedding_manager is None:
            from .embedding import EmbeddingManager

            if self.embedding_path is None:
                # Try default paths
                default_paths = [
                    "data/embeddings/protein_embeddings_base.npz",
                    "templ_pipeline/data/embeddings/protein_embeddings_base.npz",
                    "/home/ubuntu/mcs/templ_pipeline/data/embeddings/protein_embeddings_base.npz",
                ]

                for path in default_paths:
                    if os.path.exists(path):
                        self.embedding_path = path
                        break

                if self.embedding_path is None:
                    raise FileNotFoundError(
                        "No embedding database found. Please specify embedding_path."
                    )

            self._embedding_manager = EmbeddingManager(self.embedding_path)

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
        sdf_path = "templ_pipeline/data/ligands/processed_ligands_new_unzipped.sdf"

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
    ) -> Dict[str, Tuple[Any, Dict[str, float]]]:
        """
        Generate poses using template-based conformer generation.

        Args:
            query_mol: Query molecule
            template_mols: List of template molecules
            num_conformers: Number of conformers to generate
            n_workers: Number of worker processes
            use_aligned_poses: If True, return aligned poses. If False, return original conformers with scores.

        Returns:
            Dictionary with poses and metadata, including MCS info
        """
        from .mcs import generate_conformers, find_mcs
        from .scoring import select_best

        if not template_mols:
            raise ValueError("No template molecules provided")

        logger.info(
            f"Generating poses with {len(template_mols)} templates, {num_conformers} conformers, {n_workers} workers"
        )

        # Generate conformers with MCS-based constraints
        logger.info("Finding MCS and generating conformers")
        conformers, mcs_info = generate_conformers(
            query_mol, template_mols, n_conformers=num_conformers, n_workers=n_workers
        )

        # Store MCS info in results
        if mcs_info:
            logger.info(f"MCS info captured for template selection")

        if conformers is None:
            raise RuntimeError("Failed to generate conformers")

        n_generated = conformers.GetNumConformers()
        logger.info(f"Generated {n_generated}/{num_conformers} conformers")

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
    ) -> str:
        """
        Save poses to SDF file using unified workspace management.

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

        Returns:
            Path to the saved SDF file
        """
        if not poses:
            logger.warning("No poses to save")
            return ""

        try:
            # Generate filename using unified workspace approach
            if self.use_unified_workspace and self.workspace_manager:
                # Use workspace manager for output file naming
                if custom_prefix:
                    base_filename = f"{custom_prefix}_poses"
                elif target_pdb:
                    base_filename = f"{target_pdb}_poses"
                elif template_pdb and template_pdb != "unknown":
                    base_filename = f"{template_pdb}_poses"
                else:
                    base_filename = "poses"
                
                sdf_filename = f"{base_filename}.sdf"
                
                # Create SDF content
                sdf_content = self._create_sdf_content(poses, template_pdb, target_pdb)
                
                # Save using workspace manager
                output_path = self.workspace_manager.save_output(
                    sdf_filename, 
                    sdf_content, 
                    category="result"
                )
                
                poses_written = len(poses)
                best_scores = self._extract_best_scores(poses)
                
            else:
                # Fallback to legacy approach with output manager
                output_path = self._save_results_legacy(
                    poses, template_pdb, target_pdb, ligand_smiles, 
                    ligand_file, template_source, batch_id, custom_prefix
                )
                poses_written = len(poses)
                best_scores = self._extract_best_scores(poses)

            # Generate comprehensive metadata if requested
            if generate_fair_metadata:
                try:
                    metadata = self._create_comprehensive_metadata(
                        output_path=output_path,
                        poses_count=poses_written,
                        best_scores=best_scores,
                        template_pdb=template_pdb,
                        target_pdb=target_pdb,
                        ligand_smiles=ligand_smiles,
                        ligand_file=ligand_file,
                        template_source=template_source,
                        batch_id=batch_id
                    )
                    
                    # Save metadata using workspace manager
                    if self.use_unified_workspace and self.workspace_manager:
                        base_name = Path(output_path).stem
                        self.workspace_manager.save_metadata(base_name, metadata)
                    else:
                        # Legacy metadata saving
                        metadata_path = output_path.replace(".sdf", "_metadata.json")
                        with open(metadata_path, 'w') as f:
                            import json
                            json.dump(metadata, f, indent=2)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate metadata: {e}")

            logger.info(f"Saved {poses_written} poses to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving poses: {e}")
            # Final fallback to simple SDF file
            return self._save_fallback_sdf(poses)

    def _create_sdf_content(self, poses: Dict[str, Tuple[Any, Dict[str, float]]], 
                           template_pdb: str, target_pdb: Optional[str]) -> str:
        """Create SDF content from poses dictionary."""
        Chem, _ = self._get_rdkit()
        
        # Create temporary file to write SDF content
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.sdf', delete=False) as tmp:
            writer = Chem.SDWriter(tmp.name)
            
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

                    writer.write(mol)

                except Exception as e:
                    logger.warning(f"Failed to process pose {pose_id}: {e}")
                    continue
            
            writer.close()
            
            # Read the content back
            with open(tmp.name, 'r') as f:
                content = f.read()
            
            # Clean up temporary file
            os.unlink(tmp.name)
            
            return content

    def _extract_best_scores(self, poses: Dict[str, Tuple[Any, Dict[str, float]]]) -> Dict[str, float]:
        """Extract best scores from poses dictionary."""
        best_scores = {}
        
        for pose_id, (mol, scores) in poses.items():
            for score_name, score_value in scores.items():
                if (score_name not in best_scores or 
                    score_value > best_scores[score_name]):
                    best_scores[score_name] = score_value
        
        return best_scores

    def _create_comprehensive_metadata(self, output_path: str, poses_count: int,
                                     best_scores: Dict[str, float], template_pdb: str,
                                     target_pdb: Optional[str], ligand_smiles: Optional[str],
                                     ligand_file: Optional[str], template_source: Optional[str],
                                     batch_id: Optional[str]) -> Dict[str, Any]:
        """Create comprehensive metadata for the prediction results."""
        
        metadata = {
            "pipeline_info": {
                "version": "2.0.0",
                "run_id": self.run_id,
                "timestamp": datetime.now().isoformat(),
                "workspace_unified": self.use_unified_workspace
            },
            "input": {
                "target_pdb": target_pdb,
                "ligand_smiles": ligand_smiles,
                "ligand_file": ligand_file,
                "template_source": template_source or "database",
                "batch_id": batch_id
            },
            "output": {
                "file_path": output_path,
                "format": "sdf",
                "poses_count": poses_count,
                "best_scores": best_scores
            },
            "templates": {
                "primary_template": template_pdb
            },
            "processing": {
                "workspace_structure": "unified" if self.use_unified_workspace else "legacy"
            }
        }
        
        # Add molecular descriptors if SMILES available
        if ligand_smiles:
            try:
                # Try to calculate basic descriptors
                from rdkit import Descriptors
                from rdkit.Chem import Crippen
                
                Chem, _ = self._get_rdkit()
                mol = Chem.MolFromSmiles(ligand_smiles)
                if mol:
                    metadata["molecular_properties"] = {
                        "molecular_weight": Descriptors.MolWt(mol),
                        "logp": Crippen.MolLogP(mol),
                        "num_atoms": mol.GetNumAtoms(),
                        "num_bonds": mol.GetNumBonds(),
                        "num_rings": Descriptors.RingCount(mol)
                    }
            except Exception as e:
                logger.debug(f"Could not calculate molecular descriptors: {e}")
        
        return metadata

    def _save_results_legacy(self, poses, template_pdb, target_pdb, ligand_smiles,
                           ligand_file, template_source, batch_id, custom_prefix) -> str:
        """Legacy save method using OutputManager if available."""
        try:
            from ..core.output_manager import OutputManager, PredictionContext

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

        except ImportError:
            # Fallback to simple naming if OutputManager not available
            if custom_prefix:
                filename = f"{custom_prefix}_poses.sdf"
            elif target_pdb:
                filename = f"{target_pdb}_poses.sdf"
            else:
                filename = f"poses_{self.run_id}.sdf"
            
            # Use output directory
            filename = str(self.output_dir / filename)

        # Create SDF writer
        Chem, _ = self._get_rdkit()
        writer = Chem.SDWriter(filename)

        # Write poses to file
        for pose_id, (mol, scores) in poses.items():
            try:
                mol.SetProp("_Name", pose_id)
                mol.SetProp("Template_PDB", template_pdb)
                if target_pdb:
                    mol.SetProp("Target_PDB", target_pdb)

                for score_name, score_value in scores.items():
                    mol.SetProp(score_name, str(score_value))

                writer.write(mol)

            except Exception as e:
                logger.warning(f"Failed to write pose {pose_id}: {e}")
                continue

        writer.close()
        return filename

    def _save_fallback_sdf(self, poses: Dict[str, Tuple[Any, Dict[str, float]]]) -> str:
        """Final fallback method to save poses."""
        try:
            fallback_filename = str(self.output_dir / "poses_fallback.sdf")
            Chem, _ = self._get_rdkit()
            writer = Chem.SDWriter(fallback_filename)
            
            for pose_id, (mol, scores) in poses.items():
                mol.SetProp("_Name", pose_id)
                writer.write(mol)
            
            writer.close()
            logger.warning(f"Used fallback SDF save: {fallback_filename}")
            return fallback_filename
            
        except Exception as e:
            logger.error(f"Even fallback save failed: {e}")
            return ""



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
