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

RDLogger.DisableLog('rdApp.*')
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
    
    def __init__(self, embedding_path: Optional[str] = None, output_dir: str = "output", run_id: Optional[str] = None):
        """
        Initialize the TEMPL pipeline.
        
        Args:
            embedding_path: Path to embedding database
            output_dir: Base directory for output files
            run_id: Custom run identifier (default: timestamp)
        """
        self.embedding_path = embedding_path
        
        # Create timestamped output directory
        if run_id:
            self.output_dir = Path(f"{output_dir}_{run_id}")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"{output_dir}_{timestamp}")
        
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components lazily
        self._embedding_manager = None
        self._rdkit_modules = None
        
    def _get_embedding_manager(self):
        """Lazy initialization of embedding manager."""
        if self._embedding_manager is None:
            from .embedding import EmbeddingManager
            
            if self.embedding_path is None:
                # Try default paths
                default_paths = [
                    "data/embeddings/protein_embeddings_base.npz",
                    "templ_pipeline/data/embeddings/protein_embeddings_base.npz",
                    "/home/ubuntu/mcs/templ_pipeline/data/embeddings/protein_embeddings_base.npz"
                ]
                
                for path in default_paths:
                    if os.path.exists(path):
                        self.embedding_path = path
                        break
                
                if self.embedding_path is None:
                    raise FileNotFoundError("No embedding database found. Please specify embedding_path.")
            
            self._embedding_manager = EmbeddingManager(self.embedding_path)
            
        return self._embedding_manager
    
    def _get_rdkit(self):
        """Lazy initialization of RDKit modules."""
        if self._rdkit_modules is None:
            from rdkit import Chem, RDLogger
            from rdkit.Chem import AllChem
            RDLogger.DisableLog('rdApp.*')
            self._rdkit_modules = (Chem, AllChem)
        return self._rdkit_modules
    
    def generate_embedding(self, protein_file: str, chain: Optional[str] = None) -> Tuple[Any, str]:
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
                    logger.info(f"Retrieved cached embedding with shape {embedding.shape} for chain {chain_id or 'A'}")
                    return embedding, chain_id or "A"
        
        # Fallback to generating new embedding
        if pdb_id:
            logger.info(f"PDB {pdb_id} not in cache, generating new embedding")
        else:
            logger.info("Could not extract PDB ID from filename, generating new embedding")
        
        embedding, chain_id = get_protein_embedding(protein_file, chain)
        
        if embedding is None:
            raise RuntimeError("Failed to generate protein embedding")
        
        logger.info(f"Generated embedding with shape {embedding.shape} for chain {chain_id}")
        return embedding, chain_id
    
    def find_templates(self, 
                      protein_file: Optional[str] = None,
                      protein_embedding: Optional[Any] = None,
                      num_templates: int = 10,
                      similarity_threshold: Optional[float] = None,
                      allow_self_as_template: bool = False,
                      exclude_pdb_ids: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
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
                raise ValueError("Either protein_file or protein_embedding must be provided")
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
            allow_self_as_template=allow_self_as_template
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
                raise FileNotFoundError(f"Template loading failed: {loading_stats['error']}")
            
            if loading_stats.get("missing_pdbs"):
                missing_pdbs = loading_stats["missing_pdbs"]
                logger.warning(f"Could not find templates: {', '.join(missing_pdbs)}")
            
            logger.info(f"Loaded {len(templates)} template molecules using standardized loader")
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
        sdf_path = 'templ_pipeline/data/ligands/processed_ligands_new_unzipped.sdf'
        
        if not os.path.exists(sdf_path):
            logger.error(f"Database file not found: {sdf_path}")
            return []
        
        try:
            supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
            
            for mol in supplier:
                if mol is None:
                    continue
                
                # Get molecule name
                mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else ''
                
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
    
    def prepare_query_molecule(self, 
                              ligand_smiles: Optional[str] = None,
                              ligand_file: Optional[str] = None) -> Any:
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
            
            if ligand_file.endswith('.sdf'):
                supplier = Chem.SDMolSupplier(ligand_file, removeHs=False)
            elif ligand_file.endswith('.mol'):
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
        
        logger.info(f"Query molecule prepared (standardized): {mol.GetNumAtoms()} atoms, {mol.GetNumBonds()} bonds")
        return mol
    
    def generate_poses(self,
                      query_mol: Any,
                      template_mols: List[Any],
                      num_conformers: int = 100,
                      n_workers: int = 4,
                      use_aligned_poses: bool = True) -> Dict[str, Tuple[Any, Dict[str, float]]]:
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
        
        logger.info(f"Generating poses with {len(template_mols)} templates, {num_conformers} conformers, {n_workers} workers")
        
        # Generate conformers with MCS-based constraints
        logger.info("Finding MCS and generating conformers")
        conformers, mcs_info = generate_conformers(
            query_mol, 
            template_mols, 
            n_conformers=num_conformers, 
            n_workers=n_workers
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
        
        # Score and select best poses - use alignment control parameter
        logger.info(f"Scoring and selecting best poses (aligned_poses={use_aligned_poses})")
        best_poses = select_best(
            conformers, 
            template_mols[0], 
            n_workers=n_workers, 
            return_all_ranked=False,
            no_realign=not use_aligned_poses  # Invert the logic: no_realign=True means return original conformers
        )
        
        # Also get all ranked poses for comprehensive download
        all_ranked_poses = select_best(
            conformers, 
            template_mols[0], 
            n_workers=n_workers, 
            return_all_ranked=True,
            no_realign=not use_aligned_poses
        )
        
        if not best_poses:
            raise RuntimeError("Failed to score poses")
        
        logger.info(f"Generated {len(best_poses)} best poses, {len(all_ranked_poses)} total ranked poses")
        
        # Create results dictionary with MCS info and all poses
        results = {
            'poses': best_poses,
            'mcs_info': mcs_info,
            'all_ranked_poses': all_ranked_poses,
            'alignment_used': use_aligned_poses
        }
        
        return results
    
    def save_results(self, poses: Dict[str, Tuple[Any, Dict[str, float]]], template_pdb: str = "unknown", target_pdb: Optional[str] = None) -> str:
        """
        Save poses to SDF file.
        Now includes optional target_pdb so output files are named <pdb_id>_poses.sdf for easier identification.
        
        Args:
            poses: Dictionary of poses from generate_poses
            template_pdb: Template PDB ID for metadata (kept for backwards compatibility)
            target_pdb: Target PDB ID to embed in filename (optional)
        Returns:
            Path to output SDF file
        """
        from .scoring import generate_properties_for_sdf
        Chem, _ = self._get_rdkit()

        # Determine filename – embed PDB ID if provided
        if target_pdb:
            filename = f"{target_pdb.lower()}_poses.sdf"
        else:
            filename = "poses.sdf"

        output_file = self.output_dir / filename
        logger.info(f"Saving poses to {output_file}")

        with Chem.SDWriter(str(output_file)) as writer:
            for method, (pose, scores) in poses.items():
                if pose is None:
                    logger.warning(f"No valid pose for {method}")
                    continue

                # Add properties
                pose_with_props = generate_properties_for_sdf(
                    pose,
                    method,
                    scores.get(method, 0.0),
                    template_pdb,
                    {
                        "shape_score": f"{scores.get('shape', 0.0):.3f}",
                        "color_score": f"{scores.get('color', 0.0):.3f}",
                        "combo_score": f"{scores.get('combo', 0.0):.3f}"
                    }
                )

                writer.write(pose_with_props)

        logger.info(f"Saved poses to {output_file}")
        return str(output_file)
    
    def run_full_pipeline(self,
                         protein_file: Optional[str] = None,
                         protein_pdb_id: Optional[str] = None,
                         ligand_smiles: Optional[str] = None,
                         ligand_file: Optional[str] = None,
                         num_templates: int = 100,
                         num_conformers: int = 200,
                         n_workers: int = None,
                         similarity_threshold: Optional[float] = None,
                         use_aligned_poses: bool = True,
                         exclude_pdb_ids: Optional[Set[str]] = None) -> Dict[str, Any]:
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
                results['embedding'] = embedding
                results['chain_id'] = chain_id
            elif protein_pdb_id:
                logger.info("Step 1: Getting protein embedding from database")
                embedding_manager = self._get_embedding_manager()
                
                # Fast path check
                if embedding_manager.has_embedding(protein_pdb_id):
                    logger.info(f"✅ Fast path: Using cached embedding for {protein_pdb_id}")
                    embedding, chain_id = embedding_manager.get_embedding(protein_pdb_id)
                    if embedding is not None:
                        logger.info(f"Retrieved embedding with shape {embedding.shape} for chain {chain_id or 'A'}")
                    else:
                        raise ValueError(f"Failed to retrieve embedding for PDB ID {protein_pdb_id}")
                else:
                    logger.warning(f"PDB {protein_pdb_id} not in database - this should not happen for known PDBs")
                    raise ValueError(f"PDB ID {protein_pdb_id} not found in database")
                
                results['embedding'] = embedding
                results['chain_id'] = chain_id
            else:
                raise ValueError("Either protein_file or protein_pdb_id must be provided")
            
            # Step 2: Find templates
            logger.info("Step 2: Finding similar templates")
            # Allow self as template when using PDB ID input
            allow_self = protein_pdb_id is not None
            templates = self.find_templates(
                protein_embedding=embedding,
                num_templates=num_templates,
                similarity_threshold=similarity_threshold,
                allow_self_as_template=allow_self,
                exclude_pdb_ids=exclude_pdb_ids
            )
            results['templates'] = templates
            
            if not templates:
                raise RuntimeError("No templates found")
            
            # Step 3: Load template molecules
            logger.info("Step 3: Loading template molecules")
            template_pdbs = [t[0] for t in templates]  # Use all templates
            template_mols = self.load_template_molecules(template_pdbs)
            results['template_molecules'] = template_mols
            
            if not template_mols:
                raise RuntimeError("No template molecules loaded")
            
            # Step 4: Prepare query molecule
            logger.info("Step 4: Preparing query molecule")
            query_mol = self.prepare_query_molecule(ligand_smiles, ligand_file)
            results['query_molecule'] = query_mol
            
            # Step 5: Generate poses
            logger.info(f"Step 5: Generating poses (alignment={'enabled' if use_aligned_poses else 'disabled'})")
            pose_results = self.generate_poses(
                query_mol,
                template_mols,
                num_conformers,
                n_workers,
                use_aligned_poses=use_aligned_poses
            )
            
            # Handle new dict format from generate_poses
            if isinstance(pose_results, dict):
                results['poses'] = pose_results['poses']
                results['mcs_info'] = pose_results.get('mcs_info')
                results['all_ranked_poses'] = pose_results.get('all_ranked_poses')
                results['alignment_used'] = pose_results.get('alignment_used', use_aligned_poses)
            else:
                results['poses'] = pose_results  # Backward compatibility
                results['alignment_used'] = use_aligned_poses
            
            # Step 6: Save results
            logger.info("Step 6: Saving results")
            output_file = self.save_results(
                results['poses'],
                template_pdbs[0],
                target_pdb=protein_pdb_id if protein_pdb_id else None
            )
            results['output_file'] = output_file
            
            logger.info(f"Pipeline completed successfully (alignment_used={results.get('alignment_used', 'unknown')})")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            results['error'] = str(e)
            raise 
        finally:
            # Explicit cleanup of large objects
            self._cleanup_pipeline_objects(embedding, template_mols, query_mol)
    
    def _cleanup_pipeline_objects(self, embedding=None, template_mols=None, query_mol=None):
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