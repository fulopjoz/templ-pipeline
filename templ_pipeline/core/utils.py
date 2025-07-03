"""
TEMPL Pipeline Utility Functions

This module provides utility functions for the TEMPL pipeline:
1. Pocket detection and chain identification
2. PDB file handling and structure manipulation
3. Benchmark utilities (ligand loading, RMSD calculation)
4. General helper functions
"""

import os
import gzip
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
from Bio.PDB import PDBParser, Structure, Selection
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings

# Benchmark-related imports
try:
    from rdkit import Chem
    from rdkit.Chem import rdShapeAlign  # alignment tool
    from spyrmsd.molecule import Molecule
    from spyrmsd.rmsd import rmsdwrapper
except ImportError:
    Chem = None
    rdShapeAlign = None
    Molecule = None
    rmsdwrapper = None

# Configure logging
logger = logging.getLogger(__name__)

# Global cache manager for shared molecules across processes
_GLOBAL_MOLECULE_CACHE = None


def initialize_global_molecule_cache():
    """Initialize global molecule cache that persists across processes."""
    global _GLOBAL_MOLECULE_CACHE
    if _GLOBAL_MOLECULE_CACHE is None:
        _GLOBAL_MOLECULE_CACHE = {}
    return _GLOBAL_MOLECULE_CACHE


def get_global_molecule_cache():
    """Get the global molecule cache."""
    global _GLOBAL_MOLECULE_CACHE
    return _GLOBAL_MOLECULE_CACHE


def set_global_molecule_cache(molecules: List[Any]):
    """Set molecules in global cache."""
    global _GLOBAL_MOLECULE_CACHE
    if _GLOBAL_MOLECULE_CACHE is None:
        _GLOBAL_MOLECULE_CACHE = {}
    _GLOBAL_MOLECULE_CACHE["molecules"] = molecules


def get_pocket_chain_ids_from_file(pocket_file: str) -> List[str]:
    """Extract chain IDs from a pocket PDB file.

    Args:
        pocket_file: Path to pocket PDB file

    Returns:
        List of chain IDs in the pocket
    """
    if not os.path.exists(pocket_file):
        logger.warning(f"Pocket file not found: {pocket_file}")
        return []

    with open(pocket_file) as f:
        chains = sorted(
            {
                line[21].strip()
                for line in f
                if line.startswith("ATOM") and len(line) > 21 and line[21].strip()
            }
        )

    return chains


def find_pocket_chains(
    protein_path: str, ligand_path: str = None, distance_cutoff: float = 5.0
) -> List[str]:
    """Identify binding pocket chains based on proximity to ligand.

    Args:
        protein_path: Path to protein PDB file
        ligand_path: Path to ligand PDB/SDF file (if None, tries to find *_ligand.pdb/*.sdf in same dir)
        distance_cutoff: Maximum distance (Å) between protein and ligand atoms to define pocket

    Returns:
        List of chain IDs in the binding pocket
    """
    # First check if we have a dedicated pocket file
    protein_dir = os.path.dirname(protein_path)
    pdb_id = Path(protein_path).stem
    if pdb_id.endswith("_protein"):
        pdb_id = pdb_id[:-8]  # Remove '_protein' suffix if present

    # Check for pocket file in PDBbind format
    pocket_file = os.path.join(protein_dir, f"{pdb_id}_pocket.pdb")
    if os.path.exists(pocket_file):
        logger.info(f"Found dedicated pocket file: {pocket_file}")
        return get_pocket_chain_ids_from_file(pocket_file)

    # If no dedicated pocket file, try to find ligand and compute pocket
    if ligand_path is None:
        # Try common ligand file naming patterns
        ligand_candidates = [
            os.path.join(protein_dir, f"{pdb_id}_ligand.pdb"),
            os.path.join(protein_dir, f"{pdb_id}_ligand.sdf"),
            os.path.join(protein_dir, f"{pdb_id}.sdf"),
        ]
        for lig_path in ligand_candidates:
            if os.path.exists(lig_path):
                ligand_path = lig_path
                break

    if ligand_path is None or not os.path.exists(ligand_path):
        logger.warning(
            f"No ligand file found for {pdb_id}. Cannot identify pocket chains."
        )
        return []

    # Load protein structure
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure("protein", protein_path)
        except Exception as e:
            logger.error(f"Failed to parse protein structure {protein_path}: {str(e)}")
            return []

    # Load ligand coordinates
    ligand_coords = []
    if ligand_path.endswith(".pdb"):
        # Parse ligand from PDB
        try:
            ligand_structure = parser.get_structure("ligand", ligand_path)
            for atom in ligand_structure.get_atoms():
                ligand_coords.append(atom.get_coord())
        except Exception as e:
            logger.error(f"Failed to parse ligand PDB {ligand_path}: {str(e)}")
            return []
    else:
        # Assume SDF format and use RDKit
        try:
            from rdkit import Chem

            mol = Chem.SDMolSupplier(ligand_path, removeHs=False)[0]
            if mol is None:
                logger.error(f"Failed to read ligand SDF {ligand_path}")
                return []
            conf = mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                ligand_coords.append(np.array([pos.x, pos.y, pos.z]))
        except Exception as e:
            logger.error(f"Failed to parse ligand SDF {ligand_path}: {str(e)}")
            return []

    if not ligand_coords:
        logger.warning(f"No ligand coordinates found in {ligand_path}")
        return []

    # Find protein atoms within cutoff distance of any ligand atom
    pocket_chains = set()
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom_coord = atom.get_coord()
                    # Check distance to each ligand atom
                    for lig_coord in ligand_coords:
                        distance = np.linalg.norm(atom_coord - lig_coord)
                        if distance <= distance_cutoff:
                            pocket_chains.add(chain.id)
                            break  # No need to check other ligand atoms

    logger.info(
        f"Identified {len(pocket_chains)} pocket chains using distance cutoff {distance_cutoff}Å"
    )
    return sorted(list(pocket_chains))


def find_pdbbind_paths(pdb_id: str, data_dir: str = "data") -> Dict[str, str]:
    """Find paths to PDBbind files for a given PDB ID.

    Args:
        pdb_id: PDB ID to find files for
        data_dir: Root directory for PDBbind data

    Returns:
        Dictionary with paths to protein, pocket, and ligand files
    """
    result = {"protein": None, "pocket": None, "ligand": None}

    # Try multiple potential base paths
    potential_base_paths = [
        data_dir,  # Original path
        os.path.join("templ_pipeline", data_dir),
        os.path.join("templ_pipeline", "data", "PDBBind"),
        os.path.join("/home/ubuntu/mcs", "templ_pipeline", "data", "PDBBind"),
        os.path.join("/home/ubuntu/mcs", "mcs_bench", "data"),
    ]

    # Try both refined and general sets in each base path
    for base_path in potential_base_paths:
        # Check refined set
        refined_dir = os.path.join(
            base_path, "PDBbind_v2020_refined", "refined-set", pdb_id
        )
        if os.path.exists(refined_dir):
            protein_path = os.path.join(refined_dir, f"{pdb_id}_protein.pdb")
            if os.path.exists(protein_path):
                result["protein"] = protein_path
                result["pocket"] = os.path.join(refined_dir, f"{pdb_id}_pocket.pdb")

                # Check for ligand in different formats
                ligand_sdf = os.path.join(refined_dir, f"{pdb_id}_ligand.sdf")
                ligand_pdb = os.path.join(refined_dir, f"{pdb_id}_ligand.pdb")
                if os.path.exists(ligand_sdf):
                    result["ligand"] = ligand_sdf
                elif os.path.exists(ligand_pdb):
                    result["ligand"] = ligand_pdb

                logger.info(
                    f"Found protein file for {pdb_id} in refined set at {protein_path}"
                )
                return result

        # Check general set
        general_dir = os.path.join(
            base_path, "PDBbind_v2020_other_PL", "v2020-other-PL", pdb_id
        )
        if os.path.exists(general_dir):
            protein_path = os.path.join(general_dir, f"{pdb_id}_protein.pdb")
            if os.path.exists(protein_path):
                result["protein"] = protein_path
                result["pocket"] = os.path.join(general_dir, f"{pdb_id}_pocket.pdb")

                # Check for ligand in different formats
                ligand_sdf = os.path.join(general_dir, f"{pdb_id}_ligand.sdf")
                ligand_pdb = os.path.join(general_dir, f"{pdb_id}_ligand.pdb")
                if os.path.exists(ligand_sdf):
                    result["ligand"] = ligand_sdf
                elif os.path.exists(ligand_pdb):
                    result["ligand"] = ligand_pdb

                logger.info(
                    f"Found protein file for {pdb_id} in general set at {protein_path}"
                )
                return result

    logger.warning(
        f"Could not find PDBbind files for {pdb_id} in any of the expected locations"
    )
    return result


# -----------------------------------------------------------------------------
# Benchmark Utilities
# -----------------------------------------------------------------------------


def load_sdf_molecules_cached(
    sdf_path: Path, cache: Dict = None, memory_limit_gb: float = 8.0
) -> List[Any]:
    """Load molecules from processed ligands SDF.gz file with memory optimization.

    Args:
        sdf_path: Path to SDF file
        cache: Optional cache dictionary
        memory_limit_gb: Memory limit for loading (GB)

    Returns:
        List of loaded molecules
    """
    if Chem is None:
        raise RuntimeError("RDKit not available - required for benchmark utilities")

    if cache is None:
        cache = {}

    cache_key = str(sdf_path)
    if cache_key in cache:
        logger.debug(f"Using cached molecules: {len(cache[cache_key])}")
        return cache[cache_key]

    molecules = []
    if not sdf_path.exists():
        logger.error(f"SDF file not found: {sdf_path}")
        return molecules

    # Memory monitoring
    try:
        import psutil

        process = psutil.Process()
        initial_memory_gb = process.memory_info().rss / (1024**3)

        if initial_memory_gb > memory_limit_gb:
            logger.warning(
                f"Process already using {initial_memory_gb:.1f}GB, above limit {memory_limit_gb}GB"
            )
    except ImportError:
        initial_memory_gb = 0

    logger.info(f"Loading molecules from {sdf_path.name}...")

    # Single attempt with memory-optimized approach
    try:
        if sdf_path.suffix == ".gz":
            # Memory-efficient gzip handling
            import gzip
            import io

            # Read compressed file in chunks to manage memory
            with gzip.open(sdf_path, "rb") as fh:
                content = fh.read()

            # Process from memory buffer to avoid file handle conflicts
            with io.BytesIO(content) as buffer:
                supplier = Chem.ForwardSDMolSupplier(
                    buffer, removeHs=False, sanitize=False
                )

                processed_count = 0
                skipped_count = 0

                for idx, mol in enumerate(supplier):
                    try:
                        if mol is None:
                            skipped_count += 1
                            continue

                        if not mol.HasProp("_Name"):
                            skipped_count += 1
                            continue

                        # Basic molecule validation
                        if mol.GetNumAtoms() == 0:
                            skipped_count += 1
                            continue

                        mol_name = mol.GetProp("_Name")
                        mol.SetProp("original_name", mol_name)
                        mol.SetProp("molecule_index", str(idx))
                        molecules.append(mol)
                        processed_count += 1

                        # Memory check every 1000 molecules
                        if processed_count % 1000 == 0:
                            try:
                                current_memory_gb = process.memory_info().rss / (
                                    1024**3
                                )
                                memory_delta = current_memory_gb - initial_memory_gb

                                if memory_delta > memory_limit_gb:
                                    logger.warning(
                                        f"Memory limit exceeded at {processed_count} molecules, stopping"
                                    )
                                    break

                            except Exception:
                                pass

                    except Exception as mol_err:
                        skipped_count += 1
                        continue
        else:
            # Handle uncompressed SDF files with memory monitoring
            with open(sdf_path, "rb") as fh:
                supplier = Chem.ForwardSDMolSupplier(fh, removeHs=False, sanitize=False)

                processed_count = 0
                skipped_count = 0

                for idx, mol in enumerate(supplier):
                    try:
                        if mol is None:
                            skipped_count += 1
                            continue

                        if not mol.HasProp("_Name"):
                            skipped_count += 1
                            continue

                        if mol.GetNumAtoms() == 0:
                            skipped_count += 1
                            continue

                        mol_name = mol.GetProp("_Name")
                        mol.SetProp("original_name", mol_name)
                        mol.SetProp("molecule_index", str(idx))
                        molecules.append(mol)
                        processed_count += 1

                        # Memory check every 1000 molecules
                        if processed_count % 1000 == 0:
                            try:
                                current_memory_gb = process.memory_info().rss / (
                                    1024**3
                                )
                                memory_delta = current_memory_gb - initial_memory_gb

                                if memory_delta > memory_limit_gb:
                                    logger.warning(
                                        f"Memory limit exceeded at {processed_count} molecules, stopping"
                                    )
                                    break

                            except Exception:
                                pass

                    except Exception as mol_err:
                        skipped_count += 1
                        continue

        logger.info(
            f"Loaded {len(molecules)} molecules from {sdf_path.name} (skipped {skipped_count})"
        )

        # Final memory check
        try:
            final_memory_gb = process.memory_info().rss / (1024**3)
            memory_used = final_memory_gb - initial_memory_gb
            logger.info(f"Memory used for loading: {memory_used:.1f}GB")
        except Exception:
            pass

        # Cache results
        cache[cache_key] = molecules

        return molecules

    except Exception as e:
        logger.error(f"Failed to load SDF file {sdf_path}: {e}")

        # Try fallback to uncompressed version
        if sdf_path.suffix == ".gz":
            alt_path = sdf_path.with_suffix("").with_suffix(".sdf")
            if alt_path.exists() and alt_path != sdf_path:
                logger.info(f"Trying fallback to uncompressed file: {alt_path}")
                return load_sdf_molecules_cached(alt_path, cache, memory_limit_gb)

        return molecules


def get_pdb_id_from_mol(mol: Any) -> Optional[str]:
    """Extract PDB ID from molecule name."""
    if Chem is None:
        return None

    for prop in ["original_name", "_Name"]:
        if mol.HasProp(prop):
            name = mol.GetProp(prop)
            return name[:4].lower()
    return None


def find_ligand_by_pdb_id(
    pdb_id: str, molecules: List[Any]
) -> Tuple[Optional[str], Any]:
    """Find ligand SMILES and molecule for given PDB ID with error handling."""
    if Chem is None:
        return None, None

    pdb_id_lower = pdb_id.lower()

    for mol in molecules:
        try:
            mol_pdb = get_pdb_id_from_mol(mol)
            if mol_pdb == pdb_id_lower:
                # Handle missing conformers
                if mol.GetNumConformers() == 0:
                    try:
                        from rdkit.Chem import AllChem

                        AllChem.EmbedMolecule(mol, randomSeed=42)
                    except Exception as e:
                        logger.debug(f"Could not embed conformer for {pdb_id}: {e}")
                        continue

                # Generate SMILES with error handling
                try:
                    mol_copy = Chem.Mol(mol)
                    Chem.SanitizeMol(mol_copy)
                    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol_copy))
                    return smiles, mol
                except Exception as e:
                    logger.debug(f"Could not generate SMILES for {pdb_id}: {e}")
                    # Try with more aggressive cleaning
                    try:
                        mol_clean = Chem.RemoveHs(Chem.Mol(mol))
                        Chem.SanitizeMol(mol_clean)
                        smiles = Chem.MolToSmiles(mol_clean)
                        return smiles, mol
                    except Exception as e2:
                        logger.warning(f"Failed to process molecule for {pdb_id}: {e2}")
                        continue
        except Exception as e:
            logger.debug(f"Error processing molecule in search for {pdb_id}: {e}")
            continue

    return None, None


def calculate_rmsd(pose_mol: Any, crystal_mol: Any) -> float:
    """Calculate RMSD after aligning *pose_mol* onto *crystal_mol* with RDKit ShapeAlign.

    Steps:
    1. Heavy-atom copies are generated and guaranteed to have at least one conformer.
    2. If `rdShapeAlign` is available the pose is aligned onto the crystal (combo alignment).
    3. RMSD is computed with ``spyrmsd`` (no further minimisation / symmetry correction handled internally).
    4. If any stage fails we fall back to a direct sPyRMSD calculation without alignment.
    """

    # Verify dependencies
    if Chem is None or rmsdwrapper is None:
        return float("nan")

    # Quick sanity checks
    if pose_mol is None or crystal_mol is None:
        return float("nan")

    try:
        # Create independent heavy-atom copies
        pose = Chem.RemoveHs(Chem.Mol(pose_mol))
        ref = Chem.RemoveHs(Chem.Mol(crystal_mol))

        def _ensure_conformer(m):
            if m.GetNumConformers() == 0:
                from rdkit.Chem import AllChem

                AllChem.EmbedMolecule(m, randomSeed=42)

        _ensure_conformer(pose)
        _ensure_conformer(ref)

        # Attempt alignment (silently skip if rdShapeAlign missing)
        if rdShapeAlign is not None:
            try:
                rdShapeAlign.AlignMol(
                    ref, pose, refConfId=0, probeConfId=0, useColors=True
                )
            except Exception:
                # Alignment failure should not abort RMSD computation
                pass

        # Compute RMSD with sPyRMSD
        return rmsdwrapper(
            Molecule.from_rdkit(pose),
            Molecule.from_rdkit(ref),
            minimize=False,
            strip=True,
            symmetry=True,
        )[0]

    except Exception:
        # Fallback – try raw RMSD on original inputs, else NaN
        try:
            return rmsdwrapper(
                Molecule.from_rdkit(Chem.RemoveHs(pose_mol)),
                Molecule.from_rdkit(Chem.RemoveHs(crystal_mol)),
                minimize=False,
                strip=True,
                symmetry=True,
            )[0]
        except Exception:
            return float("nan")


def get_protein_file_paths(pdb_id: str, data_dir: Path) -> List[Path]:
    """Get possible protein file paths for PDB ID."""
    return [
        data_dir
        / "PDBBind"
        / "PDBbind_v2020_refined"
        / "refined-set"
        / pdb_id
        / f"{pdb_id}_protein.pdb",
        data_dir
        / "PDBBind"
        / "PDBbind_v2020_other_PL"
        / "v2020-other-PL"
        / pdb_id
        / f"{pdb_id}_protein.pdb",
        data_dir
        / "PDBbind_v2020_refined"
        / "refined-set"
        / pdb_id
        / f"{pdb_id}_protein.pdb",
        data_dir
        / "PDBbind_v2020_other_PL"
        / "v2020-other-PL"
        / pdb_id
        / f"{pdb_id}_protein.pdb",
    ]


def find_ligand_file_paths(data_dir: Path) -> List[Path]:
    """Get possible ligand file paths."""
    return [
        data_dir / "ligands" / "processed_ligands_new.sdf.gz",
        data_dir / "ligands" / "processed_ligands_new_unzipped.sdf",
        data_dir / "processed_ligands_new.sdf.gz",
        data_dir / "processed_ligands_new_unzipped.sdf",
    ]


def load_split_pdb_ids(split_file: Path, data_dir: Path) -> Set[str]:
    """Load PDB IDs from split file, filtered by available embeddings."""
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    pdbs = set()
    with open(split_file) as f:
        for line in f:
            if line.strip():
                pdbs.add(line.strip().lower())

    # Filter by available embeddings
    embedding_files = [
        data_dir / "protein_embeddings_base.npz",
        data_dir / "embeddings" / "protein_embeddings_base.npz",
    ]

    for embedding_file in embedding_files:
        if embedding_file.exists():
            try:
                data = np.load(embedding_file, allow_pickle=True)
                available_pdbs = set(str(pdb).lower() for pdb in data["pdb_ids"])
                filtered_pdbs = pdbs.intersection(available_pdbs)
                logger.info(
                    f"Loaded {len(pdbs)} PDBs from {split_file}, {len(filtered_pdbs)} have embeddings"
                )
                return filtered_pdbs
            except Exception as e:
                logger.warning(
                    f"Could not filter by embeddings from {embedding_file}: {e}"
                )
                break

    logger.info(f"Loaded {len(pdbs)} PDBs from {split_file} (no embedding filtering)")
    return pdbs


def get_worker_config(benchmark_workers: int) -> Dict[str, int]:
    """Get optimized worker configuration to prevent oversubscription."""
    return {
        "benchmark_workers": benchmark_workers,
        "pipeline_workers": 1 if benchmark_workers > 1 else benchmark_workers,
    }


def get_shared_molecule_cache() -> Optional[Dict]:
    """Get shared molecule cache if available."""
    global _GLOBAL_MOLECULE_CACHE
    # First try our global cache
    if _GLOBAL_MOLECULE_CACHE is not None:
        return _GLOBAL_MOLECULE_CACHE

    # Then try to import from timesplit module
    try:
        from templ_pipeline.benchmark.timesplit import SHARED_MOLECULE_CACHE

        if SHARED_MOLECULE_CACHE is not None:
            return SHARED_MOLECULE_CACHE
    except (ImportError, AttributeError):
        pass

    return None


def load_molecules_with_shared_cache(
    data_dir: Path, cache_key: str = "molecules"
) -> List[Any]:
    """Load molecules using shared cache if available, fallback to local cache."""

    # Try shared cache first
    shared_cache = get_shared_molecule_cache()
    if shared_cache and cache_key in shared_cache:
        logger.info(f"Using shared cache: {len(shared_cache[cache_key])} molecules")
        return shared_cache[cache_key]

    # Fallback to individual loading
    logger.info("Shared cache not available, loading individually")
    ligand_file_paths = find_ligand_file_paths(data_dir)

    for path in ligand_file_paths:
        if path.exists():
            try:
                molecules = load_sdf_molecules_cached(
                    path, memory_limit_gb=6.0
                )  # Lower memory limit
                if molecules:
                    # Store in global cache for reuse
                    set_global_molecule_cache(molecules)
                    return molecules
            except Exception as e:
                logger.warning(f"Failed to load ligands from {path}: {e}")
                continue

    logger.error("Could not load molecules from any path")
    return []
