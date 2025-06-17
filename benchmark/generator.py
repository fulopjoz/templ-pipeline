from pathlib import Path
from typing import Iterator, Tuple, Optional, Set, Any
import gzip
import io
import gc
from rdkit import Chem


def load_molecules_streaming(sdf_paths: list, pdb_filter: Optional[Set[str]] = None) -> Iterator[Tuple[str, Any]]:
    """
    Generator that yields molecules one at a time from SDF files.
    Prevents loading entire datasets into memory.
    """
    for sdf_path in sdf_paths:
        if not Path(sdf_path).exists():
            continue
            
        try:
            for pdb_id, mol in _load_from_single_sdf(sdf_path, pdb_filter):
                yield pdb_id, mol
                # Explicit cleanup after yielding
                del mol
                gc.collect()
        except Exception as e:
            continue


def _load_from_single_sdf(sdf_path: Path, pdb_filter: Optional[Set[str]] = None) -> Iterator[Tuple[str, Any]]:
    """Load molecules from a single SDF file with streaming."""
    try:
        if sdf_path.suffix == '.gz':
            with gzip.open(sdf_path, 'rb') as fh:
                content = fh.read()
            
            with io.BytesIO(content) as buffer:
                supplier = Chem.ForwardSDMolSupplier(buffer, removeHs=False, sanitize=False)
                
                for mol in supplier:
                    if mol is None or not mol.HasProp("_Name"):
                        continue
                    
                    mol_name = mol.GetProp("_Name")
                    pdb_id = mol_name[:4].lower()
                    
                    if pdb_filter and pdb_id not in pdb_filter:
                        continue
                    
                    yield pdb_id, mol
        else:
            with Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False) as supplier:
                for mol in supplier:
                    if mol is None or not mol.HasProp("_Name"):
                        continue
                    
                    mol_name = mol.GetProp("_Name")
                    pdb_id = mol_name[:4].lower()
                    
                    if pdb_filter and pdb_id not in pdb_filter:
                        continue
                    
                    yield pdb_id, mol
                    
    except Exception as e:
        return


def find_ligand_streaming(sdf_paths: list, target_pdb: str) -> Tuple[Optional[str], Optional[Any]]:
    """Find specific ligand using streaming approach to minimize memory usage."""
    target_pdb_lower = target_pdb.lower()
    
    for pdb_id, mol in load_molecules_streaming(sdf_paths, {target_pdb_lower}):
        if pdb_id == target_pdb_lower:
            try:
                # Extract SMILES
                smiles = Chem.MolToSmiles(mol)
                return smiles, mol
            except:
                continue
    
    return None, None


def batch_process_streaming(pdb_ids: Set[str], data_dir: Path, batch_size: int = 10) -> Iterator[Tuple[str, Optional[str], Optional[Any]]]:
    """
    Process PDB IDs in batches using streaming to control memory usage.
    Yields (pdb_id, smiles, molecule) tuples.
    """
    from templ_pipeline.core.utils import find_ligand_file_paths
    
    ligand_file_paths = find_ligand_file_paths(data_dir)
    pdb_id_list = list(pdb_ids)
    
    for i in range(0, len(pdb_id_list), batch_size):
        batch = pdb_id_list[i:i + batch_size]
        batch_set = set(pdb.lower() for pdb in batch)
        
        # Process current batch
        found_in_batch = set()
        for pdb_id, mol in load_molecules_streaming(ligand_file_paths, batch_set):
            if pdb_id in batch_set:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    yield pdb_id, smiles, mol
                    found_in_batch.add(pdb_id)
                except:
                    yield pdb_id, None, None
        
        # Yield None results for missing PDBs in batch
        for pdb_id in batch_set - found_in_batch:
            yield pdb_id, None, None
        
        # Force cleanup after each batch
        gc.collect()


class StreamingMoleculeLoader:
    """Manages streaming access to molecular data."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._ligand_paths = None
    
    def get_ligand_paths(self):
        """Lazy load ligand file paths."""
        if self._ligand_paths is None:
            from templ_pipeline.core.utils import find_ligand_file_paths
            self._ligand_paths = find_ligand_file_paths(self.data_dir)
        return self._ligand_paths
    
    def load_single_molecule(self, pdb_id: str) -> Tuple[Optional[str], Optional[Any]]:
        """Load single molecule using streaming approach."""
        return find_ligand_streaming(self.get_ligand_paths(), pdb_id)
    
    def load_batch(self, pdb_ids: Set[str], batch_size: int = 10) -> Iterator[Tuple[str, Optional[str], Optional[Any]]]:
        """Load molecules in batches."""
        yield from batch_process_streaming(pdb_ids, self.data_dir, batch_size)
    
    def cleanup(self):
        """Cleanup cached paths."""
        self._ligand_paths = None
        gc.collect() 