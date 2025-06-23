
"""
Standardized Molecular Processing Module

Provides unified molecule loading and processing for both CLI and web app interfaces.
Ensures consistent sanitization, coordinate generation, and validation.
"""

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from typing import Optional, Dict, Any
import gzip
import os

RDLogger.DisableLog('rdApp.*')

class StandardizedMoleculeProcessor:
    """Unified molecule processing for TEMPL pipeline."""
    
    def __init__(self, sdf_paths: Dict[str, str]):
        self.sdf_paths = sdf_paths
    
    def load_molecule(self, identifier: str, source: str = "database") -> Optional[Chem.Mol]:
        """Load and process molecule with standardized pipeline."""
        if source == "database":
            return self._load_from_database(identifier)
        elif source == "smiles":
            return self._load_from_smiles(identifier)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def _load_from_database(self, pdb_id: str) -> Optional[Chem.Mol]:
        """Load molecule from SDF database."""
        for source_name, path in self.sdf_paths.items():
            if not os.path.exists(path):
                continue
            
            try:
                opener = gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'r')
                
                with opener as f:
                    supplier = Chem.ForwardSDMolSupplier(f, removeHs=False, sanitize=False)
                    
                    for mol in supplier:
                        if mol is None:
                            continue
                        
                        props = mol.GetPropsAsDict()
                        mol_pdb = props.get('template_pid', props.get('_Name', ''))
                        
                        if pdb_id.lower() in mol_pdb.lower():
                            return self._standardize_molecule(mol)
            
            except Exception:
                continue
        
        return None
    
    def _load_from_smiles(self, smiles: str) -> Optional[Chem.Mol]:
        """Load molecule from SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate 3D coordinates
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        
        return self._standardize_molecule(mol)
    
    def _standardize_molecule(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Apply standardized processing."""
        try:
            mol_processed = Chem.Mol(mol)
            
            # Standardized sanitization
            Chem.SanitizeMol(mol_processed)
            
            # Standardized aromaticity
            Chem.SetAromaticity(mol_processed)
            
            # Ensure 3D coordinates
            if mol_processed.GetNumConformers() == 0:
                mol_processed = Chem.AddHs(mol_processed, addCoords=True)
                AllChem.EmbedMolecule(mol_processed)
                AllChem.MMFFOptimizeMolecule(mol_processed)
                mol_processed = Chem.RemoveHs(mol_processed)
            
            return mol_processed
            
        except Exception:
            return None
