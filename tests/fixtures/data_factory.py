# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Test data factory for consistent test data generation.

This module provides centralized test data creation with standardized
patterns for molecules, proteins, embeddings, and pipeline results.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem


class TestDataFactory:
    """Factory for creating standardized test data."""
    
    # Standard test molecules with different complexities
    STANDARD_MOLECULES = {
        'simple_alkane': {
            'smiles': 'CCO',
            'name': 'ethanol',
            'atoms': 3,
            'type': 'simple'
        },
        'aromatic': {
            'smiles': 'c1ccccc1',
            'name': 'benzene', 
            'atoms': 6,
            'type': 'aromatic'
        },
        'complex_drug': {
            'smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
            'name': 'ibuprofen',
            'atoms': 13,
            'type': 'complex'
        },
        'organometallic': {
            'smiles': '[Fe+2]',
            'name': 'iron_cation',
            'atoms': 1,
            'type': 'organometallic'
        },
        'invalid': {
            'smiles': 'INVALID_SMILES',
            'name': 'invalid',
            'atoms': 0,
            'type': 'invalid'
        }
    }
    
    # Standard protein data
    STANDARD_PROTEINS = {
        'minimal': {
            'content': """HEADER    TEST PROTEIN                            01-JAN-00   TEST            
ATOM      1  N   ALA A   1      20.154  16.967  12.784  1.00 10.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  12.345  1.00 10.00           C  
ATOM      3  C   ALA A   1      18.899  14.793  13.116  1.00 10.00           C  
ATOM      4  O   ALA A   1      19.505  14.548  14.163  1.00 10.00           O  
END""",
            'chains': ['A'],
            'atoms': 4,
            'type': 'minimal'
        },
        'multi_chain': {
            'content': """HEADER    MULTI-CHAIN PROTEIN                     01-JAN-00   MULT            
ATOM      1  N   ALA A   1      20.154  16.967  12.784  1.00 10.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  12.345  1.00 10.00           C  
ATOM      3  C   ALA A   1      18.899  14.793  13.116  1.00 10.00           C  
ATOM      4  O   ALA A   1      19.505  14.548  14.163  1.00 10.00           O  
ATOM      5  N   GLY B   1      25.154  20.967  15.784  1.00 10.00           N  
ATOM      6  CA  GLY B   1      24.030  20.101  15.345  1.00 10.00           C  
END""",
            'chains': ['A', 'B'],
            'atoms': 6,
            'type': 'multi_chain'
        }
    }
    
    @staticmethod
    def create_molecule_data(molecule_type: str = 'simple_alkane') -> Dict[str, Any]:
        """
        Create standardized molecule data.
        
        Args:
            molecule_type: Type of molecule from STANDARD_MOLECULES
            
        Returns:
            Dict containing molecule data and RDKit mol object
        """
        if molecule_type not in TestDataFactory.STANDARD_MOLECULES:
            raise ValueError(f"Unknown molecule type: {molecule_type}")
            
        mol_data = TestDataFactory.STANDARD_MOLECULES[molecule_type].copy()
        
        # Create RDKit molecule if SMILES is valid
        if mol_data['type'] != 'invalid':
            mol = Chem.MolFromSmiles(mol_data['smiles'])
            if mol:
                # Add 3D coordinates
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
                mol_data['mol'] = mol
            else:
                mol_data['mol'] = None
        else:
            mol_data['mol'] = None
            
        return mol_data
    
    @staticmethod
    def create_protein_data(protein_type: str = 'minimal') -> Dict[str, Any]:
        """
        Create standardized protein data.
        
        Args:
            protein_type: Type of protein from STANDARD_PROTEINS
            
        Returns:
            Dict containing protein data
        """
        if protein_type not in TestDataFactory.STANDARD_PROTEINS:
            raise ValueError(f"Unknown protein type: {protein_type}")
            
        return TestDataFactory.STANDARD_PROTEINS[protein_type].copy()
    
    @staticmethod
    def create_embedding_data(size: int = 1280, num_proteins: int = 10) -> np.ndarray:
        """
        Create standardized embedding data.
        
        Args:
            size: Embedding dimension
            num_proteins: Number of protein embeddings
            
        Returns:
            Numpy array of embeddings
        """
        # Use fixed seed for reproducible test data
        np.random.seed(42)
        embeddings = np.random.rand(num_proteins, size).astype(np.float32)
        np.random.seed()  # Reset seed
        return embeddings
    
    @staticmethod
    def create_mcs_test_pairs() -> List[Tuple[str, str, str]]:
        """
        Create standardized MCS test pairs.
        
        Returns:
            List of (mol1_smiles, mol2_smiles, description) tuples
        """
        return [
            ('CCO', 'CCC', 'Similar alkanes'),
            ('CCO', 'CCOC', 'Alcohol vs ether'),
            ('c1ccccc1', 'c1ccccc1O', 'Benzene vs phenol'),
            ('CCCC', 'C', 'Chain vs single carbon'),
            ('CCO', 'O', 'Ethanol vs water'),
            ('C1CCCCC1', 'c1ccccc1', 'Cyclohexane vs benzene'),
            ('CC(C)C', 'CCC', 'Branched vs linear')
        ]
    
    @staticmethod
    def create_mock_pipeline_results(success: bool = True) -> Dict[str, Any]:
        """
        Create standardized mock pipeline results.
        
        Args:
            success: Whether to create successful or failed results
            
        Returns:
            Dict containing mock pipeline results
        """
        if success:
            mol = Chem.MolFromSmiles('CCO')
            return {
                'poses': {'combo': (mol, {'combo_score': 0.8})},
                'mcs_info': {'smarts': 'CCO'},
                'templates': [('1abc', 0.9)],
                'embedding': TestDataFactory.create_embedding_data(size=1280, num_proteins=1)[0],
                'output_file': 'test_output.sdf'
            }
        else:
            return {
                'poses': {},
                'mcs_info': {},
                'templates': [],
                'embedding': None,
                'output_file': None,
                'error': 'Mock pipeline failure'
            }
    
    @staticmethod
    def create_test_files(temp_dir: Path) -> Dict[str, Path]:
        """
        Create standardized test files in temporary directory.
        
        Args:
            temp_dir: Temporary directory to create files in
            
        Returns:
            Dict mapping file types to file paths
        """
        files = {}
        
        # Create minimal SDF file
        sdf_file = temp_dir / "test_ligand.sdf"
        sdf_content = """
  Mrv2014 01010100002D          

  3  2  0  0  0  0            999 V2000
   -0.4125    0.7145    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4125    0.7145    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  3  1  0  0  0  0
  2  3  1  0  0  0  0
M  END
$$$$
"""
        sdf_file.write_text(sdf_content)
        files['sdf'] = sdf_file
        
        # Create minimal PDB file
        pdb_file = temp_dir / "test_protein.pdb"
        protein_data = TestDataFactory.create_protein_data('minimal')
        pdb_file.write_text(protein_data['content'])
        files['pdb'] = pdb_file
        
        # Create mock embedding file
        embedding_file = temp_dir / "test_embeddings.npz"
        embeddings = TestDataFactory.create_embedding_data()
        pdb_ids = [f"test{i:03d}" for i in range(embeddings.shape[0])]
        np.savez_compressed(
            embedding_file,
            embeddings=embeddings,
            pdb_ids=pdb_ids
        )
        files['embeddings'] = embedding_file
        
        return files
    
    @staticmethod
    def create_benchmark_target_data() -> List[Dict[str, Any]]:
        """
        Create standardized benchmark target data.
        
        Returns:
            List of benchmark target dictionaries
        """
        return [
            {
                'pdb_id': 'test001',
                'protein_file': 'test001_protein.pdb',
                'ligand_smiles': 'CCO',
                'expected_score': 0.8
            },
            {
                'pdb_id': 'test002', 
                'protein_file': 'test002_protein.pdb',
                'ligand_smiles': 'c1ccccc1',
                'expected_score': 0.7
            },
            {
                'pdb_id': 'test003',
                'protein_file': 'test003_protein.pdb', 
                'ligand_smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
                'expected_score': 0.9
            }
        ]
    
    @staticmethod
    def create_error_test_data() -> Dict[str, Any]:
        """
        Create standardized error testing data.
        
        Returns:
            Dict containing various error scenarios
        """
        return {
            'invalid_smiles': ['', 'INVALID', 'C[C', 'C(C', '[C]', 'XYZ123'],
            'invalid_files': ['nonexistent.pdb', '/invalid/path/file.sdf'],
            'malformed_data': {
                'empty_dict': {},
                'none_values': {'key': None},
                'wrong_types': {'string': 123, 'number': 'abc'}
            },
            'edge_cases': {
                'very_long_smiles': 'C' * 10000,
                'unicode_strings': 'test_运行_🧪_αβγ',
                'special_chars': 'test-id_with.special/chars'
            }
        }
    
    @staticmethod
    def create_timesplit_data(temp_dir: Path, num_train: int = 80, num_val: int = 10, num_test: int = 10) -> Dict[str, Path]:
        """
        Create standardized timesplit test data files.
        
        Args:
            temp_dir: Temporary directory to create files in
            num_train: Number of training PDB IDs
            num_val: Number of validation PDB IDs  
            num_test: Number of test PDB IDs
            
        Returns:
            Dict mapping split names to file paths
        """
        files = {}
        
        # Generate synthetic PDB IDs following realistic patterns
        # Use fixed seed for reproducible test data
        np.random.seed(42)
        
        # Create realistic PDB ID patterns (4 characters: digit+letter+digit+letter)
        letters = 'abcdefghijklmnopqrstuvwxyz'
        digits = '0123456789'
        
        all_pdbs = []
        for i in range(num_train + num_val + num_test):
            pdb_id = f"{np.random.choice(list(digits))}{np.random.choice(list(letters))}{np.random.choice(list(digits))}{np.random.choice(list(letters))}"
            all_pdbs.append(pdb_id)
        
        # Ensure unique IDs
        all_pdbs = list(set(all_pdbs))
        while len(all_pdbs) < (num_train + num_val + num_test):
            pdb_id = f"{np.random.choice(list(digits))}{np.random.choice(list(letters))}{np.random.choice(list(digits))}{np.random.choice(list(letters))}"
            if pdb_id not in all_pdbs:
                all_pdbs.append(pdb_id)
        
        np.random.seed()  # Reset seed
        
        # Split the PDB IDs
        train_pdbs = all_pdbs[:num_train]
        val_pdbs = all_pdbs[num_train:num_train + num_val]
        test_pdbs = all_pdbs[num_train + num_val:num_train + num_val + num_test]
        
        # Create split files
        splits = {
            'train': train_pdbs,
            'val': val_pdbs,
            'test': test_pdbs
        }
        
        for split_name, pdb_list in splits.items():
            split_file = temp_dir / f"{split_name}_pdbs.txt"
            split_file.write_text('\n'.join(pdb_list) + '\n')
            files[split_name] = split_file
            
        return files
    
    @staticmethod
    def create_mock_embeddings_with_splits(temp_dir: Path, split_files: Dict[str, Path]) -> Path:
        """
        Create mock embeddings file that matches the timesplit PDB IDs.
        
        Args:
            temp_dir: Temporary directory to create files in
            split_files: Dict of split files from create_timesplit_data
            
        Returns:
            Path to embeddings file
        """
        # Read all PDB IDs from split files
        all_pdbs = []
        for split_file in split_files.values():
            with open(split_file, 'r') as f:
                pdbs = [line.strip() for line in f if line.strip()]
                all_pdbs.extend(pdbs)
        
        # Create embeddings for all PDBs
        embeddings = TestDataFactory.create_embedding_data(size=1280, num_proteins=len(all_pdbs))
        chain_data = [f"A:{i*10}:{i*10+50}" for i in range(len(all_pdbs))]  # Mock chain data
        
        # Create embeddings file
        embedding_file = temp_dir / "mock_templ_protein_embeddings_v1.0.0.npz"
        np.savez_compressed(
            embedding_file,
            pdb_ids=np.array(all_pdbs, dtype=object),
            embeddings=embeddings,
            chain_ids=np.array(chain_data, dtype=object)
        )
        
        return embedding_file
    
    @staticmethod
    def create_mock_ligand_files(temp_dir: Path, num_ligands: int = 5) -> Dict[str, Path]:
        """
        Create mock ligand SDF files for testing.
        
        Args:
            temp_dir: Temporary directory to create files in
            num_ligands: Number of ligand files to create
            
        Returns:
            Dict mapping ligand names to file paths
        """
        ligand_files = {}
        
        # Standard test ligands with different properties
        test_ligands = [
            {
                'name': 'ethanol',
                'smiles': 'CCO',
                'sdf_content': """
  Mrv2014 01010100002D          

  3  2  0  0  0  0            999 V2000
   -0.4125    0.7145    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4125    0.7145    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  3  1  0  0  0  0
  2  3  1  0  0  0  0
M  END
$$$$"""
            },
            {
                'name': 'benzene',
                'smiles': 'c1ccccc1',
                'sdf_content': """
  Mrv2014 01010100002D          

  6  6  0  0  0  0            999 V2000
    1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.5000    0.8660    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5000    0.8660    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5000   -0.8660    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.5000   -0.8660    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  2  3  1  0  0  0  0
  3  4  2  0  0  0  0
  4  5  1  0  0  0  0
  5  6  2  0  0  0  0
  6  1  1  0  0  0  0
M  END
$$$$"""
            },
            {
                'name': 'propanol',
                'smiles': 'CCCO',
                'sdf_content': """
  Mrv2014 01010100002D          

  4  3  0  0  0  0            999 V2000
   -0.8250    0.7145    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.7145    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.8250    0.7145    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.8250   -0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  3  4  1  0  0  0  0
M  END
$$$$"""
            },
            {
                'name': 'phenol',
                'smiles': 'c1ccc(O)cc1',
                'sdf_content': """
  Mrv2014 01010100002D          

  7  7  0  0  0  0            999 V2000
    1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.5000    0.8660    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5000    0.8660    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.5000   -0.8660    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.5000   -0.8660    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0  0  0  0
  2  3  1  0  0  0  0
  3  4  2  0  0  0  0
  4  5  1  0  0  0  0
  5  6  2  0  0  0  0
  6  1  1  0  0  0  0
  4  7  1  0  0  0  0
M  END
$$$$"""
            },
            {
                'name': 'acetone',
                'smiles': 'CC(=O)C',
                'sdf_content': """
  Mrv2014 01010100002D          

  4  3  0  0  0  0            999 V2000
   -0.8250    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.8250    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.8250    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
  2  4  2  0  0  0  0
M  END
$$$$"""
            }
        ]
        
        # Create ligand files up to the requested number
        for i in range(min(num_ligands, len(test_ligands))):
            ligand = test_ligands[i]
            ligand_file = temp_dir / f"{ligand['name']}_ligand.sdf"
            ligand_file.write_text(ligand['sdf_content'].strip())
            ligand_files[ligand['name']] = ligand_file
        
        return ligand_files
    
    @staticmethod
    def create_mock_pdb_files(temp_dir: Path, pdb_ids: List[str]) -> Dict[str, Path]:
        """
        Create mock PDB files for testing.
        
        Args:
            temp_dir: Temporary directory to create files in
            pdb_ids: List of PDB IDs to create files for
            
        Returns:
            Dict mapping PDB IDs to file paths
        """
        pdb_files = {}
        
        # Base PDB template
        base_pdb_content = """HEADER    TEST PROTEIN                            01-JAN-00   {pdb_id}            
ATOM      1  N   ALA A   1      20.154  16.967  12.784  1.00 10.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  12.345  1.00 10.00           C  
ATOM      3  C   ALA A   1      18.899  14.793  13.116  1.00 10.00           C  
ATOM      4  O   ALA A   1      19.505  14.548  14.163  1.00 10.00           O  
ATOM      5  CB  ALA A   1      17.756  16.925  12.345  1.00 10.00           C  
END"""
        
        for pdb_id in pdb_ids:
            pdb_file = temp_dir / f"{pdb_id}_protein.pdb"
            content = base_pdb_content.format(pdb_id=pdb_id.upper())
            pdb_file.write_text(content)
            pdb_files[pdb_id] = pdb_file
        
        return pdb_files
    
    @staticmethod
    def create_mcs_test_environment(temp_dir: Path) -> Dict[str, Any]:
        """
        Create a complete test environment for MCS testing.
        
        Args:
            temp_dir: Temporary directory to create files in
            
        Returns:
            Dict containing all test data and file paths
        """
        # Create ligand files
        ligand_files = TestDataFactory.create_mock_ligand_files(temp_dir, num_ligands=5)
        
        # Create PDB files for the ligands
        pdb_ids = ['1a0q', '1a1e', '1abc', '2def', '3ghi']
        pdb_files = TestDataFactory.create_mock_pdb_files(temp_dir, pdb_ids)
        
        # Create embeddings for the PDBs
        embeddings = TestDataFactory.create_embedding_data(size=1280, num_proteins=len(pdb_ids))
        chain_data = [f"A:{i*10}:{i*10+50}" for i in range(len(pdb_ids))]
        
        embedding_file = temp_dir / "test_embeddings.npz"
        np.savez_compressed(
            embedding_file,
            pdb_ids=np.array(pdb_ids, dtype=object),
            embeddings=embeddings,
            chain_ids=np.array(chain_data, dtype=object)
        )
        
        # Load test molecules from SMILES for direct use
        test_molecules = {}
        ligand_data = [
            ('ethanol', 'CCO'),
            ('benzene', 'c1ccccc1'),
            ('propanol', 'CCCO'),
            ('phenol', 'c1ccc(O)cc1'),
            ('acetone', 'CC(=O)C')
        ]
        
        for name, smiles in ligand_data:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
                test_molecules[name] = mol
        
        return {
            'temp_dir': temp_dir,
            'ligand_files': ligand_files,
            'pdb_files': pdb_files,
            'embedding_file': embedding_file,
            'test_molecules': test_molecules,
            'pdb_ids': pdb_ids
        }