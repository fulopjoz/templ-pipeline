"""Mock data for CLI tests."""

import numpy as np

# Mock protein sequences and structures
MOCK_PROTEIN_PDB = """HEADER    TRANSFERASE/DNA                         01-JUL-97   1ABC              
ATOM      1  N   ALA A   1      20.154  16.967  14.421  1.00 25.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  14.849  1.00 25.00           C  
ATOM      3  C   ALA A   1      17.693  16.849  15.035  1.00 25.00           C  
ATOM      4  O   ALA A   1      17.534  17.904  14.425  1.00 25.00           O  
END"""

MOCK_LIGAND_SDF = """
  Mrv2014 01012100002D          

  3  2  0  0  0  0            999 V2000
   -0.4125    0.7145    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4125   -0.7145    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.8250    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  3  1  0  0  0  0
  2  3  1  0  0  0  0
M  END
$$$$
"""

# Simple valid SMILES
MOCK_SMILES = "CCO"
MOCK_INVALID_SMILES = "C[C@H](C)C(=O)N[C@@H](C[C@H](O)C[NH3+])C(=O)N[invalid"

# Mock embeddings
MOCK_EMBEDDINGS = np.random.rand(10, 1280).astype(np.float32)
MOCK_PDB_IDS = ["1abc", "2def", "3ghi", "4jkl", "5mno"]

# Mock template info
MOCK_TEMPLATE_INFO = {
    "embedding_similarity": "0.85",
    "ref_chains": "A",
    "mob_chains": "B"
} 