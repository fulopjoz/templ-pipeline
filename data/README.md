# TEMPL Pipeline Data Directory

This directory contains data files required for the TEMPL (Template-based Protein-Ligand) pipeline.

## Directory Structure

- `embeddings/`: Contains pre-computed protein embeddings
  - `templ_protein_embeddings_v1.0.0.npz`: Pre-computed ESM2 embeddings for proteins in PDBbind

- `ligands/`: Contains ligand structure files
  - `templ_processed_ligands_v1.0.0.sdf.gz`: Processed ligand structures from PDBbind

- `splits/`: Contains dataset split definitions for benchmarking
  - `timesplit_test`: Test set PDB IDs for time-split benchmarking
  - `timesplit_train`: Training set PDB IDs for time-split benchmarking
  - `timesplit_val`: Validation set PDB IDs for time-split benchmarking

## File Formats

### Embeddings

The `templ_protein_embeddings_v1.0.0.npz` file is a NumPy compressed archive containing:
- `pdb_ids`: Array of PDB IDs
- `embeddings`: Array of ESM2 embeddings (shape: [num_proteins, embedding_dim])
- `chain_ids`: Array of chain IDs used for each protein

### Ligands

The `templ_processed_ligands_v1.0.0.sdf.gz` file is a compressed Structure-Data File (SDF) containing:
- 3D structures of ligands from PDBbind
- Additional properties for each ligand

### Dataset Splits

The time-split files contain PDB IDs for different subsets of the data, split by deposition date:
- Each line contains a single PDB ID
- The splits ensure proper temporal separation (no future information leakage)
- The `no_lig_overlap` splits ensure that similar ligands don't appear across train/val/test sets
