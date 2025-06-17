#!/usr/bin/env python3
"""
Debug script to analyze SDF.gz ligand file and identify quality issues.
"""

import gzip
import sys
from pathlib import Path
from collections import defaultdict, Counter
import logging

import numpy as np
from rdkit import Chem

def analyze_sdf_file(sdf_path: Path, max_molecules: int = None, problem_pdbs: list = None):
    """Analyze SDF.gz file for quality issues."""
    
    print(f"Analyzing SDF file: {sdf_path}")
    
    stats = {
        "total_molecules": 0,
        "valid_molecules": 0,
        "no_conformers": 0,
        "extreme_coords": 0,
        "sanitization_fails": 0,
        "coordinate_stats": {"x": [], "y": [], "z": []},
        "problematic_pdbs": {},
        "atom_counts": [],
        "heavy_atom_counts": []
    }
    
    if problem_pdbs:
        print(f"Focusing on problematic PDBs: {problem_pdbs}")
    
    try:
        with gzip.open(sdf_path, 'rb') as fh:
            supplier = Chem.ForwardSDMolSupplier(fh, removeHs=False, sanitize=False)
            
            for i, mol in enumerate(supplier):
                if max_molecules and i >= max_molecules:
                    break
                    
                stats["total_molecules"] += 1
                
                if i % 1000 == 0:
                    print(f"Processed {i} molecules...")
                
                if not mol:
                    continue
                
                # Get PDB ID
                pdb_id = "unknown"
                if mol.HasProp("_Name"):
                    mol_name = mol.GetProp("_Name")
                    pdb_id = mol_name[:4].lower()
                
                # Skip if not in problem list (when specified)
                if problem_pdbs and pdb_id not in problem_pdbs:
                    continue
                
                # Check conformers
                if mol.GetNumConformers() == 0:
                    stats["no_conformers"] += 1
                    if problem_pdbs:
                        print(f"PDB {pdb_id}: No conformers")
                    continue
                
                # Analyze coordinates
                conf = mol.GetConformer(0)
                extreme_found = False
                coords_all = {"x": [], "y": [], "z": []}
                
                for atom_idx in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(atom_idx)
                    coords_all["x"].append(pos.x)
                    coords_all["y"].append(pos.y)
                    coords_all["z"].append(pos.z)
                    
                    if any(abs(coord) > 1000 for coord in [pos.x, pos.y, pos.z]):
                        extreme_found = True
                
                if extreme_found:
                    stats["extreme_coords"] += 1
                    max_coord = max(max(coords_all["x"]), max(coords_all["y"]), max(coords_all["z"]))
                    min_coord = min(min(coords_all["x"]), min(coords_all["y"]), min(coords_all["z"]))
                    stats["problematic_pdbs"][pdb_id] = {
                        "issue": "extreme_coordinates",
                        "max_coord": max_coord,
                        "min_coord": min_coord,
                        "atoms": mol.GetNumAtoms()
                    }
                    if problem_pdbs:
                        print(f"PDB {pdb_id}: Extreme coords - max={max_coord:.1f}, min={min_coord:.1f}")
                    continue
                
                # Test sanitization
                try:
                    test_mol = Chem.Mol(mol)
                    Chem.SanitizeMol(test_mol)
                except Exception as e:
                    stats["sanitization_fails"] += 1
                    stats["problematic_pdbs"][pdb_id] = {
                        "issue": "sanitization_failure",
                        "error": str(e),
                        "atoms": mol.GetNumAtoms()
                    }
                    if problem_pdbs:
                        print(f"PDB {pdb_id}: Sanitization failed - {e}")
                    continue
                
                # Valid molecule - collect stats
                stats["valid_molecules"] += 1
                stats["atom_counts"].append(mol.GetNumAtoms())
                
                # Heavy atom count
                mol_no_h = Chem.RemoveHs(mol)
                stats["heavy_atom_counts"].append(mol_no_h.GetNumAtoms())
                
                # Coordinate ranges for valid molecules
                stats["coordinate_stats"]["x"].extend(coords_all["x"])
                stats["coordinate_stats"]["y"].extend(coords_all["y"])
                stats["coordinate_stats"]["z"].extend(coords_all["z"])
    
    except Exception as e:
        print(f"Error analyzing SDF: {e}")
        return stats
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total molecules: {stats['total_molecules']}")
    print(f"Valid molecules: {stats['valid_molecules']}")
    print(f"No conformers: {stats['no_conformers']}")
    print(f"Extreme coordinates: {stats['extreme_coords']}")
    print(f"Sanitization failures: {stats['sanitization_fails']}")
    print(f"Success rate: {stats['valid_molecules']/stats['total_molecules']*100:.1f}%")
    
    if stats["atom_counts"]:
        print(f"\nAtom count stats:")
        print(f"  Mean: {np.mean(stats['atom_counts']):.1f}")
        print(f"  Median: {np.median(stats['atom_counts']):.1f}")
        print(f"  Range: {min(stats['atom_counts'])} - {max(stats['atom_counts'])}")
    
    if stats["coordinate_stats"]["x"]:
        print(f"\nCoordinate ranges (valid molecules):")
        for axis in ["x", "y", "z"]:
            coords = stats["coordinate_stats"][axis]
            print(f"  {axis}: {min(coords):.2f} to {max(coords):.2f}")
    
    if stats["problematic_pdbs"]:
        print(f"\nProblematic PDBs ({len(stats['problematic_pdbs'])}):")
        for pdb_id, info in list(stats["problematic_pdbs"].items())[:10]:
            print(f"  {pdb_id}: {info['issue']}")
        if len(stats["problematic_pdbs"]) > 10:
            print(f"  ... and {len(stats['problematic_pdbs']) - 10} more")
    
    return stats

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Find SDF file
    sdf_paths = [
        Path("/home/ubuntu/mcs/templ_pipeline/data/ligands/processed_ligands_new.sdf.gz"),
        Path("/home/ubuntu/mcs/templ_pipeline/data/processed_ligands_new.sdf.gz"),
        Path("templ_pipeline/data/ligands/processed_ligands_new.sdf.gz")
    ]
    
    sdf_file = None
    for path in sdf_paths:
        if path.exists():
            sdf_file = path
            break
    
    if not sdf_file:
        print(f"Could not find SDF file in any of: {sdf_paths}")
        return
    
    # Problem PDBs from benchmark output
    problem_pdbs = [
        "6cjj", "6cyg", "6h14", "6m7h", "6mjq", "6eeb", "6gdy", "6mji", 
        "6o0h", "6q36", "6iby", "6d40", "5zxk", "6np3", "6k1s", "6e3o"
    ]
    
    # Run analysis
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        print("Running full analysis...")
        analyze_sdf_file(sdf_file, max_molecules=None)
    else:
        print("Running focused analysis on problem PDBs...")
        analyze_sdf_file(sdf_file, max_molecules=20000, problem_pdbs=problem_pdbs)

if __name__ == "__main__":
    main() 