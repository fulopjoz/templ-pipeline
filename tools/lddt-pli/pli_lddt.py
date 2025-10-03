#!/usr/bin/env python3

import ost
from ost import mol, io
from ost.mol.alg.ligand_scoring_lddtpli import LDDTPLIScorer
from ost.mol.alg.scoring_base import PDBPrep
import sys
import os
import csv
from pathlib import Path

def calculate_pli_lddt_sdf(apo_pdb_path, reference_sdf_path, model_sdf_path, ligand_radius=6.0):
    try:
        receptor = PDBPrep(apo_pdb_path)

        ref_ligand_entity = io.LoadEntity(reference_sdf_path, format="sdf")
        reference_ligands = ref_ligand_entity.GetResidueList()

        
        
        model_ligand_entity = io.LoadEntity(model_sdf_path, format="sdf") 
        model_ligands = model_ligand_entity.GetResidueList() 
        scorer = LDDTPLIScorer(model=receptor,      # same receptor for both
            target=receptor,                        # same receptor for both
            model_ligands=model_ligands,
            target_ligands=reference_ligands,
            lddt_pli_radius=ligand_radius,
            substructure_match=True
        )
        return scorer.score_matrix[0, 0]
    except Exception as e:
        print("Exception!!!",e)
        return 0.0
        

def main():
    apo_pdb_path = sys.argv[1]
    reference_sdf_path = sys.argv[2]
    model_sdf_path = sys.argv[3]
    radius = 6.0
    score = calculate_pli_lddt_sdf(apo_pdb_path, reference_sdf_path, model_sdf_path, radius)
    return score

if __name__ == "__main__":
    final_score = main()
    print(f"Final PLI LDDT Score: {final_score:.4f}")
    with open(sys.argv[4], 'a') as f:
        writer = csv.writer(f)
        writer.writerow([Path(sys.argv[1]).stem,Path(sys.argv[3]).stem,str(final_score)])
