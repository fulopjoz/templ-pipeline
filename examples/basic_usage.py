#!/usr/bin/env python3
"""
Basic TEMPL Pipeline Usage Example

This example demonstrates how to use the TEMPL pipeline components to:
1. Load protein structures and generate embeddings
2. Select templates based on protein similarity
3. Find MCS between query and template ligands
4. Generate conformers using constrained embedding
5. Score and select poses

For a real-world application, you would need:
- PDB files for proteins
- SDF files for ligands
- Pre-computed embeddings (optional, but recommended for speed)
"""

import os
import sys
import logging
from pathlib import Path
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Add parent directory to path to allow imports when run from examples directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import TEMPL components
from templ_pipeline.core import (
    EmbeddingManager,
    find_mcs,
    constrained_embed,
    select_best,
    generate_properties_for_sdf,
    rmsd_raw,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("templ-example")


def run_example(output_dir="example_output"):
    """Run the TEMPL pipeline example."""
    logger.info("Starting TEMPL pipeline example")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Create simple test molecules ---
    logger.info("Creating test molecules")

    # Create simple query molecule (ethanol)
    query_mol = Chem.MolFromSmiles("CCO")
    if not query_mol:
        logger.error("Failed to create query molecule")
        return
    query_mol.SetProp("_Name", "query_ethanol")
    AllChem.EmbedMolecule(query_mol)
    AllChem.MMFFOptimizeMolecule(query_mol)

    # Create a few template molecules
    template_mols = []
    template_smiles = [
        "CCOC",
        "CCCCO",
        "CC(O)C",
    ]  # Methyl ethyl ether, butanol, isopropanol
    for i, smiles in enumerate(template_smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            logger.warning(f"Failed to create template molecule {i+1}")
            continue
        mol.SetProp("_Name", f"template_{i+1}")
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        template_mols.append(mol)

    # --- 2. Find maximum common substructure ---
    logger.info("Finding maximum common substructure")

    idx, smarts = find_mcs(query_mol, template_mols)
    if idx is None:
        logger.error("MCS search failed")
        return

    logger.info(f"MCS found with template {idx+1}: {smarts}")

    # Get the winning template
    template_mol = template_mols[idx]

    # --- 3. Generate conformers with constrained embedding ---
    logger.info("Generating conformers with constrained embedding")

    confs = constrained_embed(
        query_mol, template_mol, smarts, n_conformers=10, n_workers=2
    )
    n_confs = confs.GetNumConformers()
    logger.info(f"Generated {n_confs} conformers")

    # --- 4. Score and select poses ---
    logger.info("Scoring and selecting poses")

    best_poses = select_best(confs, template_mol, no_realign=False, n_workers=2)

    # --- 5. Save results ---
    logger.info("Saving results")

    # Save SDF files with pose information
    sdf_path = os.path.join(output_dir, "poses.sdf")
    with Chem.SDWriter(sdf_path) as writer:
        for metric, (pose, scores) in best_poses.items():
            if pose is None:
                logger.warning(f"No valid pose for {metric}")
                continue

            # Add properties to the molecule
            pose_with_props = generate_properties_for_sdf(
                pose,
                metric,
                scores[metric],
                template_mol.GetProp("_Name"),
                {
                    "shape_score": f"{scores['shape']:.3f}",
                    "color_score": f"{scores['color']:.3f}",
                    "combo_score": f"{scores['combo']:.3f}",
                },
            )

            # Write to SDF
            writer.write(pose_with_props)

    logger.info(f"Saved poses to {sdf_path}")

    # --- 6. Evaluate RMSD to original query (in a real scenario, this would be to crystal) ---
    logger.info("Evaluating RMSD to query structure")

    for metric, (pose, scores) in best_poses.items():
        if pose is None:
            continue

        rms = rmsd_raw(pose, query_mol)
        logger.info(
            f"RMSD for {metric} pose: {rms:.2f} Å (score: {scores[metric]:.3f})"
        )

    logger.info("Example completed successfully")

    return sdf_path, best_poses


if __name__ == "__main__":
    run_example()
