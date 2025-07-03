#!/usr/bin/env python3
"""
TEMPL CLI Demo Script

This script demonstrates how to use the TEMPL CLI with real example data.
It shows various workflows using the 1iky and 5eqy protein-ligand pairs.

Usage:
    python cli_demo.py [--output-dir OUTPUT_DIR]
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("templ-cli-demo")


def run_command(cmd, description):
    """Run a command and log the results."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("✓ Command completed successfully")
        if result.stdout:
            logger.info(f"Output:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Command failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(f"Error:\n{e.stderr}")
        return False


def check_files():
    """Check if required example files exist."""
    required_files = [
        "data/example/1iky_protein.pdb",
        "data/example/1iky_ligand.sdf",
        "data/example/5eqy_protein.pdb",
        "data/example/5eqy_ligand.sdf",
        "data/embeddings/protein_embeddings_base.npz",
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)

    if missing_files:
        logger.error("Missing required files:")
        for file_path in missing_files:
            logger.error(f"  - {file_path}")
        return False

    logger.info("All required example files found")
    return True


def demo_embed(output_dir):
    """Demo: Generate protein embeddings."""
    logger.info("\n" + "=" * 50)
    logger.info("DEMO 1: Generate Protein Embeddings")
    logger.info("=" * 50)

    # Embed 1iky protein
    cmd = [
        "templ",
        "embed",
        "--protein-file",
        "data/example/1iky_protein.pdb",
        "--output-dir",
        output_dir,
    ]

    return run_command(cmd, "Generate embedding for 1iky protein")


def demo_find_templates(output_dir):
    """Demo: Find protein templates."""
    logger.info("\n" + "=" * 50)
    logger.info("DEMO 2: Find Protein Templates")
    logger.info("=" * 50)

    cmd = [
        "templ",
        "find-templates",
        "--protein-file",
        "data/example/1iky_protein.pdb",
        "--embedding-file",
        "data/embeddings/protein_embeddings_base.npz",
        "--num-templates",
        "5",
        "--output-dir",
        output_dir,
    ]

    return run_command(cmd, "Find templates for 1iky protein")


def demo_generate_poses_smiles(output_dir):
    """Demo: Generate poses using SMILES input."""
    logger.info("\n" + "=" * 50)
    logger.info("DEMO 3: Generate Poses (SMILES Input)")
    logger.info("=" * 50)

    # 1iky ligand SMILES
    smiles = "COc1ccc(C(C)=O)c(O)c1[C@H]1C[C@H]1NC(=S)Nc1ccc(C#N)cn1"

    cmd = [
        "templ",
        "generate-poses",
        "--protein-file",
        "data/example/1iky_protein.pdb",
        "--ligand-smiles",
        smiles,
        "--template-pdb",
        "5eqy",
        "--num-conformers",
        "20",
        "--output-dir",
        output_dir,
    ]

    return run_command(cmd, "Generate poses for 1iky ligand using SMILES")


def demo_generate_poses_sdf(output_dir):
    """Demo: Generate poses using SDF input."""
    logger.info("\n" + "=" * 50)
    logger.info("DEMO 4: Generate Poses (SDF Input)")
    logger.info("=" * 50)

    cmd = [
        "templ",
        "generate-poses",
        "--protein-file",
        "data/example/5eqy_protein.pdb",
        "--ligand-file",
        "data/example/5eqy_ligand.sdf",
        "--template-pdb",
        "1iky",
        "--num-conformers",
        "20",
        "--output-dir",
        output_dir,
    ]

    return run_command(cmd, "Generate poses for 5eqy ligand using SDF file")


def demo_full_pipeline(output_dir):
    """Demo: Run the full TEMPL pipeline."""
    logger.info("\n" + "=" * 50)
    logger.info("DEMO 5: Full TEMPL Pipeline")
    logger.info("=" * 50)

    # 1iky ligand SMILES
    smiles = "COc1ccc(C(C)=O)c(O)c1[C@H]1C[C@H]1NC(=S)Nc1ccc(C#N)cn1"

    cmd = [
        "templ",
        "run",
        "--protein-file",
        "data/example/1iky_protein.pdb",
        "--ligand-smiles",
        smiles,
        "--embedding-file",
        "data/embeddings/protein_embeddings_base.npz",
        "--num-templates",
        "3",
        "--num-conformers",
        "50",
        "--output-dir",
        output_dir,
    ]

    return run_command(cmd, "Run full pipeline for 1iky protein-ligand pair")


def main():
    """Run all CLI demos."""
    parser = argparse.ArgumentParser(description="TEMPL CLI Demo")
    parser.add_argument(
        "--output-dir", default="cli_demo_output", help="Directory for demo outputs"
    )
    args = parser.parse_args()

    logger.info("TEMPL CLI Demo Starting...")
    logger.info(f"Output directory: {args.output_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check required files
    if not check_files():
        logger.error("Cannot run demo without required files")
        return 1

    # Run demos
    demos = [
        ("Embed", demo_embed),
        ("Find Templates", demo_find_templates),
        ("Generate Poses (SMILES)", demo_generate_poses_smiles),
        ("Generate Poses (SDF)", demo_generate_poses_sdf),
        ("Full Pipeline", demo_full_pipeline),
    ]

    results = {}
    for name, demo_func in demos:
        try:
            success = demo_func(args.output_dir)
            results[name] = "✓ PASSED" if success else "✗ FAILED"
        except Exception as e:
            logger.error(f"Demo '{name}' crashed: {e}")
            results[name] = "✗ CRASHED"

    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("DEMO SUMMARY")
    logger.info("=" * 50)
    for name, result in results.items():
        logger.info(f"{name}: {result}")

    # Check if all passed
    all_passed = all("PASSED" in result for result in results.values())
    if all_passed:
        logger.info("\n🎉 All demos completed successfully!")
        return 0
    else:
        logger.error("\nERROR: Some demos failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
