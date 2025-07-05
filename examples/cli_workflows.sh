#!/bin/bash
# TEMPL CLI Example Workflows
# 
# This script demonstrates various workflows using the TEMPL CLI
# with real example data (1iky and 5eqy protein-ligand pairs).
#
# Usage: bash cli_workflows.sh

set -e  # Exit on any error

echo "=========================================="
echo "TEMPL CLI Example Workflows"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "data/example/1iky_protein.pdb" ]; then
    echo "Error: Please run this script from the templ_pipeline directory"
    echo "Expected to find: data/example/1iky_protein.pdb"
    exit 1
fi

# Create output directory
OUTPUT_DIR="workflow_output"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# Workflow 1: Quick test with 1iky ligand
echo "=== Workflow 1: Testing with 1iky ligand ==="
echo "Protein: 1iky (kinase)"
echo "Ligand: COc1ccc(C(C)=O)c(O)c1[C@H]1C[C@H]1NC(=S)Nc1ccc(C#N)cn1"
echo ""

templ run \
    --protein-file data/example/1iky_protein.pdb \
    --ligand-smiles "COc1ccc(C(C)=O)c(O)c1[C@H]1C[C@H]1NC(=S)Nc1ccc(C#N)cn1" \
    --embedding-file data/embeddings/templ_protein_embeddings_v1.0.0.npz \
    --num-templates 3 \
    --num-conformers 50 \
    --output-dir "$OUTPUT_DIR/1iky_workflow"

echo ""
echo "✓ Workflow 1 completed"
echo ""

# Workflow 2: Test with 5eqy using SDF input
echo "=== Workflow 2: Testing with 5eqy ligand (SDF input) ==="
echo "Protein: 5eqy"
echo "Ligand: From SDF file"
echo ""

templ run \
    --protein-file data/example/5eqy_protein.pdb \
    --ligand-file data/example/5eqy_ligand.sdf \
    --embedding-file data/embeddings/templ_protein_embeddings_v1.0.0.npz \
    --num-templates 3 \
    --num-conformers 50 \
    --output-dir "$OUTPUT_DIR/5eqy_workflow"

echo ""
echo "✓ Workflow 2 completed"
echo ""

# Workflow 3: Step-by-step pipeline
echo "=== Workflow 3: Step-by-step pipeline ==="
echo "Running each step separately for 1iky"
echo ""

# Step 1: Generate embedding
echo "Step 1: Generate protein embedding..."
templ embed \
    --protein-file data/example/1iky_protein.pdb \
    --output-dir "$OUTPUT_DIR/step_by_step"

# Step 2: Find templates
echo ""
echo "Step 2: Find protein templates..."
templ find-templates \
    --protein-file data/example/1iky_protein.pdb \
    --embedding-file data/embeddings/templ_protein_embeddings_v1.0.0.npz \
    --num-templates 5 \
    --output-dir "$OUTPUT_DIR/step_by_step"

# Step 3: Generate poses
echo ""
echo "Step 3: Generate poses..."
templ generate-poses \
    --protein-file data/example/1iky_protein.pdb \
    --ligand-smiles "COc1ccc(C(C)=O)c(O)c1[C@H]1C[C@H]1NC(=S)Nc1ccc(C#N)cn1" \
    --template-pdb 5eqy \
    --num-conformers 30 \
    --output-dir "$OUTPUT_DIR/step_by_step"

echo ""
echo "✓ Workflow 3 completed"
echo ""

# Workflow 4: Cross-template comparison
echo "=== Workflow 4: Cross-template comparison ==="
echo "Using 1iky ligand with 5eqy protein as template"
echo ""

templ generate-poses \
    --protein-file data/example/1iky_protein.pdb \
    --ligand-smiles "COc1ccc(C(C)=O)c(O)c1[C@H]1C[C@H]1NC(=S)Nc1ccc(C#N)cn1" \
    --template-pdb 5eqy \
    --num-conformers 50 \
    --output-dir "$OUTPUT_DIR/cross_template"

echo ""
echo "✓ Workflow 4 completed"
echo ""

# Summary
echo "=========================================="
echo "All workflows completed successfully!"
echo "=========================================="
echo "Results saved in: $OUTPUT_DIR/"
echo ""
echo "Directory structure:"
ls -la "$OUTPUT_DIR/"
echo ""
echo "To explore results:"
echo "  cd $OUTPUT_DIR"
echo "  ls -la */  # View output directories"
echo "  # Check SDF files for generated poses"
echo "  # Check log files for detailed information" 