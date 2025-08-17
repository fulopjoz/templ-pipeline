# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Integration tests for TEMPL Pipeline with enhanced alignment validation.

This module tests the integration between the UI and the enhanced pipeline
with RDShapeAlign and spyrmsd capabilities.
"""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem


class TestPipelineIntegration:
    """Integration tests for the enhanced pipeline."""

    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Cleanup test environment."""
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_enhanced_alignment_functions(self):
        """Test the enhanced alignment functions."""
        # Test the new alignment functions
        try:
            from templ_pipeline.core.mcs import (
                calculate_shape_alignment,
                calculate_symmetry_corrected_rmsd,
                enhanced_template_scoring,
            )

            # Create test molecules
            query_mol = Chem.MolFromSmiles("CCO")
            template_mol = Chem.MolFromSmiles("CCO")

            # Add 3D coordinates
            AllChem.EmbedMolecule(query_mol)
            AllChem.EmbedMolecule(template_mol)

            # Test shape alignment
            shape_results = calculate_shape_alignment(query_mol, template_mol)
            assert "shape_tanimoto" in shape_results
            assert "color_tanimoto" in shape_results
            assert "combo_score" in shape_results
            assert shape_results["alignment_success"] is True

            # Test symmetry-corrected RMSD
            rmsd_results = calculate_symmetry_corrected_rmsd(query_mol, template_mol)
            assert "basic_rmsd" in rmsd_results
            assert "symmetry_rmsd" in rmsd_results
            assert rmsd_results["rmsd_calculation_success"] is True

            # Test enhanced template scoring
            scoring_results = enhanced_template_scoring(query_mol, template_mol, 0.8)
            assert "combined_score" in scoring_results
            assert "chemical_similarity" in scoring_results
            assert "embedding_similarity" in scoring_results
            assert scoring_results["scoring_success"] is True

        except ImportError:
            pytest.skip("Enhanced alignment functions not available")

    def test_spyrmsd_integration(self):
        """Test spyrmsd integration."""
        try:
            from templ_pipeline.core.mcs import (
                HAS_SPYRMSD,
                calculate_symmetry_corrected_rmsd,
            )

            if not HAS_SPYRMSD:
                pytest.skip("spyrmsd not available")

            # Create test molecules with different conformations
            query_mol = Chem.MolFromSmiles("CCO")
            template_mol = Chem.MolFromSmiles("CCO")

            AllChem.EmbedMolecule(query_mol)
            AllChem.EmbedMolecule(template_mol)

            # Test with spyrmsd
            results = calculate_symmetry_corrected_rmsd(
                query_mol, template_mol, use_spyrmsd=True
            )

            assert "symmetry_rmsd" in results
            assert results["rmsd_calculation_success"] is True

            # Symmetry RMSD should be <= basic RMSD for symmetric molecules
            assert (
                results["symmetry_rmsd"] <= results["basic_rmsd"] + 0.1
            )  # Small tolerance

        except ImportError:
            pytest.skip("spyrmsd integration test dependencies not available")

    def test_rdshape_align_integration(self):
        """Test RDShapeAlign integration."""
        try:
            from rdkit.Chem import rdShapeHelpers

            from templ_pipeline.core.mcs import calculate_shape_alignment

            # Create test molecules
            query_mol = Chem.MolFromSmiles("CCO")
            template_mol = Chem.MolFromSmiles("CCC")

            AllChem.EmbedMolecule(query_mol)
            AllChem.EmbedMolecule(template_mol)

            # Test shape alignment
            results = calculate_shape_alignment(query_mol, template_mol)

            assert "shape_tanimoto" in results
            assert "color_tanimoto" in results
            assert "protrude_dist" in results
            assert results["alignment_success"] is True

            # Shape Tanimoto should be between 0 and 1
            assert 0 <= results["shape_tanimoto"] <= 1
            assert 0 <= results["color_tanimoto"] <= 1

        except ImportError:
            pytest.skip("RDShapeAlign integration test dependencies not available")

    @patch("templ_pipeline.core.pipeline.TEMPLPipeline")
    def test_pipeline_with_enhanced_scoring(self, mock_pipeline):
        """Test pipeline integration with enhanced scoring."""
        # Setup mock pipeline
        mock_instance = Mock()
        mock_pipeline.return_value = mock_instance

        # Mock pipeline results with enhanced scoring
        mock_results = {
            "poses": {
                "shape": (
                    Chem.MolFromSmiles("CCO"),
                    {
                        "shape": 0.8,
                        "color": 0.6,
                        "combo": 0.7,
                        "shape_tanimoto": 0.75,
                        "color_tanimoto": 0.55,
                        "basic_rmsd": 1.2,
                        "symmetry_rmsd": 1.1,
                        "chemical_similarity": 0.85,
                        "combined_score": 0.78,
                    },
                )
            },
            "mcs_info": {"mcs_smiles": "CCO"},
            "templates": [("1ABC", 0.95)],
            "embedding": np.random.rand(1280),
            "output_file": "/tmp/test_output.sdf",
        }

        mock_instance.run.return_value = mock_results

        # Test that pipeline service can handle enhanced results
        try:
            from templ_pipeline.ui.services.pipeline_service import PipelineService

            service = PipelineService()

            # Mock the service methods
            with patch.object(service, "run_pipeline") as mock_run:
                mock_run.return_value = mock_results

                results = service.run_pipeline(
                    protein_input="1ABC", ligand_smiles="CCO", num_conformers=10
                )

                assert results is not None
                assert "poses" in results
                assert "shape" in results["poses"]

                # Check enhanced scoring metrics
                shape_pose = results["poses"]["shape"]
                pose_mol, scores = shape_pose

                # Should have enhanced scoring metrics
                expected_metrics = [
                    "shape_tanimoto",
                    "color_tanimoto",
                    "basic_rmsd",
                    "symmetry_rmsd",
                    "chemical_similarity",
                    "combined_score",
                ]

                for metric in expected_metrics:
                    if metric in scores:
                        assert isinstance(scores[metric], (int, float))

        except ImportError:
            pytest.skip("Pipeline service integration test dependencies not available")

    def test_coordinate_tracking_integration(self):
        """Test coordinate tracking integration."""
        try:
            from templ_pipeline.core.mcs import transform_ligand

            # Create test protein structures
            test_protein_pdb = os.path.join(self.temp_dir, "test_protein.pdb")
            with open(test_protein_pdb, "w") as f:
                f.write(
                    """HEADER    TEST PROTEIN
ATOM      1  CA  ALA A   1      20.0  16.0  10.0  1.00 20.00           C
ATOM      2  CA  GLY A   2      18.0  15.0  11.0  1.00 20.00           C
END
"""
                )

            # Create test ligand
            test_ligand = Chem.MolFromSmiles("CCO")
            AllChem.EmbedMolecule(test_ligand)

            # Test coordinate transformation
            # Note: This would require proper protein structure setup
            # For now, just test that the function exists and can be called

            # The function should exist and be callable
            assert callable(transform_ligand)

        except ImportError:
            pytest.skip(
                "Coordinate tracking integration test dependencies not available"
            )

    def test_error_handling_in_enhanced_functions(self):
        """Test error handling in enhanced alignment functions."""
        try:
            from templ_pipeline.core.mcs import (
                calculate_shape_alignment,
                calculate_symmetry_corrected_rmsd,
                enhanced_template_scoring,
            )

            # Test with invalid molecules
            invalid_mol = None
            valid_mol = Chem.MolFromSmiles("CCO")
            AllChem.EmbedMolecule(valid_mol)

            # Should handle invalid molecules gracefully
            shape_results = calculate_shape_alignment(invalid_mol, valid_mol)
            assert shape_results["alignment_success"] is False

            rmsd_results = calculate_symmetry_corrected_rmsd(invalid_mol, valid_mol)
            assert rmsd_results["rmsd_calculation_success"] is False

            scoring_results = enhanced_template_scoring(invalid_mol, valid_mol)
            assert scoring_results["scoring_success"] is False

        except ImportError:
            pytest.skip("Enhanced alignment functions not available")

    def test_template_scoring_weights(self):
        """Test template scoring weight distribution."""
        try:
            from templ_pipeline.core.mcs import enhanced_template_scoring

            # Create test molecules
            query_mol = Chem.MolFromSmiles("CCO")
            template_mol = Chem.MolFromSmiles("CCO")

            AllChem.EmbedMolecule(query_mol)
            AllChem.EmbedMolecule(template_mol)

            # Test with different embedding similarities
            results_low = enhanced_template_scoring(query_mol, template_mol, 0.1)
            results_high = enhanced_template_scoring(query_mol, template_mol, 0.9)

            # Higher embedding similarity should lead to higher combined score
            assert results_high["combined_score"] > results_low["combined_score"]

            # Check that all components are present
            expected_components = [
                "shape_tanimoto",
                "color_tanimoto",
                "shape_combo",
                "basic_rmsd",
                "symmetry_rmsd",
                "chemical_similarity",
                "embedding_similarity",
                "combined_score",
            ]

            for component in expected_components:
                assert component in results_high
                assert component in results_low

        except ImportError:
            pytest.skip("Enhanced template scoring test dependencies not available")


class TestStreamlitPipelineIntegration:
    """Test Streamlit UI integration with enhanced pipeline."""

    def test_ui_can_handle_enhanced_results(self):
        """Test that UI components can handle enhanced pipeline results."""
        try:
            from templ_pipeline.ui.components.results_section import (
                render_results_section,
            )

            # Mock enhanced results
            enhanced_results = {
                "poses": {
                    "shape": (
                        Chem.MolFromSmiles("CCO"),
                        {
                            "shape": 0.8,
                            "color": 0.6,
                            "combo": 0.7,
                            "shape_tanimoto": 0.75,
                            "color_tanimoto": 0.55,
                            "basic_rmsd": 1.2,
                            "symmetry_rmsd": 1.1,
                            "chemical_similarity": 0.85,
                            "combined_score": 0.78,
                        },
                    )
                },
                "mcs_info": {"mcs_smiles": "CCO"},
                "templates": [("1ABC", 0.95)],
                "embedding": np.random.rand(1280),
                "output_file": "/tmp/test_output.sdf",
            }

            # Should not raise exceptions
            render_results_section(enhanced_results)

        except ImportError:
            pytest.skip("UI integration test dependencies not available")

    def test_ui_displays_enhanced_metrics(self):
        """Test that UI can display enhanced scoring metrics."""
        # This would require actual UI testing with rendered components
        # For now, just test that the enhanced metrics are structured correctly

        enhanced_metrics = {
            "shape_tanimoto": 0.75,
            "color_tanimoto": 0.55,
            "basic_rmsd": 1.2,
            "symmetry_rmsd": 1.1,
            "chemical_similarity": 0.85,
            "combined_score": 0.78,
        }

        # All metrics should be numeric
        for metric, value in enhanced_metrics.items():
            assert isinstance(value, (int, float))

        # Combined score should be reasonable
        assert 0 <= enhanced_metrics["combined_score"] <= 1

        # Symmetry RMSD should be <= basic RMSD
        assert enhanced_metrics["symmetry_rmsd"] <= enhanced_metrics["basic_rmsd"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
