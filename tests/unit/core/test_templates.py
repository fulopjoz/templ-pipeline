# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Tests for template selection and filtering utilities.
"""

from unittest.mock import patch

import pytest
from rdkit import Chem

from templ_pipeline.core.templates import (
    CA_RMSD_FALLBACK_THRESHOLDS,
    filter_templates_by_ca_rmsd,
    get_templates_with_progressive_fallback,
)


class TestFilterTemplatesByCArmsd:
    """Test CA RMSD filtering functionality."""

    def test_filter_with_infinite_threshold(self):
        """Test that infinite threshold returns all templates."""
        templates = [Chem.MolFromSmiles("CCO") for _ in range(3)]
        result = filter_templates_by_ca_rmsd(templates, float("inf"))
        assert len(result) == 3
        assert result == templates

    def test_filter_with_valid_ca_rmsd_properties(self):
        """Test filtering with valid CA RMSD properties."""
        templates = []

        # Create templates with different CA RMSD values
        for i, ca_rmsd in enumerate([1.0, 2.5, 4.0]):
            mol = Chem.MolFromSmiles("CCO")
            mol.SetProp("ca_rmsd", str(ca_rmsd))
            templates.append(mol)

        # Test with threshold 2.0 - should pass first template only
        result = filter_templates_by_ca_rmsd(templates, 2.0)
        assert len(result) == 1
        assert float(result[0].GetProp("ca_rmsd")) == 1.0

        # Test with threshold 3.0 - should pass first two templates
        result = filter_templates_by_ca_rmsd(templates, 3.0)
        assert len(result) == 2
        assert float(result[0].GetProp("ca_rmsd")) == 1.0
        assert float(result[1].GetProp("ca_rmsd")) == 2.5

    def test_filter_with_missing_ca_rmsd_properties(self):
        """Test that templates without CA RMSD properties are included."""
        templates = []

        # Template with CA RMSD
        mol1 = Chem.MolFromSmiles("CCO")
        mol1.SetProp("ca_rmsd", "1.5")
        templates.append(mol1)

        # Template without CA RMSD
        mol2 = Chem.MolFromSmiles("CCC")
        templates.append(mol2)

        result = filter_templates_by_ca_rmsd(templates, 1.0)
        # Should include only mol2 (no CA RMSD property)
        assert len(result) == 1
        assert not result[0].HasProp("ca_rmsd")

    def test_filter_with_invalid_ca_rmsd_values(self):
        """Test handling of invalid CA RMSD values."""
        templates = []

        # Template with invalid CA RMSD
        mol1 = Chem.MolFromSmiles("CCO")
        mol1.SetProp("ca_rmsd", "invalid")
        templates.append(mol1)

        # Template with valid CA RMSD
        mol2 = Chem.MolFromSmiles("CCC")
        mol2.SetProp("ca_rmsd", "1.5")
        templates.append(mol2)

        result = filter_templates_by_ca_rmsd(templates, 2.0)
        # Should include only the valid template
        assert len(result) == 1
        assert float(result[0].GetProp("ca_rmsd")) == 1.5

    def test_filter_with_empty_list(self):
        """Test filtering with empty template list."""
        result = filter_templates_by_ca_rmsd([], 2.0)
        assert len(result) == 0


class TestGetTemplatesWithProgressiveFallback:
    """Test progressive fallback functionality."""

    def test_successful_filtering_with_primary_threshold(self):
        """Test successful filtering with primary threshold."""
        templates = []
        for ca_rmsd in [1.0, 1.5, 3.0]:
            mol = Chem.MolFromSmiles("CCO")
            mol.SetProp("ca_rmsd", str(ca_rmsd))
            templates.append(mol)

        result_templates, threshold_used, use_central_atom = (
            get_templates_with_progressive_fallback(templates, [2.0, 3.0, 5.0])
        )

        assert len(result_templates) == 2  # Should pass 2 templates with threshold 2.0
        assert threshold_used == 2.0
        assert not use_central_atom

    def test_fallback_to_relaxed_threshold(self):
        """Test fallback to relaxed thresholds."""
        templates = []
        for ca_rmsd in [3.5, 4.0, 6.0]:
            mol = Chem.MolFromSmiles("CCO")
            mol.SetProp("ca_rmsd", str(ca_rmsd))
            templates.append(mol)

        with patch("templ_pipeline.core.templates.log") as _:
            result_templates, threshold_used, use_central_atom = (
                get_templates_with_progressive_fallback(templates, [2.0, 3.0, 5.0])
            )

        assert len(result_templates) == 2  # Should pass 2 templates with threshold 5.0
        assert threshold_used == 5.0
        assert not use_central_atom

    def test_central_atom_fallback(self):
        """Test central atom fallback when no templates pass thresholds."""
        templates = []
        for ca_rmsd in [15.0, 20.0, 25.0]:  # All above thresholds
            mol = Chem.MolFromSmiles("CCO")
            mol.SetProp("ca_rmsd", str(ca_rmsd))
            templates.append(mol)

        with patch("templ_pipeline.core.templates.log") as _:
            result_templates, threshold_used, use_central_atom = (
                get_templates_with_progressive_fallback(templates, [2.0, 3.0, 5.0])
            )

        assert len(result_templates) == 1  # Should return best template
        assert threshold_used == float("inf")
        assert use_central_atom

    def test_fallback_with_no_ca_rmsd_properties(self):
        """Test fallback when templates have no CA RMSD properties."""
        templates = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCC")]

        with patch("templ_pipeline.core.templates.log") as _:
            result_templates, threshold_used, use_central_atom = (
                get_templates_with_progressive_fallback(templates, [2.0, 3.0, 5.0])
            )

        # Templates without CA RMSD properties are included in filtered results
        assert len(result_templates) == 2  # Both templates pass filtering
        assert threshold_used == 2.0  # First threshold succeeded
        assert not use_central_atom  # No fallback needed

    def test_empty_template_list(self):
        """Test behavior with empty template list."""
        with patch("templ_pipeline.core.templates.log") as _:
            result_templates, threshold_used, use_central_atom = (
                get_templates_with_progressive_fallback([], [2.0, 3.0, 5.0])
            )

        assert len(result_templates) == 0
        assert threshold_used == float("inf")
        assert not use_central_atom

    def test_default_fallback_thresholds(self):
        """Test using default fallback thresholds."""
        templates = []
        mol = Chem.MolFromSmiles("CCO")
        mol.SetProp("ca_rmsd", "1.5")
        templates.append(mol)

        result_templates, threshold_used, use_central_atom = (
            get_templates_with_progressive_fallback(templates)  # No explicit thresholds
        )

        assert len(result_templates) == 1
        assert threshold_used == CA_RMSD_FALLBACK_THRESHOLDS[0]
        assert not use_central_atom


if __name__ == "__main__":
    pytest.main([__file__])
