"""
Tests for template selection and filtering utilities.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import patch, MagicMock
from rdkit import Chem

from templ_pipeline.core.templates import (
    filter_templates_by_ca_rmsd,
    get_templates_with_progressive_fallback,
    find_best_ca_rmsd_template,
    load_uniprot_exclude,
    get_uniprot_mapping,
    load_pdb_filter,
    standardize_atom_arrays,
    DEFAULT_CA_RMSD_FALLBACK_THRESHOLDS,
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
            get_templates_with_progressive_fallback(
                templates, [2.0, 3.0, 5.0], primary_threshold=2.0
            )
        )

        assert len(result_templates) == 2  # 1.0 and 1.5 pass 2.0 threshold
        assert threshold_used == 2.0
        assert use_central_atom is False

    def test_fallback_to_relaxed_threshold(self):
        """Test fallback to relaxed thresholds."""
        templates = []
        for ca_rmsd in [3.5, 4.0, 6.0]:
            mol = Chem.MolFromSmiles("CCO")
            mol.SetProp("ca_rmsd", str(ca_rmsd))
            templates.append(mol)

        with patch("templ_pipeline.core.templates.log") as mock_log:
            result_templates, threshold_used, use_central_atom = (
                get_templates_with_progressive_fallback(
                    templates, [2.0, 3.0, 5.0], primary_threshold=2.0
                )
            )

        assert len(result_templates) == 2  # 3.5 and 4.0 pass 5.0 threshold
        assert threshold_used == 5.0
        assert use_central_atom is False
        mock_log.warning.assert_called()

    def test_central_atom_fallback(self):
        """Test ultimate fallback to central atom positioning."""
        templates = []
        for ca_rmsd in [10.0, 15.0, 8.0]:  # All above fallback thresholds
            mol = Chem.MolFromSmiles("CCO")
            mol.SetProp("ca_rmsd", str(ca_rmsd))
            templates.append(mol)

        with patch("templ_pipeline.core.templates.log") as mock_log:
            result_templates, threshold_used, use_central_atom = (
                get_templates_with_progressive_fallback(templates, [2.0, 3.0, 5.0])
            )

        assert len(result_templates) == 1  # Best template (8.0 RMSD)
        assert float(result_templates[0].GetProp("ca_rmsd")) == 8.0
        assert threshold_used == float("inf")
        assert use_central_atom is True
        mock_log.warning.assert_called()

    def test_fallback_with_no_ca_rmsd_properties(self):
        """Test fallback when templates have no CA RMSD properties."""
        templates = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCC")]

        with patch("templ_pipeline.core.templates.log") as mock_log:
            result_templates, threshold_used, use_central_atom = (
                get_templates_with_progressive_fallback(templates, [2.0, 3.0, 5.0])
            )

        # Templates without CA RMSD properties pass filtering, so all templates returned
        assert len(result_templates) == 2  # All templates pass filtering
        assert threshold_used == 2.0  # First threshold succeeded
        assert use_central_atom is False  # No fallback needed
        mock_log.info.assert_called()  # Info message for successful filtering

    def test_empty_template_list(self):
        """Test behavior with empty template list."""
        with patch("templ_pipeline.core.templates.log") as mock_log:
            result_templates, threshold_used, use_central_atom = (
                get_templates_with_progressive_fallback([], [2.0, 3.0, 5.0])
            )

        assert len(result_templates) == 0
        assert threshold_used == float("inf")
        assert use_central_atom is False
        mock_log.error.assert_called()

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
        assert threshold_used == DEFAULT_CA_RMSD_FALLBACK_THRESHOLDS[0]
        assert use_central_atom is False


class TestFindBestCArmsdTemplate:
    """Test finding best CA RMSD template."""

    def test_find_best_template(self):
        """Test finding template with lowest CA RMSD."""
        templates = []
        ca_rmsd_values = [3.0, 1.5, 2.0, 1.0]

        for ca_rmsd in ca_rmsd_values:
            mol = Chem.MolFromSmiles("CCO")
            mol.SetProp("ca_rmsd", str(ca_rmsd))
            templates.append(mol)

        with patch("templ_pipeline.core.templates.log") as mock_log:
            best_idx = find_best_ca_rmsd_template(templates)

        assert best_idx == 3  # Index of template with 1.0 RMSD
        mock_log.info.assert_called()

    def test_find_best_with_invalid_values(self):
        """Test finding best template when some have invalid CA RMSD values."""
        templates = []

        # Template with invalid CA RMSD
        mol1 = Chem.MolFromSmiles("CCO")
        mol1.SetProp("ca_rmsd", "invalid")
        templates.append(mol1)

        # Template with valid CA RMSD
        mol2 = Chem.MolFromSmiles("CCC")
        mol2.SetProp("ca_rmsd", "2.5")
        templates.append(mol2)

        with patch("templ_pipeline.core.templates.log") as mock_log:
            best_idx = find_best_ca_rmsd_template(templates)

        assert best_idx == 1  # Index of valid template
        mock_log.info.assert_called()

    def test_find_best_with_no_ca_rmsd(self):
        """Test fallback when no templates have CA RMSD properties."""
        templates = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCC")]

        with patch("templ_pipeline.core.templates.log") as mock_log:
            best_idx = find_best_ca_rmsd_template(templates)

        assert best_idx == 0  # Default to first template
        mock_log.info.assert_called()

    def test_find_best_empty_list(self):
        """Test behavior with empty template list."""
        with patch("templ_pipeline.core.templates.log") as mock_log:
            best_idx = find_best_ca_rmsd_template([])

        assert best_idx == 0  # Default value
        mock_log.info.assert_called()


class TestLoadUniprotExclude:
    """Test loading UniProt exclude lists."""

    def test_load_valid_exclude_file(self):
        """Test loading a valid exclude file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("P12345\n")
            f.write("Q67890\n")
            f.write("  R11111  \n")  # Test whitespace handling
            f.write("\n")  # Empty line
            temp_file = f.name

        try:
            with patch("templ_pipeline.core.templates.log") as mock_log:
                result = load_uniprot_exclude(temp_file)

            assert result == {"P12345", "Q67890", "R11111"}
            mock_log.info.assert_called()
        finally:
            os.unlink(temp_file)

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file."""
        with patch("templ_pipeline.core.templates.log") as mock_log:
            result = load_uniprot_exclude("nonexistent_file.txt")

        assert result == set()
        mock_log.warning.assert_called()

    def test_load_file_with_read_error(self):
        """Test handling file read errors."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("P12345\n")
            temp_file = f.name

        try:
            # Simulate a read error by changing file permissions
            os.chmod(temp_file, 0o000)

            with patch("templ_pipeline.core.templates.log") as mock_log:
                result = load_uniprot_exclude(temp_file)

            assert result == set()
            mock_log.error.assert_called()
        finally:
            os.chmod(temp_file, 0o644)  # Restore permissions
            os.unlink(temp_file)


class TestGetUniprotMapping:
    """Test loading UniProt mappings."""

    def test_load_valid_mapping_file(self):
        """Test loading a valid UniProt mapping file."""
        mapping_data = {
            "1abc": {"uniprot": "P12345", "other": "data"},
            "2def": {"uniprot": "Q67890"},
            "3ghi": {"no_uniprot": "data"},  # Missing uniprot field
        }

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            json.dump(mapping_data, f)
            temp_file = f.name

        try:
            with patch("templ_pipeline.core.templates.log") as mock_log:
                result = get_uniprot_mapping(temp_file)

            expected = {"1abc": "P12345", "2def": "Q67890"}
            assert result == expected
            mock_log.info.assert_called()
        finally:
            os.unlink(temp_file)

    def test_load_nonexistent_mapping_file(self):
        """Test loading a non-existent mapping file."""
        with patch("templ_pipeline.core.templates.log") as mock_log:
            result = get_uniprot_mapping("nonexistent_file.json")

        assert result == {}
        mock_log.warning.assert_called()

    def test_load_invalid_json_file(self):
        """Test loading an invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            f.write("invalid json content")
            temp_file = f.name

        try:
            with patch("templ_pipeline.core.templates.log") as mock_log:
                result = get_uniprot_mapping(temp_file)

            assert result == {}
            mock_log.error.assert_called()
        finally:
            os.unlink(temp_file)


class TestLoadPdbFilter:
    """Test loading PDB filter lists."""

    def test_load_valid_pdb_filter(self):
        """Test loading a valid PDB filter file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("1ABC\n")  # Should be lowercased
            f.write("2def\n")
            f.write("# Comment line\n")  # Should be ignored
            f.write("  3GHI  \n")  # Test whitespace handling
            f.write("\n")  # Empty line
            temp_file = f.name

        try:
            with patch("templ_pipeline.core.templates.log") as mock_log:
                result = load_pdb_filter(temp_file)

            assert result == {"1abc", "2def", "3ghi"}
            mock_log.info.assert_called()
        finally:
            os.unlink(temp_file)

    def test_load_nonexistent_pdb_filter(self):
        """Test loading a non-existent PDB filter file."""
        with patch("templ_pipeline.core.templates.log") as mock_log:
            result = load_pdb_filter("nonexistent_file.txt")

        assert result == set()
        mock_log.warning.assert_called()

    def test_load_pdb_filter_with_read_error(self):
        """Test handling read errors."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("1abc\n")
            temp_file = f.name

        try:
            # Simulate a read error
            os.chmod(temp_file, 0o000)

            with patch("templ_pipeline.core.templates.log") as mock_log:
                result = load_pdb_filter(temp_file)

            assert result == set()
            mock_log.error.assert_called()
        finally:
            os.chmod(temp_file, 0o644)
            os.unlink(temp_file)


class TestStandardizeAtomArrays:
    """Test atom array standardization."""

    def test_standardize_with_empty_list(self):
        """Test standardization with empty list."""
        result = standardize_atom_arrays([])
        assert result == []

    def test_standardize_with_single_array(self):
        """Test standardization with single array."""
        mock_array = MagicMock()
        result = standardize_atom_arrays([mock_array])
        assert result == [mock_array]

    def test_standardize_with_no_annotations(self):
        """Test standardization when arrays have no annotations."""
        mock_array1 = MagicMock()
        mock_array2 = MagicMock()
        del mock_array1.annotations  # Simulate missing annotations
        del mock_array2.annotations

        result = standardize_atom_arrays([mock_array1, mock_array2])
        assert result == [mock_array1]  # Should return first array

    def test_standardize_with_common_annotations(self):
        """Test standardization with arrays having common annotations."""
        # Create mock arrays with annotations
        mock_array1 = MagicMock()
        mock_array1.annotations = {
            "common1": "value1",
            "common2": "value2",
            "unique1": "value3",
        }
        mock_array1.copy.return_value = mock_array1

        mock_array2 = MagicMock()
        mock_array2.annotations = {
            "common1": "value4",
            "common2": "value5",
            "unique2": "value6",
        }
        mock_array2.copy.return_value = mock_array2

        result = standardize_atom_arrays([mock_array1, mock_array2])

        assert len(result) == 2
        # Verify that copy was called
        mock_array1.copy.assert_called_once()
        mock_array2.copy.assert_called_once()

    def test_standardize_with_exception(self):
        """Test standardization when exception occurs."""
        mock_array1 = MagicMock()
        mock_array2 = MagicMock()

        # Simulate exception during processing
        mock_array1.annotations = {"key": "value"}
        mock_array2.annotations = MagicMock()
        mock_array2.annotations.__iter__.side_effect = Exception("Test exception")

        result = standardize_atom_arrays([mock_array1, mock_array2])
        assert result == [mock_array1]  # Should fallback to first array


if __name__ == "__main__":
    pytest.main([__file__])
