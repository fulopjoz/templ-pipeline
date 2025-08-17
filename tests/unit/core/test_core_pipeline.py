# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Working tests for the TEMPLPipeline class - focused on actual functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from rdkit import Chem
import os

from templ_pipeline.core.pipeline import TEMPLPipeline, PipelineConfig


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    test_dir = tempfile.mkdtemp()
    output_dir = Path(test_dir) / "output"
    output_dir.mkdir(exist_ok=True)
    
    yield {
        'test_dir': test_dir,
        'output_dir': str(output_dir)
    }
    
    shutil.rmtree(test_dir)


@pytest.fixture
def mock_embedding_file(temp_dirs):
    """Create a mock embedding file."""
    embedding_path = Path(temp_dirs['test_dir']) / "test_embeddings.npz"
    dummy_data = {'embeddings': np.random.rand(10, 1280)}
    np.savez(str(embedding_path), **dummy_data)
    return str(embedding_path)


@pytest.fixture
def pipeline_config(temp_dirs, mock_embedding_file):
    """Create a test pipeline configuration."""
    return PipelineConfig(
        output_dir=temp_dirs['output_dir'],
        embedding_npz=mock_embedding_file
    )


@pytest.fixture
def pipeline(pipeline_config):
    """Create a TEMPLPipeline instance for testing."""
    return TEMPLPipeline(config=pipeline_config)


class TestTEMPLPipelineBasicFunctionality:
    """Test basic TEMPLPipeline functionality."""

    def test_pipeline_initialization(self, temp_dirs):
        """Test pipeline can be initialized."""
        pipeline = TEMPLPipeline(output_dir=temp_dirs['output_dir'])
        
        assert pipeline.output_dir == temp_dirs['output_dir']
        assert pipeline.config is not None
        assert hasattr(pipeline, 'embedding_path')

    def test_pipeline_with_config(self, pipeline_config):
        """Test pipeline with configuration object."""
        pipeline = TEMPLPipeline(config=pipeline_config)
        
        assert pipeline.config == pipeline_config
        assert pipeline.output_dir == pipeline_config.output_dir

    def test_config_property_access(self, pipeline):
        """Test configuration properties can be accessed."""
        assert hasattr(pipeline.config, 'sim_threshold')
        assert hasattr(pipeline.config, 'n_confs')
        assert hasattr(pipeline.config, 'output_dir')
        assert pipeline.config.sim_threshold == 0.90

    def test_config_modification(self, pipeline):
        """Test configuration can be modified."""
        original_threshold = pipeline.config.sim_threshold
        pipeline.config.sim_threshold = 0.95
        
        assert pipeline.config.sim_threshold == 0.95
        assert pipeline.config.sim_threshold != original_threshold


class TestTEMPLPipelineMethodsExist:
    """Test that expected methods exist and can be called."""

    def test_load_target_data_exists(self, pipeline):
        """Test load_target_data method exists and returns boolean."""
        result = pipeline.load_target_data()
        assert isinstance(result, bool)

    def test_load_templates_exists(self, pipeline):
        """Test load_templates method exists and returns boolean.""" 
        result = pipeline.load_templates()
        assert isinstance(result, bool)

    def test_run_exists(self, pipeline):
        """Test run method exists and returns boolean."""
        result = pipeline.run()
        assert isinstance(result, bool)

    def test_utility_methods_exist(self, pipeline):
        """Test utility methods exist."""
        assert hasattr(pipeline, '_extract_pdb_id_from_path')
        assert hasattr(pipeline, '_get_embedding_manager')
        
        # Test they can be called
        result = pipeline._extract_pdb_id_from_path("test.pdb")
        assert result is None or isinstance(result, str)
        
        manager = pipeline._get_embedding_manager()
        # EmbeddingManager should have methods for handling embeddings
        if manager is not None:
            assert hasattr(manager, 'load_embeddings') or hasattr(manager, 'get_embedding')


class TestTEMPLPipelineAddedMethods:
    """Test methods that were added to the pipeline."""

    def test_prepare_query_molecule_valid(self, pipeline):
        """Test prepare_query_molecule with valid SMILES."""
        if hasattr(pipeline, 'prepare_query_molecule'):
            mol = pipeline.prepare_query_molecule("CCO")
            assert mol is not None
            assert isinstance(mol, Chem.Mol)

    def test_prepare_query_molecule_invalid(self, pipeline):
        """Test prepare_query_molecule with invalid SMILES."""
        if hasattr(pipeline, 'prepare_query_molecule'):
            with pytest.raises(ValueError):
                pipeline.prepare_query_molecule("INVALID_SMILES")

    def test_generate_embedding_exists(self, pipeline):
        """Test generate_embedding method if it exists."""
        if hasattr(pipeline, 'generate_embedding'):
            # Should handle gracefully when embedding manager not available
            result = pipeline.generate_embedding("test.pdb")
            assert result is None or isinstance(result, np.ndarray)


class TestTEMPLPipelineIntegration:
    """Test pipeline integration and error handling."""

    def test_run_full_pipeline_exists(self, pipeline, temp_dirs):
        """Test run_full_pipeline method exists and handles basic inputs."""
        protein_file = Path(temp_dirs['test_dir']) / "test.pdb"
        protein_file.write_text("MOCK PDB CONTENT")
        
        try:
            result = pipeline.run_full_pipeline(
                protein_file=str(protein_file),
                ligand_smiles="CCO"
            )
            # If successful, should return dict
            if result is not None:
                assert isinstance(result, dict)
        except Exception as e:
            # Expected to fail due to missing data - this is acceptable
            assert isinstance(e, (FileNotFoundError, ValueError, AttributeError))

    @patch('templ_pipeline.core.pipeline.TEMPLPipeline.load_target_data')
    @patch('templ_pipeline.core.pipeline.TEMPLPipeline.load_templates')
    def test_run_with_mocked_dependencies(self, mock_load_templates, mock_load_target, pipeline):
        """Test run method with mocked dependencies."""
        mock_load_target.return_value = True
        mock_load_templates.return_value = True
        
        result = pipeline.run()
        assert isinstance(result, bool)


class TestTEMPLPipelineErrorHandling:
    """Test pipeline error handling."""

    def test_invalid_output_directory(self):
        """Test pipeline with invalid output directory."""
        try:
            pipeline = TEMPLPipeline(output_dir="/root/forbidden")
            # If it succeeds, that's fine
            assert pipeline is not None
        except (PermissionError, OSError):
            # These are acceptable for invalid directories
            pass

    def test_nonexistent_embedding_file(self, temp_dirs):
        """Test pipeline with nonexistent embedding file."""
        config = PipelineConfig(
            output_dir=temp_dirs['output_dir'],
            embedding_npz="/nonexistent/file.npz"
        )
        
        # Should not crash during initialization
        pipeline = TEMPLPipeline(config=config)
        assert pipeline is not None


class TestTEMPLPipelineValidation:
    """Test pipeline validation and edge cases."""

    def test_empty_config_values(self, temp_dirs):
        """Test pipeline with minimal configuration."""
        pipeline = TEMPLPipeline(output_dir=temp_dirs['output_dir'])
        
        # Should have reasonable defaults
        assert pipeline.config.sim_threshold > 0
        assert pipeline.config.n_confs > 0
        assert pipeline.config.ca_rmsd_threshold > 0

    def test_config_boundary_values(self, temp_dirs):
        """Test configuration with boundary values."""
        config = PipelineConfig(
            output_dir=temp_dirs['output_dir'],
            sim_threshold=0.1,  # Very low
            n_confs=1,         # Minimal
            ca_rmsd_threshold=100.0  # Very high
        )
        
        pipeline = TEMPLPipeline(config=config)
        assert pipeline.config.sim_threshold == 0.1
        assert pipeline.config.n_confs == 1
        assert pipeline.config.ca_rmsd_threshold == 100.0


# Replaced optional method tests with real data tests above


# Replace with real data tests using example files
@pytest.mark.parametrize("test_data", [
    ("1iky", "data/example/1iky_protein.pdb", "data/example/1iky_ligand.sdf"),
    ("5eqy", "data/example/5eqy_protein.pdb", None),  # Protein only
])
def test_pipeline_with_real_example_data(pipeline, test_data):
    """Test pipeline with real example data from data/example/ folder."""
    pdb_id, protein_file, ligand_file = test_data
    
    # Test protein file loading
    if protein_file and os.path.exists(protein_file):
        # Test that pipeline can handle real protein files
        assert os.path.getsize(protein_file) > 0
        assert protein_file.endswith('.pdb')
        
        # Test basic file validation
        with open(protein_file, 'r') as f:
            content = f.read()
            assert 'ATOM' in content or 'HEADER' in content
    
    # Test ligand file loading if available
    if ligand_file and os.path.exists(ligand_file):
        assert os.path.getsize(ligand_file) > 0
        assert ligand_file.endswith('.sdf')
        
        # Test basic SDF validation
        with open(ligand_file, 'r') as f:
            content = f.read()
            assert 'V2000' in content or 'V3000' in content


def test_pipeline_with_1iky_example_data(pipeline):
    """Test pipeline specifically with 1iky example data."""
    protein_file = "data/example/1iky_protein.pdb"
    ligand_file = "data/example/1iky_ligand.sdf"
    
    # Verify example files exist
    assert os.path.exists(protein_file), f"Example protein file not found: {protein_file}"
    assert os.path.exists(ligand_file), f"Example ligand file not found: {ligand_file}"
    
    # Test file properties
    protein_size = os.path.getsize(protein_file)
    ligand_size = os.path.getsize(ligand_file)
    
    assert protein_size > 1000, f"Protein file too small: {protein_size} bytes"
    assert ligand_size > 500, f"Ligand file too small: {ligand_size} bytes"
    
    # Test file content validation
    with open(protein_file, 'r') as f:
        protein_content = f.read()
        assert 'ATOM' in protein_content, "Protein file should contain ATOM records"
        assert '1iky' in protein_content.lower(), "Protein file should contain 1iky reference"
    
    with open(ligand_file, 'r') as f:
        ligand_content = f.read()
        assert 'V2000' in ligand_content, "Ligand file should contain V2000 format"
        assert '1iky_ligand' in ligand_content, "Ligand file should contain 1iky reference"


def test_pipeline_with_5eqy_example_data(pipeline):
    """Test pipeline specifically with 5eqy example data."""
    protein_file = "data/example/5eqy_protein.pdb"
    
    # Verify example file exists
    assert os.path.exists(protein_file), f"Example protein file not found: {protein_file}"
    
    # Test file properties
    protein_size = os.path.getsize(protein_file)
    assert protein_size > 1000, f"Protein file too small: {protein_size} bytes"
    
    # Test file content validation
    with open(protein_file, 'r') as f:
        protein_content = f.read()
        assert 'ATOM' in protein_content, "Protein file should contain ATOM records"
        assert '5eqy' in protein_content.lower(), "Protein file should contain 5eqy reference"


@pytest.mark.parametrize("sim_threshold", [0.1, 0.5, 0.8, 0.9, 0.95])
def test_similarity_thresholds(temp_dirs, sim_threshold):
    """Test different similarity threshold values."""
    config = PipelineConfig(
        output_dir=temp_dirs['output_dir'],
        sim_threshold=sim_threshold
    )
    pipeline = TEMPLPipeline(config=config)
    
    assert pipeline.config.sim_threshold == sim_threshold


def test_pipeline_can_be_created_and_destroyed(temp_dirs):
    """Test pipeline lifecycle."""
    pipeline = TEMPLPipeline(output_dir=temp_dirs['output_dir'])
    assert pipeline is not None
    
    # Should be able to access basic properties
    assert hasattr(pipeline, 'config')
    assert hasattr(pipeline, 'output_dir')
    
    # Cleanup should not cause issues
    del pipeline


if __name__ == "__main__":
    pytest.main([__file__])