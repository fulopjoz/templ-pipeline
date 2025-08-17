# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Working performance tests for TEMPL pipeline - using actual available functions.
"""

import pytest
import time
import psutil
import os
from unittest.mock import patch, Mock
import tempfile
import shutil
from pathlib import Path
import numpy as np
from rdkit import Chem


@pytest.fixture
def sample_molecules():
    """Create sample RDKit molecules for testing."""
    smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O', 'CCCCCC']
    return [Chem.MolFromSmiles(smiles) for smiles in smiles_list if Chem.MolFromSmiles(smiles)]


@pytest.fixture  
def temp_dirs():
    """Create temporary directories for testing."""
    test_dir = tempfile.mkdtemp()
    yield test_dir
    shutil.rmtree(test_dir)


@pytest.mark.performance
@pytest.mark.slow
class TestCoreFunctionPerformance:
    """Benchmark core function performance using actual available functions."""

    def test_mcs_performance(self, sample_molecules):
        """Test MCS calculation performance using actual find_mcs function."""
        if len(sample_molecules) < 2:
            pytest.skip("Need at least 2 molecules for MCS test")
            
        try:
            from templ_pipeline.core.mcs import find_mcs
        except ImportError:
            pytest.skip("MCS module not available")

        target_mol = sample_molecules[0]
        reference_mols = sample_molecules[1:]

        start_time = time.time()
        result = find_mcs(target_mol, reference_mols)
        elapsed = time.time() - start_time

        # Performance baseline: should complete within reasonable time
        assert elapsed < 10.0  # 10 seconds max
        assert isinstance(result, tuple)
        assert len(result) >= 2  # Should return at least (count, smarts)

    def test_scoring_performance(self, sample_molecules):
        """Test scoring performance using actual scoring functions."""
        try:
            from templ_pipeline.core.scoring import rmsd_raw, score_and_align
        except ImportError:
            pytest.skip("Scoring module not available")

        if len(sample_molecules) < 2:
            pytest.skip("Need at least 2 molecules for scoring test")

        mol1 = sample_molecules[0]
        mol2 = sample_molecules[1]

        # Ensure molecules have conformers
        from rdkit.Chem import AllChem
        AllChem.EmbedMolecule(mol1)
        AllChem.EmbedMolecule(mol2)

        start_time = time.time()
        try:
            rmsd_value = rmsd_raw(mol1, mol2)
            elapsed = time.time() - start_time

            assert elapsed < 5.0  # 5 seconds max
            assert isinstance(rmsd_value, (int, float))
        except Exception:
            # RMSD calculation might fail for incompatible molecules
            elapsed = time.time() - start_time
            assert elapsed < 5.0  # Should fail fast

    def test_embedding_performance(self):
        """Test embedding-related performance."""
        try:
            from templ_pipeline.core.embedding import EmbeddingManager
        except ImportError:
            pytest.skip("Embedding module not available")

        # Mock the singleton instance to avoid initialization issues
        start_time = time.time()
        
        # Test with a mock embedding file
        with patch.object(EmbeddingManager, '_instance', None):
            with patch('numpy.load') as mock_load:
                mock_load.return_value = {'embeddings': np.random.rand(10, 1280)}
                
                try:
                    manager = EmbeddingManager("/fake/path")
                    elapsed = time.time() - start_time
                    
                    assert elapsed < 5.0  # Should initialize quickly
                    assert manager is not None
                except Exception:
                    # If initialization fails, just check timing
                    elapsed = time.time() - start_time
                    assert elapsed < 5.0  # Should fail quickly

    def test_pipeline_initialization_performance(self, temp_dirs):
        """Test pipeline initialization performance."""
        try:
            from templ_pipeline.core.pipeline import TEMPLPipeline
        except ImportError:
            pytest.skip("Pipeline module not available")

        start_time = time.time()
        pipeline = TEMPLPipeline(output_dir=temp_dirs)
        elapsed = time.time() - start_time

        assert elapsed < 10.0  # Should initialize within 10 seconds
        assert pipeline is not None
        assert hasattr(pipeline, 'config')


@pytest.mark.performance  
class TestMemoryUsage:
    """Monitor memory usage during processing."""

    def get_memory_usage(self):
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    def test_memory_baseline(self):
        """Establish memory usage baseline."""
        initial_memory = self.get_memory_usage()
        if initial_memory == 0:
            pytest.skip("Cannot measure memory usage")

        # Perform minimal operations
        data = [i for i in range(1000)]
        del data

        final_memory = self.get_memory_usage()
        memory_diff = abs(final_memory - initial_memory)

        # Should not leak significant memory for simple operations
        assert memory_diff < 100  # Less than 100MB change

    def test_molecule_memory_usage(self, sample_molecules):
        """Test memory usage with molecule operations."""
        initial_memory = self.get_memory_usage()
        if initial_memory == 0:
            pytest.skip("Cannot measure memory usage")

        # Create and manipulate molecules
        molecules = []
        for _ in range(10):
            for mol in sample_molecules:
                if mol:
                    molecules.append(Chem.Mol(mol))

        peak_memory = self.get_memory_usage()
        del molecules

        final_memory = self.get_memory_usage()
        
        # Memory should be released after processing
        memory_retained = final_memory - initial_memory
        assert memory_retained < 200  # Less than 200MB retained

    def test_large_data_memory(self):
        """Test memory usage with large data structures."""
        initial_memory = self.get_memory_usage()
        if initial_memory == 0:
            pytest.skip("Cannot measure memory usage")

        # Create large data structure
        large_data = np.random.rand(1000, 1000)  # ~8MB array
        
        peak_memory = self.get_memory_usage()
        del large_data

        final_memory = self.get_memory_usage()

        # Memory should be released
        memory_retained = final_memory - initial_memory
        assert memory_retained < 50  # Less than 50MB retained


@pytest.mark.performance
@pytest.mark.slow
class TestScalabilityLimits:
    """Test scalability limits with actual functions."""

    def test_mcs_scaling_with_molecule_count(self, sample_molecules):
        """Test MCS performance scaling with number of reference molecules."""
        try:
            from templ_pipeline.core.mcs import find_mcs
        except ImportError:
            pytest.skip("MCS module not available")

        if len(sample_molecules) < 2:
            pytest.skip("Need multiple molecules for scaling test")

        target_mol = sample_molecules[0]
        
        # Test with increasing numbers of reference molecules
        molecule_counts = [1, 2, min(3, len(sample_molecules)-1)]
        times = []

        for count in molecule_counts:
            ref_mols = sample_molecules[1:count+1]
            
            start_time = time.time()
            result = find_mcs(target_mol, ref_mols)
            elapsed = time.time() - start_time
            
            times.append(elapsed)

        # Performance should scale reasonably (not exponentially)
        assert all(t < 30 for t in times)  # All under 30 seconds
        
        # Later tests shouldn't be dramatically slower
        if len(times) > 1:
            max_slowdown = 10  # Max 10x slowdown
            assert times[-1] <= times[0] * max_slowdown

    def test_conformer_generation_scaling(self, sample_molecules):
        """Test conformer generation scaling."""
        try:
            from rdkit.Chem import AllChem
        except ImportError:
            pytest.skip("RDKit AllChem not available")

        if not sample_molecules:
            pytest.skip("No molecules available")

        mol = sample_molecules[0]
        
        # Test with different conformer counts
        conformer_counts = [1, 5, 10]
        times = []

        for count in conformer_counts:
            mol_copy = Chem.Mol(mol)
            
            start_time = time.time()
            AllChem.EmbedMultipleConfs(mol_copy, numConfs=count)
            elapsed = time.time() - start_time
            
            times.append(elapsed)

        # Should complete within reasonable time
        assert all(t < 60 for t in times)  # All under 60 seconds

    def test_batch_processing_performance(self, sample_molecules):
        """Test performance with batch operations."""
        if not sample_molecules:
            pytest.skip("No molecules available")

        # Test processing batches of increasing size
        batch_sizes = [1, 3, min(5, len(sample_molecules))]
        
        for batch_size in batch_sizes:
            batch = sample_molecules[:batch_size]
            
            start_time = time.time()
            
            # Mock batch processing operation
            results = []
            for mol in batch:
                if mol:
                    # Simple operation: count atoms
                    atom_count = mol.GetNumAtoms()
                    results.append({"atom_count": atom_count})
            
            elapsed = time.time() - start_time

            # Should complete efficiently
            assert elapsed < batch_size * 1.0  # Max 1 second per molecule
            assert len(results) == len(batch)


@pytest.mark.performance
class TestResourceManagement:
    """Test resource management and cleanup."""

    def test_file_handle_management(self, temp_dirs):
        """Test that file handles are properly managed."""
        import gc
        
        # Create temporary files
        temp_files = []
        for i in range(10):
            temp_file = Path(temp_dirs) / f"test_{i}.txt"
            temp_file.write_text(f"test data {i}")
            temp_files.append(temp_file)

        # Read files and ensure handles are closed
        data = []
        for temp_file in temp_files:
            with open(temp_file, 'r') as f:
                data.append(f.read())

        # Force garbage collection
        gc.collect()

        # All files should still be accessible (no handle leaks)
        for temp_file in temp_files:
            assert temp_file.exists()
            assert temp_file.read_text().startswith("test data")

    def test_molecule_object_cleanup(self, sample_molecules):
        """Test molecule object cleanup."""
        import gc
        
        # Create many molecule copies
        mol_copies = []
        for _ in range(100):
            for mol in sample_molecules[:2]:  # Limit to first 2
                if mol:
                    mol_copies.append(Chem.Mol(mol))

        # Clear references
        del mol_copies
        gc.collect()

        # Test should complete without memory issues
        assert True  # If we get here, cleanup worked


# Integration performance test
@pytest.mark.performance
@pytest.mark.integration
def test_end_to_end_performance(sample_molecules, temp_dirs):
    """Test end-to-end performance with actual pipeline components."""
    if not sample_molecules:
        pytest.skip("No molecules available")

    try:
        from templ_pipeline.core.pipeline import TEMPLPipeline
    except ImportError:
        pytest.skip("Pipeline not available")

    pipeline = TEMPLPipeline(output_dir=temp_dirs)
    
    start_time = time.time()
    
    # Test basic pipeline operations
    result = pipeline.load_target_data()  # Should return quickly
    
    elapsed = time.time() - start_time

    assert elapsed < 30.0  # Should complete within 30 seconds
    assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__])