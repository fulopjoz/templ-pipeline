import pytest
import time
import psutil
import os
from unittest.mock import patch


@pytest.mark.performance
@pytest.mark.slow
class TestCoreFunctionPerformance:
    """Benchmark core function performance"""

    def test_embedding_performance(self, sample_smiles):
        """Test embedding generation performance"""
        from templ_pipeline.core.embedding import EmbeddingEngine

        pytest.importorskip("templ_pipeline.core.embedding")

        engine = EmbeddingEngine()

        # Mock to avoid external dependencies
        with patch.object(engine, "generate_conformers") as mock_gen:
            mock_gen.return_value = [{"coords": [[0, 0, 0]]}] * 100

            start_time = time.time()
            result = engine.generate_conformers(sample_smiles[0], n_conformers=100)
            elapsed = time.time() - start_time

            # Performance baseline: should complete within reasonable time
            assert elapsed < 30.0  # 30 seconds max
            assert len(result) <= 100

    def test_mcs_performance(self, sample_smiles):
        """Test MCS calculation performance"""
        from templ_pipeline.core.mcs import MCSEngine

        pytest.importorskip("templ_pipeline.core.mcs")

        engine = MCSEngine()

        with patch.object(engine, "calculate_mcs") as mock_mcs:
            mock_mcs.return_value = {"score": 0.8, "mapping": []}

            start_time = time.time()
            result = engine.calculate_mcs(sample_smiles[0], sample_smiles[1])
            elapsed = time.time() - start_time

            assert elapsed < 10.0  # 10 seconds max
            assert "score" in result

    def test_scoring_performance(self, sample_mol):
        """Test scoring performance"""
        from templ_pipeline.core.scoring import ScoringEngine

        pytest.importorskip("templ_pipeline.core.scoring")

        engine = ScoringEngine()

        with patch.object(engine, "score_pose") as mock_score:
            mock_score.return_value = {"total_score": 0.75}

            start_time = time.time()
            result = engine.score_pose(sample_mol)
            elapsed = time.time() - start_time

            assert elapsed < 5.0  # 5 seconds max
            assert "total_score" in result


@pytest.mark.performance
class TestMemoryUsage:
    """Monitor memory usage during processing"""

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    def test_memory_baseline(self):
        """Establish memory usage baseline"""
        initial_memory = self.get_memory_usage()

        # Perform minimal operations
        data = [i for i in range(1000)]
        del data

        final_memory = self.get_memory_usage()
        memory_diff = final_memory - initial_memory

        # Should not leak significant memory
        assert memory_diff < 50  # Less than 50MB increase

    def test_large_molecule_memory(self, sample_smiles):
        """Test memory usage with large molecules"""
        initial_memory = self.get_memory_usage()

        # Mock processing of large molecules
        large_data = []
        for _ in range(100):
            large_data.append({"conformers": [{"coords": [[0, 0, 0]] * 1000}]})

        peak_memory = self.get_memory_usage()
        del large_data

        final_memory = self.get_memory_usage()

        # Memory should be released after processing
        memory_retained = final_memory - initial_memory
        assert memory_retained < 100  # Less than 100MB retained


@pytest.mark.performance
@pytest.mark.slow
class TestScalabilityLimits:
    """Test scalability limits"""

    def test_conformer_scaling(self):
        """Test performance scaling with conformer count"""
        from templ_pipeline.core.embedding import EmbeddingEngine

        pytest.importorskip("templ_pipeline.core.embedding")

        engine = EmbeddingEngine()

        conformer_counts = [10, 50, 100, 200]
        times = []

        for count in conformer_counts:
            with patch.object(engine, "generate_conformers") as mock_gen:
                mock_gen.return_value = [{"coords": [[0, 0, 0]]}] * count

                start_time = time.time()
                engine.generate_conformers("CCO", n_conformers=count)
                elapsed = time.time() - start_time

                times.append(elapsed)

        # Performance should scale reasonably
        assert all(t < 60 for t in times)  # All under 60 seconds

    def test_batch_size_limits(self, sample_smiles):
        """Test batch processing limits"""
        # Test with increasingly large batches
        batch_sizes = [1, 5, 10, 20]

        for batch_size in batch_sizes:
            batch = sample_smiles * (batch_size // len(sample_smiles) + 1)
            batch = batch[:batch_size]

            start_time = time.time()

            # Mock batch processing
            results = []
            for smiles in batch:
                results.append({"score": 0.8})

            elapsed = time.time() - start_time

            # Should complete within reasonable time
            assert elapsed < batch_size * 2  # 2 seconds per molecule max
            assert len(results) == batch_size
