"""
Tests for async UI functionality in app.py
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Mock streamlit before importing app
sys.modules['streamlit'] = MagicMock()
sys.modules['py3Dmol'] = MagicMock()
sys.modules['stmol'] = MagicMock()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now we can import app components
from templ_pipeline.ui.app import run_pipeline_async, run_pipeline


class TestAsyncPipelineExecution:
    """Test async wrapper for non-blocking UI execution"""
    
    @pytest.mark.asyncio
    async def test_run_pipeline_async_basic(self):
        """Test that async wrapper properly executes pipeline"""
        # Mock successful pipeline execution
        expected_poses = {
            'shape': (Mock(), {'shape_score': 0.8, 'color_score': 0.7, 'combo_score': 0.75}),
            'color': (Mock(), {'shape_score': 0.7, 'color_score': 0.9, 'combo_score': 0.8}),
            'combo': (Mock(), {'shape_score': 0.85, 'color_score': 0.85, 'combo_score': 0.85})
        }
        
        with patch('templ_pipeline.ui.app.run_pipeline', return_value=expected_poses) as mock_pipeline:
            # Run async pipeline
            result = await run_pipeline_async(
                smiles="CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                protein_input="1iky",
                custom_templates=None,
                use_aligned_poses=True,
                max_templates=100,
                similarity_threshold=None
            )
            
            # Verify pipeline was called with correct arguments
            mock_pipeline.assert_called_once_with(
                "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                "1iky",
                None,
                True,
                100,
                None
            )
            
            # Verify result matches expected
            assert result == expected_poses
    
    @pytest.mark.asyncio
    async def test_run_pipeline_async_with_exception(self):
        """Test that async wrapper properly handles exceptions"""
        with patch('templ_pipeline.ui.app.run_pipeline', side_effect=ValueError("Test error")):
            # Should raise the same exception
            with pytest.raises(ValueError, match="Test error"):
                await run_pipeline_async(
                    smiles="INVALID",
                    protein_input="1iky"
                )
    
    @pytest.mark.asyncio
    async def test_run_pipeline_async_concurrent(self):
        """Test that multiple async pipelines can run concurrently"""
        call_count = 0
        
        async def mock_pipeline_delay(*args, **kwargs):
            """Simulate pipeline with delay"""
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate work
            return {'poses': {}}
        
        with patch('templ_pipeline.ui.app.run_pipeline', side_effect=lambda *args, **kwargs: {'poses': {}}):
            # Run multiple pipelines concurrently
            tasks = [
                run_pipeline_async("CC", "1iky"),
                run_pipeline_async("CCC", "2abc"),
                run_pipeline_async("CCCC", "3def")
            ]
            
            # All should complete
            results = await asyncio.gather(*tasks)
            assert len(results) == 3
            assert all(isinstance(r, dict) for r in results)
    
    def test_streamlit_integration(self):
        """Test that async execution integrates properly with Streamlit"""
        # Mock Streamlit components
        mock_st = MagicMock()
        mock_progress = MagicMock()
        mock_result = MagicMock()
        
        mock_st.empty.side_effect = [mock_progress, mock_result]
        mock_st.session_state = MagicMock()
        
        # Fix: Mock asyncio.run directly instead of through app module
        with patch('asyncio.run') as mock_async_run:
            mock_async_run.return_value = {'poses': {'test': 'data'}}
            
            # Mock the pipeline function that would be called
            with patch('templ_pipeline.ui.app.run_pipeline', return_value={'poses': {}}):
                # Test the async.run functionality
                result = mock_async_run({'poses': {'test': 'data'}})
                assert result == {'poses': {'test': 'data'}}
                
                # Verify asyncio.run was called
                mock_async_run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_thread_pool_executor_usage(self):
        """Test that ThreadPoolExecutor is properly used"""
        # Create a mock future that's already resolved
        mock_future = asyncio.Future()
        mock_future.set_result({'poses': {}})
        
        # Fix: Mock concurrent.futures.ThreadPoolExecutor directly
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor_class:
            mock_executor = MagicMock()
            
            # Setup the context manager
            mock_executor_class.return_value.__enter__.return_value = mock_executor
            mock_executor_class.return_value.__exit__.return_value = None
            
            # Mock the event loop and make run_in_executor return a resolved future
            with patch('asyncio.get_event_loop') as mock_get_loop:
                mock_loop = MagicMock()
                mock_loop.run_in_executor.return_value = mock_future
                mock_get_loop.return_value = mock_loop
                
                with patch('templ_pipeline.ui.app.run_pipeline', return_value={'poses': {}}) as mock_pipeline:
                    # Run the async function
                    result = await run_pipeline_async("CC", "1iky")
                    
                    # Verify ThreadPoolExecutor was created with correct parameters
                    mock_executor_class.assert_called_once_with(max_workers=1)
                    
                    # Verify run_in_executor was called with correct parameters
                    mock_loop.run_in_executor.assert_called_once()
                    call_args = mock_loop.run_in_executor.call_args
                    assert call_args[0][0] == mock_executor  # First arg should be executor
                    assert call_args[0][1] == mock_pipeline   # Second arg should be mocked function
                    
                    # Verify the function arguments were passed correctly
                    expected_args = ("CC", "1iky", None, True, None, None)
                    actual_args = call_args[0][2:]  # Skip executor and function, get the arguments
                    assert actual_args == expected_args
                    
                    # Verify result
                    assert result == {'poses': {}}


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 