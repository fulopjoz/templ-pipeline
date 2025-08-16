"""
Test cases for UI components.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Configure for Streamlit testing
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

try:
    from templ_pipeline.ui.components import header, status_bar
    from templ_pipeline.ui.config.settings import AppConfig
    from templ_pipeline.ui.core.session_manager import SessionManager
    from templ_pipeline.ui.config.constants import VERSION, COLORS
except ImportError:
    # Fall back to local imports for development
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from templ_pipeline.ui.components import header, status_bar
    from templ_pipeline.ui.config.settings import AppConfig
    from templ_pipeline.ui.core.session_manager import SessionManager
    from templ_pipeline.ui.config.constants import VERSION, COLORS


class TestHeaderComponent(unittest.TestCase):
    """Test cases for header component."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=AppConfig)
        self.mock_config.app_name = "TEMPL Pipeline"
        self.mock_config.app_description = "Template-based Protein-Ligand Pose Prediction"
        self.mock_config.ui_settings = {"show_technical_details": False}
        
        self.mock_session = Mock(spec=SessionManager)
        self.mock_session.get_session_info.return_value = {"session_id": "test_123"}

    @patch('templ_pipeline.ui.components.header.st')
    def test_render_header_basic(self, mock_st):
        """Test basic header rendering."""
        header.render_header(self.mock_config, self.mock_session)
        
        # Verify streamlit markdown was called
        self.assertTrue(mock_st.markdown.called)
        
        # Check that markdown was called at least twice (CSS + content)
        self.assertGreaterEqual(mock_st.markdown.call_count, 2)
        
        # Verify header content includes app name
        calls = mock_st.markdown.call_args_list
        header_content = str(calls)
        self.assertIn("TEMPL Pipeline", header_content)
        self.assertIn("Template-based Protein-Ligand Pose Prediction", header_content)

    @patch('templ_pipeline.ui.components.header.st')
    def test_render_header_with_version(self, mock_st):
        """Test header rendering includes version."""
        header.render_header(self.mock_config, self.mock_session)
        
        calls = mock_st.markdown.call_args_list
        header_content = str(calls)
        self.assertIn(f"Version {VERSION}", header_content)

    @patch('templ_pipeline.ui.components.header.st')
    def test_render_header_with_debug_mode(self, mock_st):
        """Test header rendering with debug mode enabled."""
        self.mock_config.ui_settings = {"show_technical_details": True}
        
        # Mock the expander context manager
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = Mock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = Mock(return_value=None)
        
        header.render_header(self.mock_config, self.mock_session)
        
        # Verify expander was created for session information
        mock_st.expander.assert_called_once_with("Session Information", expanded=False)
        
        # Verify session info was retrieved and displayed
        self.mock_session.get_session_info.assert_called_once()

    @patch('templ_pipeline.ui.components.header.st')
    def test_render_header_css_styling(self, mock_st):
        """Test that header includes CSS styling."""
        header.render_header(self.mock_config, self.mock_session)
        
        calls = mock_st.markdown.call_args_list
        css_content = str(calls[0])  # First call should be CSS
        
        # Verify key CSS classes are present
        self.assertIn("header-container", css_content)
        self.assertIn("header-title", css_content)
        self.assertIn("header-subtitle", css_content)
        self.assertIn("header-version", css_content)
        
        # Verify colors are used
        self.assertIn(COLORS['background'], css_content)
        self.assertIn(COLORS['text'], css_content)

    @patch('templ_pipeline.ui.components.header.st')
    def test_render_header_no_debug_mode(self, mock_st):
        """Test header rendering without debug mode."""
        self.mock_config.ui_settings = {"show_technical_details": False}
        
        header.render_header(self.mock_config, self.mock_session)
        
        # Verify expander was not created
        mock_st.expander.assert_not_called()
        
        # Verify session info was not retrieved
        self.mock_session.get_session_info.assert_not_called()


class TestStatusBarComponent(unittest.TestCase):
    """Test cases for status bar component."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock(spec=SessionManager)
        
        # Mock the columns return value with context manager support
        self.mock_columns = []
        for i in range(4):
            mock_col = MagicMock()
            mock_col.__enter__ = Mock(return_value=mock_col)
            mock_col.__exit__ = Mock(return_value=None)
            self.mock_columns.append(mock_col)

    @patch('templ_pipeline.ui.components.status_bar.st')
    def test_render_status_bar_idle(self, mock_st):
        """Test status bar rendering in idle state."""
        from templ_pipeline.ui.components.status_bar import render_status_bar
        
        # Mock session info
        self.mock_session.get_session_info.return_value = {
            "session_id": "test_123",
            "start_time": None,
            "pipeline_runs": 0,
            "has_results": False,
            "has_input": False,
            "memory_stats": {}
        }
        
        # Mock st.columns to return our mock columns
        mock_st.columns.return_value = self.mock_columns
        
        render_status_bar(self.mock_session)
        
        # Verify streamlit calls were made
        self.assertTrue(mock_st.markdown.called)
        self.assertTrue(mock_st.columns.called)
        self.assertTrue(mock_st.caption.called)

    @patch('templ_pipeline.ui.components.status_bar.st')
    def test_render_status_bar_with_results(self, mock_st):
        """Test status bar rendering with results available."""
        from templ_pipeline.ui.components.status_bar import render_status_bar
        
        # Mock session info with results
        self.mock_session.get_session_info.return_value = {
            "session_id": "test_456",
            "start_time": "2025-01-01T00:00:00",
            "duration_seconds": 300,
            "pipeline_runs": 2,
            "has_results": True,
            "has_input": True,
            "memory_stats": {"cache_size_mb": 25.5}
        }
        
        # Mock st.columns to return our mock columns
        mock_st.columns.return_value = self.mock_columns
        
        render_status_bar(self.mock_session)
        
        # Verify streamlit calls were made
        self.assertTrue(mock_st.markdown.called)
        self.assertTrue(mock_st.columns.called)
        self.assertTrue(mock_st.caption.called)
        
        # Verify session info was retrieved
        self.mock_session.get_session_info.assert_called_once()

    @patch('templ_pipeline.ui.components.status_bar.st')
    def test_render_status_bar_minimal_info(self, mock_st):
        """Test status bar rendering with minimal session info."""
        from templ_pipeline.ui.components.status_bar import render_status_bar
        
        # Mock session info with minimal data
        self.mock_session.get_session_info.return_value = {
            "session_id": "minimal_test"
        }
        
        # Mock st.columns to return our mock columns  
        mock_st.columns.return_value = self.mock_columns
        
        render_status_bar(self.mock_session)
        
        # Verify streamlit calls were made
        self.assertTrue(mock_st.markdown.called)
        self.assertTrue(mock_st.columns.called)
        self.assertTrue(mock_st.caption.called)


class TestUIComponentsIntegration(unittest.TestCase):
    """Integration tests for UI components."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=AppConfig)
        self.mock_config.app_name = "TEMPL Pipeline"
        self.mock_config.app_description = "Test Description"
        self.mock_config.ui_settings = {}
        
        self.mock_session = Mock(spec=SessionManager)

    @patch('templ_pipeline.ui.components.header.st')
    @patch('templ_pipeline.ui.components.status_bar.st')
    def test_components_work_together(self, mock_status_st, mock_header_st):
        """Test that components can be rendered together."""
        from templ_pipeline.ui.components.status_bar import render_status_bar
        
        # Mock session info for status bar
        self.mock_session.get_session_info.return_value = {"session_id": "test"}
        
        # Mock st.columns for status bar with context manager support
        mock_columns = []
        for i in range(4):
            mock_col = MagicMock()
            mock_col.__enter__ = Mock(return_value=mock_col)
            mock_col.__exit__ = Mock(return_value=None)
            mock_columns.append(mock_col)
        mock_status_st.columns.return_value = mock_columns
        
        # Render both components
        header.render_header(self.mock_config, self.mock_session)
        render_status_bar(self.mock_session)
        
        # Verify both components made streamlit calls
        self.assertTrue(mock_header_st.markdown.called)
        self.assertTrue(mock_status_st.markdown.called)

    def test_constants_are_defined(self):
        """Test that required constants are properly defined."""
        self.assertIsNotNone(VERSION)
        self.assertIsInstance(VERSION, str)
        
        self.assertIsInstance(COLORS, dict)
        self.assertIn('background', COLORS)
        self.assertIn('text', COLORS)

    def test_component_imports(self):
        """Test that components can be imported successfully."""
        # Test that we can import all components
        from templ_pipeline.ui.components import header, status_bar
        
        # Verify they have the expected functions
        self.assertTrue(hasattr(header, 'render_header'))
        self.assertTrue(hasattr(status_bar, 'render_status_bar'))

    @patch('templ_pipeline.ui.components.header.st')
    def test_header_error_handling(self, mock_st):
        """Test header component error handling."""
        # Test with None config
        with self.assertRaises(AttributeError):
            header.render_header(None, self.mock_session)
        
        # Test with missing attributes
        broken_config = Mock()
        del broken_config.app_name  # Remove required attribute
        
        with self.assertRaises(AttributeError):
            header.render_header(broken_config, self.mock_session)

    def test_config_validation(self):
        """Test configuration object validation."""
        # Valid config should work
        valid_config = Mock(spec=AppConfig)
        valid_config.app_name = "Test App"
        valid_config.app_description = "Test Description"
        valid_config.ui_settings = {}
        
        # Should not raise any exceptions
        self.assertIsNotNone(valid_config.app_name)
        self.assertIsInstance(valid_config.ui_settings, dict)


if __name__ == "__main__":
    unittest.main()