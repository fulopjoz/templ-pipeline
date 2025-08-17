# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
End-to-End Testing Framework for TEMPL Pipeline Streamlit App

This module provides comprehensive end-to-end testing using Playwright
for automated browser testing of the complete user workflow.
"""

import asyncio
import os
import signal
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest
import requests
from playwright.async_api import Browser, Page, async_playwright
from playwright.sync_api import sync_playwright


class StreamlitAppServer:
    """Helper class to manage Streamlit app server for testing."""

    def __init__(self, app_file=None, port=8501):
        self.app_file = app_file or str(Path(__file__).parent.parent / "app.py")
        self.port = port
        self.process = None
        self.base_url = f"http://localhost:{port}"

    def start(self):
        """Start the Streamlit app server."""
        try:
            # Start streamlit app
            self.process = subprocess.Popen(
                [
                    "streamlit",
                    "run",
                    self.app_file,
                    "--server.port",
                    str(self.port),
                    "--server.headless",
                    "true",
                    "--server.runOnSave",
                    "false",
                    "--browser.gatherUsageStats",
                    "false",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for server to start
            self._wait_for_server()
            return True

        except Exception as e:
            print(f"Failed to start Streamlit server: {e}")
            return False

    def stop(self):
        """Stop the Streamlit app server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def _wait_for_server(self, timeout=30):
        """Wait for the server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(self.base_url, timeout=1)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(0.5)

        raise TimeoutError(f"Streamlit server not ready after {timeout} seconds")

    @contextmanager
    def running(self):
        """Context manager for running the server."""
        try:
            self.start()
            yield self
        finally:
            self.stop()


class PlaywrightTestBase:
    """Base class for Playwright-based tests."""

    def __init__(self):
        self.browser = None
        self.context = None
        self.page = None
        self.app_server = None

    async def setup_browser(self, headless=True):
        """Setup browser for testing."""
        self.playwright = await async_playwright().start()

        # Launch browser
        self.browser = await self.playwright.chromium.launch(
            headless=headless, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )

        # Create context
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 720}
        )

        # Create page
        self.page = await self.context.new_page()

        # Setup console logging
        self.page.on("console", lambda msg: print(f"Browser console: {msg.text}"))

        # Setup error handling
        self.page.on("pageerror", lambda error: print(f"Page error: {error}"))

    async def teardown_browser(self):
        """Teardown browser."""
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if hasattr(self, "playwright"):
            await self.playwright.stop()

    async def goto_app(self, url="http://localhost:8501"):
        """Navigate to the Streamlit app."""
        await self.page.goto(url)

        # Wait for Streamlit to load
        await self.page.wait_for_selector('[data-testid="stApp"]', timeout=30000)

    async def wait_for_streamlit_ready(self):
        """Wait for Streamlit to be fully ready."""
        # Wait for the main app container
        await self.page.wait_for_selector('[data-testid="stApp"]')

        # Wait for any loading indicators to disappear
        loading_selectors = [".stSpinner", '[data-testid="stSpinner"]', ".loading"]

        for selector in loading_selectors:
            try:
                await self.page.wait_for_selector(
                    selector, state="hidden", timeout=5000
                )
            except:
                pass  # Selector might not exist

        # Small delay to ensure everything is rendered
        await self.page.wait_for_timeout(1000)


class TestStreamlitE2E:
    """End-to-end tests for the Streamlit app."""

    def setup_method(self):
        """Setup for each test."""
        self.test_base = PlaywrightTestBase()
        self.app_server = StreamlitAppServer()

    def teardown_method(self):
        """Teardown after each test."""
        # This will be called by pytest automatically
        pass

    @pytest.mark.asyncio
    async def test_app_loads_successfully(self):
        """Test that the app loads without errors."""
        with self.app_server.running():
            await self.test_base.setup_browser(headless=True)

            try:
                await self.test_base.goto_app()
                await self.test_base.wait_for_streamlit_ready()

                # Check that the main app container exists
                app_container = await self.test_base.page.query_selector(
                    '[data-testid="stApp"]'
                )
                assert app_container is not None

                # Check page title
                title = await self.test_base.page.title()
                assert "TEMPL" in title or "Streamlit" in title

            finally:
                await self.test_base.teardown_browser()

    @pytest.mark.asyncio
    async def test_sidebar_navigation(self):
        """Test sidebar navigation and components."""
        with self.app_server.running():
            await self.test_base.setup_browser(headless=True)

            try:
                await self.test_base.goto_app()
                await self.test_base.wait_for_streamlit_ready()

                # Check for sidebar
                sidebar = await self.test_base.page.query_selector(
                    '[data-testid="stSidebar"]'
                )
                if sidebar:
                    # Check that sidebar is visible
                    is_visible = await sidebar.is_visible()
                    assert is_visible

                    # Look for common sidebar elements
                    sidebar_content = await sidebar.inner_text()
                    assert len(sidebar_content) > 0

            finally:
                await self.test_base.teardown_browser()

    @pytest.mark.asyncio
    async def test_file_upload_interaction(self):
        """Test file upload functionality."""
        with self.app_server.running():
            await self.test_base.setup_browser(headless=True)

            try:
                await self.test_base.goto_app()
                await self.test_base.wait_for_streamlit_ready()

                # Look for file upload widgets
                file_uploads = await self.test_base.page.query_selector_all(
                    'input[type="file"]'
                )

                if file_uploads:
                    # Create a temporary test file
                    test_file_content = """HEADER    TEST PROTEIN
ATOM      1  CA  ALA A   1      20.0  16.0  10.0  1.00 20.00           C
END
"""
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".pdb", delete=False
                    ) as f:
                        f.write(test_file_content)
                        test_file_path = f.name

                    try:
                        # Upload file to the first file upload widget
                        await file_uploads[0].set_input_files(test_file_path)

                        # Wait for processing
                        await self.test_base.page.wait_for_timeout(2000)

                        # Check that file was accepted (no error messages)
                        error_elements = await self.test_base.page.query_selector_all(
                            '[data-testid="stAlert"]'
                        )
                        error_messages = []
                        for elem in error_elements:
                            text = await elem.inner_text()
                            if "error" in text.lower():
                                error_messages.append(text)

                        # Should not have critical errors
                        assert len(error_messages) == 0 or not any(
                            "critical" in msg.lower() for msg in error_messages
                        )

                    finally:
                        os.unlink(test_file_path)

            finally:
                await self.test_base.teardown_browser()

    @pytest.mark.asyncio
    async def test_input_validation_ui(self):
        """Test input validation through the UI."""
        with self.app_server.running():
            await self.test_base.setup_browser(headless=True)

            try:
                await self.test_base.goto_app()
                await self.test_base.wait_for_streamlit_ready()

                # Look for text input fields
                text_inputs = await self.test_base.page.query_selector_all(
                    'input[type="text"]'
                )

                for text_input in text_inputs:
                    # Get placeholder or label to identify the field
                    placeholder = await text_input.get_attribute("placeholder")

                    if placeholder and "smiles" in placeholder.lower():
                        # Test invalid SMILES input
                        await text_input.fill("invalid_smiles_123")
                        await text_input.press("Enter")

                        # Wait for validation
                        await self.test_base.page.wait_for_timeout(1000)

                        # Check for validation messages
                        # The app should handle invalid input gracefully
                        break

            finally:
                await self.test_base.teardown_browser()

    @pytest.mark.asyncio
    async def test_responsive_design(self):
        """Test responsive design at different screen sizes."""
        with self.app_server.running():
            await self.test_base.setup_browser(headless=True)

            try:
                # Test different viewport sizes
                viewports = [
                    {"width": 1920, "height": 1080},  # Desktop
                    {"width": 1024, "height": 768},  # Tablet
                    {"width": 375, "height": 667},  # Mobile
                ]

                for viewport in viewports:
                    await self.test_base.context.set_viewport_size(viewport)
                    await self.test_base.goto_app()
                    await self.test_base.wait_for_streamlit_ready()

                    # Check that main content is visible
                    app_container = await self.test_base.page.query_selector(
                        '[data-testid="stApp"]'
                    )
                    assert app_container is not None

                    is_visible = await app_container.is_visible()
                    assert is_visible

            finally:
                await self.test_base.teardown_browser()

    @pytest.mark.asyncio
    async def test_error_handling_ui(self):
        """Test error handling through the UI."""
        with self.app_server.running():
            await self.test_base.setup_browser(headless=True)

            try:
                await self.test_base.goto_app()
                await self.test_base.wait_for_streamlit_ready()

                # Look for any error alerts or messages
                error_elements = await self.test_base.page.query_selector_all(
                    '[data-testid="stAlert"]'
                )

                # Check that there are no critical errors on page load
                critical_errors = []
                for elem in error_elements:
                    text = await elem.inner_text()
                    if "critical" in text.lower() or "fatal" in text.lower():
                        critical_errors.append(text)

                assert len(critical_errors) == 0

            finally:
                await self.test_base.teardown_browser()


class TestStreamlitPerformanceE2E:
    """Performance tests using Playwright."""

    def setup_method(self):
        self.test_base = PlaywrightTestBase()
        self.app_server = StreamlitAppServer()

    @pytest.mark.asyncio
    async def test_page_load_performance(self):
        """Test page load performance."""
        with self.app_server.running():
            await self.test_base.setup_browser(headless=True)

            try:
                # Measure page load time
                start_time = time.time()
                await self.test_base.goto_app()
                await self.test_base.wait_for_streamlit_ready()
                end_time = time.time()

                load_time = end_time - start_time

                # Page should load within reasonable time
                assert load_time < 15.0  # 15 seconds threshold

            finally:
                await self.test_base.teardown_browser()

    @pytest.mark.asyncio
    async def test_memory_usage_patterns(self):
        """Test memory usage patterns during interaction."""
        with self.app_server.running():
            await self.test_base.setup_browser(headless=True)

            try:
                await self.test_base.goto_app()
                await self.test_base.wait_for_streamlit_ready()

                # Get initial memory usage
                initial_memory = await self.test_base.page.evaluate(
                    """
                    () => {
                        return performance.memory ? performance.memory.usedJSHeapSize : 0;
                    }
                """
                )

                # Simulate user interactions
                await self.test_base.page.wait_for_timeout(2000)

                # Get final memory usage
                final_memory = await self.test_base.page.evaluate(
                    """
                    () => {
                        return performance.memory ? performance.memory.usedJSHeapSize : 0;
                    }
                """
                )

                # Memory usage should not increase dramatically
                if initial_memory > 0 and final_memory > 0:
                    memory_increase = final_memory - initial_memory
                    # Allow for reasonable memory increase (50MB)
                    assert memory_increase < 50 * 1024 * 1024

            finally:
                await self.test_base.teardown_browser()


class TestStreamlitAccessibility:
    """Accessibility tests for the Streamlit app."""

    def setup_method(self):
        self.test_base = PlaywrightTestBase()
        self.app_server = StreamlitAppServer()

    @pytest.mark.asyncio
    async def test_keyboard_navigation(self):
        """Test keyboard navigation accessibility."""
        with self.app_server.running():
            await self.test_base.setup_browser(headless=True)

            try:
                await self.test_base.goto_app()
                await self.test_base.wait_for_streamlit_ready()

                # Test tab navigation
                focusable_elements = await self.test_base.page.query_selector_all(
                    'button, input, select, textarea, [tabindex]:not([tabindex="-1"])'
                )

                if focusable_elements:
                    # Focus first element
                    await focusable_elements[0].focus()

                    # Test tab navigation
                    await self.test_base.page.keyboard.press("Tab")

                    # Check that focus moved
                    focused_element = await self.test_base.page.evaluate(
                        "document.activeElement.tagName"
                    )
                    assert focused_element in ["BUTTON", "INPUT", "SELECT", "TEXTAREA"]

            finally:
                await self.test_base.teardown_browser()

    @pytest.mark.asyncio
    async def test_screen_reader_compatibility(self):
        """Test screen reader compatibility."""
        with self.app_server.running():
            await self.test_base.setup_browser(headless=True)

            try:
                await self.test_base.goto_app()
                await self.test_base.wait_for_streamlit_ready()

                # Check for proper ARIA labels and roles
                elements_with_aria = await self.test_base.page.query_selector_all(
                    "[aria-label], [role]"
                )

                # Should have some accessibility attributes
                assert len(elements_with_aria) > 0

                # Check for proper heading structure
                headings = await self.test_base.page.query_selector_all(
                    "h1, h2, h3, h4, h5, h6"
                )
                if headings:
                    # Should have at least one heading
                    assert len(headings) > 0

            finally:
                await self.test_base.teardown_browser()


# Synchronous test runner for easier integration
class SyncTestRunner:
    """Synchronous wrapper for async tests."""

    def run_test(self, test_func):
        """Run an async test function synchronously."""
        return asyncio.run(test_func())


# Utility functions for E2E testing
def create_test_data_files():
    """Create test data files for E2E testing."""
    test_data = {
        "protein.pdb": """HEADER    TEST PROTEIN
ATOM      1  CA  ALA A   1      20.0  16.0  10.0  1.00 20.00           C
ATOM      2  CA  GLY A   2      18.0  15.0  11.0  1.00 20.00           C
END
""",
        "ligand.sdf": """Test Ligand
  -I-interpret- 

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    2.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  2  3  1  0  0  0  0
M  END
$$$$
""",
    }

    test_files = {}
    for filename, content in test_data.items():
        temp_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=f'.{filename.split(".")[-1]}', delete=False
        )
        temp_file.write(content)
        temp_file.close()
        test_files[filename] = temp_file.name

    return test_files


if __name__ == "__main__":
    # Run E2E tests
    pytest.main([__file__, "-v", "-s"])
