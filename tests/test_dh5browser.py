"""
Tests for dh5browser module.

These tests verify the browser functionality without actually opening GUI windows.
They test data source creation, viewer configuration, and command-line argument parsing.
"""

import os
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

# Set Qt to use offscreen platform for headless testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Try to import ephyviewer for real browser tests
try:
    import ephyviewer

    EPHYVIEWER_AVAILABLE = True
except ImportError:
    EPHYVIEWER_AVAILABLE = False
    # Mock ephyviewer for tests that don't need it
    sys.modules["ephyviewer"] = MagicMock()

import dh5cli.dh5browser as browser_module  # noqa: E402
from dh5neo import DH5IO


class TestBrowserModule:
    """Test basic browser module functionality."""

    def test_dh5mainviewer_class_exists(self):
        """Test that DH5MainViewer class exists."""
        assert hasattr(browser_module, "DH5MainViewer")

    def test_imports(self):
        """Test that browser module can be imported."""
        assert hasattr(browser_module, "create_browser")
        assert hasattr(browser_module, "main")

    def test_create_browser_signature(self):
        """Test that create_browser has the expected signature."""
        import inspect

        sig = inspect.signature(browser_module.create_browser)
        params = list(sig.parameters.keys())
        assert "dh5_file" in params
        assert "trial_index" in params
        assert "cache_size" in params

    def test_segment_cache_class_exists(self):
        """Test that SegmentCache class exists."""
        assert hasattr(browser_module, "SegmentCache")

    def test_navigation_widget_class_exists(self):
        """Test that TrialNavigationWidget class exists."""
        assert hasattr(browser_module, "TrialNavigationWidget")


class TestCommandLineInterface:
    """Test command-line argument parsing."""

    def test_help_argument(self):
        """Test that --help argument works."""
        with patch.object(sys, "argv", ["dh5browser", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                browser_module.main()
            # argparse exits with 0 for --help
            assert exc_info.value.code == 0

    def test_missing_filename(self):
        """Test that missing filename causes error."""
        with patch.object(sys, "argv", ["dh5browser"]):
            with pytest.raises(SystemExit) as exc_info:
                browser_module.main()
            # argparse exits with 2 for missing required argument
            assert exc_info.value.code == 2

    def test_nonexistent_file(self):
        """Test that nonexistent file causes error."""
        with patch.object(sys, "argv", ["dh5browser", "nonexistent_file.dh5"]):
            with pytest.raises(SystemExit) as exc_info:
                browser_module.main()
            # Should exit with 1 for file not found
            assert exc_info.value.code == 1


@pytest.mark.skip(reason="Browser functionality under development")
@pytest.mark.skipif(not EPHYVIEWER_AVAILABLE, reason="ephyviewer not installed")
class TestBrowserWithRealData:
    """Test browser with actual DH5 file (no GUI)."""

    @pytest.fixture(scope="class")
    def qapp(self):
        """Create Qt application once for all tests in this class."""
        app = ephyviewer.mkQApp()
        yield app
        # Clean up after all tests
        app.processEvents()

    @pytest.fixture
    def test_dh5_file(self):
        """Get path to test DH5 file."""
        test_file = pathlib.Path(__file__).parent / "test.dh5"
        if not test_file.exists():
            pytest.skip("Test DH5 file not available")
        return test_file

    @pytest.mark.timeout(10)
    def test_browser_loads_file(self, qapp, test_dh5_file):
        """Test that browser can load a DH5 file without errors."""
        win = None
        try:
            # Create browser - should not raise any exceptions
            win, filename, cache = browser_module.create_browser(
                test_dh5_file, trial_index=0
            )

            # Verify MainViewer was created
            assert win is not None
            assert isinstance(win, browser_module.DH5MainViewer)

            # Verify viewers were added
            assert len(win.viewers) > 0

            # Check that navigation widget exists for multi-trial file
            assert "Trial Navigation" in win.viewers
        finally:
            # Clean up - close window and process events to avoid hanging
            if win is not None:
                win.close()
                qapp.processEvents()

    @pytest.mark.timeout(10)
    def test_browser_invalid_segment(self, qapp, test_dh5_file):
        """Test that invalid trial index raises appropriate error."""
        # Try to create browser with invalid trial index
        # This should exit with error, not raise exception
        import io
        from contextlib import redirect_stderr, redirect_stdout

        captured_output = io.StringIO()
        with redirect_stdout(captured_output), redirect_stderr(captured_output):
            with pytest.raises(SystemExit) as exc_info:
                browser_module.create_browser(test_dh5_file, trial_index=999)
            assert exc_info.value.code == 1

        # Clean up
        qapp.processEvents()

    def test_browser_single_segment_no_navigation(self, tmp_path):
        """Test that single-trial files don't show navigation widget."""
        # This test would require creating a single-trial DH5 file
        # Skipping for now as it requires file creation
        pytest.skip("Requires creating test file with single trial")

    @pytest.mark.timeout(10)
    def test_navigation_widget_created(self, qapp, test_dh5_file):
        """Test that TrialNavigationWidget is created for multi-trial files."""
        win = None
        try:
            win, filename, cache = browser_module.create_browser(
                test_dh5_file, trial_index=0
            )

            # Verify navigation widget exists
            assert "Trial Navigation" in win.viewers
            # ephyviewer stores widgets in a dict with 'dock' and 'widget' keys
            nav_widget_dict = win.viewers["Trial Navigation"]
            nav_widget = nav_widget_dict["widget"]

            # Verify it's a TrialNavigationWidget
            assert isinstance(nav_widget, browser_module.TrialNavigationWidget)

            # Verify it has the name attribute required by ephyviewer
            assert hasattr(nav_widget, "name")
            assert nav_widget.name == "Trial Navigation"

            # Verify it has required ephyviewer methods
            assert hasattr(nav_widget, "get_settings")
            assert hasattr(nav_widget, "set_settings")
        finally:
            # Clean up
            if win is not None:
                win.close()
                qapp.processEvents()

    @pytest.mark.timeout(10)
    def test_navigation_button_clicks(self, qapp, test_dh5_file):
        """Test that navigation buttons work correctly and preserve the widget."""
        win = None
        try:
            win, filename, cache = browser_module.create_browser(
                test_dh5_file, trial_index=0
            )

            # Get navigation widget
            assert "Trial Navigation" in win.viewers
            nav_widget_dict = win.viewers["Trial Navigation"]
            nav_widget = nav_widget_dict["widget"]

            # Initially should be on trial 0
            assert nav_widget.current_trial == 0

            # Click next button (simulate the click)
            nav_widget._on_next()
            qapp.processEvents()

            # Navigation widget should still exist and be on trial 1
            assert "Trial Navigation" in win.viewers
            nav_widget_dict = win.viewers["Trial Navigation"]
            nav_widget = nav_widget_dict["widget"]
            assert nav_widget.current_trial == 1

            # Click previous button
            nav_widget._on_previous()
            qapp.processEvents()

            # Should be back on trial 0
            assert "Trial Navigation" in win.viewers
            nav_widget_dict = win.viewers["Trial Navigation"]
            nav_widget = nav_widget_dict["widget"]
            assert nav_widget.current_trial == 0

        finally:
            # Clean up
            if win is not None:
                win.close()
                qapp.processEvents()


@pytest.mark.skip(reason="Browser functionality under development")
class TestSegmentCache:
    """Test segment caching functionality."""

    @pytest.fixture
    def test_dh5_file(self):
        """Get path to test DH5 file."""
        test_file = pathlib.Path(__file__).parent / "test.dh5"
        if not test_file.exists():
            pytest.skip("Test DH5 file not available")
        return test_file

    @pytest.fixture
    def reader(self, test_dh5_file):
        """Create a DH5IO reader."""
        reader = DH5IO(test_dh5_file)
        reader.parse_header()
        return reader

    def test_cache_initialization(self, reader):
        """Test that cache can be initialized."""
        cache = browser_module.SegmentCache(reader, max_cache_size=5)
        assert cache.max_cache_size == 5
        assert len(cache._cache) == 0
        assert len(cache._access_order) == 0

    def test_cache_miss_loads_segment(self, reader):
        """Test that cache miss loads segment from disk."""
        cache = browser_module.SegmentCache(reader, max_cache_size=5)
        segment = cache.get_segment(0)
        assert segment is not None
        assert len(cache._cache) == 1
        assert 0 in cache._cache

    def test_cache_hit_returns_same_object(self, reader):
        """Test that cache hit returns the same segment object."""
        cache = browser_module.SegmentCache(reader, max_cache_size=5)
        segment1 = cache.get_segment(0)
        segment2 = cache.get_segment(0)
        assert segment1 is segment2
        assert len(cache._cache) == 1

    def test_cache_eviction(self, reader):
        """Test that cache evicts LRU segment when full."""
        cache = browser_module.SegmentCache(reader, max_cache_size=3)

        # Load 3 segments (fill cache)
        seg0 = cache.get_segment(0)
        seg1 = cache.get_segment(1)
        seg2 = cache.get_segment(2)
        assert len(cache._cache) == 3

        # Access segment 0 again to make it most recent
        cache.get_segment(0)

        # Load segment 3 (should evict segment 1 - least recently used)
        seg3 = cache.get_segment(3)
        assert len(cache._cache) == 3
        assert 1 not in cache._cache
        assert 0 in cache._cache
        assert 2 in cache._cache
        assert 3 in cache._cache

    def test_cache_clear(self, reader):
        """Test that cache can be cleared."""
        cache = browser_module.SegmentCache(reader, max_cache_size=5)
        cache.get_segment(0)
        cache.get_segment(1)
        assert len(cache._cache) == 2

        cache.clear()
        assert len(cache._cache) == 0
        assert len(cache._access_order) == 0

    def test_cache_multiple_segments(self, reader):
        """Test caching multiple segments."""
        cache = browser_module.SegmentCache(reader, max_cache_size=10)

        # Load several segments
        for i in range(5):
            seg = cache.get_segment(i)
            assert seg is not None

        assert len(cache._cache) == 5

        # All should be cache hits now
        for i in range(5):
            seg = cache.get_segment(i)
            assert i in cache._cache


@pytest.mark.skip(reason="Browser functionality under development")
@pytest.mark.skipif(not EPHYVIEWER_AVAILABLE, reason="ephyviewer not installed")
class TestChannelSelection:
    """Test channel selection and settings persistence."""

    @pytest.fixture(scope="class")
    def qapp(self):
        """Create Qt application once for all tests in this class."""
        app = ephyviewer.mkQApp()
        yield app
        # Clean up after all tests
        app.processEvents()

    @pytest.fixture
    def test_dh5_file(self):
        """Get path to test DH5 file."""
        test_file = pathlib.Path(__file__).parent / "test.dh5"
        if not test_file.exists():
            pytest.skip("Test DH5 file not available")
        return test_file

    @pytest.mark.timeout(10)
    def test_channel_visibility_control(self, qapp, test_dh5_file):
        """Test that channel visibility can be controlled."""
        win = None
        try:
            win, filename, cache = browser_module.create_browser(
                test_dh5_file, trial_index=0
            )

            # Find the trace viewer
            trace_viewer = None
            for viewer_dict in win.viewers.values():
                widget = viewer_dict["widget"]
                if isinstance(widget, ephyviewer.TraceViewer):
                    trace_viewer = widget
                    break

            assert trace_viewer is not None, "TraceViewer not found"

            # Check that trace viewer has by_channel_params
            assert hasattr(trace_viewer, "by_channel_params")

            # Get number of channels
            nb_channels = trace_viewer.source.nb_channel
            assert nb_channels > 0

            # Get initial visibility state (may be restored from settings)
            initial_ch0_visible = trace_viewer.by_channel_params["ch0", "visible"]

            # Toggle first channel visibility
            trace_viewer.by_channel_params["ch0", "visible"] = not initial_ch0_visible
            visible = trace_viewer.by_channel_params["ch0", "visible"]
            assert visible == (not initial_ch0_visible)

            # Toggle it back
            trace_viewer.by_channel_params["ch0", "visible"] = initial_ch0_visible
            visible = trace_viewer.by_channel_params["ch0", "visible"]
            assert visible == initial_ch0_visible

        finally:
            if win is not None:
                win.close()
                qapp.processEvents()

    @pytest.mark.timeout(10)
    def test_channel_settings_in_get_settings(self, qapp, test_dh5_file):
        """Test that channel visibility is included in get_settings()."""
        win = None
        try:
            win, filename, cache = browser_module.create_browser(
                test_dh5_file, trial_index=0
            )

            # Find the trace viewer
            trace_viewer = None
            for viewer_dict in win.viewers.values():
                widget = viewer_dict["widget"]
                if isinstance(widget, ephyviewer.TraceViewer):
                    trace_viewer = widget
                    break

            assert trace_viewer is not None

            # Hide a channel
            trace_viewer.by_channel_params["ch0", "visible"] = False

            # Get settings
            settings = trace_viewer.get_settings()
            assert settings is not None

            # Check that Channels are in settings
            assert "children" in settings
            children = settings["children"]
            assert "Channels" in children

            # Check that ch0 visibility is False in settings
            channels_settings = children["Channels"]
            ch_children = channels_settings.get("children", {})
            ch0_settings = ch_children.get("ch0", {})
            ch0_params = ch0_settings.get("children", {})
            visible_param = ch0_params.get("visible", {})
            assert visible_param.get("value") is False

        finally:
            if win is not None:
                win.close()
                qapp.processEvents()

    @pytest.mark.timeout(10)
    def test_per_channel_parameters(self, qapp, test_dh5_file):
        """Test that per-channel parameters (gain, offset, color) exist."""
        win = None
        try:
            win, filename, cache = browser_module.create_browser(
                test_dh5_file, trial_index=0
            )

            # Find the trace viewer
            trace_viewer = None
            for viewer_dict in win.viewers.values():
                widget = viewer_dict["widget"]
                if isinstance(widget, ephyviewer.TraceViewer):
                    trace_viewer = widget
                    break

            assert trace_viewer is not None

            # Check that first channel has all expected parameters
            # These are the default per-channel parameters in ephyviewer
            gain = trace_viewer.by_channel_params["ch0", "gain"]
            assert isinstance(gain, (int, float))
            assert gain == 1.0  # Default gain

            offset = trace_viewer.by_channel_params["ch0", "offset"]
            assert isinstance(offset, (int, float))
            assert offset == 0.0  # Default offset

            color = trace_viewer.by_channel_params["ch0", "color"]
            # Color can be a QColor object or string depending on Qt backend
            from PyQt5.QtGui import QColor

            assert isinstance(color, (str, QColor))

            # Test that we can modify these parameters
            trace_viewer.by_channel_params["ch0", "gain"] = 2.0
            new_gain = trace_viewer.by_channel_params["ch0", "gain"]
            assert new_gain == 2.0

        finally:
            if win is not None:
                win.close()
                qapp.processEvents()


@pytest.mark.skip(reason="Browser functionality under development")
class TestBrowserIntegration:
    """Integration tests for browser functionality."""

    def test_example_script_exists(self):
        """Test that example script exists."""
        example_file = (
            pathlib.Path(__file__).parent.parent / "examples" / "example_dh5_browser.py"
        )
        assert example_file.exists()

    def test_readme_exists(self):
        """Test that browser README exists."""
        readme_file = (
            pathlib.Path(__file__).parent.parent
            / "src"
            / "dh5cli"
            / "BROWSER_README.md"
        )
        assert readme_file.exists()

    def test_entry_point_defined(self):
        """Test that dh5browser entry point is defined in pyproject.toml."""
        pyproject_file = pathlib.Path(__file__).parent.parent / "pyproject.toml"
        assert pyproject_file.exists()

        content = pyproject_file.read_text()
        assert "dh5browser" in content
        assert "dh5cli.dh5browser:main" in content

    def test_browser_dependencies_in_pyproject(self):
        """Test that browser dependencies are defined in pyproject.toml."""
        pyproject_file = pathlib.Path(__file__).parent.parent / "pyproject.toml"
        content = pyproject_file.read_text()

        # Check for browser optional dependency group
        assert "browser" in content
        assert "ephyviewer" in content

    def test_segment_navigation_documentation_exists(self):
        """Test that trial navigation documentation exists."""
        doc_file = (
            pathlib.Path(__file__).parent.parent / "docs" / "SEGMENT_NAVIGATION.md"
        )
        assert doc_file.exists()

        content = doc_file.read_text()
        # Note: Documentation may still reference "Segment" in some contexts
        # since Neo uses segment as the data structure name
        assert "SegmentCache" in content

    def test_channel_selection_documentation_exists(self):
        """Test that channel selection documentation exists."""
        doc_file = (
            pathlib.Path(__file__).parent.parent / "docs" / "CHANNEL_SELECTION_GUIDE.md"
        )
        assert doc_file.exists()

        content = doc_file.read_text()
        assert "channel" in content.lower()
        assert "visible" in content.lower()


class TestBrowserDocumentation:
    """Test that documentation is complete."""

    def test_module_docstring(self):
        """Test that module has a docstring."""
        assert browser_module.__doc__ is not None
        assert len(browser_module.__doc__) > 0

    def test_create_browser_docstring(self):
        """Test that create_browser function has docstring."""
        assert browser_module.create_browser.__doc__ is not None
        assert "Parameters" in browser_module.create_browser.__doc__
        assert "Returns" in browser_module.create_browser.__doc__

    def test_main_docstring(self):
        """Test that main function has docstring."""
        assert browser_module.main.__doc__ is not None


class TestDH5MainViewerSettings:
    """Test settings persistence for DH5MainViewer."""

    @pytest.fixture
    def qapp(self):
        """Create a QApplication instance for tests that need one."""
        app = ephyviewer.mkQApp()
        yield app
        app.processEvents()

    def test_dh5mainviewer_class_exists(self):
        """Test that DH5MainViewer class exists."""
        assert hasattr(browser_module, "DH5MainViewer")

    def test_dh5mainviewer_extends_mainviewer(self):
        """Test that DH5MainViewer extends ephyviewer.MainViewer."""
        assert issubclass(
            browser_module.DH5MainViewer, browser_module.ephyviewer.MainViewer
        )

    def test_dh5mainviewer_has_save_method(self):
        """Test that DH5MainViewer has save_all_settings method."""
        assert hasattr(browser_module.DH5MainViewer, "save_all_settings")

    @pytest.fixture
    def test_dh5_file(self):
        """Get path to test DH5 file."""
        test_file = pathlib.Path(__file__).parent / "test.dh5"
        if not test_file.exists():
            pytest.skip("Test DH5 file not available")
        return test_file

    def test_settings_saved_on_close(self, qapp, test_dh5_file):
        """Test that window state is saved when browser is closed."""
        try:
            from PySide6.QtCore import QSettings
        except ImportError:
            from PyQt5.QtCore import QSettings

        settings_name = f"dh5browser_{test_dh5_file.stem}_test"
        win = None
        try:
            # Create browser with test settings name
            win, filename, cache = browser_module.create_browser(
                test_dh5_file, trial_index=0
            )
            # Override settings name for testing
            win.settings_name = settings_name
            win.settings = QSettings("ephyviewer", settings_name)

            # Manually trigger save
            win.save_all_settings()

            # Verify settings were saved
            assert win.settings.value("window_geometry") is not None
            assert win.settings.value("window_state") is not None

        finally:
            # Clean up
            if win is not None:
                win.close()
                qapp.processEvents()
            # Clean up test settings
            settings = QSettings("ephyviewer", settings_name)
            settings.clear()


@pytest.mark.skip(reason="Browser functionality under development")
def test_browser_with_neo_objects():
    """Test that browser works with Neo objects (if neo is available)."""
    # This test just verifies the integration point exists
    # Actual GUI testing would require Qt and is not done here
    assert hasattr(browser_module, "create_browser")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
