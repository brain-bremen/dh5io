"""
DH5 Browser - Interactive data viewer for DH5 files using ephyviewer.

This module provides an interactive browser for DH5 (DAQ-HDF5) files based on ephyviewer.
The browser displays signals, spikes, and events from a single trial.

Usage:
    dh5browser <filename.dh5> [--trial TRIAL_INDEX]

Examples:
    dh5browser mydata.dh5
    dh5browser mydata.dh5 --trial 0
"""

import argparse
import logging
import pathlib
import sys
import time
from typing import Dict, List, Optional, Tuple

try:
    import ephyviewer
except ImportError:
    print("Error: ephyviewer is required for dh5browser")
    print("Install with: pip install ephyviewer")
    sys.exit(1)

import neo
import numpy as np

from dh5neo import DH5IO

# Configure logging
logger = logging.getLogger(__name__)

try:
    import matplotlib

    # Set backend before importing pyplot to avoid premature Qt initialization
    matplotlib.use("Agg")  # Use non-interactive backend
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available, event colors will not be customized")

try:
    from PySide6 import QtCore, QtWidgets
    from PySide6.QtCore import Signal

    Qt = QtCore.Qt
except ImportError:
    print("Error: PySide6 is required for dh5browser")
    print("Install with: pip install PySide6")
    sys.exit(1)


class SegmentCache:
    """Cache for loaded Neo segments to avoid re-reading from disk."""

    def __init__(self, reader: DH5IO, max_cache_size: int = 10) -> None:
        """
        Parameters
        ----------
        reader : DH5IO
            The DH5 reader instance
        max_cache_size : int
            Maximum number of segments to keep in cache
        """
        self.reader = reader
        self.max_cache_size = max_cache_size
        self._cache: Dict[int, Tuple[neo.Segment, float]] = {}
        self._access_order: List[int] = []

    def get_segment(self, segment_index: int) -> neo.Segment:
        """
        Get a segment, loading from disk if not cached.

        Parameters
        ----------
        segment_index : int
            Index of segment to retrieve

        Returns
        -------
        neo.Segment
            The requested segment
        """
        if segment_index in self._cache:
            logger.debug(f"Cache hit for segment {segment_index}")
            # Update access order
            self._access_order.remove(segment_index)
            self._access_order.append(segment_index)
            return self._cache[segment_index][0]

        # Cache miss - load from disk
        logger.debug(f"Cache miss for segment {segment_index}, loading from disk")
        step_start = time.perf_counter()
        segment = self.reader.read_segment(
            block_index=0, seg_index=segment_index, lazy=False
        )
        logger.debug(
            f"  read_segment({segment_index}): {time.perf_counter() - step_start:.3f}s"
        )

        # Add to cache
        self._cache[segment_index] = (segment, time.time())
        self._access_order.append(segment_index)

        # Evict oldest if cache is full
        if len(self._cache) > self.max_cache_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
            logger.debug(f"Evicted segment {oldest} from cache (LRU)")

        return segment

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
        logger.debug("Cache cleared")


class TrialNavigationWidget(QtWidgets.QWidget):  # type: ignore[misc]
    """Widget for navigating between trials with Previous/Next buttons."""

    trial_changed = Signal(int)  # Signal emitted when trial changes
    time_changed = Signal(float, float)  # Required by ephyviewer (t_start, t_stop)

    def __init__(
        self,
        nb_trials: int,
        initial_trial: int = 0,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """
        Parameters
        ----------
        nb_trials : int
            Total number of trials
        initial_trial : int
            Initial trial index
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        self.name = "Trial Navigation"  # Required by ephyviewer
        self.source = (
            None  # Required by ephyviewer (no data source for navigation widget)
        )
        self.nb_trials = nb_trials
        self.current_trial = initial_trial

        # Create layout
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Previous button
        self.prev_btn = QtWidgets.QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(self._on_previous)
        layout.addWidget(self.prev_btn)

        # Segment label
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self.label.setMinimumWidth(150)
        self._update_label()
        layout.addWidget(self.label)

        # Next button
        self.next_btn = QtWidgets.QPushButton("Next ▶")
        self.next_btn.clicked.connect(self._on_next)
        layout.addWidget(self.next_btn)

        # Stretch to keep buttons compact
        layout.addStretch()

        # Update button states
        self._update_buttons()

    def _update_label(self) -> None:
        """Update the trial label."""
        self.label.setText(f"Trial {self.current_trial + 1} of {self.nb_trials}")

    def _update_buttons(self) -> None:
        """Enable/disable buttons based on current trial."""
        self.prev_btn.setEnabled(self.current_trial > 0)
        self.next_btn.setEnabled(self.current_trial < self.nb_trials - 1)

    def _on_previous(self) -> None:
        """Handle Previous button click."""
        if self.current_trial > 0:
            self.current_trial -= 1
            self._update_label()
            self._update_buttons()
            self.trial_changed.emit(self.current_trial)

    def _on_next(self) -> None:
        """Handle Next button click."""
        if self.current_trial < self.nb_trials - 1:
            self.current_trial += 1
            self._update_label()
            self._update_buttons()
            self.trial_changed.emit(self.current_trial)

    def set_trial(self, trial_index: int) -> None:
        """Programmatically set the current trial."""
        if 0 <= trial_index < self.nb_trials:
            self.current_trial = trial_index
            self._update_label()
            self._update_buttons()

    def get_settings(self) -> Dict[str, int]:
        """Get widget settings for ephyviewer persistence.

        Note: We intentionally don't persist the trial selection.
        Users expect to start at trial 0 (or the --trial argument) each time,
        not to resume where they left off in a previous session.
        """
        return {}

    def set_settings(self, settings: Dict[str, int]) -> None:
        """Set widget settings for ephyviewer persistence.

        Note: We don't restore trial selection from saved settings.
        """
        pass  # Intentionally don't restore trial - use initial_trial instead

    def seek(self, t: float) -> None:
        """Dummy seek method required by ephyviewer (navigation widget doesn't need to seek)."""
        pass  # Navigation widget doesn't display time-based data


class CombinedAnalogSignalSource:
    """
    Custom ephyviewer source that combines multiple analog signals into one multi-channel source.
    Preserves the original channel names from Neo array_annotations.
    Supports updating signals dynamically for efficient trial switching.
    """

    def __init__(self, analog_signals: List[neo.AnalogSignal]) -> None:
        """
        Parameters
        ----------
        analog_signals : list of neo.AnalogSignal
            List of analog signals to combine
        """
        init_start: float = time.perf_counter()

        self.signals: List[neo.AnalogSignal] = analog_signals
        self.type: str = "analogsignal"
        self.with_scatter: bool = False

        self._process_signals(analog_signals)

        logger.debug(
            f"CombinedAnalogSignalSource.__init__ total: {time.perf_counter() - init_start:.3f}s"
        )

    def _process_signals(self, analog_signals: List[neo.AnalogSignal]) -> None:
        """Process analog signals and update internal data structures."""
        # All signals should have the same sampling rate and t_start
        param_start = time.perf_counter()
        self.sample_rate: float = float(analog_signals[0].sampling_rate.magnitude)  # type: ignore[union-attr]
        self.t_start: float = float(analog_signals[0].t_start.magnitude)
        self.t_stop: float = float(analog_signals[0].t_stop.magnitude)
        logger.debug(f"  Extract parameters: {time.perf_counter() - param_start:.3f}s")

        # Collect channel names from array_annotations
        names_start = time.perf_counter()
        self.channel_names: List[str] = []
        for sig in analog_signals:
            if (
                hasattr(sig, "array_annotations")
                and "channel_names" in sig.array_annotations
            ):
                # Extract channel names from array_annotations
                for ch_name in sig.array_annotations["channel_names"]:
                    self.channel_names.append(str(ch_name))
            else:
                # Fallback to signal name
                for i in range(sig.shape[1]):
                    self.channel_names.append(f"{sig.name}/{i}")

        self.nb_channel = len(self.channel_names)
        logger.debug(
            f"  Collect channel names ({self.nb_channel} channels): {time.perf_counter() - names_start:.3f}s"
        )

        # Concatenate all signals along channel axis
        concat_start = time.perf_counter()
        signal_arrays = [sig.magnitude for sig in analog_signals]
        logger.debug(
            f"    Extract magnitude arrays: {time.perf_counter() - concat_start:.3f}s"
        )

        concat2_start = time.perf_counter()
        self._data: np.ndarray = np.concatenate(signal_arrays, axis=1)
        self.nb_channel: int = len(self.channel_names)
        logger.debug(f"    np.concatenate: {time.perf_counter() - concat2_start:.3f}s")
        logger.debug(
            f"  Total concatenation: {time.perf_counter() - concat_start:.3f}s"
        )

    def update_signals(self, analog_signals: List[neo.AnalogSignal]) -> None:
        """
        Update the source with new analog signals.
        This allows reusing the same source object with different data.

        Parameters
        ----------
        analog_signals : list of neo.AnalogSignal
            New list of analog signals to display
        """
        self.signals = analog_signals
        self._process_signals(analog_signals)

    def get_shape(self) -> Tuple[int, int]:
        shape = self._data.shape
        return (shape[0], shape[1])

    def get_length(self) -> int:
        return self._data.shape[0]

    def get_chunk(
        self, i_start: Optional[int] = None, i_stop: Optional[int] = None
    ) -> np.ndarray:
        """Get a chunk of data."""
        if i_start is None:
            i_start = 0
        if i_stop is None:
            i_stop = self._data.shape[0]
        return self._data[i_start:i_stop, :]

    def get_channel_name(
        self, chan: Optional[int] = None, channel_index: Optional[int] = None
    ) -> str:
        """Get the name of a specific channel."""
        # Accept both 'chan' (used by ephyviewer) and 'channel_index'
        idx = chan if chan is not None else channel_index
        if idx is None:
            raise ValueError("Either chan or channel_index must be provided")
        return self.channel_names[idx]

    def index_to_time(self, index: int) -> float:
        """Convert sample index to time."""
        return self.t_start + index / self.sample_rate

    def time_to_index(self, time: float) -> int:
        """Convert time to sample index."""
        return int((time - self.t_start) * self.sample_rate)


def create_browser(
    dh5_file: pathlib.Path, trial_index: int = 0, cache_size: int = 10
) -> ephyviewer.MainViewer:
    """
    Create an ephyviewer MainViewer window for a DH5 file with trial navigation.

    Parameters
    ----------
    dh5_file : pathlib.Path
        Path to the DH5 file to open
    trial_index : int, optional
        Index of the trial to display (default: 0)
    cache_size : int, optional
        Maximum number of trials to cache (default: 10)

    Returns
    -------
    ephyviewer.MainViewer
        The configured main viewer window
    """
    overall_start = time.perf_counter()

    # Load DH5 file using Neo
    logger.info(f"Loading DH5 file: {dh5_file}")
    step_start = time.perf_counter()
    reader = DH5IO(dh5_file)
    logger.debug(f"  DH5IO initialization: {time.perf_counter() - step_start:.3f}s")

    # Get segment count without reading all data
    reader.parse_header()
    nb_segments = reader.segment_count(0)

    if nb_segments == 0:
        print("Error: No trials found in DH5 file")
        sys.exit(1)

    if trial_index >= nb_segments:
        print(f"Error: Trial index {trial_index} out of range")
        print(f"File contains {nb_segments} trial(s)")
        sys.exit(1)

    # Create segment cache
    cache = SegmentCache(reader, max_cache_size=cache_size)

    # Load initial segment
    segment = cache.get_segment(trial_index)

    logger.info(f"Loading trial {trial_index} of {nb_segments}")
    logger.debug(f"  Analog signals: {len(segment.analogsignals)}")
    logger.debug(f"  Spike trains: {len(segment.spiketrains)}")
    logger.debug(f"  Events: {len(segment.events)}")
    logger.debug(f"  Epochs: {len(segment.epochs)}")

    # Report data sizes
    if segment.analogsignals:
        total_samples = sum(sig.shape[0] for sig in segment.analogsignals)
        total_channels = sum(sig.shape[1] for sig in segment.analogsignals)
        total_mb = sum(sig.magnitude.nbytes for sig in segment.analogsignals) / (
            1024**2
        )
        logger.info(
            f"  Total analog data: {total_samples:,} samples × {total_channels} channels = {total_mb:.1f} MB"
        )

    # Create the main viewer window
    logger.debug("Creating MainViewer window")
    step_start = time.perf_counter()
    win = ephyviewer.MainViewer(
        debug=False,
        show_auto_scale=True,
        show_global_xsize=True,
        settings_name=f"dh5browser_{dh5_file.stem}",
    )
    logger.debug(f"  MainViewer created: {time.perf_counter() - step_start:.3f}s")

    # Store references to reusable viewer widgets
    trace_viewer_widget = None
    spike_viewer_widgets = []
    epoch_viewer_widgets = []

    # Function to load and display a trial
    def load_trial(trial_idx: int) -> None:
        """Load and display a specific trial by updating data sources in existing widgets."""
        nonlocal trace_viewer_widget, spike_viewer_widgets, epoch_viewer_widgets

        load_start = time.perf_counter()
        logger.info(f"Loading trial {trial_idx}...")

        # Get segment from cache
        seg = cache.get_segment(trial_idx)

        # Get sources from the Neo segment
        logger.debug("Creating ephyviewer sources from Neo segment")
        step_start = time.perf_counter()
        sources = ephyviewer.get_sources_from_neo_segment(seg)
        logger.debug(
            f"  get_sources_from_neo_segment(): {time.perf_counter() - step_start:.3f}s"
        )

        logger.debug(f"  Created {len(sources)} data source groups:")
        for name, source_list in sources.items():
            if isinstance(source_list, list):
                logger.debug(f"    - {name} ({len(source_list)} sources)")
            else:
                logger.debug(f"    - {name} ({type(source_list).__name__})")

        # Add viewers for different data types
        view_count = 0

        # Extract individual sources from the lists
        spike_sources = sources.get("spike", [])
        epoch_sources = sources.get("epoch", [])
        # Event epochs are now created in the Neo reader as proper epochs

        # Combine all analog signals into a single multi-channel source
        initial_t_start = None
        initial_t_stop = None
        if seg.analogsignals:
            logger.info(
                f"Combining {len(seg.analogsignals)} analog signals into single viewer"
            )

            # Reuse or create TraceViewer
            if trace_viewer_widget is None:
                # Create new viewer on first load
                step_start = time.perf_counter()
                combined_source = CombinedAnalogSignalSource(seg.analogsignals)
                logger.debug(
                    f"  CombinedAnalogSignalSource created: {time.perf_counter() - step_start:.3f}s"
                )
                logger.debug(f"  Shape: {combined_source.get_shape()}")
                logger.debug(f"  Channels: {combined_source.nb_channel}")
                logger.debug(
                    f"  Memory: {combined_source._data.nbytes / (1024**2):.1f} MB"
                )

                step_start = time.perf_counter()
                trace_view = ephyviewer.TraceViewer(
                    source=combined_source, name="All Signals"
                )
                logger.debug(
                    f"  TraceViewer created: {time.perf_counter() - step_start:.3f}s"
                )

                trace_view.params["scale_mode"] = "same_for_all"
                trace_view.params["display_labels"] = True
                trace_view.params["xsize"] = 10.0  # Show 10 seconds initially

                step_start = time.perf_counter()
                win.add_view(trace_view)
                logger.debug(
                    f"  TraceViewer added to window: {time.perf_counter() - step_start:.3f}s"
                )
                trace_viewer_widget = trace_view
                view_count += 1
            else:
                # Update existing viewer's source with new data
                logger.debug("Reusing existing TraceViewer, updating source data")
                trace_viewer_widget.source.update_signals(seg.analogsignals)
                trace_viewer_widget.refresh()
                trace_viewer_widget.initialize_plot()

            # Store the time range from the actual source being used by the viewer
            initial_t_start = trace_viewer_widget.source.t_start
            initial_t_stop = trace_viewer_widget.source.t_stop

        # Reuse or create spike train viewers
        logger.debug("Updating spike/event/epoch viewers")
        step_start = time.perf_counter()
        if spike_sources:
            for i, source in enumerate(spike_sources):
                if i < len(spike_viewer_widgets):
                    # Reuse existing viewer - update source data in-place
                    spike_viewer_widgets[i].source.all = source.all
                    spike_viewer_widgets[i].refresh()
                    spike_viewer_widgets[i].initialize_plot()
                else:
                    # Create new viewer
                    spike_view = ephyviewer.SpikeTrainViewer(
                        source=source, name=f"Spike Trains {i + 1}"
                    )
                    spike_view.params["xsize"] = 10.0  # Match trace viewer
                    win.add_view(spike_view)
                    spike_viewer_widgets.append(spike_view)
                    view_count += 1

        # Reuse or create epoch viewers (including event epochs from Neo reader)
        if epoch_sources:
            for i, source in enumerate(epoch_sources):
                if i < len(epoch_viewer_widgets):
                    # Reuse existing viewer - update source data in-place
                    epoch_viewer_widgets[i].source.all = source.all
                    epoch_viewer_widgets[i].refresh()
                    epoch_viewer_widgets[i].initialize_plot()
                else:
                    # Create new viewer
                    epoch_view = ephyviewer.EpochViewer(
                        source=source, name=f"Epochs {i + 1}"
                    )
                    epoch_view.params["xsize"] = 10.0  # Match trace viewer
                    win.add_view(epoch_view)
                    epoch_viewer_widgets.append(epoch_view)
                    view_count += 1

                    # Apply progressive colors to distinguish event types
                    import matplotlib.cm
                    import matplotlib.colors
                    import numpy

                    n_channels = source.nb_channel
                    cmap = matplotlib.cm.get_cmap("Dark2", n_channels)

                    epoch_view.by_channel_params.blockSignals(True)
                    for c in range(n_channels):
                        color = [
                            int(e * 255)
                            for e in matplotlib.colors.ColorConverter().to_rgb(cmap(c))
                        ]
                        epoch_view.by_channel_params[f"ch{c}", "color"] = color
                    epoch_view.by_channel_params.blockSignals(False)
                    epoch_view.refresh()

        if view_count > 0:
            logger.debug(
                f"  Added {view_count} new viewers: {time.perf_counter() - step_start:.3f}s"
            )

        if (
            trace_viewer_widget is None
            and len(spike_viewer_widgets) == 0
            and len(epoch_viewer_widgets) == 0
        ):
            logger.warning("No viewable data found in segment")

        # Auto-scale the trace viewers (only on first load)
        if view_count > 0:
            logger.debug("Auto-scaling viewers")
            step_start = time.perf_counter()
            win.auto_scale()
            logger.debug(f"  auto_scale(): {time.perf_counter() - step_start:.3f}s")

        # Update navigation toolbar's global time range and seek to the new trial's data start time
        if initial_t_start is not None:
            # Update the navigation toolbar's global time limits
            # This is crucial - otherwise the toolbar keeps the old trial's time range
            win.navigation_toolbar.set_start_stop(
                initial_t_start, initial_t_stop, seek=False
            )

            logger.info(f"Seeking all viewers to t={initial_t_start}")
            step_start = time.perf_counter()
            # Explicitly seek each viewer to the new trial's start time
            if trace_viewer_widget is not None:
                trace_viewer_widget.seek(initial_t_start)
            for spike_viewer in spike_viewer_widgets:
                spike_viewer.seek(initial_t_start)
            for epoch_viewer in epoch_viewer_widgets:
                epoch_viewer.seek(initial_t_start)
            # Also call MainViewer's seek to synchronize
            win.seek(initial_t_start)
            logger.info(
                f"  seek({initial_t_start}): {time.perf_counter() - step_start:.3f}s"
            )

        load_elapsed = time.perf_counter() - load_start
        logger.info(f"Trial {trial_idx} loaded in {load_elapsed:.3f}s")

    # Load initial trial
    load_trial(trial_index)

    # Add navigation widget if there are multiple trials
    if nb_segments > 1:
        logger.debug(f"Adding navigation widget ({nb_segments} trials)")
        nav_widget = TrialNavigationWidget(nb_segments, initial_trial=trial_index)

        # Connect trial change signal
        def on_trial_changed(new_trial_index: int) -> None:
            logger.info(f"Switching to trial {new_trial_index}")
            load_trial(new_trial_index)

        nav_widget.trial_changed.connect(on_trial_changed)

        # Add to main viewer
        win.add_view(nav_widget, location="top", orientation="horizontal")
        logger.debug("Navigation widget added")

    overall_elapsed = time.perf_counter() - overall_start
    logger.info(f"Browser ready in {overall_elapsed:.3f}s")

    return win


def main() -> None:
    """Main entry point for dh5browser command."""
    parser = argparse.ArgumentParser(
        description="Interactive browser for DH5 (DAQ-HDF5) files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dh5browser mydata.dh5
  dh5browser mydata.dh5 --trial 0
  dh5browser mydata.dh5 -t 2

The browser displays signals, spikes, events, and epochs from a single trial
using an interactive viewer based on ephyviewer.
        """,
    )

    parser.add_argument("filename", type=str, help="Path to DH5 file to open")
    parser.add_argument(
        "-t",
        "--trial",
        type=int,
        default=0,
        help="Trial index to display (default: 0)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=10,
        help="Maximum number of trials to cache in memory (default: 10)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging with detailed timing information",
    )

    args = parser.parse_args()

    # Configure logging based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
    )

    # Validate file path
    dh5_file = pathlib.Path(args.filename)
    if not dh5_file.exists():
        print(f"Error: File not found: {dh5_file}")
        sys.exit(1)

    if not dh5_file.suffix == ".dh5":
        print(f"Warning: File does not have .dh5 extension: {dh5_file}")

    # Create Qt application
    app = ephyviewer.mkQApp()

    # Create and show browser window
    try:
        win = create_browser(
            dh5_file, trial_index=args.trial, cache_size=args.cache_size
        )

        logger.info("Showing browser window...")
        show_start = time.perf_counter()
        win.show()
        logger.debug(f"win.show() took: {time.perf_counter() - show_start:.3f}s")

        print("\n" + "=" * 60)
        print("DH5 Browser")
        print("=" * 60)
        print(f"File: {dh5_file}")
        print(f"Trial: {args.trial}")
        print("=" * 60)
        print("\nBrowser window opened. Close window to exit.")

        # Run the Qt event loop
        app.exec()

    except Exception as e:
        print(f"Error creating browser: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
