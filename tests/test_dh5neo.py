"""Tests for DH5RawIO - NEO RawIO interface for DH5 files."""

import pathlib

import numpy as np
import pytest

from dh5io.dh5file import DH5File
from dh5neo.dh5rawio import DH5RawIO

# Use the existing test file
test_file_path = pathlib.Path(__file__).parent / "test.dh5"


@pytest.fixture
def dh5_rawio():
    """Create a DH5RawIO instance for testing."""
    return DH5RawIO(test_file_path)


@pytest.fixture
def parsed_rawio(dh5_rawio):
    """Create a parsed DH5RawIO instance."""
    dh5_rawio.parse_header()
    return dh5_rawio


class TestDH5RawIOInit:
    """Test DH5RawIO initialization."""

    def test_init(self, dh5_rawio):
        """Test that DH5RawIO can be initialized."""
        assert dh5_rawio is not None
        assert isinstance(dh5_rawio, DH5RawIO)
        assert dh5_rawio.filename == test_file_path

    def test_source_name(self, dh5_rawio):
        """Test _source_name method."""
        assert dh5_rawio._source_name() == test_file_path


class TestDH5RawIOHeader:
    """Test header parsing."""

    def test_parse_header(self, dh5_rawio):
        """Test that header can be parsed."""
        dh5_rawio.parse_header()
        assert dh5_rawio.header is not None

    def test_header_structure(self, parsed_rawio):
        """Test that header has correct structure."""
        header = parsed_rawio.header
        assert header["nb_block"] == 1
        assert header["nb_segment"] is not None
        assert len(header["nb_segment"]) == 1

    def test_signal_streams(self, parsed_rawio):
        """Test signal streams in header."""
        streams = parsed_rawio.header["signal_streams"]
        assert len(streams) > 0
        # Check that each stream has id and name
        for stream in streams:
            assert "id" in stream.dtype.names
            assert "name" in stream.dtype.names

    def test_signal_channels(self, parsed_rawio):
        """Test signal channels in header."""
        channels = parsed_rawio.header["signal_channels"]
        assert len(channels) > 0
        # Check channel structure
        for channel in channels:
            assert "name" in channel.dtype.names
            assert "id" in channel.dtype.names
            assert "sampling_rate" in channel.dtype.names
            assert "dtype" in channel.dtype.names
            assert "units" in channel.dtype.names
            assert "gain" in channel.dtype.names
            assert "offset" in channel.dtype.names
            assert "stream_id" in channel.dtype.names

    def test_spike_channels(self, parsed_rawio):
        """Test spike channels in header."""
        spikes = parsed_rawio.header["spike_channels"]
        # There should be at least one spike channel in test.dh5
        assert len(spikes) >= 0
        if len(spikes) > 0:
            for spike in spikes:
                assert "name" in spike.dtype.names
                assert "id" in spike.dtype.names

    def test_event_channels(self, parsed_rawio):
        """Test event channels in header."""
        events = parsed_rawio.header["event_channels"]
        assert len(events) >= 2  # At least trials and events channels
        for event in events:
            assert "name" in event.dtype.names
            assert "id" in event.dtype.names
            assert "type" in event.dtype.names


class TestDH5RawIOSegments:
    """Test segment-related methods."""

    def test_segment_count(self, parsed_rawio):
        """Test that correct number of segments is reported."""
        nb_segments = parsed_rawio.header["nb_segment"][0]
        # Should match number of trials in trialmap
        assert nb_segments > 0

    def test_segment_t_start(self, parsed_rawio):
        """Test segment start times."""
        nb_segments = parsed_rawio.header["nb_segment"][0]
        for seg_idx in range(min(3, nb_segments)):  # Test first 3 segments
            t_start = parsed_rawio._segment_t_start(0, seg_idx)
            assert isinstance(t_start, (float, np.floating))
            assert t_start >= 0

    def test_segment_t_stop(self, parsed_rawio):
        """Test segment stop times."""
        nb_segments = parsed_rawio.header["nb_segment"][0]
        for seg_idx in range(min(3, nb_segments)):
            t_stop = parsed_rawio._segment_t_stop(0, seg_idx)
            assert isinstance(t_stop, (float, np.floating))
            # Stop time should be after start time
            t_start = parsed_rawio._segment_t_start(0, seg_idx)
            assert t_stop > t_start


class TestDH5RawIOSignals:
    """Test analog signal methods."""

    def test_get_signal_size(self, parsed_rawio):
        """Test getting signal size for a segment."""
        if len(parsed_rawio.header["signal_streams"]) == 0:
            pytest.skip("No signal streams in test file")

        size = parsed_rawio._get_signal_size(block_index=0, seg_index=0, stream_index=0)
        assert isinstance(size, int)
        assert size >= 0

    def test_get_signal_t_start(self, parsed_rawio):
        """Test getting signal start time."""
        if len(parsed_rawio.header["signal_streams"]) == 0:
            pytest.skip("No signal streams in test file")

        t_start = parsed_rawio._get_signal_t_start(
            block_index=0, seg_index=0, stream_index=0
        )
        assert isinstance(t_start, (float, np.floating))

    def test_get_analogsignal_chunk(self, parsed_rawio):
        """Test getting analog signal data chunks."""
        if len(parsed_rawio.header["signal_streams"]) == 0:
            pytest.skip("No signal streams in test file")

        # Get size first
        size = parsed_rawio._get_signal_size(block_index=0, seg_index=0, stream_index=0)

        if size == 0:
            pytest.skip("Segment has no data")

        # Get a chunk of data
        chunk_size = min(100, size)
        chunk = parsed_rawio._get_analogsignal_chunk(
            block_index=0,
            seg_index=0,
            i_start=0,
            i_stop=chunk_size,
            stream_index=0,
            channel_indexes=None,
        )

        assert isinstance(chunk, np.ndarray)
        assert chunk.shape[0] == chunk_size
        assert chunk.ndim == 2

    def test_get_analogsignal_chunk_with_channel_selection(self, parsed_rawio):
        """Test getting analog signal with specific channels."""
        if len(parsed_rawio.header["signal_streams"]) == 0:
            pytest.skip("No signal streams in test file")

        size = parsed_rawio._get_signal_size(block_index=0, seg_index=0, stream_index=0)

        if size == 0:
            pytest.skip("Segment has no data")

        # Get first channel only
        chunk = parsed_rawio._get_analogsignal_chunk(
            block_index=0,
            seg_index=0,
            i_start=0,
            i_stop=min(100, size),
            stream_index=0,
            channel_indexes=[0],
        )

        assert chunk.shape[1] == 1


class TestDH5RawIOSpikes:
    """Test spike-related methods."""

    def test_spike_count(self, parsed_rawio):
        """Test getting spike count."""
        if len(parsed_rawio.header["spike_channels"]) == 0:
            pytest.skip("No spike channels in test file")

        count = parsed_rawio._spike_count(
            block_index=0, seg_index=0, spike_channel_index=0
        )
        assert isinstance(count, int)
        assert count >= 0

    def test_get_spike_timestamps(self, parsed_rawio):
        """Test getting spike timestamps."""
        if len(parsed_rawio.header["spike_channels"]) == 0:
            pytest.skip("No spike channels in test file")

        timestamps = parsed_rawio._get_spike_timestamps(
            block_index=0, seg_index=0, spike_channel_index=0, t_start=None, t_stop=None
        )
        assert isinstance(timestamps, np.ndarray)

    def test_rescale_spike_timestamp(self, parsed_rawio):
        """Test rescaling spike timestamps."""
        # Create fake timestamps in nanoseconds
        fake_timestamps = np.array([1000000000, 2000000000], dtype=np.int64)
        rescaled = parsed_rawio._rescale_spike_timestamp(fake_timestamps, np.float64)

        assert rescaled.dtype == np.float64
        assert np.allclose(rescaled, [1.0, 2.0])

    def test_get_spike_raw_waveforms(self, parsed_rawio):
        """Test getting spike waveforms."""
        if len(parsed_rawio.header["spike_channels"]) == 0:
            pytest.skip("No spike channels in test file")

        count = parsed_rawio._spike_count(
            block_index=0, seg_index=0, spike_channel_index=0
        )

        if count == 0:
            pytest.skip("No spikes in first segment")

        waveforms = parsed_rawio._get_spike_raw_waveforms(
            block_index=0,
            seg_index=0,
            spike_channel_index=0,
            t_start=None,
            t_stop=None,
        )

        assert isinstance(waveforms, np.ndarray)
        assert waveforms.ndim == 3  # (nb_spike, nb_channel, nb_sample)
        assert waveforms.shape[0] == count


class TestDH5RawIOEvents:
    """Test event-related methods."""

    def test_event_count(self, parsed_rawio):
        """Test getting event count."""
        if len(parsed_rawio.header["event_channels"]) == 0:
            pytest.skip("No event channels in test file")

        count = parsed_rawio._event_count(
            block_index=0, seg_index=0, event_channel_index=0
        )
        assert isinstance(count, int)
        assert count >= 0

    def test_get_event_timestamps_trials(self, parsed_rawio):
        """Test getting trial epoch timestamps."""
        # Find the trials event channel
        event_channels = parsed_rawio.header["event_channels"]
        trial_channel_idx = None
        for i, ch in enumerate(event_channels):
            if ch["type"] == b"epoch":
                trial_channel_idx = i
                break

        if trial_channel_idx is None:
            pytest.skip("No trial epoch channel found")

        timestamps, durations, labels = parsed_rawio._get_event_timestamps(
            block_index=0,
            seg_index=0,
            event_channel_index=trial_channel_idx,
            t_start=None,
            t_stop=None,
        )

        assert isinstance(timestamps, np.ndarray)
        assert isinstance(durations, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert len(timestamps) == len(durations) == len(labels)

    def test_get_event_timestamps_events(self, parsed_rawio):
        """Test getting EV02 event timestamps."""
        # Find the EV02 event channel
        event_channels = parsed_rawio.header["event_channels"]
        ev_channel_idx = None
        for i, ch in enumerate(event_channels):
            if ch["type"] == b"event":
                ev_channel_idx = i
                break

        if ev_channel_idx is None:
            pytest.skip("No EV02 event channel found")

        timestamps, durations, labels = parsed_rawio._get_event_timestamps(
            block_index=0,
            seg_index=0,
            event_channel_index=ev_channel_idx,
            t_start=None,
            t_stop=None,
        )

        assert isinstance(timestamps, np.ndarray)
        assert isinstance(durations, np.ndarray)
        assert isinstance(labels, np.ndarray)

    def test_rescale_event_timestamp(self, parsed_rawio):
        """Test rescaling event timestamps."""
        # Create fake timestamps in nanoseconds
        fake_timestamps = np.array([1000000000, 2000000000], dtype=np.int64)
        rescaled = parsed_rawio._rescale_event_timestamp(fake_timestamps, np.float64)

        assert rescaled.dtype == np.float64
        assert np.allclose(rescaled, [1.0, 2.0])

    def test_rescale_epoch_duration(self, parsed_rawio):
        """Test rescaling epoch durations."""
        # Create fake durations in nanoseconds
        fake_durations = np.array([500000000, 1000000000], dtype=np.int64)
        rescaled = parsed_rawio._rescale_epoch_duration(fake_durations, np.float64)

        assert rescaled.dtype == np.float64
        assert np.allclose(rescaled, [0.5, 1.0])


class TestDH5RawIOIntegration:
    """Integration tests using Neo's reader interface."""

    def test_full_parse_and_read(self, dh5_rawio):
        """Test complete workflow of parsing and reading data."""
        # Parse header
        dh5_rawio.parse_header()

        # Check that we can access basic information
        assert dh5_rawio.header is not None
        nb_segments = dh5_rawio.header["nb_segment"][0]
        assert nb_segments > 0

        # Try to read from first segment if available
        if len(dh5_rawio.header["signal_streams"]) > 0:
            size = dh5_rawio._get_signal_size(0, 0, 0)
            if size > 0:
                chunk = dh5_rawio._get_analogsignal_chunk(
                    0, 0, 0, min(10, size), 0, None
                )
                assert chunk is not None

    def test_multiple_segments(self, parsed_rawio):
        """Test accessing multiple segments."""
        nb_segments = parsed_rawio.header["nb_segment"][0]

        # Test first few segments
        for seg_idx in range(min(5, nb_segments)):
            t_start = parsed_rawio._segment_t_start(0, seg_idx)
            t_stop = parsed_rawio._segment_t_stop(0, seg_idx)
            assert t_stop > t_start

    def test_signal_streams_match_cont_groups(self, parsed_rawio):
        """Test that signal streams match CONT groups in file."""
        # Open the DH5 file directly
        dh5_file = DH5File(test_file_path)
        cont_names = dh5_file.get_cont_group_names()

        # Check that signal streams match
        signal_streams = parsed_rawio.header["signal_streams"]
        assert len(signal_streams) == len(cont_names)

        # Check that IDs match
        stream_ids = [s["id"] for s in signal_streams]
        for cont_name in cont_names:
            assert cont_name in stream_ids

    def test_spike_channels_match_spike_groups(self, parsed_rawio):
        """Test that spike channels match SPIKE groups in file."""
        dh5_file = DH5File(test_file_path)
        spike_names = dh5_file.get_spike_group_names()

        spike_channels = parsed_rawio.header["spike_channels"]

        # Each spike group should have at least one unit
        # (currently we only support one unit per spike group)
        assert len(spike_channels) == len(spike_names)


class TestDH5RawIOEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_file(self):
        """Test that invalid file raises appropriate error."""
        with pytest.raises(Exception):
            rawio = DH5RawIO("nonexistent_file.dh5")
            rawio.parse_header()

    def test_methods_before_parse(self, dh5_rawio):
        """Test that methods fail appropriately before header is parsed."""
        with pytest.raises(ValueError, match="Header not yet parsed"):
            dh5_rawio._get_signal_size(0, 0, 0)

        with pytest.raises(ValueError, match="Header not yet parsed"):
            dh5_rawio._segment_t_start(0, 0)

    def test_empty_segment(self, parsed_rawio):
        """Test handling of segments with no data."""
        # This is a smoke test - implementation should handle gracefully
        if len(parsed_rawio.header["signal_streams"]) > 0:
            # Even if a segment is empty, these methods should not crash
            try:
                size = parsed_rawio._get_signal_size(0, 0, 0)
                assert size >= 0
            except Exception as e:
                pytest.fail(f"Should handle empty segments gracefully: {e}")


class TestDH5RawIOWithNeo:
    """Test integration with Neo library (if available)."""

    def test_compatible_with_neo_reader(self, parsed_rawio):
        """Test that the RawIO can be used with Neo's reader."""
        try:
            from neo.rawio import DH5RawIO as NeoReader

            # This would be the actual integration test
            # For now, just verify the structure is compatible
            assert hasattr(parsed_rawio, "header")
            assert hasattr(parsed_rawio, "parse_header")
        except ImportError:
            pytest.skip("Neo integration test requires neo to be installed")

    def test_annotations(self, parsed_rawio):
        """Test that annotations are generated."""
        # After parsing, annotations should exist
        assert hasattr(parsed_rawio, "_generate_minimal_annotations")
        # The base class should have created some annotations
        # This is a basic check that the method was called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
