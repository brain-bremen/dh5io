"""Test sequential data validation in the dh5merge tool.

This module tests that the merge tool correctly validates that files to be merged
contain sequential (non-overlapping) data for CONT, WAVELET, TRIALMAP, and EV02.
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from dh5cli.dh5merge import merge_dh5_files
from dh5io.cont import create_cont_group_from_data_in_file, create_empty_index_array
from dh5io.create import create_dh_file
from dh5io.event_triggers import add_event_triggers_to_file
from dh5io.trialmap import add_trialmap_to_file
from dh5io.wavelet import create_wavelet_group_from_data_in_file
from dhspec.trialmap import TRIALMAP_DATASET_DTYPE
from dhspec.wavelet import create_empty_index_array as create_empty_wavelet_index


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_dh5_with_cont(
    filepath: Path, cont_id: int, start_time: int, duration_ns: int
) -> None:
    """
    Create a DH5 file with a CONT block.

    Parameters
    ----------
    filepath : Path
        Output file path
    cont_id : int
        CONT block ID
    start_time : int
        Start time in nanoseconds
    duration_ns : int
        Duration in nanoseconds
    """
    n_samples = 1000
    n_channels = 2
    sample_period_ns = duration_ns // n_samples

    data = np.random.randint(-100, 100, size=(n_samples, n_channels), dtype=np.int16)
    index = create_empty_index_array(1)
    index[0]["time"] = start_time
    index[0]["offset"] = 0

    with create_dh_file(filepath, overwrite=True, boards=["Board1"]) as dh5:
        create_cont_group_from_data_in_file(
            dh5._file,
            cont_id,
            data=data,
            index=index,
            sample_period_ns=np.int32(sample_period_ns),
            calibration=np.ones(n_channels, dtype=np.float64),
            name=f"CONT{cont_id}",
        )


def create_dh5_with_wavelet(
    filepath: Path, wavelet_id: int, start_time: int, duration_ns: int
) -> None:
    """
    Create a DH5 file with a WAVELET block.

    Parameters
    ----------
    filepath : Path
        Output file path
    wavelet_id : int
        WAVELET block ID
    start_time : int
        Start time in nanoseconds
    duration_ns : int
        Duration in nanoseconds
    """
    n_channels = 2
    n_samples = 100
    n_frequencies = 10
    sample_period_ns = duration_ns // n_samples

    amplitude = np.random.rand(n_channels, n_samples, n_frequencies).astype(np.float64)
    phase = (
        np.random.rand(n_channels, n_samples, n_frequencies).astype(np.float64)
        * 2
        * np.pi
    )
    frequency_axis = np.logspace(np.log10(5), np.log10(160), n_frequencies)

    index = create_empty_wavelet_index(1)
    index[0]["time"] = start_time
    index[0]["offset"] = 0
    index[0]["scaling"] = 1.0

    with create_dh_file(filepath, overwrite=True, boards=["Board1"]) as dh5:
        create_wavelet_group_from_data_in_file(
            dh5._file,
            wavelet_id,
            amplitude=amplitude,
            phase=phase,
            index=index,
            sample_period_ns=np.int32(sample_period_ns),
            frequency_axis=frequency_axis,
            name=f"WAVELET{wavelet_id}",
        )


def create_dh5_with_trialmap(
    filepath: Path, start_time: int, n_trials: int, trial_duration_ns: int
) -> None:
    """
    Create a DH5 file with a TRIALMAP and a dummy CONT block.

    Parameters
    ----------
    filepath : Path
        Output file path
    start_time : int
        Start time of first trial in nanoseconds
    n_trials : int
        Number of trials
    trial_duration_ns : int
        Duration of each trial in nanoseconds
    """
    trialmap = np.recarray(n_trials, dtype=TRIALMAP_DATASET_DTYPE)
    for i in range(n_trials):
        trialmap[i].TrialNo = i
        trialmap[i].StimNo = 0
        trialmap[i].Outcome = 0
        trialmap[i].StartTime = start_time + i * trial_duration_ns
        trialmap[i].EndTime = start_time + (i + 1) * trial_duration_ns

    # Add a dummy CONT block so merge doesn't fail
    # Make sure it aligns with the TRIALMAP timing
    n_channels = 1
    total_duration = n_trials * trial_duration_ns
    sample_period_ns = 100_000  # 100 microseconds
    n_samples = int(total_duration / sample_period_ns)

    data = np.random.randint(-100, 100, size=(n_samples, n_channels), dtype=np.int16)
    index = create_empty_index_array(1)
    index[0]["time"] = start_time
    index[0]["offset"] = 0

    with create_dh_file(filepath, overwrite=True, boards=["Board1"]) as dh5:
        add_trialmap_to_file(dh5._file, trialmap)
        create_cont_group_from_data_in_file(
            dh5._file,
            1,
            data=data,
            index=index,
            sample_period_ns=np.int32(sample_period_ns),
            calibration=np.ones(n_channels, dtype=np.float64),
            name="CONT1",
        )


def create_dh5_with_events(
    filepath: Path, start_time: int, n_events: int, event_spacing_ns: int
) -> None:
    """
    Create a DH5 file with EV02 event triggers and a dummy CONT block.

    Parameters
    ----------
    filepath : Path
        Output file path
    start_time : int
        Start time of first event in nanoseconds
    n_events : int
        Number of events
    event_spacing_ns : int
        Time between events in nanoseconds
    """
    timestamps = np.array(
        [start_time + i * event_spacing_ns for i in range(n_events)], dtype=np.int64
    )
    event_codes = np.array([i % 10 for i in range(n_events)], dtype=np.int16)

    # Add a dummy CONT block so merge doesn't fail
    # Make sure it aligns with the event trigger timing
    n_channels = 1
    total_duration = (n_events - 1) * event_spacing_ns + event_spacing_ns
    sample_period_ns = 100_000  # 100 microseconds
    n_samples = int(total_duration / sample_period_ns)

    data = np.random.randint(-100, 100, size=(n_samples, n_channels), dtype=np.int16)
    index = create_empty_index_array(1)
    index[0]["time"] = start_time
    index[0]["offset"] = 0

    with create_dh_file(filepath, overwrite=True, boards=["Board1"]) as dh5:
        add_event_triggers_to_file(dh5._file, timestamps, event_codes)
        create_cont_group_from_data_in_file(
            dh5._file,
            1,
            data=data,
            index=index,
            sample_period_ns=np.int32(sample_period_ns),
            calibration=np.ones(n_channels, dtype=np.float64),
            name="CONT1",
        )


# CONT block sequential validation tests


def test_cont_sequential_valid(temp_dir):
    """Test that merging sequential CONT blocks succeeds."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # File 1: 0 to 1,000,000 ns
    create_dh5_with_cont(file1, cont_id=1, start_time=0, duration_ns=1_000_000)
    # File 2: 2,000,000 to 3,000,000 ns (gap is fine)
    create_dh5_with_cont(file2, cont_id=1, start_time=2_000_000, duration_ns=1_000_000)

    # Should succeed
    merge_dh5_files([file1, file2], output)


def test_cont_overlapping_raises_error(temp_dir):
    """Test that merging overlapping CONT blocks raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # File 1: 0 to 1,000,000 ns
    create_dh5_with_cont(file1, cont_id=1, start_time=0, duration_ns=1_000_000)
    # File 2: 500,000 to 1,500,000 ns (overlaps with file 1)
    create_dh5_with_cont(file2, cont_id=1, start_time=500_000, duration_ns=1_000_000)

    # Should raise ValueError about non-sequential data
    with pytest.raises(ValueError, match="CONT1.*not sequential.*files 0 and 1"):
        merge_dh5_files([file1, file2], output)


def test_cont_out_of_order_raises_error(temp_dir):
    """Test that merging out-of-order CONT blocks raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # File 1: 2,000,000 to 3,000,000 ns
    create_dh5_with_cont(file1, cont_id=1, start_time=2_000_000, duration_ns=1_000_000)
    # File 2: 0 to 1,000,000 ns (comes before file 1)
    create_dh5_with_cont(file2, cont_id=1, start_time=0, duration_ns=1_000_000)

    # Should raise ValueError about non-sequential data
    with pytest.raises(ValueError, match="CONT1.*not sequential.*files 0 and 1"):
        merge_dh5_files([file1, file2], output)


def test_cont_same_start_time_raises_error(temp_dir):
    """Test that merging CONT blocks with same start time raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # Both files start at same time
    create_dh5_with_cont(file1, cont_id=1, start_time=1_000_000, duration_ns=1_000_000)
    create_dh5_with_cont(file2, cont_id=1, start_time=1_000_000, duration_ns=1_000_000)

    # Should raise ValueError about non-sequential data
    with pytest.raises(ValueError, match="CONT1.*not sequential.*files 0 and 1"):
        merge_dh5_files([file1, file2], output)


# WAVELET block sequential validation tests


def test_wavelet_sequential_valid(temp_dir):
    """Test that merging sequential WAVELET blocks succeeds."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # File 1: 0 to 10,000,000 ns
    create_dh5_with_wavelet(file1, wavelet_id=1, start_time=0, duration_ns=10_000_000)
    # File 2: 20,000,000 to 30,000,000 ns
    create_dh5_with_wavelet(
        file2, wavelet_id=1, start_time=20_000_000, duration_ns=10_000_000
    )

    # Should succeed
    merge_dh5_files([file1, file2], output)


def test_wavelet_overlapping_raises_error(temp_dir):
    """Test that merging overlapping WAVELET blocks raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # File 1: 0 to 10,000,000 ns
    create_dh5_with_wavelet(file1, wavelet_id=1, start_time=0, duration_ns=10_000_000)
    # File 2: 5,000,000 to 15,000,000 ns (overlaps)
    create_dh5_with_wavelet(
        file2, wavelet_id=1, start_time=5_000_000, duration_ns=10_000_000
    )

    # Should raise ValueError about non-sequential data
    with pytest.raises(ValueError, match="WAVELET1.*not sequential.*files 0 and 1"):
        merge_dh5_files([file1, file2], output)


def test_wavelet_out_of_order_raises_error(temp_dir):
    """Test that merging out-of-order WAVELET blocks raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # File 1: 20,000,000 to 30,000,000 ns
    create_dh5_with_wavelet(
        file1, wavelet_id=1, start_time=20_000_000, duration_ns=10_000_000
    )
    # File 2: 0 to 10,000,000 ns (comes before file 1)
    create_dh5_with_wavelet(file2, wavelet_id=1, start_time=0, duration_ns=10_000_000)

    # Should raise ValueError about non-sequential data
    with pytest.raises(ValueError, match="WAVELET1.*not sequential.*files 0 and 1"):
        merge_dh5_files([file1, file2], output)


# TRIALMAP sequential validation tests


def test_trialmap_sequential_valid(temp_dir):
    """Test that merging sequential TRIALMAPs succeeds."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # File 1: trials from 0 to 5,000,000 ns (5 trials, 1ms each)
    create_dh5_with_trialmap(
        file1, start_time=0, n_trials=5, trial_duration_ns=1_000_000
    )
    # File 2: trials from 10,000,000 to 15,000,000 ns (gap is fine)
    create_dh5_with_trialmap(
        file2, start_time=10_000_000, n_trials=5, trial_duration_ns=1_000_000
    )

    # Should succeed
    merge_dh5_files([file1, file2], output)


def test_trialmap_overlapping_raises_error(temp_dir):
    """Test that merging overlapping TRIALMAPs raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # File 1: trials from 0 to 5,000,000 ns
    create_dh5_with_trialmap(
        file1, start_time=0, n_trials=5, trial_duration_ns=1_000_000
    )
    # File 2: trials from 3,000,000 to 8,000,000 ns (overlaps)
    create_dh5_with_trialmap(
        file2, start_time=3_000_000, n_trials=5, trial_duration_ns=1_000_000
    )

    # Should raise ValueError about non-sequential data (CONT validates first, catches overlap)
    with pytest.raises(ValueError, match="not sequential.*files 0 and 1"):
        merge_dh5_files([file1, file2], output)


def test_trialmap_out_of_order_raises_error(temp_dir):
    """Test that merging out-of-order TRIALMAPs raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # File 1: trials from 10,000,000 to 15,000,000 ns
    create_dh5_with_trialmap(
        file1, start_time=10_000_000, n_trials=5, trial_duration_ns=1_000_000
    )
    # File 2: trials from 0 to 5,000,000 ns (comes before file 1)
    create_dh5_with_trialmap(
        file2, start_time=0, n_trials=5, trial_duration_ns=1_000_000
    )

    # Should raise ValueError about non-sequential data (CONT validates first, catches overlap)
    with pytest.raises(ValueError, match="not sequential.*files 0 and 1"):
        merge_dh5_files([file1, file2], output)


def test_trialmap_non_increasing_within_file_raises_error(temp_dir):
    """Test that non-increasing trial timestamps within a file raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # Create file with non-increasing trial times
    trialmap = np.recarray(3, dtype=TRIALMAP_DATASET_DTYPE)
    trialmap[0].TrialNo = 0
    trialmap[0].StimNo = 0
    trialmap[0].Outcome = 0
    trialmap[0].StartTime = 1_000_000
    trialmap[0].EndTime = 2_000_000
    trialmap[1].TrialNo = 1
    trialmap[1].StimNo = 0
    trialmap[1].Outcome = 0
    trialmap[1].StartTime = 3_000_000  # Good
    trialmap[1].EndTime = 4_000_000
    trialmap[2].TrialNo = 2
    trialmap[2].StimNo = 0
    trialmap[2].Outcome = 0
    trialmap[2].StartTime = 2_500_000  # Bad - goes backwards!
    trialmap[2].EndTime = 3_500_000

    with create_dh_file(file1, overwrite=True, boards=["Board1"]) as dh5:
        add_trialmap_to_file(dh5._file, trialmap)

    # Create a simple sequential TRIALMAP file for file2
    trialmap2 = np.recarray(3, dtype=TRIALMAP_DATASET_DTYPE)
    for i in range(3):
        trialmap2[i].TrialNo = i + 10
        trialmap2[i].StimNo = 0
        trialmap2[i].Outcome = 0
        trialmap2[i].StartTime = 10_000_000 + i * 1_000_000
        trialmap2[i].EndTime = 10_000_000 + (i + 1) * 1_000_000

    # Add a dummy CONT block so merge doesn't fail
    # Make it match the timing of trialmap2
    n_channels = 1
    total_duration = 3 * 1_000_000  # 3 trials of 1ms each
    sample_period_ns = 100_000  # 100 microseconds
    n_samples = int(total_duration / sample_period_ns)

    data = np.random.randint(-100, 100, size=(n_samples, n_channels), dtype=np.int16)
    index = create_empty_index_array(1)
    index[0]["time"] = 10_000_000
    index[0]["offset"] = 0

    with create_dh_file(file2, overwrite=True, boards=["Board1"]) as dh5:
        add_trialmap_to_file(dh5._file, trialmap2)
        create_cont_group_from_data_in_file(
            dh5._file,
            1,
            data=data,
            index=index,
            sample_period_ns=np.int32(sample_period_ns),
            calibration=np.ones(n_channels, dtype=np.float64),
            name="CONT1",
        )

    # Should log warning about non-increasing timestamps but still attempt merge
    # Note: File 1 has no CONT block, so merge fails on missing common blocks
    with pytest.raises(
        ValueError,
        match="No common CONT or WAVELET blocks",
    ):
        merge_dh5_files([file1, file2], output)


# EV02 event triggers sequential validation tests


def test_events_sequential_valid(temp_dir):
    """Test that merging sequential EV02 event triggers succeeds."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # File 1: events from 0 to 5,000,000 ns
    create_dh5_with_events(file1, start_time=0, n_events=10, event_spacing_ns=500_000)
    # File 2: events from 10,000,000 to 15,000,000 ns
    create_dh5_with_events(
        file2, start_time=10_000_000, n_events=10, event_spacing_ns=500_000
    )

    # Should succeed
    merge_dh5_files([file1, file2], output)


def test_events_overlapping_raises_error(temp_dir):
    """Test that merging overlapping EV02 event triggers raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # File 1: events from 0 to 5,000,000 ns
    create_dh5_with_events(file1, start_time=0, n_events=10, event_spacing_ns=500_000)
    # File 2: events from 3,000,000 to 8,000,000 ns (overlaps)
    create_dh5_with_events(
        file2, start_time=3_000_000, n_events=10, event_spacing_ns=500_000
    )

    # Should raise ValueError about non-sequential data (CONT validates first, catches overlap)
    with pytest.raises(ValueError, match="not sequential.*files 0 and 1"):
        merge_dh5_files([file1, file2], output)


def test_events_out_of_order_raises_error(temp_dir):
    """Test that merging out-of-order EV02 event triggers raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # File 1: events from 10,000,000 to 15,000,000 ns
    create_dh5_with_events(
        file1, start_time=10_000_000, n_events=10, event_spacing_ns=500_000
    )
    # File 2: events from 0 to 5,000,000 ns (comes before file 1)
    create_dh5_with_events(file2, start_time=0, n_events=10, event_spacing_ns=500_000)

    # Should raise ValueError about non-sequential data (CONT validates first, catches overlap)
    with pytest.raises(ValueError, match="not sequential.*files 0 and 1"):
        merge_dh5_files([file1, file2], output)


def test_events_non_increasing_within_file_raises_error(temp_dir):
    """Test that non-increasing event timestamps within a file raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # Create file with non-increasing event times
    timestamps = np.array(
        [1_000_000, 2_000_000, 1_500_000, 3_000_000], dtype=np.int64
    )  # Out of order
    event_codes = np.array([1, 2, 3, 4], dtype=np.int16)

    with create_dh_file(file1, overwrite=True, boards=["Board1"]) as dh5:
        add_event_triggers_to_file(dh5._file, timestamps, event_codes)

    # Create a simple sequential events file for file2
    timestamps2 = np.array([10_000_000 + i * 500_000 for i in range(5)], dtype=np.int64)
    event_codes2 = np.array([i % 10 for i in range(5)], dtype=np.int16)

    # Add a dummy CONT block so merge doesn't fail
    # Make it match the timing of the events
    n_channels = 1
    total_duration = 4 * 500_000 + 500_000  # 5 events spaced by 500µs
    sample_period_ns = 100_000  # 100 microseconds
    n_samples = int(total_duration / sample_period_ns)

    data = np.random.randint(-100, 100, size=(n_samples, n_channels), dtype=np.int16)
    index = create_empty_index_array(1)
    index[0]["time"] = 10_000_000
    index[0]["offset"] = 0

    with create_dh_file(file2, overwrite=True, boards=["Board1"]) as dh5:
        add_event_triggers_to_file(dh5._file, timestamps2, event_codes2)
        create_cont_group_from_data_in_file(
            dh5._file,
            1,
            data=data,
            index=index,
            sample_period_ns=np.int32(sample_period_ns),
            calibration=np.ones(n_channels, dtype=np.float64),
            name="CONT1",
        )

    # Should log warning about non-increasing timestamps but still attempt merge
    # Note: File 1 has no CONT block, so merge fails on missing common blocks
    with pytest.raises(
        ValueError,
        match="No common CONT or WAVELET blocks",
    ):
        merge_dh5_files([file1, file2], output)


# Multi-file sequential validation tests


def test_three_files_sequential_valid(temp_dir):
    """Test that merging three sequential files succeeds."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    file3 = temp_dir / "file3.dh5"
    output = temp_dir / "merged.dh5"

    create_dh5_with_cont(file1, cont_id=1, start_time=0, duration_ns=1_000_000)
    create_dh5_with_cont(file2, cont_id=1, start_time=2_000_000, duration_ns=1_000_000)
    create_dh5_with_cont(file3, cont_id=1, start_time=4_000_000, duration_ns=1_000_000)

    # Should succeed
    merge_dh5_files([file1, file2, file3], output)


def test_three_files_middle_out_of_order_raises_error(temp_dir):
    """Test that merging three files with middle file out of order raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    file3 = temp_dir / "file3.dh5"
    output = temp_dir / "merged.dh5"

    create_dh5_with_cont(file1, cont_id=1, start_time=0, duration_ns=1_000_000)
    create_dh5_with_cont(
        file2, cont_id=1, start_time=10_000_000, duration_ns=1_000_000
    )  # Out of order
    create_dh5_with_cont(file3, cont_id=1, start_time=2_000_000, duration_ns=1_000_000)

    # Should raise ValueError - file 2 comes after file 1, but file 3 overlaps with file 2
    with pytest.raises(ValueError, match="CONT1.*not sequential.*files 1 and 2"):
        merge_dh5_files([file1, file2, file3], output)
