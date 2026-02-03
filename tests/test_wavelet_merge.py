"""Test the wavelet merging functionality in dh5merge."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dh5cli.dh5merge import (
    concatenate_wavelet_data,
    determine_wavelet_blocks_to_merge,
    merge_dh5_files,
    validate_wavelet_blocks_compatible,
)
from dh5io.create import create_dh_file
from dh5io.dh5file import DH5File
from dh5io.wavelet import create_wavelet_group_from_data_in_file
from dhspec.wavelet import create_empty_index_array as create_empty_wavelet_index


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_test_wavelet_data(
    n_channels: int,
    n_samples: int,
    n_frequencies: int,
    start_time: int = 0,
    seed: int = 42,
):
    """
    Create test wavelet data (amplitude and phase).

    Parameters
    ----------
    n_channels : int
        Number of channels
    n_samples : int
        Number of time samples
    n_frequencies : int
        Number of frequency bins
    start_time : int
        Start time in nanoseconds
    seed : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (amplitude, phase, index, frequency_axis)
    """
    np.random.seed(seed)

    # Create random amplitude and phase data
    amplitude = np.random.rand(n_channels, n_samples, n_frequencies) * 100.0
    phase = (np.random.rand(n_channels, n_samples, n_frequencies) - 0.5) * 2 * np.pi

    # Create index with a single recording region
    index = create_empty_wavelet_index(1)
    index[0]["time"] = start_time
    index[0]["offset"] = 0
    index[0]["scaling"] = (
        1.0 / 65535.0 * np.max(amplitude) * 1.1
    )  # Scaling for amplitude

    # Create frequency axis
    frequency_axis = np.linspace(1.0, 100.0, n_frequencies)

    return amplitude, phase, index, frequency_axis


def create_test_dh5_file_with_wavelet(
    filepath: Path,
    wavelet_id: int,
    n_channels: int,
    n_samples: int,
    n_frequencies: int,
    start_time: int = 0,
    seed: int = 42,
) -> None:
    """
    Create a test DH5 file with a single WAVELET block.

    Parameters
    ----------
    filepath : Path
        Output file path
    wavelet_id : int
        WAVELET block ID
    n_channels : int
        Number of channels
    n_samples : int
        Number of time samples
    n_frequencies : int
        Number of frequency bins
    start_time : int
        Start time in nanoseconds
    seed : int
        Random seed
    """
    amplitude, phase, index, frequency_axis = create_test_wavelet_data(
        n_channels, n_samples, n_frequencies, start_time, seed
    )

    # Create file
    with create_dh_file(filepath, overwrite=True, boards=["TestBoard"]) as dh5:
        create_wavelet_group_from_data_in_file(
            dh5._file,
            wavelet_id,
            amplitude=amplitude,
            phase=phase,
            index=index,
            sample_period_ns=np.int32(10000000),  # 10 ms
            frequency_axis=frequency_axis,
            name=f"WAVELET{wavelet_id}",
            comment="Test wavelet",
        )


def test_merge_two_wavelet_files(temp_dir):
    """Test merging two DH5 files with the same WAVELET block."""
    # Create two test files
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    n_channels = 4
    n_frequencies = 35
    n_samples_1 = 100
    n_samples_2 = 150
    wavelet_id = 1

    create_test_dh5_file_with_wavelet(
        file1, wavelet_id, n_channels, n_samples_1, n_frequencies, start_time=0, seed=42
    )
    create_test_dh5_file_with_wavelet(
        file2,
        wavelet_id,
        n_channels,
        n_samples_2,
        n_frequencies,
        start_time=1000000000,
        seed=43,
    )

    # Merge files
    merge_dh5_files([file1, file2], output)

    # Verify merged file
    with DH5File(output, mode="r") as merged:
        wavelet = merged.get_wavelet_group_by_id(wavelet_id)

        assert wavelet is not None

        # Check dimensions
        assert wavelet.n_samples == n_samples_1 + n_samples_2
        assert wavelet.n_channels == n_channels
        assert wavelet.n_frequencies == n_frequencies

        # Check index has two regions
        assert wavelet.n_regions == 2
        assert wavelet.index[0]["time"] == 0
        assert wavelet.index[0]["offset"] == 0
        assert wavelet.index[1]["time"] == 1000000000
        assert wavelet.index[1]["offset"] == n_samples_1


def test_merge_multiple_wavelet_files(temp_dir):
    """Test merging three DH5 files with WAVELET blocks."""
    files = [temp_dir / f"file{i}.dh5" for i in range(3)]
    output = temp_dir / "merged.dh5"

    n_channels = 2
    n_frequencies = 20
    n_samples = [50, 75, 100]
    wavelet_id = 0
    start_times = [0, 500000000, 1500000000]

    for i, (file, n_samp, start_time) in enumerate(zip(files, n_samples, start_times)):
        create_test_dh5_file_with_wavelet(
            file,
            wavelet_id,
            n_channels,
            n_samp,
            n_frequencies,
            start_time,
            seed=42 + i,
        )

    # Merge files
    merge_dh5_files(files, output)

    # Verify merged file
    with DH5File(output, mode="r") as merged:
        wavelet = merged.get_wavelet_group_by_id(wavelet_id)

        assert wavelet is not None

        # Check total samples
        assert wavelet.n_samples == sum(n_samples)
        assert wavelet.n_channels == n_channels
        assert wavelet.n_frequencies == n_frequencies

        # Check index offsets
        assert wavelet.n_regions == 3
        expected_offsets = [0, n_samples[0], n_samples[0] + n_samples[1]]
        for i, expected_offset in enumerate(expected_offsets):
            assert wavelet.index[i]["offset"] == expected_offset
            assert wavelet.index[i]["time"] == start_times[i]


def test_merge_multiple_wavelet_blocks(temp_dir):
    """Test merging files with multiple WAVELET blocks."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    n_channels = 3
    n_frequencies = 25
    n_samples_1 = 80
    n_samples_2 = 120

    # Create files with multiple WAVELET blocks
    with create_dh_file(file1, overwrite=True) as dh5:
        for wavelet_id in [0, 1, 1001]:
            amplitude, phase, index, frequency_axis = create_test_wavelet_data(
                n_channels,
                n_samples_1,
                n_frequencies,
                start_time=0,
                seed=42 + wavelet_id,
            )
            create_wavelet_group_from_data_in_file(
                dh5._file,
                wavelet_id,
                amplitude=amplitude,
                phase=phase,
                index=index,
                sample_period_ns=np.int32(10000000),
                frequency_axis=frequency_axis,
            )

    with create_dh_file(file2, overwrite=True) as dh5:
        for wavelet_id in [0, 1, 1001]:
            amplitude, phase, index, frequency_axis = create_test_wavelet_data(
                n_channels,
                n_samples_2,
                n_frequencies,
                start_time=800000000,
                seed=100 + wavelet_id,
            )
            create_wavelet_group_from_data_in_file(
                dh5._file,
                wavelet_id,
                amplitude=amplitude,
                phase=phase,
                index=index,
                sample_period_ns=np.int32(10000000),
                frequency_axis=frequency_axis,
            )

    # Merge files
    merge_dh5_files([file1, file2], output)

    # Verify merged file has all three WAVELET blocks
    with DH5File(output, mode="r") as merged:
        wavelet_ids = merged.get_wavelet_group_ids()
        assert sorted(wavelet_ids) == [0, 1, 1001]

        # Check each merged wavelet
        for wavelet_id in wavelet_ids:
            wavelet = merged.get_wavelet_group_by_id(wavelet_id)
            assert wavelet.n_samples == n_samples_1 + n_samples_2
            assert wavelet.n_regions == 2


def test_incompatible_channel_count(temp_dir):
    """Test that merging files with different channel counts raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    wavelet_id = 0
    n_frequencies = 30

    create_test_dh5_file_with_wavelet(
        file1, wavelet_id, n_channels=2, n_samples=100, n_frequencies=n_frequencies
    )
    create_test_dh5_file_with_wavelet(
        file2, wavelet_id, n_channels=3, n_samples=100, n_frequencies=n_frequencies
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="Number of channels mismatch"):
        merge_dh5_files([file1, file2], output)


def test_incompatible_frequency_count(temp_dir):
    """Test that merging files with different frequency counts raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    wavelet_id = 0

    create_test_dh5_file_with_wavelet(
        file1, wavelet_id, n_channels=2, n_samples=100, n_frequencies=30
    )
    create_test_dh5_file_with_wavelet(
        file2, wavelet_id, n_channels=2, n_samples=100, n_frequencies=35
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="Number of frequencies mismatch"):
        merge_dh5_files([file1, file2], output)


def test_incompatible_sample_period(temp_dir):
    """Test that merging files with different sample periods raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    wavelet_id = 0
    n_channels = 2
    n_frequencies = 30

    # Create files with different sample periods
    amplitude1, phase1, index1, freq_axis1 = create_test_wavelet_data(
        n_channels, 100, n_frequencies
    )
    with create_dh_file(file1, overwrite=True) as dh5:
        create_wavelet_group_from_data_in_file(
            dh5._file,
            wavelet_id,
            amplitude=amplitude1,
            phase=phase1,
            index=index1,
            sample_period_ns=np.int32(10000000),  # 10 ms
            frequency_axis=freq_axis1,
        )

    amplitude2, phase2, index2, freq_axis2 = create_test_wavelet_data(
        n_channels, 100, n_frequencies
    )
    with create_dh_file(file2, overwrite=True) as dh5:
        create_wavelet_group_from_data_in_file(
            dh5._file,
            wavelet_id,
            amplitude=amplitude2,
            phase=phase2,
            index=index2,
            sample_period_ns=np.int32(20000000),  # 20 ms - different!
            frequency_axis=freq_axis2,
        )

    output = temp_dir / "merged.dh5"

    # Should raise ValueError
    with pytest.raises(ValueError, match="Sample period mismatch"):
        merge_dh5_files([file1, file2], output)


def test_incompatible_frequency_axis(temp_dir):
    """Test that merging files with different frequency axes raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    wavelet_id = 0
    n_channels = 2
    n_frequencies = 30

    # Create files with different frequency axes
    amplitude1, phase1, index1, _ = create_test_wavelet_data(
        n_channels, 100, n_frequencies
    )
    freq_axis1 = np.linspace(1.0, 100.0, n_frequencies)

    with create_dh_file(file1, overwrite=True) as dh5:
        create_wavelet_group_from_data_in_file(
            dh5._file,
            wavelet_id,
            amplitude=amplitude1,
            phase=phase1,
            index=index1,
            sample_period_ns=np.int32(10000000),
            frequency_axis=freq_axis1,
        )

    amplitude2, phase2, index2, _ = create_test_wavelet_data(
        n_channels, 100, n_frequencies
    )
    freq_axis2 = np.linspace(2.0, 120.0, n_frequencies)  # Different!

    with create_dh_file(file2, overwrite=True) as dh5:
        create_wavelet_group_from_data_in_file(
            dh5._file,
            wavelet_id,
            amplitude=amplitude2,
            phase=phase2,
            index=index2,
            sample_period_ns=np.int32(10000000),
            frequency_axis=freq_axis2,
        )

    output = temp_dir / "merged.dh5"

    # Should raise ValueError
    with pytest.raises(ValueError, match="Frequency axis mismatch"):
        merge_dh5_files([file1, file2], output)


def test_no_common_wavelet_blocks(temp_dir):
    """Test merging when files have no common WAVELET blocks."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # Create files with different WAVELET IDs and some CONT blocks to merge
    from dh5io.cont import create_cont_group_from_data_in_file, create_empty_index_array

    # Create file1 with CONT0 and WAVELET1
    with create_dh_file(file1, overwrite=True) as dh5:
        # Add CONT block so merge doesn't fail entirely
        data = np.random.randint(-100, 100, size=(50, 2), dtype=np.int16)
        index = create_empty_index_array(1)
        index[0] = (0, 0)
        create_cont_group_from_data_in_file(dh5._file, 0, data, index, np.int32(1000))

        # Add WAVELET1
        amplitude, phase, idx, frequency_axis = create_test_wavelet_data(
            2, 50, 20, start_time=0, seed=42
        )
        create_wavelet_group_from_data_in_file(
            dh5._file,
            1,
            amplitude=amplitude,
            phase=phase,
            index=idx,
            sample_period_ns=np.int32(10000000),
            frequency_axis=frequency_axis,
        )

    # Create file2 with CONT0 and WAVELET2
    with create_dh_file(file2, overwrite=True) as dh5:
        # Add same CONT block
        data = np.random.randint(-100, 100, size=(50, 2), dtype=np.int16)
        index = create_empty_index_array(1)
        index[0] = (50000, 0)
        create_cont_group_from_data_in_file(dh5._file, 0, data, index, np.int32(1000))

        # Add WAVELET2
        amplitude, phase, idx, frequency_axis = create_test_wavelet_data(
            2, 50, 20, start_time=50000, seed=43
        )
        create_wavelet_group_from_data_in_file(
            dh5._file,
            2,
            amplitude=amplitude,
            phase=phase,
            index=idx,
            sample_period_ns=np.int32(10000000),
            frequency_axis=frequency_axis,
        )

    # This should succeed but with no common WAVELET blocks
    merge_dh5_files([file1, file2], output)

    # Verify merged file has CONT but no WAVELET blocks
    with DH5File(output, mode="r") as merged:
        wavelet_ids = merged.get_wavelet_group_ids()
        assert len(wavelet_ids) == 0  # No common WAVELET blocks


def test_determine_wavelet_blocks_to_merge(temp_dir):
    """Test the determine_wavelet_blocks_to_merge function."""
    # Create test files with different WAVELET blocks
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    file3 = temp_dir / "file3.dh5"

    n_channels = 2
    n_frequencies = 20

    with create_dh_file(file1, overwrite=True) as dh5:
        for wavelet_id in [0, 1, 2]:
            amplitude, phase, index, frequency_axis = create_test_wavelet_data(
                n_channels, 50, n_frequencies, seed=wavelet_id
            )
            create_wavelet_group_from_data_in_file(
                dh5._file,
                wavelet_id,
                amplitude=amplitude,
                phase=phase,
                index=index,
                sample_period_ns=np.int32(10000000),
                frequency_axis=frequency_axis,
            )

    with create_dh_file(file2, overwrite=True) as dh5:
        for wavelet_id in [0, 2, 3]:
            amplitude, phase, index, frequency_axis = create_test_wavelet_data(
                n_channels, 50, n_frequencies, seed=wavelet_id + 10
            )
            create_wavelet_group_from_data_in_file(
                dh5._file,
                wavelet_id,
                amplitude=amplitude,
                phase=phase,
                index=index,
                sample_period_ns=np.int32(10000000),
                frequency_axis=frequency_axis,
            )

    with create_dh_file(file3, overwrite=True) as dh5:
        for wavelet_id in [0, 1, 3]:
            amplitude, phase, index, frequency_axis = create_test_wavelet_data(
                n_channels, 50, n_frequencies, seed=wavelet_id + 20
            )
            create_wavelet_group_from_data_in_file(
                dh5._file,
                wavelet_id,
                amplitude=amplitude,
                phase=phase,
                index=index,
                sample_period_ns=np.int32(10000000),
                frequency_axis=frequency_axis,
            )

    # Open files
    dh5_files = [DH5File(f, mode="r") for f in [file1, file2, file3]]

    try:
        # Test finding common blocks (should be only WAVELET0)
        common = determine_wavelet_blocks_to_merge(dh5_files)
        assert common == {0}

    finally:
        for dh5_file in dh5_files:
            dh5_file._file.close()


def test_merge_with_missing_wavelets(temp_dir):
    """Test merging when some files have wavelets and others don't."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    file3 = temp_dir / "file3.dh5"
    output = temp_dir / "merged.dh5"

    from dh5io.cont import create_cont_group_from_data_in_file, create_empty_index_array

    # File 1 with CONT and WAVELET
    with create_dh_file(file1, overwrite=True) as dh5:
        data = np.random.randint(-100, 100, size=(50, 2), dtype=np.int16)
        index = create_empty_index_array(1)
        index[0] = (0, 0)
        create_cont_group_from_data_in_file(dh5._file, 0, data, index, np.int32(1000))

        # Add WAVELET1
        amplitude, phase, idx, frequency_axis = create_test_wavelet_data(
            2, 50, 20, start_time=0, seed=42
        )
        create_wavelet_group_from_data_in_file(
            dh5._file,
            1,
            amplitude=amplitude,
            phase=phase,
            index=idx,
            sample_period_ns=np.int32(10000000),
            frequency_axis=frequency_axis,
        )

    # File 2 WITHOUT WAVELET
    with create_dh_file(file2, overwrite=True) as dh5:
        data = np.random.randint(-100, 100, size=(50, 2), dtype=np.int16)
        index = create_empty_index_array(1)
        index[0] = (1000000, 0)
        create_cont_group_from_data_in_file(dh5._file, 0, data, index, np.int32(1000))
        # No wavelet

    # File 3 with CONT and WAVELET
    with create_dh_file(file3, overwrite=True) as dh5:
        data = np.random.randint(-100, 100, size=(50, 2), dtype=np.int16)
        index = create_empty_index_array(1)
        index[0] = (2000000, 0)
        create_cont_group_from_data_in_file(dh5._file, 0, data, index, np.int32(1000))

        # Add WAVELET1
        amplitude, phase, idx, frequency_axis = create_test_wavelet_data(
            2, 50, 20, start_time=2000000, seed=43
        )
        create_wavelet_group_from_data_in_file(
            dh5._file,
            1,
            amplitude=amplitude,
            phase=phase,
            index=idx,
            sample_period_ns=np.int32(10000000),
            frequency_axis=frequency_axis,
        )

    # Merge - should succeed with CONT but no WAVELET (since not all files have it)
    merge_dh5_files([file1, file2, file3], output)

    # Verify merged file
    with DH5File(output, mode="r") as merged:
        # CONT should be merged
        cont_ids = merged.get_cont_group_ids()
        assert 0 in cont_ids

        # No WAVELET blocks should be merged (not common to all files)
        wavelet_ids = merged.get_wavelet_group_ids()
        assert len(wavelet_ids) == 0


def test_concatenate_wavelet_data(temp_dir):
    """Test the concatenate_wavelet_data function directly."""
    # Create test wavelets
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"

    n_channels = 2
    n_frequencies = 15
    n_samples_1 = 50
    n_samples_2 = 75

    create_test_dh5_file_with_wavelet(
        file1,
        wavelet_id=0,
        n_channels=n_channels,
        n_samples=n_samples_1,
        n_frequencies=n_frequencies,
    )
    create_test_dh5_file_with_wavelet(
        file2,
        wavelet_id=0,
        n_channels=n_channels,
        n_samples=n_samples_2,
        n_frequencies=n_frequencies,
    )

    # Load wavelets
    dh5_1 = DH5File(file1, mode="r")
    dh5_2 = DH5File(file2, mode="r")

    try:
        wavelet1 = dh5_1.get_wavelet_group_by_id(0)
        wavelet2 = dh5_2.get_wavelet_group_by_id(0)

        # Concatenate
        merged_amplitude, merged_phase, merged_index = concatenate_wavelet_data(
            [wavelet1, wavelet2]
        )

        # Check shapes
        assert merged_amplitude.shape == (
            n_channels,
            n_samples_1 + n_samples_2,
            n_frequencies,
        )
        assert merged_phase.shape == (
            n_channels,
            n_samples_1 + n_samples_2,
            n_frequencies,
        )

        # Check index
        assert len(merged_index) == 2
        assert merged_index[0]["offset"] == 0
        assert merged_index[1]["offset"] == n_samples_1

    finally:
        dh5_1._file.close()
        dh5_2._file.close()


def test_validate_wavelet_blocks_compatible(temp_dir):
    """Test the validate_wavelet_blocks_compatible function."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"

    create_test_dh5_file_with_wavelet(
        file1, wavelet_id=0, n_channels=2, n_samples=50, n_frequencies=20, start_time=0
    )
    create_test_dh5_file_with_wavelet(
        file2,
        wavelet_id=0,
        n_channels=2,
        n_samples=75,
        n_frequencies=20,
        start_time=1_000_000_000,
    )

    dh5_1 = DH5File(file1, mode="r")
    dh5_2 = DH5File(file2, mode="r")

    try:
        wavelet1 = dh5_1.get_wavelet_group_by_id(0)
        wavelet2 = dh5_2.get_wavelet_group_by_id(0)

        # Should not raise - compatible wavelets
        validate_wavelet_blocks_compatible([wavelet1, wavelet2], 0)

    finally:
        dh5_1._file.close()
        dh5_2._file.close()


def test_wavelet_data_preservation(temp_dir):
    """Test that wavelet data is correctly preserved after merging."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    n_channels = 2
    n_frequencies = 10
    n_samples_1 = 20
    n_samples_2 = 30
    wavelet_id = 0

    # Create files with known data
    np.random.seed(123)
    amp1 = np.random.rand(n_channels, n_samples_1, n_frequencies) * 50.0
    phase1 = (np.random.rand(n_channels, n_samples_1, n_frequencies) - 0.5) * 2 * np.pi

    np.random.seed(456)
    amp2 = np.random.rand(n_channels, n_samples_2, n_frequencies) * 50.0
    phase2 = (np.random.rand(n_channels, n_samples_2, n_frequencies) - 0.5) * 2 * np.pi

    freq_axis = np.linspace(1.0, 100.0, n_frequencies)
    index1 = create_empty_wavelet_index(1)
    index1[0] = (0, 0, 1.0 / 65535.0 * np.max(amp1) * 1.1)

    index2 = create_empty_wavelet_index(1)
    # File 1 has 20 samples at 10ms each = 200ms total, so start file 2 after that
    index2[0] = (250_000_000, 0, 1.0 / 65535.0 * np.max(amp2) * 1.1)

    with create_dh_file(file1, overwrite=True) as dh5:
        create_wavelet_group_from_data_in_file(
            dh5._file,
            wavelet_id,
            amplitude=amp1,
            phase=phase1,
            index=index1,
            sample_period_ns=np.int32(10000000),
            frequency_axis=freq_axis,
        )

    with create_dh_file(file2, overwrite=True) as dh5:
        create_wavelet_group_from_data_in_file(
            dh5._file,
            wavelet_id,
            amplitude=amp2,
            phase=phase2,
            index=index2,
            sample_period_ns=np.int32(10000000),
            frequency_axis=freq_axis,
        )

    # Merge
    merge_dh5_files([file1, file2], output)

    # Verify data
    with DH5File(output, mode="r") as merged:
        wavelet = merged.get_wavelet_group_by_id(wavelet_id)

        # Get merged data
        merged_amp = wavelet.get_amplitude_calibrated()
        merged_phase_rad = wavelet.get_phase_radians()

        # Check that first part matches file1 data (approximately, due to quantization)
        assert merged_amp.shape == (
            n_channels,
            n_samples_1 + n_samples_2,
            n_frequencies,
        )

        # Due to quantization, we expect some loss but should be close
        np.testing.assert_allclose(
            merged_amp[:, :n_samples_1, :], amp1, rtol=0.01, atol=1.0
        )
        np.testing.assert_allclose(
            merged_amp[:, n_samples_1:, :], amp2, rtol=0.01, atol=1.0
        )

        # Phase should be more accurate (int8 quantization)
        np.testing.assert_allclose(
            merged_phase_rad[:, :n_samples_1, :], phase1, atol=0.03
        )
        np.testing.assert_allclose(
            merged_phase_rad[:, n_samples_1:, :], phase2, atol=0.03
        )
