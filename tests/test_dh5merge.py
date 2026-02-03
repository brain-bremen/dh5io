"""Test the dh5merge CLI tool."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from dh5cli.dh5merge import (
    determine_cont_blocks_to_merge,
    merge_dh5_files,
    merge_index_arrays_with_shapes,
    suggest_merged_filename,
)
from dh5io.cont import create_cont_group_from_data_in_file, create_empty_index_array
from dh5io.create import create_dh_file
from dh5io.dh5file import DH5File


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def create_test_dh5_file(
    filepath: Path,
    cont_id: int,
    n_samples: int,
    n_channels: int,
    start_time: int = 0,
) -> None:
    """
    Create a test DH5 file with a single CONT block.

    Parameters
    ----------
    filepath : Path
        Output file path
    cont_id : int
        CONT block ID
    n_samples : int
        Number of samples
    n_channels : int
        Number of channels
    start_time : int
        Start time in nanoseconds
    """
    # Create test data
    data = np.random.randint(-1000, 1000, size=(n_samples, n_channels), dtype=np.int16)

    # Create index with a single recording region
    index = create_empty_index_array(1)
    index[0]["time"] = start_time
    index[0]["offset"] = 0

    # Create file
    with create_dh_file(filepath, overwrite=True, boards=["TestBoard"]) as dh5:
        create_cont_group_from_data_in_file(
            dh5._file,
            cont_id,
            data=data,
            index=index,
            sample_period_ns=np.int32(1000),  # 1 microsecond
            calibration=np.ones(n_channels, dtype=np.float64),
            name=f"CONT{cont_id}",
        )


def test_merge_two_files(temp_dir):
    """Test merging two DH5 files with the same CONT block."""
    # Create two test files
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    n_channels = 3
    n_samples_1 = 100
    n_samples_2 = 200
    cont_id = 0

    create_test_dh5_file(file1, cont_id, n_samples_1, n_channels, start_time=0)
    create_test_dh5_file(file2, cont_id, n_samples_2, n_channels, start_time=1000000)

    # Merge files
    merge_dh5_files([file1, file2], output)

    # Verify merged file
    with DH5File(output, mode="r") as merged:
        cont = merged.get_cont_group_by_id(cont_id)

        # Check dimensions
        assert cont.n_samples == n_samples_1 + n_samples_2
        assert cont.n_channels == n_channels

        # Check index has two regions
        assert cont.n_regions == 2
        assert cont.index[0]["time"] == 0
        assert cont.index[0]["offset"] == 0
        assert cont.index[1]["time"] == 1000000
        assert cont.index[1]["offset"] == n_samples_1


def test_merge_multiple_files(temp_dir):
    """Test merging three DH5 files."""
    files = [temp_dir / f"file{i}.dh5" for i in range(3)]
    output = temp_dir / "merged.dh5"

    n_channels = 2
    n_samples = [50, 75, 100]
    cont_id = 1
    start_times = [0, 500000, 1000000]

    for file, n_samp, start_time in zip(files, n_samples, start_times):
        create_test_dh5_file(file, cont_id, n_samp, n_channels, start_time)

    # Merge files
    merge_dh5_files(files, output)

    # Verify merged file
    with DH5File(output, mode="r") as merged:
        cont = merged.get_cont_group_by_id(cont_id)

        # Check total samples
        assert cont.n_samples == sum(n_samples)
        assert cont.n_channels == n_channels

        # Check index offsets
        assert cont.n_regions == 3
        expected_offsets = [0, n_samples[0], n_samples[0] + n_samples[1]]
        for i, expected_offset in enumerate(expected_offsets):
            assert cont.index[i]["offset"] == expected_offset
            assert cont.index[i]["time"] == start_times[i]


def test_merge_specific_cont_ids(temp_dir):
    """Test merging only specific CONT blocks."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # Create files with multiple CONT blocks
    with create_dh_file(file1, overwrite=True) as dh5:
        for cont_id in [0, 1, 2]:
            data = np.random.randint(-100, 100, size=(50, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (0, 0)
            create_cont_group_from_data_in_file(
                dh5._file, cont_id, data, index, np.int32(1000)
            )

    with create_dh_file(file2, overwrite=True) as dh5:
        for cont_id in [0, 1, 2]:
            data = np.random.randint(-100, 100, size=(50, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (1000000, 0)
            create_cont_group_from_data_in_file(
                dh5._file, cont_id, data, index, np.int32(1000)
            )

    # Merge only CONT blocks 0 and 2
    merge_dh5_files([file1, file2], output, cont_ids=[0, 2])

    # Verify merged file
    with DH5File(output, mode="r") as merged:
        cont_ids = merged.get_cont_group_ids()
        assert sorted(cont_ids) == [0, 2]

        # CONT1 should not be present
        assert 1 not in cont_ids


def test_incompatible_channel_count(temp_dir):
    """Test that merging files with different channel counts raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    cont_id = 0
    create_test_dh5_file(file1, cont_id, 100, n_channels=2)
    create_test_dh5_file(file2, cont_id, 100, n_channels=3)  # Different channel count

    # Should raise ValueError
    with pytest.raises(ValueError, match="Number of channels mismatch"):
        merge_dh5_files([file1, file2], output)


def test_incompatible_sample_period(temp_dir):
    """Test that merging files with different sample periods raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"

    # Create files with different sample periods
    with create_dh_file(file1, overwrite=True) as dh5:
        data = np.random.randint(-100, 100, size=(50, 2), dtype=np.int16)
        index = create_empty_index_array(1)
        index[0] = (0, 0)
        create_cont_group_from_data_in_file(
            dh5._file,
            0,
            data,
            index,
            np.int32(1000),  # 1000 ns
        )

    with create_dh_file(file2, overwrite=True) as dh5:
        data = np.random.randint(-100, 100, size=(50, 2), dtype=np.int16)
        index = create_empty_index_array(1)
        index[0] = (1000000, 0)
        create_cont_group_from_data_in_file(
            dh5._file,
            0,
            data,
            index,
            np.int32(2000),  # 2000 ns - different!
        )

    output = temp_dir / "merged.dh5"

    # Should raise ValueError
    with pytest.raises(ValueError, match="Sample period mismatch"):
        merge_dh5_files([file1, file2], output)


def test_no_common_cont_blocks(temp_dir):
    """Test that merging files with no common CONT blocks raises an error."""
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    create_test_dh5_file(file1, cont_id=0, n_samples=100, n_channels=2)
    create_test_dh5_file(file2, cont_id=1, n_samples=100, n_channels=2)

    # Should raise ValueError
    with pytest.raises(ValueError, match="No common CONT or WAVELET blocks"):
        merge_dh5_files([file1, file2], output)


def test_merge_index_arrays():
    """Test the merge_index_arrays_with_shapes function."""
    # Create test index arrays
    index1 = create_empty_index_array(2)
    index1[0] = (0, 0)
    index1[1] = (1000000, 50)

    index2 = create_empty_index_array(1)
    index2[0] = (2000000, 0)

    index3 = create_empty_index_array(2)
    index3[0] = (3000000, 0)
    index3[1] = (4000000, 30)

    # Data shapes
    shapes = [(100, 2), (75, 2), (50, 2)]

    # Merge indices
    merged = merge_index_arrays_with_shapes([index1, index2, index3], shapes)

    # Check result
    assert len(merged) == 5

    # First file's indices should be unchanged
    assert merged[0]["time"] == 0
    assert merged[0]["offset"] == 0
    assert merged[1]["time"] == 1000000
    assert merged[1]["offset"] == 50

    # Second file's indices should be offset by 100
    assert merged[2]["time"] == 2000000
    assert merged[2]["offset"] == 100

    # Third file's indices should be offset by 100 + 75 = 175
    assert merged[3]["time"] == 3000000
    assert merged[3]["offset"] == 175
    assert merged[4]["time"] == 4000000
    assert merged[4]["offset"] == 175 + 30


def test_determine_cont_blocks_to_merge(temp_dir):
    """Test the determine_cont_blocks_to_merge function."""
    # Create test files with different CONT blocks
    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    file3 = temp_dir / "file3.dh5"

    with create_dh_file(file1, overwrite=True) as dh5:
        for cont_id in [0, 1, 2]:
            data = np.random.randint(-100, 100, size=(10, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (0, 0)
            create_cont_group_from_data_in_file(
                dh5._file, cont_id, data, index, np.int32(1000)
            )

    with create_dh_file(file2, overwrite=True) as dh5:
        for cont_id in [0, 2, 3]:
            data = np.random.randint(-100, 100, size=(10, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (0, 0)
            create_cont_group_from_data_in_file(
                dh5._file, cont_id, data, index, np.int32(1000)
            )

    with create_dh_file(file3, overwrite=True) as dh5:
        for cont_id in [0, 1, 3]:
            data = np.random.randint(-100, 100, size=(10, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (0, 0)
            create_cont_group_from_data_in_file(
                dh5._file, cont_id, data, index, np.int32(1000)
            )

    # Open files
    dh5_files = [DH5File(f, mode="r") for f in [file1, file2, file3]]

    try:
        # Test finding common blocks (should be only CONT0)
        common = determine_cont_blocks_to_merge(dh5_files, cont_ids=None)
        assert common == {0}

        # Test with specific IDs that exist in all files
        specific = determine_cont_blocks_to_merge(dh5_files, cont_ids=[0])
        assert specific == {0}

        # Test with specific IDs that don't exist in all files (should raise error)
        with pytest.raises(ValueError, match="not found in file"):
            determine_cont_blocks_to_merge(dh5_files, cont_ids=[0, 1, 2])

    finally:
        for dh5_file in dh5_files:
            dh5_file._file.close()


def test_merge_trialmaps(temp_dir):
    """Test that TRIALMAPs are merged correctly."""
    from dh5io.trialmap import add_trialmap_to_file
    from dhspec.trialmap import TRIALMAP_DATASET_DTYPE

    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    output = temp_dir / "merged.dh5"

    # Create trialmaps
    trialmap1 = np.recarray(5, dtype=TRIALMAP_DATASET_DTYPE)
    for i in range(5):
        trialmap1[i].TrialNo = i + 1
        trialmap1[i].StimNo = 1
        trialmap1[i].Outcome = 1
        trialmap1[i].StartTime = i * 1000000
        trialmap1[i].EndTime = (i + 1) * 1000000

    trialmap2 = np.recarray(3, dtype=TRIALMAP_DATASET_DTYPE)
    for i in range(3):
        trialmap2[i].TrialNo = i + 6
        trialmap2[i].StimNo = 2
        trialmap2[i].Outcome = 0
        trialmap2[i].StartTime = (i + 5) * 1000000
        trialmap2[i].EndTime = (i + 6) * 1000000

    # Create files with TRIALMAPs
    with create_dh_file(file1, overwrite=True) as dh5:
        data = np.random.randint(-100, 100, size=(50, 2), dtype=np.int16)
        index = create_empty_index_array(1)
        index[0] = (0, 0)
        create_cont_group_from_data_in_file(dh5._file, 0, data, index, np.int32(1000))
        add_trialmap_to_file(dh5._file, trialmap1)

    with create_dh_file(file2, overwrite=True) as dh5:
        data = np.random.randint(-100, 100, size=(50, 2), dtype=np.int16)
        index = create_empty_index_array(1)
        index[0] = (5000000, 0)
        create_cont_group_from_data_in_file(dh5._file, 0, data, index, np.int32(1000))
        add_trialmap_to_file(dh5._file, trialmap2)

    # Merge
    merge_dh5_files([file1, file2], output)

    # Verify merged TRIALMAP
    with DH5File(output, mode="r") as merged:
        trialmap = merged.get_trialmap()
        assert trialmap is not None
        assert len(trialmap) == 8  # 5 + 3

        # Check first trial from file 1
        assert trialmap.trial_numbers[0] == 1
        assert trialmap.trial_type_numbers[0] == 1

        # Check first trial from file 2
        assert trialmap.trial_numbers[5] == 6
        assert trialmap.trial_type_numbers[5] == 2


def test_merge_with_missing_trialmaps(temp_dir):
    """Test merging when some files don't have TRIALMAPs."""
    from dh5io.trialmap import add_trialmap_to_file
    from dhspec.trialmap import TRIALMAP_DATASET_DTYPE

    file1 = temp_dir / "file1.dh5"
    file2 = temp_dir / "file2.dh5"
    file3 = temp_dir / "file3.dh5"
    output = temp_dir / "merged.dh5"

    # Create trialmap for file 1
    trialmap1 = np.recarray(3, dtype=TRIALMAP_DATASET_DTYPE)
    for i in range(3):
        trialmap1[i].TrialNo = i + 1
        trialmap1[i].StimNo = 1
        trialmap1[i].Outcome = 1
        trialmap1[i].StartTime = i * 1000000
        trialmap1[i].EndTime = (i + 1) * 1000000

    # File 1 with TRIALMAP
    with create_dh_file(file1, overwrite=True) as dh5:
        data = np.random.randint(-100, 100, size=(50, 2), dtype=np.int16)
        index = create_empty_index_array(1)
        index[0] = (0, 0)
        create_cont_group_from_data_in_file(dh5._file, 0, data, index, np.int32(1000))
        add_trialmap_to_file(dh5._file, trialmap1)

    # File 2 WITHOUT TRIALMAP
    with create_dh_file(file2, overwrite=True) as dh5:
        data = np.random.randint(-100, 100, size=(50, 2), dtype=np.int16)
        index = create_empty_index_array(1)
        index[0] = (3000000, 0)
        create_cont_group_from_data_in_file(dh5._file, 0, data, index, np.int32(1000))
        # No trialmap

    # File 3 with TRIALMAP
    trialmap3 = np.recarray(2, dtype=TRIALMAP_DATASET_DTYPE)
    for i in range(2):
        trialmap3[i].TrialNo = i + 4
        trialmap3[i].StimNo = 2
        trialmap3[i].Outcome = 0
        trialmap3[i].StartTime = (i + 3) * 1000000
        trialmap3[i].EndTime = (i + 4) * 1000000

    with create_dh_file(file3, overwrite=True) as dh5:
        data = np.random.randint(-100, 100, size=(50, 2), dtype=np.int16)
        index = create_empty_index_array(1)
        index[0] = (5000000, 0)
        create_cont_group_from_data_in_file(dh5._file, 0, data, index, np.int32(1000))
        add_trialmap_to_file(dh5._file, trialmap3)

    # Merge
    merge_dh5_files([file1, file2, file3], output)

    # Verify merged TRIALMAP (should have 3 + 2 = 5 trials, skipping file2)
    with DH5File(output, mode="r") as merged:
        trialmap = merged.get_trialmap()
        assert trialmap is not None
        assert len(trialmap) == 5  # 3 from file1 + 2 from file3

        # Check trials are in correct order
        assert trialmap.trial_numbers[0] == 1
        assert trialmap.trial_numbers[2] == 3
        assert trialmap.trial_numbers[3] == 4
        assert trialmap.trial_numbers[4] == 5


def test_suggest_merged_filename():
    """Test the suggest_merged_filename function."""
    # Test case 1: Files with common prefix
    files = [
        Path("session1_day1.dh5"),
        Path("session1_day2.dh5"),
        Path("session1_day3.dh5"),
    ]
    result = suggest_merged_filename(files)
    assert result is not None
    assert result.name == "session1_day_merged.dh5"

    # Test case 2: Files with underscore separator
    files = [Path("experiment_A.dh5"), Path("experiment_B.dh5")]
    result = suggest_merged_filename(files)
    assert result is not None
    assert result.name == "experiment_merged.dh5"

    # Test case 3: Files with no overlap (should return None)
    files = [Path("file1.dh5"), Path("data2.dh5")]
    result = suggest_merged_filename(files)
    assert result is None

    # Test case 4: Files with very short overlap (1 char, should return None)
    files = [Path("x1.dh5"), Path("x2.dh5")]
    result = suggest_merged_filename(files)
    assert result is None

    # Test case 5: Files with trailing underscore in prefix
    files = [Path("mouse123_trial_1.dh5"), Path("mouse123_trial_2.dh5")]
    result = suggest_merged_filename(files)
    assert result is not None
    assert result.name == "mouse123_trial_merged.dh5"

    # Test case 6: Complete overlap except for extension
    files = [Path("data.dh5"), Path("data.dh5")]
    result = suggest_merged_filename(files)
    assert result is not None
    assert result.name == "data_merged.dh5"

    # Test case 7: Files in different directories
    files = [Path("/tmp/data_session1.dh5"), Path("/tmp/data_session2.dh5")]
    result = suggest_merged_filename(files)
    assert result is not None
    assert result.name == "data_session_merged.dh5"
    assert result.parent == Path("/tmp")

    # Test case 8: Partial word overlap
    files = [
        Path("recording_file1.dh5"),
        Path("recording_file2.dh5"),
        Path("recording_data3.dh5"),
    ]
    result = suggest_merged_filename(files)
    assert result is not None
    assert result.name == "recording_merged.dh5"

    # Test case 9: Less than 2 files (should return None)
    files = [Path("file1.dh5")]
    result = suggest_merged_filename(files)
    assert result is None

    # Test case 10: Empty list (should return None)
    files = []
    result = suggest_merged_filename(files)
    assert result is None


def test_merge_without_output_filename(temp_dir):
    """Test merging files with auto-suggested output filename."""
    # Create test files with common prefix
    file1 = temp_dir / "session_part1.dh5"
    file2 = temp_dir / "session_part2.dh5"

    n_channels = 2
    cont_id = 0

    create_test_dh5_file(file1, cont_id, 50, n_channels, start_time=0)
    create_test_dh5_file(file2, cont_id, 75, n_channels, start_time=1000000)

    # Suggest output filename
    suggested = suggest_merged_filename([file1, file2])
    assert suggested is not None
    assert suggested.name == "session_part_merged.dh5"
    assert suggested.parent == temp_dir

    # Use suggested filename to merge
    merge_dh5_files([file1, file2], suggested)

    # Verify merged file exists and is correct
    assert suggested.exists()
    with DH5File(suggested, mode="r") as merged:
        cont = merged.get_cont_group_by_id(cont_id)
        assert cont.n_samples == 125  # 50 + 75
