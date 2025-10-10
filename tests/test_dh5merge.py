"""Test the dh5merge CLI tool."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from dh5io.create import create_dh_file
from dh5io.cont import create_cont_group_from_data_in_file, create_empty_index_array
from dh5io.dh5file import DH5File
from dh5cli.dh5merge import (
    merge_dh5_files,
    determine_cont_blocks_to_merge,
    merge_index_arrays_with_shapes,
)


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
            dh5.file,
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
                dh5.file, cont_id, data, index, np.int32(1000)
            )

    with create_dh_file(file2, overwrite=True) as dh5:
        for cont_id in [0, 1, 2]:
            data = np.random.randint(-100, 100, size=(50, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (1000000, 0)
            create_cont_group_from_data_in_file(
                dh5.file, cont_id, data, index, np.int32(1000)
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
            dh5.file,
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
            dh5.file,
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
    with pytest.raises(ValueError, match="No common CONT blocks"):
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
                dh5.file, cont_id, data, index, np.int32(1000)
            )

    with create_dh_file(file2, overwrite=True) as dh5:
        for cont_id in [0, 2, 3]:
            data = np.random.randint(-100, 100, size=(10, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (0, 0)
            create_cont_group_from_data_in_file(
                dh5.file, cont_id, data, index, np.int32(1000)
            )

    with create_dh_file(file3, overwrite=True) as dh5:
        for cont_id in [0, 1, 3]:
            data = np.random.randint(-100, 100, size=(10, 2), dtype=np.int16)
            index = create_empty_index_array(1)
            index[0] = (0, 0)
            create_cont_group_from_data_in_file(
                dh5.file, cont_id, data, index, np.int32(1000)
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
            dh5_file.file.close()
