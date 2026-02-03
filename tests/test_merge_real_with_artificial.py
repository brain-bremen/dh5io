"""
Integration test for merging real test.dh5 with artificial continuation data.

This test creates an artificial DH5 file with CONT, WAVELET, and TRIALMAP data
that continues after the last data point in tests/test.dh5, then merges them
and validates the results.
"""

import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from dh5cli.dh5merge import merge_dh5_files
from dh5io.dh5file import DH5File


@pytest.fixture
def real_test_file():
    """Path to the real test.dh5 file."""
    test_file = Path(__file__).parent / "test.dh5"
    if not test_file.exists():
        pytest.skip("test.dh5 not found")
    return str(test_file)


@pytest.fixture
def artificial_continuation_file(tmp_path):
    """
    Create an artificial DH5 file that continues after test.dh5's last data point.

    Based on test.dh5 analysis:
    - Last TRIALMAP end time: 5379267000000 ns
    - CONT blocks: sample period 1000000 ns (1ms), 1 channel each
    - WAVELET blocks: sample period 10000000 ns (10ms), 1 channel, 35 frequencies (5-160 Hz)
    - Block names: CONT1, CONT1001, CONT60-64, WAVELET1, WAVELET1001
    """
    artificial_file = tmp_path / "artificial_continuation.dh5"

    # Start time: 1 second after the last trial end time from test.dh5
    start_time = 5379267000000 + 1_000_000_000  # 5380267000000 ns

    with h5py.File(artificial_file, "w") as f:
        # Create CONT blocks matching test.dh5 structure
        cont_blocks = [
            "CONT1",
            "CONT1001",
            "CONT60",
            "CONT61",
            "CONT62",
            "CONT63",
            "CONT64",
        ]
        cont_sample_period = 1_000_000  # 1 ms in nanoseconds
        cont_duration = 10_000_000_000  # 10 seconds of data
        cont_num_samples = int(cont_duration / cont_sample_period)  # 10,000 samples

        for block_name in cont_blocks:
            grp = f.create_group(block_name)
            grp.attrs["SamplePeriod"] = np.int32(cont_sample_period)

            # Add Calibration only for CONT1 and CONT1001 to match test.dh5
            # CONT60-64 don't have Calibration in test.dh5
            if block_name in ["CONT1", "CONT1001"]:
                grp.attrs["Calibration"] = np.array([1.0172526e-07])

            # Channels dtype matching test.dh5
            # Note: CONT1001 in test.dh5 doesn't have Channels attribute
            if block_name != "CONT1001":
                channels_dtype = np.dtype(
                    [
                        ("GlobalChanNumber", np.int16),
                        ("BoardChanNo", np.int16),
                        ("ADCBitWidth", np.int16),
                        ("MaxVoltageRange", np.float32),
                        ("MinVoltageRange", np.float32),
                        ("AmplifChan0", np.float32),
                    ]
                )
                channels_data = np.array(
                    [(1, 1, 16, 10.0, -10.0, 1.0)], dtype=channels_dtype
                )
                grp.attrs["Channels"] = channels_data

            # Create continuous data (1 channel)
            # Simulate a simple sine wave with channel-specific frequency
            channel_freq = hash(block_name) % 100 + 1  # 1-100 Hz
            t = (
                np.arange(cont_num_samples) * cont_sample_period * 1e-9
            )  # time in seconds
            data = np.sin(2 * np.pi * channel_freq * t).astype(np.int16)
            data = data.reshape(
                -1, 1
            )  # Shape: (N samples, 1 channel) to match test.dh5

            grp.create_dataset("DATA", data=data, dtype=np.int16)

            # Create INDEX with 3 regions
            num_regions = 3
            samples_per_region = cont_num_samples // num_regions

            index_dtype = np.dtype([("time", np.int64), ("offset", np.int64)])

            index_data = np.zeros(num_regions, dtype=index_dtype)
            for i in range(num_regions):
                index_data[i]["time"] = (
                    start_time + i * samples_per_region * cont_sample_period
                )
                index_data[i]["offset"] = i * samples_per_region

            grp.create_dataset("INDEX", data=index_data)

        # Create WAVELET blocks matching test.dh5 structure
        wavelet_blocks = ["WAVELET1", "WAVELET1001"]
        wavelet_sample_period = 10_000_000  # 10 ms in nanoseconds
        wavelet_num_samples = int(
            cont_duration / wavelet_sample_period
        )  # 1,000 samples
        num_frequencies = 35

        # Frequency axis matching test.dh5: 5 to 160 Hz, 35 bins
        frequency_axis = np.logspace(np.log10(5), np.log10(160), num_frequencies)

        for block_name in wavelet_blocks:
            grp = f.create_group(block_name)
            grp.attrs["SamplePeriod"] = np.int32(wavelet_sample_period)
            grp.attrs["FrequencyAxis"] = frequency_axis

            # Create wavelet data (1 channel, N samples, 35 frequencies)
            # Simulate wavelet transform with frequency-dependent amplitude and phase
            wavelet_dtype = np.dtype(
                [
                    ("a", np.uint16),  # amplitude (stored as uint16)
                    ("phi", np.int8),  # phase (stored as int8)
                ]
            )

            data = np.zeros(
                (1, wavelet_num_samples, num_frequencies), dtype=wavelet_dtype
            )

            for freq_idx in range(num_frequencies):
                # Amplitude varies with frequency and time
                amp_base = 1000.0 / (
                    freq_idx + 1
                )  # Higher frequencies have lower amplitude
                for t_idx in range(wavelet_num_samples):
                    # Time-varying amplitude
                    amp_modulation = 1.0 + 0.3 * np.sin(
                        2 * np.pi * t_idx / wavelet_num_samples
                    )
                    amplitude = amp_base * amp_modulation

                    # Store amplitude (will be scaled by INDEX scaling factor)
                    data[0, t_idx, freq_idx]["a"] = np.clip(int(amplitude), 0, 65535)

                    # Phase varies with time
                    phase_rad = 2 * np.pi * t_idx / wavelet_num_samples + freq_idx * 0.1
                    # Convert to int8: phi = phase_rad * 127 / pi
                    phi_stored = int(np.clip(phase_rad * 127.0 / np.pi, -127, 127))
                    data[0, t_idx, freq_idx]["phi"] = phi_stored

            grp.create_dataset("DATA", data=data)

            # Create INDEX with 3 regions
            num_regions = 3
            samples_per_region = wavelet_num_samples // num_regions

            index_dtype = np.dtype(
                [("time", np.int64), ("offset", np.int64), ("scaling", np.float64)]
            )

            index_data = np.zeros(num_regions, dtype=index_dtype)
            for i in range(num_regions):
                index_data[i]["time"] = (
                    start_time + i * samples_per_region * wavelet_sample_period
                )
                index_data[i]["offset"] = i * samples_per_region
                index_data[i]["scaling"] = 1.0  # Simple scaling factor

            grp.create_dataset("INDEX", data=index_data)

        # Create TRIALMAP entries
        # Add 5 new trials continuing from the last trial in test.dh5
        # Last trial in test.dh5: (679, 82, 0, 5376111360000, 5379267000000)
        trialmap_dtype = np.dtype(
            [
                ("TrialNo", np.int32),
                ("StimNo", np.int32),
                ("Outcome", np.int32),
                ("StartTime", np.int64),
                ("EndTime", np.int64),
            ]
        )

        num_new_trials = 5
        trialmap_data = np.zeros(num_new_trials, dtype=trialmap_dtype)

        trial_duration = 3_000_000_000  # 3 seconds per trial
        last_trial_no = 679
        last_stim_no = 82

        for i in range(num_new_trials):
            trial_start = start_time + i * trial_duration
            trial_end = trial_start + trial_duration - 1

            trialmap_data[i]["TrialNo"] = last_trial_no + i + 1
            trialmap_data[i]["StimNo"] = (
                last_stim_no + (i % 3) + 1
            )  # Cycle through stim numbers
            trialmap_data[i]["Outcome"] = i % 2  # Alternate between 0 and 1
            trialmap_data[i]["StartTime"] = trial_start
            trialmap_data[i]["EndTime"] = trial_end

        f.create_dataset("TRIALMAP", data=trialmap_data)

    return str(artificial_file)


def test_merge_real_with_artificial(
    real_test_file, artificial_continuation_file, tmp_path
):
    """
    Test merging real test.dh5 with artificial continuation data.

    This test validates:
    1. CONT data is correctly concatenated
    2. WAVELET data is correctly concatenated
    3. TRIALMAP entries are combined
    4. INDEX offsets are properly updated
    5. Timestamps are in correct order
    """
    merged_file = tmp_path / "merged_output.dh5"

    # Perform the merge
    merge_dh5_files(
        [Path(real_test_file), Path(artificial_continuation_file)], merged_file
    )

    # Validate the merged file
    with h5py.File(merged_file, "r") as merged:
        # Get info from original files
        with h5py.File(real_test_file, "r") as real:
            with h5py.File(artificial_continuation_file, "r") as artificial:
                # Test CONT blocks
                cont_blocks = [
                    "CONT1",
                    "CONT1001",
                    "CONT60",
                    "CONT61",
                    "CONT62",
                    "CONT63",
                    "CONT64",
                ]
                for block_name in cont_blocks:
                    print(f"\nValidating {block_name}...")

                    # Check that block exists in merged file
                    assert block_name in merged, (
                        f"{block_name} missing from merged file"
                    )

                    real_block = real[block_name]
                    artificial_block = artificial[block_name]
                    merged_block = merged[block_name]

                    # Validate DATA concatenation
                    real_data = real_block["DATA"][:]
                    artificial_data = artificial_block["DATA"][:]
                    merged_data = merged_block["DATA"][:]

                    # CONT data shape: (n_samples, n_channels) in test.dh5
                    real_samples = real_data.shape[0]
                    artificial_samples = artificial_data.shape[0]
                    merged_samples = merged_data.shape[0]

                    expected_samples = real_samples + artificial_samples
                    assert merged_samples == expected_samples, (
                        f"{block_name} DATA samples mismatch: {merged_samples} != {expected_samples}"
                    )

                    # Validate INDEX
                    real_index = real_block["INDEX"][:]
                    artificial_index = artificial_block["INDEX"][:]
                    merged_index = merged_block["INDEX"][:]

                    expected_index_len = len(real_index) + len(artificial_index)
                    assert len(merged_index) == expected_index_len, (
                        f"{block_name} INDEX length mismatch"
                    )

                    # Check timestamps are monotonic
                    times = merged_index["time"]
                    assert np.all(times[1:] >= times[:-1]), (
                        f"{block_name} INDEX times not monotonic"
                    )

                    # Check that real file times come first
                    assert np.all(
                        merged_index["time"][: len(real_index)] == real_index["time"]
                    ), f"{block_name} real file times don't match"

                    # Check that artificial file times are appended
                    assert np.all(
                        merged_index["time"][len(real_index) :]
                        == artificial_index["time"]
                    ), f"{block_name} artificial file times don't match"

                    # Check that offsets are updated correctly
                    # Second file's offsets should be shifted by real file's total samples
                    # Note: real_samples is the first dimension (number of samples)
                    artificial_offsets_in_merged = merged_index["offset"][
                        len(real_index) :
                    ]
                    expected_offsets = artificial_index["offset"] + real_samples
                    assert np.all(artificial_offsets_in_merged == expected_offsets), (
                        f"{block_name} offset update incorrect"
                    )

                # Test WAVELET blocks
                wavelet_blocks = ["WAVELET1", "WAVELET1001"]
                for block_name in wavelet_blocks:
                    print(f"\nValidating {block_name}...")

                    assert block_name in merged, (
                        f"{block_name} missing from merged file"
                    )

                    real_block = real[block_name]
                    artificial_block = artificial[block_name]
                    merged_block = merged[block_name]

                    # Validate DATA concatenation
                    real_data = real_block["DATA"][:]
                    artificial_data = artificial_block["DATA"][:]
                    merged_data = merged_block["DATA"][:]

                    # WAVELET shape: (channels, time_samples, frequencies)
                    real_samples = real_data.shape[1]
                    artificial_samples = artificial_data.shape[1]
                    merged_samples = merged_data.shape[1]

                    expected_samples = real_samples + artificial_samples
                    assert merged_samples == expected_samples, (
                        f"{block_name} DATA samples mismatch: {merged_samples} != {expected_samples}"
                    )

                    # Check channels and frequencies match
                    assert merged_data.shape[0] == real_data.shape[0], (
                        f"{block_name} channel count mismatch"
                    )
                    assert merged_data.shape[2] == real_data.shape[2], (
                        f"{block_name} frequency count mismatch"
                    )

                    # Validate attributes
                    assert (
                        merged_block.attrs["SamplePeriod"]
                        == real_block.attrs["SamplePeriod"]
                    ), f"{block_name} SamplePeriod mismatch"
                    assert np.allclose(
                        merged_block.attrs["FrequencyAxis"],
                        real_block.attrs["FrequencyAxis"],
                    ), f"{block_name} FrequencyAxis mismatch"

                    # Validate INDEX
                    real_index = real_block["INDEX"][:]
                    artificial_index = artificial_block["INDEX"][:]
                    merged_index = merged_block["INDEX"][:]

                    expected_index_len = len(real_index) + len(artificial_index)
                    assert len(merged_index) == expected_index_len, (
                        f"{block_name} INDEX length mismatch"
                    )

                    # Check timestamps are monotonic
                    times = merged_index["time"]
                    assert np.all(times[1:] >= times[:-1]), (
                        f"{block_name} INDEX times not monotonic"
                    )

                    # Check offset updates
                    artificial_offsets_in_merged = merged_index["offset"][
                        len(real_index) :
                    ]
                    expected_offsets = artificial_index["offset"] + real_samples
                    assert np.all(artificial_offsets_in_merged == expected_offsets), (
                        f"{block_name} offset update incorrect"
                    )

                    # Check scaling factors are preserved
                    if "scaling" in artificial_index.dtype.names:
                        assert np.allclose(
                            merged_index["scaling"][len(real_index) :],
                            artificial_index["scaling"],
                        ), f"{block_name} scaling factors not preserved"

                # Test TRIALMAP
                print("\nValidating TRIALMAP...")
                assert "TRIALMAP" in merged, "TRIALMAP missing from merged file"

                real_trialmap = real["TRIALMAP"][:]
                artificial_trialmap = artificial["TRIALMAP"][:]
                merged_trialmap = merged["TRIALMAP"][:]

                expected_trialmap_len = len(real_trialmap) + len(artificial_trialmap)
                assert len(merged_trialmap) == expected_trialmap_len, (
                    f"TRIALMAP length mismatch: {len(merged_trialmap)} != {expected_trialmap_len}"
                )

                # Check that real trials come first
                for i in range(len(real_trialmap)):
                    for field in real_trialmap.dtype.names:
                        assert merged_trialmap[i][field] == real_trialmap[i][field], (
                            f"TRIALMAP entry {i} field {field} mismatch"
                        )

                # Check that artificial trials are appended
                for i in range(len(artificial_trialmap)):
                    merged_idx = len(real_trialmap) + i
                    for field in artificial_trialmap.dtype.names:
                        assert (
                            merged_trialmap[merged_idx][field]
                            == artificial_trialmap[i][field]
                        ), f"TRIALMAP entry {merged_idx} field {field} mismatch"

                # Check temporal ordering
                start_times = merged_trialmap["StartTime"]
                assert np.all(start_times[1:] >= start_times[:-1]), (
                    "TRIALMAP StartTime not monotonic"
                )

                # Verify no temporal overlap between real and artificial data
                last_real_end = real_trialmap[-1]["EndTime"]
                first_artificial_start = artificial_trialmap[0]["StartTime"]
                assert first_artificial_start > last_real_end, (
                    "Artificial data should start after real data ends"
                )

                print(f"\n✓ Merge validation complete!")
                print(f"  Total TRIALMAP entries: {len(merged_trialmap)}")
                print(f"  Real file trials: {len(real_trialmap)}")
                print(f"  Artificial file trials: {len(artificial_trialmap)}")
                print(
                    f"  Time range: {merged_trialmap[0]['StartTime']} - {merged_trialmap[-1]['EndTime']}"
                )


def test_merged_data_continuity(real_test_file, artificial_continuation_file, tmp_path):
    """
    Test that merged data maintains continuity without gaps.
    """
    merged_file = tmp_path / "merged_continuity.dh5"

    merge_dh5_files(
        [Path(real_test_file), Path(artificial_continuation_file)], merged_file
    )

    with h5py.File(merged_file, "r") as f:
        # Check CONT block continuity
        block = f["CONT1"]
        index = block["INDEX"][:]
        times = index["time"]
        offsets = index["offset"]
        sample_period = block.attrs["SamplePeriod"]

        # Verify no duplicate timestamps
        assert len(np.unique(times)) == len(times), (
            "Duplicate timestamps in merged CONT INDEX"
        )

        # Check that INDEX offsets are strictly increasing
        assert np.all(np.diff(offsets) > 0), (
            "CONT INDEX offsets not strictly increasing"
        )

        # Check WAVELET block continuity
        block = f["WAVELET1"]
        index = block["INDEX"][:]
        times = index["time"]
        offsets = index["offset"]

        assert len(np.unique(times)) == len(times), (
            "Duplicate timestamps in merged WAVELET INDEX"
        )
        assert np.all(np.diff(offsets) > 0), (
            "WAVELET INDEX offsets not strictly increasing"
        )


def test_merged_file_structure(real_test_file, artificial_continuation_file, tmp_path):
    """
    Test that merged file has correct overall structure.
    """
    merged_file = tmp_path / "merged_structure.dh5"

    merge_dh5_files(
        [Path(real_test_file), Path(artificial_continuation_file)], merged_file
    )

    with h5py.File(merged_file, "r") as f:
        # Check that all expected blocks exist
        expected_cont = [
            "CONT1",
            "CONT1001",
            "CONT60",
            "CONT61",
            "CONT62",
            "CONT63",
            "CONT64",
        ]
        expected_wavelet = ["WAVELET1", "WAVELET1001"]

        for block_name in expected_cont:
            assert block_name in f, f"Missing {block_name}"
            assert "DATA" in f[block_name], f"Missing DATA in {block_name}"
            assert "INDEX" in f[block_name], f"Missing INDEX in {block_name}"

        for block_name in expected_wavelet:
            assert block_name in f, f"Missing {block_name}"
            assert "DATA" in f[block_name], f"Missing DATA in {block_name}"
            assert "INDEX" in f[block_name], f"Missing INDEX in {block_name}"
            assert "FrequencyAxis" in f[block_name].attrs, (
                f"Missing FrequencyAxis in {block_name}"
            )

        assert "TRIALMAP" in f, "Missing TRIALMAP"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
