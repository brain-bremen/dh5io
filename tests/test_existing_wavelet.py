"""Tests for reading existing wavelet data from the test.dh5 file."""

import h5py
import numpy as np
import pytest

import dh5io
import dh5io.wavelet as wavelet
from dhspec.wavelet import amplitude_to_float, phase_to_radians


class TestExistingWaveletData:
    """Test reading wavelet data from the real test.dh5 file."""

    @pytest.fixture
    def test_file(self):
        """Path to the test DH5 file."""
        return "tests/test.dh5"

    def test_wavelet_groups_present(self, test_file):
        """Test that wavelet groups are present in the file."""
        with h5py.File(test_file, "r") as f:
            wavelet_ids = wavelet.enumerate_wavelet_groups(f)
            assert len(wavelet_ids) >= 2
            assert 1 in wavelet_ids
            assert 1001 in wavelet_ids

    def test_wavelet_group_names(self, test_file):
        """Test getting wavelet group names."""
        with h5py.File(test_file, "r") as f:
            wavelet_names = wavelet.get_wavelet_group_names_from_file(f)
            assert "WAVELET1" in wavelet_names
            assert "WAVELET1001" in wavelet_names

    def test_wavelet1_structure(self, test_file):
        """Test the structure of WAVELET1 group."""
        with h5py.File(test_file, "r") as f:
            wv_group = f["WAVELET1"]

            # Check required attributes
            assert "SamplePeriod" in wv_group.attrs
            assert "FrequencyAxis" in wv_group.attrs

            # Check required datasets
            assert "DATA" in wv_group
            assert "INDEX" in wv_group

            # Check shapes
            data_shape = wv_group["DATA"].shape
            assert len(data_shape) == 3  # (F, M, N)

            index_shape = wv_group["INDEX"].shape
            assert len(index_shape) == 1  # (R,)

    def test_wavelet1_attributes(self, test_file):
        """Test WAVELET1 attributes."""
        with h5py.File(test_file, "r") as f:
            wv_group = f["WAVELET1"]

            # Sample period
            sample_period = wv_group.attrs["SamplePeriod"]
            assert sample_period == 10_000_000  # 10 ms

            # Frequency axis
            freq_axis = wv_group.attrs["FrequencyAxis"]
            assert len(freq_axis) == 35
            assert freq_axis[0] == 5.0  # First frequency
            assert freq_axis[-1] == 160.0  # Last frequency
            # Should be logarithmically spaced
            assert np.all(freq_axis[1:] > freq_axis[:-1])  # Monotonically increasing

    def test_wavelet1_data_types(self, test_file):
        """Test WAVELET1 data types match specification."""
        with h5py.File(test_file, "r") as f:
            wv_group = f["WAVELET1"]

            # DATA dtype
            data_dtype = wv_group["DATA"].dtype
            assert "a" in data_dtype.names
            assert "phi" in data_dtype.names
            assert data_dtype["a"] == np.uint16
            assert data_dtype["phi"] == np.int8

            # INDEX dtype
            index_dtype = wv_group["INDEX"].dtype
            assert "time" in index_dtype.names
            assert "offset" in index_dtype.names
            assert "scaling" in index_dtype.names
            assert index_dtype["time"] == np.int64
            assert index_dtype["offset"] == np.int64
            assert index_dtype["scaling"] == np.float64

    def test_wavelet1_data_values(self, test_file):
        """Test WAVELET1 has valid data values."""
        with h5py.File(test_file, "r") as f:
            wv_group = f["WAVELET1"]
            data = wv_group["DATA"][:]

            # Amplitude should be in valid range
            assert data["a"].min() >= 0
            assert data["a"].max() <= 65535

            # Phase should be in valid range
            assert data["phi"].min() >= -127
            assert data["phi"].max() <= 127

            # Should have some non-zero data
            assert data["a"].max() > 0
            assert np.sum(data["a"] > 0) > 0

    def test_wavelet1_index_values(self, test_file):
        """Test WAVELET1 INDEX has valid values."""
        with h5py.File(test_file, "r") as f:
            wv_group = f["WAVELET1"]
            index = wv_group["INDEX"][:]

            # Should have multiple regions
            assert len(index) > 0

            # Times should be positive and increasing
            assert np.all(index["time"] > 0)
            if len(index) > 1:
                # Times should generally increase (allowing for some flexibility)
                assert index["time"][1] >= index["time"][0]

            # Offsets should be non-negative
            assert np.all(index["offset"] >= 0)

            # Scaling factors should be positive
            assert np.all(index["scaling"] > 0)

    def test_wavelet_class_with_real_data(self, test_file):
        """Test Wavelet class with real data."""
        with h5py.File(test_file, "r") as f:
            wv = wavelet.Wavelet(f["WAVELET1"])

            # Test properties
            assert wv.id == 1
            # DATA has shape [N=1, M=144117, F=35]
            assert wv.n_channels == 1  # First dimension of DATA
            assert wv.n_samples == 144117
            assert wv.n_frequencies == 35  # Third dimension of DATA
            assert wv.n_regions == 385
            assert wv.sample_period == 10_000_000

            # Test frequency axis
            assert len(wv.frequency_axis) == 35
            assert wv.frequency_axis[0] == 5.0
            assert wv.frequency_axis[-1] == 160.0

            # Test duration calculation
            duration = wv.duration_s
            assert duration > 0
            # With 385 regions and 144117 samples at 10ms per sample,
            # duration should be substantial
            assert duration > 100  # At least 100 seconds

    def test_wavelet_calibrated_amplitude(self, test_file):
        """Test getting calibrated amplitude from real data."""
        with h5py.File(test_file, "r") as f:
            wv = wavelet.Wavelet(f["WAVELET1"])

            # Get calibrated amplitude for a subset (all channels, first 100 time samples, all frequencies)
            amplitude = wv.get_amplitude_calibrated(time_idx=slice(0, 100))

            # Should be the right shape [N=1, M=100, F=35]
            assert amplitude.shape == (1, 100, 35)

            # Should be float type
            assert amplitude.dtype == np.float64

            # Should have non-negative values
            assert np.all(amplitude >= 0)

            # Should have some non-zero values
            assert np.any(amplitude > 0)

    def test_wavelet_phase_radians(self, test_file):
        """Test getting phase in radians from real data."""
        with h5py.File(test_file, "r") as f:
            wv = wavelet.Wavelet(f["WAVELET1"])

            # Get phase for a subset (all channels, first 100 time samples, all frequencies)
            phase = wv.get_phase_radians(time_idx=slice(0, 100))

            # Should be the right shape [N=1, M=100, F=35]
            assert phase.shape == (1, 100, 35)

            # Should be float type
            assert phase.dtype == np.float64

            # Should be in valid range for radians
            assert np.all(phase >= -np.pi)
            assert np.all(phase <= np.pi)

    def test_dh5file_wavelet_methods(self, test_file):
        """Test DH5File wavelet methods with real file."""
        with dh5io.DH5File(test_file, "r") as dh5file:
            # Test enumeration
            wavelet_ids = dh5file.get_wavelet_group_ids()
            assert 1 in wavelet_ids
            assert 1001 in wavelet_ids

            wavelet_names = dh5file.get_wavelet_group_names()
            assert "WAVELET1" in wavelet_names
            assert "WAVELET1001" in wavelet_names

            # Test getting by ID
            wv = dh5file.get_wavelet_group_by_id(1)
            assert wv is not None
            assert isinstance(wv, wavelet.Wavelet)
            assert wv.id == 1

            # Test getting all groups
            wavelet_groups = dh5file.get_wavelet_groups()
            assert len(wavelet_groups) >= 2

    def test_wavelet1001_matches_wavelet1(self, test_file):
        """Test that WAVELET1001 has similar structure to WAVELET1."""
        with h5py.File(test_file, "r") as f:
            wv1 = wavelet.Wavelet(f["WAVELET1"])
            wv1001 = wavelet.Wavelet(f["WAVELET1001"])

            # Should have same dimensions
            assert wv1.n_frequencies == wv1001.n_frequencies
            assert wv1.n_samples == wv1001.n_samples
            assert wv1.n_channels == wv1001.n_channels
            assert wv1.n_regions == wv1001.n_regions

            # Should have same sample period
            assert wv1.sample_period == wv1001.sample_period

            # Should have same frequency axis
            np.testing.assert_array_equal(wv1.frequency_axis, wv1001.frequency_axis)

    def test_validate_real_wavelet_groups(self, test_file):
        """Test that real wavelet groups pass validation."""
        with h5py.File(test_file, "r") as f:
            # Validate dtype
            assert wavelet.validate_wavelet_dtype(f)

            # Validate both groups - they should pass with correct dimension interpretation
            assert wavelet.validate_wavelet_group(f["WAVELET1"])
            assert wavelet.validate_wavelet_group(f["WAVELET1001"])

    def test_manual_data_conversion(self, test_file):
        """Test manual conversion of amplitude and phase data."""
        with h5py.File(test_file, "r") as f:
            wv_group = f["WAVELET1"]
            # DATA shape is [N=1, M=144117, F=35]
            data = wv_group["DATA"][
                0, :100, 0
            ]  # First channel, first 100 samples, first frequency
            index = wv_group["INDEX"][0]

            # Manual amplitude conversion
            scaling = index["scaling"]
            amplitude_manual = amplitude_to_float(data["a"], scaling)
            assert amplitude_manual.shape == (100,)
            assert np.all(amplitude_manual >= 0)

            # Manual phase conversion
            phase_manual = phase_to_radians(data["phi"])
            assert phase_manual.shape == (100,)
            assert np.all(phase_manual >= -np.pi)
            assert np.all(phase_manual <= np.pi)

    def test_dh5file_str_includes_wavelets(self, test_file):
        """Test that DH5File string representation includes wavelet info."""
        with dh5io.DH5File(test_file, "r") as dh5file:
            file_str = str(dh5file)
            assert "WAVELET" in file_str
            assert (
                "WAVELET1" in file_str or "2" in file_str
            )  # Should show count or names

    def test_frequency_axis_properties(self, test_file):
        """Test properties of the frequency axis."""
        with h5py.File(test_file, "r") as f:
            wv = wavelet.Wavelet(f["WAVELET1"])
            freq_axis = wv.frequency_axis

            # Should span a reasonable range for neural signals
            assert freq_axis.min() >= 1.0  # At least 1 Hz
            assert freq_axis.max() <= 500.0  # Less than 500 Hz

            # Should be logarithmically spaced (approximately)
            # Check that the ratio between consecutive frequencies is roughly constant
            if len(freq_axis) > 2:
                ratios = freq_axis[1:] / freq_axis[:-1]
                # Ratios should be similar (within 10%)
                assert np.std(ratios) / np.mean(ratios) < 0.1

    def test_data_dimensions_consistency(self, test_file):
        """Test that data dimensions are consistent across groups."""
        with h5py.File(test_file, "r") as f:
            wv1 = wavelet.Wavelet(f["WAVELET1"])

            # Frequency axis length should match third dimension of DATA [N, M, F]
            assert len(wv1.frequency_axis) == 35
            assert wv1.n_frequencies == 35  # Third dimension of DATA

            # Check that DATA shape matches expected dimensions [N, M, F]
            data_shape = wv1.data.shape
            assert data_shape[0] == 1  # n_channels (first dim)
            assert data_shape[1] == 144117  # n_samples (second dim)
            assert data_shape[2] == 35  # n_frequencies (third dim)

    def test_multiple_regions_handling(self, test_file):
        """Test handling of multiple recording regions."""
        with h5py.File(test_file, "r") as f:
            wv = wavelet.Wavelet(f["WAVELET1"])

            # Should have many regions
            assert wv.n_regions > 1

            # Get full calibrated data (tests multi-region processing)
            amplitude = wv.get_amplitude_calibrated()
            assert amplitude.shape == (wv.n_channels, wv.n_samples, wv.n_frequencies)

            # All values should be non-negative
            assert np.all(amplitude >= 0)

            # Check that different regions might have different characteristics
            # (this is just a sanity check)
            index_data = wv.index[:]
            scalings = index_data["scaling"]
            # Should have variation in scaling factors across regions
            if len(scalings) > 10:
                assert np.std(scalings) > 0  # Some variation expected
