import h5py
import numpy as np
import pytest

import dh5io
import dh5io.spike as spike
from dh5io.create import create_dh_file
from dh5io.errors import DH5Error
from dh5io.validation import validate_dh5_file
from dhspec.spike import SpikeParams


def test_spike_block_with_waveforms(tmp_path):
    filename = tmp_path / "test.dh5"
    rng = np.random.default_rng(0)
    waveforms = rng.integers(-1000, 1000, size=(5, 32, 4), dtype=np.int16)
    timestamps = np.array([10, 20, 35, 50, 70], dtype=np.int64) * 1_000_000
    with create_dh_file(filename) as dh5file:
        spike.create_spike_group_from_data_in_file(
            dh5file._file, 3, timestamps, sample_period_ns=33_333, waveforms=waveforms,
            pre_trigger_samples=10, cluster_info=[1, 2, 1, 1, 2], calibration=np.full(4, 0.5),
            name="tetrode 1",
        )

    validate_dh5_file(filename)
    with dh5io.DH5File(filename, "r") as dh5file:
        block = dh5file.get_spike_by_id(3)
        assert block.name == "tetrode 1" and block.has_waveforms
        assert block.n_spikes == 5 and block.n_channels == 4
        assert block.params.spikeSamples == 32 and block.params.preTrigSamples == 10
        assert block._group["DATA"].shape == (5 * 32, 4)
        np.testing.assert_array_equal(block.index, timestamps)
        np.testing.assert_array_equal(block.waveforms, waveforms)
        np.testing.assert_array_equal(block.read_waveforms(2, 2), waveforms[2:4])
        np.testing.assert_array_equal(block.cluster_info, [1, 2, 1, 1, 2])
        np.testing.assert_allclose(block.calibrated_waveforms, waveforms * 0.5)
        assert [s.id for s in dh5file.get_spikes()] == [3]


def test_spike_block_without_waveforms(tmp_path):
    """spikeSamples = 0: timestamps only, with the mean waveform as MEAN_WAVEFORM."""
    filename = tmp_path / "test.dh5"
    timestamps = np.arange(1000, dtype=np.int64) * 1_000_000
    mean = (np.sin(np.linspace(0, 3, 61)) * 1000).astype(np.int16)[:, None]
    with create_dh_file(filename) as dh5file:
        spike.create_spike_group_from_data_in_file(
            dh5file._file, None, timestamps, sample_period_ns=33_333, n_channels=1,
            pre_trigger_samples=20, calibration=np.array([1e-6]), mean_waveform=mean,
        )

    validate_dh5_file(filename)
    with dh5io.DH5File(filename, "r") as dh5file:
        block = dh5file.get_spike_by_id(0)
        assert not block.has_waveforms
        assert block._group["DATA"].shape == (0, 1)
        assert block.waveforms.shape == (1000, 0, 1)
        np.testing.assert_array_equal(block.index, timestamps)
        np.testing.assert_array_equal(block.mean_waveform, mean)
        np.testing.assert_allclose(block.calibrated_mean_waveform, mean * 1e-6)
        assert block.cluster_info is None


def test_write_in_chunks(tmp_path):
    filename = tmp_path / "test.dh5"
    with create_dh_file(filename) as dh5file:
        group = spike.create_empty_spike_group_in_file(
            dh5file._file, 1, nSpikes=10, nChannels=2, spikeParams=SpikeParams(8, 2, 0),
            sample_period_ns=33_333, compression="gzip",
        )
        block = dh5io.Spike(group)
        for first in (0, 5):
            block.write_spikes(first, np.arange(first, first + 5) * 100,
                               np.full((5, 8, 2), first, dtype=np.int16))
        np.testing.assert_array_equal(block.index, np.arange(10) * 100)
        assert block.read_waveforms(5, 1).max() == 5
    validate_dh5_file(filename)


def test_errors(tmp_path):
    with create_dh_file(tmp_path / "test.dh5") as dh5file:
        file = dh5file._file
        spike.create_spike_group_from_data_in_file(file, 1, [1, 2], sample_period_ns=1000)
        with pytest.raises(DH5Error, match="already exists"):
            spike.create_spike_group_from_data_in_file(file, 1, [1, 2], sample_period_ns=1000)
        with pytest.raises(DH5Error, match="shape"):
            spike.create_spike_group_from_data_in_file(file, 2, [1, 2], 1000, waveforms=np.zeros((3, 4, 1)))
        with pytest.raises(DH5Error, match="stores no waveforms"):
            dh5io.Spike(file["SPIKE1"]).write_spikes(0, [1], np.zeros((1, 4, 1), np.int16))
        with pytest.raises(DH5Error, match="only for blocks without waveforms"):
            spike.create_spike_group_from_data_in_file(
                file, 3, [1], 1000, waveforms=np.zeros((1, 4, 1), np.int16), mean_waveform=np.zeros((4, 1)))


def test_validation_detects_wrong_data_size(tmp_path):
    filename = tmp_path / "test.dh5"
    with create_dh_file(filename) as dh5file:
        spike.create_spike_group_from_data_in_file(
            dh5file._file, 1, [1, 2, 3], 1000, waveforms=np.zeros((3, 4, 1), np.int16))
    with h5py.File(filename, "r+") as file:  # one waveform for three spikes
        del file["SPIKE1/DATA"]
        file["SPIKE1"].create_dataset("DATA", data=np.zeros((4, 1), np.int16))
    with pytest.raises(DH5Error, match="expected spikeSamples"):
        validate_dh5_file(filename)


def test_channel_info_from_create_channel_info(tmp_path):
    from dhspec.cont import create_channel_info

    with create_dh_file(tmp_path / "test.dh5") as dh5file:
        group = spike.create_spike_group_from_data_in_file(
            dh5file._file, 1, [1, 2], 1000, channels=create_channel_info(5, 2, 16, 0.0, 0.0, 0), calibration=1e-6)
        assert dh5io.Spike(group).channels["GlobalChanNumber"][0] == 5
