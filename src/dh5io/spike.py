"""Spike data in SPIKE blocks of DAQ-HDF files.

A ``SPIKEn`` group holds the spikes of one nTrode (see :mod:`dhspec.spike` for the format):
``DATA`` (int16, ``spikeSamples * nSpikes`` rows by ``nChannels``), ``INDEX`` (int64
timestamps in ns, one per spike, at the trigger point) and the optional ``CLUSTER_INFO``
(uint8 cluster number per spike).

Waveforms are handled as arrays of shape ``(nSpikes, spikeSamples, nChannels)``; spike ``k``
occupies rows ``k * spikeSamples`` to ``(k + 1) * spikeSamples`` of ``DATA``. The format
does not require waveforms: with ``spikeSamples = 0``, ``DATA`` is empty (0 rows) and the
block holds timestamps (and cluster numbers) only, e.g. for spike-sorter output. Such a
block can have a ``MEAN_WAVEFORM`` dataset (dh5io extension, see :mod:`dhspec.spike`).
"""

from __future__ import annotations

import warnings

import h5py
import numpy as np
import numpy.typing as npt

from dh5io.errors import DH5CalibrationMissingWarning, DH5DataTypeConversionWarning, DH5Error
from dh5io.hdf5_strings import ascii_str, decode_str_attr
from dhspec.cont import CalibrationType
from dhspec.spike import (
    CLUSTER_INFO_DATASET_NAME,
    DATA_DATASET_NAME,
    INDEX_DATASET_NAME,
    MEAN_WAVEFORM_DATASET_NAME,
    SPIKE_CHANNELS_DTYPE,
    SPIKE_PARAMS_DTYPE,
    SPIKE_PREFIX,
    SpikeParams,
    spike_id_from_name,
    spike_name_from_id,
)

MAX_SPIKE_ID = 65535


def _spike_params_array(params: SpikeParams) -> np.ndarray:
    array = np.zeros((), dtype=SPIKE_PARAMS_DTYPE)
    array["spikeSamples"] = params.spikeSamples
    array["preTrigSamples"] = params.preTrigSamples
    array["lockOutSamples"] = params.lockOutSamples
    return array


class Spike:
    """Abstraction for the HDF5 group of a SPIKE block."""

    def __init__(self, group: h5py.Group):
        self._group = group

    def __repr__(self):
        return f"Spike(group={self._group.name}, file={self._group.file.filename})"

    def __str__(self):
        return f"""    {self._group.name} in {self._group.file.filename}
        ├─── id: {self.id}
        ├─── name: {self.name}
        ├─── comment: {self.comment}
        ├─── sample_period: {self.sample_period} ns
        ├─── n_channels: {self.n_channels}
        ├─── n_spikes: {self.n_spikes}
        ├─── params: {self.params}
        ├─── cluster_info: {"yes" if self.has_cluster_info else "no"}
        └─── mean_waveform: {"yes" if self.mean_waveform is not None else "no"}
            """

    @property
    def id(self) -> int:
        return spike_id_from_name(self._group.name)

    @property
    def name(self) -> str:
        return decode_str_attr(self._group.attrs.get("Name", ""))

    @property
    def comment(self) -> str:
        return decode_str_attr(self._group.attrs.get("Comment", ""))

    @property
    def params(self) -> SpikeParams:
        p = self._group.attrs["SpikeParams"]
        return SpikeParams(
            spikeSamples=np.int16(p["spikeSamples"]),
            preTrigSamples=np.int16(p["preTrigSamples"]),
            lockOutSamples=np.int16(p["lockOutSamples"]),
        )

    @property
    def sample_period(self) -> int:
        """Sample period of the waveforms in nanoseconds."""
        return int(self._group.attrs["SamplePeriod"])

    @property
    def calibration(self) -> CalibrationType | None:
        return self._group.attrs.get("Calibration")

    @property
    def channels(self) -> np.ndarray | None:
        return self._group.attrs.get("Channels")

    @property
    def n_channels(self) -> int:
        return self._group[DATA_DATASET_NAME].shape[1]

    @property
    def n_spikes(self) -> int:
        return self._group[INDEX_DATASET_NAME].shape[0]

    @property
    def has_waveforms(self) -> bool:
        return int(self.params.spikeSamples) > 0

    @property
    def index(self) -> npt.NDArray[np.int64]:
        """Spike timestamps (trigger points) in nanoseconds."""
        return self._group[INDEX_DATASET_NAME][()]

    @property
    def has_cluster_info(self) -> bool:
        return CLUSTER_INFO_DATASET_NAME in self._group

    @property
    def cluster_info(self) -> npt.NDArray[np.uint8] | None:
        if not self.has_cluster_info:
            return None
        return self._group[CLUSTER_INFO_DATASET_NAME][()]

    @property
    def mean_waveform(self) -> npt.NDArray[np.int16] | None:
        """Mean waveform (samples, nChannels) of a block without waveforms, or None."""
        if MEAN_WAVEFORM_DATASET_NAME not in self._group:
            return None
        return self._group[MEAN_WAVEFORM_DATASET_NAME][()]

    @property
    def calibrated_mean_waveform(self) -> npt.NDArray[np.float64] | None:
        mean = self.mean_waveform
        if mean is None:
            return None
        calibration = self.calibration
        return mean.astype(np.float64) if calibration is None else mean * calibration

    def read_waveforms(self, first_spike: int = 0, count: int | None = None) -> npt.NDArray[np.int16]:
        """Waveforms of ``count`` spikes from ``first_spike``: (count, spikeSamples, nChannels)."""
        samples = int(self.params.spikeSamples)
        if count is None:
            count = self.n_spikes - first_spike
        rows = self._group[DATA_DATASET_NAME][first_spike * samples : (first_spike + count) * samples]
        return rows.reshape(count, samples, self.n_channels)

    @property
    def waveforms(self) -> npt.NDArray[np.int16]:
        """All waveforms: (nSpikes, spikeSamples, nChannels); spikeSamples is 0 without waveforms."""
        return self.read_waveforms()

    @property
    def calibrated_waveforms(self) -> npt.NDArray[np.float64]:
        calibration = self.calibration
        if calibration is None:
            warnings.warn(
                DH5CalibrationMissingWarning(
                    f"Calibration attribute is missing from {self._group.name}", cont_id=self.id
                )
            )
            return self.waveforms.astype(np.float64)
        return self.waveforms * calibration

    def write_spikes(
        self,
        first_spike: int,
        timestamps_ns: npt.ArrayLike,
        waveforms: npt.ArrayLike | None = None,
        cluster_info: npt.ArrayLike | None = None,
    ) -> None:
        """Write consecutive spikes from ``first_spike`` on (e.g. to write a block in chunks).

        ``waveforms`` (count, spikeSamples, nChannels) is required if the block has waveforms
        and is stored as int16.
        """
        timestamps = np.asarray(timestamps_ns, dtype=np.int64)
        count = timestamps.shape[0]
        if first_spike < 0 or first_spike + count > self.n_spikes:
            raise DH5Error(f"spikes {first_spike}..{first_spike + count} outside 0..{self.n_spikes}")
        samples = int(self.params.spikeSamples)
        if samples > 0:
            if waveforms is None:
                raise DH5Error(f"{self._group.name} stores waveforms ({samples} samples per spike)")
            waveforms = np.asarray(waveforms)
            if waveforms.shape != (count, samples, self.n_channels):
                raise DH5Error(
                    f"waveforms have shape {waveforms.shape}, expected {(count, samples, self.n_channels)}"
                )
            if waveforms.dtype != np.int16:
                warnings.warn(
                    DH5DataTypeConversionWarning(f"Waveforms were converted from {waveforms.dtype} to int16")
                )
                waveforms = waveforms.astype(np.int16)
            self._group[DATA_DATASET_NAME][first_spike * samples : (first_spike + count) * samples] = (
                waveforms.reshape(count * samples, self.n_channels)
            )
        elif waveforms is not None and np.size(waveforms) > 0:
            raise DH5Error(f"{self._group.name} has spikeSamples = 0 and stores no waveforms")
        self._group[INDEX_DATASET_NAME][first_spike : first_spike + count] = timestamps
        if cluster_info is not None:
            if not self.has_cluster_info:
                self._group.create_dataset(CLUSTER_INFO_DATASET_NAME, shape=(self.n_spikes,), dtype=np.uint8)
            self._group[CLUSTER_INFO_DATASET_NAME][first_spike : first_spike + count] = np.asarray(
                cluster_info, dtype=np.uint8
            )

    @classmethod
    def from_group(cls, group: h5py.Group) -> Spike:
        return cls(group)


# create
def create_empty_spike_group_in_file(
    file: h5py.File,
    spike_group_id: int | None,
    nSpikes: int,
    nChannels: int,
    spikeParams: SpikeParams,
    sample_period_ns: np.int32 | int,
    calibration: CalibrationType | None = None,
    channels: np.ndarray | None = None,
    name: str | None = None,
    comment: str | None = None,
    cluster_info: bool = False,
    compression: str | None = None,
) -> h5py.Group:
    """Create a ``SPIKEn`` group with zero-filled ``DATA`` and ``INDEX`` for ``nSpikes`` spikes.

    ``spike_group_id=None`` takes the next free id. ``spikeParams.spikeSamples = 0`` creates a
    block without waveforms. ``cluster_info=True`` also creates the ``CLUSTER_INFO`` dataset.
    ``compression`` (e.g. ``"gzip"``) is passed to h5py.
    """
    if file.mode not in ("r+", "w", "a"):
        raise DH5Error(f"File must be opened with write access but is open with {file.mode}")
    existing = enumerate_spike_groups(file)
    if spike_group_id is None:
        spike_group_id = max(existing, default=-1) + 1
    if spike_group_id in existing:
        raise DH5Error(f"{spike_name_from_id(spike_group_id)} already exists in {file.filename}")
    if not 0 <= spike_group_id <= MAX_SPIKE_ID:
        raise DH5Error(f"SPIKE id {spike_group_id} outside 0..{MAX_SPIKE_ID}")
    if spikeParams.spikeSamples < 0:
        raise DH5Error("SpikeParams.spikeSamples must not be negative")
    if calibration is not None and len(calibration) != nChannels:
        raise DH5Error(f"Calibration has {len(calibration)} values for {nChannels} channels")
    if channels is not None and len(channels) != nChannels:
        raise DH5Error(f"Channels has {len(channels)} entries for {nChannels} channels")

    group = file.create_group(spike_name_from_id(spike_group_id))
    rows = int(nSpikes) * int(spikeParams.spikeSamples)
    # h5py cannot compress empty datasets
    group.create_dataset(
        DATA_DATASET_NAME, shape=(rows, nChannels), dtype=np.int16, compression=compression if rows else None
    )
    group.create_dataset(
        INDEX_DATASET_NAME, shape=(nSpikes,), dtype=np.int64, compression=compression if nSpikes else None
    )
    if cluster_info:
        group.create_dataset(
            CLUSTER_INFO_DATASET_NAME, shape=(nSpikes,), dtype=np.uint8, compression=compression if nSpikes else None
        )
    group.attrs["SpikeParams"] = _spike_params_array(spikeParams)
    group.attrs["SamplePeriod"] = np.int32(sample_period_ns)
    if calibration is not None:
        group.attrs["Calibration"] = np.asarray(calibration, dtype=np.float64)
    if channels is not None:
        group.attrs["Channels"] = np.asarray(channels, dtype=SPIKE_CHANNELS_DTYPE)
    group.attrs["Name"] = ascii_str(name if name is not None else spike_name_from_id(spike_group_id))
    group.attrs["Comment"] = ascii_str(comment or "")
    return group


def create_spike_group_from_data_in_file(
    file: h5py.File,
    spike_group_id: int | None,
    timestamps_ns: npt.ArrayLike,
    sample_period_ns: np.int32 | int,
    waveforms: npt.ArrayLike | None = None,
    n_channels: int = 1,
    pre_trigger_samples: int = 0,
    lockout_samples: int = 0,
    cluster_info: npt.ArrayLike | None = None,
    calibration: CalibrationType | None = None,
    channels: np.ndarray | None = None,
    name: str | None = None,
    comment: str | None = None,
    compression: str | None = None,
    mean_waveform: npt.ArrayLike | None = None,
) -> h5py.Group:
    """Create a SPIKE block from timestamps (ns) and optionally waveforms
    (nSpikes, spikeSamples, nChannels).

    Without ``waveforms`` the block has ``spikeSamples = 0`` and ``n_channels`` channels, and
    can store ``mean_waveform`` (samples, nChannels; int16, scaled by ``calibration``) with
    ``pre_trigger_samples`` samples before the trigger point.
    """
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    if waveforms is None:
        samples = 0
    else:
        waveforms = np.asarray(waveforms)
        if waveforms.ndim != 3 or waveforms.shape[0] != timestamps.shape[0]:
            raise DH5Error(
                f"waveforms must have shape (nSpikes={timestamps.shape[0]}, spikeSamples, nChannels), "
                f"not {waveforms.shape}"
            )
        _, samples, n_channels = waveforms.shape
    if mean_waveform is not None:
        if waveforms is not None:
            raise DH5Error("MEAN_WAVEFORM is only for blocks without waveforms")
        mean_waveform = np.asarray(mean_waveform)
        if mean_waveform.ndim != 2 or mean_waveform.shape[1] != n_channels:
            raise DH5Error(f"mean_waveform must have shape (samples, {n_channels}), not {mean_waveform.shape}")
        if mean_waveform.dtype != np.int16:
            warnings.warn(
                DH5DataTypeConversionWarning(f"Mean waveform was converted from {mean_waveform.dtype} to int16")
            )
            mean_waveform = mean_waveform.astype(np.int16)
    group = create_empty_spike_group_in_file(
        file,
        spike_group_id,
        nSpikes=timestamps.shape[0],
        nChannels=n_channels,
        spikeParams=SpikeParams(np.int16(samples), np.int16(pre_trigger_samples), np.int16(lockout_samples)),
        sample_period_ns=sample_period_ns,
        calibration=calibration,
        channels=channels,
        name=name,
        comment=comment,
        cluster_info=cluster_info is not None,
        compression=compression,
    )
    if mean_waveform is not None:
        group.create_dataset(MEAN_WAVEFORM_DATASET_NAME, data=mean_waveform)
    if timestamps.shape[0]:
        Spike(group).write_spikes(0, timestamps, waveforms, cluster_info)
    return group


# read
def enumerate_spike_groups(file: h5py.File) -> list[int]:
    return [spike_id_from_name(name) for name in get_spike_group_names_from_file(file)]


def get_spike_group_names_from_file(file: h5py.File) -> list[str]:
    return [
        name for name in file.keys() if name.startswith(SPIKE_PREFIX) and isinstance(file[name], h5py.Group)
    ]


def get_spike_groups_from_file(file: h5py.File) -> list[h5py.Group]:
    """Get all SPIKE groups of the file."""
    return [file[name] for name in get_spike_group_names_from_file(file)]


def get_spike_group_by_id_from_file(file: h5py.File, id: int) -> h5py.Group | None:
    """Get a SPIKE group by id, or None if there is none."""
    name = spike_name_from_id(id)
    if name in file:
        return file[name]
    return None


# validate
def validate_spike_group(group: h5py.Group) -> None:
    """Check the required attributes and datasets of a SPIKE group and their sizes."""
    name = group.name
    for attribute in ("SpikeParams", "SamplePeriod"):
        if attribute not in group.attrs:
            raise DH5Error(f"{attribute} attribute is missing from SPIKE group {name}")
    params = group.attrs["SpikeParams"]
    if params.dtype.names != SPIKE_PARAMS_DTYPE.names:
        raise DH5Error(
            f"SpikeParams in {name} has fields {params.dtype.names}, expected {SPIKE_PARAMS_DTYPE.names}"
        )
    for dataset in (DATA_DATASET_NAME, INDEX_DATASET_NAME):
        if dataset not in group or not isinstance(group[dataset], h5py.Dataset):
            raise DH5Error(f"{dataset} dataset is missing from SPIKE group {name}")
    data, index = group[DATA_DATASET_NAME], group[INDEX_DATASET_NAME]
    if data.ndim != 2 or data.dtype != np.int16:
        raise DH5Error(f"DATA in {name} must be a 2D int16 dataset, is {data.ndim}D {data.dtype}")
    if index.ndim != 1 or index.dtype != np.int64:
        raise DH5Error(f"INDEX in {name} must be a 1D int64 dataset, is {index.ndim}D {index.dtype}")
    samples = int(params["spikeSamples"])
    if samples < 0:
        raise DH5Error(f"SpikeParams.spikeSamples in {name} is negative ({samples})")
    if data.shape[0] != samples * index.shape[0]:
        raise DH5Error(
            f"DATA in {name} has {data.shape[0]} rows, expected spikeSamples * nSpikes = {samples * index.shape[0]}"
        )
    if CLUSTER_INFO_DATASET_NAME in group:
        clusters = group[CLUSTER_INFO_DATASET_NAME]
        if clusters.shape != index.shape or clusters.dtype != np.uint8:
            raise DH5Error(f"CLUSTER_INFO in {name} must be uint8 with one value per spike")
    if MEAN_WAVEFORM_DATASET_NAME in group:
        mean = group[MEAN_WAVEFORM_DATASET_NAME]
        if samples != 0 or mean.ndim != 2 or mean.dtype != np.int16 or mean.shape[1] != data.shape[1]:
            raise DH5Error(
                f"MEAN_WAVEFORM in {name} must be int16 (samples, nChannels) in a block with spikeSamples = 0"
            )
    calibration = group.attrs.get("Calibration")
    if calibration is not None and len(calibration) != data.shape[1]:
        raise DH5Error(f"Calibration in {name} has {len(calibration)} values for {data.shape[1]} channels")
    channels = group.attrs.get("Channels")
    if channels is not None and len(channels) != data.shape[1]:
        raise DH5Error(f"Channels in {name} has {len(channels)} entries for {data.shape[1]} channels")
