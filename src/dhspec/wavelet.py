"""Wavelet data in WAVELET blocks in DAQ-HDF files.

WAVELET blocks store time-frequency analysis data, typically obtained through
wavelet transforms of continuous signals. Each block represents a multi-channel
nTrode with time-frequency decomposition data.

WAVELET blocks are stored in groups named `WAVELETn`, where n can have values
between 0 and 65535. `CONT`, `SPIKE`, and `WAVELET` blocks can share the same
identifier numbers.

`WAVELETn` group **must** have the following **attributes**:

- `SamplePeriod` (`int32` scalar) - specified in nanoseconds. It's the time
  interval between two consecutive time samples in the wavelet analysis.

- `FrequencyAxis` (`double` array[F]) - specifies the frequency values (in Hz)
  corresponding to each frequency bin in the wavelet analysis. There are F
  frequency bins, where F is the first dimension of the DATA dataset.

`WAVELETn` group **must** have the following **datasets**:

- `DATA` (`struct` array[N,M,F]):

  | Offset | Name | Type     |
  | ------ | ---- | -------- |
  | 0      | a    | `uint16` |
  | 2      | phi  | `int8`   |

- `INDEX` (`struct` array[R]):

  | Offset | Name    | Type     |
  | ------ | ------- | -------- |
  | 0      | time    | `int64`  |
  | 8      | offset  | `int64`  |
  | 16     | scaling | `double` |

Here, N is the number of channels in the nTrode; M is the total number of time
samples stored; F is the number of frequency bins; R is the number of
recording regions.

Description of the **attributes**:

- `SamplePeriod` plays the same role as in `CONT` and `SPIKE` blocks,
  specifying the time resolution of the wavelet analysis in nanoseconds.

- `FrequencyAxis` provides the frequency values for each frequency bin. This
  allows reconstruction of the frequency domain of the wavelet transform. The
  array has F elements, one for each frequency bin.

Description of the **datasets**:

- `DATA` stores the wavelet transform results as a 3-dimensional array with
  shape [N,M,F] (channels, time samples, frequencies) with compound structure.
  For each time-frequency point and channel, two values are stored:
  - `a` (amplitude/magnitude) is stored as unsigned 16-bit integers with
    values from 0 to 65535. These are scaled values that must be converted to
    floating-point representation using scaling factors from the INDEX dataset.
  - `phi` (phase) is stored as signed 8-bit integers with values from -127 to
  127. To convert to radians, use the formula: `phi_rad = phi * pi / 127.0`.

- `INDEX` characterizes each recording region with three values:
  - `time` is the timestamp of the first time sample in nanoseconds.
  - `offset` specifies the sample offset within the `DATA` dataset where the
    first sample of a particular region is stored (1-based indexing).
  - `scaling` is a floating-point scaling factor used to restore the actual
    magnitude values. To obtain the true magnitude value for samples in a given
    region, multiply the raw `a` values by the corresponding `scaling` factor.

The INDEX system for `WAVELET` blocks is similar to that of `CONT` blocks,
allowing for piecewise-continuous recording with gaps. All samples within a
contiguous recording region share the same scaling factor. To restore the
value of an arbitrary sample, one must first determine which region it belongs
to, fetch that region's scaling value, and multiply it by the sample's raw
magnitude value.
"""

import numpy as np
import numpy.typing as npt

# specification
WAVELET_PREFIX = "WAVELET"
DATA_DATASET_NAME = "DATA"
INDEX_DATASET_NAME = "INDEX"
WAVELET_DTYPE_NAME = "WAVELET_INDEX_ITEM"

# Data type for wavelet DATA: amplitude (a) and phase (phi)
DATA_DTYPE = np.dtype([("a", np.uint16), ("phi", np.int8)])

# Index type for wavelet INDEX: time, offset, and scaling
INDEX_DTYPE = np.dtype(
    [("time", np.int64), ("offset", np.int64), ("scaling", np.float64)]
)

FrequencyAxisType = npt.NDArray[np.float64]


def create_empty_index_array(n_index_items: int) -> np.ndarray:
    """Create an empty index array for WAVELET blocks.

    Args:
        n_index_items: Number of recording regions

    Returns:
        Empty numpy array with WAVELET INDEX_DTYPE
    """
    return np.zeros(n_index_items, dtype=INDEX_DTYPE)


def wavelet_name_from_id(id: int) -> str:
    """Convert wavelet ID to group name.

    Args:
        id: Wavelet block identifier (0-65535)

    Returns:
        Group name in format "WAVELETn"
    """
    return f"{WAVELET_PREFIX}{id}"


def wavelet_id_from_name(name: str) -> int:
    """Extract wavelet ID from group name.

    Args:
        name: Group name in format "WAVELETn" or "/WAVELETn"

    Returns:
        Wavelet block identifier
    """
    return int(name.lstrip("/").lstrip(WAVELET_PREFIX))


def phase_to_radians(phi: np.ndarray) -> np.ndarray:
    """Convert phase values from int8 storage format to radians.

    Args:
        phi: Phase values as int8 array (range -127 to 127)

    Returns:
        Phase values in radians
    """
    return phi * np.pi / 127.0


def radians_to_phase(phi_rad: np.ndarray) -> np.ndarray:
    """Convert phase values from radians to int8 storage format.

    Args:
        phi_rad: Phase values in radians

    Returns:
        Phase values as int8 array (range -127 to 127)
    """
    return np.clip(np.round(phi_rad * 127.0 / np.pi), -127, 127).astype(np.int8)


def amplitude_to_float(a: np.ndarray, scaling: float) -> np.ndarray:
    """Convert amplitude values from uint16 storage format to float.

    Args:
        a: Amplitude values as uint16 array (range 0-65535)
        scaling: Scaling factor from INDEX dataset

    Returns:
        Amplitude values as float array
    """
    return a.astype(np.float64) * scaling


def float_to_amplitude(a_float: np.ndarray, scaling: float) -> np.ndarray:
    """Convert amplitude values from float to uint16 storage format.

    Args:
        a_float: Amplitude values as float array
        scaling: Scaling factor to use for conversion

    Returns:
        Amplitude values as uint16 array (range 0-65535)
    """
    return np.clip(np.round(a_float / scaling), 0, 65535).astype(np.uint16)
