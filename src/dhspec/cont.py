from enum import Enum
import numpy as np
import numpy.typing as npt

# specification
CONT_PREFIX = "CONT"
DATA_DATASET_NAME = "DATA"
INDEX_DATASET_NAME = "INDEX"
CONT_DTYPE_NAME = "CONT_INDEX_ITEM"
INDEX_DTYPE = np.dtype([("time", np.int64), ("offset", np.int64)])


CalibrationType = npt.NDArray[np.float64]


class ContSignalType(Enum):
    LFP = "LFP"
    MUA = "MUA/ESA"
    ANALOG = "ANALOG"
    CSD = "CSD"


class ContBlockGroup(Enum):
    """Groups of CONT block ids used by the lab's converters (a convention, not part of the format)."""

    RAW = "RAW"  # raw neural data (wideband or AP band), one block per channel
    ANALOG = "ANALOG"  # raw analog inputs (eye position, mains reference, ...)
    LFP = "LFP"  # downsampled neural data
    ANALOG_DOWNSAMPLED = "ANALOG_DOWNSAMPLED"  # downsampled analog and derived behavioural signals
    MUA = "MUA/ESA"  # MUA / ESA envelope
    AP = "AP"  # high-pass filtered raw data (not for long-term storage)
    CSD = "CSD"  # current source density


# First and last CONT block id of each group: 1600 neural channels, followed by 400 analog
# channels for the raw and downsampled data. Channel k (0-based) of a group is CONT<first + k>.
CONT_BLOCK_RANGES: dict[ContBlockGroup, tuple[int, int]] = {
    ContBlockGroup.RAW: (1, 1600),
    ContBlockGroup.ANALOG: (1601, 2000),
    ContBlockGroup.LFP: (2001, 3600),
    ContBlockGroup.ANALOG_DOWNSAMPLED: (3601, 4000),
    ContBlockGroup.MUA: (4001, 5600),
    ContBlockGroup.AP: (6001, 7600),
    ContBlockGroup.CSD: (8001, 9600),
}


def cont_block_group(id: int) -> ContBlockGroup | None:
    """The group whose id range contains CONT block id ``id``, or None."""
    for group, (first, last) in CONT_BLOCK_RANGES.items():
        if first <= id <= last:
            return group
    return None


CHANNELS_DTYPE = np.dtype(
    [
        ("GlobalChanNumber", np.int16),
        ("BoardChanNo", np.int16),
        ("ADCBitWidth", np.int16),
        ("MaxVoltageRange", np.float32),
        ("MinVoltageRange", np.float32),
        ("AmplifChan0", np.float32),
    ]
)


def create_channel_info(
    GlobalChanNumber: int,
    BoardChanNo: int,
    ADCBitWidth: int,
    MaxVoltageRange: float,
    MinVoltageRange: float,
    AmplifChan0: float,
) -> np.recarray:
    return np.rec.array(
        (
            GlobalChanNumber,
            BoardChanNo,
            ADCBitWidth,
            MaxVoltageRange,
            MinVoltageRange,
            AmplifChan0,
        ),
        dtype=CHANNELS_DTYPE,
    )


def create_empty_index_array(n_index_items: int) -> np.ndarray:
    return np.zeros(n_index_items, dtype=INDEX_DTYPE)


def cont_name_from_id(id: int) -> str:
    return f"{CONT_PREFIX}{id}"


def cont_id_from_name(name: str) -> int:
    return int(name.lstrip("/").lstrip(CONT_PREFIX))


CHANNELS_DTYPE = np.dtype(
    [
        ("GlobalChanNumber", np.int16),
        ("BoardChanNo", np.int16),
        ("ADCBitWidth", np.int16),
        ("MaxVoltageRange", np.float32),
        ("MinVoltageRange", np.float32),
        ("AmplifChan0", np.float32),
    ]
)
