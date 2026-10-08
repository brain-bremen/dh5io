import numpy as np

from dhspec.intervals import INTERVAL_DATASET_DTYPE


def test_interval_dtype_matches_specification():
    # Member names, offsets and types as in the specification and the original C++ library
    assert INTERVAL_DATASET_DTYPE.names == ("StartTime", "EndTime")
    assert INTERVAL_DATASET_DTYPE.fields["StartTime"] == (np.dtype(np.int64), 0)
    assert INTERVAL_DATASET_DTYPE.fields["EndTime"] == (np.dtype(np.int64), 8)
    assert INTERVAL_DATASET_DTYPE.itemsize == 16
