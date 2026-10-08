import logging
import os.path
import pathlib

import h5py
import h5py.h5t as h5t
import numpy

from dh5io.dh5file import DH5File
from dh5io.hdf5_strings import str_array_attr_value
from dh5io.operations import add_operation_to_file
from dh5io.validation import ValidationOptions, validate_dh5_file
from dhspec.dh5file import CURRENT_FILEVERSION

logger = logging.getLogger(__name__)


def create_dh_file(
    filename: str | pathlib.Path,
    overwrite=False,
    file_version: int = CURRENT_FILEVERSION,
    validate: bool = True,
    boards: list[str] = [],
) -> DH5File:
    """Create a new DH5 file.

    Files of version 3 (the default) store strings as variable-length UTF-8.
    Use file_version=2 for fixed-length ASCII strings, which the original
    dhfun.dll and its MEX files require.
    """
    if not overwrite and os.path.exists(filename):
        raise FileExistsError(f"File {filename} already exists.")

    # create new empty HDF5 file so DH5File (which requires an existing file) can open it
    h5py.File(filename, mode="w").close()

    dh5File = DH5File(filename, mode="r+")
    h5file = dh5File._file
    h5file.attrs["FILEVERSION"] = file_version

    # after FILEVERSION, which selects the string representation
    h5file.attrs["BOARDS"] = str_array_attr_value(h5file, boards)

    tid = h5t.py_create(numpy.dtype([("time", numpy.int64), ("offset", numpy.int64)]))
    tid.commit(h5file.id, b"CONT_INDEX_ITEM")

    add_operation_to_file(h5file, "create_file", tool="dh5io", id=0)

    if validate:
        validate_dh5_file(
            h5file,
            options=ValidationOptions(
                validate_trialmap=False,
                validate_cont=False,
                validate_event_triggers=False,
            ),
        )

    return dh5File
