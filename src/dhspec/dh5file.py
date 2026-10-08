"""
DAQ-HDF (DH5) is a set of specifications on how to store electrophysiological
data in files based on the HDF5 file format.

The following kinds of data can be stored in DAQ-HDF files:

-   signal data (CONT groups)
-   spike data (SPIKE groups)
-   trialmap (TRIALMAP dataset)
-   time markers (Markers group) and intervals (Intervals group)
-   processing history (Operations groups)

There are additional two kinds of data which are specified to
accommodate the respective streams from DAQ-files and other similar file
formats such as UFF:

-   event triggers (EV02 dataset)
-   trial descriptor records (TD01 dataset)

These sets of data are only used for subsequent generation of trialmap,
time markers and intervals based on the information from them.

DAQ-HDF has the following **attributes** associated with the root group:

- `FILEVERSION` (`int32` scalar) – version of the DAQ-HDF. The current version number is 3.
Versions 2 and 3 have the same structure and differ only in how strings are stored (see
*Strings* below). If this attribute is missing, version 1 is assumed. Version 1 is obsolete,
and it has substantial differences in data structures compared to version 2.
- `BOARDS` (`string` array) – names of the A/D boards used during recording of data. If
initial data was acquired by means other than analog recording, for example, if it was
generated in software, this attribute may contain some description of the creation process
instead.

The root group must also contain a shared *datatype* named `CONT_INDEX_ITEM`
if there are `CONT` blocks present in the file. See description of this
datatype in the `CONT` blocks description.

### Strings

All string attributes (for example `BOARDS`, the `Name` and `Comment` of blocks
and the attributes of the processing history) are stored as follows:

- Version 3: variable-length strings with UTF-8 character set. This is the
  default representation of h5py and other modern HDF5 libraries.
- Version 2: fixed-length strings with ASCII character set, terminated or
  padded with null bytes. This is the representation of the original
  dhfun.dll and its MEX files, which cannot read variable-length strings.
  Readers should accept null-terminated as well as null-padded strings, and
  null-terminated strings whose stored size does not include the terminator.

Readers should accept both representations in files of either version.

"""

import numpy as np

FILEVERSION_ATTRIBUTE_NAME = "FILEVERSION"
FILEVERSION_ATTRIBUTE_DTYPE = np.int32
# Version written by default. Versions 2 and 3 differ only in the string representation.
CURRENT_FILEVERSION = 3
FIRST_FILEVERSION_WITH_UTF8_STRINGS = 3
BOARDS_ATTRIBUTE_NAME = "BOARDS"
BOARDS_ATTRIBUTE_DTYPE = np.int32
