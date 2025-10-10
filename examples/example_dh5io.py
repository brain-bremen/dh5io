from dh5io.dh5file import DH5File
import pathlib

# use the existing file path relative to this script (based on __file__)
current_dir = pathlib.Path(__file__).parent
example_filename = current_dir / "example.dh5"


dh5 = DH5File(example_filename, "r")
# inspect file content
print(dh5)

# Will show the following...
#
# DAQ-HDF5 File (version 2) <path>\example.dh5 containing:
#     ├───CONT Groups (7):
#     │   ├─── CONT1
#     │   ├─── CONT60
#     │   ├─── CONT61
#     │   ├─── CONT62
#     │   ├─── CONT63
#     │   ├─── CONT64
#     │   └─── CONT1001
#     ├───SPIKE Groups (1):
#     │   └─── SPIKE0
#     ├─── 10460 Events
#     └─── 385 Trials in TRIALMAP

cont = dh5.get_cont_group_by_id(1)  # Get CONT group with id 1
print(cont)

# Will show the following...
#
# CONT1 in C:\Code\cog-neurophys-lab\dh5io\examples\example.dh5
#     ├─── id: 1
#     ├─── name:
#     ├─── comment:
#     ├─── sample_period: 1000000 ns (1000.0 Hz)
#     ├─── n_channels: 1
#     ├─── n_samples: 1443184
#     ├─── duration: 3021.76 s
#     ├─── n_regions: 385
#     ├─── signal_type: None
#     ├─── calibration: [1.0172526e-07]
#     ├─── data: (1443184, 1)
#     └─── index: (385,)

trialmap = dh5.get_trialmap()
print(trialmap)
