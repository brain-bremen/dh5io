# DAQ-HDF file format (dh5)

Document revision 3.1 from 03.02.2026

## Abstract

DAQ-HDF is a set of specifications on how to store electrophysiological
data in files based on the HDF5 file format [1].

HDF5 is itself a file format for storage of generic scientific data.
HDF5 offers great flexibility in choosing a particular way to organize
the information in a file. In this sense, DAQ-HDF restricts this
flexibility and specifies file organization for the case of
electrophysiological data.

- [Design goals and history](#design-goals-and-history)
- [HDF-5 file format in brief](#hdf-5-file-format-in-brief)
- [DAQ-HDF concepts](#daq-hdf-concepts)
  - [Signal data](#signal-data)
  - [Spike data](#spike-data)
  - [Trialmap](#trialmap)
  - [Time markers and intervals](#time-markers-and-intervals)
  - [Processing history](#processing-history)
- [DAQ-HDF specifications](#daq-hdf-specifications)
  - [Base file](#base-file)
  - [`CONT` blocks](#cont-blocks)
  - [`SPIKE` blocks](#spike-blocks)
  - [Trialmap](#trialmap-1)
  - [Time markers](#time-markers)
  - [Time intervals](#time-intervals)
  - [Processing history](#processing-history-1)
  - [Event triggers and Trial descriptor records](#event-triggers-and-trial-descriptor-records)
- [References](#references)

## Design goals and history

DAQ-HDF format was designed to have the following desirable properties:

- Electrophysiological data related to a single experiment or
  recording session is stored in a single file
- Each data stream within a file is stored in a contiguous way, so
  that both random and sequential access to parts of the data streams
  is efficient
- Determining of data size does not require to perform complex or
  time-consuming operations
- Cross-platform conversion of data types, such as different formats
  of floating-point or integer numbers, is possible in an automated
  and transparent way
- Data structures are organized in an intuitive manner which reduces
  the probability of errors and facilitates diagnostics
- It is possible to save the history of data processing which further
  facilitates diagnostics and makes it easy to determine the condition
  of a data file
- DAQ-HDF format is easily extensible which makes it possible to add
  some attribute information to the data streams, as well as parallel
  streams and additional streams, which may be produced during the
  data processing.

Some of the mentioned properties are inherent for the underlying HDF5
file format, others are provided by the DAQ-HDF specifications.

DAQ-HDF was initially developed as a format to convert into it
electrophysiological recording data files (DAQ-files) produced by Data
Acquisition Program (DAP) by Andreas Kreiter, while transferring all the
information from DAQ-files and allowing for storage of intermediate data
processing results as well as extensibility.

Data Acquisition Program deals with several streams of data produced at
different rates, acquired in realtime, and the final size of data is
unknown to the program at the time of recording. DAP writes its streams
in a single recording file, in an interleaved fashion, which is one of a
few ways to accomplish this task. Single interleaved file saves the
recording PC's computational resources, it does not require specific
hardware and does not create file fragmentation on the disk. Interleaved
files are also resource-saving when there is a need to play the data
back in real time. Successful examples are AVI, MPEG and other file
formats used to store video and audio streams.

This is an example how two streams are stored in one file, interleaved:

|           |         |         |         |         |         |         |         |
| --------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| File data |         |         |         |         |         |         |         |
| stream1   | stream2 | stream1 | stream2 | stream1 | stream2 | stream1 | stream2 |

It is not the playback what is usually done with electrophysiological
data, however. Data processing algorithms usually require that the total
size of the data in each stream is known, and that random access to the
data in each stream is possible. All this is possible with interleaved
files, however it is to a large extent inefficient in terms of algorithm
complexity, computational and data-throughput resources, and the time
required for data processing. To address the mentioned two needs:
predictability of the data size and possibility of random access,
interleaved data files must be converted.

Underlying HDF5 file format was chosen as a highly elaborated scientific
data format which has an associated software library, continuous support
and ongoing development by the NCSA.

Initial DAQ-HDF specification was just to store the same information as
contained in DAQ files. This specification was extended during the
development of data processing software. As time went on, the Data
Acquisition Program went under substantial extension of capabilities,
and there were needs to import data from other data acquisition systems
into DAQ-HDF format. This all led to a new generation of DAQ-HDF files
and further refinement of specifications and support libraries to
address the upcoming needs. This document describes the current DAQ-HDF
specifications which are likely to extend in the future while
maintaining backwards compatibility when possible.

## HDF-5 file format in brief

HDF5 file format is, crudely speaking, a file system. A file system
within a file. There are substantial differences from a typical file
system, however.

Subdirectory – in HDF5 format it is called a Group. Directory tree in a
file system and group tree in HDF5 files have the same meaning and serve
the same purpose – to provide means for hiearchically arranging the
units of data.

File is analogous to a Dataset within HDF5, and it serves as a named
unit of data. Each subdirectory in a file system and each group in HDF5
files can contain files, or datasets, respectively. Dataset has a more
strict specification of contents than a file. Dataset can be
multidimensional, and each individual data items have associated type,
be it an integer, floating point number or a compound datatype.

Attributes in HDF5 files have a broader meaning than in file systems. In
HDF5, attribute is basically the same as dataset, however it has
different purpose. Rather than storing data, attributes are typically
small compared to datasets, and are used to characterize data. Each
group, dataset and other objects within HDF5 files can have associated
attributes. For the attributes there are also strict specifications of
data types. Unlike file systems, attributes are not used in HDF5 files
to control access to other objects. For example there is no 'read-only'
attribute in HDF5 files.

There are links which make it possible to acces the same dataset from
different group locations within a HDF5 file. The same is possible for
modern file systems.

Data types for datasets and attributes in HDF5 files include, but are
not limited by, the following:

- integers, with various size, with or without sign, little-endian or
  big-endian (conversion, if required, is performed by the NCSA HDF5
  library on the fly);
- floating point numbers, with various size, endianness and format
  (conversion is done by the NCSA HDF library);
- strings of characters;
- enumeration data type which is an integer, each value of which has
  an associated name;
- compound data type, analogous to structures in programming languages
  like C, Matlab; this is a named combination of values of any other
  possible datatype.

There are software tools from NCSA to perform generic operations with
HDF5 files in a comfortable way, such as creating groups, datasets and
attributes, renaming, copying and moving them, and viewing them. Thanks
that the data type is always known, so generic viewing program will
almost always display meaningful result for a dataset of arbitrary type,
without any additional information specific for a particular application
of the HDF5 format.

There are also specifications from NCSA about the structure of HDF5
files to store images, sounds and other common higher-level kinds of
data.

## DAQ-HDF concepts

The following kinds of data can be stored in DAQ-HDF files:

- signal data
- spike data
- wavelet data
- trialmap
- time markers and intervals
- processing history

There are additional two kinds of data which are specified to
accommodate the respective streams from DAQ-files and other similar file
formats such as UFF:

- event triggers
- trial descriptor records

These sets of data are only used for subsequent generation of trialmap,
time markers and intervals based on the information from them.

### Signal data

Signal data for DAQ-HDF means continuously or piecewise-continuously
(trial-based) recorded continuous-time signal. Sampling is supposed to
be equidistant. It is possible to store multiple signals with different
sampling rates and different regions of recording. Signal data is
represented in DAQ-HDF files in the form of `CONT` blocks (from the
continuous-time signal concept).

Each `CONT` block stores data for a single nTrode which is a multi-channel
electrode. Therefore, a `CONT` block can contain multiple channels of
piecewise-continuous signal recording. All channels within a `CONT` block
share the same sampling rate and the same regions (pieces) of recording.
Different `CONT` blocks, however, can have both of the mentioned
parameters independent of each other. Each `CONT` block has an unique
identifying number, and apart from a range limit, there are no other
restrictions which Ids to assign to them. In contrast, channels within a
single `CONT` block are numbered from 0 to N-1, where N is the number of
channels. Gaps in numbering are not possible.

In general, `CONT` blocks are thought of to be separable units of data,
whereas the channels within a single `CONT` block are supposed be stored
and processed together.

### Spike data

Brain signals recorded from selective electrodes contain spikes. Most
analysis algorithms for such signals are interested in spikes only, not
the signal waveform between them. During the recording or after the
recording, in an off-line processing, spikes are detected and extracted
from continuous-time signals. For each spike, a piece of signal waveform
that contains the spike, is stored as well as a timestamp. Spikes are
usually sorted in the first stages of data processing where, depending
on their waveform, each of them is assigned to one of several spike
clusters. It is possible to save this clustering information in DAQ-HDF
files, too. Spike data is represented in DAQ-HDF files in the form of
`SPIKE` blocks.

`SPIKE` blocks also use the concept of nTrodes, like the `CONT` blocks.
Multichannel data is possible within a single `SPIKE` block, however, all
the channels have the same sampling rate, the same time windows for
stored waveforms, and the same spike timestamps.

### Wavelet data

Wavelet data represents time-frequency decomposition of neural signals,
typically obtained by applying wavelet transforms to continuous signal data.
This analysis method provides simultaneous information about the temporal and
spectral characteristics of the signal, which is particularly useful for
studying oscillatory neural activity and transient spectral features.

Wavelet data is stored as complex values (magnitude and phase) across three
dimensions: frequency bins, time samples, and channels. The frequency axis
specifies which frequencies were analyzed, while the time samples maintain the
temporal resolution of the original signal (though often downsampled). Wavelet
data is represented in DAQ-HDF files in the form of `WAVELET` blocks.

Like `CONT` and `SPIKE` blocks, `WAVELET` blocks use the nTrode concept for
multi-channel electrodes. All channels within a `WAVELET` block share the same
sampling rate, the same frequency axis, and the same recording regions.
Different `WAVELET` blocks can have independent frequency axes and sampling
rates.

### Trialmap

Electrophysiological recording sessions typically consist of trials. In
different trials, experimental conditions repeat or have other important
similarities. Trialmap information in DAQ-HDF files characterizes each
trial and makes it possible to determine which parts of signal or spike
data corresponds to a particular trial.

The information contained in DAQ-HDF trialmap is:

- Trial numbers as generated by the stimulation PC;
- Stimulus numbers (encoded type of stimulation);
- Outcome number (encoded behavioral data, such as successful or
  unsuccessful performance of the experiment subject in each trial)
- Timestamps for the start and the end of each trial;

### Time markers and intervals

These are named time points and intervals in time which represent some
experiment-dependent timing information. For example, in each trial
there may be a particular time point which is located at a variable time
offset from the beginning of trial. Markers are used to describe these
time points.

A marker set in a DAQ-HDF file has a symbolic name and associated list
of timestamps when the named event had occurred.

Each Interval set in a DAQ-HDF file also has a symbolic name and
associated list of beginning and ending time points of the named
interval occurences.

### Processing history

During the analysis of data, contents of a DAQ-HDF file may change. For
example, user can apply a filter to the signal data. For each such
operation, a history record is added to a DAQ-HDF file, in order to let
the user to trace these operations back in time and check a file's
condition when necessary.

For each operation, typically, the program name and version number is
written, as well as the operator's name, and other parameters relevant
to a specific operation.

## DAQ-HDF specifications

The reader should become familiar with the concepts and definitions
relevant to HDF5-files before reading this section.

### Base file

DAQ-HDF has the following **attributes** associated with the root group:

- `FILEVERSION` (`int32` scalar) – version of the DAQ-HDF. The current version number is 2,
  and this is the only version described in this document. If this attribute is missing,
  version 1 is assumed. Version 1 is obsolete, and it has substantial differences in data
  structures compared to version 2.
- `BOARDS` (`string` array) – names of the A/D boards used during recording of data. If
  initial data was acquired by means other than analog recording, for example, if it was
  generated in software, this attribute may contain some description of the creation process
  instead.

The root group must also contain a shared _datatype_ named `CONT_INDEX_ITEM`
if there are `CONT` blocks present in the file. See description of this
datatype in the `CONT` blocks description.

### `CONT` blocks

There can be several `CONT` blocks in a DAQ-HDF file. Each of them is
stored in a group named `CONTn`, where n is the identifier number of each
`CONT` block. This identifier must be in the range from 0 to 65535.

`CONTn` group **must** have the following **attributes**:

- `Channels` (`struct` array\[N\]):

  | Offset | Name             | Type    |
  | ------ | ---------------- | ------- |
  | 0      | GlobalChanNumber | `int16` |
  | 2      | BoardChanNo      | `int16` |
  | 4      | ADCBitWidth      | `int16` |
  | 6      | MaxVoltageRange  | `float` |
  | 10     | MinVoltageRange  | `float` |
  | 14     | AmplifChan0      | `float` |

- `SamplePeriod` (`int32` scalar).

_Optional attribute_:

- `Calibration` (`double` array\[N\])

`CONTn` group **must** have the following **datasets**:

- `DATA` (`int16` array\[M,N\])

- `INDEX` (`struct` array\[R\]):

  | Offset | Name   | Type    |
  | ------ | ------ | ------- |
  | 0      | time   | `int64` |
  | 8      | offset | `int64` |

Here, N is number of channels in nTrode; M is the total number of
samples stored for every channel in the `CONT` block; R is the number of
recording regions.

Shared HDF5 datatype `/CONT_INDEX_ITEM` is used in each `CONT` block to
describe the INDEX dataset.

Description of the **attributes**:

Signal data may be recorded from multiple A/D boards within a single PC.
Data Acquisition Program enumerates all available A/D channels from all
A/D boards present, so that each channel gets an unique number at the
time of recording. This is stored in the `GlobalChanNumber` member of the
structure. A/D channels which compose an nTrode may have very different
numbers, they may also belong to different A/D boards in the recording
setup. This information is normally not needed during the data
processing, but may be needed for documentation of the experiment.

- `BoardChanNo` – this is the number of channel within the A/D board from
  which it was acquired.

- `ADCBitWidth` – number of bits in the A/D converter. Note, however, that
  the signals are always stored in 16-bit format regardless of the value
  of this parameter.

- `MaxVoltageRange`, `MinVoltageRange` – these two values specify the A/D
  converter's input voltage range. Knowing them, it is possible to convert
  the unitless signal data into volts.

- `AmplifChan0` – If an A/D board has some programmable-gain amplifier
  (PGA), this value specifies amplification gain for each recording
  channel. If this value is zero, then there is no PGA on the board.

- `SamplePeriod` is specified in nanoseconds. It's the time interval between
  two consecutive samples of the signal.

- `Calibration` attribute stores a real number for every channel belonging
  to the nTrode. If you multiply this calibration value with the raw
  channel data, you get value in volts. `Calibration` attribute is normally
  not present in a freshly recorded and converted file, because there is
  not enough information to produce the calibration value. It must be
  obtained from other source of information, typically these are
  special-purpose calibration recording files.

`Calibration` value is supposed to encapsulate all the gains throughout
the whole amplification/recording chain. By multiplying calibration
value with channel data it should be possible, therefore, to get the
very initial voltage as it was on the electrode tip.

Description of the **datasets**:

- `DATA` dataset stores the signal samples as a single 2-dimensional block in the form of
  16-bit integers, whose minimum value is -32768, and the maximum value is 32767. Contiguous
  pieces of recording are merged together. It is possible to determine where these pieces
  (regions) are located by using information from the INDEX dataset.

- `INDEX` structure dataset characterizes each recording region with two numbers: 'time' is
  the timestamp of the first signal sample, in nanoseconds, and 'offset' member specifies the
  sample offset within the `DATA` dataset where is the first sample of a particular region
  stored.

Knowing these two values for each recording region, and knowing the
total number of samples, it is possible to calculate the following
information: offsets of starting and ending sample for each recording
region, and their respective time stamps.

### `SPIKE` blocks

`SPIKE` blocks are stored in groups named `SPIKEn`, where n can have values
between 0 and 65535. `CONT` and `SPIKE` blocks can have the same
identifiers.

`SPIKEn` group **must** have the following **attributes**:

- `SpikeParams` (`struct` scalar):

  | Offset | Name           | Type    |
  | ------ | -------------- | ------- |
  | 0      | spikeSamples   | `int16` |
  | 2      | preTrigSamples | `int16` |
  | 4      | lockOutSamples | `int16` |

- `Channels` (`struct` array\[N\]), with the same members and their meaning as
  the 'Channels' attribute for `CONTn` groups;

- `SamplePeriod` (`int32` scalar).

_Optional attribute_:

- `Calibration` (`double` array\[N\])

`SPIKEn` group **must** have the following **datasets**:

- `DATA` (`int16` array\[M,N\]);
- `INDEX` (`int64` array\[S\]);

_Optional dataset_:

- `CLUSTER_INFO` (unsigned `int8` array\[S\]).

Here, `N` is number of channels in nTrode; `M` is the total number of
samples stored for every channel in the `SPIKE` block; S is the total
number of spikes.

Description of the **attributes**:

- `Channels`, `SamplePeriod` and `Calibration` attributes play the same
  role here as in the `CONT` blocks.

- `SpikeParams` describes some spike parameters common for all the
  channels within this nTrode. `spikeSamples` member tells how many
  samples of the signal waveform are stored in total for each spike;
  `PreTrigSamples` member specifies how many samples of the signal
  waveform are stored prior to the spike trigger point. `lockOutSamples`
  is a parameter which was used for detection of spikes and tells the
  minimum number of samples between the trigger points of two consecutive
  spikes.

Description of the **datasets**:

- `DATA` stores all spike waveforms, merged together. Therefore the total
  number of samples M is equal to the product of SpikeParams.spikeSamples
  and the total number of spikes S. Waveforms are stored in 16-bit signed
  format, same as with `CONT` blocks.

- `INDEX` stores spike timestamps, specified in nanoseconds. There are S
  timestamps, one for each spike, and it tells the time of the spike
  trigger point which is not the beginning of a particular waveform if
  spikeParams.PreTrigSamples is nonzero.

Because the length of all spike waveforms is the same, it is simple to
extract waveform for a particular spike: sample offset is calculated by
multiplying the spike number with the `spikeParams.spikeSamples`
parameter.

If there are multiple channels in the nTrode, spike trigger points are
common for all these channels, as well as other parameters except the
waveforms themselves.

- `CLUSTER_INFO` dataset is created during the spike sorting process. Each
  spike is assigned a cluster number, so the `CLUSTER_INFO` dataset simply
  stores these numbers for every spike. There can be up to 256 clusters
  for every `SPIKE` block, which is far more than enough, since a typical
  spike sorting process creates 2 to 4 clusters.

### `WAVELET` blocks

`WAVELET` blocks store time-frequency analysis data, typically obtained through
wavelet transforms of continuous signals. Each block represents a multi-channel
nTrode with time-frequency decomposition data. `WAVELET` blocks are stored in
groups named `WAVELETn`, where n can have values between 0 and 65535. `CONT`,
`SPIKE`, and `WAVELET` blocks can share the same identifier numbers.

`WAVELETn` group **must** have the following **attributes**:

- `SamplePeriod` (`int32` scalar) - specified in nanoseconds. It's the time
  interval between two consecutive time samples in the wavelet analysis.

- `FrequencyAxis` (`double` array\[F\]) - specifies the frequency values (in Hz)
  corresponding to each frequency bin in the wavelet analysis. There are F
  frequency bins, where F is the first dimension of the DATA dataset.

`WAVELETn` group **must** have the following **datasets**:

- `DATA` (`struct` array\[N,M,F\]):

  | Offset | Name | Type     |
  | ------ | ---- | -------- |
  | 0      | a    | `uint16` |
  | 2      | phi  | `int8`   |

- `INDEX` (`struct` array\[R\]):

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
  shape \[N,M,F\] (channels, time samples, frequencies) with compound structure.
  For each time-frequency point and channel, two values are stored:
  - `a` (amplitude/magnitude) is stored as unsigned 16-bit integers with
    values from 0 to 65535. These are scaled values that must be converted to
    floating-point representation using scaling factors from the INDEX dataset.
  - `phi` (phase) is stored as signed 8-bit integers with values from -127 to
  127. To convert to radians, use the formula: `phi_rad = phi * pi / 127.0`.

- `INDEX` characterizes each recording region with three values:
  - `time` is the timestamp of the first time sample in nanoseconds.
  - `offset` specifies the sample offset within the `DATA` dataset where the
    first sample of a particular region is stored (0-based indexing).
  - `scaling` is a floating-point scaling factor used to restore the actual
    magnitude values. To obtain the true magnitude value for samples in a given
    region, multiply the raw `a` values by the corresponding `scaling` factor.

The INDEX system for `WAVELET` blocks is similar to that of `CONT` blocks,
allowing for piecewise-continuous recording with gaps. All samples within a
contiguous recording region share the same scaling factor. To restore the
value of an arbitrary sample, one must first determine which region it belongs
to, fetch that region's scaling value, and multiply it by the sample's raw
magnitude value.

### Trialmap

Trialmap is a dataset in the root group of a DAQ-HDF file:

`TRIALMAP` (`struct` array\[T\]):

| Offset | Name      | Type    |
| ------ | --------- | ------- |
| 0      | TrialNo   | `int32` |
| 4      | StimNo    | `int32` |
| 8      | Outcome   | `int32` |
| 12     | StartTime | `int64` |
| 20     | EndTime   | `int64` |

Here, `T` is the total number of trials in the file.

- **TrialNo** is a sequence number generated by the stimulation program and
  then transferred to the recording program and stored in the data
  acquisition file. This number can be used to combine trialmap
  information with whatever other information about the trials from other
  sources of data than the DAQ-HDF file itself. If DAQ-HDF file and the
  trialmap were created in such a way that the above mentioned
  considerations are not applicable, this structure member may be ignored
  altogether or, better, filled with a sequence of ascending numbers.

- **StimNo** is so-called Stimulus Number. Trials which have the same stimulus
  numbers can be usually grouped together for analysis. So, StimNo
  contains some encoded information about the type of trial and, possibly,
  some other conditions.

- **Outcome** – behavioral data. Outcome code specifies the type of behavior
  observed and discriminated from the experimental subject. Typically,
  Outcome member specifies whether the subject performed his task during a
  trial successfully or not, and if not, what particular kind of error was
  made by him.

- **StartTime** and **EndTime** are timestamps, in nanoseconds, for the beginning
  and ending of each trial. All timestamps throughout a DAQ-HDF file have
  the same base value, so timestamps from `CONT` and `SPIKE` blocks as well as
  the `TRIALMAP`, are comparable with each other. It is typically needed,
  based on the timestamps from the TRIALMAP, to determine location of the
  corresponding piece of signal within `CONT` or `SPIKE` blocks.

### Time markers

Markers are used to describe some important points in time during the
recording session, other than trial starts and ends.

Each marker has a symbolic name and a set of times of occurrence.

In DAQ-HDF files, markers are stored under the '/Markers' group. This
group can contain multiple datasets. Each of these datasets is named
after the marker symbolic name, and stores a one-dimensional array of
64-bit integers. These are timestamps in nanoseconds.

If no markers are specified for a DAQ-HDF file, the '/Markers' group may
be absent altogether.

### Time intervals

Intervals are similar to markers, however they describe not single
points in time, but ranges in time, or intervals.

Each interval has a symbolic name and a set of time ranges of its
occurrence.

In DAQ-HDF files, intervals are stored under the `/Intervals` group.
This group can contain multiple datasets. Each of these datasets is
named after the interval's symbolic name, and stores a one-dimensional
array of structures:

| Offset | Name      | Type    |
| ------ | --------- | ------- |
| 0      | StartTime | `int64` |
| 8      | EndTime   | `int64` |

StartTime and EndTime are specified in nanoseconds, and they tell the
starting and ending points in time for each occurrence of a given
interval.

Composition of this structure is stored as a shared HDF5 datatype in the
`/Intervals` group under the name `INTERVAL`. So, the `/Intervals` group
contains datasets as well as one shared datatype.

If no intervals are specified for a DAQ-HDF file, the `/Intervals` group
can be absent as well as the `INTERVAL` shared datatype.

### Processing history

Processing history is stored in DAQ-HDF files under the group named
`/Operations`. Usually processing history contains at least one entry
which describes how a file was created. Software tools which change
DAQ-HDF files should add history entries.

Each history entry is stored in a subgroup in the `/Operations` group.
The name of this subgroup should be given as follows:

```
nnn_OperationName
```

Where `nnn` is a 3-digit number with leading zeros if this number is less
than 100. Numeration starts from 0. When a new history entry is added,
it gets a number which equals the number of the last existing history
entry plus one.

Numeration is necessary to preserve the order of history entries,
because group and dataset names in HDF5-files are automatically sorted.

Any information about a particular operation performed on the file is
stored as attributes of this operation's subgroup. At the moment,
datasets are not allowed here.

There are no strict definitions about what information should be written
about each operation. However, typically the following attributes are
written at all cases:

- **Tool** (`string` scalar) – Title and version of the software tool used to
  perform named operation;

- **Operator name** (`string` scalar) – Name of the person who initiated and
  controlled the named operation; preferably full name;

- **Date** (`struct` scalar) – Date and time when this operation was
  performed:

      | Offset | Name | Type |
      |-----|--------|-------|
      | 0   | Year   | `int16` |
      | 2   | Month  | `int8`  |
      | 3   | Day    | `int8`  |
      | 4   | Hour   | `int8`  |
      | 5   | Minute | `int8`  |
      | 6   | Second | `int8`  |

- **Original file name** (`string` scalar) – If the processing involved
  creation of a new DAQ-HDF file and copying some of the initial file's
  contents, instead of just modifying the old file, this attribute is used
  to specify the name of the initial file. For example, when a DAQ-file is
  converted into a DAQ-HDF file, original DAQ-filename is written here.
  Preferably, original file name should be specified exactly as provided
  to the processing tool, which means no truncation of the file path.

It is recommended that data processing tools always write all important
parameters provided to these tools from their operators, unless storing
such information would dramatically increase the size of a DAQ-HDF file.

If a reversible modification was performed on a DAQ-HDF file and then it
was undone, history entries should be neither deleted nor modified. An
undo entry must be added to the processing history instead.

### Event triggers and Trial descriptor records

Event triggers are a low-level piece of information. They represent the
stream of event triggers in DAQ files, along with the trial descriptor
records. Based on the information from event triggers and trial
descriptor records only, it is not possible to reconstruct a trialmap.
But combined with additional information from the user, these two
datasets are used at the early stages of data processing to produce
information which is then written into the Trialmap, Markers and
Intervals. Event triggers and trial descriptor records can also be used
as some intermediate information storage facility during conversion from
other file formats into DAQ-HDF.

Event triggers and trial-descriptor records, imported from DAQ or other
file formats, are likely to contain some sequence errors which increase
the complexity of production of Trialmap, Markers and Intervals. A
conversion program should perform thorough checks of event trigger
stream before saving the results. It is supposed that the information
stored in Trialmap, Markers and Intervals is consistent and has been
checked, so that other analysis software would not need to perform it
over again.

Event triggers are stored in DAQ-HDF files in the form of a dataset
named `EV02` in the root group. `EV02` is an array of structures:

| Offset | Name  | Type    |
| ------ | ----- | ------- |
| 0      | time  | `int64` |
| 8      | event | `int32` |

Each event trigger therefore has a timestamp specified in nanoseconds,
and an encoded event type. Encoding may vary across different
experimental setups and depending on other conditions. No assumptions
about encoding are made in general. Processing and conversion software
should receive the information about the event trigger encoding from
other sources than the DAQ-HDF file.

Trial descriptor records are stored in DAQ-HDF files in the form of a
dataset named `TD01` in the root group.

`TD01` is an array of structures:

| Offset | Name      | Type               |
| ------ | --------- | ------------------ |
| 0      | time      | `int64`            |
| 8      | TrialNo   | `int32`            |
| 12     | StimNo    | `int32`            |
| 16     | reserved1 | `Unsigned` `int32` |
| 20     | reserved2 | `Unsigned` `int32` |

Each `TD01` record describes a trial. It is written by the Data
Acquisition Program at the trial start. Typically, at the same time
another EV02 record is written which has the same timestamp, expressed
in nanoseconds. TrialNo and StimNo are fields with the same meaning as
in the Trialmap. Typically, during production of the Trialmap, the
information from these fields is copied into the respective Trialmap
fields. However, from a `TD01` record alone it is not possible to
reconstruct the information about trial end and trial outcome. Fields
reserved1 and reserved2 are present in DAQ files, however so far they
were never used to convey any information. In various applications of
DAQ-HDF files these fields may eventually find some use. Higher-level
analysis software usually does not read or interpret the information
from EV02 and TD01 datasets.

## References

[1] http://hdf.ncsa.uiuc.edu/HDF5/

DAQ-HDF specifications are designed and documented by Michael Borisov,
modified by Joscha Schmiedt.

Copyright © 2005-2026, University of Bremen
