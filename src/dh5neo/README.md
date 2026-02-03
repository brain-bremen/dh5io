# DH5 Neo Interface

`dh5neo` provides access to electrophysiology data stored in the DAQ-HDF5 (DH5) file format using the [Neo](https://github.com/NeuralEnsemble/python-neo) object model.

## Overview

Neo is a Python package for working with electrophysiology data in Python, together with support for reading a wide range of neurophysiology file formats. The `dh5neo` module implements Neo's I/O interfaces to enable reading DH5 files as Neo objects.

## Features

- **Full Neo compatibility**: Read DH5 files as Neo `Block`, `Segment`, `AnalogSignal`, `SpikeTrain`, `Event`, and `Epoch` objects
- **Two-level interface**:
  - **`DH5RawIO`**: Low-level interface for direct data access (inherits from `neo.rawio.BaseRawIO`)
  - **`DH5IO`**: High-level interface that creates Neo objects (inherits from `neo.io.BaseFromRaw`)
- **Supports all DH5 data types**:
  - Continuous signals (CONT groups) → Neo `AnalogSignal`
  - Spike data (SPIKE groups) → Neo `SpikeTrain` with waveforms
  - Event triggers (EV02) → Neo `Event`
  - Trial map (TRIALMAP) → Neo `Epoch`

## Installation

The `dh5neo` module is part of the `dh5io` package. Install with Neo support:

```bash
pip install dh5io neo
```

## Quick Start

### High-Level Interface (Recommended)

Use `DH5IO` to read DH5 files as Neo objects:

```python
from dh5neo import DH5IO

# Open a DH5 file
reader = DH5IO('mydata.dh5')

# Read entire file as a Neo Block
block = reader.read_block()

# Access segments (trials)
for segment in block.segments:
    print(f"Segment duration: {segment.t_stop - segment.t_start} s")

    # Access analog signals
    for signal in segment.analogsignals:
        print(f"  Signal: {signal.shape}, {signal.sampling_rate}")

    # Access spike trains
    for spiketrain in segment.spiketrains:
        print(f"  Spikes: {len(spiketrain)} spikes")
        print(f"  Waveforms: {spiketrain.waveforms.shape}")

    # Access events
    for event in segment.events:
        print(f"  Events: {len(event.times)} events")
```

### Low-Level Interface (Advanced)

Use `DH5RawIO` for direct access to raw data:

```python
from dh5neo import DH5RawIO
import numpy as np

# Open and parse file
reader = DH5RawIO('mydata.dh5')
reader.parse_header()

# Inspect header information
print(f"Blocks: {reader.header['nb_block']}")
print(f"Segments: {reader.header['nb_segment']}")
print(f"Signal streams: {len(reader.header['signal_streams'])}")

# Read analog signal chunk
chunk = reader._get_analogsignal_chunk(
    block_index=0,
    seg_index=0,
    i_start=0,
    i_stop=1000,
    stream_index=0,
    channel_indexes=[0, 1, 2]  # Read specific channels
)

# Get spike timestamps
timestamps = reader._get_spike_timestamps(
    block_index=0,
    seg_index=0,
    spike_channel_index=0,
    t_start=None,
    t_stop=None
)

# Rescale to seconds
timestamps_s = reader._rescale_spike_timestamp(timestamps, np.float64)
```

## DH5 to Neo Mapping

| DH5 Concept    | Neo Concept            | Description                                 |
| -------------- | ---------------------- | ------------------------------------------- |
| DH5 File       | `Block`                | One file = one block                        |
| TRIALMAP entry | `Segment`              | Each trial = one segment                    |
| CONT group     | `AnalogSignal` stream  | Continuous signals grouped by sampling rate |
| CONT channel   | `AnalogSignal` channel | Individual signal channel                   |
| SPIKE group    | `SpikeTrain`           | Spike timestamps with waveforms             |
| EV02 dataset   | `Event`                | Event triggers with codes                   |
| TRIALMAP       | `Epoch`                | Trial information with metadata             |

### Important Note on Time Representation

**DH5 Files**: All timestamps (CONT INDEX, SPIKE INDEX, TRIALMAP, EV02) share a **common absolute time base** starting at t=0 for the entire recording session. Timestamps are stored in nanoseconds.

**Neo Representation**: The implementation **preserves absolute times** from the DH5 file:

- Segment `t_start` = TRIALMAP StartTime (in seconds)
- Segment `t_stop` = TRIALMAP EndTime (in seconds)
- All data timestamps (signals, spikes, events) maintain their **absolute times** from the DH5 file
- Conversion applied: nanoseconds → seconds only

This preserves the complete temporal context from the recording session. For trial-aligned analysis (relative to trial start), you can easily convert times in your analysis code.

## Data Organization

### Segments (Trials)

Each entry in the DH5 TRIALMAP corresponds to one Neo `Segment`. Segments contain:

- Time boundaries from `StartTime` and `EndTime`
- All analog signals, spike trains, and events within that time window
- Annotations with trial metadata (`TrialNo`, `StimNo`, `Outcome`)

### Signal Streams

Each CONT group in the DH5 file becomes a signal stream. All channels in a CONT group share:

- Same sampling rate
- Same time base (INDEX array)
- Aligned samples

### Spike Channels

Each SPIKE group becomes a spike channel with:

- Spike timestamps (from INDEX dataset)
- Waveforms (from DATA dataset)
- Calibration and metadata
- Currently one unit per SPIKE group (cluster info support planned)

### Events and Epochs

- **Events** (`Event`): Individual event triggers from EV02 dataset
- **Epochs** (`Epoch`): Trial information from TRIALMAP with duration

## Implementation Details

### Time Representation and Conversion

**DH5 Storage Format**:

- All timestamps stored as absolute times in **nanoseconds** (int64)
- Common time base for entire file (t=0 at recording start)
- CONT INDEX times, SPIKE INDEX times, TRIALMAP times, EV02 times all comparable

**Neo Representation**:

- Times in **seconds** (float64)
- **Absolute times preserved** from DH5 file
- Each Segment's `t_start` and `t_stop` reflect actual recording times

**Automatic Conversion Applied**:

1. **Nanoseconds → Seconds**: Division by 1e9 (only unit conversion)
   - Segment `t_start = StartTime / 1e9` (absolute time in seconds)
   - Segment `t_stop = EndTime / 1e9` (absolute time in seconds)
   - Spike times: `spike_time / 1e9` (absolute time in seconds)
   - Event times: `event_time / 1e9` (absolute time in seconds)

**For Trial-Aligned Analysis**:

If you need relative times (e.g., time from trial start), simply subtract the segment's `t_start`:

```python
segment = block.segments[0]
for spiketrain in segment.spiketrains:
    # Convert to relative times (time from trial start)
    relative_times = spiketrain.times - segment.t_start
```

This approach preserves complete temporal context while allowing easy conversion to relative times when needed.

### Calibration

Analog signals and spike waveforms are automatically calibrated using the `Calibration` attribute from DH5 CONT/SPIKE groups. The gain is applied when creating Neo objects.

### Multi-Region Signals

DH5 CONT groups can have multiple regions (gaps in recording). The implementation:

- Concatenates all regions within a trial
- Handles discontinuities correctly
- Maps trial time windows to data indices

## Examples

See `examples/example_dh5_neo.py` for comprehensive examples including:

- Basic file inspection
- Reading analog signals
- Reading spike data with waveforms
- Reading events and epochs
- Creating Neo objects

## API Reference

### DH5RawIO

Low-level interface for reading DH5 files.

**Constructor:**

```python
DH5RawIO(filename: str | pathlib.Path)
```

**Key Methods:**

- `parse_header()`: Parse file header and populate metadata
- `_get_analogsignal_chunk()`: Read analog signal data
- `_get_spike_timestamps()`: Get spike times for a segment
- `_get_spike_raw_waveforms()`: Get spike waveforms
- `_get_event_timestamps()`: Get event times and labels

### DH5IO

High-level interface that creates Neo objects.

**Constructor:**

```python
DH5IO(filename: str | pathlib.Path)
```

**Key Methods:**

- `read_block()`: Read entire file as Neo Block
- `read_segment()`: Read specific segment
- All standard Neo IO methods

## Testing

Run the test suite:

```bash
pytest tests/test_dh5neo.py -v
```

Tests cover:

- Header parsing
- Signal reading
- Spike data access
- Event handling
- Edge cases and error conditions

## Data Integrity

The implementation preserves all timing relationships from the original DH5 file:

- **Absolute times preserved exactly** - all timestamps maintain their original values from the DH5 file
- Trial durations are preserved exactly
- Multi-region CONT data (gaps) handled correctly
- All temporal relationships maintained (spike-to-event, trial-to-trial, etc.)

**Converting to Trial-Relative Times**:

Since absolute times are preserved, you can easily convert to relative times (e.g., time from trial start) for analysis:

```python
reader = DH5RawIO('data.dh5')
reader.parse_header()

# Read segment (trial) - times are absolute
segment_index = 0
spike_times_ns = reader._get_spike_timestamps(0, segment_index, 0, None, None)
spike_times_sec = spike_times_ns / 1e9  # Absolute times in seconds

# Get trial start time for reference
segment_t_start = reader._segment_t_start(0, segment_index)  # Absolute time

# Convert to relative times (time from trial start)
spike_times_relative = spike_times_sec - segment_t_start

# Or using Neo objects:
io = DH5IO('data.dh5')
block = io.read_block()
segment = block.segments[0]
for spiketrain in segment.spiketrains:
    relative_times = spiketrain.times - segment.t_start
```

## Requirements

- Python >= 3.10
- numpy
- h5py
- neo >= 0.10
- dh5io (parent package)

## Contributing

Issues and pull requests welcome at the [dh5io repository](https://github.com/cog-neurophys-lab/dh5io).

## License

See LICENSE file in the dh5io repository.

## References

- [Neo documentation](https://neo.readthedocs.io/)
- [DAQ-HDF5 specification](https://github.com/cog-neurophys-lab/DAQ-HDF5)
- [DH5IO documentation](../../../README.md)
