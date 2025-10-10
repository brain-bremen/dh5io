# dh5cli - CLI tools for inspecting and manipulating DH5 data files

## Available Commands

### dh5tree

Display the contents of a DH5 file as a tree structure.

```bash
dh5tree file.dh5
```

### dh5merge

Merge multiple DH5 files containing the same channels (CONT blocks) recorded at different non-overlapping times.

#### Basic Usage

Merge all common CONT blocks from multiple files:

```bash
dh5merge file1.dh5 file2.dh5 file3.dh5 -o merged.dh5
```

#### Options

- `-o, --output`: Output file path (required)
- `--cont-ids`: Specific CONT block IDs to merge (optional). If not specified, all common blocks will be merged.
- `--overwrite`: Overwrite output file if it exists
- `-v, --verbose`: Enable verbose logging

#### Examples

Merge only specific CONT blocks:
```bash
dh5merge file1.dh5 file2.dh5 -o merged.dh5 --cont-ids 0 1 2
```

Merge with overwrite:
```bash
dh5merge file1.dh5 file2.dh5 -o merged.dh5 --overwrite
```

Enable verbose logging:
```bash
dh5merge file1.dh5 file2.dh5 -o merged.dh5 -v
```

#### How It Works

The `dh5merge` tool:

1. **Validates compatibility**: Ensures all CONT blocks to be merged have:
   - Same sample period
   - Same number of channels
   - Compatible calibration values

2. **Concatenates data**: Merges DATA arrays from each file along the time axis

3. **Updates INDEX datasets**: Adjusts the offset values in the INDEX dataset to reflect the concatenated data structure. The INDEX dataset describes recording regions with:
   - `time`: timestamp of first sample in nanoseconds
   - `offset`: sample offset within the DATA array

4. **Preserves metadata**: Copies attributes like calibration, channel configuration, and signal type from the first file

#### Requirements

- All input files must contain the CONT blocks being merged
- CONT blocks must have matching sample periods and channel counts
- Recording times should be non-overlapping (though this is not strictly validated)

