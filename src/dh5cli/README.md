# dh5cli - CLI tools for inspecting and manipulating DH5 data files

## Available Commands

### dh5tree

Display the contents of a DH5 file as a tree structure.

```bash
dh5tree file.dh5
```

### dh5merge

Merge multiple DH5 files containing the same channels (CONT blocks) recorded at different
non-overlapping times. `dh5merge` is powered by Claude Sonnet 4.5.

#### Basic Usage

**GUI Mode** (no arguments):

```bash
dh5merge
```

Running without arguments opens a graphical file selector to choose input files and output location.

**Command-line Mode**:
Merge all common CONT blocks from multiple files:

```bash
# With explicit output filename
dh5merge file1.dh5 file2.dh5 file3.dh5 -o merged.dh5

# Auto-suggest output filename based on common prefix
dh5merge session_part1.dh5 session_part2.dh5
# Output: session_part_merged.dh5
```

#### Options

- `-o, --output`: Output file path (optional). If not specified, will auto-suggest based on common filename prefix
- `--cont-ids`: Specific CONT block IDs to merge (optional). If not specified, all common blocks will be merged.
- `--overwrite`: Overwrite output file if it exists
- `-v, --verbose`: Enable verbose logging

#### Examples

Auto-suggest output filename:

```bash
# Files with common prefix automatically generate output name
dh5merge experiment_session1.dh5 experiment_session2.dh5
# Output: experiment_session_merged.dh5
```

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

Files with no common prefix (must specify output):

```bash
# These files have no common prefix, so you must use -o
dh5merge recording1.dh5 data2.dh5 -o merged.dh5
```

#### Auto-Filename Suggestion

When no output filename is specified with `-o/--output`, the tool automatically suggests a merged filename based on the common prefix of input files:

- **Finds common prefix**: Determines the longest character-by-character prefix
- **Cleans up**: Removes trailing underscores, hyphens, or spaces
- **Validates**: Requires at least 2 characters of overlap
- **Generates**: Appends `_merged.dh5` to create the output filename
- **Aborts if no overlap**: If files have no common prefix, the operation aborts with an error

**Examples:**

- `session1_day1.dh5`, `session1_day2.dh5` → `session1_day_merged.dh5`
- `experiment_A.dh5`, `experiment_B.dh5` → `experiment_merged.dh5`
- `file1.dh5`, `data2.dh5` → Error (no common prefix)

For more details, see [Auto-Filename Suggestion Documentation](../../docs/AUTO_FILENAME_SUGGESTION.md).

#### How It Works

The `dh5merge` tool:

1. **Suggests output filename**: If `-o/--output` not provided, auto-suggests based on common prefix or aborts if no overlap exists

2. **Validates compatibility**: Ensures all CONT blocks to be merged have:
   - Same sample period
   - Same number of channels
   - Compatible calibration values

3. **Validates sequential data**: Ensures files contain sequential, non-overlapping data across:
   - CONT blocks (continuous recording data)
   - WAVELET blocks (time-frequency data)
   - TRIALMAP entries (trial information)
   - EV02 events (event triggers)

4. **Concatenates data**: Merges DATA arrays from each file along the time axis

5. **Updates INDEX datasets**: Adjusts the offset values in the INDEX dataset to reflect the concatenated data structure. The INDEX dataset describes recording regions with:
   - `time`: timestamp of first sample in nanoseconds
   - `offset`: sample offset within the DATA array

6. **Merges TRIALMAPs**: Concatenates TRIALMAP datasets from all input files, preserving trial information including:
   - Trial numbers
   - Stimulus numbers
   - Outcome codes
   - Start/end timestamps

7. **Merges EV02 (Event Triggers)**: Concatenates event trigger datasets from all input files, preserving:
   - Event timestamps (nanoseconds)
   - Event codes

8. **Records merge operation**: Adds a processing history entry to the output file documenting:
   - The merge operation
   - Source file names
   - Number of files merged
   - Timestamp and tool version

9. **Preserves metadata**: Copies attributes like calibration, channel configuration, and signal type from the first file

#### Requirements

- All input files must contain the CONT blocks being merged
- CONT blocks must have matching sample periods and channel counts
- Recording times must be sequential and non-overlapping (validated automatically)
- For auto-filename suggestion: input files must have a common prefix of at least 2 characters

#### See Also

- [Auto-Filename Suggestion Documentation](../../docs/AUTO_FILENAME_SUGGESTION.md)
- [Sequential Data Validation](../../docs/SEQUENTIAL_DATA_VALIDATION.md)
- [WAVELET Merge Feature](../../WAVELET_MERGE_README.md)
