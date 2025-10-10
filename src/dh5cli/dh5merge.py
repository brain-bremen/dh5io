"""CLI tool for merging multiple DH5 files.

This tool merges two or more DH5 files that contain the same CONT blocks
(channels) recorded at different non-overlapping times. The DATA arrays
are concatenated and the INDEX datasets are updated to reflect the new offsets.
"""

import argparse
import sys
import logging
from pathlib import Path
import numpy as np
from typing import List, Optional, Set

from dh5io.dh5file import DH5File
from dh5io.create import create_dh_file
from dh5io.cont import (
    create_cont_group_from_data_in_file,
    Cont,
)

logger = logging.getLogger(__name__)


def merge_dh5_files(
    input_files: List[Path],
    output_file: Path,
    cont_ids: Optional[List[int]] = None,
    overwrite: bool = False,
) -> None:
    """
    Merge multiple DH5 files into a single output file.

    Parameters
    ----------
    input_files : List[Path]
        List of input DH5 file paths to merge
    output_file : Path
        Output file path
    cont_ids : Optional[List[int]]
        If provided, only merge these CONT block IDs. Otherwise, merge all common blocks.
    overwrite : bool
        Whether to overwrite the output file if it exists
    """
    if len(input_files) < 2:
        raise ValueError("At least two input files are required for merging")

    # Verify all input files exist
    for file_path in input_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

    # Load all input files
    input_dh5_files = [DH5File(str(f), mode="r") for f in input_files]

    try:
        # Determine which CONT blocks to merge
        cont_blocks_to_merge = determine_cont_blocks_to_merge(input_dh5_files, cont_ids)

        if not cont_blocks_to_merge:
            raise ValueError("No common CONT blocks found to merge")

        logger.info(f"Merging CONT blocks: {sorted(cont_blocks_to_merge)}")

        # Get boards attribute from first file
        boards = input_dh5_files[0].file.attrs.get("BOARDS", [])

        # Create output file
        output_dh5 = create_dh_file(
            str(output_file),
            overwrite=overwrite,
            boards=list(boards) if len(boards) > 0 else [],
        )

        try:
            # Merge each CONT block
            for cont_id in sorted(cont_blocks_to_merge):
                logger.info(f"Merging CONT{cont_id}...")
                merge_cont_block(input_dh5_files, output_dh5, cont_id)

            logger.info(
                f"Successfully merged {len(input_files)} files into {output_file}"
            )

        finally:
            output_dh5.file.close()

    finally:
        # Close all input files
        for dh5_file in input_dh5_files:
            dh5_file.file.close()


def determine_cont_blocks_to_merge(
    input_files: List[DH5File], cont_ids: Optional[List[int]] = None
) -> Set[int]:
    """
    Determine which CONT blocks should be merged.

    If cont_ids is provided, verify those blocks exist in all files.
    Otherwise, find the intersection of CONT block IDs across all files.

    Parameters
    ----------
    input_files : List[DH5File]
        List of input DH5 files
    cont_ids : Optional[List[int]]
        Specific CONT block IDs to merge

    Returns
    -------
    Set[int]
        Set of CONT block IDs to merge
    """
    # Get CONT block IDs from each file
    file_cont_ids = [set(f.get_cont_group_ids()) for f in input_files]

    if cont_ids is not None:
        # Use specified CONT IDs
        requested_ids = set(cont_ids)

        # Verify all requested IDs exist in all files
        for i, ids in enumerate(file_cont_ids):
            missing = requested_ids - ids
            if missing:
                raise ValueError(
                    f"CONT blocks {sorted(missing)} not found in file {i}: "
                    f"{input_files[i].file.filename}"
                )

        return requested_ids
    else:
        # Find intersection of all CONT block IDs
        common_ids = file_cont_ids[0]
        for ids in file_cont_ids[1:]:
            common_ids &= ids

        if not common_ids:
            # Show what's available in each file
            for i, ids in enumerate(file_cont_ids):
                logger.warning(
                    f"File {i} ({input_files[i].file.filename}) has CONT blocks: {sorted(ids)}"
                )

        return common_ids


def merge_cont_block(
    input_files: List[DH5File], output_file: DH5File, cont_id: int
) -> None:
    """
    Merge a single CONT block from multiple files.

    This concatenates the DATA arrays and updates the INDEX dataset
    to reflect the new offsets.

    Parameters
    ----------
    input_files : List[DH5File]
        List of input DH5 files
    output_file : DH5File
        Output DH5 file
    cont_id : int
        CONT block ID to merge
    """
    # Load CONT blocks from all files
    cont_blocks = [f.get_cont_group_by_id(cont_id) for f in input_files]

    # Validate compatibility
    validate_cont_blocks_compatible(cont_blocks, cont_id)

    # Use attributes from the first file
    first_cont = cont_blocks[0]
    sample_period = first_cont.sample_period
    calibration = first_cont.calibration
    channels = first_cont.channels
    name = first_cont.name
    comment = first_cont.comment
    signal_type = first_cont.signal_type

    # Concatenate data and merge indices
    merged_data, merged_index = concatenate_cont_data(cont_blocks)

    # Create merged CONT block in output file
    create_cont_group_from_data_in_file(
        output_file.file,
        cont_id,
        data=merged_data,
        index=merged_index,
        sample_period_ns=np.int32(sample_period),
        calibration=calibration,
        channels=channels,
        name=name,
        comment=comment,
        signal_type=signal_type,
    )


def validate_cont_blocks_compatible(cont_blocks: List[Cont], cont_id: int) -> None:
    """
    Validate that CONT blocks from different files are compatible for merging.

    All blocks must have the same:
    - Sample period
    - Number of channels
    - Calibration (if present)
    - Channel configuration

    Parameters
    ----------
    cont_blocks : List[Cont]
        List of CONT blocks to validate
    cont_id : int
        CONT block ID (for error messages)
    """
    first = cont_blocks[0]

    for i, cont in enumerate(cont_blocks[1:], start=1):
        # Check sample period
        if cont.sample_period != first.sample_period:
            raise ValueError(
                f"CONT{cont_id}: Sample period mismatch in file {i}. "
                f"Expected {first.sample_period} ns, got {cont.sample_period} ns"
            )

        # Check number of channels
        if cont.n_channels != first.n_channels:
            raise ValueError(
                f"CONT{cont_id}: Number of channels mismatch in file {i}. "
                f"Expected {first.n_channels}, got {cont.n_channels}"
            )

        # Check calibration
        first_calib = first.calibration
        cont_calib = cont.calibration

        if (first_calib is None) != (cont_calib is None):
            raise ValueError(
                f"CONT{cont_id}: Calibration presence mismatch in file {i}"
            )

        if first_calib is not None and cont_calib is not None:
            if not np.allclose(first_calib, cont_calib):
                logger.warning(f"CONT{cont_id}: Calibration values differ in file {i}")


def merge_index_arrays_with_shapes(
    index_arrays: List[np.ndarray], data_shapes: List[tuple[int, int]]
) -> np.ndarray:
    """
    Merge multiple INDEX arrays with knowledge of data array shapes.

    Parameters
    ----------
    index_arrays : List[np.ndarray]
        List of INDEX arrays to merge
    data_shapes : List[tuple[int, int]]
        List of data array shapes (nSamples, nChannels)

    Returns
    -------
    np.ndarray
        Merged INDEX array with updated offsets
    """
    merged_indices = []
    cumulative_samples = 0

    for index_array, shape in zip(index_arrays, data_shapes):
        # Copy the index array
        adjusted_index = index_array.copy()

        # Adjust offsets by the cumulative sample count
        adjusted_index["offset"] += cumulative_samples

        merged_indices.append(adjusted_index)

        # Update cumulative sample count (nSamples is shape[0])
        cumulative_samples += shape[0]

    return np.concatenate(merged_indices)


# Update concatenate_cont_data to use the improved version
def concatenate_cont_data(cont_blocks: List[Cont]) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate DATA arrays and merge INDEX datasets from multiple CONT blocks.

    The INDEX dataset is updated so that offsets reflect the concatenated DATA array.
    Time stamps are preserved from the original recordings.

    Parameters
    ----------
    cont_blocks : List[Cont]
        List of CONT blocks to concatenate

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Merged data array and merged index array
    """
    # Collect all data arrays
    data_arrays = [cont.data for cont in cont_blocks]

    # Concatenate along the samples axis (axis 0)
    merged_data = np.concatenate(data_arrays, axis=0)

    # Merge index arrays with knowledge of data shapes
    data_shapes = [arr.shape for arr in data_arrays]
    merged_index = merge_index_arrays_with_shapes(
        [cont.index for cont in cont_blocks], data_shapes
    )

    return merged_data, merged_index


def main():
    """Main entry point for the dh5merge CLI tool."""
    parser = argparse.ArgumentParser(
        description="Merge multiple DH5 files containing the same channels recorded at different times.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all common CONT blocks from multiple files
  dh5merge file1.dh5 file2.dh5 file3.dh5 -o merged.dh5
  
  # Merge only specific CONT blocks
  dh5merge file1.dh5 file2.dh5 -o merged.dh5 --cont-ids 0 1 2
  
  # Overwrite existing output file
  dh5merge file1.dh5 file2.dh5 -o merged.dh5 --overwrite
  
  # Enable verbose logging
  dh5merge file1.dh5 file2.dh5 -o merged.dh5 -v
        """,
    )

    parser.add_argument(
        "input_files",
        type=str,
        nargs="+",
        help="Input DH5 files to merge (at least 2 required)",
    )

    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output DH5 file path"
    )

    parser.add_argument(
        "--cont-ids",
        type=int,
        nargs="+",
        help="CONT block IDs to merge (if not specified, all common blocks will be merged)",
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite output file if it exists"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    try:
        # Convert input file paths
        input_files = [Path(f) for f in args.input_files]
        output_file = Path(args.output)

        # Perform merge
        merge_dh5_files(
            input_files=input_files,
            output_file=output_file,
            cont_ids=args.cont_ids,
            overwrite=args.overwrite,
        )

        print(f"\n✓ Successfully merged {len(input_files)} files into {output_file}")

    except Exception as e:
        logger.error(f"Error merging files: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
