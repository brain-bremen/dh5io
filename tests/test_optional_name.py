"""Test merging DH5 files without optional Name attribute."""

import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dh5io.create import create_dh_file
from dh5io.cont import create_empty_index_array
from dh5io.dh5file import DH5File
from dh5cli.dh5merge import merge_dh5_files


def test_merge_without_name_attribute():
    """Test merging files where CONT blocks don't have Name attribute."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create two test files without Name attribute
        file1 = tmpdir / "test1.dh5"
        file2 = tmpdir / "test2.dh5"
        output = tmpdir / "merged.dh5"

        # Create file 1
        with create_dh_file(file1, overwrite=True) as dh5:
            data1 = np.random.randint(-1000, 1000, size=(100, 3), dtype=np.int16)
            index1 = create_empty_index_array(1)
            index1[0] = (0, 0)

            # Create CONT group directly without Name attribute
            cont_group = dh5._file.create_group("CONT3")
            cont_group.create_dataset("DATA", data=data1)
            cont_group.create_dataset(
                "INDEX", data=index1, dtype=dh5._file["CONT_INDEX_ITEM"]
            )
            cont_group.attrs["SamplePeriod"] = np.int32(1000)
            cont_group.attrs["Calibration"] = np.ones(3, dtype=np.float64)
            # Intentionally NOT setting Name attribute

        # Create file 2
        with create_dh_file(file2, overwrite=True) as dh5:
            data2 = np.random.randint(-1000, 1000, size=(150, 3), dtype=np.int16)
            index2 = create_empty_index_array(1)
            index2[0] = (1000000, 0)

            # Create CONT group directly without Name attribute
            cont_group = dh5._file.create_group("CONT3")
            cont_group.create_dataset("DATA", data=data2)
            cont_group.create_dataset(
                "INDEX", data=index2, dtype=dh5._file["CONT_INDEX_ITEM"]
            )
            cont_group.attrs["SamplePeriod"] = np.int32(1000)
            cont_group.attrs["Calibration"] = np.ones(3, dtype=np.float64)
            # Intentionally NOT setting Name attribute

        # Try to merge - this should now work without errors
        try:
            merge_dh5_files([file1, file2], output, overwrite=True)

            # Verify the merge worked
            with DH5File(output, mode="r") as merged:
                cont = merged.get_cont_group_by_id(3)
                assert cont.n_samples == 250  # 100 + 150
                assert cont.n_channels == 3
                assert cont.n_regions == 2

            print("✓ Test passed! Files without Name attribute merged successfully.")
            return True

        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            return False


if __name__ == "__main__":
    success = test_merge_without_name_attribute()
    sys.exit(0 if success else 1)
