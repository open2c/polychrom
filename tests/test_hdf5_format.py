"""
Comprehensive tests for hdf5_format module
"""

import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from polychrom.hdf5_format import (
    HDF5Reporter,
    _convert_to_hdf5_array,
    _read_h5_group,
    _write_group,
    list_URIs,
    load_hdf5_file,
    load_URI,
    save_hdf5_file,
)


def test_convert_to_hdf5_array():
    """Test conversion of various data types to HDF5-compatible format"""
    # Test string conversion
    datatype, converted = _convert_to_hdf5_array("test_string")
    assert datatype == "item"
    assert converted == b"test_string"

    # Test integer conversion
    datatype, converted = _convert_to_hdf5_array(42)
    assert datatype == "item"
    assert converted == 42

    # Test float conversion
    datatype, converted = _convert_to_hdf5_array(3.14)
    assert datatype == "item"
    assert abs(converted - 3.14) < 1e-10

    # Test array conversion
    arr = np.array([1, 2, 3, 4])
    datatype, converted = _convert_to_hdf5_array(arr)
    assert datatype == "ndarray"
    assert np.array_equal(converted, arr)

    # Test 2D array
    arr2d = np.array([[1, 2], [3, 4]])
    datatype, converted = _convert_to_hdf5_array(arr2d)
    assert datatype == "ndarray"
    assert np.array_equal(converted, arr2d)

    # Test object that can't be converted (should return None, None)
    class CustomObject:
        pass

    obj = CustomObject()
    datatype, converted = _convert_to_hdf5_array([obj, obj])
    assert datatype is None
    assert converted is None


def test_read_write_h5_group(tmp_path):
    """Test reading and writing HDF5 groups"""
    test_file = tmp_path / "test_group.h5"

    test_data = {
        "scalar_int": 42,
        "scalar_float": 3.14,
        "string": "test_string",
        "array_1d": np.array([1, 2, 3, 4, 5]),
        "array_2d": np.random.random((10, 3)),
    }

    # Write data to HDF5 group
    with h5py.File(test_file, "w") as f:
        _write_group(test_data, f)

    # Read data back
    with h5py.File(test_file, "r") as f:
        read_data = _read_h5_group(f)

    # Verify data
    assert read_data["scalar_int"] == 42
    assert abs(read_data["scalar_float"] - 3.14) < 1e-10
    assert read_data["string"] == "test_string"  # Strings are decoded back
    assert np.array_equal(read_data["array_1d"], test_data["array_1d"])
    assert np.allclose(read_data["array_2d"], test_data["array_2d"])


def test_save_load_hdf5_file(tmp_path):
    """Test saving and loading complete HDF5 files"""
    test_file = tmp_path / "test_save_load.h5"

    test_data = {
        "pos": np.random.random((100, 3)),
        "vel": np.random.random((100, 3)),
        "time": 1.234,
        "step": 100,
        "metadata": "simulation_data",
    }

    # Save file
    save_hdf5_file(str(test_file), test_data)

    # Load file
    loaded_data = load_hdf5_file(str(test_file))

    # Verify
    assert np.allclose(loaded_data["pos"], test_data["pos"])
    assert np.allclose(loaded_data["vel"], test_data["vel"])
    assert abs(loaded_data["time"] - test_data["time"]) < 1e-10
    assert loaded_data["step"] == test_data["step"]
    assert loaded_data["metadata"] == "simulation_data"  # Strings are decoded back


def test_list_URIs(tmp_path):
    """Test listing URIs from a trajectory folder"""
    # Create some mock block files
    with h5py.File(tmp_path / "blocks_0-4.h5", "w") as f:
        for i in range(5):
            f.create_group(str(i))

    with h5py.File(tmp_path / "blocks_5-9.h5", "w") as f:
        for i in range(5, 10):
            f.create_group(str(i))

    # Test list format
    uris = list_URIs(str(tmp_path), return_dict=False)
    assert len(uris) == 10
    assert uris[0].endswith("blocks_0-4.h5::0")
    assert uris[9].endswith("blocks_5-9.h5::9")

    # Test dict format
    uri_dict = list_URIs(str(tmp_path), return_dict=True)
    assert len(uri_dict) == 10
    assert 0 in uri_dict
    assert 9 in uri_dict
    assert uri_dict[0].endswith("blocks_0-4.h5::0")

    # Test empty folder error
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No files found"):
        list_URIs(str(empty_dir))

    # Test with empty_error=False
    uris_empty = list_URIs(str(empty_dir), empty_error=False)
    assert uris_empty == []

    # Test duplicate block detection
    with h5py.File(tmp_path / "blocks_8-12.h5", "w") as f:
        for i in range(8, 13):
            f.create_group(str(i))

    with pytest.raises(ValueError, match="Block .* exists more than once"):
        list_URIs(str(tmp_path))


def test_load_URI(tmp_path):
    """Test loading individual blocks via URI"""
    # Create a test file with blocks
    test_file = tmp_path / "blocks_0-2.h5"
    with h5py.File(test_file, "w") as f:
        for i in range(3):
            group = f.create_group(str(i))
            group.create_dataset("pos", data=np.ones((10, 3)) * i)
            group.attrs["block_num"] = i

    # Load a block
    uri = f"{test_file}::1"
    data = load_URI(uri)

    assert np.allclose(data["pos"], np.ones((10, 3)))
    assert data["block_num"] == 1

    # Test invalid URI format
    with pytest.raises(ValueError, match="Invalid URI format"):
        load_URI(str(test_file))


def test_hdf5_reporter_basic(tmp_path):
    """Test basic HDF5Reporter functionality"""
    folder = tmp_path / "trajectory"
    reporter = HDF5Reporter(str(folder), max_data_length=3)

    assert os.path.exists(folder)
    assert reporter.max_data_length == 3
    assert reporter.folder == str(folder)
    assert not reporter.blocks_only

    # Report some data blocks
    for i in range(5):
        reporter.report("data", {"pos": np.random.random((10, 3)), "time": float(i), "step": i * 100})

    # Force dump of remaining data
    reporter.dump_data()

    # Check files were created
    files = list(folder.glob("blocks_*.h5"))
    assert len(files) == 2  # 5 blocks with max_data_length=3 -> 2 files

    # Check URIs
    uris = list_URIs(str(folder))
    assert len(uris) == 5


def test_hdf5_reporter_overwrite(tmp_path):
    """Test HDF5Reporter overwrite functionality"""
    folder = tmp_path / "trajectory"

    # Create initial reporter and save some data
    reporter1 = HDF5Reporter(str(folder), max_data_length=2)
    for i in range(3):
        reporter1.report("data", {"pos": np.ones(3) * i})
    reporter1.dump_data()

    # Verify data exists
    assert len(list_URIs(str(folder))) == 3

    # Try to create new reporter without overwrite (should fail)
    with pytest.raises(RuntimeError, match="folder .* is not empty"):
        HDF5Reporter(str(folder))

    # Create with overwrite=True
    reporter2 = HDF5Reporter(str(folder), overwrite=True, max_data_length=2)
    for i in range(2):
        reporter2.report("data", {"pos": np.ones(3) * (i + 10)})
    reporter2.dump_data()

    # Verify old data was overwritten
    uris = list_URIs(str(folder))
    assert len(uris) == 2
    data0 = load_URI(uris[0])
    assert np.allclose(data0["pos"], np.ones(3) * 10)


def test_hdf5_reporter_blocks_only(tmp_path):
    """Test blocks_only mode"""
    folder = tmp_path / "trajectory"
    reporter = HDF5Reporter(str(folder), blocks_only=True)

    # Report non-data items (should be ignored in blocks_only mode)
    reporter.report("initArgs", {"N": 100, "dt": 0.01})
    reporter.report("applied_forces", {"force1": "harmonic"})

    # Report data blocks
    reporter.report("data", {"pos": np.ones((10, 3))})
    reporter.dump_data()

    # Check that only blocks were saved
    files = list(folder.glob("*.h5"))
    assert len(files) == 1
    assert "blocks_" in files[0].name
    assert not any("initArgs" in f.name for f in files)
    assert not any("applied_forces" in f.name for f in files)


def test_hdf5_reporter_continue_trajectory(tmp_path):
    """Test continuing a trajectory"""
    folder = tmp_path / "trajectory"

    # Create initial trajectory
    reporter1 = HDF5Reporter(str(folder), max_data_length=2)
    for i in range(5):
        reporter1.report("data", {"pos": np.ones((10, 3)) * i, "step": i})
    reporter1.dump_data()

    # Continue trajectory
    reporter2 = HDF5Reporter(str(folder), max_data_length=2, check_exists=False)
    block_num, last_data = reporter2.continue_trajectory()

    assert block_num == 4  # Last block is 4 (0-indexed)
    assert np.allclose(last_data["pos"], np.ones((10, 3)) * 4)
    assert last_data["step"] == 4

    # Add more data
    for i in range(5, 8):
        reporter2.report("data", {"pos": np.ones((10, 3)) * i, "step": i})
    reporter2.dump_data()

    # Verify complete trajectory
    uris = list_URIs(str(folder))
    assert len(uris) == 8

    # Test continuing from specific block (with higher continue_max_delete)
    reporter3 = HDF5Reporter(str(folder), max_data_length=2, check_exists=False)
    block_num, data = reporter3.continue_trajectory(continue_from=2, continue_max_delete=10)
    assert block_num == 2
    assert data["step"] == 2

    # After continuing, add one more block and dump
    reporter3.report("data", {"pos": np.ones((10, 3)) * 99, "step": 99})
    reporter3.dump_data()

    # Verify blocks after continue_from were deleted and new block added
    uris_after = list_URIs(str(folder))
    # Should have blocks 0, 1, 2, 3 (block 3 is the new one we added)
    block_nums = sorted([int(uri.split("::")[-1]) for uri in uris_after])
    assert 0 in block_nums  # Original blocks preserved
    assert 1 in block_nums
    assert 2 in block_nums
    assert 3 in block_nums  # New block added


def test_hdf5_reporter_non_data_reports(tmp_path):
    """Test reporting non-data information"""
    folder = tmp_path / "trajectory"
    reporter = HDF5Reporter(str(folder))

    # Report various types of information
    reporter.report("initArgs", {"N": 1000, "dt": 0.001, "temperature": 300.0, "description": "test simulation"})

    reporter.report("starting_conformation", {"pos": np.random.random((1000, 3)), "source": "random_walk"})

    # Note: nested dicts cannot be saved to HDF5, use flat structure
    reporter.report(
        "applied_forces", {"harmonic_bonds_k": 100, "harmonic_bonds_r0": 1.0, "excluded_volume_radius": 0.5}
    )

    # Check files were created
    assert (folder / "initArgs_0.h5").exists()
    assert (folder / "starting_conformation_0.h5").exists()
    assert (folder / "applied_forces_0.h5").exists()

    # Load and verify data
    init_data = load_hdf5_file(str(folder / "initArgs_0.h5"))
    assert init_data["N"] == 1000
    assert abs(init_data["dt"] - 0.001) < 1e-10
    assert init_data["description"] == "test simulation"  # Strings are decoded

    start_data = load_hdf5_file(str(folder / "starting_conformation_0.h5"))
    assert start_data["pos"].shape == (1000, 3)
    assert start_data["source"] == "random_walk"  # Strings are decoded

    forces_data = load_hdf5_file(str(folder / "applied_forces_0.h5"))
    assert forces_data["harmonic_bonds_k"] == 100
    assert abs(forces_data["harmonic_bonds_r0"] - 1.0) < 1e-10
    assert abs(forces_data["excluded_volume_radius"] - 0.5) < 1e-10


def test_hdf5_compression(tmp_path):
    """Test that compression works and reduces file size"""
    folder = tmp_path / "trajectory"

    # Create reporter with compression
    reporter_compressed = HDF5Reporter(
        str(folder / "compressed"), max_data_length=1, h5py_dset_opts={"compression": "gzip", "compression_opts": 9}
    )

    # Create reporter without compression
    reporter_uncompressed = HDF5Reporter(str(folder / "uncompressed"), max_data_length=1, h5py_dset_opts={})

    # Create data with repeated values (highly compressible)
    data = {"pos": np.ones((10000, 3)) * 42.0}

    reporter_compressed.report("data", data)
    reporter_compressed.dump_data()

    reporter_uncompressed.report("data", data)
    reporter_uncompressed.dump_data()

    # Compare file sizes
    compressed_size = (folder / "compressed" / "blocks_0-0.h5").stat().st_size
    uncompressed_size = (folder / "uncompressed" / "blocks_0-0.h5").stat().st_size

    assert compressed_size < uncompressed_size
    # For highly repetitive data, compression should be significant
    assert compressed_size < uncompressed_size * 0.5


def test_reporter_with_extras(tmp_path):
    """Test saving extra data with blocks"""
    folder = tmp_path / "trajectory"
    reporter = HDF5Reporter(str(folder), max_data_length=2)

    # Report data with extras
    for i in range(3):
        reporter.report(
            "data",
            {
                "pos": np.random.random((10, 3)),
                "custom_scalar": i * 1.5,
                "custom_array": np.arange(5) * i,
                "custom_string": f"block_{i}",
            },
        )
    reporter.dump_data()

    # Load and verify
    uris = list_URIs(str(folder))
    for i, uri in enumerate(uris):
        data = load_URI(uri)
        assert abs(data["custom_scalar"] - i * 1.5) < 1e-10
        assert np.array_equal(data["custom_array"], np.arange(5) * i)
        assert data["custom_string"] == f"block_{i}"  # Strings are decoded


def test_edge_cases(tmp_path):
    """Test various edge cases and error conditions"""
    folder = tmp_path / "trajectory"

    # Test max_data_length edge cases
    reporter = HDF5Reporter(str(folder / "test1"), max_data_length=1)
    reporter.report("data", {"pos": np.ones(3)})
    reporter.dump_data()
    assert len(list_URIs(str(folder / "test1"))) == 1

    # Test with max_data_length = 1 (immediate dump after reaching limit)
    reporter2 = HDF5Reporter(str(folder / "test2"), max_data_length=1)
    reporter2.report("data", {"pos": np.ones(3)})
    # After 1 item with max_data_length=1, should have auto-dumped
    assert len(reporter2.datas) == 0  # Should be empty after auto-dump
    assert len(list_URIs(str(folder / "test2"))) == 1

    # Add another to test multiple single-block files
    reporter2.report("data", {"pos": np.ones(3) * 2})
    reporter2.dump_data()
    assert len(list_URIs(str(folder / "test2"))) == 2

    # Test continue_trajectory with invalid block number
    reporter3 = HDF5Reporter(str(folder / "test1"), check_exists=False)
    with pytest.raises(ValueError, match="block .* not in folder"):
        reporter3.continue_trajectory(continue_from=999)

    # Test warning for non-convertible data
    with pytest.warns(UserWarning, match="Could not convert record"):
        reporter4 = HDF5Reporter(str(folder / "test3"))
        _write_group({"bad_data": [object(), object()]}, h5py.File(folder / "test3" / "test.h5", "w"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
