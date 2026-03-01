#!/usr/bin/env python3
"""Tests for pmpiv.helper utility functions."""
import pytest
import numpy as np

import pmpiv
from pmpiv import helper


class TestFolderSanity:
    def test_existing_folder_does_not_raise(self, tmp_path):
        helper._folder_sanity(str(tmp_path))  # must not raise

    def test_missing_folder_raises(self):
        with pytest.raises(FileNotFoundError):
            helper._folder_sanity("/nonexistent/path/xyz")


class TestFileSanity:
    def test_existing_file_does_not_raise(self, tmp_path):
        f = tmp_path / "dummy.txt"
        f.write_text("hello")
        helper._file_sanity(str(f))  # must not raise

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            helper._file_sanity("/nonexistent/file.txt")


class TestCheckMetadata:
    def test_both_none_raises(self):
        with pytest.raises(ValueError):
            helper._check_metadata(None, None)

    def test_metadata_object_returned_as_is(self, md):
        result = helper._check_metadata(md, None)
        assert result is md

    def test_metadata_loaded_from_file(self, metadata_file):
        result = helper._check_metadata(None, str(metadata_file))
        assert isinstance(result, pmpiv.metadata.Metadata)
        assert result.FEATURE_SIZE == 11


class TestReadTif:
    def test_returns_numpy_array(self, test_image_tif):
        from pathlib import Path
        p = Path(test_image_tif)
        arr = helper.read_tif(str(p.parent), p.name)
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 2

    def test_missing_folder_raises(self):
        with pytest.raises(FileNotFoundError):
            helper.read_tif("/nonexistent/dir", "image.tif")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            helper.read_tif(str(tmp_path), "not_there.tif")
