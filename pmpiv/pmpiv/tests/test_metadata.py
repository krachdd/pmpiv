#!/usr/bin/env python3
"""Tests for pmpiv.metadata.Metadata."""
import pytest
from pathlib import Path

from conftest import _write_metadata
import pmpiv


class TestMetadataParsing:
    """Valid metadata file is parsed into correct types."""

    def test_string_fields(self, md):
        assert isinstance(md.IN_PATH, str)
        assert isinstance(md.WORKING_DIR, str)
        assert isinstance(md.IN_FORMAT, str)

    def test_float_fields(self, md):
        assert isinstance(md.PIXELSIZE, float)
        assert isinstance(md.HEIGHT, float)
        assert isinstance(md.FPS, float)
        assert isinstance(md.STATIC_DEV_PARAMETER, float)

    def test_int_fields(self, md):
        for attr in (
            "START_FRAME", "END_FRAME", "RATE",
            "FEATURE_SIZE", "FEATURE_MIN_SIZE",
            "MAX_PARTICLE_SPEED", "MEMORY", "DURATION",
            "CHECK_STATIC",
        ):
            assert isinstance(getattr(md, attr), int), f"{attr} should be int"

    def test_list_fields_empty(self, md):
        assert md.REMOVAL == []
        assert md.EXTRACTION == []

    def test_values(self, md):
        assert md.PIXELSIZE == pytest.approx(1.3e-6)
        assert md.HEIGHT == pytest.approx(100.0e-6)
        assert md.FPS == pytest.approx(70.0)
        assert md.FEATURE_SIZE == 11
        assert md.DURATION == 5
        assert md.CHECK_STATIC == 5


class TestMetadataErrors:
    """Invalid configs raise the appropriate exceptions."""

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            pmpiv.metadata.Metadata("/nonexistent/path/does_not_exist.txt")

    def test_invalid_format_raises(self, tmp_path):
        cfg = _write_metadata(tmp_path, IN_FORMAT="jpg")
        with pytest.raises(ValueError, match="Format"):
            pmpiv.metadata.Metadata(str(cfg))

    def test_empty_in_path_raises(self, tmp_path):
        cfg = _write_metadata(tmp_path, IN_PATH="")
        with pytest.raises(ValueError, match="IN_PATH"):
            pmpiv.metadata.Metadata(str(cfg))

    def test_empty_working_dir_raises(self, tmp_path):
        cfg = _write_metadata(tmp_path, WORKING_DIR="")
        with pytest.raises(ValueError, match="WORKING_DIR"):
            pmpiv.metadata.Metadata(str(cfg))

    def test_json_file_missing_raises(self, tmp_path):
        cfg = _write_metadata(
            tmp_path,
            JSON_PATH=str(tmp_path),
            REMOVAL="nonexistent.json",
        )
        with pytest.raises(FileNotFoundError):
            pmpiv.metadata.Metadata(str(cfg))


class TestMetadataWithAnnotations:
    """REMOVAL / EXTRACTION lists are parsed correctly when files exist."""

    def test_removal_list_populated(self, tmp_path):
        j = tmp_path / "removal.json"
        j.write_text("{}")
        cfg = _write_metadata(tmp_path, JSON_PATH=str(tmp_path), REMOVAL="removal.json")
        md = pmpiv.metadata.Metadata(str(cfg))
        assert md.REMOVAL == ["removal.json"]

    def test_extraction_multiple(self, tmp_path):
        for name in ("a.json", "b.json"):
            (tmp_path / name).write_text("{}")
        cfg = _write_metadata(
            tmp_path,
            JSON_PATH=str(tmp_path),
            EXTRACTION="a.json, b.json",
        )
        md = pmpiv.metadata.Metadata(str(cfg))
        assert md.EXTRACTION == ["a.json", "b.json"]
