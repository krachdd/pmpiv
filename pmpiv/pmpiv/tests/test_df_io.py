#!/usr/bin/env python3
"""Tests for pmpiv.df_io (CSV read / write helpers)."""
import pytest
import numpy as np
import pandas as pd

from pmpiv import df_io


def _sample_df() -> pd.DataFrame:
    """Small trajectory-style DataFrame."""
    rows = [
        {"frame": 0, "particle": 0, "x": 10.5, "y": 20.3, "mass": 900.0},
        {"frame": 0, "particle": 1, "x": 50.1, "y": 60.7, "mass": 1100.0},
        {"frame": 1, "particle": 0, "x": 11.0, "y": 21.0, "mass": 910.0},
        {"frame": 1, "particle": 1, "x": 51.0, "y": 61.5, "mass": 1050.0},
    ]
    df = pd.DataFrame(rows)
    df = df.set_index("frame", drop=False)
    return df


class TestWrite2Csv:
    def test_creates_file(self, tmp_path):
        df_io.write2csv(_sample_df(), str(tmp_path), "out.csv")
        assert (tmp_path / "out.csv").exists()

    def test_missing_folder_raises(self):
        with pytest.raises(FileNotFoundError):
            df_io.write2csv(_sample_df(), "/nonexistent/dir", "out.csv")


class TestReadCsv:
    def test_roundtrip_values(self, tmp_path):
        orig = _sample_df().reset_index(drop=True)
        df_io.write2csv(orig, str(tmp_path), "data.csv")
        loaded = df_io.read_csv(str(tmp_path), "data.csv")
        # values match
        pd.testing.assert_frame_equal(
            loaded.reset_index(drop=True)[["frame", "particle", "x", "y", "mass"]],
            orig[["frame", "particle", "x", "y", "mass"]],
        )

    def test_frame_set_as_index(self, tmp_path):
        df_io.write2csv(_sample_df().reset_index(drop=True), str(tmp_path), "data.csv")
        loaded = df_io.read_csv(str(tmp_path), "data.csv")
        assert loaded.index.name == "frame"

    def test_missing_folder_raises(self):
        with pytest.raises(FileNotFoundError):
            df_io.read_csv("/nonexistent/dir", "data.csv")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            df_io.read_csv(str(tmp_path), "does_not_exist.csv")
