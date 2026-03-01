#!/usr/bin/env python3
"""Tests for pmpiv.fstats: Frame_Statistics and Sequence_Statistics."""
import json
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

import pmpiv
from pmpiv import fstats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_locate_result(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Mimic a single tp.locate() output DataFrame."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "mass": rng.uniform(800, 1200, n),
            "size": rng.uniform(2.5, 3.5, n),
            "ecc": rng.uniform(0.0, 0.3, n),
            "x": rng.uniform(0, 200, n),
            "y": rng.uniform(0, 200, n),
        }
    )


_EXPECTED_KEYS = [
    f"{method}_{param}"
    for method in (
        "mean", "max", "min", "std",
        "percentile3", "percentile97",
        "percentile5", "percentile95",
        "percentile10", "percentile90",
        "percentile15", "percentile85",
    )
    for param in ("mass", "size", "ecc")
]


# ---------------------------------------------------------------------------
# Sequence_Statistics – statistical wrapper methods
# ---------------------------------------------------------------------------

class TestSequenceStatisticsMethods:
    """The standalone statistical wrapper methods return correct values."""

    @pytest.fixture
    def ss(self, md):
        return fstats.Sequence_Statistics([], m_metadata=md)

    def test_mean(self, ss):
        a = np.array([1.0, 2.0, 3.0])
        assert ss.mean(a) == pytest.approx(2.0)

    def test_max(self, ss):
        a = np.array([1.0, 5.0, 3.0])
        assert ss.max(a) == pytest.approx(5.0)

    def test_min(self, ss):
        a = np.array([1.0, 5.0, 3.0])
        assert ss.min(a) == pytest.approx(1.0)

    def test_std(self, ss):
        a = np.array([2.0, 2.0, 2.0])
        assert ss.std(a) == pytest.approx(0.0)

    def test_percentile5(self, ss):
        a = np.arange(100, dtype=float)
        assert ss.percentile5(a) == pytest.approx(np.percentile(a, 5.0))

    def test_percentile95(self, ss):
        a = np.arange(100, dtype=float)
        assert ss.percentile95(a) == pytest.approx(np.percentile(a, 95.0))


# ---------------------------------------------------------------------------
# Sequence_Statistics – sget (sequential)
# ---------------------------------------------------------------------------

class TestSequenceStatisticsSget:
    def test_returns_dict(self, md):
        n_frames = 3
        mock_frames = [np.zeros((50, 50), dtype=np.uint8)] * n_frames
        fake_result = _synthetic_locate_result()

        with patch("trackpy.locate", return_value=fake_result):
            ss = fstats.Sequence_Statistics(mock_frames, m_metadata=md, parallel=False)
            result = ss.sget()

        assert isinstance(result, dict)

    def test_all_expected_keys_present(self, md):
        mock_frames = [np.zeros((50, 50), dtype=np.uint8)] * 2
        fake_result = _synthetic_locate_result()

        with patch("trackpy.locate", return_value=fake_result):
            ss = fstats.Sequence_Statistics(mock_frames, m_metadata=md, parallel=False)
            result = ss.sget()

        for key in _EXPECTED_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_mean_mass_within_range(self, md):
        mock_frames = [np.zeros((50, 50), dtype=np.uint8)] * 5
        fake_result = _synthetic_locate_result()

        with patch("trackpy.locate", return_value=fake_result):
            ss = fstats.Sequence_Statistics(mock_frames, m_metadata=md, parallel=False)
            result = ss.sget()

        assert result["min_mass"] <= result["mean_mass"] <= result["max_mass"]

    def test_percentile_ordering(self, md):
        mock_frames = [np.zeros((50, 50), dtype=np.uint8)] * 3
        fake_result = _synthetic_locate_result()

        with patch("trackpy.locate", return_value=fake_result):
            ss = fstats.Sequence_Statistics(mock_frames, m_metadata=md, parallel=False)
            result = ss.sget()

        assert result["percentile5_mass"] <= result["percentile95_mass"]


# ---------------------------------------------------------------------------
# Sequence_Statistics – get (caching)
# ---------------------------------------------------------------------------

class TestSequenceStatisticsGet:
    def test_writes_json_cache(self, md, tmp_path):
        md.WORKING_DIR = str(tmp_path)
        mock_frames = [np.zeros((50, 50), dtype=np.uint8)] * 2
        fake_result = _synthetic_locate_result()

        with patch("trackpy.locate", return_value=fake_result):
            ss = fstats.Sequence_Statistics(mock_frames, m_metadata=md, parallel=False)
            ss.get()

        assert (tmp_path / "seq_stats.json").exists()

    def test_reads_from_cache_on_second_call(self, md, tmp_path):
        md.WORKING_DIR = str(tmp_path)
        mock_frames = [np.zeros((50, 50), dtype=np.uint8)] * 2
        fake_result = _synthetic_locate_result()

        with patch("trackpy.locate", return_value=fake_result) as mock_locate:
            ss = fstats.Sequence_Statistics(mock_frames, m_metadata=md, parallel=False)
            first = ss.get()
            call_count_after_first = mock_locate.call_count
            second = ss.get()
            call_count_after_second = mock_locate.call_count

        # tp.locate should not be called again when reading from cache
        assert call_count_after_second == call_count_after_first
        assert first == second

    def test_force_recompute_ignores_cache(self, md, tmp_path):
        md.WORKING_DIR = str(tmp_path)
        mock_frames = [np.zeros((50, 50), dtype=np.uint8)] * 2
        fake_result = _synthetic_locate_result()

        with patch("trackpy.locate", return_value=fake_result) as mock_locate:
            ss = fstats.Sequence_Statistics(mock_frames, m_metadata=md, parallel=False)
            ss.get()
            calls_first = mock_locate.call_count
            ss.get(force_recompute=True)
            calls_second = mock_locate.call_count

        assert calls_second > calls_first


# ---------------------------------------------------------------------------
# Frame_Statistics – histogram
# ---------------------------------------------------------------------------

class TestFrameStatistics:
    def test_histogram_saves_files(self, md, tmp_path):
        md.WORKING_DIR = str(tmp_path)
        df0 = _synthetic_locate_result()
        fs = fstats.Frame_Statistics(df0, m_metadata=md)
        fs.histogram("mass", save=True)
        assert (tmp_path / "frame_hist_mass.pdf").exists()
        assert (tmp_path / "frame_hist_mass.png").exists()

    def test_custom_bins(self, md, tmp_path):
        md.WORKING_DIR = str(tmp_path)
        df0 = _synthetic_locate_result()
        fs = fstats.Frame_Statistics(df0, m_metadata=md)
        fs.histogram("mass", save=True, bins=10)  # should not raise
