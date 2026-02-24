#!/usr/bin/env python3
"""Tests for pmpiv.motion_stats.Motion_Statistics."""
import pytest
import numpy as np
import pandas as pd

import pmpiv
from pmpiv import motion_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linear_trajectory(
    n_particles: int = 3,
    n_frames: int = 5,
    dx: float = 3.0,
    dy: float = 2.0,
) -> pd.DataFrame:
    """Particles moving at constant (dx, dy) per frame."""
    rows = []
    for p in range(n_particles):
        for f in range(n_frames):
            rows.append(
                {
                    "x": p * 100.0 + f * dx,
                    "y": p * 50.0 + f * dy,
                    "frame": f,
                    "particle": p,
                }
            )
    df = pd.DataFrame(rows)
    return df.set_index("frame", drop=False)


# ---------------------------------------------------------------------------
# Sanity checks on construction
# ---------------------------------------------------------------------------

class TestMotionStatisticsInit:
    def test_no_metadata_raises(self):
        df = _linear_trajectory()
        with pytest.raises(FileNotFoundError):
            motion_stats.Motion_Statistics(df)

    def test_valid_construction(self, md):
        df = _linear_trajectory()
        ms = motion_stats.Motion_Statistics(df, m_metadata=md)
        assert ms.m_df is df


# ---------------------------------------------------------------------------
# displacement_2frames
# ---------------------------------------------------------------------------

class TestDisplacement2Frames:
    def test_returns_dataframe(self, md):
        df = _linear_trajectory()
        ms = motion_stats.Motion_Statistics(df, m_metadata=md)
        result = ms.displacement_2frames(0, 1)
        assert isinstance(result, pd.DataFrame)

    def test_has_displacement_columns(self, md):
        df = _linear_trajectory()
        ms = motion_stats.Motion_Statistics(df, m_metadata=md)
        result = ms.displacement_2frames(0, 1)
        for col in ("dx", "dy", "dr"):
            assert col in result.columns

    def test_same_frame_raises(self, md):
        df = _linear_trajectory()
        ms = motion_stats.Motion_Statistics(df, m_metadata=md)
        with pytest.raises(ValueError):
            ms.displacement_2frames(2, 2)

    def test_reverse_order_raises(self, md):
        df = _linear_trajectory()
        ms = motion_stats.Motion_Statistics(df, m_metadata=md)
        with pytest.raises(ValueError):
            ms.displacement_2frames(3, 1)

    def test_displacement_magnitude(self, md):
        df = _linear_trajectory(dx=3.0, dy=4.0)  # dr = 5.0 per frame
        ms = motion_stats.Motion_Statistics(df, m_metadata=md)
        result = ms.displacement_2frames(0, 1)
        assert result["dr"].mean() == pytest.approx(5.0, abs=0.01)


# ---------------------------------------------------------------------------
# mean_displacement_2frames / abs_mean_displacement_2frames
# ---------------------------------------------------------------------------

class TestMeanDisplacement:
    @pytest.fixture
    def ms(self, md):
        return motion_stats.Motion_Statistics(_linear_trajectory(dx=3.0, dy=2.0), m_metadata=md)

    def test_mean_displacement_keys(self, ms):
        result = ms.mean_displacement_2frames(0, 1)
        assert set(result.keys()) == {"dx", "dy", "dr"}

    def test_mean_displacement_values(self, ms):
        result = ms.mean_displacement_2frames(0, 1)
        assert result["dx"] == pytest.approx(3.0, abs=0.01)
        assert result["dy"] == pytest.approx(2.0, abs=0.01)

    def test_abs_mean_displacement_positive(self, ms):
        result = ms.abs_mean_displacement_2frames(0, 1)
        assert result["dx"] >= 0
        assert result["dy"] >= 0
        assert result["dr"] >= 0

    def test_max_displacement_keys(self, ms):
        result = ms.max_displacement_2frames(0, 1)
        assert set(result.keys()) == {"dx", "dy", "dr"}


# ---------------------------------------------------------------------------
# mean_displacement_allframes / mean_velocity_allframes
# ---------------------------------------------------------------------------

class TestAllFramesStats:
    @pytest.fixture
    def ms(self, md):
        return motion_stats.Motion_Statistics(_linear_trajectory(dx=3.0, dy=2.0), m_metadata=md)

    def test_mean_displacement_allframes_keys(self, ms):
        result = ms.mean_displacement_allframes()
        assert set(result.keys()) == {"dx", "dy", "dr"}

    def test_mean_displacement_allframes_values(self, ms):
        result = ms.mean_displacement_allframes()
        assert result["dx"] == pytest.approx(3.0, abs=0.1)
        assert result["dy"] == pytest.approx(2.0, abs=0.1)

    def test_velocity_equals_displacement_times_fps_pixelsize(self, md):
        df = _linear_trajectory(dx=3.0, dy=2.0)
        ms = motion_stats.Motion_Statistics(df, m_metadata=md)
        disp = ms.mean_displacement_allframes()
        vel = ms.mean_velocity_allframes()
        scale = md.FPS * md.PIXELSIZE
        assert vel["dx"] == pytest.approx(disp["dx"] * scale, rel=1e-6)
        assert vel["dy"] == pytest.approx(disp["dy"] * scale, rel=1e-6)
        assert vel["dr"] == pytest.approx(disp["dr"] * scale, rel=1e-6)

    def test_abs_displacement_non_negative(self, ms):
        result = ms.mean_abs_displacement_allframes()
        assert result["dx"] >= 0
        assert result["dy"] >= 0
        assert result["dr"] >= 0

    def test_abs_velocity_non_negative(self, ms):
        result = ms.mean_abs_velocity_allframes()
        assert result["dx"] >= 0
        assert result["dy"] >= 0
        assert result["dr"] >= 0

    def test_freq_not_one_raises(self, ms):
        with pytest.raises(NotImplementedError):
            ms.mean_displacement_allframes(freq=2)

    def test_abs_freq_not_one_raises(self, ms):
        with pytest.raises(NotImplementedError):
            ms.mean_abs_displacement_allframes(freq=2)
