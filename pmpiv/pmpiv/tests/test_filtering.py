#!/usr/bin/env python3
"""Tests for pmpiv.filtering: Filtering, Annotation_Filtering, filter_static, filter_stubs."""
import pytest
import numpy as np
import pandas as pd

import pmpiv
from pmpiv import filtering


# ---------------------------------------------------------------------------
# Filtering.percentile_filter
# ---------------------------------------------------------------------------

class TestPercentileFilter:
    """Filtering.percentile_filter keeps particles within the specified percentile band."""

    def _make_stats(self, df, percentile):
        """Build a minimal fstats dict for the given percentile."""
        lo, hi = percentile, 100 - percentile
        return {
            f"percentile{lo}_mass": float(np.percentile(df["mass"], lo)),
            f"percentile{hi}_mass": float(np.percentile(df["mass"], hi)),
        }

    def test_reduces_dataframe(self, trajectory_df, md):
        stats = self._make_stats(trajectory_df, 5)
        f = filtering.Filtering(trajectory_df, m_metadata=md)
        result = f.percentile_filter(5, ["mass"], stats)
        assert len(result) < len(trajectory_df)

    def test_mass_within_bounds(self, trajectory_df, md):
        stats = self._make_stats(trajectory_df, 5)
        f = filtering.Filtering(trajectory_df, m_metadata=md)
        result = f.percentile_filter(5, ["mass"], stats)
        assert result["mass"].min() > stats["percentile5_mass"]
        assert result["mass"].max() < stats["percentile95_mass"]

    def test_invalid_percentile_raises(self, trajectory_df, md):
        f = filtering.Filtering(trajectory_df, m_metadata=md)
        with pytest.raises(ValueError, match="Perctile"):
            f.percentile_filter(50, ["mass"], {"percentile50_mass": 0})

    def test_invalid_parameter_raises(self, trajectory_df, md):
        f = filtering.Filtering(trajectory_df, m_metadata=md)
        with pytest.raises(ValueError, match="Parameters"):
            f.percentile_filter(5, ["flux"], {})

    def test_empty_stats_raises(self, trajectory_df, md):
        f = filtering.Filtering(trajectory_df, m_metadata=md)
        with pytest.raises(ValueError, match="empty"):
            f.percentile_filter(5, ["mass"], {})

    def test_verbose_typo_fixed(self, trajectory_df, md):
        """Ensure the 'verbose' attribute is correctly set (not 'vebose')."""
        f = filtering.Filtering(trajectory_df, m_metadata=md)
        assert hasattr(f, "verbose")
        assert not hasattr(f, "vebose")


# ---------------------------------------------------------------------------
# filter_static
# ---------------------------------------------------------------------------

class TestFilterStatic:
    def test_static_particle_removed(self, mixed_df, md):
        result = filtering.filter_static(mixed_df, m_metadata=md)
        assert 0 not in result["particle"].values

    def test_moving_particle_kept(self, mixed_df, md):
        result = filtering.filter_static(mixed_df, m_metadata=md)
        assert 1 in result["particle"].values

    def test_frame_index_preserved(self, mixed_df, md):
        result = filtering.filter_static(mixed_df, m_metadata=md)
        assert result.index.name == "frame"

    def test_missing_columns_raises(self, md):
        bad_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises((ValueError, KeyError)):
            filtering.filter_static(bad_df, m_metadata=md)

    def test_no_metadata_raises(self, mixed_df):
        with pytest.raises(FileNotFoundError):
            filtering.filter_static(mixed_df)


# ---------------------------------------------------------------------------
# filter_stubs
# ---------------------------------------------------------------------------

class TestFilterStubs:
    def test_short_trajectory_removed(self, stubs_df, md):
        result = filtering.filter_stubs(stubs_df, m_metadata=md)
        assert 0 not in result["particle"].values

    def test_long_trajectory_kept(self, stubs_df, md):
        result = filtering.filter_stubs(stubs_df, m_metadata=md)
        assert 1 in result["particle"].values

    def test_no_metadata_raises(self, stubs_df):
        with pytest.raises(FileNotFoundError):
            filtering.filter_stubs(stubs_df)

    def test_missing_columns_raises(self, md):
        bad_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises((ValueError, KeyError)):
            filtering.filter_stubs(bad_df, m_metadata=md)


# ---------------------------------------------------------------------------
# Annotation_Filtering
# ---------------------------------------------------------------------------

class TestAnnotationFiltering:
    """Annotation_Filtering removes / extracts particles by spatial region."""

    def test_missing_x_column_raises(self, md):
        bad_df = pd.DataFrame({"y": [1.0], "frame": [0], "particle": [0]})
        with pytest.raises(ValueError, match="missing"):
            filtering.Annotation_Filtering(bad_df, m_metadata=md)

    def test_missing_y_column_raises(self, md):
        bad_df = pd.DataFrame({"x": [1.0], "frame": [0], "particle": [0]})
        with pytest.raises(ValueError, match="missing"):
            filtering.Annotation_Filtering(bad_df, m_metadata=md)

    def test_remove_reduces_count(self, annotation_region_df, annotation_filter_df, md):
        af = filtering.Annotation_Filtering(annotation_region_df, m_metadata=md)
        result = af.remove(annotation_filter_df)
        assert len(result) < len(annotation_region_df)

    def test_remove_drops_inside_particles(self, annotation_region_df, annotation_filter_df, md):
        af = filtering.Annotation_Filtering(annotation_region_df, m_metadata=md)
        result = af.remove(annotation_filter_df)
        # particle 0 (inside region x≈30, y≈30) must be gone
        assert 0 not in result["particle"].values

    def test_remove_keeps_outside_particles(self, annotation_region_df, annotation_filter_df, md):
        af = filtering.Annotation_Filtering(annotation_region_df, m_metadata=md)
        result = af.remove(annotation_filter_df)
        # particle 1 (outside region x≈80, y≈80) must remain
        assert 1 in result["particle"].values

    def test_extract_keeps_only_inside(self, annotation_region_df, annotation_filter_df, md):
        af = filtering.Annotation_Filtering(annotation_region_df, m_metadata=md)
        result = af.extract(annotation_filter_df)
        assert 0 in result["particle"].values
        assert 1 not in result["particle"].values

    def test_invalid_json_type_remove_raises(self, annotation_region_df, annotation_filter_df, md):
        af = filtering.Annotation_Filtering(annotation_region_df, m_metadata=md)
        with pytest.raises(ValueError):
            af.remove(annotation_filter_df, json_type="INVALID")

    def test_coco_json_type_remove_raises_not_implemented(
        self, annotation_region_df, annotation_filter_df, md
    ):
        af = filtering.Annotation_Filtering(annotation_region_df, m_metadata=md)
        with pytest.raises(NotImplementedError):
            af.remove(annotation_filter_df, json_type="COCO")

    def test_extract_invalid_json_type_raises(self, annotation_region_df, annotation_filter_df, md):
        af = filtering.Annotation_Filtering(annotation_region_df, m_metadata=md)
        with pytest.raises(ValueError):
            af.extract(annotation_filter_df, json_type="COCO")

    def test_filter_df_missing_x_raises(self, annotation_region_df, md):
        bad_filter = pd.DataFrame({"y": [[(0, 10)]]})
        af = filtering.Annotation_Filtering(annotation_region_df, m_metadata=md)
        with pytest.raises(ValueError, match="missing"):
            af.remove(bad_filter)
