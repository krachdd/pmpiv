#!/usr/bin/env python3
"""Tests for pmpiv.annotations: Annotation_Handler and Annotation_Reader."""
import pytest
import pandas as pd

import pmpiv
from pmpiv import annotations


pycocotools = pytest.importorskip("pycocotools", reason="pycocotools not installed")


class TestAnnotationHandlerInit:
    def test_missing_json_raises(self, md):
        with pytest.raises(FileNotFoundError):
            annotations.Annotation_Handler("/nonexistent/file.json", m_metadata=md)

    def test_invalid_json_type_raises(self, synthetic_coco_json, md):
        with pytest.raises(ValueError, match="json_type"):
            annotations.Annotation_Handler(
                synthetic_coco_json, m_metadata=md, json_type="INVALID"
            )

    def test_coco_type_warns(self, synthetic_coco_json, md, capsys):
        annotations.Annotation_Handler(
            synthetic_coco_json, m_metadata=md, json_type="COCO"
        )
        captured = capsys.readouterr()
        assert "super slow" in captured.out

    def test_no_metadata_raises(self, synthetic_coco_json):
        with pytest.raises(FileNotFoundError):
            annotations.Annotation_Handler(synthetic_coco_json)


class TestAnnotations2DF:
    def test_returns_dataframe(self, synthetic_coco_json, md):
        ah = annotations.Annotation_Handler(synthetic_coco_json, m_metadata=md)
        df = ah.annotations2DF()
        assert isinstance(df, pd.DataFrame)

    def test_has_x_y_columns(self, synthetic_coco_json, md):
        ah = annotations.Annotation_Handler(synthetic_coco_json, m_metadata=md)
        df = ah.annotations2DF()
        assert "x" in df.columns
        assert "y" in df.columns

    def test_coco_range_y_contains_tuples(self, synthetic_coco_json, md):
        ah = annotations.Annotation_Handler(
            synthetic_coco_json, m_metadata=md, json_type="COCO_range"
        )
        df = ah.annotations2DF()
        # Each y value is a list of (min, max) tuples
        for val in df["y"]:
            assert isinstance(val, list)
            for tpl in val:
                assert len(tpl) == 2
                assert tpl[0] <= tpl[1]

    def test_x_values_within_image_bounds(self, synthetic_coco_json, md):
        ah = annotations.Annotation_Handler(synthetic_coco_json, m_metadata=md)
        df = ah.annotations2DF()
        # Image is 100×100; annotation rectangle is at x∈[20,40]
        assert df["x"].min() >= 0
        assert df["x"].max() <= 100

    def test_writeto_creates_csv(self, synthetic_coco_json, md, tmp_path):
        out = str(tmp_path / "annotation.csv")
        ah = annotations.Annotation_Handler(synthetic_coco_json, m_metadata=md)
        ah.annotations2DF(writeto=out)
        assert (tmp_path / "annotation.csv").exists()


class TestAnnotationHandlerArea:
    def test_area_positive(self, synthetic_coco_json, md):
        ah = annotations.Annotation_Handler(synthetic_coco_json, m_metadata=md)
        area = ah.area()
        assert area > 0

    def test_area_in_square_meters(self, synthetic_coco_json, md):
        ah = annotations.Annotation_Handler(synthetic_coco_json, m_metadata=md)
        area = ah.area()
        # 400 px² × (1.3e-6 m/px)² ≈ 6.76e-10 m²
        expected = 400.0 * (md.PIXELSIZE ** 2)
        assert area == pytest.approx(expected, rel=0.01)

    def test_area_callable_twice(self, synthetic_coco_json, md):
        """area() must not overwrite itself (previously stored as self.area)."""
        ah = annotations.Annotation_Handler(synthetic_coco_json, m_metadata=md)
        a1 = ah.area()
        a2 = ah.area()  # would fail if area() overwrote the method with a float
        assert a1 == a2

    def test_area_stored_as_private(self, synthetic_coco_json, md):
        """Result is stored as _area, not area (no method overwrite)."""
        ah = annotations.Annotation_Handler(synthetic_coco_json, m_metadata=md)
        ah.area()
        assert callable(ah.area)  # method still callable
        assert hasattr(ah, "_area")


class TestAnnotationHandlerVolume:
    def test_volume_positive(self, synthetic_coco_json, md):
        ah = annotations.Annotation_Handler(synthetic_coco_json, m_metadata=md)
        vol = ah.volume()
        assert vol > 0

    def test_volume_equals_area_times_height(self, synthetic_coco_json, md):
        ah = annotations.Annotation_Handler(synthetic_coco_json, m_metadata=md)
        area = ah.area()
        vol = ah.volume()
        assert vol == pytest.approx(area * md.HEIGHT, rel=1e-9)


class TestAnnotationReader:
    def test_missing_csv_raises(self, md):
        with pytest.raises(FileNotFoundError):
            annotations.Annotation_Reader("/nonexistent/file.csv", m_metadata=md)

    def test_invalid_json_type_raises(self, synthetic_coco_json, md, tmp_path):
        with pytest.raises(ValueError, match="json_type"):
            annotations.Annotation_Reader(
                str(tmp_path / "dummy.csv"), m_metadata=md, json_type="BAD"
            )

    def test_read_roundtrip_coco_range(self, synthetic_coco_json, md, tmp_path):
        out_csv = str(tmp_path / "annotation.csv")
        ah = annotations.Annotation_Handler(synthetic_coco_json, m_metadata=md)
        orig_df = ah.annotations2DF(writeto=out_csv)

        reader = annotations.Annotation_Reader(
            out_csv, m_metadata=md, json_type="COCO_range"
        )
        loaded = reader.read()
        assert list(loaded["x"]) == list(orig_df["x"])

    def test_read_y_parsed_as_list(self, synthetic_coco_json, md, tmp_path):
        out_csv = str(tmp_path / "annotation.csv")
        ah = annotations.Annotation_Handler(synthetic_coco_json, m_metadata=md)
        ah.annotations2DF(writeto=out_csv)

        reader = annotations.Annotation_Reader(out_csv, m_metadata=md)
        loaded = reader.read()
        for val in loaded["y"]:
            assert isinstance(val, list)


class TestAnnotationHandlerBigCapillary:
    """Integration-level checks using the real test JSON file."""

    def test_annotations2df_nonempty(self, big_coco_json, md):
        ah = annotations.Annotation_Handler(big_coco_json, m_metadata=md)
        df = ah.annotations2DF()
        assert len(df) > 0

    def test_area_matches_json(self, big_coco_json, md):
        ah = annotations.Annotation_Handler(big_coco_json, m_metadata=md)
        area_m2 = ah.area()
        # JSON annotation area is 284001.5 px²
        expected = 284001.5 * (md.PIXELSIZE ** 2)
        assert area_m2 == pytest.approx(expected, rel=1e-3)
