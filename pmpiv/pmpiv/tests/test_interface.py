#!/usr/bin/env python3
"""Tests for pmpiv.interface (Interface_Tracking and module numerics)."""
import numpy as np
import pandas as pd
import pytest

import pmpiv
from pmpiv import interface
from conftest import _synthetic_channel_image, _arc_y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step_edge_image(shape=(200, 200), edge_y=100, low=0.0, high=255.0):
    """Image with a horizontal step: rows < edge_y are `low`, rows >= edge_y are `high`."""
    img = np.full(shape, low, dtype=float)
    img[edge_y:, :] = high
    return img


LEGACY = dict(preprocess='gaussian', fit_model='poly')

WALLS = ((100, 112), (400, 412), (700, 712))
ARC = {"cx": 556, "cy": 150, "r": 200, "thickness": 20}
X_SEARCH = (420, 692)
WALL_X = (412, 700)
BAND = (250, 400)


def _arc_frames(n_frames, dy=1.0, start=0, empty=()):
    """Synthetic channel frames with the arc moving dy px/frame; frames in `empty` have no arc/beads."""
    frames = []
    for f in range(n_frames):
        if f in empty:
            frames.append(_synthetic_channel_image(walls=WALLS, arcs=(), beads_in=(), seed=f))
        else:
            arc = dict(ARC, cy=ARC["cy"] + dy * (start + f))
            frames.append(_synthetic_channel_image(walls=WALLS, arcs=(arc,), beads_in=(1,), seed=f))
    return frames


def _tracker(md, **kw):
    params = dict(x_search=X_SEARCH, y_search=BAND, wall_x=WALL_X, verbose=False, savgol_window=5)
    params.update(kw)
    return interface.Interface_Tracking(m_metadata=md, **params)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestInterfaceTrackingInit:
    def test_no_metadata_raises(self):
        with pytest.raises(FileNotFoundError):
            interface.Interface_Tracking(x_search=(0, 10), y_search=(0, 10))

    def test_missing_x_search_raises(self, md):
        with pytest.raises(ValueError):
            interface.Interface_Tracking(m_metadata=md, x_search=None, y_search=(0, 10))

    def test_missing_y_search_raises(self, md):
        with pytest.raises(ValueError):
            interface.Interface_Tracking(m_metadata=md, x_search=(0, 10), y_search=None)

    def test_invalid_bounds_raises(self, md):
        with pytest.raises(ValueError):
            interface.Interface_Tracking(m_metadata=md, x_search=(10, 5), y_search=(0, 10))

    def test_invalid_preprocess_raises(self, md):
        with pytest.raises(ValueError):
            interface.Interface_Tracking(m_metadata=md, x_search=(0, 10), y_search=(0, 10), preprocess='magic')

    def test_invalid_fit_model_raises(self, md):
        with pytest.raises(ValueError):
            interface.Interface_Tracking(m_metadata=md, x_search=(0, 10), y_search=(0, 10), fit_model='magic')

    def test_valid_construction(self, md):
        it = interface.Interface_Tracking(m_metadata=md, x_search=(0, 10), y_search=(20, 30))
        assert it.x_search == (0, 10)
        assert it.y_search == (20, 30)
        assert it.m_metadata is md
        assert it.active is True

    def test_poly_default_percentile_trim(self, md):
        it = interface.Interface_Tracking(m_metadata=md, x_search=(0, 10), y_search=(0, 10), fit_model='poly')
        assert it.percentile_trim == (10, 90)
        it2 = interface.Interface_Tracking(m_metadata=md, x_search=(0, 10), y_search=(0, 10))
        assert it2.percentile_trim is None


# ---------------------------------------------------------------------------
# Module numerics
# ---------------------------------------------------------------------------

class TestPreprocessBand:
    def test_shapes_nlm(self):
        img = _synthetic_channel_image()
        band, y0 = interface.preprocess_band(img, (50, 150), (20, 120), mode='nlm_highpass')
        assert band.shape == (100, 100)
        assert y0 == 20

    def test_shapes_gaussian(self):
        img = _synthetic_channel_image()
        band, y0 = interface.preprocess_band(img, (50, 150), (20, 120), mode='gaussian', blur_sigma=5)
        assert band.shape == (100, 100)
        assert y0 == 20

    def test_bad_mode_raises(self):
        img = _synthetic_channel_image()
        with pytest.raises(ValueError):
            interface.preprocess_band(img, (50, 150), (20, 120), mode='magic')


class TestDetectRidge:
    def test_ridge_at_band_centre(self):
        rng = np.random.default_rng(0)
        band = rng.normal(0, 1.0, (150, 100))
        band[60:80, :] -= 50.0
        x_idx, y_idx, snr = interface.detect_ridge(band, ridge_sigma=4)
        assert len(x_idx) >= 90
        assert abs(np.median(y_idx) - 69.5) <= 3
        assert np.median(snr) > 5


class TestCircleFits:
    def _points(self, cx=10.0, cy=-5.0, r=30.0, n=40, seed=0):
        rng = np.random.default_rng(seed)
        t = np.linspace(0.2, np.pi - 0.2, n)
        x = cx + r * np.cos(t)
        y = cy + r * np.sin(t)
        return x, y, rng

    def test_kasa_exact(self):
        x, y, _ = self._points()
        cx, cy, r = interface.fit_circle_kasa(x, y)
        assert cx == pytest.approx(10.0, abs=1e-6)
        assert cy == pytest.approx(-5.0, abs=1e-6)
        assert r == pytest.approx(30.0, abs=1e-6)

    def test_kasa_collinear_returns_none(self):
        x = np.arange(10, dtype=float)
        assert interface.fit_circle_kasa(x, 2 * x + 1) is None

    def test_kasa_nan_returns_none(self):
        assert interface.fit_circle_kasa([1.0, np.nan, 3.0], [1.0, 2.0, 3.0]) is None

    def test_ransac_with_outliers(self):
        x, y, rng = self._points(n=60)
        n_out = 18
        idx = rng.choice(len(x), n_out, replace=False)
        y = y.copy()
        y[idx] += rng.uniform(15, 40, n_out)
        res = interface.fit_circle_ransac(x, y, n_iter=100, tol_px=1.0, rng=1)
        assert res is not None
        assert res['cx'] == pytest.approx(10.0, abs=0.5)
        assert res['cy'] == pytest.approx(-5.0, abs=0.5)
        assert res['r'] == pytest.approx(30.0, abs=0.5)
        assert res['inlier_mask'].sum() >= len(x) - n_out
        assert res['residual_rms'] < 0.5

    def test_ransac_collinear_returns_none(self):
        x = np.arange(20, dtype=float)
        assert interface.fit_circle_ransac(x, 3 * x - 2) is None

    def test_circle_arc_lower_branch(self):
        y = interface.circle_arc(0.0, 0.0, 10.0, np.array([0.0, 6.0, 11.0]))
        assert y[0] == pytest.approx(10.0)
        assert y[1] == pytest.approx(8.0)
        assert np.isnan(y[2])


class TestContactAngle:
    def test_sixty_degrees(self):
        assert interface.contact_angle_deg(0.0, 0.0, 10.0, 5.0) == pytest.approx(60.0, abs=1e-9)

    def test_flat_is_ninety(self):
        assert interface.contact_angle_deg(0.0, 0.0, 10.0, 0.0) == pytest.approx(90.0)

    def test_hemispherical_is_zero(self):
        assert interface.contact_angle_deg(0.0, 0.0, 10.0, 10.0) == pytest.approx(0.0)

    def test_outside_is_nan(self):
        assert np.isnan(interface.contact_angle_deg(0.0, 0.0, 10.0, 11.0))


# ---------------------------------------------------------------------------
# detect_frame / fit_frame (legacy contracts)
# ---------------------------------------------------------------------------

class TestDetectFrame:
    def test_recovers_step_edge_position(self, md):
        img = _step_edge_image(edge_y=100)
        it = interface.Interface_Tracking(
            m_metadata=md, x_search=(50, 150), y_search=(0, 199), blur_sigma=5, **LEGACY,
        )
        x_valid, y_valid = it.detect_frame(img)

        assert len(x_valid) > 0
        assert np.median(y_valid) == pytest.approx(100, abs=3)

    def test_returns_columns_within_roi(self, md):
        img = _step_edge_image(edge_y=100)
        it = interface.Interface_Tracking(
            m_metadata=md, x_search=(50, 150), y_search=(0, 199), blur_sigma=5, **LEGACY,
        )
        x_valid, _ = it.detect_frame(img)

        assert x_valid.min() >= 50
        assert x_valid.max() < 150

    def test_ridge_detects_synthetic_arc(self, md):
        img = _synthetic_channel_image(walls=WALLS, arcs=(ARC,), beads_in=(1,))
        it = _tracker(md)
        x_valid, y_valid = it.detect_frame(img)
        # cv2.circle draws the band centred on the radius
        expected = _arc_y(x_valid, ARC["cx"], ARC["cy"], ARC["r"])
        assert len(x_valid) >= 0.6 * (X_SEARCH[1] - X_SEARCH[0])
        assert np.median(np.abs(y_valid - expected)) <= 3


class TestFitFrame:
    def test_fits_linear_data(self, md):
        it = interface.Interface_Tracking(
            m_metadata=md, x_search=(50, 150), y_search=(0, 199), poly_deg=1, **LEGACY,
        )
        x_valid = np.arange(50, 150, dtype=float)
        y_valid = 2.0 * x_valid + 3.0

        result = it.fit_frame(x_valid, y_valid)
        assert result is not None

        x_arc, y_arc, y_median = result
        assert y_arc[0] == pytest.approx(2.0 * x_arc[0] + 3.0, abs=1e-6)
        assert y_arc[-1] == pytest.approx(2.0 * x_arc[-1] + 3.0, abs=1e-6)

    def test_degenerate_input_returns_none(self, md):
        it = interface.Interface_Tracking(
            m_metadata=md, x_search=(0, 10), y_search=(0, 10), poly_deg=1, **LEGACY,
        )
        x_valid = np.array([1.0, 2.0, np.nan, 4.0])
        y_valid = np.array([1.0, 2.0, 3.0, 4.0])

        assert it.fit_frame(x_valid, y_valid) is None

    def test_circle_fit_full(self, md):
        it = _tracker(md)
        x = np.arange(*X_SEARCH, dtype=float)
        y = _arc_y(x, ARC["cx"], ARC["cy"], ARC["r"])
        fit = it.fit_frame_full(x, y)
        assert fit is not None
        assert fit['cx'] == pytest.approx(ARC["cx"], abs=0.5)
        assert fit['r'] == pytest.approx(ARC["r"], abs=0.5)
        assert fit['y_apex'] == pytest.approx(ARC["cy"] + ARC["r"], abs=0.5)
        assert 0 < fit['contact_left'] < 90
        assert fit['contact_left'] == pytest.approx(fit['contact_right'], abs=0.5)

    def test_circle_fit_nan_returns_none(self, md):
        it = _tracker(md)
        assert it.fit_frame(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0])) is None


# ---------------------------------------------------------------------------
# compute_velocity
# ---------------------------------------------------------------------------

class TestComputeVelocity:
    def test_velocity_accounts_for_frame_gaps(self, md):
        it = interface.Interface_Tracking(
            m_metadata=md, x_search=(0, 10), y_search=(0, 10), rolling_window=3,
        )
        df = pd.DataFrame({
            'frame': [0, 1, 3, 4],
            'y_median_px': [100.0, 110.0, 130.0, 140.0],
        })

        result = it.compute_velocity(df)
        expected_velocity = 10.0 * md.PIXELSIZE * md.FPS

        row_1frame_gap = result[result['frame'] == 1].iloc[0]
        row_2frame_gap = result[result['frame'] == 3].iloc[0]

        assert row_1frame_gap['velocity_m_s'] == pytest.approx(expected_velocity, rel=1e-6)
        assert row_2frame_gap['velocity_m_s'] == pytest.approx(expected_velocity, rel=1e-6)

    def test_first_row_velocity_is_nan(self, md):
        it = interface.Interface_Tracking(m_metadata=md, x_search=(0, 10), y_search=(0, 10))
        df = pd.DataFrame({'frame': [0, 1], 'y_median_px': [100.0, 110.0]})
        result = it.compute_velocity(df)
        assert np.isnan(result.iloc[0]['velocity_m_s'])

    def test_savgol_velocity_linear_motion(self, md):
        it = interface.Interface_Tracking(m_metadata=md, x_search=(0, 10), y_search=(0, 10),
                                          savgol_window=5, max_misses=2)
        n = 12
        df = pd.DataFrame({'frame': np.arange(n), 'y_median_px': 100.0 + 2.0 * np.arange(n),
                           'radius_px': np.full(n, 200.0)})
        result = it.compute_velocity(df)
        expected = 2.0 * md.PIXELSIZE * md.FPS
        assert np.allclose(result['velocity_savgol_m_s'], expected, rtol=1e-6)
        assert np.allclose(result['y_smooth_px'], df['y_median_px'], atol=1e-6)
        assert result['curvature_1_m'].iloc[0] == pytest.approx(1.0 / (200.0 * md.PIXELSIZE))

    def test_nan_position_propagates(self, md):
        it = interface.Interface_Tracking(m_metadata=md, x_search=(0, 10), y_search=(0, 10),
                                          savgol_window=5, max_misses=3)
        y = 100.0 + 2.0 * np.arange(12)
        y[5] = np.nan
        df = pd.DataFrame({'frame': np.arange(12), 'y_median_px': y})
        result = it.compute_velocity(df)
        for c in ('velocity_m_s', 'velocity_rolling_mean_m_s', 'velocity_savgol_m_s'):
            assert np.isnan(result.loc[5, c])
        assert np.isfinite(result.loc[8, 'velocity_savgol_m_s'])


# ---------------------------------------------------------------------------
# step / track
# ---------------------------------------------------------------------------

class TestTrack:
    def test_tracks_moving_edge_legacy(self, md):
        n_frames = 5
        frames = [_step_edge_image(edge_y=100 - 5 * f) for f in range(n_frames)]

        it = interface.Interface_Tracking(
            m_metadata=md, x_search=(50, 150), y_search=(50, 150),
            blur_sigma=3, poly_deg=1, verbose=False, **LEGACY,
        )
        df = it.track(frames)

        assert len(df) == n_frames
        assert (df['status'] == 'ok').all()
        assert set(it._arcs.keys()) == set(df['frame'])
        assert df['y_median_px'].iloc[0] > df['y_median_px'].iloc[-1]

    def test_tracks_moving_arc(self, md):
        n = 8
        frames = _arc_frames(n, dy=1.0)
        it = _tracker(md)
        df = it.track(frames)

        assert list(df.columns) == interface.RECORD_COLUMNS
        assert len(df) == n
        assert (df['status'] == 'ok').all(), df[['frame', 'status', 'n_valid', 'snr', 'residual_px']]
        expected_apex = ARC["cy"] + ARC["r"] + np.arange(n)
        assert np.all(np.abs(df['y_apex_px'].to_numpy() - expected_apex) <= 3)
        assert np.all(np.abs(df['radius_px'].to_numpy() - ARC["r"]) <= 0.1 * ARC["r"])
        assert np.all((df['contact_angle_left_deg'] > 0) & (df['contact_angle_left_deg'] < 90))

    def test_weak_then_recover(self, md):
        frames = _arc_frames(9, dy=0.0, empty=(3, 4, 5))
        it = _tracker(md, max_misses=5)
        df = it.track(frames)

        assert list(df['status']) == ['ok'] * 3 + ['weak'] * 3 + ['ok'] * 3
        assert df.loc[3:5, 'y_median_px'].isna().all()
        assert it.active is True

    def test_lost_after_max_misses(self, md):
        frames = _arc_frames(12, dy=0.0, empty=range(3, 12))
        it = _tracker(md, max_misses=3)
        df = it.track(frames)

        assert df['status'].iloc[-1] == 'lost'
        assert len(df) == 7            # 3 ok + 3 weak + terminal
        assert it.active is False
        assert it.status == 'lost'

    def test_out_of_frame(self, md):
        n = 16
        frames = _arc_frames(n, dy=25.0)
        it = _tracker(md)
        df = it.track(frames)

        assert df['status'].iloc[-1] == 'out_of_frame'
        assert len(df) < n
        assert it.active is False

    def test_step_returns_none_when_inactive(self, md):
        frames = _arc_frames(2, dy=0.0)
        it = _tracker(md)
        it.active = False
        assert it.step(frames[0], 0) is None


# ---------------------------------------------------------------------------
# plot_annotated_tiffs
# ---------------------------------------------------------------------------

class TestPlotAnnotatedTiffs:
    def test_requires_track_first(self, md):
        it = interface.Interface_Tracking(m_metadata=md, x_search=(0, 10), y_search=(0, 10))
        with pytest.raises(ValueError):
            it.plot_annotated_tiffs(pd.DataFrame(), 'outfolder', [np.zeros((10, 10))])

    def test_writes_tiff_files(self, md, tmp_path):
        frames = _arc_frames(3, dy=1.0)
        it = _tracker(md)
        df = it.track(frames)

        outfolder = tmp_path / 'tiffs_interface'
        it.plot_annotated_tiffs(df, str(outfolder), frames, parallel=False)

        for fidx in df['frame']:
            assert (outfolder / f'frame_{fidx:06d}.tif').exists()
