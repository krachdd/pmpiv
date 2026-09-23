#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 20 2026

@author: David Krach
         david.krach@mib.uni-stuttgart.de
"""

### HEADER ------------------------------------------------------------------------
from __future__ import division, unicode_literals, print_function

import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')          # non-interactive backend — safe for headless/multiprocessing use
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import savgol_filter
import cv2

import pandas as pd
import multiprocessing

import pmpiv as pmpiv

# OpenCV's internal thread pool can deadlock inside forked Pool workers.
cv2.setNumThreads(1)

###--------------------------------------------------------------------------------

RECORD_COLUMNS = ['frame', 'status', 'y_median_px', 'y_apex_px', 'cx_px', 'cy_px',
                  'radius_px', 'residual_px', 'n_valid', 'snr',
                  'contact_angle_left_deg', 'contact_angle_right_deg',
                  'band_y0', 'band_y1']

PREPROCESS_MODES = ('nlm_highpass', 'gaussian')
FIT_MODELS       = ('circle', 'poly')


# ---------------------------------------------------------------------------------
# Module-level numerics (no metadata needed; reused by interface_discovery)
# ---------------------------------------------------------------------------------

def _to_uint8(crop):
    if crop.dtype == np.uint8:
        return np.ascontiguousarray(crop)
    lo, hi = float(crop.min()), float(crop.max())
    if hi <= lo:
        return np.zeros(crop.shape, dtype = np.uint8)
    return np.ascontiguousarray(((crop - lo) / (hi - lo) * 255.0).astype(np.uint8))


def preprocess_band(img, x_search, y_search, mode = 'nlm_highpass',
                    blur_sigma = 30, nlm_h = 10, highpass_sigma = 60):
    """
    Cut the (x_search, y_search) band out of img and preprocess it.

    The crop is padded before filtering so the filters see real
    neighbouring image content instead of a mirrored edge (blur_sigma /
    highpass_sigma can be large relative to the band).

    mode = 'nlm_highpass': non-local-means denoise (suppresses tracer
    beads while keeping the meniscus band) then subtract a large-scale
    Gaussian background so the band stands out as a negative ridge.
    mode = 'gaussian'    : legacy heavy Gaussian blur.

    Returns (band, y_min) with band.shape == (y_max - y_min, x_max - x_min).
    """
    if mode not in PREPROCESS_MODES:
        raise ValueError(f'mode must be in {PREPROCESS_MODES}')

    x_min, x_max = x_search
    y_min, y_max = y_search

    def padded(margin):
        x0 = max(0, x_min - margin)
        x1 = min(img.shape[1], x_max + margin)
        y0 = max(0, y_min - margin)
        y1 = min(img.shape[0], y_max + margin)
        return img[y0:y1, x0:x1], (y_min - y0, x_min - x0)

    def to_band(arr, off):
        return arr[off[0]:off[0] + (y_max - y_min), off[1]:off[1] + (x_max - x_min)]

    if mode == 'gaussian':
        crop, off = padded(int(3 * blur_sigma))
        return to_band(gaussian_filter(crop.astype(float), sigma = blur_sigma), off), y_min

    # Denoise only the band plus a small margin (NLM is the expensive step);
    # the smooth background comes from a downsampled blur of a wide crop.
    crop, off = padded(32)
    den = cv2.fastNlMeansDenoising(_to_uint8(crop), None, h = float(nlm_h),
                                   templateWindowSize = 7, searchWindowSize = 21).astype(float)
    den_band = to_band(den, off)

    wide, off_w = padded(int(3 * highpass_sigma))
    ds = 4
    small = cv2.resize(wide.astype(np.float32), (max(1, wide.shape[1] // ds), max(1, wide.shape[0] // ds)),
                       interpolation = cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, highpass_sigma / ds))
    bg = cv2.resize(small, (wide.shape[1], wide.shape[0]), interpolation = cv2.INTER_LINEAR).astype(float)
    bg_band = to_band(bg, off_w)

    return den_band - bg_band, y_min


def _column_snr(resp, y_idx):
    """(peak - median) / robust noise, per column."""
    cols = np.arange(resp.shape[1])
    peak = resp[y_idx, cols]
    med  = np.median(resp, axis = 0)
    mad  = np.median(np.abs(resp - med), axis = 0) * 1.4826 + 1e-9
    return (peak - med) / mad


def _continuity_mask(y_idx, continuity_window, continuity_tolerance):
    local_median = median_filter(y_idx, size = continuity_window, mode = 'nearest')
    return np.abs(y_idx - local_median) <= continuity_tolerance


def detect_ridge(band, ridge_sigma = 8, continuity_window = 15, continuity_tolerance = 15):
    """
    Per-column position of a dark band (ridge) in a preprocessed band.

    Returns (x_idx, y_idx, snr) in band coordinates for the columns that
    pass the local-continuity check.
    """
    resp  = -gaussian_filter(band, sigma = ridge_sigma)
    y_all = np.argmax(resp, axis = 0)
    snr   = _column_snr(resp, y_all)
    mask  = _continuity_mask(y_all, continuity_window, continuity_tolerance)
    x_idx = np.arange(band.shape[1])
    return x_idx[mask], y_all[mask], snr[mask]


def detect_gradient(band, continuity_window = 15, continuity_tolerance = 15):
    """
    Legacy detector: per-column argmax of |d/dy| of a blurred band.

    Returns (x_idx, y_idx, snr) in band coordinates.
    """
    resp  = np.abs(np.gradient(band, axis = 0))
    y_all = np.argmax(resp, axis = 0)
    snr   = _column_snr(resp, y_all)
    mask  = _continuity_mask(y_all, continuity_window, continuity_tolerance)
    x_idx = np.arange(band.shape[1])
    return x_idx[mask], y_all[mask], snr[mask]


def percentile_trim(x, y, percentiles = (10, 90)):
    """Drop points whose y lies outside the given percentiles."""
    if len(y) == 0:
        return x, y
    lo, hi = np.percentile(y, percentiles[0]), np.percentile(y, percentiles[1])
    mask = (y >= lo) & (y <= hi)
    return x[mask], y[mask]


def fit_circle_kasa(x, y):
    """
    Algebraic (Kåsa) least-squares circle fit. Returns (cx, cy, r) or None
    if the points are degenerate (collinear, fewer than 3, non-finite).
    """
    x = np.asarray(x, dtype = float)
    y = np.asarray(y, dtype = float)
    if len(x) < 3 or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None

    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    sol, _, rank, _ = np.linalg.lstsq(A, b, rcond = None)
    if rank < 3:
        return None

    D, E, F = sol
    cx, cy = -D / 2.0, -E / 2.0
    r2 = cx ** 2 + cy ** 2 - F
    if not np.isfinite(r2) or r2 <= 0:
        return None

    return float(cx), float(cy), float(np.sqrt(r2))


def fit_circle_ransac(x, y, n_iter = 50, tol_px = 3.0, min_inliers_frac = 0.5, rng = None):
    """
    RANSAC circle fit: minimal samples take one point from each x-third
    (avoids near-collinear triples), the best inlier set is refit with
    fit_circle_kasa. Returns dict(cx, cy, r, residual_rms, inlier_mask)
    or None.
    """
    x = np.asarray(x, dtype = float)
    y = np.asarray(y, dtype = float)
    n = len(x)
    if n < 3 or not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return None

    rng    = np.random.default_rng(rng)
    order  = np.argsort(x)
    thirds = [t for t in np.array_split(order, 3) if len(t) > 0]

    best_mask, best_count = None, 0
    for _ in range(int(n_iter)):
        if len(thirds) == 3:
            idx = [rng.choice(t) for t in thirds]
        else:
            idx = rng.choice(n, size = 3, replace = False)
        c = fit_circle_kasa(x[idx], y[idx])
        if c is None:
            continue
        cx, cy, r = c
        d    = np.abs(np.hypot(x - cx, y - cy) - r)
        mask = d <= tol_px
        if mask.sum() > best_count:
            best_mask, best_count = mask, int(mask.sum())

    if best_mask is None or best_count < max(3, int(np.ceil(min_inliers_frac * n))):
        return None

    c = fit_circle_kasa(x[best_mask], y[best_mask])
    if c is None:
        return None
    cx, cy, r = c
    d = np.abs(np.hypot(x - cx, y - cy) - r)
    inliers = d <= tol_px
    if inliers.sum() < 3:
        return None
    residual_rms = float(np.sqrt(np.mean(d[inliers] ** 2)))

    return dict(cx = cx, cy = cy, r = r, residual_rms = residual_rms, inlier_mask = inliers)


def circle_arc(cx, cy, r, x, branch = +1):
    """y on the circle for the given x (branch=+1: lower arc, larger y). NaN outside |x-cx|>r."""
    dx = np.asarray(x, dtype = float) - cx
    y  = np.full(dx.shape, np.nan)
    inside = np.abs(dx) <= r
    y[inside] = cy + branch * np.sqrt(r ** 2 - dx[inside] ** 2)
    return y


def contact_angle_deg(cx, cy, r, x_wall, branch = +1):
    """
    Angle (deg) between the arc's tangent at x_wall and the (vertical)
    wall, for a circular cap pinned at the wall: 90 deg = flat interface,
    -> 0 deg = hemispherical cap. NaN if the wall is outside the circle.
    """
    dx = abs(float(x_wall) - cx)
    if not np.isfinite(dx) or r <= 0 or dx > r:
        return float('nan')
    ang = float(np.degrees(np.arccos(min(1.0, dx / r))))
    return ang if branch > 0 else 180.0 - ang


def _nan_record(frame, band):
    rec = {c: np.nan for c in RECORD_COLUMNS}
    rec['frame']   = int(frame)
    rec['status']  = 'weak'
    rec['band_y0'] = int(band[0])
    rec['band_y1'] = int(band[1])
    return rec


# ---------------------------------------------------------------------------------
# Single-interface tracker
# ---------------------------------------------------------------------------------

class Interface_Tracking:
    """

    Track one fluid-fluid interface (e.g. a retreating meniscus arc) in an
    image sequence.

    Per frame (default settings):
      1. Cut a tight, padded band around the last known position and
         preprocess it (non-local-means denoise + high-pass, see
         preprocess_band) so tracer beads are suppressed while the dark
         meniscus band survives.
      2. Find the band's centre per column with a ridge detector
         (detect_ridge) and drop columns inconsistent with their
         neighbours.
      3. Fit a circular cap with RANSAC (fit_circle_ransac): apex, radius,
         contact angles at the walls.
      4. Gate the result on valid-column fraction, fit residual and SNR.
         Weak frames are recorded with NaN geometry; too many misses in a
         row -> 'lost'; a band pinned at the image edge -> 'out_of_frame'.
         Both are terminal (self.active becomes False).
      5. Re-center the band on the predicted position (last position +
         recent velocity), keeping its height fixed.

    Legacy behaviour is available with preprocess='gaussian' (heavy blur
    + column-wise |gradient| argmax) and fit_model='poly'.

    x_search : (min, max) pixel bounds of the search columns, fixed.
    y_search : (min, max) pixel bounds of the *initial* y band; its height
               is kept, its position follows the interface. Size it to the
               interface itself, not the range it moves across: a tall
               band lets blurred bead texture dominate the column signal.
    wall_x   : optional (x_left, x_right) wall positions for contact angles.

    """

    def __init__(self, m_metadata = None, m_metadata_file = None,
                 x_search = None, y_search = None, wall_x = None,
                 preprocess = 'nlm_highpass', detector = None,
                 blur_sigma = 30, nlm_h = 10, highpass_sigma = 60, ridge_sigma = 8,
                 fit_model = 'circle', poly_deg = 2, percentile_trim = None,
                 ransac_iters = 50, ransac_tol_px = 3.0,
                 min_valid_frac = 0.4, max_residual_px = 4.0, min_snr = 3.0, max_radius_px = None,
                 max_misses = 10, edge_margin_px = 40, max_band_shift_frac = 0.25,
                 rolling_window = 10, savgol_window = 15, savgol_polyorder = 2,
                 continuity_window = 15, continuity_tolerance = 15,
                 verbose = True):

        # check if metadata is avail
        try:
            self.m_metadata = pmpiv.helper._check_metadata(m_metadata, m_metadata_file)
        except:
            raise FileNotFoundError('Metadata information not available!')

        for name, bounds in (('x_search', x_search), ('y_search', y_search)):
            if bounds is None:
                raise ValueError(f'{name} must be given as a (min, max) tuple!')
            if len(bounds) != 2 or not (bounds[0] < bounds[1]):
                raise ValueError(f'{name} must be a (min, max) tuple with min < max!')

        if preprocess not in PREPROCESS_MODES:
            raise ValueError(f'preprocess must be in {PREPROCESS_MODES}')
        if fit_model not in FIT_MODELS:
            raise ValueError(f'fit_model must be in {FIT_MODELS}')
        if detector is None:
            detector = 'ridge' if preprocess == 'nlm_highpass' else 'gradient'
        if detector not in ('ridge', 'gradient'):
            raise ValueError("detector must be in ('ridge', 'gradient')")

        self.x_search = (int(x_search[0]), int(x_search[1]))
        self.y_search = (int(y_search[0]), int(y_search[1]))
        self.wall_x   = None if wall_x is None else (float(wall_x[0]), float(wall_x[1]))

        self.preprocess     = preprocess
        self.detector       = detector
        self.blur_sigma     = blur_sigma
        self.nlm_h          = nlm_h
        self.highpass_sigma = highpass_sigma
        self.ridge_sigma    = ridge_sigma

        self.fit_model      = fit_model
        self.poly_deg       = poly_deg
        if percentile_trim is None and fit_model == 'poly':
            percentile_trim = (10, 90)
        self.percentile_trim = percentile_trim
        self.ransac_iters   = ransac_iters
        self.ransac_tol_px  = ransac_tol_px

        self.min_valid_frac      = min_valid_frac
        self.max_residual_px     = max_residual_px
        self.min_snr             = min_snr
        # a straight dark line (wall edge, image border) fits a circle of
        # huge radius with tiny residual; cap the radius to reject it
        self.max_radius_px       = max_radius_px
        self.max_misses          = max_misses
        self.edge_margin_px      = edge_margin_px
        self.max_band_shift_frac = max_band_shift_frac

        self.rolling_window   = rolling_window
        self.savgol_window    = savgol_window
        self.savgol_polyorder = savgol_polyorder

        self.continuity_window    = continuity_window
        self.continuity_tolerance = continuity_tolerance

        self.verbose = verbose

        self.reset()

    def reset(self):
        self.active    = True
        self.status    = None
        self._band     = self.y_search
        self._history  = []      # (frame, y_median_px) of ok frames
        self._misses   = 0
        self._arcs     = {}
        self._records  = {}

    def quiet(self):
        self.verbose = False

    # ---------------------------------------------------------------- detection

    def detect(self, img, y_search = None):
        """
        Preprocess the band and locate the interface per column.

        Returns dict(x, y, snr, n_cols) with x, y in absolute image pixels.
        """
        ys = self.y_search if y_search is None else y_search
        band, y0 = preprocess_band(img, self.x_search, ys, mode = self.preprocess,
                                   blur_sigma = self.blur_sigma, nlm_h = self.nlm_h,
                                   highpass_sigma = self.highpass_sigma)

        if self.detector == 'ridge':
            x_idx, y_idx, snr = detect_ridge(band, self.ridge_sigma,
                                             self.continuity_window, self.continuity_tolerance)
        else:
            x_idx, y_idx, snr = detect_gradient(band, self.continuity_window, self.continuity_tolerance)

        x = x_idx + self.x_search[0]
        y = y_idx + y0

        if self.percentile_trim is not None and len(y) > 0:
            lo, hi = np.percentile(y, self.percentile_trim[0]), np.percentile(y, self.percentile_trim[1])
            keep = (y >= lo) & (y <= hi)
            x, y, snr = x[keep], y[keep], snr[keep]

        return dict(x = x, y = y, snr = snr, n_cols = band.shape[1])

    def detect_frame(self, img, y_search = None):
        """Return (x_valid, y_valid) for columns within the search ROI."""
        det = self.detect(img, y_search = y_search)
        return det['x'], det['y']

    # ---------------------------------------------------------------- fitting

    def fit_frame_full(self, x_valid, y_valid):
        """
        Fit the interface model to the detected points.

        Returns dict(x_arc, y_arc, y_median, y_apex, cx, cy, r, residual_rms,
        n_inliers, contact_left, contact_right) or None if the fit fails.
        """
        x_min, x_max = self.x_search
        x_arc = np.arange(x_min, x_max)
        x_valid = np.asarray(x_valid, dtype = float)
        y_valid = np.asarray(y_valid, dtype = float)

        if self.fit_model == 'circle':
            res = fit_circle_ransac(x_valid, y_valid, n_iter = self.ransac_iters,
                                    tol_px = self.ransac_tol_px)
            if res is None:
                return None
            cx, cy, r = res['cx'], res['cy'], res['r']
            # cap pointing down (+y) has its centre above the points
            branch = 1 if cy <= np.median(y_valid) else -1
            y_arc  = circle_arc(cx, cy, r, x_arc, branch = branch)
            if not np.any(np.isfinite(y_arc)):
                return None
            y_apex = cy + branch * r
            cl = contact_angle_deg(cx, cy, r, self.wall_x[0], branch) if self.wall_x else np.nan
            cr = contact_angle_deg(cx, cy, r, self.wall_x[1], branch) if self.wall_x else np.nan
            return dict(x_arc = x_arc, y_arc = y_arc, y_median = float(np.nanmedian(y_arc)),
                        y_apex = float(y_apex), cx = cx, cy = cy, r = r,
                        residual_rms = res['residual_rms'], n_inliers = int(res['inlier_mask'].sum()),
                        contact_left = cl, contact_right = cr)

        try:
            coeffs = np.polyfit(x_valid, y_valid, deg = self.poly_deg)
        except (np.linalg.LinAlgError, TypeError, ValueError):
            return None
        if not np.all(np.isfinite(coeffs)):
            return None
        y_arc = np.polyval(coeffs, x_arc)
        resid = y_valid - np.polyval(coeffs, x_valid)
        return dict(x_arc = x_arc, y_arc = y_arc, y_median = float(np.median(y_arc)),
                    y_apex = float(np.max(y_arc)), cx = np.nan, cy = np.nan, r = np.nan,
                    residual_rms = float(np.sqrt(np.mean(resid ** 2))), n_inliers = int(len(x_valid)),
                    contact_left = np.nan, contact_right = np.nan)

    def fit_frame(self, x_valid, y_valid):
        """Return (x_arc, y_arc, y_median) or None."""
        fit = self.fit_frame_full(x_valid, y_valid)
        if fit is None:
            return None
        return fit['x_arc'], fit['y_arc'], fit['y_median']

    # ---------------------------------------------------------------- tracking

    def _velocity_estimate(self):
        """Median px/frame velocity over the last few ok frames (0 if unknown)."""
        if len(self._history) < 2:
            return 0.0
        h = np.asarray(self._history[-6:], dtype = float)
        dy = np.diff(h[:, 1]) / np.maximum(np.diff(h[:, 0]), 1.0)
        return float(np.median(dy))

    def _update_band(self, y_center_now, fidx, img_height):
        """Re-center the fixed-height band on the predicted next position."""
        band_h = self.y_search[1] - self.y_search[0]
        cur_c  = 0.5 * (self._band[0] + self._band[1])

        pred  = y_center_now + self._velocity_estimate()
        shift = np.clip(pred - cur_c, -self.max_band_shift_frac * band_h, self.max_band_shift_frac * band_h)
        new_c = cur_c + shift

        y0 = int(round(new_c - band_h / 2.0))
        y1 = y0 + band_h
        if y0 < 0:
            y0, y1 = 0, band_h
        if y1 > img_height:
            y1 = img_height
            y0 = max(0, y1 - band_h)
        self._band = (y0, y1)

    def step(self, img, fidx):
        """
        Process one frame. Returns the record dict for this frame (see
        RECORD_COLUMNS), or None if the tracker is no longer active.
        """
        if not self.active:
            return None

        H   = img.shape[0]
        det = self.detect(img, y_search = self._band)
        n_valid = int(len(det['x']))
        snr_med = float(np.median(det['snr'])) if n_valid > 0 else np.nan

        fit = self.fit_frame_full(det['x'], det['y']) if n_valid >= 3 else None
        ok  = (fit is not None
               and n_valid / max(1, det['n_cols']) >= self.min_valid_frac
               and fit['residual_rms'] <= self.max_residual_px
               and (np.isnan(snr_med) or snr_med >= self.min_snr)
               and (self.max_radius_px is None or not np.isfinite(fit['r']) or fit['r'] <= self.max_radius_px))

        rec = _nan_record(fidx, self._band)
        rec['n_valid'] = n_valid
        rec['snr']     = snr_med

        if ok:
            rec.update(status = 'ok', y_median_px = fit['y_median'], y_apex_px = fit['y_apex'],
                       cx_px = fit['cx'], cy_px = fit['cy'], radius_px = fit['r'],
                       residual_px = fit['residual_rms'],
                       contact_angle_left_deg = fit['contact_left'],
                       contact_angle_right_deg = fit['contact_right'])
            self._arcs[fidx] = (fit['x_arc'], fit['y_arc'])
            self._history.append((fidx, fit['y_median']))
            self._misses = 0
            y_now = fit['y_median']
        else:
            self._misses += 1
            y_now = 0.5 * (self._band[0] + self._band[1]) + self._velocity_estimate()

        at_bottom = self._band[1] >= H
        if at_bottom and (not ok or fit['y_apex'] > H - self.edge_margin_px):
            rec['status'] = 'out_of_frame'
            self.active = False
        elif self._misses > self.max_misses:
            rec['status'] = 'lost'
            self.active = False

        if self.verbose and (fidx % 50 == 0 or rec['status'] not in ('ok', 'weak')):
            print(f'Frame {fidx:05d}: status={rec["status"]} y={rec["y_median_px"]:.1f} '
                  f'n_valid={n_valid} snr={snr_med:.1f}')

        self._records[fidx] = rec
        if self.active:
            self._update_band(y_now, fidx, H)
        else:
            self.status = rec['status']

        return rec

    def track(self, image_sequence, start_frame = 0):
        """
        Run step() over image_sequence until the interface is lost or leaves
        the frame. Returns a DataFrame with RECORD_COLUMNS (one row per
        processed frame; NaN geometry on weak frames; no rows after
        termination). Fitted arcs are kept for plot_annotated_tiffs().
        """
        records = []
        for fidx in range(start_frame, len(image_sequence)):
            img = np.asarray(image_sequence[fidx])
            if img.ndim == 3:
                img = img[..., 0]
            rec = self.step(img, fidx)
            if rec is None:
                break
            records.append(rec)
            if not self.active:
                break

        return pd.DataFrame(records, columns = RECORD_COLUMNS)

    # ---------------------------------------------------------------- velocity

    def compute_velocity(self, df, position = 'y_median_px'):
        """
        Add displacement and velocity columns.

        Legacy columns (dt from the actual frame gap, not a constant):
        y_median_m, dy_px, dy_m, velocity_m_s, velocity_rolling_mean_m_s.
        New: y_smooth_px and velocity_savgol_m_s (Savitzky-Golay over the
        interpolated position, per contiguous run), curvature_1_m when a
        radius is available. Velocities are NaN wherever the raw position
        is NaN.
        """
        md = self.m_metadata
        df = df.copy()

        pos       = df[position].astype(float)
        frame_gap = df['frame'].diff()
        dt        = frame_gap / md.FPS

        df['y_median_m']   = df['y_median_px'] * md.PIXELSIZE
        df['dy_px']        = pos.diff()
        df['dy_m']         = df['dy_px'] * md.PIXELSIZE
        df['velocity_m_s'] = df['dy_m'] / dt
        df['velocity_rolling_mean_m_s'] = (
            df['velocity_m_s']
            .rolling(window = self.rolling_window, center = True, min_periods = 1)
            .mean()
        )

        df['y_smooth_px']         = np.nan
        df['velocity_savgol_m_s'] = np.nan
        win = int(self.savgol_window) if self.savgol_window else 0
        if win > self.savgol_polyorder + 1:
            if win % 2 == 0:
                win += 1
            interp = pos.interpolate(limit = self.max_misses, limit_area = 'inside').to_numpy()
            finite = np.isfinite(interp)
            frames = df['frame'].to_numpy(dtype = float)
            # process each contiguous (in index and frame number) finite run
            i = 0
            n = len(interp)
            while i < n:
                if not finite[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < n and finite[j + 1] and frames[j + 1] - frames[j] == 1:
                    j += 1
                if j - i + 1 >= win:
                    seg = interp[i:j + 1]
                    df.iloc[i:j + 1, df.columns.get_loc('y_smooth_px')] = \
                        savgol_filter(seg, win, self.savgol_polyorder)
                    df.iloc[i:j + 1, df.columns.get_loc('velocity_savgol_m_s')] = \
                        savgol_filter(seg, win, self.savgol_polyorder, deriv = 1, delta = 1.0 / md.FPS) * md.PIXELSIZE
                i = j + 1

        raw_nan = pos.isna().to_numpy()
        for c in ('velocity_m_s', 'velocity_rolling_mean_m_s', 'velocity_savgol_m_s'):
            df.loc[raw_nan, c] = np.nan

        if 'radius_px' in df.columns:
            df['curvature_1_m'] = 1.0 / (df['radius_px'] * md.PIXELSIZE)

        return df

    # ---------------------------------------------------------------- plotting

    def _plot_frames(self, frame_idx_list):
        """
        """
        for fidx in frame_idx_list:
            img = np.asarray(self._image_sequence[fidx])
            if img.ndim == 3:
                img = img[..., 0]

            x_arc, y_arc = self._arcs[fidx]

            fig, ax = plt.subplots(figsize = self._figsize)
            ax.imshow(img, cmap = 'gray', origin = 'upper')
            ax.plot(x_arc, y_arc, color = self._color, linewidth = 1.5)
            ax.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
            ax.tick_params(axis = 'y', which = 'both', left = False, right = False, labelleft = False)
            plt.tight_layout()
            plt.savefig(f'{self._outfolder}/frame_{fidx:06d}.tif', dpi = self._dpi)
            plt.close()
            plt.clf()

            if self.verbose:
                print(f'PID {os.getpid()}: saved frame {fidx:06d}')

    def plot_annotated_tiffs(self, df, outfolder, image_sequence,
                              color = 'cyan', dpi = 100, parallel = True):
        """
        Save one annotated TIFF per tracked frame, overlaying the fitted
        interface arc. Requires track() to have been called first.
        """

        if not self._arcs:
            raise ValueError('No fitted arcs available. Call track() first.')

        os.makedirs(outfolder, exist_ok = True)
        if self.verbose:
            print(f'Create folder if not existing: {outfolder}')

        img0 = np.asarray(image_sequence[0])
        if img0.ndim == 3:
            img0 = img0[..., 0]
        H, W = img0.shape
        figsize = list(np.array([W, H]) * 1.7 / plt.rcParams['figure.dpi'])

        valid_frames = sorted(self._arcs.keys())

        self._image_sequence = image_sequence
        self._outfolder      = outfolder
        self._color          = color
        self._dpi            = dpi
        self._figsize        = figsize

        if parallel:
            ncpus = max(1, int(0.5 * multiprocessing.cpu_count()))
            if self.verbose:
                print(f'Using {ncpus} worker processes for TIFF output.')

            frame_lists = list(np.array_split(valid_frames, ncpus))
            for i in range(len(frame_lists)): frame_lists[i] = list(frame_lists[i])

            pool = multiprocessing.Pool(ncpus)
            pool.map(self._plot_frames, frame_lists)
            pool.close()
            pool.join()
        else:
            self._plot_frames(valid_frames)


# ---------------------------------------------------------------------------------
# Multi-interface tracker
# ---------------------------------------------------------------------------------

class Multi_Interface_Tracking:
    """

    Discover every curved interface in a frame and track all of them
    through an image sequence in a single pass (each frame is read once).

    discovery : 'walls' | 'hough' | 'both' (see
                pmpiv.interface_discovery.discover_interfaces).
    tracker_kwargs are forwarded to every Interface_Tracking.

    """

    DEFAULT_COLORS = ('cyan', 'magenta', 'yellow', 'lime', 'orange', 'red')

    def __init__(self, m_metadata = None, m_metadata_file = None,
                 discovery = 'both', band_height = 300, discovery_kwargs = None,
                 verbose = True, **tracker_kwargs):

        # check if metadata is avail
        try:
            self.m_metadata = pmpiv.helper._check_metadata(m_metadata, m_metadata_file)
        except:
            raise FileNotFoundError('Metadata information not available!')

        self.discovery        = discovery
        self.band_height      = int(band_height)
        self.discovery_kwargs = dict(discovery_kwargs or {})
        self.tracker_kwargs   = dict(tracker_kwargs)
        self.verbose          = verbose

        self.seeds    = []
        self.trackers = {}
        self._dfs     = None

    def quiet(self):
        self.verbose = False
        for t in self.trackers.values():
            t.quiet()

    def add_interface(self, label, x_search, y_search, wall_x = None):
        kwargs = dict(self.tracker_kwargs)
        kwargs.setdefault('max_radius_px', 3.0 * (x_search[1] - x_search[0]))
        self.trackers[label] = Interface_Tracking(m_metadata = self.m_metadata,
                                                  x_search = x_search, y_search = y_search,
                                                  wall_x = wall_x, verbose = self.verbose,
                                                  **kwargs)
        return self.trackers[label]

    def discover(self, img):
        """Find interfaces in img (usually the first frame) and create one tracker per seed."""
        from pmpiv import interface_discovery

        img = np.asarray(img)
        if img.ndim == 3:
            img = img[..., 0]

        self.seeds = interface_discovery.discover_interfaces(img, method = self.discovery,
                                                             band_height = self.band_height,
                                                             verbose = self.verbose,
                                                             **self.discovery_kwargs)
        for s in self.seeds:
            self.add_interface(s['label'], s['x_search'], s['y_search'], s.get('wall_x'))

        if self.verbose:
            print(f'Discovered {len(self.seeds)} interface(s): '
                  + ', '.join(f"{s['label']} x={s['x_search']} y={s['y_search']} ({s['seed_type']})"
                              for s in self.seeds))
        return self.seeds

    def track(self, image_sequence, start_frame = 0):
        """Track all interfaces; returns {label: DataFrame}."""
        if not self.trackers:
            raise ValueError('No interfaces to track. Call discover() or add_interface() first.')

        records = {label: [] for label in self.trackers}

        for fidx in range(start_frame, len(image_sequence)):
            active = [t for t in self.trackers.values() if t.active]
            if not active:
                break

            img = np.asarray(image_sequence[fidx])
            if img.ndim == 3:
                img = img[..., 0]

            for label, t in self.trackers.items():
                if not t.active:
                    continue
                rec = t.step(img, fidx)
                if rec is not None:
                    records[label].append(rec)

        self._dfs = {label: pd.DataFrame(recs, columns = RECORD_COLUMNS)
                     for label, recs in records.items()}
        return self._dfs

    def compute_velocity(self, position = 'y_median_px'):
        if self._dfs is None:
            raise ValueError('Nothing tracked yet. Call track() first.')
        self._dfs = {label: self.trackers[label].compute_velocity(df, position = position)
                     for label, df in self._dfs.items()}
        return self._dfs

    def to_dataframe(self):
        """Long-format DataFrame with a leading 'interface' column."""
        if self._dfs is None:
            raise ValueError('Nothing tracked yet. Call track() first.')
        parts = []
        for label, df in self._dfs.items():
            d = df.copy()
            d.insert(0, 'interface', label)
            parts.append(d)
        if not parts:
            return pd.DataFrame(columns = ['interface'] + RECORD_COLUMNS)
        return pd.concat(parts, ignore_index = True)

    def write_csv(self, folder, filename = 'interface_tracking.csv'):
        pmpiv.df_io.write2csv(self.to_dataframe(), folder, filename)

    def _plot_frames(self, frame_idx_list):
        """
        """
        for fidx in frame_idx_list:
            img = np.asarray(self._image_sequence[fidx])
            if img.ndim == 3:
                img = img[..., 0]

            fig, ax = plt.subplots(figsize = self._figsize)
            ax.imshow(img, cmap = 'gray', origin = 'upper')

            for k, (label, t) in enumerate(self.trackers.items()):
                color = self._colors[k % len(self._colors)]
                if t.wall_x is not None:
                    for xw in t.wall_x:
                        ax.axvline(xw, color = color, linewidth = 0.6, linestyle = '--', alpha = 0.6)
                rec = t._records.get(fidx)
                if fidx in t._arcs:
                    x_arc, y_arc = t._arcs[fidx]
                    ax.plot(x_arc, y_arc, color = color, linewidth = 1.5)
                    if rec is not None and np.isfinite(rec['cx_px']):
                        ax.plot(rec['cx_px'], rec['y_apex_px'], marker = '+', color = color, markersize = 8)
                if rec is not None:
                    txt = f"{label}: y={rec['y_median_px']:.0f} r={rec['radius_px']:.0f} {rec['status']}"
                    ax.text(t.x_search[0], max(0, rec['band_y0'] - 10), txt, color = color, fontsize = 8)

            ax.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
            ax.tick_params(axis = 'y', which = 'both', left = False, right = False, labelleft = False)
            plt.tight_layout()
            plt.savefig(f'{self._outfolder}/frame_{fidx:06d}.tif', dpi = self._dpi)
            plt.close()
            plt.clf()

            if self.verbose:
                print(f'PID {os.getpid()}: saved frame {fidx:06d}')

    def plot_annotated_tiffs(self, outfolder, image_sequence, colors = None,
                              dpi = 100, parallel = True):
        """
        One annotated TIFF per frame with every tracked arc overlaid.
        """
        frames = sorted(set().union(*[set(t._arcs.keys()) for t in self.trackers.values()]))
        if not frames:
            raise ValueError('No fitted arcs available. Call track() first.')

        os.makedirs(outfolder, exist_ok = True)
        if self.verbose:
            print(f'Create folder if not existing: {outfolder}')

        img0 = np.asarray(image_sequence[0])
        if img0.ndim == 3:
            img0 = img0[..., 0]
        H, W = img0.shape

        self._image_sequence = image_sequence
        self._outfolder      = outfolder
        self._colors         = tuple(colors) if colors else self.DEFAULT_COLORS
        self._dpi            = dpi
        self._figsize        = list(np.array([W, H]) * 1.7 / plt.rcParams['figure.dpi'])

        if parallel:
            ncpus = max(1, int(0.5 * multiprocessing.cpu_count()))
            if self.verbose:
                print(f'Using {ncpus} worker processes for TIFF output.')
            frame_lists = [list(a) for a in np.array_split(frames, ncpus)]
            pool = multiprocessing.Pool(ncpus)
            pool.map(self._plot_frames, frame_lists)
            pool.close()
            pool.join()
        else:
            self._plot_frames(frames)
