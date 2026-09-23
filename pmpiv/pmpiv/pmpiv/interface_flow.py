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
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, label

import pmpiv as pmpiv
from pmpiv.interface import circle_arc

###--------------------------------------------------------------------------------


def particle_velocity_frame(df_linked, frame, m_metadata = None, m_metadata_file = None,
                            dt_frames = 1):
    """
    Per-particle velocity between frame and frame + dt_frames, in physical units.

    Reuses pmpiv.motion_stats.Motion_Statistics.displacement_2frames (which wraps
    trackpy.motion.relate_frames) instead of duplicating the frame-pairing logic.

    Returns a DataFrame indexed by particle with columns x, y (position at `frame`),
    dx, dy, dr (px), vx, vy, v (m/s), frame1.
    """
    try:
        md = pmpiv.helper._check_metadata(m_metadata, m_metadata_file)
    except:
        raise FileNotFoundError('Metadata information not available!')

    ms = pmpiv.motion_stats.Motion_Statistics(df_linked, m_metadata = md, verbose = False)
    d = ms.displacement_2frames(frame, frame + dt_frames)

    scale = md.FPS / dt_frames * md.PIXELSIZE
    d = d.copy()
    d['vx']     = d['dx'] * scale
    d['vy']     = d['dy'] * scale
    d['v']      = d['dr'] * scale
    d['frame1'] = frame

    return d


def restrict_to_roi(df_linked, x_range, frame_range = None):
    """
    Filter df_linked to x in [x_range[0], x_range[1]] and, optionally, frame in
    [frame_range[0], frame_range[1]]. Meant to be applied BEFORE constructing
    Motion_Statistics for a many-frame trend series, so relate_frames stays cheap.
    """
    x0, x1 = x_range
    d = df_linked[(df_linked['x'] >= x0) & (df_linked['x'] <= x1)]
    if frame_range is not None:
        f0, f1 = frame_range
        d = d[(d['frame'] >= f0) & (d['frame'] <= f1)]
    return d.copy()


def interface_relative_velocity(df_vel_frame, cx, cy, r, x_search, branch = 1):
    """
    Decompose particle velocities relative to a circular-cap interface.

    Adds y_interface (via pmpiv.interface.circle_arc), depth = y - y_interface
    (>= 0 means below/inside the liquid), and tangential/normal velocity
    components v_t, v_n (v_n > 0 means moving away from the interface, deeper
    into the liquid). Rows above the interface (depth < 0) or outside the
    circle's domain (|x-cx| > r) are dropped.
    """
    df = df_vel_frame.reset_index() if df_vel_frame.index.name == 'particle' else df_vel_frame.copy()
    if 'particle' not in df.columns:
        df = df.reset_index().rename(columns = {'index': 'particle'})

    x0, x1 = x_search
    df = df[(df['x'] >= x0) & (df['x'] <= x1)].copy()

    y_if = circle_arc(cx, cy, r, df['x'].to_numpy(), branch = branch)
    df['y_interface'] = y_if
    df['depth'] = df['y'] - y_if

    df = df[np.isfinite(df['y_interface']) & (df['depth'] >= 0)].copy()
    if len(df) == 0:
        for c in ('tx', 'ty', 'nx', 'ny', 'v_t', 'v_n'):
            df[c] = pd.Series(dtype = float)
        return df

    dxc = df['x'].to_numpy() - cx
    denom = np.sqrt(np.maximum(r ** 2 - dxc ** 2, 1e-12))
    dydx = -branch * dxc / denom               # d(circle_arc)/dx

    tnorm = np.sqrt(1.0 + dydx ** 2)
    tx, ty = 1.0 / tnorm, dydx / tnorm          # unit tangent
    nx, ny = -ty, tx                            # unit normal, rotate tangent +90deg
    # normal must point away from the interface (into deeper liquid, +y-ish)
    flip = ny < 0
    nx[flip] *= -1
    ny[flip] *= -1

    df['tx'], df['ty'] = tx, ty
    df['nx'], df['ny'] = nx, ny
    df['v_t'] = df['vx'] * tx + df['vy'] * ty
    df['v_n'] = df['vx'] * nx + df['vy'] * ny

    return df


def velocity_field_grid(df_rel, x_range, depth_range, grid_step = 10.0, method = 'linear'):
    """
    Interpolate vx, vy (from interface_relative_velocity's output) onto a regular
    (x, depth) grid. Cells outside the particles' convex hull are NaN (never
    fabricated as zero velocity).

    Returns dict(X, Y, U, V, dx, dy, n_particles).
    """
    x0, x1 = x_range
    y0, y1 = depth_range  # depth-space bounds, but Y below is absolute y = y_interface-independent grid in depth units mapped by caller

    nx = max(2, int(round((x1 - x0) / grid_step)) + 1)
    ny = max(2, int(round((y1 - y0) / grid_step)) + 1)
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys)

    pts = df_rel[['x', 'depth']].to_numpy()
    U = griddata(pts, df_rel['vx'].to_numpy(), (X, Y), method = method)
    V = griddata(pts, df_rel['vy'].to_numpy(), (X, Y), method = method)

    dx = xs[1] - xs[0] if nx > 1 else grid_step
    dy = ys[1] - ys[0] if ny > 1 else grid_step

    return dict(X = X, Y = Y, U = U, V = V, dx = dx, dy = dy, n_particles = len(df_rel))


def vorticity(X, Y, U, V, dx, dy, smooth_sigma = 1.0):
    """
    omega = dV/dx - dU/dy [1/s].

    U, V are lightly Gaussian-smoothed first (smooth_sigma in grid cells,
    NaN-aware): griddata + finite differences on sparse scattered particles
    otherwise double-differentiates noise. NaN cells propagate to their
    neighbours' derivatives by construction; downstream code must treat NaN
    vorticity as "unknown", never as zero.
    """
    def _nan_gaussian(a, sigma):
        if sigma is None or sigma <= 0:
            return a
        mask = np.isfinite(a)
        if not mask.any():
            return a
        filled = np.where(mask, a, 0.0)
        num = gaussian_filter(filled, sigma = sigma)
        den = gaussian_filter(mask.astype(float), sigma = sigma)
        with np.errstate(invalid = 'ignore', divide = 'ignore'):
            out = num / den
        out[den < 1e-6] = np.nan
        return out

    Us = _nan_gaussian(U, smooth_sigma)
    Vs = _nan_gaussian(V, smooth_sigma)

    dVdx = np.gradient(Vs, dx, axis = 1)
    dUdy = np.gradient(Us, dy, axis = 0)

    return dVdx - dUdy


def detect_eddies(omega, X, Y, dx, dy, noise_floor = None, noise_percentile = 80.0,
                  min_area_px2 = None, threshold_frac = 0.3):
    """
    Identify eddy cores as connected regions of same-signed vorticity exceeding
    a noise floor, then measure each core's size from the |omega| >
    threshold_frac*peak contour within that same connected region (labeling
    each sign separately keeps counter-rotating neighbours from merging).

    noise_floor defaults to the noise_percentile-th percentile of |omega| over
    valid cells (adapts per frame/channel). A plain median (50th percentile)
    is too permissive on sparse, griddata-interpolated PIV data — validated on
    real data, it passed roughly half of all cells and fragmented into 30+
    spurious "eddies" per frame; the 80th percentile keeps only the frame's
    genuinely coherent regions. min_area_px2 defaults to 8*dx*dy (an area, so
    it is robust to the chosen grid_step) — also tuned against real data,
    where the mean nearest-neighbour particle spacing is comparable to a few
    grid cells, so smaller regions are indistinguishable from interpolation
    noise rather than a real recirculation cell.

    Returns a list of dicts: sign, peak_omega, core_x, core_y, radius_eq, area,
    dominant (bool, the single largest |peak_omega|*area eddy is flagged True).
    """
    valid = np.isfinite(omega)
    if not valid.any():
        return []

    if noise_floor is None:
        noise_floor = float(np.percentile(np.abs(omega[valid]), noise_percentile))
    if min_area_px2 is None:
        min_area_px2 = 8.0 * dx * dy
    cell_area = dx * dy

    eddies = []
    for sign in (1, -1):
        mask = valid & (np.sign(omega) == sign) & (np.abs(omega) > noise_floor)
        if not mask.any():
            continue
        labels, n = label(mask)
        for lbl in range(1, n + 1):
            region = labels == lbl
            if region.sum() * cell_area < min_area_px2:
                continue
            peak = np.nanmax(np.abs(np.where(region, omega, np.nan)))
            core_idx = np.unravel_index(np.nanargmax(np.where(region, np.abs(omega), np.nan)), omega.shape)
            grown = region & (np.abs(omega) > threshold_frac * peak)
            area = float(grown.sum() * cell_area)
            eddies.append(dict(
                sign = sign, peak_omega = float(sign * peak),
                core_x = float(X[core_idx]), core_y = float(Y[core_idx]),
                radius_eq = float(np.sqrt(area / np.pi)), area = area,
                dominant = False,
            ))

    if eddies:
        best = max(eddies, key = lambda e: abs(e['peak_omega']) * e['area'])
        best['dominant'] = True

    return eddies


def penetration_depth(X, Y, omega, decay_frac = 0.1, near_rows = 3, min_valid_frac = 0.5):
    """
    Depth (in Y's units, i.e. below the interface) at which the x-averaged
    |omega| profile first decays to decay_frac of its near-interface value.

    Rows (constant-depth lines of the grid) with fewer than min_valid_frac
    finite cells are skipped rather than averaged with fabricated data.
    near_interface_value is the mean of the first `near_rows` valid rows, not
    a single noisy row.

    Returns dict(depth, censored). censored=True (depth=NaN) means the
    profile never decayed within the grid's depth range — reported honestly
    rather than substituting the grid's maximum depth as if it were the
    answer.
    """
    depths = Y[:, 0]
    abs_omega = np.abs(omega)
    valid_frac = np.mean(np.isfinite(abs_omega), axis = 1)
    row_mean = np.full(abs_omega.shape[0], np.nan)
    ok_rows = valid_frac >= min_valid_frac
    row_mean[ok_rows] = np.nanmean(abs_omega[ok_rows], axis = 1)

    valid_idx = np.where(np.isfinite(row_mean))[0]
    if len(valid_idx) < near_rows + 1:
        return dict(depth = np.nan, censored = True)

    near_value = float(np.mean(row_mean[valid_idx[:near_rows]]))
    if near_value <= 0:
        return dict(depth = np.nan, censored = True)
    threshold = decay_frac * near_value

    for i in range(1, len(valid_idx)):
        j0, j1 = valid_idx[i - 1], valid_idx[i]
        v0, v1 = row_mean[j0], row_mean[j1]
        if v1 <= threshold <= v0 or v1 <= threshold < v0:
            if v0 == v1:
                d = depths[j1]
            else:
                frac = (v0 - threshold) / (v0 - v1)
                d = depths[j0] + frac * (depths[j1] - depths[j0])
            return dict(depth = float(d), censored = False)

    return dict(depth = np.nan, censored = True)
