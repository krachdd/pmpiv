#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface tracking for Marangoni PIV experiments.

Detects the retreating fluid-fluid interface (meniscus arc) in each frame via:
  1. Heavy Gaussian blur  → suppresses tracer particles
  2. Column-wise gradient → raw y-position per column, restricted to ROI
  3. Outlier rejection    → drop percentile extremes within ROI
  4. Polynomial fit       → smooth interface curve across ROI columns
  5. Velocity computation → dy/dt with rolling mean

Usage:
    python interface_tracking.py studies/2026_marangoni/2026_marangoni.txt

Outputs:
    {WORKING_DIR}/tiffs_interface/frame_*.tif   annotated TIFFs (cyan curve)
    {WORKING_DIR}/interface_tracking.csv         per-frame position + velocity
"""

from __future__ import division, unicode_literals, print_function

import sys
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — must come before pyplot
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import multiprocessing
import pandas as pd
import pims

import pmpiv

# ---------------------------------------------------------------------------
# Tuning parameters — adjust as needed
# ---------------------------------------------------------------------------
BLUR_SIGMA     = 30    # px — suppresses tracer particles, keeps large-scale intensity step
POLY_DEG       = 2     # degree of polynomial fit
ROLLING_WINDOW = 10    # frames — window for rolling-mean velocity smoothing

# Search region of interest (ROI) — must be set to the channel opening.
# Columns and rows outside this rectangle are ignored entirely.
# Determined empirically from difference-image diagnostics and peak-profile analysis:
#   - Y: just below the bottom channel wall edge (y≈1058) up to near image bottom
#   - X: inner x-span of the meniscus arc
Y_SEARCH_MIN   = 1100  # px — top of y-search band (below wall edge at y≈1058)
Y_SEARCH_MAX   = 1850  # px — bottom of y-search band
X_SEARCH_MIN   = 1300  # px — left inner edge of arc
X_SEARCH_MAX   = 1820  # px — right inner edge of arc
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Module-level globals used by multiprocessing workers.
# On Linux the default start method is 'fork', so child processes inherit
# these without pickling.
# ---------------------------------------------------------------------------
_g_img_seq   = None   # pims ImageSequence (sliced to START:END)
_g_outfolder = None   # output folder for TIFFs
_g_coeffs    = None   # dict: frame_idx -> (x_arc, y_fit) arrays for plotting
_g_figsize   = None   # matplotlib figsize in inches


def _worker_plot_frame(frame_idx):
    """Plot and save one annotated TIFF with the cyan interface curve."""
    img = np.asarray(_g_img_seq[frame_idx])
    if img.ndim == 3:
        img = img[..., 0]

    x_arc, y_arc = _g_coeffs[frame_idx]   # pre-evaluated arc coordinates

    fig, ax = plt.subplots(figsize=_g_figsize)
    ax.imshow(img, cmap='gray', origin='upper')
    ax.plot(x_arc, y_arc, color='cyan', linewidth=1.5)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(f'{_g_outfolder}/frame_{frame_idx:06d}.tif', dpi=100)
    plt.close()
    plt.clf()
    print(f'PID {os.getpid()}: saved frame {frame_idx:06d}')


# ---------------------------------------------------------------------------
# Core detection helpers
# ---------------------------------------------------------------------------

def _detect_interface_position(img):
    """
    Return (x_valid, y_valid) for columns within the x/y search ROI.

    Blurs the ROI strip, finds the per-column maximum-|gradient| y-position,
    then drops the extreme 10th and 90th percentile outliers before fitting.
    """
    # Crop to the y-search band for efficiency; blur only that strip
    img_roi  = img[Y_SEARCH_MIN:Y_SEARCH_MAX, X_SEARCH_MIN:X_SEARCH_MAX].astype(float)
    img_blur = gaussian_filter(img_roi, sigma=BLUR_SIGMA)

    # Gradient along y for all columns at once (vectorised)
    img_grad    = np.gradient(img_blur, axis=0)
    y_peaks_roi = np.argmax(np.abs(img_grad), axis=0)   # index within ROI strip
    y_peaks     = y_peaks_roi + Y_SEARCH_MIN             # absolute image y

    x_cols = np.arange(X_SEARCH_MIN, X_SEARCH_MAX)

    # Drop extreme outliers (columns where gradient peak is spurious)
    p10 = np.percentile(y_peaks, 10)
    p90 = np.percentile(y_peaks, 90)
    mask = (y_peaks >= p10) & (y_peaks <= p90)

    return x_cols[mask], y_peaks[mask]


def _fit_interface(x_valid, y_valid):
    """
    Fit a degree-POLY_DEG polynomial over the valid columns and return
    (x_arc, y_arc, y_median) spanning the full x search ROI.
    """
    coeffs   = np.polyfit(x_valid, y_valid, deg=POLY_DEG)
    x_arc    = np.arange(X_SEARCH_MIN, X_SEARCH_MAX)
    y_arc    = np.polyval(coeffs, x_arc)
    y_median = float(np.median(y_arc))
    return x_arc, y_arc, y_median


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _g_img_seq, _g_outfolder, _g_coeffs, _g_figsize

    # ------------------------------------------------------------------
    # Parse arguments and load metadata
    # ------------------------------------------------------------------
    try:
        metadata_file = sys.argv[1]
    except IndexError:
        raise SystemExit('Usage: python interface_tracking.py <metadata_file.txt>')

    md = pmpiv.metadata.Metadata(metadata_file)

    # ------------------------------------------------------------------
    # Load image sequence (same slice as main processing script)
    # ------------------------------------------------------------------
    img_seq_full = pims.ImageSequence(f'{md.IN_PATH}/*{md.IN_FORMAT}')
    start        = md.START_FRAME
    end          = md.END_FRAME if md.END_FRAME != 0 else len(img_seq_full)
    img_seq      = img_seq_full[start:end]

    n_frames = len(img_seq)
    print(f'Image sequence: {n_frames} frames  ({start} … {start + n_frames - 1})')
    print(f'Search ROI: y=[{Y_SEARCH_MIN}, {Y_SEARCH_MAX})  x=[{X_SEARCH_MIN}, {X_SEARCH_MAX})')

    # Image geometry from first frame
    img0 = np.asarray(img_seq[0])
    if img0.ndim == 3:
        img0 = img0[..., 0]
    H, W    = img0.shape
    figsize = list(np.array([W, H]) * 1.7 / plt.rcParams['figure.dpi'])

    # ------------------------------------------------------------------
    # Per-frame interface detection and polynomial fitting
    # ------------------------------------------------------------------
    records    = []
    all_arcs   = {}   # frame_idx -> (x_arc, y_arc) for TIFF plotting

    for fidx in range(n_frames):
        img = np.asarray(img_seq[fidx])
        if img.ndim == 3:
            img = img[..., 0]

        x_valid, y_valid = _detect_interface_position(img)

        if len(x_valid) < POLY_DEG + 1:
            print(f'Frame {fidx}: too few valid columns ({len(x_valid)}), skipping.')
            continue

        x_arc, y_arc, y_median_px = _fit_interface(x_valid, y_valid)
        all_arcs[fidx] = (x_arc, y_arc)

        records.append({'frame': fidx, 'y_median_px': y_median_px})

        if fidx % 50 == 0:
            print(f'Frame {fidx:05d}: interface y = {y_median_px:.1f} px')

    # ------------------------------------------------------------------
    # Build DataFrame and compute velocity
    # ------------------------------------------------------------------
    df = pd.DataFrame(records)
    df = df.sort_values('frame').reset_index(drop=True)

    dt = 1.0 / md.FPS          # seconds per frame

    df['y_median_m']              = df['y_median_px'] * md.PIXELSIZE
    df['dy_px']                   = df['y_median_px'].diff()
    df['dy_m']                    = df['y_median_m'].diff()
    df['velocity_m_s']            = df['dy_px'] * md.PIXELSIZE / dt
    df['velocity_rolling_mean_m_s'] = (
        df['velocity_m_s']
        .rolling(window=ROLLING_WINDOW, center=True, min_periods=1)
        .mean()
    )

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------
    os.makedirs(md.WORKING_DIR, exist_ok=True)
    csv_path = os.path.join(md.WORKING_DIR, 'interface_tracking.csv')
    df[['frame', 'y_median_px', 'y_median_m', 'dy_px', 'dy_m',
        'velocity_m_s', 'velocity_rolling_mean_m_s']].to_csv(
        csv_path, sep=';', index=False
    )
    print(f'\nSaved interface data → {csv_path}')

    # ------------------------------------------------------------------
    # Output annotated TIFFs in parallel
    # ------------------------------------------------------------------
    out_tiff_dir = os.path.join(md.WORKING_DIR, 'tiffs_interface')
    os.makedirs(out_tiff_dir, exist_ok=True)
    print(f'Saving annotated TIFFs → {out_tiff_dir}')

    # Store in module-level globals so forked workers can access them
    _g_img_seq   = img_seq
    _g_outfolder = out_tiff_dir
    _g_coeffs    = all_arcs
    _g_figsize   = figsize

    valid_frames = sorted(all_arcs.keys())
    ncpus        = max(1, int(0.5 * multiprocessing.cpu_count()))
    print(f'Using {ncpus} worker processes for TIFF output.')

    pool = multiprocessing.Pool(ncpus)
    pool.map(_worker_plot_frame, valid_frames)
    pool.close()
    pool.join()

    print('\nInterface tracking complete.')


if __name__ == '__main__':
    main()
