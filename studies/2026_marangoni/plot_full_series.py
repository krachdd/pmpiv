#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full annotated-frame image series for the Marangoni study: every frame either
interface was tracked, showing its fitted arc plus the PIV tracer particles
attributed to it (inside its x-range, within DEPTH_MAX below the interface —
the same attribution used for the eddy/velocity analysis in flow_report.py).

Reuses the cached interface_tracking_full.csv / df_linked_full.pkl from
flow_report.py — run that first if they don't exist yet.

Output is PNG, not TIFF: the pre-existing tiffs_interface/ (uncompressed
TIFF, ~600 frames) is ~32GB, i.e. ~53MB/frame; at that rate the ~2900 frames
here would be >100GB. PNG is lossless and compresses this mostly-flat
grayscale content to a few hundred KB/frame instead.

Usage:
    python plot_full_series.py studies/2026_marangoni/2026_marangoni.txt
"""

from __future__ import division, unicode_literals, print_function

import os
import sys
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pmpiv
from pmpiv.interface import circle_arc
from flow_report import discover_and_track_full, DEPTH_MAX, X_PAD

OUTDIR_NAME = 'tiffs_interface_full'
COLORS = ('cyan', 'magenta', 'yellow', 'lime')
DPI = 100

# Module-level globals for forked Pool workers (matches the existing
# multiprocessing convention elsewhere in this package).
_g_seq        = None
_g_by_frame   = None
_g_track_dfs  = None
_g_trackers   = None
_g_colors     = None
_g_outdir     = None
_g_figsize    = None


def _plot_frames(frame_list):
    for fidx in frame_list:
        img = np.asarray(_g_seq[fidx])
        if img.ndim == 3:
            img = img[..., 0]

        fig, ax = plt.subplots(figsize = _g_figsize)
        ax.imshow(img, cmap = 'gray', vmin = 0, vmax = 255)

        particles = _g_by_frame.get(fidx)

        for label, color in _g_colors.items():
            tracker = _g_trackers[label]
            df = _g_track_dfs[label]
            if fidx not in df.index:
                continue
            rec = df.loc[fidx]
            if rec['status'] != 'ok':
                continue

            x0, x1 = tracker.x_search
            cx, cy, r = rec['cx_px'], rec['cy_px'], rec['radius_px']

            x_arc = np.arange(x0, x1)
            y_arc = circle_arc(cx, cy, r, x_arc, branch = 1)
            ax.plot(x_arc, y_arc, color = color, linewidth = 1.5)

            if particles is not None and len(particles):
                sel = particles[(particles['x'] >= x0 - X_PAD) & (particles['x'] <= x1 + X_PAD)].copy()
                depth = sel['y'].to_numpy() - circle_arc(cx, cy, r, sel['x'].to_numpy(), branch = 1)
                sel = sel[(depth >= 0) & (depth <= DEPTH_MAX)]
                if len(sel):
                    ax.scatter(sel['x'], sel['y'], s = 4, color = color, alpha = 0.7, linewidths = 0)

        ax.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
        ax.tick_params(axis = 'y', which = 'both', left = False, right = False, labelleft = False)
        plt.tight_layout()
        plt.savefig(f'{_g_outdir}/frame_{fidx:06d}.png', dpi = DPI)
        plt.close()

        if fidx % 100 == 0:
            print(f'PID {os.getpid()}: frame {fidx:06d}')


def main():
    try:
        metadata_file = sys.argv[1]
    except IndexError:
        raise SystemExit('Usage: python plot_full_series.py <metadata_file.txt>')

    md = pmpiv.metadata.Metadata(metadata_file)
    seq = pmpiv.image_sequence.Image_Sequence(m_metadata = md).read()

    interface_csv = os.path.join(md.WORKING_DIR, 'interface_tracking_full.csv')
    linked_path = os.path.join(md.WORKING_DIR, 'df_linked_full.pkl')
    if not (os.path.isfile(interface_csv) and os.path.isfile(linked_path)):
        raise SystemExit(f'Missing cached data — run flow_report.py first.\n'
                         f'Expected: {interface_csv}\n          {linked_path}')

    mt, track_dfs = discover_and_track_full(md, seq, interface_csv)
    track_dfs = {label: df.set_index('frame', drop = False) for label, df in track_dfs.items()}

    df_linked = pd.read_pickle(linked_path)
    if df_linked.index.name == 'frame':
        # tp.filter_stubs leaves 'frame' as both the index and a column,
        # which groupby('frame') below rejects as ambiguous.
        df_linked = df_linked.reset_index(drop = True)
    print('Splitting particles by frame...')
    by_frame = {f: g for f, g in df_linked.groupby('frame')}

    n_frames = max(int(df['frame'].max()) for df in track_dfs.values()) + 1
    print(f'Rendering frames 0..{n_frames - 1} for {list(track_dfs)}')

    outdir = os.path.join(md.WORKING_DIR, OUTDIR_NAME)
    os.makedirs(outdir, exist_ok = True)

    img0 = np.asarray(seq[0])
    if img0.ndim == 3:
        img0 = img0[..., 0]
    H, W = img0.shape
    figsize = list(np.array([W, H]) * 1.0 / plt.rcParams['figure.dpi'])

    global _g_seq, _g_by_frame, _g_track_dfs, _g_trackers, _g_colors, _g_outdir, _g_figsize
    _g_seq       = seq
    _g_by_frame  = by_frame
    _g_track_dfs = track_dfs
    _g_trackers  = mt.trackers
    _g_colors    = dict(zip(mt.trackers, COLORS))
    _g_outdir    = outdir
    _g_figsize   = figsize

    frames = list(range(n_frames))
    ncpus = max(1, int(0.5 * multiprocessing.cpu_count()))
    print(f'Using {ncpus} worker processes for {len(frames)} frames -> {outdir}')

    frame_lists = [list(a) for a in np.array_split(frames, ncpus)]
    # Python 3.14 changed the default multiprocessing start method on Linux
    # from 'fork' to 'forkserver'. This script's workers rely on inheriting
    # the module-level _g_* globals set just above via copy-on-write memory,
    # which only happens with a true fork (forkserver children are forked
    # from an early, separate server process that never sees these
    # assignments) -- so the context is pinned to 'fork' explicitly here.
    pool = multiprocessing.get_context('fork').Pool(ncpus)
    pool.map(_plot_frames, frame_lists)
    pool.close()
    pool.join()

    print('done')


if __name__ == '__main__':
    main()
