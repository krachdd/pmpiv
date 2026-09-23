#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface-relative particle velocities, eddy size and penetration-depth
analysis for the Marangoni study, written up as a compiled LaTeX report.

Extends interface tracking to the full image sequence to find each channel's
actual valid lifetime, extends the particle-tracking (trackpy) cache to match
(new cache files distinct from the study's existing 600-frame cache), then
for each channel:
  - decomposes tracer-particle velocities into components tangential/normal
    to the interface (pmpiv.interface_flow.interface_relative_velocity),
  - interpolates a velocity field below the interface and computes its
    vorticity, on a handful of representative frames (annotated figures) and,
    more coarsely, across the channel's whole lifetime (trend plots),
  - detects recirculating eddies and measures their size,
  - measures how deep the vorticity signal penetrates into the liquid,
and assembles the results into report.tex / report.pdf.

Usage:
    python flow_report.py studies/2026_marangoni/2026_marangoni.txt
"""

from __future__ import division, unicode_literals, print_function

import copy
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trackpy as tp

import pmpiv
from pmpiv import interface_flow as flow

GRID_STEP    = 20.0     # px, velocity-field interpolation grid — tuned to the real
                        # data's mean inter-particle spacing (~14px); a finer grid
                        # oversamples the data and detect_eddies fragments into
                        # dozens of spurious sub-particle-spacing "eddies"
DEPTH_MAX    = 400.0    # px below the interface searched for eddies
N_REP        = 5        # representative fully-annotated frames per channel
TREND_STRIDE = dict(ch0 = 5, ch1 = 20)
DEFAULT_STRIDE = 10
X_PAD        = 20.0     # px padding when restricting df_linked to a channel


def discover_and_track_full(md, seq, out_csv):
    """Full-sequence multi-interface discovery/tracking, cached to out_csv."""
    mt = pmpiv.interface.Multi_Interface_Tracking(m_metadata = md, discovery = 'both', verbose = False)
    mt.discover(seq[0])

    if os.path.isfile(out_csv):
        long_df = pmpiv.df_io.read_csv(os.path.dirname(out_csv) or '.', os.path.basename(out_csv))
        dfs = {label: long_df[long_df['interface'] == label].drop(columns = ['interface']).reset_index(drop = True)
               for label in mt.trackers}
        for label, df in dfs.items():
            mt.trackers[label]._dfs_cached = df  # not used elsewhere; keeps intent obvious
        mt._dfs = dfs
        return mt, dfs

    dfs = mt.track(seq)
    mt.compute_velocity()
    mt.write_csv(os.path.dirname(out_csv) or '.', os.path.basename(out_csv))
    return mt, dfs


def ensure_piv_cache(md_full, seq_full, batch_path, linked_path, verbose = True):
    """Run (or load cached) tp.batch/tp.link/tp.filter_stubs over seq_full."""
    if os.path.isfile(linked_path):
        if verbose:
            print(f'Loading cached linked results from {linked_path}')
        return pd.read_pickle(linked_path)

    if os.path.isfile(batch_path):
        if verbose:
            print(f'Loading cached batch results from {batch_path}')
        df_all = pd.read_pickle(batch_path)
    else:
        if verbose:
            print(f'Running tp.batch on {len(seq_full)} frames...')
        df_all = tp.batch(seq_full, md_full.FEATURE_SIZE, minmass = md_full.FEATURE_MIN_SIZE,
                          invert = md_full.FEATURES_ARE_DARK)
        df_all.to_pickle(batch_path)
        if verbose:
            print(f'Saved batch results to {batch_path}')

    if verbose:
        print('Linking...')
    df_linked = tp.link(df_all, md_full.MAX_PARTICLE_SPEED, memory = md_full.MEMORY)
    df_linked = tp.filter_stubs(df_linked, md_full.DURATION)
    df_linked.to_pickle(linked_path)
    if verbose:
        print(f'Saved linked results to {linked_path}')
    return df_linked


def channel_field(df_linked, md_full, tracker, rec, dt_frames = 1):
    """
    Build the interface-relative velocity field for one channel at one 'ok'
    tracking record. Returns dict(grid, omega, rel) or None if there weren't
    enough particles.
    """
    x0, x1 = tracker.x_search
    frame = int(rec['frame'])

    roi = flow.restrict_to_roi(df_linked, x_range = (x0 - X_PAD, x1 + X_PAD),
                               frame_range = (frame, frame + dt_frames))
    if roi['frame'].nunique() < 2:
        return None

    vel = flow.particle_velocity_frame(roi, frame, m_metadata = md_full, dt_frames = dt_frames)
    if len(vel) < 8:
        return None

    rel = flow.interface_relative_velocity(vel, rec['cx_px'], rec['cy_px'], rec['radius_px'],
                                           x_search = tracker.x_search, branch = 1)
    rel = rel[rel['depth'] <= DEPTH_MAX]
    if len(rel) < 8:
        return None

    grid = flow.velocity_field_grid(rel, x_range = tracker.x_search, depth_range = (0, DEPTH_MAX),
                                    grid_step = GRID_STEP)
    omega = flow.vorticity(grid['X'], grid['Y'], grid['U'], grid['V'], grid['dx'], grid['dy'])

    return dict(grid = grid, omega = omega, rel = rel)


def annotate_frame(label, frame, img, tracker, rec, field, outpath):
    """Quiver overlay + vorticity heatmap with eddy contours for one frame."""
    grid, omega, rel = field['grid'], field['omega'], field['rel']
    eddies = flow.detect_eddies(omega, grid['X'], grid['Y'], grid['dx'], grid['dy'])
    pdepth = flow.penetration_depth(grid['X'], grid['Y'], omega)

    fig, axes = plt.subplots(1, 2, figsize = (14, 7))

    x0, x1 = tracker.x_search
    y0 = int(rec['cy_px'] - rec['radius_px']) if np.isfinite(rec['cy_px']) else int(rec['y_apex_px'] - DEPTH_MAX)
    y1 = int(rec['y_apex_px'] + DEPTH_MAX)

    ax = axes[0]
    ax.imshow(img, cmap = 'gray')
    if frame in tracker._arcs:
        xa, ya = tracker._arcs[frame]
        ax.plot(xa, ya, color = 'cyan', linewidth = 1.5)
    step = max(1, len(rel) // 200)
    ax.quiver(rel['x'].to_numpy()[::step], rel['y'].to_numpy()[::step],
              rel['vx'].to_numpy()[::step], -rel['vy'].to_numpy()[::step],
              color = 'red', scale = None, width = 0.002)
    ax.set_xlim(x0 - 40, x1 + 40)
    ax.set_ylim(y1, y0 - 20)
    ax.set_title(f'{label} frame {frame}: particle velocities')

    ax = axes[1]
    pc = ax.pcolormesh(grid['X'], grid['Y'], omega, shading = 'auto', cmap = 'RdBu_r',
                       vmin = -np.nanpercentile(np.abs(omega), 95), vmax = np.nanpercentile(np.abs(omega), 95))
    plt.colorbar(pc, ax = ax, label = 'vorticity [1/s]')
    for e in eddies:
        circle = plt.Circle((e['core_x'], e['core_y']), e['radius_eq'], fill = False,
                            color = 'lime' if e['dominant'] else 'black', linewidth = 2 if e['dominant'] else 1)
        ax.add_patch(circle)
    if not pdepth['censored']:
        ax.axhline(pdepth['depth'], color = 'black', linestyle = '--', linewidth = 1)
    ax.invert_yaxis()
    ax.set_xlabel('x [px]'); ax.set_ylabel('depth below interface [px]')
    title = f'vorticity (n_eddies={len(eddies)}'
    title += f', depth={pdepth["depth"]:.0f}px)' if not pdepth['censored'] else ', depth=censored)'
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(outpath, dpi = 120)
    plt.close()

    dominant = next((e for e in eddies if e['dominant']), None)
    return dict(frame = frame, n_eddies = len(eddies),
               dominant_radius_px = dominant['radius_eq'] if dominant else np.nan,
               dominant_sign = dominant['sign'] if dominant else np.nan,
               penetration_depth_px = pdepth['depth'], censored = pdepth['censored'])


def analyze_channel(label, md_full, seq_full, df_linked, tracker, track_df, report_dir, verbose = True):
    """Trend series (cheap stride) + representative annotated frames for one channel."""
    ok = track_df[track_df['status'] == 'ok'].reset_index(drop = True)
    if len(ok) == 0:
        if verbose:
            print(f'{label}: no ok frames, skipping.')
        return None

    stride = TREND_STRIDE.get(label, DEFAULT_STRIDE)
    trend_rows = []
    for i in range(0, len(ok), stride):
        rec = ok.iloc[i]
        field = channel_field(df_linked, md_full, tracker, rec)
        if field is None:
            continue
        eddies = flow.detect_eddies(field['omega'], field['grid']['X'], field['grid']['Y'],
                                    field['grid']['dx'], field['grid']['dy'])
        pdepth = flow.penetration_depth(field['grid']['X'], field['grid']['Y'], field['omega'])
        dominant = next((e for e in eddies if e['dominant']), None)
        trend_rows.append(dict(
            frame = int(rec['frame']), n_eddies = len(eddies),
            dominant_radius_px = dominant['radius_eq'] if dominant else np.nan,
            penetration_depth_px = pdepth['depth'], censored = pdepth['censored'],
        ))
        if verbose and i % (stride * 10) == 0:
            print(f'{label} frame {int(rec["frame"])}: n_eddies={len(eddies)}, '
                  f'depth={pdepth["depth"]}, censored={pdepth["censored"]}')

    trend_df = pd.DataFrame(trend_rows)

    rep_idx = np.linspace(0, len(ok) - 1, min(N_REP, len(ok))).round().astype(int)
    rep_idx = sorted(set(rep_idx))
    fig_dir = os.path.join(report_dir, 'figures')
    os.makedirs(fig_dir, exist_ok = True)

    rep_rows = []
    for i in rep_idx:
        rec = ok.iloc[i]
        frame = int(rec['frame'])
        field = channel_field(df_linked, md_full, tracker, rec)
        if field is None:
            continue
        img = np.asarray(seq_full[frame])
        if img.ndim == 3:
            img = img[..., 0]
        outpath = os.path.join(fig_dir, f'{label}_frame{frame:05d}.png')
        rep_rows.append(annotate_frame(label, frame, img, tracker, rec, field, outpath))

    return dict(trend = trend_df, representative = pd.DataFrame(rep_rows), n_rep = len(rep_idx))


def main():
    try:
        metadata_file = sys.argv[1]
    except IndexError:
        raise SystemExit('Usage: python flow_report.py <metadata_file.txt>')

    md = pmpiv.metadata.Metadata(metadata_file)
    seq_all = pmpiv.image_sequence.Image_Sequence(m_metadata = md).read()

    interface_csv = os.path.join(md.WORKING_DIR, 'interface_tracking_full.csv')
    mt, track_dfs = discover_and_track_full(md, seq_all, interface_csv)

    n_final = 0
    for label, df in track_dfs.items():
        n_final = max(n_final, int(df['frame'].max()))
    print(f'Extending PIV coverage to frame {n_final} (channels: '
          + ', '.join(f'{l}<={int(d["frame"].max())}' for l, d in track_dfs.items()) + ')')

    md_full = copy.copy(md)
    md_full.END_FRAME = n_final + 1
    seq_full = pmpiv.image_sequence.Image_Sequence(m_metadata = md_full).subsection_range()

    batch_path  = os.path.join(md.WORKING_DIR, 'df_batch_full.pkl')
    linked_path = os.path.join(md.WORKING_DIR, 'df_linked_full.pkl')
    df_linked = ensure_piv_cache(md_full, seq_full, batch_path, linked_path)

    report_dir = os.path.join(md.WORKING_DIR, 'report')
    os.makedirs(report_dir, exist_ok = True)

    results = {}
    for label, tracker in mt.trackers.items():
        print(f'\n=== Analyzing {label} ===')
        results[label] = analyze_channel(label, md_full, seq_all, df_linked, tracker,
                                         track_dfs[label], report_dir)

    sections = []
    tables = {}
    figures = {}

    sections.append(('Overview', (
        f"Study: 2026\\_marangoni. Frame rate {md.FPS:.0f}\\,fps, pixel size {md.PIXELSIZE*1e6:.2f}\\,$\\mu$m. "
        f"Interfaces analyzed: {', '.join(mt.trackers)}. "
        f"Velocity fields interpolated on a {GRID_STEP:.0f}px grid down to {DEPTH_MAX:.0f}px below each interface."
    )))

    sections.append(('Implementation', (
        "This section describes the code that produced this report, not just the physical "
        "quantities it computes (those are in Section~\\ref{sec:methods}). The pipeline lives "
        "in the \\texttt{pmpiv} package as four modules, driven by two short study scripts.\n\n"
        "\\subsection{Package layout}\n"
        "\\begin{itemize}\n"
        "\\item \\texttt{pmpiv.interface\\_discovery} -- finds channel walls and menisci in a frame "
        "with no hand-set region of interest.\n"
        "\\item \\texttt{pmpiv.interface} -- \\texttt{Interface\\_Tracking} tracks one meniscus "
        "through an image sequence; \\texttt{Multi\\_Interface\\_Tracking} discovers and tracks "
        "every interface in a single pass over the frames.\n"
        "\\item \\texttt{pmpiv.interface\\_flow} -- turns tracked particle trajectories and interface "
        "geometry into interface-relative velocities, a vorticity field, detected eddies and a "
        "penetration depth.\n"
        "\\item \\texttt{pmpiv.report} -- assembles this document and compiles it with "
        "\\texttt{pdflatex}.\n"
        "\\item \\texttt{studies/2026\\_marangoni/flow\\_report.py} -- the driver that produced this "
        "report; \\texttt{plot\\_full\\_series.py} -- a companion script producing an annotated PNG "
        "for every tracked frame (interface arcs plus the particles attributed to each), too large "
        "to embed here.\n"
        "\\end{itemize}\n\n"
        "\\subsection{Interface discovery}\n"
        "\\texttt{discover\\_interfaces} first finds the dark vertical channel walls from a "
        "column-wise intensity profile of the lower part of the frame, then classifies each gap "
        "between walls as liquid-filled or empty by the standard deviation of its high-pass-filtered "
        "texture (tracer beads make a liquid channel visibly noisier than an empty one). For each "
        "liquid channel it seeds the initial search band either from a Hough-circle detection of the "
        "meniscus arc, or, if none is found, from a coarse scan of candidate bands scored by how "
        "cleanly the ridge detector (below) responds in each -- this scan is what makes discovery "
        "work even when the arc is too shallow or noisy for the Hough transform.\n\n"
        "\\subsection{Interface tracking}\n"
        "For each frame, \\texttt{Interface\\_Tracking.step} crops a band around the last known "
        "position, denoises it (non-local means) and high-pass filters it so the meniscus survives "
        "as a dark ridge while tracer-bead texture is suppressed, then locates that ridge's centre "
        "per image column. A circle is fit to the ridge points with RANSAC, which tolerates a "
        "fraction of columns being corrupted by nearby beads or wall artefacts. A frame is only "
        "accepted (\\texttt{status = ok}) if enough columns agreed, the fit residual is small and "
        "the ridge stood out clearly from the background; otherwise it is marked \\texttt{weak} and "
        "the previous band is kept. The band is a fixed height, re-centred each frame on a "
        "short-horizon velocity prediction rather than left static -- a band tall enough to "
        "statically cover the interface's whole range of motion would, once blurred, be dominated "
        "by bead texture rather than the meniscus itself, which was confirmed to break detection on "
        "this data before the moving-band design was adopted. Tracking ends (\\texttt{lost} or "
        "\\texttt{out\\_of\\_frame}) once too many consecutive frames fail or the interface reaches "
        "the edge of the image.\n\n"
        "\\subsection{Particle velocities, eddies and penetration depth}\n"
        "Tracer-bead trajectories are located and linked frame-to-frame with trackpy "
        "(\\texttt{tp.batch}, \\texttt{tp.link}), cached to disk since this is the most expensive "
        "step. \\texttt{interface\\_flow.particle\\_velocity\\_frame} turns consecutive-frame "
        "displacement into a physical velocity per particle; "
        "\\texttt{interface\\_relative\\_velocity} then expresses it in coordinates tied to the "
        "interface (depth below it, and velocity tangential/normal to its local slope) using the "
        "circle fitted for that frame. The velocities of particles within a fixed depth window are "
        "interpolated onto a regular grid (\\texttt{velocity\\_field\\_grid}) and differentiated to "
        "vorticity (\\texttt{vorticity}). \\texttt{detect\\_eddies} labels connected regions of "
        "same-signed vorticity above an adaptive (percentile-based) noise floor as recirculation "
        "cells and reports their equivalent radius; \\texttt{penetration\\_depth} reads off the depth "
        "at which the vorticity magnitude, averaged across the channel width, decays to a fixed "
        "fraction of its near-interface value.\n\n"
        "\\subsection{Report generation}\n"
        "\\texttt{flow\\_report.py} calls each of the above in turn for every discovered interface, "
        "saves the resulting figures and summary tables, and hands them to "
        "\\texttt{pmpiv.report.build\\_latex\\_report}, which writes this document as a \\texttt{.tex} "
        "file and compiles it. Everything expensive (interface discovery over the full sequence, "
        "particle linking) is cached to disk, so re-running the driver after a code change to the "
        "analysis or report layout only redoes the cheap parts."
    )))

    for label, res in results.items():
        if res is None:
            continue
        tracker = mt.trackers[label]
        trend = res['trend']
        tdf = track_dfs[label]

        vel_col = 'velocity_savgol_m_s' if tdf['velocity_savgol_m_s'].notna().any() else 'velocity_rolling_mean_m_s'
        vel_mm_s = tdf[vel_col] * 1e3

        fig, ax = plt.subplots(1, 2, figsize = (11, 4))
        ax[0].plot(tdf['frame'], tdf['y_median_px'] * md.PIXELSIZE * 1e3, '-')
        ax[0].set_xlabel('frame'); ax[0].set_ylabel('interface position [mm]')
        ax[0].set_title('interface retreat position')
        ax[1].plot(tdf['frame'], vel_mm_s, '-')
        ax[1].set_xlabel('frame'); ax[1].set_ylabel('retreat velocity [mm/s]')
        ax[1].set_title('interface retreat velocity')
        plt.tight_layout()
        vel_fig = os.path.join(report_dir, 'figures', f'{label}_interface_velocity.png')
        os.makedirs(os.path.dirname(vel_fig), exist_ok = True)
        plt.savefig(vel_fig, dpi = 120)
        plt.close()
        figures[f'{label}_interface_velocity.png'] = vel_fig

        vel_ok = vel_mm_s[tdf['status'] == 'ok'].dropna()
        vel_summary = (
            f"Interface retreat velocity (Savitzky-Golay smoothed): "
            f"mean {vel_ok.mean():.3f}\\,mm/s, peak {vel_ok.max():.3f}\\,mm/s "
            f"(frame {int(tdf.loc[vel_ok.idxmax(), 'frame'])})."
            if len(vel_ok) else "Interface retreat velocity: insufficient data."
        )

        fig, ax = plt.subplots(1, 2, figsize = (11, 4))
        ax[0].plot(trend['frame'], trend['dominant_radius_px'] * md.PIXELSIZE * 1e6, 'o-')
        ax[0].set_xlabel('frame'); ax[0].set_ylabel('dominant eddy radius [$\\mu$m]')
        ax[1].plot(trend['frame'], trend['penetration_depth_px'] * md.PIXELSIZE * 1e6, 'o-')
        ax[1].plot(trend.loc[trend['censored'], 'frame'],
                   np.full(trend['censored'].sum(), np.nanmax(trend['penetration_depth_px']) * md.PIXELSIZE * 1e6),
                   'rx', label = 'censored (no decay within window)')
        ax[1].set_xlabel('frame'); ax[1].set_ylabel('penetration depth [$\\mu$m]')
        ax[1].legend(fontsize = 8)
        plt.tight_layout()
        trend_fig = os.path.join(report_dir, 'figures', f'{label}_trend.png')
        os.makedirs(os.path.dirname(trend_fig), exist_ok = True)
        plt.savefig(trend_fig, dpi = 120)
        plt.close()
        figures[f'{label}_trend.png'] = trend_fig

        body = (
            f"Wall $x$-range: {tracker.wall_x}. Tracked {int(track_dfs[label]['status'].eq('ok').sum())} "
            f"`ok' frames (frame {int(track_dfs[label]['frame'].min())}--{int(track_dfs[label]['frame'].max())}). "
            f"{vel_summary}\n\n"
            f"\\includegraphics[width=\\linewidth]{{{label}_interface_velocity.png}}\n\n"
            f"\\includegraphics[width=\\linewidth]{{{label}_trend.png}}\n\n"
            "{{table:%s_summary}}\n\n" % label
        )
        for rf in sorted(res['representative']['frame']) if len(res['representative']) else []:
            fname = f'{label}_frame{int(rf):05d}.png'
            figures[fname] = os.path.join(report_dir, 'figures', fname)
            body += f"\\includegraphics[width=\\linewidth]{{{fname}}}\n\n"

        summary = trend.describe().loc[['mean', 'std', 'min', 'max']].reset_index()
        tables[f'{label}_summary'] = summary

        sections.append((f'Channel {label}', body))

    sections.append(('Methods and limitations', (
        "\\label{sec:methods}\n"
        "Particle velocities are obtained from trackpy-linked trajectories via consecutive-frame "
        "displacement (pmpiv.motion\\_stats), decomposed into components tangential/normal to the "
        "tracked circular-cap interface. A velocity field is interpolated (linear griddata) below the "
        "interface on a regular grid, lightly Gaussian-smoothed, and differentiated to vorticity "
        "$\\omega = \\partial v/\\partial x - \\partial u/\\partial y$. Eddies are connected regions of "
        "same-signed vorticity above a per-frame noise floor; their size is the equivalent radius of the "
        "$|\\omega| > 0.3\\times$peak contour. Penetration depth is the depth at which the "
        "$x$-averaged $|\\omega|$ profile decays to 10\\% of its near-interface value; frames where it "
        "never decays within the search window are reported as censored, not silently truncated. "
        "Interpolating and differentiating a sparse scattered field is the main source of numerical "
        "noise in these results; treat single-frame eddy sizes as order-of-magnitude estimates."
    )))

    report_path = pmpiv.report.build_latex_report(
        md.WORKING_DIR, title = 'Marangoni interface flow analysis',
        sections = sections, tables = tables, figures = figures,
    )
    print(f'\nReport written to {report_path}')


if __name__ == '__main__':
    main()
