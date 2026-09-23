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
from scipy.ndimage import gaussian_filter, median_filter
import cv2

import pmpiv as pmpiv
from pmpiv.interface import preprocess_band, detect_ridge, circle_arc

###--------------------------------------------------------------------------------

DISCOVERY_METHODS = ('walls', 'hough', 'both')

_PREPROCESS_KEYS = ('mode', 'blur_sigma', 'nlm_h', 'highpass_sigma')
_RIDGE_KEYS      = ('ridge_sigma', 'continuity_window', 'continuity_tolerance')


def column_profile(img, y_from, y_to = None):
    """Column-mean intensity of img[y_from:y_to]."""
    return img[y_from:y_to].astype(float).mean(axis = 0)


def _dark_runs(profile, rel_threshold, min_width, max_width, merge_gap = 3):
    """
    Contiguous index runs where profile / local_baseline < rel_threshold,
    merged when closer than merge_gap and filtered by width.
    """
    n = len(profile)
    size = min(201, n if n % 2 == 1 else n - 1)
    baseline = median_filter(profile, size = max(3, size), mode = 'nearest')
    ratio = profile / np.maximum(baseline, 1e-9)
    dark = ratio < rel_threshold

    edges  = np.diff(dark.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0] + 1)
    if dark[0]:
        starts = [0] + starts
    if dark[-1]:
        ends = ends + [n]

    runs = []
    for s, e in zip(starts, ends):
        if runs and s - runs[-1][1] <= merge_gap:
            runs[-1] = (runs[-1][0], e)
        else:
            runs.append((s, e))

    return [(int(s), int(e)) for s, e in runs if min_width <= e - s <= max_width]


def find_channel_walls(img, y_from = None, rel_threshold = 0.7, min_width = 6, max_width = 40):
    """
    Dark vertical wall bands as [(x0, x1)], from the column-mean profile of
    the lower part of the image (default rows from 0.55*H).
    """
    if y_from is None:
        y_from = int(0.55 * img.shape[0])
    return _dark_runs(column_profile(img, y_from), rel_threshold, min_width, max_width)


def _wall_top(img, x0, x1, dark_ratio, pad = 4, min_run = 40):
    """
    Start row of the first dark run of at least min_run rows in the wall
    columns (widened by pad, minimum over columns so a slightly tilted
    wall still counts), or None. Short dark runs (border line, a bead
    next to the wall) are skipped.
    """
    prof = img[:, max(0, x0 - pad):x1 + pad].astype(float).min(axis = 1)
    dark = prof < dark_ratio * float(np.median(img))
    if not dark.any():
        return None
    edges  = np.diff(dark.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0] + 1)
    if dark[0]:
        starts = [0] + starts
    if dark[-1]:
        ends = ends + [len(dark)]
    for s, e in zip(starts, ends):
        if e - s >= min_run:
            return int(s)
    return None


def find_top_wall(img, walls = None, x_range = None, dark_ratio = 0.6, min_fraction = 0.5,
                  min_width = 4, max_width = 120):
    """
    y of the channel mouth (top of the channel walls), or None.

    With `walls` (from find_channel_walls): the vertical walls exist only
    below the mouth, so the top end of the longest dark run in each wall's
    own columns marks it (median over walls). If the walls reach the top
    of the image this is uninformative and the row method below is used.

    Row method: bottom edge of the lowest near-full-width dark horizontal
    line within x_range; a row counts when at least min_fraction of its
    pixels are darker than dark_ratio * median intensity. Restrict x_range
    to a solid (non-channel) region when possible: a meniscus apex can
    darken ~25 % of a row. The lowest line is taken because the
    field-of-view border can add a dark line at the very top of the image.
    """
    if walls:
        tops = [t for t in (_wall_top(img, x0, x1, dark_ratio) for x0, x1 in walls) if t is not None]
        if tops:
            top = int(np.median(tops))
            if top > 0:
                return top

    x0, x1 = (0, img.shape[1]) if x_range is None else x_range
    sub  = img[:, x0:x1].astype(float)
    dark_level = dark_ratio * float(np.median(sub))
    frac = (sub < dark_level).mean(axis = 1)
    rows = frac >= min_fraction

    edges  = np.diff(rows.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    ends   = list(np.where(edges == -1)[0] + 1)
    if rows[0]:
        starts = [0] + starts
    if rows[-1]:
        ends = ends + [len(rows)]

    found = None
    for s, e in zip(starts, ends):
        if min_width <= e - s <= max_width:
            found = int(e)
    return found


def channel_texture(img, x0, x1, y_from, y_to = None):
    """Std of the high-passed crop img[y_from:y_to, x0:x1] (8-bit units)."""
    crop = img[y_from:y_to, x0:x1].astype(float)
    if crop.size == 0:
        return 0.0
    hp = crop - gaussian_filter(crop, 10)
    return float(hp.std())


def channels_from_walls(img, walls, y_from, texture_threshold = 14.0,
                        include_border = False, verbose = True):
    """
    Gaps between consecutive walls as [dict(x0, x1, texture, liquid)].
    A gap is 'liquid' when its high-pass texture (std, 8-bit units, rows
    from y_from) exceeds texture_threshold: bead-seeded liquid measured
    ~23 on the reference data, empty solid gaps 7-9 (sensor noise plus
    faint dirt). Textures are printed when verbose to help retuning.
    """
    W = img.shape[1]
    walls = sorted(walls)
    gaps = [(walls[i][1], walls[i + 1][0]) for i in range(len(walls) - 1)]
    if include_border and walls:
        gaps = [(0, walls[0][0])] + gaps + [(walls[-1][1], W)]

    channels = []
    for x0, x1 in gaps:
        if x1 - x0 < 4:
            continue
        tex = channel_texture(img, x0, x1, y_from)
        liquid = tex >= texture_threshold
        channels.append(dict(x0 = int(x0), x1 = int(x1), texture = tex, liquid = bool(liquid)))
        if verbose:
            print(f'Channel gap x=[{x0}, {x1}): texture={tex:.2f} -> {"liquid" if liquid else "empty"}')
    return channels


def find_menisci_hough(img, blur_sigma = 6, dp = 1.5, min_dist = 150, param1 = 80,
                       param2_range = (40, 25, -5), min_radius = 120, max_radius = 450):
    """
    Circular caps via cv2.HoughCircles, sweeping the accumulator threshold
    from strict to loose and keeping new circles farther than min_dist from
    already found ones. Circles whose apex (cy + r) lies outside the image
    are dropped. Returns [(cx, cy, r)].
    """
    H = img.shape[0]
    img8 = img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(np.ascontiguousarray(img8), (0, 0), float(blur_sigma))

    found = []
    for p2 in range(*param2_range):
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp = dp, minDist = min_dist,
                                   param1 = param1, param2 = p2,
                                   minRadius = int(min_radius), maxRadius = int(max_radius))
        if circles is None:
            continue
        for cx, cy, r in circles[0]:
            if not (0 <= cy + r <= H):
                continue
            if any(np.hypot(cx - fx, cy - fy) < min_dist for fx, fy, _ in found):
                continue
            found.append((float(cx), float(cy), float(r)))
    return found


def _split_detector_kwargs(detector_kwargs):
    kw = dict(detector_kwargs or {})
    pre = {k: kw[k] for k in _PREPROCESS_KEYS if k in kw}
    rid = {k: kw[k] for k in _RIDGE_KEYS if k in kw}
    return pre, rid


def scan_channel_for_band(img, x_search, y_from, y_to, band_height, detector_kwargs = None):
    """
    Slide a band of band_height down the channel columns and return the
    window (y0, y1, score) where the ridge detector responds best
    (score = valid-column fraction * median SNR).
    """
    pre_kw, ridge_kw = _split_detector_kwargs(detector_kwargs)
    y_to = min(int(y_to), img.shape[0])
    y_from = max(0, int(y_from))
    band_height = int(band_height)

    if y_to - y_from <= band_height:
        starts = [y_from]
    else:
        step = max(1, band_height // 2)
        starts = list(range(y_from, y_to - band_height + 1, step))
        if starts[-1] != y_to - band_height:
            starts.append(y_to - band_height)

    best = None
    for y0 in starts:
        y1 = min(y_to, y0 + band_height)
        if y1 - y0 < 2:
            continue
        band, _ = preprocess_band(img, x_search, (y0, y1), **pre_kw)
        x_idx, y_idx, snr = detect_ridge(band, **ridge_kw)
        if len(x_idx) == 0:
            score, y_c = 0.0, None
        else:
            score = (len(x_idx) / band.shape[1]) * float(np.median(snr))
            y_c = y0 + float(np.median(y_idx))
        if best is None or score > best[2]:
            best = (int(y0), int(y1), float(score), y_c)

    if best is None:
        return None
    y0, y1, score, y_c = best
    # Re-center on the detected ridge so the interface is not left sitting
    # at the window edge (where points get clamped and the first fit is biased).
    if y_c is not None:
        y0, y1 = _band_around(y_c, band_height, img.shape[0])
    return (int(y0), int(y1), float(score))


def _band_around(y_center, band_height, H):
    y0 = int(round(y_center - band_height / 2.0))
    y0 = max(0, min(y0, H - 2))
    y1 = min(H, y0 + int(band_height))
    return (y0, y1)


def discover_interfaces(img, method = 'both', band_height = 300, inset = 8,
                        walls_kwargs = None, hough_kwargs = None, detector_kwargs = None,
                        texture_threshold = 14.0, texture_y_from = None,
                        include_border = False, verbose = True):
    """
    Find curved interfaces in a single frame.

    'walls': detect vertical channel walls, keep gaps with bead texture,
             locate the band per channel with scan_channel_for_band.
    'hough': circular caps from cv2.HoughCircles anywhere in the frame.
    'both' : channels from the walls; a Hough circle inside a channel seeds
             its band, otherwise the scan does.

    Returns a list of seeds: dict(label, x_search, y_search, wall_x,
    seed_type, circle), ordered left to right.
    """
    if method not in DISCOVERY_METHODS:
        raise ValueError(f'method must be in {DISCOVERY_METHODS}')

    img = np.asarray(img)
    if img.ndim == 3:
        img = img[..., 0]
    H, W = img.shape
    seeds = []

    if method == 'hough':
        circles = find_menisci_hough(img, **(hough_kwargs or {}))
        for cx, cy, r in sorted(circles, key = lambda c: c[0]):
            x0 = max(0, int(round(cx - 0.8 * r)))
            x1 = min(W, int(round(cx + 0.8 * r)))
            if x1 - x0 < 8:
                continue
            ys = circle_arc(cx, cy, r, np.arange(x0, x1))
            if not np.any(np.isfinite(ys)):
                continue
            seeds.append(dict(label = f'ch{len(seeds)}', x_search = (x0, x1),
                              y_search = _band_around(float(np.nanmean(ys)), band_height, H),
                              wall_x = None, seed_type = 'hough', circle = (cx, cy, r)))
        return seeds

    walls = find_channel_walls(img, **(walls_kwargs or {}))
    # Texture is judged on the lower part of the image only: below any
    # meniscus a liquid channel is full of beads while an empty gap is flat,
    # whereas the region above the channel mouths is busy in both.
    if texture_y_from is None:
        texture_y_from = int(0.55 * H)
    channels = channels_from_walls(img, walls, texture_y_from, texture_threshold = texture_threshold,
                                   include_border = include_border, verbose = verbose)
    liquid = [c for c in channels if c['liquid']]

    # Measure the top wall over solid gaps (clean full-width line there);
    # fall back to the whole width.
    top = find_top_wall(img, walls = walls)
    if top is None:
        for c in sorted((c for c in channels if not c['liquid']), key = lambda c: c['x0'] - c['x1']):
            top = find_top_wall(img, x_range = (c['x0'], c['x1']))
            if top is not None:
                break
    if top is None:
        top = find_top_wall(img)
    # Start searching half a band above the mouth: a fresh meniscus can sit
    # right at the channel opening. Inside the channel columns the mouth
    # line itself does not exist, so the band cannot lock onto it.
    y_from = max(0, top - band_height // 2) if top is not None else 0
    if verbose:
        print(f'Walls: {walls}; top wall y={top}; search from y={y_from}')

    circles = find_menisci_hough(img, **(hough_kwargs or {})) if method == 'both' else []
    used = set()

    for c in liquid:
        x_search = (c['x0'] + inset, c['x1'] - inset)
        if x_search[1] - x_search[0] < 8:
            continue
        wall_x = (float(c['x0']), float(c['x1']))

        assigned = [(k, circ) for k, circ in enumerate(circles)
                    if k not in used and c['x0'] < circ[0] < c['x1'] and y_from < circ[1] + circ[2] < H]
        if assigned:
            k, (cx, cy, r) = max(assigned, key = lambda kc: kc[1][2])
            used.add(k)
            ys = circle_arc(cx, cy, r, np.arange(*x_search))
            if np.any(np.isfinite(ys)):
                seeds.append(dict(label = f'ch{len(seeds)}', x_search = x_search,
                                  y_search = _band_around(float(np.nanmean(ys)), band_height, H),
                                  wall_x = wall_x, seed_type = 'hough', circle = (cx, cy, r)))
                continue

        best = scan_channel_for_band(img, x_search, y_from, H, band_height, detector_kwargs)
        if best is None:
            continue
        seeds.append(dict(label = f'ch{len(seeds)}', x_search = x_search,
                          y_search = (best[0], best[1]), wall_x = wall_x,
                          seed_type = 'scan', circle = None))

    if verbose:
        for k, circ in enumerate(circles):
            if k not in used:
                print(f'Hough circle {tuple(round(v) for v in circ)} not inside any liquid channel; dropped.')

    return seeds
