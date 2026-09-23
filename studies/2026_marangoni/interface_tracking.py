#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interface tracking driver for the Marangoni PIV experiment.

Discovers every curved interface (meniscus) in the first frame — channel
walls + Hough circles, no hand-set ROI — tracks each one through the
sequence and writes per-interface positions, radii, contact angles and
retreat velocities. All logic lives in pmpiv.interface /
pmpiv.interface_discovery; this script only wires it to the metadata.

Usage:
    python interface_tracking.py studies/2026_marangoni/2026_marangoni.txt

Outputs:
    {WORKING_DIR}/interface_tracking.csv         long format, one row per
                                                 interface and frame
    {WORKING_DIR}/tiffs_interface/frame_*.tif   annotated TIFFs
"""

from __future__ import division, unicode_literals, print_function

import sys

import pmpiv


def main():
    try:
        metadata_file = sys.argv[1]
    except IndexError:
        raise SystemExit('Usage: python interface_tracking.py <metadata_file.txt>')

    md = pmpiv.metadata.Metadata(metadata_file)

    image_sequence = pmpiv.image_sequence.Image_Sequence(m_metadata = md)
    frames = image_sequence.subsection_range()

    tracker = pmpiv.interface.Multi_Interface_Tracking(m_metadata = md, discovery = 'both')
    tracker.discover(frames[0])
    # Manual seed if discovery misses an interface:
    # tracker.add_interface('extra', x_search = (1357, 1741), y_search = (1000, 1300), wall_x = (1357, 1741))

    tracker.track(frames)
    tracker.compute_velocity()

    tracker.write_csv(md.WORKING_DIR, 'interface_tracking.csv')
    tracker.plot_annotated_tiffs(f'{md.WORKING_DIR}/tiffs_interface', frames)

    print('\nInterface tracking complete.')


if __name__ == '__main__':
    main()
