#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 08 2024

@author: David Krach 
         david.krach@mib.uni-stuttgart.de
"""

### HEADER ------------------------------------------------------------------------
from __future__ import division, unicode_literals, print_function 

import numpy as np
import os, glob, sys
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
import time
from ast import literal_eval
import numba

import trackpy as tp        # https://soft-matter.github.io/trackpy/v0.6.1/index.html
import pims                 # https://soft-matter.github.io/pims/v0.6.1/
import pmpiv as pmpiv

###--------------------------------------------------------------------------------

# GET USER DEFINED PARAMS 
try:
    metadata_file = sys.argv[1]
except:
    raise FileNotFoundError('Please define input file!')

md = pmpiv.metadata.Metadata(metadata_file)


# define image_sequence <class 'pims.image_sequence.ImageSequence'>
image_sequence = pmpiv.image_sequence.Image_Sequence(m_metadata = md)

# Activate parallel ploting
image_sequence.plot_parallel()

# Extract subsection <class 'slicerator.Slicerator'>
frames = image_sequence.subsection_range()

# Compute statistics to see if batching is good
# It is only used later for percentile filtering
# Do not compute if not necessary, since expensive
_sq_stats = pmpiv.fstats.Sequence_Statistics(frames, m_metadata = md)
sq_stats = _sq_stats.get()
# print(sq_stats)

# Locate Gaussian-like blobs of some approximate size in the set of images
# This takes place fully parallel
#tp.quiet()  # Turn off progress reports for best performance
df_all = tp.batch(frames, md.FEATURE_SIZE, minmass=md.FEATURE_MIN_SIZE, invert = md.FEATURES_ARE_DARK)

# Linking particle positions 
# sequential computation
df_filtered_linked = tp.link(df_all, md.MAX_PARTICLE_SPEED, memory = md.MEMORY)

# Remove spurious trajectories
df_filtered_sp = tp.filter_stubs(df_filtered_linked, md.DURATION)

# Save all trajectory data in a copy
df_all = df_filtered_sp.copy(deep=True)
dfs = {}

for s in range(len(md.EXTRACTION)):
    selection_ann_handler = pmpiv.annotations.Annotation_Handler(f'{md.JSON_PATH}/{md.EXTRACTION[s]}', m_metadata = md)
    filtered_area         = selection_ann_handler.area()
    print(f'Extracted Area: {filtered_area}')

    filter_selection_df   = selection_ann_handler.annotations2DF()
    selection_ann_filter  = pmpiv.filtering.Annotation_Filtering(df_all.copy(deep = True), filter_selection_df)
    df_filtered           = selection_ann_filter.extract()

    key = md.EXTRACTION[s].replace('.json', '')
    df_save = df_filtered.copy(deep = True)
    df_filter_static = df_filtered.copy(deep = True)
    # dfs[key] = df_save

    pmpiv.df_io.write2csv(df_filtered, md.WORKING_DIR, f'df_{key}_short.csv')


    # Remove static trajectories--------------------------------------------------------------------
    static_handler   = pmpiv.filtering.Static_Filtering(df_filter_static, m_metadata = md)
    df_filter_static = static_handler.remove()

    pmpiv.df_io.write2csv(df_filter_static, md.WORKING_DIR, f'df_{key}_fstatic_short.csv')
