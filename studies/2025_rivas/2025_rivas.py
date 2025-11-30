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
import scipy

import pandas as pd
import time
from ast import literal_eval
import numba

import trackpy as tp
import pims
import pmpiv as pmpiv

# import pims             # https://soft-matter.github.io/pims/v0.6.1/
# import trackpy as tp    # https://soft-matter.github.io/trackpy/v0.6.1/index.html

# sys.path.append('src')
# import fstats, image_sequence, filtering, metadata, annotations
# import df_io

###--------------------------------------------------------------------------------

# GET USER DEFINED PARAMS 
try:
    metadata_file = sys.argv[1]
except:
    raise FileNotFoundError('Please define input file!')

md = pmpiv.metadata.Metadata(metadata_file)

# Define image sequence class
#  <class 'pims.image_sequence.ImageSequence'>
image_sequence = pmpiv.image_sequence.Image_Sequence(m_metadata = md)
# Activate parallel ploting
image_sequence.plot_parallel()


# TASK = "BATCH"
TASK = "VELOCITY"
# TASK = "ALL"

if TASK == "BATCH" or TASK == "ALL":

    # Extract subsection <class 'slicerator.Slicerator'>
    frames = image_sequence.subsection_range()

    # Annotation of frame 0 to doi frame statistics
    df0 = tp.locate(frames[0], md.FEATURE_SIZE, invert = md.FEATURES_ARE_DARK, minmass = md.FEATURE_MIN_SIZE)
    hmf1 = pmpiv.fstats.Frame_Statistics(df0, m_metadata = md)
    hmf1.histogram('mass', bins = 50)


    # Compute statistics to see if batching is good
    # It is only used later for percentile filtering
    # Do not compute if not necessary, since expensive
    _sq_stats = pmpiv.fstats.Sequence_Statistics(frames, m_metadata = md)
    # sq_stats = _sq_stats.get(force_recompute = True)
    sq_stats = _sq_stats.get()
    # print(sq_stats)

    # Locate Gaussian-like blobs of some approximate size in the set of images
    # This takes place fully parallel
    #tp.quiet()  # Turn off progress reports for best performance
    df_all = tp.batch(frames, md.FEATURE_SIZE, minmass=md.FEATURE_MIN_SIZE, invert = md.FEATURES_ARE_DARK)

    # Linking particle positions
    df_filtered = tp.link(df_all.copy( deep = True ), md.MAX_PARTICLE_SPEED, memory = 0)

    df_filtered = pmpiv.filtering.complete_removal( df_filtered.copy( deep = True ), m_metadata = md )
    # image_sequence.plot_annotated_pngs(df_filtered, os.path.join(md.WORKING_DIR, f'png_filtered_removal_all'), color = 'red')
    image_sequence.plot_annotated_tiffs(df_filtered, os.path.join(md.WORKING_DIR, f'tiff_filtered_removal_all'), color = 'red')


if TASK == "VELOCITY" or TASK == "ALL":
    
    df_filtered = pmpiv.df_io.read_csv( md.WORKING_DIR, 'df_complete_removal.csv' )

    # df_filtered = pmpiv.filter_static(df_filtered, m_metadata = md)
    # image_sequence.plot_annotated_tiffs(df_filtered, os.path.join(md.WORKING_DIR, f'tiff_filtered_static_all'), color = 'purple')
    # df_filtered = pmpiv.filter_stubs(df_filtered, m_metadata = md)
    # image_sequence.plot_annotated_tiffs(df_filtered, os.path.join(md.WORKING_DIR, f'tiff_filtered_stubs_all'), color = 'orange')
    
    pmpiv.ploting.ymapped_velocity(df_filtered, m_metadata = md)
    pmpiv.ploting.trajectory(df_filtered, m_metadata = md)







