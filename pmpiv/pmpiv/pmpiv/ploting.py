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
import os, glob
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd

import pims             # https://soft-matter.github.io/pims/v0.6.1/
import trackpy as tp    # https://soft-matter.github.io/trackpy/v0.6.1/index.html

import pmpiv
###--------------------------------------------------------------------------------

def trajectory( df, 
                m_metadata = None, 
                m_metadata_file = None, 
                figsize = [np.float64(43.52), np.float64(32.571999999999996)],
                fn = str('traj'),
                verbose = True):
    """
    """

    # check if metadata is avail
    try:
        m_metadata = pmpiv.helper._check_metadata(m_metadata, m_metadata_file)
    except:
        raise FileNotFoundError('Metadata information not available!')


    fig, ax = plt.subplots(1, 1, figsize = figsize)
    image = tp.plot_traj(df, ax = ax)
    plt.savefig(f'{m_metadata.WORKING_DIR}/{fn}.pdf')
    plt.savefig(f'{m_metadata.WORKING_DIR}/{fn}.png')
    plt.close()
    plt.clf()


def ymapped_velocity(df, 
                     m_metadata = None, 
                     m_metadata_file = None, 
                     figsize = [np.float64(43.52), np.float64(32.571999999999996)],
                     fn = str('mapped_vel'),
                     verbose = True):

    # check if metadata is avail
    try:
        m_metadata = pmpiv.helper._check_metadata(m_metadata, m_metadata_file)
    except:
        raise FileNotFoundError('Metadata information not available!')


    total_vel = pmpiv.motion_stats.Motion_Statistics(df, m_metadata = m_metadata)

    # y-positions, x velocities
    ys = []
    vx = []

    # get list of all frames
    all_frames = np.unique((df['frame']).to_numpy())

    for i in range(all_frames.shape[0]-1):
        tmp_df = total_vel.displacement_2frames(i, i+1)
        ys += tmp_df['y'].to_list()
        vx += tmp_df['dx'].to_list()

    # Bin data to get mean per position
    bins = np.linspace(0, 1913, int(1914))
    data = np.asarray(ys)
    digitized = np.digitize(data, bins)

    velsx = np.abs(np.asarray(vx))

    pd_data = pd.DataFrame({'bin': digitized, 'velocity': velsx})

    pd_data = pd_data.loc[pd_data['velocity'] > 0.5]

    means = pd_data.groupby('bin').mean().reset_index()

    moving_avg = means['velocity'].rolling(window = 30).mean()

    fig, ax = plt.subplots(1, 1, figsize = figsize)
    plt.plot(means['bin'].to_numpy(), moving_avg)
    plt.savefig(f'{m_metadata.WORKING_DIR}/{fn}.pdf')
    plt.savefig(f'{m_metadata.WORKING_DIR}/{fn}.png')
    plt.close()
    plt.clf()



