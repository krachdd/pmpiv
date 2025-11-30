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
import multiprocessing
import collections

import pims             # https://soft-matter.github.io/pims/v0.6.1/
import trackpy as tp    # https://soft-matter.github.io/trackpy/v0.6.1/index.html


import pmpiv as pmpiv

###--------------------------------------------------------------------------------


class Motion_Statistics:
    """
    
    Compute statistics of Motion

    frame : <class 'pandas.core.frame.DataFrame'>

    save : bool 

    """

    def __init__(self, m_df, 
                 m_metadata = None, m_metadata_file = None, 
                 verbose = True):
        """
        """
        self.m_df = m_df
        self.verbose = verbose

        # check if metadata is avail
        try:
            self.m_metadata = pmpiv.helper._check_metadata(m_metadata, m_metadata_file)
        except:
            raise FileNotFoundError('Metadata information not available!')

        
    def displacement_2frames(self, frame1, frame2):
        """
        """
        # sanity
        frame1 = int(frame1)
        frame2 = int(frame2)

        if frame2 == frame1:
            raise ValueError('Frame number must not be the same!')
        if frame2 < frame1:
            raise ValueError('Frame number of frame2 is smaller than frame1. This is not allowed.')

        return tp.motion.relate_frames(self.m_df, frame1, frame2)


    def mean_displacement_2frames(self, frame1, frame2):
        """
        """
        rdict = {}
        df_displacement = self.displacement_2frames(frame1, frame2)
        _mean_displacement = df_displacement.mean(axis=0, 
                                                  skipna=True, 
                                                  numeric_only=True)

        rdict['dx'] = _mean_displacement['dx']
        rdict['dy'] = _mean_displacement['dy']
        rdict['dr'] = _mean_displacement['dr']

        return rdict

    def abs_mean_displacement_2frames(self, frame1, frame2):
        """
        """
        rdict = {}
        df_displacement = self.displacement_2frames(frame1, frame2)
        
        df_displacement['dx'] = df_displacement['dx'].abs()
        df_displacement['dy'] = df_displacement['dy'].abs()
        df_displacement['dr'] = df_displacement['dr'].abs()

        _mean_displacement = df_displacement.mean(axis=0, 
                                                  skipna=True, 
                                                  numeric_only=True)

        rdict['dx'] = _mean_displacement['dx']
        rdict['dy'] = _mean_displacement['dy']
        rdict['dr'] = _mean_displacement['dr']

        return rdict


    def max_displacement_2frames(self, frame1, frame2):
        """
        """
        rdict = {}
        df_displacement = self.displacement_2frames(frame1, frame2)
        _max_displacement = df_displacement.max(axis=0, 
                                                  skipna=True, 
                                                  numeric_only=True)

        rdict['dx'] = _max_displacement['dx']
        rdict['dy'] = _max_displacement['dy']
        rdict['dr'] = _max_displacement['dr']

        return rdict

    def mean_displacement_allframes(self, freq = 1):
        """
        """
        rdict = {}
        
        all_frames = np.unique((self.m_df['frame']).to_numpy())

        dx, dy, dr = [], [], []

        if freq != 1:
            raise NotImplementedError

        for i in range(all_frames.shape[0]-1):
            _rdict = self.mean_displacement_2frames(i, i+1)
            dx.append(_rdict['dx'])
            dy.append(_rdict['dy'])
            dr.append(_rdict['dr'])

        rdict['dx'] = np.mean(np.asarray(dx))
        rdict['dy'] = np.mean(np.asarray(dy))
        rdict['dr'] = np.mean(np.asarray(dr))

        return rdict


    def mean_abs_displacement_allframes(self, freq = 1):
        """
        """
        rdict = {}
        
        all_frames = np.unique((self.m_df['frame']).to_numpy())

        dx, dy, dr = [], [], []

        if freq != 1:
            raise NotImplementedError

        for i in range(all_frames.shape[0]-1):
            _rdict = self.abs_mean_displacement_2frames(i, i+1)
            dx.append(_rdict['dx'])
            dy.append(_rdict['dy'])
            dr.append(_rdict['dr'])


        rdict['dx'] = np.mean(np.abs(np.asarray(dx)))
        rdict['dy'] = np.mean(np.abs(np.asarray(dy)))
        rdict['dr'] = np.mean(np.abs(np.asarray(dr)))

        return rdict

    def mean_velocity_allframes(self, freq = 1):
        """
        """

        if freq != 1:
            raise NotImplementedError

        rdict = self.mean_displacement_allframes(freq = freq)

        rdict['dx'] *= (self.m_metadata.FPS * self.m_metadata.PIXELSIZE)
        rdict['dy'] *= (self.m_metadata.FPS * self.m_metadata.PIXELSIZE)
        rdict['dr'] *= (self.m_metadata.FPS * self.m_metadata.PIXELSIZE)

        return rdict



    def mean_abs_velocity_allframes(self, freq = 1):
        """
        """

        if freq != 1:
            raise NotImplementedError

        rdict = self.mean_abs_displacement_allframes(freq = freq)

        rdict['dx'] *= (self.m_metadata.FPS * self.m_metadata.PIXELSIZE)
        rdict['dy'] *= (self.m_metadata.FPS * self.m_metadata.PIXELSIZE)
        rdict['dr'] *= (self.m_metadata.FPS * self.m_metadata.PIXELSIZE)

        return rdict



