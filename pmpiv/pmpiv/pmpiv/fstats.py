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
import json

import pims             # https://soft-matter.github.io/pims/v0.6.1/
import trackpy as tp    # https://soft-matter.github.io/trackpy/v0.6.1/index.html

import pmpiv as pmpiv
###--------------------------------------------------------------------------------


class Frame_Statistics:
    """
    
    Compute statistics of a frame

    frame : <class 'pandas.core.frame.DataFrame'>

    save : bool 

    """

    def __init__(self, 
                 frame, 
                 m_metadata = None, 
                 m_metadata_file = None,
                 bins = 20):

        self.frame = frame
        self.bins  = bins

        # check if metadata is avail
        try:
            self.m_metadata = pmpiv.helper._check_metadata(m_metadata, m_metadata_file)
        except:
            raise FileNotFoundError('Metadata information not available!')


    def histogram(self, parameter, save = True, bins = None):
        """
        """
        if bins is not None:
            self.bins = bins

        parameter = str(parameter)

        fig, ax = plt.subplots()
        ax.hist(self.frame[parameter], bins = self.bins)
        # Optionally, label the axes.
        ax.set(xlabel = parameter, ylabel='count [ # ]')
        
        if save:
            plt.savefig(f'{self.m_metadata.WORKING_DIR}/frame_hist_{parameter}.pdf')
            plt.savefig(f'{self.m_metadata.WORKING_DIR}/frame_hist_{parameter}.png')
        else:
            plt.show()
        
        plt.close()
        plt.clf()
        

class Sequence_Statistics:
    """
    
    Compute statistics of image sequence

    """

    def __init__(self, frames, 
                       m_metadata = None, 
                       m_metadata_file = None,
                       verbose = True,
                       parallel = True): 
        """
        """

        # check if metadata is avail
        try:
            self.m_metadata = pmpiv.helper._check_metadata(m_metadata, m_metadata_file)
        except:
            raise FileNotFoundError('Metadata information not available!')

        self.frames            = frames
        self.min_feature_size  = self.m_metadata.FEATURE_MIN_SIZE
        self.feature_size      = self.m_metadata.FEATURE_SIZE
        self.features_are_dark = self.m_metadata.FEATURES_ARE_DARK
        self.verbose           = verbose

        self.parallel          = parallel
        self.computed          = False

        self.methods = [
                      'mean', 
                      'max', 
                      'min', 
                      'std', 
                      'percentile3', 
                      'percentile97',
                      'percentile5',
                      'percentile95',
                      'percentile10',
                      'percentile90',
                      'percentile15',
                      'percentile85'
                     ]
        
        self.parameters = [
                        'mass', 'size', 'ecc'
                     ]

    # wrapper 
    def mean(self, a):
        return np.mean(a)

    def max(self, a):
        return np.max(a)

    def min(self, a):
        return np.min(a)

    def std(self, a):
        return np.std(a)

    def percentile3(self, a):
        return np.percentile(a, 3.)

    def percentile97(self, a):
        return np.percentile(a, 97.)

    def percentile5(self, a):
        return np.percentile(a, 5.)

    def percentile95(self, a):
        return np.percentile(a, 95.)

    def percentile10(self, a):
        return np.percentile(a, 10.)

    def percentile90(self, a):
        return np.percentile(a, 90.)

    def percentile15(self, a):
        return np.percentile(a, 15.)

    def percentile85(self, a):
        return np.percentile(a, 85.)

    # end wrappers

    def quiet(self):
        self.verbose = False

    def get(self, force_recompute = False):
        """
        """

        _f = os.path.join(self.m_metadata.WORKING_DIR, 'seq_stats.json')
        
        if not force_recompute:
            if os.path.isfile( _f ):
                with open( _f, 'r' ) as f:
                    if self.verbose:
                        print(f'Read statistics from {_f}')
                    _s = dict(json.load(f))
            else:
                if self.parallel:
                    _s = self.pget()
                else:
                    _s = self.sget()

                with open( _f, 'w' ) as f:
                    if self.verbose:
                        print(f'Writing statistics to {_f}')
                    json.dump(_s, f)

        else:
            # remove the stats file if existing
            if os.path.isfile( _f ):
                os.remove( _f )

            if self.parallel:
                _s = self.pget()
            else:
                _s = self.sget()

            # fresh dump
            with open( _f, 'w' ) as f:
                if self.verbose:
                    print(f'Writing statistics to {_f}')
                json.dump(_s, f)

        return _s



    def sget(self):
        """
        sequentiell computation
        """
        self.n_frames = len(self.frames)

        sq_data = {}
        self.sq_stats = {} 

        # initalize 
        for m in self.methods:
            for p in self.parameters:
                sq_data[f'{m}_{p}'] = []

        for i in range(self.n_frames):
            if self.verbose:
                print(f'Computing frames statistics: frame {i} from {self.n_frames}')
            f = tp.locate(self.frames[i], self.feature_size, 
                          invert = self.features_are_dark, 
                          minmass = self.min_feature_size)
            for m in self.methods:
                for p in self.parameters:
                    m_method = getattr(self, m)
                    sq_data[f'{m}_{p}'].append(m_method(f[p]))

        for m in self.methods:
            for p in self.parameters:
                m_method = getattr(self, m)
                self.sq_stats[f'{m}_{p}'] = m_method(sq_data[f'{m}_{p}'])

        self.computed = True
        return self.sq_stats


    def _get(self, myframes):
        """
        """

        self.sq_data = {}

        # initalize 
        for m in self.methods:
            for p in self.parameters:
                self.sq_data[f'{m}_{p}'] = []
        for frame_i in myframes: 
            if self.verbose:
                print(f'PID {os.getpid()} is computing frame statistics: frame {frame_i} from {self.n_frames}')
            f = tp.locate(self.frames[frame_i], self.feature_size, 
                              invert = self.features_are_dark, 
                              minmass = self.min_feature_size)
            for m in self.methods:
                for p in self.parameters:
                    m_method = getattr(self, m)
                    self.sq_data[f'{m}_{p}'].append(m_method(f[p]))

        return self.sq_data

    
    def pget(self):
        """
        parallel computation 
        """
        self.n_frames = len(self.frames)
        self.sq_stats = {} 

        _keys = []
        for m in self.methods:
            for p in self.parameters:
                _keys.append(f'{m}_{p}')

        framelist = np.arange(self.n_frames)

        ncpus = int(0.5 * multiprocessing.cpu_count())

        framelist = list(np.array_split(framelist, ncpus))
        for i in range(len(framelist)): framelist[i] = list(framelist[i])

        pool = multiprocessing.Pool(ncpus)
        collected = pool.map(self._get, framelist)

        # Merge dicts
        coll_sq_data = {}
        for k in _keys:
            coll_sq_data[k] = np.concatenate(list(dd[k] for dd in collected))

        for m in self.methods:
            for p in self.parameters:
                m_method = getattr(self, m)
                self.sq_stats[f'{m}_{p}'] = m_method(coll_sq_data[f'{m}_{p}'])

        pool.close()
        pool.join()

        self.computed = True
        
        return self.sq_stats







