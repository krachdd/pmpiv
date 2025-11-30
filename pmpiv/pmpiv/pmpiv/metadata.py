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


import pmpiv as pmpiv

###--------------------------------------------------------------------------------



class Metadata:
    """
    
    Get Metadata from text file

    """

    def __init__(self, infile):
        self.infile = infile

        self._is_avail()
        _md = self.read()

        self.IN_PATH              = self._md['IN_PATH']
        self.WORKING_DIR          = self._md['WORKING_DIR']
        self.IN_FORMAT            = self._md['IN_FORMAT']
        self.PIXELSIZE            = self._md['PIXELSIZE']
        self.HEIGHT               = self._md['HEIGHT']
        self.START_FRAME          = self._md['START_FRAME']
        self.END_FRAME            = self._md['END_FRAME']
        self.RATE                 = self._md['RATE']
        self.FEATURE_SIZE         = self._md['FEATURE_SIZE']
        self.FEATURE_MIN_SIZE     = self._md['FEATURE_MIN_SIZE']
        self.FEATURES_ARE_DARK    = self._md['FEATURES_ARE_DARK']
        self.FPS                  = self._md['FPS']
        self.MAX_PARTICLE_SPEED   = self._md['MAX_PARTICLE_SPEED']
        self.MEMORY               = self._md['MEMORY']
        self.REMOVE_STATIC        = self._md['REMOVE_STATIC']
        self.CHECK_STATIC         = self._md['CHECK_STATIC']
        self.STATIC_DEV_PARAMETER = self._md['STATIC_DEV_PARAMETER']
        self.DURATION             = self._md['DURATION']
        self.JSON_PATH            = self._md['JSON_PATH']
        self.REMOVAL              = self._md['REMOVAL']
        self.EXTRACTION           = self._md['EXTRACTION']

    def _is_avail(self):
        """
        """
        if os.path.isfile(self.infile):
            pass
        else:
            raise FileNotFoundError('Infile not found!')

    def read(self):
        """
        """

        # Define metaparameter
        self._md = {}
        self._keys = [
            'IN_PATH',
            'WORKING_DIR',
            'IN_FORMAT',
            'PIXELSIZE',
            'HEIGHT',
            'START_FRAME',
            'END_FRAME',
            'RATE',
            'FEATURE_SIZE', 
            'FEATURE_MIN_SIZE', 
            'FEATURES_ARE_DARK',
            'FPS', 
            'MAX_PARTICLE_SPEED',
            'MEMORY',
            'DURATION',
            'REMOVE_STATIC',
            'CHECK_STATIC',
            'STATIC_DEV_PARAMETER',
            'JSON_PATH', 
            'REMOVAL',
            'EXTRACTION' 
                    ]

        self._str_keys = ['IN_PATH', 'WORKING_DIR', 'IN_FORMAT', 'FEATURES_ARE_DARK', 'JSON_PATH', 'REMOVE_STATIC']
        self._int_keys = ['FEATURE_SIZE', 'FEATURE_MIN_SIZE', 'RATE', 'MAX_PARTICLE_SPEED', 'MEMORY', 'DURATION', 'CHECK_STATIC', 'START_FRAME', 'END_FRAME']
        self._flt_keys = ['FPS', 'PIXELSIZE', 'HEIGHT', 'STATIC_DEV_PARAMETER']

        with open(self.infile, 'r') as fh:
            for line in fh:
                for k in range(len(self._keys)):
                    if line.startswith(self._keys[k]):
                        if   self._keys[k] in self._str_keys:
                            self._md[str(self._keys[k])] = str(line.replace(f'{self._keys[k]} ', '').replace('\n', ''))
                        elif self._keys[k] in self._int_keys:
                            self._md[str(self._keys[k])] = int(line.replace(f'{self._keys[k]} ', ''))
                        elif self._keys[k] in self._flt_keys:
                            self._md[str(self._keys[k])] = float(line.replace(f'{self._keys[k]} ', ''))
                        elif self._keys[k] in ['REMOVAL', 'EXTRACTION']:
                            self._md[str(self._keys[k])] = list((line.replace(f'{self._keys[k]} ', '').replace('\n', '')).split(','))
                            if self._md[str(self._keys[k])] == list(['']):
                                self._md[str(self._keys[k])] = list([])
                            for f in range(len(self._md[str(self._keys[k])])):
                                self._md[str(self._keys[k])][f] = self._md[str(self._keys[k])][f].replace(' ', '')


        print(f'Metadata: {self._md}')

        self._sanity_check()

        return self._md

    def is_odd(self, a):
        """
        """
        return a % 2 != 0

    def _sanity_check(self):
        """
        """
        # Missing the odd 

        if not self._md['IN_PATH']:
            raise ValueError('Path IN_PATH is empty!')

        if not self._md['WORKING_DIR']:
            raise ValueError('Path WORKING_DIR is empty!')

        # if not self._md['JSON_PATH']:
        #     raise ValueError('Path JSON_PATH is empty!')

        if self._md['IN_FORMAT'] not in ['tif', 'tiff', 'TIF', 'TIFF']: 
            raise ValueError('Format should be [tif, tiff, TIF, TIFF]!')

        for i in self._int_keys:
            if not (isinstance(self._md[i], int)):
                raise ValueError(f'{i} must be int type!')

        for i in self._flt_keys:
            if not (isinstance(self._md[i], float)):
                raise ValueError(f'{i} must be float type!')

        for i in ['REMOVAL', 'EXTRACTION']:
            if self._md[i]:
                for j in self._md[i]:
                    _m_path = self._md['JSON_PATH']
                    if not os.path.isfile(f'{_m_path}/{j}'):
                        raise FileNotFoundError(f'File {_m_path}/{j} does not exist!')









