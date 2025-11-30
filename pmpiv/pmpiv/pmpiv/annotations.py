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
import json
from pycocotools.coco import COCO
import itertools
from ast import literal_eval

import pims             # https://soft-matter.github.io/pims/v0.6.1/
import trackpy as tp    # https://soft-matter.github.io/trackpy/v0.6.1/index.html

import pmpiv as pmpiv

###--------------------------------------------------------------------------------

class Annotation_Handler:
    """
    """
    def __init__(self, json_filename, m_metadata = None, m_metadata_file = None, json_type = 'COCO_range', verbose = True):

        # initialize json file with coco annotations
        self.json_filename = json_filename
        self.json_type = json_type
        self.verbose   = verbose

        # check if metadata is avail
        try:
            self.m_metadata = pmpiv.helper._check_metadata(m_metadata, m_metadata_file)
        except:
            raise FileNotFoundError('Metadata information not available!')

        # Do not use blank COCO, will be obsolete soon
        if self.json_type not in ['COCO_range', 'COCO']:
            raise ValueError('self.json_type must be in [COCO_range, COCO]')
        elif self.json_type == 'COCO':
            print('self.json_type is COCO. Be aware that this is super slow!')

        self._json_file_exists()
        self._plot = False


    def _json_file_exists(self):
        if not os.path.isfile(self.json_filename):
            raise FileNotFoundError(f'Can not find file: {self.json_filename}')

    def show_plots(self):
        self._plot = True

    def _to_ranges(self, iterable):
        iterable = sorted(set(iterable))
        for key, group in itertools.groupby(enumerate(iterable), lambda t: t[1] - t[0]):
            group = list(group)
            yield group[0][1], group[-1][1]


    def annotations2DF(self, writeto = None):
        """
        """
        # if self._plot == True:
        #     if mytif == None:
        #         raise ValueError('If ploting is enablet')

        # load annotations
        coco = COCO(self.json_filename)

        # load annotation polygons
        img = coco.imgs[1]
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)

        mask = coco.annToMask(anns[0])

        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])

        mask[mask != 0] = 1

        xlist = []
        ylist = []

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] == 1:
                    ylist.append(i)
                    xlist.append(j)

        pd_data = np.column_stack((xlist, ylist))

        _dead_pixels = pd.DataFrame(data = pd_data, columns = ['x', 'y'])
        
        if self.json_type == 'COCO_range':
            _dead_pixels = _dead_pixels.groupby('x')['y'].apply(list)

            myindex = _dead_pixels.index
            myvalues = _dead_pixels.values

            dead_pixels = pd.DataFrame(columns = ['x', 'y'])

            for i in range(len(myindex)):
                dead_pixels.loc[i] = [myindex[i] , list(self._to_ranges(myvalues[i]))]
        elif self.json_type == 'COCO':
            dead_pixels = _dead_pixels

        if writeto is not None:
            dpc = dead_pixels.copy(deep = True)
            dpc.to_csv(writeto, index = False, sep = ';')

        if self._plot == True:
            plt.imshow(mask)
            plt.show()
            plt.close()
            plt.clf()

        return dead_pixels

    def area(self):
        """
        returns combined area of all annotations in json file in [m^2].
        """
        m_area = 0
        coco = COCO(self.json_filename)

        if len(coco.anns.keys()) == 0:
            raise ValueError('JSON File does not hold annotations with area!')
        else:
            for key in coco.anns.keys():
                m_area += coco.anns[key]['area']

        m_area *= (self.m_metadata.PIXELSIZE * self.m_metadata.PIXELSIZE)

        self.area = m_area

        return self.area

    
    def volume(self):
        """
        returns combined volume of all annotations in json file in [m^3].
        """
        self.volume = self.area() * self.md.HEIGHT
        return self.volume


class Annotation_Reader:
    """
    """
    def __init__(self, csv_filename, m_metadata = None, m_metadata_file = None, json_type = 'COCO_range', verbose = True):

        # initialize filter with df
        self.csv_filename = csv_filename
        self.json_type    = json_type
        self.verbose      = verbose

        # check if metadata is avail
        try:
            self.m_metadata = pmpiv.helper._check_metadata(m_metadata, m_metadata_file)
        except:
            raise FileNotFoundError('Metadata information not available!')

        if self.json_type not in ['COCO_range', 'COCO']:
            raise ValueError('self.json_type must be in [COCO_range, COCO]')
        elif self.json_type == 'COCO':
            print('self.json_type is COCO. Be aware that this is super slow!')

        self._csv_file_exists()

    def _csv_file_exists(self):
        if not os.path.isfile(self.csv_filename):
            raise FileNotFoundError(f'Can not find file: {self.csv_filename}')

    def read(self):
        """
        """
        self._csv_file_exists()

        if self.json_type == 'COCO':
            return pd.read_csv(self.csv_filename, sep = ';')
        elif self.json_type == 'COCO_range':
            return pd.read_csv(self.csv_filename, sep = ';', converters={'y': literal_eval})