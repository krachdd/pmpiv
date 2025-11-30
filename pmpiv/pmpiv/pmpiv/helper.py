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
import cv2

import pandas as pd
from PIL import Image

import pims             # https://soft-matter.github.io/pims/v0.6.1/
import trackpy as tp    # https://soft-matter.github.io/trackpy/v0.6.1/index.html


import pmpiv as pmpiv

###--------------------------------------------------------------------------------

def _folder_sanity(m_folder):
    """
    """
    
    if not os.path.isdir(m_folder):
        FileNotFoundError(f'Folder {m_folder} does not exist!')


def _file_sanity(m_infile):
    """
    """
    
    if not os.path.exists(m_infile):
        FileNotFoundError(f'File {m_infile} does not exist!')



def read_tif(m_folder, m_filename):
    """
    """

    _folder_sanity(m_folder)

    infile = os.path.join(m_folder, m_filename)

    _file_sanity(infile)

    im = Image.open(infile)
    array = np.array(im)

    return array


def _check_metadata( m , f ):
    """
    m: metadata dict 
    f: file path to input file 
    """
    if ( m is None) and ( f is None):
        raise ValueError('Metadata of Metadata-File must be not None!')
    elif ( m is None ) and ( f is not None ):
        return pmpiv.metadata.Metadata( f )
    else:
        return m





