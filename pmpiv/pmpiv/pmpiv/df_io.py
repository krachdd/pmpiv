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



def read_csv(m_folder, m_filename):
    """
    """

    _folder_sanity(m_folder)
    
    infile = os.path.join(m_folder, m_filename)

    _file_sanity(infile)

    df = pd.read_csv(infile, sep = ';')
    df.set_index('frame', drop = False, append = False, inplace = True)

    return df


def write2csv(df, m_folder, m_filename):
    """
    """

    _folder_sanity(m_folder)

    outfile = os.path.join(m_folder, m_filename)

    df.to_csv(outfile, sep = ';', index = False)

    print(f'DataFrame saved to file {outfile} !')




