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
import PIL
import multiprocessing
import warnings

import pims             # https://soft-matter.github.io/pims/v0.6.1/
import trackpy as tp    # https://soft-matter.github.io/trackpy/v0.6.1/index.html

import pmpiv as pmpiv

###--------------------------------------------------------------------------------


class Image_Sequence:
    """
    
    """

    def __init__(self, m_metadata = None, m_metadata_file = None):
        
        
        # check if metadata is avail
        try:
            self.m_metadata = pmpiv.helper._check_metadata(m_metadata, m_metadata_file)
        except:
            raise FileNotFoundError('Metadata information not available!')
        
        self.folder  = self.m_metadata.IN_PATH
        self.ftype   = self.m_metadata.IN_FORMAT
        self._is_read = False

        self.verbose = True
        self._plot_parallel = False

    def quiet(self):
        self.verbose = False

    def plot_parallel(self):
        self._plot_parallel = True

    def __info(self):
        """
        Shall be private.
        """
        print(f'Number of frames in folder: {len(self.image_sequence)}')
        print(f'Size of frames            : {self.image_sequence[0].shape}')
        print(f'Metadata of frames:       : {self.image_sequence[0].metadata}')

    
    def _read(self):
        """
        """
        # Read image sequence
        self.image_sequence = pims.ImageSequence(f'{self.folder}/*{self.ftype}')
        
        self._is_read = True
        self.__info()

    def read(self):
        """
        """
        # Read image sequence
        if not self._is_read:
            self._read()

        return self.image_sequence

    # get parts of the image sequence to use
    def subsection_framerate(self):
        """
        """
        
        if not self._is_read:
            self._read()

        self.image_sequence = self.image_sequence[::self.m_metadata.RATE]
        return self.image_sequence

    def subsection_range(self):
        """
        """

        if not self._is_read:
            self._read()

        # if END_FRAME == 0 return entire image sequence
        if self.m_metadata.END_FRAME != 0:
            self.image_sequence = self.image_sequence[self.m_metadata.START_FRAME:self.m_metadata.END_FRAME]
        

        return self.image_sequence



    def _output_annotated_pngs(self, df_annotations, outfolder, color = 'red', dpi = 100):
        """
        """

        if self._is_read is False:
            self._read()

        # compute figsize 
        figsize = list(np.array((self.image_sequence[0].shape)[::-1]) * 1.7/plt.rcParams['figure.dpi'])  # pixel in inches

        # get all frames from df_annotations
        out_frames = list(np.unique(np.asarray(df_annotations['frame'])))

        os.system(f'mkdir -p {outfolder}')
        if self.verbose: print(f'Create folder if not existing: {outfolder}')

        for myf in out_frames:
            # section of df
            _df = df_annotations[df_annotations['frame'] == myf]

            # create fig
            fig, ax = plt.subplots(figsize=figsize)
            ax = tp.annotate(_df, self.image_sequence[myf], ax = ax, color = color)
            plt.tight_layout()
            plt.savefig(f'{outfolder}/frame_{myf:06d}.png', dpi = dpi)
            plt.close()
            plt.clf()


    def _output_annotated_tiffs(self, df_annotations, outfolder, color = 'red', dpi = 100):
        """
        """

        if self._is_read is False:
            self._read()

        # compute figsize 
        figsize = list(np.array((self.image_sequence[0].shape)[::-1]) * 1.7/plt.rcParams['figure.dpi'])  # pixel in inches

        # get all frames from df_annotations
        out_frames = list(np.unique(np.asarray(df_annotations['frame'])))

        os.system(f'mkdir -p {outfolder}')
        if self.verbose: print(f'Create folder if not existing: {outfolder}')

        for myf in out_frames:
            # section of df
            _df = df_annotations[df_annotations['frame'] == myf]

            # create fig
            fig, ax = plt.subplots(figsize=figsize)
            ax = tp.annotate(_df, self.image_sequence[myf], ax = ax, color = color)
            plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
            plt.tick_params(axis = 'y', which = 'both', bottom = False, top = False, labelbottom = False)
            plt.tight_layout()
            plt.savefig(f'{outfolder}/frame_{myf:06d}.tif', dpi = dpi)
            plt.close()
            plt.clf()


    def _plot_output_annotated_tiffs(self, out_frames_list):
        """
        """

        # compute figsize 
        figsize = list(np.array((self.image_sequence[0].shape)[::-1]) * 1.7/plt.rcParams['figure.dpi'])  # pixel in inches
        
        # print(figsize)

        for myf in out_frames_list:
            if self.verbose:
                print(f'PID {os.getpid()}: Parallel save tiff No. {myf:06d} from {len(self.image_sequence):06d}.')
            # section of df
            _df = self._df_annotations[self._df_annotations['frame'] == myf]

            # create fig
            fig, ax = plt.subplots(figsize=figsize)
            ax = tp.annotate(_df, self.image_sequence[myf], ax = ax, color = self._color)
            plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
            plt.tick_params(axis = 'y', which = 'both', bottom = False, top = False, labelbottom = False)
            plt.tight_layout()
            plt.savefig(f'{self._outfolder}/frame_{myf:06d}.tif', dpi = self._dpi)
            plt.close()
            plt.clf()



    def _parallel_output_annotated_tiffs(self, df_annotations, outfolder, color = 'red', dpi = 100):
        """
        """

        if self._is_read is False:
            self._read()


        # get all frames from df_annotations
        out_frames = list(np.unique(np.asarray(df_annotations['frame'])))

        os.system(f'mkdir -p {outfolder}')
        if self.verbose: print(f'Create folder if not existing: {outfolder}')

        ncpus = int(0.5 * multiprocessing.cpu_count())

        out_frames_lists = list(np.array_split(out_frames, ncpus))
        for i in range(len(out_frames_lists)): out_frames_lists[i] = list(out_frames_lists[i])

        # make all important parmeters class objects to make stuff easy
        self._df_annotations = df_annotations
        self._outfolder = outfolder
        self._color = color
        self._dpi = dpi

        p = multiprocessing.Pool(ncpus)
        p.map(self._plot_output_annotated_tiffs, out_frames_lists)
        p.close()
        p.join()




    def _output_compare_annotated_tiffs(self, df_annotations, df_annotations_mod, outfolder, colors = ['red', 'blue'], dpi = 100):
        """
        """

        if self._is_read is False:
            self._read()

        # compute figsize 
        figsize = list(np.array((self.image_sequence[0].shape)[::-1]) * 1.7/plt.rcParams['figure.dpi'])  # pixel in inches

        figsize[0] *= 2 

        # get all frames from df_annotations
        out_frames = list(np.unique(np.asarray(df_annotations['frame'])))
        out_frames_mod = list(np.unique(np.asarray(df_annotations_mod['frame'])))

        if not (out_frames == out_frames_mod):
            raise ValueError('Frames must be same!')

        if self.verbose: print(f'Create folder if not existing: {outfolder}')
        os.system(f'mkdir -p {outfolder}')

        for myf in out_frames:
            if self.verbose:
                print(f'Save tiffs {myf:06d} from {len(out_frames):06d}.')
            # section of df
            _df = df_annotations[df_annotations['frame'] == myf]
            _df_mod = df_annotations_mod[df_annotations_mod['frame'] == myf]

            # create fig
            fig, ax = plt.subplots(1, 2,figsize=figsize)
            ax[0] = tp.annotate(_df, self.image_sequence[myf], ax = ax[0], color = colors[0])
            ax[1] = tp.annotate(_df_mod, self.image_sequence[myf], ax = ax[1], color = colors[1])
            
            ax[0].tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
            ax[0].tick_params(axis = 'y', which = 'both', bottom = False, top = False, labelbottom = False)
            ax[1].tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
            ax[1].tick_params(axis = 'y', which = 'both', bottom = False, top = False, labelbottom = False)
            
            plt.tight_layout()
            
            plt.savefig(f'{outfolder}/frame_{myf:06d}.tif', dpi = dpi)
            plt.close()
            plt.clf()


    def _plot_output_compare_annotated_tiffs(self, out_frames_list):
        """
        """
        # compute figsize 
        figsize = list(np.array((self.image_sequence[0].shape)[::-1]) * 1.7/plt.rcParams['figure.dpi'])  # pixel in inches

        figsize[0] *= 2 

        for myf in out_frames_list:
            if self.verbose:
                print(f'PID {os.getpid()}: Parallel save tiff No. {myf:06d} from {len(self.image_sequence):06d}.')
            # section of df
            _df = self._df_annotations[self._df_annotations['frame'] == myf]
            _df_mod = self._df_annotations_mod[self._df_annotations_mod['frame'] == myf]

            # create fig
            fig, ax = plt.subplots(1, 2,figsize=figsize)
            ax[0] = tp.annotate(_df, self.image_sequence[myf], ax = ax[0], color = self._colors[0])
            ax[1] = tp.annotate(_df_mod, self.image_sequence[myf], ax = ax[1], color = self._colors[1])
            
            ax[0].tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
            ax[0].tick_params(axis = 'y', which = 'both', bottom = False, top = False, labelbottom = False)
            ax[1].tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
            ax[1].tick_params(axis = 'y', which = 'both', bottom = False, top = False, labelbottom = False)
            
            plt.tight_layout()
            
            plt.savefig(f'{self._outfolder}/frame_{myf:06d}.tif', dpi = self._dpi)
            plt.close()
            plt.clf()




    def _parallel_output_compare_annotated_tiffs(self, df_annotations, df_annotations_mod, outfolder, colors = ['red', 'blue'], dpi = 100):
        """
        """

        if self._is_read is False:
            self._read()

        # get all frames from df_annotations
        out_frames = list(np.unique(np.asarray(df_annotations['frame'])))
        out_frames_mod = list(np.unique(np.asarray(df_annotations_mod['frame'])))

        if not (out_frames == out_frames_mod):
            raise ValueError('Frames must be same!')

        if self.verbose: print(f'Create folder if not existing: {outfolder}')
        os.system(f'mkdir -p {outfolder}')

        ncpus = int(0.5 * multiprocessing.cpu_count())

        out_frames_lists = list(np.array_split(out_frames, ncpus))
        for i in range(len(out_frames_lists)): out_frames_lists[i] = list(out_frames_lists[i])

        # make all important parmeters class objects to make stuff easy
        self._df_annotations = df_annotations
        self._df_annotations_mod = df_annotations_mod
        self._outfolder = outfolder
        self._colors = colors
        self._dpi = dpi

        p = multiprocessing.Pool(ncpus)
        p.map(self._plot_output_compare_annotated_tiffs, out_frames_lists)
        p.close()
        p.join()






    def plot_annotated_pngs(self, df_annotations, outfolder, color = 'red', dpi = 100):
        """
        """
        if self._plot_parallel:
            warnings.warn('Parallel png dump not implemented, using sequential dump.', UserWarning)
            self._output_annotated_pngs(df_annotations, outfolder, color = color, dpi = 100)
        else:
            self._output_annotated_pngs(df_annotations, outfolder, color = color, dpi = 100)



    def plot_annotated_tiffs(self, df_annotations, outfolder, color = 'red', dpi = 100):
        """
        """

        if self._plot_parallel:
            self._parallel_output_annotated_tiffs(df_annotations, outfolder, 
                                                  color = color, dpi = dpi)
        else:
            self._output_annotated_tiffs(df_annotations, outfolder, 
                                                  color = color, dpi = dpi)

    def plot_compare_annotated_tiffs(self, df_annotations, df_annotations_mod, outfolder, colors = ['red', 'blue'], dpi = 100):
        """
        """
        
        if self._plot_parallel:
            self._parallel_output_compare_annotated_tiffs(df_annotations, df_annotations_mod, 
                                                          outfolder, colors = colors, dpi = dpi)
        else:
            self._output_compare_annotated_tiffs(df_annotations, df_annotations_mod, 
                                                          outfolder, colors = colors, dpi = dpi)






