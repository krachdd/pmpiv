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
import itertools

import pims             # https://soft-matter.github.io/pims/v0.6.1/
import trackpy as tp    # https://soft-matter.github.io/trackpy/v0.6.1/index.html

import pmpiv as pmpiv

###--------------------------------------------------------------------------------

class Filtering:
    """
    """
    def __init__(self, init_df, m_metadata = None, m_metadata_file = None, m_verbose = True):
        
        # initialize filter with df
        self.init_df = init_df

        try:
            self.m_metadata = pmpiv.helper._check_metadata(m_metadata, m_metadata_file)
        except:
            raise FileNotFoundError('Metadata information not available!')

        self.vebose = m_verbose 


    
    def percentile_filter(self, percentile, parameters, m_fstats):
        
        """
        """

        # Sanity

        if percentile not in [3, 5, 10, 15]:
            raise ValueError(f'Perctile must be ONE value from [3, 5, 10, 15].')

        for p in parameters:
            if p not in ['mass', 'size', 'ecc']:
                raise ValueError('Parameters must only be [mass, size, ecc].')

        if not m_fstats:
            raise ValueError('Statistics dict must not be empty!')

        # Get percentile keys
        _k0 = f'percentile{int(percentile)}_'
        _k1 = f'percentile{int(100 - int(percentile))}_'

        # create deep copy of init df
        filtered_df = self.init_df.copy(deep = True)

        for i in parameters:
            k0 = f'{_k0}{i}'
            k1 = f'{_k1}{i}'

            threshold0 = m_fstats[k0]
            threshold1 = m_fstats[k1]

            filtered_df = filtered_df[((filtered_df[i] > threshold0) & 
                                       (filtered_df[i] < threshold1) )]

            print(f'Filtering done: {threshold0} < {i} < {threshold1}')

        return filtered_df


class Annotation_Filtering:
    """
    """

    def __init__(self,
                 init_df, 
                 m_metadata = None, m_metadata_file = None,
                 fuzziness = 0.5, 
                 verbose = True,
                 save = True):
        
        # check if metadata is avail
        try:
            self.m_metadata = pmpiv.helper._check_metadata(m_metadata, m_metadata_file)
        except:
            raise FileNotFoundError('Metadata information not available!')

        if ('x' not in init_df.columns) or ('y' not in init_df.columns):
            raise ValueError('Index missing in initial DF!')

        self.init_df     = init_df
        self.fuzziness   = fuzziness
        self.verbose     = verbose
        self.save        = save


    def _distributed_thresholding(self, mydelpos):
        """
        """
        del_list = []

        counter = 0
        for i in mydelpos:
            if counter%10 == 0 and self.verbose:
                print(f'PID {os.getpid()}: Filtering: {counter} from {len(mydelpos)}')
            _dxpos        = self._dfpos[i, 0]
            _dypos_ranges = self._dfpos[i, 1]
            # print(f'PID {os.getpid()}: Filtering: _dxpos {_dxpos} ranges: {_dypos_ranges}, {_dypos_ranges[0]}, type1 {type(_dypos_ranges)}, type2 {len(_dypos_ranges)}')
            # Check if list of tuples is empty
            if not _dypos_ranges:
                print(f'PID {os.getpid()}: Position {_dxpos}: List of Pixels to include is empty!')
            else:
                # loop over all tuples in list of ranges
                for tpl in _dypos_ranges:
                    _dypos_min = tpl[0]
                    _dypos_max = tpl[1]
                    for j in range(self._all_xpos.shape[0]):
                        _xpos  = self._all_xpos[j]
                        _ypos  = self._all_ypos[j]

                        if ((_dxpos - self.fuzziness < _xpos) & (_xpos < _dxpos + self.fuzziness) & 
                            (_dypos_min - self.fuzziness < _ypos) & (_ypos < _dypos_max + self.fuzziness)):
                            del_list.append(int(j))
            counter += 1

        print(f'PID {os.getpid()}: Filtering: Done!')
        return del_list



    def _df_threshold_handler(self, m_type = None, filter_df = None):

        """

        Private function handling the parallel removal or extraction of features 
        throughout all the frames of a image sequence (in terms of a dataframe).
        It distributes all positions to be removed on the number of available cores. 
        Each core produces a list of features to be removed/extracted from original 
        dataframe.

        This function uses all threads.


        """

        # Sanity check

        if m_type is None:
            raise ValueError('m_type must not be None!')
        if m_type not in ['remove', 'extract']:
            raise ValueError('m_type must be in [remove, extract]!')

        # Initial dataframe 
        f_df = self.init_df.copy(deep = True)
        f_df.index.names = ['idx']
        f_df.reset_index(inplace = True)

        # Save as array to make handling easier
        self._all_xpos = np.asarray(f_df['x'])
        self._all_ypos = np.asarray(f_df['y'])

        self._dfpos = np.asarray(filter_df[['x','y']])


        framelist = np.arange(self._dfpos.shape[0])

        ncpus = int(multiprocessing.cpu_count())

        framelist = list(np.array_split(framelist, ncpus))
        for i in range(len(framelist)): framelist[i] = list(framelist[i])

        pool = multiprocessing.Pool(ncpus)
        collected = pool.map(self._distributed_thresholding, framelist)
        pool.close()
        pool.join()

        all_collected = list(itertools.chain.from_iterable(collected))
        all_collected = list(np.unique(np.asarray(all_collected, dtype = int)))


        if m_type == 'extract':
            s_all_collected = set(all_collected)
            s_index         = set(f_df.index)
            all_inverse     = list(s_index.difference(s_all_collected))
            f_df.drop(f_df.index[all_inverse], inplace=True)
        elif m_type == 'remove':
            f_df.drop(f_df.index[all_collected], inplace=True)
        
        # f_df = f_df.drop(all_collected)

        print(f'Filtering removed {len(self.init_df.index) - len(f_df.index)} from {len(self.init_df.index)} tracks.')

        # put frame number as index again
        # drop column 'idx'
        f_df.index = f_df['frame']
        f_df.drop(columns = ['idx'], inplace = True)

        return f_df


    def _check_filter_df(self, filter_df):
        """
        """

        if ('x' not in filter_df.columns) or ('y' not in filter_df.columns):
            raise ValueError('Index missing extraction DF!')

    #---------------------------------------------------------------------------------------------
    # Wrapper function for removal of particles/annotatiions
    #---------------------------------------------------------------------------------------------

    def remove(self, filter_df, json_type = 'COCO_range'):
        """
        """

        self._check_filter_df(filter_df = filter_df)

        if json_type not in ['COCO_range', 'COCO']:
            raise ValueError('json_type must be in [COCO_range, COCO]')
        elif json_type == 'COCO':
            print('json_type is COCO. Be aware that this is super slow!')

        if json_type == 'COCO':
            my_df = self._premove()
        elif json_type == 'COCO_range':
            my_df = self._df_threshold_handler(m_type = 'remove', filter_df = filter_df)

        return my_df

    #---------------------------------------------------------------------------------------------
    # Wrapper function for extraction of particles/annotations
    #---------------------------------------------------------------------------------------------

    def extract(self, filter_df, json_type = 'COCO_range'):

        self._check_filter_df(filter_df = filter_df)

        if json_type not in ['COCO_range']:
            raise ValueError('json_type must be in [COCO_range]')

        if json_type == 'COCO_range':
            my_df = self._df_threshold_handler(m_type = 'extract', filter_df = filter_df)

        return my_df

    #---------------------------------------------------------------------------------------------


def complete_removal(init_df, m_metadata = None, m_metadata_file = None, verbose = True, fuzziness = 0.5, save = True):
    """
    """

    try:
        m_metadata = pmpiv.helper._check_metadata(m_metadata, m_metadata_file)
    except:
        raise FileNotFoundError('Metadata information not available!')

    my_df = init_df.copy( deep = True )
    for s in range(len(m_metadata.REMOVAL)):
        if verbose:
            print(f'\n### Remove all annotations given in {m_metadata.JSON_PATH}/{m_metadata.REMOVAL[s]}')
        selection_ann_handler = pmpiv.annotations.Annotation_Handler(f'{m_metadata.JSON_PATH}/{m_metadata.REMOVAL[s]}', m_metadata = m_metadata)
        filter_selection_df   = selection_ann_handler.annotations2DF()
        removal_ann_filter    = pmpiv.filtering.Annotation_Filtering(my_df.copy(deep = True), 
                                                                     m_metadata = m_metadata, 
                                                                     m_metadata_file = m_metadata_file,
                                                                     fuzziness = fuzziness,
                                                                     verbose = verbose,
                                                                     save = save )
        my_df                 = removal_ann_filter.remove( filter_df = filter_selection_df )

    if save:
        fn = 'df_complete_removal.csv'
        _f = os.path.join(m_metadata.WORKING_DIR, fn)
        
        # Remove file if existing
        if os.path.isfile( _f ):
            os.remove( _f )

        pmpiv.df_io.write2csv( my_df.copy( deep = True ), m_metadata.WORKING_DIR, fn )

    return my_df





# class Static_Filtering:
#     """
#     """

#     def __init__(self, init_df,
#                  m_metadata = None, m_metadata_file = None,
#                  verbose = True, parallel = True):
        
#         # check if metadata is avail
#         try:
#             self.m_metadata = pmpiv.helper._check_metadata(m_metadata, m_metadata_file)
#         except:
#             raise FileNotFoundError('Metadata information not available!')

#         if ('x' not in init_df.columns) or ('y' not in init_df.columns):
#             raise ValueError('Index missing in initial DF!')

#         self.init_df  = init_df
#         self.verbose  = verbose
#         self.parallel = parallel


#     def _distributed_static_check(self, my_trajlist):
#         """
#         """
#         del_list = []
#         counter = 0

#         for i in my_trajlist:
#             if counter == 0 and self.verbose:
#                 print(f'PID {os.getpid()}: Start Static filtering')
#             if counter%1000 == 0 and counter > 0 and self.verbose:
#                 print(f'PID {os.getpid()}: Filtering: {counter} from {len(my_trajlist)}')
#             m_traj = self._tdata[self._tdata[:, 2] == i]
#             index  = list(m_traj[:, 3])
#             # print(m_traj)
#             # print(index)

#             pos_threshold = self.m_metadata.STATIC_DEV_PARAMETER
#             stat_threshold = self.m_metadata.CHECK_STATIC
#             # if the standart deviation of all positions (both components) of a trajectory is smaller 
#             # than threshold we assume it is stagnant and we remove it
            
#             # if np.std(m_traj[:, 0]) < pos_threshold and np.std(m_traj[:, 1]) < pos_threshold:
#             if np.ptp(m_traj[:, 0]) < stat_threshold and np.ptp(m_traj[:, 1]) < stat_threshold:
#                 del_list.extend(index)

#             counter += 1

#         return del_list



#     def _static_handler(self, m_type):
#         """
#         Remove static annotations by a threshold for minimum required movement.
        
#         """
#         if m_type is None:
#             raise ValueError('m_type must not be None!')
#         if m_type not in ['remove', 'extract']:
#             raise ValueError('m_type must be in [remove, extract]!')

#         # Initial dataframe 
#         f_df = self.init_df.copy(deep = True)
#         f_df.index.names = ['idx']
#         f_df.reset_index(inplace = True)
#         f_df['idx'] = f_df.index


#         self._traj  = f_df['particle'].to_numpy()
#         self._unique_traj = np.unique(self._traj)
#         init_number_traj = self._unique_traj.shape[0]

#         self._tdata = f_df[['x', 'y', 'particle', 'idx']].to_numpy()

#         ncpus = int(multiprocessing.cpu_count())

#         trajlist = list(self._unique_traj)

#         trajlist = list(np.array_split(trajlist, ncpus))
#         for i in range(len(trajlist)): trajlist[i] = list(trajlist[i])

#         pool = multiprocessing.Pool(ncpus)
#         collected = pool.map(self._distributed_static_check, trajlist)
#         pool.close()
#         pool.join()

#         all_collected = list(itertools.chain.from_iterable(collected))
#         all_collected = list(np.unique(np.asarray(all_collected, dtype = int)))

#         if m_type == 'extract':
#             s_all_collected = set(all_collected)
#             s_index         = set(f_df.index)
#             all_inverse     = list(s_index.difference(s_all_collected))
#             f_df.drop(f_df.index[all_inverse], inplace=True)
#         elif m_type == 'remove':
#             f_df.drop(f_df.index[all_collected], inplace=True)



#         # put frame number as index again
#         # drop column 'idx'
#         f_df.index = f_df['frame']
#         f_df.drop(columns = ['idx'], inplace = True)
#         final_traj_number = np.unique(f_df['particle']).shape[0]
#         print(f'Number of removed trajectories: {init_number_traj-final_traj_number} from {init_number_traj}.')

#         return f_df



#     # wrapper function for the removal of static annotations
#     def remove(self):
        
#         """
#         """

#         # loop based
#         if self.parallel:
#             return self._static_handler(m_type = 'remove')
#         else:
#             raise NotImplementedError
#             # return self._sc_static_handler()



def filter_static(init_df, 
                  m_metadata = None, 
                  m_metadata_file = None,
                  verbose = True):
    """
    """


    # check if metadata is avail
    try:
        m_metadata = pmpiv.helper._check_metadata(m_metadata, m_metadata_file)
    except:
        raise FileNotFoundError('Metadata information not available!')

    try:
        init_df['frame']
        init_df['particle']
        init_df['x']
        init_df['y']
    except KeyError:
        raise ValueError("Init DF must contain columns 'frame', 'particle', 'x', 'y'.")

    grouped = init_df.reset_index(drop=True).groupby('particle', dropna=True)
    
    # Check if the change in position over the whole sequence is bigger than 
    # CHECK_STATIC (given in pixels, not meter)
    filtered = grouped.filter(lambda f: (f.x.max() - f.x.min()) >= m_metadata.CHECK_STATIC and 
                                        (f.y.max() - f.y.min()) >= m_metadata.CHECK_STATIC     )
    
    df_filtered = filtered.set_index('frame', drop=False)

    if verbose:
        particles_filtered = np.unique(df_filtered['particle'].to_numpy()).shape[0]
        particles_init     = np.unique(init_df['particle'].to_numpy()).shape[0]
        print(f'\n### Removed static trajectories:\nInitial number: {particles_init}\nRemoved: {particles_init-particles_filtered}\nResulting: {particles_filtered}')

    return df_filtered



def filter_stubs(init_df, 
                 m_metadata = None, 
                 m_metadata_file = None,
                 verbose = True):
    
    """
    """
    # check if metadata is avail
    try:
        m_metadata = pmpiv.helper._check_metadata(m_metadata, m_metadata_file)
    except:
        raise FileNotFoundError('Metadata information not available!')

    try:
        init_df['frame']
        init_df['particle']
        init_df['x']
        init_df['y']
    except KeyError:
        raise ValueError("Init DF must contain columns 'frame', 'particle', 'x', 'y'.")

    df_filtered = tp.filter_stubs(init_df, m_metadata.DURATION)

    if verbose:
        particles_filtered = np.unique(df_filtered['particle'].to_numpy()).shape[0]
        particles_init     = np.unique(init_df['particle'].to_numpy()).shape[0]
        print(f'\n### Removed sporious trajectories:\nInitial number: {particles_init}\nRemoved: {particles_init-particles_filtered}\nResulting: {particles_filtered}')

    return df_filtered



