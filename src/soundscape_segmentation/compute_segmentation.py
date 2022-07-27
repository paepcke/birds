'''
Created on Jul 13, 2022

@author: paepcke
'''

import argparse
from enum import Enum
import os
import sys

from birdsong.utils.utilities import FileUtils
from data_augmentation.utils import Interval, Utils
from experiment_manager.experiment_manager import Datatype, ExperimentManager
from logging_service.logging_service import LoggingService
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from powerflock.signal_analysis import SignalAnalyzer
from result_analysis.charting import Charter
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------- Enums --------------
class Bound(str, Enum):
    START = 'start'
    STOP  = 'stop'

class Direction(str, Enum):
    AHEAD   = 'ahead'
    BEHIND  = 'behind'

# ---------------------------------- Class SegmentationComputer --------------

SPECTRO_PATH = '/Users/paepcke/EclipseWorkspacesNew/birds/data/am02_20190717_052958_spectro.csv'
class SegmentationComputer:
    '''
    classdocs
    '''
    CUR_DIR = os.path.dirname(__file__)
    PROJ_ROOT  = os.path.abspath(os.path.join(CUR_DIR, '../../'))
    EXP_ROOT = os.path.join(PROJ_ROOT, 'experiments/SoundscapeSegmentation')

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, experiment, sel_tbl_key, peaks_key, spectro_key_or_path):
        '''
        '''

        self.peaks_key = peaks_key
        
        self.log = LoggingService()
        self.exp = experiment

        self.sels = experiment.read(sel_tbl_key, Datatype.tabular)
        if spectro_key_or_path is None:
            self.spectro = None
            self.exp_timestamp = FileUtils.file_timestamp()
            return
            
        self._obtain_spectrogram(spectro_key_or_path)
        
        # Timestamp to be used in all experiment keys:
        # first try to find the timestamp in the spectro fname:
        self.exp_timestamp = Utils.timestamp_from_exp_path(spectro_key_or_path)
        if self.exp_timestamp is None:
            self.exp_timestamp = FileUtils.file_timestamp()

        # Make the string 'floats' into real floats:
        self.spectro.columns = self.spectro.columns.values.astype(float)

        # Get the upper freq bands:: [0.0, 492.1875, 992.1875, 1492.1875, ... 15992.1875]
        num_channels, _num_times = self.spectro.shape
        
        # Get sorted list of new freq band boundaries, like:
        # [0.0, 492.1875, 992.1875, 1492.1875, ...]
        band_upper_bounds = list(reversed(self.spectro.index[np.arange(0,num_channels,
                                                                  self.num_freq_bands)]))

        # Get 2-tuples of (low_band_freq, high_band_freq):
        self.freq_bounds = list(zip(band_upper_bounds, band_upper_bounds[1:]))

    #------------------------------------
    # event_mask_sigs
    #-------------------

    def event_mask_sigs(self, acorr_peaks_key, sig_types, peak_lower_limit=0.9):
    
        if type(sig_types) != list:
            sig_types = [sig_types]

        self.peaks_key = acorr_peaks_key
        acorr_df_raw = self.exp.read(acorr_peaks_key, Datatype.tabular)

        # Pull out just the columns we need:
        self.acorr_peaks = acorr_df_raw[acorr_df_raw.lag == 1][['time','sig_type', 'acorr','freq_low', 'freq_high']]

        #**************
        ax = None # No plot of rects started yet
        #**************

        # Only look at acorr peaks that were
        # computed on one of the requested sig types:
        peaks_df_for_sigs = self.acorr_peaks[self.acorr_peaks.sig_type.isin(sig_types)] \
                                            [['time','acorr','freq_low', 'freq_high']]
        # window size 2336 ~= 250msec, polynomial order 3
        #**************
        # peaks_smooth = peaks_df_for_sigs
        # peaks_smooth   = pd.Series(savgol_filter(peaks_df_for_sigs.acorr, 2337, 3),
        #                            index=peaks_df_for_sigs.index)
        # peaks_df_for_sigs = peaks_df_for_sigs.assign(acorr = peaks_smooth)
        #**************
        
        # Limit to peaks >= peak_lower_limit:
        high_peaks = peaks_df_for_sigs[peaks_df_for_sigs.acorr >= peak_lower_limit]
         
        # Create a df with same dimensions as spectrogram,
        # initialized to all False. The df will be set to
        # True selectively where acorr meets the peak_lower_limit:
        mask_height, mask_width = self.spectro.shape 
        sels_mask = pd.DataFrame(np.array([False] * mask_height * mask_width).reshape((mask_height, mask_width)),
                                 index = self.spectro.index,
                                 columns=self.spectro.columns.astype(float))
        
        # Get row numbers grouped be frequency band. 
        # I.e. row numbers for band (1992.1875 to 2492.1875), 
        # band (2492.1875 to 2992.1875), etc. Only the bands
        # still at play after the above filter by peak_lower_level
        # are included, not all 32 bands:
        
        freq_grps = high_peaks.groupby(by=['freq_low', 'freq_high'])
        spectro_freqs = pd.Series(sorted(self.spectro.index.values), name='spectro_freqs')
        spectro_freqs_min = spectro_freqs.min()
        spectro_freqs_max = spectro_freqs.max()
        
        spectro_times = pd.Series(self.spectro.columns, name='spectro_times', dtype=float)
        
        # Create a spectrogram-shaped mask of True/False, and
        # a list of bounding rectangles that were found. We'll
        # fill bounding_rects with tuples:
        #    (freq_low_left, time_low_left, freq_high_right, time_high_right)
        
        bounding_rects = []

        # freq_grps.groups is a dict mapping freq_low/freq_high pairs
        # to row numbers:
        for fband_low, fband_high in freq_grps.groups.keys():
            
            # For now, don't include the freq band that covers
            # the entire height of the spectro:
            if (fband_low == spectro_freqs_min and fband_high == spectro_freqs_max):
                continue 
            
            self.log.info(f"Analyzing band {fband_low}Hz to {fband_high}Hz")
            
            # Each freq band combines several frequency channels in 
            # the original spectrogram. Get the set of those channels
            # that comprise the fband under consideration in this round
            # of the loop. The modulo len(spectro_freqs) accounts for
            # each frequency band include rows with all spectro times:

            spectro_freqs_to_mask = spectro_freqs[spectro_freqs.between(fband_low, fband_high)]
            
            # Get the series of times at which acorr values were above threshold that
            # were detected in this loop's freq band. Some of these times will
            # be contiguous spectro frame times. Separate events in the freq
            # band will show as 'missing spectro frame times' in the time series
            # (see follow-on comment below):
            spectro_times_to_mask = high_peaks[np.logical_and(high_peaks.freq_low >= fband_low, 
                                                              high_peaks.freq_high <= fband_high)].time
                                                              
            # We now have:
            #         event times:       3,4,5,        10,11,12,13,       50,...
            #  spectro timeframes: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,...
            # 
            # We find each event start, and follow it along the spectro timeframes.
            # an interruption is the end of one event.
            # The event_start is the first in the sequences of rectangle (i.e. event) 
            # start times:
            event_start = spectro_times_to_mask.iloc[0]
            # Pt to the corresponding spectro timeframe. The .index.values[0]
            # turns the weird Pandas index-class number into a normal integer 
            # row number. spectro_new_time_idx will be pointer along the timeframe
            # seq:
            spectro_new_time_idx = spectro_times[spectro_times == event_start].index.values[0]
            spectro_start_event_time = spectro_times[spectro_new_time_idx]
            # Pointer along the event times seq:
            event_idx = 0
            
            # Actual times (i.e. dereference the pointers):
            spectro_time = spectro_times.iloc[spectro_new_time_idx]
            event_time   = spectro_times_to_mask.iloc[event_idx]
            try:
                while True:
                    # Have we found a 'hole' in the event time seq?
                    while spectro_time == event_time:
                        spectro_new_time_idx += 1
                        event_idx += 1
                        spectro_time = spectro_times.iloc[spectro_new_time_idx]
                        event_time   = spectro_times_to_mask.iloc[event_idx]
                        
                    # Found a rectangle righ-side time boundary (right bound non-inclusive):
                    new_sel_rect = (spectro_start_event_time, fband_low, 
                                    spectro_time, fband_high)
                    bounding_rects.append(new_sel_rect)
                    #***************
                    # Add the rectangle to the evolving chart:
                    ax = self.plot_rectangles(new_sel_rect, 
                                              spectro_times, 
                                              spectro_freqs, 
                                              ax=ax)
                    #***************
                    # Get ready for feeling along the next rectangle:
                    spectro_new_time_idx = \
                       spectro_times[spectro_times == spectro_times_to_mask.iloc[event_idx]].index.values[0]
                    spectro_start_event_time = spectro_times[spectro_new_time_idx]
                    spectro_time = spectro_start_event_time
            except IndexError:
                # Accounted for all hypothesized selection regions:
                pass
            
            # Set the mask areas of high acorrs to True:
            sels_mask.loc[spectro_freqs_to_mask, spectro_times_to_mask] = True

        self.plot_rectangles(bounding_rects, spectro_times, spectro_freqs)

        # Reshape sels_mask from spectrogram dimensions
        # (index is freq, columns are time frames) to 
        # something easier to visualize:
        #        time     in_selection
        #  freq   
        #  ...   0.008       True/False
        #  ...    ...         ...
        freq_time_coords = pd.melt(sels_mask, 
                                   var_name='time', 
                                   value_name='in_selection', 
                                   ignore_index=False)
        exp_key = f"spectral_events_{FileUtils.extract_file_timestamp(self.peaks_key)}"
        self.exp.save(exp_key, freq_time_coords)
        
        Charter.spectrogram_plot(sels_mask)
        #print(sels_mask)

    #------------------------------------
    # def event_mask_z_scores
    #-------------------
    
    def event_mask_z_scores(self, z_scores_key):

        #**************
        ax = None # No plot of rects started yet
        #**************

        self.log.info("Reading z-scores df...")
        z_scores_df = self.exp(z_scores_key, Datatype.tabular) 

        # The z_scores_df has columns:
        #    freq, time, mean_band_zscore
        #
        # The freqs are upper bounds of the freq bands. Create
        # two new columns: freq_low and freq_high:
        z_scores_df[['freq_low', 'freq_high']] = self.freq_bounds

        # Get row numbers grouped be frequency band. 
        # I.e. row numbers for band (1992.1875 to 2492.1875), 
        # band (2492.1875 to 2992.1875), etc. Only the bands
        # still at play after the above filter by peak_lower_level
        # are included, not all 32 bands:
        
        
        freq_grps         = z_scores_df.groupby(by=['freq_low', 'freq_high'])
        spectro_freqs     = pd.Series(sorted(self.spectro.index.values), 
                                      name='spectro_freqs')
        spectro_freqs_min = spectro_freqs.min()
        spectro_freqs_max = spectro_freqs.max()
        
        spectro_times = pd.Series(self.spectro.columns, name='spectro_times', dtype=float)
        
        # Create a spectrogram-shaped mask of True/False, and
        # a list of bounding rectangles that were found. We'll
        # fill bounding_rects with tuples:
        #    (freq_low_left, time_low_left, freq_high_right, time_high_right)
        
        bounding_rects = []

        # freq_grps.groups is a dict mapping freq_low/freq_high pairs
        # to row numbers:
        for fband_low, fband_high in freq_grps.groups.keys():
            
            # For now, don't include the freq band that covers
            # the entire height of the spectro:
            if (fband_low == spectro_freqs_min and fband_high == spectro_freqs_max):
                continue 
            
            self.log.info(f"Analyzing band {fband_low}Hz to {fband_high}Hz")
            
            # Each freq band combines several frequency channels in 
            # the original spectrogram. Get the set of those channels
            # that comprise the fband under consideration in this round
            # of the loop. The modulo len(spectro_freqs) accounts for
            # each frequency band include rows with all spectro times:

            #spectro_freqs_to_mask = spectro_freqs[spectro_freqs.between(fband_low, fband_high)]
            
            # Get the series of times at which acorr values were above threshold that
            # were detected in this loop's freq band. Some of these times will
            # be contiguous spectro frame times. Separate events in the freq
            # band will show as 'missing spectro frame times' in the time series
            # (see follow-on comment below):
            spectro_times_to_mask = z_scores_df[np.logical_and(z_scores_df.freq_low >= fband_low, 
                                                               z_scores_df.freq_high <= fband_high)].time
                                                              
            # We now have:
            #         event times:       3,4,5,        10,11,12,13,       50,...
            #  spectro timeframes: 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,...
            # 
            # We find each event start, and follow it along the spectro timeframes.
            # an interruption is the end of one event.
            # The event_start is the first in the sequences of rectangle (i.e. event) 
            # start times:
            event_start = spectro_times_to_mask.iloc[0]
            # Pt to the corresponding spectro timeframe. The .index.values[0]
            # turns the weird Pandas index-class number into a normal integer 
            # row number. spectro_new_time_idx will be pointer along the timeframe
            # seq:
            spectro_new_time_idx = spectro_times[spectro_times == event_start].index.values[0]
            spectro_start_event_time = spectro_times[spectro_new_time_idx]
            # Pointer along the event times seq:
            event_idx = 0
            
            # Actual times (i.e. dereference the pointers):
            spectro_time = spectro_times.iloc[spectro_new_time_idx]
            event_time   = spectro_times_to_mask.iloc[event_idx]
            try:
                while True:
                    # Have we found a 'hole' in the event time seq?
                    while spectro_time == event_time:
                        spectro_new_time_idx += 1
                        event_idx += 1
                        spectro_time = spectro_times.iloc[spectro_new_time_idx]
                        event_time   = spectro_times_to_mask.iloc[event_idx]
                        
                    # Found a rectangle righ-side time boundary (right bound non-inclusive):
                    new_sel_rect = (spectro_start_event_time, fband_low, 
                                    spectro_time, fband_high)
                    bounding_rects.append(new_sel_rect)
                    #***************
                    # Add the rectangle to the evolving chart:
                    ax = self.plot_rectangles(new_sel_rect, 
                                              spectro_times, 
                                              spectro_freqs, 
                                              ax=ax)
                    #***************
                    # Get ready for feeling along the next rectangle:
                    spectro_new_time_idx = \
                       spectro_times[spectro_times == spectro_times_to_mask.iloc[event_idx]].index.values[0]
                    spectro_start_event_time = spectro_times[spectro_new_time_idx]
                    spectro_time = spectro_start_event_time
            except IndexError:
                # Accounted for all hypothesized selection regions:
                pass
            
            # Set the mask areas of high acorrs to True:
            #****sels_mask.loc[spectro_freqs_to_mask, spectro_times_to_mask] = True
            print('foo')

    #------------------------------------
    # event_mask_energy
    #-------------------
    
    def event_mask_energy(self,
                          energy_peaks_key,
                          energy_thresholds=0.25,
                          smoothing_win=3
                          ):
        
        # Get df the following columns/index for each frequency:
        #
        #     freq                       : freq band upper-bound (in the index)
        #     time                       : time into spectrogram
        #     mean_band                  : one energy peak value mean-normalized
        #     prominence                 : prominence of the peak
        #     height_above_left_neighbor : difference between the peak & its left neighbor
        #     height_above_right_neighbor: difference between the peak & its left neighbor

        if type(smoothing_win) != list:
            smoothing_win = [smoothing_win]
            
        if type(energy_thresholds) != list:
            energy_thresholds = [energy_thresholds]
            
        peaks = self.exp.read(energy_peaks_key, Datatype.tabular, index_col='freq')
        # Better name for the mean_band column:
        peaks.rename({'mean_band' : 'energy'}, inplace=True, axis=1)
        
        # List of  threshold, window, start_time, stop_time dfs
        events_arr = []
        for thresh in thresholds:
            for win_width in smoothing_win:
                
                # Place to collect rows of smoothed energy df
                # that indicate a spectral event is happening:
                events = []
                
                # Roll a window across the Series, taking the
                # mean() over each:
                smooth_energy_ser = peaks['energy'].rolling(win_width, center=True).mean()
                
                # Turn into a df with index (i.e. 'freq') moved
                # to separate column, and index being default row nums:
                smooth_energy = smooth_energy_ser.reset_index()
                
                # Depending on window size, there will be nan
                # values at the start. Replace them with the first
                # non-nan in the result:
                first_valid_idx = smooth_energy.energy.first_valid_index() 
                first_valid_val = smooth_energy.iloc[first_valid_idx]
                smooth_energy[:first_valid_idx] = first_valid_val
                
                # Same with end of energies:
                last_valid_idx = smooth_energy.energy.last_valid_index() 
                last_valid_val = smooth_energy.iloc[last_valid_idx]
                smooth_energy[last_valid_idx:] = last_valid_val
                
                
                # Process each frequency band separately:
                freq_grp = smooth_energy.groupby(by='freq')
                # For each freq, get indices into smooth_energy whose
                # energy value is >= thresh
                meets_thresh = freq_grp.apply(lambda row, thresh=thresh : row.energy >= thresh)
                
                # Now have multiindexed Series; name the multiindex levels,
                # and also name the bools column indicating whether thresh is
                # met:
                meets_thresh.index.rename(['freq', 'smooth_energy_idx'], inplace=True)
                meets_thresh.name = 'above_thres'
                
                # Now have in meets_thresh something like:
                # freq        smooth_energy_idx
                # 492.1875    0                    False
                #             1                    False
                #             2                    True
                #              ...                    ...  
                # 15992.1875  27805                False
                #             27806                True
                #               ...
                # Name: above_thres, Length: 27810, dtype: bool
                
                # Separate the below-thresh entries from the
                # above-thresh. The ones above are part of a
                # predicted spectral event; the others are not:
                
                event_idxs     = meets_thresh[meets_thresh].index.get_level_values('smooth_energy_idx')
                non_event_idxs = meets_thresh[~meets_thresh].index.get_level_values('smooth_energy_idx')
                
                # Now have in event_idxs ( and non_event idxs):
                # freq  smooth_energy_idx
                # 400   1                    True  (False in non_event_idxs) 
                # 600   3                    True
                #       5                    True
                # Name: above_thres, dtype: bool

                # Get the indices 'column' (actually second level in multiindex)
                # for both data structs. Those are indices into the original
                # peaks df that has info about peaks:
                above_thresh_idxs = event_idxs.get_level_values('smooth_energy_idx')
                below_thresh_idxs = non_event_idxs.get_level_values('smooth_energy_idx')
                
                # We want start and stop times of events.
                # A start time is when an event_idx comes right
                # after a non_event_idx:
                start_time_idxs = above_thresh_idxs[
                    above_thresh_idxs.isin(below_thresh_idxs + 1)]
                stop_time_idxs  = below_thresh_idxs[
                    below_thresh_idxs.isin(above_thresh_idxs + 1)]
                
                # Number of rows in the result df:
                num_rows = len(start_time_idxs)

                # Buld result df...
                events = pd.concat(
                    {'freq'         : smooth_energy.iloc[start_time_idxs].reset_index().freq,
                     'start_time'   : peaks.iloc[start_time_idxs].reset_index().time,
                     'stop_time'    : peaks.iloc[stop_time_idxs].reset_index().time,
                     'threshold'    : pd.Series([thresh]*num_rows),
                     'win_width'    : pd.Series([win_width]*num_rows),
                     'energy_smooth': smooth_energy.iloc[start_time_idxs].reset_index().energy,
                     'energy_raw'   : peaks.iloc[start_time_idxs].reset_index().energy,
                     'prominence'   : peaks.iloc[start_time_idxs].reset_index().prominence,
                     'ht_abv_rght_neighbor' : peaks.iloc[start_time_idxs].reset_index().height_above_right_neighbor,
                     'ht_abv_lft_neighbor'  : peaks.iloc[start_time_idxs].reset_index().height_above_left_neighbor
                     }, axis=1)

                events_arr.append(events)
        # Done with all thresholds and window widths.
        # Stitch the results of those settings into one
        # df:

        res = pd.concat(events_arr, ignore_index=True)
        exp_key = f"spectral_events_{FileUtils.extract_file_timestamp(energy_peaks_key)}"
        self.exp.save(exp_key, res)
        return res

    #------------------------------------
    # plot_event_durations
    #-------------------
    
    def plot_event_durations(self, events_df):
        '''
        Given a df with identified spectral event start/stop times,
        plot lines that mirror the predicted start and stop times of
        spectrogram events. X-axis is time, Y-axis is frequency
         
        :param events_df:
        :type events_df:
        '''
        pass

    #------------------------------------
    # plot_rectangles
    #-------------------

    def plot_rectangles(self, 
                        rect_spec_list,
                        spectro_times,
                        spectro_freqs, 
                        edgecolor='black', 
                        facecolor='none',
                        ax=None):

        if rect_spec_list != pd.DataFrame:
            if type(rect_spec_list) != list:
                # Got just a tuple for a single rect:
                rect_spec_list = [rect_spec_list]
            data = pd.DataFrame(rect_spec_list, 
                                columns=['x', 'y', 'width', 'height'])
            
        ax = Charter.rectangles(data,
                                spectro_times,
                                spectro_freqs,
                                ylabel='Frequency (Hz)', 
                                xlabel='Time (s)',
                                edgecolor=edgecolor,
                                facecolor=facecolor,
                                ax=ax
                                )
        return ax


    #------------------------------------
    # find_next_labeled_selection
    #-------------------
    
    def find_next_labeled_selection(self, freq):
        pass


    #------------------------------------
    # _closest_labeled_sel_time_bound
    #-------------------
    
    def _closest_labeled_sel_time_bound(self,
                                ref_time, 
                                which_bound=Bound.START, 
                                direction=Direction.AHEAD,
                                freq_constraint=None):
        '''
        Find closest ref_time at which any of the selections in self.selections_df
        begins or ends. If which_bound is Bound.START, the start
        ref_time of the selection is sought. Else for Bound.STOP the
        closest selection stop ref_time is sought.  
        
        If direction is Direction.AHEAD, only selection boundaries
        later than the given ref_time are considered. For Direction.BEHIND, only 
        selections that start earlier are considered. Or use Direction.BOTH.
        
        All times and frequencies are snapped to the spectrogram that
        is under examination (self.spectro).
        
        Returns the selection ID. ITERATOR
        
        Assumption: dataframe self.selections_df is available, and contains cols:
            spectro_times,
            snapped_start_time,
            snapped_stop_time,
            sel_id,
            sel_dur,
            is_sel_start,
            is_sel_stop,
            snapped_freq_low,
            snapped_freq_high

        :param ref_time: reference time for which closest bound
            is to be found
        :type ref_time: float
        :param which_bound: whether to look for closeness of
            selection starts, or of selection stops
        :type which_bound: Bound
        :param direction: whether to limit search for selections
            that are ahead of given ref_time, or behind the given ref_time
        :type direction: Direction
        :returns
        :rtype 
        '''
        if type(freq_constraint) == Interval:
            low_freq_bound  = freq_constraint['low_val']
            high_freq_bound = freq_constraint['high_val']
        elif freq_constraint is None:
            low_freq_bound   = min(self.sels.snapped_freq_low)
            high_freq_bound  = max(self.sels.snapped_freq_high)
        else:
            low_freq_bound  = freq_constraint
            high_freq_bound = max(self.sels.snapped_freq_high)
            
        # Sel starts only, later than ref_time 
        if which_bound == Bound.START and direction == Direction.AHEAD:
            sel_excerpt = self.sels[(self.sels.snapped_start_time >= ref_time) &
                                    (self.sels.snapped_freq_low >= low_freq_bound) &
                                    (self.sels.snapped_freq_high <= high_freq_bound)]

        elif which_bound == Bound.START and direction == Direction.BEHIND:
            sel_excerpt = self.sels[(self.sels.snapped_start_time <= ref_time) &
                                    (self.sels.snapped_freq_low >= low_freq_bound) &
                                    (self.sels.snapped_freq_high <= high_freq_bound)]

        elif which_bound == Bound.STOP and direction == Direction.AHEAD:
            sel_excerpt = self.sels[(self.sels.snapped_stop_time >= ref_time) &
                                    (self.sels.snapped_freq_low >= low_freq_bound) &
                                    (self.sels.snapped_freq_high <= high_freq_bound)]

        elif which_bound == Bound.STOP and direction == Direction.BEHIND:
            sel_excerpt = self.sels[(self.sels.snapped_stop_time <= ref_time) &
                                    (self.sels.snapped_freq_low >= low_freq_bound) &
                                    (self.sels.snapped_freq_high <= high_freq_bound)]

        cols_of_interest = sel_excerpt[['sel_id', 
                                        'snapped_start_time', 
                                        'snapped_freq_low', 
                                        'snapped_freq_high']]

        return cols_of_interest

    #------------------------------------
    # cartesian
    #-------------------
    
    def cartesian(self, arrays, out=None):
        """
        Generate a cartesian product of input arrays.
    
        Parameters
        ----------
        arrays : list of array-like
            1-D arrays to form the cartesian product of.
        out : ndarray
            Array to place the cartesian product in.
    
        Returns
        -------
        out : ndarray
            2-D array of shape (M, len(arrays)) containing cartesian products
            formed of input arrays.
    
        Examples
        --------
        >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
        array([[1, 4, 6],
               [1, 4, 7],
               [1, 5, 6],
               [1, 5, 7],
               [2, 4, 6],
               [2, 4, 7],
               [2, 5, 6],
               [2, 5, 7],
               [3, 4, 6],
               [3, 4, 7],
               [3, 5, 6],
               [3, 5, 7]])
    
        """
    
        arrays = [np.asarray(x) for x in arrays]
        dtype = arrays[0].dtype
    
        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)
    
        #m = n / arrays[0].size
        m = int(n / arrays[0].size) 
        out[:,0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            self.cartesian(arrays[1:], out=out[0:m, 1:])
            for j in range(1, arrays[0].size):
            #for j in xrange(1, arrays[0].size):
                out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
        return out        

    #------------------------------------
    # _obtain_spectrogram
    #-------------------
    
    def _obtain_spectrogram(self, fname_or_key):
        if fname_or_key.endswith('.csv'):
            # Filename, either absolute, or relative to
            # <proj_root>/data: 
            if not os.path.isabs(fname_or_key):
                # Relative:
                fname_or_key = os.path.join(SegmentationComputer.PROJ_ROOT,
                                            f"data/{fname_or_key}")
                if not os.path.exists(fname_or_key):
                    raise FileNotFoundError(f"Spectram file {fname_or_key} not found")
                
                self.log.info("Reading spectrogram from .csv file...")
                self.spectro = pd.read_csv(fname_or_key,
                                           index_col='freq',
                                           header=0,
                                           engine='pyarrow'  # Faster csv reader
                                           )
                self.log.info("Done reading spectrogram from .csv file.")
        else:
            self.log.info(f"Reading spectrogram from experiment ({fname_or_key})...")
            self.spectro = self.exp.read(fname_or_key, Datatype.tabular)
            self.log.info(f"Done reading spectrogram from experiment ({fname_or_key}).")
        
        return self.spectro


# ----------------------- Class Rect ---------------

class Rect:
    
    #------------------------------------
    #  Constructor
    #-------------------
    
    def __init__(self, ll=None, ur=None, width=None, height=None):

        self._check_arguments(ll, ur, width, height)
        
    #------------------------------------
    # intersection_over_union
    #-------------------
    
    def intersection_over_union(self, other, epsilon=1e-5):
        '''
        Given two rectangles return Intersection over Union (IoU)
        value. These values range from 0 (no overlap) to 1 (exact
        overlap).
        
        Adapted from http://ronny.rest/tutorials/module/localization_001/iou/

        :param other: the rectangle whose overlap with self is
            to be computed
        :type other: Rect
        :param epsilon: small number to avoid division by zeor
        :type epsilon: float
        :return the IoU
        :rtype float
        '''

        # I'm stealing this code from Ronny Restpro's blog. His function
        # takes two four-tuples, one for the self-rect, the other for
        # the other-rect. Each four tuple numbers are:
        # where:
        #     x1,y1 represent the upper left corner
        #     x2,y2 represent the lower right corner
        # To avoid having to think about converting that to our
        # Rect class' lower-left, upper-right convention:
        
        a = np.array([self.ll_x, self.ur_y, self.ur_x, self.ll_y])
        b = np.array([other.ll_x, other.ur_y, other.ur_x, other.ll_y])
        
        # COORDINATES OF THE INTERSECTION BOX
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        # AREA OF OVERLAP - Area where the boxes intersect
        width  = (x2 - x1)
        height = (y2 - y1)
        # handle case where there is NO overlap
        if (width<0) or (height <0):
            return 0.0
        area_overlap = width * height
    
        # COMBINED AREA
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        area_combined = area_a + area_b - area_overlap
    
        # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
        iou = area_overlap / (area_combined+epsilon)
        return iou

    #------------------------------------
    # _check_arguments
    #-------------------

    def _check_arguments(self, ll, ur, width, height):
        '''
        Given a combination of lower-left, upper-right,
        width and height, compute or initialize attributes
        self.ll_x, self.ll_y, self.width, and self.height.
        
        :param ll:
        :type ll:
        :param ur:
        :type ur:
        :param width:
        :type width:
        :param height:
        :type height:
        '''

        # Ensure we have a full spec:
        # At least one of lower-left or upper-right must be provided
        if ll is None and ur is None:
            raise ValueError("Must provide lower-left or upper-right, or both")

        # If only one of ll and ur is provided, then
        # both width and height must be provided:
        if ll is None or ur is None:
            if width is None or height is None:
                raise ValueError("With only one of lower-left and upper-right provided, both width and height are needed")

        if ll is not None and ur is not None:
            # Given lower-left and upper-right: compute width and height:
            ll_x = ll[0]
            ll_y = ll[1]
            ur_x = ur[0]
            ur_y = ur[1]
            width  = ur_x - ll_x 
            height = ur_y - ll_y
        else:
            # One of ll or ur is None:
            if ll is not None:
                ll_x = ll[0]
                ll_y = ll[1]
                ur_x = ll_x + width
                ur_y = ll_y + height
            else:
                # ll is None, ur is not:
                ur_x = ur[0]
                ur_y = ur[1]
                ll_x = ur_x - width
                ll_y = ur_y - height

        self.ll_x = ll_x
        self.ll_y = ll_y
        self.ur_x = ur_x
        self.ur_y = ur_y
        self.width  = width
        self.height = height

# ------------------------ Main ------------
if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Compute spectrogram event starts and stops"
                                     )

    parser.add_argument('-c', '--command',
                        choices=['event_mask'],
                        required=True,
                        help='Command to perform',
                        )

    parser.add_argument('-e', '--experiment',
                        required=True,
                        help=f"Experiment root: subdir of {SegmentationComputer.EXP_ROOT}",
                        )

    parser.add_argument('-s', '--selection_tbl',
                        type=str,
                        help=f"path to selection table; if not an absolute path, looks in <proj_root>/data",
                        default=None
                        )

    parser.add_argument('-p', '--spectrogram',
                        type=str,
                        help=(f"For peaks only: fname of audio or spectrogram;\n",
                              'or spectrogram experiment key') ,
                        default=None
                        )
        
    parser.add_argument('-m', '--measures',
                        choices=['sigs', 'z_scores', 'energy'],
                        help='Which info to use for computing call times',
                        default=None
                        )
    parser.add_argument('-t', '--thresholds',
                        type=float,
                        nargs='+',
                        help='Repeatable: measure threshold above which event is concluded')

    parser.add_argument('-w', '--window',
                        type=int,
                        help=(f"Width of smoothing window") ,
                        default=None
                        )

    parser.add_argument('source_info',
                        type=str,
                        help='Filename or experiment key for signal peaks file to use'
                        )
    args = parser.parse_args()

    # The root of the experiment where the .csv files are:
    exp_root = os.path.join(SegmentationComputer.EXP_ROOT, args.experiment)
    exp = ExperimentManager(exp_root)

    #acorr_peaks_key = 'significant_acorrs_2022-07-13T17_19_19'
    #acorr_peaks_key = 'peaks_2022-07-20T10_47_48'
    #sel_info_key    = 'selections_infoAM02_20190717_052958'
    #sel_info_key    = 'selections_infoAM02_20190717_052958'
    #spectro_key     = 'am02_20190717_052958_spectro'
    #spectro_key     = None
    #z_score_key     = 'am02_20190717_052958_spectro_z_scores'

    sel_info_key = args.selection_tbl
    if args.spectrogram is None:
        spectro_key = None
        
    smoothing_win = args.window
    if smoothing_win is not None and type(smoothing_win) != int:
        print(f"Window option must be an integer")
        sys.exit(1)
        
    thresholds = args.thresholds
    measures = args.measures

    seg_computer = SegmentationComputer(exp, sel_info_key, args.source_info, spectro_key)
    
    # OPTIONS:
    #seg_computer.event_mask_sigs(['continuity', 'energy_sum'], peak_lower_limit=0.95)
    #seg_computer.event_mask_sigs(['flatness', 'continuity', 'pitch', 'freq_mod', 'energy_sum'],
    #                        peak_lower_limit=0.95)
    #seg_computer.event_mask_z_scores(z_score_key)

    if args.command == 'event_mask':
        if measures == 'energy':
            seg_computer.event_mask_energy(args.source_info,
                                           thresholds,
                                           smoothing_win
                                           )
        elif measures == 'z_scores':
            seg_computer.event_mask_z_scores(args.source_info)

        elif measures == 'sigs':
            #seg_computer.event_mask_sigs()
            print("Signatures based even masking not implemented")
    print('Done')