'''
Created on Jul 13, 2022

@author: paepcke
'''

from enum import Enum
import os

from data_augmentation.utils import Interval
from experiment_manager.experiment_manager import Datatype, ExperimentManager
from logging_service.logging_service import LoggingService

import numpy as np
import pandas as pd

from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

from birdsong.utils.utilities import FileUtils
from result_analysis.charting import Charter
from powerflock.signal_analysis import SignalAnalyzer


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

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, experiment, acorr_peaks_key, sel_tbl_key):
        '''
        '''
        #*******
        self.spectro_path = SPECTRO_PATH
        #*******

        self.acorr_peaks_key = acorr_peaks_key
        
        self.log = LoggingService()
        self.exp = experiment
        acorr_df_raw = experiment.read(acorr_peaks_key, Datatype.tabular)

        # Pull out just the columns we need:
        self.acorr_peaks = acorr_df_raw[acorr_df_raw.lag == 1][['time','sig_type', 'acorr','freq_low', 'freq_high']]
        
        self.sels = experiment.read(sel_tbl_key, Datatype.tabular)
        
        self.log.info("Reading spectrogram from .csv file...")
        self.spectro = pd.read_csv(self.spectro_path,
                                   index_col='freq',
                                   header=0,
                                   engine='pyarrow'  # Faster csv reader
                                   )
        # Make the string 'floats' into real floats:
        self.spectro.columns = self.spectro.columns.values.astype(float)

    #------------------------------------
    # event_mask
    #-------------------

    def event_mask(self, sig_types, peak_lower_limit=0.9):
    
        if type(sig_types) != list:
            sig_types = [sig_types]

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
        exp_key = f"spectral_events_{FileUtils.extract_file_timestamp(self.acorr_peaks_key)}"
        self.exp.save(exp_key, freq_time_coords)

        Charter.spectrogram_plot(sels_mask)
        #print(sels_mask)

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


# ------------------------ Main ------------
if __name__ == '__main__':
    
    CUR_DIR = os.path.dirname(__file__)
    PROJ_ROOT  = os.path.abspath(os.path.join(CUR_DIR, '../../'))
    EXP_ROOT = os.path.join(PROJ_ROOT, 'experiments/SoundscapeSegmentation')
    exp_root = os.path.join(EXP_ROOT, 'ExpAM02_20190717_052958_AllData1')
    exp = ExperimentManager(exp_root)

    #acorr_peaks_key = 'significant_acorrs_2022-07-13T17_19_19'
    acorr_peaks_key = 'peaks_2022-07-13T17_19_19'
    sel_info_key    = 'selections_infoAM02_20190717_052958'
    seg_computer = SegmentationComputer(exp, acorr_peaks_key, sel_info_key)
    
    seg_computer.event_mask(['continuity', 'energy_sum'], peak_lower_limit=0.95)
    
    ref_time = 19.0
    sel_excerpt = seg_computer._closest_labeled_sel_time_bound(ref_time, which_bound=Bound.START, direction=Direction.AHEAD)
    
    print(sel_excerpt)
    