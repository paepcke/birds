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

    #------------------------------------
    # event_mask
    #-------------------

    def event_mask(self, sig_types, peak_lower_limit=0.9):
    
        if type(sig_types) != list:
            sig_types = [sig_types]
            
        # Only look at acorr peaks that were
        # computed on one of the requested sig types:
        df = self.acorr_peaks[np.logical_and(self.acorr_peaks.sig_type.isin(sig_types), 
                                             self.acorr_peaks.acorr >= peak_lower_limit)]\
                                             [['time','acorr','freq_low', 'freq_high']]

        time_axis = df.time.unique()
        freq_axis = df.freq_low.unique()
        
        mask_height, mask_width = self.spectro.shape 
        
        sels_mask = pd.DataFrame(np.array([False] * mask_height * mask_width).reshape((mask_height, mask_width)),
                                 index = self.spectro.index,
                                 columns=self.spectro.columns.astype(float))
        
        spectro_freqs = pd.Series(sorted(self.spectro.index.values), name='spectro_freqs')
        spectro_times = pd.Series(sorted(self.spectro.columns.values.astype(float)), name='spectro_times')
        fband_bounds = sorted(set(zip(df.freq_low, df.freq_high)))
        for fband_low, fband_high in fband_bounds:
            spectro_idxs  = np.where((np.logical_and((spectro_freqs >= fband_low), 
                                                     (spectro_freqs < fband_high))))
            event_spectro_freqs = spectro_freqs.iloc[spectro_idxs]
            coord_pairs = self.cartesian((event_spectro_freqs, time_axis))
            for freq, time in coord_pairs:
                sels_mask.at[freq, time] = True
                
            spectro_detail_freqs = self.spectro.iloc[spectro_idxs].index
            fband_times = df[df.freq_low == fband_low].time
            mask_row = pd.Series([False]*mask_width, index=sels_mask.columns)
            mask_row.loc[fband_times] = True
            #******* NEXT Set all sels_mask rows spectro_detail_freqs to mask_row
            
            

        print(df)

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

    acorr_peaks_key = 'significant_acorrs_2022-07-13T17_19_19'
    sel_info_key    = 'selections_infoAM02_20190717_052958'
    seg_computer = SegmentationComputer(exp, acorr_peaks_key, sel_info_key)
    
    seg_computer.event_mask(['energy_sum'])
    
    ref_time = 19.0
    sel_excerpt = seg_computer._closest_labeled_sel_time_bound(ref_time, which_bound=Bound.START, direction=Direction.AHEAD)
    
    print(sel_excerpt)
    