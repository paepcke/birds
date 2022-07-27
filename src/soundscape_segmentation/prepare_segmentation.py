#!/usr/bin/env python
'''
Created on May 30, 2022

Given a spectrogram, computes two files:
   o significant_acorrs_<timestamp>.csv
   o peaks_<timestamp>.csv
   
The two timestamps will match. The acorrs file is produced
by calling compute_autocorrelations(), and contains:

   time,sig_type,freq_low,freq_high,lag,acorr,ci_low,ci_high

The sig_type is 'continuity', 'pitch', 'freq_mod', 'flatness', and 'energy_sum'
   
Calling peak_positions_acorrs() then finds peaks among the autocorrelations.
The peaks file contains:

   time,sig_type,freq_low,freq_high,lag,acorr,ci_low,ci_high,plateau_width,prominence
   
All computations are done separately for (modifiable) 500Hz frequency bands.

@author: paepcke
'''

#************
import sys
sys.path.insert(0,'/Users/paepcke/EclipseWorkspacesNew/birds/src')
#************

from _functools import partial
import argparse
from enum import Enum
import json
import os
from pathlib import Path
import warnings

from birdsong.utils.utilities import FileUtils
from data_augmentation.utils import RavenSelectionTable, Utils
from experiment_manager.experiment_manager import ExperimentManager, \
    JsonDumpableMixin, Datatype
from logging_service import LoggingService
from powerflock.signal_analysis import SignalAnalyzer
from powerflock.signatures import Signature
from result_analysis.charting import Charter
from scipy.signal import find_peaks
from scipy.signal._peak_finding_utils import PeakPropertyWarning

import multiprocessing as mp
import numpy as np
import pandas as pd

class PatternLookbackDurationUnits(str, Enum):
    '''
    Distinguish between numbers that indicate
    lag, vs. seconds (the two are roughtly 
    interchangabe. Note: the values are chosen
    to be strings to make conversion to and 
    from json work without a special encoder.
    Same for the addition of str to the inheritance.
    '''
    LAGS = 'lags'
    SECONDS = 'time'

    @classmethod
    def from_value(self, val):
        '''
        Given the 'value side' of an instance
        return a corresponding instance. Used
        when recreating from Json string
         
        :param val: either 'lags' or 'time'
        :type val: str
        :return fresh instance of PatternLookbackDurationUnits
        :rtype PatternLookbackDurationUnits
        '''
        if val == 'lags':
            return PatternLookbackDurationUnits.LAGS
        else:
            return PatternLookbackDurationUnits.SECONDS

#from result_analysis.charting import Charter
CUR_DIR    = os.path.dirname(__file__)
PROJ_ROOT  = os.path.abspath(os.path.join(CUR_DIR, '../../'))
TEST_DATA  = os.path.join(PROJ_ROOT, 'data')
AUDIO_PATH =  os.path.join(TEST_DATA, 'kelleyRecommendedFldRec_AM02_20190717_052958.wav')
TEST_SEL_TBL = os.path.join(TEST_DATA, 'DS_AM02_20190717_052958.Table.1.selections.txt')
SPECTRO_PATH = os.path.join(TEST_DATA, 'am02_20190717_052958_spectro.csv')

EXP_ROOT = os.path.join(PROJ_ROOT, 'experiments/SoundscapeSegmentation')

#from data_augmentation.sound_processor import SoundProcessor 

class SegmentationPreparer:
    '''
    
    '''
    
    PATTERN_LOOKBACK_DURATION = 0.5 # sec
    '''Number of seconds to consider during autocorrelation computations for detecting patterns'''

    ROUND_TO = 5
    '''Number of digits to which time floats are rounded throughout'''

    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    def __init__(self, experimenter, settings):
        '''
        Constructor
        '''

        self.experimenter  = experimenter
        self.settings      = settings
        self.spectro_path  = settings.spectro_path
        self.audio_path    = Path(settings.audio_path) if settings.audio_path is not None else None
        selection_tbl_path = settings.selection_tbl_path
        self.round_to      = settings.round_to
        self.freq_low, self.freq_high, self.band_height = settings.freq_split
        self.num_freq_bands = int(self.freq_high / self.band_height)
        raw_lookback_duration  = settings.pattern_lookback_dur
        # Whether to remove selections span most of the
        # entire spectrogram:
        remove_long_selections = settings.remove_long_selections
        do_noisereduce = settings.do_noisereduce

        self.log = LoggingService()
        
        # Get the time for one autocorrelation lag.
        # The extra_granularity doubles the frequency
        # range:
        self.lag_duration = SignalAnalyzer.spectro_timeframe_width(sample_rate=settings.sample_rate,
                                                                   extra_granularity=True
                                                                   )

        self.compression_window_width = settings.compression_window_width // self.lag_duration

        # Need number of lags to compute for each autocorrelation.
        # The lag may be given in seconds, or directly in lags:
        if settings.pattern_lookback_dur_units == PatternLookbackDurationUnits.LAGS:
            # Given in lags:
            self.num_lags = raw_lookback_duration
            self.lookback_duration = raw_lookback_duration * self.lag_duration
        else:
            # Lookback duration given in number of seconds:
            self.num_lags = int(raw_lookback_duration / self.lag_duration)
            self.lookback_duration = raw_lookback_duration 

        # Add lag and lookback duration as seconds to the settings:
        settings.num_lags = self.num_lags
        settings.lookback_duration = self.lookback_duration
        
        # Save the settings of this run:
        experimenter.save('exp_settings', settings)
        
        # Figure out how caller pointed to spectrogram, if we are to
        # use an existing one. Could be an experiment key, or
        # an absolute file name, or None
        
        try:
            # Read spectrogram csv into df, turning the 'freq' col
            # into the index
            self.spectro = experimenter.read(self.spectro_path, 
                                             Datatype.tabular,
                                             index_col='freq'
                                             )
        except (FileNotFoundError, TypeError):
            self.spectro = self._obtain_spectrogram(do_noisereduce)

        # Get the spectrogram times as np array of rounded floats:
        spectro_times_rounded = self.spectro.columns.to_numpy(dtype=float).round(self.round_to)
        self.spectro_times = pd.Series(spectro_times_rounded, 
                                       index=spectro_times_rounded, 
                                       name='spectro_times')

        # Freqs in low to high order:
        self.spectro_freqs = pd.Series(self.spectro.index,
                                       index=self.spectro.index, 
                                       name='spectro_freqs').sort_values()
        
        # Get frequency step size of the spectro. Any
        # difference in successive spectro index values
        # will do, except that the ones at the end 
        # are half steps. Index runs highest-freq to 0:
        self.channel_steps = self.spectro.index[-3] - self.spectro.index[-2]

        if selection_tbl_path is not None:
            self.selections_df = self._process_selection_tbl(self.spectro_times, 
                                                             selection_tbl_path,
                                                             remove_long_selections)
        else:
            # No selection table available:
            sel_id  = pd.Series(np.nan, name='sel_id', index=self.spectro_times)
            is_sel_start = pd.Series(np.nan, name='is_sel_start', index=self.spectro_times)
            is_sel_stop  = pd.Series(np.nan, name='is_sel_stop', index=self.spectro_times)
            sel_duration = pd.Series(np.nan, name='sel_duration', index=self.spectro_times)
            
            self.selections_df = pd.DataFrame({'sel_id'    : sel_id,
                                               'labeled_sel_start' : is_sel_start,
                                               'labeled_sel_stop'  : is_sel_stop,
                                               'labeled_sel_dur'   : sel_duration
                                               }) 

        # Freq lookup tbl: human-marked freq to spectro freq
        # in low to high order:
        self.freqs_sel2spectro = {}
        for freq_intval in self.sel_tbl.freq_intervals:
            freq_low  = freq_intval['low_val']
            freq_high = freq_intval['high_val']
            
            self.freqs_sel2spectro[freq_low]  = Utils.nearest_in_array(self.spectro_freqs, freq_low)
            self.freqs_sel2spectro[freq_high] = Utils.nearest_in_array(self.spectro_freqs, freq_high)

    #------------------------------------
    # _obtain_spectrogram
    #-------------------
    
    def _obtain_spectrogram(self, do_noisereduce):
        if type(self.spectro_path) != str or not os.path.exists(self.spectro_path):
            
            # Read audio file and produce spectrogram. Audio
            # will first be noise-reduced, if do_noisereduce 
            # was True in settings. Save the spectrogram to the experiment.
            
            # First, create an experiment key for the new
            # spectrogram:
            
            # Timestamp to be used in all experiment keys:
            self.exp_timestamp = FileUtils.file_timestamp()

            # Build a memorable experiment key for the
            # new spectrogram. Something like:
            #    AM02_20190717_052958_spectro_2022-07-24T11_44_28.csv
            
            audiomoth_id = FileUtils.extract_audiomoth_id(self.audio_path)
            if audiomoth_id is None:
                self.log.warn(f"No audiomoth ID found in fname '{self.audio_path}'")
                audiomoth_id = ''

            exp_spectro_key = f"{audiomoth_id}_spectro_{self.exp_timestamp}{'_noisereduced' if do_noisereduce else ''}"
            
            self.spectro = self.make_test_spectro(self.audio_path,
                                                  sr=self.settings.sample_rate,
                                                  dest_spectro_key=exp_spectro_key, 
                                                  do_noisereduce=do_noisereduce)
            
        else:
            self.log.info("Reading spectrogram from .csv file...")
            self.spectro = pd.read_csv(self.spectro_path,
                                       index_col='freq',
                                       header=0,
                                       engine='pyarrow'  # Faster csv reader
                                       )
            # Timestamp to be used in all experiment keys:
            # first try to find the timestamp in the spectro fname:
            self.exp_timestamp = Utils.timestamp_from_exp_path(self.spectro_path)
            if self.exp_timestamp is None:
                self.exp_timestamp = FileUtils.file_timestamp()
        
        return self.spectro

    #------------------------------------
    # slice_spectro
    #-------------------
    
    def slice_spectro(self, 
                      slice_height=None, 
                      low_freq=None, 
                      high_freq=8000,
                      save_slices=False
                      ):
        '''
        Slice given spectrogram into slice_height Hz high horizontal slices. If
        requested, save each slice dataframe to the experiment using 
        spectro file name with center freq appended as save key.
        
        Returns an array of SpectroSlice instances. The final 'slice'
        covers the entire spectrogram.
        
        Note: frequency values are 'snapped' to the nearest frequency
        available in the spectrogram.
        
        Assumption: self.spectro contains the full spectrogram

        :param slice_height: frequency band height in Hz. Default: self.band_height
        :type slice_height: {int | float}
        :param low_freq: frequency at which to start. Default is slice_height 
        :type low_freq: {None | int | float}
        :param high_freq: frequency beyond which no 
            slices are computed.
        :type high_freq: {int | float}
        :param save_slices: whether or not to save each
            slice to the current experiment
        :type save_slices: bool
        :returns list of SpectroSlice instances
        :rtype [SpectroSlice]
        '''
        
        if slice_height is None:
            slice_height = self.band_height
        if low_freq is None:
            low_freq = slice_height
        
        self.log.info(f"Slicing spectro into {int((high_freq - low_freq) / slice_height)} freq bands: {slice_height}Hz high [{low_freq}KHz, {high_freq}KHz]")
        slice_arr = []
        freqs = pd.Series(self.spectro.index, name='freqs')
        
        for target_freq_low in np.arange(low_freq, high_freq, slice_height):
            low_freq = max(freqs[freqs <= target_freq_low])
            high_freq = max(freqs[freqs <= target_freq_low + slice_height])

            # Cut the slice. Weirdly the following returns 0 rows:
            #    slice = spectro.loc[low_freq:high_freq,:] 
            spectro_slice = self.spectro[(self.spectro.index >= low_freq) & (self.spectro.index <= high_freq)]
            # Round the time columns to 5 places:
            spectro_slice.columns = pd.Series(spectro_slice.columns.astype(float)).round(self.round_to)
            
            slice_arr.append(SpectroSlice(low_freq, high_freq, slice_height, spectro_slice)) 
            
            if save_slices:
                self.experimenter.save(f"{self.recording_id}_{target_freq_low}Hz", spectro_slice)
                
        # Add a slice that covers the entire spectrogram freq range:
        slice_arr.append(SpectroSlice(min(self.spectro.index), # low slice freq
                                      max(self.spectro.index), # high slice freq
                                      max(self.spectro.index), # slice height
                                      self.spectro
                                      ))
        
        return slice_arr

    #------------------------------------
    # compute_sigs 
    #-------------------
    
    def compute_sigs(self, spectro_slice_or_spectro):
        '''
        For the given spectrogram, or SpectroSlice instance,
        compute the signature values ('continuity', 'pitch', etc)
        
        Returns a Signature instance in which all measures are
        normalized.
        
        :param spectro_slice_or_spectro:
        :type spectro_slice_or_spectro:
        :return: signature holding df with requested values, all normalized
        :rtype: pd.DataFrame
        '''
        
        if type(spectro_slice_or_spectro) == SpectroSlice:
            slice_desc = f"slice {spectro_slice_or_spectro.name}"
            spectro = spectro_slice_or_spectro.spectro
        else:
            slice_desc = f"spectro {min(spectro.index)}Hz to {max(spectro.index)}Hz"
            
        self.log.info(f"Computing signatures for {slice_desc}")
        
        # Make an empty sig, and fill it with the
        # four signatures across the whole spectro:
        sig_for_whole_spectro = Signature('test_species', 
                                          spectro, 
                                          fname=self.audio_path,
                                          sig_id=0)
        sig = SignalAnalyzer.spectral_measures_each_timeframe(spectro, 
                                                              sig_for_whole_spectro,
                                                              self.settings.sig_types)
        
        # Normalize the values using regular standardization, column by column:
        # Could do sig_normalized = (sig.sig - sig.sig.mean()) / sig.sig.std(). But
        # safer to use the Signature class' way of doing it.
        
        scale_info = {'flatness' : {'mean' : sig.sig.flatness.mean(),
                                    'standard_measure' : sig.sig.flatness.std()
                                    },
                      'continuity' : {'mean' : sig.sig.continuity.mean(),
                                      'standard_measure' : sig.sig.continuity.std()
                                    },
                      'pitch' : {'mean' : sig.sig.pitch.mean(),
                                 'standard_measure' : sig.sig.pitch.std()
                                    },
                      'freq_mod' : {'mean' : sig.sig.freq_mod.mean(),
                                    'standard_measure' : sig.sig.freq_mod.std()
                                    },
                      'energy_sum' : {'mean' : sig.sig.energy_sum.mean(),
                                    'standard_measure' : sig.sig.energy_sum.std()
                                    },
                                    
        }
        sig.normalize_self(scale_info)
        return sig

    #------------------------------------
    # compute_autocorrelations
    #-------------------
    
    def compute_autocorrelations(self, settings):
        '''
        Given a signature and a list of signature column names: ['flatness', 'pitch', ...],
        apply autocorrelation to each column up to number of lags equivalent
        to lookback_duration. This computation is done separately for each
        frequency slice.
        
        Writes result to significant_acorrs_<timestamp>.csv 
        
        Return:
                   flatness_rho, flatness_sum_significant, pitch_rho, pitch_sum_significant, ...
            time
              0          ...               True/False         ...          True/False
              0.0232
                ...
                
        Assumption: the settings instance contains a lookback_duration attribute
                    that specifies the length in seconds of the autocorrelations.

        
        :param col_names: name of measure(s) for which to compute autocorrelations
        :type col_names: {str | [str]}
        :param lookback_duration: time that covers the number of lags
            given the timeframes of the underlying spectrogram
        :type lookback_duration: float
        '''
        
        col_names = settings.sig_types
        lookback_duration = settings.lookback_duration

        low_freq, high_freq, slice_height = settings.freq_split
        # The highest-freq slice can only be as high
        # as the highest spectrogram frequency. Therefore
        # the min(...) below:
        slice_arr = self.slice_spectro(slice_height=slice_height,
                                       low_freq=low_freq,
                                       high_freq=min(high_freq, self.spectro.index.max())
                                       )
        if type(col_names) != list:
            col_names = [col_names]
        
        #*********
        # Uncomment to do one at a time without parallelism:
        #acorr_by_slice = []
        #for one_slice in slice_arr:
        #    acorr_by_slice.append(self._process_one_freq_slice(one_slice, col_names, lookback_duration))
        #*********

        #********* Read DF INSTEAD OF COMPUTING IT
        #self.log.info("Temporary test: read acors df instead of computing it...")
        #final_res_df = pd.read_csv('/Users/paepcke/EclipseWorkspacesNew/birds/experiments/SoundscapeSegmentation/ExpAM02_20190717_052958_AllData1/csv_files/significant_acorrs_2022-07-07T12_12_17.csv',
        #                           engine='pyarrow')
        #********* Read DF INSTEAD OF COMPUTING IT

        # Leave one CPU core for others:
        pool = mp.Pool(mp.cpu_count() - 1)
        
        res_objs = [pool.apply_async(self._process_one_freq_slice,
                                     (one_slice, col_names, lookback_duration)) 
                    for one_slice in slice_arr]
        
        final_res_arrs = [res_obj.get() for res_obj in res_objs]
        pool.close()
        pool.join()
        
        # final_res_arrs is a list of list of dataframes.
        # Each sublist has a dataframe for one of the time
        # series (sig values). E.g. 5 dfs for flatness, pitch,
        # etc. There are as many of these df lists as there are
        # frequency bands. For 15 bands that would be be
        # 15*5 == 75 sublists.
        # We must flatten this list into a simple list
        # of dfs, and then concat them into one big result:
        
        final_res_arrs_flat = []
        for several_res_dfs in final_res_arrs:
            final_res_arrs_flat.extend(several_res_dfs)
        #
        # Turn the list of dataframes into one large df.
        # The index of that big df will be repeating the
        # the index of the constituent correlation dfs, which
        # is not useful; so reset the index to simple row nums.
        # The 'drop=True' is required b/c the index cols of the dfs
        # are named 'Time', and there already is a 'Time' column.
        # So: failure if trying to copy the index to a new col:
        final_res_df = pd.concat(final_res_arrs_flat).reset_index(drop=True)

        # The df has an extra column with an empty str as a
        # name. Get rid of that. Try/Except in case it's not
        # there after all:
        try:
            final_res_df.drop('', axis=1, inplace=True)
        except Exception:
            pass
        exp_key = f"significant_acorrs_{self.exp_timestamp}"
        self.experimenter.save(exp_key, final_res_df)
        
        return final_res_df

    #------------------------------------
    # peak_positions_acorrs
    #-------------------
    
    def peak_positions_acorrs(self, df_or_path):
        '''
        Given a acorr_df produced by compute_autocorrelations(), 
        find the peaks in each autocorrelation series. Of
        those there are [num-of-freq-bands] * [num-of-sig-types]
        where sig_types is 'flatness', 'pitch' etc.
        
        Uses scypi.signal.find_peak(), which considers prominence
        relative to other peaks.
        
        Saves and returns a acorr_df that includes:
          'time' spectrogram time when peak occurred, 
          'sig_type'      over which signature values was the autocorrelation
                             computed, 
          'freq_low'      low edge of frequency band
          'freq_high'     high edge of frequency band
          'lag'           lag value for which a row's autocorrelation
                             was computed
          'acorr'         the autocorrelation value 
          'plateau_width' width of the detected plateau 
          'prominence'    prominence of the peak
        
        :param df_or_path: either a acorr_df created by compute_autocorrelations(),
            or the file path to such a acorr_df stored as csv
        :type df_or_path: {pd.DataFrame | str}
        :returns a new acorr_df with peaks computed for every
            freqband/sig_type pair
        :rtype pd.DataFrame
        '''

        if type(df_or_path) == str:
            # Was given a path or experiment key:
            if df_or_path.endswith('.csv'):
                fname = Path(df_or_path).name
                self.log.info(f"Reading autocorrelations from {fname}...")
                acorr_df = pd.read_csv(df_or_path, engine='pyarrow') # pyarrow is a fast csv reader
                self.log.info(f"Done reading autocorrelations from {fname}.")
            else:
                self.log.info(f"Reading autocorrelations from experiment: {df_or_path}...")
                acorr_df = self.experimenter.read(df_or_path, Datatype.tabular)
                self.log.info(f"Done reading autocorrelations from experiment: {df_or_path}.")
            in_fname_timestamp = FileUtils.extract_file_timestamp(df_or_path)
        else:
            # Was given a df:
            acorr_df = df_or_path
            in_fname_timestamp = None

        # Partition the acorr_df by sig_type ('flatness', 'pitch' etc.),
        # and frequency band. The number of acorr_df extracts in the group:
        #    num_sig_types * num_freq_bands
        df_grp = acorr_df.groupby(by=['sig_type', 'freq_low'])
        
        # Get a list of dfs, each containing the original
        # acorr_df rows for one combination sig_type/freq_band
        # Each acorr_df in the arr will have some junk cols from 
        # the group multiindex, but otherwise have the same
        # cols as the original acorr_df. 
        self.log.info(f"Partitioning data into {self.num_freq_bands} bands...")
        fband_sig_type_dfs = [data.reset_index() 
                              for _grp_key, data 
                              in df_grp]
        self.log.info(f"Done partitioning data into {self.num_freq_bands} bands...")
        
        # Remove the junk columns:
        list(map(lambda df: df.drop(['', 'index'], axis=1, inplace=True), 
                 fband_sig_type_dfs))
        
        # For each extract, build acorr_df:
        peaks_df_arr = []
        for df_extract in fband_sig_type_dfs:
            # Isolate the acorr values that are significant,
            # and are not the trivial value of 1 that occurs
            # with lag==0. Note: its index will be fragmented,
            # showing the chosen indexes of df_extract:
            df_extract.acorr = df_extract.acorr.abs()
            # Get the indexes in *extract_acorrs* (not directly
            # those in df_extract) that are peaks:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=PeakPropertyWarning)
                peak_idxs, peak_stats = find_peaks(
                    df_extract.acorr,
                    height=(None, None), 
                    width=(None, None), 
                    prominence=(None, None), 
                    plateau_size=(None,None))

            # To get the indexes in the original df_extract,
            # that are peaks, first get the rows from extract_acorrs
            # that are peaks. Then get the index entries of those
            # rows. Those indexes point into df_extract:
            df_extract_idxs = df_extract.iloc[peak_idxs].index
            # Need the copy, else peaks_df will be a view,
            # and subsequent addition of columns will fail:
            peaks_df = df_extract.iloc[df_extract_idxs].copy()

            # Add some of the find_peaks() stats:
            peaks_df['plateau_width'] = peak_stats['plateau_sizes']
            peaks_df['prominence'] = peak_stats['prominences']

            peaks_df_arr.append(peaks_df)
            
        peaks_df = pd.concat(peaks_df_arr)
        
        if in_fname_timestamp is not None:
            # Match the input acorr_df's timestamp:
            exp_key = f"peaks_acorrs_{in_fname_timestamp}"
        else:
            exp_key = f"peaks_acorrs_{self.exp_timestamp}"
        self.log.info(f"Saving autocorrelation peaks to experiment: {exp_key}")
        self.experimenter.save(exp_key, peaks_df)

        return peaks_df

    #------------------------------------
    # peak_positions_z_scores
    #-------------------
    
    def peak_positions_z_scores(self, df_or_path):
        '''
        Given a z-score computation as produced by 
        compressed_spectrograms(), find the peaks 
        along time in each frequency band, and also
        along freqs in each timeframe.  
        
        Uses scypi.signal.find_peak(), which considers prominence
        relative to other peaks.
        
        Expected like:
			 freq,    time,    z_left,      z_right,      z_centered
			 492.1875 ,0.0,  -0.48808396, -0.48808396,   -0.48808396
			 992.1875 ,0.0,  -0.5756462,  -0.5756462,    -0.5756462
			 1492.1875,0.0,  -0.6076283,  -0.6076283,    -0.6076283
			 1992.1875,0.0,  -0.6267458,  -0.6267458,    -0.6267458
			         
        
        
        Saves and returns a df that includes:
        
        :param df_or_path: either a zscores_df created by compute_autocorrelations(),
            or the file path to such a zscores_df stored as csv
        :type df_or_path: {pd.DataFrame | str}
        :returns a new zscores_df with peaks computed for every
            freqband/sig_type pair
        :rtype pd.DataFrame
        '''

        if type(df_or_path) == str:
            # Was given a path or experiment key:
            if df_or_path.endswith('.csv'):
                fname = Path(df_or_path).name
                self.log.info(f"Reading z_score values from {fname}...")
                zscores_wide = pd.read_csv(df_or_path, engine='pyarrow') # pyarrow is a fast csv reader
                self.log.info(f"Done reading z_score values from {fname}.")
            else:
                # An experiment key:
                self.log.info(f"Reading z_score values from experiment: {df_or_path}...")
                zscores_wide = self.experimenter.read(df_or_path, Datatype.tabular)
                self.log.info(f"Done reading z_score values from experiment: {df_or_path}.")
            in_fname_timestamp = FileUtils.extract_file_timestamp(df_or_path)
        else:
            # Was given a df:
            zscores_wide = df_or_path
            in_fname_timestamp = None
        
        # Now have zscores_df:
        #             freq  time     z_left    z_right  z_centered
        #     0   492.1875   0.0  -0.488084  -0.488084   -0.488084
        #     1   992.1875   0.0  -0.575646  -0.575646   -0.575646
        #     2  1492.1875   0.0  -0.607628  -0.607628   -0.607628
        #
        # Transform to:
        #        freq    time      z_bias    z_score
        #      492.1875   0.0      z_left  -0.488084
        #      992.1875   0.0      z_left  -0.575646
        #            ...
        #      492.1875   0.008    z_left  -0.488084
        #      992.1875   0.008    z_left  -0.575646
        #            ...
        #      492.1875   0.0      z_right  -0.626746
        #      992.1875   0.0      z_right  -0.626904
        #      492.1875   0.008    z_right  -0.488084
        #      992.1875   0.008    z_right  -0.575646
        #                   ...
        #      492.1875  60.0      z_centered   1.112056
        #      992.1875 60.0       z_centered   1.368559

        z_scores = pd.melt(zscores_wide, 
                           id_vars=['freq','time'], 
                           value_vars=['z_left', 'z_right', 'z_centered'], 
                           var_name='z_bias', 
                           value_name='z_score') 

        # Partition the zscores_df by z-score 'bias' (centered, left, or right)
        # and frequency band. The number of z_scores extracted into the group:
        #    num_z_score_weightings * num_freq_bands = 3 * 32
        
        df_grp = z_scores.groupby(by=['freq', 'z_bias'])
        
        # Get a list of dfs, each containing the original
        # zscores_df rows for one combination sig_type/freq_band
        # Each zscores_df in the arr will have some junk cols from 
        # the group multiindex, but otherwise have the same
        # cols as the original zscores_df. 
        self.log.info(f"Partitioning data into {self.num_freq_bands} bands...")
        fband_z_bias_dfs = [data.reset_index() 
                              for _grp_key, data 
                              in df_grp]
        self.log.info(f"Done partitioning data into {self.num_freq_bands} bands...")
        
        # Remove the junk columns:
        list(map(lambda df: df.drop(['index'], axis=1, inplace=True), 
                 fband_z_bias_dfs))
        
        # For each extract, build zscores_df:
        #    peak_idx   measure_name   measure_val  freq_low  freq_high   peak_width  prominence  
        # time
        
        peaks_df_arr = []
        for df_extract in fband_z_bias_dfs:
            # Isolate the acorr values that are significant,
            # and are not the trivial value of 1 that occurs
            # with lag==0. Note: its index will be fragmented,
            # showing the chosen indexes of df_extract:
            df_extract.z_score = df_extract.z_score.abs()
            # Get peaks. The (None,None) key values stand
            # for 'min' and 'max' or the respective key. Must
            # included them to have find_peaks() return corresponding
            # result values:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=PeakPropertyWarning)
                peak_idxs, peak_stats = find_peaks(
                    df_extract.z_score,
                    height=(None, None), 
                    width=(None, None), 
                    prominence=(None, None), 
                    plateau_size=(None,None),
                    threshold=(None, None)
                    )

            # To get the indexes in the original df_extract,
            # that are peaks, first get the rows from df_extract
            # that are peaks. Then get the index entries of those
            # rows. Those indexes point into df_extract:
            df_extract_idxs = df_extract.iloc[peak_idxs].index
            # Need the copy, else peaks_df will be a view,
            # and subsequent addition of columns will fail:
            peaks_df = df_extract.iloc[df_extract_idxs].copy()

            # Add some of the find_peaks() stats:
            peaks_df['plateau_width'] = peak_stats['plateau_sizes']
            peaks_df['prominence']    = peak_stats['prominences']
            peaks_df['peak_height']   = peak_stats['peak_heights']
            peaks_df['left_edge']     = peak_stats['left_edges']
            peaks_df['right_edge']     = peak_stats['right_edges']
            peaks_df['height_above_left_neighbor']  = peak_stats['left_thresholds']
            peaks_df['height_above_right_neighbor'] = peak_stats['right_thresholds']

            peaks_df_arr.append(peaks_df)
            
        peaks_df = pd.concat(peaks_df_arr)
        
        # Create an experiment key:
        if in_fname_timestamp is not None:
            # Match the input zscores_df's timestamp:
            exp_key = f"peaks_zscores_{in_fname_timestamp}"
        else:
            exp_key = f"peaks_zscores_{self.exp_timestamp}"
        self.log.info(f"Saving z-score peaks to experiment: {exp_key}")
        self.experimenter.save(exp_key, peaks_df)

        return peaks_df

    #------------------------------------
    # peak_positions_energy
    #-------------------
    
    def peak_positions_energy(self, df_or_path):
        '''
        Given a z-score computation as produced by 
        compressed_spectrograms(), find the peaks 
        along time in each frequency band, and also
        along freqs in each timeframe.  
        
        Uses scypi.signal.find_peak(), which considers prominence
        relative to other peaks.
        
        Expected like:
			 freq,    time,    z_left,      z_right,      z_centered
			 492.1875 ,0.0,  -0.48808396, -0.48808396,   -0.48808396
			 992.1875 ,0.0,  -0.5756462,  -0.5756462,    -0.5756462
			 1492.1875,0.0,  -0.6076283,  -0.6076283,    -0.6076283
			 1992.1875,0.0,  -0.6267458,  -0.6267458,    -0.6267458
			         
        
        
        Saves and returns a df that includes:
        
        :param df_or_path: either a zscores_df created by compute_autocorrelations(),
            or the file path to such a zscores_df stored as csv
        :type df_or_path: {pd.DataFrame | str}
        :returns a new zscores_df with peaks computed for every
            freqband/sig_type pair
        :rtype pd.DataFrame
        '''
        
        if type(df_or_path) == str:
            # Was given a path or experiment key:
            if df_or_path.endswith('.csv'):
                fname = Path(df_or_path).name
                self.log.info(f"Reading freq/time energy values from {fname}...")
                df_wide = pd.read_csv(df_or_path, engine='pyarrow') # pyarrow is a fast csv reader
                self.log.info(f"Done reading freq/time energy values from {fname}.")
            else:
                # Get 
                #        freq   time   mean_band
                # index
                #   1
                #   2
                self.log.info(f"Reading energy values from experiment: {df_or_path}...")
                df_wide = self.experimenter.read(df_or_path, Datatype.tabular)
                self.log.info(f"Done reading energy values from experiment: {df_or_path}.")
            in_fname_timestamp = FileUtils.extract_file_timestamp(df_or_path)
            
        else:
            # Was given a df:
            df_wide = df_or_path
            in_fname_timestamp = None
        
        # We now have:
        #        freq          time  mean_band
        # index
        #   0      492.1875      0.0   0.029274
        #   1      992.1875      0.0   0.018127
        #   2      ...           ...     ...
        #   ...    492.1875     60.0     ...
        #          992.1875     60.0     ...
        #         13992.1875   60.0   0.024160
        #         14492.1875   60.0   0.021063        

        # Partition the energy values by frequency band:
        # This provides for each freq a list of index values
        # in df_wide that relate to that frequency. Since the
        # index is itself the freqs, that looks like:
        #
        #   {492.1875: [0, 32, 64,...], 992.1875: [1, 33, 65]}
        #
        #    num_freq_bands * num_of_timeframes, something like 32 * 7501
        
        df_grp = df_wide.groupby(by='freq')
        
        # Get a list of dfs, each containing the energy for each
        # time at a single frequency:
        #
        #            index      freq    time  mean_band
        #     0          0  492.1875   0.000   0.029274
        #     1         32  492.1875   0.008   0.039264
        #     2         64  492.1875   0.016   0.051820
        #     ...      ...       ...     ...        ...
        #     7496  239872  492.1875  59.968   0.025690        

        self.log.info(f"Partitioning data into {self.num_freq_bands} bands...")
        fband_energy_dfs = [data.reset_index() 
                            for _grp_key, data 
                            in df_grp]
        self.log.info(f"Done partitioning data into {self.num_freq_bands} bands...")
        
        # Remove the 'index' column:
        list(map(lambda df: df.drop(['index'], axis=1, inplace=True), 
                 fband_energy_dfs))
        
        # For each single-freqency extract of the original df, build: 
        #    peak_idx   measure_name   measure_val  freq   peak_width  prominence  
        # time
        
        peaks_df_arr = []
        for df_extract in fband_energy_dfs:

            # Get peaks. The (None,None) key values stand
            # for 'min' and 'max' or the respective key. Must
            # included them to have find_peaks() return corresponding
            # result values in the stats dict:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=PeakPropertyWarning)
                peak_idxs, peak_stats = find_peaks(
                    df_extract.mean_band,
                    height=(None, None), 
                    width=(None, None), 
                    prominence=(None, None), 
                    plateau_size=(None,None), # all come out as 1
                    threshold=(None, None)
                    )

            # To get the indexes in the original df_extract,
            # that are peaks, first get the rows from df_extract
            # that are peaks. Then get the index entries of those
            # rows. Those indexes point into df_extract:
            df_extract_idxs = df_extract.iloc[peak_idxs].index
            # Need the copy, else peaks_df will be a view,
            # and subsequent addition of columns will fail:
            peaks_df = df_extract.iloc[df_extract_idxs].copy()

            # Add some of the find_peaks() stats:
            peaks_df['prominence']    = peak_stats['prominences']
            peaks_df['height_above_left_neighbor']  = peak_stats['left_thresholds']
            peaks_df['height_above_right_neighbor'] = peak_stats['right_thresholds']

            peaks_df_arr.append(peaks_df)
            
        peaks_df = pd.concat(peaks_df_arr)
        
        # Create an experiment key:
        if in_fname_timestamp is not None:
            # Match the input energy's timestamp:
            exp_key = f"peaks_energy_{in_fname_timestamp}"
        else:
            exp_key = f"peaks_energy_{self.exp_timestamp}"
        self.log.info(f"Saving energy peaks to experiment: {exp_key}")
        self.experimenter.save(exp_key, peaks_df)

        return peaks_df


    #------------------------------------
    # compressed_spectrograms
    #-------------------

    def compressed_spectrograms(self, source_info=None, normalize=False):
        '''
        Creates four dfs with number of rows equal to number of freq bands:
        
           1. a new spectrogram, in which, for example 500Hz
              high channels are combined at each timeframe via mean. 
           2. A second spectro in which the combination is done by median.
           3. A third spectro in which energy values are replaced by the
              distance of the energy from the band's mean across all times,
              i.e. each value's z-scores.

        Exlanation of 1: A new spectrogram from the original audio file's 
        	self.spectro. The new df will combine self.band_height Hz channels into 
        	frequency bands. Example: if self.band_height is 500Hz, then the 
        	original's spectrogram's lowest 500hz will be combined into one row of the new
        	spectrogram. The next 500Hz are combined into the second row, etc.
        
        	New rows will be the MEAN of the original channel energy values. 
        	The shape of the new spectrogram is (num_of_bands, width_of_original)
        
        Explanation of 2: same as 1, but using median of channels to compute 
           the values for each band
        
        Explanation of 3: for each band, compute the mean along all timeframes.
           Each df cell will contain the distance in standard deviations from
           that mean, i.e. the value's z-score: (band - band.mean()) / band.std()

        Writes all three dfs to experiment, using the key of the form:
        
            <filename-of-spectro>_compressed_mean_<exp_timestamp>
            <filename-of-spectro>_compressed_median_<exp_timestamp>
            <filename-of-spectro>_compressed_z_scores_<exp_timestamp>            

        If normalize is True, the dataframe is mean-normalized before any other
        action is taken. Normalization includes *all* df values, as if the
        df was flattened to a series S. Then:
        
                (S - S.mean()) / S.std()

        Assumption: If source_info is None, then self.spectro contains 
             a dataframe of the original spectrogram.
        
        :param source_info: either a path to full spectrogram, or
            experiment key to it. If None, then self.spectro must
            have been initialized.
        :type source_info: {None | str|
        :param normalize: whether or not to normalize the given spectrogram
            before any action
        :type normalize: bool
        :return compressed spectrogram
        :rtyp pd.Dataframe
        '''
        if source_info is None:
            # The self.spectro df must exist:
            try:
                spectro = self.spectro
            except NameError:
                self.log.err("Fatal: source_info not provided, nor self.spectro defined.")
                sys.exit(1)
        else:
            if source_info.endswith('.csv'):
                # Full filename:
                try:
                    spectro = pd.read_csv(source_info, engine='pyarrow')
                except FileNotFoundError:
                    self.log.err(f"Fatal: spectro file not found: {source_info}")
                    sys.exit(1)
            else:
                # Experiment key:
                spectro = self.experimenter.read(source_info, Datatype.tabular)
        
        if normalize:
            spectro = Utils.normalize_df(spectro, axis='all')
        
        # Get the upper freq bands:: [0.0, 492.1875, 992.1875, 1492.1875, ... 15992.1875]
        num_channels, _num_times = spectro.shape
        
        # Get sorted list of new freq band boundaries, like:
        # [0.0, 492.1875, 992.1875, 1492.1875, ...]
        band_upper_bounds = list(reversed(spectro.index[np.arange(0,num_channels,
                                                                  self.num_freq_bands)]))
        
        # Get 2-tuples of (low_band_freq, high_band_freq):
        freq_bounds = list(zip(band_upper_bounds, band_upper_bounds[1:]))

        # MEAN COMPRESSION
                
        # Create, and write to experiment both the mean-,
        # and median-compressed versions: the means/medians are 
        # taken separately over the spectro channels of each freq band:
        
        self.log.info(f"Computing mean-compressed {'(normalized) ' if normalize else ''}spectro...")
        spectro_compressed_mean   = self._band_energy_compression(spectro, 
                                                                  band_upper_bounds, 
                                                                  freq_bounds, 
                                                                  'mean')
        # If spectrogram file is given, derive the first part of
        # the spectrogram key:
        if self.spectro_path.endswith('.csv'):
            spectro_exp_key_root = Path(self.spectro_path).stem
        else:
            spectro_exp_key_root = self.spectro_path # Already is a key:
        
        if normalize:
            exp_compressed_key_mean = f"{spectro_exp_key_root}_normed_compressed_mean_{self.exp_timestamp}"
        else:
            exp_compressed_key_mean = f"{spectro_exp_key_root}_compressed_mean_{self.exp_timestamp}"
        
        
        # The saved version of the dfs are two columns and
        # the index:
        #
        #             time  mean_band
        # 492.1875     0.0   0.210266
        # 992.1875     0.0   0.171045
        # 1492.1875    0.0   0.149459
        #           ...
        #
        # Same for all the following dfs:
        compressed_mean_for_viz = pd.melt(spectro_compressed_mean, 
                                          var_name='time', 
                                          value_name='mean_band', 
                                          ignore_index=False)
         
        self.log.info(f"Writing mean_compressed {'(normalized) ' if normalize else ''}spectro to experiment ({exp_compressed_key_mean})")
        self.experimenter.save(exp_compressed_key_mean, 
                               compressed_mean_for_viz,
                               index_col='freq')

        # MEDIAN COMPRESSION
        self.log.info(f"Computing median-compressed {'(normalized) ' if normalize else ''}spectro...")        
        spectro_compressed_median = self._band_energy_compression(spectro, 
                                                                  band_upper_bounds, 
                                                                  freq_bounds, 
                                                                  'median')
        
        compressed_median_for_viz = pd.melt(spectro_compressed_median, 
                                          var_name='time', 
                                          value_name='mean_band', 
                                          ignore_index=False) 

        if normalize:
            exp_compressed_key_median = f"{spectro_exp_key_root}_normed_compressed_median_{self.exp_timestamp}"
        else:
            exp_compressed_key_median = f"{spectro_exp_key_root}_compressed_median_{self.exp_timestamp}"
        self.log.info(f"Writing median_compressed {'(normalized) ' if normalize else ''}spectro to experiment ({exp_compressed_key_median})")
        # Transpose makes df narrower: 32 (i.e. num_bands) wide, 
        # and one row per time. Makes examination with Tableau easier:
        self.experimenter.save(exp_compressed_key_median, 
                               compressed_median_for_viz,
                               index_col='freq') 

        # DISTANCE FROM MEAN (Z-SCORES)
        
        # This computation is over the already compressed 
        # energy values. So the channels are already bundled
        # into bands via mean.
        # 
        # The computation occurs across one freq band at a time. 
        # Each energy value is replaced by the z-score of the value 
        # relative to the mean over a sliding window whose width is 
        # defined by self.compression_window_width:
        
        self.log.info("Computing distances from mean (i.e. z-score)...")
        # Next: a df with fields being the distance in
        # standard deviations of each energy value from its band's mean
        df_distance_from_mean = self._distances_from_mean(spectro_compressed_mean)

        if normalize:
            exp_compressed_key_zscores = f"{spectro_exp_key_root}_normed_z_scores_{self.exp_timestamp}"
        else:
            exp_compressed_key_zscores = f"{spectro_exp_key_root}_normed_z_scores_{self.exp_timestamp}"
            
        self.log.info(f"Writing z-scores to experiment ({exp_compressed_key_zscores})")
        # Transpose makes df narrower: 32 (i.e. num_bands) wide, 
        # and one row per time. Makes examination with Tableau easier:
        self.experimenter.save(exp_compressed_key_zscores, 
                               df_distance_from_mean,
                               index_col='freq')
        
    #------------------------------------
    # _quartile_membership
    #-------------------
    
    def _quartile_membership(self, df):
        '''
        For each row, replace each value with the name
        of the quartile within the row of which it is a member.
        The quartile names will be 1,2,3,4:
        
        Return a new df with all values replaced
        
        :param df: dataframe of numbers
        :type df: pd.DataFrame
        :return: new df with values replaced by 
            quartile names based on a value's position
            in the row's distribution
        :rtype pd.DataFrame
        '''
        pass

    #------------------------------------
    # _distances_from_mean
    #-------------------
    
    def _distances_from_mean(self, df):
        '''
        For each row, replace each value with its
        distance from the mean over a window around
        the value. This result is computed once for
        the window 'sticking out' more on the left
        of each value, once centered, and once sticking
        out more on the right. 
        
        The left side bias is best to discover starting
        edges of calls. The right side bias is better
        to find endings of call. The centered version will
        tend to have z-score of 0 inside a call.
        
        Returns a df:
        
                        time     z_left    z_right  z_centered
              freq
            492.1875     0.0  -0.937635  -0.937635   -0.937635
            992.1875     0.0  -1.011013  -1.011013   -1.011013
            ...          ...        ...        ...         ...
            13992.1875  60.0   1.092039   1.092039    1.092039
            15992.1875  60.0  11.100916  11.100916   11.100916
            
            [240032 rows x 4 columns]
        
        Rows are one full time series for each frequency:
        
        :param df: dataframe of numbers
        :type df: pd.DataFrame
        :return: new df with values replaced by their 
            distance from the mean in standard deviations.
        :rtype pd.DataFrame
        '''

        for weighted_neighbors in ('left', 'right', None):
            
            self.log.info(f"Computing z scores with {weighted_neighbors} neighbors favored...")
            
            # Func to compute the distance of each 
            # value in a row from the mean of a sliding window
            # around the time of each energy value. The distance
            # is measured in (fractional) standard deviations.
            # Returns a new Series with the z-scores:
            z_scores_func = partial(self._z_scores, 
                                    win=self.compression_window_width,
                                    weighted_neighbors=weighted_neighbors)
            
            # Apply the func to each row,
            # building a new df:
            if weighted_neighbors == 'left':
                z_scores_left = df.apply(z_scores_func, axis=1)
            elif weighted_neighbors == 'right':
                z_scores_right = df.apply(z_scores_func, axis=1)
            else:
                z_scores_centered = df.apply(z_scores_func, axis=1)

        z_left_melted = pd.melt(z_scores_left,
                        var_name='time',
                        value_name='z_left',
                        ignore_index=False)
        
        z_centered_melted = pd.melt(z_scores_centered,
                                    var_name='time',
                                    value_name='z_centered',
                                    ignore_index=False)
        
        z_right_melted = pd.melt(z_scores_right,
                                 var_name='time',
                                 value_name='z_right',
                                 ignore_index=False)

        # Combine scores into on df:
        z_scores = z_left_melted
        z_scores['z_right'] = z_right_melted.z_right
        z_scores['z_centered'] = z_centered_melted.z_centered

        z_scores.index.name = 'freq'
        
        return z_scores

    #------------------------------------
    # _z_scores
    #-------------------
    
    def _z_scores(self, row, win, weighted_neighbors):
        '''
        Given a Series of values, and a window width 'win',
        successively place the window's weighted_neighbors over each value
        in turn. Compute the mean and std just over the values 
        within that window, and use the results to compute the 
        value's z-score. Return  series as long as row, but with
        each value being the z-score (distance from mean in standard
        units) of the original value.
        
        The weighted_neighbors is either 'left', 'center' or 'right'. 
        If weighted_neighbors is 'left, the window will be placed on
        each element 2/3 into the window. If set to 'right', placement
        will be 1/3 into the window. If 'None', window is placed centered.
        
               |--------------|
               |  el          | weighted_neighbors == 'right'
               |--------------|

               |--------------|
               |      el      | weighted_neighbors == None
               |--------------|

               |--------------|
               |          el  | weighted_neighbors == 'left'
               |--------------|
        
        
        A left weighted_neighbors will weight row elements to the left
        of the element more heavily than elements on the reight.
        Vice versa for right weighted_neighbors.
        
        A win of None uses the mean/std of the entire row
        to compute the z-score, and weighted_neighbors is ignored
        
        :param row: a sequence of values
        :type row: pd.Series
        :param win: width of window over which the z-score
             means/stds are computed
        :type win: int
        :param weighted_neighbors: whether to favor neighbors on the left
            or right more heavily for each element than other neighbors
        :type weighted_neighbors: {None | 'left' | 'right'
        :return a new Series in which values are replaced
            by their windowed z-scores.
        :rtype pd.Series
        '''
        
        if win is None:
            row_mean = row.mean()
            row_std  = row.std()
            z_values = (row - row_mean) / row_std
            return z_values
        

        if weighted_neighbors is None:
            # Half the window will be on the left, the
            # other half on the right of a value whose
            # z score is being computed
            win_left  = int(win//2)
        elif weighted_neighbors == 'left':
            win_left  = 2 * int(win // 3)
        elif weighted_neighbors == 'right':
            win_left  = int(win // 3)
        else:
            raise ValueError(f"Arg weighted_neighbor must be None, 'left', or 'right', not {weighted_neighbors}")

        # Right 
        win_right = int(win) - win_left
        
        # In the map() below, the iterable has to 
        # start width of the left window side into 
        # the window in, because else the
        # left portion of the window will have negative
        # values for indexing into the row:
        win_idxs = np.arange(win_left, len(row) - win_right)
         
        # Get the means for all row values, except
        # 1/2 window size worth of values on the left.
        # The '+1' includes the right edge of the window:
        means_center = pd.Series(map(lambda i, row=row, win_left=win_left, win_right=win_right : 
                                     row.iloc[i-win_left : i+win_right+1].mean(), 
                                     win_idxs))
        # Fill the means for the left-most values of the row
        # that we skipped with the first mean that we did
        # compute:
        means_leader  = pd.Series([means_center.iloc[0]]*win_left)
        means_tail    = pd.Series([means_center.iloc[-1]]*win_right)
        sliding_means = pd.concat([means_leader, means_center, means_tail])

        # Set index to be same as what row has:
        sliding_means.set_axis(row.index, inplace=True)

        # Do the same for the standard deviation that we 
        # just did for the mean:
        stds_center  = pd.Series(map(lambda i, row=row, win_left=win_left, win_right=win_right : 
                                     row.iloc[i-win_left : i+win_right+1].std(), 
                                     win_idxs))
        stds_leader  = pd.Series([stds_center.iloc[0]]*win_left)
        stds_tail    = pd.Series([stds_center.iloc[-1]]*win_right)
        sliding_stds = pd.concat([stds_leader, stds_center, stds_tail])
        sliding_stds.set_axis(row.index, inplace=True)
        
        # Finally: the z-scores for each member of the row:
        z_values = (row - sliding_means) / sliding_stds
        return z_values

    #------------------------------------
    # _band_energy_compression
    #-------------------

    def _band_energy_compression(self, 
                                 spectro, 
                                 band_upper_bounds, 
                                 freq_bounds, 
                                 compression_method):

        # For each freq band, for each timeframe get the energy
        # in the spectrogram across the band's constituent freqs:
        # This will be a list of Series. Each series will be as
        # long as the spectro is wide, there will be as
        # many Series as there are freq bands:
    
        if compression_method == 'mean':
            band_energy_compressed = list(map(lambda freqs_low_high, self=self:
                                              spectro.loc[np.logical_and(
                                                  spectro.index >= freqs_low_high[0], 
                                                  spectro.index >= freqs_low_high[1])].mean(), 
                                              freq_bounds))
        elif compression_method == 'median':
            band_energy_compressed = list(map(lambda freqs_low_high, self=self:
                                              spectro.loc[np.logical_and(
                                                  spectro.index >= freqs_low_high[0], 
                                                  spectro.index >= freqs_low_high[1])].median(), 
                                              freq_bounds))
        else:
            raise NotImplementedError(f"Compression method '{compression_method}' not implemented")

        band_dict = {upper_freq:compression_series for upper_freq, compression_series in 
                     zip(band_upper_bounds, band_energy_compressed)}
            
        # Build a compressed spectrogram df: index will be the upper
        # freqs of the freq bands (therefore the [1:], which excludes
        # the initial 0.0):
        #
        #               t1        t2    ... 16K
        #
        # bound1    mean_1_1  mean_1_2, ...
        # bound2
        #   ...
        # bound32
        spectro_compressed = pd.DataFrame(band_dict.values(), 
            index=band_upper_bounds[1:], 
            columns=spectro.columns)
        return spectro_compressed


    #------------------------------------
    # __getstate__ and __setstate__
    #-------------------
    
    def __getstate__(self):
        '''
        Called when this object is pickled. Returns a dict
        with the instance attributes that are to be saved.
        
        :returns attributes and their values to be saved
        :rtype dict
        '''
        return {'settings' : self.settings.json_dumps(),
                'round_to' : self.round_to,
                'lag_duration' : self.lag_duration,
                'num_lags' : self.num_lags,
                'spectro_times' : self.spectro_times,
                'sel_tbl' : self.sel_tbl,
                'selections_df' : self.selections_df,
                'audio_path' : self.audio_path,
                'spectro_path' : self.spectro_path,
                'experimenter_root' : self.experimenter.root
                }
        
    def __setstate__(self, state):
        '''
        Called during unpickling. The state parameter is
        the dict that was saved with instructions of __getstate__()
        
        :param state: attributes to initialize in the initially
            empty self instance
        :type state: dict
        '''
        # Initialize instance vars
        
        self.__dict__.update(state)
        self.log = LoggingService()
        # Create an ExperimentManager, but prevent it from
        # modifying its experiment.json file. Else the multiple
        # processes will mix up the content:
        self.experimenter = ExperimentManager(state['experimenter_root'], freeze_state_file=True)
        self.settings = SoundSegmentationSetting.json_loads(state['settings'])

    #------------------------------------
    # _process_one_freq_slice
    #-------------------
    
    def _process_one_freq_slice(self, spectro_freq_slice, col_names, lookback_duration):
        
        final_res_arr = []

        exp_key = f"slice_sig_{spectro_freq_slice.low_freq}_{spectro_freq_slice.high_freq}"
        try:
            slice_sig = self.experimenter.read(exp_key, Signature)
        except FileNotFoundError:
            slice_sig = self.compute_sigs(spectro_freq_slice)
            self.experimenter.save(exp_key, slice_sig)
            
        spectro_freq_slice.sig = slice_sig
        
        freq_low  = spectro_freq_slice.low_freq
        freq_high = spectro_freq_slice.high_freq
        sig_df    = spectro_freq_slice.sig.sig
        
        self.log.info(f"Starting frequency band {round(freq_low,2)}-{round(freq_high,2)}")
        # Treat each signature measure separately: one requested
        # df col after the other:
        for col_name in col_names:
            
            #self.log.info(f"Autocorrelation for {col_name}")
            one_measure_res = self._one_measure_acorr_significance(sig_df[col_name],
                                                                   lookback_duration)
            # Add cols for the current frequency low and high:
            one_measure_res.insert(2, 
                                      'freq_low', 
                                      pd.Series([freq_low]*len(one_measure_res), index=one_measure_res.index))
            one_measure_res.insert(3, 
                                      'freq_high', 
                                      pd.Series([freq_high]*len(one_measure_res), index=one_measure_res.index))
            
            final_res_arr.append(one_measure_res)

        # Return array of dataframes:
        return final_res_arr

    #------------------------------------
    # _one_measure_acorr_significance
    #-------------------
    
    def _one_measure_acorr_significance(self, sig_measure, lookback_duration):
        '''
        Return df with cols:
        
          'sig_type', 'time', 'lag', 'acorr', 'ci_low', 'ci_high'
           
        The number of lags that correspond to lookback_duration are computed
        from the spectrogram timeframe width. The autocorrelation is computed
        repeatedly every lookback_duration seconds along the measure value
        time series.
        
            Example o flatness measure values, assuming a measure
                      every second:
                      1  2  3  4  5  6  7
                    o lookback_duration: 2 seconds
                    o assume lags that corrend to 2 seconds is 2
                    
          Step1      1     2     3     4     5     6     7
                 acorr(2)
                 Count number of significant autocorrelations 
                 
          Step2      1     2     3     4     5     6     7
                               acorr(2)
                               Count number of significant autocorrelations 
                 
          Step3      1     2     3     4     5     6     7
                                           acorr(2)
                                           Count number of significant autocorrelations 
                                           
                          ...
                          
        Result is the series of counts.
        
        The sig_measure is a time series, in this context one of the quad sigs measures: 
        flatness or pitch, or, etc. ...
        
        An autocorrelation with the computed lag is performed repeatedly, starting 
        every lookback_duration seconds along the sig_measure Series. Each time the
        number of autocorrelation coefficients that are statistically significant
        are counted and noted. 
        
        :param sig_measure: time series along which autocorrelations
            are to be performed
        :type sig_measure: pd.Series
        :param lookback_duration: number of seconds up to which to
            slide the time series.
        :type lookback_duration:
        '''
        
        # Build as array:
        #            RowName            NumSignificantAcorrs
        #   [[0.0230_434.12Hz_640.34Hz,    4],
        #    [0.0460_434.12Hz_640.34Hz,    2],
        #         ...
        #    ]
        res_arr = []
        
        stop_time = max(sig_measure.index)
        
        self.log.info(f"Autocorrelations for measure {sig_measure.name}: [0,{stop_time}] sec")

        # Do autocorrelation for every timeframe:
        #for spectro_start_time in sig_measure.index[::self.num_lags]:
        for spectro_start_time in sig_measure.index:
            
            # Start autocorrelation at current start time,
            # and compute it down twice the lookback_duration
            # elements. To find the precise time that is about
            # lookback_duration seconds beyond spectro_start_time,
            # consider that the sig_measure.index are the times. We
            # find the closes index value to spectro_start_time
            # plus lookback_duration. We find all the indexes that are
            # greater than that value, and grab the first:
            
            try:
                spectro_end_time = sig_measure[sig_measure.index >= float(spectro_start_time) +
                                               lookback_duration].index[0]
            except IndexError:
                # Went through the whole series:
                break
            
            measures = sig_measure.loc[spectro_start_time:spectro_end_time]
            
            # Get df w/ cols like 'flatness', 'is_significant':
            acorr_res = SignalAnalyzer.autocorrelation(measures, nlags=self.num_lags)
            
            # Ignore non-significant acorrs, and remove the is_significant col:
            acorr_res_sig = acorr_res[acorr_res.is_significant].drop('is_significant', axis=1)

            # Append the new results, but removing 
            # autocorrelations of lag 0, which is trivially 1.0:
            res_arr.append(acorr_res_sig[acorr_res_sig['lag'] > 0])

        res_df = pd.concat(res_arr)
        res_df.columns = ['time', 'lag', 'acorr', 'ci_low', 'ci_high']
        res_df.index = res_df.time

        # Add a signature type column ('flatness' or 'pitch', etc.)
        res_df.insert(1, 'sig_type', [sig_measure.name]*len(res_df)) 

        return res_df

    #------------------------------------
    # make_test_spectro
    #-------------------
    
    def make_test_spectro(self, 
                          audio_path, 
                          sr, 
                          dest_spectro_key=None,
                          dest_audio_key=None, 
                          do_noisereduce=False):
        '''
        Create a spectrogram from an adio file. Sampling rate
        is controlled by the 'sr' parameter. If do_noisereduce 
        is True, then noise reduction will first be applied to
        the audio.
        
        Returns the spectrogram.
        
        If dest_spectro_key is a string, writes spectro to the experiment
        under that key.
        
        If do_noisereduce is True, and dest_audio_key is a string,
        the noise-reduced audio is also saved to the experiment. 
        
        :param audio_path: path to audio file
        :type audio_path: str
        :param dest_spectro_key: key under which new spectro is to
            be saved in the experiment.
        :type dest_spectro_key: str
        :param dest_audio_key: key under which the noise-reduced
            audio is to be saved in the experiment. Ignored if
            do_noisereduce is False
        :type dest_spectro_key: str
        :param sr: desired sampling rate 
        :type sr: int
        :param do_noisereduce: whether or not to noise-reduce
        :type do_noisereduce: bool
        :return spectrogram
        :rtype df
        '''
        
        self.log.info(f"Creating spectrogram for {audio_path}")
        
        # The raven_spectrogram() method returns a spectrogram
        # df AND the noise-reduced audio if noise reduction is
        # requested. Else it only returns the spectro:
        if do_noisereduce:
            spectro, audio = SignalAnalyzer.raven_spectrogram(audio_path, 
                                                              to_db=False,
                                                              sr=sr,
                                                              do_noisereduce=True,
                                                              extra_granularity=True)
        else:
            spectro = SignalAnalyzer.raven_spectrogram(audio_path, 
                                                       to_db=False,
                                                       sr=sr,
                                                       do_noisereduce=True,
                                                       extra_granularity=True)

        if dest_spectro_key is not None:
            log_msg = f"Saving {'noise reduced ' if do_noisereduce else ''}spectrogram to {dest_spectro_key}..." 
            self.log.info(log_msg)
            self.experimenter.save(dest_spectro_key, spectro, index_col='freq')
            self.log.info(f"Done: {log_msg}")
        
            # Get the actual path to the just-saved spectro,
            # and init self.spectro_path:
            
            self.spectro_path = self.experimenter.abspath(dest_spectro_key, Datatype.tabular)

        # If noisereduction, also save the noise-reduced audio:
        
        if do_noisereduce and type(dest_audio_key) == str:
            log_msg = f"Saving noise-reduced audio to {dest_audio_key}..." 
            self.log.info(log_msg)
            self.experimenter.save(dest_audio_key, audio)
            self.log.info(f"Done: {log_msg}")
            
            

        return spectro

    #------------------------------------
    # _process_selection_tbl
    #-------------------
    
    def _process_selection_tbl(self, 
                               spectro_times,
                               selection_tbl_path,
                               remove_long_selections=False
                               ):
        '''
        Imports a Raven selection table and the time frames
        in seconds of a spectrogram. Creates df with information
        about each selection (i.e. rectangle in Raven). The 
        df has shape (<spectrogram_width>, 8), the columns describing
        an aspect of the selection:
           
            snapped_start_time start time snapped to spectrogram timeframe
            snapped_stop_time  stop time snapped to spectrogram timeframe
            sel_id             Raven selection ID
            sel_dur            duration of selection in fractional seconds 
            is_sel_start       whether or not the respective time frame is a selection start
            is_sel_stop        whether or not the respective time frame is a selection end
            snapped_freq_low   low frequency bound snapped to spectrogram freq bands
            snapped_freq_high  high frequency bound snapped to spectrogram freq bands

        The dataframe is returned, but also saved to the experiment
        under name 
              'selections_info{rec_id}', 
        where recording ID is parsed from the selection table file path 
        if possible. Parser looks for the 'AM<recorderID>_<date><time>' 
        pattern in this example:
        
             /foo/bar/DS_AM02_20190717_052958.Table.1.selections.txt

        HOWEVER: some of the code depends on the selection table
                 being a RavenSelectionTable instance. So the saved
                 selection table df is of limited use. 
        
        :param spectro_times: times in seconds of spectrogram
            time frames
        :type spectro_times: [float]
        :param selection_tbl_path: path to a Raven selection table.
        :type selection_tbl_path: str
        :param remove_long_selections: remove selections that span
            95% or more of the entire spectrogram
        :type remove_long_selections: bool
        :return for each time, whether or not it is the start or
            end of a Raven selection.
        :rtype pd.DataFrame
        '''

        self.log.info("Reading selection table from file...")
        self.sel_tbl = RavenSelectionTable(selection_tbl_path)
        
        # By default each entry's freq_interval has 
        # a standard step size, near-0 or 1. Correct
        # those, because at this point we know the freq
        # stepsize:
        
        for entry in self.sel_tbl.entries:
            entry.freq_interval['step'] = self.channel_steps

        # If requested, only keep selections that are less
        # than 95% of the total spectrogram in length:
        if remove_long_selections:
            removal_threshold = spectro_times.max() * 95/100
            sel_entries = list(filter(lambda entry, removal_threshold=removal_threshold: 
                                      entry.stop_time - entry.start_time < removal_threshold, 
                                      self.sel_tbl.entries))
        else:
            sel_entries = self.sel_tbl.entries
        
        # Get dicts of sel_id to labeled start/stop times from selection table,
        # rounded to our standard ROUND_TO decimal places:
        
        labeled_start_times = {sel_entry.sel_id : np.round(sel_entry.start_time, 
                                                           self.round_to)
                                                           for sel_entry
                                                           in sel_entries}
        labeled_stop_times  = {sel_entry.sel_id : np.round(sel_entry.stop_time,
                                                           self.round_to)
                                                           for sel_entry
                                                           in sel_entries}
        sel_durs  = {sel_entry.sel_id : sel_entry.delta_time
                     for sel_entry
                     in sel_entries}


        # Get dict: {sel_id : (snapped_start_time, snapped_stop_time)}:
        snapped_times = {}
        for sel_id, labeled_start_time, labeled_stop_time \
            in zip(labeled_start_times.keys(),
                   labeled_start_times.values(),
                   labeled_stop_times.values()):
            # For this selection table start time,
            # find the nearest spectrogram time frame:
            snapped_start_time = Utils.nearest_in_array(spectro_times,
                                                        labeled_start_time,
                                                        is_sorted=True)
            snapped_stop_time  = Utils.nearest_in_array(spectro_times,
                                                        labeled_stop_time,
                                                        is_sorted=True)
            snapped_times[sel_id] = (snapped_start_time, snapped_stop_time)

        # Analogously for frequency ranges:
        
        labeled_freqs = {sel_entry.sel_id : (sel_entry.low_freq, sel_entry.high_freq)
                                 for sel_entry
                                 in sel_entries}
        snapped_freqs = {}
        for sel_id, (labeled_freq_low, labeled_freq_high) \
            in zip(labeled_freqs.keys(),
                   labeled_freqs.values()
                   ):
            # For this selection table's frequency bounds,
            # find the nearest spectrogram freq band:
            snapped_freq_low = Utils.nearest_in_array(self.spectro_freqs,
                                                      labeled_freq_low,
                                                      is_sorted=False)
            snapped_freq_high  = Utils.nearest_in_array(self.spectro_freqs,
                                                        labeled_freq_high,
                                                        is_sorted=False)
            snapped_freqs[sel_id] = (snapped_freq_low, snapped_freq_high)
        
        snapped_times_df = pd.DataFrame(snapped_times).transpose()
        snapped_times_df.columns = ['snapped_start_time', 'snapped_stop_time']
        snapped_times_df['sel_id'] = snapped_times_df.index
        
        sel_durs_df = pd.DataFrame(sel_durs.values(), 
                                   index=sel_durs.keys(),
                                   columns=['sel_dur'])
        # We'll set the following two cols properly
        # further down:
        sel_durs_df['is_sel_start'] = False
        sel_durs_df['is_sel_stop'] = False
        sel_durs_df['sel_id'] = sel_durs_df.index
        snapped_times_durs_df = snapped_times_df.merge(sel_durs_df, on='sel_id')
        
        snapped_freqs_df = pd.DataFrame(snapped_freqs).transpose()
        snapped_freqs_df.columns = ['snapped_freq_low', 'snapped_freq_high']
        snapped_freqs_df['sel_id'] = snapped_freqs_df.index
        
        snapped_time_freqs = snapped_times_durs_df.merge(snapped_freqs_df, on='sel_id')
        snapped_time_freqs.index = snapped_time_freqs.snapped_start_time
        
        # Prepare a df: 
        #      sel_id   snapped_start_time  snapped_stop_time snapped_freq_low snapped_freq_high
        # time
        #
        # The df will have width spectrogram:
        exploded_cols = list(map(lambda _idx, the_len=len(self.spectro_times): 
                                 pd.Series([np.nan]*the_len, dtype=float), 
                                 np.arange(len(snapped_time_freqs.columns))))
        exploded_time_freqs = pd.concat(exploded_cols, axis=1)
        exploded_time_freqs.columns = snapped_time_freqs.columns
        exploded_time_freqs.index = spectro_times
        # Set is_sel_start and is_sel_stop to False to start with:
        exploded_time_freqs.loc[exploded_time_freqs.index, 'is_sel_start'] = False
        exploded_time_freqs.loc[exploded_time_freqs.index, 'is_sel_stop'] = False
        
        # Fill in the 'border' times for the selections,
        # i.e. freq bounds at the time bounds:
        exploded_time_freqs.loc[snapped_time_freqs.snapped_start_time] = snapped_time_freqs
        
        # Fill in the (time) rows that are between the bounds.
        # Ex: if row 2.0 sec and 2.5 seconds have high and low freqs
        #     filled in because they are the start and stop of a selection, 
        #     then fill the rows between 2.0 and 2.5 with duplicates of
        #     the row at 2.0.
        for start_time, stop_time in zip(snapped_time_freqs['snapped_start_time'].to_list(),
                                         snapped_time_freqs['snapped_stop_time'].to_list()):
            start_row = snapped_time_freqs[np.logical_and(snapped_time_freqs.snapped_start_time == start_time,
                                                          snapped_time_freqs.snapped_stop_time == stop_time)].values
            exploded_time_freqs.loc[np.logical_and(exploded_time_freqs.index >= start_time,
                                                   exploded_time_freqs.index <= stop_time)] = start_row

        # Finally, mark the selection start and stop times in
        # the is_sel_start and is_sel_stop cols:
        
        exploded_time_freqs.loc[snapped_time_freqs['snapped_start_time'], 'is_sel_start'] = True
        exploded_time_freqs.loc[snapped_time_freqs['snapped_stop_time'], 'is_sel_stop'] = True
        
        # Try to get the recording identifier from the Raven
        # selection table name, which for us looks like:
        rec_id = FileUtils.extract_audiomoth_id(selection_tbl_path)

        exp_key = f"selections_info{rec_id}"
        self.experimenter.save(exp_key, exploded_time_freqs)

        return exploded_time_freqs


# ------------------------ Class SpectroSlice ----------

class SpectroSlice:
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, low_freq, high_freq, slice_height, spectro, sig=None):
        
        self.name = f"{low_freq}_to_{high_freq}"
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.slice_height = slice_height
        self.spectro = spectro
        self.sig = sig
        
    #------------------------------------
    # __repr__
    #-------------------

    def __str__(self):
        return f"<Spectro Slice {round(self.low_freq,2)}Hz to {round(self.high_freq, 2)}Hz {hex(id(self))}>"

    #------------------------------------
    # __repr__
    #-------------------

    def __repr__(self):
        return self.__str__()

# -------------------------- Class SoundSegmentationSettings --------------

class SoundSegmentationSetting(JsonDumpableMixin):
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self,
                 sig_types=['flatness', 'continuity', 'pitch', 'freq_mod', 'energy_sum'],
                 sr=22050, # sampling rate
                 pattern_lookback_dur=2,
                 pattern_lookback_dur_units=PatternLookbackDurationUnits.LAGS,
                 freq_split=(0,20,500), # 0KHz-20KHz in 500Hz increments
                 spectro_path=None,
                 audio_path=None,
                 selection_tbl_path=None,
                 recording_id=None,
                 round_to=5,
                 remove_long_selections=False,
                 compression_window_width=2, # seconds
                 do_noisereduce=False
                 ):
        '''
        Holds parameters for a soundscape segmentation run.
        Lookback duration may be specified either in fractional
        seconds, or in a number of lags. The pattern_lookback_dur_units
        parameter must specify which it is. The spectrogram time frame
        times are used to convert from one to the other.
        NOTE: the SegmentationPreparer's __init__() adds two instances vars:
              lookback_duration, and num_lags that disambiguate.
        
        Either spectrogram path or audio path may be specified. If
        spectrogram path is provided, it is used, and the audio path
        is ignored. Else a spectrogram is created from the audio.
        
        If a Raven selection is provided, its selection start and end
        times will be available in the final result dataframe.  
        
        The recording_id is an optional string that identifies the soundscape
        recording indpendent of file location. Ex: AM02_20190717_052958
        If provided it is used when creating file names. 
        
        :param sig_types: list of signatures to compute ('continuity', 'pitch', etc)
        :type sig_types: [str]
        :param sr: desired sampling rate
        :type sr: {float | int} 
        :param pattern_lookback_dur: number of fractional seconds, or
            number of lags to use for autocorrelation
        :type pattern_lookback_dur: {float | int}
        :param pattern_lookback_dur_units: whether pattern_lookback_dur is
            in units of seconds or of lags
        :type pattern_lookback_dur_units: PatternLookbackDurationUnits
        :param freq_split: low/high/step of how spectrogram is partitioned
            into frequency bands. low and high are in KHz, step is in Hz
        :type freq_split: (number, number, number)
        :param spectro_path: path to an already created spectrogram 
        :type spectro_path: {str | None}
        :param audio_path: path to a soundscape audio file
        :type audio_path: {str | None}
        :param selection_tbl_path: path to Raven selection table, if one
            is available
        :type selection_tbl_path: {str | None}
        :param recording_id: optional string that identifies the soundscape
            recording indpendent of file location.
        :type recording_id: {None | str}
        :param round_to: number of decimal places to which all time 
            measurements are to be rounded. 
        :type round_to: int
        :param compression_window_width: width in seconds of sliding window
            around each energy value when the value's z-score is computed.
        :type compression_window_width: {None | int}
        :param do_noisereduce: whether or not to apply noise reduction
            to audio file, if a new spectrogram is to be created.
        :type do_noisereduce: bool
        '''
        
        # If no recording_id is provided, but an audio_path
        # was passed in, try to extract the recording ID
        # from the filename:
        
        if recording_id is None and audio_path is not None:
            # Could still be None, if fname is non-standard;
            # that's fine, at least we tried:
            recording_id = FileUtils.extract_audiomoth_id(audio_path)
        
        self.sig_types = sig_types
        self.sample_rate = sr
        self.pattern_lookback_dur = pattern_lookback_dur
        self.pattern_lookback_dur_units = pattern_lookback_dur_units
        self.freq_split = freq_split
        self.spectro_path = spectro_path
        # The experiment key for spectrogram: file name without .csv:
        self.spectro_key  = Path(spectro_path).stem if self.spectro_path is not None else None
        self.audio_path = audio_path
        self.selection_tbl_path = selection_tbl_path
        self.recording_id = recording_id
        self.round_to = round_to
        self.remove_long_selections = remove_long_selections
        self.compression_window_width = compression_window_width
        self.do_noisereduce = do_noisereduce

    #------------------------------------
    # json_dumps
    #-------------------
    
    def json_dumps(self):
        
        # Create a dict with the instance vars.
        return json.dumps(self.__dict__)
    
    #------------------------------------
    # json_loads
    #-------------------
    
    @classmethod
    def json_loads(cls, jstr):
        '''
        Given a json string created by
        json_dumps(), materialize a SoundSegmentationSetting
        instance filled with the proper instance
        var values.
        
        :param jstr: json string created via json_dumps()
        :type jstr: str
        :return: new instance of SoundSegmentationSetting
        :rtype SoundSegmentationSetting
        '''
        
        as_dict = json.loads(jstr)
        # Recover the PatternLookbackDurationUnits instance:
        dur_units = as_dict['pattern_lookback_dur_units']
        # Create a fresh PatternLookbackDurationUnits instance:
        as_dict['pattern_lookback_dur_units'] = PatternLookbackDurationUnits.from_value(dur_units)
        res = SoundSegmentationSetting(
            as_dict['sig_types'],
            as_dict['sample_rate'],
            as_dict['pattern_lookback_dur'],
            as_dict['pattern_lookback_dur_units'],
            as_dict['freq_split'],
            as_dict['spectro_path'],
            as_dict['audio_path'],
            as_dict['selection_tbl_path'],
            as_dict['recording_id'],
            as_dict['round_to'],
            as_dict['remove_long_selections']
            )
        return res

    #------------------------------------
    # json_dump 
    #-------------------
    
    def json_dump(self, fpath):
        '''
        Render self onto disk, json encoded.
        Recoverable via SoundSegmentationSetting.json_load()
        
        :param fpath: destination path
        :type fpath: str
        '''
        with open(fpath, 'w') as fd:
            fd.write(self.json_dumps())
            
    #------------------------------------
    # json_load 
    #-------------------

    @classmethod
    def json_load(cls, fpath):
        '''
        Return a SoundSegmentationSetting instance materialized
        from a previously stored, json encoded file.
        This method is the inverse of json_dump() 

        :param fpath: file with json-rendered SoundSegmentationSetting instance
        :type fpath: str
        :return new SoundSegmentationSetting instance
        :rtype SoundSegmentationSetting
        '''
        with open(fpath, 'r') as fd:
            jstr = fd.read()
        return cls.json_loads(jstr)
    
    #------------------------------------
    # toJson
    #-------------------
    
    def toJson(self):
        return self.json_dumps()
    
    #------------------------------------
    # __getstate__ and __setstate__
    #-------------------
    
    def __getstate__(self):
        '''
        Called when this object is pickled. Returns a dict
        with the instance attributes that are to be saved.
        
        :returns attributes and their values to be saved
        :rtype dict
        '''
        return self.__dict__
        
    def __setstate__(self, state):
        '''
        Called during unpickling. The state parameter is
        the dict that was saved with instructions of __getstate__()
        
        :param state: attributes to initialize in the initially
            empty self instance
        :type state: dict
        '''
        # Initialize instance vars
        
        self.__dict__.update(state)
        # The PatternLookbackDurationUnits is still
        # in json form. Turn into an instance of the
        # Enum:
        self.pattern_lookback_dur_units = PatternLookbackDurationUnits.from_value(state['pattern_lookback_dur_units'])


# ------------------------ Main ------------
if __name__ == '__main__':
    
    def expand_path_to_data_dir(source_info, experimenter):
        '''
        Checks whether the given string has a file extension,
        such as .csv, .txt, etc. If not, assume source_info is
        an experiment key. Check the corresponding file's existence.
        If not there, raise FileNotFoundError. Else return the
        key as it was passed in.
        
        Checks whether given string is an absolute
        path with a file extension. If so, checks 
        existence, and raises FileNotFound if not 
        found. If the absolute path exists, it is returned.
        
        If the path is not absolute (but does have and extension), 
        construct a path to the project root's 'data' subdirectory,
        and see whether the given relative file is there. If so,
        return the full path, else raise FileNotFound.
        
        :param source_info: absolute file name, or fname in
            <proj-root>/data
        :type source_info: str
        :return absolute path
        :rtype src
        :raise FileNotFoundErr 
        '''
        if source_info is None:
            return None
        if os.path.isabs(source_info):
            if os.path.exists(source_info):
                return source_info
            else:
                raise FileNotFoundError(f"File {source_info} not found")

        # Is either a file relative to <proj-root>/data,
        # or an experiment key:
        if Path(source_info).suffix in ('.csv', '.txt', '.wav'):
            # It's a file name:
            path = os.path.join(PROJ_ROOT, f"data/{source_info}")
            if os.path.exists(path):
                return path
            else:
                raise FileNotFoundError(f"Given relative path ({source_info}), but file {path} does not exist")
        else:
            # It's an experimenter key:
            path = experimenter.abspath(source_info, Datatype.tabular)
            if os.path.exists(path):
                # Return the key:
                return source_info
            else:
                raise FileNotFoundError(f"Given experiment key ({source_info}), but file {path} does not exist")

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Compute signature autocorrelations and their peaks"
                                     )

    parser.add_argument('-c', '--command',
                        choices=['acorrs', 'peaks', 'spectro_compression'],
                        required=True,
                        help='Command to perform',
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
                        choices=['acorrs', 'z_scores', 'energy'],
                        help='For peaks only: find peaks in autocorrelations, z-scores, or energy',
                        default=None
                        )
    
    parser.add_argument('source_info',
                        type=str,
                        help=('For acorrs: fname of audio or spectrogram;\n'
                              'or spectrogram experiment key;\n'
                              'For spectro_compression fname to spectrogram; \n'
                              'For peaks, either full fname to acorr result, or experiment manager key'
                              ),
                        )

    args = parser.parse_args()

    #exp_root = os.path.join(EXP_ROOT, 'ExpAM02_20190717_052958_HalfSecondLags')
    #exp_root = os.path.join(EXP_ROOT, 'ExpAM02_20190717_052958_AllData1')
    exp_root = os.path.join(EXP_ROOT, 'ExpAM02_20190717_052958_NoiseReduced')
    exp = ExperimentManager(exp_root)
    
    # Process selection table argument:
    if args.selection_tbl is not None:
        selection_tbl = expand_path_to_data_dir(args.selection_tbl, exp)
    else:
        selection_tbl = None
    
    cmd = args.command
    
    # Determine meaning of filename:
    if cmd == 'acorrs':
        # Filenam is either an audio file, or a 
        # precomputed spectrogram file, which would
        # be a csv file:
        if args.source_info.endswith('.csv'):
            # A spectrogram:
            spectro_path = expand_path_to_data_dir(args.source_info, exp)
            audio_path   = None 
        else:
            audio_path   = expand_path_to_data_dir(args.source_info, exp)
            spectro_path = None

    elif cmd == 'peaks':
        
        # Are the required options set?
        if args.measures is None or args.spectrogram is None:
            print("For 'peaks' command, both --measures and --spectrogram are required")
            sys.exit(1)

        audio_path   = None
        
        # Could be an experiment manager key, in which case it would
        # not have a .csv extension:
        measures_fname_or_key = expand_path_to_data_dir(args.source_info, exp)
        spectro_path = expand_path_to_data_dir(args.spectrogram, exp)

    elif cmd == 'spectro_compression':
        # Could be an experiment manager key, would
        # not have a .csv or .wav extension, or a full name:
        if args.source_info.endswith('.csv'):
            spectro_path = expand_path_to_data_dir(args.source_info, exp)
            audio_path   = None
        elif args.source_info.endswith('.wav'):
            spectro_path = None
            audio_path   = args.source_info
            if not os.path.exists(audio_path):
                print(f"Cannot find audio file {audio_path}. If intending an experiment key, don't include '.wav'")
                sys.exit(1)
        else:
            # Not a file path, but an experiment
            # key to the spectrogram:
            spectro_path = exp.abspath(args.source_info, Datatype.tabular)
            audio_path   = None

    # About number of lags: ~25ms and ~50ms seems to be a good times
    # for look-back. The number of corresponding lags depends on
    # the spectrogram timeframe width, which is correlated with
    # the sampling frequency:
    #
    # At sr=22050; hop_len=512; timedelta~=23ms. ==> 2 lags at ~23ms each
    # At sr=22050: hop_len=256; timedelta~=12ms. ==> 4 lags at ~.12ms (extra granulatity)
    # At sr=31000: hop_len=256; timedelta~=8ms   ==> 6 lags at ~.8ms
    
    settings = SoundSegmentationSetting(
        #****sig_types = ['flatness', 'continuity', 'pitch', 'freq_mod', 'energy_sum'],
        sig_types = ['energy_sum'],
        #******sig_types = ['continuity', 'energy_sum'],
        #******sig_types = ['continuity', 'energy_sum', 'pitch'],
        #******sig_types = ['continuity'],
        sr = 32000, # sampling rate
        pattern_lookback_dur=0.5, # up to 500ms lookback
        pattern_lookback_dur_units=PatternLookbackDurationUnits.SECONDS,
        spectro_path = spectro_path,
        freq_split = (0,16000,500),  # Spectro frequency range, and freq band height
        audio_path = audio_path,
        selection_tbl_path=selection_tbl,
        recording_id = 'AM02_20190717_052958',
        round_to = SegmentationPreparer.ROUND_TO,
        remove_long_selections = True,
        compression_window_width = 2,  # seconds: Sliding window width during compression.
        do_noisereduce=True # Only used if spectro_path is None, so that a new spectro is made
        )
    segmenter = SegmentationPreparer(exp, settings)
    
    #*************
    #cmd = 'peaks'
    #cmd = 'acorrs'
    #cmd = 'spectro_compression'
    #*************
    #**************
    #df = segmenter.experimenter.read('significant_acorrs_2022-07-13T17_19_19', Datatype.tabular)
    #df1 = segmenter.experimenter.read('significant_acorrs_2022-07-09T10_00_15', Datatype.tabular)
    #**************
    if cmd == 'acorrs':
        acorrs = segmenter.compute_autocorrelations(settings)
    elif cmd == 'peaks':
        if args.measures == 'acorrs':
            res = segmenter.peak_positions_acorrs(measures_fname_or_key)
        elif args.measures == 'z_scores':
            res = segmenter.peak_positions_z_scores(measures_fname_or_key)
        else:
            # 'energy':
            res = segmenter.peak_positions_energy(measures_fname_or_key)
    elif cmd == 'spectro_compression':
        res = segmenter.compressed_spectrograms(normalize=True)
    else:
        print("Arg must be 'acorrs', 'peaks', or 'spectro_compression")
    print('Done')