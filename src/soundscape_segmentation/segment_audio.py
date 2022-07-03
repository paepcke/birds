#!/usr/bin/env python
'''
Created on May 30, 2022

@author: paepcke
'''
from enum import Enum
import json
import os
from pathlib import Path

from scipy.signal import find_peaks

from birdsong.utils.utilities import FileUtils
from data_augmentation.utils import RavenSelectionTable, Utils
from experiment_manager.experiment_manager import ExperimentManager, \
    JsonDumpableMixin
from logging_service import LoggingService
from powerflock.signal_analysis import SignalAnalyzer
from powerflock.signatures import Signature
from result_analysis.charting import Charter

import multiprocessing as mp
import numpy as np
import pandas as pd


class PatternLookbackDurationUnits(Enum):
    '''
    Distinguish between numbers that indicate
    lag, vs. seconds (the two are roughtly 
    interchangabe.
    '''
    LAGS = 0
    SECONDS = 1

#from result_analysis.charting import Charter
CUR_DIR    = os.path.dirname(__file__)
PROJ_ROOT  = os.path.abspath(os.path.join(CUR_DIR, '../../'))
TEST_DATA  = os.path.join(PROJ_ROOT, 'data')
AUDIO_PATH =  os.path.join(TEST_DATA, 'kelleyRecommendedFldRec_AM02_20190717_052958.wav')
TEST_SEL_TBL = os.path.join(TEST_DATA, 'DS_AM02_20190717_052958.Table.1.selections.txt')
SPECTRO_PATH = os.path.join(TEST_DATA, 'am02_spectro.csv')

EXP_ROOT = os.path.join(PROJ_ROOT, 'experiments/SoundscapeSegmentation')

#from data_augmentation.sound_processor import SoundProcessor 

class AudioSegmenter:
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
        self.spectro_path  = settings.spectro_path
        self.audio_path    = Path(settings.audio_path) if settings.audio_path is not None else None
        selection_tbl_path = settings.selection_tbl_path
        self.round_to      = settings.round_to
        raw_lookback_duration  = settings.pattern_lookback_dur
        remove_long_selections = settings.remove_long_selections

        self.log = LoggingService()
        
        # Get the time for one autocorrelation lag:
        self.lag_duration = SignalAnalyzer.spectro_timeframe_width()

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

        if not os.path.exists(self.spectro_path):
            self.spectro = self.make_test_spectro(self.audio_path)
        else:
            self.log.info("Reading spectrogram from .csv file...")
            self.spectro = pd.read_csv(self.spectro_path,
                                       index_col='freq',
                                       header=0
                                       )
        # Get the spectrogram times as np array of rounded floats:
        spectro_times_rounded = self.spectro.columns.to_numpy(dtype=float).round(self.round_to)
        self.spectro_times = pd.Series(spectro_times_rounded, 
                                       index=spectro_times_rounded, 
                                       name='spectro_times')
        
        if selection_tbl_path is not None:
            self.selections_df = self._process_selection_tbl(self.spectro_times, 
                                                             selection_tbl_path,
                                                             remove_long_selections)
            experimenter.save('sel_tbl', self.sel_tbl)
        else:
            # No selection table available:
            sel_id  = pd.Series(np.nan, name='sel_id', index=self.spectro_times)
            is_sel_start = pd.Series(np.nan, name='is_sel_start', index=self.spectro_times)
            is_sel_stop  = pd.Series(np.nan, name='is_sel_stop', index=self.spectro_times)
            sel_duration = pd.Series(np.nan, name='sel_duration', index=self.spectro_times)
            
            self.selections_df = pd.DataFrame({'true_sel_id'    : sel_id,
                                               'true_sel_start' : is_sel_start,
                                               'true_sel_stop'  : is_sel_stop,
                                               'true_sel_dur'   : sel_duration
                                               }) 

    #------------------------------------
    # slice_spectro
    #-------------------
    
    def slice_spectro(self, 
                      slice_height=500, 
                      low_freq=500, 
                      high_freq=8000,
                      save_slices=False
                      ):
        '''
        Slice given spectrogram into 500Hz horizontal slices. If
        requested, save each slice dataframe to the experiment using 
        spectro file name with center freq appended as save key.
        
        Returns an array of SpectroSlice instances. The final 'slice'
        covers the entire spectrogram.
        
        Note: frequency values are 'snapped' to the nearest frequency
        available in the spectrogram.
        
        Assumption: self.spectro contains the full spectrogram

        :param slice_height: frequency band width
        :type slice_height: {int | float}
        :param low_freq: frequency at which to start 
        :type low_freq: {int | float}
        :param high_freq: frequency beyond which no 
            slices are computed
        :type high_freq: {int | float}
        :param save_slices: whether or not to save each
            slice to the current experiment
        :type save_slices: bool
        :returns list of SpectroSlice instances
        :rtype [SpectroSlice]
        '''
        
        self.log.info(f"Slicing spectro into {int((high_freq - low_freq) / slice_height)} freq bands: {slice_height} high [{low_freq}, {high_freq}]")
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
    # compute_sig_whole_spectro 
    #-------------------
    
    def compute_sig_whole_spectro(self, spectro_slice_or_spectro):
        
        if type(spectro_slice_or_spectro) == SpectroSlice:
            slice_desc = f"slice {spectro_slice_or_spectro.name}"
            spectro = spectro_slice_or_spectro.spectro
        else:
            slice_desc = f"spectro {min(spectro.index)}Hz to {max(spectro.index)}Hz"
            
        self.log.info(f"Computing signatures for {slice_desc}")
        
        # Make an empty sig, and fill it with the
        # four signatures across the whole spectro: ******
        sig_for_whole_spectro = Signature('test_species', 
                                          spectro, 
                                          fname=self.audio_path,
                                          sig_id=0)
        sig = SignalAnalyzer.spectral_measures_each_timeframe(spectro, sig_for_whole_spectro)
        
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
    
    def compute_autocorrelations(self, col_names, lookback_duration=0.5):
        '''
        Given a signature and a list of signature column names: ['flatness', 'pitch', ...],
        apply autocorrelation to each column up to number of lags equivalent
        to lookback_duration. This computation is done separately for each
        frequency slice. 
        
        Return:
                   flatness_rho, flatness_sum_significant, pitch_rho, pitch_sum_significant, ...
            time
              0          ...               True/False         ...          True/False
              0.0232
                ...
        
        :param col_names: name of measure(s) for which to compute autocorrelations
        :type col_names: {str | [str]}
        :param lookback_duration: time that covers the number of lags
            given the timeframes of the underlying spectrogram
        :type lookback_duration: float
        '''
        
        slice_arr = self.slice_spectro()
        if type(col_names) != list:
            col_names = [col_names]
        
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
        
        # Turn the list of dataframes into one large df.
        # The index of that big df will be repeating the
        # the index of the constituent correlation dfs, which
        # is not useful; so reset the index to simple row nums:
        final_res_df = pd.concat(final_res_arrs_flat).reset_index()
        
        # Get all the true selection start times
        # snapped to spectro timeframes:
        true_start_times = self.selections_df[self.selections_df.true_sel_start].index.to_numpy(dtype=float)
        true_stop_times  = self.selections_df[self.selections_df.true_sel_stop].index.to_numpy(dtype=float)
        
        # Add boolean columns is_sel_start and is_sel_stop
        final_res_df.loc[final_res_df.time.isin(true_start_times), 'is_sel_start'] = True
        final_res_df.loc[final_res_df.time.isin(true_stop_times), 'is_sel_stop'] = True
        final_res_df.is_sel_start.fillna(False, inplace=True)
        final_res_df.is_sel_stop.fillna(False, inplace=True)

        # Add a column in_selection, which has the respective selection's ID
        # if a time falls within a selection from the selection table.
        # Also added: a column 'sel_dur' with the respective selection's 
        # time duration:
        for sel_tbl_entry in self.sel_tbl.entries:
            start_time_snapped = Utils.nearest_in_array(final_res_df.time, sel_tbl_entry.start_time)
            stop_time_snapped  = Utils.nearest_in_array(final_res_df.time, sel_tbl_entry.stop_time)
            sel_id  = sel_tbl_entry.sel_id
            sel_dur = sel_tbl_entry.delta_time
            final_res_df.loc[final_res_df['time'].between(start_time_snapped, 
                                                          stop_time_snapped), 'is_in_selection'] = sel_id
            final_res_df.loc[final_res_df['time'].between(start_time_snapped, 
                                                          stop_time_snapped), 'sel_dur'] = sel_dur
        
        exp_key = f"significant_acorrs_{FileUtils.file_timestamp()}"
        self.experimenter.save(exp_key, final_res_df)
        
        return final_res_df

    #------------------------------------
    # peak_positions
    #-------------------
    
    def peak_positions(self, df_or_path):
        '''
        Given a df produced by compute_autocorrelations(), 
        find the peaks in each autocorrelation series. Of
        those there are [num-of-freq-bands] * [num-of-sig-types]
        where sig_types is 'flatness', 'pitch' etc.
        
        Uses scypi.signal.find_peak(), which considers prominence
        relative to other peaks.
        
        Saves and returns a df that includes:
          'time' spectrogram time when peak occurred, 
          'sig_type'      over which signature values was the autocorrelation
                             computed, 
          'freq_low'      low edge of frequency band
          'freq_high'     high edge of frequency band
          'is_in_selection' whether or not the time is 
                            within a Raven selection
          'lag'           lag value for which a row's autocorrelation
                             was computed
          'sel_dur'       duration in seconds of the Raven selection 
          'acorr'         the autocorrelation value 
          'peak_height'   the height of the peak in the same
                          units as the autocorrelation acorr column
          'plateau_width' width of the detected plateau 
          'prominence'    prominence of the peak
        
        :param df_or_path: either a df created by compute_autocorrelations(),
            or the file path to such a df stored as csv
        :type df_or_path: {pd.DataFrame | str}
        :returns a new df with peaks computed for every
            freqband/sig_type pair
        :rtype pd.DataFrame
        '''
        
        if type(df_or_path) == str:
            df = pd.read_csv(df_or_path)
            in_fname_timestamp = FileUtils.extract_file_timestamp(df_or_path)
        else:
            df = df_or_path
            in_fname_timestamp = None
            
        # Partition the df by sig_type ('flatness', 'pitch' etc.),
        # and frequency band. The number of df extracts in the group:
        #    num_sig_types * num_freq_bands
        df_grp = df.groupby(by=['sig_type', 'freq_low', 'freq_high'])
        
        # Get a list of dfs, each containing the original
        # df rows for one combination sig_type/freq_band
        # Each df in the arr will have some junk cols from 
        # the group multiindex, but otherwise have the same
        # cols as the original df. 
        fband_sig_type_dfs = [data.reset_index() 
                              for _grp_key, data 
                              in df_grp]
        
        # For each extract, build df:
        #    peak_idx   measure_name   measure_val  freq_low  freq_high   peak_height  peak_width  prominence  
        # time
        
        peaks_df_arr = []
        for df_extract in fband_sig_type_dfs:
            # Isolate the acorr values that are significant,
            # and are not the trivial value of 1 that occurs
            # with lag==0. Note: its index will be fragmented,
            # showing the chosen indexes of df_extract:
            extract_acorrs = df_extract[np.logical_and(df_extract.is_significant, 
                                                       df_extract.lag > 0)].acorr.abs()
            # Get the indexes in *extract_acorrs* (not directly
            # those in df_extract) that are peaks:
            peak_idxs, peak_stats = find_peaks(
                extract_acorrs,
                height=(None, None), 
                width=(None, None), 
                prominence=(None, None), 
                plateau_size=(None,None))

            # To get the indexes in the original df_extract,
            # that are peaks, first get the rows from extract_acorrs
            # that are peaks. Then get the index entries of those
            # rows. Those indexes point into df_extract:
            df_extract_idxs = extract_acorrs.iloc[peak_idxs].index
            peaks_df = df_extract.iloc[df_extract_idxs][['time', 
                                                         'sig_type', 
                                                         'freq_low', 
                                                         'freq_high', 
                                                         'is_in_selection', 
                                                         'sel_dur',
                                                         'is_sel_start',
                                                         'is_sel_stop', 
                                                         'lag', 
                                                         'acorr']]

            # Time column needs to match the significant-digits
            # convention to match other dataframes:
            peaks_df['time'] = peaks_df['time'].round(self.round_to)
            
            # Add some of the find_peaks() stats:
            peaks_df['peak_height'] = peak_stats['peak_heights']
            peaks_df['plateau_width'] = peak_stats['plateau_sizes']
            peaks_df['prominence'] = peak_stats['prominences']
            
            peaks_df_arr.append(peaks_df)
            
        peaks_df = pd.concat(peaks_df_arr)
        # Flip the negative correlations to the top
        # of the acorr == 0 line:
        peaks_df['acorr'] = np.abs(peaks_df['acorr'])
        
        if in_fname_timestamp is not None:
            # Match the input df's timestamp:
            exp_key = f"peaks_{in_fname_timestamp}"
        else:
            exp_key = f"peaks_{FileUtils.in_fname_timestamp()}"
        self.experimenter.save(exp_key, peaks_df)

        return peaks_df

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
        return {'round_to' : self.round_to,
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

    #------------------------------------
    # _process_one_freq_slice
    #-------------------
    
    def _process_one_freq_slice(self, spectro_freq_slice, col_names, lookback_duration):
        
        final_res_arr = []

        exp_key = f"slice_sig_{spectro_freq_slice.low_freq}_{spectro_freq_slice.high_freq}"
        try:
            slice_sig = self.experimenter.read(exp_key, Signature)
        except FileNotFoundError:
            slice_sig = self.compute_sig_whole_spectro(spectro_freq_slice)
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
        Return:
                    time   flatness_sum_acorr_cnts
            0    0.00000                         2
            1    1.99692                         2
            2    3.99383                         0
                    
        where xxx_sum_acorr_cnts is a count of statistically significant
        autocorrelation values when repeatedly looking back up to lookback_duration seconds.
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
        res_df = pd.DataFrame()
        
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
                spectro_end_time = sig_measure[sig_measure.index >= spectro_start_time + lookback_duration].index[0]
            except IndexError:
                # Went through the whole series:
                break
            
            measures = sig_measure.loc[spectro_start_time:spectro_end_time]
            
            # Get df w/ cols like 'flatness', 'is_significant':
            acorr_res = SignalAnalyzer.autocorrelation(measures, nlags=self.num_lags)
            
            # Add a signature type column ('flatness' or 'pitch', etc.)
            acorr_res.insert(1, 'sig_type', [sig_measure.name]*len(acorr_res))

            # Append the new results:
            res_df = pd.concat([res_df, acorr_res], axis=0)

        return res_df

    #------------------------------------
    # make_test_spectro
    #-------------------
    
    def make_test_spectro(self, test_audio):
        
        self.log.info(f"Creating spectrogram for {test_audio}")
        
        spectro = SignalAnalyzer.raven_spectrogram(test_audio, 
                                                   to_db=False, 
                                                   extra_granularity=False)
        spectro.to_csv(self.spectro_path,
                       header=spectro.columns,
                       index=True,
                       index_label='freq'
                       )
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
        in seconds of a spectrogram. Returns a two-column
        boolean df with columns is_sel_start and is_sel_stop.
        
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
        
        # If requested, only keep selections that are less
        # than 95% of the total spectrogram in length:
        if remove_long_selections:
            removal_threshold = spectro_times.max() * 95/100
            sel_entries = list(filter(lambda entry, removal_threshold=removal_threshold: 
                                      entry.stop_time - entry.start_time < removal_threshold, 
                                      self.sel_tbl.entries))
        else:
            sel_entries = self.sel_tbl.entries
        
        # Get list of true start/stop times from selection table,
        # rounded to our standard ROUND_TO decimal places:
        
        true_start_times = pd.Series([np.round(sel_entry.start_time, self.round_to)
                                 for sel_entry
                                 in sel_entries], name='true_sel_starts')
        true_stop_times = pd.Series([np.round(sel_entry.stop_time, self.round_to)
                                for sel_entry
                                in sel_entries], name='true_sel_stops')
        
        # Two Series with values all False, one False for each
        # spectrogram time frame. One series for start, one
        # for stop times: 
        is_sel_start = pd.Series([False]*len(self.spectro_times), 
                                 name='is_sel_start',
                                 index=self.spectro_times)
        is_sel_stop  = pd.Series([False]*len(self.spectro_times), 
                                 name='is_sel_stop',
                                 index=self.spectro_times)
        
        # Another two full-length columns, one for the selection ID,
        # and one for the duration of each selection:
        sel_id = pd.Series([np.nan]*len(self.spectro_times), 
                           name='sel_id',
                           index=self.spectro_times)
        sel_duration = pd.Series([np.nan]*len(self.spectro_times), 
                                 name='sel_duration',
                                 index=self.spectro_times)
        
        
        
        for start_time in true_start_times:
            # For this selection table start time,
            # find the nearest spectrogram time frame:
            snapped_start_time = Utils.nearest_in_array(spectro_times,
                                                        start_time,
                                                        is_sorted=True)
            # Indicate the time among all spectro timeframes
            # that is a selection start:
            is_sel_start.loc[snapped_start_time] = True
            # Set corresponding selection ID
            sel_id.loc[snapped_start_time] = \
                int(next(filter(lambda entry, 
                                start_time=start_time: round(entry.start_time,self.round_to) == start_time,
                                sel_entries)).sel_id)
            # Set corresponding length of selection:
            sel_duration.loc[snapped_start_time] = \
                next(filter(lambda entry, 
                            start_time=start_time: round(entry.start_time, self.round_to) == start_time,
                            sel_entries)).delta_time
        for stop_time in true_stop_times:
            # For this selection table stop time,
            # find the nearest spectrogram time frame:
            snapped_stop_time = Utils.nearest_in_array(spectro_times,
                                                       stop_time,
                                                       is_sorted=True)
            # Indicate the time among all spectro timeframes
            # that is a selection stop:
            is_sel_stop.loc[snapped_stop_time] = True
            
            # Set corresponding selection ID
            sel_id.loc[snapped_stop_time] = \
                int(next(filter(lambda entry, 
                                stop_time=stop_time: round(entry.stop_time,self.round_to) == stop_time,
                                sel_entries)).sel_id)
            # Set corresponding length of selection:
            sel_duration.loc[snapped_stop_time] = \
                next(filter(lambda entry, 
                            stop_time=stop_time: round(entry.stop_time, self.round_to) == stop_time,
                            sel_entries)).delta_time

        true_selections_df = pd.DataFrame({'true_sel_sel_id': sel_id,
                                           'true_sel_start' : is_sel_start,
                                           'true_sel_stop'  : is_sel_stop,
                                           'true_sel_dur'   : sel_duration
                                           })
        
        return true_selections_df
    
    #------------------------------------
    # _nearest_spectro_times
    #-------------------
    
    def _nearest_spectro_times(self, arr_like):
        '''
        Given an array of time floats, look up for each
        time the closest spectrogram time.
        
        Assumption: self.spectro_time is a sorted list of
        spectrogram times.
        
        Returns an np array of new times the same length
        as arr_like.
        
        :param arr_like: times for which to find closest
            spectro times
        :type arr_like: {[float] | pd.Series([float])
        :return list of nearest spectrogram times
        :rtype np.ndarray
        '''
        
        spectro_times = []
        for one_time in arr_like:
            spectro_times.append(Utils.nearest_in_array(self.spectro_times,
                                                        one_time,
                                                        is_sorted=True))
        return np.array(spectro_times)


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
                 pattern_lookback_dur,
                 pattern_lookback_dur_units,
                 spectro_path=None,
                 audio_path=None,
                 selection_tbl_path=None,
                 recording_id=None,
                 round_to=5,
                 remove_long_selections=False
                 ):
        '''
        Holds parameters for a soundscape segmentation run.
        Lookback duration may be specified either in fractional
        seconds, or in a number of lags. The pattern_lookback_dur_units
        parameter must specify which it is. The spectrogram time frame
        times are used to convert from one to the other.
        
        Either spectrogram path or audio path may be specified. If
        spectrogram path is provided, it is used, and the audio path
        is ignored. Else a spectrogram is created from the audio.
        
        If a Raven selection is provided, its selection start and end
        times will be available in the final result dataframe.  
        
        The recording_id is an optional string that identifies the soundscape
        recording indpendent of file location. Ex: AM02_20190717_052958
        If provided it is used when creating file names. 
         
        :param pattern_lookback_dur: number of fractional seconds, or
            number of lags to use for autocorrelation
        :type pattern_lookback_dur: {float | int}
        :param pattern_lookback_dur_units: whether pattern_lookback_dur is
            in units of seconds or of lags
        :type pattern_lookback_dur_units: PatternLookbackDurationUnits
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
        '''
        self.pattern_lookback_dur = pattern_lookback_dur
        self.pattern_lookback_dur_units = pattern_lookback_dur_units
        self.spectro_path = spectro_path
        self.audio_path = audio_path
        self.selection_tbl_path = selection_tbl_path
        self.recording_id = recording_id
        self.round_to = round_to
        self.remove_long_selections = remove_long_selections

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
        res = SoundSegmentationSetting(
            as_dict['pattern_lookback_dur'],
            as_dict['pattern_lookback_dur_units'],
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

# ------------------------ Main ------------
if __name__ == '__main__':
    #exp_root = os.path.join(EXP_ROOT, 'ExpAM02_20190717_052958_HalfSecondLags')
    exp_root = os.path.join(EXP_ROOT, 'ExpAM02_20190717_052958_AllData')
    exp = ExperimentManager(exp_root)
    settings = SoundSegmentationSetting(
        10, # 10 lags at ~23ms each. No significatnt acorrs beyond that.
        PatternLookbackDurationUnits.LAGS,
        spectro_path = SPECTRO_PATH,
        audio_path = AUDIO_PATH,
        selection_tbl_path=TEST_SEL_TBL,
        recording_id = 'AM02_20190717_052958',
        round_to = AudioSegmenter.ROUND_TO,
        remove_long_selections = True
        )
    segmenter = AudioSegmenter(exp, settings)
    
    #*******acorrs = segmenter.compute_autocorrelations(['flatness', 'continuity', 'pitch', 'freq_mod', 'energy_sum'])
    res = segmenter.peak_positions('/Users/paepcke/EclipseWorkspacesNew/birds/experiments/SoundscapeSegmentation/ExpAM02_20190717_052958_AllData/csv_files/significant_acorrs_2022-07-01T12_45_10.csv')
        