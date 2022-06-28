'''
Created on May 30, 2022

@author: paepcke
'''
from copy import deepcopy
from enum import Enum
import json
import os

from birdsong.utils.utilities import FileUtils
from data_augmentation.utils import RavenSelectionTable, Utils, Interval
from experiment_manager.experiment_manager import ExperimentManager, \
    JsonDumpableMixin
from logging_service import LoggingService
from powerflock.signal_analysis import SignalAnalyzer
from powerflock.signatures import Signature
from result_analysis.charting import Charter

import numpy as np
import pandas as pd


class PatternLookbackDurationUnits(Enum):
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
        self.audio_path    = settings.Path(settings.audio_path) if settings.audio_path is not None else None
        selection_tbl_path = settings.selection_tbl_path
        self.round_to      = settings.round_to
        raw_lookback_duration = settings.pattern_lookback_dur
        
            
        
        
        self.log = LoggingService()
        
        # Get the time for one autocorrelation lag:
        self.lag_duration = SignalAnalyzer.spectro_timeframe_width()

        # Need number of lags to compute for each autocorrelation.
        # The lag may be given in seconds, or directly in lags:
        if settings.pattern_lookback_dur_units == PatternLookbackDurationUnits.LAGS:
            # Given in lags:
            self.num_lags = raw_lookback_duration
        else:
            # Lookback duration given in number of seconds:
            self.num_lags = int(raw_lookback_duration / self.lag_duration) 

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
            self.selections_df = self._process_selection_tbl(self.spectro_times, selection_tbl_path)
            experimenter.save('sel_tbl', self.sel_tbl)
        else:
            # No selection table available:
            is_sel_start = pd.Series([np.nan]*len(self.spectro_times), 
                                     name='is_sel_start',
                                     index=self.spectro_times)
            is_sel_stop  = pd.Series([np.nan]*len(self.spectro_times), 
                                     name='is_sel_stop',
                                     index=self.spectro_times)
            self.selections_df = pd.DataFrame({'true_sel_start' : is_sel_start,
                                               'true_sel_stop'  : is_sel_stop
                                               }) 

    #------------------------------------
    # slice_spectro
    #-------------------
    
    def slice_spectro(self, slice_height=500, low_freq=500, high_freq=8000):
        '''
        Slice given spectrogram into 500Hz horizontal slices. Save
        each slice dataframe to the experiment using spectro file 
        name with center freq appended as save key.
        
        Returns dict of slices keys being low freq of the slice, 
        and values being the slice dataframes.
        
        Note: frequency values are 'snapped' to the nearest frequency
        available in the spectrogram.
        
        :param spectro: Spectrogram to slice
        :type spectro: pd.DataFrame
        :returns array of spectrogram slices
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
            
            # Uncomment to save all spectro slices separately:
            #self.experimenter.save(f"{self.recording_id}_{target_freq_low}Hz", spectro_slice)

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
        apply autocorrelation to each column every lookback_duration seconds.
        
        Return:
                   flatness_rho, flatness_sum_significant, pitch_rho, pitch_sum_significant, ...
            time
              0          ...               True/False         ...          True/False
              0.0232
                ...
        
        Time is in seconds of true spectrogram time. The xxx_sum_significant
        is the number of autocorrelation measures that are significant during
        the lookback_duration. 
        
        The lookback_duration is converted to lags for the purpose of autocorrelation.
        
        Time is in seconds
        :param col_names:
        :type col_names:
        :param lookback_duration:
        :type lookback_duration:
        '''
        
        slice_arr = self.slice_spectro()
        
        # For each horizontal one_slice, compute signatures separately:
        for spectro_slice in slice_arr:
            exp_key = f"slice_sig_{spectro_slice.low_freq}_{spectro_slice.high_freq}"
            try:
                slice_sig = self.experimenter.read(exp_key, Signature)
            except FileNotFoundError:
                slice_sig = segmenter.compute_sig_whole_spectro(spectro_slice)
                self.experimenter.save(exp_key, slice_sig)
                
            spectro_slice.sig = slice_sig

        final_res_arr = []
        
        for one_slice in slice_arr:

            freq_low  = one_slice.low_freq
            freq_high = one_slice.high_freq
            sig_df    = one_slice.sig.sig
            
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
        
        # Turn the list of dataframes into one large df.
        # The index of that big df will be repeating the
        # the index of the constituent correlation dfs, which
        # is not useful; so reset the index to simple row nums:
        final_res_df = pd.concat(final_res_arr).reset_index()
        
        # Add boolean columns is_sel_start and is_sel_top.
        # Remember: each time occurs several times, once for
        # each frequency band:
        
        is_sel_start_col = pd.Series([False]*len(final_res_df), index=final_res_df.time, name='is_sel_start')
        is_sel_stop_col = pd.Series([False]*len(final_res_df), index=final_res_df.time, name='is_sel_start')
        
        # Get all the true selection start times:
        true_start_times = self.selections_df[self.selections_df.true_sel_start].index.to_numpy(dtype=float)
        true_stop_times  = self.selections_df[self.selections_df.true_sel_stop].index.to_numpy(dtype=float)
        
        # In the new is_sel_start and is_sel_stop columns
        # that will go into the final_res_df, set to true
        # the times when true selection starts/stops happen
        is_sel_start_col.loc[is_sel_start_col.index.isin(true_start_times)] = True
        is_sel_stop_col.loc[is_sel_stop_col.index.isin(true_stop_times)] = True
        
        # Must harmonize index between final_res_df and
        # the two new columns to add the cols to
        # final_res_df:
        is_sel_start_col.index = final_res_df.index
        is_sel_stop_col.index = final_res_df.index

        # Add the cols to final_res_df:
        final_res_df['is_sel_start'] = is_sel_start_col
        final_res_df['is_sel_stop'] = is_sel_stop_col

        exp_key = f"significant_acorrs_{FileUtils.file_timestamp()}"
        self.experimenter.save(exp_key, final_res_df)
        
        return final_res_df

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
        res_df = pd.DataFrame(index=self.spectro_times)
        
        stop_time = max(sig_measure.index)
        
        self.log.info(f"Autocorrelations for measure {sig_measure.name}: [0,{stop_time}] sec")

        # Do autocorrelation every lookback_duration seconds:
        for spectro_start_time in sig_measure.index[::self.num_lags]:
            
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
            
            res_df.loc[acorr_res.index, acorr_res.columns] = acorr_res

        return res_df


    #------------------------------------
    # save_sel_tbl_and_sigs
    #-------------------
    
    def save_sel_tbl_and_sigs(self, spectro_slice, sel_tbl):


        self.log.info(f"Saving selection tables and signatures for slice {spectro_slice.name}")

        # Get the quadsig values themselves:
        sig_vals_df = spectro_slice.sig.sig
        slice_freq_interval = Interval(spectro_slice.low_freq, spectro_slice.high_freq)

        # We will remove selections that cover almost the entire
        # recording (95%):
        most_of_recording = 95. * (sig_vals_df.index[-1] - sig_vals_df.index[0]) / 100. 

        # Remove selections that are outside the
        # frequency coverage of the signature, but
        # do that on a copy:
        sel_tbl_culled = deepcopy(sel_tbl)
        
        # Copy selection table entries in iterator 
        # b/c we will remove entries from the selection
        # table copy's entry list in the loop:
         
        for selection_entry in deepcopy(sel_tbl_culled.entries):
            # Remove selections in frequency ranges that are not of interest
            # or that span most of the spectrogram:
            sel_freq_interval = Interval(selection_entry.low_freq, selection_entry.high_freq)
            if not sel_freq_interval.overlaps(slice_freq_interval) or \
               (selection_entry.stop_time - selection_entry.start_time) > most_of_recording:
                sel_tbl_culled.remove(selection_entry)
            
        # Get ground truth start times of all calls from the selection table:
        sel_ids      = [tbl_entry.sel_id for tbl_entry in sel_tbl_culled]
        start_times  = [tbl_entry.start_time for tbl_entry in sel_tbl_culled]
        stop_times   = [tbl_entry.stop_time for tbl_entry in sel_tbl_culled]
        low_freqs    = [tbl_entry.low_freq for tbl_entry in sel_tbl_culled]
        high_freqs   = [tbl_entry.high_freq for tbl_entry in sel_tbl_culled]
        # Main species is the first of the species_list
        # See utils.py RavenSelectionTableEntry init method.
        species       = [tbl_entry.species_list[0] for tbl_entry in sel_tbl_culled]
        species_lists = [','.join(tbl_entry.species_list) for tbl_entry in sel_tbl_culled]
        
        # Start times snapped to the spectrogram time frames. Selection tables
        # don't have massive numbers of entries, so loop is fine:
        
        sig_vals_df.index = sig_vals_df.index.astype(float).to_series().round(self.round_to)
        sig_vals_df['is_sel_start'] = False
        sig_vals_df['is_sel_stop'] = False
        
        snapped_starts = {}
        snapped_stops  = {}

        for start, stop in zip(start_times, stop_times):

            snapped_start = Utils.nearest_in_array(sig_vals_df.index,
                                                   round(start,self.round_to),
                                                   is_sorted=True)
            snapped_starts[start] = snapped_start
            sig_vals_df.loc[snapped_start, 'is_sel_start'] = True
    
            snapped_stop = Utils.nearest_in_array(sig_vals_df.index,
                                                  round(stop,self.round_to),
                                                  is_sorted=True)
            snapped_stops[stop] = snapped_stop
            sig_vals_df.loc[snapped_stop, 'is_sel_stop'] = True
    
        #********* Add selection_id into each row that is in_selection:
    
        # Add a column 'inSelection' to indicate whether at each row's
        # signatures we were in a selection box:
        sig_vals_df['in_selection'] = False
        for start_sel, stop_sel in zip(snapped_starts.values(), snapped_stops.values()):
            sig_vals_df.loc[start_sel:stop_sel, 'in_selection'] = True

        sel_tbl_df   = pd.DataFrame({'start_time' : start_times,
                                     'stop_time'  : stop_times,
                                     'low_freqs'  : low_freqs,
                                     'high_freqs' : high_freqs,
                                     'species'    : species,
                                     'species_lists' : species_lists,
                                     'snapped_start' : [snapped_starts[start]
                                                        for start
                                                        in start_times
                                                        ],
                                     'snapped_stop'  : [snapped_stops[stop]
                                                        for stop
                                                        in stop_times
                                                        ],
                                     },
                                     index = sel_ids,
                                     )
        
        save_key_root = f"{spectro_slice.low_freq}_to_{spectro_slice.high_freq}"
        self.experimenter.save(
            f"sig_vals_{save_key_root}", 
            sig_vals_df, 
            index_col='time')
        self.experimenter.save(
            f"sel_tbl_culled_{save_key_root}", 
            sel_tbl_df, 
            index_col='selection_id')

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
    
    def _process_selection_tbl(self, spectro_times, selection_tbl_path):
        '''
        Imports a Raven selection table and the time frames
        in seconds of a spectrogram. Returns a two-column
        boolean df with columns is_sel_start and is_sel_stop.
        
        :param spectro_times: times in seconds of spectrogram
            time frames
        :type spectro_times: [float]
        :param selection_tbl_path: path to a Raven selection table.
        :type selection_tbl_path: str
        :return for each time, whether or not it is the start or
            end of a Raven selection.
        :rtype pd.DataFrame
        '''
        
        self.log.info("Reading selection table from file...")
        self.sel_tbl = RavenSelectionTable(selection_tbl_path)
        
        # Get list of true start/stop times from selection table,
        # rounded to our standard ROUND_TO decimal places:
        
        true_start_times = pd.Series([np.round(sel_entry.start_time, self.round_to)
                                 for sel_entry
                                 in self.sel_tbl], name='true_sel_starts')
        true_stop_times = pd.Series([np.round(sel_entry.stop_time, self.round_to)
                                for sel_entry
                                in self.sel_tbl], name='true_sel_stops')
        
        # Two Series with values all False, one False for each
        # spectrogram time frame. One series for start, one
        # for stop times: 
        is_sel_start = pd.Series([False]*len(self.spectro_times), 
                                 name='is_sel_start',
                                 index=self.spectro_times)
        is_sel_stop  = pd.Series([False]*len(self.spectro_times), 
                                 name='is_sel_stop',
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
        
        for stop_time in true_stop_times:
            # For this selection table stop time,
            # find the nearest spectrogram time frame:
            snapped_stop_time = Utils.nearest_in_array(spectro_times,
                                                       stop_time,
                                                       is_sorted=True)
            # Indicate the time among all spectro timeframes
            # that is a selection stop:
            is_sel_stop.loc[snapped_stop_time] = True

        true_selections_df = pd.DataFrame({'true_sel_start' : is_sel_start,
                                           'true_sel_stop'  : is_sel_stop
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
                 round_to=5
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
            as_dict['round_to']
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
    exp_root = os.path.join(EXP_ROOT, 'ExpAM02_20190717_052958_OneLag')
    exp = ExperimentManager(exp_root)
    settings = SoundSegmentationSetting(
        1, # One lag only
        PatternLookbackDurationUnits.LAGS,
        spectro_path = SPECTRO_PATH,
        selection_tbl_path=TEST_SEL_TBL,
        recording_id = 'AM02_20190717_052958'
        )
    segmenter = AudioSegmenter(exp, settings)
    
    acorrs = segmenter.compute_autocorrelations(['flatness', 'continuity', 'pitch', 'freq_mod', 'energy_sum'])
        