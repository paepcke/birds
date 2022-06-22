'''
Created on May 30, 2022

@author: paepcke
'''
from copy import deepcopy
import os
from pathlib import Path

from data_augmentation.utils import RavenSelectionTable, Utils, Interval
from experiment_manager.experiment_manager import ExperimentManager
from powerflock.signal_analysis import SignalAnalyzer
from powerflock.signatures import Signature
from result_analysis.charting import Charter

import numpy as np
import pandas as pd

from logging_service import LoggingService


#from result_analysis.charting import Charter
CUR_DIR    = os.path.dirname(__file__)
PROJ_ROOT  = os.path.abspath(os.path.join(CUR_DIR, '../../'))
TEST_DATA  = os.path.join(PROJ_ROOT, 'data')
AUDIO_PATH =  os.path.join(TEST_DATA, 'kelleyRecommendedFldRec_AM02_20190717_052958.wav')
TEST_SEL_TBL = os.path.join(TEST_DATA, 'DS_AM02_20190717_052958.Table.1.selections.txt')
SPECTRO_PATH = os.path.join(TEST_DATA, 'am02_spectro.csv')

EXP_ROOT = os.path.join(PROJ_ROOT, 'experiment')

#from data_augmentation.sound_processor import SoundProcessor 

class AudioSegmenter:
    '''
    
    '''
    
    PATTERN_LOOKBACK_DURATION = 3 # sec
    '''Number of seconds to consider during autocorrelation computations for detecting patterns'''

    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

    def __init__(self, experimenter, spectro_path, audio_path=None, selection_tbl_path=None):
        '''
        Constructor
        '''

        self.experimenter = experimenter
        self.spectro_path = spectro_path
        self.audio_path   = Path(audio_path)
        
        self.log = LoggingService()
        
        # Get the time for one autocorrelation lag:
        self.lag_duration = SignalAnalyzer.spectro_timeframe_width()

        self.exp_root = os.path.join(AudioSegmenter.proj_root, 'experiment')

        if not os.path.exists(spectro_path):
            self.spectro = self.make_test_spectro(audio_path)
        else:
            self.spectro = pd.read_csv(spectro_path,
                                       index_col='freq',
                                       header=0
                                       )
        if selection_tbl_path is not None:
            self.sel_tbl = RavenSelectionTable(selection_tbl_path)
            experimenter.save('sel_tbl', self.sel_tbl)

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
        
        self.log.info(f"Slicing spectro into {high_freq - low_freq} freq bands: {slice_height} high [{low_freq}, {high_freq}]")
        slice_arr = []
        
        # Get /foo/bar  from /foo/bar.wav
        #out_path_root = Path.joinpath(self.audio_path.parent, self.audio_path.stem)
        out_key_root = self.audio_path.stem
        freqs = pd.Series(self.spectro.index, name='freqs')
        
        for target_freq_low in np.arange(low_freq, high_freq, slice_height):
            low_freq = max(freqs[freqs <= target_freq_low])
            high_freq = max(freqs[freqs <= target_freq_low + slice_height])

            # Cut the slice. Weirdly the following returns 0 rows:
            #    slice = spectro.loc[low_freq:high_freq,:] 
            spectro_slice = self.spectro[(self.spectro.index >= low_freq) & (self.spectro.index <= high_freq)]
            # Round the time columns to 5 places:
            spectro_slice.columns = pd.Series(spectro_slice.columns.astype(float)).round(5)
            
            slice_arr.append(SpectroSlice(low_freq, high_freq, slice_height, spectro_slice)) 
            self.experimenter.save(f"{out_key_root}_{target_freq_low}Hz", spectro_slice)

        return slice_arr

    #------------------------------------
    # compute_sig_whole_spectro 
    #-------------------
    
    def compute_sig_whole_spectro(self, spectro_slice_or_spectro):
        
        if type(spectro_slice_or_spectro) == SpectroSlice:
            slice_desc = f"slice spectro_slice.name"
            spectro = spectro_slice_or_spectro.spectro
        else:
            slice_desc = f"spectro {min(spectro.index)}Hz to {max(spectro.index)}Hz"
            
        self.log.info(f"Computing signatures for {slice_desc}")
        
        # Make an empty sig, and fill it with the
        # four signatures across the whole spectro: ******
        sig_for_whole_spectro = Signature('test_species', 
                                          spectro, 
                                          fname=AUDIO_PATH, 
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
    
    def compute_autocorrelations(self, sig, col_names):
        
        res_df = pd.DataFrame()
        sig_df = sig.sig
        # Number of lags: PATTERN_LOOKBACK_DURATION seconds:
        num_lags = int(AudioSegmenter.PATTERN_LOOKBACK_DURATION / SignalAnalyzer.spectro_timeframe_width())

        for col_name in col_names:
            
            self.log.info(f"Autocorrelation for {col_name}")
            
            measures = sig_df[col_name]
            acorr_res = SignalAnalyzer.autocorrelation(measures, nlags=num_lags)
            acorr_res.columns = [f"{col_name}_{acorr_col}"
                                 for acorr_col
                                 in acorr_res.columns]
            res_df = pd.concat([res_df, acorr_res], axis=1)
            
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
        
        sig_vals_df.index = sig_vals_df.index.astype(float).to_series().round(5)
        sig_vals_df['is_sel_start'] = False
        sig_vals_df['is_sel_stop'] = False
        
        snapped_starts = {}
        snapped_stops  = {}

        for start, stop in zip(start_times, stop_times):

            snapped_start = Utils.nearest_in_array(sig_vals_df.index,
                                                   round(start,5),
                                                   is_sorted=True)
            snapped_starts[start] = snapped_start
            sig_vals_df.loc[snapped_start, 'is_sel_start'] = True
    
            snapped_stop = Utils.nearest_in_array(sig_vals_df.index,
                                                  round(stop,5),
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


# ------------------------ Main ------------
if __name__ == '__main__':
    exp = ExperimentManager(EXP_ROOT)
    segmenter = AudioSegmenter(exp,
                               SPECTRO_PATH,
                               AUDIO_PATH,
                               selection_tbl_path=TEST_SEL_TBL)
    # Slice spectro into horizontal bands:
    slice_arr = segmenter.slice_spectro()
    # For each horizontal slice, compute signatures separately:
    for spectro_slice in slice_arr:
        slice_sig = segmenter.compute_sig_whole_spectro(spectro_slice)
        spectro_slice.sig = slice_sig
        #*********
        #acorr = segmenter.compute_autocorrelations(spectro_slice.sig, ['flatness'])
        acorr = segmenter.compute_autocorrelations(spectro_slice.sig, ['flatness', 'pitch'])
        #*********
        segmenter.save_sel_tbl_and_sigs(spectro_slice, segmenter.sel_tbl)
        acorrs = segmenter.compute_autocorrelations(spectro_slice.sig, 
                                                    ['flatness', 'continuity', 'pitch', 'freq_mod', 'energy_sum'])
        acorr_key = f"autocorrelations_{spectro_slice.name}"  
        exp.save(acorr_key, acorrs, index_col='lag')
        