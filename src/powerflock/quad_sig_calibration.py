'''
Created on Dec 20, 2021

@author: paepcke
'''
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from logging_service import LoggingService

from data_augmentation.utils import Utils, Interval
import numpy as np
import pandas as pd
from powerflock.signal_analysis import SignalAnalyzer


class QuadSigCalibrator:
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, 
                 species=None, 
                 cal_data_root=None,
                 cal_outdir=None,
                 unittesting=False):
        
        '''
        For each species, use ~10 clean calls to establish
        typical distributions of harmonic pitch, frequency
        modulation, percentage of frequencies with contours,
        and spectral flatness. Each quantity is computed as
        one number for each timeframe. Then, a single unitless
        number is computed for each of the four measures. It  
        is a unit of change figured as the median distance of 
        the measure from its mean within each time frame.
        
        Assumptions for default arguments:
           o for cal_data_root None, a subdirectory called 
             species_calibration_data below this file's dir
             is assumed to hold one subdirectory for each species.
             Within each subdirectory two files are expected. One
             with extension .txt, which must be a Raven selection
             table, and another, which must be a sound file.
             
             The sound file is assumed to hold a series of calls
             by one species, labeled via the selection table. 

           o for species None, all species with subdirectories in
             cal_data_root are processed
             
           o cal_outdir is the directory where one json file
             is written with the four results of one species.
             The dict will be of form:
             
                {
                  'pitch'      : ...,
                  'freq_mods'  : ...,
                  'flatness'   : ...,
                  'continuity' : ...
                  }
                  
             The json files will have the same name as the species subdirectories. 
              
        :param species: individual or list of (for example) five-letter species
            names. The species names must match the subdirectories under
            cal_data_root. Default: all species under cal_data_root.
        :type species: {None | str | [str]}
        :param cal_data_root: root of directory with the species specific
            subdirectories of selection table and sound files
        :type cal_data_root: {None | str}
        :param cal_outdir: directory for result json files
        :type cal_outdir: {None | str}
        :param unittesting: if True, return without initializing anything
        :type unittesting: bool
        '''
        
        self.log = LoggingService()

        if unittesting:
            return

        self.cur_dir = os.path.dirname(__file__)
        
        if cal_data_root is None:
            cal_data_root = os.path.join(self.cur_dir, 'species_calibration_data')
            
        if cal_outdir is None:
            # Put json of finished calibration into 
            # root of the sound/sel-table files: 
            cal_outdir = os.path.join(cal_data_root, 'species_calibration_results')
            try:
                if not os.path.exists(cal_outdir):
                    os.makedirs(cal_outdir)
            except FileExistsError:
                pass

        self.cal_data_root = cal_data_root
        self.cal_outdir = cal_outdir

        if species is None:
            # Process all species: expect the directory
            # names in cal_data_root to be species names.
            # Ignore non-directory files:
            self.species_list = [entry 
                            for entry in os.listdir(cal_data_root) 
                            if os.path.isdir(os.path.join(cal_data_root, entry))] 
        elif type(species) != list:
            self.species_list = [species]
        else:
            self.species_list = species

    #------------------------------------
    # calibrate_species 
    #-------------------

    def calibrate_species(self):

        # Place to hold the results for each species:
        species_scales = {}

        for species in self.species_list:
            # The subdir with species sound and selection table file:
            species_dir = os.path.join(self.cal_data_root, species)
            
            # From the two expected files, identify the
            # selection table and the sound file;
            # Sanity check: should only have two files in
            # species calibrations data dir: a sound file and a sel tbl:
 
            for file_no, fname in enumerate(os.listdir(species_dir)):
                if file_no > 1:
                    raise ValueError(f"Should only have one sound file and one selection table file in {species_dir}")
                self.log.info(f"Checking samples for {fname}")
                if fname.endswith('.txt'):
                    sel_tbl_file = os.path.join(species_dir, fname)
                else:
                    sound_file = os.path.join(species_dir, fname)
                    
             
            # Calibrate the four per-timeframe power spectrum values:
            species_scales[species] = self.calibrate_one_species(sound_file, sel_tbl_file)

        # If cal_data_root already has a species signature 
        # json file, load it, and update it with the current
        # run's result:
        signatures_fname = os.path.join(self.cal_data_root, 'signatures.json')
        if os.path.exists(signatures_fname):
            try:
                cur_sigs = self.sigs_json_load(signatures_fname)
            except Exception as e:
                self.log.err(f"Loading current sigs from {signatures_fname} failed: {repr(e)}...")
                self.log.err(f"    ... Moving that file to .bak")
                shutil.move(signatures_fname, signatures_fname+'.bak')
                cur_sigs = {}
        else:
            cur_sigs = {}
        # Update current sigs with results from this run:
        self.log.info(f"Saving/updating all sigs to {signatures_fname}")
        cur_sigs.update(species_scales)
        self.sigs_json_dump(cur_sigs, signatures_fname)
        
        self.signatures_fname = signatures_fname
        return cur_sigs

    #------------------------------------
    # calibrate_one_species
    #-------------------
    
    def calibrate_one_species(self, sound_file, sel_tbl_file, extract=True):
        '''
        Return a dict with unit-less scale factors for
        all four power measures:
        
            sig = {
                'pitch'      : np.abs((pitches - pitches.mean()).median()),
                'freq_mods'  : np.abs((freq_mods - freq_mods.mean()).median()),
                'flatness'   : np.abs((flatness - flatness.mean()).median()),
                'continuity' : np.abs((continuity - continuity.mean()).median())
            }
        
        Timing: for a sound file with 17 calls the method
           takes about 1:10 minutes. 
        
        :param sound_file:
        :type sound_file:
        :param sel_tbl_file:
        :type sel_tbl_file:
        :param extract:
        :type extract:
        '''
        
        # For informative logging: find species name
        # from parent dir:
        species = Path(sound_file).stem
        
        selection_dicts = Utils.read_raven_selection_table(sel_tbl_file)
        
        self.log.info(f"Processing {len(selection_dicts)} calls for species {species}")
        
        low_freqs  = [sel['Low Freq (Hz)'] for sel in selection_dicts]
        high_freqs = [sel['High Freq (Hz)'] for sel in selection_dicts]
        max_freq   = max(high_freqs) 
        min_freq   = min(low_freqs)
        
        passband   = Interval(min_freq, max_freq) 
        spec_df = SignalAnalyzer.raven_spectrogram(sound_file, extra_granularity=True)
        
        spec_df_clipped = SignalAnalyzer.apply_bandpass(passband, spec_df, extract=extract)
        power_df = spec_df_clipped ** 2
        
        pitch_list    = []
        freq_mods_list  = []
        flatness_list   = []
        continuity_list = []

        for call_num, selection in enumerate(selection_dicts):
            start_time = selection['Begin Time (s)']
            end_time   = selection['End Time (s)']

            nearest_spec_start_time = power_df.columns[power_df.columns >= start_time][0]
            nearest_spec_end_time   = power_df.columns[power_df.columns < end_time][-1]
            spec_snip = power_df.loc[:,nearest_spec_start_time : nearest_spec_end_time]
            
            self.log.info(f"... call {species}:{call_num} harmonic pitch")
            pitch_list.append(SignalAnalyzer.harmonic_pitch(spec_snip))
            self.log.info(f"... call {species}:{call_num} frequency modulation")
            freq_mods_list.append(SignalAnalyzer.freq_modulations(spec_snip))
            self.log.info(f"... call {species}:{call_num} spectral flatness")
            flatness_list.append(SignalAnalyzer.spectral_flatness(spec_snip, is_power=True))
            self.log.info(f"... call {species}:{call_num}  spectral continuity")
            _long_contours_df, continuity = SignalAnalyzer.spectral_continuity(spec_snip, 
                                                                               is_power=True,
                                                                               plot_contours=False
                                                                               ) 
            continuity_list.append(continuity)
            

        # For each measure we now have an array of series,
        # each series with one element for each timeframe.
        # Concat the series of a each kind. In the process,
        # remove entries that have the same index:
        
        pitches    = self._dedup_series(pd.concat(pitch_list))
        freq_mods  = self._dedup_series(pd.concat(freq_mods_list))
        flatness   = self._dedup_series(pd.concat(flatness_list))
        continuity =self._dedup_series( pd.concat(continuity_list))
        
        # Compute median distance from mean; the
        # '.item()' turns the resulting np.float64
        # into a Python native float:
        sig = {
            'species'    : species,
            
            'pitch'      : pitches,
            'freq_mods'  : freq_mods,
            'flatness'   : flatness,
            'continuity' : continuity,

            'pitch_scale'      : np.abs((pitches - pitches.mean()).median()).item(),
            'freq_mods_scale'  : np.abs((freq_mods - freq_mods.mean()).median()).item(),
            'flatness_scale'   : np.abs((flatness - flatness.mean()).median()).item(),
            'continuity_scale' : np.abs((continuity - continuity.mean()).median()).item()
        }
        return sig

    #------------------------------------
    # _sigs_to_json 
    #-------------------
    
    def _sigs_to_json(self, sig_dict):
        new_dict = sig_dict.copy()
        new_dict['pitch'] = sig_dict['pitch'].to_json()
        new_dict['freq_mods'] = sig_dict['freq_mods'].to_json()
        new_dict['flatness'] = sig_dict['flatness'].to_json()
        new_dict['continuity'] = sig_dict['continuity'].to_json()
        return new_dict

    #------------------------------------
    # sigs_json_dump
    #-------------------
    
    def sigs_json_dump(self, sigs, fname):
        '''
        Given a dict of signatures like:
            {'CMTOG' : {'species' : 'CMTOG',
                        'pitch' : ...
                           ...
                       },
             'OtherSpec' : {...}
             },
             
        ensure that all values that are pd.Series
        are individually turned into json, and then
        write the entire nested dir to fname as a json
        file.
        
        :param sigs: nested dict of signatures
        :type sigs: {str : {str : ANY}}
        :param fname: destination path
        :type fname: str
        '''
        new_dict = {}
        for species, sig in sigs.items():
            new_dict[species] = self._sigs_to_json(sig)
        with open(fname, 'w') as fd:
            json.dump(new_dict, fd)


    #------------------------------------
    # sigs_json_load 
    #-------------------
    
    def sigs_json_load(self, fname):
        '''
        Reconstruct a dict of signatures from
        the given json file. All nested pd.Series
        will be reconstructed to be pd.Series again,
        though they are represented as nested json
        by sigs_json_dump. The result will look like:
        
            {'CMTOG' : {'species' : 'CMTOG',
                        'pitch' : ...
                           ...
                       },
             'OtherSpec' : {...}
             }
        
        :param fname: json file to load
        :type fname: str
        :return dict of signatures
        :rtype {str : {str : ANY}}
        '''
        
        with open(fname, 'r') as fd:
            new_dict = json.load(fd)
        
        for species, sig in new_dict.items():
            # Turn series into into pd.Series instances:
            sig['pitch'] = pd.Series(self.safe_eval(sig['pitch']))
            sig['freq_mods'] = pd.Series(self.safe_eval(sig['freq_mods']))
            sig['flatness'] = pd.Series(self.safe_eval(sig['flatness']))
            sig['continuity'] = pd.Series(self.safe_eval(sig['continuity']))
            new_dict[species] = sig 

        return new_dict

    #------------------------------------
    # safe_eval
    #-------------------
    
    def safe_eval(self, expr_str):
        '''
        Given a string, evaluate it as a Python
        expression. But do it safely by making
        almost all Python functions unavailable
        during the eval.

        :param expr_str: string to evaluate
        :type expr_str: str
        :return Python expression result
        :rtype Any
        '''
        res = eval(expr_str,
                   {"__builtins__":None},    # No built-ins at all
                   {}                        # No additional func
                   )
        return res
        
                
        


    #------------------------------------
    # _dedup_series
    #-------------------
    
    def _dedup_series(self, ser):
        '''
        Given a pd.Series, return a new series
        in which entries with duplicate index entries
        are deleted.
        
        :param ser: input series
        :type ser: pd.Series
        :return new series with no duplicates in the index
        :rtype pd.Series
        '''
        
        ser_dedup = ser[~ser.index.duplicated(keep='first')]
        return ser_dedup

# ------new_dict['------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Calibrate quad power for one or more species"
                                     )

    parser.add_argument('-s', '--species',
                        type=str,
                        nargs='+',
                        help='Repeatable: five-letter species code; default all in data dir',
                        default=None)

    parser.add_argument('-d', '--data',
                        help='fully qualified path to root of example calls and selection tables; \n' +\
                             "default: subdirectory of this file's dir: species_calibration_data.",
                        default=None)
    
    parser.add_argument('-o', '--outdir',
                        help='where to place the json files with calibration numbers; \n' +\
                             "default: subdirectory of this file's dir: species_calibration_results.",
                        default=None)

    args = parser.parse_args()

    QuadSigCalibrator(args.species,
                      args.data,
                      args.outdir 
                      )