'''
Created on Dec 20, 2021

@author: paepcke
'''
import argparse
import json
import os
from pathlib import Path
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
            cal_outdir = os.path.join(self.cur_dir, 'species_calibration_results')
            try:
                os.makedirs(cal_outdir)
            except FileExistsError:
                pass
            
        if species is None:
            # Process all species: expect the directory
            # names in cal_data_root to be species names:
            species_list = os.listdir(cal_data_root)
        elif type(species) != list:
            species_list = [species]
        else:
            species_list = species
            
        for species in species_list:
            # The subdir with species sound and selection table file:
            species_dir = os.path.join(cal_data_root, species)
            
            # From the two expected files, identify the
            # selection table and the sound file: 
            for fname in os.listdir(species_dir):
                self.log.info(f"Checking samples for {fname}")
                if fname.endswith('.txt'):
                    sel_tbl_file = os.path.join(cal_data_root, fname)
                else:
                    sound_file = os.path.join(cal_data_root, fname) 

            # Ensure an output dir for this species exists:
            try:
                outdir = os.path.join(cal_outdir, species) 
                os.makedirs(outdir)
                self.log.info(f"Created output dir {outdir}")
            except FileExistsError:
                pass
             
            # Calibrate the four per-timeframe power spectrum values:
            self.species_scales[species] = self.calibrate_species(sound_file, sel_tbl_file, outdir)
            
            with open(os.path.join(outdir, species), 'w') as fd:
                json.dump(self.species_scales, fd)

    #------------------------------------
    # calibrate_species
    #-------------------
    
    def calibrate_species(self, sound_file, sel_tbl_file, extract=True):
        '''
        Return a dict with unit-less scale factors for
        all four power measures:
        
            scales = {
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
        # Concat the series of a each kind:
        
        pitches    = pd.concat(pitch_list)
        freq_mods  = pd.concat(freq_mods_list) 
        flatness   = pd.concat(flatness_list)
        continuity = pd.concat(continuity_list)

        # Compute median distance from mean; the
        # '.item()' turns the resulting np.float64
        # into a Python native float:
        scales = {
            'pitch'      : np.abs((pitches - pitches.mean()).median()).item(),
            'freq_mods'  : np.abs((freq_mods - freq_mods.mean()).median()).item(),
            'flatness'   : np.abs((flatness - flatness.mean()).median()).item(),
            'continuity' : np.abs((continuity - continuity.mean()).median()).item()
        }
        return scales
    
# ------------------------ Main ------------
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