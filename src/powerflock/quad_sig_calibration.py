#!/usr/bin/env python
'''
Created on Dec 20, 2021

@author: paepcke
'''
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys

from experiment_manager.experiment_manager import JsonDumpableMixin, \
    ExperimentManager, Datatype
from logging_service import LoggingService

from data_augmentation.utils import Utils, Interval, RavenSelectionTable
import numpy as np
import pandas as pd
from powerflock.power_evaluation import PowerEvaluator, Action
from powerflock.power_member import PowerResult, PowerMember
from powerflock.signal_analysis import SignalAnalyzer
from powerflock.signatures import Signature, SpectralTemplate

from result_analysis.charting import Charter

class QuadSigCalibrator(JsonDumpableMixin):
    '''
    Given recordings of example calls for one or more species, 
    creates signature templates for each species. Stores the
    combined json file on disk for use in subsequent analyses. 
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, 
                 species=None, 
                 cal_data_root=None,
                 cal_outdir=None,
                 experiment=None,
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
                  'freq_mod'  : ...,
                  'flatness'   : ...,
                  'continuity' : ...
                  }
                  
             The json files will have the same name as the species subdirectories. 

        The result is saved in one or two places:
            o A json representation of the templates dict is
              saved in <dir-of-this-file>/species_calibration_data/signatures.json
              Recover as QuadSigCalibrator.json_load(<...signatures.json>)
            o If an experiment instance is provided, the result
              is additionally saved as experiment.save('templates') as
              json. Recover using experiment.read('signatures', QuadSigCalibrator)
        
        If an experiment is provided, the signatures stored
        there will take precedence over the signatures.json under dir-of-this-file.
        Meaning that the experiment version will be loaded, modified 
        and then written both to the experiment and the signatures.json file.
        So signatures.json will track what is in the experiment.

        :param species: individual or list of (for example) five-letter species
            names. The species names must match the subdirectories under
            cal_data_root. Default: all species under cal_data_root.
        :type species: {None | str | [str]}
        :param cal_data_root: root of directory with the species specific
            subdirectories of selection table and sound files
        :type cal_data_root: {None | str}
        :param cal_outdir: directory for result json files
        :type cal_outdir: {None | str}
        :param experiment: optional name or ExperimentManager instance of 
            experiment where to save the dict of SpectralTemplates
        :type experiment {None | str | ExperimentManager}
        :param unittesting: if True, return without initializing anything
        :type unittesting: bool
        '''
        
        self.log = LoggingService()

        if unittesting:
            return

        if type(experiment) == str:
            # Experiment is root of experiment info;
            # create an experiment:
            experiment = ExperimentManager(experiment)
        # Else experiment must be None or an ExperimentManager instance
        if experiment is not None and not isinstance(experiment, ExperimentManager):
            raise TypeError(f"Experiment arg must be None, the root path of an experiment, or an ExperimentManager instance; not {type(experiment)}")
        self.experiment = experiment

        self.cur_dir = os.path.dirname(__file__)
        
        if cal_data_root is None:
            cal_data_root = os.path.join(self.cur_dir, 'species_calibration_data')
            
        if cal_outdir is None:
            # Put json of finished calibration into 
            # root of the sound/sel-table files: 
            cal_outdir = cal_data_root

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
            
        # For remembering the call samples and selection table files;
        # will be like:
        #     {species : {'sound'   : <full soundfilepath>,
        #                 'sel_tbl' : <full select table path>
        #         ...
        #     }
        self.cal_files = {species : {} for species in self.species_list}
        # Fill the dict:
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
                    # Remember the sel file path for later:
                    self.cal_files[species]['sel_tbl'] = sel_tbl_file
                else:
                    sound_file = os.path.join(species_dir, fname)
                    # Remember the sound file path for later:
                    self.cal_files[species]['sound'] = sound_file

    #------------------------------------
    # make_species_templates 
    #-------------------

    def make_species_templates(self):

        # Place to hold the results for each species:
        templates = {}

        for species in self.species_list:
            # Calibrate the four per-timeframe power spectrum values:
            sound_file = self.cal_files[species]['sound']
            sel_tbl_file = self.cal_files[species]['sel_tbl']
            templates[species] = self.template_for_one_species(sound_file, sel_tbl_file)

        # If cal_data_root already has a species signature 
        # json file, load it, and update it with the current
        # run's result:
        signatures_fname = os.path.join(self.cal_data_root, 'signatures.json')

        if self.experiment is None:
            if os.path.exists(signatures_fname):
                try:
                    cur_templates = self.json_load(signatures_fname)
                except Exception as e:
                    self.log.err(f"Loading current sigs from {signatures_fname} failed: {repr(e)}...")
                    self.log.err(f"    ... Moving that file to .bak")
                    shutil.move(signatures_fname, signatures_fname+'.bak')
                    cur_templates = {}
            else:
                cur_templates = {}
        else:
            # Look in the experiment for the dict of
            # templates
            try:
                cur_templates = self.experiment.read('templates', QuadSigCalibrator)
            except FileNotFoundError:
                cur_templates = {}

        # Update current sigs with results from this run:
        self.log.info(f"Saving/updating all sigs to {signatures_fname}")
        cur_templates.update(templates)
        self.cur_templates = cur_templates
        # Save the template:
        self.json_dump(signatures_fname)
        # Additionally, save in experiment if one was provided:
        if self.experiment is not None:
            self.log.info(f"Saving/updating all sigs to 'templates' in experiment")
            self.experiment.save('templates', self)
        
        self.signatures_fname = signatures_fname
        return cur_templates

    #------------------------------------
    # template_for_one_species
    #-------------------
    
    def template_for_one_species(self, sound_file, sel_tbl_file, extract=True):
        '''
        Go through each call of one species, and create
        a Signature instance. Wrap the Signature instances
        in a SpectralTemplate, and return that SpectralTemplate 
        
        Timing: for a sound file with 17 calls the method
           takes about 1:10 minutes. 
        
        :param sound_file: example sounds 
        :type sound_file: str
        :param sel_tbl_file: Raven selection table corresponding
            to that file.
        :type sel_tbl_file: str
        :param extract: Whether or not to compute the signatures
            on the full height of the spectrogram, or only the
            spectrograms clipped around calls
        :type extract: bool
        :return: template with signatures
        :rtype: SpectralTemplate
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
        
        # Width of frequencies along y axis:
        freq_step_size = np.abs(spec_df.index[1] - spec_df.index[0])
        
        # List of low/high frequency intervals for each call:
        freq_intervals = [Interval(low_f, high_f, freq_step_size)
                          for low_f, high_f
                          in zip(low_freqs, high_freqs)]
        
        
        pitch_list    = []
        freq_mod_list  = []
        flatness_list   = []
        continuity_list = []
        
        sig_instance_list = []

        #************
        # To shorten testing cycle, 
        # can use the [0:2] version here.
        for call_num, selection in enumerate(selection_dicts):
        #for call_num, selection in enumerate(selection_dicts[0:2]):
        #************
            start_time = selection['Begin Time (s)']
            end_time   = selection['End Time (s)']

            nearest_spec_start_time = power_df.columns[power_df.columns >= start_time][0]
            nearest_spec_end_time   = power_df.columns[power_df.columns < end_time][-1]
            spec_snip = power_df.loc[:,nearest_spec_start_time : nearest_spec_end_time]
            
            # Change the spectrum snippet's times (the columns)
            # to be relative to the start of the call, rather than
            # the beginning of the file:
            spec_snip.columns = spec_snip.columns - nearest_spec_start_time
            
            # Each result will be a Series of measures across
            # one call's time duration:
            self.log.info(f"... call {species}:{call_num} harmonic pitch")
            pitch = SignalAnalyzer.harmonic_pitch(spec_snip)
            self.log.info(f"... call {species}:{call_num} frequency modulation")
            freq_mod = SignalAnalyzer.freq_modulations(spec_snip)
            self.log.info(f"... call {species}:{call_num} spectral flatness")
            flatness = SignalAnalyzer.spectral_flatness(spec_snip, is_power=True)
            self.log.info(f"... call {species}:{call_num}  spectral continuity")
            _long_contours_df, continuity = SignalAnalyzer.spectral_continuity(spec_snip, 
                                                                               is_power=True,
                                                                               plot_contours=False
                                                                               ) 
            # Make a df from the above four measures:
            sig = pd.DataFrame(
                [flatness, continuity, pitch, freq_mod],
                columns=pitch.index, # any of the indexes will do
                index=['flatness', 'continuity', 'pitch', 'freq_mod']
                ).transpose()

            # Make a signature instance, leaving the scale_info
            # uninitialized, because we don't know that values yet:
            
            sig_inst = Signature(species=species,
                                 sig_values=sig,
                                 start_idx=nearest_spec_start_time,
                                 end_idx=nearest_spec_end_time,
                                 fname=sound_file,
                                 sig_id=(call_num + 1), # want sig-ids 1-origin to match Raven table rows
                                 freq_interval=freq_intervals[call_num],
                                 bandpass_filter=passband,
                                 extract=extract
                                 )
            sig_instance_list.append(sig_inst)
            
            # Keep track of the measures series for later,
            # when we will need to compute the median of deviations
            # from the mean for each of them across all calls:
            pitch_list.append(pitch)
            freq_mod_list.append(freq_mod)
            flatness_list.append(flatness)
            continuity_list.append(continuity)
            
        # For each measure we now have an array of series,
        # each series for one call, with one element for each 
        # timeframe. Concatenate the series of each measure
        # to get the measure across all calls for computing
        # an appropriate scaling factor (see below):

        flatness   = pd.concat(flatness_list)
        continuity = pd.concat(continuity_list)
        pitch    = pd.concat(pitch_list)
        freq_mod  = pd.concat(freq_mod_list)
        
        flatness_mean = flatness.mean()
        continuity_mean = continuity.mean()
        pitch_mean = pitch.mean()
        freq_mod_mean = freq_mod.mean()

        # Compute median distance from mean across
        # all calls; the '.item()' turns the resulting 
        # np.float64 into a Python native float:
        scale_info = {
            'flatness' : {'mean' : flatness_mean.item(),
                          'standard_measure' : np.abs((flatness - flatness_mean).median()).item() 
                          },
            'continuity' : {'mean' : continuity_mean.item(),
                            'standard_measure' : np.abs((continuity - continuity_mean).median()).item() 
                            },
            'pitch' : {'mean' : pitch_mean.item(),
                       'standard_measure' : np.abs((pitch - pitch_mean).median()).item() 
                       },

            'freq_mod' : {'mean' : freq_mod_mean.item(),
                        'standard_measure' : np.abs((freq_mod - freq_mod_mean).median()).item()
                        }
            }
        
        # Set these scale factors in all 
        # the signature instances we colleced:
        for sig_inst in sig_instance_list:
            sig_inst.normalize_self(scale_info)

        # Wrap the Signature instances in a 
        # SpectralTemplate:
        
        template = SpectralTemplate(sig_instance_list, rec_fname=sound_file)
        
        return template

    #------------------------------------
    # calibrate_templates
    #-------------------
    
    def calibrate_templates(self, species_list):
        '''
        For each template, run an audio file
        analysis of the audio from which its 
        signatures were created. Then, for each
        signature, find the probability and prominence
        thresholds that yield best results for that
        signature's vocalization.
        
        This step is required because while probability
        spikes are clear, the scales of the peaks vary
        as much as 0.00123 to 0.5 among calls.
        
        :param species_list: if provided, a single, or a list
            of species which to calibrate. If None, all in
            self.cur_templates will be calibrated.
        :type species_list: {None | str | [str]}
        '''
        
        if type(species_list) != list:
            species_list = [species_list]
            
        for species, template in self.cur_templates.items():
            # Skip any unwanted species:
            if species_list is not None and species not in species_list:
                continue
            self._calibrate_one_template(species, template)
            
    #------------------------------------
    # _calibrate_one_template
    #-------------------
    
    def _calibrate_one_template(self, species, template):

        samples_sound_file = self.cal_files[species]['sound']
        sel_tbl_file = self.cal_files[species]['sel_tbl']

        # Is there already a PowerResult for a self test
        # of this species? Files look like:
        #    PwrRes_2022-01-21T16_41_54_species_BANAS.json
        pwr_res_info = None
        pattern = re.compile(r"PwrRes_[^s]*species_([a-zA-Z]{5}).*")
        for fname in self.experiment.listdir(PowerResult):
            match_obj = pattern.match(fname)
            if match_obj is not None:
                # Found the power result:
                if match_obj.group(1).upper() == species.upper():
                    pwr_res_info = fname
                    break
        # No existing power result was found, create
        # one by setting the action to ANALYSIS. Else,
        # just initialize a PowerEvaluator instance:
        evaluator = PowerEvaluator(self.experiment,
                                   species,
                                   Action.ANALYSIS if pwr_res_info is None else Action.NOOP,
                                   power_result_info=pwr_res_info,
                                   test_recording=samples_sound_file,
                                   test_sel_tbl=sel_tbl_file
                                   )
        pwr_member = PowerMember(
            species,
            spectral_template_info=template,
            experiment = self.experiment
            )         

        pwr_res = evaluator.pwr_res
        sel_tbl = RavenSelectionTable(sel_tbl_file)
        pwr_res.add_truth(sel_tbl)
        
        # Get time intervals of known vocalizations
        # from the selection table:
        true_vocalization_intervals = [sel_tbl_entry.time_interval 
                                       for sel_tbl_entry 
                                       in sel_tbl.entries]
        # Number of true vocalizations 
        num_true_vocalizations = len(true_vocalization_intervals)
        
        # Go through each signature...
        for sig_idx, sig in enumerate(template.signatures):
            sig_id = sig.sig_id
            # The time interval from which this signature
            # was created:
            sig_time_interval = sel_tbl.entries[sig_idx].time_interval
            
            # Go through progressive prominences till we find
            # the first that finds this signature's peak, while 
            # still just maintaining a precision of 1.0
            lo_prom = 0.
            hi_prom = 1.
            while hi_prom - lo_prom > 0.001:
                mid_prom = lo_prom + (hi_prom - lo_prom) / 2.0
                precision, true_positives = self._try_prominence(
                            mid_prom,
                            pwr_res, 
                            pwr_member, 
                            sig_id,
                            sig_time_interval, 
                            true_vocalization_intervals)
                if precision in [-1, 1]:
                    # Did not find the peak at all,
                    # or precision was 
                    # Need to lower the prominence
                    hi_prom = mid_prom
                else:
                    # Precision was less than 1, need
                    # to up the prominence a bit:
                    lo_prom = mid_prom + 0.001
            if true_positives is None:
                sig.usable = False
            else:
                sig.usable = True
                sig.prominence_threshold = hi_prom
                sig.recall    = true_positives / num_true_vocalizations
                sig.precision = precision
                
            # Loop for next sig

        for sig in template.signatures:
            if sig.usable:
                print(f"prom {sig.sig_id}: {sig.prominence_threshold}; prec: {sig.precision}; rec: {sig.recall}")
            else:
                print(f"prom {sig.sig_id}: not usable") 
        print('foo')

    #------------------------------------
    # _try_prominence
    #-------------------
    
    def _try_prominence(self, 
                        prominence, 
                        pwr_res, 
                        pwr_member, 
                        sig_id,
                        sig_time_interval, 
                        true_vocalization_intervals):
    
        _consensus_peaks, sig_peak_times = \
           pwr_member.find_calls(pwr_res,prominence_threshold=prominence)
    
        peak_times_this_sig = sig_peak_times[sig_id][sig_peak_times[sig_id]]
        if sig_peak_times[sig_id].sum().sum() == 0:
            # No peaks found at all
            return -1, None
    
        # Were any of the peaks for *this* sig? (That's
        # what we are looking for):
        found_this_sig_peak = False
        for center_time in peak_times_this_sig.index:
            if center_time in sig_time_interval:
                found_this_sig_peak = True
                break
        if not found_this_sig_peak:
            # Found some peaks, but not the one
            # for this sig; need to lower prominence
            # threshold:
            return -1, None
    
        # Found peak for this sig; compute precision:
        tps = 0
        for found_peak_time in peak_times_this_sig.index:
            for true_time_interval in true_vocalization_intervals:
                if found_peak_time in true_time_interval:
                    tps += 1
    
        precision = tps / len(peak_times_this_sig)
        return precision, tps


    # ---------------------- Utilities ---------------
    
    #------------------------------------
    # json_dump
    #-------------------
    
    def json_dump(self, fname):
        '''
        Given that self.cur_templates is a dict of 
        SpectralTemplate values like:
            {'CMTOG' : template_cmtog,
             'OtherSpec' : template_other_spec
             },
             
        ensure that all values are individually turned 
        into json, and then write the entire dict to fname 
        as a json file.
        
        :param sigs: nested dict of signatures
        :type sigs: {str : {str : ANY}}
        :param fname: destination path
        :type fname: str
        '''
        with open(fname, 'w') as fd:
            json.dump({key : val.json_dumps() for key,val in self.cur_templates.items()}, fd)

    #------------------------------------
    # json_load 
    #-------------------
    
    @classmethod
    def json_load(cls, fname):
        '''
        Reconstruct a dict of templates from
        the given json file. All SpectralTemplate
        instances will be reconstructed.
        The result will look like:
        
            {'CMTOG' : template_cmtog,
             'OtherSpec' : template_other_spec
             }
        
        :param fname: json file to load
        :type fname: str
        :return dict of SpectralTemplate
        :rtype {str : SpectralTemplate}
        '''
        
        with open(fname, 'r') as fd:
            dict_of_templates = json.load(fd)
        
        new_dict = {species : SpectralTemplate.json_loads(jstr)
                    for species, jstr
                    in dict_of_templates.items()
                    }

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
                                     description="Create signature templates for one or more species"
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
    
    parser.add_argument('-e', '--experiment',
                        help='root directory of experiment to be managed by ExperimentManager',
                        default=None)
    
    parser.add_argument('-o', '--outdir',
                        help=("where to place the template(s); \n"
                              "default: if experiment given, then its json_files,\n"
                              "else subdirectory of this \n"
                              "file's dir: species_calibration_results."),
                        default=None)

    args = parser.parse_args()

    calibrator = QuadSigCalibrator(args.species,
                                   cal_data_root=args.data,
                                   cal_outdir=args.outdir,
                                   experiment=args.experiment
                                   )
    #********
    #calibrator.make_species_templates()
    calibrator.cur_templates = calibrator.experiment.read('templates', QuadSigCalibrator)
    #********
    calibrator.calibrate_templates('BANAS')