#!/usr/bin/env python
'''
Created on Nov 5, 2021

@author: paepcke
'''
import argparse
from bisect import bisect_left
from datetime import datetime, timedelta
from enum import Enum
import os
from pathlib import Path
import sys

from experiment_manager.experiment_manager import ExperimentManager
from logging_service.logging_service import LoggingService
from matplotlib.patches import Rectangle

from birdsong.utils.utilities import FileUtils
from data_augmentation.utils import Interval, RavenSelectionTable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from powerflock.matplotlib_crosshair_cursor import CrosshairCursor
from powerflock.power_member import PowerMember, PowerQuantileClassifier, \
    PowerResult
from powerflock.signal_analysis import SignalAnalyzer
from powerflock.signatures import SpectralTemplate, TemplateCollection
from result_analysis.charting import Charter


class Action(Enum):
    UNITTEST = -1
    NOOP = 0
    ANALYSIS = 1
    SCORE = 2
    VIZ_PROBS = 3
    GRID_SEARCH = 4

class PowerEvaluator:
    '''
    Use an instance of this class to:
        o Analyze an audio file and find vocalizations of
          one species
        o Get the information retrieval scores for
          the result of an analysis
        o Visualize the probabilities that were computed
          in an analysis as a line chart

    Before doing any of these, quad_sig_calibration must be
    run once to compute SpectralTemplates for the species of
    interest.
    
    The output of an analysis is a PowerResult, which is then
    used for any of the other actions above.
    
    The class uses ExperimentManager to store and retrieve
    results.
    '''
    
    DEFAULT_EXPERIMENT_DIR = 'experiments/PowerSignatures'
    '''Root of ExperimentManager instances relative to project root'''
    
    # Probability thresholds for use in result_analysis
    THRESHOLDS = [0.6, 0.7, 0.8, 0.9]
    PROMINENCE_THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    DEFAULT_PROBABILITY_THRESHOLD = 0.6
    DEFAULT_PERCENTAGE_AGREEMENT  = .5
    DEFAULT_PROMINENCE_THRESHOLD  = 0.3
    SLIDE_WIDTH = 0.05 # fraction of signature
    SIG_ID = 3
    
    # Default font size of axis tick labels 
    # and axis labels in figures. Super titles 
    # will be one pt higher, subtitles will be 
    # 2pt lower
    AX_LBL_FONTSIZE = 16
    
    TRUTH_RECT_HEIGHTS = 500 # Hz
    '''Height of rectangles that indicate location of true bird calls in viz_probs'''
    
    #------------------------------------
    # Constructor
    #-------------------


    def __init__(self,
                 experiment_info,
                 species, 
                 actions,
                 templates_info=None,
                 power_result_info=None,
                 test_recording=None,
                 test_sel_tbl=None,
                 prominence_thresholds=None,
                 probability_thresholds=None,
                 outfigs=False
                 ):
        '''
        
        Notes on some parameters:
        
        The EXPERIMENT_INFO will result in an ExperimentManager instance.
        Either a new one, or the result of loading an existing one from
        disk. Options for experiment_info:
           o A ready-made ExperimentManager instance
           o A path relative to PowerEvaluator.DEFAULT_EXPERIMENT_DIR
           
        The TEMPLATES_INFO arg will result in a SpectralTemplate instance.
        Options for templates_info:
        
           o A ready-made SpectralTemplate instance
           o None, in which case attempts to read a template from 
             the given experiment. Tries these keys in order:
                 * 'templates_calibrated' for the result of a prior\
                   run of quad_sig_calibration.
                 * 'templates_info' for non-calibrated template
             Uses one of these keys to pull a TemplateCollection from
             the experiment. TemplateCollection instances behave like
             a dict mapping species to a SpectralTemplate.
           o String: 
                 * if the string is the path to a json-encoded TemplateCollection, 
                   that TemplateCollection is materialized, and the focus species' 
                   template is extracted.
                 * else the string is taken as an experiment root relative
                   to PowerEvaluator.DEFAULT_EXPERIMENT_DIR. A TemplateCollection
                   is extracted from that experiment as in second option above.

        The POWER_RESULT_INFO will result in a PowerResult instance that
        was the output of a prior analysis action. For analysis actions
        the power_result_info is left at None, b/c that action generates
        the instance in the first place. Options:
        
           o 'latest': the latest PowerResult by date of its key is
             retrieved from the experiment, if available. Else, the
             value is set to None.
           o A key (string) to use for extracting the PowerResult from
             the experiment. Keys are created by the analysis. The strings
             begin with 'PwrRes', followed by the date, and the species:
                PwrRes_2022-01-21T16_41_54_species_BANAS 
        
        :param experiment_info: either an ExperimentManager instance,
            or the path to the root of an experiment from which an
            ExperimentManager will then be created. The path is relative
            to PowerEvaluator.DEFAULT_EXPERIMENT_DIR
        :type experiment_info: {str | ExperimentManager}
        :param species: the species on which evaluations will focus.
            A given PowerEvaluator instance only deals with one species. 
        :type species: str
        :param actions: one or more actions to perform
        :type actions: Action
        :param templates_info: info on how to obtain a SpectralTemplate instance 
            for the given species
        :type templates_info: {str | SpectralTemplate | None}
        :param power_result_info: if provided, a PowerResult that was produced
            by an earlier analysis. If None, attempts to retrieve one from
            the experiment 
        :type power_result_info: {None | PowerResult}
        :param test_recording: the recording to be analyized if action is 
            Action.ANALYSIS
        :type test_recording: {None | str}
        :param test_sel_tbl: a Raven selection table 
        :type test_sel_tbl:
        :param prominence_thresholds:
        :type prominence_thresholds:
        :param probability_thresholds:
        :type probability_thresholds:
        :param outfigs:
        :type outfigs:
        '''
        '''
        Constructor
        '''
        self.log = LoggingService()
        
        self.species = species
        self.actions = actions if type(actions) == list else [actions]
        self.test_recording = test_recording
        
        # If given path to a Raven selection table,
        # import that table into an instance of 
        # RavenSelectionTable:
        if test_sel_tbl is not None and type(test_sel_tbl) == str:
            self.test_sel_tbl = RavenSelectionTable(test_sel_tbl)
        else:
            self.test_sel_tbl = test_sel_tbl
        
        self._init_data_paths()

        # Cache of loaded audio for visualizations
        # maps fileName --> np_array
        self.audio_dict = {}
        # Map audioFile to Raven spectrogram df
        self.spectro_dict = {}
        
        if type(experiment_info) == str:
            # The _init_data_paths() bound self.experiment_dir to 
            # the root of ExperimentManager roots. Assume that
            # experiment_info is relative to that path:  
            experiment_root = os.path.abspath(os.path.join(self.experiment_dir, 
                                                           experiment_info))
            # Create from existing, or create new:
            self.experiment = ExperimentManager(experiment_root)
        else:
            if type(experiment_info) != ExperimentManager:
                raise TypeError(f"Experiment info must be a string (name) or ExperimentManager, not {type(experiment_info)}")
            self.experiment = experiment_info
        
        # Was a PowerResult computed, and stored as json either
        # in experiment, or elsewhere on the file system, materialize
        # that PowerResult instance:
        if power_result_info == 'latest':
            # See whether the experiment has PowerResult instances:
            pwr_res = self._latest_power_result(self.experiment)
            if pwr_res is not None:
                self.log.info(f"Using PowerResult {repr(pwr_res)} ({pwr_res.name})")
        elif power_result_info is not None:
            pwr_res = self.experiment.read(power_result_info, PowerResult)
        else:
            # Will be computed
            pwr_res = None

        # Make accessible outside:
        self.pwr_res = pwr_res

        template = self._template_from_template_info(species, templates_info)

        self.template = template
        
        self.power_member = PowerMember(
            species_name=species, 
            spectral_template_info=template,
            the_slide_width_time=self.SLIDE_WIDTH,
            experiment=self.experiment
            )
        self.power_member.power_result = pwr_res
        
        for action in self.actions:
            if action == Action.UNITTEST:
                return
            elif action == Action.ANALYSIS:
                # Analysis of a recording (~ 20min processing for 1min recording):
                self.log.info("Running action 'analysis'...")
                pwr_res = self.run_analysis(pwr_member=self.power_member,
                                            rec_path=self.test_recording,
                                            pwr_res=self.pwr_res
                                            )
                self.log.info("Done running action 'analysis'.")
                 
            elif action == Action.SCORE:
                self.log.info("Running action 'score'...")
                scores = self.score(self.test_sel_tbl,
                                    pwr_res_info=pwr_res,
                                    pwr_member=self.power_member,
                                    prominence_thresholds=prominence_thresholds,
                                    probability_thresholds=probability_thresholds
                                    )
                self.log.info("Done running action 'score'.")
                
            elif action == Action.VIZ_PROBS:
                ax = self.viz_probs(self.power_member,
                                    self.test_recording,
                                    self.test_sel_tbl,
                                    self.SIG_ID,
                                    outfigs
                                    )
            elif action == Action.GRID_SEARCH:
                grid_res = self.grid_search()
                
            elif action == Action.NOOP:
                return
            else:
                raise ValueError(f"Unknown action: {action}")

        print("Done")

    #------------------------------------
    # _template_from_template_info
    #-------------------

    def _template_from_template_info(self, species, template_info):
        '''
        Given one of the possibilities callers have for 
        specifying a SpectralTemplate when instantiating
        a PowerEvaluator, this method tries to find a template,
        and returns it. If it fails, None is returned
        
        :param species: species of interest
        :type species: str
        :param template_info: see header comment of Constructor
        :type template_info: 
        '''
        if template_info is None: # Does the experiment have template_info from a
            # prior quad_sig_calibration run?
            try:
                templates_coll = self.experiment.read('templates_calibrated', TemplateCollection)
                template = templates_coll[species]
            except FileNotFoundError:
                # Calibrated signatures are required only for
                # scoring, so, accept non-calibrated ones if
                # not scoring:
                if Action.SCORE not in self.actions:
                    try:
                        templates_coll = self.experiment.read('template_info', TemplateCollection)
                        template = templates_coll[species]
                    except FileNotFoundError:
                        template = None
                else:
                    template = None
        elif isinstance(template_info, SpectralTemplate):
            template = template_info
        else:
            try:
                template_coll = TemplateCollection.json_load(template_info)
                template = template_coll[species]
            except FileNotFoundError:
                # String could be a path to an experiment that
                # contains templates_calibrated or template_info:
                experiment_root = os.path.abspath(os.path.join(self.experiment_dir, template_info))
                if os.path.exists(experiment_root) and os.path.isdir(experiment_root):
                    template_exp = ExperimentManager(experiment_root)
                    try:
                        template_coll = template_exp.read('templates_calibrated', TemplateCollection)
                        template = template_coll[species]
                    except KeyError:
                        raise KeyError(f"Found template collection at {experiment_root}, but no species {species} in collection")
                    except FileNotFoundError:
                        try:
                            template_coll = template_exp.read('template_info', TemplateCollection)
                            template = template_coll['species']
                        except FileNotFoundError:
                            template = None # See whether template_info is the path to signatures json file
        # stored by QuadSigCalibrator somewhere outside the experiment:
        return template

    #------------------------------------
    # run_analysis
    #-------------------
    
    def run_analysis(self, 
                     pwr_member, 
                     rec_path,
                     probability_thres=None,
                     percentage_agreement=None,
                     prominence_thres=None,
                     pwr_res=None
                     ):
                 
        '''
        Used to compute decisions on the presence of one
        species' articulation both by timeframe and at the
        granularity of calls. Use before running score()
        No labels are used/required in this method.
        
        Runs through a recording, generating a PowerResult instance
        that holds probabilities at each timeframe of the recording
        containing a vocalization by the power member's species.
        
        Also generates decision, both at the timeframe and call level.
        For timeframe level decisions the probability_threshold and 
        percentage_agreement are relevant:
           
           o probability_threshold is the probability of a class
             during one timeframe above which class membership is
             concluded.
           o percentage_agreement is the percentage of signatures
             that must have decided positively on class membership
             during a given timeframe to conclude as a final decision
             that a class is positive during the timeframe
             
        For call level decisions the prominence of probability peaks
        are used to determine whether a peak is distinguished enough
        from surrounding noise that the associated timeframe can be 
        considered the center time of a vocalization. Value are 
        in [0,1]
        
        For all three quantities, class variables are used if None
        is passed in:
           o cls.DEFAULT_PROMINENCE_THRESHOLD 
           o cls.DEFAULT_PERCENTAGE_AGREEMENT
           o cls.DEFAULT_PROMINENCE_THRESHOLD
        
        
        Saves PowerResult instance in experiment as json.
        Key will start with PwrRes.
        
        :param pwr_member: PowerMember instance to use
            for computing the PowerResult
        :type pwr_member: PowerMember
        :param rec_path: path to the recording to analyze
        :type rec_path: str
        :param percentage_agreement: percentage of signatures that
            must have voted True to conclude that a timeframe contains
            a vocalization of the focus species.
        :type percentage: {None | float}
        :param prominence_thres: the importance of a probability
            peak to be considered a vocalization.
        :type prominence_thres: {None | float}
        :param pwr_res: if pwr_res is not given, the probabilities
            are computed, else those in the PowerResult are used.
        :type pwr_res: {None | PowerResult}
        '''

        if probability_thres is None:
            probability_thres = PowerEvaluator.DEFAULT_PROBABILITY_THRESHOLD
        if percentage_agreement is None:
            percentage_agreement = PowerEvaluator.DEFAULT_PERCENTAGE_AGREEMENT

        species = pwr_member.species_name
        
        if pwr_res is None:
            self.log.info(f"Analyzing recording to detect {species} vocalizations...")
            pwr_res = pwr_member.compute_probabilities(rec_path)
            self.log.info(f"Done analyzing recording to detect {species} vocalizations.") 

            pwr_res_fname = self._make_experiment_key(
                species=pwr_res.species,
                prefix='PwrRes'
                )
            self.experiment.save(pwr_res_fname, pwr_res)
            self.log.info(f"Power result saved in experiment {self.experiment.root} under '{pwr_res_fname}' PowerResult")

        # Make accessible to the outside:
        self.pwr_res = pwr_res
        
        # Make the per-timeframe decisions based on
        timeframe_scores = []
        for sig_id in pwr_res.sig_ids():
            pwr_classifier = PowerQuantileClassifier(
                sig_id=sig_id, 
                threshold_quantile=probability_thres
                )
            pwr_classifier.fit(pwr_res)
            decision = pwr_classifier.predict(pwr_res.probabilities(sig_id))
            decision.name = sig_id
            timeframe_scores.append(decision)

        # Get:
        #           1.0  2.0  3.0  4.0  5.0  6.0   ...   12.0 13.0 14.0 15.0 16.0 17.0
        # time                                     ...                                
        # 0.162540   NaN  NaN  NaN  NaN  NaN  NaN  ...  False  NaN  NaN  NaN  NaN  NaN
        # 0.174150   NaN  NaN  NaN  NaN  NaN  NaN  ...  False  NaN  NaN  NaN  NaN  NaN
        timeframe_votes = pd.concat(timeframe_scores, axis=1)
        # Compute number of sigs that must have voted True
        # to conclude class positive:
        num_sigs = len(pwr_res.sig_ids())
        required_agreement = round(percentage_agreement * num_sigs)
        
        # Get subset of rows in timeframe_votes that
        # have sufficient True votes:
        timeframe_vote_result = timeframe_votes[timeframe_votes.sum(axis=1) >= required_agreement]
        
        # Prepare the final data structure as a 
        # Series that will have True where a class is
        # concluded, and has as many elements as 
        # the num of timeframes:
        timeframe_decision = pd.Series([False]*len(timeframe_votes), 
                                       index=timeframe_votes.index,
                                       name=f"is-{self.species}-call")
        
        timeframe_decision.loc[timeframe_vote_result.index] = True
        
        exp_key = self._make_experiment_key(
            sp=self.species,
            prefix='TmFrmDecisions'
            )
        self.experiment.save(exp_key, timeframe_decision)
        self.log.info(f"Saved timeframe level decisions to {exp_key} tabular")
        
        # Now the analysis at call level (which also saves the result
        # to the experiment):
        peaks = pwr_member.find_calls(pwr_res, prominence_threshold=prominence_thres)
        
        exp_key = self._make_experiment_key(
            sp=self.species,
            pr=pwr_res.name,
            promThres=prominence_thres,
            prefix='CallDecisions'
        )
        self.experiment.save(exp_key, peaks)
        self.log.info(f"Saved call level decisions to {exp_key} tabular")

        return pwr_res

    #------------------------------------
    # score
    #-------------------

    def score(self, 
              sel_tbl, 
              pwr_member, 
              pwr_res_info=None,
              probability_thresholds=None,
              prominence_thresholds=None
              ):
        '''
        Once a PowerResult has been computed via the
        'analyze' action, this method computes success
        scores for two levels of detail: at each spectrogram
        timeframe, and at the level of 'detected call'. Computed
        scores are:
        
            o balanced accuracy
            o accuracy
            o recall
            o precision
            o f0.5 score
            o f1 score
        
        Sets of these scores can be computed with multiple
        probability thresholds (for timeframe level) and/or
        prominence thresholds (for call level). 
        
        Results are saved in self.experiment under keys with
        prefix:
            'scores_calls...'  for call level
            'scores_frames...' for timeframe level
        the string '...' will be information about the PowerResult
        used (pr), the species (sp), and the date of the score computation:

            scores_calls_2022-01-21T18_03_49_sp_BANAS_pr_PwrRes_2022-01-21T18_03_40
            frames_calls_2022-01-21T18_03_49_sp_BANAS_pr_PwrRes_2022-01-21T18_03_40            

        Returns a dict with two entries, one with scores for calls,
        another for scores by timeframes. 
        
            {'call_level' : scores, 
             'timeframe_level' : res_df
            }

        Examples of result dicts:
            Call level:
                                   bal_acc       acc  ...      f0.5  prom_thres
            prominence_thres_0.3  0.509723  0.470501  ...  0.299982         0.3
            
            Timeframe level:
                                        bal_acc       acc  ...  sig_id  threshold
            score_sig_id1.0_thres0.6   0.521159  0.513554  ...     1.0        0.6
            score_sig_id2.0_thres0.6   0.534896  0.527988  ...     2.0        0.6
            score_sig_id3.0_thres0.6   0.343388  0.337368  ...     3.0        0.6
                              ... <more signature results>
        

        :param sel_tbl: Raven selection table with truth 
        :type sel_tbl: str
        :param pwr_member: the PowerMember instance for the species
        :type pwr_member: PowerMember
        :param pwr_res_info: the PowerResult from the prior 'analysis'
            action. If None, the PowerResult is pulled from the experiment.
        :type pwr_res_info: {None | PowerResult}
        :param probability_thresholds: optionally, a list of probability
            thresholds under which to compute the timeframe level results
            Default: self.DEFAULT_PROBABILITY_THRESHOLD
        :type {None | [float]}
        :param prominence_thresholds: probability_thresholds: optionally, a list of prominence
            thresholds under which to compute the call level results
            Default: self.DEFAULT_PROMINENCE_THRESHOLD
        :type prominence_thresholds: {None | [float]}
        :return: the timeframe level and call level scores
        :rtype: [pd.DataFrame, pd.DataFrame]
        '''
        
        # Make final decisions about timeframe level, and
        # call level classification:

        if pwr_res_info is None:
            pwr_res = self._latest_power_result(self.experiment)
        elif type(pwr_res_info) == PowerResult:
            pwr_res = pwr_res_info
        else:
            # Must be a string that's the key into the experiment:
            pwr_res = self.experiment.read(pwr_res_info, PowerResult)

        if not pwr_res.knows_truth():
            pwr_res.add_truth(sel_tbl)

        if probability_thresholds is None:
            probability_thresholds = [self.DEFAULT_PROBABILITY_THRESHOLD]

        res_df = pd.DataFrame()
        for threshold in probability_thresholds:
            for sig_id in pwr_res.sig_ids():
                quantile_evaluator = PowerQuantileClassifier(sig_id=sig_id, 
                                                             threshold_quantile=threshold)
                quantile_evaluator.fit(pwr_res)
                score_name = f"score_sig_id{sig_id}_thres{threshold}" 
                score = quantile_evaluator.score(None, None, name=score_name)
                score['sig_id'] = sig_id
                score['threshold'] = threshold
                res_df = pd.concat([res_df, score])
        
        pwr_res_nm_date = Path(pwr_res.name).stem
        exp_key = self._make_experiment_key(
            sp=pwr_res.species,
            pr=pwr_res_nm_date,
            prefix='scores_frames'
            )
        self.experiment.save(exp_key, res_df)
        self.log.info(f"Timeframe-level scores saved in experiment under '{exp_key}, 'tabular")
        
        # On to scores of finding calls, as opposed to 
        # predicting every time frame:

        call_scores = []
        if prominence_thresholds is None:
            prominence_thresholds = [None]
        for prominence_threshold in prominence_thresholds:
            peaks = pwr_member.find_calls(pwr_res, prominence_threshold=prominence_threshold)
            score = pwr_member.score_call_level(peaks, self.test_sel_tbl)
            score['prom_thres'] = prominence_threshold
            score.name = f"prominence_thres_{prominence_threshold}"
            call_scores.append(score)

        # Get like:
        #                        bal_acc       acc  ...      f0.5  prom_thres
        # prominence_thres_0.3  0.578212  0.606010  ...  0.686055         0.3
        # prominence_thres_0.6  0.551401  0.412645  ...  0.386556         0.6
        
        scores = pd.concat(call_scores, axis=1).T
        exp_key = self._make_experiment_key(
            sp=pwr_res.species,
            pr=pwr_res_nm_date,
            prefix='scores_calls'
            )

        self.experiment.save(exp_key, scores)
        self.log.info(f"Call-level scores saved in experiment under '{exp_key}, 'tabular")
        return {'call_level' : scores, 
                'timeframe_level' : res_df
                }
        #input("Press any key to quit: ")

        #call_intervals = Utils.get_call_intervals(sel_tbl_path)

        #clf = PowerQuantileClassifier(sig_id, thres)
        #clf.fit(pwr_res)
        #score = clf.score(pwr_res.probabilities(sig_id), 
        #                  pwr_res.truths(sig_id),
        #                  name=f"({self.SIG_ID}/{self.THRESHOLD}/{self.SLIDE_WIDTH})"
        #                  )
        #return score

    #------------------------------------
    # grid_search
    #-------------------
    
    def grid_search(self):
        
        # pr_disp = sklearn.metrics.PrecisionRecallDisplay.from_estimator(
        #     clf,
        #     pwr_res.prob_df.probability,
        #     pwr_res.prob_df.Truth
        #     )
        # pr_disp.plot()
        from timeit import default_timer as timer

        start = timer()
        # 1hr:20min
        _grid_res = PowerQuantileClassifier.grid_search(
            self.power_member,
            audio_file=self.test_recording,
            selection_tbl_file=self.test_sel_tbl,
            #******quantile_thresholds=[0.80, 0.85, 0.90, 0.99],
            #******quantile_thresholds=[0.18, 0.19, 0.20, 0.80],
            #******quantile_thresholds=[0.21, 0.22, 0.3],
            #******slide_widths=[0.01, 0.02, 0.03],
            #******quantile_thresholds=[0.4, 0.5, 0.6],
            #******slide_widths=[0.03, 0.04, 0.05],
            quantile_thresholds=[0.7, 0.8, 0.9],
            slide_widths=[0.03, 0.04, 0.05],
            experiment=self.experiment
            )
        end = timer()
        self.log.info(f"Runtime: {str(timedelta(seconds=end - start))}")
        self.log.info(f"Grid result in {self.experiment.root}")

# ---------------- Visualization -------------

    #------------------------------------
    # viz_probs
    #-------------------
    
    def viz_probs(self, 
                  power_member,
                  test_recording,
                  test_sel_tbl,
                  sig_id,
                  save_figs=None
                  ):
        
        if not power_member.output_ready:
            pwr_res = power_member.compute_probabilities(test_recording)
        else:
            pwr_res = power_member.power_result

        try:
            truths = pwr_res.truths(sig_id)
        except IndexError:
            # Nobody has told this power result about
            # what is true: 
            # Add a Truth column to the result:
            pwr_res.add_truth(test_sel_tbl)
            truths = pwr_res.truths(sig_id)
        
        try:
            spectro = self.spectro_dict[test_recording]
        except KeyError:
            spectro = SignalAnalyzer.raven_spectrogram(test_recording)
            self.spectro_dict[test_recording] = spectro
        
        species = power_member.species_name
        # Show spectrogram of the recording:
        mesh = plt.pcolormesh(spectro.columns, 
                              list(spectro.index), 
                              spectro, 
                              cmap='jet', 
                              shading='auto')
        
        ax = mesh.axes

        # Superimpose rectangles at the bottom of the
        # spectrogram, to show the true location of calls:
        call_intervals = test_sel_tbl.species_times(power_member.species_name)
        truth_rects_x = [interval['low_val']
                         for interval
                         in call_intervals]
        truth_rects_y = [0]*len(call_intervals)
        widths        = [interval['high_val'] - interval['low_val']
                         for interval
                         in call_intervals]
        heights       = [self.TRUTH_RECT_HEIGHTS]*len(call_intervals)

        for x,y,width,height in zip(truth_rects_x, truth_rects_y, widths, heights):
            ax.add_patch(Rectangle((x,y),
                                   width, height,
                                   facecolor=None,
                                   fill=False,
                                   edgecolor='black',
                                   hatch='*')
                                   )

        fig = ax.figure
        fig.set_size_inches(14,7)
        fig.suptitle(f"{os.path.basename(test_recording)} ({power_member.species_name})", 
                     fontsize=self.AX_LBL_FONTSIZE + 1)

        # Next, if show the probabilities as a line graph,
        # one sig at a time:
        ax_probs = ax.twinx()
        ax.set_xlabel('Time (sec)', fontsize=self.AX_LBL_FONTSIZE)
        ax.set_ylabel('Frequency (Hertz)', fontsize=self.AX_LBL_FONTSIZE)
        ax_probs.set_ylabel("Probability", fontsize=self.AX_LBL_FONTSIZE)
        
        ax.tick_params(axis='both', labelsize=self.AX_LBL_FONTSIZE)
        ax_probs.tick_params(axis='y', labelsize=self.AX_LBL_FONTSIZE)
        
        fig.show()
        
        # Offering one chart at a time to user:
        batch_mode = False
        
        # Are we to save figs? If so: to experiment or to a 
        # runtime specified dir?
        if save_figs and self.experiment is None:
            # Place for destination directory asked from human:
            dst_dir = None
        for i, probs in enumerate(pwr_res):
            Charter.linechart(probs, ax=ax_probs, color_groups={'red' : ['match_prob']})
            sig_id = pwr_res.sig_ids()[i]
            ax_probs.set_title(f"Signature {sig_id}", fontsize=self.AX_LBL_FONTSIZE - 2)
            
            if save_figs:
                fname  = f"{species}_sig{sig_id}_xparent.png"
                # Save to experiment?
                if self.experiment is not None:
                    key = Path(fname).stem
                    self.log.info(f"Saving sig-{sig_id} to experiment under key {key}...")
                    self.experiment.save(key, fig, transparent=True, format='png')
                else:
                    # Only ask dst_dir once:
                    if dst_dir is None:
                        dst_dir = input("Figures destination directory: ")
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)
                    dst    = os.path.join(dst_dir, fname)
                    self.log.info(f"Saving sig-{sig_id}...")
                    fig.savefig(dst, transparent=True)

            # If saving figs, might just want to go batch
            # mode, and save them all without visually inspecting
            # each chart, and hitting key in between:
            if not batch_mode:
                if save_figs:
                    resp = input("ENTER for next chart; q for quit; b for batch mode (don't pause): ")
                else:
                    resp = input("ENTER for next chart; q for quit: ")
                if resp in ('Q', 'q', 'Quit', 'quit'):
                    break
                if resp in ('b', 'B'):
                    batch_mode = True
            # Remove the current line chart:
            ax_probs.clear()
            # Put the right-side axis label back in:
            ax_probs.set_ylabel("Probability", fontsize=self.AX_LBL_FONTSIZE)

        # cursor = PowerInfoCursor(ax, pwr_res, truths, call_intervals, spectro)
        # _motion_conn_id = fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
        # _click_conn_id  = fig.canvas.mpl_connect('button_press_event', cursor.onclick)
        # if Action.UNITTEST in self.actions:
        #     return fig, ax, cursor
        input("Press ENTER to finish: ")

    #------------------------------------
    # analyze_result
    #-------------------
    
    def analyze_result(self, pwr_res):
        
        gp = pwr_res.prob_df.groupby('sig_id')
        ax = None
        for sig_id, probs in gp.match_prob:
            sig_id = int(sig_id)
            ax = Charter.linechart(probs,
                                   ylabel=f"Probability of Match",
                                   xlabel="Time into recording",
                                   title=f"Match Probabilities During Window Slide",
                                   ax=ax
                                   )

        print(f"Done analysis for pwr_res {pwr_res.sig_ids()}")


# --------------  Utilities -----------------

    #------------------------------------
    # make_experiment_key
    #-------------------
    
    def _make_experiment_key(self, **kwargs):
        '''
        Create an informative key (i.e. filename without its extension)
        for saving to an ExperimentManager instance.
        
        Client provides info to include in arbitrary
        keyword arguments:
        
            species='CMTOG', thres=0.6
            
        Will return:
        
            <date>_species_CMTOG_thres_0.6
            
        The special keyword 'prefix' is available:
        
            species='CMTOG', thres=0.6, prefix='PwrRes'
            
        would return:
        
            PwrRes_<date>_species_CMTOG_thres_0.6
        '''

        try:
            prefix = kwargs['prefix']
            del kwargs['prefix']
        except KeyError:
            prefix = None
            
        exp_key = FileUtils.fname_from_props(
            props_info=kwargs,
            prefix=prefix,
            incl_date=True
            )

        return exp_key

    #------------------------------------
    # _latest_power_result
    #-------------------
    
    def _latest_power_result(self, exp):
        '''
        Given an experiment check whether it contains
        PowerResult instances. If so, return the newest
        instance.
        
        :param exp: experiment to check
        :type exp: ExperimentManager
        :return: newest power result or None if none available
        :rtype {None | PowerResult}
        '''

        # Dict mapping timestamps to file names:
        time_files = {}
        jfiles = exp.listdir(PowerResult)
        for jfile in jfiles:
            # Jfile names are like PwrRes_2022-01-02T11_45_07.json
            # Get the parts:
            fparts = FileUtils.parse_filename(jfile)
            try:
                if fparts['prefix'] != 'PwrRes':
                    continue
            except KeyError:
                # Filename does not even have a 'prefix' part to it:
                continue
            # Replace the underscores w/ colons to get
            # correct iso formated times:
            timestamp = fparts['timestamp'].replace('_',':')
            time_files[datetime.fromisoformat(timestamp)] = jfile
        if len(time_files) == 0:
            return None
        newest_date = max(time_files.keys())
        exp_key = time_files[newest_date]
        pwr_res = exp.read(exp_key, PowerResult)
        pwr_res.name = exp_key
        return pwr_res

    #------------------------------------
    # _init_data_paths
    #-------------------

    def _init_data_paths(self):
        
        self.cur_dir = os.path.dirname(__file__)
        proj_root = os.path.join(self.cur_dir, '../..')
        self.experiment_dir = os.path.join(proj_root, self.DEFAULT_EXPERIMENT_DIR)

        self.sound_data = os.path.join(self.cur_dir, 'tests/signal_processing_sounds')
        self.xc_sound_data = os.path.join(self.cur_dir, 'tests/signal_processing_sounds/XenoCanto')
        
        
        # Field Recordings
        self.BAFFG_data = os.path.join(self.sound_data, 'BAFFG')
        self.CCROC_data = os.path.join(self.sound_data, 'CCROC')
        
        self.BAFFG1_rec = os.path.join(self.BAFFG_data, 'Micrastur-ruficollis-58028.mp3')
        self.BAFFG2_rec = os.path.join(self.BAFFG_data, 'Micrastur-ruficollis-75982.mp3')
        self.BAFFG3_rec = os.path.join(self.BAFFG_data, 'Micrastur-ruficollis-85219.mp3')
        
        self.CCROC1_rec = os.path.join(self.CCROC_data, 'CALL_XC332432-ClayColoredThrush_Yucatan_081216_call2.mp3')
        self.CCROC2_rec = os.path.join(self.CCROC_data, 'CALL_XC482432-R028_Clay_coloured_thrush.mp3')
        self.CCROC3_rec = os.path.join(self.CCROC_data, 'CALL_XC540584-MixPre-255_Turdus_grayi.mp3')
        
        self.DCFLC_rec_fld      = os.path.join(self.sound_data, 'Field/DCFLC/DS_AM17_20190713_172958.WAV')
        self.DCFLC_sel_tbl_fld  = os.path.join(self.sound_data, 'Field/DCFLC/JZ_DS_AM17_20190713_172958.Table.1.selections.txt')
        
        self.sel_tbl_fld = os.path.join(self.cur_dir, 'tests/selection_tables/JZ_DS_AM03_20190713_055956.Table.1.selections.txt')
        # Full field recording for selection tbl: 
        self.sel_recording_fld = os.path.join(self.sound_data, 'DS_AM03_20190713_055956.wav')

        # Xeno Canto 
        self.sel_tbl_cmto_xc1 = os.path.join(self.cur_dir, 'tests/selection_tables/XenoCanto/sel_tbl_Kelley_SONG_XC274155-429_CMTO_KEL.txt')
        self.sel_rec_cmto_xc1 = os.path.join(self.xc_sound_data, 'CMTOG/SONG_XC274155-429_Chestnut-mandibled_Toucan_2_song.mp3')
        
        self.sel_tbl_cmto_xc2 = os.path.join(self.cur_dir, 'tests/selection_tables/XenoCanto/cmto2.selections.txt')
        self.sel_rec_cmto_xc2 = os.path.join(self.xc_sound_data, 'CMTOG/SONG_XC591812-Ramphastos_ambiguus.mp3')
    
        self.sel_tbl_cmto_xc3 = os.path.join(self.cur_dir, 'tests/selection_tables/XenoCanto/cmto3.selections.txt')
        self.sel_rec_cmto_xc3 = os.path.join(self.xc_sound_data, 'CMTOG/SONG_Black-mandibled_Toucan2011-1-24-1.mp3')

# ---------------- Class PowerInfoCursor --------

class PowerInfoCursor(CrosshairCursor):
    
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, ax, pwr_res, truths, call_intervals, spectro):
        CrosshairCursor.__init__(self, ax)
        
        ax.texts[0].set_position((0.08, 0.4))
        self.pwr_res = pwr_res
        self.truths = truths
        self.call_intervals = call_intervals
        self.spectro = spectro
        
        # Get number of signatures:
        self.num_sigs = len(pwr_res.prob_df.groupby(by='sig_id'))

    #------------------------------------
    # on_mouse_move
    #-------------------
    
    def on_mouse_move(self, event):
        x, y = event.xdata, event.ydata
        super().on_mouse_move(event)

        # Entering and exiting the image 
        # delivers y as None; avoid error for that:
        if x is None or y is None:
            return
        
        is_call = Interval.binary_search_contains(self.call_intervals, x) > -1
        # For each signature, the current probability:
        probs_by_sig = []
        for sig_idx in range(self.num_sigs):
            probs_ser = self.pwr_res.probabilities(sig_idx+1)
            prob_times = probs_ser.index.values
            prob_idx = bisect_left(prob_times, x)
            try:
                probs_by_sig.append(probs_ser.iloc[prob_idx].round(2))
            except IndexError:
                # ignore
                continue
        
        # Find closest freq and time values in axes space
        # from the cursor, and get the spectrogram value there:
        freq_idx = bisect_left(list(reversed(self.spectro.index)), y)
        time_idx = bisect_left(list(self.spectro.columns), x)

        try:
            dbfs = round(self.spectro.iloc[freq_idx, time_idx], 1)
        except IndexError:
            #print(f"*********** Index error: freq_idx: {freq_idx}, time_idx: {time_idx}")
            dbfs = 0.0
        
        if len(probs_by_sig) > 0:
            best_sig_idx = np.argmax(probs_by_sig) if is_call else np.argmin(probs_by_sig)
            best_sig_id = best_sig_idx + 1 # sig IDs are 1-based
        
            txt = (f"time={x.round(2)}, freq={int(y)}\n"
                   f"is call: {is_call}\n"
                   f"power: {dbfs} dB FS \n"
                   f"probs by sig: {probs_by_sig}\n"
                   f"best sig: {best_sig_id} ({probs_by_sig[best_sig_idx]})\n"
                   f"mean_prob: {np.mean(probs_by_sig)}\n"
                   f"median_prob: {np.median(probs_by_sig)}\n"
                   f"min_prob: {np.min(probs_by_sig)}\n"
                   f"max_prob: {np.max(probs_by_sig)}"
                   )
    
            self.text.set_text(txt)
            self._update_display()

    #------------------------------------
    # _update_display
    #-------------------
    
    def _update_display(self):
        self.ax.figure.canvas.restore_region(self.background)
        self.ax.draw_artist(self.horizontal_line)
        self.ax.draw_artist(self.vertical_line)
        self.ax.draw_artist(self.text)
        self.ax.figure.canvas.blit(self.ax.bbox)

    #------------------------------------
    # onclick
    #-------------------
    
    def onclick(self, event):
        '''
        Left/right clicks: increase/decrease font size
        Middle click: switch between black and white
        
        :param event:
        :type event:
        '''
        if event.button == 2:
            cur_color = self.text.get_color()
            if cur_color == 'black':
                self.text.set_color('white')
            else:
                self.text.set_color('black')
        elif event.button == 1:
            cur_fontsize = self.text.get_fontsize()
            self.text.set_fontsize(cur_fontsize + 2)
            # Position of lower left of text block in
            # axes coordinates (0,0 is lower left, 1,1 is upper right):
            #******cur_txt_x, cur_txt_y = self.text.get_position()
            # Move txt down a bit to accommodate the larger font:
            print(f"MB1: x: {event.x}; y: {event.y}, xdata: {event.xdata}, ydata: {event.ydata}")
            #*****self.text.set_position((cur_txt_x, cur_txt_y + 0.4))
            #*****self.text.set_position((event.x, event.y))
            self.text.set_position((event.x, event.y))
            self.text.set_transform(self.ax.transAxes)
            
        elif event.button == 3:
            cur_fontsize = self.text.get_fontsize()
            self.text.set_fontsize(cur_fontsize - 2)
            #****cur_txt_x, cur_txt_y = self.text.get_position()
            # Move txt up a bit b/c txt now smaller:
            print(f"MB3: x: {event.x}; y: {event.y}, xdata: {event.xdata}, ydata: {event.ydata}")
            #*****self.text.set_position((cur_txt_x, cur_txt_y - 0.4))
            self.text.set_position((event.xdata, event.ydata))
            self.text.set_transform(self.ax.transAxes)

        if event.button in [1,2,3]:
            self._update_display()

    #------------------------------------
    # _choose_text_color
    #-------------------

    # Not working; replaced with middle-click to switch
    # between left and right:
        
    # def _choose_text_color(self, spectro, txt_area, color_map_name='jet'):
    #     '''
    #     Given a spectrogram and a dict: {'y', 'x', 'width', 'height')
    #     that define a rectangle, compute the average color 
    #     in the rectangle, and retun 'black' or 'white' 
    #     for the color to use for text on top of the rectangle
    #     when the spectrogram is rendered.
    #
    #     Units are indices into the spectrogram index and
    #     columns(, which will correspond to labels on the
    #     axis scale ticks of a pcolormesh). Thus: width is time,
    #     and height is frequencies.
    #
    #     :param spectro: dataframe of values that will
    #         be shown in pcolormesh
    #     :type spectro: pd.DataFrame
    #     :param txt_area: a rectangle over the mesh as
    #         a dict with keys ['x','y','width','height']
    #     :type txt_area: (int, int,int, int)
    #     :returns {'black' | 'white'}
    #     :rtype str
    #     '''
    #
    #     yx = (txt_area['y'], txt_area['x'])
    #     rect_df = Utils.df_extract_rect(spectro,
    #                                     yx=yx, 
    #                                     height=txt_area['height'], 
    #                                     width=txt_area['width']) 
    #     rect_df_abs = rect_df.abs()
    #     rect_df_normed = rect_df_abs / rect_df_abs.max().max()
    #     mean_rect_normed = rect_df_normed.mean().mean() 
    #     cmap = matplotlib.cm.get_cmap(color_map_name)
    #     rgb_fractions = pd.Series(cmap(mean_rect_normed), index=['R', 'G', 'B', 'A'])
    #     rgb_255 = rgb_fractions * 255
    #     txt_color = 'white' if 1 - (rgb_255['R'] * 0.299 + \
    #                                 rgb_255['G'] * 0.587 + \
    #                                 rgb_255['B'] * 0.114) / 255 < 0.5 \
    #                         else 'black'
    #

    #    return txt_color

# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Evaluate or calibrate the power signature approach"
                                     )

    parser.add_argument('--templates_info',
                        help='path to already computed templates, or path to experiment root',
                        default=None)
    parser.add_argument('--prior_result',
                        help="optional: experiment key to previously computed power result, or 'latest'",
                        default=None)
    parser.add_argument('--test_rec',
                        help='path to recording that is to be tested against a given signature',
                        default=None)
    parser.add_argument('--test_sel',
                        help='path to recording selection table for recording',
                        default=None)
    parser.add_argument('-s', '--species',
                        help='species for which action should be performed',
                        default=None)
    parser.add_argument('-e', '--experiment_name',
                        help='name for root dir of experiment where to save results; default: Species name',
                        default=None)
    parser.add_argument('-o', '--outfigs',
                        action='store_true',
                        help='write figures to disk: into experiment if given, or ask outfile',
                        default=False)
    parser.add_argument('--prominence_thres',
                        type=float,
                        nargs='+',
                        help=f"For call level scoring: Repeatable floats for prominence thresholds to try; default {PowerEvaluator.DEFAULT_PROMINENCE_THRESHOLD}")
    
    parser.add_argument('--probability_thres',
                        type=float,
                        nargs='+',
                        help=f"For timeframe level scoring: Repeatable floats for probability thresholds to try; default {PowerEvaluator.DEFAULT_PROBABILITY_THRESHOLD}")
    
    parser.add_argument('actions',
                        choices=['analyze', 'score', 'viz_probs', 'gridSearch'],
                        nargs='+',
                        help='Repeatable: actions to perform')

    args = parser.parse_args()

    cur_dir = os.path.dirname(__file__)
    # Check file existence:

    #if args.templates is None:
    #    default_templates_path = os.path.join(cur_dir, 'species_calibration_data/signatures.json')

    if args.test_rec is not None and not os.path.exists(args.test_rec):
        raise FileNotFoundError(f"Audio file for testing not found ({args.test_rec})")
    if args.test_sel is not None and not os.path.exists(args.test_sel):
        raise FileNotFoundError(f"Raven selection table file for testing not found ({args.test_sel})")

    # Convert text actions to Action enum members:
    actions = []
    for action in args.actions:
        if action == 'analyze':
            actions.append(Action.ANALYSIS)
        elif action == 'score':
            actions.append(Action.SCORE)
        elif action == 'gridSearch':
            actions.append(Action.GRID_SEARCH)
        elif action == 'viz_probs':
            actions.append(Action.VIZ_PROBS)
        else:
            raise NotImplementedError(f"Action {action} is not implemented")
    
    if args.experiment_name is None:
        args.experiment_name = args.species
    
    PowerEvaluator(args.experiment_name,
                   args.species,
                   actions,
                   args.templates_info,
                   power_result_info=args.prior_result,
                   test_recording=args.test_rec,
                   test_sel_tbl=args.test_sel,
                   prominence_thresholds=args.prominence_thres,
                   probability_thresholds=args.probability_thres,
                   outfigs=args.outfigs
                   )
