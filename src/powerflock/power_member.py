#!/usr/bin/env python
'''
Created on Oct 18, 2021

@author: paepcke


TODO:
    o Identify hyper parameters

'''
import argparse
import datetime
import json
import os
import sys

import librosa
from logging_service.logging_service import LoggingService
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

from experiment_manager.experiment_manager import JsonDumpableMixin

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils, Interval
import numpy as np
import pandas as pd
from powerflock.signal_analysis import SignalAnalyzer, TemplateSelection
from powerflock.signatures import SpectralTemplate


#from sklearn.metrics import PrecisionRecallDisplay
#import matplotlib.pyplot as plt
#from result_analysis.charting import Charter
#from seaborn.matrix import heatmap
class PowerMember:
    '''
    The equivalent to a binary classifier that is part
    of an ensemble. A member of this class is specialized
    on one bird species. Each input is an audio snippet.
    The corresponding output is a probability that the 
    snippet is of this power member's species. c
    
    Creating a PowerMember requires a SpectralTemplate, 
    which holds the spectral centroids of one or more
    call examples. A spectral centroid is a series of
    frequencies as long as one call. Each element of the
    series is a moment in time, and the frequency at which 
    spectral power was maximum among all frequencies at
    that time:
    
        f1
        f2
        f3         x       
        f4   x            x
        f5                       x
            t1     t2    t3      t4
    
    The probability output is computed by passing the 
    template across the given audio clip, and finding
    the position when the cost of matching the template
    curve to the audio's spectral centroids is smallest. 
    
    The most efficient use of the class is to provide
    a long recording, and to allow the PowerMember to 
    compute the probability of species presence during
    a window of time, such as a few milliseconds.
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, 
                 species_name, 
                 spectral_template_info=None,
                 the_slide_width_time=None,
                 experiment=None,
                 apply_bandpass=False,
                 template_selection=TemplateSelection.SIMILAR_LEN,
                 ):
        '''
        Initialize a member of a species recognition
        ensemble. SpectralTemplate instance(s) are either created
        from the file names passed in spectral_template_info, or may 
        be passed in as an individual, or list of templates.
        
        NOTE: even though multiple templates may be passed in, the
              current implementation only uses the first. Passing
              in more will currently raise a NotImplementedError
        
        If spectral_template_info is not an individual, or list of SpectralTemplate
        it is expected to be a zip object that pairs a Raven selection table
        file with the recording file that the table annotates:
        
          Zip:
                recording1-file   selection-table1-file
                recording2-file   selection-table2-file
                      ...

        The slide width is the number of fractional seconds by
        which a signature is passed across audio files. 
        
        If an ExperimentManager is passed in, results are stored there.
        
        The template_selection names a policy for selecting among
        the tmplates. See the TemplateSelection enumeration.
        
        NOTE: after initialization the PowerMember is ready to accept
              an audio recording, and to generate probabilities from
              it. Until compute_probabilities() is called with a 
              recording to process, no output is available. Clients
              can test for available output via property:
                 
                <inst>.output_ready
        
        :param species_name: name of species to which this ensemble
            member is specialized
        :type species_name: str
        :param spectral_template_info: 
        :type spectral_template_info:
        :param the_slide_width_time:
        :type the_slide_width_time:
        :param experiment:
        :type experiment:
        :param apply_bandpass: whether or not to create signatures
            with a bandpass filter derived from the selection table,
            or not. Only relevant if spectral_template_info is a
            zip for creating signatures, rather than the already-made
            templates.
        :type apply_bandpass: bool
        :param template_selection:
        :type template_selection:
        '''

        self.log = LoggingService()
        self.experiment = experiment
        
        # --------- Argument Checks -----------
        # Init slide width:
        if the_slide_width_time is None:
            the_slide_width_time = SignalAnalyzer.SIGNATURE_MATCH_SLIDE_FRACTION

        # Figure out what client passed for spectral template(s):
        if type(spectral_template_info) == SpectralTemplate:
            # A ready-made single template:
            self.spectral_templates = [spectral_template_info]
        elif type(spectral_template_info) == list:
            for maybe_tmplt in spectral_template_info:
                if type(maybe_tmplt) != SpectralTemplate:
                    raise TypeError(f"Template info {maybe_tmplt} is not a SpectralTemplate")
            # A list of ready-made templates
            self.spectral_templates = spectral_template_info
        elif type(spectral_template_info) == str:
            # Path to json file:
            try:
                self.spectral_templates = \
                    [SpectralTemplate.from_json_file(spectral_template_info)[species_name]]
            except KeyError:
                raise ValueError(f"No template for {species_name} in file {spectral_template_info}")
        else:
            # Must be a zip of recording/RavenTable pairs:
            if not isinstance(spectral_template_info, zip):
                raise TypeError(f"Template info {spectral_template_info} is not of any acceptable type")

            call_files, sel_tbl_files = list(zip(*spectral_template_info))
            
            for file in call_files + sel_tbl_files:
                if not os.path.exists(file):
                    raise FileNotFoundError(f"Could not find {file}")

            self.spectral_templates = []
            for call_file, sel_tbl_file in zip(call_files, sel_tbl_files):
                self.spectral_templates.extend(self.compute_spectral_templates(species_name, 
                                                                               call_file, 
                                                                               sel_tbl_file,
                                                                               apply_bandpass)
                                                                               )

        self.species_name = species_name
        self.template_selection = template_selection
        
        # NOTE: From here on out we only consider first spectral template:
        
        if len(self.spectral_templates) != 1:
            raise NotImplementedError(f"Current implementation only supports a single SpectralTemplate instance")

        self.spectral_template = self.spectral_templates[0]
        
        self.min_call_len = min(self.spectral_template.sig_lengths)
        
        # Number of array elements to slide
        # template across input audio when computing
        # match probabilities.

        self.slide_width_time = the_slide_width_time
        # The slide_width_samples is a setter, so the
        # width in time will be converted to samples
        # during this assignment:
        self.slide_width_samples = the_slide_width_time

        # Width of each audio clip to match against
        # the signatures of a template must be the 
        # length of the longest call for which the signature was
        # created. The signature lengths are spectrogram
        # frames. So clip width must be:
        #      samplesPerFrame * Frames,
        # which depends on hop_length and n_fft. The computation
        # is available from librosa.frames_to_samples()
        #
        # Note that in signal_analysis.SignalAnalyzer.cls._compute_prob()
        # just before comparing the clip to one of the template's
        # signatures, the clip is shortened to that signature's
        # length:
        
        self.audio_clip_width = librosa.frames_to_samples(
            max(self.spectral_template.sig_lengths), 
            self.spectral_template.hop_length, 
            self.spectral_template.n_fft)
        
        # Power member is initialized, and ready to 
        # compute probabilities on an input. But no 
        # input has been passed through yet. That happens
        # when clients call compute_probabilities()
        self.output_ready = False

    #------------------------------------
    # slide_width_time Getters/Setters
    #-------------------
    
    @property
    def slide_width_time(self):
        return self._slide_width_time
    
    @slide_width_time.setter
    def slide_width_time(self, new_slide_width_time):
        self._slide_width_time = new_slide_width_time

    @property
    def slide_width_samples(self):
        return self._slide_width_samples
    
    @slide_width_samples.setter
    def slide_width_samples(self, new_slide_width_time):
        self._slide_width_samples = int(new_slide_width_time * self.spectral_template.sr)

    #--------------------------- Computations --------------

    #------------------------------------
    # compute_probabilities
    #-------------------
    
    def compute_probabilities(self, 
                              full_audio,
                              outfile=None, 
                              sr=SignalAnalyzer.sr
                              ):
        '''
        Workhorse for generating probabilities from audio input
        
        Given an audio clip array or file longer than the shortest call
        of this PowerMember's species, or a file path to a recording. 
        Slide the signal template past the full_audio in increments 
        of self._slide_width_samples array elements. Each time, find 
        the probability of the audiosnippet being a call of this member's 
        species.
        
        If full_audio is an audio file, the sample rate does not need
        to be supplied in this call. 
        
        If self.template was computed with a bandpass filter, then
        that same filter is applied to the given audio before taking
        power samples and comparing to the template.
        
        Results are stored in self.power_result, which is returned. The 
        power_result holds probability information for every _slide_width
        multiple of time.
        
        Sample rate of full_audio is matched to this PowerMember's 
        template.
        
        :param full_audio: audio clip for which to compute probabilities
        :type full_audio: np.array
        :param sr: sample rate of the audio
        :type sr: float
        :returned a PowerResult instance that holds in the property probs_series
            a pd.Series whose index is time into recording, and whose
            values are probabilities of a call occurring at that moment.
        :rtype PowerResult
        '''

        if type(full_audio) == str:
            if not os.path.exists(full_audio):
                raise FileNotFoundError(f"Could not find audio file {full_audio}")
            self.log.info(f"Loading audio {full_audio}")
            full_audio, sr = SoundProcessor.load_audio(full_audio)

        if len(full_audio) < self.min_call_len:
            raise ValueError(f"Audio snippet length must be at at least {self.min_call_len}")
        # Adjust audio's sample rate if necessary:
        audio = self._right_size_sr(full_audio, sr)
        # Get results of matching audio against each
        # signature of the current SpectralTemplate:
        probs_df = SignalAnalyzer.match_probability(
            audio, 
            self.spectral_template,
            slide_width_time=self.slide_width_time
            )
        self.log.info("Matched all signatures")
        self.power_result = PowerResult(probs_df, self.species_name)

        if outfile is not None:
            print(f"Saving power_result to {outfile}")
            self.power_result.json_dump(outfile)

        # Indicate that probabilities were computed
        # on an input:
        self.output_ready = True
        
        return self.power_result

    # ----------------------- Visualizations --------------------

    #------------------------------------
    # plot_pr_curve
    #-------------------

    def plot_pr_curve(self, power_result):

        y_true = power_result.truth_df['Truth']
        y_pred = power_result.truth_df['Probability']
        pred_display = sklearn.metrics.PrecisionRecallDisplay.from_predictions(
            y_true, 
            y_pred, 
            name='CMTOG PowerFlock Member')
        return pred_display


    # ------------------------- Utilities ------------
    
    #------------------------------------
    # compute_spectral_templates
    #-------------------
    
    def compute_spectral_templates(self, 
                                   species, 
                                   call_files, 
                                   sel_tbl_files,
                                   apply_bandpass=False
                                   ):
        '''
        DEPRECATED: use the separate quad_sig_calibration.py.
        
        Compute templates of call signatures from 
        the call files and associated selection table files.
        
        :param species: birds species that produced the calls
        :type species: str
        :param call_files: path(s) to recordings of bird calls
        :type call_files: {str | [str]}
        :param sel_tbl_files: Raven selection table files that
            one-to-one correspond to the recording files
        :type sel_tbl_files: {str | [str]}
        :param apply_bandpass: whether or not to 
            apply a  bandpass filter during signature creation
            as per the low/high frequency spec in the Raven
            selection table.
        :type apply_bandpass: {bool | Interval}

        :return a list of SpectralTemplate instances
        :rtype [SpectralTemplate]
        '''
        templates = SignalAnalyzer.compute_species_templates(species, 
                                                             call_files, 
                                                             sel_tbl_files,
                                                             apply_bandpass=apply_bandpass)
        return templates

    #------------------------------------
    # _right_size_sr
    #-------------------
    
    def _right_size_sr(self, audio, sr):
        '''
        Compares given audio's sample rate against the 
        spectral template's sample rate. If unequal, resamples
        audio to match sample rate of template.
        
        Returns audio with sample rate matched.
        
        :param audio: audio snippet
        :type audio: np.array
        :param sr: given audio's sample rate
        :type sr: float
        :return: same or new audio, guaranteed to have
            same sample rate as template
        :rtype np.array
        '''
        if sr != self.spectral_template.sample_rate:
            return librosa.resample(audio, sr, self.spectral_template.sample_rate())
        else:
            return audio

# ------------------------ PowerQuantileClassifier ------------

class PowerQuantileClassifier(BaseEstimator, ClassifierMixin):
    '''
    Instances are sklearn-type estimators with fit() and predict()
    methods. During creation, an instance is given a PowerResult instance 
    that holds the result of matching an audio file to a template, i.e. to
    one or more power signatures. Given is therefore information like:
    
             n_samples  probability  sig_id    start     stop  start_time  stop_time  center_time    overlap  Truth
          0    43136.0     0.711948     1.0      0.0  43136.0    0.000000   1.956281     0.978141  29.090684   True 
          1    43136.0     0.662506     1.0  10784.0  53920.0    0.489070   2.445351     1.467211  54.090628   True 
          2    43136.0     0.810184     1.0  21568.0  64704.0    0.978141   2.934422     1.956281  79.090572   True 
          3    36862.0     0.725635     2.0   9215.0  46077.0    0.417914   2.089660     1.253787      0.0    False  
          4    36862.0     0.756336     2.0  18430.0  55292.0    0.835828   2.507574     1.671701      0.0    False
          
    This classifier is initialized with a quantile threshold, such as
    0.75 for the forth quartile; though any quantile is acceptable.
    The estimator considers the matching probabilities that resulted from
    a single signature being slid across the audio with some slide width.
    See compute_probabilities() in PowerMember for details on that width.
    
    Given a probability this estimator returns True if the probability is the
    GE to the given signature's threshold probability.
    
    
    Usage:
    
         estimator = PowerQuantileClassifier(pwr_result, 3, 0.53)
         estimator.fit()
         estimator.predict([0.23, 0.87, 0.51])
    '''
    
    
    
    #------------------------------------
    #  Constructor
    #-------------------
    
    def __init__(self, sig_id, threshold_quantile):
        super().__init__()
        self.sig_id = sig_id
        self.threshold_quantile = threshold_quantile
        
        # No absolute power result or probability threshold 
        # yet: use one of these None-assignment to check in predict()
        # fit() was called:
        self.power_result   = None
        self.prob_threshold = None 

    #------------------------------------
    # fit
    #-------------------
    
    def fit(self, power_result, y=None):
        '''
        Extracts the sig_id-relevant subset of power_result's prob_df
        into sig_prob_df. Extracts just the probabilities associated
        with this classifier's sig_id into inst vars X and probability, 
        and the truth values into inst vars y and truth.
        
        Inst var prob_threshold will contain the absolute probability
        that is the threshold between True and False.
        
        :param power_result:
        :type power_result:
        :param y: not used
        :type y: Any
        '''

        self.power_result = power_result
        y = power_result.prob_df.Truth
        self.classes_, y = np.unique(y, return_inverse=True)

        # Get like:
        #        n_samples  probability  sig_id    start     stop  start_time  stop_time  center_time    overlap  Truth
        #     0    43136.0     0.711948     1.0      0.0  43136.0    0.000000   1.956281     0.978141  29.090684   True 
        #     1    43136.0     0.662506     1.0  10784.0  53920.0    0.489070   2.445351     1.467211  54.090628   True 
        #     2    43136.0     0.810184     1.0  21568.0  64704.0    0.978141   2.934422     1.956281  79.090572   True 
        #     3    36862.0     0.725635     2.0   9215.0  46077.0    0.417914   2.089660     1.253787      0.0    False  
        #     4    36862.0     0.756336     2.0  18430.0  55292.0    0.835828   2.507574     1.671701      0.0    False
        
        df = self.power_result.prob_df
        
        # For convenience:
        sig_id  = self.sig_id

        # Dig out just the results for the given signature:
        df_this_sig = df[df.sig_id == sig_id]
        
        # Isolate results that pertain to the given sig_id
        # to get: 
        #         probability  mean_prob  med_prob  max_prob  center_time  Truth
        #     0      0.711948   0.406659   0.38578  0.962312     0.978141    1.0
        #     1      0.662506   0.406659   0.38578  0.962312     1.467211    1.0
        #     2      0.810184   0.406659   0.38578  0.962312     1.956281    1.0
        
        # The phrase gb.size().iloc[sig_idx] is the number of measurements
        # taken with the given signature: 
        # num_probs = len(df_this_sig)
        # sig_df = pd.concat([df_this_sig.probability.reset_index().probability,
        #                     pd.Series([df_this_sig.probability.mean()]*num_probs),
        #                     pd.Series([df_this_sig.probability.median()]*num_probs),
        #                     pd.Series([df_this_sig.probability.max()]*num_probs),
        #                     df_this_sig.center_time.reset_index().center_time,
        #                     df_this_sig.Truth.reset_index().Truth
        #                     ],
        #                     ignore_index=True,
        #                     axis='columns'
        #                     )
        # sig_df.columns = ['probability', 'mean_prob', 'med_prob', 'max_prob', 'center_time', 'Truth']
        
        probs_by_time = df_this_sig.probability
        probs_by_time.index = df_this_sig.center_time
        
        truth_by_time = df_this_sig.Truth
        truth_by_time.index = df_this_sig.center_time
         
        sig_df = pd.concat(
            [probs_by_time,
             truth_by_time
            ], axis='columns')
        
        # Compute the absolute threshold probability for
        # this signature from the given threshold quantile:
        self.prob_threshold = df_this_sig.probability.quantile(q=self.threshold_quantile)
        
        self.X = self.probabilities = df_this_sig.probability
        self.y = self.truth         = sig_df.Truth
        self.center_time            = sig_df.Truth.index
        self.mean_prob = self.probabilities.mean()
        self.median_prob = self.probabilities.median()
        self.max_prob = self.probabilities.max() 
        
        # Return the classifier:
        return self

    #------------------------------------
    # predict
    #-------------------
 
    # UNCLEAR WHETHER NEEDED AND CORRECT: UNTESTED
    def predict(self, X):
        decisions = self.decision_function(X)
        return self.classes_[np.argmax(decisions, axis=1)]


    #------------------------------------
    # decision_function
    #-------------------
    
    def decision_function(self, X):
        '''
        Given an array-like of power match probabilities,
        return True or False for each element, depending on whether
        the probability is high enough, given the probability threshold.
        
        :param X: an individual value, or an array-like of probabilities
        :type X: [float]
        :return: whether or not the probability is high
            enough to assume the corresponding audio except
            is part of a call.
        :rtype: {float | [float]}
        '''
        if self.prob_threshold is None:
            raise NotFittedError(f"Must first fit estimator with a PowerResult")
        
        return X >= self.prob_threshold

    #------------------------------------
    # score
    #-------------------
    
    def score(self, X, y, name=None):

        probs  = self.probabilities
        preds  = self.decision_function(probs)
        truth = self.truth
        score = pd.Series({
            'bal_acc'    : sklearn.metrics.balanced_accuracy_score(truth, preds),
            'acc'        : sklearn.metrics.accuracy_score(truth, preds),
            'recall'     : sklearn.metrics.recall_score(truth, preds),
            'precision'  : sklearn.metrics.precision_score(truth, preds),
            'f1'         : sklearn.metrics.f1_score(truth, preds)
            }, name=name
            )
        return score

    #------------------------------------
    # grid_search
    #-------------------
    
    @classmethod
    def grid_search(cls, 
                    power_member,
                    audio_file,
                    selection_tbl_file, 
                    sig_ids=[1.0],
                    quantile_thresholds=[0.75, 0.80, 0.90, 0.95],
                    slide_widths=[0.05, 0.1, 0.2], # seconds
                    sig_combination=[TemplateSelection.MED_PROB],
                    experiment=None
                    ):

        cls.best_f1        = 0.0
        cls.best_f1_probs  = None
        cls.best_f1_truths = None
        cls.best_f1_preds  = None
        rec_arr, rec_sr = SoundProcessor.load_audio(audio_file)
        
        # DF with schema:
        #                                 prob0    prob1    ...
        #  (sig_id, thres, slide_width)
        all_probs = pd.DataFrame()
        
        res_df = pd.DataFrame([],
                              columns=['bal_acc', 'recall', 'precision', 'f1']
                              )
        from timeit import default_timer as timer
        start = timer()
        for sig_id in sig_ids:
            for thres in quantile_thresholds:
                for slide_width in slide_widths:
                    # Set slide width fractional seconds
                    # The associated setter method will convert
                    # to samples: 
                    power_member.slide_width_time = slide_width
                    power_res = power_member.compute_probabilities(rec_arr, rec_sr)
                    power_res.add_truth(selection_tbl_file)

                    clf = PowerQuantileClassifier(sig_id, thres)
                    clf.fit(power_res)
                    probs  = clf.probabilities
                    preds  = clf.decision_function(probs)
                    truths = clf.truth
                    score = pd.Series({
                        'bal_acc'    : sklearn.metrics.balanced_accuracy_score(truths, preds),
                        'acc'        : sklearn.metrics.accuracy_score(truths, preds),
                        'recall'     : sklearn.metrics.recall_score(truths, preds),
                        'precision'  : sklearn.metrics.precision_score(truths, preds),
                        'f1'         : sklearn.metrics.f1_score(truths, preds)
                        }, name=(sig_id, thres, slide_width)
                        )
    
                    res_df = res_df.append(score)
                    
                    all_probs = all_probs.append(pd.Series(probs, name=(sig_id, thres, slide_width)))
                    # Save the best f1's probabilities and truths
                    if score.f1 > cls.best_f1:
                        cls.best_f1 = score.f1
                        cls.best_f1_probs  = probs
                        cls.best_f1_truths = truths
                        cls.best_f1_preds  = preds
        end = timer()
        experiment.save('all_probs', all_probs)
        experiment.save('res_scores', res_df)
        print(f"Grid search duration: {str(datetime.timedelta(seconds=(end-start)))}")
        return res_df

# ------------------------ Class PowerResult --------

class PowerResult(JsonDumpableMixin):
    '''
    Instances hold results from an individual PowerMember.
    Also provided are methods for accuracy and other outcome
    measures.
    '''
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, prob_df, species, sr=22050):
        '''
        Life begins with the result of a PowerMember's
        compute_probabilities() computation. To its result
        df we add columns for the walltime start/end/middle.
        
        Clients can call the add_truth() method with a source
        of the true call match/no-match values. Those will
        be added to the prob_df in yet another column. 
        
        Example prob_df:
        
               n_samples     match_prob  sig_id  start_idx   stop_idx
            0         10.0         0.85     0.0   10.0       20.0
            1         10.0         0.85     0.0   20.0       30.0
            2         10.0         0.75     0.0   31.0       40.0
            3         10.0         0.65     0.0   41.0       50.0
            4          4.0         0.65     2.0    0.0        4.0
            5          4.0         0.55     2.0    5.0        9.0
            6        100.0         0.65     3.0    0.0  1    00.0
            7        100.0         0.55     3.0  101.0  2    00.0
             

        Example summary_ser:
        
            min_prob        lowest probability among all matches
            max_prob        highest probability among all matches
            med_prob        the median of the probabilities among all matches
            best_fit_prob   max probability among matches with the 
                            longest signature

        :param prob_df: all match probabilities
        :type prob_df: pd.DataFrame
        :param summary_ser: aggregates of probabilities
        :type summary_ser: pd.Series
        :param species: the species for which the PowerMember
            is specialized
        :type species: str
        '''
        
        self.sr = sr
        
        # Add start, end, and middle wallclock times to the df:
        prob_df['start_time'] = SignalAnalyzer.hop_length  * prob_df.start_idx / sr
        prob_df['stop_time']  = SignalAnalyzer.hop_length  * prob_df.stop_idx / sr
        center_time = prob_df.start_time + (prob_df.stop_time - prob_df.start_time)/2. 

        self.prob_df = prob_df
        self.prob_df.index = center_time
        self.prob_df.index.name = 'time'
        self.species  = species
        
        # The truths column has not been added yet.
        # The column is added by a client calling add_truth().
        # The following property can be tested via 

    #------------------------------------
    # add_and_truth
    #-------------------
    
    def add_truth(self, truth_source, plot=False):
        '''
        Add two new columns: 'Overlap' and 'Truth' to the matching results
        df. For each row the Truth column is 1 or 0, depending
        on whether in the interval of that row a call did
        occur. The Overlap column will contain the percentage
        of overlap between the probability result interval and a
        true call by the species in self.species.
        
        The prob_df will permanently contain the columns.
        
        :param truth_source: either a selection table file, 
            or a list of Interval instances with time intervals
            of calls
        :type truth_source: {str | [Interval]}
        '''
        
        if type(truth_source) == str:
            # File name to a selection table that covers
            # the recording that was the input to the result
            if not os.path.exists(truth_source):
                raise FileNotFoundError(f"Could not find selection table {truth_source}")
            
            # Get list of dicts, each holding one row of
            # the selection table, plus time_interval, which
            # is a data_augmentation.utils.Interval instance
            row_dicts = Utils.read_raven_selection_table(truth_source)
            # Cannot use list comprehension here, b/c
            # self won't be known:
            call_intervals = []
            for row_dict in row_dicts:
                if row_dict['species'] == self.species:
                    call_intervals.append(row_dict['time_interval'])

        elif type(truth_source) == list:
            # Assume source is a list of Interval instances,
            # each describing the time period in the input 
            # recording where a bird call happened:
            types = [type(el) == Interval for el in truth_source]
            if sum(types) != len(truth_source):
                raise TypeError(f"List in truth_source does not contain Interval instances")
            call_intervals = truth_source

        # Find all rows shows timestamp (i.e. index)
        # lies within any of the call intervals:
        ntime_frames, _n_facts = self.prob_df.shape
        overlaps = pd.Series([False]*ntime_frames, index=self.prob_df.index)
        time_slots = pd.Series(self.prob_df.index)
        for call_intv in call_intervals:
            new_overlaps = time_slots.apply(call_intv.contains)
            new_overlaps.index = overlaps.index
            overlaps = np.logical_or(overlaps, new_overlaps)

        # New column with True when a row is
        # within a call.
        self.prob_df['truth'] = overlaps

        #**********
        # if plot:
        #     import matplotlib.pyplot as plt
        #     fig, axes = plt.subplots(nrows=3, ncols=4)
        #     match_groups = self.prob_df.groupby(by='sig_id')
        #     # Axes is an array-of-array of axes instances.
        #     # Make that a simple array of axes:
        #     axes = axes.flatten()
        #     for ax, (sig_id, sig_match_df) in zip(axes, match_groups):
        #         prob_and_overlap = pd.DataFrame([sig_match_df.probability, 
        #                                          sig_match_df.overlap/100]).transpose()
        #         prob_and_overlap.index = sig_match_df.center_time.round(2)
        #         ax_prime = Charter.linechart(prob_and_overlap,
        #                                      color_groups={'probability' : 'green', 'overlap' : 'blue'},
        #                                      ax=ax,
        #                                      rotation=45
        #                                      )
        #     ax.figure.legend()
        #
        # prob_and_overlap = pd.DataFrame([self.prob_df.probability, 
        #                                  self.prob_df.overlap/100]).transpose()
        # prob_and_overlap.index = self.prob_df.center_time.round(2)
        #

        #**********

    #------------------------------------
    # probabilities
    #-------------------
    
    def probabilities(self, sig_id):
        '''
        Return a series with all the probabilities
        :param sig_id: identifyer of signature whose
            probability results are to be returned
        :type sig_id: {int | float}
        :return: pd.Series of probabilities yielded using 
            that signature; the index are the recording times
        :rtype pd.Series(float)
        '''
        probs = self.prob_df[self.prob_df.sig_id == sig_id].probability
        probs.index = self.prob_df[self.prob_df.sig_id == sig_id].center_time
        return probs 

    #------------------------------------
    # truths
    #-------------------
    
    def truths(self, sig_id):
        '''
        Return a series with all the truth values
        :param sig_id: identifyer of signature whose
            probability results are to be returned
        :type sig_id: {int | float}
        :return: pd.Series of truth: TRUE/FALSE at each time;
            index will be the center times
        :rtype pd.Series(bool)
        '''
        if 'Truth' not in self.prob_df.columns:
            raise IndexError(f"Power result missing Truth column; call add_truth() first")
        truths = self.prob_df[self.prob_df.sig_id == sig_id].Truth
        truths.index = self.prob_df[self.prob_df.sig_id == sig_id].center_time 
        return truths

    #------------------------------------
    # json_dumps
    #-------------------
    
    def json_dumps(self):
        
        # Create a dict with the instance vars.
        # The prob_df is rendered to json in place.
        # The orient='table' is required b/c the
        # default needs the index to be unique, which
        # this one isn't:
        
        as_dict = {
            'species' : self.species,
            'sr' : self.sr,
            'prob_df' : self.prob_df.to_json(orient='table')
            }
        return json.dumps(as_dict)
    
    #------------------------------------
    # json_loads
    #-------------------
    
    @classmethod
    def json_loads(cls, jstr):
        '''
        Given a json string created by
        json_dumps(), materialize a PowerResult
        instance filled with the proper instance
        var values.
        
        :param jstr: json string created via json_dumps()
        :type jstr: str
        :return: new instance of PowerResult
        :rtype PowerResult
        '''
        
        as_dict = json.loads(jstr)
        # Read df as:

        #     'index'  'col1'   'col2'
        # 0      1       10       20
        # 1      2       30       10
        # 2      2        ...
        # 3      10       ...

        prob_df = pd.read_json(as_dict['prob_df'], orient='table')
        # Remove the auxiliary column 'time' that
        # was added by the pr.dump():
        prob_df.drop(labels='time', axis='columns', inplace=True)
        pwr_res = PowerResult(prob_df, as_dict['species'], as_dict['sr'])
        return pwr_res

    #------------------------------------
    # json_dump 
    #-------------------
    
    def json_dump(self, fpath):
        '''
        Render self onto disk, json encoded.
        Recoverable via PowerResult.json_load()
        
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
        Return a PowerResult instance materialized
        from a previously stored, json encoded file.
        This method is the inverse of json_dump() 

        :param fpath: file with json-rendered PowerResult instance
        :type fpath: str
        :return new PowerResult instance
        :rtype PowerResult
        '''
        with open(fpath, 'r') as fd:
            jstr = fd.read()
        return cls.json_loads(jstr)

    #------------------------------------
    # __repr__
    #-------------------
    
    def __repr__(self):
        return f"<PowerResult {self.species} {round(self.prob_df.index[-1])}sec {hex(id(self))}>"

    #------------------------------------
    # __str__
    #-------------------
    
    def __str__(self):
        return self.__repr__()

# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Start a species-specific vocalization detector."
                                     )

    parser.add_argument('species',
                        help='name of species to recognize')

    parser.add_argument('template',
                        help='path to calibrated-template file; run quad_sig_calibration.py if unavailable') 

    parser.add_argument('recording',
                        help='recording on which to find vocalizations'
                        )
    
    parser.add_argument('outfile',
                        help='path where to save the json-formatted result'
                        )

    parser.add_argument('-b', '--bandpass',
                        help='apply a bandpass filter; low/high frequencies as per signatures',
                        action='store_true')

    args = parser.parse_args()
    
    if not os.path.exists(args.template):
        print(f"Cannot find template file {args.template}")
        sys.exit(1)
        
    if not os.path.exists(args.recording):
        print(f"Cannot find recording {args.recording}")
        sys.exit(1)
        
    if os.path.exists(args.outfile):
        response = input(f"WARNING: outfile {args.outfile} exists; overwrite? (y/n):")
        if response not in ['Y', 'y', 'yes']:
            print("Aborting.")
            sys.exit(0)
    
    templates_dict = SpectralTemplate.from_json_file(args.template)
    try:
        cmtog_template = templates_dict[args.species]
    except KeyError:
        avail_species = templates_dict.keys()
        print(f"Species {args.species} is not available in template. Available are {avail_species}")
        sys.exit(1)
    
        
    pwr_member = PowerMember(species_name=args.species, 
                             spectral_template_info=args.template, 
                             apply_bandpass=args.bandpass
                             )
    
    pwr_result = pwr_member.compute_probabilities(args.recording, outfile=args.outfile)
    