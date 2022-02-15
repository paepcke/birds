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
from pathlib import Path
import sys

from experiment_manager.experiment_manager import JsonDumpableMixin
import librosa
from logging_service.logging_service import LoggingService
import scipy
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

from birdsong.utils.utilities import FileUtils
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Interval, RavenSelectionTable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from powerflock.signal_analysis import SignalAnalyzer
from powerflock.signatures import SpectralTemplate, TemplateCollection
from result_analysis.charting import Charter


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
    
    DEFAULT_PROMINENCE_THRES = 0.4
    '''Minimum cutoff for call to find_peaks(); higher is more discriminative'''
    
    DEFAULT_PROBABILITY_THRESHOLD = 0.6
    '''Minumum probability at which a peak is accepted'''
    
    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, 
                 species_name, 
                 spectral_template_info=None,
                 the_slide_width_time=None,
                 experiment=None
                 ):
        '''
        Initialize a member of a species recognition
        ensemble. SpectralTemplate instance(s) are either created
        from the file names passed in spectral_template_info, or may 
        be passed in ready made. The argument may be:
        
           o A SpectralTemplate instance
           o A TemplateCollection
           o A zip object that pairs a Raven selection table
             file with the recording file that the table annotates:
        
	             Zip:
	                   recording1-file   selection-table1-file
	                   recording2-file   selection-table2-file
	                         ...

        NOTE: it is the caller's responsibility to pass an 
              appropriate template. For example, if it is important
              that a template is calibrated, pass a calibrated
              template. Else a default prominence_threshold is
              used. After running quad_sig_calibration, the experiment
              will contain calibrated templates_info in key 'templates_calibrated'

        The slide width is the number of fractional seconds by
        which a signature is passed across audio files. 
        
        If an ExperimentManager is passed in, results are stored there.
        
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
        :param template_selection:
        :type template_selection:
        '''

        self.log = LoggingService()
        self.experiment = experiment
        if experiment is not None:
            # Create name to refer to the experiment: its root di:
            self.exp_name = Path(experiment.root).stem
        
        # --------- Argument Checks -----------
        # Init slide width:
        if the_slide_width_time is None:
            the_slide_width_time = SignalAnalyzer.SIGNATURE_MATCH_SLIDE_FRACTION

        # Figure out what client passed for spectral template(s):
        if type(spectral_template_info) == SpectralTemplate:
            # A ready-made single template:
            self.spectral_template = spectral_template_info
        elif type(spectral_template_info) == TemplateCollection:
            # A dict-like of ready-made templates_info, one for each
            # of several species. Pick out the one for the
            # species of this PowerMember:
            self.spectral_template = spectral_template_info[species_name]
        elif type(spectral_template_info) == str:
            # Path to json file of template:
            try:
                self.spectral_template = SpectralTemplate.json_load(spectral_template_info)
            except KeyError:
                raise ValueError(f"No template for {species_name} in file {spectral_template_info}")
        else:
            raise FileNotFoundError(f"Cannot find template for {species_name} from {spectral_template_info}")

        self.species_name = species_name

        # Preserve the template in the experiment:
        if self.experiment is not None:
            self.experiment.save(f"template_{species_name}", self.spectral_template)
        
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
        of (self.slide_width_time * <samples-in-shortest-call>) array 
        elements. Each time, find the probability of the audiosnippet 
        being a call of this member's species.
        
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
            slide_width_time_fraction=self.slide_width_time
            )
        self.log.info("Matched all signatures")
        # Uses setter method, which sets output_ready to True:
        self.power_result = PowerResult(probs_df, self.species_name)

        if outfile is not None:
            print(f"Saving power_result to {outfile}")
            self.power_result.json_dump(outfile)
        
        return self.power_result

    #------------------------------------
    # find_calls
    #-------------------
    
    def find_calls(self, 
                   pwr_res,
                   probability_threshold=None, 
                   prominence_threshold=None):
        '''
        Given a PowerResult computed by compute_probabilities()
        across an entire recording, mark an estimate of the center 
        time of each vocalization, plus a lower and upper time bound
        of each call. 
        
        Return two dfs:
                  peak_prob   low_bound       high_bound
            t0      0.123     t0 - some_num     t0 + some_num
            t8      0.543     t8 - some_num     t8 + some_num
                       ...
        and:
        
            peak_times_by_sig_id
                        1.0    2.0    3.0    4.0    5.0  ...
            1.056508   False  True  False  False  False  ...
            1.149388   False  False  True  False  False  ...
                       ...
        
        The prominence_threshold specifies how 'important'
        a probability peak must be in the context of probabilities
        neighboring in time. See scipy.signal.find_peaks and
        Wikipedia 'Topographic prominence' for details. Default
        is set in the class variable DEFAULT_PROMINENCE_THRES. 
        Values must be in [0,1]
        
        :param pwr_res: the probabilities for each timeframe
            according to every signature
        :type pwr_res: PowerResult
        :param prominence_threshold: importance of probability
            peak in context
        :rtype prominence_threshold: {None | float} 
        :result 1. center time and probability of each call, plus
            estimates of low and high time bounds of the calls.
                2. for each time, which signature detected a peak
        :rtype [pd.DataFrame pd.DataFrame]
        '''

        sig_ids = pd.Series(pwr_res.sig_ids())
        
        prob_df = pwr_res.prob_df
        
        # Pull out sig id and their smoothed and normalized
        # match probabilities:
        all_sig_probs = prob_df[['sig_id', 'match_prob']]
        
        prob_peaks_by_sig_list = []
        for sig_id in sig_ids:
            sig = self.spectral_template.get_sig(sig_id)
            if prominence_threshold is None:
                try:
                    if not sig.usable:
                        continue
                    prominence_threshold = sig.prominence_threshold
                except AttributeError:
                    # Not a calibrated sig:
                    self.log.warn(f"Sig {sig_id} is not calibrated; no minimum prominence enforced")
                    prominence_threshold  = 0.0

            if probability_threshold is None:
                try:
                    probability_threshold = sig.prob_threshold
                except AttributeError:
                    self.log.warn(f"Sig {sig_id} is not calibrated; no minimum probability enforced")
                    probability_threshold = 0.0 

            probs = all_sig_probs[all_sig_probs.sig_id == sig_id].match_prob
            peak_indices, properties = scipy.signal.find_peaks(probs,
                                                               prominence=prominence_threshold)
            peak_probs = properties['prominences']
            # Time when a peak occurred:
            times = probs.index[peak_indices]

            # Remove peaks whose probability is below
            # the prob threshold: produce list of tuples
            # with time and prob for which prob is greater
            # than threshold:
            #   [(sig_id, good_time1, good_prob1), (sig_id, good_time2, good_prob2), ...]  
            
            for time, prob in zip(times, peak_probs):
                if prob < probability_threshold:
                    continue
                else:
                    prob_peaks_by_sig_list.append((time, sig_id, prob))

        # Get a df with rows for each time when a probability
        # peak occurs.
        #   0  time1, sid1, prob1
        #   1  time2, sid1, prob2
        #   2  time1, sid2, prob3
        #       ...
        
        prob_peaks_by_sig_df = pd.DataFrame(prob_peaks_by_sig_list, 
                                            columns=['time', 'sig_id', 'match_prob'])
        
        # Create df with maximum prob of peaks, and the IDs
        # of the sigs that contributed that max in each case,
        # like 
        #              peak_prob     sig_id
        #     t3        0.4024        11
        #     t12       0.3125        2
        #     t22            ...

        peaks = prob_peaks_by_sig_df.groupby('time').max()
        if len(peaks) == 0:
            # At given probability and prominence
            # thresholds no peak is found:
            return peaks

        # Add estimated width of the call simply
        # based on the width of the 'winning' sig:
        
        all_sigs = self.spectral_template.signatures
        sig_widths = pd.Series({sig.sig_id : sig.duration()
                                for sig
                                in all_sigs
                                })
        half_widths = sig_widths[peaks.sig_id] / 2.
        half_widths.index = peaks.index
        
        # Ensure lower not negative, and upper not gt recording len:
        peaks['low_bound']  = peaks.index - half_widths
        peaks['high_bound'] = peaks.index + half_widths
        
        peaks['low_bound'] = peaks['low_bound'].where(peaks['low_bound'] >=0, 0)
        peaks['high_bound'] = peaks['high_bound'].where(peaks['high_bound'] <= peaks.index[-1], 
                                                        peaks.index[-1])

        # Add a column that repeats the prominence_threshold
        # for each row; redundant, but easy:
        peaks['prominence_threshold'] = [prominence_threshold]*len(peaks)
        return peaks
    
    #------------------------------------
    # score_call_level
    #-------------------
    
    def score_call_level(self, peaks, sel_tbl):
        scorer = CallLevelScorer(peaks, sel_tbl, self.species_name)
        call_level_score = scorer.score()
        return call_level_score

    # ----------------------- Visualizations --------------------

    #------------------------------------
    # plot_pr_curve
    #-------------------

    def plot_pr_curve(self, power_result, sig_ids=None):

        ax_cols_n = 4
        
        if sig_ids is None:
            sig_ids = power_result.sig_ids()
            num_sig_ids = len(sig_ids)
            num_subplots = num_sig_ids
            ax_rows_n = int(np.ceil(num_subplots / ax_cols_n))

        elif type(sig_ids) == list:
            num_sig_ids = len(sig_ids)
            num_subplots = num_sig_ids
            ax_rows_n = int(np.ceil(num_subplots / ax_cols_n))

        else:
            # A single sig_ids:
            num_sig_ids = 1
            ax_rows_n = 1
            ax_cols_n = 1
            # Uniformly deal with multiple charts,
            # with one being a degenerate case:
            #*****axs = (axs)
            sig_ids = [sig_ids]
        fig, axs  = plt.subplots(ax_rows_n, ax_cols_n, sharex=True, sharey=True)
        # If just one axes, make it a list to
        # match what the branches above would produce:
        if type(axs) != np.ndarray:
            axs = [axs]
        else:
            # The axes returned by multi-chart subplots are an
            # array of axes instances, get just a list:
            axs = axs.flatten()
        pred_displays = []
        fig.suptitle(f"PR Plot(s) {power_result.species} ({num_sig_ids}/{len(power_result.sig_ids())} call sigs)")
        for sig_id, ax in zip(sig_ids, axs):
            y_true = power_result.prob_df['truth'][power_result.prob_df['sig_id'] == sig_id]
            y_pred = power_result.prob_df['match_prob'][power_result.prob_df['sig_id'] == sig_id]
            pred_displays.append(sklearn.metrics.PrecisionRecallDisplay.from_predictions(
                y_true, 
                y_pred,
                ax=ax,
                name=f"Sig-{sig_id}"
                ))
        fig.show()
        return pred_displays

    #------------------------------------
    # get_sig
    #-------------------
    
    def get_sig(self, sig_id):
        '''
        Return a signature instance, given a sig_id.
        Return None if no matching sig exists.
        
        :param sig_id: ID of signature instance to find
        :type sig_id: Any
        :return Signature
        :raise KeyError if signature of given sig_id is unknown
        '''
        return self.spectral_template.get_sig(sig_id)

    #------------------------------------
    # power_result
    #-------------------
    
    @property
    def power_result(self):
        return self._power_result
    
    @power_result.setter
    def power_result(self, new_val):
        if new_val is None:
            self._power_result = None
            self.output_ready = False
            return
        
        if type(new_val) != PowerResult:
            raise TypeError(f"Power result must be of class PowerResult, not {type(new_val)}")
        self._power_result = new_val
        self.output_ready = True

    # ------------------------- Utilities ------------

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

# ------------------------- CallLevelScore -----------
class CallLevelScorer:
    '''
    Used when a labeled recording is available
    to compute recall and precision at the call level.
    Given from a call to PowerMember.find_calls() is the
    following:
    
              peak_prob   low_bound                        high_bound
        t0      0.123     t0 - 1/2 signature_width_S3     t0 + 1/2 signature_width_S3 
        t8      0.543     t8 - 1/2 signature_width_S1     t8 + 1/2 signature_width_S1
        
    where t_n are the estimated center times of a call,
    peak_prob is the highest probability amongst the probabilities
    computed from all signatures. Column low_bound is the estimated
    beginning of the call, calculated based on the width of the signature
    that generated peak_prob. Analogously for high_bound.
    
    Success criteria:
    
       o The positive classification of a call at time t_n 
         is correct if t_n lies within the start and end times
         of any call label. It is all right for the classifier
         to mark multiple time frames within a call as positive.
         However, all positive outcomes between low_bound and high_bound
         count as one single successfully identified call, i.e. true positive (TP)
         
       o A positive classification outside any labeled time interval
         is a false positive (FP)
         
       o Recall is measured by adding the number of TPs and dividing by
         the number of labeled calls. 

    '''

    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, peaks, sel_tbl, species):
        
        self.peaks = peaks
        self.species = species
        self.sel_tbl = sel_tbl

    #------------------------------------
    # score
    #-------------------
    
    def score(self):
        
        # If no peaks at all: results are all zero:
        if len(self.peaks) == 0:
            return pd.Series({
            'bal_acc'    : 0.0,
            'acc'        : 0.0,
            'recall'     : 0.0,
            'precision'  : 0.0,
            'f1'         : 0.0,
            'f0.5'       : 0.0
            }, name='score')
        # Pick out the true call Interval instances
        # from the selection table:
        true_time_intervals = self.sel_tbl.species_times(self.species) 
        # Ensure the list is sorted by start time:
        true_time_intervals.sort()
        
        # Note the (constant) step size of the truth
        # intervals, and create a list of Interval instances
        # for the predictions: 
        step_size = true_time_intervals[0]['step']  # @UnusedVariable
        # Each interval denotes the estimated start and stop
        # of one call. The low_bound/high_bound were computed
        # as center time +/ half the width of the signature that
        # generated the probability peak:
        pred_time_intervals = [Interval(low_time, high_time, step_size)
                               for low_time, high_time, step_size
                               in zip(self.peaks.low_bound, 
                                      self.peaks.high_bound,
                                      [step_size]*len(self.peaks)
                                      )
                               ]
        TPs = []
        FPs = []

        # For each predicted call interval, see
        # whether it overlaps any truth interval.
        # When it does, we call that a true positive,
        # else the predicted call is a false positive:
        for pred_intv in pred_time_intervals:
            overlap = Interval.binary_search_overlap(true_time_intervals, pred_intv)
            if overlap > -1:
                TPs.append(pred_intv)
            else:
                FPs.append(pred_intv)
                
        # To use the scypy scoring, we need two array-like
        # of 1s and 0s. One array each for the predictions
        # and one for the truth.
        # We therefore need corresponding 1/0 values for each
        # timeframe:
        #                     |     True Interval         |
        # Truth   0   0   0   1    1    1     1     1     1   0
        #
        # Pred    0   0   1   0    1    0     0     1     0   0
        #                FP   |    Turn these to 1s       |   0
        #
        # We accept some 0s inside predictions, as long as there
        # is at least one 1 to identify the call.
        # We no long have access to the spectrogram here, so
        # we create a 10ms time scale that covers the earliest 
        # to latest interval:
        
        latest =   max(true_time_intervals[-1]['high_val'],
                       pred_time_intervals[-1]['high_val'])
        earliest = min(true_time_intervals[0]['low_val'],
                       pred_time_intervals[0]['low_val'])
        
        times = np.arange(earliest, latest, 0.01)
        # Start with all 0s in the truth and predictions
        # timeline:
        truth = pd.Series(np.zeros(len(times)), index=times)
        preds = pd.Series(np.zeros(len(times)), index=times)
        
        # Set values to 1 in truth wherever a truth interval
        # straddles the timeline
        for true_intv in true_time_intervals:
            truth[(truth.index >= true_intv['low_val']) & (truth.index < true_intv['high_val'])] = 1.0
        
        # Same for predictions:
        for pred_intv in pred_time_intervals:
            preds[(preds.index >= pred_intv['low_val']) & (preds.index < pred_intv['high_val'])] = 1.0

        all_scores = {
            'bal_acc'    : sklearn.metrics.balanced_accuracy_score(truth, preds),
            'acc'        : sklearn.metrics.accuracy_score(truth, preds),
            'recall'     : sklearn.metrics.recall_score(truth, preds),
            'precision'  : sklearn.metrics.precision_score(truth, preds),
            'f1'         : sklearn.metrics.f1_score(truth, preds),
            'f0.5'       : sklearn.metrics.fbeta_score(truth, preds, beta=0.5),
            }

        score = pd.Series(all_scores, name='score') 
        return score

# ------------------------ PowerQuantileClassifier ------------

class PowerQuantileClassifier(BaseEstimator, ClassifierMixin):
    '''
    Instances are sklearn-type estimators with fit() and predict()
    methods. During creation, an instance is given a PowerResult instance 
    that holds the result of matching an audio file to a template, i.e. to
    one or more power signatures. Given is therefore information like:
    
             n_samples   match_prob  sig_id start_idx stop_idx   start_time stop_time  truth
          0    43136.0     0.711948     1.0      0.0  43136.0    0.000000   1.956281   True 
          1    43136.0     0.662506     1.0  10784.0  53920.0    0.489070   2.445351   True 
          2    43136.0     0.810184     1.0  21568.0  64704.0    0.978141   2.934422   True 
          3    36862.0     0.725635     2.0   9215.0  46077.0    0.417914   2.089660  False  
          4    36862.0     0.756336     2.0  18430.0  55292.0    0.835828   2.507574  False
          
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
    
    log = LoggingService()


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
        self.classes_ = [True, False]

        # Get like:
        #        n_samples  match_prob   sig_id start_idx stop_idx   start_time  stop_time truth
        #     0    43136.0     0.711948     1.0      0.0  43136.0    0.000000   1.956281  True 
        #     1    43136.0     0.662506     1.0  10784.0  53920.0    0.489070   2.445351  True 
        #     2    43136.0     0.810184     1.0  21568.0  64704.0    0.978141   2.934422  True 
        #     3    36862.0     0.725635     2.0   9215.0  46077.0    0.417914   2.089660 False  
        #     4    36862.0     0.756336     2.0  18430.0  55292.0    0.835828   2.507574 False
        
        df = self.power_result.prob_df
        
        # For convenience:
        sig_id  = self.sig_id

        # Dig out just the results for the given signature:
        df_this_sig = df[df.sig_id == sig_id]
                
        # Compute the absolute threshold probability for
        # this signature from the given threshold quantile:
        self.prob_threshold = df_this_sig.match_prob.quantile(q=self.threshold_quantile)

        # Some convenience instance vars:
        self.X             = df_this_sig.match_prob
        self.probabilities = df_this_sig.match_prob
        self.center_time   = df_this_sig.index
        
        # Return the classifier:
        return self

    #------------------------------------
    # predict
    #-------------------
 
    def predict(self, X):
        '''
        Given a probability of an input being of the
        focus class output True or False, and given the
        decision threshold with which this instance
        was created. This is a simple greater-than test.
                
        :param X: probability
        :type X: float
        :result whether or not the probability indictes class membership
        :rtype: bool
        '''
        decision = self.decision_function(X)
        return decision

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
        
        if not self.power_result.knows_truth():
            raise AttributeError("Score was called without updating PowerResult with a truth column")
        
        truth = self.power_result.truths(self.sig_id)
        score = pd.Series({
            'bal_acc'    : sklearn.metrics.balanced_accuracy_score(truth, preds),
            'acc'        : sklearn.metrics.accuracy_score(truth, preds),
            'recall'     : sklearn.metrics.recall_score(truth, preds),
            'precision'  : sklearn.metrics.precision_score(truth, preds),
            'f1'         : sklearn.metrics.f1_score(truth, preds),
            'f0.5'       : sklearn.metrics.fbeta_score(truth, preds, beta=0.5),
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
                    quantile_thresholds=[0.75, 0.80, 0.90, 0.95],
                    slide_widths=[0.05, 0.1, 0.2], # fraction of samples in shortest call
                    experiment=None
                    ):
        '''
        Perform a grid search sig_id x decision_threshold x slide_win_width.
        
        Repeatedly match an audio file against
        all Signature instances of the power_member's spectral template.
        Vary the quantile threshold of probability above 
        which a positive decision is made (separate for each
        Signature instance). Also vary the fraction of samples
        of the shortest Signature by which signatures are slid
        across the audio file (analogous to hop-length in STFT)
        
        Returns a dataframe with columns ['bal_acc', 'recall', 'precision', 'f1'].
        This df is also saved in the experiment.
         
        :param power_member: the PowerMember for one species; it
            contains the SpectralTemplate previously computed by
            a run of quad_sig_calibration.py
        :type power_member: PowerMember
        :param audio_file: path to audio that is to be matched
            against the signatures
        :type audio_file: str
        :param selection_tbl_file: path to Raven selection table from
            which the ground truth is determined
        :type selection_tbl_file: str
        :param quantile_thresholds: quantiles of probabilities to
            include in the search. These are numbers in [0,1].
        :type quantile_thresholds: [float]
        :param slide_widths: fraction the samples in the shortest call
            that are to be tried in the search
        :type slide_widths: [float]
        :returns: dataframe with columns 
           ['threshold', 'slide_fraction', 'sig_id', 'bal_acc', 'acc', 'recall',
            'precision', 'f1', 'f0.5']
        :rtype: pd.DataFrame
        '''

        cls.best_f1        = 0.0
        cls.best_f1_probs  = None
        cls.best_f1_truths = None
        cls.best_f1_preds  = None
        rec_arr, rec_sr = SoundProcessor.load_audio(audio_file)
        
        sig_ids = [sig.sig_id 
                   for sig 
                   in power_member.spectral_template.signatures]
        
        # DF with schema:
        #                                 prob0    prob1    ...
        #  (sig_id, thres, slide_width)
        all_prob_series = []

        from timeit import default_timer as timer
        start = timer()
        for thres in quantile_thresholds:
            for slide_width in slide_widths:
                # Set slide width fractional seconds
                # The associated setter method will convert
                # to samples: 
                power_member.slide_width_time = slide_width
                cls.log.info(f"Compute probabilities for thres {thres}/{len(quantile_thresholds)}, slide fraction {slide_width}/{len(slide_widths)}...")
                power_res = power_member.compute_probabilities(rec_arr, sr=rec_sr)
                cls.log.info(f"Done compute probabilities for thres {thres}, slide fraction {slide_width}...")
                power_res.add_truth(RavenSelectionTable(selection_tbl_file))
                all_truths = power_res.prob_df.truth
                
                scores = []

                for sig_id in sig_ids:
                    cls.log.info(f"Compute score for sig-{sig_id}/{len(sig_ids)}...")
                    clf = PowerQuantileClassifier(sig_id, thres)
                    clf.fit(power_res)
                    probs  = clf.probabilities
                    preds  = clf.decision_function(probs)
                    truths = all_truths[probs.index]
                    score = pd.Series({
                        'threshold'  : thres,
                        'slide_fraction': slide_width,
                        'sig_id'     : sig_id,
                        'bal_acc'    : sklearn.metrics.balanced_accuracy_score(truths, preds),
                        'acc'        : sklearn.metrics.accuracy_score(truths, preds),
                        'recall'     : sklearn.metrics.recall_score(truths, preds),
                        'precision'  : sklearn.metrics.precision_score(truths, preds),
                        'f1'         : sklearn.metrics.f1_score(truths, preds),
                        'f0.5'       : sklearn.metrics.fbeta_score(truths, preds, beta=0.5)
                        }, name=(sig_id, thres, slide_width)
                        )
    
                    scores.append(score)
                    new_probs = probs.copy()
                    new_probs.name = (sig_id, thres, slide_width)
                    all_prob_series.append(new_probs)
                    # Save the best f1's probabilities and truths
                    if score.f1 > cls.best_f1:
                        cls.best_f1 = score.f1
                        cls.best_f1_probs  = probs
                        cls.best_f1_truths = truths
                        cls.best_f1_preds  = preds
        end = timer()
        res_df = pd.DataFrame(scores)
        all_probs = pd.DataFrame(all_prob_series)
        if experiment is not None:
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
    
    def __init__(self, prob_df, species, sr=22050, name=None):
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
             
        In the constructor, a new column is added: raw_prob, which will
        contain the given match_prob. The values in match_prob are
        replaced with slightly smoothed, and then min-max normed values.
        Also added are start_time and stop_time columns.

        The name argument allows assignment of a client
        determined name for the new instance. No attempt
        is made to verify uniqueness. The name can be set any
        time after instance creation.

        :param prob_df: all match probabilities
        :type prob_df: pd.DataFrame
        :param species: the species for which the PowerMember
            is specialized
        :type species: str
        :param sr: sample rate
        :type sr: {None | float}
        :param name: name for this instance
        :type name: {None | str} 
        
        '''
        
        self.sr = sr
        
        if name is None:
            name = f"PwrRes_{FileUtils.file_timestamp()}"
        self.name = name
        
        # Save match_prob col as 'raw_prob', and replace
        # values in the match_prob col with normalized and
        # smoothed values:
        normed_probs = self.normalize_probs(prob_df['match_prob'])
        new_prob_df  = prob_df.assign(raw_prob=prob_df['match_prob'],
                                      match_prob=normed_probs)

        # Add start, end, and middle wallclock times to the df:
        new_prob_df['start_time'] = SignalAnalyzer.hop_length  * new_prob_df.start_idx / sr
        new_prob_df['stop_time']  = SignalAnalyzer.hop_length  * new_prob_df.stop_idx / sr
        center_time = new_prob_df.start_time + (new_prob_df.stop_time - new_prob_df.start_time)/2. 

        self.prob_df = new_prob_df
        self.prob_df.index = center_time
        self.prob_df.index.name = 'time'
        self.species  = species
        
        # Cache for normalized probabilities
        # (see method normalized_sigs())
        self.normed_sigs = None

    #------------------------------------
    # normalize_probs
    #-------------------
    
    def normalize_probs(self, probs):
        '''
        Takes a series of probabilities. Returns
        a new Series with the same index, but values
        slightly smoothed, and then min-max normalized.
        
        :param probs: probabilities to smooth and normalize
        :type probs: pd.Series
        :return: new Series with normalized and smoothed values
        :rtype: pd.Series
        '''
        
        # Smooth probs a bit:
        probs_filtered = np.abs(pd.Series(scipy.signal.savgol_filter(probs, 
                                                                     window_length=51, 
                                                                     polyorder=11), 
                                          index=probs.index))
        probs_normed = pd.Series(sklearn.preprocessing.minmax_scale(probs_filtered),
                                                                    index=probs_filtered.index)
        return probs_normed 

    #------------------------------------
    # knows_truth
    #-------------------
    
    def knows_truth(self):
        '''
        Returns True if this instance has information
        about the (timeframe level) truth values. This
        condition is met after add_truth() was called.
         
        :return: True if instance has truth label info
        :rtype bool
        '''

        return 'truth' in self.prob_df.columns

    #------------------------------------
    # add_truth
    #-------------------
    
    def add_truth(self, truth_sel_tbl_info):
        '''
        Add two new columns: 'Overlap' and 'Truth' to the matching results
        df. For each row the Truth column is 1 or 0, depending
        on whether in the interval of that row a call did
        occur. The Overlap column will contain the percentage
        of overlap between the probability result interval and a
        true call by the species in self.species.
        
        The prob_df will permanently contain the columns.
        
        :param truth_sel_tbl_info: a RavenSelectionTable instance
            or path to a Raven selection table
        :type {str | RavenSelectionTable}
        '''

        if type(truth_sel_tbl_info) == str:
            # Given path to Raven selection table:
            if not os.path.exists(truth_sel_tbl_info):
                raise FileNotFoundError(f"Selection table file {truth_sel_tbl_info} not found")
            truth_sel_tbl = RavenSelectionTable(truth_sel_tbl_info)
        elif not isinstance(truth_sel_tbl_info, RavenSelectionTable):
            raise TypeError(f"The truth_sel_tbl_info must be path to table, or a RavenSelectionTable, not {type(truth_sel_tbl_info)}")
        else:
            truth_sel_tbl = truth_sel_tbl_info
             
        call_intervals = truth_sel_tbl.species_times(self.species)
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
        probs = self.prob_df[self.prob_df.sig_id == sig_id].match_prob
        probs.index = self.prob_df[self.prob_df.sig_id == sig_id].index
        return probs 
    
    #------------------------------------
    # __iter__
    #-------------------
    
    def __iter__(self):
        '''
        Iterator over the probability Series,
        one signature at a time.
        '''
        # Init the index into the signature IDs: 
        self._cur_sig_idx = -1
        return self
    
    #------------------------------------
    # __next__
    #-------------------
    
    def __next__(self):
        '''
        Probabilities for the next signature.
        
        :return: probabilities for next signature,
            excerpted from self.prob_df.
        :type pd.Series
        :raise StopIteration
        '''
        self._cur_sig_idx += 1
        try:
            nxt_prob_sig = self.sig_ids()[self._cur_sig_idx]
        except IndexError:
            raise StopIteration
        probs = self.prob_df[self.prob_df.sig_id == nxt_prob_sig].match_prob
        probs.index = self.prob_df[self.prob_df.sig_id == nxt_prob_sig].index
        return probs
    
    #------------------------------------
    # __len__
    #-------------------
    
    def __len__(self):
        '''
        The number of signatures for which probabilities
        are contained in this PowerResult
        '''
        return len(self.sig_ids())

    #------------------------------------
    # __deepcopy__
    #-------------------
    
    def __deepcopy__(self):
        '''
        Called by copy module when its deepcopy()
        method is called. Also called from this class'
        copy() method
        
        :return: copy of self without any references back
            to the original
        :rtype PowerResult
        '''
        new_pwr_res = PowerResult(self.prob_df.copy(deep=True),
                                  self.species,
                                  sr=self.sr,
                                  name=self.name
                                  )
        return new_pwr_res

    #------------------------------------
    # copy
    #-------------------
    
    def copy(self):
        '''
        Returns a deep copy of self. I.e. no references
        of the copy's instance variables point back to
        the self's instance vars. 
        
        :return: copy of self without any references back
            to the original
        :rtype PowerResult
        '''
        return self.__deepcopy__()

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
        if 'truth' not in self.prob_df.columns:
            raise IndexError(f"Power result missing 'truth' column; call add_truth() first")
        truths = self.prob_df[self.prob_df.sig_id == sig_id].truth
        truths.index = self.prob_df[self.prob_df.sig_id == sig_id].index
        return truths

    #------------------------------------
    # sig_ids
    #-------------------
    
    def sig_ids(self):
        '''
        Return a list of all signature IDs that
        are represented in the prob_df.
        
        :return: list of signature IDs. These are usually numbers
        :rtype: [Any] 
        '''
        return self.prob_df['sig_id'].unique()

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
        #prob_df.drop(labels='time', axis='columns', inplace=True)
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
    
    templates_dict = SpectralTemplate.json_load(args.template)
    try:
        cmtog_template = templates_dict[args.species]
    except KeyError:
        avail_species = templates_dict.keys()
        print(f"Species {args.species} is not available in template. Available are {avail_species}")
        sys.exit(1)
    
        
    pwr_member = PowerMember(species_name=args.species, 
                             spectral_template_info=args.template, 
                             )
    
    pwr_result = pwr_member.compute_probabilities(args.recording, outfile=args.outfile)
    