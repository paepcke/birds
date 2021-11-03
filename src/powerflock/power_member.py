'''
Created on Oct 18, 2021

@author: paepcke


TODO:
    o Identify hyper parameters

'''
import os

import librosa
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils, Interval
import numpy as np
import pandas as pd
from powerflock.signal_analysis import SignalAnalyzer, TemplateSelection
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
                 spectral_template, 
                 template_selection=TemplateSelection.SIMILAR_LEN,
                 the_slide_width_time=None
                 ):
        '''
        Constructor
        '''
        if the_slide_width_time is None:
            the_slide_width_time = SignalAnalyzer.SIGNATURE_MATCH_SLIDE_TIME
        self.species_name = species_name
        self.spectral_template = spectral_template
        self.template_selection = template_selection
        
        self.min_call_len = min(self.spectral_template.sig_lengths)
        
        # Number of array elements to slide
        # template across input audio when computing
        # match probabilities.

        self.slide_width_time = the_slide_width_time
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
    
    def compute_probabilities(self, full_audio, sr):
        '''
        Given an audio clip array longer than the shortest call
        of this PowerMember's species. Slide the signal template
        past the full_audio in increments of self._slide_width_samples array
        elements. Each time, find the probability of the audio
        snippet being a call of this member's species.
        
        Results are returned, and stored in self.probs_by_time. The
        data structure is a dict keyed with the center time of
        each template window overlay.
        
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

        if len(full_audio) < self.min_call_len:
            raise ValueError(f"Audio snippet length must be at at least {self.min_call_len}")
        audio = self._right_size_sr(full_audio, sr)
        probs_df, summary_ser = SignalAnalyzer.match_probability(
            audio, 
            self.spectral_template,
            slide_width_time=self.slide_width_time
            )
        self.power_result = PowerResult(probs_df, summary_ser, self.species_name)
        
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
    See computer_probabilities() in PowerMember for details on that width.
    
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
    
    def score(self, X, y):
        return sklearn.metrics.accuracy_score(X, y)

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
                    slide_widths=[0.05, 0.1, 0.2] # seconds
                    ):

        cls.best_f1        = 0.0
        cls.best_f1_probs  = None
        cls.best_f1_truths = None
        cls.best_f1_preds  = None
        rec_arr, rec_sr = SoundProcessor.load_audio(audio_file)
        
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
                    power_res.add_overlap_and_truth(selection_tbl_file)

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
                    # Save the best f1's probabilities and truths
                    if score.f1 > cls.best_f1:
                        cls.best_f1 = score.f1
                        cls.best_f1_probs  = probs
                        cls.best_f1_truths = truths
                        cls.best_f1_preds  = preds
        end = timer()
        print(f"Grid search duration: {end - start}")
        return res_df

# ------------------------ Class PowerResult --------

class PowerResult:
    '''
    Instances hold results from an individual PowerMember.
    Also provided are methods for accuracy and other outcome
    measures.
    '''
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, prob_df, summary_ser, species, sr=22050):
        '''
        Life begins with the result of a PowerMember's
        compute_probabilities() computation. To its result
        df we add columns for the walltime start/end/middle.
        
        Clients can call the add_truth() method with a source
        of the true call match/no-match values. Those will
        be added to the prob_df in yet another column. 
        
        Example prob_df:
        
               n_samples    probability  sig_id  start   stop
            0         10.0         0.85     0.0   10.0   20.0
            1         10.0         0.85     0.0   20.0   30.0
            2         10.0         0.75     0.0   31.0   40.0
            3         10.0         0.65     0.0   41.0   50.0
            4          4.0         0.65     2.0    0.0    4.0
            5          4.0         0.55     2.0    5.0    9.0
            6        100.0         0.65     3.0    0.0  100.0
            7        100.0         0.55     3.0  101.0  200.0
             

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
        prob_df['start_time']  = prob_df.start / sr
        prob_df['stop_time']   = prob_df.stop / sr
        prob_df['center_time'] = prob_df.start_time + (prob_df.stop_time - prob_df.start_time)/2. 

        self.prob_df = prob_df
        self.summary_ser = summary_ser
        self.species  = species
        
        
        
    #------------------------------------
    # add_and_truth
    #-------------------
    
    def add_overlap_and_truth(self, truth_source, plot=False):
        '''
        Add two new columns: 'Overlap' and 'Truth' to the matching results
        df. For each row the Truth column is 1 or 0, depending
        on whether in the interval of that row a call did
        occur. The Overlap column will contain the percentage
        of overlap between the probability result interval and a
        true call.
        
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
            # is an data_augmentation.utils.Interval instance
            row_dicts = Utils.read_raven_selection_table(truth_source)
            call_intervals = [row_dict['time_interval']
                              for row_dict
                              in row_dicts]

        elif type(truth_source) == list:
            # Assume source is a list of Interval instances,
            # each describing the time period in the input 
            # recording where a bird call happened:
            types = [type(el) == Interval for el in truth_source]
            if sum(types) != len(truth_source):
                raise TypeError(f"List in truth_source does not contain Interval instances")
            call_intervals = truth_source
            
        # Build a Series whose index are the times from the 
        # probability series. Fill the series with the percentage
        # of overlap of the interval in each probability row:
        
        overlap_percentages = pd.Series([0.]*len(self.prob_df.index),
                                        name='sig_overlap_perc')

        cur_sig_id = self.prob_df.loc[0,'sig_id']
        truth_intervals = iter(call_intervals)
        cur_truth_interval = next(truth_intervals)
        for row_num, prob_ser in self.prob_df.iterrows():
            if prob_ser.sig_id != cur_sig_id:
                cur_truth_interval = next(truth_intervals)
                cur_sig_id = prob_ser.sig_id 
            prob_interval = Interval(prob_ser.start_time, 
                                     prob_ser.stop_time,
                                     step=1/SignalAnalyzer.sr)
            ovlp = cur_truth_interval.percent_overlap(prob_interval)
            overlap_percentages[row_num] = ovlp

        # Attach the overlap percentages to the right
        # of self.prob_df:
        self.prob_df['overlap'] = overlap_percentages

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

        # For convenience, add a boolean col Truth, with
        # 1 if any overlap between prbability row and any
        # true call intervals:
        self.prob_df['Truth'] = self.prob_df.overlap > 0
