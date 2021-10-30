'''
Created on Oct 18, 2021

@author: paepcke


TODO:
    o Identify hyper parameters

'''
import os

import librosa
import sklearn

from data_augmentation.utils import Utils, Interval
import numpy as np
import pandas as pd
from powerflock.signal_analysis import SignalAnalyzer, TemplateSelection
from result_analysis.charting import Charter


class PowerMember:
    '''
    The equivalent to a binary classifier that is part
    of an ensemble. A member of this class is specialized
    on one bird species. Each input is an audio snippet.
    The corresponding output is a probability that the 
    snippet is of this power member's species. 
    
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

    WIN_SLIDE_WIDTH_FRACTION = 0.5
    '''Fraction of minimum bird call width to slide window when finding probabilities'''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, 
                 species_name, 
                 spectral_template, 
                 template_selection=TemplateSelection.SIMILAR_LEN
                 ):
        '''
        Constructor
        '''
        self.species_name = species_name
        self.spectral_template = spectral_template
        self.template_selection = template_selection
        
        self.min_call_len = min(self.spectral_template.sig_lengths)
        
        # Number of array elements to slide
        # template across input audio when computing
        # match probabilities:
        
        self.slide_width = int(self.min_call_len * self.WIN_SLIDE_WIDTH_FRACTION)

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

    #--------------------------- Computations --------------

    #------------------------------------
    # compute_probabilities
    #-------------------
    
    def compute_probabilities(self, full_audio, sr):
        '''
        Given an audio clip array longer than the shortest call
        of this PowerMember's species. Slide the signal template
        past the full_audio in increments of self.slide_width array
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
        probs_df, summary_ser = SignalAnalyzer.match_probability(audio, 
                                                                 self.spectral_template
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
    
    def add_overlap_and_truth(self, truth_source):
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

        # For convenience, add a boolean col Truth, with
        # 1 if any overlap between prbability row and any
        # true call intervals:
        self.prob_df['Truth'] = self.prob_df.overlap > 0
