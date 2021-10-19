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
        self.win_width   = max(max(self.spectral_template.sig_lengths),
                               2048
                               )

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
        :returned a PowerResult instance that holds in property probs_series
            a pd.Series whose index is time into recording, and whose
            values are probabilities of a call occurring at that moment.
        :rtype PowerResult
        '''

        if len(full_audio) < self.min_call_len:
            raise ValueError(f"Audio snippet length must be at at least {self.min_call_len}")
        audio = self._right_size_sr(full_audio, sr)
        
        probs = {}
        for start_idx in range(0, len(full_audio), self.slide_width):
            end_idx = start_idx + self.win_width

            try:
                aud_snip = audio[start_idx:end_idx]
                prob = SignalAnalyzer.match_probability(
                    aud_snip, 
                    self.spectral_template, sr)
                center_time = np.floor(start_idx + self.slide_width / 2.) / sr
                probs[center_time] = prob
                 
            except IndexError:
                # Slid past end of given audio:
                break

        prob_series = pd.Series(probs.values(), index=probs.keys())
        self.power_result = PowerResult(prob_series, self.species_name)
        
        return self.power_result

    # ----------------------- Visualizations --------------------
    
    #------------------------------------
    # plot_pr_curve
    #-------------------

    def plot_pr_curve(self, power_result):

        y_true = power_result.truth_df['Truth']
        y_pred = power_result.truth_df['Probability']
        sklearn.metrics.PrecisionRecallDisplay.from_predictions(y_true, 
                                                                y_pred, 
                                                                name='CMTOG PowerFlock Member')


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
    
    def __init__(self, prob_series, species):
        '''
        Life begins with the result of a PowerMember's
        compute_probabilities() computation. The prob_series
        pd.Series index are time points into a recording.
        The values are the probability that the PowerMember's
        input at that time was part of a call. 
        
        :param prob_series:
        :type prob_series:
        :param species: the species for which the PowerMember
            is specialized
        :type species: str
        '''
        
        self.truth_df = pd.DataFrame([], 
                                     columns=['Probability', 'Truth'], 
                                     index=prob_series.index)
        self.truth_df['Probability'] = prob_series
        self.species = species
        
    #------------------------------------
    # add_truth
    #-------------------
    
    def add_truth(self, truth_source):
        
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
            
        # Build a df whose index are the times from the 
        # probability series. The first column is the probability
        # of a call at that time, and the second column is 
        # 1 or 0 depending on whether a call did occur at that time:
        
        # Assume no call at each time point until
        # proven otherwise:
        truth_each_timepoint = pd.Series([0]*len(self.truth_df),
                                         index=self.truth_df.index)
        for timepoint in self.truth_df.index:
            for interval in call_intervals:
                if interval.contains(timepoint):
                    truth_each_timepoint[timepoint] = 1
                    break

        self.truth_df['Truth'] = truth_each_timepoint
