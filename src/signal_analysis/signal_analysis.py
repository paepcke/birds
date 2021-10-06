'''
Created on Oct 1, 2021

@author: paepcke
'''
from pathlib import Path
from enum import Enum

import librosa

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils, Interval
import numpy as np
import pandas as pd
from result_analysis.charting import Charter


class TemplateSelection(Enum):
    '''How to select results from templates with multiple signatures'''
    SIMILAR_LEN = 0
    MAX_PROB    = 1
    MIN_PROB    = 2
    MED_PROB    = 3

class SignalAnalyzer:
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor 
    #-------------------


    def __init__(self):
        '''
        Constructor
        '''
        pass
        
    #------------------------------------
    # plot_center_freqs
    #-------------------
    
    @classmethod
    def plot_center_freqs(cls, 
                          sel_tbls, 
                          recordings, 
                          species_list,
                          color=None,
                          ax=None):
        '''
        Chart the frequencies of maximum energy along the time
        of one or more bird call(s).

        Given one or more select table paths and corresponding
        paths to audio recording file, extract calls for each 
        of the specified species as audio clips. Chart the center
        frequencies. Lines for calls by one species will be of the
        same color.
        
        Plot will be x=time, y=frequency. Each point x,y shows the frequency
        holding the maximal energy (spectral centroid) at time x.
        
        If color is None, all lines of one species
        will be of one color. Else color must be a string with 
        a pyplot color ('black', 'mediumblue', etc.). In this case
        the color will be used for all lines during the method execution.
        
        Method may be called multiple times, passing ax back in 
        each time. Lines will be added to that Axes.
        
        :param sel_tbls: path(s) to selection table(s)
        :type sel_tbls: {str | [str]}
        :param recordings: path(s) to corresponding recording(s)
        :type recordings: {str | [str]}
        :param species_list: one or more species whose calls are
            to be plotted.
        :type species_list: {str | [str]}
        :param color: None for automatic color management, else
            one color to use for all lines
        :type color: {None | str}
        :param average_calls: whether or not to average
            the curves all all calls into a single curve
        :type average_calls: bool
        :param ax: optional axes already existing, and returned 
            from earlier calls
        :type ax: matplotlib.axes
        :return axes of the plot
        :rtype plt.Axes
        '''

        # Error checks: must have same number
        # of recordings as selection tables,
        # and we want both as lists:
        if type(sel_tbls) != list:
            sel_tbls = [sel_tbls]
        if type(recordings) != list:
            recordings = [recordings]
        if type(species_list) != list:
            species_list = [species_list]
            
        # More than a sane number of species 
        # looks awful. If more wanted, use a 
        # color list locally, instead of Charter.COLORS:
        if len(species_list) > len(Charter.COLORS):
            raise IndexError(f"Plots are only set up for up to {len(Charter.COLORS)} species.")

        num_tbls = len(sel_tbls)
        num_recs = len(recordings)
        if num_tbls != num_recs:
            raise ValueError(f"Must provide same number of selection tables as recordings, not ({num_tbls}/{num_recs})")

        # One family of curves...
        for color_idx, species in enumerate(species_list):
            # ... with all calls taken from across the tables
            # and their recordings:
            spectral_centroids = cls.compute_species_templates(sel_tbls, 
                                                                recordings, 
                                                                species
                                                                )
            if color is None:
                color_group = {Charter.COLORS[color_idx] : spectral_centroids.index}
            else:
                color_group = {color : spectral_centroids.index}
            ax = Charter.linechart(spectral_centroids, 
                                   ylabel='center frequency (Hz)', 
                                   xlabel=u'time (\u03bcs)',
                                   rotation=45,
                                   color_groups=color_group,
                                   ax=ax)
                
        return ax
    
    #------------------------------------
    # spectral_centroid_each_timeframe
    #-------------------
    
    @classmethod
    def spectral_centroid_each_timeframe(cls, audio, sr):
        '''
        Return a 1D array with the centroid of energy
        across all frequencies for each time slice.
        
        :param audio:
        :type audio:
        :return a 1D Series of frequency centroids, one
            for each time slot
        :rtype pd.Series
        '''
        return pd.Series(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])

    #------------------------------------
    # zero_crossings
    #-------------------
    
    @classmethod
    def zero_crossings(cls, audio):
        '''
        Allow audio as a pd.Series, a pd.DataFrame,
        a 2-tuple of series-name and pd.Series, and
        np array.

        :param audio:
        :type audio:
        '''
        
        if type(audio) == tuple and len(audio) == 2 and type(audio[1]) == pd.Series:
            # Funky tuple with with name being 
            # Series name, and second element
            # being a Series: grap the values np-array:
            clip = audio[1].values
        elif type(audio) == pd.Series:
            clip = audio.values
        elif type(audio) == tuple or type(audio) == np.ndarray:
            # 'Normal' tuple, or np-array: zero_crossing handles it:
            clip = audio
        elif type(audio) == list:
            clip = np.array(audio)
        elif type(audio) == pd.DataFrame:
            # Multiple clips to process:
            clip = audio.to_numpy()
        else:
            raise TypeError(f"Audio must be array-like, not {type(audio)}")
            
        # Get [False, False, True, False, False, True, ...]
        zero_cross_bool_each_frame = librosa.zero_crossings(clip)
        
        # Get indices where a crossing occurs. Result 
        # will be a tuple of np-arrays. For a single audio clip
        # this will be a 1-tuple:
        res_df = pd.DataFrame()
        for row in zero_cross_bool_each_frame:
            res_df = res_df.append(pd.Series(np.nonzero(row)), ignore_index=True)

        return pd.DataFrame(res_df)

    #------------------------------------
    # compute_species_templates
    #-------------------
    
    @classmethod
    def compute_species_templates(cls, species, recordings, sel_tbls):
        '''
        Given a list of selection table paths, and a list
        of paths to the corresponding recordings, isolate
        the audio clips of calls by the given species. 
        
        For each clip in each recording, compute for each time 
        frame the frequency at which energy is maximal. This 
        computation results in one pd.Series, a time series of
        max-energy-holding frequencies over time for each call.
        This time series is the call's "signature".
        
        Package the signatures of one recording into SpectralTemplate
        instance, and return a list of such instances, one for
        each recording. Each instance contains the signatures of
        all calls in one recording. 
        
        :param sel_tbls: list of selection table paths
        :type sel_tbls: [str]
        :param recordings: list of paths to corresponding recordings
        :type recordings: [str]
        :param species: name of species whose calls are to be
            analyzed
        :type species: str
        :returns a list of SpectralTemplate instances, each holding
            the signatures of the calls in one recording. If no content
            for given species is found, returns None
        :rtype [SpectralTemplate]
        '''

        spectral_centroids = None
        res_templates = []
        if type(sel_tbls) != list:
            sel_tbls = [sel_tbls]
            
        if type(recordings) != list:
            recordings = [recordings]
            
        if len(sel_tbls) != len(recordings):
            raise ValueError(f"Number of recordings must equal number of selection tbls.")
        
        for sel_tbl_path, rec_path in zip(sel_tbls, recordings): 
            species_clips_dict, sr = SignalAnalyzer.audio_from_selection_table(
                sel_tbl_path,
                rec_path,
                species)
            # If this selection table has no calls by
            # current species: next species:
            if species not in list(species_clips_dict.keys()):
                continue
            
            spectral_centroids = []
            rec_fname = Path(rec_path).name
            for clip_num, clip in enumerate(species_clips_dict[species]):
                
                centroid = SignalAnalyzer.spectral_centroid_each_timeframe(clip, sr)
                
                num_cols = len(centroid)
                time_step = int(10**6 * 1/sr)
                col_names = list(Interval(0,time_step*num_cols,time_step).values())
                centroid.index = col_names
                centroid.name = f"{rec_fname}_call{clip_num}" 
                
                spectral_centroids.append(centroid)
                
            res_templates.append(SpectralTemplate(spectral_centroids, 
                                                  rec_fname=rec_fname,
                                                  sr=sr))

        return res_templates

    #------------------------------------
    # match_probability
    #-------------------
    
    @classmethod
    def match_probability(cls, 
                          clip, 
                          spectroid_template, 
                          sr,
                          template_selection=TemplateSelection.SIMILAR_LEN
                          ):
        '''
        Given an audio clip, match it to each
        of the given spectral centroid timelines in the
        given SpectralTemplate instance. Determine
        the match with lowest matching cost. On the basis
        of that lowest cost, return the probability that one
        of the template's signatures came from the species that produced
        the clip.
        
        When a template contains multiple signatures, 
        the template_selection arg controls how a final probability
        is chosen TemplateSelection may be:
        
           o SIMILAR_LEN  the signature closest in length to the 
                          signature of the given clip is used
           o MAX_PROB     all template signatures are computed,
                          and the maximum probability is returned.
           o MIN_PROB     all template signatures are computed,
                          and the minimum probability is returned.
           o MED_PROB     all template signatures are computed,
                          and the median probability is returned.
                          

        :param clip: audio of a single call
        :type clip: {np.array | pd.Series}
        :param spectroid_template: one or more SpectralTemplate
            instance(s) against which to compare the clip
        :type spectroid_template: {SpectralTemplate | [SpectralTemplate]}
        :param sr: sampling rate
        :type sr: int
        :param template_selection: how to select probability results
            from templates with multiple signatures
        :type template_selection: TemplateSelection 
        :return probability of clip having been created
            by the species from whose calls the templates were
            made
        :rtype float
        '''

        if type(spectroid_template) != list:
            spectroid_template = [spectroid_template]

        # For convenience:
        if template_selection == TemplateSelection.MIN_PROB:
            sel_by_min = True
            sel_by_max = sel_by_med = False
        elif template_selection == TemplateSelection.MAX_PROB:
            sel_by_max = True
            sel_by_min = sel_by_med = False
        elif template_selection == TemplateSelection.MED_PROB:
            sel_by_max = sel_by_min = False
            sel_by_med = True
        else:
            sel_by_max = sel_by_min = sel_by_med = False

        # Get centroid of the clip:
        clip_signature = cls.spectral_centroid_each_timeframe(clip, sr)
        clip_sig_len   = len(clip_signature)

        # Map signature Series instances to tuples containing
        # a match-probability with clip, and the length of the
        # signature that yielded that probability:
        matching_probs = {}
        
        # Match clip sig against the sigs within
        # each template, collecting the probs from
        # each template match into matching_probs:
        for template in spectroid_template:

            # Pick signature closest in length to the given clip's
            # signature?
            if template_selection == TemplateSelection.SIMILAR_LEN:
                # Find the template signature closest in
                # length to clip_sig_len:
                sig_diffs = []
                for sig in template:
                    sig_diffs.append(abs(clip_sig_len - len(sig)))
                sig_idx = sig_diffs.index(min(sig_diffs))
                template_sig = template[sig_idx]
                prob = cls._compute_prob(clip_signature, template_sig)
                matching_probs[template] = (prob, len(template_sig))
                
            elif sel_by_min or sel_by_max or sel_by_med:
                # Match against all of the template's sigs,
                # and pick the max or min of the resulting probs.
                
                # The following list will hold tuples:
                #    [(probability, signatureLen),
                #               ...
                #    ]
                # The lengths are needed further down:
                probs = []
                for template_sig in template.signatures:
                    probs.append(
                        (cls._compute_prob(clip_signature, template_sig),
                        len(template_sig)
                        )
                    )

                if sel_by_max:
                    # Find the max prob and the len of the sig
                    # that yielded that max probability:
                    best_prob, sig_len = max(probs, key=lambda res_tuple: res_tuple[0])
                elif sel_by_min:
                    best_prob, sig_len = min(probs, key=lambda res_tuple: res_tuple[0])
                else:
                    # Pick the median of the probs:
                    # Sort the prob/len tuples by the probability;
                    # sort() uses the first of each tuple for sorting
                    # automatically:
                    probs.sort()
                    median_idx = int(len(probs) / 2)
                    best_prob, sig_len = probs[median_idx]
                matching_probs[template] = (best_prob, sig_len) 

            else:
                raise ValueError(f"Bad template_selection value: {template_selection}")
        
        # Now matching_prob keys each template to one
        # tuple: (probality, template-sig-len)
        # Aain apply the requested aggregation given in template_selection
        if sel_by_max:
            best_prob, sig_len = max(matching_probs.values(), key=lambda prob_sig_len: prob_sig_len[0]) 
        elif sel_by_min:
            best_prob, sig_len = min(matching_probs.values(), key=lambda prob_sig_len: prob_sig_len[0])
        elif sel_by_med:
            probs_and_lengths = list(matching_probs.values())
            probs_and_lengths.sort()
            median_idx = int(len(probs_and_lengths) / 2)
            best_prob, sig_len = probs_and_lengths[median_idx]
        else:
            # Select using probability of sig with 
            # the length closest to the clip's len:
            len_diffs = []
            probs     = []
            for prob, sig_len in matching_probs.values():
                len_diffs.append(abs(clip_sig_len - sig_len))
                probs.append(prob)
            min_dist_idx = len_diffs.index(min(len_diffs))
            best_prob = probs[min_dist_idx]
            
        return best_prob

    #------------------------------------
    # _compute_prob
    #-------------------
    
    @classmethod
    def _compute_prob(cls, clip_sig, template_sig):
        '''
        Given a clip's spectral centroid signature, and
        a template signature, find the lowest cost sequence
        of steps to align the two wave forms.
        
        Normalize the resulting cost matrix to 0-1, and
        derive a probability from the total cost of the
        alignment.
        
        Both args may be pd.Series or np.array. If the
        template_sig is a Series, it may be taken from 
        a template dataframe, in which case its first
        element will be a column 'TrueLen' indicating the
        length of the signature without NaN padding. Thi
        col is not needed here, and will be detected and
        discarded.
        
        The following is thus a legal idiom:
        
            for row_idx, template_sig in template.iterrows():
                prob = cls._compute_prob(clip_sig, template_sig)
                
        or equivalently:
            pro
        
        :param clip_sig: spectral controid sig of clip
        :type clip_sig: {pd.Series | np.array}
        :param template_sig: one spectral controid sig
            from a signature template
        :type template_sig: {pd.Series | np.array}
        '''
        
        if type(clip_sig) == pd.Series:
            clip_sig_np = clip_sig.to_numpy()
        else:
            clip_sig_np = clip_sig
            
        # Turn template_sig into np, and cut away
        # the leading TrueLen, if it is there:
        
        if type(template_sig) == pd.Series:
            if template_sig.index[0] == 'TrueLen':
                template_sig_np = template_sig.to_numpy()[1:]
            else:
                template_sig_np = template_sig.to_numpy()

        # Use dynamic time warping to find the cheapest
        # way of making the clip sig similar to the template sig:
        cost_matrix, _warp_pairs = librosa.sequence.dtw(clip_sig_np, 
                                                        template_sig_np)

        # Normalize values in the cost matrix to 0-1:
        cost_normalized = (cost_matrix-cost_matrix.min())/(cost_matrix.max()-cost_matrix.min())
        # The value on the lower right of the cost 
        # matrix is the sum of all steps' costs:
        total_cost = cost_normalized[-1,-1]
        prob = 1 - total_cost
        return prob
    
    
    #------------------------------------
    # df_from_clips
    #-------------------
    
    @classmethod
    def df_from_clips(cls, clip_list):
        '''
        Given a ragged list of arrays, pad each
        array to the length of the longest, using
        the mean of the respective array.
        
           [
           [1,2,3],
           [4,5,6,7,8]
           ]
           
        Yields:
           df([1,2,3,2,2],
              [4,5,6,7,8]
              ])
              
        :param clip_list: list of arrays
        :type clip_list: [{list[list] | np.array | pd.DataFrame}
        :return dataframe with equal-length rows,
            each row being the data from one clip
        :rtype pd.DataFrame
        '''
        # Ensure each sub-array is an np array:
        if type(clip_list) != pd.DataFrame:
            # Create a ragged df from the arrays
            df = pd.DataFrame()
            for clip in clip_list:
                # Append ragged arrays; they are auto-padded
                # with nan:
                df = df.append(pd.Series(clip), ignore_index=True)
        else:
            # clip_list is already a ragged df:
            df = clip_list
            
        # Replace NaNs in each row with the means of the row:
        df = df.apply(lambda row: row.fillna(row.mean()), axis=1)
        return df

# ------------- Utilities ----------

    #------------------------------------
    # audio_from_selection_tables
    #-------------------
    
    @classmethod
    def audio_from_selection_table(cls, 
                                   sel_tbl_path,
                                   recording_path,
                                   requested_species=None,
                                   ):
        '''
        Comb through the given selection table, and
        extract audio snippets by requested_species.
        Return a dict mapping each species name to
        a list of audio clips extracted from the full
        recording that corresponds to the selection
        table.  
        
        :param sel_tbl_path: path to selection table
        :type sel_tbl_path: src
        :param recording_path: path to recording associated
            with the selection table
        :type recording_path: str
        :param requested_species: individual or list of species to
            extract. If None, extract all species
        :type requested_species: {None | [str]}
        :return dict mapping species name to list of
            audio clips, and sampling rate
        :rtype ({str : [np.array[float]}, int)
        '''
        
        # Get a list of dicts, one dict per select
        # table. Dicts hold:
        #
        #     Selection           row number <int>
        #     View                not used
        #     Channel             not used
        #     Begin Time (s)      begin of vocalization in fractional seconds <float>
        #     End Time (s)        end of vocalization in fractional seconds <float>
        #     Low Freq (Hz)       lowest frequency within the lassoed vocalization <float>
        #     High Freq (Hz)      highest frequency within the lassoed vocalization <float>
        #     species             four-letter requested_species name <str>
        #     type                {song, call, call-1, call-trill} <str>
        #     number              not used
        #     mix                 comma separated list of other audible requested_species [<str>]
        #
        #     time_interval       Inteval instance from start and end times
        #     freq_interval       Inteval instance from start and end frequencies
        
        if requested_species is not None and type(requested_species) != list:
            requested_species = [requested_species]
        # Ensure upper case:
        requested_species = [species.upper() for species in requested_species]
        
        audio_result_dict = {}
        row_dicts = Utils.read_raven_selection_table(sel_tbl_path)

        # If particular requested_species are requested,
        # find the dicts in which they are captured:
        relevant_dicts = []
        if requested_species is not None:
            for tbl_row_dict in row_dicts:
                for species in requested_species:
                    if species == tbl_row_dict['species'] or species in tbl_row_dict['species']:
                        relevant_dicts.append(tbl_row_dict)
        else:
            # Interested in all species:
            relevant_dicts = row_dicts

        # Extract the audio (slow):
        audio, sr = SoundProcessor.load_audio(recording_path)
        for sel_dict in relevant_dicts:
            # Process one row of the selection table:
            species = sel_dict['species']
            clip = SoundProcessor.extract_clip(audio, 
                                               sr, 
                                               sel_dict['Begin Time (s)'], 
                                               sel_dict['End Time (s)']
                                               )
            try:
                audio_result_dict[species].append(clip)
            except KeyError:
                # First one:
                audio_result_dict[species] = [clip]
        return audio_result_dict, sr

# -------------------- Class SpectralTemplate -----------

class SpectralTemplate:
    '''
    Hold the spectral centroid timelines (signatures)
    of all calls in one recording.
    
    Usage:
        o <inst>.signatures   : list of pd.Series with the 
                                frequency-of-max-energy timeline
        o iter(<inst>)        : return iterator over signature Series
        o <inst>[n]           : return the nth signature
        o len(<inst>)         : number of signatures
        o <inst>.sig_lengths  : list of lengths of the signatures
        o <inst>.mean_sig     : signature that is the mean of all
                                signatures' frequencies at each
                                time.
    '''
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, 
                 signatures, 
                 rec_fname=None,
                 sr=None):
        
        # Turn all ways of passing signatures
        # into a list of signatures:
        if type(signatures) == pd.Series:
            self.signatures = [signatures]
        elif type(signatures) == pd.DataFrame:
            self.signatures = []
            for row_label, sig in signatures.iterrows():
                clean_sig = sig.dropna()
                clean_sig.index = [f"t{i}" for i in range(len(sig))]
                clean_sig.name  = row_label
                self.signatures.append(clean_sig)
        # Hopefully it's a list of pandas Series:
        elif type(signatures) == list and all([type(el) == pd.Series for el in signatures]):
            self.signatures = signatures
        else:
            raise TypeError(f"Signatures must be a pd.Series, a list of pd.Series, or a pd.DataFrame, not {type(signatures)}")
        
        self.rec_fname = rec_fname
        self.sr = sr

        # Computed lazily:
        self.cached_mean_sig = None

    #------------------------------------
    # mean_sig
    #-------------------

    @property
    def mean_sig(self):
        if self.cached_mean_sig is not None:
            return self.cached_mean_sig
        
        max_sig_len = max(self.sig_lengths)
        longest_seq = [sig 
                       for sig 
                       in self.signatures 
                       if len(sig) == max_sig_len
                       ][0]
        df = pd.DataFrame()
        for sig in self.signatures:
            # Number of needed pads
            sig_len = len(sig)
            pad_len = max_sig_len - sig_len
            # Get pad value for this signature:
            mean_freq = sig.mean()
            
            # Col names for the padding columns
            # will match the corresponding col names
            # in the longes sig: 
            col_names = longest_seq.index[sig_len :]
            
            # Pads are the mean of the sig:
            pad_series = pd.Series([mean_freq]*pad_len, 
                                   index=col_names)
            
            padded_sig      = sig.append(pad_series)
            padded_sig.name = sig.name
            # Now sig is as long as the longest seq,
            # so can append to df without generating
            # NaNs:
            df = df.append(padded_sig)
        
        # Finally: take the column-wise mean,
        # i.e. the mean of frequencies at each
        # time frame, generating a single pd.Series:
        mean_sig = df.mean(axis=0)
        mean_sig.name = 'mean_sig'
        self.cached_mean_sig = mean_sig
        return mean_sig

    #------------------------------------
    # sig_lengths
    #-------------------
    
    @property
    def sig_lengths(self):
        
        return [len(sig) for sig in self.signatures]

    #------------------------------------
    # recording_fname
    #-------------------
    
    @property
    def recording_fname(self):
        return self.rec_fname

    #------------------------------------
    # sample_rate
    #-------------------
    
    @property
    def sample_rate(self):
        return self.sr

    #------------------------------------
    # __getitem_
    #-------------------

    def __getitem__(self, idx):
        return self.signatures[idx]

    #------------------------------------
    # __len__
    #-------------------
    
    def __len__(self):
        return len(self.signatures)
    
    #------------------------------------
    # __iter__
    #-------------------
    
    def __iter__(self):
        
        return iter(self.signatures)

    #------------------------------------
    # __repr__
    #-------------------
    
    def __repr__(self):
        return f"<SpectralTemplate ({len(self.signatures)} sigs) {hex(id(self))}>"

    #------------------------------------
    # __str__
    #-------------------
    
    def __str__(self):
        return self.__repr__()
    
