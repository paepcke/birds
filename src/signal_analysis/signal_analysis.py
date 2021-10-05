'''
Created on Oct 1, 2021

@author: paepcke
'''
from pathlib import Path

import librosa

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils, Interval
import numpy as np
import pandas as pd
from result_analysis.charting import Charter


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
                          average_calls=False,
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
            spectral_centroids = cls.compute_spectral_centroids(sel_tbls, 
                                                                recordings, 
                                                                species,
                                                                average_calls=average_calls)
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
    # compute_spectral_centroids
    #-------------------
    
    @classmethod
    def compute_spectral_centroids(cls, sel_tbls, recordings, species, average_calls):
        '''
        Given a list of selection table paths, and a list
        of paths to the corresponding recordings, isolate
        the audio clips of calls by the given species. 
        
        Compute for each time frame the frequency at which
        energy is maximal. This computation results in one 
        time Series for each call.
        
        If average_calls is False, a dataframe is returned
        whose rows are the maximal energy values along time, 
        and whose columns are time since start of call. The
        result df will thus contain as many rows as there are
        calls by the specified species.
        
        If average_calls is True, then at each time point the
        frequencies that hold the maximum energy are avaraged,
        resulting in a single line.

        :param sel_tbls: list of selection table paths
        :type sel_tbls: [str]
        :param recordings: list of paths to corresponding recordings
        :type recordings: [str]
        :param species: name of species whose calls are to be
            analyzed
        :type species: str
        :param average_calls: whether or not to average the 
            frequencies of each call over each time frame
        :type average_calls: bool
        :returns a dataframe with result values. If no content
            for given species is found, returns None
        :rtype pd.DataFrame
        '''

        spectral_centroids = None
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
            
            spectral_centroids = pd.DataFrame([])
            
            for clip in species_clips_dict[species]:
                centroid = SignalAnalyzer.spectral_centroid_each_timeframe(clip, sr)
                spectral_centroids = spectral_centroids.append(centroid, ignore_index=True)
    
            # Replace the nan from unequal clip lengths 
            # with the mean of the respective row:
            spectral_centroids = cls.df_from_clips(spectral_centroids)
            
            _num_rows, num_cols = spectral_centroids.shape
            time_step = int(10**6 * 1/sr)
            col_names = list(Interval(0,time_step*num_cols,time_step).values())
            spectral_centroids.columns = col_names

            if average_calls:
                spectral_centroids = spectral_centroids.mean(axis=0)

        return spectral_centroids

    #------------------------------------
    # compute_species_templates
    #-------------------
    
    @classmethod
    def compute_species_templates(cls, species, recordings, sel_tbls):
        '''
        From a set of labeled recordings, create one
        Series of frequencies at which energy is maximal
        for each successive time frame. Time frames are of
        with 1/sampleRate. 
        
        Return a dataframe with one row per recording. The 
        row contains the averaged spectral centroid (freq/time)
        for the calls in one recording. The first column holds
        the true (non-NaN-padded) length of the row.
        
        The recordings must correspond to the selection tables.
        
        The return might look like this:
        
                            TrueLen  t0       t1        ...
            recording	 
            foo_bird1.mp3     133    1000.4  1400.2    ...
            foo_bird2.mp3      45    1100.1  14001.4   ...
        
        Recordings are typically from a clean source, such as
        Xeno Canto.
        
        NOTE: the length of the spectral centroid lines are
               different for different recordings. So each
               row will contain filler NaNs. When using each
               row (i.e. one spectral line), use dropna() to
               remove the NaNs.
        
        :param species: species for which templates are to be
            computed
        :type species: str
        :param recordings: one or more paths to recordings that contain
            calls from the target species. Calls by other species 
            are ignored
        :type recordings: {str | [str]}
        :param sel_tbls: one or more paths to selection tables
            that identify where in the corresponding recording
            calls from the target species occur
        :type sel_tbls: {str | [str]}
        :return dataframe with one row per recording; each row
            holds at column 't' the frequency at which the calls
            in that recording hold max energy on average across the
            calls in that recording
        :rtype pd.DataFrame[float]
        :raise ValueError if number of selection tables does not match
            the number of recordings.
        '''

        # Get a Series of frequencies with maximal energy over time.
        # It's a single line, gained from computing spectral centroids
        # for each call of each recording, and averaging the frequencies:
        
        # spectral_centroids = cls.compute_spectral_centroids(sel_tbls,   
        #                                                     recordings, 
        #                                                     species,
        #                                                     average_calls=True)
        
        template_lines = pd.DataFrame([])
        # Keep track of how long the centroid lines
        # are for each recording:
        template_lengths = []
        
        for sel_tbl, recording in zip(sel_tbls, recordings):
            # Get average spectral line from the calls
            # of one recording:
            spectral_centroid = cls.compute_spectral_centroids(sel_tbl,
                                                               recording, 
                                                               species,
                                                               average_calls=True)
            spectral_centroid.name = Path(recording).name
            # Remember length of the series
            template_lengths.append(len(spectral_centroid))
            # Append to result df, padding with NaNs
            template_lines = template_lines.append(spectral_centroid)
            
        template_lines.index.name = 'recording'
        _num_rows, num_cols = template_lines.shape
        
        template_lines.columns=[f"t{i}" for i in range(num_cols)]
        
        # Put the true (non-padded) length of each row
        # as column 0:
        template_lines.insert(0, 'TrueLen', template_lengths) 
        
        
        # Remember: rows contain filler NaN that
        # you want to drop when using a row.
        return template_lines

    #------------------------------------
    # match_probability
    #-------------------
    
    @classmethod
    def match_probability(cls, clip, spectroid_templates, sr):
        '''
        Given an audio clip, match it to each
        of the given spectral centroid timelines. Determine
        the match with lowest matching cost. On the basis
        of that lowest cost, return the probability that one
        of the templates came from the species that produced
        the clip.
        
        Best results if the length of the audio clip is close
        to the length of the template(s).
        
        For the format for submitting multiple spectroid_templates
        to match agains, see header comment of compute_species_templates().
        
        :param clip: audio of a single call
        :type clip: {np.array | pd.Series}
        :param spectroid_template: one or more spectral centroid
            lines computed (via compute_species_templates()) from
            a known species to which the clip is to be compared
        :type spectroid_template: {pd.Series | pd.DataFrame}
        :param sr: sampling rate
        :type sr: int
        :return probability of clip having been created
            by the species from whose calls the templates were
            made
        :rtype float
        '''

        if type(spectroid_templates) == pd.Series:
            templates = pd.DataFrame(np.hstack([np.array(len(spectroid_templates)),
                                                spectroid_templates.values]))
        else:
            templates = spectroid_templates

        # Get centroid of the clip:
        clip_centroid = cls.spectral_centroid_each_timeframe(clip, sr)
        matching_probs = []
        for _row_num, template_centroid_ser in templates.iterrows():
            # Centroid is a Series whose first element is
            # its length without NaN fillers. Get a numpy
            # array without that first col, and without any
            # NaNs:

            template_centroid_np = template_centroid_ser.dropna().values[1:]
            clip_centroid_np     = clip_centroid.to_numpy()
            cost_matrix, _warp_pairs = librosa.sequence.dtw(clip_centroid_np, 
                                                            template_centroid_np)

            cost_normalized = (cost_matrix-cost_matrix.min())/(cost_matrix.max()-cost_matrix.min())
            total_cost = cost_normalized[-1,-1]
            prob = 1 - total_cost
            matching_probs.append(prob)

        return max(matching_probs)

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
