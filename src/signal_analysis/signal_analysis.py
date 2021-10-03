'''
Created on Oct 1, 2021

@author: paepcke
'''
import librosa

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils, Interval
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
    # plot_call_center_freqs
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
                spectral_centroids = spectral_centroids.apply(
                    lambda row: row.fillna(row.mean()),
                    axis=1
                    )
                
                _num_rows, num_cols = spectral_centroids.shape
                time_step = int(10**6 * 1/sr)
                col_names = list(Interval(0,time_step*num_cols,time_step).values())
                spectral_centroids.columns = col_names
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
    # energy_profile 
    #-------------------
    
    @classmethod
    def energy_profile(cls, bandwidth, spectro):
        '''
        Given a spectrogram as a dataframe or np array
        :param bandwidth:
        :type bandwidth:
        :param spectro:
        :type spectro:
        '''
        pass
    
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
