'''
Created on Oct 1, 2021

@author: paepcke
'''
from collections import namedtuple
from enum import Enum
from pathlib import Path

import librosa

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils, Interval
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from result_analysis.charting import Charter


class TemplateSelection(Enum):
    '''How to select results from templates with multiple signatures'''
    SIMILAR_LEN = 0
    MAX_PROB    = 1
    MIN_PROB    = 2
    MED_PROB    = 3
    MEAN_PROB   = 4

ClipInfo = namedtuple("ClipInfo", "clip start_idx end_idx fname sr")


class SpectralAggregation(Enum):
    '''Method for creating a signature entry from a time slice across FFT frequencies'''
    MIN    = 'Min'
    MAX    = 'Max'
    MEAN   = 'Mean'
    MEDIAN = 'Median'

DEFAULT_SPECTRAL_AGGREGATION = SpectralAggregation.MAX
'''Method for computing one signature value from the frequencies of one time frame'''

class SignalAnalyzer:
    '''
    classdocs
    '''
    # Decide on STFT parameters, ensuring that 
    # even short audio clips get a good amount of signature
    # points across a good number of frequencies:
    # Formulas:
    #
    #    num_time_frames = audio_clip_duration * sr / hop_length
    #                    = num_audio_samples / hop_length
    #    
    #    num_time_frames per sec = sr/hop_length
    #
    #    num_frequency_rows = 1 + n_fft/2
    #           n_fft       = 2 * (num_frequency_rows - 1)

    # ~88 time frames per second:
    hop_length = 256
    # 1025 frequency bins:
    n_fft = 2048
    
    sr = 22050
    
    # Meaning of the following:
    # When an audio clip is tested for degree of match
    # against a species signature, multiple measures of
    # overlap quality are taken, lifting a signature-length 
    # snippet from the clip each time. The window that defines
    # the snippet is slid over by some time duration. 
    # The following quantitiy is that time duration:
    #
    # Ex for 12 samples in a signature, and 18
    #    samples in the audio clip to test. 
    #    SIGNATURE_MATCH_SLIDE_TIME == 0.1 sec
    #
    #     Audio: abcdefghejklmnopqrs
    # Signature: xxxxxxxxxxxx
    #
    # First measurement:
    #          abcdefghejkl
    #          xxxxxxxxxxxx
    #
    # Second measurement (slide to right by 0.1 sec * sr s/sec 
    #          defghejklmno          
    #          xxxxxxxxxxxx
    #
    # Third measurement:
    #
    #          jklmnopqrsMM   <--- MM is mean of j-s
    #          xxxxxxxxxxxx          
    
    SIGNATURE_MATCH_SLIDE_TIME = 0.1 # seconds
    '''Fraction of signature length to slide test clips before each match check'''

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
        paths to audio recording files, extract calls for each 
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
            spectral_centroids = cls.compute_species_templates(species,
                                                               recordings, 
                                                               sel_tbls, 
                                                               )
            # Get df like:
            #                t0         t1        t2 ...
            #    'sparrow0'  1000.0    1100.9     ...
            #    'sparrow1'  1050.3    1030.9     ...
            
            centroids_df = pd.DataFrame(spectral_centroids[0].signatures)
            row_labels = centroids_df.index.values
            
            if color is None:
                color_group = {Charter.COLORS[color_idx] : row_labels}
            else:
                color_group = {color : row_labels}
                #******!!!!!{color : spectral_centroids[0].index}
                            
            ax = Charter.linechart(centroids_df, 
                                   ylabel='center frequency (Hz)', 
                                   xlabel=u'time (\u03bcs)',
                                   rotation=45,
                                   color_groups=color_group,
                                   ax=ax)
                
        return ax

    #------------------------------------
    # spectral_flatness
    #-------------------
    
    @classmethod
    def spectral_flatness(cls,
                          spec_df=None,
                          audio=None,
                          bandpass=None,
                          wiener_entropy=False
                          ):
        '''
        Given audio or an already computed spectrogram,
        go through each time frame, compute the 'spectral flatness'
        a.k.a. Wiener entropy. Return a time series of flatness
        values.
        
        Flatness values will be in [0,1] with 1 being very flat, corresponding
        to mostly noise. Near zero means presence of a tone with harmonics.
        A pure sine wave has no harmonics, and will have flatness of 1.
        
        Either spec_df or audio must be provided.btest
        
        :param spec_df: optionally the spectrogram computed as with
           raven_spectrogram(). 
        :type spec_df: pd.DataFrame
        :param audio: optionally an audio array
        :type audio: np.ndarray(1)
        :param bandpass: if provided, the specification of a bandpass filter
            to apply prior to computation
        :type bandpass: {None | Interval}
        :param wiener_entropy: if True, convert values from spectral
            flatness to Wiener entropy. The flatness values [0,1] are thereby
            converted to [-inf, 0], i.e. taking the log of the flatness values
        :type wiener_entropy: bool
        :return series of values [0,1] corresponding to the spectral flatness
            at each time frame.
        :rtype pd.Series
        
        '''
        if spec_df is None and audio is None:
            raise ValueError("Either spectrogram or audio must be provided.")
        
        if spec_df is None:
            # Get spectrogram from audio:
            if type(audio) == pd.Series:
                audio_np = audio.as_numpy()
            else:
                audio_np = audio
            # Spectral flatness computations need absolute energy
            # values, not decibels relative to max energy:
            spec_df = cls.raven_spectrogram(audio_np, to_db=False)
        
        if bandpass is not None:
            cls.apply_bandpass(bandpass, spec_df, inplace=True)
        
        # The 'power=2' tells librosa that we want to work
        # the the power spectrum, on which flatness is usually shown:
        flatness_arr = librosa.feature.spectral_flatness(S=spec_df.to_numpy(), power=2)
        # The result is an np arr of the form:
        #   [[0.4,0.6,...]]
        # with length equal to the number of time frames.
        # The [-1] here pulls out the numbers we want:
        flatness_ser = pd.Series(flatness_arr[-1], index=spec_df.columns, name="SigSpectralFlatness")
        
        if wiener_entropy:
            # Take the log10 of each element, taking
            # care of 0 values by setting those to -inf:  
            flatness_ser = flatness_ser.apply(lambda el: -np.inf if el == 0 else np.log10(el))
            
        return flatness_ser

    #------------------------------------
    # spectral_continuity
    #-------------------
    
    @classmethod
    def spectral_continuity(cls, 
                            spec_df=None,
                            audio=None,
                            bandpass=None,
                            edge_mag_thres=1):
        '''
        Given a spectrogram DataFrame as might be created from
        raven_spectrogram(), or an audio file. If necessary, create
        the spectrogram. Compute a single number for each time frame,
        ********CONTINUE HERE. 
        
        For each frequency band 
        
        :param cls:
        :type cls:
        :param spec_df:
        :type spec_df:
        :param audio:
        :type audio:
        :param bandpass:
        :type bandpass:
        :param edge_mag_thres:
        :type edge_mag_thres:
        '''
        
        if spec_df is None:
            spec_df = cls.raven_spectrogram(audio, to_db=False)
        if bandpass is not None:
            cls.apply_bandpass(bandpass, spec_df, inplace=True)

        num_rows, num_cols = spec_df.shape

        # Get df same size as spec_df with a True in cells
        # that are negative, and False in cells that are positive:
        #******spec_is_negative = np.signbit(spec_df)
        
        # Another df of same size as spec_df where each cell
        # has result of:
        #  T(ti,fj) = T*(abs(Wiener_entropy(col))/(abs(row_freq - mean_freq(col))))
        wiener_ser = cls.spectral_flatness(spec_df, wiener_entropy=True)
        
        # Make a df where each row is a copy of the wiener series
        # with as many rows as spec_df has:
        wiener_df = pd.DataFrame([wiener_ser.abs()] * num_rows,
                                 index=spec_df.index,
                                 columns=spec_df.columns
                                 )
        
        # For each spectrum cell, get the difference of its magnitude
        # from the mean magnitude in its column (i.e. during the time
        # frame across all freqs):
        energy_distance_from_mean = (spec_df - spec_df.mean(axis='rows')).abs()

        edge_mag_thres = edge_mag_thres * wiener_df/energy_distance_from_mean
        
        thresholded_spec_df = spec_df >= edge_mag_thres
        
        Charter.draw_contours(thresholded_spec_df,
                              title='CMTO Countours',
                              xlabel='Time',
                              ylabel='Frequency',
                              decimals_x=2,
                              decimals_y=2)
        
        print(edge_mag_thres)

    #------------------------------------
    # apply_bandpass
    #-------------------
    
    @classmethod
    def apply_bandpass(cls, bandpass, spec_df, inplace=False):
        '''
        Given a spectrogram and bandpass frequency interval, 
        such as Interval(5000, 8000, 1), return a new spectrogram
        with frequencies above and below the given interval set
        to -inf.
        
        :param bandpass: specification of frequencies to retain 
        :type bandpass: Interval
        :param spec_df: the spectrogram: index is frequencies, 
            columns are time frames.
        :type spec_df: pd.DataFrame
        :return filtered spectrogram of equal size as the original
        :rtype pd.DataFrame
        '''

        if not isinstance(bandpass, Interval):
            raise TypeError(f"Bandpass must be an Interval of floats or ints, not {bandpass}") 
        # Remember that spectrograms are
        # arranged high frequency to low frequency
        # to match what visual spectros show. The
        # "- 1" keeps the low bound included
        low_freq, high_freq = bandpass['low_val'] - 1, bandpass['high_val'] 
        # Reason for setting unwanted part of the freq spectrum
        # to -np.inf:
        # Values are in dB FS (i.e. relative to the maximum signal).
        # So the vals are all negative. When we will look
        # for, say, max values, we don't want that to be zero:
        if inplace:
            filtered_spec = spec_df
        else:
            filtered_spec = spec_df.copy()
        filtered_spec[filtered_spec.index >= high_freq] = -np.inf
        filtered_spec[filtered_spec.index < low_freq] = -np.inf
        
        return filtered_spec

    #------------------------------------
    # spectral_centroid_each_timeframe
    #-------------------

    @classmethod
    def spectral_centroid_each_timeframe(cls, 
                                         audio,
                                         bandpass=None,
                                         aggregation=None):
        '''
        Return a 1D array with the centroid of energy
        across all frequencies for each time slice.
        
        There will be ~88 signature entries (time frames)
        per second, and the mean energy at each time frame
        will be taken across 1025 frequencies.
        
        :param audio: audio in time domain
        :type audio: {np.array | pd.Series}
        :param bandpass: Interval with low and high cutoff for
            masking frequencies below and above a frequency
            range of interest.
        :type bandpass: {None | Interval}
        :param aggregation: method of summarizing the frequencies
            in one (vertical) time frame in the spectrogram
        :type aggregation: {None | SpectralAggregation}
        :return a 1D Series of frequency centroids, one
            for each time slot. Values are frequencies, index
            entries are time
        :rtype pd.Series
        '''
        # if hop_length is None:
        #     hop_length = cls.hop_length
        # if n_fft is None:
        #     n_fft = cls.n_fft
        # if normalize:
        #     audio = sklearn.preprocessing.minmax_scale(audio,feature_range=(-1,1))
        #
        # sig_ser = pd.Series(librosa.feature.spectral_centroid(y=audio, 
        #                                                       sr=sr,
        #                                                       hop_length=hop_length,
        #                                                       n_fft=n_fft
        #                                                       )[0])
        
        if aggregation is None:
            aggregation = DEFAULT_SPECTRAL_AGGREGATION
        
        spec_df = cls.raven_spectrogram(audio)
        if bandpass is not None:
            cls.apply_bandpass(bandpass, spec_df, inplace=True)

        if aggregation == SpectralAggregation.MAX:
            sig_ser = spec_df.max()
        elif aggregation == SpectralAggregation.MEAN:
            # Need to exclude the -np.inf values that we
            # introducted above from being considered for
            # mean/median/min:
            sig_ser = spec_df[~spec_df.isin([-np.inf])].mean()
        elif aggregation == SpectralAggregation.MEDIAN:
            sig_ser = spec_df[~spec_df.isin([-np.inf])].median()
        elif aggregation == SpectralAggregation.MIN:
            sig_ser = spec_df[~spec_df.isin([-np.inf])].min()
        else:
            raise ValueError(f"Aggregation method must be a member of SpectralAggregation, not {aggregation}")

        sig_ser.index = spec_df.columns
        sig_ser.name  = f"Sig{aggregation.value}"
        
        # We now have a series:
        #   t0   db_val0
        #   t1   db_val1
        #     ...
        # But we need:
        #   t0   frequency at which power is db_val0
        #   t1   frequency at which power is db_val1
        #     ...
        #
        # The needed frequencies are the values of spec_df's
        # index at the rows where power is of desired value
        # at the column's time.
        #
        # There is likely *some* way to bludgeon pandas
        # in yielding these values... But bird calls are
        # short, and therefore sig_ser is short, and we
        # can affort to loop through the columns.
        
        sig_freqs = []
        for sig_idx, desired_val in enumerate(sig_ser):
            freqs_with_max_val = spec_df.index[spec_df.iloc[:,sig_idx] == desired_val].tolist()
            # At a given time there could be multiple
            # frequencies with the desired value. Just pick
            # the first for now:
            sig_freqs.append(freqs_with_max_val[0])
            
        # Final result:
        sig_series = pd.Series(sig_freqs, index=sig_ser.index, name=sig_ser.name)

        return sig_series


    #------------------------------------
    # raven_spectrogram
    #-------------------
    
    @classmethod
    def raven_spectrogram(cls, audio, to_db=True):
        '''
        Returns a spectrogram as the default settings in 
        the Raven program would generate. Same frequency
        and time scales. The index will be frequencies at
        the centers of the fft frequency bands. The columns
        will be time in seconds. 
        
        To visualize the returned spectrogram:
        
            mesh = plt.pcolormesh(spectro.columns, list(spectro.index), spectro, cmap='jet', shading='flat')
        
        To get the axes from the mesh:
        
            mesh.axes
            
        If axes is already available:
        
            mesh = ax.pcolormesh(spectro.columns, list(spectro.index), spectro, cmap='jet', shading='flat')    
            

        :param audio: either the path to an audio file,
            or the already loaded audio array
        :type audio: {str | np.array(float)
        :param to_db: whether to convert the energy values to db
            relative to the max. Raven does this conversion, so
            the default is True. But procedures such as spectral flatness
            computations need the absolute values.
        :type to_db: bool
        :return: dataframe with a spectrogram that reflects
            what a Raven labeling software user would see
            in its spectrogram display.
        :rtype: pd.DataFrame
        '''
        sr = 22050
        hop_length = 512
        if type(audio) == str:
            audio_arr, aud_sr = librosa.load(audio)
            # Adjust sr if necessary:
            if aud_sr != sr:
                audio_arr = librosa.resample(audio_arr, aud_sr, sr)  
        else:
            audio_arr = audio 

        spec = np.abs(librosa.stft(audio_arr, n_fft=512, hop_length=512, window='hann'))
        # Convert to dB readings
        if to_db:
            spec_values = librosa.amplitude_to_db(spec, ref=np.max)
        else:
            spec_values = spec
        num_rows, num_cols = spec_values.shape
        time_ticks = librosa.core.frames_to_time(np.arange(num_cols + 1), 
                                                 sr=sr, 
                                                 hop_length=hop_length)[:-1]
        freq_ticks = cls._freq_ticks(num_rows)[:-1]
        spec_df = pd.DataFrame(spec_values, columns=time_ticks, index=freq_ticks)
        # This df is ordered freq at zero at the top; reverse the order
        # of the rows:
        spec_df = spec_df.iloc[::-1]
        
        # Normalize so that all intensity values are 
        # relative to the highest value in the spectrum:
        
        return spec_df

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
    def compute_species_templates(cls,
                                  species,
                                  recordings,
                                  sel_tbls,
                                  apply_bandpass=False,
                                  aggregation_method=DEFAULT_SPECTRAL_AGGREGATION
                                  ):
        '''
        Given a list of selection table paths, and a list
        of paths to the corresponding recordings, isolate
        the audio clips of calls by the given species. 
        
        For each clip in each recording, compute for each time 
        frame the frequency at which energy is the mean of energies
        at this frame. This computation results in one pd.Series, 
        a time series of mean-energy-holding frequencies over time 
        for each call. This time series is the call's "signature".
        
        Package the signatures of one recording into SpectralTemplate
        instance, and return a list of such instances, one for
        each recording. Each instance contains the signatures of
        all calls in one recording. 
        
        In order to match audio snippets to calls in the template,
        the snippet centroids will also be computed. That computation
        must use the same FFT parameters used when computing the
        signatures in this template. The SpectralTemplate instance
        will therefore contain:
        
            n_fft          number of frequencies (rows in spectrogram)
                           num_rows = 1 - n_fft/2
            sr
            hop_length
            win_length
            window
        
        :param sel_tbls: list of selection table paths
        :type sel_tbls: [str]
        :param recordings: list of paths to corresponding recordings
        :type recordings: [str]
        :param species: name of species whose calls are to be
            analyzed
        :type species: str
        :param apply_bandpass: optional tuple (low_frequency, high_frequency)
            below and above which the spectrogram will be set to 0 before
            computing the signatures.
        :type apply_bandpass: {None, ({int | float}, {int | float})
        :param aggregation_method: method for computing the signature
            from frequencies of a single time frame. See SpectralAggregation
            enum. Default is set in DEFAULT_SPECTRAL_AGGREGATION.
        :type aggregation_method: SpectralAggregation
        :param apply_bandpass: whether or not to apply a
            bandpass filter based on the Raven selection table's high/low
            frequencies for each selection before creating each respective
            signature
        :type apply_bandpass: bool
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
            for clip_num, clip_info in enumerate(species_clips_dict[species]):
                
                clip = clip_info['clip']
                if apply_bandpass:
                    bandpass_spec = clip_info['freq_interval']
                else:
                    bandpass_spec = None
                #****** RETURNS series of all zeroes:
                centroid = SignalAnalyzer.spectral_centroid_each_timeframe(
                    clip,
                    bandpass=bandpass_spec,
                    aggregation=aggregation_method
                    )

                sig = Signature(centroid,
                                sr=sr,
                                start_idx=clip_info['start_idx'],
                                end_idx=clip_info['end_idx'],
                                species=species,
                                fname=rec_path,
                                sig_id=clip_num,
                                audio=clip,
                                freq_interval=clip_info['freq_interval'],
                                bandpass_filter=bandpass_spec
                                )
                spectral_centroids.append(sig)
                
            res_templates.append(SpectralTemplate(spectral_centroids, 
                                                  rec_fname=rec_fname,
                                                  sr=sr,
                                                  hop_length=cls.hop_length,
                                                  n_fft=cls.n_fft
                                                  ))

        return res_templates

    #------------------------------------
    # match_probability
    #-------------------
    
    @classmethod
    def match_probability(cls, 
                          audio,
                          spectroid_template,
                          slide_width_time=None,
                          aggregation=None
                          ):
        '''
        Given an audio clip, match it to each signatures
        in each of the given spectral centroid timelines.
        For each signature of duration D, partition the audio
        into equal sized subclips of duration D.
        
        Compute the power centroid timeline for each subclip, and
        determine the probability that the subclip's content is
        the same as the audio that underlies the signature.
        
        Return a pd.DataFrame with all results, and a pd.Series with
        a summary.
        
        The full-information returned dataframe: 

        Columns of the returned dataframe looks like this:
        
               num_samples  probability  sig_id  start   stop
                
        where each row is the result of matching one subclip of the 
        given audio to one signature from one template. The size
        of the subclip will be the size of the signature being matched
        at the moment.
        
            o start is the index to the first audio sample used of the audio
            o end is one index beyond the last audio sample used
            o n_samples is the length in samples of the subclip being matched
                (i.e.: end - start)
            o probability is the probability that the row's subclip
                matches the row's signature
            o sig_id is a numeric identifier of the signature. The sig_id
                is unique across the list of templates, and is useful
                as a groupby variable for aggreation over subclip results
                per signature.

        Example return df:
        
               n_samples    probability  sig_id  start   stop
            0         10.0         0.85     0.0   10.0   20.0
            1         10.0         0.85     0.0   20.0   30.0
            2         10.0         0.75     0.0   31.0   40.0
            3         10.0         0.65     0.0   41.0   50.0
            4          4.0         0.65     2.0    0.0    4.0
            5          4.0         0.55     2.0    5.0    9.0
            6        100.0         0.65     3.0    0.0  100.0
            7        100.0         0.55     3.0  101.0  200.0

                
        Example uses of the dataframe
        
           highest_probs = df.groupby(by='sig_id').max().probability
                sig_id
                0.0    0.85
                2.0    0.65
                3.0    0.65
                Name: probability, dtype: float64
            
           highest_prob_sig_id2 = df.groupby(by='sig_id').max().probability.iloc[1]
                0.65
        
        Using the groupby instance for clarity, and to avoid retyping 
        "df.groupby(by='sig_id')": 
        
           gb = df.groupby(by='sig_id')
           gb.max().probability
           
        Example 'median of all probabilities':
        
           df.probability.median()
                0.65
                
        Example: highest probability among the longest
            signature:
            
          longest_sig = df['num_samples'].max()
          df[df['num_samples'] == longest_sig].probability.max()
                0.65

        The summary:
        
        The second returned value is a pd.Series with probabilities
        aggregated across all the matches:
        
            min_prob        lowest probability among all matches
            max_prob        highest probability among all matches
            med_prob        the median of the probabilities among all matches
            best_fit_prob   max probability among matches with the 
                            longest signature
                            
        :param audio: audio of a single call
        :type audio: {np.array | pd.Series}
        :param spectroid_template: one or more SpectralTemplate
            instance(s) against which to compare the clip
        :type spectroid_template: {SpectralTemplate | [SpectralTemplate]}
        :param slide_width_time: time in (usually fractional) seconds by
            which to slide a signature across the audio
        :type slide_width_time: float
        :param aggregation: method for computing the signature
            from frequencies of a single time frame. See SpectralAggregation
            enum. Default is set in DEFAULT_SPECTRAL_AGGREGATION.
        :type aggregation: SpectralAggregation
        :return all probabilities, and a summary of results
        :rtype (pd.DataFrame, pd.Series)
        '''

        if aggregation is None:
            aggregation = DEFAULT_SPECTRAL_AGGREGATION
        
        if slide_width_time is None:
            slide_width_time = SignalAnalyzer.SIGNATURE_MATCH_SLIDE_TIME 

        if type(spectroid_template) != list:
            spectroid_template = [spectroid_template]

        # An ID number for each signature unique across
        # the templates; simply a running number. Used
        # only for aggregation (group-by):
        sig_id = 0 
        res_df = pd.DataFrame()

        # Match clip sig against the sigs within
        # each template, collecting the probs from
        # each template match into matching_probs:
        for template in spectroid_template:

            # Match against all of the template's sigs:
            for sig in template.signatures:
                sig_id += 1
                # How many samples underly the signature?
                sig_width_samples = sig.end_idx - sig.start_idx
                # Number of samples (as opposed to time) to slide
                # sig to the right between each measurement:
                num_sample_slides = int(slide_width_time * cls.sr)
                # Create subclips to match the present signature:
                for start_idx in np.arange(0, len(audio), num_sample_slides):
                    # Take snippet of same width each time:
                    end_idx = start_idx + sig_width_samples
                    try:
                        aud_snip = audio[start_idx:end_idx]
                        # Do we have a full signature width of audio
                        # in this snippet?
                        if len(aud_snip) < sig_width_samples:
                            # Pad aud with its mean:
                            missing_width = sig_width_samples - len(aud_snip)
                            aud_snip = np.append(aud_snip, 
                                                 [aud_snip.mean()]*missing_width)
                        
                        # Get the snippet's signature:
                        clip_sig = SignalAnalyzer.spectral_centroid_each_timeframe(
                            aud_snip,
                            aggregation=aggregation,
                            bandpass=sig.bandpass_filter
                            )
                        # Probability for one subclip on one signature:
                        matching_prob = cls._compute_prob(clip_sig, sig)
                        res_df = res_df.append({
                            'start' : start_idx,
                            'stop'  : end_idx,
                            'n_samples' : sig_width_samples,
                            'probability' : matching_prob,
                            'sig_id': sig_id
                            }, 
                            ignore_index=True)
                    except IndexError:
                        # No more subclips to match against current sig.
                        # Process next signature, and then
                        # next template:
                        break

            # Matched against all sigs of one template.
        # Matched against all templates

        # Create the summary:
        min_prob = res_df['probability'].min()
        max_prob = res_df['probability'].max()
        med_prob = res_df['probability'].median()
        # Best probability for the longest signature
        longest_sig = res_df['n_samples'].max()
        best_fit_prob = res_df[res_df['n_samples'] == longest_sig].probability.max()
        
        res_summary = pd.Series({'min_prob' : min_prob,
                                 'max_prob' : max_prob,
                                 'med_prob' : med_prob,
                                 'best_fit_prob' : best_fit_prob
                                 })
        return (res_df, res_summary)

    #------------------------------------
    # _compute_prob
    #-------------------
    
    #********** Turn plot_sigs to False:
    @classmethod
    def _compute_prob(cls, clip_sig, template_sig, plot_sigs=False):
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
        elif type(clip_sig) == Signature:
            clip_sig_np = clip_sig.sig.to_numpy()
        else:
            clip_sig_np = clip_sig
            
        # Turn template_sig into np, and cut away
        # the leading TrueLen, if it is there:
        
        if type(template_sig) == Signature:
            template_sig_np = template_sig.sig.to_numpy()
        elif type(template_sig) == pd.Series:
            if template_sig.index[0] == 'TrueLen':
                template_sig_np = template_sig.to_numpy()[1:]
            else:
                template_sig_np = template_sig.to_numpy()
        elif type(template_sig) == np.ndarray:
            template_sig_np = template_sig
        else: 
            raise TypeError(f"template_sig must be a pd.Series, or np.ndarray, not {type(template_sig)}")
            
        # Shorten the clip signature to match the
        # length of the template signature with which
        # to compare. It is the caller's responsibility
        # to ensure that the clip is at least as long
        # as the signature:
        
        clip_sig_np = clip_sig_np[:len(template_sig_np)]

        # Use dynamic time warping to find the cheapest
        # way of making the clip sig similar to the template sig:
        cost_matrix, _warp_pairs = librosa.sequence.dtw(clip_sig_np, 
                                                        template_sig_np)

        # Normalize values in the cost matrix to 0-1:
        # For the case that the clip signature is identical to 
        # the template signature, the cost matrix will be zero.
        # This will occur when using the same clip with which 
        # signatures were created. In that case, probability
        # should be 1:
        if (cost_matrix == 0).all():
            prob = 1.0
        else:
            cost_normalized = (cost_matrix-cost_matrix.min())/(cost_matrix.max()-cost_matrix.min())
            # The value on the lower right of the cost 
            # matrix is the sum of all steps' costs:
            total_cost = cost_normalized[-1,-1]
            prob = 1 - total_cost
        
        if plot_sigs:
            templ_sig_series = pd.Series(template_sig_np,
                                         index=np.arange(len(clip_sig_np)),
                                         name='TemplateSig')
            clip_series = pd.Series(clip_sig_np,
                                    index=np.arange(len(clip_sig_np)),
                                    name='ClipSig')
            
            df = pd.DataFrame({'TemplateSig' : templ_sig_series, 'ClipSig' : clip_series})
            Charter.linechart(df, title="Template and Clip sigs")

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
        a named tuple that includes:

            o clip: np array of audio samples,
            o start_idx: index into the recording sample array
                where the call starts
            o end_idx: index into the recording sample array
                where the call ends
            o fname: path to the recording
            o sr: clip sample rate
            o species: species that vocalized the call
            o freq_interval: {'low_val': llll, 'high_val': hhhh, 'step': 0.000x}
                
        :param sel_tbl_path: path to selection table
        :type sel_tbl_path: src
        :param recording_path: path to recording associated
            with the selection table
        :type recording_path: str
        :param requested_species: individual or list of species to
            extract. If None, extract all species
        :type requested_species: {None | [str]}
        :return dict mapping species name to named tuples
            of clip related information
        :rtype [{str : namedTuple}]
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
            begin_time_secs = sel_dict['Begin Time (s)']
            end_time_secs = sel_dict['End Time (s)']
            freq_interval = sel_dict['freq_interval']
            clip = SoundProcessor.extract_clip(audio, 
                                               begin_time_secs,
                                               end_time_secs
                                               )
            clip_info = {'clip'      : clip,
                         'start_idx' : librosa.time_to_samples(begin_time_secs, sr),
                         'end_idx'   : librosa.time_to_samples(end_time_secs, sr),
                         'sr'        : sr,
                         'fname'     : recording_path,
                         'species'   : species,
                         'freq_interval' : freq_interval
                         } 
            try:
                audio_result_dict[species].append(clip_info)
            except KeyError:
                # First one:
                audio_result_dict[species] = [clip_info]
        return audio_result_dict, sr


    #------------------------------------
    # _freq_ticks
    #-------------------

    @classmethod
    def _freq_ticks(cls, num_rows, sr=22050):
        '''
        Get the frequencies for FFT bins, given
        the number of frequency bands, and the sample rate.
        
        Code lifted from librosa.
    
        :param num_rows: number of frequency bins in spectrogram
        :type num_rows: int
        :param sr: sample rate
        :type sr: int
        :return: array of index labels
        :rtype: np_array(float)
        '''
        n_fft = 2 * (num_rows - 1)
        # The following code centers the FFT bins at their frequencies
        # and clips to the non-negative frequency range [0, nyquist]
        basis = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)
        fmax = basis[-1]
        basis -= 0.5 * (basis[1] - basis[0])
        basis = np.append(np.maximum(0, basis), [fmax])
        return basis


# -------------------- Class SpectralTemplate -----------

class SpectralTemplate:
    '''
    Hold the spectral centroid timelines (signatures)
    of all calls in one recording.
    
    Usage:
        o <inst>.signatures    : list of Signature instances with the 
                                 frequency-of-max-energy timeline
        o iter(<inst>)         : return iterator over signatures
        o <inst>[n]            : return the nth signature
        o len(<inst>)          : number of signatures
        o <inst>.sig_lengths   : list of lengths of the signatures
        o <inst>.mean_sig      : signature that is the mean of all
                                 signatures' frequencies at each
                                 time.
        o <inst>.as_time(<signature>): sigs are pd.Series whose index are times
                                 as fractional secs. This method returns an array of 
                                 those times.
                                
    Also available as (read-only) properties:
        o sample_rate          : sample rate
        o hop_length           : samples between frames
        o n_fft                : related to number of frequency bands 
                                
    '''
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, 
                 signatures, 
                 rec_fname=None,
                 sr=SignalAnalyzer.sr,
                 hop_length=SignalAnalyzer.hop_length,
                 n_fft=SignalAnalyzer.n_fft
                 ):
        '''
        
        :param signatures:
        :type signatures:
        :param rec_fname:
        :type rec_fname:
        :param sr:
        :type sr:
        :param hop_length:
        :type hop_length:
        :param n_fft:
        :type n_fft:
        '''

        # Save the hop length and frequency bin-count-relevant
        # parameters so that clients wanting to create signatures
        # for audio clips to match against this template will
        # create proper sigs:
        
        self._hop_length = hop_length
        self._n_fft      = n_fft
        
        self.signatures = signatures
        self.rec_fname = rec_fname
        self.sr = sr

        # Computed lazily:
        self.cached_mean_sig = None
        
        # About application of bandpass filtering
        # of test audio when matching its snippets to
        # any of this template's sigs: if any of the
        # sigs were created with a species-specific bandpass
        # filter applied, then create an Interval that
        # comprises the max bandwidth from among this template's
        # constitutent sigs:
        
        sigs_with_bandpass = sum([1 if sig.bandpass_filter is not None else 0
                                  for sig
                                  in self.signatures
                                  ])
        if sigs_with_bandpass > 0:
            self.bandpass_filter = self.frequency_band()
        else:
            self.bandpass_filter = None

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
    # frequency_band
    #-------------------
    
    def frequency_band(self):
        '''
        Finds the lower and higher frequency
        bounds of each signature, and returns
        an Interval such that 'low_val' is the lowest
        frequency, and 'high_val' is the highest
        frequency from among all signature's bandwidths  
        
        :returns the lowest frequency and the highest 
            frequency from among all signatures in the template
        :rtype Interval
        '''
        
        lowest_freq = min([sig.freq_interval['low_val']
                           for sig
                           in self.signatures
                           ])
        high_freq   = max([sig.freq_interval['high_val']
                           for sig
                           in self.signatures
                           ])
        
        return Interval(lowest_freq, high_freq + 1)

    #------------------------------------
    # as_time
    #-------------------
    
    def as_time(self, signature):
        '''
        For each element of the signature, return
        the wallclock time. The last entry corresponds
        to the duration of the clip from which the signature
        was extracted. 
        
        :param signature: signature for which to 
            return times
        :type signature: pd.Series
        :return array of fractional seconds
        :rtype [float]
        '''
        
        return librosa.frames_to_time(np.arange(len(signature)),
                                      hop_length=self.hop_length)


    #------------------------------------
    # duration
    #-------------------
    
    def duration(self, signature):
        '''
        Return the duration of the signature in
        fractional seconds.
        
        :param signature: signature to analyze
        :type signature: pd.Series
        :return time duration
        :rtype float
        '''
        
        duration = librosa.frames_to_samples(
            len(signature), hop_length=self.hop_length) / self.sample_rate
        return duration


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
    # hop_length
    #-------------------
    
    @property
    def hop_length(self):
        return self._hop_length

    #------------------------------------
    # n_fft
    #-------------------
    
    @property
    def n_fft(self):
        return self._n_fft

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
    

class Signature:
    '''
    Instances hold the spectral signature of a
    single bird call. Information includes the
    following properties:
    
          o sig: a pd.Series.
            Its values are the successive frequencies over
            time. The index are the fractional seconds
            into the call for each value.
            
                        Frequency
                 Time
                0.01456   2402
                0.02912   3725
                      ...
          o fname: path to the recording that contains the
            bird call from which the signature was determined.
          o start_idx: index into the recording at which the
            call started
          o end_idx: index into the recording at which the
            call ended
          o sr: sample rate of the recording extract from which
            the call was taken. By default this is 22050, the librosa
            default to which the librosa.load() function resamples
            
    '''

    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, 
                 spectral_centroid, 
                 sr=22050, 
                 start_idx=0, 
                 end_idx=None, 
                 species=None, 
                 fname=None,
                 sig_id='na',
                 audio=None,
                 freq_interval=None,
                 bandpass_filter=None
                 ):
        '''
        The spectral_centroid is the actual signature: the
        frequencies at which energy was the median of all 
        energies at one frame in time. Maybe a pd.Series,
        or an np.ndarray. If a Series, each index value is 
        expected to be a time into the call relative to the
        beginning of the call (i.e. not relative to the start
        of the recording from which the call was lifted.   
        
        The start_idx/end_idx are the indices to the clip samples
        in the full recording from which the clip is lifted.
        The fname is the path to a file; format up to the caller;
        may or may not include hostname.
        
        :param spectral_centroid: frequencies for which 
            energy was median in their time frame
        :type spectral_centroid: {np.ndarray | pd.Series}
        :param sr: sample rate
        :type sr: int
        :param start_idx: index into the full recording's samples
            array where the call begins
        :type start_idx: int
        :param end_idx: index into the full recording's samples
            array where the call ends
        :type end_idx: int
        :param species: species that voiced the call
        :type species: str
        :param fname: path to full recording
        :type fname: str
        :param sig_id: optionally some identifier of this
            signature that is meaningful to the caller
        :type sig_id: Any
        :param audio: optionally, the audio from which the 
            signature was computed
        :type np.ndarray
        :param freq_interval: dict providing lowest, and highest 
            frequency of the signature, and the frequency steps
            with which the signature's spectrogram was created.
        :type freq_interval: {str : float}
        :param bandpass_filter: if not None: an interval that defines
            the bandpass filter that was applied before computing
            this signature.
        type bandpass_filter: {None | Interval}
        '''

        if type(spectral_centroid) == pd.Series:
            self.sig = spectral_centroid
        else:
            raise TypeError(f"The spectral_centroid must be a pd.Series, not {type(spectral_centroid)}")
        rec_fname = Path(fname).name
        self.sig.name = f"{rec_fname}_call_{sig_id}"
        self.sr = sr
        self.fname = fname
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.species = species
        self.sig_id  = sig_id
        self.audio = audio
        self.freq_interval = freq_interval
        if freq_interval is not None:
            self.freq_span = freq_interval['high_val'] - freq_interval['low_val']

        self.bandpass_filter = bandpass_filter
    #------------------------------------
    # as_walltime
    #-------------------
    
    def as_walltime(self):
        '''
        Returns a copy of the sig series with
        the index relative to the start of the 
        entire recording from which the call clip
        was taken.
        
        :return: copy of signature Series with new index
        :rtype: pd.Series
        '''
        
        sig_times = self.sig.index
        new_index = sig_times + librosa.samples_to_time(self.start_idx, self.sr)
        sig_copy  = self.sig.copy()
        sig_copy.index = new_index
        return sig_copy
    
    #------------------------------------
    # as_frames
    #-------------------
    
    def as_frames(self):
        '''
        Returns a copy of the sig series with
        the index replaced with the enumeration
        of spectral frames for each value (i.e. 1..len(sig)).
        
        :return: copy of signature Series with new index
        :rtype: pd.Series
        '''
        
        sig_copy  = self.sig.copy()
        sig_copy.index = np.arange(len(self.sig))
        return sig_copy

    #------------------------------------
    # plot
    #-------------------
    
    def plot(self, ax=None):
        '''
        Plot the signature as a line. The x axis 
        will be labeled with time into signature.
        
        The y axis will be true frequency, if the signature's
        frequency range was passed in during instance 
        creation. Else the matplotlib default y-axis labels
        are used.
         
        :param ax: optional existing Axes instance to use
        :type ax: plt.Axes
        '''
        
        if ax is None:
            self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        else:
            self.ax = ax
        self.ax.plot(self.sig)
        if self.freq_span is not None:
            self.ax.set_yticklabels(np.arange(int(self.freq_span))*1000)

    #------------------------------------
    # __len__
    #-------------------
    
    def __len__(self):
        return len(self.sig)
    
    #------------------------------------
    # __iter__
    #-------------------
    
    def __iter__(self):
        return iter(self.sig)
    
    #------------------------------------
    # __getitem__
    #-------------------
    
    def __getitem__(self, key):
        
        if type(key) == float:
            # Key is assumed to be a time in fractional seconds:
            return self.sig.loc[key] 
        elif type(key) == int:
            return self.sig[key]
        else:
            raise TypeError(f"Only floats and ints can index into sigs, not {key}")

    #------------------------------------
    # index
    #-------------------
    
    @property
    def index(self):
        '''
        Allow Signature instance to be used like
        a pd.Series by adding this index property.
        '''
        return self.sig.index

    #------------------------------------
    # name
    #-------------------
    
    @property
    def name(self):
        '''
        Allow Signature instance to be used like
        a pd.Series by adding this name property.
        '''
        return self.sig.name
