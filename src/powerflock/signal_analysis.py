'''
Created on Oct 1, 2021

@author: paepcke
'''
from collections import namedtuple
from enum import Enum
import multiprocessing as mp
import os
import warnings

import librosa
from logging_service.logging_service import LoggingService
from scipy.signal import argrelextrema

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils, Interval
import matplotlib.pyplot as plt
#from multitaper.multitaper_spectrogram_python import MultitaperSpectrogrammer
import numpy as np
import pandas as pd
from powerflock.signatures import Signature
from result_analysis.charting import Charter

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
    #    SIGNATURE_MATCH_SLIDE_FRACTION == 0.1 sec
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
    
    SIGNATURE_MATCH_SLIDE_FRACTION = 0.1
    '''Fraction of signature length to slide test clips before each match check'''
    
    log = LoggingService()

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
        NO LONGER USED.
        
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
                          spec_src=None,
                          bandpass=None,
                          wiener_entropy=False,
                          is_power=False
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
        
        Important: for a simple-magnitude spectro such as returned
           by raven_spectrogram(), set is_power to False so that the
           spectro is turned to power by squaring the values. If the
           spectro is a multitaper spectro, leave the default, and the
           values will be used as is.
        
        :param spec_src: pectrogram computed as with raven_spectrogram(),
            or path to audio file
        :type spec_src: {str | pd.DataFrame}
        :param bandpass: if provided, the specification of a bandpass filter
            to apply prior to computation
        :type bandpass: {None | Interval}
        :param wiener_entropy: if True, convert values from spectral
            flatness to Wiener entropy. The flatness values [0,1] are thereby
            converted to [-inf, 0], i.e. taking the log of the flatness values
        :type wiener_entropy: bool
        :is_power: True if the passed-in spectrogram values are already squared
            to represent power, rather than energy.
        :is_power: bool
        :return series of values [0,1] corresponding to the spectral flatness
            at each time frame.
        :rtype pd.Series
        
        '''
        if type(spec_src) == str:
            if not os.path.exists(spec_src):
                raise FileNotFoundError(f"Audio file {spec_src} does not exist.")
            spec_df = cls.raven_spectrogram(spec_src, extra_granularity=True)
        elif type(spec_src) != pd.DataFrame:
            raise TypeError(f"Arg must be file path or spectrogram, not {type(spec_src)}")
        else:
            spec_df = spec_src
        
        # Do have a passed-in spectro; square the values if needed:
        if not is_power:
            spec_df = spec_df ** 2
        
        if bandpass is not None:
            cls.apply_bandpass(bandpass, spec_df, inplace=True)
        
        # The 'power=1' tells librosa that we already giving it 
        # a power spectrum, on which flatness is usually shown:
        
        flatness_arr = librosa.feature.spectral_flatness(S=spec_df.to_numpy(), power=1)
        # The result is an np arr of the form:
        #   [[0.4,0.6,...]]
        # with length equal to the number of time frames.
        # The [-1] here pulls out the numbers we want:
        flatness_ser = pd.Series(flatness_arr[-1], index=spec_df.columns, name="SigSpectralFlatness")
        
        if wiener_entropy:
            # Take the log10 of each element, taking
            # care of 0 values by setting those to -inf:  
            flatness_ser = flatness_ser.apply(lambda el: -np.inf if el == 0 else np.log10(el))

        flatness_ser.name = 'flatness'
        return flatness_ser

    #------------------------------------
    # spectral_continuity
    #-------------------

    @classmethod
    def spectral_continuity(cls, 
                            spec_df=None,
                            audio=None,
                            bandpass=None,
                            continuity_time_thres=50,
                            is_power=False,
                            plot_contours=False
                            ):
        '''
        Given a spectrogram DataFrame as might be created from
        raven_spectrogram(), or an audio file. If given audio, create
        the spectrogram. Compute a single number for each time frame.
        The number represents whether or not the timeframe is part of
        an energy contour of time duration continuity_time_thres milliseconds
        or more. 
        
        Procedure:
           - Apply a filter that isolates the 'significant' energies
               in each time frame from noise above and below. The filter
               uses edge_mag_thres to determine how high energy in a particular
               cell of a time frame needs to be in order to be considered above
               noise. In order to adjust the threshold to the context of the 
               particular spectrogram, the given number is multiplied by:
               
                                abs(Wiener_entropy)
                       ----------------------------------------
                            abs(energy_t,f - mean(energy_t))
                            
               i.e. the ratio of overall flatness to how much the 
               energy at a given cell rises above the mean energy in 
               the time frame.
          
               The result is a boolean dataframe the same shape as spec_df with
               True in each cell that is part of an energy contour at the cell's
               time frame. This df can be plotted to yield a contour map.
               
            - Next, contours shorter than continuity_time_thres milliseconds 
              are eliminated. 
              
            - Finally, a single 'spectral continuity' number is derived for each
              time frame by computing the percentage of cells in the time frame
              that are part of a remaining ridge.

        Two values are returned: the intermediate boolean contour dataframe, and
        a Series of spectral continuity values, one value for each timeframe.
        
        Each element (i.e. information about one timeframe) in the returned Series 
        is the percentage of frequencies at the time what participate in a ridge. 

        :param spec_df:
        :type spec_df:
        :param audio:
        :type audio:
        :param bandpass:
        :type bandpass:
        :param continuity_time_thres:
        :type continuity_time_thres:
        :is_power: True if the passed-in spectrogram values are already squared
            to represent power, rather than energy.
        :is_power: bool
        :param plot_contours: whether or not to plot contour maps of
            the spectrum along the way
        :type plot_contours: bool
        :returns: a boolean df with True in cells that participate in a contour,
            and a Series with one element for each time frame
        :rtype: (pd.DataFrame, pd.Series)
        '''
        
        if spec_df is None:
            #spec_df = cls.raven_spectrogram(audio, to_db=False)
            spec_df = cls.raven_spectrogram(audio, extra_granularity=True)

        # Do have a passed-in spectro; square the values if needed:
        if not is_power:
            spec_df = spec_df ** 2
        
        if bandpass is not None:
            spec_df = cls.apply_bandpass(bandpass, spec_df)
            # Bandpass filtering sets out-of-band cells to -np.inf.
            # But the flatness detection methods below to not
            # tolerate negative values; so replace the -np.inf 
            # with zeroes: 
            spec_df.replace(-np.inf, 0, inplace=True)

        num_rows, _num_cols = spec_df.shape

        # For each timeframe get the median distance
        # of the power from the mean within that timeframe:
        med_dist_from_mean = (spec_df - spec_df.mean(axis='rows')).median().abs()
        # Contours are moments of power larger than
        # the median distance from the mean at each timeframe: 
        contour_df = spec_df >= spec_df.mean(axis='rows') + med_dist_from_mean

        if plot_contours:
            fig = plt.figure()
            axes_grid_spec = fig.add_gridspec(nrows=2, ncols=1)
            contour_ax = fig.add_subplot(axes_grid_spec[0,0])
            filtered_contour_ax = fig.add_subplot(axes_grid_spec[1, 0])
            contour_ax = Charter.draw_contours(contour_df,
                                               ax=contour_ax,
                                               title='CMTO Contours',
                                               xlabel='Time',
                                               ylabel='Frequency',
                                               decimals_x=2,
                                               decimals_y=2,
                                               fewer_labels_x=10,
                                               fewer_labels_y=12
                                               )
            fig.show()
            input(f"Contours for spectrogram; press ENTER to continue: ")
        
        # Obtain new df with spurious contours removed:
        # keep only the contours longer than continuity_time_thres
        # milliseconds:
        long_contours_df = cls.contour_length_filter(contour_df, continuity_time_thres)
        
        if plot_contours:
            _ax = Charter.draw_contours(long_contours_df,
                                        ax=filtered_contour_ax,
                                        title='CMTO Contours',
                                        xlabel='Time',
                                        ylabel='Frequency',
                                        decimals_x=2,
                                        decimals_y=2,
                                        fewer_labels_x=10,
                                        fewer_labels_y=12
                                        )
            fig.tight_layout()
            input(f"Contours filtered for minimum len; press ENTER to continue: ")
            
        # Finally, compute for each timeframe the percentage
        # of frequencies that participate in a contour:
        continuity = 100 * long_contours_df.sum(axis=0) / num_rows
        
        continuity.name = 'continuity'
        return long_contours_df, continuity 

    #------------------------------------
    # contour_mask
    #-------------------
    
    @classmethod
    def contour_mask(cls, spectro_src):
        '''
        Given the path to an audio file, or a spectrogram,
        create the spectrogram form the audio if needed. The
        index of the spectro are frequencies, the columns are
        (possibly fractional) times.
        
        Return a new boolean df with same shape/index/columns as
        the spectro. In each column the True values mark the 
        frequencies of overtones during the column's timeframe.
        
                              1.0     2.0 
                       40     True    True
                       30     False   True
                       20     True    True
                       10     False   True
                       
        At time 1.0sec the fundamental is 20Hz, with an overtone
        at 40Hz. At time 2.0sec, the fundamental is 10Hz, with 
        overtones every 10Hz above.
                       
        :param spectro_src: audio file path or spectrogram
        :type spectro_src: {str | pd.DataFrame}
        :return boolean df marking fundamental and overtones
        :rtypr pd.DataFrame(bool)
        '''
        
        if type(spectro_src) == str:
            if not os.path.exists(spectro_src):
                raise FileNotFoundError(f"Audio file {spectro_src} does not exist.")
            spec_df = cls.raven_spectrogram(spectro_src, extra_granularity=True)
        elif type(spectro_src) != pd.DataFrame:
            raise TypeError(f"Arg must be file path or spectrogram, not {type(spectro_src)}")
        else:
            spec_df = spectro_src
        
        # Func that takes a pd.Series. It computes its discrete gradient
        # at each element (i.e. the 1st derivative), then finds the zero
        # crossings. Returns a pd.Series of sample length/index/name
        # with True where a zero crossing occurs, and False elswhere.
        # Used in apply() below:
        def column_contour_mask(ser):
            grad = np.gradient(ser.values)
            zxing_idxs = np.where(np.diff(np.sign(grad)))[0]
            # Initially: all False 
            mask = pd.Series([False]*len(ser), index=ser.index, name=ser.name)
            mask.iloc[zxing_idxs] = True
            return mask
        
        # Apply the column_contour_mask func to 
        # each column of the spectrogram. The frequencies 
        # associated with True values are overtones (frequencies
        # are the values of the index):
        spec_contours = spec_df.apply(column_contour_mask, axis='rows')

        return spec_contours
    
    #------------------------------------
    # harmonic_pitch
    #-------------------
    
    @classmethod
    def harmonic_pitch(cls, spec_src):
        '''
        Given the path to an audio file, or a 
        spectrogram, generates a spectrogram if needed.
        The index are frequencies, the columns are times.
        
        Obtains a boolean mask with index/columns same as
        the spectro. True values in each column mark
        overtone frequencies.
        
        Returns the median difference between all overtones,
        
        Given a df of energy contours across time for
        each of a spectral analysis' frequencies, return
        a Series with pitch at each spectral timeframe. 
        
        :param spec_source: audio file or spectrogram.
            Note: if spectrogram, must be fine grained in 
            frequency, such as is returned by raven_spectrogram()
            with extra_granularity set to True
        :type spec_source: {str | pd.DataFrame}
        :return the harmonic pitch of the overall sound
        :rtype float
        '''
        
        if type(spec_src) == str:
            if not os.path.exists(spec_src):
                raise FileNotFoundError(f"Audio file {spec_src} does not exist.")
            spec_df = cls.raven_spectrogram(spec_src, extra_granularity=True)
        elif type(spec_src) != pd.DataFrame:
            raise TypeError(f"Arg must be file path or spectrogram, not {type(spec_src)}")
        else:
            spec_df = spec_src
        
        # Get a boolean df with True in each column for 
        # the frequencies that are overtones at that column's
        # timeframe:
        contour_mask = cls.contour_mask(spec_src)
        
        # Set spectral magnitudes that are not part of
        # a contour to negative infinity:
        masked_spec_df = spec_df.where(~contour_mask, -np.inf)
        # Discover the frequencies of local magnitude maxima.
        # Done by getting the index values of the local maxima one
        # column at a time. Order is number of neighbor points
        # to consider for maxima discovery:
        extrema_idxs_row_wise, extrema_idxs_col_wise = \
            argrelextrema(masked_spec_df.values, np.greater, order=3)

        # If we were to pair the extrema indices for rows and cols,
        # we would get:
        #     list(zip(extrema_idxs_row_wise, extrema_idxs_col_wise))
        #   --> [(6, 258),    # (index-of-row, index-of-col)
        #        (11, 257), 
        #        (11, 258),
        #           ... for 
        # Thus: frequency at masked_spec_df.index[6] has a peak
        # at timeframe number 258. Same for frequency at masked_spec_df.index[11]
        # NOTE: argrelextrema excludes results for columns of all equal values.
        # So the extrema_idxs_col_wise may be missing some cols.
        # we fill those in later:
        
        # Organize the above into a 2-col df:
        #       timeframe_idx, frequency 
        
        peak_freq_df = pd.DataFrame({'timeframe_idx' : extrema_idxs_col_wise,
                                     'freq' : spec_df.index[extrema_idxs_row_wise]}
                                     )

        # Get peak frequencies within each timeframe (i.e. column):
        timeframe_groups = peak_freq_df.groupby('timeframe_idx')
        
        # Within each timeframe (a.k.a column, a.k.a. group),
        # compute the median difference between frequencies
        # with peaks. The division by 2 is because we can only
        # observe the odd harmonics, since the even ones cancel out: 
        freq_diffs = timeframe_groups.freq.apply(np.diff)
        # Remove empty lists created by np.diff at the edges
        # of dataframes and series:
        freq_diffs = freq_diffs[freq_diffs.apply(len) > 0]
        harm_pitch = np.abs(freq_diffs.apply(np.median)) / 2.
        
        # harm_pitch may be missing pitches for some
        # timeframes (columns), because they only had 
        # the same values (e.g. all 0). Give those a 
        # median frequency of 0. First, discover the missing
        # cols:
        # set of col numbers that *should* be there:
        spec_cols_idx_set = set(np.arange(0,len(spec_df.columns)))
        # set of col numbers that are in fact there:
        freq_diffs_idxs_set = set(freq_diffs.index)
        # the missing ones:
        missing_col_idxs = spec_cols_idx_set - freq_diffs_idxs_set
        harm_pitch = harm_pitch.combine(pd.Series([0.0]*len(missing_col_idxs), 
                                                  index=missing_col_idxs,
                                                  dtype=float
                                                  ),
                                        max,
                                        fill_value=0.0)
        
        # The diff() operation above naturally introduces
        # nan values at the edges of the df to which it is
        # applied. Set those to 0.0 as well:
        
        harm_pitch.fillna(0.0, inplace=True)
        harm_pitch.name = 'pitch'
        # Set the index to the timeframe times in secs:
        harm_pitch.index = spec_df.columns
        return harm_pitch 

    #------------------------------------
    # freq_modulations
    #-------------------
    
    @classmethod
    def freq_modulations(cls, spec_src):
        '''
        Given a spectrogram, take the directional
        derivatives in the frequency and time directions
        at each point. Use trigonometry to compute the 
        true direction angle of the derivative for each point.
        Return a pd Series with the median of the absolute
        value of the angles at each timeframe (column). 

        :param spec_src: audio file or spectrogram.
            Note: if spectrogram, must be fine grained in 
            frequency, such as is returned by raven_spectrogram()
            with extra_granularity set to True
        :type spec_src: {str | pd.DataFrame}
        :return the median angle of contour normals at each timeframe
        :rtype pd.Series
        '''
        if type(spec_src) == str:
            if not os.path.exists(spec_src):
                raise FileNotFoundError(f"Audio file {spec_src} does not exist.")
            spec_df = cls.raven_spectrogram(spec_src, extra_granularity=True)
        elif type(spec_src) != pd.DataFrame:
            raise TypeError(f"Arg must be file path or spectrogram, not {type(spec_src)}")
        else:
            spec_df = spec_src

        # Compute gradients in both freq and time directions:
        gradf, gradt = np.gradient(spec_df)
        
        # At each point in the spectro, create two 
        # vectors:
        #
        #               /|
        #               /
        #              /  v1=(gradt, gradf)
        #             /
        #            /------>
        #               v0=(gradt, 0)

        zero_matrix = np.zeros(gradf.shape)
        # For v0:
        # Get pairs (gradt, 0) as np arrays, arriving
        # at an array shaped (num_freqs, num_timeframes, 2).
        # same for v1:
        
        v0 = np.stack([gradt, zero_matrix], axis=-1)
        v1 = np.stack([gradt, gradf], axis=-1)

        num_freqs, num_timeframes, _vvec_width = v0.shape
        # For ease of broadcasting the following trigonometry
        # for the direction angles, make two long arrays, one
        # holding all v0 coords, one the v1 coords:
        # 
        v0_vecs = v0.reshape(num_freqs*num_timeframes, 2)
        v1_vecs = v1.reshape(num_freqs*num_timeframes, 2)
        
        # Pairwise compute the dot product of vectors v0, v1.
        # Since v0's second element is 0, the computation
        # turns from: 
        #     v0_vecs[:,0] * v1_vecs[:,0] +  v0_vecs[:,1] * v1_vecs[:,1])
        # to:
        #      v0_vecs[:,0] * v1_vecs[:,0]
        dot_prods = v0_vecs[:,0] * v1_vecs[:,0]
        
        # Similarly with the determinants of the vec pairs.
        # Determinant of:
        #    | a    b|
        #    | c    d|
        # is:
        #    ad-bc
        # or:
        #    v0_vecs[0] * v1_vecs[1] - v0_vecs[1] * v1_vecs[0]
        #
        # Since v0_vec[1] == 0, this reduces to:
        #
        #    determinants = ad, or v0_vecs[:,0] * v1_vecs[:,1] 
        
        determinants = v0_vecs[:,0] * v1_vecs[:,1]
        # Need to apply arctan to determinant/dot-product pairs.
        
        angles = np.degrees(np.arctan2(determinants, dot_prods))

        # Turn back into shape of original spectrogram:
        angles_right_shaped = angles.reshape(num_freqs, num_timeframes)
        angles_df = pd.DataFrame(angles_right_shaped,
                                 index=spec_df.index,
                                 columns=spec_df.columns
                                 )
        # Only consider the angles of real contours:
        # create a mask same shape as the angles_df,
        # with True where a contour exists:
        contour_mask = cls.contour_mask(spec_df)
        
        # Turn all angles that are off-contour into nan:
        angles_masked = angles_df.mask(~contour_mask)
        
        # For each timeframe, compute the median angle 
        # of all contours, ignoring the sign. So we do
        # lose the rising vs. falling info of the contours:
        # Some spec frames will have no contours at all,
        # generating one of two runtime warnings. Temporarily
        # suppress these:
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=RuntimeWarning, message='All-NaN slice encountered')
            warnings.filterwarnings(action='ignore', category=RuntimeWarning, message='Mean of empty slice')
        
            angle_medians = np.abs(angles_masked).median()

        angle_medians.name = 'freq_mod'
        return angle_medians

    #------------------------------------
    # Contour_length_filter
    #-------------------

    @classmethod
    def contour_length_filter(cls, contour_df, continuity_time_thres=50):
        '''
        Given a boolean df whose True cells indicate the presence of a 
        a horizontal spectrum contour along one frequency row, set contour
        cells to False if they form a contour shorter in time
        than continuity_time_thres milliseconds. Return a copy of contour_df
        with that modification.
        
        Used to eliminate spurious contours.
        
        Example:
         
        Start with contour_df being:
        
           True  False True True  False True
           False False True False True  True
              ...

        Say the continuity_time_thres were to correspond
        to two spectrum time frames (this number is computed
        in this method): num_contour_frames == 2:
        
        For each vertical slice of width 2, add the values in 
        each row of the slice to get from the first slice:
           
           1
           0
        
        In each row, set both columns to:
           col == num_contour_frames == 2
        
        Giving:
           False False True True  False True
           False False True False True  True
              ...
        
        The second slice yields:
        
          2
          1
        ==>
           False False True  True  False True
           False False False False True  True
        
        Finally, after the last slice:
           False False True  True  False False
           False False False False True  Treu
 
        :param contour_df: boolean array identifying contour ridges
        :type contour_df: pd.DataFrame([bool])
        :param continuity_time_thres: minimum number of milliseconds at
            which a contour is considered to be significant.
        :type continuity_time_thres: int
        :return new boolean dataframe with only the 'significantly-long' 
            contours retained
        :rtype: pd.DataFrame([bool])
        '''

        # Compute the number of spectrum time frames required
        # to consider an energy contour long enough (in milliseconds):
        timeframe_width = (contour_df.columns[1] - contour_df.columns[0]) * 1000
        num_rows, _num_cols = contour_df.shape
        num_contour_frames = int(round((continuity_time_thres / timeframe_width), 0))
        long_contours = contour_df.copy() 

        num_slides = int(len(long_contours.columns) / num_contour_frames)
        for i in np.arange(0, num_slides + 1, num_contour_frames): # The column vector of one slide:
            is_contour = long_contours.iloc[:, i:i + num_contour_frames].sum(axis=1) == num_contour_frames # Turn the column vector result into a df of
            # width num_contour_frames by replicating the col_vector
            # to the width of the slide, then broadcasting the
            # assignment down the slice:
            long_contours.iloc[:, i:i + num_contour_frames] = is_contour.to_numpy().repeat(num_contour_frames).reshape((num_rows, num_contour_frames))

        return long_contours

    #------------------------------------
    # apply_bandpass
    #-------------------
    
    @classmethod
    def apply_bandpass(cls, bandpass, spec_df, inplace=False, extract=False):
        '''
        Given a spectrogram and bandpass frequency interval, 
        such as Interval(5000, 8000, 1). If extract is False, 
        return a new spectrogram with frequencies above and below 
        the given interval set to -inf. In this case the shape
        of the returned structure equals that of spec_df. 
        
        If extract is True, create a new df with all non-qualifying
        rows removed.
        
        Inputs inplace and extract cannot both be True 
        
        :param bandpass: specification of frequencies to retain 
        :type bandpass: Interval
        :param spec_df: the spectrogram: index is frequencies, 
            columns are time frames.
        :type spec_df: pd.DataFrame
        :param inplace: whether or not to modify spec_df in memory;
            deafault: new df
        :type inplace: bool
        :param extract: if True, return a (potentially) smaller df
            with only the qualifying rows
        :type extract: pd.DataFrame
        :return filtered spectrogram of equal size as the original
        :rtype pd.DataFrame
        '''

        if not isinstance(bandpass, Interval):
            raise TypeError(f"Bandpass must be an Interval of floats or ints, not {bandpass}")
        
        if inplace and extract:
            raise ValueError("Parameters inplace and extract cannot both be True")
         
        # Remember that spectrograms are
        # arranged high frequency to low frequency
        # to match what visual spectros show. The
        # "- 1" keeps the low bound included
        low_freq, high_freq = bandpass['low_val'] - 1, bandpass['high_val'] 
        if inplace or extract:
            filtered_spec = spec_df
        else:
            filtered_spec = spec_df.copy()
            
        if extract:
            # Will always make a copy:
            filtered_spec = filtered_spec[(filtered_spec.index >= low_freq) & (filtered_spec.index < high_freq)]
        else: 
            # Reason for setting unwanted part of the freq spectrum
            # to -np.inf:
            # Values are in dB FS (i.e. relative to the maximum signal).
            # So the vals are all negative. When we will look
            # for, say, max values, we don't want that to be zero:
            
            filtered_spec[filtered_spec.index >= high_freq] = -np.inf
            filtered_spec[filtered_spec.index < low_freq] = -np.inf
        
        return filtered_spec

    #------------------------------------
    # spectral_measures_each_timeframe
    #-------------------

    @classmethod
    def spectral_measures_each_timeframe(cls, 
                                         spec_snip,
                                         sig
                                         ):
        '''
        Return a dataframe 
        
                    'flatness', 'continuity', 'pitch', 'freq_mod'
          t0
          t1                    ...
        
        The df will have as many rows as spec_snip
        has timeframes.
        
        :param spec_df: spectrogram, usually several
            frames of a larger spectrogram
        :type spec_df: pd.DataFrame
        :param sig: a Signature from which to take 
            information such as bandpass filtering
        :type sig: Signature
        :return: a Signature instance containing for 
            for each timeframe the measurements
            'flatness', 'continuity', 'pitch', 'freq_mod'
        :rtype Signature
        '''

        # Compute the measures into one series each.
        # Each series covers all timeframes:
        flatness = SignalAnalyzer.spectral_flatness(spec_snip, is_power=True)
        _long_contours_df, continuity = SignalAnalyzer.spectral_continuity(spec_snip,
                                                                           is_power=True,
                                                                           plot_contours=False
                                                                           ) 
        pitch = SignalAnalyzer.harmonic_pitch(spec_snip)
        freq_mod = SignalAnalyzer.freq_modulations(spec_snip)

        # Each measure makes a col:
        snip_results = pd.concat([flatness, continuity, pitch, freq_mod], axis=1)
        
        field_recording_sig = Signature(
            sig.species,
            snip_results,
            scale_info=sig.scale_info,
            start_idx=spec_snip.columns[0],
            end_idx=spec_snip.columns[-1],
            fname='field-recording',
            sig_id=f"field_against_sig{sig.sig_id}",
            freq_interval=Interval(np.min(spec_snip.index), np.max(spec_snip.index)),
            bandpass_filter=sig.bandpass_filter,
            extract=sig.extract
            )
        return field_recording_sig


    #------------------------------------
    # raven_spectrogram
    #-------------------
    
    @classmethod
    def raven_spectrogram(cls, audio, to_db=True, extra_granularity=False):
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
            
        If extra_granularity is True, the frequency granularity
        is doubled compared to the default raven spectrogram by 
        using n_fft==2048 instead of 512, and hop_length=256 instead
        of 512
         
        
        :param audio: either the path to an audio file,
            or the already loaded audio array
        :type audio: {str | np.array(float)
        :param to_db: whether to convert the energy values to db
            relative to the max. Raven does this conversion, so
            the default is True. But procedures such as spectral flatness
            computations need the absolute values.
        :type to_db: bool
        :param extra_granularity: whether or not to double 
            resolution of the spectrogram
        :type extra_granularity: bool
        :return: dataframe with a spectrogram that reflects
            what a Raven labeling software user would see
            in its spectrogram display.
        :rtype: pd.DataFrame
        '''
        sr = 22050
        if extra_granularity:
            hop_length = 256
        else:
            hop_length = 512
        if type(audio) == str:
            audio_arr, aud_sr = librosa.load(audio)
            # Adjust sr if necessary:
            if aud_sr != sr:
                audio_arr = librosa.resample(audio_arr, aud_sr, sr)  
        else:
            audio_arr = audio 

        if extra_granularity:
            spec = np.abs(librosa.stft(audio_arr, n_fft=2048, hop_length=hop_length, window='hann'))
        else:
            spec = np.abs(librosa.stft(audio_arr, n_fft=512, hop_length=hop_length, window='hann'))
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
    # raven_spectro_duration
    #-------------------
    
    @classmethod
    def raven_spectro_duration(cls, spec, extra_granularity=False):
        '''
        Compute time duration of a given spectrogram.
        The spectrogram is assumed to have been computed
        via STFT parameters used in raven_spectro_duration().
        
        The extra_granularity argument must match the one used
        when the spectrogram was created.
        
        :param spec: spectrogram to examine
        :type spec: pd.DataFrame
        :param extra_granularity: whether or not the time
            granularity was enhanced when the spectrogram
            was created
        :type extra_granularity: bool
        :return: time in fractional seconds
        :rtype: float
        '''
        
        if extra_granularity:
            hop_length = 256
        else:
            hop_length = 512
            
        _num_freqs, num_timeframes = spec.shape
        return num_timeframes * hop_length / 22050 


#   #------------------------------------
#   # multitaper_spectrogram
#   #-------------------
#
#   @classmethod
#   def multitaper_spectrogram(cls,
#                              audio_spec,
#                              sr=22050,
#                              bandpass=None,
#                              plot=False
#                              ):
#       '''
#       SOME DOUBT ABOUT CORRECTNESS!
#
#       Given an np.ndarray audio signal, or the path
#       to an audio file, return a spectrogram constructed
#       by the multitaper method. For usage example, see
#       A procedure for an automated measurement of song similarity
# By Ofer Tchernichovski, Fernando Nottebohm, CHing Elizabeth Ho, 
# Bijan Pesaran, Partha Pratim Mitra. ANIMAL BEHAVIOUR, 2000, 59, 1167–1176.
#
# Well explained in video tutorial #2 at https://prerau.bwh.harvard.edu/multitaper/
#
# This method fixes parameters to the underlying workhorse
# in file multitaper_spectrogram_python.py to work with birds. For
# info on that program, see git@github.com:preraulab/multitaper_toolbox.git.
#
#       :param audio_spec: path to audio, or audio array
#       :type audio_spec: {str | np.ndarray}
#       :param sr: sample rate. If audio is an audio path, the sr is
#           found automatically. Default is the librosa default sr value.
#       :type sr: int
#       :param bandpass: if not None, an Interval instance with low
#           and high freqs to consider
#       :type bandpass: {None | Interval}
#       :param plot: if True, plot the result spectrogram and pause
#       :type plot: bool
#       :return a spectrogram dataframe; index will be frequencies,
#           columns will be times for each spectrogram timeframe
#       :rtype: pd.DataFrame
#       '''
#
#       if type(audio_spec) == str:
#           audio, sr = SoundProcessor.load_audio(audio_spec)
#       elif type(audio_spec) == np.ndarray:
#           audio = audio_spec
#       else:
#           raise TypeError("Audio spec must be a file path or np array of audio")
#
#       # Number of audio samples in one timeframe:
#       timeframe_samples = 512
#
#       # Width of one timeframe in seconds
#       # What the multitaper software calls data_window
#       timeframe_secs = timeframe_samples/sr
#       # Max frequency that is discernible with given sr:
#       nyquist_freq = sr/2.0
#
#       if bandpass is not None:
#           freq_range = (bandpass['low_val'], bandpass['high_val'])
#       else:
#           freq_range = (0, nyquist_freq) 
#
#       top_freq = freq_range[1]
#
#       # Sometimes called TW: timeframe_width_secs * freq_band_width / 2:
#       time_bandwidth = (timeframe_secs * top_freq) / 2.0
#
#       # Multitaper software wants the following:
#       window_params = (timeframe_secs, timeframe_samples/sr)
#
#       # Computation will use multiple cores; be nice:
#       num_cpus = int(mp.cpu_count() * 80 / 100)
#
#       # The min_nfft determines the number of frequency bands,
#       # and therefore the resolution of contour analyis:
#       spectro, times, freqs = MultitaperSpectrogrammer.multitaper_spectrogram(audio, 
#                                                                               sr,
#                                                                               frequency_range=freq_range,
#                                                                               time_bandwidth=time_bandwidth, 
#                                                                               #num_tapers,
#                                                                               window_params=window_params, 
#                                                                               min_nfft=1024,
#                                                                               # detrend_opt, 
#                                                                               multiprocess=True, 
#                                                                               cpus=num_cpus, 
#                                                                               # weighting, 
#                                                                               plot_on=False, 
#                                                                               clim_scale=False
#                                                                               # verbose, 
#                                                                               # xyflip
#                                                                               )
#
#       spectro_df = pd.DataFrame(spectro, columns=times, index=freqs)
#       # Make the top row the highest freq, rather than 0:
#       spectro_df = spectro_df.reindex(index=spectro_df.index[::-1])
#
#       if plot:
#           Charter.spectrogram_plot(spectro_df, fig_title="Multitaper Spectrogram")
#           input("Hit Enter to dismiss and continue: ")
#       return spectro_df

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
    # match_probability
    #-------------------
    
    @classmethod
    def match_probability(cls, 
                          audio,
                          spectral_template,
                          slide_width_time_fraction=None,
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
        :param spectral_template: one or more SpectralTemplate
            instance(s) against which to compare the clip
        :type spectral_template: {SpectralTemplate | [SpectralTemplate]}
        :param slide_width_time_fraction: time in (usually fractional) seconds by
            which to slide a signature across the audio
        :type slide_width_time_fraction: float
        :param aggregation: method for computing the signature
            from frequencies of a single time frame. See SpectralAggregation
            enum. Default is set in DEFAULT_SPECTRAL_AGGREGATION.
        :type aggregation: SpectralAggregation
        :return: all probabilities
        :rtype: pd.DataFrame
        '''

        if slide_width_time_fraction is None:
            slide_width_time_fraction = SignalAnalyzer.SIGNATURE_MATCH_SLIDE_FRACTION 

        cls.log.info(f"Creating spectrogram")
        spec_df = SignalAnalyzer.raven_spectrogram(audio, extra_granularity=True)
        
        power_df = spec_df ** 2

        #sr = spectral_template[0].sr
        #if type(audio) != pd.Series:
        #    # Turn into a series:
        #    audio = pd.Series(audio, np.arange(0,len(audio)/sr,1/sr))

        # Match clip sig against the sigs within
        # each template, collecting the probs from
        # each template match into matching_probs:

        # Match against all of the template's sigs:
        num_cores = mp.cpu_count()
        # Use 80% of the cores:
        #num_workers = round(num_cores * 90  / 100)
        num_workers = num_cores
        match_results = []
        
        sigs = spectral_template.signatures
        # Number of samples to slide between each 
        # measurement is slide_width_time_fraction * <shortest-sig>
        # The max(..., 1) guards against slide window
        # going to 0. Least number of samples to slide
        # is 1:
        slide_width_samples = max(round(min([len(sig) for sig in sigs]) * slide_width_time_fraction), 1)
        
        with mp.Pool(processes=num_workers) as pool:
            for sig in spectral_template.signatures:
                match_results.append(
                    pool.apply_async(cls.process_one_sig,
                                     (power_df, sig, slide_width_samples))
                    )
            res_dfs = []
            for worker in match_results:
                res_dfs.append(worker.get())
                
        # Composite the result dfs from all workers
        # into a single df:
        res = pd.concat(res_dfs)
        return res

    #------------------------------------
    # process_one_sig 
    #-------------------
    
    @classmethod
    def process_one_sig(cls, power_df, sig, slide_width_samples):

        sig_id = sig.sig_id
        _num_freqs, num_frames = power_df.shape
        
        cls.log.info(f"Matching against {sig.species} sig-{sig_id}")
        passband = sig.bandpass_filter
        if passband is not None:
            cls.log.info(f"Applying bandpass [{passband['low_val']},{passband['high_val']}]")
            power_df_clipped = SignalAnalyzer.apply_bandpass(passband, 
                                                             power_df, 
                                                             extract=sig.extract)
        else:
            power_df_clipped = power_df
        
        # How many samples underly the signature?
        sig_width_samples = len(sig)
        
        # Create subclips that match the width of the present signature;
        # the subtraction of num_sample_slides ensures
        # that we don't slide beyond the spectrogram:
        last_idx = num_frames - sig_width_samples - 1
        percentage_reported = 0
        result_series_list = []
        for start_idx in np.arange(0, last_idx, slide_width_samples):
            # Take snippet of same width as the current sig each time:
            end_idx = start_idx + sig_width_samples
            # Time for sign of life? Do that about every 10% done: 
            perc_done = 100*start_idx/last_idx
            if perc_done > percentage_reported + 10:
                cls.log.info(f"... done {int(perc_done)}% of {sig.species} sig-{sig_id}")
                percentage_reported = perc_done

            try:
                spec_snip = power_df_clipped.iloc[:, start_idx:end_idx]
                # Get the snippet's signature:
                clip_sig = SignalAnalyzer.spectral_measures_each_timeframe(
                    spec_snip,
                    sig=sig
                    )
                # Probability for one subclip on one signature:
                sig_dist = clip_sig.match_probabilities(sig)
                result_series_list.append(pd.Series(
                    {
                    'start_idx'  : start_idx,
                    'stop_idx'   : end_idx,
                    'n_samples'  : sig_width_samples,
                    'match_prob' : sig_dist, 
                    'sig_id': sig_id
                    }))
            except IndexError:
                # No more subclips to match against current sig.
                # Process next signature
                break
        res_df = pd.DataFrame(result_series_list)
        cls.log.info(f"Done matching snippets against signature {sig.species} {sig_id}")

        return res_df

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
