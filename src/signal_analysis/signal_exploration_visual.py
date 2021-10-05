#!/usr/bin/env python3
'''
Created on Oct 2, 2021

@author: paepcke
'''

# TODO:
#    o legends are wrong now that we
#      make two figures.
#    o Next: correlation

import os

import librosa

import numpy as np
import torch
import pandas as pd
from result_analysis.charting import Charter
from signal_analysis import SignalAnalyzer


#from signal_analysis.signal_analysis import SignalAnalyzer
class SignalViz:
    '''
    classdocs
    '''

    #------------------------------------
    # init_class
    #-------------------

    @classmethod
    def __init_class__(cls):
        
        cls.color_explanation = {}
        cls.ax = None

        cls.cur_dir = os.path.dirname(__file__)
        cls.sound_data = os.path.join(cls.cur_dir, 'tests/signal_processing_sounds')
        cls.xc_sound_data = os.path.join(cls.cur_dir, 'tests/signal_processing_sounds/XenoCanto')
        
        # Field selection table and associated recording
        cls.sel_tbl_fld = os.path.join(cls.cur_dir, 'tests/selection_tables/JZ_DS_AM03_20190713_055956.Table.1.selections.txt')
        cls.sel_recording_fld = os.path.join(cls.sound_data, 'DS_AM03_20190713_055956.wav')
        
        # Xeno Canto CMTOG
        cls.sel_tbl_cmto_xc1 = os.path.join(cls.cur_dir, 'tests/selection_tables/XenoCanto/cmto1.selections.txt')
        cls.sel_rec_cmto_xc1 = os.path.join(cls.xc_sound_data, 'CMTOG/SONG_XC274155-429_Chestnut-mandibled_Toucan_2_song.mp3')
        
        cls.sel_tbl_cmto_xc2 = os.path.join(cls.cur_dir, 'tests/selection_tables/XenoCanto/cmto2.selections.txt')
        cls.sel_rec_cmto_xc2 = os.path.join(cls.xc_sound_data, 'CMTOG/SONG_XC591812-Ramphastos_ambiguus.mp3')
        
        cls.sel_tbl_cmto_xc3 = os.path.join(cls.cur_dir, 'tests/selection_tables/XenoCanto/cmto3.selections.txt')
        cls.sel_rec_cmto_xc3 = os.path.join(cls.xc_sound_data, 'CMTOG/SONG_Black-mandibled_Toucan2011-1-24-1.mp3')

        # Xeno Canto CTFLG
        cls.sel_tbl_ctfl_xc1 = os.path.join(cls.cur_dir, 'tests/selection_tables/XenoCanto/ctfl1.selections.txt')
        cls.sel_rec_ctfl_xc1 = os.path.join(cls.xc_sound_data, 'CTFLG/SONG_XC331714-COMMON_TODY_FLYCATCHER_CALLS.mp3')
        
        cls.sel_tbl_ctfl_xc2 = os.path.join(cls.cur_dir, 'tests/selection_tables/XenoCanto/ctfl2.selections.txt')
        cls.sel_rec_ctfl_xc2 = os.path.join(cls.xc_sound_data, 'CTFLG/SONG_XC649369-ENPA108.mp3')
        

    #------------------------------------
    # energy_centroids_by_frame
    #-------------------

    @classmethod
    def energy_centroids_by_frame(cls,
                           sel_tbl_path,
                           recording_path,
                           species=None,
                           color='black',
                           legend_explanation='n/a',
                           origin=None,
                           ax=None
                           ):
        '''
        Given the paths to one selection table, and its
        associated recording, plus a target species, 
        draw a frequency vs. time chart with one line
        for each call by the target species.
        
        The Y-axis will be frequency, the X-axis will be 
        time. Each x/y point denotes the frequency where
        maximum energy across all frequencies at the x moment 
        in time occurs. 
        
        I.e. in contrast to a spectrogram, at each time frame only 
        the frequency with the highest signal energy at 
        the moment in time is identified.
        
        Since each selection table may contain multiple calls
        by the target species, multiple lines may be drawn.
        One for each target species call in the recording.

        :param sel_tbl_path: path to select table file
        :type sel_tbl_path: src
        :param recording_path: path to the corresponding recording
        :type recording_path: str
        :param species: target species whose calls will be plotted
        :type species: str
        :param color: matplotlib color to use for all calls
        :type color: str
        :param legend_explanation: short explanation for the 
            lines to use in legend
        :type legend_explanation: str
        :param origin: where recording originates. Ex.: 'Xeno Canto', or 'Field'
        :type origin: str
        :param ax: Axes to draw into. If None, a new figure is drawn
        :type ax: {None | plt.Axes}
        '''
        cls.ax = SignalAnalyzer.plot_center_freqs(sel_tbl_path,
                                                  recording_path,
                                                  species,
                                                  color=color,
                                                  ax=ax)
        cls.color_explanation[color] = f"{species}; {legend_explanation}"
        cls._place_legend(cls.ax, origin, cls.color_explanation)
        
        cls.ax.figure.show()
        return cls.ax

    #------------------------------------
    # energy_centroids_averaged
    #-------------------
    
    @classmethod
    def energy_centroids_averaged(
            cls,
            sel_tbl_path,
            recording_path,
            species=None,
            color='black',
            legend_explanation='n/a',
            origin=None,
            ax=None
            ):
        '''
        Like energy_centroids_by_frame, but the
        frequencies of highest energy are averaged,
        so that multiple calls to the target species
        will result in a single line.

        :param sel_tbl_path: path to select table file
        :type sel_tbl_path: src
        :param recording_path: path to the corresponding recording
        :type recording_path: str
        :param species: target species whose calls will be plotted
        :type species: str
        :param color: matplotlib color to use for all calls
        :type color: str
        :param legend_explanation: short explanation for the 
            lines to use in legend
        :type legend_explanation: str
        :param origin: where recording originates. Ex.: 'Xeno Canto', or 'Field'
        :type origin: str
        :param ax: Axes to draw into. If None, a new figure is drawn
        :type ax: {None | plt.Axes}
        '''
        
        cls.ax = SignalAnalyzer.plot_center_freqs(sel_tbl_path,
                                                  recording_path,
                                                  species,
                                                  color=color,
                                                  average_calls=True,
                                                  ax=ax)
        cls.color_explanation[color] = f"{species}; {legend_explanation}"
        cls._place_legend(cls.ax, origin, cls.color_explanation)
        
        cls.ax.figure.show()
        return cls.ax

    #------------------------------------
    # align_max_energy_curves
    #-------------------
    
    @classmethod
    def align_max_energy_curves(cls, sel_tbl_path,
                                recording_path,
                                species,
                                color='mediumblue',
                                ax=None):
        '''
        Given two x/y sequences, return an 
        array of index pairs a,b into clipA and
        clipB, respectively. Shifting the curves
        to match the indices results in a best match
        of the shapes on top of each other. 
        
        :param clipA:
        :type clipA:
        :param clipB:
        :type clipB:
        '''

        # Get df:
        #         t1    t2     t3   ...
        #         ---------------------
        #   0     f1    f2     f1
        #   1     f2    f2     f5
        
        curves_all_calls = SignalAnalyzer.compute_spectral_centroids(
            [sel_tbl_path],
            [recording_path],
            species,
            average_calls=False)
        
        # Find reference curve in the middle
        # of all the curves' frequency range:
        # Find the mean of all frequencies,
        # and then the row whose average is 
        # closes to that overall mean:

        #*****global_mean = curves_all_calls.mean().mean()
        #*****line_means  = curves_all_calls.mean(axis=1)
        #************DO THIS WITH FRESH MIND
        # for now: take the middle curve
        
        num_curves, _curve_len = curves_all_calls.shape
        ref_curve_idx = round(num_curves / 2, 0)
        ref_curve    = curves_all_calls.loc[ref_curve_idx, :]
        ref_curve_np = ref_curve.to_numpy()
        
        all_crvs_df  = pd.DataFrame(ref_curve_np)
        
        for curve in curves_all_calls.iterrows():
            # Each row is a 2-tuple whose first
            # el is a row number, and the 2nd
            # el is a pd.Series of the row values:
            row_num, series_otr = curve
            if row_num == ref_curve_idx:
                # Don't align reference curve with itself:
                continue
            cost_matrix, warp = librosa.sequence.dtw(ref_curve_np, 
                                                     series_otr.to_numpy())
            
            cost_normalized = (cost_matrix-cost_matrix.min())/(cost_matrix.max()-cost_matrix.min())
            total_cost = cost_normalized[-1,-1]
            prob = 1 - total_cost
            #**********
            print(warp)
            #**********
        


    #------------------------------------
    # zero_crossings
    #-------------------
    
    # NOT FULLY TESTED/DEBUGGED
    @classmethod
    def zero_crossings(cls, 
                       sel_tbl_path,
                       recording_path,
                       species=None,
                       color='black',
                       legend_explanation='n/a',
                       origin=None,
                       ax=None
                       ):
        clip_dict, _sr = SignalAnalyzer.audio_from_selection_table(
             sel_tbl_path,
             recording_path,
             requested_species=species,
             )
        clips_df = SignalAnalyzer.df_from_clips(clip_dict[species])
        crossing_indices = SignalAnalyzer.zero_crossings(clips_df)
        Charter.linechart(
            crossing_indices, 
            rotation=0,
            ylabel=None,
            xlabel=None,
            color_groups=None,
            ax=None,
            title=None
            )
        return ax

    #------------------------------------
    # new_figure
    #-------------------
    
    @classmethod
    def new_figure(cls):
        if cls.ax is not None:
            cls.ax.clear()

# ------------------ Utilities ------------

    #------------------------------------
    # _place_legend
    #-------------------
    
    @classmethod
    def _place_legend(cls, ax, origin, color_explanation):
        '''
        Adds a legend to the given Axes. Origin denotes
        the recording's provenance: Xeno Canto, or Field.
        Color dict maps a color to a string that explains
        lines of that color. Like
        
                'green' : "VASEG; recording4"
                 
        :param ax: already populated pyplot axes where to add legend
        :type ax: plt.Axes
        :param origin: where recording was created (e.g. Xeno Canto, or Field)
        :type origin: str
        :param color_explanation: map from color name to explanatory
            string for legend
        :type color_explanation: {str : str}
        '''
        lines = ax.get_lines()
        colors_in_legend = []
        line_artists_for_legend = {}
        for line_artist in lines:
            color = line_artist.get_color()
            if color in colors_in_legend:
                continue
            colors_in_legend.append(color)
            line_artists_for_legend[line_artist] = f"{origin}--{color_explanation[color]}"
        
        # Remove current legend, if any, then put in new legend
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.legend(list(line_artists_for_legend.keys()),
                  list(line_artists_for_legend.values())
                  )

# ------------------------ Main ------------
if __name__ == '__main__':
    
    SignalViz.__init_class__()
    
    # ------------ Energy Centroids Each Frame ----------
    
    # Show call examples of CMTOG calls
    # in one XC recording:
    # ax = SignalViz.energy_centroids_by_frame(
    #          SignalViz.sel_tbl_cmto_xc1,
    #          SignalViz.sel_rec_cmto_xc1,
    #          species='CMTOG',
    #          color='mediumblue',
    #          legend_explanation='recording1',
    #          origin='Xeno Canto')
    #
    # input("Hit ENTER to add another CMTOG XC recording: ")
    #
    # # Add calls from another XC recording:
    # ax = SignalViz.energy_centroids_by_frame(
    #          SignalViz.sel_tbl_cmto_xc2,
    #          SignalViz.sel_rec_cmto_xc2,
    #          species='CMTOG',
    #          color='black',
    #          legend_explanation='recording2',
    #          origin='Xeno Canto',
    #          ax=ax)
    # input("Hit ENTER to add another CMTOG XC recording: ")
    #
    # # Add calls from a third XC recording:
    # ax = SignalViz.energy_centroids_by_frame(
    #          SignalViz.sel_tbl_cmto_xc3,
    #          SignalViz.sel_rec_cmto_xc3,
    #          species='CMTOG',
    #          color='red',
    #          legend_explanation='recording3',
    #          origin='Xeno Canto',
    #          ax=ax)
    #
    # input("Hit ENTER to add a CMTOG field recording: ")
    #
    # # Add calls from one Field selection table:
    # ax = SignalViz.energy_centroids_by_frame(
    #         SignalViz.sel_tbl_fld,
    #         SignalViz.sel_recording_fld,
    #         species='CMTOG',
    #         color='green',
    #         legend_explanation='AM03_20190713_055956',
    #         origin='Field',
    #         ax=ax)
    #
    # input("Hit ENTER for CTFLG: ")
    # SignalViz.new_figure()
    #
    # # Same, but with different species:
    #
    # ax = SignalViz.energy_centroids_by_frame(
    #          SignalViz.sel_tbl_ctfl_xc1,
    #          SignalViz.sel_rec_ctfl_xc1,
    #          species='CTFLG',
    #          color='mediumblue',
    #          legend_explanation='recording1',
    #          origin='Xeno Canto')
    #
    # ax = SignalViz.energy_centroids_by_frame(
    #          SignalViz.sel_tbl_ctfl_xc2,
    #          SignalViz.sel_rec_ctfl_xc2,
    #          species='CTFLG',
    #          color='black',
    #          legend_explanation='recording1',
    #          origin='Xeno Canto',
    #          ax=ax)
    #
    # # Add the field recordings for CTFLG:
    # ax = SignalViz.energy_centroids_by_frame(
    #          SignalViz.sel_tbl_fld,
    #          SignalViz.sel_recording_fld,
    #          species='CTFLG',
    #          color='green',
    #          legend_explanation='AM03_20190713_055956',
    #          origin='Field',
    #          ax=ax)
    
    # ------ Energy Centroids per Frame Averaged Across Multiple Calls ---------
    
    ax = SignalViz.energy_centroids_averaged(
                SignalViz.sel_tbl_cmto_xc1,
                SignalViz.sel_rec_cmto_xc1,
                species='CMTOG',
                color='mediumblue',
                legend_explanation='mean recording1',
                origin='Xeno Canto')
    
    input("Hit ENTER to add another CMTOG XC recording: ")
    
    ax = SignalViz.energy_centroids_averaged(
                SignalViz.sel_tbl_cmto_xc2,
                SignalViz.sel_rec_cmto_xc2,
                species='CMTOG',
                color='black',
                legend_explanation='mean recording2',
                origin='Xeno Canto',
                ax=ax)
    
    ax = SignalViz.energy_centroids_averaged(
                SignalViz.sel_tbl_cmto_xc3,
                SignalViz.sel_rec_cmto_xc3,
                species='CMTOG',
                color='red',
                legend_explanation='mean recording3',
                origin='Xeno Canto',
                ax=ax)
    
    input("Hit ENTER to see mean of field recordings: ")
    
    # input("Hit ENTER to add a CMTOG field recording: ")
    #
    # ax = SignalViz.energy_centroids_averaged(
    #             SignalViz.sel_tbl_fld,
    #             SignalViz.sel_recording_fld,
    #             species='CMTOG',
    #             color='green',
    #             legend_explanation='mean AM03_20190713_055956',
    #             origin='Field',
    #             ax=ax)
    #
    # input("Hit ENTER for CTFLG: ")
    # SignalViz.new_figure()
    #
    # # Same, but with different species:
    #
    # ax1 = SignalViz.energy_centroids_averaged(
    #          SignalViz.sel_tbl_ctfl_xc1,
    #          SignalViz.sel_rec_ctfl_xc1,
    #          species='CTFLG',
    #          color='mediumblue',
    #          legend_explanation='mean recording1',
    #          origin='Xeno Canto')
    #
    # ax1 = SignalViz.energy_centroids_averaged(
    #          SignalViz.sel_tbl_ctfl_xc2,
    #          SignalViz.sel_rec_ctfl_xc2,
    #          species='CTFLG',
    #          color='black',
    #          legend_explanation='mean recording2',
    #          origin='Xeno Canto',
    #          ax=ax1)
    #
    # # Add the field recordings for CTFLG:
    # ax1 = SignalViz.energy_centroids_averaged(
    #          SignalViz.sel_tbl_fld,
    #          SignalViz.sel_recording_fld,
    #          species='CTFLG',
    #          color='green',
    #          legend_explanation='mean AM03_20190713_055956',
    #          origin='Field',
    #          ax=ax1)
    
    # ------------------------ Curve Alignment ----------------
    
    # input("Hit ENTER for curve matching")
    # SignalViz.new_figure()
    
    SignalViz.align_max_energy_curves(
        SignalViz.sel_tbl_cmto_xc1,
        SignalViz.sel_rec_cmto_xc1,
        species='CMTOG',
        color='mediumblue'
        )
    
    input("Hit ENTER to quit")
    
    print('Done')
