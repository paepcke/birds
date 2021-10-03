'''
Created on Oct 2, 2021

@author: paepcke
'''

# TODO:
#    o legends are wrong now that we
#      make two figures.
#    o Next: correlation

import os

#from signal_analysis.signal_analysis import SignalAnalyzer
from signal_analysis import SignalAnalyzer

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
        
        # Xeno Canto CTFLG
        cls.sel_tbl_ctfl_xc1 = os.path.join(cls.cur_dir, 'tests/selection_tables/XenoCanto/ctfl1.selections.txt')
        cls.sel_rec_ctfl_xc1 = os.path.join(cls.xc_sound_data, 'CTFLG/SONG_XC331714-COMMON_TODY_FLYCATCHER_CALLS.mp3')
        
        cls.sel_tbl_ctfl_xc2 = os.path.join(cls.cur_dir, 'tests/selection_tables/XenoCanto/ctfl2.selections.txt')
        cls.sel_rec_ctfl_xc2 = os.path.join(cls.xc_sound_data, 'CTFLG/SONG_XC649369-ENPA108.mp3')
        

    #------------------------------------
    # center_frequencies
    #-------------------

    @classmethod
    def center_frequencies(cls,
                           sel_tbl_path,
                           recording_path,
                           species=None,
                           color='black',
                           legend_explanation='n/a',
                           origin=None,
                           ax=None
                           ):
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
    # Show call examples of CMTOG calls
    # in one XC recording:
    ax = SignalViz.center_frequencies(
             SignalViz.sel_tbl_cmto_xc1,
             SignalViz.sel_rec_cmto_xc1,
             species='CMTOG',
             color='mediumblue',
             legend_explanation='recording1',
             origin='Xeno Canto')
    
    input("Hit ENTER to add another CMTOG XC recording: ")
    
    # Add calls from another XC recording:
    ax = SignalViz.center_frequencies(
             SignalViz.sel_tbl_cmto_xc2,
             SignalViz.sel_rec_cmto_xc2,
             species='CMTOG',
             color='black',
             legend_explanation='recording2',
             origin='Xeno Canto',
             ax=ax)

    input("Hit ENTER to add a CMTOG field recording: ")
    
    # Add calls from one Field selection table:
    ax = SignalViz.center_frequencies(
            SignalViz.sel_tbl_fld,
            SignalViz.sel_recording_fld,
            species='CMTOG',
            color='green',
            legend_explanation='AM03_20190713_055956',
            origin='Field',
            ax=ax)

    input("Hit ENTER for CTFLG: ")
    SignalViz.new_figure()
    
    # Same, but with different species:
    
    ax = SignalViz.center_frequencies(
             SignalViz.sel_tbl_ctfl_xc1,
             SignalViz.sel_rec_ctfl_xc1,
             species='CTFLG',
             color='mediumblue',
             legend_explanation='recording1',
             origin='Xeno Canto')
    
    ax = SignalViz.center_frequencies(
             SignalViz.sel_tbl_ctfl_xc2,
             SignalViz.sel_rec_ctfl_xc2,
             species='CTFLG',
             color='black',
             legend_explanation='recording1',
             origin='Xeno Canto')

    # Add the field recordings for CTFLG:
    ax = SignalViz.center_frequencies(
             SignalViz.sel_tbl_fld,
             SignalViz.sel_recording_fld,
             species='CTFLG',
             color='green',
             legend_explanation='AM03_20190713_055956',
             origin='Field')
    
    input("Hit ENTER to quit")
    
    print('Done')
