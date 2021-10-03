'''
Created on Oct 1, 2021

@author: paepcke
'''
import os
import unittest

from data_augmentation.signal_analysis import SignalAnalyzer
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils, Interval
import pandas as pd
from result_analysis.charting import Charter


#********TEST_ALL = True
TEST_ALL = False

class SignalAnalysisTester(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.sound_data = os.path.join(cls.cur_dir, 'tests/signal_processing_sounds')
        cls.xc_sound_data = os.path.join(cls.cur_dir, 'tests/signal_processing_sounds/XenoCanto')
        
        
        # Field Recordings
        cls.BAFFG_data = os.path.join(cls.sound_data, 'BAFFG')
        cls.CCROC_data = os.path.join(cls.sound_data, 'CCROC')
        
        cls.BAFFG1_rec = os.path.join(cls.BAFFG_data, 'Micrastur-ruficollis-58028.mp3')
        cls.BAFFG2_rec = os.path.join(cls.BAFFG_data, 'Micrastur-ruficollis-75982.mp3')
        cls.BAFFG3_rec = os.path.join(cls.BAFFG_data, 'Micrastur-ruficollis-85219.mp3')
        
        cls.CCROC1_rec = os.path.join(cls.CCROC_data, 'CALL_XC332432-ClayColoredThrush_Yucatan_081216_call2.mp3')
        cls.CCROC2_rec = os.path.join(cls.CCROC_data, 'CALL_XC482432-R028_Clay_coloured_thrush.mp3')
        cls.CCROC3_rec = os.path.join(cls.CCROC_data, 'CALL_XC540584-MixPre-255_Turdus_grayi.mp3')
        
        cls.sel_tbl_fld = os.path.join(cls.cur_dir, 'tests/selection_tables/JZ_DS_AM03_20190713_055956.Table.1.selections.txt')
        # Full field recording for selection tbl: 
        cls.sel_recording_fld = os.path.join(cls.sound_data, 'DS_AM03_20190713_055956.wav')

        # Xeno Canto 
        cls.sel_tbl_cmto_xc1 = os.path.join(cls.cur_dir, 'tests/selection_tables/XenoCanto/cmto1.selections.txt')
        cls.sel_rec_cmto_xc1 = os.path.join(cls.xc_sound_data, 'CMTOG/SONG_XC274155-429_Chestnut-mandibled_Toucan_2_song.mp3')
        
        cls.sel_tbl_cmto_xc2 = os.path.join(cls.cur_dir, 'tests/selection_tables/XenoCanto/cmto2.selections.txt')
        cls.sel_rec_cmto_xc2 = os.path.join(cls.xc_sound_data, 'CMTOG/SONG_XC591812-Ramphastos_ambiguus.mp3')
        

    def setUp(self):
        pass


    def tearDown(self):
        pass

# -------------- Tests ---------------
    
    #------------------------------------
    # test_audio_from_selection_table
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_audio_from_selection_table(self):
        
        species_clips_dicts = SignalAnalyzer.audio_from_selection_table(self.sel_tbl_cmto_xc1,
                                                                        self.sel_rec_cmto_xc1,
                                                                        'cmto')
        sr = 22050
        # Get one dict for each row in the selection tbl:
        row_dicts = Utils.read_raven_selection_table(self.sel_tbl_cmto_xc1)
        for row_num, row_dict in enumerate(row_dicts):
            # One selection table's species:
            species = row_dict['species']
            # Selection duration in sel tbl:
            dur = row_dict['End Time (s)'] - row_dict['Begin Time (s)']
            clip = species_clips_dicts[species][row_num]
            # Duration of the extracted clip
            clip_dur = SoundProcessor.recording_len(clip, sr)
            self.assertEqual(round(clip_dur,2), round(dur,2)) 
            

    #------------------------------------
    # test_spectral_centroid_each_timeframe
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_spectral_centroid_each_timeframe(self):

        # For test recording, sr is 22050
        cmtog_clips_xc1, sr = SignalAnalyzer.audio_from_selection_table(self.sel_tbl_cmto_xc1,
                                                                        self.sel_rec_cmto_xc1,
                                                                        'cmto')
        spectral_centroids = pd.DataFrame([])
        
        for clip in cmtog_clips_xc1['CMTOG']:
            centroid = SignalAnalyzer.spectral_centroid_each_timeframe(clip, sr)
            spectral_centroids = spectral_centroids.append(centroid, ignore_index=True)

        # Replace the nan from unequal
        # clip lengths with 0s:
        spectral_centroids.fillna(0, inplace=True)
        
        _num_rows, num_cols = spectral_centroids.shape
        time_step = int(10**6 * 1/sr)
        col_names = list(Interval(0,time_step*num_cols,time_step).values())
        spectral_centroids.columns = col_names
        color_group = {'black' : spectral_centroids.index} 
        ax = Charter.linechart(spectral_centroids, 
                               ylabel='center frequency (Hz)', 
                               xlabel=u'time (\u03bcs)',
                               rotation=45,
                               color_groups=color_group)
        ax.figure.close()

    #------------------------------------
    # test_plot_center_freqs
    #-------------------
    
    #***********@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_plot_center_freqs(self):
        
        # One selection table/recording pair:
        
        # ax = SignalAnalyzer.plot_center_freqs(self.sel_tbl_cmto_xc1,
        #                                       self.sel_rec_cmto_xc1,
        #                                       'CMTOG')
        # ax.figure.close()
        
        # Two selection table/recording pairs
        
        # All lines will be blue
        # ax = SignalAnalyzer.plot_center_freqs([self.sel_tbl_cmto_xc1, self.sel_tbl_cmto_xc2],
        #                                       [self.sel_rec_cmto_xc1, self.sel_rec_cmto_xc2],
        #                                       'CMTOG')
        # ax.figure.close()
        
        # Two selection table/recording pairs,
        # different colors for calls from different
        # selection tables, but same species:
        
        ax = SignalAnalyzer.plot_center_freqs(self.sel_tbl_cmto_xc1, 
                                              self.sel_rec_cmto_xc1, 
                                              'CMTOG',
                                              color='mediumblue')
                                              
        ax = SignalAnalyzer.plot_center_freqs(self.sel_tbl_cmto_xc2, 
                                              self.sel_rec_cmto_xc2,
                                              'CMTOG',
                                              color='black',
                                              ax=ax)
        ax.figure.close()
        

# -------------- Utilities -------------


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()