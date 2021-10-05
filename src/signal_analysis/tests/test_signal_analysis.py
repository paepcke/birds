'''
Created on Oct 1, 2021

@author: paepcke
'''
import os
import unittest

from signal_analysis.signal_analysis import SignalAnalyzer
from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils, Interval
import pandas as pd
from result_analysis.charting import Charter


#***********TEST_ALL = True
TEST_ALL = False

class SignalAnalysisTester(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.sound_data = os.path.join(cls.cur_dir, 'signal_processing_sounds')
        cls.xc_sound_data = os.path.join(cls.cur_dir, 'signal_processing_sounds/XenoCanto')
        
        
        # Field Recordings
        cls.BAFFG_data = os.path.join(cls.sound_data, 'BAFFG')
        cls.CCROC_data = os.path.join(cls.sound_data, 'CCROC')
        
        cls.BAFFG1_rec = os.path.join(cls.BAFFG_data, 'Micrastur-ruficollis-58028.mp3')
        cls.BAFFG2_rec = os.path.join(cls.BAFFG_data, 'Micrastur-ruficollis-75982.mp3')
        cls.BAFFG3_rec = os.path.join(cls.BAFFG_data, 'Micrastur-ruficollis-85219.mp3')
        
        cls.CCROC1_rec = os.path.join(cls.CCROC_data, 'CALL_XC332432-ClayColoredThrush_Yucatan_081216_call2.mp3')
        cls.CCROC2_rec = os.path.join(cls.CCROC_data, 'CALL_XC482432-R028_Clay_coloured_thrush.mp3')
        cls.CCROC3_rec = os.path.join(cls.CCROC_data, 'CALL_XC540584-MixPre-255_Turdus_grayi.mp3')
        
        cls.DCFLC_rec_fld      = os.path.join(cls.sound_data, 'Field/DCFLC/DS_AM17_20190713_172958.WAV')
        cls.DCFLC_sel_tbl_fld  = os.path.join(cls.sound_data, 'Field/DCFLC/JZ_DS_AM17_20190713_172958.Table.1.selections.txt')
        
        cls.sel_tbl_fld = os.path.join(cls.cur_dir, 'selection_tables/JZ_DS_AM03_20190713_055956.Table.1.selections.txt')
        # Full field recording for selection tbl: 
        cls.sel_recording_fld = os.path.join(cls.sound_data, 'DS_AM03_20190713_055956.wav')

        # Xeno Canto 
        cls.sel_tbl_cmto_xc1 = os.path.join(cls.cur_dir, 'selection_tables/XenoCanto/cmto1.selections.txt')
        cls.sel_rec_cmto_xc1 = os.path.join(cls.xc_sound_data, 'CMTOG/SONG_XC274155-429_Chestnut-mandibled_Toucan_2_song.mp3')
        
        cls.sel_tbl_cmto_xc2 = os.path.join(cls.cur_dir, 'selection_tables/XenoCanto/cmto2.selections.txt')
        cls.sel_rec_cmto_xc2 = os.path.join(cls.xc_sound_data, 'CMTOG/SONG_XC591812-Ramphastos_ambiguus.mp3')
    
        cls.sel_tbl_cmto_xc3 = os.path.join(cls.cur_dir, 'selection_tables/XenoCanto/cmto3.selections.txt')
        cls.sel_rec_cmto_xc3 = os.path.join(cls.xc_sound_data, 'CMTOG/SONG_Black-mandibled_Toucan2011-1-24-1.mp3')
        

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
        
        species_clips_dicts, sr = SignalAnalyzer.audio_from_selection_table(self.sel_tbl_cmto_xc1,
                                                                            self.sel_rec_cmto_xc1,
                                                                            'cmto')
        #sr = 22050
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
        _ax = Charter.linechart(spectral_centroids, 
                               ylabel='center frequency (Hz)', 
                               xlabel=u'time (\u03bcs)',
                               rotation=45,
                               color_groups=color_group)

        print("Put breakpoint here to view")


    #------------------------------------
    # test_plot_center_freqs
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
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
        ax.figure.show()
        print("Put breakpoint here to view")

    #------------------------------------
    # test_compute_species_templates
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_compute_species_templates(self):

        recordings = [self.sel_rec_cmto_xc1, self.sel_rec_cmto_xc2, self.sel_rec_cmto_xc3]
        sel_tbls   = [self.sel_tbl_cmto_xc1, self.sel_tbl_cmto_xc2, self.sel_tbl_cmto_xc3]
        templates = SignalAnalyzer.compute_species_templates('CMTOG', 
                                                             recordings, 
                                                             sel_tbls)
        self.assertEqual(templates.shape, (3,134))
        true_lengths = templates['TrueLen']
        expected = pd.Series([133,68,62])
        expected.index = ['SONG_XC274155-429_Chestnut-mandibled_Toucan_2_song.mp3',
                          'SONG_XC591812-Ramphastos_ambiguus.mp3',
                          'SONG_Black-mandibled_Toucan2011-1-24-1.mp3'
                          ]
        self.assertTrue((true_lengths == expected).all())
        print('foo')


    #------------------------------------
    # test_xc_templates_xc_in_template_clips
    #-------------------
    
    #*********@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_xc_templates_xc_in_template_clips(self):

        # Clips used in templates and to test 
        # for similarity to those templates.
        # Expect high match probs
        
        # Templates from 3 CMTOG recordings:
        recordings = [self.sel_rec_cmto_xc1, self.sel_rec_cmto_xc2, self.sel_rec_cmto_xc3]
        sel_tbls   = [self.sel_tbl_cmto_xc1, self.sel_tbl_cmto_xc2, self.sel_tbl_cmto_xc3]
        templates = SignalAnalyzer.compute_species_templates('CMTOG', 
                                                             recordings, 
                                                             sel_tbls)
        
        # Clips from one of the three recordings:
        clip_dict_in_sample, sr = SignalAnalyzer.audio_from_selection_table(
            sel_tbl_path=self.sel_tbl_cmto_xc1,
            recording_path=self.sel_rec_cmto_xc1,
            requested_species='CMTOG')
        
        all_probs = []
        for clip in clip_dict_in_sample['CMTOG']:
            prob = SignalAnalyzer.match_probability(clip, templates, sr)
            all_probs.append(prob)
            #self.assertEqual(round(prob,2), 0.35)
        print(f"Templates CTMOG (3 XC recordings); {len(all_probs)} clips XC CMTOGs: {all_probs}")
        
    #------------------------------------
    # test_xc_templates_xc_outof_template_clips
    #-------------------
    
    #********@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_xc_templates_xc_outof_template_clips(self):
        
        # XC clips NOT used when creating templates:
        #
        recordings = [self.sel_rec_cmto_xc1, self.sel_rec_cmto_xc2]
        sel_tbls   = [self.sel_tbl_cmto_xc1, self.sel_tbl_cmto_xc2]
        templates = SignalAnalyzer.compute_species_templates('CMTOG', 
                                                             recordings, 
                                                             sel_tbls)
        clip_dict_in_sample, sr = SignalAnalyzer.audio_from_selection_table(
            # Recording not involved in template creation:
            sel_tbl_path=self.sel_tbl_cmto_xc3,
            recording_path=self.sel_rec_cmto_xc3,
            requested_species='CMTOG')
        
        all_probs = []
        for test_clip in clip_dict_in_sample['CMTOG']:
            all_probs.append(SignalAnalyzer.match_probability(test_clip, templates, sr))
        
        print(f"Templates CTMOG (2 XC recordings); {len(all_probs)} clips XC CMTOGs: {all_probs}")
        

    #------------------------------------
    # test_xc_templates_field_clip
    #-------------------

    #*********@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')

    def test_xc_templates_field_clips_positive(self):
        # Templates from C 
        recordings = [self.sel_rec_cmto_xc1, self.sel_rec_cmto_xc2, self.sel_rec_cmto_xc3]
        sel_tbls   = [self.sel_tbl_cmto_xc1, self.sel_tbl_cmto_xc2, self.sel_tbl_cmto_xc3]
        templates = SignalAnalyzer.compute_species_templates('CMTOG', 
                                                             recordings, 
                                                             sel_tbls)
        
        clip_dict_outof_template, sr = SignalAnalyzer.audio_from_selection_table(
            sel_tbl_path=self.sel_tbl_fld,
            recording_path=self.sel_recording_fld,
            requested_species='CMTOG')

        all_probs = []
        for test_clip in clip_dict_outof_template['CMTOG']:
            prob = SignalAnalyzer.match_probability(test_clip, templates, sr)
            all_probs.append(prob)
            
        print(f"Templates CTMOG; {len(all_probs)} clips FIELD CMTOGs: {all_probs}")
        
    #------------------------------------
    # test_xc_templates_field_clips_negative
    #-------------------

    #*********@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_xc_templates_field_clips_negative(self):

        # Make the template from CMTOC:
        recordings = [self.sel_rec_cmto_xc1, self.sel_rec_cmto_xc2, self.sel_rec_cmto_xc3]
        sel_tbls   = [self.sel_tbl_cmto_xc1, self.sel_tbl_cmto_xc2, self.sel_tbl_cmto_xc3]
        templates = SignalAnalyzer.compute_species_templates('CMTOG', 
                                                             recordings, 
                                                             sel_tbls)
        clip_dict_outof_species, sr = SignalAnalyzer.audio_from_selection_table(
            sel_tbl_path=self.DCFLC_sel_tbl_fld,
            recording_path=self.DCFLC_rec_fld,
            requested_species='DCFLC')

        all_probs = []
        for test_clip in clip_dict_outof_species['DCFLC']:
        
            prob = SignalAnalyzer.match_probability(test_clip, templates, sr)
            # Clips not used in templating:
            #*****self.assertEqual(round(prob, 2), 0.27)
            all_probs.append(prob)
        
        print(f"Templates CTMOG; {len(all_probs)} clips FIELD DCFLCs: {all_probs}")
        
# -------------- Utilities -------------


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()