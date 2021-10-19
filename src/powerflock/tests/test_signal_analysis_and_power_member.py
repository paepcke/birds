'''
Created on Oct 1, 2021

@author: paepcke
'''
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils, Interval
from powerflock.power_member import PowerMember, PowerResult
from powerflock.signal_analysis import SignalAnalyzer, TemplateSelection
from result_analysis.charting import Charter


#*******TEST_ALL = True
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
        
        # Create a signature template to work with:
        cls.recordings = [cls.sel_rec_cmto_xc1, cls.sel_rec_cmto_xc2, cls.sel_rec_cmto_xc3]
        cls.sel_tbls   = [cls.sel_tbl_cmto_xc1, cls.sel_tbl_cmto_xc2, cls.sel_tbl_cmto_xc3]
        cls.templates = SignalAnalyzer.compute_species_templates('CMTOG', 
                                                                 cls.recordings, 
                                                                 cls.sel_tbls)
        
        # Computing the probability of CMTOG on
        # recording 1 takes a few minutes. So those
        # probs for each timepoint are seved on csv.
        # Get a version as a df:
        #   
        #                Probability
        #     Time                  
        #     0.000408      0.251428
        #     0.001224      0.261540
        #     0.002041      0.255386
        #     0.002857      0.255179
        
        cls.probs_rec1_df = pd.read_csv(
            os.path.join(cls.cur_dir, 
                         'cached_results/rec1_cmto_probabilities.csv'), 
            index_col='Time')
        
        # Get as a series as well, where index is time:
        #
        #     Time
        #     0.000408     0.251428
        #     0.001224     0.261540
        #     0.002041     0.255386
        
        cls.probs_rec1_series = cls.probs_rec1_df['Probability']

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
        # There will be 11 calls in this recording,
        # each one about 2.5s long. The clip of 11 calls
        # is ~36.5sec.
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

        # For plotting lines, need each freq
        # series over time to be columns.
        spectral_centroids_T = spectral_centroids.transpose()
        # Index of xposed df is time, which will
        # end up on the x-axis labels: 
        spectral_centroids_T.index = np.arange(0,time_step*num_cols,time_step)
        spectral_centroids_T.columns = [f"Call{i}" for i in spectral_centroids.index]
        
        
        color_group = {'black' : spectral_centroids_T.columns} 
        ax = Charter.linechart(spectral_centroids_T, 
                               ylabel='center frequency (Hz)', 
                               xlabel=u'time (ms??? or \u03bcs)',
                               rotation=45,
                               color_groups=color_group,
                               title="Each line is one call"
                               )
        ax.figure.show()
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

        # Now done once in setUpClass() 
        # recordings = [self.sel_rec_cmto_xc1, self.sel_rec_cmto_xc2, self.sel_rec_cmto_xc3]
        # sel_tbls   = [self.sel_tbl_cmto_xc1, self.sel_tbl_cmto_xc2, self.sel_tbl_cmto_xc3]
        # templates = SignalAnalyzer.compute_species_templates('CMTOG', 
        #                                                      recordings, 
        #                                                      sel_tbls)
        self.assertEqual(len(self.templates), 3)
        self.assertListEqual([len(template) for template in self.templates], [11,8,11])
        
        for tmpl_num, template in enumerate(self.templates):
            for call_num, sig in enumerate(template.signatures):
                rec_fname = Path(self.recordings[tmpl_num]).name
                self.assertTrue(sig.name == f"{rec_fname}_call{call_num}")

        expected_sig_lengths = [
            [85, 72, 133, 60, 59, 45, 48, 42, 37, 87, 107], 
            [46, 46, 41, 41, 46, 68, 48, 46], 
            [58, 54, 53, 62, 58, 54, 54, 56, 58, 56, 51]
            ]
        observed_sig_lengths = [template.sig_lengths for template in self.templates]
        self.assertListEqual(observed_sig_lengths, expected_sig_lengths)
        
        # While we are in here, test signature averaging in
        # templates:
        
        template = self.templates[0]
        mean_sig = template.mean_sig

        # The mean sig should be as long as the
        # longest among the sigs:
        longest = pd.DataFrame(expected_sig_lengths).max().max()
        self.assertEqual(len(template.mean_sig), longest)
        
        # Compute what the first element of the 
        # mean sig should be: the mean of the first
        # element of all sigs in the template:
        
        mean_first_els = np.mean([sig[0] for sig in template.signatures])
        self.assertAlmostEqual(mean_sig[0], mean_first_els, places=3)

    #------------------------------------
    # test_match_probabilty
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_match_probabilty(self):
        # NOTE: truth values for final and intermediate
        #       steps for all three templates are in the
        #       giant comment at the end:
        

        clip,_sr = SoundProcessor.load_audio(self.sel_rec_cmto_xc1)
        xc1_template = self.templates[0]
        self.assertEqual(xc1_template.recording_fname,
                         Path(self.sel_rec_cmto_xc1).name
                         ) 
        prob = SignalAnalyzer.match_probability(clip,
                                                xc1_template,
                                                xc1_template.sample_rate,
                                                template_selection=TemplateSelection.MAX_PROB
                                                )
        self.assertEqual(round(prob, 2), 0.84)
       
        # Now same with MIN_PROB
        prob = SignalAnalyzer.match_probability(clip,
                                                xc1_template,
                                                xc1_template.sample_rate,
                                                template_selection=TemplateSelection.MIN_PROB
                                                )
       
        self.assertEqual(round(prob, 2), 0.17)
       
        # Now best length-fit:
        prob = SignalAnalyzer.match_probability(clip,
                                                xc1_template,
                                                xc1_template.sample_rate,
                                                template_selection=TemplateSelection.SIMILAR_LEN
                                                )
        self.assertEqual(round(prob, 2), 0.60)

        # MULTIPLE TEMPLATES:
        
        prob = SignalAnalyzer.match_probability(clip,
                                                self.templates,
                                                xc1_template.sample_rate,
                                                template_selection=TemplateSelection.MAX_PROB
                                                )
       
        self.assertEqual(round(prob, 2), 0.84)
       
        prob = SignalAnalyzer.match_probability(clip,
                                               self.templates,
                                               xc1_template.sample_rate,
                                               template_selection=TemplateSelection.MIN_PROB
                                               )
       
        self.assertEqual(round(prob, 2), 0.08)
        
        prob = SignalAnalyzer.match_probability(clip,
                                               self.templates,
                                               xc1_template.sample_rate,
                                               template_selection=TemplateSelection.MED_PROB
                                               )
        self.assertEqual(round(prob, 2), 0.52)

    #------------------------------------
    # test_xc_templates_xc_in_template_clips
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
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
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
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

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')

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

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
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


    #------------------------------------
    # test_powermember_compute_probabilities
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_powermember_compute_probabilities(self):

        pwr_member = PowerMember('CMTOG', self.templates[0])
        rec1_arr, rec1_sr = SoundProcessor.load_audio(self.sel_rec_cmto_xc1)
        power_result = pwr_member.compute_probabilities(rec1_arr, rec1_sr)
        probs = power_result.truth_df['Probability'] 
        subsampled_probs = probs[0:len(probs):100]

        expected = pd.Series([0.2514, 0.4600, 0.2234, 0.2161],
                             index=[0.000,
                                    0.082,
                                    0.164,
                                    0.245,
                                    ],
                             name='Probability')
        observed = subsampled_probs.iloc[:4].round(4)
        observed.index = [round(idx,3) for idx in observed.index]
        self.assertTrue((observed == expected).all())
        
        ax = Charter.barchart_over_timepoints(
               subsampled_probs,
               xlabel='Time (s)',
               ylabel='Probability of CMTO',
               title='Probability of CMTO Over Time',
               round_times=1
               )

        ax.figure.show()
        print("Place breakpoint in test_powermember_compute_probabilities to see chart.")

    #------------------------------------
    # test_add_truth_to_power_result
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_add_truth_to_power_result(self):

        pwr_res = PowerResult(self.probs_rec1_series, 'CMTOG')
        rec1_sel_tbl = self.sel_tbls[0]
        pwr_res.add_truth(rec1_sel_tbl)
        
        # First line of df should be: 0.00040816326530612246,0.2514283069254051
        self.assertEqual(round(pwr_res.truth_df.index[0], 4), 0.0004)
        self.assertEqual(round(pwr_res.truth_df['Probability'].iloc[0], 2), 0.25)
        self.assertFalse(pwr_res.truth_df['Truth'].iloc[0])

    #------------------------------------
    # test_plot_pr_curve_power_result
    #-------------------

    #******@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_plot_pr_curve_power_result(self):

        pwr_member = PowerMember('CMTOG', self.templates[0])
        pwr_res    = PowerResult(self.probs_rec1_series, 'CMTOG')
        pwr_res.add_truth(self.sel_tbls[0])
        
        pwr_member.plot_pr_curve(pwr_res)
        
        print('foo')
        

# -------------- Utilities -------------

# ---------------- Truths About templates 1,2, and 3:

'''
template1: [(0.22789070903695596, 85),
            (0.5225781823859109, 72),
            (0.5957025569960972, 133), 
            (0.34903527927628286, 60),
            (0.7236715597704508, 59),
            (0.6414722130153367, 45),
            (0.1669174958705406,  48),
            (0.4247026471733423, 42),
            (0.83726286744302, 37),
            (0.27413558353488465, 87),
            (0.7552083040737554, 107)]

Sorted:
    0.1669174958705406
    0.22789070903695596
    0.27413558353488465
    0.34903527927628286
    0.4247026471733423
    0.5225781823859109
    0.5957025569960972
    0.6414722130153367
    0.7236715597704508
    0.7552083040737554
    0.83726286744302]

==> min: 0.1669174958705406
    max: 0.83726286744302
    med: 0.5225781823859109

template2: [(0.46617733072541556, 46)
            (0.5360834227086135, 46)
            (0.28249214515888, 41)
            (0.4653233417034428, 41)
            (0.08080346766887236, 46)
            (0.10961679210318265, 68)
            (0.27391544502467446, 48)
            (0.17303428671077303, 46)
            ]

Sorted:
    0.08080346766887236
    0.10961679210318265
    0.17303428671077303
    0.27391544502467446
    0.28249214515888
    0.4653233417034428
    0.46617733072541556
    0.5360834227086135

==> min: 0.08080346766887236
    max: 0.5360834227086135
    med: 0.28249214515888


Template 3: [(0.7891265487903032, 58),
             (0.34152329431595296, 54),
             (0.6921432611485201, 53),
             (0.7679901063246655, 62),
             (0.830557469110825, 58),
             (0.8109351277544644, 54),
             (0.8322004764652019, 54),
             (0.8054917354268882, 56),
             (0.772340532153984, 58),
             (0.8104237993011344, 56),
             (0.7330169705978984, 51)
             ]

Sorted:
    0.34152329431595296
    0.6921432611485201
    0.7330169705978984
    0.7679901063246655
    0.772340532153984
    0.7891265487903032
    0.8054917354268882
    0.8104237993011344
    0.8109351277544644
    0.830557469110825
    0.8322004764652019

==> min: 0.34152329431595296
    max: 0.8322004764652019
    med: 0.7891265487903032

Overall aggregation:

    min(0.08080346766887236, 0.1669174958705406, 0.34152329431595296)
       0.08080346766887236
    max(0.5360834227086135, 0.8322004764652019, 0.83726286744302])
       0.83726286744302
    med(0.28249214515888, 0.5225781823859109, 0.7891265487903032)
       0.5225781823859109

'''

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()