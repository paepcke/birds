'''
Created on Oct 1, 2021

@author: paepcke
'''
import copy
import os
import unittest

from experiment_manager.experiment_manager import ExperimentManager
import librosa

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils, RavenSelectionTable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from powerflock.power_member import PowerMember, PowerResult, PowerQuantileClassifier
from powerflock.signal_analysis import SignalAnalyzer
from powerflock.signatures import SpectralTemplate, TemplateCollection
from result_analysis.charting import Charter

# If a column is added to signature df, then:
#   o run to just past
#      cls.templates = cls.experiment.read('templates', TemplateCollection)
#   o Then update all test templates' signatures dataframes and
#     associated scale_info structs with these two loops:
#
# for tmpl in cls.templates.values():
#     for sig in tmpl.signatures:
#         num_rows = len(sig.sig)
#         sig.sig['energy_sum'] = [0.0]*num_rows

# for tmpl in cls.templates.values():
#     for sig in tmpl.signatures:
#         sig.scale_info['energy_sum'] = {'mean' : 0.0, 'standard_measure' : 0.0}
#
# Then save back to the test template:
# cls.experiment.save('templates', cls.templates)


#*****TEST_ALL = True
TEST_ALL = False

class SignalAnalysisTester(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        
        cls.cur_dir = os.path.dirname(__file__)
        cls.sound_data = os.path.join(cls.cur_dir, 'signal_processing_sounds')
        cls.xc_sound_data = os.path.join(cls.cur_dir, 'signal_processing_sounds/XenoCanto')

        cls.experiment = ExperimentManager(os.path.join(cls.cur_dir, 
                                                        'quad_sig_calibrator_experiment')) 


        # Test recordings:
        cls.triangle440 = os.path.join(cls.sound_data, 'artificial/triangle440.wav')
        cls.triangle440Declining = os.path.join(cls.sound_data, 'artificial/triangle2000Declining.wav')
        
        # PowerResult to play with:
        cls.pwr_res = PowerResult.json_load(os.path.join(cls.cur_dir, 
                                                         'data/pwr_res.json'))
        
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
        cls.sel_tbl_cmto_xc1 = os.path.join(cls.cur_dir, 'selection_tables/XenoCanto/sel_tbl_Kelley_SONG_XC274155-429_CMTO_KEL.txt')
        cls.sel_rec_cmto_xc1 = os.path.join(cls.xc_sound_data, 'CMTOG/SONG_XC274155-429_Chestnut-mandibled_Toucan_2_song.mp3')
        
        cls.sel_tbl_cmto_xc2 = os.path.join(cls.cur_dir, 'selection_tables/XenoCanto/cmto2.selections.txt')
        cls.sel_rec_cmto_xc2 = os.path.join(cls.xc_sound_data, 'CMTOG/SONG_XC591812-Ramphastos_ambiguus.mp3')
    
        cls.sel_tbl_cmto_xc3 = os.path.join(cls.cur_dir, 'selection_tables/XenoCanto/cmto3.selections.txt')
        cls.sel_rec_cmto_xc3 = os.path.join(cls.xc_sound_data, 'CMTOG/SONG_Black-mandibled_Toucan2011-1-24-1.mp3')
        
        # Create a signature template to work with:
        cls.recordings = [cls.sel_rec_cmto_xc1, cls.sel_rec_cmto_xc2, cls.sel_rec_cmto_xc3]
        cls.sel_tbls   = [cls.sel_tbl_cmto_xc1, cls.sel_tbl_cmto_xc2, cls.sel_tbl_cmto_xc3]
        
        cls.templates = cls.experiment.read('templates', TemplateCollection)
        
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
            clip = species_clips_dicts[species][row_num]['clip']
            # Duration of the extracted clip
            clip_dur = SoundProcessor.recording_len(clip, sr)
            self.assertEqual(round(clip_dur,2), round(dur,2)) 

    #------------------------------------
    # test_spectral_flatness
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_spectral_flatness(self):
        
        # For test recording, sr is 22050
        # There will be 11 calls in this recording,
        # each one about 2.5s long. The clip of 11 calls
        # is ~36.5sec.
        #cmtog_clips_xc1, _sr = SignalAnalyzer.audio_from_selection_table(self.sel_tbl_cmto_xc1,
        #                                                                self.sel_rec_cmto_xc1,
        #                                                                'cmto')
        
        
        # for clip_dict in cmtog_clips_xc1['CMTOG']:
        #     flatness = SignalAnalyzer.spectral_flatness(audio=clip_dict['clip'])
        #     print(flatness)
        flatness = SignalAnalyzer.spectral_flatness(spec_src=self.sel_rec_cmto_xc1)
        
        spectro = SignalAnalyzer.raven_spectrogram(self.sel_rec_cmto_xc1)
        mesh = plt.pcolormesh(spectro.columns, list(spectro.index), spectro, cmap='jet', shading='auto')
        ax = mesh.axes
        
        # Add the flatness curve to to the spectrogram.
        # Since flatness values are in [0,1], we need to scale
        # them to the y-axis of the spectro mesh. Else the curve
        # would be near-flat against the x-axis:
        
        flatness_scaled = Charter.scale(flatness, (spectro.index.min(), spectro.index.max()))
        flatness_scaled_no_outliers = flatness_scaled[flatness_scaled < 11000]
        ax.figure.show()
        Charter.linechart(flatness_scaled_no_outliers, 
                          ax=ax, 
                          color_groups={'SigSpectralFlatness' : 'red'},
                          title="Test of Flatness"
                          )

        # To see chart: put breakpoint here.
        # maybe have to:
        #    ax.figure.show()

    #------------------------------------
    # test_contour_mask 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_contour_mask(self):
        
        mask1 = SignalAnalyzer.contour_mask(self.triangle440Declining)
        self.assertTupleEqual(mask1.shape, (1025, 69))
        #Charter.spectrogram_plot(mask1)


    #------------------------------------
    # test_spectral_continuity
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_spectral_continuity(self):

        #continuity = SignalAnalyzer.spectral_continuity(audio=self.sel_rec_cmto_xc1,
        #                                                edge_mag_thres=1.0)
        # continuity = SignalAnalyzer.spectral_continuity(audio=self.sel_rec_cmto_xc1,
        #                                                 edge_mag_thres=0.5
        #                                                 )
        long_contours, continuity = SignalAnalyzer.spectral_continuity(audio=self.sel_rec_cmto_xc1)

        # A few spot checks:
        self.assertTupleEqual(continuity.shape, (3180,))
        self.assertTupleEqual(long_contours.shape, (1025, 3180))
        self.assertEqual(round(continuity[36.86167800453515], 4), 50.0488)

    #------------------------------------
    # test_harmonic_pitch
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_harmonic_pitch(self):
        
        # contours = pd.DataFrame([[True,  True,  False],
        #                          [True,  True,   True],
        #                          [False, True,   True],
        #                          [True,  True,   True]],
        #                          index=[10.0, 25.5, 30.0, 35.5],
        #                          columns=[0.1, 0.2, 0.4])
        
        # For column 1, frequencies with contours are [10.0, 25.5, 35.5].
        # For column 1, frequencies are [10.0, 25.5, 30.0, 35.5], and
        # for column 3, frequencies with contours are [25.5, 30.0, 35.5].
        #
        # The return Series will be the medians of the diffs among adjacent
        # frequency bands:
        #
        #    median([15.5, 10.0])     = 12.75 
        #    median([15.5, 4.5, 5.5]) =  5.5
        #    median([4.5, 5.5])       =  5.0

        # pitches = SignalAnalyzer.harmonic_pitch(contours)
        # expected = pd.Series([12.75, 5.5, 5.0],
        #                      index=contours.columns,
        #                      name='harmonic_pitch'
        #                      )
        # Utils.assertSeriesEqual(pitches, expected)
        
        harm_pitch = SignalAnalyzer.harmonic_pitch(self.triangle440)
        self.assertEqual(int(harm_pitch.median()), 436)

    #------------------------------------
    # test_freq_modulations
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_freq_modulations(self):
        
        median_angles = SignalAnalyzer.freq_modulations(self.triangle440Declining)
        
        self.assertEqual(len(median_angles), 69)
        self.assertEqual(median_angles.loc[0.7082086167800453].round(5), 46.66721)
        self.assertEqual(median_angles.loc[0.058049886621315196], 0.0)

    # #------------------------------------
    # # test_multitaper_spectrogram 
    # #-------------------
    #
    # @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    # def test_multitaper_spectrogram(self):
    #
    #     mt_spec = SignalAnalyzer.multitaper_spectrogram(self.sel_rec_cmto_xc1)
    #     ax = Charter.spectrogram_plot(mt_spec, fig_title="Multitaper Spectrogram")
    #
    #     self.assertTupleEqual(mt_spec.shape, (513, 1589))
    #     # Put breakpoint on statement below if in Eclipse: 
    #     ax.figure.show()
    #     input("Hit Enter to continue: ")

    #------------------------------------
    # test_match_probabilty
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_match_probabilty(self):
        xc1_template = self.templates['CMTOG']
        recording = os.path.join(self.cur_dir, xc1_template.recording_fname)
        clip,_sr = SoundProcessor.load_audio(recording)
        
        # To save time, only test with 2 sigs:
        small_template = SpectralTemplate(xc1_template.signatures[:2])
        df = SignalAnalyzer.match_probability(clip, small_template)
        self.assertTupleEqual(df.shape, (510,5))
        
        # Check the first row:
        first_row = df.iloc[0]
        expected = pd.Series([0.0, 115.0, 115.0, 0.000004, 1.0],
                             index=['start_idx', 'stop_idx', 'n_samples', 'match_prob', 'sig_id'],
                             name=0)
        Utils.assertSeriesEqual(first_row, expected, decimals=2)
        
    #------------------------------------
    # test_xc_templates_xc_in_template_clips
    #-------------------
    
    # @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    # def test_xc_templates_xc_in_template_clips(self):
    #
    #     # Clips used in templates and to test 
    #     # for similarity to those templates.
    #     # Expect high match probs
    #
    #     # Templates from 3 CMTOG recordings:
    #     recordings = [self.sel_rec_cmto_xc1, self.sel_rec_cmto_xc2, self.sel_rec_cmto_xc3]
    #     sel_tbls   = [self.sel_tbl_cmto_xc1, self.sel_tbl_cmto_xc2, self.sel_tbl_cmto_xc3]
    #     templates = SignalAnalyzer.compute_species_templates('CMTOG', 
    #                                                          recordings, 
    #                                                          sel_tbls)
    #
    #     # Clips from one of the three recordings:
    #     clip_dict_in_sample, sr = SignalAnalyzer.audio_from_selection_table(
    #         sel_tbl_path=self.sel_tbl_cmto_xc1,
    #         recording_path=self.sel_rec_cmto_xc1,
    #         requested_species='CMTOG')
    #
    #     all_probs = []
    #     for clip in clip_dict_in_sample['CMTOG']:
    #         prob = SignalAnalyzer.match_probability(clip, templates, sr)
    #         all_probs.append(prob)
    #         #self.assertEqual(round(prob,2), 0.35)
    #     print(f"Templates CTMOG (3 XC recordings); {len(all_probs)} clips XC CMTOGs: {all_probs}")
    #
    # #------------------------------------
    # # test_xc_templates_xc_outof_template_clips
    # #-------------------
    #
    # @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    # def test_xc_templates_xc_outof_template_clips(self):
    #
    #     # XC clips NOT used when creating templates:
    #     #
    #     recordings = [self.sel_rec_cmto_xc1, self.sel_rec_cmto_xc2]
    #     sel_tbls   = [self.sel_tbl_cmto_xc1, self.sel_tbl_cmto_xc2]
    #     templates = SignalAnalyzer.compute_species_templates('CMTOG', 
    #                                                          recordings, 
    #                                                          sel_tbls)
    #     clip_dict_in_sample, sr = SignalAnalyzer.audio_from_selection_table(
    #         # Recording not involved in template creation:
    #         sel_tbl_path=self.sel_tbl_cmto_xc3,
    #         recording_path=self.sel_rec_cmto_xc3,
    #         requested_species='CMTOG')
    #
    #     all_probs = []
    #     for test_clip in clip_dict_in_sample['CMTOG']:
    #         all_probs.append(SignalAnalyzer.match_probability(test_clip, templates, sr))
    #
    #     print(f"Templates CTMOG (2 XC recordings); {len(all_probs)} clips XC CMTOGs: {all_probs}")
    #


    # #------------------------------------
    # # test_xc_templates_field_clip
    # #-------------------
    #
    # @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    #
    # def test_xc_templates_field_clips_positive(self):
    #     # Templates from C 
    #     recordings = [self.sel_rec_cmto_xc1, self.sel_rec_cmto_xc2, self.sel_rec_cmto_xc3]
    #     sel_tbls   = [self.sel_tbl_cmto_xc1, self.sel_tbl_cmto_xc2, self.sel_tbl_cmto_xc3]
    #     templates = SignalAnalyzer.compute_species_templates('CMTOG', 
    #                                                          recordings, 
    #                                                          sel_tbls)
    #
    #     clip_dict_outof_template, sr = SignalAnalyzer.audio_from_selection_table(
    #         sel_tbl_path=self.sel_tbl_fld,
    #         recording_path=self.sel_recording_fld,
    #         requested_species='CMTOG')
    #
    #     all_probs = []
    #     for test_clip in clip_dict_outof_template['CMTOG']:
    #         prob = SignalAnalyzer.match_probability(test_clip, templates, sr)
    #         all_probs.append(prob)
    #
    #     print(f"Templates CTMOG; {len(all_probs)} clips FIELD CMTOGs: {all_probs}")
        
    # #------------------------------------
    # # test_xc_templates_field_clips_negative
    # #-------------------
    #
    # @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    # def test_xc_templates_field_clips_negative(self):
    #
    #     # Make the template from CMTOC:
    #     recordings = [self.sel_rec_cmto_xc1, self.sel_rec_cmto_xc2, self.sel_rec_cmto_xc3]
    #     sel_tbls   = [self.sel_tbl_cmto_xc1, self.sel_tbl_cmto_xc2, self.sel_tbl_cmto_xc3]
    #     templates = SignalAnalyzer.compute_species_templates('CMTOG', 
    #                                                          recordings, 
    #                                                          sel_tbls)
    #     clip_dict_outof_species, sr = SignalAnalyzer.audio_from_selection_table(
    #         sel_tbl_path=self.DCFLC_sel_tbl_fld,
    #         recording_path=self.DCFLC_rec_fld,
    #         requested_species='DCFLC')
    #
    #     all_probs = []
    #     for test_clip in clip_dict_outof_species['DCFLC']:
    #
    #         prob = SignalAnalyzer.match_probability(test_clip, templates, sr)
    #         # Clips not used in templating:
    #         #*****self.assertEqual(round(prob, 2), 0.27)
    #         all_probs.append(prob)
    #
    #     print(f"Templates CTMOG; {len(all_probs)} clips FIELD DCFLCs: {all_probs}")


    #------------------------------------
    # test_power_grid_search 
    #-------------------
    
    #********@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_power_grid_search(self):
        '''
        Takes about 3 minutes:
        '''
        # To save time, only test with 2 sigs:
        small_template = SpectralTemplate(self.templates['CMTOG'].signatures[:2])
        pwr_member = PowerMember('CMTOG', small_template)
        
        # pr_disp = sklearn.metrics.PrecisionRecallDisplay.from_estimator(
        #     clf,
        #     pwr_res.prob_df.probability,
        #     pwr_res.prob_df.Truth
        #     )
        # pr_disp.plot()
        from timeit import default_timer as timer

        start = timer()
        # 1hr:20min
        grid_res = PowerQuantileClassifier.grid_search(
            pwr_member,
            audio_file=self.sel_rec_cmto_xc1,
            selection_tbl_file=self.sel_tbl_cmto_xc1,
            quantile_thresholds=[0.80, 0.85],
            slide_widths=[0.2]
            )
        end = timer()
        print(end - start)
        f1_col = grid_res.f1
        expected = pd.Series([0.388889, 0.463768],
                              name='f1', 
                              index=[(1, 0.85, 0.2),
                                     (2, 0.85, 0.2)])
        Utils.assertSeriesEqual(f1_col, expected, decimals=3)

    #------------------------------------
    # test_powermember_compute_probabilities_single_ccall
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_PowerQuantileClassifier(self):

        power_res = self.pwr_res.copy()
        df = power_res.prob_df
        
        #**** WAS:self.assertTupleEqual(df.shape, (76, 8))
        self.assertTupleEqual(df.shape, (256,8))
        first_row = df.iloc[0]
        expected = pd.Series({
            'start_idx'   :    0.000000,
            'stop_idx'    :  115.000000,
            'n_samples'   :  115.000000,
            'match_prob'  :    0.062621,
            'sig_id'      :    1.000000,
            'start_time'  :    0.000000,
            'stop_time'   :    1.335147,
            'raw_prob'    :    0.072951},
            name=0.6675736961451247)
        Utils.assertSeriesEqual(first_row, expected, 3)

        sel_tbl = RavenSelectionTable(self.sel_tbl_cmto_xc1)
        power_res.add_truth(sel_tbl)
        
        clf = PowerQuantileClassifier(1.0,        # sig_id, 
                                      0.75        # threshold_quantile)
                                      )
        clf.fit(power_res)
        self.assertTrue(clf.decision_function(0.543))
        self.assertFalse(clf.decision_function(-4))
        self.assertTrue(clf.decision_function(0.61))
        expected = [True, True, False, True]
        self.assertTrue((clf.decision_function([0.61, 0.4, 0.0, 0.98]) == expected).all())

    #------------------------------------
    # _co_chart
    #-------------------
    
    def _co_chart(self, template, audio):
        
        # audio_clip_width = librosa.frames_to_samples(
        #     max(template.sig_lengths), 
        #         template.hop_length, 
        #         template.n_fft)
        sig = template.signatures[0].copy()
        sig.index = [round(time_tick, 2) 
                     for time_tick 
                     in template.as_time(sig)
                     ]
        ax = None
        #sr = 22050
        num_frames = len(sig)
        audio_clip_width = librosa.frames_to_samples(num_frames, 
                                                     hop_length=template.hop_length)
        for start_idx in np.arange(0, len(audio), audio_clip_width):
            end_idx = start_idx + audio_clip_width

            try:
                aud_snip = audio[start_idx:end_idx]
                clip_sig = SignalAnalyzer.spectral_measures_each_timeframe(aud_snip)
                # Make clip sig same length
                clip_sig.index = [round(time_tick, 2) 
                                  for time_tick 
                                  in template.as_time(clip_sig)
                                  ]
                df = pd.DataFrame()
                if ax is None:
                    # Show the template sig once:
                    df['Template'] = sig
                    color_groups = {'Template' : 'blue', 'Clip' : 'green'}
                else:
                    color_groups = {'Clip' : 'green'}
                df['Clip'] = clip_sig
                
                ax = Charter.linechart(df, color_groups=color_groups, ax=ax) 
                # prob = SignalAnalyzer.match_probability(
                #     aud_snip, 
                #     self.spectral_template, sr)
                # center_time = np.floor(start_idx + self.slide_width / 2.) / sr
                # probs[center_time] = prob
                 
            except IndexError:
                # Slid past end of given audio:
                break

    #------------------------------------
    # test_powermember_compute_probabilities
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_powermember_compute_probabilities(self):

        # To save time, only test with 2 sigs:
        small_template = SpectralTemplate(self.templates['CMTOG'].signatures[:2])
        
        pwr_member = PowerMember('CMTOG', small_template)
        rec1_arr, rec1_sr = SoundProcessor.load_audio(self.sel_rec_cmto_xc1)
        
        power_result = pwr_member.compute_probabilities(rec1_arr, sr=rec1_sr)
        
        probs = power_result.prob_df.match_prob
        subsampled_probs = probs.iloc[0:len(probs):100]
        expected = pd.Series([3.81311763e-06, 6.45077848e-04, 1.59876707e-03, 3.03562669e-03,
                              1.80539105e-03, 5.95967358e-03],
                             index=[0.6675736961451247,  14.59954648526077, 28.531519274376414,
                                    6.9137414965986395, 20.845714285714287,  34.77768707482993],
                             name='match_prob')
        Utils.assertSeriesEqual(subsampled_probs, expected, decimals=3)
        
        # ax = Charter.barchart_over_timepoints(
        #        subsampled_probs,
        #        xlabel='Time (s)',
        #        ylabel='Probability of CMTO',
        #        title='Probability of CMTO Over Time',
        #        round_times=1
        #        )
        #
        # ax.figure.show()
        # print("Place breakpoint in test_powermember_compute_probabilities to see chart.")

    #------------------------------------
    # test_add_truth_to_power_result
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_add_truth_to_power_result(self):

        pwr_res = self.pwr_res.copy()
        rec1_sel_tbl = RavenSelectionTable(self.sel_tbls[0])
        pwr_res.add_truth(rec1_sel_tbl)
        
        # There should be 97 timeframe entries with
        # call being True:
        self.assertEqual(pwr_res.prob_df.truth.sum(), 97) 

    #------------------------------------
    # test_plot_pr_curve_power_result
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_plot_pr_curve_power_result(self):

        # SIMILAR_LENGTH: AP: 0.42
        # pwr_member = PowerMember('CMTOG', self.templates[0])
        
        # MED_PROB: 0.42
        # pwr_member = PowerMember('CMTOG', 
        #                          self.templates[0])

        # MAX_PROB: 0.42
        # pwr_member = PowerMember('CMTOG', 
        #                          self.templates[0])

        # MIN_PROB: 0.42
        pwr_member = PowerMember('CMTOG', self.templates['CMTOG'])
        pwr_res    = self.pwr_res.copy()
        sel_tbl    = RavenSelectionTable(self.sel_tbls[0])
        pwr_res.add_truth(sel_tbl)
        
        pwr_member.plot_pr_curve(pwr_res)
        
        print("Place breakpoint in test_plot_pr_curve_power_result to see chart.")
        
    #------------------------------------
    # test_autocorrelation
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_autocorrelation(self):
        
        # Very simple series:
        s = pd.Series([1,2,3])
        rho, p_value = SignalAnalyzer.autocorrelation(s, nlags=0)
        self.assertEqual(round(rho, 2), 1.0)
        self.assertEqual(round(p_value, 2), 0.0)
        
        # Sine wave:
        rads_in_circ = np.arange(0,2*np.pi, np.pi/180)
        sin_s = pd.Series(np.sin(rads_in_circ))

        # Sign wave lag of 0 should be corr of 1, with p_value of 0.0
        rho, p_value = SignalAnalyzer.autocorrelation(sin_s, 0)
        self.assertTupleEqual((round(rho, 2), round(p_value, 2)), (1, 0.0))
        
        # Shifting sin by 90deg should give a perfect
        # correlation:
        lag = int(np.pi/2)
        rho, p_value = SignalAnalyzer.autocorrelation(sin_s, lag)
        self.assertTupleEqual((round(rho, 2), round(p_value,2)), (1.0, 0.0))
        
        # Test autocorrelation for all lags:
        df = SignalAnalyzer.autocorrelation(sin_s, [0, int(np.pi/2)])
        expected = pd.DataFrame([[1.0, 0.0],
                                 [1.0, 0.0]],
                                 columns=['rho', 'p_value'],
                                 index=[0, int(np.pi/2)]
                                 )
        Utils.df_eq(df, expected, decimals=2)
        
        # Test lag given as Python range:
        df = SignalAnalyzer.autocorrelation(sin_s, range(1,3))
        expected = pd.DataFrame([[1.0, 0.0],
                                 [1.0, 0.0]],
                                 columns=['rho', 'p_value'],
                                 index=[1, 2]
                                 )
        Utils.df_eq(df, expected, decimals=2)
        
        # Test lag given as np array:
        df = SignalAnalyzer.autocorrelation(sin_s, np.array([1,2]))
        expected = pd.DataFrame([[1.0, 0.0],
                                 [1.0, 0.0]],
                                 columns=['rho', 'p_value'],
                                 index=[1, 2]
                                 )
        Utils.df_eq(df, expected, decimals=2)
        
        with self.assertRaises(TypeError):
            SignalAnalyzer.autocorrelation(sin_s, 1.24)
        with self.assertRaises(TypeError):
            SignalAnalyzer.autocorrelation(sin_s, np.ndarray([1,2,3]))

# -------------- Utilities -------------

    #------------------------------------
    # prep_one_sig_result
    #-------------------
    
    def prep_one_sig_result(self):
        '''
        Return the PowerResult that matches the CMTOG
        recording to the first of its 11 signatures
        
        :return PowerResult of sig_id 1.0
        :trype PowerResult
        '''

        # Get the 11-call template and surgically
        # remove all but the first call from it.
        # (Not something to do for regular use):
        onecall_tmplt = copy.deepcopy(self.templates['CMTOG'])
        onecall_tmplt.signatures = [onecall_tmplt.signatures[0]]
        # Force re-calc of mean sig:
        onecall_tmplt.cached_mean_sig = None

        pwr_member = PowerMember('CMTOG', onecall_tmplt)

        rec1_arr, rec1_sr = SoundProcessor.load_audio(self.sel_rec_cmto_xc1)

        power_res = pwr_member.compute_probabilities(rec1_arr, sr=rec1_sr)
        return power_res
    
    
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