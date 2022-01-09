'''
Created on Nov 9, 2021

@author: paepcke
'''
import copy
import os
import unittest

from experiment_manager.experiment_manager import ExperimentManager

from data_augmentation.sound_processor import SoundProcessor
from data_augmentation.utils import Utils
import pandas as pd
from powerflock.power_member import PowerMember, PowerResult, CallLevelScore
from powerflock.signal_analysis import SignalAnalyzer
from powerflock.signatures import SpectralTemplate, Signature


#*******TEST_ALL = True
TEST_ALL = False

class TestPowerMember(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        
        cls.rec_XC = os.path.join(cls.cur_dir,
                                  'signal_processing_sounds/XenoCanto/CMTOG/SONG_XC274155-429_Chestnut-mandibled_Toucan_2_song.mp3')
        cls.rec_fld = os.path.join(cls.cur_dir,
                                  'signal_processing_sounds/Field/CMTOG/DS_AM03_20190713_055956.wav')

        
        cls.sel_tbl_XC = os.path.join(cls.cur_dir, 
                                      'selection_tables/XenoCanto/sel_tbl_Kelley_SONG_XC274155-429_CMTO_KEL.txt')
        cls.sel_tbl_fld = os.path.join(cls.cur_dir, 
                                       'selection_tables/JZ_DS_AM03_20190713_055956.Table.1.selections.txt')

        templates_file = os.path.join(cls.cur_dir, '../species_calibration_data/signatures.json')
        
        templates_dict = SpectralTemplate.json_load_templates(templates_file)
        cls.cmtog_template = templates_dict['CMTOG']
        
        cls.exp_root = os.path.join(cls.cur_dir, 'data/experiments_root/CMTOG_Quintus')
        cls.pwr_res_key = 'PwrRes_2022-01-07T09_53_18' 

        
    def setUp(self):
        pass


    def tearDown(self):
        pass

# ----------------- Tests -------------

    #------------------------------------
    # test_init 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_init(self):
        
        # prob_df = pd.read_csv('/tmp/powertest.csv')
        # pwr_res = PowerResult(prob_df, 'CMTOG')
        # pwr_res.json_dump('/tmp/power_result.json')
        #
        # pwr_res1 = PowerResult.json_load('/tmp/power_result.json')
        
        pmember = PowerMember(
            species_name='CMTOG',
            spectral_template_info=self.cmtog_template,
            apply_bandpass=True
            )
        power_result = pmember.compute_probabilities(self.rec_XC, '/tmp/power_result_test.json')
        power_result.add_truth(self.sel_tbl_XC)
        print(pmember)

    #------------------------------------
    # test_power_result_examination
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_power_result_examination(self):
        pwr_res_file = os.path.join(os.getenv('HOME'),
                                    'tmp/cmtog_match_with_templateDec31_2021.json')
        pwr_res = PowerResult.json_load(pwr_res_file)
        pwr_res.add_truth(self.sel_tbl_XC)
        
        
        print(pwr_res)


    #------------------------------------
    # test_matching_one_sig
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_matching_one_sig(self):

        onecall_tmplt = copy.deepcopy(self.templates[0])
        onecall_tmplt.signatures = [onecall_tmplt[0]]
        # Force re-calc of mean sig:
        onecall_tmplt.cached_mean_sig = None
        sig1 = onecall_tmplt[0]
        
        # Take the clip that underlies this
        # one sig, and get its clip signature:
        sig_times_walltime = sig1.as_walltime().index
        rec1_audio, _sr = SoundProcessor.load_audio(self.sel_rec_cmto_xc1)
        clip = SoundProcessor.extract_clip(rec1_audio,
                                           sig_times_walltime[0], 
                                           sig_times_walltime[-1])
        
        details_df, summary = SignalAnalyzer.match_probability(clip, onecall_tmplt)
        
        expected_res_df = pd.DataFrame([
           {
            'n_samples'   : 43136.0,
            'probability' : 1.000000,
            'sig_id'      : 1.0,
            'start'       : 0.0,
            'stop'        : 43136.0
            },
           {
            'n_samples'   : 43136.0,
            'probability' : 0.179206,
            'sig_id'      : 1.0,
            'start'       : 10784.0,
            'stop'        : 53920.0
            },
           {
            'n_samples'   : 43136.0,
            'probability' : 0.115142,
            'sig_id'      : 1.0,
            'start'       : 21568.0,
            'stop'        : 64704.0
            },
           {
            'n_samples'   : 43136.0,
            'probability' : 0.112187,
            'sig_id'      : 1.0,
            'start'       : 32352.0,
            'stop'        : 75488.0
            }])


        details_df['probability'] = details_df['probability'].round(6)
        self.assertTrue((details_df == expected_res_df).all().all())
        
        expected_summary = pd.Series({
                         'min_prob'      :  0.112187,
                         'max_prob'      :  1.000000,
                         'med_prob'      :  0.147174,
                         'best_fit_prob' :  1.000000
                         })

        summary = summary.round(6)
        
        self.assertTrue((summary == expected_summary).all())

    #------------------------------------
    # test_find_calls
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_find_calls(self):
        
        exp = ExperimentManager(self.exp_root)
        pwr_res = exp.read(self.pwr_res_key, PowerResult)
        pwr_member = PowerMember('CMTOG',
                                 experiment=exp,
                                 spectral_template_info=self.cmtog_template 
                                 )
        peaks = pwr_member.find_calls(pwr_res)
        
        # Expect like this:
        #            peak_prob  sig_id  low_bound  high_bound
        # time                                               
        # 2.095601    0.416110     1.0   1.433832    2.757370
        # 5.183855    0.415394     2.0   4.405986    5.961723
        # 7.024036    0.371405     3.0   6.623492    7.424580
        #                       ...
        
        # Expect as many peaks as the test template
        # has signatures
        
        num_sigs = len(self.cmtog_template)
        self.assertEqual(len(peaks), num_sigs)
        # Spot check first line:
        expected = [0.4161, 1.0, 1.4338, 2.7574]
        self.assertListEqual(peaks.iloc[0].round(4).to_list(), expected)
        self.assertEqual(peaks.index[0].round(4), 2.0956)
        

    #------------------------------------
    # test_call_level_score
    #-------------------
    
    #*****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_call_level_score(self):
        
        exp = ExperimentManager(self.exp_root)
        pwr_res = exp.read(self.pwr_res_key, PowerResult)
        pwr_member = PowerMember('CMTOG',
                                 experiment=exp,
                                 spectral_template_info=self.cmtog_template 
                                 )
        peaks = pwr_member.find_calls(pwr_res)
        
        scorer = CallLevelScore(peaks, self.sel_tbl_XC)
        
        score = scorer.score()
        expected = pd.Series({ 
            'bal_acc'   :   0.979183,
            'acc'       :   0.983672,
            'recall'    :   0.959762,
            'precision' :   0.997676,
            'f1'        :   0.978352,
            'f0.5'      :   0.989856
            }, name='score')
        Utils.assertSeriesEqual(score, expected)


    #------------------------------------
    # test_matching_multiple_sigs
    #-------------------
    
    #*********** REVISIT THIS
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_matching_multiple_sigs(self):

        template11 = copy.deepcopy(self.templates[0])
        rec1, _sr = SoundProcessor.load_audio(self.sel_rec_cmto_xc1)
        details_df, summary = SignalAnalyzer.match_probability(rec1, template11)
        pow_res = PowerResult(details_df, summary, 'CMTOG')
        pow_res.add_truth(self.sel_tbl_cmto_xc1)
        print('foo')



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()