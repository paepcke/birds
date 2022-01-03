'''
Created on Nov 9, 2021

@author: paepcke
'''
import copy
import os
import unittest

from data_augmentation.sound_processor import SoundProcessor
import pandas as pd
from powerflock.power_member import PowerMember, PowerResult
from powerflock.signal_analysis import SignalAnalyzer
from powerflock.signatures import SpectralTemplate


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
        
        templates_dict = SpectralTemplate.from_json_file(templates_file)
        cls.cmtog_template = templates_dict['CMTOG']

        
    def setUp(self):
        pass


    def tearDown(self):
        pass

# ----------------- Tests -------------

    #------------------------------------
    # test_init 
    #-------------------
    
    #*****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
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

    #------------------------------------
    # test_calibrate_probabilities
    #-------------------
    
    # @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    # def test_calibrate_probabilities(self):
    #
    #     pm = PowerMember(
    #         species_name='CMTOG', 
    #         spectral_template_info=zip([self.rec_XC], [self.sel_tbl_XC]),
    #         apply_bandpass=True
    #         )
    #
    #     pm.calibrate_probabilities(self.rec_XC, self.sel_tbl_XC, apply_bandpass=True, plot_pr=True)
    #
    #     print('foo')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()