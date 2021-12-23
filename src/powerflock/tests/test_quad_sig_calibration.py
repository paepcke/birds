'''
Created on Dec 20, 2021

@author: paepcke
'''
from datetime import timedelta
import os
from pathlib import Path
import time
import unittest

from powerflock.quad_sig_calibration import QuadSigCalibrator
from result_analysis.charting import Charter


#******TEST_ALL = True
TEST_ALL = False

class QuadSigCalibrationTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.cmto_cal_calls = os.path.join(cls.cur_dir, 
                                          'species_calibration_data/CMTOG/cmto1.wav')
        cls.cmto_sel_tbs = os.path.join(cls.cur_dir, 
                                        'species_calibration_data/CMTOG/sel_tbl_Kelley_SONG_XC274155-429_CMTO_KEL.txt')

    # ------------------------- Tests ----------------------

    #****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_calibrate(self):
        
        data_root = Path(self.cmto_cal_calls).parent.parent
        calibrator = QuadSigCalibrator(species='CMTOG', 
                                       cal_data_root=data_root)

        start_time = time.time()
        sigs_dict = calibrator.calibrate_species()
        end_time = time.time()
        
        print(timedelta(seconds=end_time - start_time))

        scales_rounded = {key : round(val, 4) 
                          for key, val 
                          in sigs_dict['CMTOG'].items()
                          if key in ['pitch_scale', 
                                     'freq_mods_scale', 
                                     'flatness_scale', 
                                     'continuity_scale']
                          }
        
        expected_keys = [
            'species',
            'pitch',
            'freq_mods',
            'flatness',
            'continuity',
            'pitch_scale',
            'freq_mods_scale',
            'flatness_scale',
            'continuity_scale',
            ]
        self.assertListEqual(list(sigs_dict['CMTOG'].keys()), expected_keys)
        
        expected_scales = {'pitch_scale': 0.0373,
                           'freq_mods_scale': 0.5824,
                           'flatness_scale': 0.0156,
                           'continuity_scale': 3.2887
                           }
        
        

        self.assertDictEqual(scales_rounded, expected_scales)

        # Read the sigs back from disk
        sigs_fname = calibrator.signatures_fname
        loaded_sigs = calibrator.sigs_json_load(sigs_fname)
        
        loaded_sigs_rounded = {key : round(val, 4) 
                               for key, val 
                               in sigs_dict['CMTOG'].items()
                               if key in ['pitch_scale', 
                                          'freq_mods_scale', 
                                          'flatness_scale', 
                                          'continuity_scale']
                               }
        loaded_sigs_keys = list(loaded_sigs['CMTOG'].keys())
        self.assertListEqual(loaded_sigs_keys, expected_keys)
        self.assertDictEqual(loaded_sigs_rounded, expected_scales)
        

    #------------------------------------
    # test_save_sig
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_save_sig(self):
        calibrator = QuadSigCalibrator(species='CMTOG', cal_outdir='/tmp/')
        
        

# ------------------------ Main ------------
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()