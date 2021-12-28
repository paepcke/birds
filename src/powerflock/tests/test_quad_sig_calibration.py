'''
Created on Dec 20, 2021

@author: paepcke
'''
from datetime import timedelta
import os
from pathlib import Path
import time
import unittest

from data_augmentation.utils import Utils
import pandas as pd
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
        templates = calibrator.calibrate_species()
        end_time = time.time()
        
        print(timedelta(seconds=end_time - start_time))

        sig1 = templates['CMTOG'][0]
        scale_factors = sig1.scale_factors
        scales_rounded = scale_factors.round(4)
        
        expected_scales = pd.Series({'flatness': 0.0156,
                                    'continuity': 3.2887,
                                    'pitch': 0.0373,
                                    'freq_mod': 0.5824
                                    }, name='scale_factors')

        Utils.assertSeriesEqual(scales_rounded, expected_scales)

        # Read the templates back from disk:
        
        sigs_fname = calibrator.signatures_fname
        loaded_templates = calibrator.sigs_json_load(sigs_fname)
        sig2 = loaded_templates['CMTOG'][0]
        
        self.assertTrue(sig2 == sig1)

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