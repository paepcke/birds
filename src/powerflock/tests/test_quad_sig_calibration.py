'''
Created on Dec 20, 2021

@author: paepcke
'''
from datetime import timedelta
import os
import time
import timeit
import unittest

from powerflock.quad_sig_calibration import QuadSigCalibrator
from result_analysis.charting import Charter


TEST_ALL = True
#TEST_ALL = False

class QuadSigCalibrationTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.cmto_cal_calls = os.path.join(cls.cur_dir, '../species_calibration_data/CMTOG/cmto1.wav')
        cls.cmto_sel_tbs = os.path.join(cls.cur_dir, '../species_calibration_data/CMTOG/sel_tbl_Kelley_SONG_XC274155-429_CMTO_KEL.txt')

    # ------------------------- Tests ----------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_calibrate(self):
        
        calibrator = QuadSigCalibrator(unittesting=True)
        start_time = time.time()
        scales = calibrator.calibrate_species(self.cmto_cal_calls, self.cmto_sel_tbs)
        end_time = time.time()
        
        print(timedelta(seconds=end_time - start_time))

        scales_rounded = {key : round(val, 4) for key, val in scales.items()}
        
        expected = {'pitch': 0.0326,
                    'freq_mods': 0.5895,
                    'flatness': 0.0156,
                    'continuity': 3.3098
                    }

        self.assertDictEqual(scales_rounded, expected)

# ------------------------ Main ------------
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()