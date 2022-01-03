'''
Created on Dec 20, 2021

@author: paepcke
'''
from datetime import timedelta
import os
from pathlib import Path
import time
import unittest

from experiment_manager.experiment_manager import ExperimentManager

from data_augmentation.utils import Utils
import pandas as pd
from powerflock.quad_sig_calibration import QuadSigCalibrator
#from result_analysis.charting import Charter


TEST_ALL = True
#TEST_ALL = False

class QuadSigCalibrationTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.cmto_cal_calls = os.path.join(cls.cur_dir, 
                                          'species_calibration_data/CMTOG/cmto1.wav')
        cls.cmto_sel_tbs = os.path.join(cls.cur_dir, 
                                        'species_calibration_data/CMTOG/sel_tbl_Kelley_SONG_XC274155-429_CMTO_KEL.txt')
        cls.experiment_root = os.path.join(cls.cur_dir, 'quad_sig_calibrator_experiment')

    # ------------------------- Tests ----------------------

    #****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_calibrate(self):
        
        data_root = Path(self.cmto_cal_calls).parent.parent
        experiment = ExperimentManager(self.experiment_root)
        calibrator = QuadSigCalibrator(species='CMTOG', 
                                       cal_data_root=data_root,
                                       experiment=experiment)

        start_time = time.time()
        templates = calibrator.calibrate_species()
        end_time = time.time()
        
        print(timedelta(seconds=end_time - start_time))

        # Get first signature of the CMTOG template:
        sig1 = templates['CMTOG'][0]
        scale_factors = sig1.scale_info
        flatness    = round(scale_factors['flatness']['standard_measure'], 4)
        continuity  = round(scale_factors['continuity']['standard_measure'], 4)
        pitch       = round(scale_factors['pitch']['standard_measure'], 4)
        freq_mod    = round(scale_factors['freq_mod']['standard_measure'], 4)
        
        measures_rounded = pd.Series([flatness, continuity, pitch, freq_mod])
        expected = pd.Series([0.0156, 3.3098, 2.5335, 0.5895])

        Utils.assertSeriesEqual(measures_rounded, expected)

        # Read the templates back from disk:
        
        sigs_fname = calibrator.signatures_fname
        loaded_templates = calibrator.json_load(sigs_fname)
        sig2 = loaded_templates['CMTOG'][0]
        
        self.assertTrue(sig2 == sig1)
        
        # Recover again, this time from the experiment:
        templates = experiment.read('signatures', QuadSigCalibrator) 
        sig3 = templates['CMTOG'][0]
        self.assertTrue(sig3 == sig1)


# ------------------------ Main ------------
    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()