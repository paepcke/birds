'''
Created on Dec 26, 2021

@author: paepcke
'''
import os
import tempfile
import unittest

from data_augmentation.utils import Utils
import pandas as pd
from powerflock.power_evaluation import PowerEvaluator, PowerExperiment
from powerflock.power_member import PowerResult
from powerflock.signatures import SpectralTemplate, Signature


#*****TEST_ALL = True
TEST_ALL = False


class PowerEvaluationTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        templates_file = os.path.join(cls.cur_dir, 'signatures.json')
        
        templates_dict = SpectralTemplate.json_load(templates_file)
        cls.cmtog_template = templates_dict['CMTOG']

    def setUp(self):
        pass


    def tearDown(self):
        pass

    # ---------------- Tests ------------

    #------------------------------------
    # test_experiment_subclass
    #-------------------
    
    #******@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_experiment_subclass(self):
        
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='pwr_exp_test') as exp_root:
            exp = PowerExperiment(exp_root)
            
            self.assertTrue(os.path.exists(exp.json_files_path))
            df = pd.DataFrame([[0.1, 10, 20, 22],
                               [0.2, 10, 23, 25]
                               ],
                              columns=['match_prob', 'sig_id', 'start_idx', 'stop_idx']
                              )
            pwr_res = PowerResult(df, 'CMTOG')
            prob_df = pwr_res.prob_df
    
            exp.save('my_pwr_res', pwr_res)
            
            self.assertTrue(os.path.exists(os.path.join(exp_root, 'json_files', 'my_pwr_res.json')))
            
            restored_pwr_res = exp.read('my_pwr_res', PowerResult)
            restored_prob_df = restored_pwr_res.prob_df
            Utils.assertDataframesEqual(restored_prob_df, prob_df)


    #------------------------------------
    # test_init 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_init(self):
        
        evaluator = PowerEvaluator(
            experiment_name='TestExp',
            species='CMTOG',
            actions=PowerEvaluator.Action.TEST,
            
            
            
            )



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()