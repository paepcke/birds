'''
Created on Sep 20, 2021

@author: paepcke
'''
import os
import unittest

from birdflock.inference_eval_summary import BinaryInferenceEvaluator


TEST_ALL = True
#TEST_ALL = False


class TestBinaryInfEvaluator(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.inf_exp_root = os.path.join(cls.cur_dir, 'data/inference_results')

    def setUp(self):
        pass


    def tearDown(self):
        pass


# ---------------- Tests ----------------

    #------------------------------------
    # test_bal_acc_fig
    #-------------------

    def test_bal_acc_fig(self):
        
        evaluator = BinaryInferenceEvaluator('2021-09-18T15_00_16', 
                                             experiments_root=self.inf_exp_root)
        
        print('FOO')
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()