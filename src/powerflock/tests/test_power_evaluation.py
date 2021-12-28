'''
Created on Dec 26, 2021

@author: paepcke
'''
import os
import unittest

from powerflock.power_evaluation import PowerEvaluator
from powerflock.signatures import SpectralTemplate, Signature


TEST_ALL = True
#TEST_ALL = False


class PowerEvaluationTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        templates_file = os.path.join(cls.cur_dir, 'tests/data/signatures.json')
        
        templates_dict = SpectralTemplate.from_json_file(templates_file)
        cls.cmtog_template = templates_dict['CMTOG']

    def setUp(self):
        pass


    def tearDown(self):
        pass

    # ---------------- Tests ------------

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