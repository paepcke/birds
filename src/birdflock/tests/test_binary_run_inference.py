'''
Created on Sep 20, 2021

@author: paepcke
'''
import os
from pathlib import Path
import unittest

from experiment_manager.experiment_manager import ExperimentManager, Datatype

from birdflock.binary_run_inference import Inferencer


#*****8TEST_ALL = True
TEST_ALL = False


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.experiments_root = os.path.join(cls.cur_dir, 'data/experiments_root/')
        cls.experiment_BTSAC = os.path.join(cls.cur_dir, 'data/experiments_root/Classifier_BTSAC_2021-09-20T10_16_59')
        cls.experiment_GRHOG = os.path.join(cls.cur_dir, 'data/experiments_root/Classifier_GRHOG_2021-09-20T10_16_59')
        cls.samples_root     = os.path.join(cls.cur_dir, 'data/binary_dataset_img_samples')

    def setUp(self):
        pass

    def tearDown(self):
        pass

# -------------------------- Tests -------------

    #------------------------------------
    # test_construction 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_construction(self):
        
        inferencer = Inferencer(
            self.experiment_BTSAC,
            self.samples_root)
        
        train_exp = list(inferencer.train_exps.values())[0]
        test_exp = list(inferencer.test_exps.values())[0]
        
        inferencer.prep_model_inference(train_exp, test_exp)
        
        # Should have created an inference experiment
        # Check just the presence of the csv files 
        # subdir as a spotcheck:
        test_exp_path = self.experiment_BTSAC + '_inference'
        test_exp = ExperimentManager(test_exp_path)
        tables_dir = Path(test_exp.abspath('predictions', Datatype.tabular)).parent
        self.assertTrue(os.path.isdir(tables_dir))

    #------------------------------------
    # test_go
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_go(self):
        
        inferencer = Inferencer(
            self.experiment_BTSAC,
            self.samples_root,
            batch_size=1
            )
        
        res_coll = inferencer.go()
        
        # Expected: {
        # 'BTSAC': {(0, 'TESTING'): <ResultTally step 0 phase TESTING conf_matrix 2 x 2>, 
        #           (1, 'TESTING'): <ResultTally step 1 phase TESTING conf_matrix 2 x 2>, 
        #           (2, 'TESTING'): <ResultTally step 2 phase TESTING conf_matrix 2 x 2>
        #           }
        # } 
        for batch_key, tally in res_coll['BTSAC'].items():
            if batch_key[0] == 0:
                self.assertListEqual(tally.labels, [0])
                self.assertListEqual(tally.preds, [0])
                self.assertEqual(tally.probs[0,0].item(), 1.0)
                self.assertEqual(tally.accuracy, 1.0)
            elif batch_key[0] == 1:
                self.assertListEqual(tally.labels, [0])
                self.assertListEqual(tally.preds, [0])
                self.assertEqual(tally.probs[0,0].item(), 1.0)
                self.assertEqual(tally.accuracy, 1.0)
                
    #------------------------------------
    # test_parallelism
    #-------------------
    
    #**********@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_parallelism(self):
        
        inferencer = Inferencer(
            [self.experiment_BTSAC, self.experiment_GRHOG],
            self.samples_root,
            batch_size=1
            )
        res_coll = inferencer.go()
        
        print(res_coll)


# ----------------- Main ---------------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()