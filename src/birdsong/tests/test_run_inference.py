'''
Created on Mar 16, 2021

@author: paepcke
'''
import os
import unittest

from birdsong.run_inference import Inferencer


TEST_ALL = True
#TEST_ALL = False


class InferenceTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(InferenceTester, cls).setUpClass()
        
        cls.curr_dir = os.path.dirname(__file__)
        data_dir = os.path.join(cls.curr_dir, 'data')
        
        cls.runs_raw_results = os.path.join(
            data_dir,
            'runs_raw_results'
            )
        cls.runs_models = os.path.join(
            data_dir,
            'runs_models'
            )


        cls.samples_path = os.path.join(
            data_dir,
            'inference_data'
            )
        cls.preds_path = os.path.join(
            cls.runs_raw_results,
            'pred_2021-03-11T10_59_02_net_resnet18_pretrain_0_lr_0.01_opt_SGD_bs_64_ks_7_folds_0_classes_10.csv'
            )
        cls.labels_path = os.path.join(
            cls.runs_raw_results,
            'labels_2021-03-11T10_59_02_net_resnet18_pretrain_0_lr_0.01_opt_SGD_bs_64_ks_7_folds_0_classes_10.csv'
            )
        cls.model_path = os.path.join(
            cls.runs_models,
            'mod_2021-03-11T11_39_13_net_resnet18_pretrain_0_ini_0_lr_0.01_opt_SGD_bs_64_ks_7_folds_0_classes_10_ep29.pth'
            )

    def setUp(self):
        pass


    def tearDown(self):
        pass

    #------------------------------------
    # test_inference
    #-------------------
    
    #******@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_inference(self):
        inferencer = Inferencer(
            self.model_path,
            self.samples_path,
            batch_size=None,
            labels_path=self.labels_path
            )
        inferencer.run_inference()
        
# ---------------------- Main -------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()