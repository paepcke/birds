'''
Created on Mar 16, 2021

@author: paepcke
'''
import os
import unittest

import torch
import pandas as pd
import numpy as np

from birdsong.run_inference import Inferencer


#*********TEST_ALL = True
TEST_ALL = False


class InferenceTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(__file__)
        
        # Training data: root of species subdirectories:
        #*****cls.samples_path = os.path.join(cls.curr_dir, 'data_other/TestSnippets')
        cls.samples_path = os.path.join(cls.curr_dir, 'data/birds')
        
        # Where to put the raw prediction/label results
        # as .csv:
        cls.runs_raw_results = os.path.join(
            cls.curr_dir,
            'runs_raw_results'
            )

        # Dir of models to test:
        cls.saved_model_dir = os.path.join(
            cls.curr_dir,'models')

        # Assume there is only one model in
        # saved_model_path; a bit of an unsafe
        # assumption...but can get pickier if
        # more than one model will be involved
        # in inference testing:
        
        cls.saved_model_path = os.path.join(
            cls.saved_model_dir,
            'mod_2021-07-08T13_20_58_net_resnet18_pre_True_frz_0_lr_0.01_opt_Adam_bs_2_ks_7_folds_2_gray_True_classes_2_ep1.pth'
            )

        cls.preds_path = os.path.join(
            cls.runs_raw_results,
            'preds_inference.csv'
            )
        cls.labels_path = os.path.join(
            cls.runs_raw_results,
            'labels_inference.csv'
            )

        cls.num_samples = 0
        cls.num_species = 0
        for _dirName, _subdirList, fileList in os.walk(cls.samples_path):
            if len(fileList) > 0:
                cls.num_species += 1
            cls.num_samples += len(fileList)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    #------------------------------------
    # test_inference
    #-------------------
    
    #********@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_inference(self):
        # We have 60 test images. So, 
        # with drop_last True, a batch size
        # of 64 returns nothing. Use 16
        # to get several batches:
        
        batch_size = 2
        
        inferencer = Inferencer(
            self.saved_model_path,
            self.samples_path,
            batch_size=batch_size,
            labels_path=self.labels_path
            )
        try:
            inferencer.prep_model_inference(self.saved_model_path)
            print('Running inference...')
            tally_coll = inferencer.run_inference()
            print('Done running inference.')
    
            # Should have num_samples // batch_size
            expected_num_tallies = self.num_samples // batch_size
            self.assertEqual(len(tally_coll), expected_num_tallies)
    
            self.assertEqual(list(tally_coll.keys()),
                             [(0, 'TESTING'), (1, 'TESTING'), (2, 'TESTING'), (3, 'TESTING'), (4, 'TESTING'), (5, 'TESTING')]
                             )
            tally0 = tally_coll[(0, 'TESTING')]
            self.assertEqual(tally0.batch_size, batch_size)
            self.assertEqual(tally0.conf_matrix.shape,
                             torch.Size([self.num_species, self.num_species])
                             )
        finally:
            inferencer.close()

    #------------------------------------
    # test__report_charted_results
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test__report_charted_results(self):
        inferencer = Inferencer(
            self.saved_model_path,
            self.samples_path,
            batch_size=2,
            labels_path=self.labels_path
            )
        try:
            inferencer.prep_model_inference(self.saved_model_path)
            print('Running inference...')
            tally_coll = inferencer.run_inference()
            print('Done running inference.')
    
            inferencer._report_charted_results(
                thresholds=[0.2, 0.4, 0.6, 0.8, 1.0])
        finally:
            inferencer.close()

    #------------------------------------
    # test_res_measures_to_df 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_res_measures_to_df(self):
        
        inferencer = Inferencer(
            self.saved_model_path,
            self.samples_path,
            batch_size=2,
            labels_path=self.labels_path
            )
        try:
            inferencer.prep_model_inference(self.saved_model_path)
            series = pd.Series([10,20,np.array([100,200]),30,np.array([1000,2000])],
                               index=['meas1', 'meas2', 'prec_by_class', 'meas3', 'recall_by_class']
                               )
            df = inferencer.res_measures_to_df(series)
            truth = pd.DataFrame([[10, 20, 100, 200, 30, 1000, 2000]],
                                 columns=['meas1', 'meas2', 'prec_DYSMEN_S', 
                                         'prec_HENLES_S', 'meas3', 'rec_DYSMEN_S', 
                                         'rec_HENLES_S']
                                 )
            self.assertTrue(all(df == truth))
        finally:
            inferencer.close()

# ---------------------- Main -------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
