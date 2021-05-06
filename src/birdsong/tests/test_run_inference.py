'''
Created on Mar 16, 2021

@author: paepcke
'''
import os
import unittest

import torch

from birdsong.run_inference import Inferencer


TEST_ALL = True
#TEST_ALL = False


class InferenceTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(InferenceTester, cls).setUpClass()
        
        cls.curr_dir = os.path.dirname(__file__)
        
        # Training data: root of species subdirectories:
        cls.samples_path = os.path.join(cls.curr_dir, 'data_other/TestSnippets')
        
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
        
        cls.saved_model = os.path.join(
            cls.saved_model_dir,
            os.listdir(cls.saved_model_dir)[0]
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
        cls.num_species = len(os.listdir(cls.samples_path))
        for _dirName, _subdirList, fileList in os.walk(cls.samples_path):
            cls.num_samples += len(fileList)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    #------------------------------------
    # test_inference
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_inference(self):
        # We have 60 test images. So, 
        # with drop_last True, a batch size
        # of 64 returns nothing. Use 16
        # to get several batches:
        
        batch_size = 16
        
        inferencer = Inferencer(
            self.saved_model,
            self.samples_path,
            batch_size=batch_size,
            labels_path=self.labels_path
            )
        print('Running inference...')
        tally_coll = inferencer.run_inference()
        print('Done running inference.')

        # Should have num_samples // batch_size
        expected_num_tallies = self.num_samples // batch_size
        self.assertEqual(len(tally_coll), expected_num_tallies)

        self.assertEqual(list(tally_coll.keys()),
                         [(0, 'TESTING')]
                         )
        tally0 = tally_coll[(0, 'TESTING')]
        self.assertEqual(tally0.batch_size, batch_size)
        self.assertEqual(tally0.conf_matrix.shape,
                         torch.Size([self.num_species, self.num_species])
                         )

    #------------------------------------
    # test__report_charted_results
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test__report_charted_results(self):
        inferencer = Inferencer(
            self.saved_model,
            self.samples_path,
            batch_size=16,
            labels_path=self.labels_path
            )
        print('Running inference...')
        tally_coll = inferencer.run_inference()
        print('Done running inference.')

        inferencer._report_charted_results(
            thresholds=[0.2, 0.4, 0.6, 0.8, 1.0])
        print('done')

# ---------------------- Main -------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
