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
        data_dir = os.path.join(cls.curr_dir, 'data_other')
        
        cls.runs_raw_results = os.path.join(
            data_dir,
            'runs_raw_results'
            )
        cls.runs_models = os.path.join(
            data_dir,
            'runs_models'
            )
        cls.saved_model_path = os.path.join(
            data_dir,
            'saved_model'
            )
        cls.saved_model = os.path.join(
            cls.saved_model_path,
            'mod_2021-03-16T18_45_47_net_resnet18_ini_0_lr_0.01_opt_SGD_bs_64_ks_7_folds_0_gray_False_classes_10_ep10.pth'
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
        
        inferencer = Inferencer(
            self.saved_model,
            self.samples_path,
            batch_size=16,
            labels_path=self.labels_path
            )
        print('Running inference...')
        tally_coll = inferencer.run_inference()
        print('Done running inference.')

        # Should have 60 // 16 == 3 tallies:
        self.assertEqual(len(tally_coll), 3)
        self.assertEqual(list(tally_coll.keys()),
                         [(0, 'TESTING'), (1, 'TESTING'), (2, 'TESTING')]
                         )
        tally0 = tally_coll[(0, 'TESTING')]
        self.assertEqual(tally0.batch_size, 16)

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
