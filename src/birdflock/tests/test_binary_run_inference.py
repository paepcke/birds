'''
Created on Sep 20, 2021

@author: paepcke
'''
import json
import os
from pathlib import Path
import shutil
import unittest

from experiment_manager.experiment_manager import ExperimentManager, Datatype

from birdflock.binary_run_inference import Inferencer


#*******TEST_ALL = True
TEST_ALL = False


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.experiments_root = os.path.join(cls.cur_dir, 'data/experiments_root/')
        cls.experiment_BTSAC = os.path.join(cls.cur_dir, 'data/experiments_root/Classifier_BTSAC_2021-09-20T10_16_59')
        cls.experiment_GRHOG = os.path.join(cls.cur_dir, 'data/experiments_root/Classifier_GRHOG_2021-09-20T10_16_59')
        cls.experiment_BTSAC_inf = cls.experiment_BTSAC + '_inference'
        cls.experiment_GRHOG_inf = cls.experiment_GRHOG + '_inference'
        cls.samples_root     = os.path.join(cls.cur_dir, 'data/binary_dataset_img_samples')

    def setUp(self):
        shutil.rmtree(self.experiment_BTSAC_inf, ignore_errors=True)
        shutil.rmtree(self.experiment_GRHOG_inf, ignore_errors=True)
        
    def tearDown(self):
        shutil.rmtree(self.experiment_BTSAC_inf, ignore_errors=True)
        shutil.rmtree(self.experiment_GRHOG_inf, ignore_errors=True)

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
                self.assertListEqual(tally.preds, [[0]])
                self.assertEqual(round(tally.probs.item(),3), 0.139)
                self.assertEqual(tally.accuracy, 1.0)
            elif batch_key[0] == 1:
                self.assertListEqual(tally.labels, [0])
                self.assertListEqual(tally.preds, [[0]])
                self.assertEqual(round(tally.probs.item(),3), 0.133)
                self.assertEqual(tally.accuracy, 1.0)
                
    #------------------------------------
    # test_parallelism
    #-------------------
    
    #*****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_parallelism(self):
        
        inferencer = Inferencer(
            [self.experiment_BTSAC, self.experiment_GRHOG],
            self.samples_root,
            batch_size=1
            )
        _res_coll = inferencer.go()
        
        self.check_inference_files('BTSAC')
        self.check_inference_files('GRHOG')

# ------------------ Utilities --------------

    def check_inference_files(self, species):
        
        if species == 'BTSAC':
            exp_root = self.experiment_BTSAC_inf
        elif species == 'GRHOG':
            exp_root = self.experiment_GRHOG_inf
        
        expected_csv_files = ['accuracy_mAP.csv',
                              'conf_matrix.csv',
                              'ir_results.csv',
                              'performance_per_class.csv',
                              'predictions.csv',
                              'probabilities.csv'
                              ]
        csv_files_path = os.path.join(exp_root, 'csv_files')
        csv_files_created = os.listdir(csv_files_path)
        self.assertEqual(len(csv_files_created),
                         len(expected_csv_files))
        
        for csv_file in csv_files_created:
            self.assertTrue(csv_file in expected_csv_files)
            
        
        figs_files_path = os.path.join(exp_root, 'figs')
        fig_files_created = os.listdir(figs_files_path)
        fig_files_expected = ['conf_matrix.pdf']
        self.assertListEqual(fig_files_created, fig_files_expected)
        
        # Hparams:
        hparams_path = os.path.join(exp_root, 'hparams/hparams.json')
        self.assertTrue(os.path.exists(hparams_path))

# ----------------- Main ---------------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()