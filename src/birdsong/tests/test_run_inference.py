'''
Created on Mar 16, 2021

@author: paepcke
'''
import os
import unittest

import torch

from birdsong.run_inference import Inferencer
import numpy as np
import pandas as pd
from result_analysis.charting import CurveSpecification


TEST_ALL = True
#TEST_ALL = False


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
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
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

    #------------------------------------
    # test_pick_pr_curve_classes 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_pick_pr_curve_classes(self):
        
         
        # This instance won't do anything,
        # due to unittesting == True:
        inferencer = Inferencer(
            self.saved_model_path,
            self.samples_path,
            batch_size=16,
            labels_path=self.labels_path,
            unittesting=True
            )
       
        # Corner case: only one spec:
        crv_specs = self.make_crv_specs(1)
        crvs = inferencer.pick_pr_curve_classes(crv_specs)
        self.assertListEqual(crvs, crv_specs)

        # Four curves with bop['f1'] from 0.0 to 0.3
        crv_specs = self.make_crv_specs(4)
        crvs = inferencer.pick_pr_curve_classes(crv_specs)
        self.assertListEqual(crvs, [crv_specs[0], crv_specs[2], crv_specs[3]])
        
        # Four curves with bop['f1'] of 0.0, 0.0, 0.2, 0.2
        # i.e. median == max:
        crv_specs = self.make_crv_specs(4)
        crv_specs[1]['best_op_pt']['f1'] = 0.0
        crv_specs[2]['best_op_pt']['f1'] = 0.2
        crv_specs[3]['best_op_pt']['f1'] = 0.2
        crvs = inferencer.pick_pr_curve_classes(crv_specs)
        for i, true_crv in enumerate([crv_specs[0], crv_specs[1], crv_specs[3]]):
            self.assertEqual(crvs[i], true_crv)

        # Four curves with bop['f1'] of 0.0, 0.0, 0.0, 0.2
        crv_specs = self.make_crv_specs(4)
        crv_specs[1]['best_op_pt']['f1'] = 0.0
        crv_specs[2]['best_op_pt']['f1'] = 0.0
        crv_specs[3]['best_op_pt']['f1'] = 0.2
        crvs = inferencer.pick_pr_curve_classes(crv_specs)
        for i, true_crv in enumerate([crv_specs[0], crv_specs[3], crv_specs[3]]):
            self.assertEqual(crvs[i], true_crv)

    #------------------------------------
    # test_build_class_id_xlation
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_build_class_id_xlation(self):
        # This instance won't do anything,
        # due to unittesting == True:
        inferencer = Inferencer(
            None, # experiment root path
            self.samples_path,
            None, # model names
            batch_size=16,
            unittesting=True
            )
        # All class names for which samples exist to 
        # be inferenced. Note that 'foo3' has 
        # class I of 1 due to its position in the list:
        
        sample_class_names = ['foo1', 'foo3']
        
        # And the classes the model was trained on.
        # Here, 'foo3' is ID 2: 
        model_class_names  = ['foo1', 'foo2', 'foo3']
        
        xlate_dict, unknown_assignment = inferencer._build_class_id_xlation(sample_class_names, 
                                                                            model_class_names
                                                                            )
        expected = {0 : 0,
                    1 : 2,
                    } 
        self.assertDictEqual(xlate_dict, expected)
        self.assertTrue(len(unknown_assignment) == 0)
        
        # Test samples having a class that model was
        # not trained for:
        model_class_names  = ['foo1', 'foo2', 'NOIS']

        xlate_dict, unknown_assignment = inferencer._build_class_id_xlation(sample_class_names, 
                                                                            model_class_names,
                                                                            unknown_species_class='NOIS'
                                                                            )
        expected = {0 : 0,
                    1 : 2,
                    } 
        self.assertDictEqual(xlate_dict, expected)
        self.assertTrue(unknown_assignment == ['foo3'])

        # Move the 'unknown' bucket to a different position
        # in model space:
        
        model_class_names  = ['ABCD', 'foo1', 'foo2', 'foo4']

        # Model class names does not include the 
        # 'NOIS' class. Expect a value error:
        try:
            xlate_dict, unknown_assignment = inferencer._build_class_id_xlation(sample_class_names, 
                                                                                model_class_names
                                                                                )
            self.fail("Should have received ValueError b/c NOIS not in model space")
        except ValueError:
            # Great, got the expected exception
            pass

        
        xlate_dict, unknown_assignment = inferencer._build_class_id_xlation(sample_class_names, 
                                                                            model_class_names,
                                                                            unknown_species_class='ABCD'
                                                                            )
        expected = {0 : 1,
                    1 : 0,
                    } 
        self.assertDictEqual(xlate_dict, expected)
        self.assertTrue(unknown_assignment == ['foo3'])


# -------------------- Utilities --------------

    #------------------------------------
    # make_crv_specs 
    #-------------------
    
    def make_crv_specs(self, num):
        '''
        Makes rudimentary CurveSpecification
        instances. Only enough for testing.
        Don't take any of the numbers seriously.
        
        :param num: number of specs to create
        :type num: int
        :returns list of CurveSpecification
        :rtype [CurveSpecification]
        '''
        
        df = pd.DataFrame([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                          columns=['Precision', 'Recall', 'f1']
                          )
                          
        crv_specs = []
        for i in range(num):
            bop = pd.Series([
                         0.000853,
                         0.115044,
                         0.928571,
                         0.204724
                         ], 
                index = ['Threshold', 'Precision', 'Recall', 'f1']
                )
            # All curve specs will be the
            # same, except for the f1, which
            # will be rising:
            bop['f1'] = i/10
            
            crv_spec = CurveSpecification(df,
                                         [0.0, 0.1, 0.2], # Thresholds
                                         bop,
                                         0.1, # AP
                                         i # class_id
                                         )
            crv_specs.append(crv_spec)
        return crv_specs

# ---------------------- Main -------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
