'''
Created on Mar 16, 2021

@author: paepcke
'''
import os
import shutil
from tempfile import TemporaryDirectory
import unittest

from experiment_manager.experiment_manager import ExperimentManager, Datatype
import torch

from birdsong.run_inference import Inferencer
import numpy as np
import pandas as pd
from result_analysis.charting import CurveSpecification


#********TEST_ALL = True
TEST_ALL = False


class InferenceTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        
        # Training data: root of species subdirectories:
        #*****cls.samples_path = os.path.join(cls.curr_dir, 'data_other/TestSnippets')
        cls.samples_path = os.path.join(cls.cur_dir, 'data/birds')
        
        # Create an ExperimentManager:
        exp_root = os.path.join(cls.cur_dir, 'training_exp_inf_test')
        cls.exp = ExperimentManager(exp_root)
        
        # And a place there inference results are kept
        # in an ExperimentManager:
        cls.testing_exp_path = os.path.join(cls.cur_dir, 'testing_exp_inf_test')

        # Populate the experiment with a test model, as if
        # that model had been saved in a training:
        test_model_path = os.path.join(cls.cur_dir, 'models/model_0.pth')
        shutil.copy(test_model_path, cls.exp.models_path)
        
        test_hparams_path = os.path.join(cls.cur_dir, 'hparams/hparams.json')
        shutil.copy(test_hparams_path, cls.exp.hparams_path)
        
        class_names = os.listdir(cls.samples_path)
        cls.exp['class_label_names'] = class_names 

        cls.num_samples = 0
        cls.num_species = 0
        for _dirName, _subdirList, fileList in os.walk(cls.samples_path):
            if len(fileList) > 0:
                cls.num_species += 1
            cls.num_samples += len(fileList)
            
        cls.exp.save()

    def setUp(self):
        self.tmp_dir = TemporaryDirectory(dir='/tmp', prefix='inference_testing_')
        self.tmp_dir_nm = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.exp.root)
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
            self.exp.root,
            self.tmp_dir_nm,
            self.samples_path,
            'model_0',
            batch_size=batch_size,
            save_logits=True
            )
        try:
            inferencer.model_key = 'model_0'
            inferencer.prep_model_inference('model_0', unknown_species_classes=None)
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
    # test_parallelism
    #-------------------
    
    #***********@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_parallelism(self):
        # We have 60 test images. So, 
        # with drop_last True, a batch size
        # of 64 returns nothing. Use 16
        # to get several batches:
        
        batch_size = 2
        
        inferencer = Inferencer(
            self.exp.root,
            self.tmp_dir_nm,
            self.samples_path,
            ['model_0', 'model_0', 'model_0', 'model_0'],
            batch_size=batch_size,
            save_logits=True
            )
        try:
            inferencer.go()
        finally:
            inferencer.close()


    #------------------------------------
    # test_report_results
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_report_results(self):
        inferencer = Inferencer(
            self.exp.root,
            self.tmp_dir_nm,
            self.samples_path,
            'model_0',
            save_logits=True,
            batch_size=2
            )
        # Under normal circumstances, inferencer.model_path
        # will pt to one model to inference with:
        inferencer.model_key = 'model_0'
        try:
            inferencer.prep_model_inference('model_0', unknown_species_classes=None)
            print('Running inference...')
            _tally_coll = inferencer.run_inference()
            print('Done running inference.')
    
            # Performance per class
            df = inferencer.testing_exp.read('performance_per_class', Datatype.tabular)
            expected = pd.DataFrame([['PLANS', 0.5, 1.0, 0.666667, 6.0],
                                     ['WBWWS', 0.0, 0.0, 0.000000, 6.0]
                                    ],
                                     columns=['species', 'precision', 'recall', 'f1-score', 'support']
                                    )
            self.assertDataframesEqual(df, expected, rounding=2)

            # IR results:
            df = inferencer.testing_exp.read('ir_results', Datatype.tabular)
            expected = pd.DataFrame([[0.250000, 0.500000, 0.250000, 
                                     0.500000, 0.500000, 0.500000, 
                                     0.333333, 0.500000, 0.333333, 
                                     0.500000, 0.500000, 0.500000, 
                                     2.000000, 2.000000]],
                                     columns=['prec_macro', 'prec_micro', 'prec_weighted', 
                                              'recall_macro','recall_micro', 'recall_weighted', 
                                              'f1_macro', 'f1_micro', 'f1_weighted',
                                              'accuracy', 'balanced_accuracy', 'mAP',
                                              'well_defined_APs', 'num_classes_total'
                                              ]
                                    )
            self.assertDataframesEqual(df, expected, rounding=2)
            


            # Get the (initially empty) inference predictions table:
            df = inferencer.testing_exp.read('predictions', Datatype.tabular)
            expected = pd.DataFrame([
                                    [0,0],
                                    [0,1],
                                    [0,0],
                                    [0,0],
                                    [0,0],
                                    [0,1],
                                    [0,0],
                                    [0,1],
                                    [0,1],
                                    [0,0],
                                    [0,1],
                                    [0,1]
                                    ], columns=['prediction', 'truth'])
            
            self.assertDataframesEqual(df, expected)
            
            try:
                # Figure is saved as pdf file, so cannot read back:
                inferencer.testing_exp.read('pr_curve', Datatype.figure)
            except TypeError:
                # Good: could not read PDF file
                pass

            df = inferencer.testing_exp.read('logits', Datatype.tabular)
            expected = pd.DataFrame(
             [
                [552470.500000,-577170.75000,0],
                [429042.062500,-455358.06250,1],
                [519468.531250,-546999.25000,0],
                [339625.406250,-358724.50000,0],
                [374372.031250,-398033.21875,0],
                [829133.375000,-842167.25000,1],
                [509088.156250,-534293.37500,0],
                [904704.812500,-916578.06250,1],
                [753957.500000,-762811.75000,1],
                [632876.125000,-651891.37500,0],
                [286517.781250,-306404.31250,1],
                [134765.640625,-168751.28125,1]
              ], columns=['PLANS','WBWWS','label']
            )
            self.assertDataframesEqual(df, expected)
            
            df = inferencer.testing_exp.read('probabilities', Datatype.tabular)
            expected = pd.DataFrame(
               [
                [1.0,0.0,0],
                [1.0,0.0,1],
                [1.0,0.0,0],
                [1.0,0.0,0],
                [1.0,0.0,0],
                [1.0,0.0,1],
                [1.0,0.0,0],
                [1.0,0.0,1],
                [1.0,0.0,1],
                [1.0,0.0,0],
                [1.0,0.0,1],
                [1.0,0.0,1]
                ], columns=['PLANS','WBWWS','label']
                )
            self.assertDataframesEqual(df, expected)

        finally:
            inferencer.close()

    #------------------------------------
    # test_res_measures_to_df 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_res_measures_to_df(self):
        
        inferencer = Inferencer(
            self.exp.root,
            self.tmp_dir_nm,
            self.samples_path,
            'model_0',
            batch_size=2,
            )
        try:
            with self.assertRaises(ValueError):
                # Expected:
                # The default unknown_species_classes kwarg
                # to the prep_model_inference specified that
                # in case of class labels unknown to the model
                # either OTHRG or NOISG should be chosen.
                # But neither are in model_0.
                inferencer.prep_model_inference('model_0')
                
            # Do it again, specifying not default placement:
            inferencer.prep_model_inference('model_0', unknown_species_classes=None)
                
            series = pd.Series([10,20,np.array([100,200]),30,np.array([1000,2000])],
                               index=['meas1', 'meas2', 'prec_by_class', 'meas3', 'recall_by_class']
                               )
            df = inferencer.res_measures_to_df(series)
            truth = pd.DataFrame([[10, 20, 100, 200, 30, 1000, 2000]],
                                 columns=['meas1', 'meas2', 'prec_PLANS', 
                                         'prec_WBWWS', 'meas3', 'rec_PLANS', 
                                         'rec_WBWWS']
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
            self.exp.root,
            self.tmp_dir_nm,
            self.samples_path,
            'model_0',
            batch_size=16,
            unittesting=True
            )
       
        # Corner case: only one spec:
        crv_specs = self.make_crv_specs(1)
        crvs = inferencer.pick_pr_curve_classes(crv_specs)
        self.assertListEqual(crvs, crv_specs)

        # Four curves with bop['f1'] from 0.0 to 0.3
        crv_specs = self.make_crv_specs(4)
        min_best_f1_crv, med_best_f1_crv, max_best_f1_crv = inferencer.pick_pr_curve_classes(crv_specs)
        self.assertTrue(min_best_f1_crv == crv_specs[0])
        self.assertTrue(med_best_f1_crv == crv_specs[1])
        self.assertTrue(max_best_f1_crv == crv_specs[3])
        
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
        for i, true_crv in enumerate([crv_specs[0], crv_specs[2], crv_specs[3]]):
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
            self.tmp_dir_nm,
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
        
        xlate_dict, unknown_assignment = inferencer._build_class_id_xlation(
                        sample_class_names, 
                        model_class_names,
                        unknown_species_classes=None
                        )
        expected = {0 : 0,
                    1 : 2,
                    } 
        self.assertDictEqual(xlate_dict, expected)
        self.assertTrue(len(unknown_assignment) == 0)
        
        # Test samples having a class that model was
        # not trained for:
        model_class_names  = ['foo1', 'foo2', 'NOIS']

        # sample_class_names: ['foo1', 'foo3']
        xlate_dict, unknown_assignment = inferencer._build_class_id_xlation(
                        sample_class_names, 
                        model_class_names,
                        unknown_species_classes='NOIS'
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
            xlate_dict, unknown_assignment = inferencer._build_class_id_xlation(
                        sample_class_names, 
                        model_class_names
                        )
            self.fail("Should have received ValueError b/c NOIS not in model space")
        except ValueError:
            # Great, got the expected exception
            pass

        
        xlate_dict, unknown_assignment = inferencer._build_class_id_xlation(
                        sample_class_names, 
                        model_class_names,
                        unknown_species_classes=['ABCD']
                        )
        expected = {0 : 1,
                    1 : 0,
                    } 
        self.assertDictEqual(xlate_dict, expected)
        self.assertTrue(unknown_assignment == ['foo3'])


# -------------------- Utilities --------------

    #------------------------------------
    # assert_dataframes_equal
    #-------------------
    
    def assertDataframesEqual(self, df1, df2, rounding=None):
        self.assertTrue((df1.columns == df2.columns).all())
        self.assertTrue((df1.index   == df2.index).all())
        if rounding is None:
            self.assertTrue((df1 == df2).all().all())
        else:
            self.assertTrue((df1.round(rounding) == df2.round(rounding)).all().all())

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
        
        df = pd.DataFrame([[0.1, 0.2, 0.3, 0.1], [0.4, 0.5, 0.6, np.nan]],
                          columns=['Precision', 'Recall', 'f1', 'Threshold']
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
                                         0.1, # AP
                                         i # class_id
                                         )
            crv_specs.append(crv_spec)
        return crv_specs

# ---------------------- Main -------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
