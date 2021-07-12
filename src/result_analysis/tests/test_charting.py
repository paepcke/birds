'''
Created on May 7, 2021

@author: paepcke
'''
import os
import tempfile
import unittest

import torch

import numpy as np
import pandas as pd
from result_analysis.charting import Charter, CELL_LABELING


TEST_ALL = True
#TEST_ALL = False

class ChartingTester(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.cur_dir  = os.path.dirname(__file__)
        cls.data_dir = os.path.join(cls.cur_dir, 'data')
        cls.viz_data_dir = os.path.join(cls.data_dir, 'visualization')
        
        cls.conf_normed_matrix_path = os.path.join(cls.data_dir, 
                                                   'conf_matrix_results.csv')
        cls.conf_un_normed_matrix_path = os.path.join(cls.data_dir, 
                                                      'conf_matrix_unnormal.csv')

    def setUp(self):
        self.cm = Charter(None)
        
        self.cm_df_normed = pd.read_csv(self.conf_normed_matrix_path, index_col=0)
        self.cm_np_normed = self.cm_df_normed.to_numpy()
        self.cm_tn_normed = torch.tensor(self.cm_np_normed)
        
        self.cm_df_un_normed = pd.read_csv(self.conf_un_normed_matrix_path, index_col=0)
        
        self.class_names = self.cm_df_normed.columns
        
        # Read the numpy array with the correct 
        # heatmap data. The file was produced like this:
        #
        #     data_np_arr = my_figure.axes[0].get_children()[0].get_array()
        #     np.save('file_name', data_np_arr)
         
        self.cm_heat_data = np.load(os.path.join(self.data_dir, 'conf_matrix_img_array.npy'))
        self.cm_heat_with_inf_data = np.load(os.path.join(self.data_dir, 'conf_matrix_with_inf_img_array.npy'))
        
        self.num_species = len(self.cm_df_normed.index)
        

    def tearDown(self):
        pass

# -------------------- TESTS -----------

    #------------------------------------
    # test_calc_conf_matrix_norm 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_calc_conf_matrix_norm(self):

        # For each version of the same matrix:
        # dataframe, np.array, and tensor: ensure
        # that the sum of the sum of all rows is
        # the number of species:
         
        cm_df_normed = self.cm.calc_conf_matrix_norm(self.cm_df_normed)
        self.assertEqual(cm_df_normed.sum(axis=1).sum(),
                         self.num_species
                         )
        
        cm_np_normed = self.cm.calc_conf_matrix_norm(self.cm_np_normed)
        self.assertEqual(cm_np_normed.sum(axis=1).sum(),
                         self.num_species
                         )

        cm_tn_normed = self.cm.calc_conf_matrix_norm(self.cm_tn_normed)
        self.assertEqual(cm_tn_normed.sum(axis=1).sum(),
                         self.num_species
                         )

    #------------------------------------
    # test_fig_from_conf_matrix_no_infinities 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_fig_from_conf_matrix_no_infinities(self):

        # fig = self.cm.fig_from_conf_matrix(self.cm_df_un_normed,
        #                                    supertitle="Confusion Matrix\n",
        #                                    subtitle="Data NOT Normalized",
        #                                    write_in_fields=True
        #                                    )
        
        fig = self.cm.fig_from_conf_matrix(
            self.cm_df_normed,
            supertitle='Confusion Matrix',
            subtitle="Data normalized to percentages along rows (roughly adds to 100)",
            write_in_fields=CELL_LABELING.NEVER
            )
        self.assert_heatmaps_equal(fig, 
                                   self.cm_heat_data, 
                                   col_names=self.class_names)

    #------------------------------------
    # test_fig_from_conf_matrix_no_infinities_diag_labeled
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_fig_from_conf_matrix_no_infinities_diag_labeled(self):

        fig = self.cm.fig_from_conf_matrix(
            self.cm_df_normed,
            supertitle='Confusion Matrix',
            subtitle="Data normalized to percentages along rows (roughly adds to 100)",
            write_in_fields=CELL_LABELING.DIAGONAL
            )
        self.assert_heatmaps_equal(fig, 
                                   self.cm_heat_data, 
                                   col_names=self.class_names)

    #------------------------------------
    # test_fig_from_conf_matrix_with_infinities 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_fig_from_conf_matrix_with_infinities(self):

        # fig = self.cm.fig_from_conf_matrix(self.cm_df_un_normed,
        #                                    supertitle="Confusion Matrix\n",
        #                                    subtitle="Data NOT Normalized",
        #                                    write_in_fields=True
        #                                    )
        
        cm_with_inf = self.cm_df_normed.copy()
        cm_with_inf.iloc[0,0] = np.inf
        fig = self.cm.fig_from_conf_matrix(
            cm_with_inf,
            supertitle='Confusion Matrix',
            subtitle="Data normalized to percentages along rows (roughly adds to 100)",
            write_in_fields=CELL_LABELING.NEVER
            )
        self.assert_heatmaps_equal(fig, 
                                   self.cm_heat_with_inf_data,
                                   col_names=self.class_names)

    #------------------------------------
    # test_read_conf_matrix_from_file 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_read_conf_matrix_from_file(self):
        
        truth = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]],
                             columns=['foo','bar','fum'], 
                             index=['foo','bar','fum']
                             )
        nice_df = self.cm.read_conf_matrix_from_file(os.path.join(self.data_dir,
                                                                  'nice_df.csv'))
        notso_nice_df = self.cm.read_conf_matrix_from_file(os.path.join(self.data_dir,
                                                                        'notso_nice_df.csv'))
        self.assert_dataframes_equal(nice_df, truth)
        self.assert_dataframes_equal(notso_nice_df, truth)


    #------------------------------------
    # test_visualize_testing_result 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_visualize_testing_result(self):
        
        truth_labels = pd.read_csv(os.path.join(self.viz_data_dir, 'truth_labels_300_samples.csv'))
        # Get a series from the one-and-only column
        # (pd.read_csv() creates a df, not a series,
        # with which we will work:
        truth_labels = truth_labels.iloc[:,0]
        
        # Probabilities of 300 samples, one
        # column for each target class
        pred_probs   = pd.read_csv(os.path.join(self.viz_data_dir, 'pred_probs_300_samples.csv'))
        
        mAP, pr_curve_specs = Charter.visualize_testing_result(truth_labels, pred_probs)
        
        self.assertEqual(round(mAP, 3), 0.091)
        
        num_samples, num_classes = pred_probs.shape
        self.assertEqual(len(pr_curve_specs), num_classes)
        self.assertEqual(len(truth_labels), num_samples)
        
        avg_prec_class_0 = pr_curve_specs[0]['avg_prec']
        self.assertEqual(round(avg_prec_class_0, 6), 0.104266)
        
        best_op_pt_10 = pr_curve_specs[10]['best_op_pt']
        self.assertEqual(round(best_op_pt_10['f1'], 4), 0.1931)
        
        self.assertEqual(round(best_op_pt_10['threshold'], 7), 0.0037106)
        self.assertEqual(round(best_op_pt_10['precision'], 7), 0.1157025)
        self.assertEqual(round(best_op_pt_10['recall'], 7), 0.5833333)


# ------------------- Utilities ---------------

    #------------------------------------
    # assert_heatmaps_equal
    #-------------------
    
    def assert_heatmaps_equal(self, fig, truth_arr, col_names=None):
        '''
        Assert whether the data in a given matplotlib figure 
        is the same as the data in a different matplotlib figure
        that was previously saved.
        
        For additional thoroughness, pass in the array of column
        names of the previously saved figure.
        
        Uses self.assertEqual().
        
        :param fig: figure whose data is to be tested for equality 
            against the data of a previously saved figure  
        :type fig: matplotlib.pyplot.Figure
        :param truth_arr: the data extracted from the reference
            figure
        :type truth_arr: np.array
        :param col_names: optionally, the column names of the 
            figures' columns
        :type col_names: [str]
        :raise AssertionError
        '''
        try:
            # Get data plotted int the given figure. 
            # For heatmaps, that data is a masked array.
            # The trailing '.data' turns that into a 
            # regular np.array:
            fig_heat_arr = fig.axes[0].get_children()[0].get_array().data

            if col_names is not None:
                num_cells = len(col_names)**2
                self.assertEqual(len(fig_heat_arr), num_cells)
                self.assertEqual(len(truth_arr), num_cells)
            
            self.assertTrue((fig_heat_arr == truth_arr).all())
        except Exception as e:
            raise AssertionError(f"Heatmaps unequal: {repr(e)}")


    #------------------------------------
    # assert_dataframes_equal 
    #-------------------
    
    def assert_dataframes_equal(self, df1, df2):
        self.assertTrue((df1.columns == df2.columns).all())
        self.assertTrue((df1.index   == df2.index).all())
        self.assertTrue((df1 == df2).all().all())

    #------------------------------------
    # np_diff
    #-------------------

    def np_diff(self, first, second, as_df=False):
        '''
        Given two numpy arrays, return a 'diff' array
        of shape (n, 2) in which each row is one of
        n differences. Col 0 is value in the first
        array, and col 1 is value in the second array.

        if as_df is True, returns a dataframe with column
        header ['first', 'second']

        :param first: one np array to compare
        :type first: np.array
        :param second: other np array to compare
        :type second: np.array
        :returns an np array with result, or a dataframe
        :rtype {np.array | pd.DataFrame}
        '''
        mask = np.stack((first != second))
        res = np.stack((first[mask], second[mask])).T
        if as_df:
            res = pd.DataFrame(res, columns=['first', 'second'])
        return res

    #------------------------------------
    # test_confusion_matrices_from_raw_results
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_confusion_matrices_from_raw_results(self):
        
        # Prepare a fake runs_raw_results directory,
        # which would normally be created during the course
        # of a training process:
        
        with tempfile.TemporaryDirectory(dir='/tmp',
                                         prefix='charting_tests') as tmp_dir_nm:
            (fname, class_names) = self.make_raw_results(tmp_dir_nm)
            res_TrainHistoryCMs = Charter.confusion_matrices_from_raw_results(fname,
                                                                              class_names, 
                                                                              normalize=False)
            step0_train_truth = [
               [0,0,0,0,0,0],
               [0,0,1,2,0,0],
               [0,0,1,0,0,0],
               [0,0,1,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0]
               ]
            truth_df = pd.DataFrame(step0_train_truth)
            truth_df.columns = class_names
            truth_df.index   = class_names
            self.assert_dataframes_equal(res_TrainHistoryCMs[0].training,
                                         truth_df) 

            step0_val_truth = [
               [0,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,2,0],
               [0,0,0,0,1,0],
               [0,0,0,0,2,0],
               [0,0,0,0,0,0]
               ]
            truth_df = pd.DataFrame(step0_val_truth)
            truth_df.columns = class_names
            truth_df.index   = class_names
            self.assert_dataframes_equal(res_TrainHistoryCMs[0].validation,
                                         truth_df) 
            
            
            step2_train_truth = [
               [0,0,0,0,0,0],
               [0,0,0,0,0,0],
               [0,0,0,0,2,2],
               [0,0,0,0,0,1],
               [0,0,0,0,0,0],
               [0,0,0,0,0,0]
               ]
            truth_df = pd.DataFrame(step2_train_truth)
            truth_df.columns = class_names
            truth_df.index   = class_names
            self.assert_dataframes_equal(res_TrainHistoryCMs[1].training,
                                         truth_df) 
            
            step2_val_truth = [
               [0,0,0,0,0,0],
               [0,0,0,0,2,0],
               [0,0,0,0,1,0],
               [0,0,0,0,0,0],
               [0,0,0,0,2,0],
               [0,0,0,0,0,0]
               ]
            truth_df = pd.DataFrame(step2_val_truth)
            truth_df.columns = class_names
            truth_df.index   = class_names
            self.assert_dataframes_equal(res_TrainHistoryCMs[1].validation,
                                         truth_df) 

# ---------------------- Utilities --------------------

    def make_raw_results(self, dst_dir):
        '''
        Create a fake runs_raw_results .csv and
        class_names.txt files. These two would normally
        be left behind by a training process.
        
        :param dir: destination of .csv and .txt files
        :type dir: src
        '''
        csv_content = ("step,train_preds,train_labels,val_preds,val_labels\n"
                       '0,"[2, 2, 3, 2, 3]","[2, 3, 1, 1, 1]","[4, 4, 4, 4, 4,]","[3, 4, 4, 2, 2]"\n'
                       '2,"[5, 5, 4, 5, 4]","[2, 3, 2, 2, 2]","[4, 4, 4, 4, 4]","[2, 4, 4, 1, 1]"\n'
                       )
        names_content = ['DYSMEN_S', 'HENLES_S', 'audi', 'bmw', 'diving_gear', 'office_supplies']
        fname = os.path.join(dst_dir, 'raw_data.csv')
        
        # Write the pred/labels csv file
        with open(fname, 'w') as fd:
            fd.write(csv_content)
        class_names_path = os.path.join(dst_dir, 'class_names.txt')
        
        # Write the class names:
        with open(class_names_path, 'w') as fd:
            fd.write(str(names_content))
        return fname, names_content

# ---------------------- Main --------------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()