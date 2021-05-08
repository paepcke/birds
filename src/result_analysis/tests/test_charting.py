'''
Created on May 7, 2021

@author: paepcke
'''
import os
import unittest

import torch

import pandas as pd
from result_analysis.charting import Charter


#*******TEST_ALL = True
TEST_ALL = False


class Test(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        super(Test, cls).setUpClass()
        cls.cur_dir  = os.path.dirname(__file__)
        cls.data_dir = os.path.join(cls.cur_dir, 'data')
        
        cls.conf_matrix_path = os.path.join(cls.data_dir, 
                                            'conf_matrix_results.csv')
        with open(cls.conf_matrix_path, 'r') as fd:
            cls.class_names = fd.readline()

    def setUp(self):
        self.cm = Charter()
        
        self.cm_df = pd.read_csv(self.conf_matrix_path, index_col=0)
        self.cm_np = self.cm_df.to_numpy()
        self.cm_tn = torch.tensor(self.cm_np)
        
        self.num_species = len(self.cm_df.index)
        

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
         
        cm_df_normed = self.cm.calc_conf_matrix_norm(self.cm_df)
        self.assertEqual(cm_df_normed.sum(axis=1).sum(),
                         self.num_species
                         )
        
        cm_np_normed = self.cm.calc_conf_matrix_norm(self.cm_np)
        self.assertEqual(cm_np_normed.sum(axis=1).sum(),
                         self.num_species
                         )

        cm_tn_normed = self.cm.calc_conf_matrix_norm(self.cm_tn)
        self.assertEqual(cm_tn_normed.sum(axis=1).sum(),
                         self.num_species
                         )

    #------------------------------------
    # test_conf_matrix 
    #-------------------

    #*****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_conf_matrix(self):

        fig = self.cm.fig_from_conf_matrix(self.cm_df)
        print(self.cm)

# ---------------------- Main --------------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()