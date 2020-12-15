'''
Created on Dec 13, 2020

@author: paepcke
'''
import os
import unittest

import numpy as np

from bird_dataloader import BirdDataLoader, SKFSampler
from bird_dataset import BirdDataset


#TEST_ALL = True
TEST_ALL = False

class TestBirdDataLoader(unittest.TestCase):

    CURR_DIR = os.path.dirname(__file__)
    TEST_FILE_PATH = os.path.join(CURR_DIR, 'data/train')

    #------------------------------------
    # setUpClass
    #-------------------
    
    @classmethod
    def setUpClass(cls):
        pass
    
    #------------------------------------
    # setUp
    #-------------------

    def setUp(self):
        self.dataset = BirdDataset(self.TEST_FILE_PATH)

    #------------------------------------
    # tearDown
    #-------------------

    def tearDown(self):
        pass


    #------------------------------------
    # testSampler 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testSampler(self):
        
        sampler = SKFSampler(self.dataset, num_folds=3, shuffle=True)
        
        # Number of splits to expect:
        self.assertEqual(sampler.get_n_splits(), 3)

        # Total of 12 samples: 6 for species 0, 6 for species 1:
        self.assertEqual(len(sampler), 12)
        
        # We shuffle, but the default random_state (i.e. random seed)
        # is set in SKFSampler's __init__() method. So
        # for each fold we expect the train sample indices and 
        # corresponding test indices to be the same. 
        # Create a dict with the expected train/test index arrays
        # for each of the expected 3 folds:
        expected = [
            {'train_indices' : np.array([ 0,  2,  4,  5,  6,  7,  9, 10]), 
             'test_indices'  : np.array([ 1,  3,  8, 11])},
            
            {'train_indices' : np.array([ 1,  2,  3,  5,  6,  8,  9, 11]), 
             'test_indices'  : np.array([ 0,  4,  7, 10])},
            
            {'train_indices' : np.array([ 0,  1,  3,  4,  7,  8, 10, 11]), 
             'test_indices'  : np.array([2, 5, 6, 9])}
            ]
        
        # Ta-taaaa:
        for i, (train_indices, test_indices) in enumerate(sampler):
            self.assertEqual(train_indices.all(), expected[i]['train_indices'].all())
            self.assertEqual(test_indices.all(), expected[i]['test_indices'].all())

    #------------------------------------
    # testLoading
    #-------------------

    #*****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testLoading(self):
        dl = BirdDataLoader(self.dataset, batch_size=2, num_folds=3)
        (sample, class_id) = next(dl)
        print(sample, class_id) 

# --------------------------------- Main --------------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()