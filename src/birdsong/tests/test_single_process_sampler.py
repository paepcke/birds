'''
Created on Dec 17, 2020

@author: paepcke
'''

import os, sys
import unittest

from birdsong.rooted_image_dataset import SingleRootImageDataset
from birdsong.samplers import SKFSampler
import numpy as np

packet_root = os.path.abspath(__file__.split('/')[0])
sys.path.insert(0,packet_root)


TEST_ALL = True
#TEST_ALL = False

class TestSingleProcessSampler(unittest.TestCase):
    
    CURR_DIR = os.path.dirname(__file__)
    TEST_FILE_PATH_BIRDS = os.path.join(CURR_DIR, 'data')

    #------------------------------------
    # setUP 
    #-------------------

    def setUp(self):
        self.dataset = SingleRootImageDataset(self.TEST_FILE_PATH_BIRDS)

    #------------------------------------
    # tearDown 
    #-------------------

    def tearDown(self):
        pass

    #------------------------------------
    # test_muiltiprocess_stratified_kfold_xval_no_shuffle
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_muiltiprocess_stratified_kfold_xval_no_shuffle(self):
        
        sampler = SKFSampler(
            self.dataset,
            num_folds=3,
            shuffle=False
            )
        
        # Now have three folds with four samples each:
        #    12 samples across 3 folds

        self.assertEqual(len(sampler), 34)
        # No folds served yet:
        self.assertEqual(sampler.folds_served, 0)
        
        # ---- First fold configuration ----
        
        # Get one pair of train and test sample ids:
        (train_sample_ids1, test_sample_ids1) = next(sampler)
        
        # They better be disjoint:
        intersect = set(train_sample_ids1) & set(test_sample_ids1)
        self.assertEqual(len(intersect), 0)
        
        # Number of train samples: 2 folds' worth:
        self.assertEqual(len(train_sample_ids1), 22)
        
        # Number of test samples: 1 fold's worth:
        self.assertEqual(len(test_sample_ids1), 12)
        
        # ---- Second fold configuration ----
        next(sampler)
        
        # ---- Third fold configuration ----
        
        (train_sample_ids3, test_sample_ids3) = next(sampler)
        # Check disjointness of train and test sample_ids once more:
        intersect = set(train_sample_ids3) & set(test_sample_ids3)
        self.assertEqual(len(intersect), 0)

        # ---- Attempted fourth fold configuration ----
        try:
            next(sampler)
            self.fail("Should have received StopIteration exception")
        except StopIteration:
            pass
            
        self.assertEqual(sampler.folds_served, 1)

        # Since we asked not to shuffle, we should 
        # get the same train/test sample_id sequences
        # if we do it again:
        
        sampler1 = SKFSampler(
             self.dataset,
             num_folds=3,
             shuffle=False
             )

        (train_sample_ids2_1, test_sample_ids2_1) = next(sampler1)
        self.assertEqual(all(train_sample_ids2_1), all(train_sample_ids1))
        self.assertEqual(all(test_sample_ids2_1), all(test_sample_ids1))

    #------------------------------------
    # test_stratified_kfold_xval_with_shuffle
    #-------------------

    #*****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_stratified_kfold_xval_with_shuffle(self):
        
        # First without shuffle:
        sampler1 = SKFSampler(
            self.dataset,
            num_folds=3,
            shuffle=False
            )

        # Get one pair of train and test sample ids:
        (train_sample_ids1, test_sample_ids1) = next(sampler1)
        
        # Get another sampler, and they should be the same:
        sampler2 = SKFSampler(
            self.dataset,
            num_folds=3,
            shuffle=False
            )

        # Should be same:
        (train_sample_ids2_1, test_sample_ids2_1) = next(sampler2)
        self.assertEqual(all(train_sample_ids2_1), all(train_sample_ids1))
        self.assertEqual(all(test_sample_ids2_1), all(test_sample_ids1))

        # Now with shuffle, but same seed:
        sampler3 = SKFSampler(
            self.dataset,
            num_folds=3,
            shuffle=True,
            seed=42
            )
        (train_sample_ids3_1, test_sample_ids3_1) = next(sampler3)

        # Should *differ* b/c xxx_3_1 is shuffled, while xxx_1 is not:
        try:
            np.testing.assert_equal(train_sample_ids3_1, train_sample_ids1)
            self.fail("train_sample_ids3_1 and train_sample_ids should differ")
        except AssertionError:
            pass
        try:
            np.testing.assert_equal(test_sample_ids3_1, test_sample_ids1)
            self.fail("test_sample_ids3_1 and test_sample_ids should differ")
        except AssertionError:
            pass
        
        # BUT: sampler4 with shuffle and another with same seed
        #      should create xxx_4_1 equal to xxx_3_1
        
        # Shuffle, but same seed:
        sampler4 = SKFSampler(
            self.dataset,
            num_folds=3,
            shuffle=True,
            seed=42
            )
        (train_sample_ids4_1, test_sample_ids4_1) = next(sampler4)
        np.testing.assert_equal(train_sample_ids4_1, train_sample_ids3_1,
                                "train_sample_ids4_1 and train_sample_ids3_1 should be equal: both shuffle, but same seed"
                                )
        np.testing.assert_equal(test_sample_ids4_1, test_sample_ids3_1,
                                "test_sample_ids4_1 and test_sample_ids3_1 should be equal: both shuffle, but same seed"
                                )
        
        # sampler5 which shuffles different seed 
        # should differ from xxx_4_1
        
        # Shuffle, but different seed:
        sampler5 = SKFSampler(
            self.dataset,
            num_folds=3,
            shuffle=True,
            seed=50
            )
        (train_sample_ids5_1, test_sample_ids5_1) = next(sampler5)
        try:
            np.testing.assert_equal(train_sample_ids5_1, train_sample_ids4_1)
            self.fail("train_sample_ids5_1 and train_sample_ids4_1) should differ: both shuffle, but different seeds")
        except AssertionError:
            pass

        try:
            np.testing.assert_equal(test_sample_ids5_1, test_sample_ids4_1)
            self.fail("test_sample_ids5_1 and test_sample_ids4_1) should differ: both shuffle, but different seeds")
        except AssertionError:
            pass

# ---------------- Main -----------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
