'''
Created on Dec 13, 2020

@author: paepcke
'''
import os, sys
import unittest

import numpy as np

from birdsong.rooted_image_dataset import SingleRootImageDataset
from birdsong.cross_validation_dataloader import CrossValidatingDataLoader
from birdsong.cross_validation_dataloader import SKFSampler, EndOfSplit

packet_root = os.path.abspath(__file__.split('/')[0])
sys.path.insert(0,packet_root)

#from birdsong.rooted_image_dataset import SingleRootImageDataset
#from birdsong.rooted_image_dataset import MultiRootImageDataset

TEST_ALL = True
#TEST_ALL = False

class TestBirdDataLoader(unittest.TestCase):

    SAMPLE_WIDTH  = 400
    SAMPLE_HEIGHT = 400
    CURR_DIR = os.path.dirname(__file__)
    TEST_FILE_PATH_BIRDS = os.path.join(CURR_DIR, 'data')

    #------------------------------------
    # setUpClass
    #-------------------
    
    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.data_root = os.path.join(cls.cur_dir, cls.TEST_FILE_PATH_BIRDS)

    #------------------------------------
    # setUp
    #-------------------

    def setUp(self):
        self.dataset = SingleRootImageDataset(
            self.data_root,
            sample_height=self.SAMPLE_HEIGHT,
            sample_width=self.SAMPLE_WIDTH,
            to_grayscale=True
            )
        
        #self.dataset = SingleRootImageDataset(self.TEST_FILE_PATH_BIRDS)

    #------------------------------------
    # tearDown
    #-------------------

    def tearDown(self):
        pass

    #------------------------------------
    # test_sampler 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_sampler(self):
        
        sampler = SKFSampler(self.dataset, num_folds=3, shuffle=True)
        
        # Number of splits to expect:
        self.assertEqual(sampler.get_n_splits(), 3)

        # Total of 12 samples: 6 for species 0, 6 for species 1:
        self.assertEqual(len(sampler), 34)
        
        # We shuffle, but the default random_state (i.e. random seed)
        # is set in SKFSampler's __init__() method. So
        # for each fold we expect the train sample indices and 
        # corresponding test indices to be the same. 
        # Create a dict with the expected train/test index arrays
        # for each of the expected 3 folds:
        expected = [
            {'train_indices' : np.array([1,3,5,6,8,9,10,11,12,13,16,19,21,22,23,26,28,29,30,31,32,33]),
             'test_indices'  : np.array([0,2,4,7,14,15,17,18,20,24,25,27])},
            
            {'train_indices' : np.array([0,2,3,4,5,7,9,10,11,13,14,15,16,17,18,20,24,25,26,27,28,29,32]),
             'test_indices'  : np.array([1,6,8,12,19,21,22,23,30,31,33])},
            
            {'train_indices' : np.array([0,1,2,4,6,7,8,12,14,15,17,18,19,20,21,22,23,24,25,27,30,31,33]), 
             'test_indices'  : np.array([3,5,9,10,11,13,16,26,28,29,32])}
            ]
        
        # Ta-taaaa:
        for i, (train_indices, test_indices) in enumerate(sampler.get_split()):
            self.assertEqual(train_indices.all(), expected[i]['train_indices'].all())
            self.assertEqual(test_indices.all(), expected[i]['test_indices'].all())

    #------------------------------------
    # test_loading_batches
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_loading_batches(self):
        '''
        Case when the batch size neatly divides
        into the number of samples. Example:
        
            num-samples = 12
            batch-size  =  2
            num-folds   = 12/2 = 6 batches
        
        In contrast to, say batch size 5, for which
        two full batches can be served, but two samples
        are left over. This case is tested in a separate
        test method 
        '''
        batch_size        = 2
        num_folds         = 3
        num_samples       = 34 # Image files under the 'data' subdir
        samples_per_fold  = num_samples // num_folds
        unique_class_ids  = list(self.dataset.class_to_id.values())
        
        dl = CrossValidatingDataLoader(self.dataset, 
                            batch_size=batch_size, 
                            num_folds=num_folds)
        
        # Number of samples in our unittest *dataset*,
        # i.e. not in the dataloader:
        
        self.assertEqual(len(dl.dataset), num_samples)
        
        # Total number of batches we expect from 
        # the dataloader; for the algorithm, see method
        # __len__() in class CrossValidatingDataLoader:

        num_batches_total = num_folds * samples_per_fold
        
        self.assertEqual(len(dl), num_batches_total)

        # Get an iterator over the data loader
        # This step would happen automatically 
        # if using a for loop: "for batch in dl:..."
        it = iter(dl)
        
        (batch1, y1) = next(it)
        # Tensor dimensions should be: batch size,
        # 3 for RGB, pixel-height, pixel-width:
         
        self.assertTupleEqual(batch1.shape, (batch_size,1,400,400))
        
        # There should be one truth label for each
        # sample in batch1:
        
        self.assertEqual(len(y1), batch_size)
        
        # Each truth label should be one of
        # the class IDs that the underlying BirdDataset
        # knows about:
        
        for class_label in y1:
            self.assertIn(class_label, unique_class_ids)
        
        # The current fold's list of sample indices
        # in the test fold should be as long as the
        # number of samples in each fold:
        #     num-samples // num-folds = 34 / 3 = 11
        # However, the sampler might balance, and 
        # serve one more or less: 

        self.assertAlmostEqual(len(dl.get_split_test_sample_ids()), 
                               samples_per_fold, 
                               delta=1)
        
        (batch2, _y2) = next(it)
        self.assertTupleEqual(batch2.shape, (batch_size,1,400,400))
        
        # Start over with a new dataloader, same data,
        # and count how many batches it servers out:

        dl = CrossValidatingDataLoader(self.dataset, 
                            batch_size=batch_size, 
                            num_folds=num_folds)

        # We should have 4batches-per-fold - 2batches-pulled = 2
        # more batches in the current first fold. Followed by 
        # num_folds - 1 = 2 additional folds' worth of more batches:
        #  
        #  (3folds * 2batches-per-fold) - 2batches-already-pulled = 4
        # 
        # more batches:

        # Count number of batches and splits:
        got_splits  = 0
        batches_per_split = num_batches_total // dl.num_folds
        
        # 'Safety count' just in case StopIteration
        # never happens due to a bug:
        
        for _safety_count in range(dl.num_folds + 2):
            try:
                got_batches = 0
                for _i, _batch in enumerate(dl):
                    got_batches += 1
            except EndOfSplit:
                self.assertEqual(got_batches, batches_per_split)
                got_splits += 1
                
        self.assertEqual(got_splits, dl.num_folds)
        
        # Last fold's number of test sample ids
        # should be the full length as in previous
        # batches:
        self.assertEqual(len(dl.get_split_test_sample_ids()), 
                         samples_per_fold)

    #------------------------------------
    # test_some_batches_partly_filled 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_some_batches_partly_filled(self):
        '''
        Nasty case, when divisions between number
        of samples, batch size, number of folds,
        and distribution of classes does not come
        out well. 
        
        We use all samples, as usual: 12. Set
        batch_size to 3, and num_folds to 5. We
        get cases of 9 training samples, and 3 test
        samples:
        
        When batches fit: 9 train, 3 test:
        
            >>> fold_train_ids
            array([ 2,  3,  4,  5,  7,  8,  9, 10, 11])
        
            >>> fold_test_ids
            array([0, 1, 6])

        And:
		When batches have one left over: 10 train, 2 test:
		
		    >>> fold_train_ids
		    array([ 0,  1,  2,  4,  5,  6,  7,  8, 10, 11])
		
		    >>> fold_test_ids
		    array([3, 9])
		
		In this case, the number of full or partial batches 
		for each of the five folds comes out like this:
		
		    Fold
			   1 : full batch
			          "           <-- Each batch is 3 samples,
			          " 	          so 9 train, 3 test      
			   2 : full batch      
			          "
			          "
			   3 : full batch
			          "
			          "            <-- case of 10 train, 2 test
			          1
			   4 : full batch
			          "
			          "
			          1
			   5 : full batch
			          "
			          "
			          1
        '''
        
        batch_size   = 3
        num_folds    = 5
        num_samples  = 34
        #samples_per_fold = num_samples // num_folds

        dl = CrossValidatingDataLoader(self.dataset, 
                            batch_size=batch_size,
                            drop_last=True, 
                            num_folds=num_folds)
        
        # Compute batches_per_split for
        # just the training folds:
        samples_per_fold  = num_samples // (num_folds - 1) # 8
        samples_per_split = (num_folds-1) * samples_per_fold # 32
        samples_total     = (num_folds-1) * samples_per_split # 128
        batches_total     = samples_total // batch_size # 128/3: 42.666666666666664
        #batches_per_split = batches_total / (num_folds-1)  # 42.666666666666664 / 4 

        # Compute batches_per_split assuming
        # all num_folds folds:
        #samples_per_split = (num_folds-1) * samples_per_fold
        #samples_total     = num_folds * samples_per_split
        #batches_total     = samples_total // batch_size
        #batches_per_split = batches_total / num_folds
        
        self.assertAlmostEqual(batches_total, len(dl), delta=batch_size-1)
        
        num_samples_seen = 0
        num_batches_seen = 0
        batches_seen_in_splits = []
        batch_count = 0
        for _safety_cnt in range(dl.num_folds + 2):
            try:
                for _i, (batch, _y) in enumerate(dl):
                    num_samples_seen += len(batch)
                    num_batches_seen += 1
                    batch_count += 1
            except EndOfSplit:
                # End of split:
                batches_seen_in_splits.append(batch_count)
                batch_count = 0
        
        # True number of batches per split is 9
        # or the average of the delivered batches per split:
        self.assertEqual(num_batches_seen, 
                         np.mean(batches_seen_in_splits)*num_folds)


# --------------------------------- Main --------------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
