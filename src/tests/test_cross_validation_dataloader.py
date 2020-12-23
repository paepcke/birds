'''
Created on Dec 13, 2020

@author: paepcke
'''
import os
import unittest

import numpy as np

from bird_dataloader import CrossValidatingDataLoader, SKFSampler
from bird_dataset import BirdDataset


TEST_ALL = True
#TEST_ALL = False

class TestBirdDataLoader(unittest.TestCase):

    CURR_DIR = os.path.dirname(__file__)
    TEST_FILE_PATH_BIRDS = os.path.join(CURR_DIR, 'data/train')

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
        self.dataset = BirdDataset(self.TEST_FILE_PATH_BIRDS)

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
    # test_loading_batches_fit_in_fold
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_loading_batches_fit_in_fold(self):
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
        num_samples       = 12 # 6 images each for 2 species
        samples_per_fold  = num_samples / num_folds
        unique_class_ids  = list(self.dataset.class_to_id.values())
        
        dl = CrossValidatingDataLoader(self.dataset, 
                            batch_size=batch_size, 
                            num_folds=num_folds)
        
        # Number of samples in our unittest *dataset*,
        # i.e. not in the dataloader: six of one species, 
        # and six of another:
        
        self.assertEqual(len(dl.dataset), num_samples)
        
        # Total number of batches we expect from 
        # the dataloader; for the algorithm, see method
        # __len__() in class CrossValidatingDataLoader:

        num_batches_total = 12
        
        # But: length of the bird loader should be
        # the number of batches it will feed out
        #    num-samples / batch-size = 12/2 = 6   #<---- 4 (12-4)/2?
        
        self.assertEqual(len(dl), num_batches_total)

        # Get an iterator over the data loader
        # This step would happen automatically 
        # if using a for loop: "for batch in dl:..."
        it = iter(dl)
        
        (batch1, y1) = next(it)
        # Tensor dimensions should be: batch size,
        # 3 for RGB, pixel-height, pixel-width:
         
        self.assertTupleEqual(batch1.shape, (batch_size,3,400,400))
        
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
        #     num-samples / num-folds = 12 / 3 = 4 

        self.assertEqual(len(dl.get_split_test_sample_ids()),
                         samples_per_fold)
        
        (batch2, _y2) = next(it)
        self.assertTupleEqual(batch2.shape, (batch_size,3,400,400))
        
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
        
        for i, _batch in enumerate(dl):
            pass
        
        # The plus-1 is b/c i is zero-origin:
        self.assertEqual(i+1, num_batches_total)
        self.assertEqual(len(dl), num_batches_total)
        
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
        _num_samples  = 12
        
        dl = CrossValidatingDataLoader(self.dataset, 
                            batch_size=batch_size, 
                            num_folds=num_folds)

        num_samples_seen = 0
        num_batches_seen = 0
        for _i, (batch, _y) in enumerate(dl):
            num_samples_seen += len(batch)
            num_batches_seen += 1

        self.assertEqual(num_batches_seen, 18)
        self.assertEqual(num_samples_seen, 48)


# --------------------------------- Main --------------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()