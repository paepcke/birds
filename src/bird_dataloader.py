'''
Created on Dec 13, 2020

@author: paepcke
'''

from sklearn.model_selection._split import StratifiedKFold
import torch
from torch import unsqueeze, cat
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

import numpy as np


class BirdDataLoader(DataLoader):
    '''

    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self,
                 dataset,
                 batch_size=32,
                 num_workers=0,
                 pin_memory=False,
                 prefetch_factor=2,
                 drop_last=False,
                 num_folds=10
                 ):
        '''
        Constructor
        Note: the shuffle keyword must not be specified, because
              we are using the SKFSampler below; specified in
              (https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler)
        '''
        
        fold_indices_sampler = SKFSampler(dataset, num_folds=num_folds)
        self.sampler     = fold_indices_sampler
        self.num_folds   = num_folds
        # Total num of batches served when
        # rotating through all folds is computed
        # the first time __len__() is called:
        
        self.num_batches = None
        
        #***** delete
#         self.sampler = BatchSampler(fold_indices_sampler,
#                                     batch_size,
#                                     drop_last
#                                     )
        #***** End delete
        
        super().__init__(
                 dataset,
                 batch_size=batch_size,
                 sampler=self.sampler,
                 num_workers=num_workers,
                 pin_memory=pin_memory,
                 prefetch_factor=prefetch_factor,
                 drop_last=drop_last
            )

    #------------------------------------
    # __len__
    #-------------------

    def __len__(self):
        '''
        Number of batches this loader will
        feed out. Example:
            o 12 samples total
            o  3 folds
            o  2 batch size
            o  4 samples in each fold (12/3)
            o  2 batches per fold (samples-each-fold / batch-size)
            o  3 number of trips through folds
            o  2 number of folds in each of the 3
                 trips (num-folds - hold-out-fold)
            o 12 batches total: batches-per-fold * folds-per-trip * num-folds 
                   2*2*3 = 12
        '''

        # Compute number of batches only once:
        if self.num_batches is None:
            
            # This computation can surely be more
            # concise and direct. But it happens only
            # once, and this step by step is easier
            # on the eyes than one minimal expression:
            num_samples      = len(self.sampler)
            
            # Rounded-down number of samples that fit into each fold:
            samples_per_fold = num_samples // self.num_folds
            batches_per_fold = samples_per_fold // self.batch_size
            # For each of the num-folds trips, (num-folds - 1)*batches-per-fold
            # are server out:
            self.num_batches = self.num_folds * batches_per_fold * (self.num_folds - 1)
            
            # May have more samples than exactly fit into
            # that total_batches number of batches. So there
            # may be one more partially filled batch for every trip
            # through the folds:
            
            remainder_samples = num_samples % self.num_batches 
            
            if not self.drop_last and remainder_samples > 0:
                # Add the final partially filled batch, 
                # if num_samples not a multiple of batches += 1
                self.num_batches += 1
            
        return self.num_batches

    #------------------------------------
    # __iter__
    #-------------------
    
    def __iter__(self):
        # Call to __next__() returns
        # a generator, which does the 
        # right thing with next(), list(),
        # and for loops. Return that iterator:
        
        return(self.__next__())
    
    #------------------------------------
    # __next__
    #-------------------

    def __next__(self):
        
        self.curr_fold_idx = 0
        for fold_train_ids, fold_test_ids in self.sampler:

            # Keep track of which fold we are working
            # on. Needed only as info for client; not
            # used for logic in this method:
            
            self.curr_fold_idx += 1
            
            # fold_train_ids has all sample IDs
            # to use for training in this fold. 
            # The fold_test_ids holds the left-out
            # sample IDs to use for testing once 
            # the fold_train_ids have been served out
            # one batch at a time.
            
            # Set this fold's test ids aside for client
            # to retrieve via: get_fold_test_sample_ids()
            # once they pulled all the batches of this
            # fold:
            self.curr_test_sample_ids = fold_test_ids
            
            num_train_sample_ids = len(fold_train_ids)
            num_batches = num_train_sample_ids // self.batch_size
            num_remainder_samples = num_train_sample_ids % self.batch_size
            batch_start_idx = 0
            
            for _batch_count in range(num_batches):
                
                batch  = None
                # Truth labels for each sample in 
                # the current batch:
                y      = []
                batch_end_idx    = batch_start_idx + self.batch_size
                curr_batch_range = range(batch_start_idx, batch_end_idx)
                
                for sample_id in curr_batch_range:
                    
                    # Get one pair: <img-tensor>, class_id_int:
                    (img_tensor, label) = self.dataset[sample_id]
                    expanded_img_tensor = unsqueeze(img_tensor, dim=0)
                    batch = (cat((batch, expanded_img_tensor), dim=0)
                              if batch is not None
                            else expanded_img_tensor)
                    y.append(label)
                    
                # Got one batch ready:
                yield (batch, torch.tensor(y))
                # Client consumed one batch in current fold.
                # Next batch: Starts another batch size
                # samples onwards in the train fold:
                batch_start_idx += self.batch_size
                continue
            
            # Done all full batches. Any partial batch 
            # left over that we should include?
            
            if num_remainder_samples > 0 and not self.drop_last:
                batch = None
                y     = []
                
                for sample_id in range(batch_start_idx, 
                                       batch_start_idx + num_remainder_samples):
                    (img_tensor, label) = self.dataset[sample_id] 
                    expanded_img_tensor = unsqueeze(img_tensor, dim=0)
                    batch = (cat((batch, expanded_img_tensor))
                              if batch is not None
                            else expanded_img_tensor)
                    y.append(label)
                yield (batch, torch.tensor(y))
            # Next fold:
            continue

    #------------------------------------
    # get_curr_fold_idx 
    #-------------------
    
    def get_curr_fold_idx(self):
        return self.curr_fold_idx


    #------------------------------------
    # get_fold_test_sample_ids 
    #-------------------
    
    def get_fold_test_sample_ids(self):
        try:
            return self.curr_test_sample_ids
        except:
            return None
    

# --------------------------- Class SKFSampler --------------

class SKFSampler(StratifiedKFold):
    
    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self,
                 dataset,
                 num_folds=10,
                 random_state=1,
                 shuffle=False
                 ):
        super().__init__(n_splits=num_folds,
                         random_state=random_state if shuffle else None, 
                         shuffle=shuffle)
        
        self.dataset = dataset
        
        # Keep track of how many folds
        # we served. Just for logging, 
        # performance analysis, and
        # debugging:
        
        self.folds_served = 0
        
        # Stratified k-fold needs only the labels 
        # in an array; the corresponding samples each 
        # have the same index as the one for each 
        # y-split (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold)
        
        self.fold_generator = self.split(np.zeros(len(dataset)), 
                                         dataset.sample_classes()
                                         )

    #------------------------------------
    # __len__ 
    #-------------------
    
    def __len__(self):
        return len(self.dataset)

    #------------------------------------
    # __iter__ 
    #-------------------

    def __iter__(self):
        return self

    #------------------------------------
    # __next__ 
    #-------------------
    
    def __next__(self):
        return next(self.fold_generator)


# -------------------- Multiprocessing Dataloader -----------

class MultiprocessingDataloader(BirdDataLoader):
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, dataset, world_size, node_rank, **kwargs):
        
        self.dataset  = dataset
        
        self.sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=node_rank
                )

        super().__init__(dataset,
                         shuffle=False,
                         num_workers=0,
                         pin_memory=True,
                         sampler=self.sampler,
                         **kwargs)

