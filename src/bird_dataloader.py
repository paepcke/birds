'''
Created on Dec 13, 2020

@author: paepcke
'''

from sklearn.model_selection._split import StratifiedKFold
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torch.utils.data.sampler import Sampler, BatchSampler

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
        self.sampler = fold_indices_sampler
        
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
        num_samples = len(self.sampler)
        
        # As many batches as can be filled
        # with samples (the // operator rounds down)):
         
        num_batches = num_samples // self.batch_size
        
        if not self.drop_last and (num_samples % self.batch_size) > 0:
            # Add the final partially filled batch, 
            # if num_samples not a multiple of batches += 1
            num_batches += 1
            
        return num_batches
    
    #------------------------------------
    # __next__ 
    #-------------------
    
    def __next__(self):
        return next(self.sampler)


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

