'''
Created on Dec 17, 2020

@author: paepcke
'''
import math

import torch
from sklearn.model_selection._split import StratifiedKFold
from torch import distributed as dist

import numpy as np


# --------------------------- Class SKFSampler --------------
class SKFSampler(StratifiedKFold):
    '''
    This is an abstract class, i.e. it cannot
    be instantiated directly. Use the DistributedSKFSampler
    class.
    
    Partitions dataset into num_folds
    sequences of batches. Manages serving
    batches one train fold at a time. Switch
    between train folds is transparent. 
    '''
    
    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self,
                 dataset,
                 num_folds=10,
                 seed=42,
                 shuffle=False
                 ):
        '''
        Arg seed, if set to an
        int ensures that successive instantiations
        of this class shuffle predictably, and thus
        ensure repeatability.
        
        If shuffle is True the samples in the dataset
        will be shuffled at the outset. However, the
        samples within each fold will still not be 
        shuffled.
          
        @param dataset: underlying map sample_id --> <img_tensor, label>
        @type dataset: BirdDataset
        @param num_folds: number k in k-fold cross validation
        @type num_folds: int
        @param seed: also known as seed
        @type seed: {None | int}
        @param shuffle: whether or not to shuffle dataset at
            the outset
        @type shuffle: bool
        '''
        super().__init__(n_splits=num_folds,
                 random_state=seed if shuffle else None, 
                 shuffle=shuffle)
        
        self.dataset = dataset
        self.seed = seed
        self.folds_served = 0
        
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = range(len(self.dataset))

        # Construct a list of class IDs corresponding
        # to each of the samples in order. I.e. the list
        # will be as long as there the number of indices
        # computed above:
        
        self.my_classes = [dataset.sample_id_to_class[indx] for indx in indices]

        # Stratified k-fold needs only the labels 
        # in an array; the corresponding samples each 
        # have the same index as the one for each 
        # y-split (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold)
        
        self.fold_generator = self.split(np.zeros(len(dataset)), 
                                         self.my_classes
                                         )

    #------------------------------------
    # __len__ 
    #-------------------
    
    def __len__(self):
        return len(self.dataset)

    #------------------------------------
    # get_split
    #-------------------
    
    def get_split(self):
        for _i, train_and_validate_samples in enumerate(
            self.split(np.zeros(len(self.dataset)), 
                       self.my_classes)
            ):
            yield train_and_validate_samples
            
    #------------------------------------
    # __iter__ 
    #-------------------

    def __iter__(self):
        return self

    #------------------------------------
    # __next__ 
    #-------------------
    
    def __next__(self):
        try:
            return next(self.fold_generator)
        except StopIteration as e:
            self.folds_served += 1
            raise StopIteration from e 


# --------------------------- Class DistributedSKFSampler --------------

class DistributedSKFSampler(SKFSampler):
    '''
    Like SKFSampler, but can operate in a
    distributed environment, where samples 
    are process in different machines and/or
    GPUs.
    '''
    
    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self,
                 dataset,
                 num_folds=10,
                 seed=42,
                 shuffle=False,
                 drop_last=True
                 ):
        '''
        Stratified k-fold cross validation sampling,
        like SKFSampler. However, this sampler operates
        in a distributed environment of (potentially) multiple
        machines, with (potentially) multiple GPUs.
        
        Assumptions:
            o dist.init_process_group() has been called 
              before creating an instance of this class.

        @param dataset: underlying map sample_id --> <img_tensor, label>
        @type dataset: BirdDataset
        @param num_folds: number k in k-fold cross validation
        @type num_folds: int
        @param seed: also known as seed
        @type seed: {None | int}
        @param shuffle: whether or not to shuffle dataset at
            the outset
        @type shuffle: bool
        @param drop_last: whether to discard partially filled folds
        @type drop_last: bool
        '''
        
        if len(dataset) == 0:
            raise ValueError("Dataset is empty, nothing to sample from")
        
        if not dist.is_initialized():
            raise RuntimeError("Must call dist.init_process_group() before instantiating distrib sampler")

        StratifiedKFold.__init__(self,
                                 n_splits=num_folds,
                                 random_state=seed if shuffle else None, 
                                 shuffle=shuffle)
        
        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        # Epoch will be changed by client via a 
        # call to set_epoch() before each epoch. 
        # The number is then used when shuffling
        # to ensure that all replicas will shuffle
        # the same way in their respective epochs:
        
        self.epoch = 0

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas
        
        # Keep track of how many folds
        # we served. Just for logging, 
        # performance analysis, and
        # debugging:
        
        self.folds_served = 0
        
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.tensor(range(len(self.dataset)))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        
        # Stratified k-fold needs only the sequence 
        # of labels (i.e. target classes) for each of the samples. 
        # (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold)
        
        # Obtain a subset of the indices in dataset that
        # this replica will work on:

        self.my_indices = indices[self.rank:self.total_size:self.num_replicas]
        self.my_classes = [dataset.sample_id_to_class[int(sample_id)] 
                              for sample_id 
                               in self.my_indices]
        
        self.fold_generator = self.split(np.zeros(len(dataset)), 
                                         self.my_classes
                                         )

    #------------------------------------
    # set_epoch 
    #-------------------


    def set_epoch(self, epoch: int):
        '''
        Sets the epoch for this sampler. When shuffle=True`, 
        updating epoch before starting each fold ensures all replicas
        use a different random ordering for each epoch. Otherwise, 
        the next iteration of this sampler will yield the same ordering.

        @param epoch: number of upcoming epoch
        @type epoch: int
        '''
        self.epoch = epoch
