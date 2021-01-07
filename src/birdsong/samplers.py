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
                 shuffle=False,
                 drop_last=True
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
        self.drop_last = drop_last
        super().__init__(n_splits=num_folds,
                 random_state=seed if shuffle else None, 
                 shuffle=shuffle)
        
        self.dataset = dataset
        self.seed = seed
        self.folds_served = 0
        self.epoch = 0
        
        # Since this is not the subclass DistributedSKFSampler,
        # we know that only one process is working. 
        # Initialize respective vars:
        
        self.num_replicas = 1
        
        # The following  var may be updated
        # by subclasses:
        self.rank = 0
        
        self.folds_served = 0
        
        # Initialize self.total_size:
        self.compute_effective_total_size()

        self.fold_generator = self.generate_folds()


    #------------------------------------
    # compute_effective_total_size 
    #-------------------
    
    def compute_effective_total_size(self):

        '''
        There may be multiple processes (replicas)
        of the training script running. Each of them
        should be working on a subset of the dataset.
        The subsets should be disjoint, so the entire
        dataset is covered, but no sample is used
        by more than one replica.
        
        If the dataset length is evenly divisible by # of 
        replicas, then there is no need to drop any data, 
        since the dataset will be split equally. Otherwise,
        depending on whether drop_last was requested, 
        an adjustment is made to exclude the pieces of data
        that do not fill a fold. 
        
        That adjustement in turn will make the effective
        total number of samples across all machines and processes
        slightly different from the number of samples in the
        dataset. Compute that effective size, and initialize
        self.total_size.
        '''
        
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

    #------------------------------------
    # set_epoch 
    #-------------------

    def set_epoch(self, new_epoch):
        
        self.epoch = new_epoch
        self.fold_generator = self.generate_folds()

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
            # The zeros dim won't be used, but must
            # match len(self.my_classes). That length
            # may be less than the length of the dataset
            # when processing the last split:
            self.split(np.zeros(len(self.my_classes)), 
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


    #------------------------------------
    # generate_folds 
    #-------------------
    
    def generate_folds(self):
        '''
        Create a new fold_generator. Called
        by the top level training script via
        calling set_epoch() on its dataloader 
        
        This method must be called at the start
        of each  new epoch (except for epoch 0, which 
        is taken care of in the constructor).
        Do not call this method during the course
        of an epoch, else samples may be reused
        in unpredictable sequences.
        
        Creates a new split, after optionally 
        shuffling the underlying dataset. No
        shuffling occurs after that.
        
        The shuffle is predictable based on 
        the seed and the epoch. All replicas
        therefore shuffle the same way.
        
        Sets self.fold_generator.
        '''
        
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
        
        # Obtain a subset of the indices into the dataset that
        # this replica will work on. The 'rank' variable
        # is a serial number assigned to the process running
        # this replica of the training script. The master 
        # process is 0, subsequent processes on the same or 
        # other machines are 1,2,3,... 
        #  
        # Example:
        #
        #   Let indices      = [30, 0, 1, 17, 10, 6, 18, 2, 26, 22, 20, 24]
        #          rank      = 0
        #     num_replicas   = 3
        #       total_size   = len(indices) = 12
        #     num_replicas   = 3
        #
        #   Then the expression indices[rank:total_size:num_replicas]
        #   produces for the three ranks (replicas of the training
        #   script:
        #
        #     rank 0: [30, 17, 18, 22]
        #     rank 1: [ 0, 10,  2, 20]
        #     rank 2: [ 1,  6, 26, 24]
        #
        #   Thus each process, whether on the machine 
        #   this process is running, or elsewhere, gets
        #   a unique slice of the samples to train on.
        #
        #   For the special case of a single process on one machine being the sole
        #   worked, the indices calculation yields the entire dataset worth of
        #   samples for the process to work on:
        #
        #      [30, 0, 1, 17, 10, 6, 18, 2, 26, 22, 20, 24]

        # Assign my_indices to an instance var,
        # even though in this file the value is only
        # used here. Unittests probe this variable from
        # the outside:
         
        self.my_indices = indices[self.rank:self.total_size:self.num_replicas]
        self.my_classes = [self.dataset.sample_id_to_class[int(sample_id)] 
                              for sample_id 
                               in self.my_indices]
        
        self.fold_generator = self.split(np.zeros(len(self.dataset)), 
                                         self.my_classes
                                         )

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
        
        # Get the true number of processes
        # expected (eventually) to work on the
        # dataset:
        self.num_replicas = dist.get_world_size()
        
        # This processe's serial number:
        self.rank = dist.get_rank()
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
