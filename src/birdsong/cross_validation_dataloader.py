'''
Created on Dec 13, 2020

@author: paepcke
'''

import torch
from torch import unsqueeze, cat
from torch.utils.data import DataLoader

from birdsong.samplers import SKFSampler, DistributedSKFSampler

# ------------------------- Class CrossValidatingDataLoader ----------------

class CrossValidatingDataLoader(DataLoader):
    '''
    
    Subclass of torch.utils.data.DataLoader. Provides
    stratified k-fold crossvalidation in single-machine,
    (optionally) single-GPU context.
    
    Instantiate this class if running only on a
    single machine, optionally using a single GPU. Else,
    instantiate the MultiprocessingDataLoader subclass 
    instead.
    
    An instance of this class wraps any dict-API dataset instance, 
    which provides tuples , for instance (<img-tensor>, class-label-int) 
    from the file system when given a sample ID.
    
    This subclass of torch.utils.data.DataLoader specilizes
    the default by using a stratified k-fold cross validation
    sampler. That underlying sampler manages partitioning of
    samples into folds, and successively feeding samples from
    the training folds. The sampler also manages the 'switching out'
    of folds to take the role of test fold in round robin fashion.
        
    This DataLoader instance also managing combination of 
    samples into batches.
    
    An instance of this class presents an iterator API, additionally
    serving the test samples whenever one set of train folds are 
    exhausted. Example: assume 
          
          o k-fold cross validation k = 5
        
          for split in range(k):
          
              for batch in my_dataloader:
                  try:
                      <feed training batch to emerging model>
                  except EndOfSplit as e:
                      print(e.message) # Just for debugging
                      break
                  
              # Exhausted all train folds of one split
              # Now test current state of the 
              # model using this split's test samples,
              # which are available as an iterator from the
              # dataloader:
              
              for (img_tensor, label) in my_ataloader.validation_samples():
                  <test model on img_tensor>
         
              # next split
              
    The validation_samples() method is a generator that provides the content of 
    the just exhausted split's validation samples.
    
    NOTE: when re-setting an instance of this class
          for a new epoch, client must call set_epoch()
          with the new epoch number to ensure proper
          shuffling randomness. Such a reset occurs implicitly
          with the often used idiom:
               
                for i,res = enumerate(dataloader)
        
          The enumerate() starts the same dataloader instance
          from the beginning. 
          
          If shuffle is False, set_epoch() needs not be called.
          But doing so does no harm.
          
    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self,
                 dataset,
                 batch_size=32,
                 shuffle=False,
                 seed=42,
                 num_workers=0,
                 pin_memory=False,
                 prefetch_factor=2,
                 drop_last=False,
                 num_folds=10,
                 sampler=None
                 ):
        '''
        This instance will use cross validation
        as it serves out samples. The client determines
        the number of folds to use. Example for 
        num_folds of 2:
        
         Split1:
           TrainFold1    TrainFold2   ValidationFold  
            sample1      sample2        sample3
            sample4      sample5        sample6

         Split2:
           TrainFold1    TrainFold2   ValidationFold  
            sample3      sample4        sample2
            sample1      sample6        sample5
            
        This dataloader will create two sequences,
        like this:
        
           For use with training:   [sample1, sample4, sample2, sample5]
           For use with validation: [sample4, sample6]
             after the training 
             sequence is used up

        Assuming batch_size of two, this dataloader's
        client will receive one row from each 
        call to next():
        
            [[sample1, sample4],
             [sample2, sample5],
             [None   , None]
             ]
             
        The None tuple indicates that this split has
        been exhausted, and it is time to validate.
        
        The client then calls validation_samples() on
        this dataloader instance to receive one validation
        sample at a time. The client will predict the
        (target) class for each of these validation samples,
        and tally successes and failures. The client should
        then compute the compute validation accuracy from
        that series of successes and failures. 

        Calling next() again will create a new split,
        and again feed out the samples in the respective
        new folds.
        
        The feed terminates after as many splits as there
        are folds. Any following call to next() will raise
        a StopIteration exception.

        @param dataset: underlying map-store that 
                supplies(img_torch, label) tuples
        @type dataset: BirdDataset
        @param batch_size: number of samples to combine into 
            a batch to feed model during training
        @type batch_size: int
        @param pin_memory: set to True if using a GPU. Speeds
            transfer of tensors from CPU to GPU
        @type pin_memory: bool
        @param prefetch_factor: how many samples to prefetch from
            underlying database to speed access to file system
        @type prefetch_factor: int
        @param drop_last: whether or not to serve only partially 
            filled batches. Those occur when samples cannot be
            evenly packed into batches. 
        @type drop_last: bool
        @param num_folds: the 'k' in k-fold cross validation
        @type num_folds: int
        @param sampler: Only used when MultiprocessingDataLoader
            is being instantiated, and that class's __init__()
            calls super(). Leave out for singleprocess/single-GPU
            use
        @param drop_last: whether to skip the last split if
            the folds would not have equal numbers of samples
        @type drop_last: bool
        @type sampler: {None | DistributedSKFSampler}
        '''
        
        if len(dataset) == 0:
            raise ValueError("Dataset is empty, nothing to load")

        self.drop_last = drop_last
        
        # Sampler will only be set if a subclass instance
        # of MultiprocessingDataLoader is being initialized.
        # Else, running single process:
        
        if sampler is None:
            self.sampler = SKFSampler(dataset, 
                                      num_folds=num_folds, 
                                      shuffle=shuffle,
                                      drop_last=drop_last,
                                      seed=seed)
        else:
            self.sampler = sampler
            
        if not isinstance(batch_size, int) or batch_size <= 0:
            msg = f"Batch size must be a positive int, not "
            
            # Complete the error msg according which of
            # the two failure conditions occurred:
            msg += type(batch_size).__name__ else f"{batch_size}"
            
            raise ValueError(msg)
        
        self.batch_size = batch_size
            
        self.num_folds   = num_folds

        # Total num of batches served when
        # rotating through all folds is computed
        # the first time __len__() is called:
        
        self.num_batches = None
        self.curr_split_idx = -1

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
            
            if num_samples == 0:
                raise ValueError("No samples to serve.")
            
            # Rounded-down number of samples that fit into each fold.
            # Having 34 samples with 3 folds, that is 34/3 == ~11

            samples_per_fold = num_samples // self.num_folds
            
            # For training we get 2 folds worth of samples,
            # with one fold held out: 11*2 = 22
            
            samples_per_split = samples_per_fold * (self.num_folds - 1)
            
            # As many permutations as there are folds: 3 * 22: 66
            
            total_train_samples = self.num_folds * samples_per_split
            
            # Convert to batches. Assume batch_size of 2:
            # 66 // 2 = 33
            
            self.total_num_batches = total_train_samples // self.batch_size
                        
            remainder_samples = total_train_samples % self.batch_size
            if not self.drop_last and remainder_samples > 0:
                # Add the final partially filled batch, 
                # if num_samples not a multiple of batches += 1
                self.total_num_batches += 1
            
        return self.total_num_batches

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
        
        # Loop over all splits (i.e. over all
        # configurations of which fold is for
        # validation.
        
        # Get one list of sample IDs that
        # covers all samples in one split.
        # And one list of sample IDs in the 
        # test split:
        for _i, (split_train_ids, split_test_ids) \
            in enumerate(self.sampler.get_split()):
            
            # Keep track of which split we are working
            # on. Needed only as info for client; not
            # used for logic in this method:
            
            self.curr_split_idx += 1
            
            # split_train_ids has all sample IDs
            # to use for training in this split. 
            # The split_test_ids holds the left-out
            # sample IDs to use for testing once 
            # the split_train_ids have been served out
            # one batch at a time.
            
            # Set this split's test ids aside for client
            # to retrieve via: get_split_test_sample_ids()
            # once they pulled all the batches of this
            # split:
            self.curr_test_sample_ids = split_test_ids
            
            # Create one batch:
            
            num_train_sample_ids = len(split_train_ids)
            num_batches = num_train_sample_ids // self.batch_size
            num_remainder_samples = num_train_sample_ids % self.batch_size
            batch_start_idx = 0
            
            # Create num_batches batches from the
            # training data of this split:
            
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
                # samples onwards in the train folds:
                
                batch_start_idx += self.batch_size
                
                # Put together next batch:
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
                
            # Let client know that all batches for one split
            # have been delivered by a None/None pair:

            yield (None, None)

            # Next split:
            continue

    #------------------------------------
    # get_curr_fold_idx 
    #-------------------
    
    def get_curr_fold_idx(self):
        return self.curr_split_idx


    #------------------------------------
    # get_split_test_sample_ids 
    #-------------------
    
    def get_split_test_sample_ids(self):
        try:
            return self.curr_test_sample_ids
        except:
            return None
    
    #------------------------------------
    # validation_samples 
    #-------------------
    
    def validation_samples(self):
        '''
        Generator that runs through every
        test sample_id of the current fold, 
        and feeds (<img_tensor, label) pairs.
        
           for (img_tensor, label) in my_bird_dataloader.validation_samples():
               <test model>
        '''
        
        for sample_id in self.get_split_test_sample_ids():
            yield self.dataset[sample_id]

    #------------------------------------
    # file_from_sample_id 
    #-------------------
    
    def file_from_sample_id(self, sample_id):
        '''
        Given a sample_id, return the absolute
        file path of the corresponding sample
        in the file system.
        
        We use the public dataset method.
        
        @param sample_id: sample ID to look up
        @type sample_id: int
        '''
        return self.dataset.file_from_sample_id(sample_id)

    #------------------------------------
    # set_epoch 
    #-------------------

    def set_epoch(self, new_epoch):
        '''
        Must be called by client every time
        a new epoch starts. The epoch number
        is used by the sampler to shuffle
        the dataset before beginning to draw
        samples.

        @param new_epoch: the epoch under which the dataloader
            is (re)started
        @type new_epoch: int
        '''
        self.sampler.set_epoch(new_epoch)

# -------------------- Multiprocessing Dataloader -----------

class MultiprocessingDataLoader(CrossValidatingDataLoader):
    '''
    Use this class for dataloader if running using
    multiple machines, or using multiple GPUs on a
    single machine.
    '''
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 dataset, 
                 batch_size=32,
                 prefetch_factor=2,
                 drop_last=False,
                 num_folds=10,
                 seed=42,
                 shuffle=True,
                 **kwargs
                 ):

        self.sampler = DistributedSKFSampler(
                dataset,
                num_folds=num_folds,
                seed=seed,
                shuffle=shuffle,
                drop_last=drop_last
                )

        super().__init__(dataset,
                         batch_size=batch_size,
                         num_folds=num_folds,
                         prefetch_factor=prefetch_factor,
                         pin_memory=True,
                         sampler=self.sampler,
                         **kwargs)
