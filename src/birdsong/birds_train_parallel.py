#!/usr/bin/env python
'''
Created on Jul 7, 2020 based heavily on
code by 
  o Jonathan Michael Gomes Selman,
  o Nikita-Girey Nechvet Demir, 
  o Leo Epstein Glikbarg
  o Amy C Dunphy

@author: paepcke
'''

import os, sys
packet_root = os.path.abspath(__file__.split('/')[0])
sys.path.insert(0,packet_root)

from _collections import OrderedDict
import argparse
import datetime
import json
import random  # Just so we can fix seed for testing
import signal
import socket

import GPUtil
from logging_service import LoggingService
from torch import cuda
from torch import hub
from torch import optim
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.resnet import ResNet, BasicBlock

from birdsong.cross_validation_dataloader import MultiprocessingDataLoader, CrossValidatingDataLoader
from birdsong.rooted_image_dataset import MultiRootImageDataset
import numpy as np
from birdsong.result_tallying import TrainResult, TrainResultCollection
import torch.distributed as dist
import torch.nn as nn

from birdsong.utils.dottable_config import DottableConfigParser
from birdsong.utils.learning_phase import LearningPhase


#from torch.nn import BCELoss
import faulthandler; faulthandler.enable()


# For parallelism:

# ------------------------ Specialty Exceptions --------

class NoGPUAvailable(Exception):
    pass

class TrainError(Exception):
    # Error in the train/validate loop
    pass

class InterruptTraining(Exception):
    # Used to handle cnt-C
    pass


class BirdTrainer(object):
    
    # Flag for subprocess to offer model saving,
    # and quitting. Used to handle cnt-C graciously:
    STOP = False
    
    # Device number for CPU (as opposed to GPUs, which 
    # are numbered as positive ints:
    CPU_DEV = -1

    SECS_IN_DAY = 24*60*60
    
    # Number of differences between accuracies from 
    # successive epochs that should be averaged to 
    # determine whether to continue training:
    
    EPOCHS_FOR_PLATEAU_DETECTION = 5

    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self,
                 config_info,
                 root_train_test_data=None,
                 batch_size=None,
                 checkpoint=None,   #******** load if given
                 logfile=None,
                 performance_log_dir=None,
                 testing_cuda_on_cpu=False,
                 started_from_launch=False,
                 unit_testing=False
                 ):

        self.curr_dir = os.path.dirname(os.path.abspath(__file__))

        # Initialize a config dict of dict with
        # the application's configurations. Sections
        # will be:
        #
        #   config['Paths']       -> dict[attr : val]
        #   config['Training']    -> dict[attr : val]
        #   config['Parallelism'] -> dict[attr : val]
        #
        # The config read method will handle config_info
        # being None. 
        #
        # If config_info is a string, it is assumed to
        # be a file with the configuration. Else it is
        # assumed to be a DottableConfigParser

        if isinstance(config_info, str):
            self.config = self.read_configuration(config_info)
        elif isinstance(config_info, DottableConfigParser):
            self.config = config_info
        else:
            print("Error: must have a config file. See config.cfg.Example in project root")
            sys.exit(1)

        # Replace None args with config file values:

        if logfile is None:
            logfile = self.config.getpath('Paths','logfile', 
                                          relative_to=self.curr_dir)
        if logfile is None:
            self.log = LoggingService()
        else:
            self.log = LoggingService(logfile=logfile)
        
        if root_train_test_data is None:
            root_train_test_data = self.config.getpath('Paths', 'root_train_test_data', 
                                                       relative_to=self.curr_dir)
        self.root_train_test_data = root_train_test_data
        
        if batch_size is None:
            batch_size = self.config.getint('Training', 'batch_size')
        self.batch_size = batch_size
        
        self.seed = self.config.getint('Training', 'seed')
        self.set_seed(self.seed)
        self.started_from_launch = started_from_launch
        self.setup_gpus()
        

        train_parms  = self.config['Training']
        
        # Not worth the hassle of juggling tensor
        # dimension to retrain the last batch if
        # it's not completely filled:
        self.drop_last = True
        
        # Install signal handler for cnt-C:
        # It will set the class var STOP,
        # which we check periodically during
        # training and evaluation. When seen
        # to be True, offers training state
        # save:

        signal.signal(signal.SIGINT, self.request_interrupt_training)

        #************
#         print(f"******WORLD_SIZE:{os.getenv('WORLD_SIZE')}")
#         print(f"******:RANK:{os.getenv('RANK')}")
#         print(f"******LOCAL_RANK:{os.getenv('LOCAL_RANK')}")
#         print(f"******:NODE_RANK:{os.getenv('NODE_RANK')}")
#         print(f"******:MASTER_ADDR:{os.getenv('MASTER_ADDR')}")
#         print(f"******:MASTER_PORT:{os.getenv('MASTER_PORT')}")
#         print("****** Exiting intentionally")
#         sys.exit()
        #************

        
        # Whether or not we are testing GPU related
        # code on a machine that only has a CPU.
        # Obviously: only set to True in that situation.
        # Don't use any computed results as real:
        
        self.testing_cuda_on_cpu = testing_cuda_on_cpu

        self.device = torch.device('cuda' if cuda.is_available() else 'cpu')
        self.cuda   = torch.device('cuda')
        self.cpu    = torch.device('cpu')
        
        dataset  = MultiRootImageDataset(self.root_train_test_data,
                                         sample_width=train_parms.getint('sample_width'),
                                         sample_height=train_parms.getint('sample_width')
                                         )
        self.num_folds = self.config.Training.getint('num_folds')
        
        # In the dataloader creation, make drop_last=True 
        # to skip the last batch if it is not full. Just 
        # not worth the trouble dealing with unusually 
        # dimensioned tensors:

        # Make an appropriate (single/multiprocessing) dataloader:
        if self.gpu_device == self.CPU_DEV:

            # CPU bound, single machine:
            
            self.dataloader = CrossValidatingDataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=False,
                prefetch_factor=2,
                drop_last=self.drop_last,
                shuffle=True,       # Shuffle underlying dataset at the outset (only)
                seed=self.seed,
                num_folds=self.num_folds
                )
        else:
            # GPUSs used, single or multiple machines:
            
            self.dataloader = MultiprocessingDataLoader(dataset,
                                                        shuffle=True,
                                                        seed=self.seed,
                                                        num_folds=self.num_folds,
                                                        drop_last=True,
                                                        batch_size=batch_size 
                                                            if batch_size is not None 
                                                            else train_parms.getint('batch_size')
                                                        )
        self.num_classes = len(dataset.class_id_list())

        # Resnet18 retain 6 layers of pretraining.
        # Train the remaining 4 layers with your own
        # dataset:
        self.model = self.get_resnet18_partially_trained(self.num_classes)

        if unit_testing:
            return

        # Optimizer
        self.optimizer = optim.SGD(self.model.parameters(), 
                                   lr=self.config.Training.getfloat('lr', 0.001), # Provide a default 
                                   momentum=self.config.Training.get('momentum', 0.9))
        if checkpoint:
            # Requested to continue training 
            # from a checkpoint:
            if not os.path.exists(checkpoint):
                msg = f"Requested continuation from checkpoint {checkpoint}, which does not exist."
                self.log.err(msg)
                print(msg)
                sys.exit(1)
            self.log.info("Loading checkpoint...")
            self.load_model_checkpoint(checkpoint, 
                                       self.model, 
                                       self.optimizer)
            self.log.info("Loading checkpoint done")
            self.log.info(f"Resume training at start of epoch {self.epoch}")
        else:
            self.epoch = None
            self._diff_avg = None
            self.tally_collection = None

        # Loss function:
        self.loss_func = nn.CrossEntropyLoss()
        
        # Scheduler:
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        
        # Move to GPU if available:
        self.to_best_device(self.model)

        self.log.info(f"Model resides on device: {self.device_residence(self.model)}")

        # Note: call to setup_logging must
        # be after checkpoint restoration above.
        # Else tally_collection will be overwritten:
        
        self.setup_logging(logfile, self.num_classes)
        self.setup_json_logging(self.config.Training.getint('kernel_size'), 
                                self.config.Training.getint('batch_size'),
                                json_log_dir=performance_log_dir
                                )
        
        if self.device == self.cuda:
            # A stack to allow clear_gpu() to
            # remove all tensors from the GPU.
            # Requires that code below always
            # pushes and pops tensors when they
            # are moved to, and removed from the GPU:
            #
            #   new_my_tensor = self.tensor_push(my_tensor)
            #   new_my_tensor = self.tensor_pop()
            
            
            self.gpu_tensor_stack = []
        else:
            # Stack used instead of the gpu stack.
            # It is part a no-op in that it allows
            # tensor_push() and tensor_pop() to be
            # used even when no GPU is available.
            # In that case, clear_gpu() will do nothing:
            
            self.cpu_tensor_stack = []

    #------------------------------------
    # setup_gpus
    #-------------------
    
    def setup_gpus(self):

        try:
            self.local_rank = int(os.environ['LOCAL_RANK'])
        except KeyError:
            # We were not called via the launch.py script.
            # Check wether there is at least one 
            # local GPU, if so, use that:
            if len(GPUtil.getGPUs()) > 0:
                self.local_rank = 0
            else:
                self.local_rank = None

        # The following call also sets self.gpu_obj
        # to a GPUtil.GPU instance, so we can check
        # on the GPU status along the way:
        
        self.gpu_device = self.enable_GPU(self.local_rank)
        if self.gpu_device != self.CPU_DEV and \
            self.local_rank is not None:
            self.setup_gpu(self.started_from_launch)
        else:
            # No GPU on this machine. Consider world_size
            # to be 1, which is the CPU struggling along:
            self.world_size = 1



    #------------------------------------
    # find_available_torch_backend 
    #-------------------
    
    def find_available_torch_backend(self):
        '''
        For parallel operation, torch needs a backend
        compiled into the torch package binaries. Options
        are nccl, mpi, and gloo. Return a string with
        the name of one of the backend names. If none are 
        available, raise NotImplementedError
        
        @return Name of torch backend
        @rtype: str
        @raise NotImplementedError 
        '''
        
        if dist.is_mpi_available():
            backend = 'mpi'
        elif dist.is_nccl_available():
            backend = 'nccl'
        elif dist.is_gloo_available():
            backend = 'gloo'
        else:
            raise NotImplementedError("None of mpi/nccl/gloo torch backends installed.")
        return backend



    #------------------------------------
    # to_best_device 
    #-------------------
    
    def to_best_device(self, item):
        if self.device == torch.device('cuda'):
            item.to(device=self.cuda)
        else:
            item.to(device=self.cpu)


    #------------------------------------
    # init_multiprocessing 
    #-------------------

    def init_multiprocessing(self,
                             master_addr,
                             master_port,
                             node_rank,
                             world_size
                             ):
        if self.node_rank == 0:
            self.log.info(f"Awaiting {self.world_size} nodes to run...")
        else:
            self.log.info("Awaiting master node's response...")
            
        if dist.is_mpi_available():
            backend = 'mpi'
        elif dist.is_nccl_available():
            backend = 'nccl'
        elif dist.is_gloo_available():
            backend = 'gloo'
        else:
            raise NotImplementedError("None of mpi/nccl/gloo torch backends installed.")
    
        #************
        # Master port seems to:
        #   1. not picked up from the URL below, and
        #   2. magically disappear from the environment
        #      before this method is called, and is 
        #      replaced with an empty string.
        # So store the port in the env var again:
        
        if len(os.environ.get('master_port', '')) == 0:
            os.environ['master_port'] = master_port
        dist.init_process_group(backend,
                                init_method=f'env://?world_size={world_size}&rank={node_rank}&master_port={master_port}'
                                ) 
#         dist.init_process_group(backend,
#                                 world_size=1,
#                                 rank=0,
#                                 init_method=f'env://'
#                                 ) 
        #************
        self.log.info("And we're off!")

    #------------------------------------
    # enable_GPU 
    #-------------------

    def enable_GPU(self, local_rank, raise_gpu_unavailable=True):
        '''
        Returns the device_residence id (an int) of an 
        available GPU. If none is exists on this 
        machine, returns self.CPU_DEV (-1).
        
        Initializes self.gpu_obj, which is a CUDA
        GPU instance. 
        
        Initializes self.cuda_dev: f"cuda:{device_id}"
        
        @param local_rank: which GPU to use. Created by the launch.py
            script. If None, just look for the next available GPU
        @type local_rank: {None|int|
        @param raise_gpu_unavailable: whether to raise error
            when GPUs exist on this machine, but none are available.
        @type raise_gpu_unavailable: bool
        @return: a GPU device_residence ID, or self.CPU_DEV
        @rtype: int
        @raise NoGPUAvailable: if exception requested via 
            raise_gpu_unavailable, and no GPU is available.
        '''

        self.gpu_obj = None

        # Get the GPU device_residence name.
        # Could be (GPU available):
        #   device_residence(type='cuda', index=0)
        # or (No GPU available):
        #   device_residence(type=self.CPU_DEV)

        gpu_objs = GPUtil.getGPUs()
        if len(gpu_objs) == 0:
            return self.CPU_DEV

        # GPUs are installed. Did caller ask for a 
        # specific GPU?
        
        if local_rank is not None:
            # Sanity check: did caller ask for a non-existing
            # GPU id?
            num_gpus = len(GPUtil.getGPUs())
            if num_gpus < local_rank + 1:
                # Definitely an error, don't revert to CPU:
                raise NoGPUAvailable(f"Request to use GPU {local_rank}, but only {num_gpus} available on this machine.")
            cuda.set_device(local_rank)
            # Part of the code uses self.cuda_dev
            self.cuda_dev = local_rank
            return local_rank
        
        # Caller did not ask for a specific GPU. Are any 
        # GPUs available, given their current memory/cpu 
        # usage? We use the default maxLoad of 0.5 and 
        # maxMemory of 0.5 as OK to use GPU:
        
        try:
            # If a GPU is available, the following returns
            # a one-element list of GPU ids. Else it throws
            # an error:
            device_id = GPUtil.getFirstAvailable()[0]
        except RuntimeError:
            # If caller wants non-availability of GPU
            # even though GPUs are installed to be an 
            # error, throw one:
            if raise_gpu_unavailable:
                raise NoGPUAvailable("Even though GPUs are installed, all are already in use.")
            else:
                # Else quietly revert to CPU
                return self.CPU_DEV
        
        # Get the GPU object that has the found
        # deviceID:
        self.gpu_obj_from_devid(device_id)
        
        # Initialize a string to use for moving 
        # tensors between GPU and cpu with their
        # to(device_residence=...) method:
        self.cuda_dev = device_id 
        return device_id 

    #------------------------------------
    # gpu_obj_from_devid 
    #-------------------
    
    def gpu_obj_from_devid(self, devid):

        for gpu_obj in GPUtil.getGPUs():
            if gpu_obj.id == devid:
                self.gpu_obj = gpu_obj
                break
        return self.gpu_obj



#     #------------------------------------
#     # zero_tallies 
#     #-------------------
#     
#     def zero_tallies(self, curr_tallies=None):
#         '''
#         Returns a dict with the running 
#         tallies that are maintained in both
#         train_one_split and eval_epoch. If curr_tallies
#         is provided, it is the dict of running values
#         from the previous epoch. In that case, the 
#         new tallies is zeroed for all entries, but
#         the 'running_loss' is carried over from
#         the old tallies.
#         
#         @param curr_tallies: if given, the tallies from
#             a current epoch
#         @type curr_tallies: {str : float}
#         @return: dict with float zeros
#         @rtype: {str : float}
#         '''
#         tallies = {'running_loss' : 0.0,
#                    'running_corrects' : 0.0,
#                    'running_samples' : 0.0,
#                    # True positives
#                    'running_tp' : 0.0,
#                    # True positives + false positives
#                    'running_tp_fp' : 0.0,
#                    # True positives + false negatives
#                    'running_tp_fn' : 0.0,
#                    # For focal loss purposes
#                    'running_true_non_zero' : 0.0
#                    }
#         if curr_tallies is not None:
#             # Carry over the accumulated loss:
#             tallies['running_loss'] = curr_tallies['running_loss']
#         return tallies

    #------------------------------------
    # tally_result
    #-------------------

    def tally_result(self,
                     split_num, 
                     labels_tns, 
                     pred_prob_tns,
                     loss,
                     learning_phase,
                     ):
        '''
        Given:
           o the results of a train or validation run
             through all batches of one split (i.e. configuration
             of folds into train + 1-validate,
           o the true labels for each sample
           o the total accumulated loss in this split
           o the phase of the process (train/validate/test)
           
        Creates and retains a confusion matrix from 
        which all other values, such as precision/recall
        can be derived in the related methods of the 
        TrainResult class. The matrix is handed to a
        new TrainResult instance, and that instance is
        added to self.tally_collection.
        
        The confusion matrix's row indices will correspond
        the true class IDs; the column indices correspond to
        the predicted classes:
        
                              pred class0 pred class2 pred class3 
             known class 0      num-pred    num_pred     num_pred
             known class 1      num_pred    num_pred     num_pred
             known class 2         ...        ...          ...
        
        As passed in, pred_prob_tns is:
        
             [num_batches x batch_size x num_classes]
                  ^             ^            ^
                  |             |            |
            all batches      result for    logits for each
            from a single   each sample    target class
          split == num_folds  in one batch
        
        This method resolves that stack to:
        
             [num_batches x batch_size]
                   ^             ^
                   |             |
         [[predicted_class_sample1_batch1, predicted_class_sample2_batch1],
         [[predicted_class_sample1_batch2, predicted_class_sample2_batch2],
                          ...
                 as many rows as there are
              folds to permute in the cross-val
                
                predicted_classes = torch.argmax(output_stack, dim=2)

        The sklearn conf_matrix method is then used.
        
        Logging Results: after the validation of each split
            is complete, means of results for the split are
            logged. Since this method is called both after
            one split's training, and again after that split's
            validation, logging is delayed until a tally for
            the validation phase is being created.

        @param labels_tns: ground truth labels for batch
        @type labels_tns: torch.Tensor (integer class IDs)
        @param pred_prob_tns: model prediction probabilities
        @type pred_prob_tns: torch.Tensor
        @param loss: accumulated result of loss function from
            run through all the folds.
        @type loss: torch.Tensor (float)
        @param learning_phase: phase under which to record this
            result: training, validation, or test.
        @type learning_phase: LearningPhase
        '''
        
        # Predictions are for one batch. Example for
        # batch_size 2 and 4 target classes:
        #  
        #    torch.tensor([[0.1, -0.2,  .3,  .42], |
        #                                          | <--- one batch's probs
        #                  [.43,  .3, -.23, -.18]  |   for batch_size == 2 
        #
        #                  ])
        
        if learning_phase == LearningPhase.TRAINING:
            pred_classes_each_batch = torch.argmax(pred_prob_tns, dim=2)
        else:
            pred_classes_each_batch = torch.argmax(pred_prob_tns, dim=1)
        
        #  For a batch size of 2 we would now have:
        #                 tensor([[3, 0],  <-- result batch 1 above
        #                         [0, 0],
        #                         [0, 0],
        #                         [0, 0],
        #                          ...
        #    num_fold'th row:     [2, 1]]

        # Conf-matrix only handles flat pred/label inputs.
        # The current shape is:
        # Tensor: tensor([[0, 0],
        #                 [0, 0],
        #                 [0, 0],
        #                 [1, 1]]
        predicted_class_ids = pred_classes_each_batch.flatten()
        truth_labels        = labels_tns.flatten()

        # Find labels that were incorrectly predicted:
        badly_predicted_labels = truth_labels[truth_labels != predicted_class_ids]

        tally = TrainResult(split_num, 
                            self.epoch, 
                            learning_phase, 
                            loss, 
                            predicted_class_ids,
                            truth_labels,
                            self.num_classes,
                            badly_predicted_labels=badly_predicted_labels
                            )
        
        self.tally_collection.add(tally)

        return tally

    #------------------------------------
    # record_json_display_results 
    #-------------------
    
    def record_json_display_results(self, epoch):
        '''
        Amy Dunphy's confusion matrix heatmaps
        that are labeled at the edges with class
        names needs a particularly formated json
        file. Write one epoch's worth of performance
        summaries to that file.
        
        Assumption: setup_json_logging() was called
           before the call to this method.
        '''

        # For json logging: get file names 
        # of samples that were incorrectly 
        # identified; just get the basenames:
        
        # Get the sample_ids that were misclassified
        # in each of this epoch's training splits:
        curr_epoch_misses = []
        [curr_epoch_misses.extend(split_tallies.badly_predicted_labels) 
            for split_tallies
             in self.tally_collection.tallies(epoch=self.epoch)
             ]
        # We have from a above a list of individual
        # sample_ids, each stuck in a tensor. Pull
        # them out:
        
        epoch_misses = [sample_id_tensor.item() for sample_id_tensor in curr_epoch_misses]

        # Get rid of duplicate samples:
        epoch_misses_unique = set(epoch_misses)
        
        
        # Get the basenames of the files that
        # correspond to each of the mis-classified
        # samples. 
        
        incorrect_paths = []
        for sample_id in epoch_misses_unique:
            absolute_file_path = self.dataloader.file_from_sample_id(sample_id)
            incorrect_paths.append(os.path.basename(absolute_file_path))

        # Mean of accuracies among the 
        # training splits of this epoch
        mean_accuracy_training = \
           self.tally_collection.mean_accuracy(epoch, 
                                               learning_phase=LearningPhase.TRAINING)
           
        mean_accuracy_validating = \
           self.tally_collection.mean_accuracy(epoch, 
                                               learning_phase=LearningPhase.VALIDATING)

        epoch_loss = self.tally_collection.cumulative_loss(epoch=epoch, 
                                                           learning_phase=LearningPhase.VALIDATING)
        epoch_mean_weighted_precision = \
            self.tally_collection.mean_weighted_precision(epoch,
                                                          learning_phase=LearningPhase.VALIDATING
                                                          )
            
        epoch_mean_weighted_recall = \
            self.tally_collection.mean_weighted_recall(epoch,
                                                       learning_phase=LearningPhase.VALIDATING
                                                       )

        # For the confusion matrix: add all 
        # the confusion matrices from Validation
        # runs:
        epoch_conf_matrix = self.tally_collection.conf_matrix_aggregated(epoch=epoch,
                                                                         learning_phase=LearningPhase.VALIDATING
                                                                         )
        with open(self.json_filepath, 'a') as f:
            f.write(json.dumps(
                    [epoch, 
                     epoch_loss,
                     mean_accuracy_training,
                     mean_accuracy_validating,
                     epoch_mean_weighted_precision,
                     epoch_mean_weighted_recall,
                     incorrect_paths,
                     epoch_conf_matrix.tolist()]) + "\n")

    #------------------------------------
    # train 
    #-------------------

    def train(self):

        self.log.info("Begin training")
        
        if self.device == self.cuda:
            self.initial_GPU_memory = cuda.memory_allocated(self.device)
            # Reset statistics on max GPU memory use:
            cuda.reset_max_memory_allocated() 

        # If restarting from checkpoint:
        # use the checkpointed epoch number:
        self.epoch = 0 if self.epoch is None else self.epoch
        
        # Similarly with self._diff_avg:
        self._diff_avg   = 100 if self._diff_avg is None else self._diff_avg
        
        time_start = datetime.datetime.now()
        num_folds  = self.config.Training.getint('num_folds')

        # We will keep track of the average accuracy
        # over the splits of each epoch. The accuracies
        # dict will key off an epoch number, and hold as
        # value the mean of accuracies among the splits
        # of the respective epoch. Use ordered dict so that
        # successively entered epochs will stay in order:
        
        accuracies = OrderedDict() 

        try: # This try ensures cleanup via the 
            #  finally expression at the end
            # Number of seconds between showing 
            # status if verbose; default is 5 seconds:
            alive_pulse = self.config.Training.getint('show_alive', 5)
            try: # This try manages cnt-C interrupts, and therefore
                #  saving of checkpoints.
                
                # Run epochs until either the max number of
                # epochs have occurred, or accuracy has not
                # improved beyond 0.05 during the past
                # self.EPOCHS_FOR_PLATEAU_DETECTION epochs:
                  
                while (self._diff_avg >= 0.05 or \
                       self.epoch <= self.config.Training.getint('min_epochs')
                       ) and \
                       self.epoch <= self.config.Training.getint('max_epochs'):
                    self.epoch += 1
                    split_num = 0
                    
                    # Get an iterator over all the 
                    # splits from the dataloader. 
                    # Since train_one_split() is a generator
                    # function, the first call here returns
                    # a generator object rather than running
                    # the method:
    
                    split_training_iterator = self.train_one_split()
                    
                    # Send the initial split_num to the
                    # generator; it is hanging to receive
                    # this near the top of train_one_split():
                    
                    split_training_iterator.send(None)
    
                    for split_num in range(num_folds):
                        self.optimizer.zero_grad()
                        try:
                            # train_one_split() is waiting for
                            # the next split_num in the second
                            # yield statement of that method:
                            split_training_iterator.send(split_num)
                        except StopIteration:
                            # Exhausted all the splits
                            self.log.warn(f"Unexpectedly ran out of splits.")
                            break
                        self.validate_one_split(split_num)
    
                        # Time for sign of life?
                        time_now = datetime.datetime.now()
                        if self.time_diff(time_start, time_now) >= alive_pulse:
                            self.log.info (f"Epoch {self.epoch}; Split number {split_num} of {num_folds}")
    
                    # Done with one epoch:
    
                    self.log.info(f"Finished epoch {self.epoch}")
                    
                    # Update the json based result record
                    # in the file system:
                    
                    self.record_json_display_results(self.epoch)
                    
                    # Compute mean accuracy over all splits
                    # of this epoch:
                    accuracies[self.epoch] = self.tally_collection.mean_accuracy(epoch=self.epoch, 
                                                                                 learning_phase=LearningPhase.TRAINING)
    
                    if self.epoch <= self.config.Training.getint('min_epochs'):
                        # Haven't trained for the minimum of epochs:
                        continue
                    
                    if self.epoch < self.EPOCHS_FOR_PLATEAU_DETECTION:
                        # Need more epochs to discover accuracy plateau:
                        continue
    
                    # Compute the mean over successive accuracy differences
                    # after the past few epochs. Where 'few' means
                    # EPOCHS_FOR_PLATEAU_DETECTION. The '1+' is needed
                    # because our epochs start at 1:
    
                    past_accuracies = [accuracies[epoch] for epoch 
                                       in range(1+len(accuracies)-self.EPOCHS_FOR_PLATEAU_DETECTION,
                                                self.epoch+1
                                                )]
                    self._diff_avg = self.mean_diff(torch.tensor(past_accuracies))
    
    
            except (KeyboardInterrupt, InterruptTraining):
                self.log.info("Early stopping due to keyboard intervention")
                do_save = self.offer_model_save()
                if do_save in ('y','Y','yes','Yes', ''):
                    have_fname = False
                    while not have_fname:
                        dest_path = input("Destination file (ex. ~/tmp/chpt.pth; $HOME/chpt.pth): ")
                        # Resolve '~' and environ vars: 
                        dest_path = os.path.expandvars(os.path.expanduser(dest_path))
                        dest_dir  = os.path.dirname(dest_path)
                        # Create any intermediate dirs:
                        if not os.path.exists(dest_dir):
                            os.makedirs(dest_dir)
                        if os.path.exists(dest_path):
                            have_fname = input("File exists, replace?") \
                              in ('y','Y','yes','Yes')
                        break
                    self.save_model_checkpoint(dest_path,
                                               self.model,
                                               self.optimizer
                                               )
                self.log.info(f"Training state saved to {dest_path}")
                self.clear_gpu()
                self.log.info("Exiting")
                sys.exit(0)

        finally:
            if self.device == self.cuda:
                self.ending_GPU_memory = cuda.memory_allocated(self.device)
                if self.ending_GPU_memory >= self.initial_GPU_memory:
                    self.log.err(f"Application did not release {self.ending_GPU_memory - self.initial_GPU_memory}")
                if len(self.gpu_tensor_stack) > 0:
                    self.log.err(f"Application leaves {len(self.gpu_tensor_stack)} tensors on GPU")
                if len(self.cpu_tensor_stack) > 0:
                    self.log.warn(f"Application used CPU, but left {len(self.cpu_tensor_stack)} tensors from push/pop regimen")
    
                self.log.info(f"Max GPU memory use: {cuda.max_memory_allocated(self.device)} bytes")
                self.clear_gpu() 

        self.log.info("Training finished")


    #------------------------------------
    # train_one_split 
    #-------------------

    def train_one_split(self): 
        '''
        This is a *generator* function, which 
        trains from one split, then yields for
        the next split_num. 
        
        Usage:
             gen = self.train_one_split()
             gen.send(None)
             while True:
                 gen.send(new_split_num)
                 
             ... till this method raises StopIteration
             
        The first gen.send(None) causes the first 
        yield statement below. All subsequent yields
        will occur in the second yield. 
        '''
        
        # Set model to train mode:
        self.model.train(True)
        output_stack = None
        label_stack  = None
        
        loss = 0.0
        
        # Await the first real call,
        # which will be via a send(split_num): 
        split_num = yield
        
        # While loop will be exited when
        # dataloader has exhausted all batches
        # from each fold from every split:
        
        for _i, (batch,targets) in enumerate(self.dataloader):

            if batch is None:
                # Exhausted this split. Record this split's
                # training results, and yield till
                # the next split's worth of training
                # is requested by the caller:
                
                self.tally_result(split_num,
                                  label_stack, 
                                  output_stack,
                                  loss, 
                                  LearningPhase.TRAINING)
                
                # Sit until next() is called on this 
                # generator again. At that point we continue feeding
                # from the same dataloader in this while
                # loop. The assignment receives a new 
                # split_num and epoch_loss via the 'send()'
                # function when train() calls next() on this
                # function:
                 
                split_num = yield
                
                # Back for a new split. Prepare:
                # Set model to train mode:
                self.model.train(True)
                output_stack = None
                label_stack  = None
                # ... and continue in the while loop,
                # using the continuing batch_feeder
                continue

            # Got another batch/targets pair:
            # Push sample and target tensors
            # to where the model is:
            batch   = self.push_tensor(batch)
            targets = self.push_tensor(targets)
            
            # Outputs will be on GPU if we are
            # using one:
            outputs = self.model(batch)
            
            loss = self.loss_func(outputs, targets)
            
            # Update the accumulating training loss
            
            # for this epoch: 
            self.tally_collection.add_loss(self.epoch, 
                                           loss,
                                           LearningPhase.TRAINING
                                           )
            loss.backward()
            self.optimizer.step()
            
            # Free GPU memory:
            targets = self.pop_tensor()
            batch   = self.pop_tensor()
            
            outputs = outputs.to('cpu')
            

            # Remember performance:
            if output_stack is None:
                output_stack = torch.unsqueeze(outputs, dim=0)
            else:
                # Did we get a full batch? Output tensors
                # of shape [batch_size, num_classes]. 
                #     [[1st_of_batch, [logit_class1, logit_class2, ... logit_num_classes]]
                #      [2nd_of_batch, [logit_class1, logit_class2, ... logit_num_classes]]
                #     ]
                have_full_batch = outputs.size() == torch.Size([self.batch_size, 
                                                                self.num_classes])
                if not have_full_batch and self.drop_last:
                    # Done with this split:
                    # Pending STOP request?
                    if BirdTrainer.STOP:
                        raise InterruptTraining()
                    continue
                    
                output_stack = torch.cat((output_stack, torch.unsqueeze(outputs, 
                                                                        dim=0))) 
            if label_stack is None:
                label_stack = torch.unsqueeze(targets, dim=0)
            else:
                label_stack = torch.cat((label_stack, torch.unsqueeze(targets, 
                                                                      dim=0)))
            # Pending STOP request?
            if BirdTrainer.STOP:
                raise InterruptTraining()


    #------------------------------------
    # validate_one_split 
    #-------------------

    def validate_one_split(self, split_num):
        # Model to eval mode:
        self.model.eval()
        with torch.no_grad():
            for (val_sample, val_target) in self.dataloader.validation_samples():
                # Push sample and target tensors
                # to where the model is; but during
                # validation we get only a single sample
                # at a time from the loader, while the 
                # model input expects batches. Turn sample
                # and target tensors to single-element batches:
                val_sample = val_sample.unsqueeze(dim=0).to(self.device_residence(self.model))
                val_target = val_target.unsqueeze(dim=0).to(self.device_residence(self.model))

#                 if (idx % 250 == 0) and self.config.Training.verbose:
#                     self.log.info (f"Batch number {idx} of {len(self.dataloader)}")
#                     self.log.info (f"Total correct predictions {self.tallies['running_tp']}, Total true positives {self.tallies['running_tp']}")
                
                # Pending STOP request?
                if BirdTrainer.STOP:
                    raise InterruptTraining()
                pred_prob_tns = self.model(val_sample)

                # Pending STOP request?
                if BirdTrainer.STOP:
                    raise InterruptTraining()
        
#                 # The Binary Cross Entropy function wants 
#                 # equal datatypes for prediction and targets:
#                 
#                 val_target = val_target.float()
                loss =  self.loss_func(pred_prob_tns, val_target)
                self.tally_collection.add_loss(self.epoch, 
                                               loss,
                                               LearningPhase.VALIDATING
                                               )

                # Free GPU memory:
                val_sample = val_sample.to('cpu')
                val_target = val_target.to('cpu')
                pred_prob_tns = pred_prob_tns.to('cpu')
                self.tally_result(split_num,
                                  val_target, 
                                  pred_prob_tns, 
                                  loss, 
                                  LearningPhase.VALIDATING
                                  )

    # ------------- Utils -----------

    #------------------------------------
    # request_interrupt_training
    #-------------------

    def request_interrupt_training(self, signum, stacktrace):
        '''
        Interrupt handler for cnt-C (SIGINT). 
        Set STOP flag. Training checks that flag
        periodically, and raises the InterruptTraining
        exception. That in turn asks user whether to
        save current training state for later resumption
        '''
        self.log.info("Requesting training interruption; waiting for clean point to interrupt...")
        BirdTrainer.STOP = True

    #------------------------------------
    # offer_model_save
    #-------------------

    def offer_model_save(self):
        do_save = input("Save partially trained model? (Y/n): ")
        return do_save
    
    #------------------------------------
    # get_lr 
    #-------------------
    
    def get_lr(self, scheduler):
        '''
        Given a scheduler instance, return its
        current learning rate. All schedulers but
        one have a get_lr() method themselves. But
        the auto-lr-reducing ReduceLROnPlateau does
        not. It maintains multiple learning rates,
        one for each 'group'. So the result of a
        get_lr() method would be a list, which would
        be contrary to all other schedulers. This
        method masks the difference. 
        
        We only have one group, so we return the
        first (and only) lr if the scheduler is an
        instance of the problem class.
        
        
        @param scheduler: scheduler whose learning rate 
            is to be retrieved
        @type scheduler:torch.optim.lr_scheduler
        @return: the scheduler's current learning rate
        @rtype: float
        '''
        
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            learning_rates = [ group['lr'] for group in scheduler.optimizer.param_groups ]
            lr = learning_rates[0]
        else:
            lr = scheduler.get_lr()
        return lr

    #------------------------------------
    # get_prec_rec_components
    #-------------------
    
    def get_prec_rec_components(self, pred_tns, label_tns):
        '''
        Given tensor of finalized predictions (0s and 1s),
        and the label tensor from the dataloader, return
        numbers of true positive, the sum of true positive
        and false positive, and the sum of true positive,
        and false negatives (tp, tp_fp, tp_fn)
        
        @param pred_tns: tensors of 1.0 and 0.0 stacked batch_size high
        @type pred_tns: torch.Tensor [batch_size, 1]
        @param label_tns: 1.0 or 2.0 true labels stacked batch_size high
        @type label_tns: torch.Tensor [batch_size, 1]
        @return: precision values tp, tp_fp, tp_fn
        @rtype: [int,int,int]
        '''
        
        # Get the values from tensor [batch_size, label_val]:
        label_vals = label_tns.view(-1)
        tp         = torch.sum(label_vals)
        pred_pos = torch.sum(pred_tns)
        fp = max(0, pred_pos - tp)
        fn = max(0, tp - pred_pos)
        tp_fp = tp + fp
        tp_fn = tp + fn
        
        return (tp, tp_fp, tp_fn)

    #------------------------------------
    # json_log_filename 
    #-------------------
    
    def json_log_filename(self):
        return self.json_filepath

    #------------------------------------
    # set_seed  
    #-------------------

    def set_seed(self, seed):
        '''
        Set the seed across all different necessary platforms
        to allow for comparison of different models and runs
        
        @param seed: random seed to set for all random num generators
        @type seed: int
        '''
        torch.manual_seed(seed)
        cuda.manual_seed_all(seed)
        # Not totally sure what these two do!
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed = seed

    #------------------------------------
    # get_resnet18_partially_trained 
    #-------------------

    def get_resnet18_partially_trained(self, num_classes, num_layers_to_retain=6):
        
        model = hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=False)
        
        # Freeze the bottom num_layers_to_retain layers:
        for (i, child) in enumerate(model.children()):
            for param in child.parameters():
                param.requires_grad = False
            if i >= num_layers_to_retain:
                break
        num_in_features = model.fc.in_features
        
        model.fc = nn.Linear(num_in_features, num_classes)
        
        # Create a property on the model that 
        # returns the number of output classes:
        model.num_classes = model.fc.out_features
        return model

    #------------------------------------
    # save_model_checkpoint 
    #-------------------
    
    def save_model_checkpoint(self, dest_path, model, optimizer):
        '''
        Save the entire current state of training.
        Re-instate that checkpoint in a new session
        by using self.load_model_checkpoint()
        
        @param dest_path: absolute path where to save the 
            pickled model. Use .pth or .tar extension
        @type dest_path: str
        @param model: the model instance to save
        @type model: torch model class
        @param optimizer: the optimizer instance to save
        @type optimizer: torch optimizer class
        @param epoch: the last finished epoch 
        @type epoch: int
        '''
        
        # Remove results from the current
        # epoch from tally_collection:
        
        clean_tally_collection = TrainResultCollection.create_from(self.tally_collection)
        for epoch, split_num, learning_phase in self.tally_collection.keys():
            if epoch == self.epoch:
                clean_tally_collection.pop((epoch,split_num,learning_phase))

        to_save = {'epoch' : self.epoch,
                   'model_state_dict' : model.state_dict(),
                   'optimizer_state_dict' : optimizer.state_dict(),
                   '_diff_avg' : self._diff_avg,
                   'tallies' : clean_tally_collection
                   }
        torch.save(to_save, dest_path)
        
    #------------------------------------
    # load_model_checkpoint 
    #-------------------
    
    def load_model_checkpoint(self, src_path, fresh_model, fresh_optimizer):
        '''
        Restore a model previously saved using
        save_model_checkpoint() such that training 
        can resume.
        
        @param src_path: path where the app_state is stored
        @type src_path: str
        @param fresh_model: a new, uninitialized instance of the model used
        @type fresh_model: torch model
        @param fresh_optimizer: a new, uninitialized instance of the optimizer used
        @type fresh_optimizer: torch optimizer
        @return: dict with the initialized model, 
            optimizer, last completed epoch, most recent
            loss measure, and tallies collection.
        @rtype {str : *}
        '''

        app_state = torch.load(src_path)
        fresh_model.load_state_dict(app_state['model_state_dict'])
        fresh_optimizer.load_state_dict(app_state['optimizer_state_dict'])
        self.epoch     = app_state['epoch']
        self._diff_avg = app_state['_diff_avg']
        self.tally_collection = app_state['tallies']

    #------------------------------------
    # read_configuration 
    #-------------------
    
    def read_configuration(self, conf_file):
        '''
        Parses config file that describes training parameters,
        various file paths, and how many GPUs different machines have.
        Syntax follows Python's configfile package, which includes
        sections, and attr/val pairs in each section.
        
        Expected sections:

           o Paths: various file paths for the application
           o Training: holds batch sizes, number of epochs, etc.
           o Parallelism: holds number of GPUs on different machines
        
        For Parallelism, expect entries like:
        
           foo.bar.com  = 4
           127.0.0.1    = 5
           localhost    = 3
           172.12.145.1 = 6
           
        Method identifies which of the entries is
        'localhost' by comparing against local hostname.
        Though 'localhost' or '127.0.0.1' may be provided.
        
        Returns a dict of dicts: 
            config[section-names][attr-names-within-section]
            
        Types of standard entries, such as epochs, batch_size,
        etc. are coerced, so that, e.g. config['Training']['epochs']
        will be an int. Clients may add non-standard entries.
        For those the client must convert values from string
        (the type in which values are stored by default) to the
        required type. This can be done the usual way: int(...),
        or using one of the configparser's retrieval methods
        getboolean(), getint(), and getfloat():
        
            config['Training'].getfloat('learning_rate')
        
        @param other_gpu_config_file: path to configuration file
        @type other_gpu_config_file: str
        @return: a dict of dicts mirroring the config file sections/entries
        @rtype: dict[dict]
        @raises ValueErr
        @raises TypeError
        '''
        
        if conf_file is None:
            return self.init_defaults()
        
        config = DottableConfigParser(conf_file)
        
        if len(config.sections()) == 0:
            # Config file exists, but empty:
            return(self.init_defaults(config))

        # Ensure that this machine's node entry is
        # 'localhost'. It is legal in the config file
        # to use 'localhost', '127.0.0.1', or the hostname's FQDN as
        # key in the config file for local host.
        # Get name of this machine. 
        
        my_hostname = socket.gethostname().split('.')[0]
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("google.com",80))
        
        # Get something like ('172.24.75.114', 44572)  
        my_global_ip_addr = s.getsockname()[0] 
    
        # Find whether some version of this host
        # is referenced in the dict; if it's anything
        # other than 'localhost', replace the entry
        # with 'localhost' as key. Must use a copy of 
        # the config dict, b/c we modify config in the
        # loop:
        
        for node_name_or_addr in config.copy()['Parallelism'].keys():
            if node_name_or_addr   == '127.0.0.1' or\
                node_name_or_addr.split('.')[0] == my_hostname or\
                node_name_or_addr  == my_global_ip_addr:
                
                # Create entry for localhost from hostname or IP
                # entry for the local host:
                config['Parallelism']['localhost'] = config[node_name_or_addr]

                # Remove the original entry (hostname or IP):
                del config['Parallelism'][node_name_or_addr]

        # If current host not in the config file at all,
        # create an entry that declares all detected GPUs on
        # this machine as usable:
        
        if 'localhost' not in config['Parallelism'].keys():
            config['Parallelism']['localhost'] = len(GPUtil.getGPUs())
    
        # Values should all be ints: number of GPUs. Ensure
        # that they are, and convert them from str to int
        # in the config dict:
        for (node, num_gpus) in config['Parallelism'].items():
            try:
                num_gpus = int(num_gpus)
            except ValueError:
                print(f"Number of GPUs at node {node} in config file {conf_file} should be an int.")
                sys.exit(1)
            config['Parallelism'][node] = num_gpus
            
        # Do type conversion also in other entries that 
        # are standard:
        
        types = {'epochs' : int,
                 'batch_size' : int,
                 'kernel_size' : int,
                 'sample_width' : int,
                 'sample_height' : int,
                 'seed' : int,
                 'pytorch_comm_port' : int,
                 
                 'root_train_test_data': str,
                 'net_name' : str,
                 }
        for section in config.sections():
            for attr_name in config[section].keys():
                try:
                    str_val = config[section][attr_name]
                    required_type = types[attr_name]
                    config[section][attr_name] = required_type(str_val)
                except KeyError:
                    # Current attribute is not standard;
                    # users of the corresponding value need
                    # to do their own type conversion when
                    # accessing this configuration entry:
                    continue
                except TypeError:
                    raise ValueError(f"Config file error: {section}.{attr_name} should be convertible to {required_type}")
    
        return config


    #------------------------------------
    # init_defaults
    #-------------------
    
    def init_defaults(self, configparser_obj):
        '''
        Used when no config file is provided. 
        Initialize the standard entries.
        
        Though not having the root of the train/test
        samples should probably be an error.
        
        @param configparser_obj: an empty configparser instance
        @type configparser_obj: configparser
        @return the same config parser object filled-in 
        @rtype configparser
        '''
        sec_paths = {
            'root_train_test_data' : '/home/data/birds/recombined_data/'
            }

        sec_training = {
            'net_name'      : 'resnet18',
            'epochs'        : 5,
            'batch_size'    : 32,
            'seed'          : 42,
            'kernel_size'   : 7,
            'sample_width'  : 400,
            'sample_height' : 400
            }

        sec_parallelism = {
            'pytorch_comm_port'  : 29920,
            'localhost'          : 1
            }
        
        configparser_obj['Paths'] = sec_paths
        configparser_obj['Training'] = sec_training
        configparser_obj['Parallelism'] = sec_parallelism
        
        return configparser_obj

    #------------------------------------
    # setup_logging
    #-------------------
    
    def setup_logging(self, logfile, num_classes):

        if logfile is None:
            default_logfile_name = os.path.join(os.path.dirname(__file__), 
                                                'bird_train.log' if self.local_rank is None 
                                                else f'bird_train_{self.local_rank}.log'
                                                )
            self.log = LoggingService(logfile=default_logfile_name)
            
            # In case there already was a logging instance,
            # ensure this new logging file is set:
            self.log.log_file = default_logfile_name

        elif logfile == 'stdout':
            self.log = LoggingService()
            print(f"Logging to stdout...")
        else:
            # Logfile name provided by caller. Still
            # need to disambiguate between multiple processes,
            # if appropriate:
            if self.local_rank is not None:
                (logfile_root, ext) = os.path.splitext(logfile)
                logfile = f"{logfile_root}_{self.local_rank}{ext}"
            self.log = LoggingService(logfile=logfile)
            # In case there already was a logging instance,
            # ensure this new logging file is set:
            self.log.log_file = logfile
            
            print(f"Logging to {logfile}...")
            
        # Likely a hack: To threshold either a batch of
        # predictions, or the prediction of a single class,
        # create a matrix of 1s and 0s from which to "where()",
        # and the same for prediction of a single sample's class
        # (used in tally_result():
        
        self.batch_sized_ones  = torch.ones(self.batch_size, num_classes)
        self.batch_sized_zeros = torch.zeros(self.batch_size, num_classes)
        self.one_sample_ones   = torch.ones(num_classes)
        self.one_sample_zeros  = torch.zeros(num_classes)
        
        self.tally_collection = TrainResultCollection() if self.tally_collection is None \
            else self.tally_collection 

    #------------------------------------
    # setup_json_logging 
    #-------------------
    
    def setup_json_logging(self, kernel_size, batch_size, json_log_dir=None):
        
        now = datetime.datetime.now()
        base_filepath = (f"{now.strftime('%d-%m-%Y')}_{now.strftime('%H-%M')}_K{kernel_size}" 
                         f"_B{batch_size}.jsonl")
        if json_log_dir is None:
            curr_dir = os.path.dirname(__file__)
            json_log_dir = os.path.join(curr_dir, 'runs_json')
            
        if not os.path.exists(json_log_dir) or not os.path.isdir(json_log_dir):
            os.mkdir(json_log_dir, 0o755)
        self.json_filepath = os.path.join(json_log_dir, base_filepath)
        
        with open(self.json_filepath, 'w') as f:
            f.write(json.dumps(['epoch', 
                                'loss', 
                                'training_accuracy', 
                                'testing_accuracy', 
                                'precision', 
                                'recall', 
                                'incorrect_paths', 
                                'confusion_matrix']) + "\n")

    #------------------------------------
    # write_json_record 
    #-------------------

    def write_json_record(self, tally, incorrect_paths):
        
        # Mean of accuracies among the 
        # training splits:
        mean_accuracy_training = \
           self.tally_collection.mean_accuracy(tally.epoch, 
                                               learning_phase=LearningPhase.TRAINING)
           
        mean_accuracy_validating = \
           self.tally_collection.mean_accuracy(tally.epoch, 
                                               learning_phase=LearningPhase.VALIDATING)
           
        with open(self.json_filepath, 'a') as f:
            f.write(json.dumps(
                    [tally.epoch, 
                     tally.loss.item(), 
                     mean_accuracy_training,
                     mean_accuracy_validating,
                     tally.precision.item(), 
                     tally.accuracy.item(),
                     incorrect_paths,
                     tally.conf_matrix.tolist()]) + "\n")

    #------------------------------------
    # setup_gpu 
    #-------------------
    
    def setup_gpu(self, started_from_launch):
        # If we were launched from the launch script,
        # rather than directly from the command line,
        # then internalize the promised instance vars

        if started_from_launch:
            # Internalize the promised env vars RANK and
            # WORLD_SIZE:
            try:
                non_initialized_vars = []
                try:
                    self.node_rank = int(os.environ['RANK'])
                except KeyError:
                    non_initialized_vars.append('RANK')
                try:
                    self.world_size = int(os.environ['WORLD_SIZE'])
                except KeyError:
                    non_initialized_vars.append('WORLD_SIZE')
                    
                try:
                    self.master_addr = os.environ['MASTER_ADDR']
                except KeyError:
                    non_initialized_vars.append('MASTER_ADDR')
                    
                try:
                    self.master_port = int(os.environ['MASTER_PORT'])
                except KeyError:
                    non_initialized_vars.append('MASTER_PORT')
    
                if len(non_initialized_vars) > 0:
                    raise ValueError(f"The following env vars are not initialize: {non_initialized_vars}")
            except KeyError as e:
                msg = (f"\nEnvironment variable {e.args[0]} not set.\n" 
                       "RANK, WORLD_SIZE, MASTER_ADDR, and MASTER_PORT\n"
                       "must be set.\n"
                       "Maybe use launch.py to run this script?"
                        )
                raise TrainError(msg)

        else:
            # Tis script was launched manually, rather
            # than through the launch.py, and WORLD_SIZE is
            # greater than 1, the init_process_group() call
            # later on will hang, waiting for the remaining
            # sister processes. Therefore: if launched manually,
            # set WORLD_SIZE to 1, and RANK to 0:
            
            self.log.info(("Setting RANK to 0, and WORLD_SIZE to 1,\n"
                           "b/c script was not started using launch.py()"
                           ))
            #***** Read config file; only 
            #      if not gotten from there, 
            #      do the following:
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '9000'
            self.node_rank  = 0
            self.world_size = 1
            self.master_addr = '127.0.0.1'
            self.master_port = '9000'
                

        self.init_multiprocessing(self.master_addr,
                                  self.master_port,
                                  self.node_rank,
                                  self.world_size
                                  )
                                  
    #------------------------------------
    # device_residence
    #-------------------
    
    def device_residence(self, model):
        '''
        Terrible hack to get around pytorch models
        not having a device() method that returns which
        device they are on. 
        
        NOTE: Works only if model parameters are not
              spread across multiple GPUs 
        
        @param model: any pytorch model
        @type model: torchvision.models
        '''
        return next(model.parameters()).device

    #------------------------------------
    # push_tensor 
    #-------------------

    def push_tensor(self, tnsr):
        '''
        If GPU is in use, moves tnsr to 
        the GPU, and returns that new tensor.
        That newly moved tensor is saved in 
        an array for tensor_pop() later.
        
        When no GPU is in use, return tnsr
        unchanged, but also track the tensor
        in a stack for subsequent tensor_pop()
        operations
         
        @param tnsr: tensor to move to GPU, or
            'pretend' to move there
        @type tnsr: torch.Tensor
        @return a tensor; either a newly GPU resident
            one, or the tnsr itself
        @raise ValueError if tnsr is not a tensor
        '''
        
        if self.device != self.gpu_device:
            self.cpu_tensor_stack.append(tnsr)
            return tnsr


        if not isinstance(tnsr, torch.Tensor):
            raise ValueError(f"Attempt to push to GPU, but '{tnsr}' is not a tensor")

        new_tnsr = tnsr.to('gpu')
        self.gpu_tensor_stack.append(new_tnsr)
        return new_tnsr
        
    #------------------------------------
    # pop_tensor 
    #-------------------
    
    def pop_tensor(self):
        '''
        Returns a previously pushed tensor.
        If a GPU is being used, moves the
        tensor to the CPU, and returns that
        new tensor.
        
        If only CPU is in use, returns a previously
        pushed tensor, but there is no GPU/CPU
        movement attempted.
        
        In both cases, popping from an empty stack
        raises a IndexError.
        
        @return: a tensor; either one that was already
            in the CPU, or one that results from moving
            a tensor from the GPU to the CPU
        @raises IndexError when popping from an empty stack
        '''
        if self.device != self.gpu_device:
            tnsr = self.cpu_tensor_stack.pop()
            return tnsr

        if len(self.gpu_tensor_stack) == 0:
            raise IndexError("Popping from empty tensor stack.")
        
        t = self.gpu_tensor_stack.pop()
        new_t = t.to('cpu')
        return new_t

    #------------------------------------
    # clear_gpu 
    #-------------------
    
    def clear_gpu(self):
        '''
        Removes all of this process's data
        from the GPU(s)
        '''
        if self.device != self.gpu_device:
            return
        
        # Move the model off the GPU
        self.model.cpu()
        
        # Move all tensors to CPU
        while not len(self.gpu_tensor_stack) > 0:
            self.pop_tensor()
            
        # Release GPU memory used by the cuda
        # allocator:
        cuda.empty_cache()

    #------------------------------------
    # time_diff 
    #-------------------
    
    def time_diff(self, datetime1, datetime2):
        '''
        Given two datetime instances as returned by
        datetime.datetime.now(), return the difference in 
        seconds. 
        
        @param datetime1: earlier time
        @type datetime1: datetime
        @param datetime2: later time
        @type datetime2: datetime
        @return difference in seconds
        @rtype int
        '''
        
        diff = datetime2 - datetime1
        return diff.seconds
        
    #------------------------------------
    # mean_diff 
    #-------------------
    
    def mean_diff(self, x):
        '''
        Given a tensor, return the absolute value
        of the mean of the differences between successive
        elements. Used to detect plateaus during
        successive computations. Only intended
        for 1D tensors.
         
        @param x: tensor of dim 1 
        @type x: torch.Tensor
        @returns: mean of differences between 
            successive elements. Or nan if 
            tensor only has one element.
        @rtype: float
        '''
        x1 = x.roll(-1)
        res = (x1 - x)[:-1]
        m = torch.mean(res.float())
        return abs(float(m))

# ---------------------- Resnet18Grayscale ---------------

class Resnet18Grayscale(ResNet):
    '''
    A Resnet18 variant that accepts single-channel
    grayscale images instead of RGB.
    
    Using this class saves space from not having 
    to replicate our single-layer spectrograms three 
    times to pretend they are RGB images.
    '''

    #------------------------------------
    # Constructor 
    #-------------------
    
    def __init__(self, *args, **kwargs):
        '''
        Args and kwargs as per https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
        class ResNet.__init__()
        
        '''
        # The [2,2,2,2] is an instruction to the
        # superclass' __init__() for how many layers
        # of each type to create. This info makes the
        # ResNet into a ResNet18:
        super().__init__(BasicBlock, [2,2,2,2], *args, **kwargs)
        
        # Change expected channels from 3 to 1
        # The superclass created first layer
        # with the first argument being a 3.
        # We just replace the first layer:
        self.inplanes = 64 #******* Should be batch size?
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        
    #------------------------------------
    # forward 
    #-------------------
    
    def forward(self, x):
        out_logit = super().forward(x)

        # Since we have binary classification,
        # the Sigmoid function does what a 
        # softmax would do for multi-class:

        out_probs  = nn.Sigmoid()(out_logit)
        return out_probs

    #------------------------------------
    # device_residence 
    #-------------------
    
    def device_residence(self):
        '''
        Returns device_residence where model resides.
        Can use like this to move a tensor
        to wherever the model is:
        
            some_tensor.to(self.device_residence(<model_instance>)

        '''
        return next(self.parameters()).device

    #------------------------------------
    # setup_logging 
    #-------------------
    
    def setup_logging(self):

        now = datetime.datetime.now()
        self.log_filepath = now.strftime("%d-%m-%Y") + '_' + now.strftime("%H-%M") + "_K" + str(
            self.kernel_size) + '_B' + str(self.batch_size) + '.jsonl'
        with open(self.log_filepath, 'w') as f:
            f.write(json.dumps(['epoch', 'loss', 'training_accuracy', 'testing_accuracy', 'precision', 'recall', 'incorrect_paths', 'confusion_matrix']) + "\n")


        # Default dest dir: ./runs
        self.writer = SummaryWriter(log_dir=None)
        tb_dir = os.path.join(os.path.dirname(__file__), 'runs')

        self.log.info(f"Tensorboard info will be at {tb_dir}")

# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Train from a set of images."
                                     )
 
 
    parser.add_argument('-r', '--resume',
                        help='fully qualified file name to a previously saved checkpoint; if not provided, start training from scratch',
                        default='');

    parser.add_argument('-c', '--config',
                        help='fully qualified file name of configuration file',
                        default=None);
    parser.add_argument('-d', '--data',
                        help='directory root of sub-directories that each contain samples of one class.',
                        default=None);
                        
    parser.add_argument('-l', '--logfile',
                        help='fully qualified log file name to which info and error messages \n' +\
                             'are directed. Default: stdout.',
                        default=None);
    parser.add_argument('-b', '--batchsize',
                        type=int,
                        help=f'how many sample to submit to training machinery together'
                        )
    parser.add_argument('-e', '--epochs',
                        type=int,
                        help=f'how many epochs to run'
                        )
    parser.add_argument('--started_from_launch',
                        action='store_true',
                        help="Used only by launch.py script! Indicate that script started via launch_training.py",
                        default=False
                        )
    args = parser.parse_args();
    
    BirdTrainer(
            config_info=args.config,
            root_train_test_data=args.data,
            checkpoint=args.resume,
            batch_size=args.batchsize,
            logfile=args.logfile
            ).train()