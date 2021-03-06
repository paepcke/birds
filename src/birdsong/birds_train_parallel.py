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

#!!!!! CURRENTLY NOT MAINTAINED !!!!!!!!!!

# TODO:
#   o NOTE: NOT maintained till time to revisit.
#   o Test model save on cnt-c, and start from savepoint

from _collections import OrderedDict
import argparse
import contextlib
import datetime
import json
import logging
import os, sys
from pathlib import Path
import random  # Just so we can fix seed for testing
import re
import signal
import socket
import numpy as np
from threading import Timer
import warnings
import traceback as tb

import GPUtil
from logging_service import LoggingService
import torch
from torch import Size
from torch import Tensor
from torch import cuda
from torch import device
from torch import no_grad
from torch import optim

import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from birdsong.class_weight_discovery import ClassWeightDiscovery
from birdsong.cross_validation_dataloader import MultiprocessingDataLoader, CrossValidatingDataLoader
from birdsong.result_tallying import ResultTally, ResultCollection
from birdsong.rooted_image_dataset import MultiRootImageDataset
from birdsong.utils import learning_phase
from birdsong.utils.dottable_config import DottableConfigParser
from birdsong.utils.learning_phase import LearningPhase
from birdsong.utils.neural_net_config import NeuralNetConfig, ConfigError
from birdsong.utils.tensorboard_plotter import TensorBoardPlotter, SummaryWriterPlus
from birdsong.nets import NetUtils

# Needed to make Eclipse's type engine happy.
# I would prefer using torch.cuda, etc in the 
# code, but Eclipse then marks those as errors
# even though the code runs fine in Eclipse. 
# Some cases are solved in the torch.pypredef
# file, but I don't know how to type-hint the
# following few:
packet_root = os.path.abspath(__file__.split('/')[0])
sys.path.insert(0,packet_root)

# For remote debugging via pydev and Eclipse:
# If uncommented, will hang if started from
# on Quatro or Quintus, and will send a trigger
# to the Eclipse debugging service on the same
# or different machine:
#*****************
# 
# if socket.gethostname() in ('quintus', 'quatro', 'sparky'):
#     # Point to where the pydev server 
#     # software is installed on the remote
#     # machine:
#     sys.path.append(os.path.expandvars("$HOME/Software/Eclipse/PyDevRemote/pysrc"))
#       
#     import pydevd
#     global pydevd
#     # Uncomment the following if you
#     # want to break right on entry of
#     # this module. But you can instead just
#     # set normal Eclipse breakpoints:
#     #*************
#     print("About to call settrace()")
#     #*************
#     pydevd.settrace('localhost', port=4040)
# **************** 

#***********
# Suppress the sklearn warnings about
# num of samples in some class being smaller
# than fold size. Only uncomment if using
#   <proj_root>/src/birdsong/tests/data
# for testing: 
#import warnings
#warnings.filterwarnings(action='ignore', category=UserWarning)
# Pytorch uses a deprecated method: 
#    "torch.distributed.reduce_op is deprecated, please use..."
# in module torch/distributed/distributed_c10d.py
# Suppress that one warning:
#

# The message kwarg takes a regex:
warnings.filterwarnings('ignore', 
                        message='torch.distributed.reduce_op is deprecated*', 
                        category=UserWarning, 
                        append=False)

#***********

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

    # Optimizers and loss functions that
    # can be chosen in config file:
    
    available_optimizers = ['Adam', 'RMSprop', 'SGD']
    
    # Currently only CrossEntropyLoss is supported
    # available_loss_fns   = ['MSELoss', 'CrossEntropyLoss']

    
    # Flag for subprocess to offer model saving,
    # and quitting. Used to handle cnt-C graciously:
    STOP = False
    
    SECS_IN_DAY = 24*60*60
    
    # Number of differences between accuracies from 
    # successive epochs that should be averaged to 
    # determine whether to continue training:
    
    EPOCHS_FOR_PLATEAU_DETECTION = 5
    
    # Shutdown delay:
    SHUTDOWN_WAIT = 3 # seconds

    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self,
                 config_info,
                 root_train_test_data=None,
                 batch_size=None,
                 checkpoint=None,   # Load if given path to saved
                 logdir=None,       #     ...partially trained model
                 logfile=None,
                 logging_level=logging.INFO,
                 performance_log_dir=None,
                 testing_cuda_on_cpu=False,
                 comm_info=None,
                 unit_testing=False
                 ):
        '''
        
        Logfile:
        
           o if logfile is None, check in config. 
             If nothing found, log to stdout.
           o if given logfile is a directory, a
             filename is constructed inside that dir.
             The name will contain this node's rank
           o if file name is given, it is augmented 
             with this node's rank.
             
        Comm Info:
        
             MASTER_ADDR = args.MASTER_ADDR
    		 MASTER_PORT = int(args.MASTER_PORT)
    		 RANK        = int(args.RANK)
    		 LOCAL_RANK  = int(args.LOCAL_RANK)
    		 MIN_RANK_THIS_MACHINE = int(args.MIN_RANK_THIS_MACHINE),
    		 GPUS_USED_THIS_MACHINE = int(args.GPUS_USED_THIS_MACHINE)
    		 WORLD_SIZE  = int(args.WORLD_SIZE)

        :param config_info: a NeuralNetConfig instance, 
            the path to a config file, or a JSON string containing
            all configuration info. If None, look for config 
            file in <proj-root>/config.cfg
        :type config_info: {None | str | NeuralNetConfig
        :param root_train_test_data: path to root of all training and 
            validation data. If none, find the path in the configuration
        :type root_train_test_data: {None | str}
        :param batch_size: size of each batch during training. If None,
            look for the value in the config
        :type batch_size: {None | int|
        :param checkpoint: optional path to a previously saved partially
            trained model. If None, train from scratch
        :type checkpoint: {None | str}
        :param logdir: root directory for Tensorboard event files. If
            None, use <script-dir>/runs
        :type logdir: {None | str}
        :param logfile: file where to log runtime activity info
        :type logfile: {None | str}
        :param logging_level: logging detail as per Python logging module
        :type logging_level: {None | LoggingLevel}
        :param performance_log_dir: directory for the JSON version of
            result logging (legacy). If None: <script-dir>/runs_json
        :type performance_log_dir: {None | str|
        :param testing_cuda_on_cpu: if True, operations normally done
            on a GPU are being tested on a CPU-only machine
        :type testing_cuda_on_cpu: bool
        :param comm_info: all information about computation distribution
        :type comm_info: {str : Any}
        :param unit_testing: whether or not to call all initialization
            methods in __init__() that are usuall called. If True, only
            a minimum of initialization is done, leaving calls to 
            remaining method to unittests
        :type unit_testing: bool
        '''

        if unit_testing:
            return
        # Just for informative info/error messages:
        self.hostname = socket.gethostname()
        
        # If any of the comm_info parameters, 
        # such as MASTER_ADDR are None, we ignore
        # the comm_info, and assume local run:

        if comm_info is None or None in comm_info.values():
            started_from_launch = False
            # To generalize some of the code
            # below: won't have to check for
            # device being cuda vs. CPU
            comm_info = {'LOCAL_RANK' : 0,
                         'RANK' : 0,
                         'MIN_RANK_THIS_MACHINE' : 0
                         }
        else:
            started_from_launch = True
            
        self.curr_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.comm_info = comm_info
        self.rank = self.comm_info['RANK']
        self.local_rank = self.comm_info['LOCAL_RANK']
        
        # The lowest rank of the processes on this
        # machine. Used for cases like downloads to
        # local file system that should only be done
        # by one process in a machine: 
        self.local_leader_rank = self.comm_info['MIN_RANK_THIS_MACHINE']

        try:
            self.config = self.initialize_config_struct(config_info)
        except Exception as e:
            # Error already reported at source:
            return

        # Is this training process collaborating with 
        # others, training on its own slice of data, 
        # or does every training process run with its
        # own settings, creating its own model?
        
        self.independent_runs = self.config.getboolean('Parallelism', 'independent_runs') 


        # The logger for runtime info/err messages:
        self.log = self.find_log_path(logfile)
        self.log.logging_level = logging_level
        
        # Replace None args with config file values:
        # Find the test data root from the configuration:
        
        if root_train_test_data is None:
            try:
                root_train_test_data = self.config.getpath('Paths', 'root_train_test_data', 
                                                           relative_to=self.curr_dir)
            except ValueError as e:
                raise ValueError("Config file must contain an entry 'root_train_test_data' in section 'Paths'") from e
        self.root_train_test_data = root_train_test_data
        
        if batch_size is None:
            batch_size = self.config.getint('Training', 'batch_size')
        self.batch_size = batch_size
        
        self.min_epochs = self.config.Training.getint('min_epochs')
        
        self.seed = self.config.getint('Training', 'seed')
        self.set_seed(self.seed)
        self.started_from_launch = started_from_launch

        self.init_process_group_called = False
        self.setup_gpus()

        train_parms  = self.config['Training']
        
        # Not worth the hassle of juggling tensor
        # dimension to retrain the last batch if
        # it's not completely filled:
        self.drop_last = True
        
        # Install signal handler for SIGTERM,
        # which is issued by launch_birds_training.py
        # It will set the class var STOP,
        # which we check periodically during
        # training and evaluation. When seen
        # to be True, offers training state
        # save:

        signal.signal(signal.SIGTERM, self.request_interrupt_training)
        
        # Whether or not we are testing GPU related
        # code on a machine that only has a CPU.
        # Obviously: only set to True in that situation.
        # Don't use any computed results as real:
        
        self.testing_cuda_on_cpu = testing_cuda_on_cpu

        self.device = device('cuda' if cuda.is_available() else 'cpu')
        self.cuda   = device('cuda')
        self.cpu    = device('cpu')
        
        self.num_folds = self.config.Training.getint('num_folds')
        
        batch_size = batch_size if batch_size is not None else train_parms.getint('batch_size')

        self.epoch = 0

        #master_addr_in_environ = self.comm_info["MASTER_ADDR"]
        #master_port_in_environ = self.comm_info["MASTER_PORT"]

        self.dataloader = self.initialize_data_access(
            batch_size,
            to_grayscale=self.config.getboolean('Training', 'to_grayscale')
            )

        dataset = self.dataloader.dataset
        
        # Ensure there is at least one batch worth
        # of samples to train with:
        
        num_samples = len(dataset)
        if num_samples < batch_size and self.drop_last:
            msg = f"Only {num_samples} samples available with batch size {batch_size}: no complete batch"
            self.log.err(msg)
            raise TrainError(msg)
        
        self.num_classes = len(dataset.class_id_list())
        self.class_names = dataset.class_names()
        
        self.log.info(f"Samples in loader: {len(self.dataloader)}")

        try:
            # Among others, set self.net_name:
            self.init_neural_net_parms(checkpoint, 
                                       unit_testing=unit_testing)
        except Exception as e:
            msg = f"During net init: {repr(e)}"
            self.log.err(msg)
            # Errors were already logged at their source
            raise ValueError(f"(Original trace above) {msg}") from e

        # Note: call to setup_tallying must
        # be after checkpoint restoration above.
        # Else tally_collection will be overwritten:
        
        self.setup_tallying(self.num_classes)
        self.setup_json_logging(self.config.Training.getint('kernel_size'), 
                                self.config.Training.getint('batch_size'),
                                json_log_dir=performance_log_dir
                                )
        
        # A LogSoftmax function to use when turning
        # losses returned from CrossEntropyLoss(loss)
        # into probabilities. The dim=2 picks the
        # list of raw predictions from a tensor 
        #   (num-batches, ?, num-classes):
        
        self.log_softmax = nn.LogSoftmax(dim=2) 
        
        # If this process is to report results,
        # set up tensorboard reporting:
        
        if self.rank == 0 or self.independent_runs:
            timestamp = datetime.datetime.now().isoformat()
            # Remove the msecs part:
            timestamp = re.sub(r'[.][0-9]{6}', '', timestamp)
            # Replace colons with underscores:
            timestamp = timestamp.replace(':', '_')
            
            # Experiment info:
            exp_info = (f"Exp{timestamp}_{self.config.Training.net_name}_" +
                        f"pretrain_{self.config.Training.num_pretrained_layers}_"
                        f"lr_{self.config.Training.lr}_"
                        f"opt_{self.config.Training.optimizer}_"
                        f"bs_{self.config.Training.batch_size}_" +
                        f"kernel_{self.config.Training.kernel_size}_" +
                        f"folds_{self.config.Training.num_folds}_" +
                        f"classes_{self.num_classes}_" +
                        f"gpus_here_{self.comm_info['GPUS_USED_THIS_MACHINE']}"
                        )

            # Tensorboard initialization:
            # The setup_tensorboard() method will create
            # the directory if needed:
            
            if logdir is None:
                logdir = os.path.join(self.curr_dir, 'runs')
            # Create a subdirectory in the Tensorboard directory
            # whose name reveals the configuration of this run.
            # This process will log all Tensorboard events under
            # this dir:
            exp_logdir = os.path.join(logdir, exp_info)
            self.setup_tensorboard(logdir=exp_logdir)

            # Log a few example spectrograms to tensorboard;
            # one per class:
            self.tensorboard_plotter.write_img_grid(self.writer,
                                                    self.root_train_test_data,
                                                    len(self.class_names), # Num of train examples
                                                    )

        # A stack to allow clear_gpu() to
        # remove all tensors from the GPU.
        # Requires that code below always
        # pushes and pops tensors when they
        # are moved to, and removed from the GPU:
        #
        #   new_my_tensor = self.tensor_push(my_tensor)
        #   new_my_tensor = self.tensor_pop()
        
        self.gpu_tensor_stack = []

        # Stack used instead of the gpu stack.
        # It is part a no-op in that it allows
        # tensor_push() and tensor_pop() to be
        # used even when no GPU is available.
        # In that case, clear_gpu() will do nothing:
        
        self.cpu_tensor_stack = []

    #------------------------------------
    # init_multiprocessing 
    #-------------------

    def init_multiprocessing(self):
        '''
        Called once by each copy of this 
        training script. Hangs until all
        copies of the script are ready to
        go.
        '''

        am_master_node = self.comm_info['RANK'] == 0
        if not am_master_node:
            master_node = self.comm_info['MASTER_ADDR']
            master_port = self.comm_info['MASTER_PORT']
        my_rank = self.comm_info['RANK']
        my_local_rank = self.comm_info['LOCAL_RANK']
        
        self.log.debug(f"***** LOCAL_RANK in init_multiprocessing(): {self.comm_info['LOCAL_RANK']}")
        info = f"{self.hostname}: RANK {my_rank}, LOCAL_RANK {my_local_rank}"
        if am_master_node:
            info += ' (master node)'
        self.log.info(info)

        # Use NCCL as distributed backend,
        # or another if, nccl not available:
        
        if dist.is_nccl_available():
            backend = 'nccl'           # Preferred
        elif dist.is_mpi_available():
            backend = 'mpi'
        elif dist.is_gloo_available():
            backend = 'gloo'
        else:
            raise NotImplementedError("None of mpi/nccl/gloo torch backends installed.")

        # Environment variables for init_process_group()
        os.environ['MASTER_ADDR'] = str(self.comm_info['MASTER_ADDR'])
        os.environ['MASTER_PORT'] = str(self.comm_info['MASTER_PORT'])
        os.environ['RANK']        = str(my_rank)
        os.environ['LOCAL_RANK']  = str(my_local_rank)
        os.environ['WORLD_SIZE']  = str(self.comm_info['WORLD_SIZE'])

        # Each process must call init_process_group()
        # exactly once. I.e. each copy of this script,
        # serving a different GPU must call the method.

        # If WORLD_SIZE > 1, method will hang:
        #    o If RANK is 0, this is the master process,
        #      and it will hang until all remaining WORLD_SIZE - 1
        #      processes have 'called in'
        #    o If RANK > 0, method will call in to the 
        #      MASTER_ADDR at MASTER_PORT, which is where
        #      the master process is listening.
        #      After calling in, will wait for master process'
        #      go-ahead, which will come after all WORLD_SIZE
        #      processes have started. Whether on the same, or
        #      other machines.
        
        if not am_master_node:
            self.log.info(f"Worker {my_rank} announcing itself to master node at {master_node}:{master_port}; then will await start signal...")
        dist.init_process_group(backend,
                                init_method='env://'
                                )
    
        self.init_process_group_called = True
        
        self.log.info("And we're off!")

    #------------------------------------
    # setup_gpus
    #-------------------
    
    def setup_gpus(self):
        
        if cuda.device_count() == 0:
            return
        
        # If we were launched from the launch script,
        # the the comm_info contains all needed
        # communications parameters.
        # Else, we initialize defaults:

        if not self.started_from_launch:
            
            # This script was launched manually, rather
            # than through the launch.py. The init_process_group() 
            # call later on will hang, waiting for remaining
            # sister processes. Therefore: if launched manually,
            # set WORLD_SIZE to 1, and RANK to 0:

            # We were not called via the launch.py script.
            # Check wether there is at least one 
            # local GPU, if so, use that:
            
            if len(GPUtil.getGPUs()) > 0:
                self.comm_info['LOCAL_RANK'] = 0
            else:
                self.comm_info['LOCAL_RANK'] = None

            self.log.info(("Setting RANK to 0, and WORLD_SIZE to 1,\n"
                           "b/c script was not started using launch.py()"
                           ))
            self.comm_info['MASTER_ADDR'] = '127.0.0.1'
            self.comm_info['MASTER_PORT'] = int(self.comm_info['MASTER_PORT'])
            self.comm_info['RANK']  = 0
            self.comm_info['WORLD_SIZE'] = 1
            
            return

        # The following call also sets self.gpu_obj
        # to a GPUtil.GPU instance, so we can check
        # on the GPU status along the way:
        
        local_rank = self.comm_info['LOCAL_RANK'] 
        if local_rank is not None:
            self.gpu_device = self.enable_GPU(local_rank)

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
        
        :return Name of torch backend
        :rtype: str
        :raise NotImplementedError 
        '''
        
        if dist.is_nccl_available():
            backend = 'nccl'
        elif dist.is_mpi_available():
            backend = 'mpi'
        elif dist.is_gloo_available():
            backend = 'gloo'
        else:
            raise NotImplementedError("None of mpi/nccl/gloo torch backends installed.")
        return backend

    #------------------------------------
    # to_best_device 
    #-------------------
    
    def to_best_device(self, item):
        if self.device == device('cuda'):
            item.to(device=self.cuda)
        else:
            item.to(device=self.cpu)

    #------------------------------------
    # prep_model 
    #-------------------
    
    def prep_model(self, model, local_rank=0):
        '''
        Takes a (usually CPU-resident) model,
        and:
           o moves is to GPU if possible
        
           o if self.independent_runs is False,
             concludes that running in distributed data parallel
             (DDP) and: 
             
             if a GPU will be used, wraps the model
                in a DistributedDataModel instance for
                participation in distributed data parallel
                operations. The result will reside on 
                GPU with ID (index) local_rank

           o Without GPU in use, returns the model unchanged
        
        :param model: a pytorch model instance
        :type model: pytorch.nn.model
        :param local_rank: id of GPU
        :type local_rank: int
        '''
        
        if self.independent_runs:
            model.to(self.device)
            return model

        # Running in distributed data parallel:
        
        if self.device == device('cuda'):
        
            # Leave out the devices and output
            # keyword args, b/c we always work
            # with one process per device, and
            # DDP will use the current device:

            return DDP(model,
                       device_ids=[local_rank],
                       output_device=local_rank
                       )

            # Leave out the devices and output
            # keyword args, b/c we always work
            # with one process per device, and
            # DDP will use the current device:

            return DDP(model)

        else:
            # A bit of a hack, maybe, for this case of
            # CPU-only training. The multi-GPU case
            # (of the branch above) needs the training
            # loop wrapped with a context available only
            # for DDP-wrapped models:
            #
            #    with ddp_wrapped_model.join():
            #        <train>
            #
            # CPU-only training uses just the straight
            # model, without the DDP wrapper, so the 
            # join() context manager is not available.
            # Add it to the model: 

            # The join() context manager needs to be
            # added to the model instance's class
            # no matter of which particular class the model
            # is an instance:
            
            model_class  = type(model)
            
            # A context that does nothing:
             
            null_context = contextlib.nullcontext
            
            # New context manager for the model:
            setattr(model_class, 'join', null_context)
            
            # Now model can be used as "with model.join():..." 
            return model

    #------------------------------------
    # enable_GPU 
    #-------------------

    def enable_GPU(self, local_rank, raise_gpu_unavailable=True):
        '''
        Returns the device_residence id (an int) of an 
        available GPU. If none is exists on this 
        machine, returns device('cpu')
        
        Initializes self.gpu_obj, which is a CUDA
        GPU instance. 
        
        Initializes self.cuda_dev: f"cuda:{device_id}"
        
        :param local_rank: which GPU to use. Created by the launch.py
            script. If None, just look for the next available GPU
        :type local_rank: {None|int|
        :param raise_gpu_unavailable: whether to raise error
            when GPUs exist on this machine, but none are available.
        :type raise_gpu_unavailable: bool
        :return: a GPU device_residence ID, or device('cpu')
        :rtype: int
        :raise NoGPUAvailable: if exception requested via 
            raise_gpu_unavailable, and no GPU is available.
        '''

        self.gpu_obj = None

        # Get the GPU device_residence name.
        # Could be (GPU available):
        #   device_residence(type='cuda', index=0)
        # or (No GPU available):
        #   device_residence(type=device('cpu'))

        gpu_objs = GPUtil.getGPUs()
        if len(gpu_objs) == 0:
            return device('cpu')

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
                return device('cpu')
        
        # Get the GPU object that has the found
        # deviceID:
        self.gpu_obj_from_devid(device_id)
        
        # Initialize a string to use for moving 
        # tensors between GPU and cpu with their
        # to(device_residence=...) method:
        self.cuda_dev = device_id
        self.log.info(f"Running on {device_id}")
        return device_id 


    #------------------------------------
    # select_loss_function
    #-------------------
    
    def select_loss_function(self, config):

        try:
            loss_fn_name = config['loss_fn']
        except KeyError:
            
            # Loss function not specified in config file
            # Use default:
            
            self.loss_fn = nn.MSELoss(reduction='mean')
            return self.loss_fn

        # Have name in optimizer. Is it one that we support?
        # Be case insensitive:
        if loss_fn_name.lower() not in [fn.lower() for fn in self.available_loss_fns]:
            msg = f"Loss function '{loss_fn_name}' in config file not implemented; use {self.available_loss_fns}"
            self.log.err(msg)
            raise ConfigError(msg)
        
        if loss_fn_name.lower() == 'mseloss':
            
            # 'sum' is default: sum squared error for each
            # class, then divice by num of classes:
            loss_fn = nn.MSELoss(reduction='mean')
            return loss_fn

        if loss_fn_name.lower() == 'crossentropyloss':
            if config.Training.getbool('weights', None):
                weights = ClassWeightDiscovery.get_weights(config.root_train_test_data)
            else:
                weights = None
            loss_fn = nn.CrossEntropyLoss(weight=weights)
            return loss_fn 

    #------------------------------------
    # select_optimizer 
    #-------------------
    
    def select_optimizer(self, config, model):

        lr = config.Training.getfloat('lr', 0.001)
        momentum = config.Training.getfloat('momentum', 0.9)
        
        try:
            optimizer_name = config.Training['optimizer'].lower()
        except KeyError:
            # Not specified in config file; use default:
            optimizer = optim.SGD(model.parameters(), 
                                  lr=lr,
                                  momentum=momentum)
            return optimizer
        
        if optimizer_name not in [opt_name.lower() 
                                      for opt_name 
                                      in self.available_optimizers]:
            msg = f"Optimizer '{optimizer_name}' in config file not implemented; use {self.available_optimizers}"
            self.log.err(msg)
            raise ValueError(msg)
            
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(),
                                   lr=lr,
                                   eps=1e-3,
                                   amsgrad=True
                                   )
            return optimizer
        
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), 
                                  lr=lr,
                                  momentum=momentum)
            return optimizer

        if optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), 
                                      lr=lr,
                                      momentum=momentum)
            return optimizer

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
#         :param curr_tallies: if given, the tallies from
#             a current epoch
#         :type curr_tallies: {str : float}
#         :return: dict with float zeros
#         :rtype: {str : float}
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

        :param labels_tns: ground truth labels for batch
        :type labels_tns: torch.Tensor (integer class IDs)
        :param pred_prob_tns: model prediction probabilities
        :type pred_prob_tns: torch.Tensor
        :param loss: accumulated result of loss function from
            run through all the folds.
        :type loss: torch.Tensor (float)
        :param learning_phase: phase under which to record this
            result: training, validation, or test.
        :type learning_phase: LearningPhase
        '''
        
        # Predictions are for one batch. Example for
        # batch_size 2 and 4 target classes:
        #  
        #    torch.tensor([[0.1, -0.2,  .3,  .42], |
        #                                          | <--- one batch's network raw outputs
        #                  [.43,  .3, -.23, -.18]  |   for batch_size == 2 
        #
        #                  ])

        # Turn something like this 6-class value:
        # tensor(
        #  [[[ -9.6347,  30.7077,  12.9497, -13.9641,  -9.8286,  -8.1160]],
        #   [[ -8.9195,  29.1933,  11.8827, -13.0813,  -9.9640,  -8.1645]],
        #            ...
        # first use softmax to turn the above into 
        # probabilities that rowise add to 1
        # 
        # tensor(
        #  [[[3.0166e-18, 1.0000e+00, 1.9400e-08, 3.9745e-20, 2.4849e-18, 1.3775e-17]],
        #    [2.8043e-17, 1.0000e+00, 3.0346e-08, 4.3689e-19, 9.8674e-18, 5.9664e-17]],
        #            ...
        #
        # Then argmax turns each row into a class prediction:
        #   [[1],
        #    [1],
        #    ...
        #    ]
        #                   ...
        pred_classes_each_batch = self.log_softmax(pred_prob_tns).argmax(dim=2)

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

        tally = ResultTally(
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

        epoch_summary = EpochSummary(self.tally_collection, 
                                     epoch,
                                     self.log)
        with open(self.json_filepath, 'a') as f:
            f.write(json.dumps(
                    [epoch, 
                     epoch_summary.epoch_loss_val,
                     epoch_summary.mean_accuracy_train,
                     epoch_summary.mean_accuracy_val,
                     epoch_summary.epoch_mean_weighted_precision,
                     epoch_summary.epoch_mean_weighted_recall,
                     incorrect_paths,
                     epoch_summary.epoch_conf_matrix.tolist()]) + "\n")

    #------------------------------------
    # record_tensorboard_results 
    #-------------------
    
    def record_tensorboard_results(self, epoch):
        '''
        Called at the end of each epoch. Writes
        all epoch-level measures to tensorboard.

        :param epoch:
        :type epoch:
        '''
        
        # A new training sample class distribution
        # barchart:

        if epoch == 1:
            # Barchart with class support only
            # needed once:
            self.tensorboard_plotter.class_support_to_tensorboard(
                self.dataloader.dataset,
                self.writer,
                step=self.epoch,
                title="Training Class Support"
                )
        
        epoch_results = EpochSummary(self.tally_collection, 
                                     epoch,
                                     self.log
                                     )
        # The following quantities will be writen
        # to tensorboard for training/validation:
        
        # training_quantities_to_report = ['mean_accuracy',
        #                                  'balanced_accuracy_score',
        #                                  'epoch_loss',
        #                                  ]
        # 
        # validation_quantities_to_report = ['mean_accuracy',
        #                                    'balanced_accuracy_score',
        #                                    'epoch_loss',
        #                                    'precision_recall',
        #                                    ]

        # Write training phase results to tensorboard:

        try:
            # epoch_loss (train):
            
            val = epoch_results.get('epoch_loss_train', None) 
            self.writer.add_scalar('epoch_loss/train', val,
                                   global_step=epoch)

            # mean_accuracy (train):
            
            val = epoch_results.get('mean_accuracy_train', None)
            self.writer.add_scalar(f"mean_accuracy/train", val, 
                                   global_step=epoch)

            # balanced_accuracy_score (train):
            
            val = epoch_results.get('balanced_adj_accuracy_score_train', None) 
            self.writer.add_scalar('balanced_accuracy_score/train', val,
                                   global_step=epoch)
                

            # Write validation phase results to tensorboard:

            # epoch_loss (validate)
            val = epoch_results.get('epoch_loss_val', None) 
            self.writer.add_scalar('epoch_loss/validate', val,
                                   global_step=epoch
                                   )
            
            # mean_accuracy (validate):
            
            val = epoch_results.get('mean_accuracy_val', None) 
            self.writer.add_scalar('mean_accuracy/validate', val,
                                    global_step=epoch)

            # balanced_accuracy_score (validate):
            
            val = epoch_results.get('balanced_adj_accuracy_score_val', None) 
            self.writer.add_scalar('balanced_accuracy_score/validate', val,
                                    global_step=epoch)


            # precision_recall (validate only):
                
            # Precision macro
            val = epoch_results.get('epoch_mean_macro_precision', None)
            self.writer.add_scalar('precision_recall_macro/precision', val,
                                   global_step=epoch
                                   )
                
            # Precision micro:
            val = epoch_results.get('epoch_mean_micro_precision', None)
            self.writer.add_scalar('precision_recall_micro/precision', val,
                                   global_step=epoch
                                   )

            # Precision weighted by class support:
            val = epoch_results.get('epoch_mean_weighted_precision', None)
            self.writer.add_scalar('precision_recall_weighted/precision', val,
                                   global_step=epoch
                                   )

            # Recall (validate only):

            # Recall macro:
            val = epoch_results.get('epoch_mean_macro_recall', None)
            self.writer.add_scalar('precision_recall_macro/recall', val,
                                   global_step=epoch
                                   )
                
            # Recall micro:
            val = epoch_results.get('epoch_mean_micro_recall', None)
            self.writer.add_scalar('precision_recall_micro/recall', val,
                                   global_step=epoch
                                   )
                
            # Recall weighted:
            val = epoch_results.get('epoch_mean_weighted_recall', None)
            self.writer.add_scalar('precision_recall_weighted/recall', val,
                                   global_step=epoch
                                   )
            
            # History of learning rate adjustments:
            lr_this_epoch = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('learning_rate', lr_this_epoch,
                                   global_step=epoch
                                   )

        except AttributeError:
            self.log.err(f"No tensorboard writer in process {self.rank}")
        except Exception as e:
            msg = f"Error while reporting to tensorboard: {repr(e)}"
            self.log.err(msg)
            raise ValueError from e


        # Submit the confusion matrix image
        # to the tensorboard. In the following:
        # do not provide a separate title, such as
        #  title=f"Confusion Matrix (Validation): Epoch{epoch}"
        # That will put each matrix into its own slot
        # on tensorboard, rather than having a time slider
        
        self.tensorboard_plotter.conf_matrix_to_tensorboard(
            self.writer,
            epoch_results['epoch_conf_matrix'], 
            self.class_names,
            step=self.epoch,
            title=f"Confusion Matrix Series"
            )

    #------------------------------------
    # train 
    #-------------------

    def train(self):
        '''
        Orchestrate the training and validation.
        This method works with cross validation,
        and interoperates closely with train_one_split()
        and validate_one_split().
        
        Assumptions:
            o self.epoch is either 0, or set to where a 
                previously saved model left off
            o self.model is ready to generate output

        Method feeds results to Tensorboard.
        
        '''

        self.log.info("Begin training")
        
        # Whether cnt-C was received and handled.
        # If True, finally clause will do nothing:
        hard_stop = False
        
        if self.device == self.cuda:
            self.initial_GPU_memory = cuda.memory_allocated(self.device)
            # Reset statistics on max GPU memory use:
            cuda.reset_peak_memory_stats()

        # Plateau discovery:
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
                
                while (self.epoch < self.config.Training.getint('max_epochs') \
                       and (self._diff_avg >= 0.05 or \
                            self.epoch <= self.min_epochs
                            )
                       ):

                    self.epoch += 1
                    self.num_train_samples_this_epoch = 0
                    self.num_val_samples = 0

                    # Places to accumulate output predictions
                    # and truth labels for training and 
                    # validation throughout one epoch:
                    
                    self.train_label_stack  = None 
                    self.train_output_stack = None
                    
                    self.val_output_stack = None
                    self.val_label_stack  = None
                    
                    # Tell dataloader that epoch changed.
                    # The dataloader will tell the sampler,
                    # which will use the information to 
                    # predictably randomize the dataset
                    # before creating splits:
                    
                    self.dataloader.set_epoch(self.epoch)
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
                            # *********
                            #with self.model.join():
                            #    split_training_iterator.send(split_num)
                            split_training_iterator.send(split_num)
                            # *********                            
                        except StopIteration:
                            # Exhausted all the splits
                            self.log.warn(f"Unexpectedly ran out of splits.")
                            break
                        except Exception as e:
                            msg = f"Error sending split_num to split_training: {repr(e)}"
                            self.log.err(msg)
                            raise ValueError(f"{msg}. Most useful: the trace *above* this one") from e

                        self.validate_one_split()
    
                        # Time for sign of life?
                        time_now = datetime.datetime.now()
                        if self.time_diff(time_start, time_now) >= alive_pulse:
                            self.log.info (f"Epoch {self.epoch}; Split number {split_num} of {num_folds}")
    
                    # Done with one epoch:
                    self.log.info(f"Finished epoch {self.epoch}")
                    
                    # Must be called after optimizer.step(),
                    # which was called as part of the epoch
                    # abobe:
                    self.scheduler.step()
                    
                    total_train_loss = self.tally_collection.cumulative_loss(epoch=self.epoch,
                                                                           learning_phase=LearningPhase.TRAINING
                                                                           )
                    
                    if self.num_train_samples_this_epoch == 0:
                        self.log.warn(f"No samples available in epoch {self.epoch}. Batch size > num_samples?")
                        avg_train_loss = total_train_loss
                    else:
                        avg_train_loss = total_train_loss / self.num_folds

                    if self.train_output_stack is None:
                        self.log.warn(f"No samples were processed for epoch {self.epoch}; no result reported")
                    else:
                        self.tally_result(self.train_label_stack, 
                                          self.train_output_stack,
                                          avg_train_loss, 
                                          LearningPhase.TRAINING)
                    
                    total_val_loss = self.tally_collection.cumulative_loss(epoch=self.epoch,
                                                                           learning_phase=LearningPhase.VALIDATING
                                                                           )
                    avg_val_loss = total_val_loss / self.num_folds
                    self.tally_result(self.val_label_stack,
                                      self.val_output_stack,
                                      avg_val_loss, 
                                      LearningPhase.VALIDATING
                                      )

                        
                    # Update the json based result record
                    # in the file system, and write results
                    # to tensorboard.
                    
                    if self.rank == 0 or self.independent_runs:
                        #******self.record_json_display_results(self.epoch)
                        self.record_tensorboard_results(self.epoch)
                    
                    # Compute mean accuracy over all splits
                    # of this epoch:
                    accuracies[self.epoch] = self.tally_collection.mean_accuracy(epoch=self.epoch, 
                                                                                 learning_phase=LearningPhase.TRAINING)
    
                    if self.epoch <= self.min_epochs:
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
    
                # All seems to have gone well. Report the 
                # overall result of the final epoch for the 
                # hparms config used in this process:
                
                self.report_hparams_summary(self.tally_collection, 
                                            self.epoch,
                                            self.net_name,
                                            self.num_pretrained_layers)
    
            except (KeyboardInterrupt, InterruptTraining) as _e:
                # Stop brutal-shutdown time; we will be gentle here:
                try:
                    self.shutdown_timer.cancel()
                except:
                    # timer might not have been set, though unlikely
                    pass
                
                self.log.info("Early stopping due to keyboard intervention")

                if self.rank != self.local_leader_rank:
                    self.log.info(f"Leaving model saving to process {self.local_leader_rank}")
                    # Return GPU used if any, though with
                    # the cnt-C it won't make a diff:
                    return self.local_rank
                    
                do_save = self.offer_model_save()
                if do_save in ('y','Y','yes','Yes', ''):
                    have_fname = False
                    while not have_fname:
                        dest_path = input("Destination file with '~' or '$' (/tmp/checkpoint.pth): ")
                        if len(dest_path) > 0:
                            # Resolve '~' and environ vars: 
                            dest_path = os.path.expandvars(os.path.expanduser(dest_path))
                            dest_dir  = os.path.dirname(dest_path)
                        else:
                            dest_path = '/tmp/checkpoint.pth)'
                            
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
                else:
                    self.log.info(f"Training aborted by user, who requested not to save the model")
                self.cleanup()
                self.log.info("Exiting")
                return

        finally:
            # If two times cnt-c, exit no matter what
            if hard_stop:
                print("******* Hard stop---no cleanup.")
                sys.exit(1)
            # Did an error occur?
            if sys.exc_info() != (None, None, None):
                # Yes; don't raise, but print the 
                # backtrace. Then continue cleaning up:
                tb.print_exc()
            if self.device == self.cuda:
                # Ensure all activity in different parts
                # of the cuda device are done; uses the
                # current device
                self.log.info("Synchronizing stream in current GPU")
                cuda.synchronize()
                
                self.model.to('cpu')
                if self.weighted:
                    self.device_resident_weights.to('cpu')
                    
                self.ending_GPU_memory = cuda.memory_allocated(self.device)
                if self.ending_GPU_memory >= self.initial_GPU_memory:
                    left_on_gpu = self.ending_GPU_memory - self.initial_GPU_memory
                    self.log.err(f"Application did not release {self.human_readable(left_on_gpu)}")
                if len(self.gpu_tensor_stack) > 0:
                    self.log.err(f"Application leaves {len(self.gpu_tensor_stack)} tensors on GPU")
                if len(self.cpu_tensor_stack) > 0:
                    self.log.warn(f"Application used CPU, but left {len(self.cpu_tensor_stack)} tensors from push/pop regimen")
                max_mem = cuda.max_memory_allocated(self.device)
                self.log.info(f"Max GPU memory use: {self.human_readable(max_mem)}")
                
                self.cleanup()
                after_cleanup_gpu_memory = cuda.memory_allocated(self.device)
                left_on_gpu_after_cleanup = after_cleanup_gpu_memory - self.initial_GPU_memory
                self.log.info(f"Memory not cleaned out from GPU: {left_on_gpu_after_cleanup}")
                                
                self.log.info(f"Process with node{self.rank} exiting.")

            self.log.info("Training finished")
            return self.local_rank

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
        
        loss = 0.0
        
        # Await the first real call,
        # which will be via a send(split_num):
        # Not used, but nice for debugging: 
        _split_num = yield
        
        # While loop will be exited when
        # dataloader has exhausted all batches
        # from each fold from every split:
        
        for _i, (batch,targets) in enumerate(self.dataloader):

            if batch is None:
                # Exhausted this split. Record this split's
                # training results, and yield till
                # the next split's worth of training
                # is requested by the caller.
                
                # Sit until next() is called on this 
                # generator again. At that point we continue feeding
                # from the same dataloader in this while
                # loop. The assignment receives a new 
                # split_num and epoch_loss via the 'send()'
                # function when train() calls next() on this
                # function:
                 
                _split_num = yield
                
                # Back for a new split. Prepare:
                # Set model to train mode (just in
                # case someone set it to eval mode...):
                self.model.train(True)
                
                # ... and continue in the while loop,
                # using the continuing batch_feeder
                continue

            # Got another batch/targets pair:
            # Push sample and target tensors
            # to where the model is:
            try:
                batch   = self.push_tensor(batch)
                targets = self.push_tensor(targets)
                self.num_train_samples_this_epoch += len(batch)

                # Outputs will be on GPU if we are
                # using one:
                outputs = self.model(batch)
                
                loss = self.loss_fn(outputs, targets)
                #self.log.debug(f"***** Train loss epoch {self.epoch}: {loss}")
                
                # Update the accumulating training loss
                
                # for this epoch: 
                self.tally_collection.add_loss(self.epoch, 
                                               loss,
                                               LearningPhase.TRAINING
                                               )
                loss.backward()
                if not self.independent_runs and \
                    self.comm_info['WORLD_SIZE'] > 1 \
                    and self.device != device('cpu'):
                    #**********
                    # Force synchronicity across all GPUs
                    # before averaging the gradients:
                    dist.barrier()
                    #**********
                    
                    # Average the model weights across
                    # all GPUs:
                    for param in self.model.parameters():
                        # Skip the frozen layers:
                        if param.grad is not None:
                            # The SUM is from torch.distributed.reduce_ops:
                            dist.all_reduce(param.grad.data, 
                                            op=dist.ReduceOp.SUM,
                                            async_op=False)
                            param.grad.data /= self.comm_info['WORLD_SIZE']
                self.optimizer.step()
                loss = loss.to('cpu')
            
            finally:
                # Free GPU memory:
                targets = self.pop_tensor()
                batch   = self.pop_tensor()
                
                outputs = outputs.to('cpu')

            self.train_output_stack, self.train_label_stack = \
                self.remember_output_and_label(outputs, 
                                               targets, 
                                               self.train_output_stack, 
                                               self.train_label_stack,
                                               learning_phase=LearningPhase.TRAINING
                                               )
            # Pending STOP request?
            if BirdTrainer.STOP:
                raise InterruptTraining()
            

    #------------------------------------
    # validate_one_split 
    #-------------------

    def validate_one_split(self):
    
        # Track num of samples we
        # validate, so we can divide cumulating
        # loss by that number:
        
        # Model to eval mode:
        self.model.eval()
        with no_grad():
            for (val_sample, val_target) in self.dataloader.validation_samples():
                self.num_val_samples += 1
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
                loss =  self.loss_fn(pred_prob_tns, val_target)
                loss = loss.to('cpu')
                self.log.debug(f"***** Validation loss epoch {self.epoch}: {loss}")

                self.tally_collection.add_loss(self.epoch, 
                                               loss,
                                               LearningPhase.VALIDATING
                                               )

                # Free GPU memory:
                val_sample = val_sample.to('cpu')
                val_target = val_target.to('cpu')
                pred_prob_tns = pred_prob_tns.to('cpu')
                self.val_output_stack, self.val_label_stack = \
                   self.remember_output_and_label(pred_prob_tns, 
                                                  val_target, 
                                                  self.val_output_stack, 
                                                  self.val_label_stack,
                                                  learning_phase=LearningPhase.VALIDATING)

    #------------------------------------
    # remember_output_and_label 
    #-------------------
    
    def remember_output_and_label(self, 
                                  outputs, 
                                  targets, 
                                  output_stack=None, 
                                  label_stack=None,
                                  learning_phase=LearningPhase.TRAINING):
        '''
        Both training and validation run through a loop
        of batches. Not every single iteration's results
        need to be represented in the performance logs.
        Instead, the output-prob/correct-label pairs are
        accumulated in an output_stack and label_stack,
        respectively.
        
        This method adds one ouput/label pair to their
        respective stacks. The tally_result() method will
        be called after the end of each split, and it will
        expect the two stacks.
        
        Clients are responsible for setting output_stack and
        label_stack to None after each split. They are also 
        responsible for preserving the returned values for the
        next call to this method. I.e. this method does not
        store output_stack or label_stack anywhere.
        
        :param outputs: an output tensor from the model 
        :type outputs: Tensor [batch_size x num_classes]
        :param targets: correct label for each sample in batch
        :type targets: Tensor [batch_size x 1]
        :param output_stack: current (partially filled) stack
            of model outputs
        :type output_stack: [Tensor]
        :param label_stack: current (partially filled) stack 
            of labels
        :type label_stack:[Tensor]
        :param learning_phase: whether currently training or validating 
        :type learning_phase: LearningPhase
        :return: tuple of updated (output_stack, label_stack)
        '''
        
        # If training we expect one row for
        # each member of one batch:
        if learning_phase == LearningPhase.TRAINING:
            expected_num_rows = self.batch_size
            
        # But if validating, it's as if batch_size is 1:
        # a single result for the single sample fed into 
        # the model under validation:

        elif learning_phase == LearningPhase.VALIDATING:
            expected_num_rows = 1
    
        else:
            raise ValueError(f"Method remember_output_and_label() called with wrong phase: {learning_phase}")

        # Remember performance:
        if output_stack is None:
            # First time, create the stack
            output_stack = torch.unsqueeze(outputs, dim=0)
        else:
            # Did we get a full batch? Output tensors
            # of shape [batch_size, num_classes]. 
            #     [[1st_of_batch, [logit_class1, logit_class2, ... logit_num_classes]]
            #      [2nd_of_batch, [logit_class1, logit_class2, ... logit_num_classes]]
            #     ]
            have_full_batch = outputs.size() == Size([expected_num_rows,
                                                      self.num_classes])
            if not have_full_batch and self.drop_last:
                # Done with this split:
                return (output_stack, label_stack)
                
            output_stack = torch.cat((output_stack, torch.unsqueeze(outputs, 
                                                                    dim=0))) 
        if label_stack is None:
            label_stack = torch.unsqueeze(targets, dim=0)
        else:
            label_stack = torch.cat((label_stack, torch.unsqueeze(targets, 
                                                                  dim=0)))

        return (output_stack, label_stack)
        


    # ------------- Utils -----------

    #------------------------------------
    # get_class_support 
    #-------------------
    
    def get_class_support(self):
        '''
        Returns the distribution of samples
        in the underlying dataset by class
        
        :return: list of tuples: (class_id, num_samples);
            one such tuple for each class_id
        :rtype: [(int, int)]
        '''
        return self.dataloader.dataset.sample_distribution()

    #------------------------------------
    # initialize_config_struct 
    #-------------------
    
    def initialize_config_struct(self, config_info):
        '''
        Initialize a config dict of dict with
        the application's configurations. Sections
        will be:
        
          config['Paths']       -> dict[attr : val]
          config['Training']    -> dict[attr : val]
          config['Parallelism'] -> dict[attr : val]
        
        The config read method will handle config_info
        being None. 
        
        If config_info is a string, it is assumed either 
        to be a file containing the configuration, or
        a JSON string that defines the config.
         
        Else config_info is assumed to be a NeuralNetConfig.
        The latter is relevant only if using this file
        as a library, rather than a command line tool.
        
        If given a NeuralNetConfig instance, it is returned
        unchanged. 
        
        :param config_info: the information needed to construct
            the structure
        :type config_info: {NeuralNetConfig | str}
        :return a NeuralNetConfig instance with all parms
            initialized
        :rtype NeuralNetConfig
        '''

        if isinstance(config_info, str):
            # Is it a JSON str? Should have a better test!
            if config_info.startswith('{'):
                # JSON String:
                config = NeuralNetConfig.from_json(config_info)
            else: 
                config = self.read_configuration(config_info)
        elif isinstance(config_info, NeuralNetConfig):
            config = config_info
        else:
            msg = f"Error: must have a config file, not {config_info}. See config.cfg.Example in project root"
            # Since logdir may be in config, need to use print here:
            print(msg)
            raise ConfigError(msg)
            
        return config

    #------------------------------------
    # initialize_data_access 
    #-------------------
    
    def initialize_data_access(self, batch_size, to_grayscale=False):
        '''
        Creates dataset and dataloader instances.
        If operating in Distributed Data Parallel,
        calls init_process_group().
        
        :param batch_size: desired batch size
        :type batch_size: int
        :param to_grayscale: do or do not tell dataset
            to convert images to grayscale as the
            are pulled from disk
        :type to_grayscale: bool
        :return: a dataloader in the pytorch sense
        :rtype CrossValidatingDataLoader, or subclass
        '''
        
        # In the dataloader creation, make drop_last=True 
        # to skip the last batch if it is not full. Just 
        # not worth the trouble dealing with unusually 
        # dimensioned tensors:
        
        dataset  = MultiRootImageDataset(self.root_train_test_data,
                                         sample_width=self.config.Training.getint('sample_width'),
                                         sample_height=self.config.Training.getint('sample_width'),
                                         to_grayscale=to_grayscale
                                         )


        # Make an appropriate (single/multiprocessing) dataloader:
        if self.device == device('cpu') or self.independent_runs:

            # CPU bound, single machine:
            
            dataloader = CrossValidatingDataLoader(
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

            self.init_multiprocessing()
            
            dataloader = MultiprocessingDataLoader(dataset,
                                                   shuffle=True,
                                                   seed=self.seed,
                                                   num_folds=self.num_folds,
                                                   drop_last=True,
                                                   batch_size=batch_size 
                                                   )
        return dataloader

    #------------------------------------
    # init_neural_net_parms 
    #-------------------
    
    def init_neural_net_parms(self, checkpoint=None, unit_testing=False):
        '''
        Finds all neural net related hyper parameter values
        in the configuration, and assigns them to instance
        vars. If checkpoint is provided it must be a file
        path to a previously saved, partially trained model.
        That checkpoint will be loaded. Else network initialization
        occurs from scratch.
        
        Creates a network  
        
        :param checkpoint: optional path to previously saved
            model
        :type checkpoint: {None | str}
        :param unit_testing: early return if unit testing
        :type unit_testing: bool
        '''

        self.net_name = self.config.Training.net_name
        
        # Resnet18 retain 6 layers of pretraining.
        # Train the remaining 4 layers with your own
        # dataset. This is the model without capability
        # for parallel training. Will be adjusted in
        # to_app

        self.num_pretrained_layers = self.config.Training.getint('num_pretrained_layers',
                                                                 0) # Default: train from scratch
        
        if self.independent_runs:
            #*************
            raw_model = NetUtils.get_net(
                self.net_name,
                num_classes=self.num_classes,  # num_classes
                num_layers_to_retain=self.num_pretrained_layers,
                to_grayscale=False
                )
            
#             raw_model = NetUtils.get_net(
#                 self.net_name,
#                 num_classes=self.num_classes,  # num_classes
#                 num_layers_to_retain=self.num_pretrained_layers,
#                 net_version=18,
#                 to_grayscale=True
#                 )
#             raw_model = BasicNet(self.num_classes, 
#                                  batch_size=self.batch_size, 
#                                  kernel_size=ks, 
#                                  processor=None)

#             raw_model = NetUtils.get_resnet_partially_trained(self.num_classes,
#                                                               num_layers_to_retain=6,
#                                                               net_version=18
#                                                               )
            #*************
        else:
            # When re-trying DDP: update this branch
            # to match the upper branch where appropriate
            raw_model = NetUtils.get_model_ddp(self.rank, 
                                               self.local_leader_rank, 
                                               self.log,
                                               net_version=18,
                                               num_layers_to_retain=6
                                               )
        
        # Wrapper to handle distributed training, and
        # move to GPU, if appropriate. 
        
        self.model = self.prep_model(raw_model,
                                     self.comm_info['LOCAL_RANK']
                                     )
        #**********
        self.log.info(f"type(model): {type(self.model)}")
        #**********
        if unit_testing:
            return
        self.optimizer = self.select_optimizer(self.config, self.model)
        
        if checkpoint:
            # Requested to continue training 
            # from a checkpoint:
            if not os.path.exists(checkpoint):
                msg = f"Requested continuation from checkpoint {checkpoint}, which does not exist."
                self.log.err(msg)
                raise FileNotFoundError(msg)
            
            self.log.info("Loading checkpoint...")
            self.load_model_checkpoint(checkpoint, 
                                       self.model, 
                                       self.optimizer)
            self.log.info("Loading checkpoint done")
            self.log.info(f"Resume training at start of epoch {self.epoch}")
        else:
            self.epoch = 0
            self._diff_avg = None
            self.tally_collection = None

        if self.config.getboolean('Training', 'weighted', True):
            
            self.weighted = True
            # Get path to samples, relative to current
            # dir, if the file path in the config file 
            # is relative:
            class_root = self.config.getpath('Paths',
                                             'root_train_test_data',
                                             relative_to = self.curr_dir)
            self.weights = ClassWeightDiscovery.get_weights(class_root)
            # Put weights to the device where the model
            # was placed: CPU, or GPU_n:
            self.device_resident_weights = self.weights.to(self.device_residence(self.model))
        else:
            self.weighted = False
            self.weights  = None
            
        # Loss function:
        #************************* Eventually: call select_loss_function!
        #self.loss_fn = self.select_loss_function(self.config)
        #******self.loss_fn = nn.CrossEntropyLoss(weight=self.device_resident_weights)
        self.loss_fn = nn.CrossEntropyLoss()
        #*************************
        
        # Scheduler: at each step (after each epoch)
        # we will call the scheduler with the latest
        # validation loss. If it has not decreased within
        # patience epochs, decrease learning rate by factor
        # of 10:
#         self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
#                                                               factor=0.1,
#                                                               patience=3
#                                                               )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                              self.min_epochs
                                                              )

        self.log.info(f"Model resides on device: {self.device_residence(self.model)}")
        


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
        # Already requested interrupt once before?
        
        if BirdTrainer.STOP:
            self.log.info("Quitting gracelessly after two cnt-C keys.")
            sys.exit(1)
            
        self.log.info("Requesting training interruption; waiting for clean point to interrupt...")
        BirdTrainer.STOP = True
        
        # Timeout in case program is not in
        # the training loop, and thus won't
        # notice the STOP flag:
        
        self.shutdown_timer = Timer(interval=self.SHUTDOWN_WAIT,
                                    function=lambda : (print("Quitting hard after timeout"), sys.exit(1))
                                    )
        self.shutdown_timer.start()

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
        
        
        :param scheduler: scheduler whose learning rate 
            is to be retrieved
        :type scheduler:torch.optim.lr_scheduler
        :return: the scheduler's current learning rate
        :rtype: float
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
        
        :param pred_tns: tensors of 1.0 and 0.0 stacked batch_size high
        :type pred_tns: torch.Tensor [batch_size, 1]
        :param label_tns: 1.0 or 2.0 true labels stacked batch_size high
        :type label_tns: torch.Tensor [batch_size, 1]
        :return: precision values tp, tp_fp, tp_fn
        :rtype: [int,int,int]
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
        
        :param seed: random seed to set for all random num generators
        :type seed: int
        '''
        torch.manual_seed(seed)
        cuda.manual_seed_all(seed)
        # Not totally sure what these two do!
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)

    #------------------------------------
    # save_model_checkpoint 
    #-------------------
    
    def save_model_checkpoint(self, dest_path, model, optimizer):
        '''
        Save the entire current state of training.
        Re-instate that checkpoint in a new session
        by using self.load_model_checkpoint()
        
        :param dest_path: absolute path where to save the 
            pickled model. Use .pth or .tar extension
        :type dest_path: str
        :param model: the model instance to save
        :type model: torch model class
        :param optimizer: the optimizer instance to save
        :type optimizer: torch optimizer class
        :param epoch: the last finished epoch 
        :type epoch: int
        '''
        
        # Remove results from the current
        # epoch from tally_collection:
        
        clean_tally_collection = ResultCollection.create_from(self.tally_collection)
        for epoch, learning_phase in self.tally_collection.keys():
            if epoch == self.epoch:
                clean_tally_collection.pop((epoch, learning_phase))

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
        
        :param src_path: path where the app_state is stored
        :type src_path: str
        :param fresh_model: a new, uninitialized instance of the model used
        :type fresh_model: torch model
        :param fresh_optimizer: a new, uninitialized instance of the optimizer used
        :type fresh_optimizer: torch optimizer
        :return: dict with the initialized model, 
            optimizer, last completed epoch, most recent
            loss measure, and tallies collection.
        :rtype {str : *}
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
        
        :param other_gpu_config_file: path to configuration file
        :type other_gpu_config_file: str
        :return: a dict of dicts mirroring the config file sections/entries
        :rtype: dict[dict]
        :raises ValueErr
        :raises TypeError
        '''
        
        if conf_file is None:
            return self.init_defaults()
        
        config = DottableConfigParser(conf_file)
        
        if len(config.sections()) == 0:
            # Config file exists, but empty:
            return(self.init_defaults(config))
    
        # Do type conversion also in other entries that 
        # are standard:
        
        types = {'epochs' : int,
                 'batch_size' : int,
                 'kernel_size' : int,
                 'sample_width' : int,
                 'sample_height' : int,
                 'seed' : int,
                 'pytorch_comm_port' : int,
                 'num_pretrained_layers' : int,
                 
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
        Not really needed any more, as a config
        file path is now required in the command
        line. 

        Initialize the standard entries.
        
        Though not having the root of the train/test
        samples should probably be an error.
        
        :param configparser_obj: an empty configparser instance
        :type configparser_obj: configparser
        :return the same config parser object filled-in 
        :rtype configparser
        '''
        sec_paths = {
            'root_train_test_data' : '/home/data/birds/recombined_data/'
            }

        sec_training = {
            'net_name'      : 'resnet18',
            'num_pretrained_layers' : 6,
            'epochs'        : 5,
            'batch_size'    : 32,
            'seed'          : 42,
            'kernel_size'   : 7,
            'sample_width'  : 400,
            'sample_height' : 400
            }

        sec_parallelism = {
            'pytorch_comm_port'  : 5678,
            }
        
        configparser_obj['Paths'] = sec_paths
        configparser_obj['Training'] = sec_training
        configparser_obj['Parallelism'] = sec_parallelism
        
        return configparser_obj

    #------------------------------------
    # setup_tallying
    #-------------------
    
    def setup_tallying(self, num_classes):

        # Likely a hack: To threshold either a batch of
        # predictions, or the prediction of a single class,
        # create a matrix of 1s and 0s from which to "where()",
        # and the same for prediction of a single sample's class
        # (used in tally_result():
        
        self.batch_sized_ones  = torch.ones(self.batch_size, num_classes)
        self.batch_sized_zeros = torch.zeros(self.batch_size, num_classes)
        self.one_sample_ones   = torch.ones(num_classes)
        self.one_sample_zeros  = torch.zeros(num_classes)
        
        self.tally_collection = ResultCollection() if self.tally_collection is None \
            else self.tally_collection 

    #------------------------------------
    # setup_tensorboard 
    #-------------------
    
    def setup_tensorboard(self, logdir):
        '''
        Initialize tensorboard. To easily compare experiments,
        use runs/exp1, runs/exp2, etc.
        
        Method creates the dir if needed.
        
        :param logdir: root for tensorboard events
        :type logdir: str
        '''
        
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
        
        # Use SummaryWriterPlus to avoid confusing
        # directory creations when calling add_hparams()
        # on the writer:
        
        self.writer = SummaryWriterPlus(log_dir=logdir)
        
        # Tensorboard image writing:
        self.tensorboard_plotter = TensorBoardPlotter()
        
        self.log.info(f"To view tensorboard charts: in shell: tensorboard --logdir {logdir}; then browser: localhost:6006")

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
    # report_hparams_summary 
    #-------------------
    
    def report_hparams_summary(self, 
                               tally_coll, 
                               epoch,
                               network_name,
                               num_pretrained_layers
                               ):
        '''
        Called at the end of training. Constructs
        a summary to report for the hyperparameters
        used in this process. Reports to the tensorboard.
        
        Hyperparameters reported:
        
           o lr
           o optimizer
           o batch_size
           o kernel_size
        
        Included in the measures are:
        
           o balanced_adj_accuracy_score_train
           o balanced_adj_accuracy_score_val
           o mean_accuracy_train
           o mean_accuracy_val
           o epoch_mean_weighted_precision
           o epoch_mean_weighted_recall
           o epoch_loss_train
           o epoch_loss_val
        
        :param tally_coll: the collection of results for
            each epoch
        :type tally_coll: TrainResultCollection
        '''
        
        summary = EpochSummary(tally_coll, epoch, logger=self.log)
        hparms_vals = {
            'pretrained_layers' : f"{network_name}_{num_pretrained_layers}",
            'lr_initial': self.config.Training.lr,
            'optimizer' : self.config.Training.optimizer,
            'batch_size': self.config.Training.batch_size,
            'kernel_size' : self.config.Training.kernel_size
            }
        
        # Epoch loss summaries may be 
        # unavailable, because not even enough
        # data for a single batch was available.
        # Put in -1 as a warning; it's unlikely
        # we could send nan to tensorboard:
        try:
            loss_train = summary.epoch_loss_train
        except AttributeError:
            loss_train = -1.0
            
        try:
            loss_val = summary.epoch_loss_val
        except AttributeError:
            loss_val = -1.0
        
        
        metric_results = {
                'zz_balanced_adj_accuracy_score_train' : summary.balanced_adj_accuracy_score_train,
                'zz_balanced_adj_accuracy_score_val':summary.balanced_adj_accuracy_score_val,
                'zz_mean_accuracy_train' : summary.mean_accuracy_train,
                'zz_mean_accuracy_val': summary.mean_accuracy_val,
                'zz_epoch_mean_weighted_precision': summary.epoch_mean_weighted_precision,
                'zz_epoch_mean_weighted_recall' : summary.epoch_mean_weighted_recall,
                'zz_epoch_loss_train' : loss_train,
                'zz_epoch_loss_val' : loss_val
                }
        
        self.writer.add_hparams(hparms_vals, metric_results)

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
        
        :param model: any pytorch model
        :type model: torchvision.models
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
         
        :param tnsr: tensor to move to GPU, or
            'pretend' to move there
        :type tnsr: torch.Tensor
        :return a tensor; either a newly GPU resident
            one, or the tnsr itself
        :raise ValueError if tnsr is not a tensor
        '''
        
        if self.device != self.cuda:
            self.cpu_tensor_stack.append(tnsr)
            return tnsr


        if not isinstance(tnsr, Tensor):
            raise ValueError(f"Attempt to push to GPU, but '{tnsr}' is not a tensor")

        new_tnsr = tnsr.to('cuda')
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
        
        :return: a tensor; either one that was already
            in the CPU, or one that results from moving
            a tensor from the GPU to the CPU
        :raises IndexError when popping from an empty stack
        '''
        if self.device != self.cuda:
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
        if self.device != self.cuda:
            return
        
        # Move the model off the GPU
        self.model.cpu()
        
        # Move all tensors to CPU
        while len(self.gpu_tensor_stack) > 0:
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
        
        :param datetime1: earlier time
        :type datetime1: datetime
        :param datetime2: later time
        :type datetime2: datetime
        :return difference in seconds
        :rtype int
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
         
        :param x: tensor of dim 1 
        :type x: torch.Tensor
        :returns: mean of differences between 
            successive elements. Or nan if 
            tensor only has one element.
        :rtype: float
        '''
        x1 = x.roll(-1)
        res = (x1 - x)[:-1]
        m = torch.mean(res.float())
        return abs(float(m))

    #------------------------------------
    # cleanup 
    #-------------------
    
    def cleanup(self):
        '''
        Recover resources taken by collaborating
        processes. OK to call multiple times.
        '''
        self.clear_gpu()
        
        try:
            self.writer.close()
        except Exception as e:
            self.log.err(f"Could not close tensorboard writer: {repr(e)}")

        if self.init_process_group_called or dist.is_initialized():
            try:
                dist.destroy_process_group()
            except RuntimeError as e:
                if str(e) == "Invalid process group specified":
                    # Weird error from distributed_c10d.py
                    # in destroy_process_group() method.
                    # We did our best to clean up:
                    pass

    #------------------------------------
    # human_readable 
    #-------------------
    
    def human_readable(self, num, suffix='B'):
        '''
        Create human readable string from a
        (usually large) number of bytes.
         
        :param num: the number to convert
        :type num: int
        :param suffix: desired suffix in the 
            output (default B for Bytes)
        :type suffix: str
        :return: human readable str. such as
            13MB, or 140TB
        '''
        for unit in ['','K','M','G','T','P','E','Z']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)

    #------------------------------------
    # find_log_path 
    #-------------------
    
    def find_log_path(self, logfile=None):
        '''
        Given a log file, which may be None, 
        find a destinating for info/err messages.
        In all cases, specify a msg_identifier for
        LoggingService to prepend to messages. The
        id specifies this node's rank to distinquish
        msg from those of other nodes. 
        
           o logfile is None, check in config. 
             If nothing found, log to stdout.
           o if given logfile is a directory, a
             filename is constructed inside that dir.
             The name will contain this node's rank
           o if file name is given, it is augmented 
             with this node's rank. 
        
        Assumptions: self.rank contains this node's rank
        
        :param logfile: specification for the log file
        :type logfile: {None | str}
        :return: a ready-to-use LoggingService instance
        :rtype: LoggingService
        '''
        if logfile is None:
            try:
                # Log file specified in configuration?
                logfile = self.config.getpath('Paths','logfile', 
                                              relative_to=self.curr_dir)
                
                log = LoggingService(logfile=logfile,
                                     msg_identifier=f"Rank {self.rank}.{self.local_rank}")
                return log
            except ValueError:
                # No logfile specified in config.cfg
                # Use stdout by omitting the logfile:
                log = LoggingService(msg_identifier=f"Rank {self.rank}.{self.local_rank}")
                return log

        # Add rank to logfile so each training process gets
        # its own log output file.

        if os.path.isdir(logfile):
            # Create logfile below given dir:
            final_file_name = os.path.join(logfile, f"node{self.rank}_{self.local_rank}.log")
            
        else:
            # Was given a file path. Splice in the rank
            # of this node to have a unique log
            # destination with distributed processing.
              
            # Use pathlib.Path for convenience:
            log_path = Path(logfile)
            
            # Construct <given_name_without_extension><rank>.<original_extension>
            new_logname = log_path.stem + str(self.rank) + log_path.suffix
            # Put the path back together, and turn 
            # from Path instance to string
            final_file_name = str(Path.joinpath(log_path.parent, new_logname))

        log = LoggingService(logfile=final_file_name, 
                             msg_identifier=f"Rank {self.rank}.{self.local_rank}")
        return log

# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Train from a set of images."
                                     )
 
 
    parser.add_argument('-r', '--resume',
                        help='fully qualified file name to a previously saved checkpoint; \n'
                             'if not provided, start training from scratch',
                        default='');

    parser.add_argument('-l', '--logfile',
                        help='fully qualified log file name to which info and error messages \n' +\
                             'are directed. Default: stdout.',
                        default=None);

    parser.add_argument('-p', '--logginglevel',
                        choices=['critical', 'error', 'warning', 'info', 'debug', 'quiet'],
                        help=f'importance of event that warrants log entries.',
                        default='info'
                        )

    parser.add_argument('-b', '--batchsize',
                        type=int,
                        help=f'how many sample to submit to training machinery together'
                        )
    parser.add_argument('-e', '--epochs',
                        type=int,
                        help=f'how many epochs to run'
                        )
    # Used only by launch.py script! Pass
    # communication parameters:

    parser.add_argument('--MASTER_ADDR',
                        help=argparse.SUPPRESS,
                        default='127.0.0.1'
                        )
    # Used only by launch.py script! Pass
    # communication parameters:

    parser.add_argument('--MASTER_PORT',
                        help=argparse.SUPPRESS,
                        type=int,
                        default=5678  # The Distributed Data Parallel port
                        )
    # Used only by launch.py script! Pass
    # communication parameters:

    parser.add_argument('--RANK',
                        help=argparse.SUPPRESS,
                        type=int,
                        default=0
                        )
    # Used only by launch.py script! Pass
    # communication parameters:

    parser.add_argument('--LOCAL_RANK',
                        help=argparse.SUPPRESS,
                        type=int,
                        default=0
                        )
        # Used only by launch.py script! Pass
    # communication parameters:
    parser.add_argument('--MIN_RANK_THIS_MACHINE',
                        help=argparse.SUPPRESS,
                        type=int,
                        default=0
                        )
    # Used only by launch.py script! Pass
    # communication parameters:

    parser.add_argument('--WORLD_SIZE',
                        help=argparse.SUPPRESS,
                        type=int,
                        default=1
                        )
    
    parser.add_argument('--GPUS_USED_THIS_MACHINE',
                        help=argparse.SUPPRESS,
                        type=int,
                        default=0
                        )
    
    parser.add_argument('-d', '--data',
                        help='directory root of sub-directories that each contain samples \n'
                             'of one class. Can be specified in config file instead',
                        default=None)
    
    parser.add_argument('config',
                        help='fully qualified file name of configuration file, or JSON string',
                        default=None)

    args = parser.parse_args();

    # Build:
    #   {'MASTER_ADDR' : args.MASTER_ADDR,
    #    'MASTER_PORT' " args.MASTER_PORT,
    #       ...
    #

    comm_info = {}
    comm_info['MASTER_ADDR'] = args.MASTER_ADDR
    comm_info['MASTER_PORT'] = int(args.MASTER_PORT)
    comm_info['RANK']        = int(args.RANK)
    comm_info['LOCAL_RANK']  = int(args.LOCAL_RANK)
    comm_info['MIN_RANK_THIS_MACHINE'] = int(args.MIN_RANK_THIS_MACHINE),
    comm_info['GPUS_USED_THIS_MACHINE'] = int(args.GPUS_USED_THIS_MACHINE)
    comm_info['WORLD_SIZE']  = int(args.WORLD_SIZE)
    
    # GPUS_USED_THIS_MACHINE may be None to indicate
    # instruction to use all GPUs on this machine:
    comm_info['GPUS_USED_THIS_MACHINE'] = \
        None if args.GPUS_USED_THIS_MACHINE is None \
             else int(args.GPUS_USED_THIS_MACHINE)

    if args.logginglevel == 'critical':
        logging_level = logging.CRITICAL
    elif args.logginglevel == 'error':
        logging_level = logging.ERROR
    elif args.logginglevel == 'warning':
        logging_level = logging.WARNING
    elif args.logginglevel == 'info':
        logging_level = logging.INFO
    elif args.logginglevel == 'debug':
        logging_level = logging.DEBUG
    elif args.logginglevel == 'quiet':
        logging_level = logging.NOTSET
    else:
        # Won't happen, b/c argparse will enforce
        raise ValueError(f"Logging level {args.logginglevel} illegal.")

    ret_code = BirdTrainer(
                config_info=args.config,
                root_train_test_data=args.data,
                checkpoint=args.resume,
                batch_size=args.batchsize,
                logfile=args.logfile,
                logging_level=logging_level,
                comm_info=comm_info
                ).train()
    sys.exit(ret_code)
