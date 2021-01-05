#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

from argparse import ArgumentParser
import argparse
import json5
from json.decoder import JSONDecodeError as JSONError
import os
from pathlib import Path
import signal
import socket
from subprocess import PIPE
import subprocess
import sys

import torch.distributed as dist

#import GPUtil

from birdsong.utils.dottable_config import DottableConfigParser


# For remote debugging via pydev and Eclipse:
#*****************
# import socket, sys, os
# hostname = socket.gethostname()
# if hostname in ('quintus', 'quatro'):
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
#     pydevd.settrace('localhost', port=5678)
#***************** 
r"""
Based on torch.distributed.launch, with additions by Andreas Paepcke
(https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py)
Many of the comments are the originals. For more relevant documentation,
see git@github.com:paepcke/bert_train_parallel.git

o Adds ability to read world configuration from JSON file.

---------------------------------------------
`torch.distributed.launch` is a module that spawns up multiple distributed
training processes on each of the training nodes.

The utility can be used for single-node distributed training, in which one or
more processes per node will be spawned. The utility can be used for either
CPU training or GPU training. If the utility is used for GPU training,
each distributed process will be operating on a single GPU. This can achieve
well-improved single-node training performance. It can also be used in
multi-node distributed training, by spawning up multiple processes on each node
for well-improved multi-node distributed training performance as well.
This will especially be benefitial for systems with multiple Infiniband
interfaces that have direct-GPU support, since all of them can be utilized for
aggregated communication bandwidth.

In both cases of single-node distributed training or multi-node distributed
training, this utility will launch the given number of processes per node
(``--nproc_per_node``). If used for GPU training, this number needs to be less
or equal to the number of GPUs on the current system (``nproc_per_node``),
and each process will be operating on a single GPU from *GPU 0 to
GPU (nproc_per_node - 1)*.

**How to use this module:**

1. Single-Node multi-process distributed training

::

    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)

2. Multi-Node multi-process distributed training: (e.g. two nodes)


Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*

::

    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

Node 2:

::

    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
               --master_port=1234 YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3
               and all other arguments of your training script)

3. To look up what optional arguments this module offers:

::

    >>> python -m torch.distributed.launch --help


**Important Notices:**

1. This utility and multi-process distributed (single-node or
multi-node) GPU training currently only achieves the best performance using
the NCCL distributed backend. Thus NCCL backend is the recommended backend to
use for GPU training.

2. In your training program, you must parse the command-line argument:
``--local_rank=LOCAL_PROCESS_RANK``, which will be provided by this module.
If your training program uses GPUs, you should ensure that your code only
runs on the GPU device of LOCAL_PROCESS_RANK. This can be done by:

Parsing the local_rank argument

::

    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument("--local_rank", type=int)
    >>> args = parser.parse_args()

Set your device to local rank using either

::

    >>> torch.cuda.set_device(arg.local_rank)  # before your code runs

or

::

    >>> with torch.cuda.device(arg.local_rank):
    >>>    # your code to run

3. In your training program, you are supposed to call the following function
at the beginning to start the distributed backend. You need to make sure that
the init_method uses ``env://``, which is the only supported ``init_method``
by this module.

::

    torch.distributed.init_process_group(backend='YOUR BACKEND')

4. In your training program, you can either use regular distributed functions
or use :func:`torch.nn.parallel.DistributedDataParallel` module. If your
training program uses GPUs for training and you would like to use
:func:`torch.nn.parallel.DistributedDataParallel` module,
here is how to configure it.

::

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[arg.local_rank],
                                                      output_device=arg.local_rank)

Please ensure that ``device_ids`` argument is set to be the only GPU device id
that your code will be operating on. This is generally the local rank of the
process. In other words, the ``device_ids`` needs to be ``[args.local_rank]``,
and ``output_device`` needs to be ``args.local_rank`` in order to use this
utility

5. Another way to pass ``local_rank`` to the subprocesses via environment variable
``LOCAL_RANK``. This behavior is enabled when you launch the script with
``--use_env=True``. You must adjust the subprocess example above to replace
``args.local_rank`` with ``os.environ['LOCAL_RANK']``; the launcher
will not pass ``--local_rank`` when you specify this flag.

.. warning::

    ``local_rank`` is NOT globally unique: it is only unique per process
    on a machine.  Thus, don't use it to decide if you should, e.g.,
    write to a networked filesystem.  See
    https://github.com/pytorch/pytorch/issues/12042 for an example of
    how things can go wrong if you don't do this correctly.

"""

# ------------------------ Specialty Exceptions --------

class ConfigError(Exception):
    pass

#num_gpus_here = len(GPUtil.getGPUs())

TESTING = True

# ----------------------------- BirdsTrainingArgumentsParser class -----------

class BirdsTrainingArgumentsParser(ArgumentParser):

    # Format helper class:
    class BlankLinesHelpFormatter (argparse.HelpFormatter):
        
        def _split_lines(self, text, width):
            if text.find('\n') == -1:
                lines = super()._split_lines(text, width)
            else:
                lines = text.split('\n')
            split_lines = []
            for line in lines:
                while True:
                    try:
                        nl_pos = line.index('\n')
                        one_line = line[:nl_pos]
                        split_lines.append(one_line)
                        line = line[nl_pos+1:]
                    except ValueError:
                        # No more NLs:
                        split_lines.append(line)
                        break
            return split_lines


    #------------------------------------
    # parse_args 
    #-------------------

    def parse_args(self):
        """
        Helper function parsing the command line options
        for both this launcher and the distributed 
        birds_train_parallel.py script.
        
        @return: a dict with keys 'script_args' and
            'launch_args' with arg-value information
            for each of the two destinations.
        @rtype: {str : {str : ANY}}
        """
    
        curr_dir = os.path.dirname(__file__)
    
        # Optional arguments for the launch helper

        self.add_argument("-q", "--quiet", 
                            action='store_true',
                            help=f"do not print status and other info messages",
                            default=False
                            )
        # The following args are destined
        # for each of the script processes:
        self.add_argument('-r', '--resume',
                            help='fully qualified file name to a previously saved checkpoint; \n'
                                 'if not provided, start training from scratch',
                            default=None);
    
        self.add_argument('-l', '--logfile',
                            help='fully qualified log file name to which info and error messages \n'
                                 'are directed. Default: stdout.',
                            default=None);
        self.add_argument('-b', '--batchsize',
                            type=int,
                            help=f'how many sample to submit to training machinery together'
                            )
        self.add_argument('-e', '--epochs',
                            type=int,
                            help=f'how many epochs to run'
                            )
        # Used only by launch_birds_training.py script to indicate that 
        # this present script was started via the launcher:
        self.add_argument('--started_from_launch',
                            action='store_true',
                            help=argparse.SUPPRESS,
                            default=True
                            )
        self.add_argument('-d', '--data',
                            help='directory root of sub-directories that each contain samples \n'
                                 'of one class. Can be specified in config file instead',
                            default=None)
        self.add_argument('config',
                            help='fully qualified file name of configuration file',
                            default=None)
    
        args = super().parse_args()
        args_dict = vars(args)
        
        # Add argument path to the script
        # that will be launched multiple times:
        args.training_script = os.path.join(curr_dir, 'birds_train_parallel.py')
        
        script_option_names = ['resume', 
                               'logfile', 
                               'batchsize', 
                               'epochs', 
                               'data', 
                               ]
        self.script_args = {arg_name : args_dict[arg_name]
                            for arg_name
                            in script_option_names}

        # Add the obligatory, i.e. non-option
        # argument destined for the training
        # script (though we also use it here in
        # the launch script:
        
        config_path = args_dict['config']
        self.script_args['config'] = config_path
 
        # Find set of all args intended for this
        # launch script, as opposed to the training
        # script. Compute the set via the set-difference 
        # between all args, and the args for the train
        # script gathered above:
        
        launch_arg_names = set(args_dict.keys()) - set(script_option_names)
        self.launch_args = {arg_name : args_dict[arg_name]
                            for arg_name
                            in launch_arg_names}

        # For convenience, add the config file path
        # to the launch args as well:
        
        self.launch_args['config'] = config_path
       
        return {'launch_args' : self.launch_args,
                'script_args' : self.script_args
                } 
        
# -------------------------- TrainScriptLauncher

class TrainScriptLauncher:
    
    #------------------------------------
    # Constructor 
    #-------------------
    
    # Use distributed torch default port:
    COMM_PORT = '5678'
    
    def __init__(self):

        # Convenience: directory of this
        # script, and project root directory
        curr_dir = Path(__file__).parent
        proj_root = curr_dir.joinpath('../..').resolve()
        self.curr_dir = str(curr_dir)
        self.proj_root = str(proj_root)
        
        args_parser = BirdsTrainingArgumentsParser(
            formatter_class=BirdsTrainingArgumentsParser.BlankLinesHelpFormatter,
            description="PyTorch distributed training launch "
            "helper to spawn multiple distributed "
            "birds_train_parallel.py processes")
    
        all_args = args_parser.parse_args()
        self.launch_args = all_args['launch_args']
        self.script_args = all_args['script_args']
        self.gather_world_layout(self.launch_args)
        
    #------------------------------------
    # gather_world_layout
    #-------------------
    
    def gather_world_layout(self, launch_args):

        try: 
            config_file = launch_args['config']
        except KeyError:
            raise RuntimeError("Error: must have a config file. See config.cfg.Example in project root")
        
        self.config = DottableConfigParser(config_file)
        
        try:
            self.world_map_path = self.config.getpath('Paths', 
                                                      'world_map',
                                                      relative_to=self.curr_dir
                                                      )
        except KeyError:
            raise RuntimeError(f"Could not find entry for 'world_map' in config file {config_file}")
        
        try:
            with open(self.world_map_path, 'r') as world_map_fd:
                self.world_map = json5.load(world_map_fd)
        except JSONError as e:
            raise JSONError(f"World map file at {self.world_map_path} contains bad JSON") from e
            
        # Now have something like:
        # {'quintus.stanford.edu' : {
        #     'rank' : 0,
        #     'gpus' : 2},
        #
        #  'quatro.stanford.edu'  : {
        #      'rank' : 1,
        #      'gpus' : 2,
        #      'devices' : [1,2]}
        # }
        
        my_hostname = socket.getfqdn()
        try:
            # Get this machine's info (sub)dict:
            my_world_info = self.world_map[my_hostname]
        except KeyError:
            raise ConfigError(f"World map file does not contain entry for this machine ({my_hostname})")

        self.my_rank = my_world_info['rank']
        self.my_gpus = my_world_info['gpus']
        try:
            self.my_gpus_to_use = my_world_info['devices']
        except KeyError:
            self.my_gpus_to_use = None
            
        # World size is number of processes, which is
        # equal to number of GPUs used on all machines with
        # lower node rank than this one, plus this node's GPUs.
        
        # Number of GPUs on this machine, plus
        # machines with lower rank
        self.world_size    = 0
        
        # Total number of GPUs, even including
        # GPUs on higher rank machines  
        self.universe_size = 0
        self.master_hostname = None

        # Go through the world map, machine (a.k.a. node)
        # one at a time:
        for machine_name in self.world_map.keys():
            machine_info   = self.world_map[machine_name]
            machine_rank = machine_info['rank']
            machine_gpus = machine_info['gpus']
            if machine_rank == 0:
                self.master_hostname = machine_name 
            
            if machine_rank <= self.my_rank: 
                self.world_size += machine_gpus
            self.universe_size  += machine_gpus

        if self.master_hostname is None:
            raise ConfigError(f"No machine ranked 0 in {self.world_map_path}")
        
        self.MASTER_ADDR = socket.gethostbyname(self.master_hostname)
        
        # Common pytorch port is either in the config file,
        # or we use the pytorch default
        self.MASTER_PORT = self.config.getint('Parallelism',
                                              'master_port',
                                              self.COMM_PORT
                                              )

        am_master_node = my_world_info['rank'] == 0
        
        # Handle special case: no GPUs anywere, and
        # we are on node 0: in that case start a single
        # copy of the training script. If it is written
        # properly, it will detect the absence of a GPU,
        # and use the CPU. This happens during debugging
        # on a laptop:

        if self.universe_size == 0 and am_master_node:
            self.world_size += 1
            
        # If trying to launch on a node without GPUs,
        # when GPUs are available elsewhere, refuse to
        # start the script:
        if not TESTING:
            if self.my_gpus == 0 and self.universe_size > 0:
                raise RuntimeError("This machine does not have any GPU, but others do; training script not started.")
        
    #------------------------------------
    # launch_scripts 
    #-------------------
    
    def launch_scripts(self):
        
        # Set PyTorch distributed related environmental variables
        current_env = os.environ.copy()
        
        current_env["MASTER_ADDR"] = self.MASTER_ADDR
        current_env["MASTER_PORT"] = str(self.MASTER_PORT)
        current_env["WORLD_SIZE"] = str(self.world_size)
        
        if 'OMP_NUM_THREADS' not in os.environ and \
            self.my_gpus > 1:
            current_env["OMP_NUM_THREADS"] = str(1)

        processes = []
        
        # Launch as many copies of the training
        # script as this machine has available
        # according to the world map. 
        #
        # Each copy is told this machine's rank,
        # and a running index of this machine's
        # GPUs. This index starts with the CPUs
        # on the master node, continues through
        # all the GPUs on machines with lower rank
        # than this one. The GPUs of higher ranked
        # machines are not included:

        # Compute a unique number for each GPU within
        # the group of nodes (machines). Starting with
        # the master node, whose numbers are 0,1,...<ngpus_here>:
        
        lower_machine_gpus = self.gpus_below_my_rank(self.world_map) 

        #***********
        print("********Calling init_multiprocessing...")
        #***********        
        self.init_multiprocessing()
        #***********
        print("********Returned from init_multiprocessing...")
        #***********        

        # Make sure to destroy the just created process
        # group in a finally statement below:
        
        try:
             
            # Spawn as many local training script
            # copies as are (to-be-used) GPUs on this
            # machine:
            
            start_local_gpu_rank = lower_machine_gpus
            for local_rank in range(start_local_gpu_rank,
                                    start_local_gpu_rank + self.my_gpus):
    
                current_env["RANK"] = str(self.my_rank)
                current_env["LOCAL_RANK"] = str(local_rank)
        
                # Spawn one training script.
                
                # Build the shell command line,
                # starting with 'python -u':
                cmd = [sys.executable, "-u"]
        
                cmd.append(self.launch_args['training_script'])
    
                # Add the args for the script:
                for arg_name in self.script_args.keys():
                    script_arg_val = self.script_args[arg_name]
                    if script_arg_val is None or arg_name == 'config':
                        # Skip over non-specified CLI args:
                        continue
                    cmd.append(f"--{arg_name}={self.script_args[arg_name]}")
    
                # Add the 'secret' arg that tells the training
                # script that it was called from this launch
                # script, and should therefore expect the various
                # environment variables to be set:
                
                cmd.append('--started_from_launch')
                
                # Finally, the obligatory non-option arg
                # to the training script: the configuration
                # file:
                
                script_name = self.script_args['config']
                cmd.append(script_name)
                    
                # Copy stdin, and give the copy to the subprocess.
                # This enables the subprocess to ask user whether
                # to save training state in case of a cnt-C:
                newstdin = os.fdopen(os.dup(sys.stdin.fileno()))
                process = subprocess.Popen(cmd,
                                           stdin=newstdin,
                                           env=current_env,
                                           stdout=PIPE,
                                           stderr=PIPE
                                           )
                processes.append(process)
            
            if not self.launch_args['quiet']:
                print(f"Node {self.my_rank} {os.path.basename(sys.argv[0])}: Num processes launched: {len(processes)}")
                if self.my_rank == 0:
                    print(f"Awaiting {self.universe_size} processes to finish...")
                else:
                    print(f"Awaiting {self.my_gpus} processes to finish...")
            
            # Let subprocesses deal with cnt-C (keyboard interrupt):
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            for process in processes:
                process.wait()
                if process.returncode != 0:
                    (the_stdout, the_stderr) = process.communicate()
                    the_stdout = the_stdout.decode('utf-8')
                    the_stderr = the_stderr.decode('utf-8')
                    train_script   = self.launch_args['training_script']
                    msg = (f"Training script {train_script} encountered error \n"
                           f"stderr: {the_stderr} \n"
                           f"stdout: {the_stdout} \n")
                    print(msg)
                    raise subprocess.CalledProcessError(returncode=process.returncode,
                                                        cmd=cmd)
        finally:
            dist.destroy_process_group()
                
    #------------------------------------
    # gpus_below_my_rank 
    #-------------------
    
    def gpus_below_my_rank(self, world_map):
        '''
        Given a world map like this:
        
        {'quintus.stanford.edu' : {
            'rank' : 0,
            'gpus' : 2},
        
         'quatro.stanford.edu'  : {
             'rank' : 1,
             'gpus' : 2,
             'devices' : [1,2]}
        }
        
        return the number of GPUs in machines
        of lower rank than this one.
        
        @param world_map: dict layout out which machine
            has how many GPUs
        @type world_map: {str : {str : ANY}}
        @return number of GPUs in machine with
            lower rank.
        @rtype: int
        '''
        
        sum_of_gpus = 0
        for machine in world_map.keys():
            if world_map[machine]['rank'] < self.my_rank:
                sum_of_gpus += world_map[machine]['gpus']
                
        return sum_of_gpus

    #------------------------------------
    # init_multiprocessing 
    #-------------------

    def init_multiprocessing(self):
        if dist.is_nccl_available():
            backend = 'nccl'           # Preferred
        elif dist.is_mpi_available():
            backend = 'mpi'
        elif dist.is_gloo_available():
            backend = 'gloo'
        else:
            raise NotImplementedError("None of mpi/nccl/gloo torch backends installed.")
        
        os.environ['MASTER_ADDR'] = str(self.MASTER_ADDR)
        os.environ['MASTER_PORT'] = str(self.MASTER_PORT)
        os.environ['WORLD_SIZE']  = str(self.world_size)
        os.environ['RANK']        = str(self.my_rank)

        # Each process must only call init_process_group()
        # once, even if spawning multiple training scripts:

        # dist.init_process_group(backend,
        #                         world_size=self.world_size,
        #                         rank=self.my_rank
        #                         )
        dist.init_process_group(backend,
                                init_method='env://'
                                )
        
        self.log.info("And we're off!")

# --------------------- Main ---------------

if __name__ == "__main__":
    launcher = TrainScriptLauncher()
    launcher.launch_scripts()
