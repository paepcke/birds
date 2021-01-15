#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

from _collections import OrderedDict
from argparse import ArgumentParser
import argparse
from json.decoder import JSONDecodeError as JSONError
import os
from pathlib import Path
import signal
import socket
from subprocess import PIPE, TimeoutExpired
import subprocess
import sys

import json5
from logging_service.logging_service import LoggingService

from birdsong.utils.dottable_config import DottableConfigParser


#import GPUtil
# For remote debugging via pydev and Eclipse:
# #*****************
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
#     print("Calling settrace to pause in Eclipse for debugging there. ")
#     pydevd.settrace('localhost', port=4040)
# **************** 
r"""
****** 
   o This script run exactly once on each machine
   o Starts as many copies of train script as are
     GPUs to be used on the machine where the script
     is started.
   o Scripts learn from world_map whether they are
     the master machine.
   o RANK vs. LOCAL_RANK
   o Coordinating process is not a separate process,
     but simply a role taken by one of the training
     script copies.
   o For fully qualified domain names for participating
     machines to match entries in the world_map, each
     host's /etc/hosts needs an entry
     
        127.0.0.1    my_machine.my_domain
     
     This means that Python's
     
        socket.getfqdn() should return my_machine.my_domain

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
    '''
    Helper class for command line argument
    parsing. Main method is parse_arg(), which 
    overrides the parent method. Knows how to 
    separate command line args destined for this
    launch script from the subsequent ones intended
    for the training scripts being launched. 
    '''

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
        Helper function to parse the command line options
        intended for this launcher and the ones destined
        to the training script copies. 
        
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
    
        self.add_argument('-f', '--logfile',
                            help='fully qualified log file name to which info and error messages \n'
                                 'are directed. Default: stdout.', 
                           default=None);
                           
        self.add_argument('-l', '--logginglevel',
                            choices=['critical', 'error', 'warning', 'info', 'debug', 'quiet'],
                            help=f'importance of event that warrants log entries.',
                            default='info'
                            )
                            
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
                               'logginglevel', 
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
    
    def __init__(self,
                 unittesting=False):

        self.hostname = socket.getfqdn()
        if unittesting:
            # Let unittests create an instance
            # and call individual methods:
            return
        
        # Logging to console during launch:
        self.log = LoggingService()
        
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
        # Separate the args for this launch script
        # from the args destined for the copies of
        # the train script:
        self.launch_args = all_args['launch_args']
        self.script_args = all_args['script_args']
        
        # Build the gpu_landscape dict:
        self.gather_world_layout(self.launch_args)
        
        self.GPUS_USED_THIS_MACHINE = self.gpu_landscape[self.hostname]['num_gpus']
        
    #------------------------------------
    # gather_world_layout
    #-------------------
    
    def gather_world_layout(self, launch_args):
        '''
        # Compute a unique number for each GPU within
        # the group of nodes (machines). Starting with
        # the master node's first GPU as 0 (if master node
        # has a GPU.
        # The resulting GPU layout is assigned to
        # variable gpu_landscape: 
        
        
        @param launch_args:
        @type launch_args:
        '''

        try: 
            config_file = launch_args['config']
            if not os.path.exists(config_file):
                raise ConfigError(f"Configuration file {config_file} that was provided as command line arg does not exist.")
        except KeyError:
            raise RuntimeError("Error: launch args must include a config file. See config.cfg.Example in project root")
        
        self.config = DottableConfigParser(config_file)
        
        try:
            self.world_map_path = self.config.getpath('Paths', 
                                                      'world_map',
                                                      relative_to=self.curr_dir
                                                      )
        except KeyError:
            raise RuntimeError(f"Could not find entry for 'world_map' in config file {config_file}")
        
        self.world_map = self.read_world_map(self.world_map_path)                    
        # Ensure that this machine has an
        # entry in the world_map:
        try:
            # Get this machine's info (sub)dict:
            _my_world_info = self.world_map[self.hostname]
        except KeyError:
            raise ConfigError(f"World map file does not contain entry for this machine ({self.hostname})")
        
        self.compute_landscape = {}
        
        # Whether or not machine running this
        # code is the master node:
        self.am_master_node = False

        # Build gpu_landscape, which maps
        # machine names to the rank range
        # that they occupy via the number of
        # their GPUs
        #
        #    {machine_name1 : [1],
        #     machine_name2 : [0],
        #     machine_name3 : [1,2,3],        

        self.gpu_landscape = self.build_compute_landscape(self.world_map)
        
        if self.master_hostname is None:
            raise ConfigError(f'No master machine in {self.world_map_path}; one entry needs to be "master" : 1')
        
        # Common pytorch port is either in the config file,
        # or we use the pytorch default
        self.MASTER_PORT = self.config.getint('Parallelism',
                                              'master_port',
                                              self.COMM_PORT
                                              )
        
        # Handle special case: no GPUs anywere, and
        # we are on node 0: in that case start a single
        # copy of the training script. If it is written
        # properly, it will detect the absence of a GPU,
        # and use the CPU. This happens during debugging
        # on a laptop:

        if self.WORLD_SIZE == 0 and self.am_master_node:
            self.WORLD_SIZE += 1
            
        # If trying to launch on a node without GPUs,
        # when GPUs are available elsewhere, refuse to
        # start the script (is this needed?):
        if not TESTING:
            if self.my_gpus == 0 and self.WORLD_SIZE > 0:
                raise RuntimeError("This machine does not have any GPU, but others do; training script not started.")
        
    #------------------------------------
    # launch_scripts 
    #-------------------
    
    def launch_scripts(self):
        '''
        Launch (possibly) multiple copies of
        the training script. Use world_map.json
        to know how many, and which GPUs this
        machine is to use.
        
        Each copy is told:
        
            o MASTER_ADDR  # Where to reach the coordinating process
            o MASTER_PORT  # Corresponding port
            o RANK         # The copy's sequence number, which is
                           # Unique across all participating machines
            o LOCAL_RANK   # Which of this machine's GPU to use (0-origin)
            o WORLD_SIZE   # How many GPUs are used on all machines together
            o GPUS_USED_THIS_MACHINE # Number of GPUs *used* on this
                                     # machine, according to the world_map.

        '''
        
        # Compute a unique number for each GPU within
        # the group of nodes (machines). Starting with
        # the master node's first GPU as 0 (if master node
        # has a GPU.
        # The resulting GPU layout is assigned to
        # variable gpu_landscape: 
        #
        #     {<machine_name> : 

        # This machine's range of ranks:
        rank_range = self.gpu_landscape[self.hostname]['rank_range']
        this_machine_gpu_ids = self.gpu_landscape[self.hostname]['gpu_device_ids']

        local_rank = 0
        # Map from process object to rank (for debug msgs):
        self.who_is_who = OrderedDict()
        for rank in rank_range: 

            cmd = self.training_script_start_cmd(rank, 
                                                 len(this_machine_gpu_ids),
                                                 local_rank,
                                                 self.launch_args,
                                                 self.script_args
                                                 )
                
            # Copy stdin, and give the copy to the subprocess.
            # This enables the subprocess to ask user whether
            # to save training state in case of a cnt-C:
            newstdin = os.fdopen(os.dup(sys.stdin.fileno()))
            
            # Spawn one training script.

            process = subprocess.Popen(cmd,
                                       stdin=newstdin,
                                       stdout=None,  # Script inherits this launch
                                       stderr=None   # ... script's stdout/stderr  
                                       )
            self.who_is_who[process] = rank
            local_rank += 1
        
        if not self.launch_args['quiet']:
            print(f"Node {self.hostname} {os.path.basename(sys.argv[0])}: Num processes launched: {len(self.who_is_who)}")
            if self.am_master_node:
                print(f"Awaiting {self.WORLD_SIZE} process(es) to finish...")
            else:
                print(f"Awaiting {self.my_gpus} process(es) to finish...")
        
        failed_processes = []
        try:
            for process in self.who_is_who.keys():
                process.wait()
                if process.returncode != 0:
                    failed_processes.append(process)
                continue
        except KeyboardInterrupt:
            # Gently kill the training scripts:
            self.handle_cnt_c(self.who_is_who.keys())
            
        num_failed = len(failed_processes)
        if num_failed > 0:
            print(f"Number of failed training scripts: {num_failed}")
            for failed_process in failed_processes:
                train_script   = self.launch_args['training_script']
                script_rank    = self.who_is_who[failed_process]
                msg = (f"Training script {train_script} (rank {script_rank}) encountered error(s); see logfile")
                print(msg)

    #------------------------------------
    # training_script_start_cmd 
    #-------------------

    def training_script_start_cmd(self, 
                                  rank,
                                  gpus_used_this_machine,
                                  local_rank,
                                  launch_args,
                                  script_args):

        # Build the shell command line,
        # starting with 'python -u':
        cmd = [sys.executable, "-u"]

        cmd.append(launch_args['training_script'])

        # Add the args for the script that were
        # in the command line:
        for arg_name in script_args.keys():
            script_arg_val = script_args[arg_name]
            if script_arg_val is None or arg_name == 'config':
                # Skip over non-specified CLI args:
                continue
            cmd.append(f"--{arg_name}={script_args[arg_name]}")

        # Add the 'secret' args that tell the training
        # script all the communication parameters:
        
        cmd.extend([f"--MASTER_ADDR={self.MASTER_ADDR}",
                    f"--MASTER_PORT={self.MASTER_PORT}",
                    f"--RANK={rank}",
                    f"--LOCAL_RANK={local_rank}",
                    f"--WORLD_SIZE={self.WORLD_SIZE}",
                    f"--GPUS_USED_THIS_MACHINE={gpus_used_this_machine}"
                    ])
        
        # Finally, the obligatory non-option arg
        # to the training script: the configuration
        # file:
        
        config_file_name = script_args['config']
        cmd.append(config_file_name)
        
        self.log.debug(f"****** Launch: the cmd is {cmd}")
        return cmd
        

    #------------------------------------
    # read_world_map 
    #-------------------
    
    def read_world_map(self, path):
        '''
        Read the JSON5 world map file, and 
        return a corresponding dict. JSON5
        allows something like:
        
        /*
            This is a block comment.
            Notice the lacking quote
            chars around the keys below.
            The are optional in JSON5
            
        */
        
        {quintus.stanford.edu : {
            "master" : Yes
            "gpus" : 2
         },
        
         quatro.stanford.edu  : {
             "gpus" : 2,
             "devices" : [1,2]
         }
        }
        
        BUT: JSON5 gets angry at dots in the 
             keys. 
        So we first read the file, and try to find 
        the machine names. We temporarily replace
        them with an acceptable marker, and then 
        convert back.
                
        @param path: path to world map file
        @type path: string
        '''
        dot_substitute = '___'
        
        try:
            # Read all the world map file lines:
            with open(path, 'r') as world_map_fd:
                tmp_world_map = world_map_fd.readlines()
        except IOError as e:
            raise IOError(f"World map file at {path} not found") from e

        # Replace occurrences of '.' with dot_substitute:
        new_text = []
        for line in tmp_world_map:
            new_text.append(line.replace('.', dot_substitute))
        
        # ... and make one string from all the lines:
        json_str = '\n'.join(new_text)

        try:
            # Hopefully, JSON5 will eat it now:
            world_map_almost = json5.loads(json_str)
        except JSONError as e:
            raise JSONError(f"World map file at {path} contains bad JSON") from e
        
        # Need to fix all the dot substitutions. 
        # At this point the data structure is
        #    { <machine_name> : {spec_attr1 : val1,
        #                        spec_attr2 : val2,
        #                       }
        #    }

        # Fix the machine names first:
        mach_names_fixed = [machine_name.replace(dot_substitute, '.')
                              for machine_name in world_map_almost.keys()]
        
        machine_specs_fixed = []
        
        # Now dig into each of the nested machine spec
        # dicts, and fix attrs and values there:
        for spec in world_map_almost.values():
            # Spec is a dict nested inside the outer one:
            spec_fixed = {key.replace(dot_substitute, '.') :
                          val.replace(dot_substitute, '.')
                            if isinstance(val, str) else val
                          for key,val in spec.items()
                          }
            machine_specs_fixed.append(spec_fixed)
        
        # Put it all together:
        world_map = {machine_name : spec_dict
                     for machine_name, spec_dict
                      in zip(mach_names_fixed, machine_specs_fixed)
                     }
        
        return world_map
    

    #------------------------------------
    # build_compute_landscape
    #-------------------
    
    def build_compute_landscape(self, world_map):
        '''
        # Using the world_map.json config file, build 
        # a dict self.gpu_landscape like this:
        #
        #    {'machine_name1' : {'start_rank'    : <int>,
        #                        'num_gpus'      : <int>,
        #                        'gpu_device_ids': [<int>,<int>,...]
        #    {'machine_name2' : {'start_rank'    : <int>,
        #                        'num_gpus'      : <int>,
        #                        'gpu_device_ids': [<int>,<int>,...]
        #    } 
        #
        # Also sets 
        #     o self.master_hostname, the hostname
        #       running the one process that coordinates all others.
        #     o self.WORLD_SIZE, number of GPUs used across all machines
        #     o self.my_gpus, the number of GPUs on this machine
        
        @param world_map:
        @type world_map:
        @return: information about how many GPUs are
            on each node
        @rtype: OrderedDict
        '''
        
        if not self.hostname in world_map.keys():
            raise ConfigError(f"World map does not contain an entry for this machine {self.hostname}")
        
        # World size is the number of training script processes, 
        # which is equal to number of GPUs used on all participating
        # machines combined:
        
        # Number of GPUs across all machines:
        self.WORLD_SIZE    = 0
        
        self.master_hostname = None
        
        # Go through the world map, machine (a.k.a. node)
        # one at a time, in alpha order of the machine
        # names to ensure all copies of this script
        # come to the same conclusions about ranks
        
        # Build gpu_landscape:
        #
        #    {'machine_name1' : {'start_rank'    : <int>,
        #                        'num_gpus'      : <int>,
        #                        'gpu_device_ids': [<int>,<int>,...]
        #    {'machine_name2' : {'start_rank'    : <int>,
        #                        'num_gpus'      : <int>,
        #                        'gpu_device_ids': [<int>,<int>,...]
        #    } 
        #
        # The structure is an OrderedDict(), containing
        # machines alphabetically by name. This discipline
        # is required so that all copies of this launch script
        # (one copy per machine) arrive at the same ordering of
        # GPUs: 
        
        gpu_landscape = OrderedDict({})
        
        for machine_name in sorted(world_map.keys()):

            # Get dict of info about the machine:
             
            machine_info = world_map[machine_name]
            
            machine_gpus = machine_info['gpus'] 
            gpu_landscape[machine_name] = {}
            gpu_landscape[machine_name]['num_gpus'] = machine_gpus
            
            # List of GPU numbers to use is optional
            # in world_maps:
            
            machine_gpus_to_use = machine_info.get('devices', None)
            if machine_gpus_to_use is None:
                # Use all GPUs on that machine:
                machine_gpus_to_use = list(range(machine_gpus))

            gpu_landscape[machine_name]['gpu_device_ids'] = machine_gpus_to_use
            
            # Accept all kinds of affirmatives as values:
            # for identification of the master node entry:
            
            is_master_node = machine_info.get('master', False) \
                in [1, 'True', 'true', 'Yes', 'yes']
                
            if is_master_node:
                self.master_hostname = machine_name
                if machine_name == self.hostname:
                    self.am_master_node = True
                self.MASTER_ADDR = socket.gethostbyname(machine_name)
            
            self.WORLD_SIZE += machine_gpus
                    
        # Go through the machine enries in gpu_landscape, and
        # assign rank ranges to each. Must start with 
        # the master node, b/c it must start with rank 0:

        gpu_landscape[self.master_hostname]['rank_range'] = \
            list(range(gpu_landscape[self.master_hostname]['num_gpus']))
        
        # Start assigning more ranks after 
        # the GPUs of the master:
        running_rank = gpu_landscape[self.master_hostname]['num_gpus']
        for machine_name in gpu_landscape.keys():
            if machine_name == self.master_hostname:
                # We already did the master node
                continue 
            num_gpus = gpu_landscape[machine_name]['num_gpus']
            gpu_landscape[machine_name]['rank_range'] = \
                list(range(running_rank, running_rank + num_gpus))
            running_rank += num_gpus
            
        self.my_gpus = gpu_landscape[self.hostname]['num_gpus']
        self.gpu_landscape = gpu_landscape
        return gpu_landscape

    #------------------------------------
    # handle_cnt_c 
    #-------------------

    def handle_cnt_c(self, procs):
        '''
        Given a list of process instances,
        Send SIGINT (cnt-C) to them:
        @param procs:
        @type procs:
        '''
        for process in procs:
            process.send_signal(signal.SIGINT)
            process.wait()

# --------------------- Main ---------------

if __name__ == "__main__":
    launcher = TrainScriptLauncher()
    launcher.launch_scripts()
