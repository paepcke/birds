'''
Created on Feb 2, 2021

@author: paepcke
'''
from _collections import OrderedDict
import argparse
from collections import UserDict
import copy
import datetime
import faulthandler
from io import StringIO
from itertools import product
from json.decoder import JSONDecodeError as JSONError
import logging
from multiprocessing.pool import Pool
import os
import signal
import socket
import sys

import json5
from logging_service.logging_service import LoggingService
import psutil

from birdsong.birds_train_parallel import BirdTrainer
from birdsong.utils.neural_net_config import NeuralNetConfig, ConfigError
import multiprocessing as mp


# TODO: 
#   nothing right now
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
#****************
#******************
def test_multi(arg):
    print(f"Test_multi was called: {arg}")
    return True
#******************

# ------------------------ Specialty Exceptions --------
class TrainScriptRunner(object):
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 starting_config_src,
                 hparms_spec, 
                 logfile=None,
                 quiet=False,
                 dryrun=False,
                 unittesting=False):
        '''
        Specifications expected like this
        *Ordered* dict (i.e. sequence of 
        keys and values always the same for
        keys()/values()/items() methods:
        
            {<hparm1> : [val1_1, val1_2, ...],
             <hparm2> : [val2_1, val2_2, ...]
             }
        
        @param starting_config_src: a configuration 
            whose neural net related parameters will 
            be modified below for each run.
        @type starting_config_src: {str | NeuralNetConfig}            
        @param hparms_spec:
        @type hparms_spec:
        @param training_script: path to the training script
            of which to run multiple copies. If None, will
            look in config for Path:train_script.
        @type training_script: {None | str}
        @param logfile: where to log runtime information. If None,
            log to console
        @type logfile: {None | str}
        @param quiet: whether or not to report progress
        @type quiet: bool
        @param unittesting: set to True if unittesting so that
            __init__() will only do a minimum, and allows unittests
            to call other methods individually
        @type bool
        '''

        if logfile is not None:
            self.log = LoggingService(logfile=logfile)
        else:
            self.log = LoggingService()

        self.quiet = quiet

        self.curr_dir = os.path.dirname(__file__)
        self.hostname = socket.getfqdn()
        # No GPUs identified so far:
        self.WORLD_SIZE = 0
        
        starting_config = NeuralNetConfig(starting_config_src)
        if unittesting:
            # Leave calling of the methods below
            # to the unittests
            return

        self.gpu_landscape = self.obtain_world_map(starting_config)
        
        # Get list of dicts of hparm-name/hparm_value pairs;
        # one for each of the runs
        
        the_run_dicts   = self.get_runs_hparm_specs(hparms_spec)
        
        # Turn the run dicts into configurations
        # that that modify the starting config:
        the_run_configs = self.gen_configurations(starting_config, 
                                                  the_run_dicts)

        if dryrun:
            print("Dryrun:")
            print(f"Would run {len(the_run_dicts)} processes with these configs:")
            for configs in the_run_dicts:
                
                print(configs)
            return
        
        # Provide support for cnt-c terminating the training
        # script processes nicely:

        self.cnt_c_received = False
        signal.signal(signal.SIGTERM, self.handle_cnt_c)
        # Start one training script for each configuration:
        self.run_configurations(the_run_configs) 
        
    #------------------------------------
    # get_runs_hparm_specs
    #-------------------
    
    def get_runs_hparm_specs(self, hparms_spec):
        '''
        Create a list of dicts. Each dict 
        holds the value for each of the hparms
        for one run.
        
        @param hparms_spec: client's dict of 
            {param_name : [val1, val2, ...]}
        @type hparms_spec: {str : [Any]}
        @return: list of dicts
        '''

        # Running example:
        
        #     {'lr'         : [0.001],
        #      'optimizer'  : ['Adam','RMSprop','SGD'],
        #      'batch_size' : [32, 64, 128],
        #      'kernel_size': [3, 7]
        #     })

        # Parameters to vary:
        parm_names = list(hparms_spec.keys())
        
        # Iterate through list of value combinations:
        #     (0.001, 'Adam', 32, 3)
        #     (0.001, 'Adam', 32, 7)
        #     (0.001, 'Adam', 64, 3)
        #        ...
        # to get a list of dicts, each with a
        # unique combination of parameter settings:
        #
        #     [{'lr': 0.001,
        #       'optimizer'  : 'Adam',
        #       'batch_size' : 32,
        #       'kernel_size': 3},
        #       {'lr': 0.001,
        #        'optimizer'  : 'Adam',
        #        'batch_size' : 32,
        #        'kernel_size': 7},
        #       {...}
        #       ...
        #     ]
        
        hparms_permutations = []
        
        for _perm_num, ordered_vals_tuple in enumerate(product(*hparms_spec.values())):
            # Have something like:
            #   (0.001, 'Adam', 32, 3)
            # Separate dict for each combo:
            conf_dict = dict(zip(parm_names, ordered_vals_tuple))
            hparms_permutations.append(conf_dict)
        
        return hparms_permutations
        
    #------------------------------------
    # gen_configurations
    #-------------------
    
    def gen_configurations(self, config, config_dicts):
        '''
        Takes a list of dicts, and returns a list
        of NeuralNetConfig instances. Each dict
        contains one hyperparameter settings combination
        that is to be tested. Such as:
             [{'lr': 0.001,
               'optimizer': 'Adam',
               'batch_size': 32,
               'kernel_size': 3},
               {'lr': 0.001,
                'optimizer': 'Adam',
                'batch_size': 32,
                'kernel_size': 7},
               {...}
               ...
             ]

        Each return configuration is a copy of the
        config, modified for the respective
        hyperparameter settings. All other parts of
        the config are kept.
        
        @param config: a configuration with
            all settings; only the hyperparameter 
            settings will be modified
        @type config: NeuralNetConfig
        @param config_dicts: one dict of hyperparm-name : value
            for each process to run independently
        @type config_dicts: [{str : Any}]
        @return: list of configurations for the classifier
            script to run
        @rtype: [NeuralNetConfig]
        '''
        
        configs = []
        for conf_dict in config_dicts:
            conf_copy = config.copy()
            for param_name, val in conf_dict.items():
                conf_copy.add_neural_net_parm(param_name, val)
            configs.append(conf_copy)
        return configs

    #------------------------------------
    # obtain_world_map 
    #-------------------
    
    def obtain_world_map(self, initial_config):
        try:
            self.world_map_path = initial_config.getpath('Paths', 
                                                         'world_map',
                                                         relative_to=self.curr_dir
                                                         )
        except KeyError:
            raise RuntimeError(f"Could not find entry for 'world_map' in initial config")
        
        self.world_map = self.read_world_map(self.world_map_path)                    
        # Ensure that this machine has an
        # entry in the world_map:
        try:
            # Get this machine's info (sub)dict:
            _my_world_info = self.world_map[self.hostname]
        except KeyError:
            raise ConfigError(f"World map file does not contain entry for this machine ({self.hostname})")
        
        self.compute_landscape = {}
        gpu_landscape = self.build_compute_landscape(self.world_map)
        return gpu_landscape

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

        machine_name = self.hostname
        machine_info = world_map[self.hostname]

        try:
            machine_gpus = machine_info['gpus']
        except KeyError:
            print("World map must include a 'gpus' entry; the value may be 0")
                 
        gpu_landscape[machine_name] = {}
        gpu_landscape[machine_name]['num_gpus'] = machine_gpus
        
        # List of GPU numbers to use is optional
        # in world_maps:
        
        machine_gpus_to_use = machine_info.get('devices', None)

        if machine_gpus_to_use is None:
            # Use all GPUs on this machine:
            machine_gpus_to_use = list(range(machine_gpus))

        gpu_landscape[machine_name]['gpu_device_ids'] = machine_gpus_to_use
        
        # Add 1 process for the on this machine,
        # which will run on its CPU, b/c no GPUs
        # are available:
        self.WORLD_SIZE += machine_gpus if machine_gpus > 0 else 1
                    
        self.my_gpus = gpu_landscape[self.hostname]['num_gpus']
        self.gpu_landscape = gpu_landscape
        return gpu_landscape

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
    # run_configurations
    #-------------------
    
    def run_configurations(self, run_configs):
        '''
        Takes a list of run configuration that 
        specify the details of a training run 
        (lr, optimizer to use, etc.) Spawns
        independent training script processes, one 
        with each of the configurations.
        
        If fewer CPUs/GPUs are available than the
        number of configs in run_configs, waits for
        processes to finish, then launches more.
        
        Configs may take one of three forms:
            o File path to a config file
            o JSON string with all the config info
            o A NeuralNetConfig instance

        Use world_map.json to know how many, and which 
        GPUs this machine is to use.
        
        Each copy of the training script is told:
        
            o RANK         # The copy's sequence number, which is
                           # Unique within this machine (but not 
                           # currently across machines, as in in 
                           # distributed data parallel (DDP)
            o LOCAL_RANK   # Which of this machine's GPU to use (0-origin)
            o WORLD_SIZE   # How many GPUs are used on all machines together
            o GPUS_USED_THIS_MACHINE # Number of GPUs *used* on this
                                     # machine, according to the world_map.
                                     # (As opposed to number of GPUs that
                                     # exist on this machine.)

        @param run_configs: list of configurations. Each config
            may either be a JSON string, the file name of
            a config file, or a NeuralNetConfig instance
        @type run_configs: [str | NeuralNetConfig]
        @return 0 for success of all processes, else 1
        @rtype int
        '''

        gpu_ids_to_use = self.gpu_landscape[self.hostname]['gpu_device_ids']
        self.gpu_id_pool = set(gpu_ids_to_use.copy())
        cpu_only = len(gpu_ids_to_use) == 0

        # Q-size: if CPU only, have room for as many
        # times 'None' as there are configs to run.
        # Else only make the queue as large as the
        # number of GPUs we are allowed:
        
        max_workers = len(run_configs) if cpu_only else len(gpu_ids_to_use)
        
        # Callbacks for Pool to when a process finishes. 
        # The currying is to get 'self' passed to the
        # methods, along with the finished process or
        # exception instance, respectively:
        
        #************ Remove
#         proc_finished_callback    = partial(self.proc_termination_callback, self)
#         proc_ended_badly_callback = partial(self.proc_error_callback, self)
#         config_launcher           = partial(self.worker_starter, self)
        #*************

        arg_tuples = [(config.to_json(),) for config in run_configs]
        with Pool(max_workers) as pool:
            queue_manager = mp.Manager()
            self.gpu_id_q = queue_manager.Queue(max_workers)
            pool.starmap_async(
                               self.worker_starter,
                               arg_tuples,
                               chunksize=1, 
                               callback=self.proc_termination_callback,
                               error_callback=self.proc_error_callback
                               )
            if cpu_only:
                num_cpus = mp.cpu_count()
                if num_cpus <= 2:
                    num_cpus = 1
                else:
                    # Be nice; leave 2 CPUs for others:
                    num_cpus -= 2
                self.gpu_id_pool = set(range(num_cpus))

            # Initially, all GPUs (or CPUs) are available:
            gpu_id_pool_copy = self.gpu_id_pool.copy()
            for gpu_id in gpu_id_pool_copy:
                self.gpu_id_pool.remove(gpu_id)
                # Provide gpu ID to one of the
                # hungry processes:
                self.gpu_id_q.put(gpu_id)

            # Wait for everyone:
            pool.close()
            pool.join()
        
    #------------------------------------
    # worker_starter 
    #-------------------
    
    def worker_starter(self, config):
        
        #***********
        #print(f"Worker starter: {config}")
        #return "It worked"
        #***********
        
        comm_info = {}
        comm_info['MASTER_ADDR'] = '127.0.0.1'
        comm_info['MASTER_PORT'] = 5678
        comm_info['RANK']        = 0 # For now
        comm_info['LOCAL_RANK']  = None
        comm_info['MIN_RANK_THIS_MACHINE'] = 0
        comm_info['WORLD_SIZE']  = 1

        curr_dir = os.path.dirname(__file__)
        log_dir = os.path.join(curr_dir, 'runs_logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Get a GPU id, waiting if needed:
        local_rank = self.gpu_id_q.get()
        rank = comm_info['RANK']
        node_id = f"node{rank}.{local_rank}"
        comm_info['LOCAL_RANK'] = local_rank 
        
        new_log_file = self.timestamped_name(prefix=f"run_{node_id}_", 
                                             suffix='.log')
        log_path = os.path.join(log_dir, new_log_file)
        BirdTrainer(
                config_info=config,
                checkpoint=False, #**********
                logfile=log_path,
                comm_info=comm_info
                ).train()
                
        # Return the gpu_id we used
        return comm_info['LOCAL_RANK']
        
    #------------------------------------
    # proc_termination_callback 
    #-------------------
    
    def proc_termination_callback(self, terminated_process_res):
        '''
        Called by Pool object when one of the workers
        finishes training a model. The result is a list
        of one or more results returned from worker_starter().
        These are GPU IDs that the terminated process(es) used.
          
        @param terminated_process_res: return code(s) from 
            worker_starter(). Hopefully each is a GPU Id that
            a terminated process used
        @type terminated_process_res: [int]
        '''
        print(f"Results: {terminated_process_res}")
        for gpu_id in terminated_process_res:
            if type(gpu_id) == int:
                self.gpu_id_pool.add(gpu_id)
            else:
                self.log.err(f"Process returned: {gpu_id}, not a GPU ID number")
    #------------------------------------
    # proc_error_callback 
    #-------------------
    
    def proc_error_callback(self, exc):
        tracebk = sys.exc_info()[-1]
        buf = StringIO()
        tracebk.print_exc(file=buf)
        self.log.err(buf.getvalue())
        buf.close()

#**************
#         for config in run_configs:
# 
#             # Get next available GPU ID, waiting
#             # for one to free up, if necessary:
#             
#             local_rank = self.gpu_manager.obtain_gpu()
# 
#             # Create a command that is fit for passing to
#             # Popen; it will start one training script
#             # process. The conditional expression accounts for 
#             # machine with no GPU (which will run on CPU):
#             
#             cmd = self.training_script_start_cmd(local_rank, config)
#                 
#             # Copy stdin, and give the copy to the subprocess.
#             # This enables the subprocess to ask user whether
#             # to save training state in case of a cnt-C:
#             newstdin = os.fdopen(os.dup(sys.stdin.fileno()))
#             
#             # Spawn one training script. Use psutil's 
#             # Popen instead of subprocess.Popen to get
#             # the wait_procs() method on the resulting
#             # process instances:
# 
#             process = psutil.Popen(cmd,
#                                    stdin=newstdin,
#                                    stdout=None,  # Script inherits this launch
#                                    stderr=None   # ... script's stdout/stderr  
#                                    )
#             
#             if cpu_only:
#                 process.wait()
#                 # CPU op is for debugging only;
#                 # Rebel right away if something
#                 # went wrong:
#                 if process.returncode != 0:
#                     print("CPU job ran with errors; see log")
#                     return
#                 continue
#             
#             # Associate process instance with
#             # the configuration it was to run.
#             
#             self.gpu_manager.process_register(RunInfo(local_rank,
#                                                       process,
#                                                       config,
#                                                       cmd
#                                                       )
#                                               )
# 
#         # Launched all configurations; wait for
#         # the last of them to be done:
#         
#         if cpu_only:
#             print("CPU job(s) ran OK")
#             return
#         
#         # Ask for GPUs until we accounted
#         # for all that we were allowed to
#         # use; that will be indication that
#         # all processes finished:
#         
#         for _i in len(gpu_ids_to_use):
#             self.gpu_manager.obtain_gpu() 
# 
#         if not self.quiet:
#             print(f"Node {self.hostname} {os.path.basename(sys.argv[0])}: " \
#                   f"Processed {len(run_configs)} configurations")
# 
#         failed_processes = self.gpu_manager.failures()
#         if len(failed_processes) > 0:
#             print(f"Failures: {len(failed_processes)} (Check log for error entries):")
#             
#             for failed_proc in failed_processes:
#                 failed_config     = self.gpu_manager.process_info(failed_proc)
#                 train_script      = self.training_script
#                 msg = (f"Training script {train_script}: {str(failed_config)}")
#                 print(msg)
#**************
    #------------------------------------
    # training_script_start_cmd 
    #-------------------

    def training_script_start_cmd(self, 
                                  local_rank, 
                                  config):
        '''
        From provided information, creates a legal 
        command string for starting the training script.
        
        @param local_rank: GPU identifier (between 0 and 
            num of GPUs in this machine)
        @type local_rank: int
        @param config: additional information in a config instance,
            or a path to a configuration file
        @type config: {NeuralNetConfig | str}
        '''

        # Build the shell command line,
        # starting with 'python -u':
        cmd = [sys.executable, "-u", f"{self.training_script}"]

        # Add the 'secret' args that tell the training
        # script all the communication parameters:
        
        cmd.extend([f"--LOCAL_RANK={local_rank}",
                    f"--WORLD_SIZE={self.WORLD_SIZE}",
                    ])
        
        # Finally, the obligatory non-option arg
        # to the training script: the configuration.
        # Could be a file, a json string, or a 
        # NeuralNetConfig instance:
        
        if isinstance(config, NeuralNetConfig):
            # Turn into a JSON str for communicating
            # to the script:
            config_arg = config.to_json()
            self.log.info(f"\nLAUNCHING TRAINING: " +\
                          f"{NeuralNetConfig.json_human_readable(config_arg)}")
        else:
            config_arg = config
            self.log.info(f"\nLAUNCHING TRAINING from file: {config_arg}")
            
        cmd.append(config_arg)
        
        #self.log.debug(f"****** Launch: the cmd is {cmd}")
        return cmd

# ------------------- Utils --------------

    #------------------------------------
    # timestamped_name
    #-------------------
    
    def timestamped_name(self, prefix=None, suffix=None):
        '''
        Creates name from current time such that 
        the timestamp can make an acceptable part of a 
        sensible filename (e.g. no spaces).
        
        Prefix and suffix are optionally placed before and
        after the timestamp
        
        @param prefix: optional string to place before timestamp
        @type prefix: {None | str}
        @param suffix: optional string to place after timestamp
        @type suffix: {None | str}
        @return: name that includes timestamp
        @rtype: str
        '''
        if prefix is not None and type(prefix) != str:
            raise TypeError(f"Prefix must be None or string, not {prefix}")
        if suffix is not None and type(suffix) != str:
            raise TypeError(f"Suffix must be None or string, not {prefix}")

        now = datetime.datetime.now()
        timestamp = f"{now.strftime('%d-%m-%Y')}_{now.strftime('%H-%M')}"
        name = timestamp if prefix is None else prefix+timestamp
        name = name if suffix is None else name+suffix
        return name
        


    #------------------------------------
    # handle_cnt_c 
    #-------------------

    def handle_cnt_c(self):
        '''
        Given a list of process instances,
        Send SIGINT (cnt-C) to them:
        @param procs:
        @type procs:
        '''

        if self.cnt_c_received:
            # Just quit after a second
            # cnt-c:
            print(f"Hard quit. May wish to check for stray birds processes")
            sys.exit(1)

        self.cnt_c_received = True
        for process in self.gpu_manager.process_list():
            # If process is no longer running,
            # forget about it:
            if process.poll is not None:
                # Process dead:
                continue
            process.send_signal(signal.SIGTERM)
            process.wait()

    #------------------------------------
    # am_master_node 
    #-------------------
    
    def am_master_node(self):
        '''
        This method allows this script to stay somewhat
        close to the Distributed Data Parallel sibling
        launch_birds_parallel(). For this script,
        though, every process is its own master.
        '''
        return True


    #------------------------------------
    # is_json_str
    #-------------------
    
    def is_json_str(self, str_to_check):
        '''
        Very primitive test whether a passed-in
        string is (legal) JSON or not.
        
        @param str_to_check: string to examine
        @type str_to_check: str
        @return True/False
        @rtype bool
        '''
        try:
            json5.loads(str_to_check)
        except JSONError:
            return False
        return True

# ----------------------- Class RunInfo ---------

class RunInfo(UserDict):
    '''
    Instances hold information about one
    process launch. Used by run_configs(),
    and the GPUManager
    '''

    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, gpu_id, proc, config, cmd):
        super().__init__()
        self['gpu_id'] = gpu_id
        self['proc']   = proc
        self['config'] = config
        self['cmd']    = cmd
        
        self['terminated'] = False

# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Run multiple trainers independently"
                                     )

    parser.add_argument('-q', '--quiet',
                        help='suppress info msgs to the display',
                        action='store_true',
                        default=False)
    
    parser.add_argument('-d', '--dryrun',
                        help='print configs of processes that would be started, but do not start them.',
                        action='store_true',
                        default=False)
    
    parser.add_argument('-c', '--config_file',
                        help='fully qualified path to config file',
                        default=None)

    args = parser.parse_args();
    
    config_file = args.config_file
    if config_file is None:
        # Check the default location: <proj_root>/config.cfg:
        curr_dir = os.path.dirname(__file__)
        config_file = os.path.join(curr_dir, '../../config.cfg')
        if not os.path.exists(config_file):
            parser.print_usage()
            raise FileNotFoundError(f"Could not find config file at {config_file}")
    else:
        if not os.path.exists(config_file):
            parser.print_usage()
            raise FileNotFoundError(f"Could not find config file at {config_file}")

    #**************
#     hparms_spec = {'lr' : [0.001],
#                    'optimizer'  : ['Adam'],
#                    'batch_size' : [32],
#                    'kernel_size': [3,7]
#                    }
    hparms_spec = {'lr' : [0.001],
                   'optimizer'  : ['Adam', 'RMSprop', 'SGD'],
                   'batch_size' : [4, 32,64,128],
                   'kernel_size': [3,7]
                   }

    #**************

    TrainScriptRunner(config_file, 
                      hparms_spec, 
                      quiet=args.quiet,
                      dryrun=args.dryrun)