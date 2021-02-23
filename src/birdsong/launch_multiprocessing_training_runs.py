#!/usr/bin/env python3
'''
Created on Feb 2, 2021

@author: paepcke
'''
from _collections import OrderedDict
import argparse
from collections import UserDict
import datetime
import faulthandler
from itertools import product
from json.decoder import JSONDecodeError as JSONError
from multiprocessing.pool import Pool
from multiprocessing.queues import SimpleQueue
import os
import signal
import socket
import sys
import time

import json5
from logging_service.logging_service import LoggingService

from birdsong.birds_train_parallel import BirdTrainer
from birdsong.utils.neural_net_config import NeuralNetConfig, ConfigError
import multiprocessing as mp


# Empty-queue exception:
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))




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
        
        # For passing on to children later:
        self.logfile = logfile
        
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
        cpu_only = len(gpu_ids_to_use) == 0

        # Q-size: if CPU only, have room for as many
        # times 'None' as there are configs to run.
        # Else only make the queue as large as the
        # number of GPUs we are allowed:
        
        if cpu_only:
            num_cpus = mp.cpu_count()
            if num_cpus <= 2:
                num_cpus = 1
            else:
                # Be nice; leave 2 CPUs for others:
                num_cpus -= 2
            gpu_id_pool = set(range(num_cpus))
        else:
            gpu_id_pool = set(gpu_ids_to_use.copy())
        
        who_is_who = {}
        
        for config_idx, config in enumerate(run_configs):
            #***************
            #self.worker_starter(config.to_json(), 0)
            #return
            #***************
            
            # Put finished processes to rest, else
            # they'll be zombies:
            while len(gpu_id_pool) == 0:
                time.sleep(3)
                # Did any of the processes finish?
                curr_procs = list(who_is_who.keys())
                for proc in curr_procs: 
                    if not proc.is_alive():
                        # Harvest the proc's GPU ID:
                        gpu_id_pool.add(who_is_who[proc])
                        proc.join()
                        del who_is_who[proc]

            gpu_id = gpu_id_pool.pop()
            proc_name = f"Config{config_idx}_gpu{gpu_id}"
            proc = mp.Process(target=self.worker_starter,
                              args=(config.to_json(), gpu_id),
                              name=proc_name
                              ) 
                                    
            proc.start()
            who_is_who[proc] = gpu_id
            
        for proc in who_is_who.keys():
            if not self.quiet:
                print(f"Waiting for proc {proc.name} to finish...")
            proc.join()

    #------------------------------------
    # worker_starter 
    #-------------------
    
    def worker_starter(self, config, gpu_id):
        '''
        This method will run in each CHILD process,
        i.e. not in the run_configurations() that
        forks the child.
        @param config:
        @type config:
        @param gpu_id:
        @type gpu_id:
        @param parent_logfile:
        @type parent_logfile:
        '''
        
        #***********
        #print(f"Worker starter: {config}")
        print("Child checking in")
        #***********

        comm_info = {}
        comm_info['MASTER_ADDR'] = '127.0.0.1'
        comm_info['MASTER_PORT'] = 5678
        comm_info['RANK']        = 0 # For now
        comm_info['LOCAL_RANK']  = gpu_id
        comm_info['MIN_RANK_THIS_MACHINE'] = 0
        comm_info['WORLD_SIZE']  = 1
        comm_info['GPUS_USED_THIS_MACHINE']  = self.my_gpus

        curr_dir = os.path.dirname(__file__)
        log_dir = os.path.join(curr_dir, 'runs_logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        res = BirdTrainer(
                  config_info=config,
                  checkpoint=False, #**********
                  logfile=log_dir,
                  comm_info=comm_info
                  ).train()

        sys.exit(res)

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
    hparms_spec = {'lr' : [0.01],
                   'optimizer'  : ['RMSProp'],
                   'batch_size' : [4],
                   'kernel_size': [7]
                   }
    
#     hparms_spec = {'lr' : [0.001],
#                    'optimizer'  : ['Adam'],
#                    'batch_size' : [32],
#                    'kernel_size': [3,7]
#                    }
#     hparms_spec = {'lr' : [0.001],
#                    'optimizer'  : ['Adam', 'RMSprop', 'SGD'],
#                    'batch_size' : [4, 32,64,128],
#                    'kernel_size': [3,7]
#                    }

    #**************

    TrainScriptRunner(config_file, 
                      hparms_spec, 
                      quiet=args.quiet,
                      dryrun=args.dryrun)