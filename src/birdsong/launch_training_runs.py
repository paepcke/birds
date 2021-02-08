'''
Created on Feb 2, 2021

@author: paepcke
'''
from _collections import OrderedDict
import argparse
from itertools import product
from json.decoder import JSONDecodeError as JSONError
import os
import socket
import subprocess
import sys
import time

import json5
from logging_service.logging_service import LoggingService

from birdsong.birds_train_parallel import BirdTrainer
from birdsong.utils.neural_net_config import NeuralNetConfig


# TODO: 
#   o In run configs: take care of 
#     more configs than CPUs/GPUs

# For remote debugging via pydev and Eclipse:
# If uncommented, will hang if started from
# on Quatro or Quintus, and will send a trigger
# to the Eclipse debugging service on the same
# or different machine:
#*****************
#
if socket.gethostname() in ('quintus', 'quatro', 'sparky'):
    # Point to where the pydev server
    # software is installed on the remote
    # machine:
    sys.path.append(os.path.expandvars("$HOME/Software/Eclipse/PyDevRemote/pysrc"))

    import pydevd
    global pydevd
    # Uncomment the following if you
    # want to break right on entry of
    # this module. But you can instead just
    # set normal Eclipse breakpoints:
    #*************
    print("About to call settrace()")
    #*************
    pydevd.settrace('localhost', port=4040)
#****************

# ------------------------ Specialty Exceptions --------
class ConfigError(Exception):
    pass

class TrainScriptRunner(object):
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 hparms_spec, 
                 starting_config_src,
                 training_script,
                 logfile=None,
                 quiet=False,
                 unittesting=False):
        '''
        Specifications expected like this
        *Ordered* dict (i.e. sequence of 
        keys and values always the same for
        keys()/values()/items() methods:
        
            {<hparm1> : [val1_1, val1_2, ...],
             <hparm2> : [val2_1, val2_2, ...]
             }
        
        @param hparms_spec:
        @type hparms_spec:
        @param starting_config_src: file path to config file,
            or a NeuralNetConfig instance
        @type starting_config_src: {str | NeuralNetConfig}
        '''

        if logfile is None:
            self.log = LoggingService(logfile=logfile)
        else:
            self.log = LoggingService()

        self.training_script = training_script
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
        
        the_run_dicts   = self.get_runs_hparm_specs(hparms_spec)
        the_run_configs = self.gen_configurations(starting_config, 
                                                  the_run_dicts)
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
        #      'kernel'     : [3, 7]
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
        #       'optimizer': 'Adam',
        #       'batch_size': 32,
        #       'kernel': 3},
        #       {'lr': 0.001,
        #        'optimizer': 'Adam',
        #        'batch_size': 32,
        #        'kernel': 7},
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
               'kernel': 3},
               {'lr': 0.001,
                'optimizer': 'Adam',
                'batch_size': 32,
                'kernel': 7},
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
                conf_copy[param_name] = val
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

        this_machine_gpu_ids = self.gpu_landscape[self.hostname]['gpu_device_ids']
        num_local_processors = 1 if len(this_machine_gpu_ids) == 0 \
            else len(this_machine_gpu_ids)

        # Map from process object to GPU ID (for debug msgs):
        self.who_is_who = OrderedDict()

        procs_started = 0
        
        for config_num, config in enumerate(run_configs):

            # Used up all CPUs/GPUs?
            if procs_started >= num_local_processors:
                # Wait for a GPU to free up:
                ret_code = self.hold_for_free_processor(procs_started)
                if ret_code != 0:
                    # If the process that terminated
                    # had an error, stop spawning new ones.
                    # (Fail early):
                    break

            # Create a command that is fit for passing to
            # Popen; it will start one training script
            # process. The max expression accounts for 
            # machine with no GPU (which will run on CPU):
            
            local_rank = 0 if len(this_machine_gpu_ids) == 0 \
                          else this_machine_gpu_ids[config_num]
            cmd = self.training_script_start_cmd(local_rank, config)
                
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
            # Associate process instance with
            # the GPU ID for error reporting later:
            
            self.who_is_who[process] = local_rank
            procs_started += 1
        
        if not self.'quiet':
            print(f"Node {self.hostname} {os.path.basename(sys.argv[0])}: " \
                  f"Num processes launched: {len(self.who_is_who)}")
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
            self.handle_cnt_c()
            pass # See which processes get the interrupt
            
        num_failed = len(failed_processes)
        if num_failed > 0:
            print(f"Number of failed training scripts: {num_failed}")
            for failed_process in failed_processes:
                train_script   = self.training_script
                failed_local_rank = self.who_is_who[failed_process]
                msg = (f"Training script {train_script} (GPU ID {failed_local_rank}) encountered error(s); see logfile")
                print(msg)

    #------------------------------------
    # hold_for_free_processor 
    #-------------------
    
    def hold_for_free_processor(self, procs_started):
        
        while True:
            for proc in self.who_is_who.keys():
                if proc.poll() is not None:
                    procs_started -= 1
                    return proc.returncode
            time.sleep(3) # Seconds
            


    #------------------------------------
    # training_script_start_cmd 
    #-------------------

    def training_script_start_cmd(self, local_rank, config):
        '''
        From provided information, creates a legal 
        command string for starting the training script.
        
        @param gpus_used_this_machine: number of GPU devices to 
            be used, according to the world_map; may be less than
            number of available GPUs
        @type gpus_used_this_machine: int
        @param local_rank: index into the local sequence of GPUs
            for for the GPU that the script is to use
        @type local_rank: int
        @param script_args: additional args for the train script
        @type script_args: {str : Any}
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
        else:
            config_arg = config
            
        cmd.append(config_arg)
        
        self.log.debug(f"****** Launch: the cmd is {cmd}")
        return cmd

# ------------------- Utils --------------

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
                        default=False)
    parser.add_argument('-c', '--config_file',
                        help='fully qualified path to config file',
                        default=None)
    parser.add_argument('training_script',
                        help='fully qualified path to the training script',
                        default=None)

    args = parser.parse_args();
    
    config_file = args.config_file
    if config_file is None:
        curr_dir = os.path.dirname(__file__)
        config_file = os.path.join(curr_dir, '../../config.cfg')
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Could not find config file at {config_file}")
    else:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Could not find config file at {config_file}")
    
    training_script = args.training_script
    if not os.path.exists(training_script):
        raise FileNotFoundError(f"Could not find training script at {training_script}")
    
    hparms_spec = {'lr' : [0.001],
                   'optimizer'  : ['Adam', 'RMSprop', 'SGD'],
                   'batch_size' : [32,64,128],
                   'kernel'     : [3,7]
                   }

    TrainScriptRunner(hparms_spec, 
                      config_file, 
                      args.training_script,
                      quiet=args.quiet)