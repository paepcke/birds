'''
Created on Feb 2, 2021

@author: paepcke
'''
from _collections import OrderedDict
import argparse
from itertools import product
import json5
from json.decoder import JSONDecodeError as JSONError
import os
import socket
import sys
import copy

from birdsong.birds_train_parallel import BirdTrainer
from birdsong.utils.neural_net_config import NeuralNetConfig


# ------------------------ Specialty Exceptions --------
class ConfigError(Exception):
    pass


class TrainScripRunner(object):
    '''
    classdocs
    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 hparms_spec, 
                 starting_config_src,
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
        
        for perm_num, ordered_vals_tuple in enumerate(product(*hparms_spec.values())):
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
        of DottableConfigParser instances. Each dict
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
        @type config: DottableConfigParser
        @param config_dicts: one dict of hyperparm-name : value
            for each process to run independently
        @type config_dicts: [{str : Any}]
        @return: list of configurations for the classifier
            script to run
        @rtype: [DottableConfigParser]
        '''
        
        configs = []
        for conf_dict in config_dicts:
            conf_copy = copy.copy(config)
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
        
        self.WORLD_SIZE += machine_gpus
                    
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
        pass

# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Run multiple trainers independently"
                                     )

    parser.add_argument('-c', '--config_file',
                        help='fully qualified path to config file',
                        default=None)

    args = parser.parse_args();
    
    config_file = args.config_file
    if config_file is None:
        curr_dir = os.path.dirname(__file__)
        config_file = os.path.join(curr_dir, '../../config.cfg')
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Could not find config file at {config_file}")
    
    hparms_spec = {'lr' : [0.001],
                   'optimizer'  : ['Adam', 'RMSprop', 'SGD'],
                   'batch_size' : [32,64,128],
                   'kernel'     : [3,7]
                   }

    TrainScripRunner(hparms_spec, config_file)