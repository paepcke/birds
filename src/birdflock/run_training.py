#!/usr/bin/env python
'''
Created on Sep 13, 2021

@author: paepcke

Command line script to conveniently start
a training. Opportunity to balance the dataset
via undersampling the majority class is available
in the --balance option, which takes a target ratio
of 
      num-minority-samples
      --------------------
      num-majority-samples

If oversampling the focal species is preferred, change
the call that instantiates BinaryBirdsTrainer.

'''

import argparse
import os
#import socket
import sys

import torch.multiprocessing as mp

from birdflock.birds_train_binaries import BinaryBirdsTrainer
from experiment_manager.neural_net_config import NeuralNetConfig

if __name__ == '__main__':
    # Must be first statement: needed because
    # multiprocess_runner.py uses spawn to create
    # subprocesses, which then each initialize CUDA
    mp.freeze_support()

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Run the binary classifier training for all species"
                                     )
    
    parser.add_argument('t', '--timestamp',
                        help='timestamp string to use in experiment directory names)'
                        )

    parser.add_argument('-s', '--species',
                        type=str,
                        nargs='+',
                        help='Repeatable: species for which to train classifiers; default: all',
                        default=None
                        )
    
    parser.add_argument('config_file',
                        help='path to the training config file (often called config.cfg)'
                        )


    args = parser.parse_args()
    
    cnf_path = os.path.abspath(args.config_file)
    if not os.path.exists(cnf_path):
        print(f"Cannot find {cnf_path}")
        sys.exit(1)
    
    try:
        config = NeuralNetConfig(cnf_path)
    except Exception as e:
        print("Error loading config file:")
        print(f"{repr(e)}")
        sys.exit(1)
        
    snips_root = config['Paths']['root_train_test_data']

    if not os.path.isdir(snips_root):
        print(f"Cannot find {snips_root}")
        sys.exit(1)
    
    # From config file, get the focal species
    # to train on this machine:
    
    #this_machine = socket.gethostname()
    # Uncomment to train some on quatro, and others on quintus:
    # if this_machine == 'quatro':
    #     focal_species = config.getarray('Training', 'focal_species_quatro')
    #     print("Will train only species for machine quatro")
    # elif this_machine == 'quintus':
    #     focal_species = config.getarray('Training', 'focal_species_quintus')
    #     print("Will train only species for machine quintus")
    # else:
    #     # Running on some other machine, just
    #     # use the species in the samples root dir
    #     # as given in the config file:
    #     focal_species = None
    
    # Only create classifiers for the 40 focal species:
    trainer = BinaryBirdsTrainer(config,
                                 focals_list=args.species
                                 )
    trainer.train()

