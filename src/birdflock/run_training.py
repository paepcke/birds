#!/usr/bin/env python3
'''
Created on Sep 13, 2021

@author: paepcke
'''

import argparse
import os
import sys

from birdflock.birds_train_binaries import BinaryBirdsTrainer
import multiprocessing as mp


if __name__ == '__main__':
    # Must be first statement: needed because
    # multiprocess_runner.py uses spawn to create
    # subprocesses, which then each initialize CUDA
    mp.freeze_support()

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Run the binary classifier training for all species"
                                     )

    parser.add_argument('snippets_dir',
                        help='path to root of all species spectrogram snippets'
                        )

    args = parser.parse_args()
    
    snips_root = args.snippets_dir
    if not os.path.isdir(snips_root):
        print(f"Cannot find {snips_root}")
        sys.exit(1)
    trainer = BinaryBirdsTrainer(snips_root)
    trainer.train()

