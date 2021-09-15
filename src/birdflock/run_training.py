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
import sys

from birdflock.binary_dataset import BalancingStrategy
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
    parser.add_argument('-b', '--balance',
                        type=float,
                        help='balance dataset so that num-minority/num-majority is given value',
                        default=None
                        )


    parser.add_argument('snippets_dir',
                        help='path to root of all species spectrogram snippets'
                        )

    args = parser.parse_args()
    
    snips_root = args.snippets_dir
    if not os.path.isdir(snips_root):
        print(f"Cannot find {snips_root}")
        sys.exit(1)
    
    focal_species = ['BANAG','BBFLG','BCMMG','BHPAG','BHTAG',
                     'BTSAC','BTSAS','CCROC','CCROS','CFPAG',
                     'CMTOG','CTFLG','DCFLC','DCFLS','FBARG',
                     'GCFLG','GHCHG','GHTAG','GRHEG','LEGRG',
                     'NOISG','OBNTC','OBNTS','OLPIG','PATYG',
                     'RBWRC','RBWRS','SHWCG','SOFLG','SPTAG',
                     'SQCUC','SQCUS','STTAG','TRGNC','TRGNS',
                     'WCPAG','WTDOG','WTROC','WTROS','YCEUG'] 

    # Only create classifiers for the 40 focal species:
    trainer = BinaryBirdsTrainer(snips_root,
                                 focal_species=focal_species,
                                 balancing_strategy=BalancingStrategy.UNDERSAMPLE,
                                 balancing_ratio=args.balance
                                 )
    trainer.train()

