'''
Created on Dec 19, 2020

@author: paepcke
'''
from datetime import datetime
import os
from pathlib import Path
import unittest

import torch

from birds_train_parallel import BirdTrainer 
from utils.dottable_config import DottableConfigParser


#*****TEST_ALL = True
TEST_ALL = False

class TestBirdsTrainingParallel(unittest.TestCase):

    #------------------------------------
    # setUpClass 
    #-------------------
    
    @classmethod
    def setUpClass(cls):
        curr_dir = os.path.dirname(__file__)
        cls.json_logdir = os.path.join(curr_dir,'runs_json')
        if os.path.exists(cls.json_logdir):
            # Get list of absolute paths of
            # .jsonl files created by earlier
            # test runs: 
            json_tst_result_files = \
                [os.path.join(cls.json_logdir, base_file)
                 for base_file
                  in os.listdir(cls.json_logdir)]
            if len(json_tst_result_files) == 0:
                # Nothing to delete:
                return

            # Create a dict {file_name : file_creation_time}
            file_creation_times = {file : Path(file).stat().st_birthtime
                                    for file
                                     in json_tst_result_files
                                     }
            # Start with first file as current
            # 'most recent', which will be a 
            # file_name (just the basename):
            
            most_recent = next(iter(file_creation_times.keys()))
            
            for (file_name, this_ctime) in file_creation_times.items():
                # Compare creation times
                if this_ctime > file_creation_times[most_recent]:
                    most_recent = file_name
                    
            [os.remove(os.path.abspath(file_name))
             for file_name
             in file_creation_times.keys()
             if file_name != most_recent
             ]

    #------------------------------------
    # setUp 
    #-------------------

    def setUp(self):
        self.config_file = os.path.join(os.path.dirname(__file__), 'bird_trainer_tst.cfg')

        # Our own copy of the configuration:
        self.config = DottableConfigParser(self.config_file)
        #******** Remove
        # Set the root of train/val data to this
        # test dir:
        #data_root = os.path.join(os.path.dirname(__file__), 'data')
        #self.config.Paths.root_train_test_data = data_root
        #*********

    #------------------------------------
    # tearDown 
    #-------------------

    def tearDown(self):
        pass

    #------------------------------------
    # test_training_init 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_training_init(self):

        trainer = BirdTrainer(self.config)
        self.assertEqual(trainer.get_lr(trainer.scheduler),
                         float(self.config.Training.lr)
                         )
        print(trainer.model)
        
    #------------------------------------
    # test_train
    #-------------------

    #******@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_train(self):
        trainer = BirdTrainer(self.config)
        trainer.train()
        # With this mini dataset, we converge
        # to plateau after epoch 7:
        self.assertEqual(trainer.epoch, 7)
        
        # Everything should be on CPU, not GPU
        # after running:
        self.assertEqual(trainer.device, torch.device('cpu'))
        self.assertEqual(trainer.device_residence(trainer.model),
                         torch.device('cpu'))
        
        # Expected number of results is 28:
        #   4 results (3 train + 1 validation) for the splits
        #   in each of the 7 epochs: 4*7=28
        
        expected_intermediate_results = (trainer.epoch * (1+trainer.dataloader.num_folds))
        self.assertEqual(len(trainer.tally_collection),
                         expected_intermediate_results
                         )
        
        # Our test dataset has 6 target classes:
        self.assertEqual(trainer.model.num_classes, 6)
        trainer.device
        print(trainer)


# ----------------------- Main -----------------

if __name__ == "__main__":
    
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()