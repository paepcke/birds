'''
Created on Dec 19, 2020

@author: paepcke
'''


import json
import os
from pathlib import Path
import unittest
from datetime import datetime
import socket

import torch

from birdsong.birds_train_parallel import BirdTrainer 
from birdsong.utils.dottable_config import DottableConfigParser

TEST_ALL = True
#TEST_ALL = False

#*****************
#
# import sys
# if socket.gethostname() in ('quintus', 'quatro'):
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
# ****************

class TestBirdsTrainingParallel(unittest.TestCase):

    DEFAULT_COMM_PORT = '9012' 

    #------------------------------------
    # setUpClass 
    #-------------------
    
    @classmethod
    def setUpClass(cls):

        cls.curr_dir = os.path.dirname(__file__)
        cls.json_logdir = os.path.join(cls.curr_dir,'runs_json')
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
            
            # Structure of json log entries:
            cls.json_entry_struct = \
                {measure_name : i 
                   for measure_name, i
                    in enumerate(
                        ["epoch", 
                         "loss", 
                         "training_accuracy", 
                         "testing_accuracy", 
                         "precision", 
                         "recall", 
                         "incorrect_paths", 
                         "confusion_matrix"])
                    }


    #------------------------------------
    # setUp 
    #-------------------

    def setUp(self):
        self.config_file = os.path.join(os.path.dirname(__file__), 'bird_trainer_tst.cfg')

        # Our own copy of the configuration:
        self.config = DottableConfigParser(self.config_file)

        # The stand-alone, single process distribution
        # parameter defaults:
        
        self.comm_info = {
            'MASTER_ADDR' :'127.0.0.1',
            'MASTER_PORT' : self.DEFAULT_COMM_PORT,
            'RANK' : 0,
            'LOCAL_RANK'  : 0,
            'MIN_RANK_THIS_MACHINE' : 0,
            'GPUS_USED_THIS_MACHINE': 2,
            'WORLD_SIZE'  : 1
            }


    #------------------------------------
    # tearDown 
    #-------------------

    def tearDown(self):
        pass

    #------------------------------------
    # test_train
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_train(self):
        self.set_distribution_env_vars()
        try:
            trainer = BirdTrainer(self.config,
                                  comm_info=self.comm_info
                                  )
        except Exception as e:
            print(f"****train: {repr(e)}")
            print(f"****MASTER_PORT: {os.environ['MASTER_PORT']}")
            raise

        try:
            device = trainer.device
            print(f"Running on {device}")
            predicted_time = f"about 3 minutes"\
                if device == trainer.cpu\
              else f"about 15 seconds" 
            print(f"Start test training a small dataset ({predicted_time})...")
            t_start = datetime.now()
            trainer.train()
            t_end = datetime.now()
            delta = t_end - t_start
            print(f"Done training checking result ({str(delta)})")
         
            # With this mini dataset, we converge
            # to plateau after epoch 7:
            # We don't know how many epoch will run,
            # b/c that depends on when accuracy levels
            # out. Which depends on the outcome of 
            # shuffling.
            #self.assertEqual(trainer.epoch, 7)
             
            # Everything should be on CPU, not GPU
            # after running:
            self.assertEqual(trainer.device_residence(trainer.model), 
                             torch.device('cpu'))
             
            # Expected number of results is 28:
            #   4 results (3 train + 1 validation) for the splits
            #   in each of the 7 epochs: 4*7=28
             
            expected_intermediate_results = trainer.epoch * 2 * trainer.dataloader.num_folds
            self.assertEqual(len(trainer.tally_collection),
                             expected_intermediate_results
                             )
             
            # Our test dataset has 6 target classes:
            self.assertEqual(trainer.num_classes, 6)
             
            # The JSON log record file:
             
            # Very superficial check of json results
            # log file: get last line:
             
            with open(trainer.json_log_filename()) as f:
                for line in f:
                    pass
                last_line = line
            # Last line should look like this:
            # [5, 86.2037582397461, 0.407407, 0.0, 0.0, 0.0, ["audi2.jpg", "audi3.jpg", "audi4.jpg", "audi6.jpg"], [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]]]
            last_entry = json.loads(last_line)
             
            # Last_entry is list; turn into dict: 
            measures = self.json_record_from_list(last_entry)
             
            # First number is the last epoch:
            self.assertEqual(measures['epoch'], trainer.epoch)
             
            # Next five elements should be floats:
            for measure_name in ['loss', 
                                 'training_accuracy',
                                 'testing_accuracy',
                                 'precision',
                                 'recall'
                                 ]:
                measure_type = type(measures[measure_name])
                self.assertEqual(measure_type, float)
             
            incorrect_paths_type = type(measures['incorrect_paths'])
            self.assertEqual(incorrect_paths_type, list)
     
            conf_matrix = torch.tensor(measures['confusion_matrix'])
            self.assertEqual(conf_matrix.shape,
                             (trainer.num_classes, trainer.num_classes) 
                         )
        finally:
            trainer.cleanup()


# -------------------- Utils --------------

    def json_record_from_list(self, record_list):
        '''
        Takes a list recovered from one line
        of a json log file, like:
        
           [5, 86.2037582397461, 0.407407, 0.0, ...]
           
        Returns a dict in which keys are measure
        names, and values are the above numbers and
        sublists.
         
        @param record_list: one line imported from a json log file
        @type record_list: list
        @return dict mapping measure-name to value
        @rtype {str : {str|list}}
        '''

        record_dict = \
            {measure_name : record_list[i]
                for i, measure_name
                 in enumerate([
                     "epoch", 
                     "loss", 
                     "training_accuracy", 
                     "testing_accuracy", 
                     "precision", 
                     "recall", 
                     "incorrect_paths", 
                     "confusion_matrix"])
             }
        return record_dict
    
    #------------------------------------
    # set_distribution_env_vars 
    #-------------------
    
    def set_distribution_env_vars(self):
        my_addr = socket.getfqdn()        
        # Mock up distributed processing:
        os.environ['WORLD_SIZE'] = '1'   # 1 GPU or CPU
        os.environ['RANK'] = '0'         # Master node
        os.environ['MASTER_ADDR'] = my_addr

        # The nose2 test framework runs the tests in
        # parallel. So all BirdTrainer instances need
        # to operate on a different port We use:
        #    test_birds_training_parallell_initialization.py: 5678
        #    test_birds_training_parallell_train.py         : 9012
        #    test_birds_training_parallel_save_model.py     : 3456
        # respectively 
        
        os.environ['MASTER_PORT'] = '9012'


        
        os.environ['MASTER_PORT'] = '5678'


# ----------------------- Main -----------------

if __name__ == "__main__":
    
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
