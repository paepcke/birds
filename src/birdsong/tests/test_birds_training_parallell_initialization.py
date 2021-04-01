'''
Created on Dec 19, 2020

@author: paepcke
'''


import os
from pathlib import Path
import socket
import unittest

from birdsong.birds_train_parallel import BirdTrainer 
from birdsong.utils.neural_net_config import NeuralNetConfig


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

    DEFAULT_COMM_PORT = 5678

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
        self.config = NeuralNetConfig(self.config_file)

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
    # test_training_init 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_training_init(self):

        self.set_distribution_env_vars()
        try:
            trainer = BirdTrainer(self.config,
                                  comm_info=self.comm_info
                                  )
        except Exception as e:
            print(f"****init: {repr(e)}")
            print(f"****MASTER_PORT: {os.environ['MASTER_PORT']}")
            raise

        try:
            self.assertEqual(trainer.get_lr(trainer.scheduler),
                             float(self.config.Training.lr)
                             )
        finally:
            trainer.cleanup()

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
        
        os.environ['MASTER_PORT'] = '5678'

# ----------------------- Main -----------------

if __name__ == "__main__":
    
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
