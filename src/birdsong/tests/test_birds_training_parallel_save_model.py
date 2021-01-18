'''
Created on Dec 19, 2020

@author: paepcke
'''


import os
from pathlib import Path
import unittest
import socket

import torch

from birdsong.birds_train_parallel import BirdTrainer 
from birdsong.utils.dottable_config import DottableConfigParser
from birdsong.utils.learning_phase import LearningPhase

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
    # test_model_saving 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_model_saving(self):

        four_results = torch.tensor(               # Label
            [[0.5922, 0.6546, 0.7172, 0.0139],     #   2
        	 [0.9124, 0.9047, 0.6819, 0.9329],     #   3
        	 [0.2345, 0.1733, 0.5420, 0.4659],     #   2
        	 [0.5954, 0.8958, 0.2294, 0.5529]      #   1
             ])

        # Make the results same as in real life:
        four_results_training = torch.unsqueeze(four_results, dim=1)
        four_truths = torch.tensor([1,2,3,4])
    
        self.set_distribution_env_vars()
        trainer = BirdTrainer(self.config,
                              comm_info=self.comm_info
                              )
            
        try:
            tally1 = trainer.tally_result(
                        0, # Split number
                        four_truths,
                        four_results_training,  # using the 3D result
                        10.1,
                        LearningPhase.TRAINING
                        )
            tally2 = trainer.tally_result(
                        1, # Split number
                        four_truths,
                        four_results,  # using the 2D result
                        20.2,
                        LearningPhase.VALIDATING
                        )
    
            
            # Pretend the trainer is in the 10th epoch...
            trainer.epoch = 10
            # ... and the tally1 above was 
            # created during the 9th epoch:
            tally1.epoch = 10
            tally2.epoch = 9
            # Have to correct the tallies' keys
            # in the tally_collection. For tally1:
            trainer.tally_collection[(10, 0, 'Training')] = \
                trainer.tally_collection[(None, 0, 'Training')]
            trainer.tally_collection.pop((None, 0, 'Training'))
            
            # For tally2:
            trainer.tally_collection[(9, 1, 'Validating')] = \
                trainer.tally_collection[(None, 1, 'Validating')]
            trainer.tally_collection.pop((None, 1, 'Validating'))
            
            
            # Make sure the surgery worked:
            self.assertEqual(trainer.tally_collection[(10,0,'Training')].epoch,10)
            self.assertEqual(trainer.tally_collection[(9,1,'Validating')].epoch,9)
            
            save_file = os.path.join(self.curr_dir, 'saved_model.pth')
            trainer.save_model_checkpoint(save_file, 
                                          trainer.model, 
                                          trainer.optimizer)
            
            # Make a new trainer from the old trainer's
            # saved checkpoint:
            self.set_distribution_env_vars()
        finally:
            trainer.cleanup()

        try:
            trainer1 = BirdTrainer(self.config,
                                   comm_info=self.comm_info,
                                   checkpoint=save_file)
            self.assertEqual(trainer1.epoch, 10)
            
            # Tally2 should be in the new trainer's
            # tallies collection:
            self.assertEqual(trainer1.tally_collection[(9,1,'Validating')].epoch,9)
            # Tally1 should have been removed during the
            # saving process because its epoch was same as
            # trainer's current epoch:
            self.assertEqual(len(trainer1.tally_collection), 1)
            
            tally2Prime = trainer1.tally_collection[(9,1,'Validating')]
            
            # Check two of the constants:
            self.assertEqual(tally2.learning_phase, tally2Prime.learning_phase)
            self.assertEqual(tally2Prime.loss, 20.2)
    
            # One of the lazily computed values:
            self.assertEqual(tally2Prime.recall, tally2.recall)
            
            # But tally2Prime must be a *copy* of tally2:
            self.assertNotEqual(tally2Prime, tally2)
        finally:
            try:
                trainer1.cleanup()
            except (UnboundLocalError, NameError):
                # Error before trainer1 was created:
                pass
            
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
        os.environ['MASTER_PORT'] = '5678'

# ----------------------- Main -----------------

if __name__ == "__main__":
    
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
