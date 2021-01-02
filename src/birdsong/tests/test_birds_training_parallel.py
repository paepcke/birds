'''
Created on Dec 19, 2020

@author: paepcke
'''
import json
import os
from pathlib import Path
import unittest
from datetime import datetime

import torch

from birdsong.birds_train_parallel import BirdTrainer 
from birdsong.utils.dottable_config import DottableConfigParser
from birdsong.utils.learning_phase import LearningPhase

#*******TEST_ALL = True
TEST_ALL = False

class TestBirdsTrainingParallel(unittest.TestCase):

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
        trainer = BirdTrainer(self.config)
        self.assertEqual(trainer.get_lr(trainer.scheduler),
                         float(self.config.Training.lr)
                         )

    #------------------------------------
    # test_train
    #-------------------

    #*****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_train(self):
        self.set_distribution_env_vars()
        trainer = BirdTrainer(self.config)
        print("Start test training a small dataset (about 3 minutes)...")
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
        trainer = BirdTrainer(self.config)
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
        trainer1 = BirdTrainer(self.config, 
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
        
    #------------------------------------
    # test_process_group_init
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_process_group_init(self):
        self.set_distribution_env_vars()
        trainer = BirdTrainer(self.config)
        
        self.assertEqual(os.environ.get('WORLD_SIZE', None), '1')
        self.assertEqual(os.environ.get('MASTER_ADDR', None), '127.0.0.1')
        self.assertEqual(os.environ.get('MASTER_PORT', None), '9000')
        self.assertEqual(os.environ.get('RANK', None), '0')
                         
        trainer.init_multiprocessing(
                                  master_addr='127.0.0.1',
                                  master_port=9000,
                                  node_rank=0,
                                  world_size=1
            )


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
        # Mock up distributed processing:
        os.environ['WORLD_SIZE'] = '1'   # 1 GPU or CPU
        os.environ['RANK'] = '0'         # Master node
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '9000'



# ----------------------- Main -----------------

if __name__ == "__main__":
    
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
