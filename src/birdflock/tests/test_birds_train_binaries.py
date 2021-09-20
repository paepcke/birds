'''
Created on Sep 11, 2021

@author: paepcke
'''
import os
from pathlib import Path
import unittest

from birdflock.birds_train_binaries import BinaryBirdsTrainer
from data_augmentation.multiprocess_runner import Task
from data_augmentation.utils import Utils
import multiprocessing as mp


#from sched import scheduler
TEST_ALL = True
#TEST_ALL = False


class BirdsBinaryTrainerTester(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cur_dir = os.path.dirname(__file__)
        cls.snippet_root = os.path.join(cls.cur_dir, 'data/snippets')
        cls.species1     = 'VASEG'
        cls.species2     = 'PATYG'
        cls.species3     = 'YOFLG'
        
        # Read the standard config file, and 
        # make some changes to match this testing
        # context:
        proj_root = Path(cls.cur_dir).parent.parent.parent
        config_path = proj_root.joinpath('config.cfg')
        cls.config  = Utils.read_configuration(str(config_path))
        # Change the species root to this test suite's data
        cls.config['Paths']['root_train_test_data'] = cls.snippet_root

    def setUp(self):
        pass


    def tearDown(self):
        pass

# -------------------- Tests ---------------

    #------------------------------------
    # test_constructor
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_constructor(self):
        
        trainer = BinaryBirdsTrainer(self.config)
        
        self.assertEqual(len(trainer.tasks_to_run), 3)
        
        trainer.train()
        for task in trainer.tasks_to_run:
            self.assertIsNone(task.error)
        print('Nothing was tested, but fit() finished')
        
        
    #------------------------------------
    # test_callback_scoring
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_callback_scoring(self):
        # To be added; or not.
        pass
    
    #------------------------------------
    # test_supplying_species_list
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_supplying_species_list(self):
        species_to_train = ['VASEG', 'YOFLG']
        trainer = BinaryBirdsTrainer(self.config,
                                     focals_list=species_to_train
                                     )
        self.assertEqual(trainer.tasks, species_to_train)
        
        trainer.train()
        
        for task in trainer.tasks_to_run:
            self.assertIsNone(task.error)

        trainer.train()

# ---------------------- Utilities ----------------

    def make_task(self, name):
        
        task_evnt = mp.Event()
        task = Task('name',
                    None, # Target function
                    shared_return_dict = {name : f"task_{name}"},
                    done_event=task_evnt
                    )
        return task

# -------------------- Main ---------------

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
    