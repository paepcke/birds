'''
Created on Dec 19, 2020

@author: paepcke
'''
import os
import unittest

from birds_train_parallel import BirdTrainer 
from utils.dottable_config import DottableConfigParser

#*****TEST_ALL = True
TEST_ALL = False

class TestBirdsTrainingParallel(unittest.TestCase):


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
        print(trainer)


# ----------------------- Main -----------------

if __name__ == "__main__":
    
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()