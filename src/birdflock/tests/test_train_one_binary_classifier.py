'''
Created on Sep 8, 2021

@author: paepcke
'''
import os
from pathlib import Path
import tempfile
import unittest

from experiment_manager.experiment_manager import ExperimentManager

from birdflock.train_one_binary_classifier import BinaryClassificationTrainer
from data_augmentation.utils import Utils


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        cls.cur_dir = os.path.dirname(__file__)
        cls.species_root = os.path.join(cls.cur_dir, 'data/snippets/') 

        # Read the standard config file, and 
        # make some changes to match this testing
        # context:
        proj_root = Path(cls.cur_dir).parent.parent.parent
        config_path = proj_root.joinpath('config.cfg')
        cls.config  = Utils.read_configuration(str(config_path))
        # Change the species root to this test suite's data
        cls.config['Paths']['root_train_test_data'] = cls.species_root

    def setUp(self):
        pass


    def tearDown(self):
        pass


# ----------------------- Tests ----------------

    def test_classifier_creation(self):
        
        with tempfile.TemporaryDirectory(dir='/tmp', prefix='tst_1_classifier') as dir_name:
            exp_root = os.path.join(dir_name, 'Experiments') 
            os.mkdir(exp_root)
            exp = ExperimentManager(exp_root)
            trainer = BinaryClassificationTrainer(self.config, 
                                                  'PATYG',
                                                  experiment=exp
                                                  )
        
        print('foo')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()