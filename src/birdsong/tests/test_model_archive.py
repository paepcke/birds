'''
Created on Mar 16, 2021

@author: paepcke
'''
import os
import unittest

from birdsong.utils.model_archive import ModelArchive
from birdsong.utils.neural_net_config import NeuralNetConfig


TEST_ALL = True
#TEST_ALL = False

class Test(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(__file__)
        cls.main_test_dir = os.path.abspath(os.path.join(cls.curr_dir, 
                                                    '../tests'))
        cls.data_dir = os.path.abspath(os.path.join(cls.main_test_dir,
                                                    'data'))
        cls.config_path = os.path.join(cls.main_test_dir,
                                       'bird_trainer_tst.cfg'
                                       )
        cls.model_dir = os.path.join(cls.main_test_dir,
                                     'data_other/runs_models'
                                     )
        cls.config  = NeuralNetConfig(cls.config_path)
        
        # Doctor the path to the samples, which in
        # the test config is relative to the main test
        # dir:
        cls.config.Paths.root_train_test_data = cls.data_dir 

        cls.archive = ModelArchive(
                    cls.config, 
                    4,            # num_classes,
                    history_len=8,
                    model_root=cls.model_dir,
                    log=None)

    def setUp(self):
        pass


    def tearDown(self):
        pass
    
# -------------- Tests ---------------

    #------------------------------------
    # test_construct_run_subdir 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_construct_run_subdir(self):

        try:
            subdir = self.archive._construct_run_subdir(
                self.config, 
                4,               # Number of classes
                self.model_dir)
            self.assertEqual(len(os.listdir(subdir)),0)
            
            # Make another one and make sure
            # it gets a unique name:
            # If subdir was not unique among its
            # siblings, a '_r<n>' will be added
            # else no such suffix is added. We
            # use that to check whether disambiguation
            # will happen when we add another subdir: 
    
            # Get either '_r<n>' or not:
            suffix = subdir.split('_')[-1]
            
            # To make things easier, ensure that
            # this first suffix looks as if it
            # had ended in 'r<>'
            if suffix[-2] != 'r':
                suffix += 'r0'
            
            # Get the '0' or other num after
            # the 'r'
            disambig_num = int(suffix[1:])
            
            subdir1 = self.archive._construct_run_subdir(
                self.config, 
                4,               # Number of classes
                self.model_dir)
            
            self.assertEqual(len(os.listdir(subdir1)),0)
            
            suffix1 = subdir1.split('_')[-1]
            disambig_num1 = int(suffix1[1:])
    
            self.assertEqual(disambig_num1, disambig_num + 1)
        finally:
            for one_model_dir in os.listdir(self.model_dir):
                try:
                    os.rmdir(os.path.join(self.model_dir, one_model_dir))
                except Exception:
                    pass
            
    #------------------------------------
    # test__instantiate_model
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test__instantiate_model(self):
        
        model = self.archive._instantiate_model(config=self.config)
        
        self.assertEqual(model.__class__.__name__, 'ResNet')
        
        # We have six classes in our test samples:
        # bmw, audi, two bird species, 
        # diving gear, and office supplies:
        
        self.assertEqual(model.num_classes, 6)


# ---------------------- Main ------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()