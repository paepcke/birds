'''
Created on Feb 4, 2021

@author: paepcke
'''
import os
import unittest

from birdsong.utils.neural_net_config import NeuralNetConfig

#*****8TEST_ALL = True
TEST_ALL = False

class NeuralNetConfigTest(unittest.TestCase):


    def setUp(self):
        cfg_file = os.path.join(os.path.dirname(__file__), 'dottable_config_tst.cfg')
        self.config = NeuralNetConfig(cfg_file)

    def tearDown(self):
        pass

    # ------------ Tests -----------

    #------------------------------------
    # test_add_section 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_add_section(self):
        self.config.add_section('FoodleDoodle')
        secs = self.config.sections()
        self.assertIn('FoodleDoodle', secs)
        self.assertEqual(len(self.config.FoodleDoodle), 0)
        self.assertEqual(len(self.config['FoodleDoodle']), 0)
        
        self.config.FoodleDoodle = 10
        self.assertEqual(self.config.FoodleDoodle, 10)
        self.assertEqual(self.config['FoodleDoodle'], 10)
        
    #------------------------------------
    # test_setter_evals 
    #-------------------

    #****@unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_setter_evals(self):
        
        # A non-neural-net name:
        self.config.foo = 10
        self.assertEqual(self.config.foo, 10)
        
        # A nn-special parameter:
        self.config.batch_size = 128
        self.assertEqual(self.config.batch_size, 128)

    #------------------------------------
    # test_setter_methods
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_setter_methods(self):
        #****self.config.net_name = 'Foobar'
        self.config.set_net_name('Foobar')
        self.assertEqual(self.config.Training.net_name, 'Foobar')
        
        # Wrong type for epoch:
        with self.assertRaises(AssertionError):
            self.config.set_min_epochs('foo')
            
        # min_epoch > max_epoch:
        self.config.set_max_epochs(10)
        with self.assertRaises(AssertionError):
            self.config.set_min_epochs('20')
            
        self.config.set_batch_size(32)
        self.assertEqual(self.config.Training.batch_size, 32)
        with self.assertRaises(AssertionError):
            self.config.set_batch_size(-20)
        self.assertEqual(self.config.Training.batch_size, 32)
        
        with self.assertRaises(AssertionError):
            self.config.set_num_folds(-20)
        
        with self.assertRaises(AssertionError):
            self.config.set_all_procs_log(-20)
        self.config.set_all_procs_log(True)
        self.assertTrue(self.config.Parallelism.all_procs_log)
        


# -------------- Main -----------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()