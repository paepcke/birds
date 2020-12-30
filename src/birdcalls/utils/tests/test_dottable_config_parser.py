'''
Created on Dec 19, 2020

@author: paepcke
'''
import os
import unittest

from utils.dottable_config import DottableConfigParser


class TestDottableConfigParser(unittest.TestCase):

    
    def setUp(self):
        pass


    def tearDown(self):
        pass


    def test_basic(self):
        
        cfg_file = os.path.join(os.path.dirname(__file__), 'dottable_config_tst.cfg')
        dcp = DottableConfigParser(cfg_file)
        
        self.assertEqual(dcp.Paths.my_path, '/foo/bar.txt')
        self.assertEqual(dcp.Training.train_str, 'resnet18')
        
        self.assertEqual(dcp.getint('Training', 'train_int'), 5)
        self.assertEqual(dcp.getfloat('Training', 'train_float'), 3.14159)
        
        self.assertTrue(dcp.getboolean('Training', 'train_yes'))
        self.assertTrue(dcp.getboolean('Training', 'train_yescap'))
        self.assertTrue(dcp.getboolean('Training', 'train_YesCap'))
        self.assertTrue(dcp.getboolean('Training', 'train_1'))
        self.assertTrue(dcp.getboolean('Training', 'train_On'))
        self.assertFalse(dcp.getboolean('Training', 'train_no'))
        self.assertFalse(dcp.getboolean('Training', 'train_0'))
        self.assertFalse(dcp.getboolean('Training', 'train_off'))
       
        try:
            dcp.getarray('Paths', 'roots')
            self.assertFail("Expected value error due to enclosing brackets")
        except ValueError:
            pass
        
        self.assertEqual(dcp.getarray('Paths', 'toots'),
                         ['/foo/bar.txt', 'blue.jpg', '10', '3.14'])

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()