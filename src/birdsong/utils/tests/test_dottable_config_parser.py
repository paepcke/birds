'''
Created on Dec 19, 2020

@author: paepcke
'''
import os, sys
packet_root = os.path.abspath(__file__.split('/')[0])
sys.path.insert(0,packet_root)

import unittest

from birdsong.utils.dottable_config import DottableConfigParser

TEST_ALL = True
#TEST_ALL = False

class TestDottableConfigParser(unittest.TestCase):

    
    def setUp(self):
        pass


    def tearDown(self):
        pass

    #------------------------------------
    # test_basics
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_basics(self):
        
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

    #------------------------------------
    # test_expand_paths 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_expand_paths(self):
        cfg_file = os.path.join(os.path.dirname(__file__), 'dottable_config_tst.cfg')
        dcp = DottableConfigParser(cfg_file)

                
        new_path = dcp.expand_path('./src', '/Users/doe/code')
        self.assertEqual('/Users/doe/code/src', new_path)
        
        new_path = dcp.expand_path('./src/.', '/Users/doe/code')
        self.assertEqual(new_path, '/Users/doe/code/src')

        new_path = dcp.expand_path('./src/..', '/Users/doe/code')
        self.assertEqual(new_path, '/Users/doe/code')
        
        new_path = dcp.expand_path('./../src/..', '/Users/doe/code')
        self.assertEqual(new_path, '/Users/doe')

# ------------------- Main --------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()