'''
Created on Dec 13, 2020

@author: paepcke
'''
import os
import sys
import unittest

from bird_dataset import BirdDataset


sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


TEST_ALL = True
#TEST_ALL = False


class TestBirdDataset(unittest.TestCase):

    CURR_DIR = os.path.dirname(__file__)
    
    TEST_FILE_PATH = os.path.join(CURR_DIR, 'data/train')
    
    #------------------------------------
    # setUp
    #-------------------

    def setUp(self):
        pass

    #------------------------------------
    # tearDown
    #-------------------

    def tearDown(self):
        pass

    #------------------------------------
    # testStructuresCreation 
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testStructuresCreation(self):
        
        ds = BirdDataset(self.TEST_FILE_PATH)
        print(ds)
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()