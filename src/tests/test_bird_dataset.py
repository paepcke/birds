'''
Created on Dec 13, 2020

@author: paepcke
'''
from _collections import OrderedDict
import os
import unittest

from bird_dataset import BirdDataset

TEST_ALL = True
#TEST_ALL = False


class TestBirdDataset(unittest.TestCase):

    CURR_DIR = os.path.dirname(__file__)
    
    TEST_FILE_PATH = os.path.join(CURR_DIR, 'data/train')

    #------------------------------------
    # setUpClass
    #-------------------
    
    @classmethod
    def setUpClass(cls):
        cls.sample_class_assignments = \
            OrderedDict([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), 
                         (6, 1), (7, 1), (8, 1), (9, 1), (10, 1), (11,1)])

    
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
        self.assertEqual(ds.class_to_id['DYSMEN_S'], 0)
        self.assertEqual(ds.class_to_id['HENLES_S'], 1)
        self.assertEqual(ds.sample_id_to_class,self.sample_class_assignments)
        self.assertEqual(len(ds.sample_id_to_path), len(ds.sample_id_to_class))
        
    #------------------------------------
    # testGetItem 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def testGetItem(self):
        ds = BirdDataset(self.TEST_FILE_PATH)
        (_img, class_id) = ds[0]
        self.assertEqual(class_id, 0)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()