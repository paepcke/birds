'''
Created on Sep 8, 2021

@author: paepcke
'''
import os
import unittest

from birdsong.train_one_binary_classifier import BinaryClassificationTrainer


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        cls.cur_dir = os.path.dirname(__file__)
        cls.species_root = os.path.join(cls.cur_dir, 'data/birds/') 

    def setUp(self):
        pass


    def tearDown(self):
        pass


# ----------------------- Tests ----------------

    def test_classifier_creation(self):
        
        trainer = BinaryClassificationTrainer(self.species_root, 'PLANS')
        
        print('foo')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()