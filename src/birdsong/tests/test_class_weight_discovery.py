'''
Created on Jan 21, 2021

@author: paepcke
'''
import os
import unittest

import natsort
import torch

from birdsong.class_weight_discovery import ClassWeightDiscovery


class Test(unittest.TestCase):

    #------------------------------------
    # setUpClass 
    #-------------------


    @classmethod
    def setUpClass(cls):
        # True number of samples in 
        # test data hierarchy:
        
        cls.true_class_sizes = {'DYSMEN_S'       : 6,
                                'HNENLES_C'      : 6,
                                'audi'           : 6,
                                'bmw'            : 6,
                                'diving_gear'    : 5,
                                'office_supplies': 5
                                }

        cls.file_root = os.path.join(os.path.dirname(__file__), 'data')
        cls.sorted_classes = natsort.natsorted(cls.true_class_sizes.keys())
        cls.true_weights   = [cls.true_class_sizes[class_name] / 6.0
                                for class_name in cls.sorted_classes
                                ]

    #------------------------------------
    # test_class_weight_discovery 
    #-------------------

    def test_class_weight_discovery(self):
        
        computed_weights = ClassWeightDiscovery.get_weights(self.file_root)
        true_weight_tensor = torch.tensor(self.true_weights)
        self.assertTrue(true_weight_tensor.eq(computed_weights).all())


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()