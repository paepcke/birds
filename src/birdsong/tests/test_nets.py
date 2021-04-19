'''
Created on Mar 2, 2021

@author: paepcke
'''
import os
import unittest

import torch

from birdsong.nets import NetUtils


TEST_ALL = True
#TEST_ALL = False


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.curr_dir = os.path.dirname(__file__)
        cls.tst_config = os.path.join(cls.curr_dir, 'bird_trainer_tst.cfg')
        cls.num_classes = 4

    def setUp(self):
        pass


    def tearDown(self):
        pass

# -------------- Tests ------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_densenet(self):
        num_pretrained_layers = 6
        #trainer = BirdTrainer(self.tst_config)
        model = NetUtils.get_net(
            'densenet161',
            num_classes=self.num_classes,  # num_classes
            num_layers_to_retain=num_pretrained_layers,
            to_grayscale=True
            )
        self.assertEqual(model.state_dict()['features.conv0.weight'].shape,
                         torch.Size([96, 1, 7, 7])
                         )
        self.assertEqual(model.classifier.out_features, 4)

    #------------------------------------
    # test_resnet 
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_resnet(self):
        
        num_pretrained_layers = 6
        model = NetUtils.get_net(
            'resnet18',
            num_classes=self.num_classes,  # num_classes
            num_layers_to_retain=num_pretrained_layers,
            to_grayscale=True
            )
        self.assertEqual(model.state_dict()['conv1.weight'].shape,
                         torch.Size([64, 1, 7, 7])
                         )
        
        self.assertEqual(model.fc.out_features, 4)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
