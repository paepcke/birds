'''
Created on Sep 4, 2021

@author: paepcke
'''
import os
import unittest

import torch

from birdsong.binary_dataset import BinaryDataset


TEST_ALL = True
#TEST_ALL = False


class BinaryDatasetTest(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        
        cls.cur_dir = os.path.dirname(__file__)
        cls.bin_ds_img_sample_root = os.path.join(cls.cur_dir, 'data/binary_dataset_img_samples')
        cls.bin_ds_aud_sample_root = os.path.join(cls.cur_dir, 'data/binary_dataset_sound_samples')
        

    def setUp(self):
        pass


    def tearDown(self):
        pass

# -------------------- Tests --------------------

    #------------------------------------
    # test_constructor
    #-------------------

    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_constructor(self):
        
        #     BTSAC:
        #        btsac1.png
        #     GRHOG
        #        grhog1.png
        #        grhog2.png
        #     HOWPG
        #        howpg1.png
        #        howpg2.png
        #        howpg3.png
        
        # Images:
        target_species = 'BTSAC'
        ds = BinaryDataset(self.bin_ds_img_sample_root,
                           target_species
                           )
        self.assertTrue(len(ds), 6)
        
        # Sounds:
        target_species = 'ROHAG'
        ds = BinaryDataset(self.bin_ds_aud_sample_root,
                           target_species
                           )
        self.assertTrue(len(ds), 6)

    #------------------------------------
    # test__getitem__
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test__getitem__(self):
        
        # Get an image
        target_species = 'BTSAC'
        ds = BinaryDataset(self.bin_ds_img_sample_root,
                           target_species
                           )
        spectro, label = ds[0]
        self.assertEqual(type(spectro), torch.Tensor)
        self.assertEqual(type(label), torch.Tensor)
        self.assertTrue(label.item() in [0,1])
        
        # Get an audio:
        target_species = 'ROHAG'
        ds = BinaryDataset(self.bin_ds_aud_sample_root,
                           target_species
                           )
        sound_tensor, label = ds[0]
        self.assertEqual(type(sound_tensor), torch.Tensor)
        self.assertEqual(type(label), torch.Tensor)
        self.assertTrue(label.item() in [0,1])
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()