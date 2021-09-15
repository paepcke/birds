'''
Created on Sep 4, 2021

@author: paepcke
'''
import os
import unittest

import torch
from torchvision import transforms

from birdflock.binary_dataset import BinaryDataset
from birdflock.binary_dataset import ClassBalancer, BalancingStrategy

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
        

    #------------------------------------
    # test_getitem_with_transforms
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_getitem_with_transforms(self):

        
        the_transforms = transforms.Compose(
            [transforms.RandomCrop([100,100]),
             transforms.ToTensor()
             ])

        target_species = 'BTSAC'
        ds = BinaryDataset(self.bin_ds_img_sample_root,
                           target_species,
                           transforms=the_transforms
                           )
        (img_tensor, _label) = ds[0]
        self.assertListEqual(list(img_tensor.shape), [1,100,100])

    #------------------------------------
    # test_dataset_splitting
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_dataset_splitting(self):
        
        # Get an image
        target_species = 'BTSAC'
        ds = BinaryDataset(self.bin_ds_img_sample_root,
                           target_species
                           )
        
        tst_fold_size = int(len(ds.data) / 5.)
        
        for i, (train_indices, test_indices) in enumerate(ds.split_generator(2, test_percentage=20)):
            self.assertEqual(len(train_indices), len(ds.data) - tst_fold_size)
            self.assertEqual(len(test_indices), tst_fold_size)
            
        # Should have received 2 splits:
        self.assertEqual(i+1, 2)

    #------------------------------------
    # test_class_balancer
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_class_balancer(self):
        
        target_species = 'BTSAC'
        ds = BinaryDataset(self.bin_ds_img_sample_root,
                           target_species
                           )
        
        # There is only one BTSAC in the 
        # test collection, so we can find it:
        minority_idx = ds.focal_indices[0]
        
        balancer = ClassBalancer(ds, 
                                 BalancingStrategy.OVERSAMPLE,
                                 minority_to_majority_target=1.0)

        data = balancer.new_data
        # Number of occurrences of the minority
        # index should equal the number of other
        # indices (since we by default asked for 1:1
        # ratio:
        self.assertEqual(data.count(minority_idx),
                         len(list(filter(lambda el: el != minority_idx, data)))
                         )
        
        # Try undersampling:
        balancer = ClassBalancer(ds, 
                                 BalancingStrategy.UNDERSAMPLE,
                                 minority_to_majority_target=1.0)
        data = balancer.new_data
        
        # Should have the single minority sample,
        # and one of the majority:
        self.assertEqual(len(data), 2)
        
        # Try a non-1:1 ratio:
        balancer = ClassBalancer(ds, 
                                 BalancingStrategy.UNDERSAMPLE,
                                 minority_to_majority_target=0.5)
        data = balancer.new_data
        # We should have 2 remaining majority
        # members:
        self.assertEqual(len(data), 3)
        self.assertEqual(data.count(minority_idx), 1)
        

    #------------------------------------
    # test_dataset_class_balancing
    #-------------------
    
    @unittest.skipIf(TEST_ALL != True, 'skipping temporarily')
    def test_dataset_class_balancing(self):
        target_species = 'BTSAC'
        ds = BinaryDataset(self.bin_ds_img_sample_root,
                           target_species,
                           balancing_strategy=BalancingStrategy.UNDERSAMPLE
                           )
        # Are undersampling the majority down
        # having only as many samples as BTSAC,
        # which is 1:
        self.assertEqual(len(ds.data), 2)
        


# ------------------------- Main -----------------
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()